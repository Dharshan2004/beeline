"""Train deterministic global fusion weights from the Slice 09 replay cache.

The search is deliberately model-free. It reconstructs candidate pools from
cached normalized route scores, orders those pools with cached reranker logits,
and applies the official session-level metrics to the frozen trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Callable, Sequence

from retrieval.fusion import ROUTE_NAMES
from retrieval.reranker import order_by_scores
from tools.benchmark_reranker import session_metrics
from tools.build_fusion_dataset import MAX_TRAINING_POOL_DEPTH, load_artifact


REPORT_VERSION = "fusion-weight-search-v1"
CURRENT_WEIGHTS = {"structured": 0.15, "bm25": 0.55, "dense": 0.3}
DEFAULT_DEPTH = 50
TOP_K = 10


def simplex_weights(units: int) -> list[dict[str, float]]:
    """Return every three-route simplex point on an exact integer grid."""
    if units <= 0:
        raise ValueError("simplex units must be positive")
    return [
        {
            "structured": structured / units,
            "bm25": bm25 / units,
            "dense": (units - structured - bm25) / units,
        }
        for structured in range(units + 1)
        for bm25 in range(units - structured + 1)
    ]


def _route_items(record: dict, route_name: str) -> list[tuple[str, float]]:
    return [
        (str(item["parent_asin"]), float(item["normalized_score"]))
        for item in record["route_candidates"][route_name]
    ]


def rank_weighted_pool(
    record: dict,
    weights: dict[str, float],
    *,
    depth: int,
) -> list[str]:
    """Reconstruct weighted fusion from cached per-route normalized scores."""
    if set(weights) != set(ROUTE_NAMES):
        raise ValueError("weights must define every Retrieval Route")
    numeric_weights = {name: float(weights[name]) for name in ROUTE_NAMES}
    if (
        any(not isfinite(value) or value < 0.0 for value in numeric_weights.values())
        or abs(sum(numeric_weights.values()) - 1.0) > 1e-9
    ):
        raise ValueError("weights must be finite, non-negative, and sum to one")
    if depth <= 0:
        raise ValueError("depth must be positive")
    fused: dict[str, float] = {}
    for route_name in ROUTE_NAMES:
        weight = numeric_weights[route_name]
        for identifier, score in _route_items(record, route_name):
            fused[identifier] = fused.get(identifier, 0.0) + weight * score
    return [
        identifier
        for identifier, _score in sorted(
            fused.items(), key=lambda item: (-item[1], item[0])
        )[:depth]
    ]


def _rank_single_pool(record: dict, route_name: str, depth: int) -> list[str]:
    return [
        identifier
        for identifier, _score in sorted(
            _route_items(record, route_name),
            key=lambda item: (-item[1], item[0]),
        )[:depth]
    ]


def _rank_rrf_pool(record: dict, depth: int, rank_constant: int = 60) -> list[str]:
    fused: dict[str, float] = {}
    for route_name in ROUTE_NAMES:
        ordered = _rank_single_pool(record, route_name, MAX_TRAINING_POOL_DEPTH)
        for rank, identifier in enumerate(ordered, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (
                rank_constant + rank
            )
    return [
        identifier
        for identifier, _score in sorted(
            fused.items(), key=lambda item: (-item[1], item[0])
        )[:depth]
    ]


def _rerank(record: dict, pool: Sequence[str]) -> list[str]:
    score_by_id = {
        str(item["parent_asin"]): float(item["score"])
        for item in record["reranker_scores"]
    }
    try:
        scores = [score_by_id[identifier] for identifier in pool]
    except KeyError as error:
        raise ValueError(
            f"cached reranker score missing for {error.args[0]}"
        ) from error
    return order_by_scores(pool, scores)


def _pool_recall(records: Sequence[dict], pools: dict[tuple[str, int], list[str]]) -> dict:
    eligible = [record for record in records if record.get("hit_eligible", True)]
    turn_hits = sum(
        str(record["target"]) in pools[(str(record["sample_id"]), int(record["turn"]))]
        for record in eligible
    )
    by_session: dict[str, bool] = {}
    for record in eligible:
        sample_id = str(record["sample_id"])
        hit = str(record["target"]) in pools[(sample_id, int(record["turn"]))]
        by_session[sample_id] = by_session.get(sample_id, False) or hit
    return {
        "turn_pool_recall": round(turn_hits / len(eligible), 6) if eligible else 0.0,
        "session_pool_recall": (
            round(sum(by_session.values()) / len(by_session), 6)
            if by_session else 0.0
        ),
    }


def _official_metrics(
    records: Sequence[dict], ranked: dict[tuple[str, int], list[str]]
) -> dict:
    metrics = session_metrics(records, ranked)
    metrics["recommended_technical_score"] = round(
        0.50 * metrics["hit_rate_at_10"]
        + 0.30 * metrics["mrr"]
        + 0.20 * metrics["efficiency"],
        6,
    )
    return metrics


def _evaluate_ranker(
    records: Sequence[dict],
    ranker: Callable[[dict], list[str]],
    *,
    rerank: bool = True,
) -> dict:
    pools = {
        (str(record["sample_id"]), int(record["turn"])): ranker(record)
        for record in records
    }
    ranked = {
        key: (
            _rerank(record, pools[key])[:TOP_K]
            if rerank else pools[key][:TOP_K]
        )
        for record in records
        for key in [(str(record["sample_id"]), int(record["turn"]))]
    }
    recall = _pool_recall(records, pools)
    metrics = _official_metrics(records, ranked)
    scenarios = sorted({str(record["scenario_type"]) for record in records})
    scenario_metrics = {}
    for scenario in scenarios:
        scenario_records = [
            record for record in records
            if str(record["scenario_type"]) == scenario
        ]
        scenario_keys = {
            (str(record["sample_id"]), int(record["turn"]))
            for record in scenario_records
        }
        scenario_metrics[scenario] = {
            "pool_recall": _pool_recall(scenario_records, pools),
            "metrics": _official_metrics(
                scenario_records,
                {key: ranked[key] for key in scenario_keys},
            ),
        }
    return {
        "reranked": rerank,
        "pool_recall": recall,
        "metrics": metrics,
        "recall_to_hit_conversion": (
            round(metrics["hit_rate_at_10"] / recall["session_pool_recall"], 6)
            if recall["session_pool_recall"] else 0.0
        ),
        "scenario_metrics": scenario_metrics,
    }


def evaluate_weighted_policy(
    records: Sequence[dict],
    weights: dict[str, float],
    *,
    depth: int = DEFAULT_DEPTH,
) -> dict:
    result = _evaluate_ranker(
        records,
        lambda record: rank_weighted_pool(record, weights, depth=depth),
    )
    return {"weights": dict(weights), "depth": depth, **result}


def _candidate_key(candidate: dict) -> tuple:
    recall = candidate["pool_recall"]
    metrics = candidate["metrics"]
    weights = candidate["weights"]
    distance = sum(
        abs(float(weights[name]) - CURRENT_WEIGHTS[name]) for name in ROUTE_NAMES
    )
    return (
        recall["session_pool_recall"],
        metrics["recommended_technical_score"],
        metrics["hit_rate_at_10"],
        metrics["mrr"],
        recall["turn_pool_recall"],
        -round(distance, 12),
        tuple(float(weights[name]) for name in ROUTE_NAMES),
    )


def select_best(candidates: Sequence[dict]) -> dict:
    if not candidates:
        raise ValueError("weight search produced no candidates")
    return max(candidates, key=_candidate_key)


def _search_summary(candidate: dict) -> dict:
    return {
        "weights": candidate["weights"],
        "session_pool_recall": candidate["pool_recall"]["session_pool_recall"],
        "turn_pool_recall": candidate["pool_recall"]["turn_pool_recall"],
        "recommended_technical_score": candidate["metrics"][
            "recommended_technical_score"
        ],
        "hit_rate_at_10": candidate["metrics"]["hit_rate_at_10"],
        "mrr": candidate["metrics"]["mrr"],
    }


def _local_weights(
    winner: dict[str, float],
    *,
    units: int,
    radius: float,
) -> list[dict[str, float]]:
    return [
        candidate for candidate in simplex_weights(units)
        if all(abs(candidate[name] - winner[name]) <= radius + 1e-12 for name in ROUTE_NAMES)
    ]


def _baseline(
    records: Sequence[dict],
    name: str,
    *,
    depth: int,
) -> dict:
    if name == "rrf":
        result = _evaluate_ranker(records, lambda record: _rank_rrf_pool(record, depth))
        return {"policy": "fixed-rrf-v1", "depth": depth, **result}
    if name.startswith("single_"):
        route_name = name.removeprefix("single_")
        result = _evaluate_ranker(
            records,
            lambda record: _rank_single_pool(record, route_name, depth),
        )
        return {"policy": f"single-{route_name}-v1", "depth": depth, **result}
    raise ValueError(f"unknown baseline {name}")


def load_fused30_baseline(path: str | Path) -> dict:
    """Load the complete no-reranker control from the Slice 7 decision gate."""
    source = Path(path)
    raw = source.read_bytes()
    report = json.loads(raw)
    metadata = report.get("cache_metadata") or {}
    configuration = metadata.get("retrieval_configuration") or {}
    rows = [
        row for row in report.get("rows", [])
        if row.get("identity") == "none (fused-30 baseline)"
    ]
    if (
        len(rows) != 1
        or metadata.get("split_version") != "public-split-v1"
        or metadata.get("session_count") != 160
        or metadata.get("turn_count") != 1009
        or configuration.get("policy_version") != "fixed-hybrid-v1"
        or configuration.get("fused_candidate_depth") != 30
        or configuration.get("weights") != CURRENT_WEIGHTS
    ):
        raise ValueError("fused-30 report has stale or incomplete provenance")
    row = rows[0]
    baseline_metrics = metadata.get("baseline_metrics") or {}
    recall = {
        "session_pool_recall": row.get("session_pool_recall"),
        "turn_pool_recall": row.get("turn_pool_recall"),
    }
    if (
        row.get("depth") != 30
        or row.get("reranked") is not False
        or row.get("session_count") != 160
        or metadata.get("pool_recall_by_depth", {}).get("30") != recall
        or row.get("session_count") != baseline_metrics.get("sample_count")
        or any(
            row.get(key) != baseline_metrics.get(key)
            for key in (
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
            )
        )
    ):
        raise ValueError("fused-30 control row does not match its cache metadata")
    return {
        "policy": "fixed-hybrid-v1",
        "weights": dict(CURRENT_WEIGHTS),
        "depth": 30,
        "reranked": False,
        "pool_recall": recall,
        "metrics": {
            key: row[key]
            for key in (
                "session_count",
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
            )
        },
        "recall_to_hit_conversion": row.get("recall_to_hit_conversion"),
        "scenario_metrics": baseline_metrics.get("scenario_metrics", {}),
        "provenance": {
            "source_report_sha256": hashlib.sha256(raw).hexdigest(),
            "cache_sha256": metadata.get("cache_sha256"),
            "turn_count": metadata.get("turn_count"),
        },
    }


def train_fusion_policy(
    records: Sequence[dict],
    manifest: dict,
    fused30_baseline: dict,
    *,
    coarse_units: int = 10,
    local_units: int = 50,
    local_radius: float = 0.1,
    depth: int = DEFAULT_DEPTH,
) -> dict:
    """Run the coarse and local searches and return a reproducible report."""
    coarse = [
        evaluate_weighted_policy(records, weights, depth=depth)
        for weights in simplex_weights(coarse_units)
    ]
    coarse_winner = select_best(coarse)
    local_grid = _local_weights(
        coarse_winner["weights"], units=local_units, radius=local_radius
    )
    local = [
        evaluate_weighted_policy(records, weights, depth=depth)
        for weights in local_grid
    ]
    winner = select_best(local)
    full_union_sizes = [
        len({
            str(item["parent_asin"])
            for route_name in ROUTE_NAMES
            for item in record["route_candidates"][route_name]
        })
        for record in records
    ]
    baselines = {
        "fused_30": fused30_baseline,
        "current_fixed": evaluate_weighted_policy(
            records, CURRENT_WEIGHTS, depth=depth
        ),
        "rrf": _baseline(records, "rrf", depth=depth),
        **{
            f"single_{route_name}": _baseline(
                records, f"single_{route_name}", depth=depth
            )
            for route_name in ROUTE_NAMES
        },
    }
    return {
        "report_version": REPORT_VERSION,
        "provenance": {
            key: manifest.get(key)
            for key in (
                "artifact_version",
                "split_version",
                "public_set_sha256",
                "artifact_sha256",
                "configuration_sha256",
                "identities_sha256",
            )
            if manifest.get(key) is not None
        },
        "dataset": {
            "session_count": len({str(record["sample_id"]) for record in records}),
            "turn_count": len(records),
            "scenario_counts": dict(sorted(Counter(
                str(record["scenario_type"])
                for record in records
                if int(record["turn"]) == 1
            ).items())),
            "holdout_opened": False,
        },
        "selection_rule": [
            "session_pool_recall_at_selected_depth",
            "recommended_technical_score",
            "hit_rate_at_10",
            "mrr",
            "turn_pool_recall",
            "closest_to_current_weights",
            "deterministic_weight_tuple",
        ],
        "pool_recall_tolerance": 0.0,
        "search": {
            "depth": depth,
            "coarse_step": 1.0 / coarse_units,
            "coarse_candidate_count": len(coarse),
            "coarse_winner": _search_summary(coarse_winner),
            "coarse_candidates": [_search_summary(candidate) for candidate in coarse],
            "local_step": 1.0 / local_units,
            "local_radius_per_weight": local_radius,
            "local_candidate_count": len(local),
            "local_winner": _search_summary(winner),
            "local_candidates": [_search_summary(candidate) for candidate in local],
        },
        "winner": winner,
        "winner_depth_comparison": {
            "depth_30": evaluate_weighted_policy(
                records, winner["weights"], depth=30
            ),
            f"depth_{depth}": winner,
            "full_union": evaluate_weighted_policy(
                records, winner["weights"], depth=MAX_TRAINING_POOL_DEPTH
            ),
            "full_union_size": {
                "minimum": min(full_union_sizes) if full_union_sizes else 0,
                "maximum": max(full_union_sizes) if full_union_sizes else 0,
                "mean": (
                    round(sum(full_union_sizes) / len(full_union_sizes), 6)
                    if full_union_sizes else 0.0
                ),
            },
        },
        "baselines": baselines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", default="benchmarks/fusion_training.jsonl"
    )
    parser.add_argument("--output", default="docs/fusion_policy_training.json")
    parser.add_argument(
        "--fused30-report", default="docs/reranker_benchmark.json"
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    arguments = parser.parse_args()
    records, manifest = load_artifact(arguments.artifact)
    report = train_fusion_policy(
        records,
        manifest,
        load_fused30_baseline(arguments.fused30_report),
        depth=arguments.depth,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "winner": report["winner"],
        "coarse_candidate_count": report["search"]["coarse_candidate_count"],
        "local_candidate_count": report["search"]["local_candidate_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
