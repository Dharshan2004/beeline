"""Compare cross-encoders and deep Candidate Pool depths on one cached replay.

Slice 07 is a decision gate: it selects the offline reranker and the deepest
base-route union that still fits the official evaluator's runtime limits. That
decision is only trustworthy if every model and depth sees exactly the same
Candidate Pools, so the tool runs in stages.

    # 1. Replay the shipped fused-30 trajectory once, recording the deep pool
    #    and the relevance label for every scored turn.
    python -m tools.benchmark_reranker cache --output benchmarks/rerank_cache.jsonl

    # 2. Score that cache with one model, in a fresh process so the peak-memory
    #    figure is comparable. Repeat per candidate model.
    python -m tools.benchmark_reranker score \
        --identity cross-encoder/ms-marco-MiniLM-L-2-v2 \
        --output benchmarks/rerank_MiniLM-L-2.json

    # 3. Merge the per-model results and the no-reranker baseline into one report.
    python -m tools.benchmark_reranker summarize benchmarks/rerank_*.json \
        --output benchmarks/rerank_summary.json

Every stage is CPU-only and network-disabled: the process pins torch to CPU and
sets the offline flags before any model is loaded, so a missing local model is a
hard failure rather than a silent download.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import time
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

# Enforced before torch or transformers can be imported by anything below.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
)
from retrieval.product_text import product_text
from retrieval.reranker import (
    DEFAULT_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    RERANKER_CANDIDATES,
    CrossEncoderReranker,
    order_by_scores,
)
from retrieval.resources import peak_rss_bytes
from starter.agent import Agent
from starter.retrieval import RERANK_TEXT_CHAR_LIMIT
from tools.dataset_split import (
    SPLIT_VERSION,
    development_samples,
    stratified_subset,
)


# Structured, BM25, and dense retrieval each admit at most 100 candidates.
MAX_POOL_DEPTH = 300
DEFAULT_DEPTHS = (30, 50, 100, 150, 200, 250, 300)
FUSED_BASELINE_DEPTH = 30
SUBSET_SEED = 20260830
TOP_K = 10


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _model_provenance(model_dir: Path, identity: str) -> dict:
    """Return the immutable fetch identity used to reproduce a local model."""
    fetched_path = model_dir / "FETCHED.json"
    if not fetched_path.is_file():
        raise RuntimeError(f"missing model provenance: {fetched_path}")
    fetched = json.loads(fetched_path.read_text(encoding="utf-8"))
    if fetched.get("identity") != identity or not fetched.get("revision"):
        raise RuntimeError(
            f"model provenance does not match {identity}: {fetched_path}"
        )
    return {
        "identity": identity,
        "revision": fetched["revision"],
        "model_dir": str(model_dir),
        "model_bytes": _directory_size_bytes(model_dir),
    }


def build_cache(arguments: argparse.Namespace) -> dict:
    """Replay the shipped configuration once and record every scored turn."""
    samples = development_samples(load_jsonl(arguments.dataset))
    if arguments.sessions:
        samples = stratified_subset(samples, arguments.sessions, seed=SUBSET_SEED)
    catalog_ids, categories, products = catalog_index(arguments.catalog)
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("network is disabled during reranker benchmarking"),
    ):
        agent = Agent(arguments.catalog, trace_pool_depths=DEFAULT_DEPTHS)
        dense_metrics = agent.get_dense_route_metrics()
        if dense_metrics.get("status") != "available":
            raise RuntimeError(
                "Slice 7 requires the dense Retrieval Route; readiness was "
                f"{dense_metrics.get('status')!r}: "
                f"{dense_metrics.get('disabled_reason') or 'no reason reported'}"
            )

        started = time.perf_counter()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        wall_seconds = time.perf_counter() - started
        dense_metrics = agent.get_dense_route_metrics()

    traces = agent.get_candidate_traces()
    session_ids = list(traces)
    if len(session_ids) != len(result["sessions"]):
        raise RuntimeError(
            "trace and evaluator session counts disagree; the cache would "
            "attribute Candidate Pools to the wrong sessions"
        )

    records: list[dict] = []
    union_sizes: list[int] = []
    for session_id, sample, session in zip(
        session_ids, samples, result["sessions"], strict=True
    ):
        if str(session["sample_id"]) != str(sample["sample_id"]):
            raise RuntimeError("evaluator session order does not match the samples")
        target = str(sample["ground_truth"]["parent_asin"])
        _intent_card, behavior = materialize_hidden_fields(sample, products)
        override = behavior.get("override") or {}
        override_turn = int(override.get("turn", 1))
        for entry in traces[session_id]:
            pools = entry["pools"]
            if entry["response_pool"] != pools[str(FUSED_BASELINE_DEPTH)]:
                raise RuntimeError(
                    "the fused-30 response and cached depth-30 pool disagree"
                )
            union_sizes.append(len(pools[str(MAX_POOL_DEPTH)]))
            records.append({
                "sample_id": str(sample["sample_id"]),
                "scenario_type": str(sample["scenario_type"]),
                "turn": int(entry["turn"]),
                "query": entry["query"],
                "target": target,
                "hit_eligible": (
                    sample["scenario_type"] != "intent_override"
                    or int(entry["turn"]) >= override_turn
                ),
                "pools": pools,
                "response_pool": entry["response_pool"],
            })

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "stage": "cache",
        "split_version": SPLIT_VERSION,
        "session_count": len(samples),
        "turn_count": len(records),
        "max_pool_depth": MAX_POOL_DEPTH,
        "union_size_mean": round(statistics.fmean(union_sizes), 2) if union_sizes else 0,
        "union_size_median": statistics.median(union_sizes) if union_sizes else 0,
        "union_size_min": min(union_sizes) if union_sizes else 0,
        "union_size_max": max(union_sizes) if union_sizes else 0,
        "pool_recall_by_depth": {
            str(depth): pool_recall(records, depth)
            for depth in (*DEFAULT_DEPTHS, MAX_POOL_DEPTH)
        },
        "baseline_wall_seconds": round(wall_seconds, 2),
        "baseline_peak_rss_bytes": peak_rss_bytes(),
        "baseline_metrics": {
            key: value for key, value in result.items() if key != "sessions"
        },
        "retrieval_configuration": agent.get_retrieval_configuration(),
        "dense_route_metrics": dense_metrics,
        "cache_path": str(output),
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_cache(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def subset_records(records: Sequence[dict], sessions: int) -> list[dict]:
    """Keep a reproducible scenario-proportional subset of the cached sessions.

    Scoring every cached turn with every candidate model costs more CPU than the
    decision is worth. Subsetting whole sessions - never individual turns - keeps
    the replay stopping rule intact, and every model still sees exactly the same
    Candidate Pools because the subset is derived from the cache, not resampled.
    """
    seen: dict[str, dict] = {}
    for record in records:
        seen.setdefault(
            record["sample_id"],
            {
                "sample_id": record["sample_id"],
                "scenario_type": record["scenario_type"],
            },
        )
    selected = {
        sample["sample_id"]
        for sample in stratified_subset(list(seen.values()), sessions, seed=SUBSET_SEED)
    }
    return [record for record in records if record["sample_id"] in selected]


def session_metrics(
    records: Sequence[dict],
    ranked_top_k: dict[tuple[str, int], list[str]],
) -> dict:
    """Score a replay the way the evaluator scores a live run.

    The trajectory is frozen by the cache, so a session converts on the earliest
    cached turn whose reranked top ten contains the Target Product. That mirrors
    the evaluator's stopping rule for the trajectory that was actually replayed.
    """
    by_session: dict[str, list[dict]] = {}
    for record in records:
        by_session.setdefault(record["sample_id"], []).append(record)
    session_rows: list[dict] = []
    for sample_id, turns in by_session.items():
        best_rank: int | None = None
        first_hit_turn: int | None = None
        for record in sorted(turns, key=lambda item: item["turn"]):
            if not record.get("hit_eligible", True):
                continue
            ranked = ranked_top_k[(sample_id, record["turn"])]
            if record["target"] in ranked:
                best_rank = ranked.index(record["target"]) + 1
                first_hit_turn = int(record["turn"])
                break
        session_rows.append({
            "hit": best_rank is not None,
            "first_hit_turn": first_hit_turn,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    session_count = len(session_rows)
    hit_rate = (
        sum(int(row["hit"]) for row in session_rows) / session_count
        if session_count else 0.0
    )
    reciprocal_ranks = [row["reciprocal_rank"] for row in session_rows]
    mttc = (
        statistics.fmean(
            row["first_hit_turn"] if row["first_hit_turn"] is not None else 11
            for row in session_rows
        )
        if session_rows else None
    )
    efficiency = (
        max(0.0, min(1.0, (11.0 - mttc) / 10.0))
        if mttc is not None else 0.0
    )
    mrr = statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return {
        "session_count": session_count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": None if mttc is None else round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(
            0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency,
            6,
        ),
    }


def _pool_at_depth(record: dict, depth: int) -> list[str]:
    try:
        return list(record["pools"][str(depth)])
    except KeyError as error:
        raise ValueError(f"cache record has no exact pool for depth {depth}") from error


def pool_recall(records: Sequence[dict], depth: int) -> dict:
    """Session-level and turn-level probability that the pool contains the target."""
    eligible_records = [record for record in records if record.get("hit_eligible", True)]
    turn_hits = sum(
        record["target"] in _pool_at_depth(record, depth)
        for record in eligible_records
    )
    by_session: dict[str, bool] = {}
    for record in records:
        if not record.get("hit_eligible", True):
            continue
        found = record["target"] in _pool_at_depth(record, depth)
        by_session[record["sample_id"]] = by_session.get(record["sample_id"], False) or found
    session_count = len(by_session)
    return {
        "turn_pool_recall": (
            round(turn_hits / len(eligible_records), 6) if eligible_records else 0.0
        ),
        "session_pool_recall": (
            round(sum(by_session.values()) / session_count, 6) if session_count else 0.0
        ),
    }


def baseline_rows(records: Sequence[dict]) -> dict:
    """The shipped fused-30 configuration, with no reranking at all."""
    ranked = {
        (record["sample_id"], record["turn"]):
            _pool_at_depth(record, FUSED_BASELINE_DEPTH)[:TOP_K]
        for record in records
    }
    recall = pool_recall(records, FUSED_BASELINE_DEPTH)
    metrics = session_metrics(records, ranked)
    return {
        "identity": "none (fused-30 baseline)",
        "depth": FUSED_BASELINE_DEPTH,
        "reranked": False,
        **recall,
        **metrics,
        "recall_to_hit_conversion": (
            round(metrics["hit_rate_at_10"] / recall["session_pool_recall"], 6)
            if recall["session_pool_recall"] else 0.0
        ),
        "turn_latency_p50_ms": 0.0,
        "turn_latency_p95_ms": 0.0,
        "turn_latency_mean_ms": 0.0,
        "added_seconds_total": 0.0,
    }


def score_cache(arguments: argparse.Namespace) -> dict:
    """Score exact cached pools, reusing work after proving prefix extension."""
    depths = sorted(set(arguments.depths))
    if depths[-1] > MAX_POOL_DEPTH:
        raise ValueError(f"depths must not exceed {MAX_POOL_DEPTH}")
    records = load_cache(Path(arguments.cache))
    if arguments.sessions:
        records = subset_records(records, arguments.sessions)
    if not records:
        raise ValueError("the reranker cache contains no scored turns")
    model_dir = Path(arguments.model_dir or Path("models") / RERANKER_CANDIDATES[arguments.identity])

    documents: dict[str, str] = {}
    wanted = {
        parent_asin
        for record in records
        for parent_asin in _pool_at_depth(record, depths[-1])
    }
    with Path(arguments.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin in wanted and parent_asin not in documents:
                documents[parent_asin] = product_text(product)[:RERANK_TEXT_CHAR_LIMIT]

    reranker = CrossEncoderReranker(
        model_dir,
        identity=arguments.identity,
        batch_size=arguments.batch_size,
        max_sequence_length=arguments.max_sequence_length,
        torch_threads=arguments.torch_threads,
    )
    # One warm-up turn: the first forward pass pays a one-off allocation cost
    # that would otherwise land in the p95 figure.
    warmup_record = next(
        (record for record in records if _pool_at_depth(record, depths[-1])),
        None,
    )
    if warmup_record is not None:
        first_identifier = _pool_at_depth(warmup_record, depths[-1])[0]
        reranker.score(
            warmup_record["query"],
            [documents.get(first_identifier, "")],
        )

    ranked_by_depth: dict[int, dict[tuple[str, int], list[str]]] = {
        depth: {} for depth in depths
    }
    latency_by_depth: dict[int, list[float]] = {depth: [] for depth in depths}
    started_all = time.perf_counter()
    for index, record in enumerate(records, start=1):
        deepest_pool = _pool_at_depth(record, depths[-1])
        texts = [documents.get(parent_asin, "") for parent_asin in deepest_pool]
        scores: list[float] = []
        elapsed = 0.0
        for depth in depths:
            exact_pool = _pool_at_depth(record, depth)
            if deepest_pool[: len(exact_pool)] != exact_pool:
                raise RuntimeError(
                    f"exact depth-{depth} pool is not a prefix of depth-{depths[-1]}"
                )
            segment = texts[len(scores) : len(exact_pool)]
            if segment:
                started = time.perf_counter()
                scores.extend(reranker.score(record["query"], segment))
                elapsed += time.perf_counter() - started
            latency_by_depth[depth].append(elapsed * 1000)
            ranked_by_depth[depth][(record["sample_id"], record["turn"])] = order_by_scores(
                exact_pool, scores[: len(exact_pool)]
            )[:TOP_K]
        if arguments.progress and index % arguments.progress == 0:
            done = time.perf_counter() - started_all
            print(
                f"  {index}/{len(records)} turns  {done:.0f}s elapsed  "
                f"{done / index * (len(records) - index):.0f}s remaining",
                flush=True,
            )

    rows = []
    for depth in depths:
        recall = pool_recall(records, depth)
        metrics = session_metrics(records, ranked_by_depth[depth])
        latencies = latency_by_depth[depth]
        rows.append({
            "identity": arguments.identity,
            "depth": depth,
            "reranked": True,
            **recall,
            **metrics,
            "recall_to_hit_conversion": (
                round(metrics["hit_rate_at_10"] / recall["session_pool_recall"], 6)
                if recall["session_pool_recall"] else 0.0
            ),
            "turn_latency_p50_ms": round(statistics.median(latencies), 1),
            "turn_latency_p95_ms": round(_percentile(latencies, 0.95), 1),
            "turn_latency_mean_ms": round(statistics.fmean(latencies), 1),
            "added_seconds_total": round(sum(latencies) / 1000, 1),
        })

    provenance = _model_provenance(model_dir, arguments.identity)
    return {
        "stage": "score",
        "identity": arguments.identity,
        "model_dir": str(model_dir),
        "model_bytes": provenance["model_bytes"],
        "model_revision": provenance["revision"],
        "max_sequence_length": arguments.max_sequence_length,
        "batch_size": arguments.batch_size,
        "torch_threads": arguments.torch_threads,
        "rerank_text_char_limit": RERANK_TEXT_CHAR_LIMIT,
        "cache_path": str(arguments.cache),
        "turn_count": len(records),
        "device": "cpu",
        "network": "disabled",
        "peak_rss_bytes": peak_rss_bytes(),
        "scoring_wall_seconds": round(time.perf_counter() - started_all, 1),
        "rows": rows,
        "baseline_row": baseline_rows(records),
    }


def summarize(arguments: argparse.Namespace) -> dict:
    """Merge the per-model reports and apply the documented selection rule."""
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in arguments.reports
    ]
    models = []
    for report in sorted(reports, key=lambda item: item["identity"]):
        if "model_dir" not in report:
            # Synthetic unit-test reports need not create a model directory.
            models.append({
                "identity": report["identity"],
                "revision": report.get("model_revision"),
                "model_dir": None,
                "model_bytes": report["model_bytes"],
            })
            continue
        provenance = _model_provenance(Path(report["model_dir"]), report["identity"])
        if report.get("model_revision") not in (None, provenance["revision"]):
            raise RuntimeError(
                f"report revision does not match local provenance for {report['identity']}"
            )
        if report["model_bytes"] != provenance["model_bytes"]:
            raise RuntimeError(
                f"report size does not match local model for {report['identity']}"
            )
        models.append(provenance)
    metadata = json.loads(Path(arguments.cache_meta).read_text(encoding="utf-8"))
    cached_sessions = int(metadata["session_count"])
    cached_turns = int(metadata["turn_count"])
    baseline_seconds = float(metadata["baseline_wall_seconds"])

    # Everything is projected onto one full 200-session evaluator run so that a
    # subset benchmark and the shipped configuration are directly comparable.
    full_sessions = arguments.full_run_sessions
    scale = full_sessions / cached_sessions
    projected_baseline = baseline_seconds * scale
    projected_turns = cached_turns * scale

    def _row(row: dict, model_bytes: int, peak_rss: int | None) -> dict:
        added = row["turn_latency_mean_ms"] / 1000 * projected_turns
        projected = projected_baseline + added
        return {
            **row,
            "model_bytes": model_bytes,
            "peak_rss_bytes": peak_rss,
            "projected_added_seconds": round(added, 1),
            "projected_wall_seconds": round(projected, 1),
            "within_budget": (
                projected <= arguments.runtime_budget_seconds
                and row["turn_latency_p95_ms"] <= arguments.p95_budget_ms
            ),
        }

    rows = [_row(
        reports[0]["baseline_row"],
        0,
        metadata.get("baseline_peak_rss_bytes"),
    )]
    for report in sorted(reports, key=lambda item: item["identity"]):
        for row in report["rows"]:
            rows.append(_row(row, report["model_bytes"], report["peak_rss_bytes"]))

    baseline = rows[0]
    baseline_technical_score = baseline.get(
        "recommended_technical_score",
        0.50 * baseline["hit_rate_at_10"]
        + 0.30 * baseline["mrr"]
        + 0.20 * baseline.get("efficiency", 0.0),
    )
    feasible = [
        row for row in rows
        if (
            row["within_budget"]
            and row["reranked"]
            and row["hit_rate_at_10"] >= baseline["hit_rate_at_10"]
            and row.get(
                "recommended_technical_score",
                0.50 * row["hit_rate_at_10"]
                + 0.30 * row["mrr"]
                + 0.20 * row.get("efficiency", 0.0),
            ) > baseline_technical_score
        )
    ]
    # Preserve the deepest quality-improving pool that passes both runtime
    # gates. Later Fusion Policy work cannot recover candidates excluded here.
    winner = max(
        feasible,
        key=lambda row: (
            row["depth"],
            row.get("recommended_technical_score", 0.0),
            row["mrr"],
            -row["turn_latency_p95_ms"],
            -row["model_bytes"],
        ),
        default=None,
    )
    ceiling = max(row["session_pool_recall"] for row in rows)
    return {
        "stage": "summary",
        "runtime_budget_seconds": arguments.runtime_budget_seconds,
        "p95_budget_ms": arguments.p95_budget_ms,
        "full_run_sessions": full_sessions,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_threads": reports[0].get("torch_threads"),
        },
        "projected_baseline_seconds": round(projected_baseline, 1),
        "benchmark_session_count": rows[0]["session_count"],
        "cache_metadata": metadata,
        "cache_pool_recall": metadata.get("pool_recall_by_depth"),
        "models": models,
        "rows": rows,
        "selected": winner,
        "decision": "reranker_selected" if winner is not None else "no_reranker",
        "pool_recall_ceiling": ceiling,
        "pool_recall_lost_to_truncation": (
            None if winner is None
            else round(ceiling - winner["session_pool_recall"], 6)
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    cache = subparsers.add_parser("cache", help="record deep Candidate Pools once")
    cache.add_argument("--catalog", default="data/catalog.jsonl")
    cache.add_argument("--dataset", default="data/public_set.jsonl")
    cache.add_argument("--output", default="benchmarks/rerank_cache.jsonl")
    cache.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="reproducible scenario-proportional subset of the development split",
    )

    score = subparsers.add_parser("score", help="benchmark one model over all depths")
    score.add_argument("--identity", choices=sorted(RERANKER_CANDIDATES), required=True)
    score.add_argument("--model-dir", default=None)
    score.add_argument("--catalog", default="data/catalog.jsonl")
    score.add_argument("--cache", default="benchmarks/rerank_cache.jsonl")
    score.add_argument("--output", required=True)
    score.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    score.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    score.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    score.add_argument("--torch-threads", type=int, default=8)
    score.add_argument("--progress", type=int, default=25)
    score.add_argument(
        "--sessions",
        type=int,
        default=0,
        help="reproducible scenario-proportional subset of the cached sessions",
    )

    summary = subparsers.add_parser("summarize", help="merge per-model reports")
    summary.add_argument("reports", nargs="+")
    summary.add_argument("--cache-meta", default="benchmarks/rerank_cache.jsonl.meta.json")
    summary.add_argument("--output", required=True)
    summary.add_argument(
        "--full-run-sessions",
        type=int,
        default=200,
        help="session count the projections are expressed for",
    )
    summary.add_argument(
        "--runtime-budget-seconds",
        type=float,
        default=900.0,
        help="evaluator wall-clock limit a configuration must stay within",
    )
    summary.add_argument(
        "--p95-budget-ms",
        type=float,
        default=1500.0,
        help="maximum admissible p95 added rerank latency per turn",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.stage == "cache":
        report = build_cache(arguments)
    elif arguments.stage == "score":
        report = score_cache(arguments)
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        report = summarize(arguments)
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(
        {key: value for key, value in report.items() if key != "cache_metadata"},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
