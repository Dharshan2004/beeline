"""Validate and freeze the pool-aware Fusion Policy on development folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

from retrieval.fusion import (
    FROZEN_GLOBAL_WEIGHTS,
    FROZEN_POLICY_VERSION as ACTIVE_POLICY_VERSION,
    NORMALIZER_VERSION,
    ROUTE_NAMES,
)
from starter.planning import PLANNING_PROMPT_SHA256
from starter.replacement_evidence import (
    REPLACEMENT_EVIDENCE_SHA256,
    REPLACEMENT_EVIDENCE_VERSION,
)
from tools.build_fusion_dataset import load_artifact
from tools.train_fusion_policy import (
    CURRENT_WEIGHTS,
    DEFAULT_DEPTH,
    evaluate_weighted_policy,
    simplex_weights,
)


VALIDATION_VERSION = "fusion-policy-freeze-v2"
FROZEN_POLICY_VERSION = "pool-aware-global-v2"
FOLD_COUNT = 4
FOLD_SEED = 20260830
MAX_SCENARIO_HR_DROP = 0.05
MIN_GAP_CLOSURE = 0.25
RUNTIME_BUDGET_SECONDS = 900.0
RERANK_P95_BUDGET_SECONDS = 1.5


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario_stratified_folds(
    records: Sequence[dict],
    *,
    fold_count: int = FOLD_COUNT,
    seed: int = FOLD_SEED,
) -> list[list[dict]]:
    """Partition whole sessions into deterministic scenario-balanced folds."""
    if fold_count <= 1:
        raise ValueError("fold_count must be greater than one")
    sessions: dict[str, str] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        scenario = str(record["scenario_type"])
        if sample_id in sessions and sessions[sample_id] != scenario:
            raise ValueError(f"session {sample_id} changes scenario")
        sessions[sample_id] = scenario
    grouped: dict[str, list[str]] = {}
    for sample_id, scenario in sessions.items():
        grouped.setdefault(scenario, []).append(sample_id)
    fold_ids = [set() for _ in range(fold_count)]
    for scenario, sample_ids in sorted(grouped.items()):
        if len(sample_ids) < fold_count:
            raise ValueError(f"scenario {scenario} cannot represent every fold")
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(
                f"{seed}:{scenario}:{sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        for index, sample_id in enumerate(ordered):
            fold_ids[index % fold_count].add(sample_id)
    folds = [
        [record for record in records if str(record["sample_id"]) in selected]
        for selected in fold_ids
    ]
    if set().union(*fold_ids) != set(sessions) or sum(map(len, fold_ids)) != len(sessions):
        raise ValueError("fold assignment is incomplete or overlapping")
    return folds


def scenario_hit_guardrail(
    candidate: dict,
    baseline: dict,
    *,
    max_drop: float = MAX_SCENARIO_HR_DROP,
) -> dict:
    if set(candidate) != set(baseline):
        raise ValueError("candidate and baseline scenario sets differ")
    scenarios = {}
    for scenario in sorted(baseline):
        baseline_hr = float(baseline[scenario]["metrics"]["hit_rate_at_10"])
        candidate_hr = float(candidate[scenario]["metrics"]["hit_rate_at_10"])
        delta = round(candidate_hr - baseline_hr, 6)
        scenarios[scenario] = {
            "baseline_hit_rate_at_10": baseline_hr,
            "candidate_hit_rate_at_10": candidate_hr,
            "delta": delta,
            "passed": delta >= -max_drop - 1e-12,
        }
    return {
        "maximum_allowed_drop": max_drop,
        "passed": all(row["passed"] for row in scenarios.values()),
        "scenarios": scenarios,
    }


def plateau_neighbor_count(
    candidate: dict,
    candidates: Sequence[dict],
    *,
    step: float,
) -> int:
    """Count admissible points reachable by one local simplex-grid move."""
    weights = candidate["weights"]
    count = 0
    for other in candidates:
        differences = [
            abs(float(weights[name]) - float(other["weights"][name]))
            for name in ROUTE_NAMES
        ]
        if (
            any(difference > 1e-12 for difference in differences)
            and max(differences) <= step + 1e-12
            and sum(differences) <= 2 * step + 1e-12
        ):
            count += 1
    return count


def select_stable_candidate(candidates: Sequence[dict]) -> dict:
    if not candidates:
        raise ValueError("no Fusion Policy passes the validation gates")

    def key(candidate: dict) -> tuple:
        stability = candidate["fold_stability"]
        metrics = candidate["full"]["metrics"]
        return (
            candidate["plateau_neighbor_count"],
            stability["minimum_session_pool_recall"],
            -stability["session_pool_recall_pstdev"],
            stability["minimum_technical_score"],
            metrics["recommended_technical_score"],
            metrics["hit_rate_at_10"],
            metrics["mrr"],
            tuple(float(candidate["weights"][name]) for name in ROUTE_NAMES),
        )

    return max(candidates, key=key)


def _fold_stability(fold_results: Sequence[dict]) -> dict:
    recalls = [row["pool_recall"]["session_pool_recall"] for row in fold_results]
    scores = [row["metrics"]["recommended_technical_score"] for row in fold_results]
    hit_rates = [row["metrics"]["hit_rate_at_10"] for row in fold_results]
    return {
        "minimum_session_pool_recall": min(recalls),
        "mean_session_pool_recall": round(statistics.fmean(recalls), 6),
        "session_pool_recall_pstdev": round(statistics.pstdev(recalls), 6),
        "minimum_technical_score": min(scores),
        "mean_technical_score": round(statistics.fmean(scores), 6),
        "technical_score_pstdev": round(statistics.pstdev(scores), 6),
        "minimum_hit_rate_at_10": min(hit_rates),
    }


def _fold_manifest(folds: Sequence[Sequence[dict]]) -> list[dict]:
    output = []
    for index, fold in enumerate(folds, start=1):
        session_rows = {
            str(record["sample_id"]): str(record["scenario_type"])
            for record in fold
        }
        session_ids = sorted(session_rows)
        output.append({
            "fold": index,
            "session_count": len(session_ids),
            "turn_count": len(fold),
            "scenario_counts": dict(sorted(Counter(session_rows.values()).items())),
            "session_ids_sha256": hashlib.sha256(
                json.dumps(session_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })
    return output


def _candidate_report(row: dict) -> dict:
    return {
        "weights": row["weights"],
        "pool_recall": row["full"]["pool_recall"],
        "metrics": row["full"]["metrics"],
        "recall_to_hit_conversion": row["full"]["recall_to_hit_conversion"],
        "fold_stability": row["fold_stability"],
        "scenario_guardrail": row["scenario_guardrail"],
        "within_pool_recall_tolerance": row["within_pool_recall_tolerance"],
        "improves_official_objective": row["improves_official_objective"],
        "admissible": row["admissible"],
        "plateau_neighbor_count": row["plateau_neighbor_count"],
    }


def validated_local_weights(training_report: dict) -> list[dict[str, float]]:
    """Reject a truncated or altered Slice 10 local search grid."""
    search = training_report.get("search") or {}
    step = float(search.get("local_step", 0.0))
    radius = float(search.get("local_radius_per_weight", -1.0))
    coarse_weights = (search.get("coarse_winner") or {}).get("weights") or {}
    if step <= 0.0 or radius < 0.0 or set(coarse_weights) != set(ROUTE_NAMES):
        raise ValueError("training report has incomplete local-grid configuration")
    units = round(1.0 / step)
    if abs(1.0 / units - step) > 1e-12:
        raise ValueError("training report local step is not an exact simplex grid")
    expected = [
        weights for weights in simplex_weights(units)
        if all(
            abs(weights[name] - float(coarse_weights[name])) <= radius + 1e-12
            for name in ROUTE_NAMES
        )
    ]
    actual_rows = search.get("local_candidates") or []
    actual = [dict(row.get("weights") or {}) for row in actual_rows]
    expected_keys = {
        tuple(float(weights[name]) for name in ROUTE_NAMES) for weights in expected
    }
    actual_keys = {
        tuple(float(weights[name]) for name in ROUTE_NAMES)
        for weights in actual
        if set(weights) == set(ROUTE_NAMES)
    }
    if (
        len(actual) != len(actual_keys)
        or actual_keys != expected_keys
        or search.get("local_candidate_count") != len(expected)
    ):
        raise ValueError("training report local candidate grid is incomplete or stale")
    return expected


def validate_training_provenance(training_report: dict, artifact_manifest: dict) -> None:
    provenance = training_report.get("provenance") or {}
    keys = (
        "artifact_version",
        "split_version",
        "public_set_sha256",
        "artifact_sha256",
        "configuration_sha256",
        "identities_sha256",
    )
    if any(provenance.get(key) != artifact_manifest.get(key) for key in keys):
        raise ValueError("training report does not match the fusion artifact")


def validate_live_evidence(
    runtime_report: dict,
    selected: dict,
    artifact_manifest: dict,
    training_report: dict,
    frozen_configuration: dict,
) -> dict:
    """Prove the activated Agent matches cached quality and runtime evidence."""
    runtime = runtime_report.get("runtime") or {}
    reranker_runtime = runtime.get("reranker") or {}
    dense_runtime = runtime.get("dense_route") or {}
    projected_wall = float(runtime.get("projected_wall_seconds_200_sessions", 0.0))
    rerank_p95 = float(reranker_runtime.get("latency_p95_seconds", 0.0))
    source_reranker = artifact_manifest["identities"]["reranker"]
    live_configuration = runtime_report.get("configuration") or {}
    live_fusion = live_configuration.get(
        "fusion_and_retrieval", {}
    )
    live_metrics = runtime_report.get("metrics") or {}
    selected_metrics = selected["full"]["metrics"]
    pool_metrics = runtime_report.get("candidate_pool_metrics") or {}
    baseline = runtime_report.get("baseline") or {}
    baseline_fusion = baseline.get("configuration", {}).get(
        "fusion_and_retrieval", {}
    )
    expected_baseline = training_report["baselines"]["fused_30"]
    metric_keys = (
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
    )
    if (
        runtime_report.get("session_count") != 160
        or live_configuration.get("dense_index_and_model", {}).get("status")
        != "available"
        or live_configuration.get("dense_index_and_model", {}).get(
            "disabled_reason"
        ) is not None
        or dense_runtime.get("status") != "available"
        or dense_runtime.get("disabled_reason") is not None
        or not isinstance(dense_runtime.get("query_count"), int)
        or dense_runtime.get("query_count") <= 0
        or live_configuration.get("catalog") != frozen_configuration["catalog"]
        or live_configuration.get("dense_index_and_model", {}).get("manifest", {}).get(
            "embedding_model"
        ) != frozen_configuration["embedding_model"]
        or live_configuration.get("dense_index_and_model", {}).get("manifest", {}).get(
            "embedding_checksum"
        ) != frozen_configuration["embedding_checksum"]
        or live_configuration.get("dense_index_and_model", {}).get("manifest", {}).get(
            "catalog"
        ) != frozen_configuration["dense_catalog"]
        or live_configuration.get("dense_index_and_model", {}).get("manifest", {}).get(
            "vector_store"
        ) != frozen_configuration["dense_vector_store"]
        or live_configuration.get("planning") != frozen_configuration["planning"]
        or live_fusion.get("policy_version") != frozen_configuration["policy_version"]
        or live_fusion.get("weights") != frozen_configuration["weights"]
        or live_fusion.get("normalizer") != frozen_configuration["normalizer"]
        or live_fusion.get("route_depth") != frozen_configuration["route_depth"]
        or live_fusion.get("rerank_depth") != frozen_configuration["rerank_depth"]
        or live_fusion.get("fused_candidate_depth") != frozen_configuration["rerank_depth"]
        or live_fusion.get("rerank_deadline_seconds")
        != frozen_configuration["rerank_deadline_seconds"]
        or live_fusion.get("reranker_identity")
        != frozen_configuration["reranker"]["identity"]
        or live_fusion.get("reranker_revision")
        != frozen_configuration["reranker"]["revision"]
        or live_fusion.get("reranker_directory_sha256")
        != frozen_configuration["reranker"]["directory_sha256"]
        or live_metrics.get("sample_count") != selected_metrics.get("session_count")
        or any(live_metrics.get(key) != selected_metrics.get(key) for key in metric_keys)
        or pool_metrics.get("session_pool_recall")
        != selected["full"]["pool_recall"]["session_pool_recall"]
        or pool_metrics.get("post_rerank_hit_rate_at_10")
        != selected_metrics.get("hit_rate_at_10")
        or baseline_fusion.get("policy_version") != "fixed-hybrid-v1"
        or baseline_fusion.get("weights") != CURRENT_WEIGHTS
        or baseline_fusion.get("fused_candidate_depth") != 30
        or any(
            baseline.get("metrics", {}).get(key)
            != expected_baseline["metrics"].get(key)
            for key in metric_keys
        )
        or projected_wall <= 0.0
        or projected_wall > RUNTIME_BUDGET_SECONDS
        or rerank_p95 <= 0.0
        or rerank_p95 > RERANK_P95_BUDGET_SECONDS
        or reranker_runtime.get("depth") != DEFAULT_DEPTH
        or reranker_runtime.get("identity") != source_reranker.get("identity")
        or reranker_runtime.get("revision") != source_reranker.get("revision")
        or reranker_runtime.get("directory_sha256")
        != source_reranker.get("directory_sha256")
        or reranker_runtime.get("status") != "available"
        or reranker_runtime.get("failure_cause") is not None
        or reranker_runtime.get("attempt_count") != reranker_runtime.get("query_count")
        or not isinstance(reranker_runtime.get("query_count"), int)
        or reranker_runtime.get("query_count") <= 0
    ):
        raise ValueError("live activated policy does not match frozen evidence")
    for scenario, expected in selected["full"]["scenario_metrics"].items():
        actual = pool_metrics.get("scenario_metrics", {}).get(scenario, {})
        live_scenario = live_metrics.get("scenario_metrics", {}).get(scenario, {})
        if (
            actual.get("session_pool_recall")
            != expected["pool_recall"]["session_pool_recall"]
            or actual.get("post_rerank_hit_rate_at_10")
            != expected["metrics"]["hit_rate_at_10"]
            or actual.get("recall_to_hit_conversion")
            != expected["recall_to_hit_conversion"]
            or live_scenario.get("mrr") != expected["metrics"]["mrr"]
        ):
            raise ValueError("live scenario metrics do not match frozen evidence")
    return {
        "projected_wall_seconds_200_sessions": projected_wall,
        "rerank_latency_p95_seconds": rerank_p95,
        "rerank_query_count": reranker_runtime["query_count"],
        "turn_count": runtime.get("turn_latency", {}).get("turn_count"),
        "passed": True,
    }


def validate_and_freeze(
    records: Sequence[dict],
    artifact_manifest: dict,
    training_report: dict,
    runtime_report: dict,
    *,
    artifact_report_sha256: str,
    training_report_sha256: str,
    runtime_report_sha256: str,
) -> dict:
    """Apply all validation gates and assemble the immutable policy record."""
    validate_training_provenance(training_report, artifact_manifest)
    if training_report.get("dataset", {}).get("holdout_opened") is not False:
        raise ValueError("training report does not prove the holdout stayed locked")
    folds = scenario_stratified_folds(records)
    fold_manifest = _fold_manifest(folds)
    expected_fold_scenarios = {
        "boundary": 2,
        "browsing": 16,
        "buying": 16,
        "intent_override": 6,
    }
    if any(
        fold["session_count"] != 40
        or fold["scenario_counts"] != expected_fold_scenarios
        for fold in fold_manifest
    ):
        raise ValueError("development folds do not preserve scenario representation")

    baseline = evaluate_weighted_policy(records, CURRENT_WEIGHTS, depth=DEFAULT_DEPTH)
    baseline_score = baseline["metrics"]["recommended_technical_score"]
    weight_sets = validated_local_weights(training_report)

    evaluated = []
    for weights in weight_sets:
        full = evaluate_weighted_policy(records, weights, depth=DEFAULT_DEPTH)
        fold_results = [
            evaluate_weighted_policy(fold, weights, depth=DEFAULT_DEPTH)
            for fold in folds
        ]
        guardrail = scenario_hit_guardrail(
            full["scenario_metrics"], baseline["scenario_metrics"]
        )
        evaluated.append({
            "weights": weights,
            "full": full,
            "folds": fold_results,
            "fold_stability": _fold_stability(fold_results),
            "scenario_guardrail": guardrail,
        })

    best_recall = max(
        row["full"]["pool_recall"]["session_pool_recall"] for row in evaluated
    )
    tolerance = float(training_report.get("pool_recall_tolerance", 0.0))
    for row in evaluated:
        row["within_pool_recall_tolerance"] = (
            row["full"]["pool_recall"]["session_pool_recall"]
            >= best_recall - tolerance - 1e-12
        )
        row["improves_official_objective"] = (
            row["full"]["metrics"]["recommended_technical_score"] > baseline_score
        )
        row["admissible"] = (
            row["within_pool_recall_tolerance"]
            and row["improves_official_objective"]
            and row["scenario_guardrail"]["passed"]
        )
    admissible = [row for row in evaluated if row["admissible"]]
    local_step = float(training_report["search"]["local_step"])
    for row in evaluated:
        row["plateau_neighbor_count"] = (
            plateau_neighbor_count(row, admissible, step=local_step)
            if row["admissible"] else 0
        )
    selected = select_stable_candidate(admissible)
    if selected["plateau_neighbor_count"] < 1:
        raise ValueError("selected optimum is isolated rather than a stable plateau")
    if (
        selected["weights"] != FROZEN_GLOBAL_WEIGHTS
        or ACTIVE_POLICY_VERSION != FROZEN_POLICY_VERSION
    ):
        raise ValueError("live default Fusion Policy does not match validation selection")

    fused30_recall = float(
        training_report["baselines"]["fused_30"]["pool_recall"]["session_pool_recall"]
    )
    current_recall = baseline["pool_recall"]["session_pool_recall"]
    union_recall = float(
        training_report["winner_depth_comparison"]["full_union"]["pool_recall"]
        ["session_pool_recall"]
    )
    selected_recall = selected["full"]["pool_recall"]["session_pool_recall"]

    def closure(start: float) -> float:
        gap = union_recall - start
        return round((selected_recall - start) / gap, 6) if gap > 0 else 1.0

    gap_closure = {
        "fused_30_pool_recall": fused30_recall,
        "current_depth_50_pool_recall": current_recall,
        "selected_depth_50_pool_recall": selected_recall,
        "full_union_pool_recall": union_recall,
        "fused_30_to_union_gap_closed": closure(fused30_recall),
        "current_depth_50_to_union_gap_closed": closure(current_recall),
        "minimum_required_fraction": MIN_GAP_CLOSURE,
    }
    if gap_closure["fused_30_to_union_gap_closed"] < MIN_GAP_CLOSURE:
        raise ValueError("selected policy does not materially close the pool-recall gap")

    source_configuration = artifact_manifest["configuration"]
    source_fusion = source_configuration["fusion_and_retrieval"]
    source_reranker = artifact_manifest["identities"]["reranker"]
    dense_manifest = artifact_manifest["identities"]["dense_index_and_model"]["manifest"]
    intent_override = selected["full"]["scenario_metrics"]["intent_override"]
    frozen_configuration = {
        "policy_version": FROZEN_POLICY_VERSION,
        "weights": selected["weights"],
        "normalizer": NORMALIZER_VERSION,
        "route_depth": source_fusion["route_depth"],
        "rerank_depth": DEFAULT_DEPTH,
        "rerank_deadline_seconds": source_fusion["rerank_deadline_seconds"],
        "reranker": source_reranker,
        "dense_catalog": dense_manifest["catalog"],
        "embedding_checksum": dense_manifest["embedding_checksum"],
        "embedding_model": dense_manifest["embedding_model"],
        "dense_vector_store": dense_manifest["vector_store"],
        "catalog": source_configuration["catalog"],
        "planning": {
            **source_configuration["planning"],
            "prompt_sha256": PLANNING_PROMPT_SHA256,
            "replacement_evidence_version": REPLACEMENT_EVIDENCE_VERSION,
            "replacement_evidence_sha256": REPLACEMENT_EVIDENCE_SHA256,
        },
    }
    runtime_evidence = validate_live_evidence(
        runtime_report,
        selected,
        artifact_manifest,
        training_report,
        frozen_configuration,
    )
    return {
        "validation_version": VALIDATION_VERSION,
        "decision": "freeze_and_activate",
        "activation_verified": True,
        "dataset": {
            "session_count": 160,
            "turn_count": len(records),
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "folds": fold_manifest,
            "holdout_session_count": 40,
            "holdout_opened": False,
        },
        "gates": {
            "pool_recall_tolerance": tolerance,
            "maximum_scenario_hit_rate_drop": MAX_SCENARIO_HR_DROP,
            "minimum_gap_closure": MIN_GAP_CLOSURE,
            "runtime_budget_seconds_200_sessions": RUNTIME_BUDGET_SECONDS,
            "rerank_p95_budget_seconds": RERANK_P95_BUDGET_SECONDS,
        },
        "baseline": baseline,
        "gap_closure": gap_closure,
        "candidate_count": len(evaluated),
        "admissible_candidate_count": len(admissible),
        "candidates": [_candidate_report(row) for row in evaluated],
        "selected": selected,
        "intent_override": {
            **intent_override,
            "recall_to_hit_gap": round(
                intent_override["pool_recall"]["session_pool_recall"]
                - intent_override["metrics"]["hit_rate_at_10"],
                6,
            ),
            "recall_to_hit_conversion": round(
                intent_override["metrics"]["hit_rate_at_10"]
                / intent_override["pool_recall"]["session_pool_recall"],
                6,
            ) if intent_override["pool_recall"]["session_pool_recall"] else 0.0,
        },
        "runtime_evidence": runtime_evidence,
        "frozen_configuration": frozen_configuration,
        "provenance": {
            "artifact_sha256": artifact_manifest["artifact_sha256"],
            "artifact_configuration_sha256": artifact_manifest["configuration_sha256"],
            "artifact_identities_sha256": artifact_manifest["identities_sha256"],
            "artifact_report_sha256": artifact_report_sha256,
            "training_report_sha256": training_report_sha256,
            "runtime_report_sha256": runtime_report_sha256,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", default="benchmarks/fusion_training.jsonl"
    )
    parser.add_argument(
        "--artifact-report", default="docs/fusion_training_dataset.json"
    )
    parser.add_argument(
        "--training-report", default="docs/fusion_policy_training.json"
    )
    parser.add_argument(
        "--runtime-report", default="docs/fusion_policy_live_evaluation.json"
    )
    parser.add_argument("--output", default="docs/fusion_policy_freeze.json")
    arguments = parser.parse_args()
    records, artifact_manifest = load_artifact(arguments.artifact)
    artifact_report = json.loads(
        Path(arguments.artifact_report).read_text(encoding="utf-8")
    )
    if artifact_report.get("artifact_sha256") != artifact_manifest.get("artifact_sha256"):
        raise ValueError("checked-in artifact report does not match local artifact")
    training_report = json.loads(
        Path(arguments.training_report).read_text(encoding="utf-8")
    )
    runtime_report = json.loads(
        Path(arguments.runtime_report).read_text(encoding="utf-8")
    )
    report = validate_and_freeze(
        records,
        artifact_manifest,
        training_report,
        runtime_report,
        artifact_report_sha256=_file_sha256(arguments.artifact_report),
        training_report_sha256=_file_sha256(arguments.training_report),
        runtime_report_sha256=_file_sha256(arguments.runtime_report),
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "decision": report["decision"],
        "selected_weights": report["selected"]["weights"],
        "plateau_neighbor_count": report["selected"]["plateau_neighbor_count"],
        "gap_closure": report["gap_closure"],
        "metrics": report["selected"]["full"]["metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
