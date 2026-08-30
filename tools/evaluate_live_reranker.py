"""Evaluate the live frozen reranker on development sessions only.

This command exercises the official Agent/evaluator seam while refusing to
deserialize the reserved rows. It records candidate reachability separately
from the reranker's conversion of reachable Target Products into top-ten hits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, materialize_hidden_fields
from retrieval.dense_route import DenseRetrievalRoute
from retrieval.reranker import FROZEN_RERANK_DEPTH, UnavailableReranker
from starter.agent import Agent
from tools.dataset_split import load_frozen_development_samples, stratified_subset


SUBSET_SEED = 20260830


def _group_summary(records: list[dict]) -> dict:
    sample_count = len(records)
    reached = sum(int(record["pool_reached"]) for record in records)
    hits = sum(int(record["hit"]) for record in records)
    return {
        "sample_count": sample_count,
        "session_pool_recall": round(reached / sample_count, 6) if sample_count else 0.0,
        "post_rerank_hit_rate_at_10": round(hits / sample_count, 6) if sample_count else 0.0,
        "recall_to_hit_conversion": round(hits / reached, 6) if reached else 0.0,
    }


def pool_conversion_metrics(
    samples: list[dict],
    sessions: list[dict],
    traces: dict[str, list[dict]],
    products: dict[str, dict],
    *,
    depth: int = FROZEN_RERANK_DEPTH,
) -> dict:
    """Summarize eligible fused-pool reachability and live top-ten conversion."""
    session_ids = list(traces)
    if len(samples) != len(sessions) or len(samples) != len(session_ids):
        raise ValueError("samples, evaluator sessions, and Agent traces must align")
    records: list[dict] = []
    for sample, session_id, session in zip(
        samples,
        session_ids,
        sessions,
        strict=True,
    ):
        if str(sample["sample_id"]) != str(session["sample_id"]):
            raise ValueError("evaluator session order does not match development samples")
        target = str(sample["ground_truth"]["parent_asin"])
        _intent_card, behavior = materialize_hidden_fields(sample, products)
        override_turn = int((behavior.get("override") or {}).get("turn", 1))
        eligible_pools = [
            trace["pools"][str(depth)]
            for trace in traces[session_id]
            if (
                sample["scenario_type"] != "intent_override"
                or int(trace["turn"]) >= override_turn
            )
        ]
        pool_reached = any(target in pool for pool in eligible_pools)
        hit = bool(session["hit"])
        if hit and not pool_reached:
            raise ValueError("a live reranker hit escaped its eligible fused Candidate Pool")
        records.append({
            "scenario_type": str(sample["scenario_type"]),
            "pool_reached": pool_reached,
            "hit": hit,
        })

    scenarios = sorted({record["scenario_type"] for record in records})
    return {
        "depth": depth,
        **_group_summary(records),
        "scenario_metrics": {
            scenario: _group_summary([
                record for record in records
                if record["scenario_type"] == scenario
            ])
            for scenario in scenarios
        },
    }


def _scored_metrics(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"sessions", "turn_latency"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/live_reranker_evaluation.json")
    parser.add_argument(
        "--sessions",
        type=int,
        default=None,
        help="optional deterministic development subset for smoke testing",
    )
    arguments = parser.parse_args()

    samples = load_frozen_development_samples(arguments.dataset)
    if arguments.sessions is not None:
        samples = stratified_subset(samples, arguments.sessions, seed=SUBSET_SEED)
    catalog_ids, categories, products = catalog_index(arguments.catalog)
    dense_route = DenseRetrievalRoute(arguments.catalog)

    baseline_agent = Agent(
        arguments.catalog,
        dense_route=dense_route,
        reranker=UnavailableReranker("fused_30_live_baseline"),
        candidate_pool_depth=30,
    )
    baseline_started = time.perf_counter()
    try:
        baseline_result = evaluate(
            baseline_agent,
            samples,
            catalog_ids,
            categories,
            products,
        )
        baseline_wall_seconds = time.perf_counter() - baseline_started
        baseline_configuration = baseline_agent.get_runtime_configuration()
    finally:
        baseline_agent.reranker.close()

    agent = Agent(
        arguments.catalog,
        dense_route=dense_route,
        trace_pool_depths=(FROZEN_RERANK_DEPTH,),
    )
    started = time.perf_counter()
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
        wall_seconds = time.perf_counter() - started
        reranker_metrics = agent.get_reranker_metrics()
        pool_metrics = pool_conversion_metrics(
            samples,
            result["sessions"],
            agent.get_candidate_traces(),
            products,
        )
        output = {
            "evaluation": "live-reranker-development-v1",
            "session_count": len(samples),
            "metrics": _scored_metrics(result),
            "candidate_pool_metrics": pool_metrics,
            "runtime": {
                "wall_seconds": round(wall_seconds, 3),
                "projected_wall_seconds_200_sessions": round(
                    wall_seconds * 200 / len(samples),
                    3,
                ),
                "turn_latency": result["turn_latency"],
                "reranker": reranker_metrics,
                "dense_route": agent.get_dense_route_metrics(),
            },
            "configuration": agent.get_runtime_configuration(),
            "baseline": {
                "identity": "fused-30 no-reranker live baseline",
                "metrics": _scored_metrics(baseline_result),
                "runtime": {
                    "wall_seconds": round(baseline_wall_seconds, 3),
                    "projected_wall_seconds_200_sessions": round(
                        baseline_wall_seconds * 200 / len(samples),
                        3,
                    ),
                    "turn_latency": baseline_result["turn_latency"],
                },
                "configuration": baseline_configuration,
            },
        }
    finally:
        agent.close()

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
