"""Development-only Candidate Pool depth experiment through the official scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate
from starter.agent import Agent
from tools.dataset_split import load_frozen_development_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog_ids, categories, products = catalog_index(args.catalog)
    samples = load_frozen_development_samples(args.dataset)
    agent = Agent(args.catalog, candidate_pool_depth=args.depth)
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.close()
    result.pop("runtime_configuration", None)
    result["experiment"] = {"candidate_pool_depth": args.depth}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "depth": args.depth,
        **{key: result[key] for key in (
            "hit_rate_at_10", "mrr", "mttc", "efficiency",
            "recommended_technical_score",
        )},
        "p95_turn_latency": result["turn_latency"]["p95_seconds"],
    }, indent=1))


if __name__ == "__main__":
    main()
