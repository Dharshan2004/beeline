from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from retrieval.fusion import POLICY_NAMES
from retrieval.reranker import UnavailableReranker
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed hybrid, RRF, or single Retrieval Route policy",
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--policy", choices=POLICY_NAMES, default="fixed")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(
        args.catalog,
        fusion_policy=args.policy,
        reranker=UnavailableReranker("transparent_retrieval_baseline"),
        semantic_ranker=None,
        candidate_pool_depth=30,
    )
    result = evaluate(
        agent,
        samples,
        catalog_ids,
        categories,
        products,
    )
    agent.close()
    result["retrieval_policy"] = args.policy
    result["retrieval_configuration"] = agent.get_retrieval_configuration()
    Path(args.output).write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {key: value for key, value in result.items() if key != "sessions"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
