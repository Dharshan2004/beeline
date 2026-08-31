"""Development-only measurement of the LLM semantic-ranking stage.

Runs the unmodified official evaluator with the Agent's optional semantic
ranker enabled. Requires OPENAI_API_KEY (loaded from the ignored .env file)
and explicit awareness that development messages and public catalog snippets
are sent to OpenAI with store=false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_openai_evaluation_settings,
    _load_dotenv,
)
from starter.agent import Agent
from starter.llm_ranker import OpenAISemanticRanker
from starter.openai_planning import DevelopmentBudget
from tools.dataset_split import load_frozen_development_samples, stratified_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--config", default="config/semantic_ranker.json")
    parser.add_argument(
        "--role",
        default="fast_ranker",
        choices=("fast_ranker", "nano_ranker", "lower_cost", "quality_reference"),
    )
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--ranker-timeout",
        type=float,
        default=None,
        help="Override the ranker timeout from the config file.",
    )
    parser.add_argument(
        "--chunk-timeout",
        type=float,
        default=1.2,
        help="Tournament chunk-ranker timeout in seconds.",
    )
    parser.add_argument(
        "--openai-rewriter-role",
        default=None,
        help="Also enable the LLM query-rewriting stage with this role.",
    )
    parser.add_argument(
        "--rewriter-config", default="config/semantic_ranker_nano.json"
    )
    parser.add_argument(
        "--tournament",
        action="store_true",
        help=(
            "Parallel chunked tournament: nano ranks pool chunks "
            "concurrently, the configured role ranks the finalists."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _load_dotenv(args.env_file)
    settings = load_openai_evaluation_settings(args.config, args.role)
    ranker = OpenAISemanticRanker(
        model=settings.model,
        pricing=settings.pricing,
        budget=DevelopmentBudget(
            limit_usd=settings.budget_limit_usd,
            warning_usd=settings.warning_usd,
            review_boundary_usd=settings.review_boundary_usd,
            absolute_stop_usd=settings.absolute_stop_usd,
        ),
        reasoning_effort=settings.reasoning_effort,
        timeout_seconds=(
            args.ranker_timeout
            if args.ranker_timeout is not None
            else settings.timeout_seconds
        ),
        max_output_tokens=settings.max_output_tokens,
        max_candidates=args.max_candidates,
    )
    if args.tournament:
        from starter.llm_ranker import TournamentSemanticRanker

        nano_settings = load_openai_evaluation_settings(
            "config/semantic_ranker_nano.json", "nano_ranker"
        )
        chunk_ranker = OpenAISemanticRanker(
            model=nano_settings.model,
            pricing=nano_settings.pricing,
            budget=DevelopmentBudget(
                limit_usd=nano_settings.budget_limit_usd,
                warning_usd=nano_settings.warning_usd,
                review_boundary_usd=nano_settings.review_boundary_usd,
                absolute_stop_usd=nano_settings.absolute_stop_usd,
            ),
            reasoning_effort=nano_settings.reasoning_effort,
            timeout_seconds=args.chunk_timeout,
            max_output_tokens=nano_settings.max_output_tokens,
            max_candidates=12,
        )
        final_ranker = OpenAISemanticRanker(
            model=settings.model,
            pricing=settings.pricing,
            budget=DevelopmentBudget(
                limit_usd=settings.budget_limit_usd,
                warning_usd=settings.warning_usd,
                review_boundary_usd=settings.review_boundary_usd,
                absolute_stop_usd=settings.absolute_stop_usd,
            ),
            reasoning_effort=settings.reasoning_effort,
            timeout_seconds=(
                args.ranker_timeout if args.ranker_timeout is not None else 1.6
            ),
            max_output_tokens=settings.max_output_tokens,
            max_candidates=12,
        )
        ranker = TournamentSemanticRanker(chunk_ranker, final_ranker)
    rewriter = None
    if args.openai_rewriter_role is not None:
        from starter.llm_rewriter import OpenAIQueryRewriter

        rewriter_settings = load_openai_evaluation_settings(
            args.rewriter_config, args.openai_rewriter_role
        )
        rewriter = OpenAIQueryRewriter(
            model=rewriter_settings.model,
            pricing=rewriter_settings.pricing,
            budget=DevelopmentBudget(
                limit_usd=rewriter_settings.budget_limit_usd,
                warning_usd=rewriter_settings.warning_usd,
                review_boundary_usd=rewriter_settings.review_boundary_usd,
                absolute_stop_usd=rewriter_settings.absolute_stop_usd,
            ),
            reasoning_effort=rewriter_settings.reasoning_effort,
            timeout_seconds=1.2,
            api_key=None,
        )
    catalog_ids, categories, products = catalog_index(args.catalog)
    samples = load_frozen_development_samples(args.dataset)
    if args.sessions:
        samples = stratified_subset(samples, args.sessions, seed=20260831)
    agent = Agent(args.catalog, semantic_ranker=ranker, query_rewriter=rewriter)
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.close()
    result.pop("runtime_configuration", None)
    result["semantic_ranker"] = {
        "configuration": ranker.configuration(),
        "metrics": ranker.metrics(),
    }
    if rewriter is not None:
        result["query_rewriter"] = {
            "configuration": rewriter.configuration(),
            "metrics": rewriter.metrics(),
        }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        **{key: result[key] for key in (
            "sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency",
            "recommended_technical_score",
        )},
        "tokens": result["reported_token_usage"],
        "ranker": ranker.metrics(),
        "p95_turn_latency": result["turn_latency"]["p95_seconds"],
    }, indent=1))


if __name__ == "__main__":
    main()
