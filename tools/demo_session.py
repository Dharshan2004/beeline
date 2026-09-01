"""Replay one public session as an annotated, demo-ready transcript.

Runs the real Agent against the unmodified official evaluator loop for a
single released session and prints each turn: the customer's message, the
agent's clarification choice with its information-value rationale, and the
top-10 recommendations with the Target Product's live rank.

Usage:
    .venv/bin/python -m tools.demo_session --sample public_0007
    .venv/bin/python -m tools.demo_session --sample public_0012 --openai-ranker-role fast_ranker
"""
from __future__ import annotations

import argparse
import json

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class DemoAgent(Agent):
    """Records clarification rationale and per-turn context for narration."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn_log: list[dict] = []

    def _next_ask_attribute(self, session_id, state, candidate_pool=()):
        pool = set(candidate_pool)
        splits = {}
        for attribute, values in self.retrieval.value_index.items():
            counts = [
                len(pool.intersection(members)) for members in values.values()
            ]
            present = [count for count in counts if count > 0]
            if len(present) >= 2:
                splits[attribute] = {
                    "distinct_values": len(present),
                    "min_eliminated": len(pool) - max(present),
                }
        chosen = super()._next_ask_attribute(session_id, state, candidate_pool)
        self.turn_log.append({
            "pool_size": len(pool),
            "splits": splits,
            "chosen": chosen,
        })
        return chosen

    def respond(self, session_id, user_message, turn, top_k):
        response = super().respond(session_id, user_message, turn, top_k)
        if self.turn_log:
            self.turn_log[-1]["user_message"] = user_message
            self.turn_log[-1]["turn"] = turn
            self.turn_log[-1]["response"] = response
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample", default="public_0007")
    parser.add_argument("--openai-ranker-role", default=None)
    parser.add_argument("--ranker-config", default="config/semantic_ranker.json")
    args = parser.parse_args()

    samples = [
        sample for sample in load_jsonl(args.dataset)
        if sample["sample_id"] == args.sample
    ]
    if not samples:
        raise SystemExit(f"unknown sample id: {args.sample}")
    catalog_ids, categories, products = catalog_index(args.catalog)
    target = str(samples[0]["ground_truth"]["parent_asin"])

    # "auto" mirrors what judges get from the bare evaluator command: the
    # shipped nano tournament when a key is available, plain offline otherwise.
    semantic_ranker = "auto"
    if args.openai_ranker_role is not None:
        from evaluator.local_evaluator import (
            load_openai_evaluation_settings,
            _load_dotenv,
        )
        from starter.llm_ranker import OpenAISemanticRanker
        from starter.openai_planning import DevelopmentBudget

        _load_dotenv(".env")
        settings = load_openai_evaluation_settings(
            args.ranker_config, args.openai_ranker_role
        )
        semantic_ranker = OpenAISemanticRanker(
            model=settings.model,
            pricing=settings.pricing,
            budget=DevelopmentBudget(
                limit_usd=settings.budget_limit_usd,
                warning_usd=settings.warning_usd,
                review_boundary_usd=settings.review_boundary_usd,
                absolute_stop_usd=settings.absolute_stop_usd,
            ),
            reasoning_effort=settings.reasoning_effort,
            timeout_seconds=settings.timeout_seconds,
            max_output_tokens=settings.max_output_tokens,
            max_candidates=12,
        )

    agent = DemoAgent(args.catalog, semantic_ranker=semantic_ranker)
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.close()

    session = result["sessions"][0]
    product = products[target]
    print(f"{BOLD}Session {args.sample}{RESET}  scenario: "
          f"{samples[0]['scenario_type']}")
    print(f"{DIM}Hidden target: {target} — "
          f"{str(product.get('title'))[:80]}{RESET}\n")
    for entry in agent.turn_log:
        if "turn" not in entry:
            continue
        print(f"{BOLD}Turn {entry['turn']}{RESET}")
        print(f"{CYAN}Customer:{RESET} {entry['user_message']}")
        response = entry["response"]
        identifiers = [
            item["parent_asin"] for item in response["recommendations"]
        ]
        rank = identifiers.index(target) + 1 if target in identifiers else None
        listing = ", ".join(
            (f"{GREEN}{BOLD}{identifier}◀{RESET}" if identifier == target
             else f"{DIM}{identifier}{RESET}")
            for identifier in identifiers[:10]
        )
        print(f"Agent recommends: [{listing}]")
        if rank:
            print(f"{GREEN}Target in top 10 at rank {rank}{RESET}")
        ask = entry.get("chosen")
        if ask:
            split = entry["splits"].get(ask)
            rationale = (
                f"splits {entry['pool_size']} candidates into "
                f"{split['distinct_values']} groups, guarantees eliminating "
                f">= {split['min_eliminated']}" if split else
                "fallback question order"
            )
            print(f"{YELLOW}Agent asks about '{ask}'{RESET} "
                  f"{DIM}({rationale}){RESET}")
        print()
    outcome = (
        f"{GREEN}{BOLD}Converted on turn {session['first_hit_turn']} at rank "
        f"{session['best_rank']}{RESET}"
        if session["hit"]
        else f"{YELLOW}No conversion within 10 turns{RESET}"
    )
    print(outcome)
    print(f"{DIM}Reported tokens: "
          f"{json.dumps(result['reported_token_usage'])}{RESET}")


if __name__ == "__main__":
    main()
