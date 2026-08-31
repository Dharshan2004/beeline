"""Anti-overfitting robustness evaluation for the Shopping Agent.

Reports three separately scored conditions through the UNMODIFIED official
evaluator:

- ``exact``: the released public sessions as-is.
- ``paraphrase``: the same sessions, but every customer message is rewritten
  by a deterministic meaning-preserving paraphraser before the Agent sees it.
  A robust agent's score must not collapse under wording changes.
- ``novel``: freshly generated sessions whose target products never appear in
  the released public labels, exercising generalization to unseen targets.

This is development tooling only. The paraphrase rules intentionally know the
simulator's message wording so they can perturb it; production Agent code must
never contain such knowledge, and the adversarial review checks that it does
not. The evaluator itself is imported unchanged and never edited.

Usage:
    .venv/bin/python -m tools.robustness_eval --output benchmarks/robustness.json
    .venv/bin/python -m tools.robustness_eval --conditions exact,paraphrase \
        --sessions 60 --output benchmarks/robustness_60.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
    load_openai_evaluation_settings,
    _load_dotenv,
)
from starter.agent import Agent
from tools.dataset_split import stratified_subset

NOVEL_SESSION_SEED = 20260901
# The official public scenario mix: 40% buying, 40% browsing, 15% intent
# override, 5% boundary.
NOVEL_SCENARIO_CYCLE = (
    "buying", "browsing", "buying", "browsing", "intent_override",
    "buying", "browsing", "buying", "browsing", "intent_override",
    "buying", "browsing", "buying", "browsing", "intent_override",
    "buying", "browsing", "buying", "browsing", "boundary",
)

# Deterministic, meaning-preserving rewrites of the simulated customer's
# phrasing. Applied in order; each (pattern, replacement) keeps the factual
# payload (category, requirements, disclosed values) byte-identical while
# changing every scaffold word around it.
PARAPHRASE_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^I'm looking for (.+), but I'm still exploring\.$"),
     r"just browsing around for \1 at the moment, nothing fixed yet"),
    (re.compile(r"^I'm looking for (.+)\. A key requirement is: (.+)\.$"),
     r"i want to buy \1 -- one thing i definitely need: \2"),
    (re.compile(r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$"),
     r"scratch what i said before... these days what i really need is \1"),
    (re.compile(r"^For that, what matters is: (.+)\.$"),
     r"hmm, i mostly care about \1 there"),
    (re.compile(r"^I don't have a preference for (.+); please use your judgment\.$"),
     r"no strong feelings about \1, you pick"),
    (re.compile(r"^I don't have an additional preference for (.+)\.$"),
     r"nothing else comes to mind about \1"),
    (re.compile(r"^I'm looking for (.+)\. (.+)$"),
     r"i could use \1. \2"),
    (re.compile(
        r"^Those options are not quite right yet\. "
        r"Ask me about one specific attribute\.$"),
     r"not quite what i had in mind, feel free to ask me something specific"),
)


def paraphrase(message: str) -> str:
    for pattern, replacement in PARAPHRASE_RULES:
        rewritten, count = pattern.subn(replacement, message)
        if count:
            # Secondary perturbation: change list separators too.
            return rewritten.replace("; ", " and ")
    return message


class ParaphrasingAgentProxy:
    """Delegates to the real Agent after rewriting each customer message."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._agent.respond(session_id, paraphrase(user_message), turn, top_k)

    def get_planning_history(self, session_id: str) -> list[dict]:
        return self._agent.get_planning_history(session_id)


def build_novel_sessions(
    catalog_ids: set[str],
    public_samples: list[dict],
    count: int,
) -> list[dict]:
    """Generate sessions whose targets never appear in the public labels."""
    public_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in public_samples
    }
    candidates = sorted(catalog_ids - public_targets)
    rng = random.Random(NOVEL_SESSION_SEED)
    targets = rng.sample(candidates, count)
    profiles = [sample["user_profile"] for sample in public_samples]
    sessions = []
    for index, target in enumerate(targets):
        sessions.append({
            "sample_id": f"novel_{index + 1:04d}",
            "scenario_type": NOVEL_SCENARIO_CYCLE[index % len(NOVEL_SCENARIO_CYCLE)],
            "user_profile": rng.choice(profiles),
            "ground_truth": {"parent_asin": target},
        })
    return sessions


def strip_sessions(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "sessions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="benchmarks/robustness.json")
    parser.add_argument(
        "--conditions",
        default="exact,paraphrase,novel",
        help="Comma-separated subset of exact,paraphrase,novel.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=None,
        help="Deterministic stratified public subset size (default: all 200).",
    )
    parser.add_argument(
        "--novel-count",
        type=int,
        default=100,
        help="Number of generated novel-target sessions.",
    )
    parser.add_argument(
        "--openai-ranker-role",
        default=None,
        help=(
            "Enable the LLM semantic-ranking stage with this configured role "
            "from --ranker-config (sends messages and catalog snippets to "
            "OpenAI with store=false)."
        ),
    )
    parser.add_argument("--ranker-config", default="config/semantic_ranker.json")
    parser.add_argument("--ranker-max-candidates", type=int, default=12)
    parser.add_argument(
        "--tournament",
        action="store_true",
        help="Use the parallel nano chunk tournament as the ranking stage.",
    )
    parser.add_argument(
        "--chunk-timeout",
        type=float,
        default=1.2,
        help="Tournament chunk-ranker timeout in seconds.",
    )
    parser.add_argument(
        "--final-timeout",
        type=float,
        default=1.6,
        help="Tournament final-ranker timeout in seconds.",
    )
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    def build_agent() -> Agent:
        if args.openai_ranker_role is None:
            return Agent(args.catalog)
        from starter.llm_ranker import OpenAISemanticRanker
        from starter.openai_planning import DevelopmentBudget

        _load_dotenv(args.env_file)
        settings = load_openai_evaluation_settings(
            args.ranker_config, args.openai_ranker_role
        )
        def build_stage(stage_settings, timeout_seconds, max_candidates):
            return OpenAISemanticRanker(
                model=stage_settings.model,
                pricing=stage_settings.pricing,
                budget=DevelopmentBudget(
                    limit_usd=stage_settings.budget_limit_usd,
                    warning_usd=stage_settings.warning_usd,
                    review_boundary_usd=stage_settings.review_boundary_usd,
                    absolute_stop_usd=stage_settings.absolute_stop_usd,
                ),
                reasoning_effort=stage_settings.reasoning_effort,
                timeout_seconds=timeout_seconds,
                max_output_tokens=stage_settings.max_output_tokens,
                max_candidates=max_candidates,
            )

        if args.tournament:
            from starter.llm_ranker import TournamentSemanticRanker

            ranker = TournamentSemanticRanker(
                build_stage(settings, args.chunk_timeout, 12),
                build_stage(settings, args.final_timeout, 12),
            )
        else:
            ranker = build_stage(
                settings, settings.timeout_seconds, args.ranker_max_candidates
            )
        return Agent(args.catalog, semantic_ranker=ranker)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]

    catalog_ids, categories, products = catalog_index(args.catalog)
    public_samples = load_jsonl(args.dataset)
    samples = (
        stratified_subset(public_samples, args.sessions, seed=20260831)
        if args.sessions
        else public_samples
    )

    report: dict = {
        "conditions": {},
        "llm_ranker_role": args.openai_ranker_role,
    }
    for condition in conditions:
        agent = build_agent()
        try:
            if condition == "exact":
                result = evaluate(agent, samples, catalog_ids, categories, products)
            elif condition == "paraphrase":
                result = evaluate(
                    ParaphrasingAgentProxy(agent),
                    samples,
                    catalog_ids,
                    categories,
                    products,
                )
            elif condition == "novel":
                novel_sessions = build_novel_sessions(
                    catalog_ids, public_samples, args.novel_count
                )
                result = evaluate(
                    agent, novel_sessions, catalog_ids, categories, products
                )
            else:
                raise ValueError(f"unknown condition: {condition}")
        finally:
            agent.close()
        result.pop("runtime_configuration", None)
        ranker = getattr(agent, "semantic_ranker", None)
        if ranker is not None:
            result["semantic_ranker_metrics"] = ranker.metrics()
        report["conditions"][condition] = strip_sessions(result)
        print(condition, json.dumps({
            key: report["conditions"][condition].get(key)
            for key in (
                "sample_count", "hit_rate_at_10", "mrr", "mttc",
                "efficiency", "recommended_technical_score",
            )
        }))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
