"""Team-side evaluation harness around the pristine official evaluator.

The competition rules disallow modifying evaluator files, so every extension
the team needs for development lives here instead: connected-model settings
loading, explicit evaluation-scope guards, per-turn latency measurement, and
execution diagnostics. The official ``evaluator/local_evaluator.py`` is
byte-identical to the published original (verify:
``git diff 2a6cc8e HEAD -- evaluator/`` is empty) and is imported unchanged —
scoring, the customer simulator, and labels are never touched.

Usage (equivalent to the previously extended evaluator CLI):
    .venv/bin/python -m tools.evaluation_harness --development-only \
        --openai-role lower_cost --output benchmarks/connected.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    evaluate,
    load_jsonl,
)
from starter.agent import Agent
from starter.openai_planning import (
    DevelopmentBudget,
    ModelPricing,
    OpenAIPlanningProvider,
)
from tools.dataset_split import (
    FROZEN_PUBLIC_SET_SHA256,
    load_frozen_development_samples,
    stratified_subset,
)

EVALUATION_SUBSET_SEED = 20260831


@dataclass(frozen=True)
class OpenAIEvaluationSettings:
    model: str
    pricing: ModelPricing
    timeout_seconds: float
    max_input_tokens_estimate: int
    max_output_tokens: int
    reasoning_effort: str
    budget_limit_usd: float
    warning_usd: float
    review_boundary_usd: float
    absolute_stop_usd: float


def load_openai_evaluation_settings(
    config_path: str | Path,
    role: str,
) -> OpenAIEvaluationSettings:
    """Load one connected model from the versioned benchmark configuration."""
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = next(
        (item for item in configuration.get("models", ()) if item.get("role") == role),
        None,
    )
    if model is None:
        raise ValueError(f"OpenAI model role {role!r} is not configured")
    prices = model["pricing_usd_per_million_tokens"]
    api = configuration["api"]
    budget = configuration["budget_usd"]
    return OpenAIEvaluationSettings(
        model=str(model["model"]),
        pricing=ModelPricing(
            input_per_million_usd=float(prices["input"]),
            output_per_million_usd=float(prices["output"]),
        ),
        timeout_seconds=float(api["timeout_seconds"]),
        max_input_tokens_estimate=int(api["max_input_tokens_estimate"]),
        max_output_tokens=int(api["max_output_tokens"]),
        reasoning_effort=str(api["reasoning_effort"]),
        budget_limit_usd=float(budget["phase_a_limit"]),
        warning_usd=float(budget["warning"]),
        review_boundary_usd=float(budget["review_boundary"]),
        absolute_stop_usd=float(budget["absolute_stop"]),
    )


def build_evaluation_agent(
    catalog_path: str | Path,
    openai_settings: OpenAIEvaluationSettings | None = None,
) -> Agent:
    """Build the official Agent in offline or explicitly connected mode."""
    provider = None
    if openai_settings is not None:
        provider = OpenAIPlanningProvider(
            model=openai_settings.model,
            timeout_seconds=openai_settings.timeout_seconds,
            max_output_tokens=openai_settings.max_output_tokens,
            max_input_tokens_estimate=openai_settings.max_input_tokens_estimate,
            reasoning_effort=openai_settings.reasoning_effort,
            pricing=openai_settings.pricing,
            budget=DevelopmentBudget(
                limit_usd=openai_settings.budget_limit_usd,
                warning_usd=openai_settings.warning_usd,
                review_boundary_usd=openai_settings.review_boundary_usd,
                absolute_stop_usd=openai_settings.absolute_stop_usd,
            ),
        )
    return Agent(catalog_path, planning_provider=provider)


def load_evaluation_samples(
    dataset_path: str | Path,
    *,
    development_only: bool,
    full_exposed_public_set: bool = False,
    session_count: int | None,
) -> list[dict]:
    """Load the requested development or explicitly exposed public scope."""
    if full_exposed_public_set:
        if session_count is not None:
            raise ValueError("full public-set evaluation requires all 200 sessions")
        path = Path(dataset_path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != FROZEN_PUBLIC_SET_SHA256:
            raise ValueError(
                "full public-set evaluation requires the frozen public dataset"
            )
        samples = load_jsonl(path)
        if len(samples) != 200:
            raise ValueError("full public-set evaluation requires all 200 sessions")
        return samples
    samples = (
        load_frozen_development_samples(dataset_path)
        if development_only
        else load_jsonl(dataset_path)
    )
    if session_count is None:
        return samples
    if session_count <= 0:
        raise ValueError("session_count must be positive")
    return stratified_subset(
        samples,
        session_count,
        seed=EVALUATION_SUBSET_SEED,
    )


def validate_connected_evaluation_scope(
    *,
    connected: bool,
    development_only: bool,
    full_exposed_public_set: bool,
) -> str:
    """Require connected runs to state whether protected rows are included."""
    if development_only and full_exposed_public_set:
        raise ValueError("evaluation scopes are mutually exclusive")
    if connected and not (development_only or full_exposed_public_set):
        raise ValueError(
            "connected evaluation requires an explicit evaluation scope"
        )
    if development_only:
        return "development_only"
    if full_exposed_public_set:
        return "full_exposed_public_set"
    return "public_set"


def _load_dotenv(path: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "python-dotenv is required for connected evaluation; install "
            "requirements-openai.txt"
        ) from error
    load_dotenv(dotenv_path=Path(path), override=False)


def turn_latency_summary(latencies: list[float]) -> dict:
    """Summarize complete ``Agent.respond`` latency with nearest-rank tails."""
    if not latencies:
        return {
            "turn_count": 0,
            "p50_seconds": 0.0,
            "p95_seconds": 0.0,
            "mean_seconds": 0.0,
        }
    ordered = sorted(latencies)

    def percentile(fraction: float) -> float:
        index = max(0, ceil(len(ordered) * fraction) - 1)
        return ordered[index]

    return {
        "turn_count": len(ordered),
        "p50_seconds": round(percentile(0.50), 6),
        "p95_seconds": round(percentile(0.95), 6),
        "mean_seconds": round(statistics.fmean(ordered), 6),
    }


class _InstrumentedAgent:
    """Records latency and diagnostics without touching the pristine scorer."""

    def __init__(self, agent) -> None:
        self._agent = agent
        self.turn_latencies: list[float] = []
        self.response_exception_count = 0
        self.invalid_response_count = 0
        self.planning_source_counts: Counter[str] = Counter()
        self.fallback_reason_counts: Counter[str] = Counter()
        self._observed_entries: dict[str, int] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        try:
            response = self._agent.respond(session_id, user_message, turn, top_k)
        except Exception:
            self.response_exception_count += 1
            raise
        finally:
            self.turn_latencies.append(time.perf_counter() - started)
            self._observe_planning(session_id)
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            self.invalid_response_count += 1
        return response

    def _observe_planning(self, session_id: str) -> None:
        history_method = getattr(self._agent, "get_planning_history", None)
        if not callable(history_method):
            return
        history = history_method(session_id)
        seen = self._observed_entries.get(session_id, 0)
        for entry in history[seen:]:
            source = entry.get("source")
            if isinstance(source, str):
                self.planning_source_counts[source] += 1
            fallback_reason = entry.get("fallback_reason")
            if isinstance(fallback_reason, str) and fallback_reason:
                self.fallback_reason_counts[fallback_reason] += 1
        self._observed_entries[session_id] = len(history)


def evaluate_with_diagnostics(
    agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict,
    products: dict,
) -> dict:
    """Score with the pristine official ``evaluate`` plus team diagnostics."""
    instrumented = _InstrumentedAgent(agent)
    result = evaluate(instrumented, samples, catalog_ids, categories, products)
    result["turn_latency"] = turn_latency_summary(instrumented.turn_latencies)
    result["execution_diagnostics"] = {
        "response_exception_count": instrumented.response_exception_count,
        "invalid_response_count": instrumented.invalid_response_count,
        "connected_plan_turn_count": instrumented.planning_source_counts["connected"],
        "fallback_plan_turn_count": instrumented.planning_source_counts["fallback"],
        "fallback_reason_counts": dict(
            sorted(instrumented.fallback_reason_counts.items())
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument(
        "--full-exposed-public-set",
        action="store_true",
        help=(
            "Acknowledge that all 200 public sessions include the previously "
            "exposed former reserved split and are not untouched evidence."
        ),
    )
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument(
        "--openai-role",
        choices=("quality_reference", "lower_cost"),
        default=None,
        help="Explicitly enable connected planning with a configured model role.",
    )
    parser.add_argument(
        "--openai-config",
        default="config/openai_phase_a_benchmark.json",
    )
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        scope = validate_connected_evaluation_scope(
            connected=args.openai_role is not None,
            development_only=args.development_only,
            full_exposed_public_set=args.full_exposed_public_set,
        )
    except ValueError as error:
        parser.error(str(error))
    samples = load_evaluation_samples(
        args.dataset,
        development_only=args.development_only,
        full_exposed_public_set=args.full_exposed_public_set,
        session_count=args.sessions,
    )
    catalog_ids, categories, products = catalog_index(args.catalog)
    openai_settings = None
    if args.openai_role is not None:
        _load_dotenv(args.env_file)
        openai_settings = load_openai_evaluation_settings(
            args.openai_config,
            args.openai_role,
        )
    agent = build_evaluation_agent(args.catalog, openai_settings)
    try:
        result = evaluate_with_diagnostics(
            agent, samples, catalog_ids, categories, products
        )
        result["evaluation_scope"] = {
            "scope": scope,
            "development_only": args.development_only,
            "full_exposed_public_set": args.full_exposed_public_set,
            "untouched_holdout_evidence": False,
            "requested_sessions": args.sessions,
            "evaluated_sessions": len(samples),
            "subset_seed": EVALUATION_SUBSET_SEED if args.sessions else None,
        }
        runtime_configuration = getattr(agent, "get_runtime_configuration", None)
        if callable(runtime_configuration):
            result["runtime_configuration"] = runtime_configuration()
        planning_loop = getattr(agent, "planning_loop", None)
        provider = getattr(planning_loop, "provider", None)
        if provider is not None:
            metrics = getattr(provider, "metrics", None)
            result["connected_planning"] = {
                "role": args.openai_role,
                "metrics": metrics() if callable(metrics) else None,
            }
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "sessions"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
