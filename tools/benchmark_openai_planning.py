"""Benchmark connected planners on canonical development-only Turn Plans.

Canonical ``PlanningRequest`` fixtures are generated once from the local,
authoritative interpreter. Both connected models receive those exact requests;
their outputs never influence a later fixture. This preserves input parity while
still scoring validated state transitions and downstream local retrieval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Callable, Mapping, Sequence

from evaluator.local_evaluator import MAX_TURNS, catalog_index, materialize_hidden_fields
from retrieval.fusion import build_fusion_policy
from starter.constraint_state import (
    AddConstraint,
    ConstraintState,
    DismissAttribute,
    PlanValidationError,
    ReintroduceConstraint,
    ReplaceConstraint,
    ReplaceProductIntent,
    TurnPlan,
)
from starter.openai_planning import (
    DevelopmentBudget,
    ModelPricing,
    OPENAI_TRANSPORT_SCHEMA_VERSION,
    OpenAIPlanningProvider,
    openai_transport_schema_sha256,
)
from starter.planning import (
    APPROVED_RETRIEVAL_TOOLS,
    PLANNING_INSTRUCTIONS,
    PLANNING_PROMPT_SHA256,
    PLANNING_PROMPT_VERSION,
    PlanningRequest,
    ProviderResponse,
    TURN_PLAN_JSON_SCHEMA,
    decode_plan,
    planning_request_as_dict,
)
from starter.retrieval import CatalogRetrieval
from starter.turn_interpreter import interpret_turn
from tools.dataset_split import (
    FROZEN_HOLDOUT_SAMPLE_IDS,
    SPLIT_VERSION,
    load_frozen_development_samples,
    stratified_subset,
)


BENCHMARK_VERSION = "openai-planning-phase-a-v1"
FIXTURE_VERSION = "canonical-planning-fixtures-v1"
SUBSET_SEED = 20260831
REQUIRED_MODEL_ROLES = {"quality_reference", "lower_cost"}
REPORT_SCHEMA_PATH = (
    Path(__file__).parents[1] / "docs" / "openai_model_benchmark_report.schema.json"
)
DEFAULT_QUALITY_TOLERANCE = {
    "aggregate_state_accuracy_max_regression": 0.02,
    "scenario_state_accuracy_max_regression": 0.05,
    "downstream_hit_rate_at_10_max_regression": 0.05,
    "additional_failure_rate_max": 0.0,
}


class BenchmarkConfigurationError(ValueError):
    """The benchmark could violate its frozen comparison contract."""


def validate_benchmark_report(report: Mapping[str, object]) -> None:
    """Validate the stable, machine-readable Phase A comparison shape."""

    schema = json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))

    def validate(value: object, node: Mapping[str, object], path: str) -> None:
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/"):
            resolved: object = schema
            for part in reference[2:].split("/"):
                resolved = resolved[part]
            node = resolved
        expected_type = node.get("type")
        matches = {
            "object": isinstance(value, Mapping),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }
        if expected_type in matches and not matches[expected_type]:
            raise BenchmarkConfigurationError(
                f"report field {path} must be {expected_type}"
            )
        if expected_type == "object" and isinstance(value, Mapping):
            for key in node.get("required", []):
                if key not in value:
                    raise BenchmarkConfigurationError(
                        f"report is missing required field {path}.{key}"
                    )
            properties = node.get("properties", {})
            for key, child in properties.items():
                if key in value:
                    validate(value[key], child, f"{path}.{key}")
            additional = node.get("additionalProperties")
            if isinstance(additional, Mapping):
                for key in value.keys() - properties.keys():
                    validate(value[key], additional, f"{path}.{key}")
        if expected_type == "array" and isinstance(value, list):
            items = node.get("items")
            if isinstance(items, Mapping):
                for index, item in enumerate(value):
                    validate(item, items, f"{path}[{index}]")

    validate(report, schema, "report")


@dataclass(frozen=True)
class ModelBenchmarkConfig:
    role: str
    model: str
    input_per_million_usd: float
    output_per_million_usd: float

    @property
    def pricing(self) -> ModelPricing:
        return ModelPricing(
            input_per_million_usd=self.input_per_million_usd,
            output_per_million_usd=self.output_per_million_usd,
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    models: dict[str, ModelBenchmarkConfig]
    budget_limit_usd: float
    warning_usd: float
    review_boundary_usd: float
    absolute_stop_usd: float
    dataset_split: str
    timeout_seconds: float
    max_output_tokens: int
    max_input_tokens_estimate: int
    reasoning_effort: str
    quality_tolerance: dict
    sha256: str


@dataclass(frozen=True)
class PlanningFixture:
    fixture_id: str
    sample_id: str
    scenario_type: str
    target_asin: str
    user_message: str
    state_before: ConstraintState
    expected_plan: TurnPlan
    expected_tools: tuple[str, ...]
    request: PlanningRequest
    supported_values: dict[str, set[str]]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    raw_bytes = config_path.read_bytes()
    raw = json.loads(raw_bytes)
    models = {
        item["role"]: ModelBenchmarkConfig(
            role=str(item["role"]),
            model=str(item["model"]),
            input_per_million_usd=float(item["pricing_usd_per_million_tokens"]["input"]),
            output_per_million_usd=float(item["pricing_usd_per_million_tokens"]["output"]),
        )
        for item in raw["models"]
    }
    if set(models) != REQUIRED_MODEL_ROLES or len(raw["models"]) != 2:
        raise BenchmarkConfigurationError(
            "benchmark must define exactly quality_reference and lower_cost"
        )
    if len({item.model for item in models.values()}) != 2:
        raise BenchmarkConfigurationError("benchmark model identities must be distinct")
    dataset_split = str(raw["dataset_split"])
    if dataset_split != f"{SPLIT_VERSION}-development-only":
        raise BenchmarkConfigurationError("benchmark must use the frozen development split")
    budget = raw["budget_usd"]
    absolute_stop = float(budget["absolute_stop"])
    limit = float(budget["phase_a_limit"])
    if limit > absolute_stop:
        raise BenchmarkConfigurationError("Phase A limit exceeds the absolute stop")
    api = raw["api"]
    return BenchmarkConfig(
        models=models,
        budget_limit_usd=limit,
        warning_usd=float(budget["warning"]),
        review_boundary_usd=float(budget["review_boundary"]),
        absolute_stop_usd=absolute_stop,
        dataset_split=dataset_split,
        timeout_seconds=float(api["timeout_seconds"]),
        max_output_tokens=int(api["max_output_tokens"]),
        max_input_tokens_estimate=int(api["max_input_tokens_estimate"]),
        reasoning_effort=str(api["reasoning_effort"]),
        quality_tolerance=dict(raw["quality_tolerance"]),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def assert_development_only_samples(samples: Sequence[dict]) -> None:
    identifiers = [str(sample.get("sample_id", "")) for sample in samples]
    protected = sorted(set(identifiers).intersection(FROZEN_HOLDOUT_SAMPLE_IDS))
    if protected:
        raise BenchmarkConfigurationError(
            "Locked Holdout identifiers are forbidden in the connected benchmark"
        )
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkConfigurationError("development samples must be unique")


def _canonical_messages_for_sample(sample: dict) -> list[str]:
    card = sample.get("intent_card") or {}
    category = str(card.get("target_category") or "product")
    scenario = str(sample["scenario_type"])
    messages = [
        f"I'm looking for {category}, but I'm still exploring."
        if scenario != "buying"
        else f"I'm looking for {category}."
    ]
    for value in card.get("hard_constraints") or []:
        message = f"I need {value}."
        if str(value).lower() not in messages[0].lower():
            messages.append(message)
    for value in card.get("soft_preferences") or []:
        message = f"I prefer {value}."
        if str(value).lower() not in " ".join(messages).lower():
            messages.append(message)
    if scenario == "boundary":
        messages.append(
            "I don't have a preference for material; please use your judgment."
        )
    if scenario == "intent_override":
        override = (sample.get("behavior") or {}).get("override") or {}
        override_turn = max(2, min(MAX_TURNS, int(override.get("turn", 3))))
        while len(messages) < override_turn - 1:
            messages.append("Show me more options.")
        messages.insert(
            override_turn - 1,
            str(override.get("message") or "Actually, replace my earlier request."),
        )
    return messages[:MAX_TURNS]


def build_planning_fixtures(
    samples: Sequence[dict],
    catalog_path: str | Path,
) -> list[PlanningFixture]:
    assert_development_only_samples(samples)
    retrieval = CatalogRetrieval(catalog_path)
    _catalog_ids, _categories, products = catalog_index(catalog_path)
    supported_values = retrieval.supported_values
    fixtures: list[PlanningFixture] = []
    for sample in samples:
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {
            **sample,
            "intent_card": intent_card,
            "behavior": behavior,
        }
        state = ConstraintState()
        recent_history: list[dict] = []
        sample_id = str(sample["sample_id"])
        for turn, user_message in enumerate(
            _canonical_messages_for_sample(effective_sample),
            start=1,
        ):
            expected_plan = interpret_turn(
                user_message,
                turn=turn,
                state=state,
                supported_values=supported_values,
            )
            expected_draft = deepcopy(state)
            try:
                expected_draft.apply(expected_plan, supported_values)
            except PlanValidationError:
                expected_plan = TurnPlan(
                    expected_state_revision=state.revision,
                    source_turn=turn,
                )
            request = PlanningRequest(
                session_id=sample_id,
                turn=turn,
                user_message=user_message,
                state_snapshot=state.as_dict(),
                recent_history=tuple(deepcopy(recent_history[-4:])),
                supported_values={
                    attribute: tuple(sorted(values))
                    for attribute, values in supported_values.items()
                },
                allowed_tools=APPROVED_RETRIEVAL_TOOLS,
                prompt_version=PLANNING_PROMPT_VERSION,
                instructions=PLANNING_INSTRUCTIONS,
                response_schema=deepcopy(TURN_PLAN_JSON_SCHEMA),
            )
            fixtures.append(PlanningFixture(
                fixture_id=f"{sample_id}:turn-{turn}",
                sample_id=sample_id,
                scenario_type=str(sample["scenario_type"]),
                target_asin=str(sample["ground_truth"]["parent_asin"]),
                user_message=user_message,
                state_before=deepcopy(state),
                expected_plan=expected_plan,
                expected_tools=APPROVED_RETRIEVAL_TOOLS,
                request=request,
                supported_values={
                    key: set(values) for key, values in supported_values.items()
                },
            ))
            state.apply(expected_plan, supported_values)
            recent_history.append({
                "turn": turn,
                "user_message": user_message,
                "state_revision": state.revision,
                "source": "canonical_local",
                "retrieval_tools": list(APPROVED_RETRIEVAL_TOOLS),
            })
    return fixtures


def fixture_corpus_sha256(fixtures: Sequence[PlanningFixture]) -> str:
    payload = [planning_request_as_dict(fixture.request) for fixture in fixtures]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _state_signature(state: ConstraintState) -> dict:
    return {
        "active_product_intent_id": state.active_product_intent_id,
        "constraints": [
            {
                "attribute": item.attribute,
                "values": list(item.values),
                "match_rule": item.match_rule,
                "classification": item.classification,
                "scope": item.scope,
                "status": item.status,
                "transition_reason": item.transition_reason,
            }
            for item in state.constraints
        ],
        "dismissed_attributes": sorted(state.dismissed_attributes),
        "transition_types": [
            event.get("type") for event in state.transition_history
        ],
    }


def _mutation_signature(plan: TurnPlan) -> tuple[tuple, ...]:
    signatures = []
    for mutation in plan.mutations:
        if isinstance(mutation, ReplaceConstraint):
            signatures.append((
                "replace_constraint", mutation.constraint_id, mutation.attribute,
                mutation.values, mutation.match_rule, mutation.classification,
                mutation.scope,
            ))
        elif isinstance(mutation, ReintroduceConstraint):
            signatures.append((
                "reintroduce_constraint", mutation.attribute, mutation.values,
                mutation.match_rule, mutation.classification, mutation.scope,
            ))
        elif isinstance(mutation, AddConstraint):
            signatures.append((
                "add_constraint", mutation.attribute, mutation.values,
                mutation.match_rule, mutation.classification, mutation.scope,
            ))
        elif isinstance(mutation, DismissAttribute):
            signatures.append(("dismiss_attribute", mutation.attribute))
        elif isinstance(mutation, ReplaceProductIntent):
            signatures.append(("replace_product_intent", mutation.product_intent_id))
    return tuple(signatures)


def _replacement_signature(plan: TurnPlan) -> tuple[tuple, ...]:
    return tuple(
        signature
        for signature in _mutation_signature(plan)
        if signature[0] in {"replace_constraint", "replace_product_intent"}
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _summarize(records: Sequence[dict]) -> dict:
    count = len(records)
    if not count:
        return {
            "fixture_count": 0,
            "state_accuracy": 0.0,
            "constraint_decision_accuracy": 0.0,
            "replacement_decision_accuracy": 0.0,
            "tool_decision_accuracy": 0.0,
            "clarification_protocol_quality": 0.0,
            "downstream_hit_rate_at_10": 0.0,
            "downstream_mrr": 0.0,
            "failure_rate": 0.0,
        }
    return {
        "fixture_count": count,
        "state_accuracy": round(sum(item["state_correct"] for item in records) / count, 6),
        "constraint_decision_accuracy": round(
            sum(item["constraint_decision_correct"] for item in records) / count,
            6,
        ),
        "replacement_decision_accuracy": round(
            sum(item["replacement_correct"] for item in records) / count,
            6,
        ),
        "tool_decision_accuracy": round(sum(item["tools_correct"] for item in records) / count, 6),
        "clarification_protocol_quality": round(
            sum(item["clarification_valid"] for item in records) / count,
            6,
        ),
        "downstream_hit_rate_at_10": round(sum(item["hit"] for item in records) / count, 6),
        "downstream_mrr": round(sum(item["reciprocal_rank"] for item in records) / count, 6),
        "failure_rate": round(sum(item["failed"] for item in records) / count, 6),
    }


def evaluate_provider(
    *,
    provider: object,
    fixtures: Sequence[PlanningFixture],
    catalog_path: str | Path,
    model: ModelBenchmarkConfig,
) -> dict:
    retrieval = CatalogRetrieval(catalog_path)
    fusion = build_fusion_policy("fixed", candidate_limit=50)
    records: list[dict] = []
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    failure_causes: Counter[str] = Counter()
    provider_metrics_before = _provider_metrics(provider)
    for fixture in fixtures:
        started = time.perf_counter()
        record = {
            "scenario_type": fixture.scenario_type,
            "state_correct": 0,
            "constraint_decision_correct": 0,
            "replacement_correct": 0,
            "tools_correct": 0,
            "selected_tools": (),
            "clarification_valid": 0,
            "hit": 0,
            "reciprocal_rank": 0.0,
            "failed": 1,
        }
        try:
            raw = provider.plan(fixture.request)
            response = raw if isinstance(raw, ProviderResponse) else ProviderResponse(raw)
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            decoded = decode_plan(
                response.output,
                user_message=fixture.user_message,
                turn=fixture.request.turn,
                state=fixture.state_before,
                supported_values=fixture.supported_values,
                grounding_mutations=fixture.expected_plan.mutations,
            )
            expected_state = deepcopy(fixture.state_before)
            expected_state.apply(fixture.expected_plan, fixture.supported_values)
            candidate_state = deepcopy(fixture.state_before)
            candidate_state.apply(decoded.turn_plan, fixture.supported_values)
            route_scores = retrieval.hybrid_route_scores(
                fixture.user_message,
                candidate_state.constraints,
                [],
                route_limit=100,
                enabled_routes=set(decoded.retrieval_tools),
            )
            ranked = fusion.rank(route_scores, candidate_limit=50)[:10]
            rank = (
                ranked.index(fixture.target_asin) + 1
                if fixture.target_asin in ranked
                else None
            )
            record.update({
                "state_correct": int(
                    _state_signature(candidate_state) == _state_signature(expected_state)
                ),
                "constraint_decision_correct": int(
                    _mutation_signature(decoded.turn_plan)
                    == _mutation_signature(fixture.expected_plan)
                ),
                "replacement_correct": int(
                    _replacement_signature(decoded.turn_plan)
                    == _replacement_signature(fixture.expected_plan)
                ),
                "tools_correct": int(
                    set(decoded.retrieval_tools) == set(fixture.expected_tools)
                ),
                "selected_tools": decoded.retrieval_tools,
                "clarification_valid": 1,
                "hit": int(rank is not None),
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "failed": 0,
            })
        except Exception as error:  # noqa: BLE001 - failures are benchmark data
            failure_causes[error.__class__.__name__] += 1
        finally:
            latencies.append(time.perf_counter() - started)
            records.append(record)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["scenario_type"]].append(record)
    token_estimated_cost = model.pricing.estimate(prompt_tokens, completion_tokens)
    provider_metrics_after = _provider_metrics(provider)
    accounted_cost = max(
        0.0,
        _budget_spend(provider_metrics_after) - _budget_spend(provider_metrics_before),
    )
    estimated_cost = max(token_estimated_cost, accounted_cost)
    successful_records = [record for record in records if not record["failed"]]
    route_selection_rates = {
        route: round(
            sum(route in record["selected_tools"] for record in successful_records)
            / len(successful_records),
            6,
        ) if successful_records else 0.0
        for route in APPROVED_RETRIEVAL_TOOLS
    }
    return {
        "role": model.role,
        "model": model.model,
        "input_corpus_sha256": fixture_corpus_sha256(fixtures),
        "aggregate": _summarize(records),
        "scenario_metrics": {
            name: _summarize(grouped[name]) for name in sorted(grouped)
        },
        "route_selection_rates": route_selection_rates,
        "latency": {
            "p50_seconds": round(_percentile(latencies, 0.50), 6),
            "p95_seconds": round(_percentile(latencies, 0.95), 6),
            "mean_seconds": round(statistics.fmean(latencies), 6) if latencies else 0.0,
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "failure_causes": dict(sorted(failure_causes.items())),
        "estimated_cost_usd": round(estimated_cost, 9),
        "cost_accounting": {
            "successful_token_estimate_usd": round(token_estimated_cost, 9),
            "provider_budget_delta_usd": round(accounted_cost, 9),
            "includes_pessimistic_failed_call_reservations": (
                accounted_cost > token_estimated_cost + 1e-12
            ),
        },
        "pricing_usd_per_million_tokens": {
            "input": model.input_per_million_usd,
            "output": model.output_per_million_usd,
        },
    }


def _provider_metrics(provider: object) -> Mapping[str, object]:
    metrics = getattr(provider, "metrics", None)
    if not callable(metrics):
        return {}
    value = metrics()
    return value if isinstance(value, Mapping) else {}


def _budget_spend(metrics: Mapping[str, object]) -> float:
    budget = metrics.get("budget")
    if not isinstance(budget, Mapping):
        return 0.0
    spent = budget.get("spent_usd", 0.0)
    return float(spent) if isinstance(spent, (int, float)) else 0.0


def compare_providers(
    *,
    fixtures: Sequence[PlanningFixture],
    catalog_path: str | Path,
    providers: Mapping[str, object],
    models: Mapping[str, ModelBenchmarkConfig],
    configuration_sha256: str,
    quality_tolerance: Mapping[str, object] | None = None,
) -> dict:
    if set(providers) != REQUIRED_MODEL_ROLES or set(models) != REQUIRED_MODEL_ROLES:
        raise BenchmarkConfigurationError("both configured model roles are required")
    reports = {
        role: evaluate_provider(
            provider=providers[role],
            fixtures=fixtures,
            catalog_path=catalog_path,
            model=models[role],
        )
        for role in ("quality_reference", "lower_cost")
    }
    hashes = {report["input_corpus_sha256"] for report in reports.values()}
    if len(hashes) != 1:
        raise BenchmarkConfigurationError("model input corpora are not identical")
    corpus_sha256 = hashes.pop()
    tolerance = {
        **DEFAULT_QUALITY_TOLERANCE,
        **dict(quality_tolerance or {}),
    }
    reference = reports["quality_reference"]
    lower_cost = reports["lower_cost"]
    aggregate_state_regression = (
        reference["aggregate"]["state_accuracy"]
        - lower_cost["aggregate"]["state_accuracy"]
    )
    downstream_regression = (
        reference["aggregate"]["downstream_hit_rate_at_10"]
        - lower_cost["aggregate"]["downstream_hit_rate_at_10"]
    )
    additional_failure_rate = (
        lower_cost["aggregate"]["failure_rate"]
        - reference["aggregate"]["failure_rate"]
    )
    scenario_state_regressions = {
        scenario: (
            reference["scenario_metrics"][scenario]["state_accuracy"]
            - lower_cost["scenario_metrics"][scenario]["state_accuracy"]
        )
        for scenario in sorted(reference["scenario_metrics"])
    }
    within_tolerance = (
        aggregate_state_regression
        <= float(tolerance["aggregate_state_accuracy_max_regression"])
        and downstream_regression
        <= float(tolerance["downstream_hit_rate_at_10_max_regression"])
        and additional_failure_rate
        <= float(tolerance["additional_failure_rate_max"])
        and all(
            regression
            <= float(tolerance["scenario_state_accuracy_max_regression"])
            for regression in scenario_state_regressions.values()
        )
    )
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "fixture_version": FIXTURE_VERSION,
        "status": "provisional_phase_a",
        "selection_status": "provisional_no_default_activation",
        "configuration_sha256": configuration_sha256,
        "planning_contract": {
            "prompt_version": PLANNING_PROMPT_VERSION,
            "prompt_sha256": PLANNING_PROMPT_SHA256,
            "transport_schema": {
                "version": OPENAI_TRANSPORT_SCHEMA_VERSION,
                "sha256": openai_transport_schema_sha256(TURN_PLAN_JSON_SCHEMA),
            },
            "allowed_tools": list(APPROVED_RETRIEVAL_TOOLS),
        },
        "input_parity": {
            "identical": True,
            "fixture_count": len(fixtures),
            "corpus_sha256": corpus_sha256,
        },
        "models": reports,
        "provisional_comparison": {
            "lower_cost_within_tolerance": within_tolerance,
            "aggregate_state_accuracy_regression": round(
                aggregate_state_regression,
                6,
            ),
            "scenario_state_accuracy_regressions": {
                key: round(value, 6)
                for key, value in scenario_state_regressions.items()
            },
            "downstream_hit_rate_at_10_regression": round(
                downstream_regression,
                6,
            ),
            "additional_failure_rate": round(additional_failure_rate, 6),
            "quality_tolerance": tolerance,
            "default_activation_allowed": False,
            "activation_gate": "Slice 13 must merge before the complete Phase B rerun",
        },
    }
    validate_benchmark_report(report)
    return report


def _load_dotenv(path: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError(
            "python-dotenv is required for the credentialed benchmark; install "
            "requirements-openai.txt"
        ) from error
    load_dotenv(dotenv_path=Path(path), override=False)


def _write_report(report: dict, output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/openai_phase_a_benchmark.json")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", default="benchmarks/openai_phase_a_report.json")
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()

    configuration = load_benchmark_config(arguments.config)
    samples = load_frozen_development_samples(arguments.dataset)
    if arguments.sessions is not None:
        samples = stratified_subset(samples, arguments.sessions, seed=SUBSET_SEED)
    assert_development_only_samples(samples)
    fixtures = build_planning_fixtures(samples, arguments.catalog)
    if arguments.validate_only:
        report = {
            "benchmark_version": BENCHMARK_VERSION,
            "fixture_version": FIXTURE_VERSION,
            "status": "validated_without_api_calls",
            "selection_status": "provisional_no_default_activation",
            "configuration_sha256": configuration.sha256,
            "development_session_count": len(samples),
            "fixture_count": len(fixtures),
            "input_corpus_sha256": fixture_corpus_sha256(fixtures),
            "locked_holdout_loaded": False,
        }
        _write_report(report, arguments.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    _load_dotenv(arguments.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment")
    budget = DevelopmentBudget(
        limit_usd=configuration.budget_limit_usd,
        warning_usd=configuration.warning_usd,
        review_boundary_usd=configuration.review_boundary_usd,
        absolute_stop_usd=configuration.absolute_stop_usd,
    )
    providers = {
        role: OpenAIPlanningProvider(
            model=model.model,
            timeout_seconds=configuration.timeout_seconds,
            max_output_tokens=configuration.max_output_tokens,
            max_input_tokens_estimate=configuration.max_input_tokens_estimate,
            reasoning_effort=configuration.reasoning_effort,
            pricing=model.pricing,
            budget=budget,
        )
        for role, model in configuration.models.items()
    }
    report = compare_providers(
        fixtures=fixtures,
        catalog_path=arguments.catalog,
        providers=providers,
        models=configuration.models,
        configuration_sha256=configuration.sha256,
        quality_tolerance=configuration.quality_tolerance,
    )
    report["budget"] = budget.as_dict()
    report["quality_tolerance"] = configuration.quality_tolerance
    report["locked_holdout_loaded"] = False
    _write_report(report, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
