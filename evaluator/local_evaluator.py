from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path

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


MAX_TURNS = 10
TOP_K = 10
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
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
    samples = load_frozen_development_samples(dataset_path) if development_only else load_jsonl(dataset_path)
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


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def behavior_for(scenario: str, card: dict, rng: random.Random) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard = card["hard_constraints"]
        soft = card["soft_preferences"]
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]),
            "old_value": old_value,
            "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return behavior


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= TOP_K:
            break
    return result


def catalog_index(catalog_path: str | Path) -> tuple[set[str], dict[str, list[str]], dict[str, dict]]:
    identifiers: set[str] = set()
    categories: dict[str, list[str]] = {}
    products: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            identifiers.add(parent_asin)
            categories[parent_asin] = [str(value) for value in product.get("categories") or []]
            products[parent_asin] = product
    return identifiers, categories, products


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def initial_message(sample: dict, category: str, disclosed: set[str]) -> str:
    scenario = sample["scenario_type"]
    if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
        constraint = str(sample["intent_card"]["hard_constraints"][0])
        disclosed.add(constraint)
        return f"I'm looking for {category}. A key requirement is: {constraint}."
    if scenario == "intent_override":
        old_value = str(sample["behavior"]["override"]["old_value"])
        return f"I'm looking for {category}. {old_value}"
    return f"I'm looking for {category}, but I'm still exploring."


def customer_reply(sample: dict, ask_attribute: object, disclosed: set[str], boundary_used: bool) -> tuple[str, bool]:
    attribute = ask_attribute if isinstance(ask_attribute, str) else None
    if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
        return f"I don't have a preference for {attribute}; please use your judgment.", True
    if not attribute:
        return "Those options are not quite right yet. Ask me about one specific attribute.", boundary_used
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"
    constraints = [
        *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
        *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
    ]
    matches = [
        value for value in constraints
        if value not in disclosed and (attribute == "other" or classify_constraint(value) == attribute)
    ][:2]
    if not matches:
        return f"I don't have an additional preference for {attribute}.", boundary_used
    disclosed.update(matches)
    return "For that, what matters is: " + "; ".join(matches) + ".", boundary_used


def metric_summary(sessions: list[dict]) -> dict:
    if not sessions:
        return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mttc": None}
    hit_rate = sum(int(item["hit"]) for item in sessions) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1 for item in sessions
    )
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
    }


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


def materialize_hidden_fields(sample: dict, products: dict[str, dict]) -> tuple[dict, dict]:
    if "intent_card" in sample and "behavior" in sample:
        return sample["intent_card"], sample["behavior"]
    target = str(sample["ground_truth"]["parent_asin"])
    product = products[target]
    card = intent_card(product)
    seed_source = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    rng = random.Random(seed_source)
    behavior = behavior_for(str(sample["scenario_type"]), card, rng)
    return card, behavior


def evaluate(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    sessions: list[dict] = []
    turn_latencies: list[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    response_exception_count = 0
    invalid_response_count = 0
    planning_source_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    for sample in samples:
        session_id = f"public_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        hit_turn: int | None = None
        best_rank: int | None = None
        observed_planning_entries = 0
        for turn in range(1, MAX_TURNS + 1):
            turn_started = time.perf_counter()
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response_exception_count += 1
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            finally:
                turn_latencies.append(time.perf_counter() - turn_started)
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                invalid_response_count += 1
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            history_method = getattr(agent, "get_planning_history", None)
            if callable(history_method):
                planning_history = history_method(session_id)
                for entry in planning_history[observed_planning_entries:]:
                    source = entry.get("source")
                    if isinstance(source, str):
                        planning_source_counts[source] += 1
                    fallback_reason = entry.get("fallback_reason")
                    if isinstance(fallback_reason, str) and fallback_reason:
                        fallback_reason_counts[fallback_reason] += 1
                observed_planning_entries = len(planning_history)
            usage = response.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                    total_prompt_tokens += usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int) and usage["completion_tokens"] >= 0:
                    total_completion_tokens += usage["completion_tokens"]
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "execution_diagnostics": {
            "response_exception_count": response_exception_count,
            "invalid_response_count": invalid_response_count,
            "connected_plan_turn_count": planning_source_counts["connected"],
            "fallback_plan_turn_count": planning_source_counts["fallback"],
            "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "turn_latency": turn_latency_summary(turn_latencies),
        "sessions": sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TechJam public-set local evaluator")
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
        result = evaluate(agent, samples, catalog_ids, categories, products)
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
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
