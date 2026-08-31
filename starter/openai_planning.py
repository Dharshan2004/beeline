"""Connected OpenAI adapter for the provider-neutral planning boundary.

The adapter is deliberately stateless: every call receives the authoritative
local Constraint State and recent history through ``PlanningRequest``. It does
not expose hosted tools and uses strict Structured Outputs for the existing
Turn Plan schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from starter.planning import (
    BudgetExceededError,
    MissingCredentialsError,
    PlanningRequest,
    PlanningSchemaError,
    ProviderResponse,
    planning_request_as_dict,
)


DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_OUTPUT_TOKENS = 2_000
DEFAULT_MAX_INPUT_TOKENS_ESTIMATE = 32_000
ABSOLUTE_BUDGET_CEILING_USD = 600.0


@dataclass(frozen=True)
class ModelPricing:
    """Standard text-token prices in USD per one million tokens."""

    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.input_per_million_usd < 0 or self.output_per_million_usd < 0:
            raise ValueError("model prices must be non-negative")

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (
            input_tokens * self.input_per_million_usd
            + output_tokens * self.output_per_million_usd
        ) / 1_000_000


@dataclass
class DevelopmentBudget:
    """In-process accounting with authorization before every API call."""

    limit_usd: float
    warning_usd: float = 40.0
    review_boundary_usd: float = 50.0
    absolute_stop_usd: float = ABSOLUTE_BUDGET_CEILING_USD
    spent_usd: float = 0.0
    review_approved: bool = False

    def __post_init__(self) -> None:
        if self.limit_usd <= 0:
            raise ValueError("development budget limit must be positive")
        if self.limit_usd > self.absolute_stop_usd:
            raise ValueError("development budget cannot exceed the absolute stop")
        if self.limit_usd > self.review_boundary_usd and not self.review_approved:
            raise ValueError(
                "development budget above the review boundary requires approval"
            )
        if self.spent_usd < 0 or self.spent_usd > self.limit_usd:
            raise ValueError("initial development spend is outside the budget")

    def authorize(self, maximum_call_cost_usd: float) -> None:
        if maximum_call_cost_usd < 0:
            raise ValueError("maximum call cost must be non-negative")
        if self.spent_usd + maximum_call_cost_usd > self.limit_usd:
            raise BudgetExceededError(
                "connected development budget would be exceeded before this call"
            )

    def record(self, actual_cost_usd: float) -> None:
        if actual_cost_usd < 0:
            raise ValueError("actual call cost must be non-negative")
        new_total = self.spent_usd + actual_cost_usd
        if new_total > self.limit_usd or new_total > self.absolute_stop_usd:
            raise BudgetExceededError("connected development budget was exceeded")
        self.spent_usd = new_total

    def as_dict(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 9),
            "limit_usd": self.limit_usd,
            "warning_usd": self.warning_usd,
            "review_boundary_usd": self.review_boundary_usd,
            "absolute_stop_usd": self.absolute_stop_usd,
            "warning_reached": self.spent_usd >= self.warning_usd,
            "review_approved": self.review_approved,
        }


def _request_input(request: PlanningRequest) -> str:
    payload = planning_request_as_dict(request)
    # Instructions and schema travel in dedicated Responses API fields. Keeping
    # one canonical representation prevents benchmark/API payload drift.
    payload.pop("instructions")
    payload.pop("response_schema")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


class OpenAIPlanningProvider:
    """Use the Responses API behind ``PlanningProvider.plan``."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_input_tokens_estimate: int = DEFAULT_MAX_INPUT_TOKENS_ESTIMATE,
        reasoning_effort: str = "low",
        pricing: ModelPricing | None = None,
        budget: DevelopmentBudget | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise MissingCredentialsError("OPENAI_API_KEY is not configured")
        if not model.strip():
            raise ValueError("OpenAI model identity must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be positive")
        if max_output_tokens <= 0 or max_input_tokens_estimate <= 0:
            raise ValueError("OpenAI token bounds must be positive")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens_estimate = max_input_tokens_estimate
        self.reasoning_effort = reasoning_effort
        self.pricing = pricing or ModelPricing()
        self.budget = budget or DevelopmentBudget(limit_usd=10.0)
        self._client = client if client is not None else self._build_client(key)
        self._submitted_call_count = 0
        self._pessimistically_accounted_call_count = 0

    @staticmethod
    def _build_client(api_key: str):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "the optional OpenAI SDK is not installed; install "
                "requirements-openai.txt"
            ) from error
        return OpenAI(api_key=api_key)

    def plan(self, request: PlanningRequest) -> ProviderResponse:
        request_input = _request_input(request)
        request_bytes = sum((
            len(request_input.encode("utf-8")),
            len(request.instructions.encode("utf-8")),
            len(json.dumps(
                request.response_schema,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")),
        ))
        reserved_cost = self.pricing.estimate(
            max(self.max_input_tokens_estimate, request_bytes),
            self.max_output_tokens,
        )
        self.budget.authorize(reserved_cost)
        client = self._client.with_options(timeout=self.timeout_seconds)
        self._submitted_call_count += 1
        try:
            response = client.responses.create(
                model=self.model,
                instructions=request.instructions,
                input=request_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "shopping_turn_plan",
                        "schema": request.response_schema,
                        "strict": True,
                    },
                },
                tools=[],
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except Exception as error:
            self._record_pessimistic_cost(reserved_cost)
            if (
                isinstance(error, TimeoutError)
                or error.__class__.__name__ == "APITimeoutError"
            ):
                raise TimeoutError("OpenAI planning request timed out") from error
            raise

        try:
            usage = getattr(response, "usage", None)
            input_tokens = _non_negative_token_count(usage, "input_tokens")
            output_tokens = _non_negative_token_count(usage, "output_tokens")
        except Exception:
            self._record_pessimistic_cost(reserved_cost)
            raise
        self.budget.record(self.pricing.estimate(input_tokens, output_tokens))
        status = getattr(response, "status", "completed")
        if status != "completed":
            raise PlanningSchemaError(
                f"OpenAI response status was {status or 'unknown'}"
            )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise PlanningSchemaError("OpenAI response did not contain output text")
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise PlanningSchemaError(
                "OpenAI structured response was not valid JSON"
            ) from error
        return ProviderResponse(
            output=output,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )

    def configuration(self) -> dict:
        """Return a secret-free, versioned identity for reports and manifests."""
        return {
            "api": "responses-v1",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
            "max_output_tokens": self.max_output_tokens,
            "structured_outputs": True,
            "store": False,
            "pricing_usd_per_million_tokens": {
                "input": self.pricing.input_per_million_usd,
                "output": self.pricing.output_per_million_usd,
            },
            "budget": self.budget.as_dict(),
        }

    def metrics(self) -> dict:
        return {
            "submitted_call_count": self._submitted_call_count,
            "pessimistically_accounted_call_count": (
                self._pessimistically_accounted_call_count
            ),
            "budget": self.budget.as_dict(),
        }

    def _record_pessimistic_cost(self, reserved_cost: float) -> None:
        self.budget.record(reserved_cost)
        self._pessimistically_accounted_call_count += 1


def _non_negative_token_count(usage: object, attribute: str) -> int:
    value = getattr(usage, attribute, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanningSchemaError(
            f"OpenAI usage {attribute} must be a non-negative integer"
        )
    return value
