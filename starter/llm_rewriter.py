"""Optional LLM conversational query rewriter for the retrieval stage.

Conversational query rewriting restates the accumulated dialog as one
standalone search query — the highest-evidence technique for session search.
The rewrite consumes only the customer's own words, the active constraint
summary, and the aggregate profile hint; it fails open to the agent's
deterministic accumulated-dialog query on timeout, malformed output, or
budget exhaustion, and is sized for sub-second latency (no reasoning,
tiny structured output).
"""
from __future__ import annotations

import json
import os

from starter.openai_planning import (
    BudgetExceededError,
    DevelopmentBudget,
    ModelPricing,
    _non_negative_token_count,
)

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "search_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["query", "search_terms"],
    "additionalProperties": False,
}

INSTRUCTIONS = (
    "You turn a shopping conversation into one standalone catalog search "
    "query. Combine everything the customer still wants into 'query' (a "
    "single natural search phrase covering product type and every stated "
    "requirement; omit anything they retracted), and put 3-6 additional "
    "likely catalog words (synonyms, category terms, attribute words) into "
    "'search_terms'. Be literal; never invent requirements the customer did "
    "not express."
)


class OpenAIQueryRewriter:
    """Stateless per-turn conversational query rewriting."""

    def __init__(
        self,
        *,
        model: str,
        pricing: ModelPricing,
        budget: DevelopmentBudget,
        timeout_seconds: float = 1.2,
        reasoning_effort: str = "none",
        max_output_tokens: int = 200,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.pricing = pricing
        self.budget = budget
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for query rewriting")
        from openai import OpenAI

        self._client = OpenAI(api_key=key, max_retries=0)
        self._call_count = 0
        self._failure_count = 0
        self._failure_causes: dict[str, int] = {}

    def configuration(self) -> dict:
        return {
            "stage": "query_rewriting",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
        }

    def metrics(self) -> dict:
        return {
            "call_count": self._call_count,
            "failure_count": self._failure_count,
            "failure_causes": dict(sorted(self._failure_causes.items())),
            "budget": self.budget.as_dict(),
        }

    def rewrite(
        self,
        dialog_messages: list[str],
        constraint_summary: str,
        profile_hint: str,
    ) -> tuple[str, int, int] | None:
        """Return (standalone query text, prompt_tokens, completion_tokens)."""
        conversation = "\n".join(
            f"Customer: {message.strip()}"
            for message in dialog_messages
            if message.strip()
        )
        if not conversation:
            return None
        request_input = conversation
        if constraint_summary:
            request_input += f"\nActive requirements: {constraint_summary}"
        if profile_hint:
            request_input += f"\n{profile_hint}"
        reserved = self.pricing.estimate(
            len(request_input.encode("utf-8")) + len(INSTRUCTIONS.encode("utf-8")),
            self.max_output_tokens,
        )
        try:
            self.budget.authorize(reserved)
        except BudgetExceededError:
            self._failure_count += 1
            return None
        self._call_count += 1
        try:
            response = self._client.with_options(
                timeout=self.timeout_seconds
            ).responses.create(
                model=self.model,
                instructions=INSTRUCTIONS,
                input=request_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "search_query_rewrite",
                        "schema": REWRITE_SCHEMA,
                        "strict": True,
                    },
                },
                tools=[],
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
            usage = getattr(response, "usage", None)
            input_tokens = _non_negative_token_count(usage, "input_tokens")
            output_tokens = _non_negative_token_count(usage, "output_tokens")
            self.budget.record(self.pricing.estimate(input_tokens, output_tokens))
            if getattr(response, "status", "completed") != "completed":
                raise ValueError("incomplete response")
            payload = json.loads(getattr(response, "output_text", "") or "")
            query = str(payload.get("query", "")).strip()
            terms = [
                str(term).strip()
                for term in payload.get("search_terms", [])
                if str(term).strip()
            ]
            if not query:
                raise ValueError("empty rewritten query")
        except Exception as error:  # noqa: BLE001 - stage must always fail open
            cause = f"{error.__class__.__name__}:{error}"[:120]
            self._failure_causes[cause] = self._failure_causes.get(cause, 0) + 1
            self._failure_count += 1
            try:
                self.budget.record(reserved)
            except Exception:  # noqa: BLE001 - budget exhaustion is not fatal
                pass
            return None
        combined = query if not terms else f"{query} {' '.join(terms[:6])}"
        return combined, input_tokens, output_tokens
