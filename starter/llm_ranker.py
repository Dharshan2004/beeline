"""Optional LLM semantic-ranking stage for the final Candidate Pool head.

Implements the "Multi-Route Retrieval -> LLM Semantic Ranking" pillar: after
local fusion and cross-encoder reranking, a language model reorders the top
candidates against the accumulated customer dialog. The stage consumes only
the customer's own messages, the validated constraint summary, and public
catalog renderings of the candidates — no session identifiers, no evaluator
knowledge, no templates.

Safety and cost posture mirrors the connected planning adapter: explicit
opt-in, a hard development budget with worst-case reservation before every
call, `store=false`, a strict JSON-schema response, and unconditional fail-open
to the local ordering on timeout, malformed output, budget exhaustion, or any
provider error.
"""
from __future__ import annotations

import json
import os
import sys

from starter.openai_planning import (
    BudgetExceededError,
    DevelopmentBudget,
    ModelPricing,
    _non_negative_token_count,
)

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["ranking"],
    "additionalProperties": False,
}

INSTRUCTIONS = (
    "You are the final ranking stage of a shopping assistant. You receive the "
    "customer's conversation so far and a numbered list of candidate "
    "products, already ordered best-first by a local relevance model. Reorder "
    "the candidate numbers from most to least likely to be exactly the "
    "product this customer is trying to buy, weighing every requirement and "
    "preference they have expressed, including price limits. Keep a "
    "candidate's existing position unless the conversation gives you a "
    "concrete reason to move it. Return JSON with a 'ranking' array that "
    "lists every candidate number exactly once, best first."
)


class RankingUnavailable(Exception):
    """Raised internally when a ranking cannot be produced this turn."""


# The shipped connected configuration: gpt-5.4-nano chunk tournament with the
# gate-validated tight timeouts (exact 0.806 / paraphrase 0.763 / novel 0.756,
# p95 3.4 s per turn).
DEFAULT_NANO_CONFIG_PATH = "config/semantic_ranker_nano.json"
DEFAULT_CHUNK_TIMEOUT_SECONDS = 1.2
DEFAULT_FINAL_TIMEOUT_SECONDS = 1.6


def _openai_key_available(env_file: str) -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        with open(env_file, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("OPENAI_API_KEY="):
                    value = stripped.split("=", 1)[1].strip().strip("'\"")
                    if value:
                        os.environ.setdefault("OPENAI_API_KEY", value)
                        return True
    except OSError:
        return False
    return False


def build_default_tournament_ranker(
    config_path: str = DEFAULT_NANO_CONFIG_PATH,
    env_file: str = ".env",
):
    """Best-effort construction of the shipped connected ranking stage.

    Returns the gate-validated nano tournament when an OpenAI key is
    available (environment or .env) and the shipped configuration loads;
    returns None — plain offline mode — otherwise. Never raises, so the
    offline agent remains the guaranteed worst case. Set BEELINE_OFFLINE=1
    to force offline mode regardless of credentials.
    """
    if os.environ.get("BEELINE_OFFLINE", "").strip() not in ("", "0"):
        return None
    # Unit tests must stay offline, free, and deterministic no matter how the
    # suite is invoked. The launch command is the reliable signal (module
    # presence is not: torch imports unittest in every process).
    argv0 = sys.argv[0] if sys.argv else ""
    argv0_name = os.path.basename(argv0)
    if (
        "unittest" in argv0
        or "pytest" in argv0_name
        or argv0_name.startswith("test_")
    ):
        return None
    try:
        if not _openai_key_available(env_file):
            return None
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        model_entry = next(
            entry for entry in config["models"] if entry["role"] == "nano_ranker"
        )
        rates = model_entry["pricing_usd_per_million_tokens"]
        pricing = ModelPricing(
            input_per_million_usd=rates["input"],
            output_per_million_usd=rates["output"],
        )
        budget_config = config["budget_usd"]

        def stage(timeout_seconds: float) -> "OpenAISemanticRanker":
            return OpenAISemanticRanker(
                model=model_entry["model"],
                pricing=pricing,
                budget=DevelopmentBudget(
                    limit_usd=budget_config["phase_a_limit"],
                    warning_usd=budget_config["warning"],
                    review_boundary_usd=budget_config["review_boundary"],
                    absolute_stop_usd=budget_config["absolute_stop"],
                ),
                reasoning_effort=config["api"]["reasoning_effort"],
                timeout_seconds=timeout_seconds,
                max_output_tokens=config["api"]["max_output_tokens"],
                max_candidates=12,
            )

        return TournamentSemanticRanker(
            stage(DEFAULT_CHUNK_TIMEOUT_SECONDS),
            stage(DEFAULT_FINAL_TIMEOUT_SECONDS),
        )
    except Exception:
        return None


class TournamentSemanticRanker:
    """Parallel chunked LLM ranking so the whole pool gets a semantic read.

    A single listwise call only sees the pool head; at a fixed wall-clock
    budget, concurrency buys coverage instead of time. The pool is split into
    chunks ranked simultaneously by a fast model, each chunk's leaders advance
    to one final listwise call, and every node fails open: a dead chunk keeps
    its local order, a dead final keeps the concatenated chunk leaders.
    """

    def __init__(
        self,
        chunk_ranker: "OpenAISemanticRanker",
        final_ranker: "OpenAISemanticRanker",
        *,
        chunk_size: int = 12,
        chunk_count: int = 4,
        finalists_per_chunk: int = 3,
    ) -> None:
        self.chunk_ranker = chunk_ranker
        self.final_ranker = final_ranker
        self.chunk_size = chunk_size
        self.chunk_count = chunk_count
        self.finalists_per_chunk = finalists_per_chunk
        self.max_candidates = chunk_size * chunk_count

    def configuration(self) -> dict:
        return {
            "stage": "semantic_ranking_tournament",
            "chunk_ranker": self.chunk_ranker.configuration(),
            "final_ranker": self.final_ranker.configuration(),
            "chunk_size": self.chunk_size,
            "chunk_count": self.chunk_count,
            "finalists_per_chunk": self.finalists_per_chunk,
        }

    def metrics(self) -> dict:
        return {
            "chunk_ranker": self.chunk_ranker.metrics(),
            "final_ranker": self.final_ranker.metrics(),
        }

    def rank(
        self,
        dialog_messages: list[str],
        constraint_summary: str,
        candidates: list[tuple[str, str]],
    ) -> tuple[list[str], int, int] | None:
        from concurrent.futures import ThreadPoolExecutor

        head = candidates[: self.max_candidates]
        if len(head) < 2:
            return None
        chunks = [
            head[start:start + self.chunk_size]
            for start in range(0, len(head), self.chunk_size)
        ]
        prompt_tokens = 0
        completion_tokens = 0
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(
                    self.chunk_ranker.rank,
                    dialog_messages,
                    constraint_summary,
                    chunk,
                )
                for chunk in chunks
            ]
            chunk_results = [future.result() for future in futures]
        ordered_chunks: list[list[str]] = []
        for chunk, result in zip(chunks, chunk_results):
            if result is None:
                ordered_chunks.append([parent_asin for parent_asin, _ in chunk])
            else:
                chunk_order, chunk_prompt, chunk_completion = result
                prompt_tokens += chunk_prompt
                completion_tokens += chunk_completion
                ordered_chunks.append(chunk_order)
        finalists: list[str] = []
        for ordered in ordered_chunks:
            finalists.extend(ordered[: self.finalists_per_chunk])
        remainder: list[str] = []
        for ordered in ordered_chunks:
            remainder.extend(ordered[self.finalists_per_chunk:])
        text_by_id = dict(head)
        final_result = self.final_ranker.rank(
            dialog_messages,
            constraint_summary,
            [(parent_asin, text_by_id[parent_asin]) for parent_asin in finalists],
        )
        if final_result is not None:
            final_order, final_prompt, final_completion = final_result
            prompt_tokens += final_prompt
            completion_tokens += final_completion
            finalists = final_order
        if prompt_tokens == 0 and completion_tokens == 0:
            return None
        return [*finalists, *remainder], prompt_tokens, completion_tokens


class OpenAISemanticRanker:
    """Stateless per-turn candidate reordering through the Responses API."""

    def __init__(
        self,
        *,
        model: str,
        pricing: ModelPricing,
        budget: DevelopmentBudget,
        timeout_seconds: float = 6.0,
        reasoning_effort: str = "low",
        max_output_tokens: int = 300,
        max_candidates: int = 20,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.pricing = pricing
        self.budget = budget
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_candidates = max_candidates
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for semantic ranking")
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "the openai package is required; install requirements-openai.txt"
            ) from error
        # No SDK auto-retries: a slow provider turn fails open to the local
        # ordering instead of stretching customer-facing latency.
        self._client = OpenAI(api_key=key, max_retries=0)
        self._call_count = 0
        self._failure_count = 0
        self._failure_causes: dict[str, int] = {}

    def configuration(self) -> dict:
        return {
            "stage": "semantic_ranking",
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "max_candidates": self.max_candidates,
        }

    def metrics(self) -> dict:
        return {
            "call_count": self._call_count,
            "failure_count": self._failure_count,
            "failure_causes": dict(sorted(self._failure_causes.items())),
            "budget": self.budget.as_dict(),
        }

    def rank(
        self,
        dialog_messages: list[str],
        constraint_summary: str,
        candidates: list[tuple[str, str]],
    ) -> tuple[list[str], int, int] | None:
        """Return (reordered head, prompt_tokens, completion_tokens) or None.

        The returned ordering contains exactly the input candidates: model
        picks come first in model order, and anything the model omitted keeps
        its local relative order behind them. Any failure returns None and the
        caller keeps the local ordering.
        """
        head = candidates[: self.max_candidates]
        if len(head) < 2:
            return None
        conversation = "\n".join(
            f"Customer: {message.strip()}"
            for message in dialog_messages
            if message.strip()
        )
        lines = [
            f"{index + 1}. {text}" for index, (_, text) in enumerate(head)
        ]
        request_input = (
            f"Conversation so far (oldest first):\n{conversation}\n\n"
            + (
                f"Structured requirements: {constraint_summary}\n\n"
                if constraint_summary
                else ""
            )
            + "Candidate products:\n" + "\n".join(lines)
        )
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
                        "name": "candidate_ranking",
                        "schema": RANKING_SCHEMA,
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
            status = getattr(response, "status", "completed")
            if status != "completed":
                detail = getattr(response, "incomplete_details", None)
                reason = getattr(detail, "reason", None) or status
                raise RankingUnavailable(f"incomplete:{reason}")
            output_text = getattr(response, "output_text", "") or ""
            ranking = json.loads(output_text).get("ranking")
            if not isinstance(ranking, list):
                raise RankingUnavailable("missing ranking array")
        except Exception as error:  # noqa: BLE001 - this stage must always fail open
            cause = f"{error.__class__.__name__}:{error}"[:120]
            self._failure_causes[cause] = self._failure_causes.get(cause, 0) + 1
            self._failure_count += 1
            try:
                # Pessimistic charge: a failed call still spends its
                # reservation so the budget cannot be overrun by retries.
                self.budget.record(reserved)
            except Exception:  # noqa: BLE001 - budget exhaustion is not fatal
                pass
            return None
        ordered: list[str] = []
        seen: set[int] = set()
        for value in ranking:
            if isinstance(value, int) and 1 <= value <= len(head) and value not in seen:
                seen.add(value)
                ordered.append(head[value - 1][0])
        for index, (parent_asin, _) in enumerate(head):
            if (index + 1) not in seen:
                ordered.append(parent_asin)
        return ordered, input_tokens, output_tokens
