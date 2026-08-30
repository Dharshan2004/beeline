from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from retrieval.dense_route import DenseRetrievalRoute
from retrieval.fusion import FusionPolicy, build_fusion_policy
from retrieval.reranker import DEFAULT_RERANKER_DIR, RerankRoute
from starter.constraint_state import ConstraintState, PlanValidationError, TurnPlan
from starter.planning import (
    DEFAULT_RETRIEVAL_TOOLS,
    PlanningLoop,
    PlanningOutcome,
    PlanningProvider,
)
from starter.retrieval import CatalogRetrieval
from starter.turn_interpreter import interpret_turn


QUESTION_ORDER = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)
DENSE_CANDIDATE_DEPTH = 100

# The deep Candidate Pool handed to the cross-encoder, drawn from the base-route
# union rather than a pre-truncated fused top 30. Slice 07 freezes this number
# from its quality-versus-runtime rule; until that benchmark is run against a
# dense-enabled baseline these are provisional defaults taken from its
# preliminary sizing measurement. Changing the selection is a change here and
# in retrieval.reranker.DEFAULT_RERANKER_IDENTITY, nowhere else.
RERANK_CANDIDATE_DEPTH = 50

# A hard per-turn ceiling on cross-encoder work. Exceeding it abandons reranking
# for the turn and returns the fused ordering, so a slow host degrades ranking
# quality instead of risking a turn the evaluator scores as a miss.
RERANK_DEADLINE_SECONDS = 1.5


class DenseRoute(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...

    def metrics(self) -> dict: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        documents: Mapping[str, str],
    ) -> list[str]: ...

    def metrics(self) -> dict: ...


class Agent:
    """Offline Shopping Agent with validated Constraint State transitions."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        dense_route: DenseRoute | None = None,
        reranker: Reranker | None = None,
        fusion_policy: FusionPolicy | str = "fixed",
        planning_provider: PlanningProvider | None = None,
        candidate_pool_depth: int | None = None,
        trace_pool_depth: int | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retrieval = CatalogRetrieval(self.catalog_path)
        self.dense_route: DenseRoute = dense_route or DenseRetrievalRoute(
            self.catalog_path
        )
        self.reranker: Reranker = reranker or RerankRoute(
            DEFAULT_RERANKER_DIR,
            deadline_seconds=RERANK_DEADLINE_SECONDS,
        )
        self.fusion_policy = (
            build_fusion_policy(fusion_policy)
            if isinstance(fusion_policy, str)
            else fusion_policy
        )
        # The depth the response is drawn from, and an optional deeper pool
        # recorded for offline benchmarking. Tracing never changes the response,
        # so a cached benchmark run replays the shipped trajectory exactly.
        self.candidate_pool_depth = (
            RERANK_CANDIDATE_DEPTH
            if candidate_pool_depth is None
            else candidate_pool_depth
        )
        if self.candidate_pool_depth <= 0:
            raise ValueError("candidate_pool_depth must be positive")
        self.trace_pool_depth = trace_pool_depth
        self._candidate_traces: dict[str, list[dict]] = {}
        self._sessions: dict[str, ConstraintState] = {}
        self._last_asked_attributes: dict[str, str | None] = {}
        self._session_ids_by_state: dict[int, str] = {}
        self._planning_history: dict[str, list[dict]] = {}
        self.planning_loop = PlanningLoop(planning_provider)

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for later personalization.
        previous = self._sessions.get(session_id)
        if previous is not None:
            self._session_ids_by_state.pop(id(previous), None)
        state = ConstraintState()
        self._sessions[session_id] = state
        self._last_asked_attributes[session_id] = None
        self._planning_history[session_id] = []
        self._candidate_traces[session_id] = []
        self._session_ids_by_state[id(state)] = session_id

    def get_constraint_state(self, session_id: str) -> list[dict]:
        """Return a copy of the inspectable Constraint State history."""
        return [
            constraint.as_dict()
            for constraint in self._state(session_id).constraints
        ]

    def get_constraint_revision(self, session_id: str) -> int:
        return self._state(session_id).revision

    def get_dismissed_attributes(self, session_id: str) -> list[dict]:
        """Return Boundary Response history without exposing mutable state."""
        return [
            dict(dismissal)
            for dismissal in self._state(session_id).dismissed_attributes.values()
        ]

    def get_transition_history(self, session_id: str) -> list[dict]:
        """Return an inspectable copy of Intent Override and Boundary history."""
        return [
            dict(event)
            for event in self._state(session_id).transition_history
        ]

    def get_planning_history(self, session_id: str) -> list[dict]:
        """Return local planning evidence without provider conversation state."""
        self._state(session_id)
        return deepcopy(self._planning_history[session_id])

    def get_candidate_traces(self) -> dict[str, list[dict]]:
        """Return the recorded deep Candidate Pools, keyed by session identifier.

        Empty unless the Agent was constructed with a ``trace_pool_depth``. The
        insertion order matches the order sessions were reset.
        """
        return deepcopy(self._candidate_traces)

    def get_dense_route_metrics(self) -> dict:
        """Return measurable load/query evidence for the dense Retrieval Route."""
        return dict(self.dense_route.metrics())

    def get_reranker_metrics(self) -> dict:
        """Return load, deadline, and fallback evidence for the rerank stage."""
        return dict(self.reranker.metrics())

    def get_retrieval_configuration(self) -> dict:
        """Return the versioned retrieval settings used by scored turns."""
        configuration = {
            "policy_version": self.fusion_policy.version,
            "route_depth": DENSE_CANDIDATE_DEPTH,
            # Fusion now produces exactly the pool the cross-encoder reranks,
            # so this one depth is both the fused depth and the rerank boundary.
            "fused_candidate_depth": self.candidate_pool_depth,
            "rerank_deadline_seconds": RERANK_DEADLINE_SECONDS,
            "reranker_status": self.reranker.metrics().get("status"),
            "reranker_identity": self.reranker.metrics().get("identity"),
        }
        weights = getattr(self.fusion_policy, "weights", None)
        if weights is not None:
            configuration["weights"] = dict(weights)
        return configuration

    def _state(self, session_id: str) -> ConstraintState:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before reading constraint state")
        return self._sessions[session_id]

    def _interpret_turn(
        self,
        user_message: str,
        turn: int,
        state: ConstraintState,
    ) -> TurnPlan:
        return self._fallback_turn(user_message, turn, state)

    def _fallback_turn(
        self,
        user_message: str,
        turn: int,
        state: ConstraintState,
    ) -> TurnPlan:
        return interpret_turn(
            user_message,
            turn=turn,
            state=state,
            supported_values=self.retrieval.supported_values,
            last_asked_attribute=self._last_asked_attributes.get(
                self._session_ids_by_state.get(id(state), "")
            ),
        )

    def _next_ask_attribute(
        self,
        session_id: str,
        state: ConstraintState,
    ) -> str | None:
        active_attributes = {
            constraint.attribute
            for constraint in state.constraints
            if constraint.status == "active"
        }
        for attribute in QUESTION_ORDER:
            if not self.retrieval.supported_values.get(attribute):
                continue
            if attribute in active_attributes or attribute in state.dismissed_attributes:
                continue
            self._last_asked_attributes[session_id] = attribute
            return attribute
        self._last_asked_attributes[session_id] = None
        return None

    def _dense_query(self, user_message: str, state: ConstraintState) -> str:
        active_evidence = [
            f"{constraint.attribute}: {', '.join(constraint.values)}"
            for constraint in state.constraints
            if (
                constraint.status == "active"
                and constraint.classification == "hard"
            )
        ]
        if not active_evidence:
            return user_message
        return (
            f"{user_message.strip()}\n"
            f"Active constraints: {'; '.join(active_evidence)}"
        )

    def _rerank(self, query: str, candidates: list[str]) -> list[str]:
        """Reorder the deep Candidate Pool, falling back to the fused ordering.

        The rerank stage is never allowed to change which products are eligible,
        only their order, so a failure returns exactly the same candidate set.
        """
        if not candidates:
            return candidates
        ranked = self.reranker.rerank(
            query,
            candidates,
            self.retrieval.rerank_text,
        )
        if sorted(ranked) != sorted(candidates):
            # A reranker that dropped, duplicated, or invented an identifier has
            # broken the contract; the fused ordering is the safe answer.
            return candidates
        return ranked

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        if self.planning_loop.provider is None:
            outcome = PlanningOutcome(
                turn_plan=self._interpret_turn(user_message, turn, state),
                retrieval_tools=DEFAULT_RETRIEVAL_TOOLS,
                clarification=None,
                source="fallback",
                attempts=0,
                prompt_tokens=0,
                completion_tokens=0,
                fallback_reason="missing_credentials",
            )
        else:
            outcome = self.planning_loop.run(
                session_id=session_id,
                turn=turn,
                user_message=user_message,
                state=state,
                supported_values=self.retrieval.supported_values,
                recent_history=self._planning_history[session_id],
                fallback_plan=lambda: self._fallback_turn(
                    user_message,
                    turn,
                    state,
                ),
            )
        try:
            state.apply(outcome.turn_plan, self.retrieval.supported_values)
        except PlanValidationError:
            fallback = self._fallback_turn(user_message, turn, state)
            try:
                state.apply(fallback, self.retrieval.supported_values)
            except PlanValidationError:
                # Retrieval still uses the original unchanged state.
                pass

        selected_tools = set(outcome.retrieval_tools)
        dense_candidates = []
        if "dense" in selected_tools:
            try:
                dense_candidates = self.dense_route.search(
                    self._dense_query(user_message, state),
                    DENSE_CANDIDATE_DEPTH,
                )
            except Exception:  # noqa: BLE001 - optional route must fail open
                dense_candidates = []
        route_scores = self.retrieval.hybrid_route_scores(
            user_message,
            state.constraints,
            dense_candidates,
            route_limit=DENSE_CANDIDATE_DEPTH,
            enabled_routes=selected_tools.intersection(
                {"structured", "bm25", "dense"}
            ),
        )
        dense_query = self._dense_query(user_message, state)
        fused_candidates = self.fusion_policy.rank(
            route_scores,
            candidate_limit=self.candidate_pool_depth,
        )
        if not fused_candidates and outcome.source == "fallback":
            fused_candidates = self.retrieval.recommend(
                user_message,
                state.constraints,
                self.candidate_pool_depth,
            )
        if self.trace_pool_depth is not None:
            self._candidate_traces[session_id].append({
                "turn": turn,
                "query": dense_query,
                "pool": self.fusion_policy.rank(
                    route_scores,
                    candidate_limit=self.trace_pool_depth,
                ),
                "response_pool": list(fused_candidates),
            })
        recommendation_limit = max(0, min(top_k, 10))
        # The reranker sees the whole deep pool, not a pre-truncated top ten:
        # reordering only the products fusion already ranked first would leave
        # the deeper candidates it was introduced to reach permanently unread.
        ranked_candidates = self._rerank(dense_query, fused_candidates)
        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in ranked_candidates[:recommendation_limit]
        ]
        message = "Here are the closest matches I found."
        if outcome.source == "connected":
            ask_attribute = (
                outcome.clarification.ask_attribute
                if outcome.clarification is not None
                else None
            )
            if outcome.clarification is not None:
                message = f"{message} {outcome.clarification.message}"
                self._last_asked_attributes[session_id] = ask_attribute
        else:
            ask_attribute = self._next_ask_attribute(session_id, state)
            if ask_attribute is not None:
                message = (
                    f"{message} Do you have a preference for "
                    f"{ask_attribute.replace('_', ' ')}?"
                )
        self._planning_history[session_id].append({
            "turn": turn,
            "user_message": user_message,
            "state_revision": state.revision,
            "source": outcome.source,
            "attempts": outcome.attempts,
            "retrieval_tools": list(outcome.retrieval_tools),
            "ask_attribute": ask_attribute,
            "fallback_reason": outcome.fallback_reason,
            "errors": list(outcome.errors),
        })
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
            },
        }
