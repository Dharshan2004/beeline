from __future__ import annotations

from pathlib import Path
from typing import Protocol

from retrieval.dense_route import DenseRetrievalRoute
from retrieval.fusion import FusionPolicy, build_fusion_policy
from starter.constraint_state import ConstraintState, PlanValidationError, TurnPlan
from starter.retrieval import CatalogRetrieval
from starter.turn_interpreter import interpret_turn


QUESTION_ORDER = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)
DENSE_CANDIDATE_DEPTH = 100


class DenseRoute(Protocol):
    def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...

    def metrics(self) -> dict: ...


class Agent:
    """Offline Shopping Agent with validated Constraint State transitions."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        dense_route: DenseRoute | None = None,
        fusion_policy: FusionPolicy | str = "fixed",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.retrieval = CatalogRetrieval(self.catalog_path)
        self.dense_route: DenseRoute = dense_route or DenseRetrievalRoute(
            self.catalog_path
        )
        self.fusion_policy = (
            build_fusion_policy(fusion_policy)
            if isinstance(fusion_policy, str)
            else fusion_policy
        )
        self._sessions: dict[str, ConstraintState] = {}
        self._last_asked_attributes: dict[str, str | None] = {}
        self._session_ids_by_state: dict[int, str] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for later personalization.
        previous = self._sessions.get(session_id)
        if previous is not None:
            self._session_ids_by_state.pop(id(previous), None)
        state = ConstraintState()
        self._sessions[session_id] = state
        self._last_asked_attributes[session_id] = None
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

    def get_dense_route_metrics(self) -> dict:
        """Return measurable load/query evidence for the dense Retrieval Route."""
        return dict(self.dense_route.metrics())

    def get_retrieval_configuration(self) -> dict:
        """Return the versioned retrieval settings used by scored turns."""
        configuration = {
            "policy_version": self.fusion_policy.version,
            "route_depth": DENSE_CANDIDATE_DEPTH,
            "fused_candidate_depth": self.fusion_policy.candidate_limit,
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
        plan = self._interpret_turn(user_message, turn, state)
        try:
            state.apply(plan, self.retrieval.supported_values)
        except PlanValidationError:
            fallback = self._fallback_turn(user_message, turn, state)
            try:
                state.apply(fallback, self.retrieval.supported_values)
            except PlanValidationError:
                # Retrieval still uses the original unchanged state.
                pass

        try:
            dense_candidates = self.dense_route.search(
                user_message,
                DENSE_CANDIDATE_DEPTH,
            )
        except Exception:  # noqa: BLE001 - an optional route cannot invalidate a turn
            dense_candidates = []
        route_scores = self.retrieval.hybrid_route_scores(
            user_message,
            state.constraints,
            dense_candidates,
            route_limit=DENSE_CANDIDATE_DEPTH,
        )
        fused_candidates = self.fusion_policy.rank(route_scores)
        recommendation_limit = max(0, min(top_k, 10))
        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in fused_candidates[:recommendation_limit]
        ]
        ask_attribute = self._next_ask_attribute(session_id, state)
        message = "Here are the closest matches I found."
        if ask_attribute is not None:
            message = (
                f"{message} Do you have a preference for "
                f"{ask_attribute.replace('_', ' ')}?"
            )
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
