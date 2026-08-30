from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol

from retrieval.dense_route import DenseRetrievalRoute
from retrieval.fusion import FusionPolicy, build_fusion_policy
from starter.constraint_state import ConstraintState, PlanValidationError, TurnPlan
from starter.planning import (
    DEFAULT_RETRIEVAL_TOOLS,
    PLANNING_PROMPT_VERSION,
    PlanningLoop,
    PlanningOutcome,
    PlanningProvider,
)
from starter.retrieval import CatalogRetrieval
from starter.telemetry import Tracer
from starter.turn_interpreter import interpret_turn


QUESTION_ORDER = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)
DENSE_CANDIDATE_DEPTH = 100


def reason_codes(errors: tuple[str, ...] | list[str]) -> list[str]:
    """Return only the error identity of each rejected planning attempt.

    Planning error text can quote customer phrasing, so telemetry keeps the
    classified cause and discards the message.
    """
    return [str(error).split(":", 1)[0].strip() for error in errors]


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
        planning_provider: PlanningProvider | None = None,
        tracer: Tracer | None = None,
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
        self._planning_history: dict[str, list[dict]] = {}
        self.planning_loop = PlanningLoop(planning_provider)
        self.tracer = tracer if tracer is not None else Tracer.from_environment()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for later personalization.
        # It is never recorded in telemetry.
        previous = self._sessions.get(session_id)
        if previous is not None:
            self._session_ids_by_state.pop(id(previous), None)
        state = ConstraintState()
        self._sessions[session_id] = state
        self._last_asked_attributes[session_id] = None
        self._planning_history[session_id] = []
        self._session_ids_by_state[id(state)] = session_id
        self.tracer.start_session(session_id, self.get_configuration_identity())

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

    def get_configuration_identity(self) -> dict:
        """Return the versioned identity every trace is grouped under."""
        dense_metrics = self.get_dense_route_metrics()
        return {
            "retrieval": self.get_retrieval_configuration(),
            "planning_prompt_version": PLANNING_PROMPT_VERSION,
            "planning_source": (
                "connected" if self.planning_loop.provider is not None else "local"
            ),
            "dense_route_status": dense_metrics.get("status"),
            "dense_route_disabled_reason": dense_metrics.get("disabled_reason"),
            "catalog_product_count": len(self.retrieval.product_text),
        }

    def get_telemetry_metrics(self) -> dict:
        """Return export evidence without exposing buffered trace payloads."""
        return self.tracer.metrics()

    def flush_telemetry(self) -> bool:
        """Export buffered traces. Safe to call at evaluator completion."""
        return self.tracer.flush()

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
        with self.tracer.turn(session_id, turn) as trace:
            return self._respond_traced(
                session_id,
                user_message,
                turn,
                top_k,
                state,
                trace,
            )

    def _respond_traced(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
        state: ConstraintState,
        trace,
    ) -> dict:
        with trace.operation("interpretation") as observation:
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
            observation.record(
                source=outcome.source,
                attempts=outcome.attempts,
                fallback_reason=outcome.fallback_reason,
                reason_codes=reason_codes(outcome.errors),
                retrieval_tools=list(outcome.retrieval_tools),
                mutation_count=len(outcome.turn_plan.mutations),
                clarification_requested=outcome.clarification is not None,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
            )

        with trace.operation("state_validation") as observation:
            revision_before = state.revision
            applied = "turn_plan"
            try:
                state.apply(outcome.turn_plan, self.retrieval.supported_values)
            except PlanValidationError:
                applied = "fallback_plan"
                fallback = self._fallback_turn(user_message, turn, state)
                try:
                    state.apply(fallback, self.retrieval.supported_values)
                except PlanValidationError:
                    # Retrieval still uses the original unchanged state.
                    applied = "unchanged_state"
            observation.record(
                applied_plan=applied,
                revision_before=revision_before,
                revision_after=state.revision,
                active_constraint_count=sum(
                    1 for item in state.constraints if item.status == "active"
                ),
                dismissed_attribute_count=len(state.dismissed_attributes),
                constraint_decisions=[
                    {
                        "constraint_id": item.constraint_id,
                        "attribute": item.attribute,
                        "values": list(item.values),
                        "match_rule": item.match_rule,
                        "classification": item.classification,
                        "scope": item.scope,
                        "status": item.status,
                        "confidence": item.confidence,
                        "source_turn": item.source_turn,
                    }
                    for item in state.constraints
                ],
            )
            if applied == "fallback_plan":
                observation.fail("rejected_turn_plan")
            elif applied == "unchanged_state":
                observation.fail("rejected_fallback_plan")

        selected_tools = set(outcome.retrieval_tools)
        with trace.operation("retrieval") as observation:
            observation.record(requested_routes=sorted(selected_tools))
            dense_candidates = []
            with trace.operation("retrieval.dense") as dense_observation:
                dense_requested = "dense" in selected_tools
                dense_observation.record(requested=dense_requested)
                if dense_requested:
                    try:
                        dense_candidates = self.dense_route.search(
                            self._dense_query(user_message, state),
                            DENSE_CANDIDATE_DEPTH,
                        )
                    except Exception as error:  # noqa: BLE001 - must fail open
                        dense_candidates = []
                        dense_observation.fail(type(error).__name__)
                    dense_observation.record(
                        candidate_count=len(dense_candidates),
                        route_status=self._dense_route_status(),
                    )
            route_scores = self.retrieval.hybrid_route_scores(
                user_message,
                state.constraints,
                dense_candidates,
                route_limit=DENSE_CANDIDATE_DEPTH,
                enabled_routes=selected_tools.intersection(
                    {"structured", "bm25", "dense"}
                ),
            )
            observation.record(
                route_depth=DENSE_CANDIDATE_DEPTH,
                candidate_counts={
                    route: len(candidates)
                    for route, candidates in route_scores.items()
                },
            )

        with trace.operation("fusion") as observation:
            fused_candidates = self.fusion_policy.rank(route_scores)
            fusion_backfill = not fused_candidates and outcome.source == "fallback"
            if fusion_backfill:
                fused_candidates = self.retrieval.recommend(
                    user_message,
                    state.constraints,
                    self.fusion_policy.candidate_limit,
                )
            observation.record(
                fused_candidate_count=len(fused_candidates),
                deterministic_backfill=fusion_backfill,
                **self.get_retrieval_configuration(),
            )

        with trace.operation("reranking") as observation:
            # The local reranker arrives with Slice 08. Recording the requested
            # tool keeps a connected plan that asks for it diagnosable.
            observation.record(
                requested="local_rerank" in selected_tools,
                applied=False,
                skipped_reason="reranker_not_enabled",
            )

        recommendation_limit = max(0, min(top_k, 10))
        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in fused_candidates[:recommendation_limit]
        ]
        message = "Here are the closest matches I found."
        with trace.operation("clarification") as observation:
            if outcome.source == "connected":
                ask_attribute = (
                    outcome.clarification.ask_attribute
                    if outcome.clarification is not None
                    else None
                )
                clarification_source = "connected_plan"
                if outcome.clarification is not None:
                    message = f"{message} {outcome.clarification.message}"
                    self._last_asked_attributes[session_id] = ask_attribute
            else:
                clarification_source = "question_order"
                ask_attribute = self._next_ask_attribute(session_id, state)
                if ask_attribute is not None:
                    message = (
                        f"{message} Do you have a preference for "
                        f"{ask_attribute.replace('_', ' ')}?"
                    )
            observation.record(
                clarification_source=clarification_source,
                ask_attribute=ask_attribute,
                asked=ask_attribute is not None,
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
        response = {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": outcome.prompt_tokens,
                "completion_tokens": outcome.completion_tokens,
            },
        }
        with trace.operation("response") as observation:
            observation.record(
                requested_top_k=top_k,
                recommendation_limit=recommendation_limit,
                recommendation_count=len(recommendations),
                message_characters=len(message),
                usage=dict(response["usage"]),
            )
        trace.record(
            source=outcome.source,
            fallback_reason=outcome.fallback_reason,
            state_revision=state.revision,
            recommendation_count=len(recommendations),
            ask_attribute=ask_attribute,
        )
        return response

    def _dense_route_status(self) -> str | None:
        try:
            return self.dense_route.metrics().get("status")
        except Exception:  # noqa: BLE001 - metrics must never change behavior
            return None
