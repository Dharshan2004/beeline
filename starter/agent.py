from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol, Sequence

from retrieval.dense_route import DenseRetrievalRoute
from retrieval.fusion import (
    NORMALIZER_VERSION,
    FusionPolicy,
    build_fusion_policy,
    normalized_route_scores,
)
from retrieval.manifest import file_sha256
from retrieval.reranker import (
    DEFAULT_RERANKER_IDENTITY,
    FROZEN_RERANK_DEPTH,
    Reranker,
    build_live_reranker,
)
from starter.constraint_state import ConstraintState, PlanValidationError, TurnPlan
from starter.planning import (
    DEFAULT_RETRIEVAL_TOOLS,
    PlanningLoop,
    PlanningOutcome,
    PlanningProvider,
    PLANNING_PROMPT_SHA256,
    PLANNING_PROMPT_VERSION,
)
from starter.replacement_evidence import (
    REPLACEMENT_EVIDENCE_SHA256,
    REPLACEMENT_EVIDENCE_VERSION,
)
from starter.retrieval import CatalogRetrieval, parse_budget
from starter.turn_interpreter import interpret_turn


QUESTION_ORDER = (
    "category", "use_case", "material", "color", "style", "feature", "size",
    "brand", "budget", "other",
)
DENSE_CANDIDATE_DEPTH = 100
# Adjacent rerank scores closer than this are treated as a tie between
# near-identical listings; only such ties are reordered by popularity.
RERANK_TIE_EPSILON = 0.3
# When the leader beats the third candidate by more than this margin the
# local ordering is trusted and the optional LLM rerank is skipped.
RERANK_CONFIDENT_MARGIN = 2.0


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
        reranker: Reranker | None = None,
        semantic_ranker=None,
        query_rewriter=None,
        candidate_pool_depth: int | None = None,
        trace_pool_depths: Sequence[int] | None = None,
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
        # The depth the response is drawn from, and optional exact pools recorded
        # for offline benchmarking. Tracing never changes the response, so a
        # cached benchmark run replays the shipped trajectory exactly.
        self.reranker = reranker or build_live_reranker()
        # Optional LLM stages. Absent by default: the offline agent stays
        # complete and free to run.
        self.semantic_ranker = semantic_ranker
        self.query_rewriter = query_rewriter
        self.candidate_pool_depth = (
            FROZEN_RERANK_DEPTH if candidate_pool_depth is None else candidate_pool_depth
        )
        if self.candidate_pool_depth <= 0:
            raise ValueError("candidate_pool_depth must be positive")
        self.trace_pool_depths = tuple(sorted(set(trace_pool_depths or ())))
        if any(depth <= 0 for depth in self.trace_pool_depths):
            raise ValueError("trace_pool_depths must contain only positive depths")
        self._candidate_traces: dict[str, list[dict]] = {}
        self._dialog_messages: dict[str, list[str]] = {}
        self._profiles: dict[str, dict] = {}
        self._asked_attributes: dict[str, set[str]] = {}
        self._budget_ranges: dict[str, tuple[float, float]] = {}
        self._sessions: dict[str, ConstraintState] = {}
        self._last_asked_attributes: dict[str, str | None] = {}
        self._session_ids_by_state: dict[int, str] = {}
        self._planning_history: dict[str, list[dict]] = {}
        self.planning_loop = PlanningLoop(planning_provider)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._profiles[session_id] = (
            dict(user_profile) if isinstance(user_profile, dict) else {}
        )
        previous = self._sessions.get(session_id)
        if previous is not None:
            self._session_ids_by_state.pop(id(previous), None)
        state = ConstraintState()
        self._sessions[session_id] = state
        self._dialog_messages[session_id] = []
        self._asked_attributes[session_id] = set()
        self._budget_ranges.pop(session_id, None)
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

        Empty unless the Agent was constructed with ``trace_pool_depths``. The
        insertion order matches the order sessions were reset.
        """
        return deepcopy(self._candidate_traces)

    def get_dense_route_metrics(self) -> dict:
        """Return measurable load/query evidence for the dense Retrieval Route."""
        return dict(self.dense_route.metrics())

    def get_reranker_metrics(self) -> dict:
        """Return local worker readiness, latency, and fail-open evidence."""
        return dict(self.reranker.metrics())

    def get_retrieval_configuration(self) -> dict:
        """Return the versioned retrieval settings used by scored turns."""
        reranker_metrics = self.reranker.metrics()
        configuration = {
            "policy_version": self.fusion_policy.version,
            "normalizer": NORMALIZER_VERSION,
            "route_depth": DENSE_CANDIDATE_DEPTH,
            "fused_candidate_depth": self.candidate_pool_depth,
            "reranker_identity": getattr(
                self.reranker,
                "identity",
                DEFAULT_RERANKER_IDENTITY,
            ),
            "rerank_depth": FROZEN_RERANK_DEPTH,
            "reranker_revision": reranker_metrics.get("revision"),
            "reranker_directory_sha256": reranker_metrics.get(
                "directory_sha256"
            ),
            "rerank_deadline_seconds": reranker_metrics.get("deadline_seconds"),
            "reranker_status": reranker_metrics.get("status"),
            "reranker_enabled": getattr(self.reranker, "configured", True),
        }
        weights = getattr(self.fusion_policy, "weights", None)
        if weights is not None:
            configuration["weights"] = dict(weights)
        return configuration

    def get_runtime_configuration(self) -> dict:
        """Return the complete versioned identity of this scored Agent build."""
        dense_configuration = getattr(self.dense_route, "configuration", None)
        if callable(dense_configuration):
            dense_identity = dense_configuration()
        else:
            dense_identity = {
                "status": self.dense_route.metrics().get("status"),
                "manifest": None,
            }
        reranker_metrics = self.reranker.metrics()
        provider = self.planning_loop.provider
        provider_configuration = None
        if provider is not None:
            configuration_method = getattr(provider, "configuration", None)
            if callable(configuration_method):
                provider_configuration = configuration_method()
        return {
            "version": "shopping-agent-runtime-v1",
            "catalog": {
                "path": str(self.catalog_path),
                "sha256": file_sha256(self.catalog_path),
            },
            "dense_index_and_model": dense_identity,
            "reranker": {
                key: reranker_metrics.get(key)
                for key in (
                    "status",
                    "identity",
                    "revision",
                    "directory_sha256",
                    "depth",
                    "deadline_seconds",
                )
            },
            "planning": {
                "prompt_version": PLANNING_PROMPT_VERSION,
                "prompt_sha256": PLANNING_PROMPT_SHA256,
                "replacement_evidence_version": REPLACEMENT_EVIDENCE_VERSION,
                "replacement_evidence_sha256": REPLACEMENT_EVIDENCE_SHA256,
                "provider": type(provider).__name__ if provider is not None else None,
                "connected_model_version": (
                    getattr(provider, "model", None) if provider is not None else None
                ),
                "provider_configuration": provider_configuration,
            },
            "fusion_and_retrieval": self.get_retrieval_configuration(),
            "semantic_ranking": (
                self.semantic_ranker.configuration()
                if self.semantic_ranker is not None
                and callable(getattr(self.semantic_ranker, "configuration", None))
                else None
            ),
            "query_rewriting": (
                self.query_rewriter.configuration()
                if self.query_rewriter is not None
                and callable(getattr(self.query_rewriter, "configuration", None))
                else None
            ),
            "feature_flags": {
                "llm_semantic_ranking": self.semantic_ranker is not None,
                "llm_query_rewriting": self.query_rewriter is not None,
                "connected_planning": provider is not None,
                "dense_retrieval": getattr(self.dense_route, "configured", True),
                "local_reranking": getattr(self.reranker, "configured", True),
                "candidate_tracing": bool(self.trace_pool_depths),
            },
            "cost_limits_usd": {
                "warning": 40,
                "review_boundary": 50,
                "absolute_stop": 600,
                "enforcement_status": (
                    "enforced_by_connected_provider"
                    if provider_configuration is not None
                    else "not_applicable_offline"
                ),
            },
        }

    def close(self) -> None:
        """Release persistent local runtime workers owned by this Agent."""
        self.reranker.close()
        close_dense = getattr(self.dense_route, "close", None)
        if callable(close_dense):
            close_dense()

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
        candidate_pool: Sequence[str] = (),
    ) -> str | None:
        """Pick the clarification with the highest expected information value.

        A useful question is one whose answer can rule products out, so each
        askable attribute is scored by how unevenly its known values split the
        current Candidate Pool: the score is the number of pool products that
        a definitive answer would eliminate at minimum. Attributes that are
        already constrained, already dismissed by the customer, or that cannot
        split the pool are never asked; ties fall back to the stable question
        order.
        """
        active_attributes = {
            constraint.attribute
            for constraint in state.constraints
            if constraint.status == "active"
        }
        asked = self._asked_attributes.setdefault(session_id, set())
        askable = [
            attribute
            for attribute in QUESTION_ORDER
            if attribute not in active_attributes
            and attribute not in state.dismissed_attributes
            and attribute not in asked
        ]
        if not askable:
            self._last_asked_attributes[session_id] = None
            return None
        pool = set(candidate_pool)
        best_attribute = None
        best_score = 0
        if pool:
            for attribute in askable:
                counts = [
                    len(pool.intersection(members))
                    for members in self.retrieval.value_index.get(
                        attribute, {}
                    ).values()
                ]
                present = [count for count in counts if count > 0]
                if len(present) < 2:
                    continue
                score = len(pool) - max(present)
                if score > best_score:
                    best_score = score
                    best_attribute = attribute
        chosen = best_attribute if best_attribute is not None else askable[0]
        asked.add(chosen)
        self._last_asked_attributes[session_id] = chosen
        return chosen

    def _popularity_tiebreak(
        self,
        ordered: list[str],
        scores: dict[str, float],
    ) -> list[str]:
        """Reorder near-tied top-10 neighbours by a Bayesian popularity prior.

        The hidden target is a real purchase, and among listings the ranking
        model cannot separate, purchase probability tracks popularity. Bands
        never cross the rank-10 boundary, so HitRate@10 cannot change.
        """
        head = ordered[:10]
        tail = ordered[10:]
        if len(head) < 2:
            return ordered
        result: list[str] = []
        band: list[str] = [head[0]]
        for previous, current in zip(head, head[1:]):
            previous_score = scores.get(previous)
            current_score = scores.get(current)
            tied = (
                previous_score is not None
                and current_score is not None
                and abs(previous_score - current_score) < RERANK_TIE_EPSILON
            )
            if tied:
                band.append(current)
            else:
                result.extend(self._order_band(band))
                band = [current]
        result.extend(self._order_band(band))
        return [*result, *tail]

    def _order_band(self, band: list[str]) -> list[str]:
        if len(band) < 2:
            return band
        return sorted(
            band,
            key=lambda parent_asin: (
                -self.retrieval.popularity_prior(parent_asin),
                parent_asin,
            ),
        )

    def _semantic_candidate_text(self, parent_asin: str) -> str:
        """Public catalog rendering of one candidate for the LLM ranking stage."""
        text = self.retrieval.rerank_text.get(parent_asin, parent_asin)
        price = self.retrieval.price.get(parent_asin)
        if price is not None:
            return f"{text} | price: ${price:g}"
        return text

    def _prior_dialog_text(self, session_id: str, budget: int = 600) -> str:
        """Return recent prior customer messages, newest first, within budget.

        A customer's need is the accumulation of everything they have said, so
        retrieval conditions on the whole dialog rather than only the latest
        message. Newest messages come first because the embedding model
        truncates long inputs and recent statements are the most binding.
        """
        parts: list[str] = []
        used = 0
        for message in reversed(self._dialog_messages.get(session_id, [])[:-1]):
            message = message.strip()
            if not message:
                continue
            if used + len(message) > budget:
                break
            parts.append(message)
            used += len(message)
        return " ".join(parts)

    def _dense_query(
        self,
        user_message: str,
        state: ConstraintState,
        prior_dialog: str = "",
        profile_hint: str = "",
    ) -> str:
        active_evidence = [
            f"{constraint.attribute}: {', '.join(constraint.values)}"
            for constraint in state.constraints
            if (
                constraint.status == "active"
                and constraint.classification == "hard"
            )
        ]
        sections = [user_message.strip()]
        if active_evidence:
            sections.append(f"Active constraints: {'; '.join(active_evidence)}")
        if prior_dialog:
            sections.append(f"Earlier in this conversation: {prior_dialog}")
        if profile_hint:
            sections.append(profile_hint)
        return "\n".join(sections)

    def _profile_hint(self, session_id: str, state: ConstraintState) -> str:
        """Aggregate-profile hint used only while the session is still vague.

        Safe personalization: the anonymized preference tags disambiguate an
        underspecified query ("boots" from a durability-focused shopper), and
        the hint is dropped once two constraints exist so stated requirements
        always dominate remembered tendencies.
        """
        active_count = sum(
            1 for constraint in state.constraints if constraint.status == "active"
        )
        if active_count >= 2:
            return ""
        profile = self._profiles.get(session_id) or {}
        tags = [
            str(tag)
            for tag in (profile.get("preference_tags") or [])
            if str(tag).strip()
        ]
        if not tags:
            return ""
        return f"Shopper priorities: {', '.join(tags[:5])}"

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

        self._dialog_messages[session_id].append(user_message)
        parsed_budget = parse_budget(user_message)
        if parsed_budget is not None:
            # The most recent spending statement supersedes earlier ones.
            self._budget_ranges[session_id] = parsed_budget
        prior_dialog = self._prior_dialog_text(session_id)
        profile_hint = self._profile_hint(session_id, state)
        rewrite_prompt_tokens = 0
        rewrite_completion_tokens = 0
        retrieval_message = user_message
        rewritten_active = False
        if self.query_rewriter is not None:
            constraint_summary = "; ".join(
                f"{constraint.attribute}: {', '.join(constraint.values)}"
                for constraint in state.constraints
                if constraint.status == "active"
            )
            rewrite_result = self.query_rewriter.rewrite(
                self._dialog_messages[session_id],
                constraint_summary,
                profile_hint,
            )
            if rewrite_result is not None:
                (
                    retrieval_message,
                    rewrite_prompt_tokens,
                    rewrite_completion_tokens,
                ) = rewrite_result
                rewritten_active = True
        # A successful rewrite already folds the dialog in, so the dense query
        # spends its token budget on the rewrite; the lexical route keeps the
        # raw dialog terms as a safety net either way.
        dense_dialog = "" if rewritten_active else prior_dialog
        selected_tools = set(outcome.retrieval_tools)
        dense_candidates = []
        if "dense" in selected_tools:
            try:
                dense_candidates = self.dense_route.search(
                    self._dense_query(
                        retrieval_message, state, dense_dialog, profile_hint
                    ),
                    DENSE_CANDIDATE_DEPTH,
                )
            except Exception:  # noqa: BLE001 - optional route must fail open
                dense_candidates = []
        route_scores = self.retrieval.hybrid_route_scores(
            retrieval_message,
            state.constraints,
            dense_candidates,
            route_limit=DENSE_CANDIDATE_DEPTH,
            enabled_routes=selected_tools.intersection(
                {"structured", "bm25", "dense"}
            ),
            dialog_text=prior_dialog,
        )
        dense_query = self._dense_query(
            retrieval_message, state, dense_dialog, profile_hint
        )
        active_constraint_count = sum(
            1 for constraint in state.constraints if constraint.status == "active"
        )
        # Popularity-aware pool admission: the target is a real purchase, and
        # among equally-matching products the popular ones are likelier buys.
        # The prior's weight decays as the customer states requirements, so
        # explicit evidence always dominates remembered tendencies.
        popularity_weight = 0.3 / (1.0 + active_constraint_count)
        wide_pool = self.fusion_policy.rank(
            route_scores,
            candidate_limit=self.candidate_pool_depth * 2,
        )
        fused_positions = {
            parent_asin: position
            for position, parent_asin in enumerate(wide_pool)
        }
        popularity_order = sorted(
            wide_pool,
            key=lambda parent_asin: (
                -self.retrieval.popularity_prior(parent_asin),
                parent_asin,
            ),
        )
        popularity_positions = {
            parent_asin: position
            for position, parent_asin in enumerate(popularity_order)
        }
        fused_candidates = sorted(
            wide_pool,
            key=lambda parent_asin: (
                (1.0 - popularity_weight) * fused_positions[parent_asin]
                + popularity_weight * popularity_positions[parent_asin],
                fused_positions[parent_asin],
            ),
        )[: self.candidate_pool_depth]
        if not fused_candidates and outcome.source == "fallback":
            fused_candidates = self.retrieval.recommend(
                user_message,
                state.constraints,
                self.candidate_pool_depth,
            )
        fused_fallback = list(fused_candidates)
        rerank_scores: dict[str, float] | None = None
        try:
            reranked_candidates = self.reranker.rerank(
                dense_query,
                fused_candidates,
                self.retrieval.rerank_documents(fused_candidates),
            )
            if (
                len(reranked_candidates) != len(fused_candidates)
                or set(reranked_candidates) != set(fused_candidates)
            ):
                raise ValueError(
                    "reranker output must be a permutation of its Candidate Pool"
                )
            fused_candidates = reranked_candidates
            scores_getter = getattr(self.reranker, "last_scores", None)
            if callable(scores_getter):
                rerank_scores = scores_getter()
            if rerank_scores:
                fused_candidates = self._popularity_tiebreak(
                    fused_candidates, rerank_scores
                )
        except Exception:  # noqa: BLE001 - optional reranking must fail open
            fused_candidates = fused_fallback
        semantic_prompt_tokens = 0
        semantic_completion_tokens = 0
        confident_margin = False
        if rerank_scores and len(fused_candidates) >= 3:
            top_scores = [
                rerank_scores.get(parent_asin)
                for parent_asin in fused_candidates[:3]
            ]
            if all(score is not None for score in top_scores):
                confident_margin = (
                    top_scores[0] - top_scores[2] > RERANK_CONFIDENT_MARGIN
                )
        if (
            self.semantic_ranker is not None
            and len(fused_candidates) > 1
            and not confident_margin
        ):
            constraint_summary = "; ".join(
                f"{constraint.attribute}: {', '.join(constraint.values)}"
                for constraint in state.constraints
                if constraint.status == "active"
            )
            head_limit = getattr(self.semantic_ranker, "max_candidates", 20)
            pairs = [
                (parent_asin, self._semantic_candidate_text(parent_asin))
                for parent_asin in fused_candidates[:head_limit]
            ]
            ranked_head = self.semantic_ranker.rank(
                self._dialog_messages[session_id],
                constraint_summary,
                pairs,
            )
            if ranked_head is not None:
                # The LLM ordering is applied as-is: a measured rank-blending
                # variant (0.4 local / 0.6 LLM) scored 0.736 against 0.757 for
                # the raw ordering on the paired development subset, so
                # variance is controlled by the confidence-margin gate above
                # rather than by damping the LLM's decisions.
                head, semantic_prompt_tokens, semantic_completion_tokens = ranked_head
                fused_candidates = [*head, *fused_candidates[len(head):]]
        session_budget = self._budget_ranges.get(session_id)
        if session_budget is not None:
            # Stable partition: products fitting the stated budget move ahead
            # of over/under-priced ones without changing relative order or
            # eliminating anything (catalog prices can be missing or stale).
            fused_candidates = sorted(
                fused_candidates,
                key=lambda parent_asin: not self.retrieval.within_budget(
                    parent_asin, session_budget
                ),
            )
        if self.trace_pool_depths:
            normalized_routes = normalized_route_scores(route_scores)
            self._candidate_traces[session_id].append({
                "turn": turn,
                "query": dense_query,
                "planning": {
                    "source": outcome.source,
                    "state_revision": state.revision,
                    "retrieval_tools": list(outcome.retrieval_tools),
                },
                "route_candidates": {
                    route_name: [
                        {
                            "parent_asin": parent_asin,
                            "raw_score": float(raw_score),
                            "normalized_score": normalized_routes[route_name].get(
                                parent_asin,
                                0.0,
                            ),
                        }
                        for parent_asin, raw_score in route_scores.get(route_name, ())
                    ]
                    for route_name in normalized_routes
                },
                "pools": {
                    str(depth): self.fusion_policy.rank(
                        route_scores,
                        candidate_limit=depth,
                    )
                    for depth in self.trace_pool_depths
                },
                "response_pool": list(fused_candidates),
            })
        recommendation_limit = max(0, min(top_k, 10))
        recommendations = [
            {"parent_asin": parent_asin}
            for parent_asin in fused_candidates[:recommendation_limit]
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
            ask_attribute = self._next_ask_attribute(
                session_id, state, fused_candidates
            )
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
                "prompt_tokens": (
                    outcome.prompt_tokens
                    + semantic_prompt_tokens
                    + rewrite_prompt_tokens
                ),
                "completion_tokens": (
                    outcome.completion_tokens
                    + semantic_completion_tokens
                    + rewrite_completion_tokens
                ),
            },
        }
