from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Literal, Mapping, Protocol, Sequence

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
from starter.replacement_evidence import (
    ReplacementEvidenceError,
    validate_replacement_evidence,
)
from starter.session_policy import ALLOWED_ASK_ATTRIBUTES, SESSION_MODES, SessionMode


RetrievalTool = Literal["structured", "bm25", "dense"]
APPROVED_RETRIEVAL_TOOLS: tuple[RetrievalTool, ...] = (
    "structured",
    "bm25",
    "dense",
)
DEFAULT_RETRIEVAL_TOOLS: tuple[RetrievalTool, ...] = (
    "structured",
    "bm25",
    "dense",
)
MAX_PLANNING_ATTEMPTS = 2
RECENT_HISTORY_LIMIT = 4
PLANNING_PROMPT_VERSION = "shopping-turn-planner-v3"
PLANNING_INSTRUCTIONS = """You are the semantic planner for a Shopping Agent.
Return exactly one complete Turn Plan that matches the supplied JSON Schema.
Use the supplied Constraint State revision and source turn unchanged. Propose only
explicit state transitions supported by the customer message and current state.
Replace one Constraint for an explicit attribute correction. Replace Product Intent
only when the latest message explicitly replaces a product type or withdraws the
whole prior intent and establishes a distinct supported successor; a different
product mention or "actually" alone is insufficient.
Use only supplied attributes, normalized values, Constraint identifiers, Product
Intent identifiers, and approved Candidate Pool-producing Retrieval Routes.
Local reranking is fixed downstream and is not a selectable tool. Never request shell, web,
arbitrary code, or catalog mutation capabilities. Local Constraint State is
authoritative; recent history is context, not permission to replay old mutations.
Revise Session Mode on every turn as buying, browsing, or uncertain. Explicit
current-turn requirements outrank recent history and aggregate profile hints.
Profile hints may only break ties between otherwise useful Clarifications; never
turn them into Constraints. Ask at most one allowed attribute, never ask an active,
dismissed, previously asked, or unsupported attribute, and return null clarification
when no useful supported attribute should be asked. Recommendations are produced
downstream even when a Clarification is selected.
"""


TURN_PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "expected_state_revision",
        "source_turn",
        "mutations",
        "retrieval_tools",
        "session_mode",
        "clarification",
    ],
    "properties": {
        "expected_state_revision": {"type": "integer", "minimum": 0},
        "source_turn": {"type": "integer", "minimum": 1, "maximum": 10},
        "mutations": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "type", "attribute", "values", "match_rule",
                            "classification", "scope", "raw_phrase", "confidence",
                        ],
                        "properties": {
                            "type": {
                                "enum": ["add_constraint", "reintroduce_constraint"],
                            },
                            "attribute": {"type": "string"},
                            "values": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "match_rule": {"enum": ["any", "all"]},
                            "classification": {"enum": ["hard", "soft"]},
                            "scope": {"enum": ["product_intent", "session"]},
                            "raw_phrase": {"type": "string", "minLength": 1},
                            "confidence": {
                                "type": "number", "minimum": 0, "maximum": 1,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "type", "constraint_id", "attribute", "values",
                            "match_rule", "classification", "scope", "raw_phrase",
                            "confidence",
                        ],
                        "properties": {
                            "type": {"const": "replace_constraint"},
                            "constraint_id": {"type": "string", "minLength": 1},
                            "attribute": {"type": "string"},
                            "values": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "match_rule": {"enum": ["any", "all"]},
                            "classification": {"enum": ["hard", "soft"]},
                            "scope": {"enum": ["product_intent", "session"]},
                            "raw_phrase": {"type": "string", "minLength": 1},
                            "confidence": {
                                "type": "number", "minimum": 0, "maximum": 1,
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "attribute", "raw_phrase"],
                        "properties": {
                            "type": {"const": "dismiss_attribute"},
                            "attribute": {"type": "string"},
                            "raw_phrase": {"type": "string", "minLength": 1},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "product_intent_id", "raw_phrase"],
                        "properties": {
                            "type": {"const": "replace_product_intent"},
                            "product_intent_id": {"type": "string", "minLength": 1},
                            "raw_phrase": {"type": "string", "minLength": 1},
                        },
                    },
                ],
            },
        },
        "retrieval_tools": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"enum": list(APPROVED_RETRIEVAL_TOOLS)},
        },
        "session_mode": {"enum": list(SESSION_MODES)},
        "clarification": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["ask_attribute", "message"],
                    "properties": {
                        "ask_attribute": {"enum": list(ALLOWED_ASK_ATTRIBUTES)},
                        "message": {"type": "string", "minLength": 1},
                    },
                },
            ],
        },
    },
}

PLANNING_PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "instructions": PLANNING_INSTRUCTIONS,
            "schema": TURN_PLAN_JSON_SCHEMA,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class PlanningError(ValueError):
    """Base error for a rejected connected planning attempt."""


class PlanningSchemaError(PlanningError):
    """The provider returned a payload outside the strict planning schema."""


class PlanningToolError(PlanningError):
    """The provider requested a capability outside the approved tool set."""


class PlanningStateError(PlanningError):
    """The proposed Turn Plan was rejected by Constraint State validation."""


class MissingCredentialsError(RuntimeError):
    """The connected provider cannot run without configured credentials."""


@dataclass(frozen=True)
class Clarification:
    ask_attribute: str
    message: str


@dataclass(frozen=True)
class PlanningRequest:
    session_id: str
    turn: int
    user_message: str
    state_snapshot: dict
    recent_history: tuple[dict, ...]
    supported_values: dict[str, tuple[str, ...]]
    allowed_tools: tuple[RetrievalTool, ...]
    previous_session_mode: SessionMode
    profile_hints: tuple[str, ...]
    previously_asked_attributes: tuple[str, ...]
    allowed_ask_attributes: tuple[str, ...]
    prompt_version: str
    instructions: str
    response_schema: dict
    validation_error: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    output: object
    prompt_tokens: int = 0
    completion_tokens: int = 0


class PlanningProvider(Protocol):
    def plan(self, request: PlanningRequest) -> ProviderResponse | Mapping: ...


@dataclass(frozen=True)
class DecodedPlan:
    turn_plan: TurnPlan
    retrieval_tools: tuple[RetrievalTool, ...]
    session_mode: SessionMode
    clarification: Clarification | None


@dataclass(frozen=True)
class PlanningOutcome:
    turn_plan: TurnPlan
    retrieval_tools: tuple[RetrievalTool, ...]
    session_mode: SessionMode
    clarification: Clarification | None
    source: Literal["connected", "fallback"]
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    fallback_reason: str | None = None
    errors: tuple[str, ...] = ()


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise PlanningSchemaError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping,
    *,
    required: set[str],
    label: str,
) -> None:
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        raise PlanningSchemaError(
            f"{label} has invalid keys; missing={missing}, extra={extra}"
        )


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanningSchemaError(f"{label} must be an integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanningSchemaError(f"{label} must be a non-empty string")
    return value


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanningSchemaError("mutation confidence must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise PlanningSchemaError("mutation confidence must be between zero and one")
    return result


def _constraint_fields(value: Mapping) -> dict:
    attribute = _string(value["attribute"], "mutation attribute")
    raw_values = value["values"]
    if not isinstance(raw_values, list) or not raw_values:
        raise PlanningSchemaError("mutation values must be a non-empty array")
    values = tuple(_string(item, "mutation value") for item in raw_values)
    if len(set(values)) != len(values):
        raise PlanningSchemaError("mutation values must be unique")
    match_rule = value["match_rule"]
    if match_rule not in ("any", "all"):
        raise PlanningSchemaError("mutation match_rule is invalid")
    classification = value["classification"]
    if classification not in ("hard", "soft"):
        raise PlanningSchemaError("mutation classification is invalid")
    scope = value["scope"]
    if scope not in ("product_intent", "session"):
        raise PlanningSchemaError("mutation scope is invalid")
    return {
        "attribute": attribute,
        "values": values,
        "match_rule": match_rule,
        "classification": classification,
        "scope": scope,
        "raw_phrase": _string(value["raw_phrase"], "mutation raw_phrase"),
        "confidence": _confidence(value["confidence"]),
    }


def _decode_mutation(value: object):
    mutation = _mapping(value, "mutation")
    mutation_type = _string(mutation.get("type"), "mutation type")
    constraint_keys = {
        "type", "attribute", "values", "match_rule", "classification",
        "scope", "raw_phrase", "confidence",
    }
    if mutation_type in ("add_constraint", "reintroduce_constraint"):
        _exact_keys(mutation, required=constraint_keys, label=mutation_type)
        mutation_class = (
            AddConstraint
            if mutation_type == "add_constraint"
            else ReintroduceConstraint
        )
        return mutation_class(**_constraint_fields(mutation))
    if mutation_type == "replace_constraint":
        _exact_keys(
            mutation,
            required=constraint_keys | {"constraint_id"},
            label=mutation_type,
        )
        return ReplaceConstraint(
            constraint_id=_string(
                mutation["constraint_id"],
                "replacement constraint_id",
            ),
            **_constraint_fields(mutation),
        )
    if mutation_type == "dismiss_attribute":
        _exact_keys(
            mutation,
            required={"type", "attribute", "raw_phrase"},
            label=mutation_type,
        )
        return DismissAttribute(
            attribute=_string(mutation["attribute"], "dismissal attribute"),
            raw_phrase=_string(mutation["raw_phrase"], "dismissal raw_phrase"),
        )
    if mutation_type == "replace_product_intent":
        _exact_keys(
            mutation,
            required={"type", "product_intent_id", "raw_phrase"},
            label=mutation_type,
        )
        return ReplaceProductIntent(
            product_intent_id=_string(
                mutation["product_intent_id"],
                "replacement product_intent_id",
            ),
            raw_phrase=_string(mutation["raw_phrase"], "replacement raw_phrase"),
        )
    raise PlanningSchemaError(f"unknown mutation type: {mutation_type}")


def decode_plan(
    payload: object,
    *,
    user_message: str,
    turn: int,
    state: ConstraintState,
    supported_values: Mapping[str, set[str]],
    grounding_mutations: tuple[object, ...] | None = None,
    allowed_ask_attributes: Sequence[str] | None = None,
    previously_asked_attributes: Sequence[str] = (),
) -> DecodedPlan:
    value = _mapping(payload, "Turn Plan")
    _exact_keys(
        value,
        required={
            "expected_state_revision", "source_turn", "mutations",
            "retrieval_tools", "session_mode", "clarification",
        },
        label="Turn Plan",
    )
    expected_revision = _integer(
        value["expected_state_revision"],
        "expected_state_revision",
    )
    source_turn = _integer(value["source_turn"], "source_turn")
    if expected_revision != state.revision:
        raise PlanningStateError("Turn Plan revision does not match local state")
    if source_turn != turn:
        raise PlanningStateError("Turn Plan source turn does not match the request")

    raw_mutations = value["mutations"]
    if not isinstance(raw_mutations, list):
        raise PlanningSchemaError("mutations must be an array")
    mutations = tuple(_decode_mutation(item) for item in raw_mutations)

    raw_tools = value["retrieval_tools"]
    if not isinstance(raw_tools, list) or not raw_tools:
        raise PlanningToolError("retrieval_tools must be a non-empty array")
    if any(not isinstance(tool, str) for tool in raw_tools):
        raise PlanningToolError("retrieval tool names must be strings")
    if len(set(raw_tools)) != len(raw_tools):
        raise PlanningToolError("retrieval tools must be unique")
    unknown_tools = set(raw_tools) - set(APPROVED_RETRIEVAL_TOOLS)
    if unknown_tools:
        raise PlanningToolError(
            f"unapproved retrieval tools: {sorted(unknown_tools)}"
        )
    retrieval_tools = tuple(raw_tools)

    session_mode = value["session_mode"]
    if session_mode not in SESSION_MODES:
        raise PlanningSchemaError("session_mode must be buying, browsing, or uncertain")

    clarification_value = value["clarification"]
    clarification = None
    if clarification_value is not None:
        clarification_mapping = _mapping(clarification_value, "clarification")
        _exact_keys(
            clarification_mapping,
            required={"ask_attribute", "message"},
            label="clarification",
        )
        ask_attribute = _string(
            clarification_mapping["ask_attribute"],
            "clarification ask_attribute",
        )
        if ask_attribute not in supported_values:
            raise PlanningStateError(
                f"clarification uses unknown attribute: {ask_attribute}"
            )
        if ask_attribute not in ALLOWED_ASK_ATTRIBUTES:
            raise PlanningStateError(
                f"clarification uses disallowed ask_attribute: {ask_attribute}"
            )
        if ask_attribute in previously_asked_attributes:
            raise PlanningStateError(
                "clarification repeats a previously asked low-value attribute"
            )
        if (
            allowed_ask_attributes is not None
            and ask_attribute not in allowed_ask_attributes
        ):
            raise PlanningStateError(
                "clarification has no useful expected value for the current mode"
            )
        clarification = Clarification(
            ask_attribute=ask_attribute,
            message=_string(
                clarification_mapping["message"],
                "clarification message",
            ),
        )

    turn_plan = TurnPlan(
        expected_state_revision=expected_revision,
        source_turn=source_turn,
        mutations=mutations,
    )
    try:
        validate_replacement_evidence(
            user_message,
            state,
            mutations,
            grounding_mutations=grounding_mutations,
        )
    except ReplacementEvidenceError as error:
        raise PlanningStateError(str(error)) from error
    draft = deepcopy(state)
    try:
        draft.apply(turn_plan, supported_values)
    except PlanValidationError as error:
        raise PlanningStateError(str(error)) from error
    if clarification is not None:
        if clarification.ask_attribute in draft.dismissed_attributes:
            raise PlanningStateError(
                "clarification repeats an attribute dismissed by the resulting state"
            )
        if any(
            constraint.status == "active"
            and constraint.attribute == clarification.ask_attribute
            for constraint in draft.constraints
        ):
            raise PlanningStateError(
                "clarification repeats an active Constraint attribute"
            )
    return DecodedPlan(
        turn_plan=turn_plan,
        retrieval_tools=retrieval_tools,
        session_mode=session_mode,
        clarification=clarification,
    )


class PlanningLoop:
    """Validate connected Turn Plans and take over locally after one retry."""

    def __init__(
        self,
        provider: PlanningProvider | None,
        *,
        max_attempts: int = MAX_PLANNING_ATTEMPTS,
    ) -> None:
        if max_attempts < 1 or max_attempts > MAX_PLANNING_ATTEMPTS:
            raise ValueError("planning attempts must be between one and two")
        self.provider = provider
        self.max_attempts = max_attempts

    def run(
        self,
        *,
        session_id: str,
        turn: int,
        user_message: str,
        state: ConstraintState,
        supported_values: Mapping[str, set[str]],
        recent_history: list[dict],
        previous_session_mode: SessionMode,
        profile_hints: tuple[str, ...],
        previously_asked_attributes: tuple[str, ...],
        allowed_ask_attributes: tuple[str, ...],
        fallback_session_mode: SessionMode,
        fallback_plan: Callable[[], TurnPlan],
    ) -> PlanningOutcome:
        if self.provider is None:
            return self._fallback(
                fallback_plan,
                state,
                supported_values,
                reason="missing_credentials",
                attempts=0,
                prompt_tokens=0,
                completion_tokens=0,
                errors=(),
                session_mode=fallback_session_mode,
            )

        prompt_tokens = 0
        completion_tokens = 0
        errors: list[str] = []
        fallback_reason = "provider_error"
        grounding_plan = fallback_plan()
        for attempt in range(1, self.max_attempts + 1):
            request = PlanningRequest(
                session_id=session_id,
                turn=turn,
                user_message=user_message,
                state_snapshot=state.as_dict(),
                recent_history=tuple(
                    deepcopy(recent_history[-RECENT_HISTORY_LIMIT:])
                ),
                supported_values={
                    attribute: tuple(sorted(values))
                    for attribute, values in supported_values.items()
                },
                allowed_tools=APPROVED_RETRIEVAL_TOOLS,
                previous_session_mode=previous_session_mode,
                profile_hints=profile_hints,
                previously_asked_attributes=previously_asked_attributes,
                allowed_ask_attributes=allowed_ask_attributes,
                prompt_version=PLANNING_PROMPT_VERSION,
                instructions=PLANNING_INSTRUCTIONS,
                response_schema=deepcopy(TURN_PLAN_JSON_SCHEMA),
                validation_error=errors[-1] if errors else None,
            )
            try:
                raw_response = self.provider.plan(request)
                response = self._provider_response(raw_response)
                prompt_tokens += response.prompt_tokens
                completion_tokens += response.completion_tokens
                decoded = decode_plan(
                    response.output,
                    user_message=user_message,
                    turn=turn,
                    state=state,
                    supported_values=supported_values,
                    grounding_mutations=grounding_plan.mutations,
                    allowed_ask_attributes=allowed_ask_attributes,
                    previously_asked_attributes=previously_asked_attributes,
                )
                return PlanningOutcome(
                    turn_plan=decoded.turn_plan,
                    retrieval_tools=decoded.retrieval_tools,
                    session_mode=decoded.session_mode,
                    clarification=decoded.clarification,
                    source="connected",
                    attempts=attempt,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    errors=tuple(errors),
                )
            except MissingCredentialsError as error:
                fallback_reason = "missing_credentials"
                errors.append(self._safe_error(error))
                break
            except TimeoutError as error:
                fallback_reason = "timeout"
                errors.append(self._safe_error(error))
            except PlanningToolError as error:
                fallback_reason = "unapproved_tool"
                errors.append(self._safe_error(error))
            except PlanningStateError as error:
                fallback_reason = "rejected_state_change"
                errors.append(self._safe_error(error))
            except PlanningSchemaError as error:
                fallback_reason = "invalid_schema"
                errors.append(self._safe_error(error))
            except Exception as error:  # noqa: BLE001 - provider failure must fail open
                fallback_reason = "provider_error"
                errors.append(self._safe_error(error))

        return self._fallback(
            lambda: grounding_plan,
            state,
            supported_values,
            reason=fallback_reason,
            attempts=min(self.max_attempts, max(1, len(errors))),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            errors=tuple(errors),
            session_mode=fallback_session_mode,
        )

    def _fallback(
        self,
        fallback_plan: Callable[[], TurnPlan],
        state: ConstraintState,
        supported_values: Mapping[str, set[str]],
        *,
        reason: str,
        attempts: int,
        prompt_tokens: int,
        completion_tokens: int,
        errors: tuple[str, ...],
        session_mode: SessionMode,
    ) -> PlanningOutcome:
        plan = fallback_plan()
        draft = deepcopy(state)
        try:
            draft.apply(plan, supported_values)
        except PlanValidationError:
            plan = TurnPlan(
                expected_state_revision=state.revision,
                source_turn=max(1, plan.source_turn),
            )
        return PlanningOutcome(
            turn_plan=plan,
            retrieval_tools=DEFAULT_RETRIEVAL_TOOLS,
            session_mode=session_mode,
            clarification=None,
            source="fallback",
            attempts=attempts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            fallback_reason=reason,
            errors=errors,
        )

    @staticmethod
    def _provider_response(value: object) -> ProviderResponse:
        response = value if isinstance(value, ProviderResponse) else ProviderResponse(value)
        for name, count in (
            ("prompt_tokens", response.prompt_tokens),
            ("completion_tokens", response.completion_tokens),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise PlanningSchemaError(f"{name} must be a non-negative integer")
        return response

    @staticmethod
    def _safe_error(error: Exception) -> str:
        if not isinstance(error, PlanningError):
            return error.__class__.__name__
        message = str(error).strip() or error.__class__.__name__
        return f"{error.__class__.__name__}: {message}"[:300]
