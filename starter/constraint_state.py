from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Literal, Mapping


Classification = Literal["hard", "soft"]
ConstraintScope = Literal["product_intent", "session"]
ConstraintStatus = Literal["active", "superseded", "dismissed"]
MatchRule = Literal["any", "all"]


class PlanValidationError(ValueError):
    """Raised when a Turn Plan cannot be applied without corrupting state."""


@dataclass(frozen=True)
class AddConstraint:
    attribute: str
    values: tuple[str, ...]
    match_rule: MatchRule
    classification: Classification
    scope: ConstraintScope
    raw_phrase: str
    confidence: float


@dataclass(frozen=True)
class ReintroduceConstraint(AddConstraint):
    pass


@dataclass(frozen=True)
class ReplaceConstraint(AddConstraint):
    constraint_id: str


@dataclass(frozen=True)
class DismissAttribute:
    attribute: str
    raw_phrase: str


@dataclass(frozen=True)
class ReplaceProductIntent:
    product_intent_id: str
    raw_phrase: str


StateMutation = AddConstraint | DismissAttribute | ReplaceProductIntent


@dataclass(frozen=True)
class TurnPlan:
    expected_state_revision: int
    source_turn: int
    mutations: tuple[StateMutation, ...] = ()


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    product_intent_id: str | None
    scope: ConstraintScope
    attribute: str
    values: tuple[str, ...]
    match_rule: MatchRule
    classification: Classification
    raw_phrase: str
    source_turn: int
    confidence: float
    status: ConstraintStatus = "active"
    transition_reason: str = "added"
    superseded_by: str | None = None

    @property
    def normalized_value(self) -> str:
        return self.values[0]

    def as_dict(self) -> dict:
        return {
            "constraint_id": self.constraint_id,
            "product_intent_id": self.product_intent_id,
            "scope": self.scope,
            "attribute": self.attribute,
            "normalized_value": self.normalized_value,
            "normalized_values": list(self.values),
            "match_rule": self.match_rule,
            "raw_phrase": self.raw_phrase,
            "classification": self.classification,
            "source_turn": self.source_turn,
            "confidence": self.confidence,
            "status": self.status,
            "transition_reason": self.transition_reason,
            "superseded_by": self.superseded_by,
        }


@dataclass
class ConstraintState:
    constraints: list[Constraint] = field(default_factory=list)
    dismissed_attributes: dict[str, dict] = field(default_factory=dict)
    transition_history: list[dict] = field(default_factory=list)
    revision: int = 0
    active_product_intent_id: str = "intent-1"
    _next_constraint_number: int = 1
    _next_product_intent_number: int = 2

    def as_dict(self) -> dict:
        return {
            "revision": self.revision,
            "active_product_intent_id": self.active_product_intent_id,
            "constraints": [constraint.as_dict() for constraint in self.constraints],
            "dismissed_attributes": deepcopy(self.dismissed_attributes),
            "transition_history": deepcopy(self.transition_history),
            "next_constraint_number": self._next_constraint_number,
            "next_product_intent_number": self._next_product_intent_number,
        }

    def active_constraints(self, attribute: str | None = None) -> list[Constraint]:
        """Return active constraints, optionally limited to one attribute."""

        return [
            constraint
            for constraint in self.constraints
            if constraint.status == "active"
            and (attribute is None or constraint.attribute == attribute)
        ]

    def apply(
        self,
        plan: TurnPlan,
        supported_values: Mapping[str, set[str]],
    ) -> None:
        if plan.expected_state_revision != self.revision:
            raise PlanValidationError(
                f"stale Turn Plan: expected revision {plan.expected_state_revision}, "
                f"current revision is {self.revision}"
            )
        if plan.source_turn < 1:
            raise PlanValidationError("source turn must be positive")

        draft = deepcopy(self)
        unknown = [
            mutation
            for mutation in plan.mutations
            if not isinstance(
                mutation,
                (AddConstraint, DismissAttribute, ReplaceProductIntent),
            )
        ]
        if unknown:
            raise PlanValidationError("unknown state mutation")
        intent_replacements = [
            mutation
            for mutation in plan.mutations
            if isinstance(mutation, ReplaceProductIntent)
        ]
        if len(intent_replacements) > 1:
            raise PlanValidationError("a Turn Plan may replace one Product Intent")
        if intent_replacements:
            draft._apply_product_intent_replacement(
                intent_replacements[0],
                plan.source_turn,
            )

        dismissals = [
            mutation
            for mutation in plan.mutations
            if isinstance(mutation, DismissAttribute)
        ]
        dismissed_attributes = {mutation.attribute for mutation in dismissals}
        if len(dismissed_attributes) != len(dismissals):
            raise PlanValidationError("an attribute may be dismissed once per Turn Plan")
        reintroduced_attributes = {
            mutation.attribute
            for mutation in plan.mutations
            if isinstance(mutation, ReintroduceConstraint)
        }
        if dismissed_attributes & reintroduced_attributes:
            raise PlanValidationError(
                "dismissal and reintroduction of one attribute are contradictory"
            )
        for mutation in dismissals:
            draft._apply_dismissal(mutation, plan.source_turn, supported_values)

        constraint_replacements = [
            mutation
            for mutation in plan.mutations
            if isinstance(mutation, ReplaceConstraint)
        ]
        replacement_targets = {
            mutation.constraint_id for mutation in constraint_replacements
        }
        if len(replacement_targets) != len(constraint_replacements):
            raise PlanValidationError("a Constraint may be replaced once per Turn Plan")
        if any(
            mutation.attribute in dismissed_attributes
            for mutation in constraint_replacements
        ):
            raise PlanValidationError(
                "replacement and dismissal of one attribute are contradictory"
            )

        added_attributes: set[tuple[str, str]] = set()
        for mutation in constraint_replacements:
            draft._apply_constraint_replacement(
                mutation,
                plan.source_turn,
                supported_values,
                added_attributes,
            )
        for mutation in plan.mutations:
            if isinstance(mutation, AddConstraint) and not isinstance(
                mutation,
                ReplaceConstraint,
            ):
                draft._apply_add(
                    mutation,
                    plan.source_turn,
                    supported_values,
                    added_attributes,
                    allow_reintroduction=isinstance(
                        mutation,
                        ReintroduceConstraint,
                    ),
                )

        if plan.mutations:
            draft.revision += 1
        self.constraints = draft.constraints
        self.dismissed_attributes = draft.dismissed_attributes
        self.transition_history = draft.transition_history
        self.revision = draft.revision
        self.active_product_intent_id = draft.active_product_intent_id
        self._next_constraint_number = draft._next_constraint_number
        self._next_product_intent_number = draft._next_product_intent_number

    def _apply_product_intent_replacement(
        self,
        mutation: ReplaceProductIntent,
        source_turn: int,
    ) -> None:
        if mutation.product_intent_id != self.active_product_intent_id:
            raise PlanValidationError("Product Intent replacement must target the active intent")
        if not mutation.raw_phrase.strip():
            raise PlanValidationError("Product Intent replacement requires provenance")
        new_intent_id = f"intent-{self._next_product_intent_number}"
        self._next_product_intent_number += 1
        self.constraints = [
            replace(
                constraint,
                status="superseded",
                transition_reason="product_intent_replaced",
                superseded_by=new_intent_id,
            )
            if constraint.status == "active"
            and constraint.scope == "product_intent"
            and constraint.product_intent_id == mutation.product_intent_id
            else constraint
            for constraint in self.constraints
        ]
        self.transition_history.append({
            "type": "product_intent_replaced",
            "source_turn": source_turn,
            "raw_phrase": mutation.raw_phrase,
            "from_product_intent_id": mutation.product_intent_id,
            "to_product_intent_id": new_intent_id,
        })
        self.active_product_intent_id = new_intent_id

    def _apply_constraint_replacement(
        self,
        mutation: ReplaceConstraint,
        source_turn: int,
        supported_values: Mapping[str, set[str]],
        added_attributes: set[tuple[str, str]],
    ) -> None:
        target = next(
            (
                constraint
                for constraint in self.constraints
                if constraint.constraint_id == mutation.constraint_id
            ),
            None,
        )
        if target is None or target.status != "active":
            raise PlanValidationError("replacement must target an active Constraint")
        self.constraints = [
            replace(
                constraint,
                status="superseded",
                transition_reason="constraint_replaced",
            )
            if constraint.constraint_id == mutation.constraint_id
            else constraint
            for constraint in self.constraints
        ]
        replacement = self._apply_add(
            mutation,
            source_turn,
            supported_values,
            added_attributes,
            allow_reintroduction=False,
        )
        self.constraints = [
            replace(constraint, superseded_by=replacement.constraint_id)
            if constraint.constraint_id == mutation.constraint_id
            else constraint
            for constraint in self.constraints
        ]

    def _apply_dismissal(
        self,
        mutation: DismissAttribute,
        source_turn: int,
        supported_values: Mapping[str, set[str]],
    ) -> Constraint:
        if mutation.attribute not in supported_values:
            raise PlanValidationError(f"unknown attribute: {mutation.attribute}")
        if not mutation.raw_phrase.strip():
            raise PlanValidationError("Boundary Response requires provenance")
        self.constraints = [
            replace(
                constraint,
                status="dismissed",
                transition_reason="boundary_response",
            )
            if constraint.status == "active"
            and constraint.attribute == mutation.attribute
            else constraint
            for constraint in self.constraints
        ]
        self.dismissed_attributes[mutation.attribute] = {
            "attribute": mutation.attribute,
            "raw_phrase": mutation.raw_phrase,
            "source_turn": source_turn,
            "status": "dismissed",
        }
        self.transition_history.append({
            "type": "attribute_dismissed",
            "source_turn": source_turn,
            "raw_phrase": mutation.raw_phrase,
            "attribute": mutation.attribute,
        })

    def _apply_add(
        self,
        mutation: AddConstraint,
        source_turn: int,
        supported_values: Mapping[str, set[str]],
        added_attributes: set[tuple[str, str]],
        allow_reintroduction: bool,
    ) -> Constraint:
        if mutation.attribute not in supported_values:
            raise PlanValidationError(f"unknown attribute: {mutation.attribute}")
        if not mutation.values or len(set(mutation.values)) != len(mutation.values):
            raise PlanValidationError("constraint values must be non-empty and unique")
        unsupported = set(mutation.values) - supported_values[mutation.attribute]
        if unsupported:
            raise PlanValidationError(
                f"unsupported values for {mutation.attribute}: {sorted(unsupported)}"
            )
        if mutation.match_rule not in ("any", "all"):
            raise PlanValidationError("invalid match rule")
        if mutation.classification not in ("hard", "soft"):
            raise PlanValidationError("invalid classification")
        if mutation.scope not in ("product_intent", "session"):
            raise PlanValidationError("invalid constraint scope")
        if not 0.0 <= mutation.confidence <= 1.0:
            raise PlanValidationError("confidence must be between zero and one")
        if not mutation.raw_phrase.strip():
            raise PlanValidationError("Constraint requires provenance")
        was_dismissed = mutation.attribute in self.dismissed_attributes
        if was_dismissed and not allow_reintroduction:
            raise PlanValidationError(
                f"dismissed {mutation.attribute} requires explicit reintroduction"
            )
        if allow_reintroduction and not was_dismissed:
            raise PlanValidationError(
                f"{mutation.attribute} is not dismissed and cannot be reintroduced"
            )
        scope_key = (
            self.active_product_intent_id
            if mutation.scope == "product_intent"
            else "session"
        )
        attribute_key = (scope_key, mutation.attribute)
        if attribute_key in added_attributes:
            raise PlanValidationError(
                "multiple additions for one attribute require one multi-value constraint"
            )
        if any(
            constraint.status == "active"
            and constraint.scope == mutation.scope
            and constraint.product_intent_id
            == (
                self.active_product_intent_id
                if mutation.scope == "product_intent"
                else None
            )
            and constraint.attribute == mutation.attribute
            for constraint in self.constraints
        ):
            raise PlanValidationError(
                f"active {mutation.attribute} constraint requires explicit replacement"
            )
        added_attributes.add(attribute_key)
        constraint_id = f"constraint-{self._next_constraint_number}"
        self._next_constraint_number += 1
        if allow_reintroduction:
            self.dismissed_attributes.pop(mutation.attribute, None)
        constraint = Constraint(
            constraint_id=constraint_id,
            product_intent_id=(
                self.active_product_intent_id
                if mutation.scope == "product_intent"
                else None
            ),
            scope=mutation.scope,
            attribute=mutation.attribute,
            values=mutation.values,
            match_rule=mutation.match_rule,
            classification=mutation.classification,
            raw_phrase=mutation.raw_phrase,
            source_turn=source_turn,
            confidence=mutation.confidence,
        )
        self.constraints.append(constraint)
        return constraint
