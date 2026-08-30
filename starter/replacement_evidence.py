from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal, Sequence

from starter.constraint_state import AddConstraint, Constraint, ConstraintState


REPLACEMENT_EVIDENCE_VERSION = "explicit-replacement-evidence-v1"

_CORRECTION_CUE_RE = re.compile(
    r"\b(?:actually|instead|rather\s+than|no\s+longer|changed\s+my\s+mind|"
    r"switch(?:ing)?\s+to|ignore|forget|not\s+.+?\bbut\b)\b",
    re.IGNORECASE,
)
_WHOLE_INTENT_WITHDRAWAL_RE = re.compile(
    r"\b(?:ignore|forget)\s+(?:everything|all|my\s+(?:entire|whole)\s+request|"
    r"what\s+i\s+(?:said|asked\s+for))\b|\bstart\s+over\b",
    re.IGNORECASE,
)
_PREFERENCE_WITHDRAWAL_RE = re.compile(
    r"\b(?:ignore|forget)\s+(?:my\s+)?(?:earlier|previous)\s+preference\b",
    re.IGNORECASE,
)

_REPLACEMENT_EVIDENCE_POLICY = {
    "product_intent": {
        "distinct_successor_required": True,
        "obsolete_active_category_must_be_relation_target": True,
        "whole_intent_withdrawal_allowed": True,
        "successor_only_relations": ["changed_mind", "switch_to", "successor_instead"],
    },
    "constraint": {
        "same_attribute_unique_target_preferred": True,
        "single_non_category_fallback": True,
        "ambiguous_target_rejected": True,
    },
    "connected_plan": {
        "replacement_provenance_equals_latest_turn": True,
        "successor_signature_equals_grounded_plan": True,
    },
}

REPLACEMENT_EVIDENCE_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": REPLACEMENT_EVIDENCE_VERSION,
            "patterns": [
                _CORRECTION_CUE_RE.pattern,
                _WHOLE_INTENT_WITHDRAWAL_RE.pattern,
                _PREFERENCE_WITHDRAWAL_RE.pattern,
            ],
            "policy": _REPLACEMENT_EVIDENCE_POLICY,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ReplacementEvidence:
    scope: Literal["constraint", "product_intent"]
    target_constraint_id: str | None = None


class ReplacementEvidenceError(ValueError):
    """A proposed replacement is not grounded in the latest customer turn."""


def has_replacement_cue(user_message: str) -> bool:
    """Return whether a turn merits deterministic replacement analysis."""

    return bool(_CORRECTION_CUE_RE.search(user_message))


def _is_distinct_successor(
    successor: AddConstraint,
    active: Sequence[Constraint],
) -> bool:
    return not any(
        constraint.attribute == successor.attribute
        and constraint.values == successor.values
        and constraint.scope == successor.scope
        for constraint in active
    )


def _successor_signature(successor: AddConstraint) -> tuple[object, ...]:
    return (
        successor.attribute,
        successor.values,
        successor.match_rule,
        successor.classification,
        successor.scope,
    )


def _term_pattern(*terms: str) -> str:
    variants = {
        term.strip().lower()
        for term in terms
        if term and term.strip()
    }
    expanded = set(variants)
    for term in variants:
        if re.fullmatch(r"[a-z0-9]+", term) and not term.endswith("s"):
            expanded.add(f"{term}s")
    return "(?:" + "|".join(
        re.escape(term).replace(r"\ ", r"[\s-]+")
        for term in sorted(expanded, key=len, reverse=True)
    ) + ")"


def _successor_terms(successor: AddConstraint) -> tuple[str, ...]:
    return (successor.raw_phrase, *successor.values)


def _constraint_terms(constraint: Constraint) -> tuple[str, ...]:
    return (constraint.raw_phrase, *constraint.values)


def _product_replacement_relation(
    user_message: str,
    successor: AddConstraint,
    active_categories: Sequence[Constraint],
) -> bool:
    successor_pattern = _term_pattern(*_successor_terms(successor))
    if not re.search(rf"(?<![a-z0-9]){successor_pattern}(?![a-z0-9])", user_message, re.I):
        return False
    if re.search(
        rf"\b(?:switch(?:ing)?\s+to|changed\s+my\s+mind(?:\s+to|\s*,?)?)\s+"
        rf"{successor_pattern}(?![a-z0-9])",
        user_message,
        re.I,
    ):
        return True
    if re.search(
        rf"(?<![a-z0-9]){successor_pattern}(?![a-z0-9]).{{0,24}}\binstead\b"
        rf"(?!\s+of\b)",
        user_message,
        re.I,
    ):
        return True
    for active_category in active_categories:
        active_pattern = _term_pattern(*_constraint_terms(active_category))
        successor_then_active = re.search(
            rf"(?<![a-z0-9]){successor_pattern}(?![a-z0-9]).{{0,32}}"
            rf"\b(?:instead\s+of|rather\s+than)\s+{active_pattern}(?![a-z0-9])",
            user_message,
            re.I,
        )
        relation_then_active = re.search(
            rf"\b(?:instead\s+of|rather\s+than|no\s+longer(?:\s+want)?)\s+"
            rf"{active_pattern}(?![a-z0-9])",
            user_message,
            re.I,
        )
        not_active_but_successor = re.search(
            rf"\bnot\s+{active_pattern}(?![a-z0-9]).{{0,32}}\bbut\s+"
            rf"{successor_pattern}(?![a-z0-9])",
            user_message,
            re.I,
        )
        if successor_then_active or relation_then_active or not_active_but_successor:
            return True
    return False


def classify_replacement_evidence(
    user_message: str,
    state: ConstraintState,
    successors: Sequence[AddConstraint],
) -> ReplacementEvidence | None:
    """Classify explicit latest-turn evidence against supported successors."""

    active = state.active_constraints()
    product_successors = [
        successor
        for successor in successors
        if successor.scope == "product_intent"
        and _is_distinct_successor(successor, active)
    ]
    incoming_category = next(
        (
            successor
            for successor in product_successors
            if successor.attribute == "category"
        ),
        None,
    )
    active_category = state.active_constraints("category")
    distinct_category = bool(
        incoming_category
        and (
            not active_category
            or set(incoming_category.values) != set(active_category[-1].values)
        )
    )
    if active and product_successors and (
        _WHOLE_INTENT_WITHDRAWAL_RE.search(user_message)
        or (
            distinct_category
            and incoming_category is not None
            and _product_replacement_relation(
                user_message,
                incoming_category,
                active_category,
            )
        )
    ):
        return ReplacementEvidence(scope="product_intent")

    if _PREFERENCE_WITHDRAWAL_RE.search(user_message):
        same_attribute_targets = {
            constraint.constraint_id: constraint
            for successor in product_successors
            for constraint in state.active_constraints(successor.attribute)
            if _is_distinct_successor(successor, (constraint,))
        }
        if len(same_attribute_targets) == 1:
            target = next(iter(same_attribute_targets.values()))
        elif same_attribute_targets:
            target = None
        else:
            non_category_targets = [
                constraint
                for constraint in active
                if constraint.attribute != "category"
            ]
            target = (
                non_category_targets[0]
                if len(non_category_targets) == 1
                else None
            )
        if target is not None and product_successors:
            return ReplacementEvidence(
                scope="constraint",
                target_constraint_id=target.constraint_id,
            )

    if has_replacement_cue(user_message):
        for successor in successors:
            same_attribute = state.active_constraints(successor.attribute)
            if (
                len(same_attribute) == 1
                and _is_distinct_successor(successor, same_attribute)
            ):
                return ReplacementEvidence(
                    scope="constraint",
                    target_constraint_id=same_attribute[-1].constraint_id,
                )
    return None


def validate_replacement_evidence(
    user_message: str,
    state: ConstraintState,
    mutations: Sequence[object],
    *,
    grounding_mutations: Sequence[object] | None = None,
) -> None:
    """Reject model-proposed replacements not licensed by explicit evidence."""

    from starter.constraint_state import ReplaceConstraint, ReplaceProductIntent

    evidence_mutations = (
        mutations if grounding_mutations is None else grounding_mutations
    )
    successors = [
        mutation
        for mutation in evidence_mutations
        if isinstance(mutation, AddConstraint)
    ]
    proposed_successors = [
        mutation
        for mutation in mutations
        if isinstance(mutation, AddConstraint)
    ]
    evidence = classify_replacement_evidence(user_message, state, successors)
    product_replacements = [
        mutation
        for mutation in mutations
        if isinstance(mutation, ReplaceProductIntent)
    ]
    constraint_replacements = [
        mutation
        for mutation in mutations
        if isinstance(mutation, ReplaceConstraint)
    ]
    replacements = [*product_replacements, *constraint_replacements]
    if not replacements and evidence is None:
        return
    if not replacements:
        raise ReplacementEvidenceError(
            "explicit replacement evidence requires a matching replacement"
        )
    if evidence is None:
        raise ReplacementEvidenceError(
            "replacement lacks Explicit Replacement Evidence in the latest turn"
        )
    if any(
        mutation.raw_phrase.strip() != user_message.strip()
        for mutation in replacements
    ):
        raise ReplacementEvidenceError(
            "replacement provenance must equal the latest customer turn"
        )
    if evidence.scope == "product_intent":
        if len(product_replacements) != 1 or constraint_replacements:
            raise ReplacementEvidenceError(
                "Product Intent evidence requires one Product Intent replacement"
            )
        if product_replacements[0].product_intent_id != state.active_product_intent_id:
            raise ReplacementEvidenceError(
                "Product Intent evidence must replace the active Product Intent"
            )
        expected_successors = {
            _successor_signature(successor)
            for successor in successors
            if successor.scope == "product_intent"
            and not isinstance(successor, ReplaceConstraint)
        }
        actual_successors = {
            _successor_signature(successor)
            for successor in proposed_successors
            if successor.scope == "product_intent"
            and not isinstance(successor, ReplaceConstraint)
        }
        if actual_successors != expected_successors:
            raise ReplacementEvidenceError(
                "Product Intent successor does not match the latest customer turn"
            )
        return
    if product_replacements or len(constraint_replacements) != 1:
        raise ReplacementEvidenceError(
            "attribute evidence may replace only one Constraint"
        )
    if constraint_replacements[0].constraint_id != evidence.target_constraint_id:
        raise ReplacementEvidenceError(
            "attribute evidence targets a different active Constraint"
        )
    expected_replacements = {
        _successor_signature(successor)
        for successor in successors
        if isinstance(successor, ReplaceConstraint)
    }
    actual_replacements = {
        _successor_signature(successor)
        for successor in constraint_replacements
    }
    if actual_replacements != expected_replacements:
        raise ReplacementEvidenceError(
            "Constraint successor does not match the latest customer turn"
        )
