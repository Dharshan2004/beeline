from __future__ import annotations

import re
from collections import defaultdict
from typing import Mapping

from starter.constraint_state import (
    AddConstraint,
    Constraint,
    ConstraintState,
    DismissAttribute,
    ReintroduceConstraint,
    ReplaceConstraint,
    ReplaceProductIntent,
    TurnPlan,
)


WORD_NORMALIZATIONS = {
    "grey": "gray",
    "shoes": "shoe",
    "boots": "boot",
    "slippers": "slipper",
    "sandals": "sandal",
    "sneakers": "sneaker",
    "dresses": "dress",
    "shirts": "shirt",
    "tops": "top",
    "jackets": "jacket",
    "coats": "coat",
    "socks": "sock",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SOFT_RE = re.compile(
    r"\b(?:prefer|preference|ideally|would\s+like|nice\s+to\s+have)\b",
    re.IGNORECASE,
)
HARD_RE = re.compile(
    r"\b(?:must|need|require|requirement|only|have\s+to|has\s+to|what\s+matters)\b",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|rather\s+than|no\s+longer|changed\s+my\s+mind|"
    r"switch(?:ing)?\s+to|ignore\s+(?:my\s+)?(?:earlier|previous)|"
    r"forget\s+(?:my\s+)?(?:earlier|previous))\b",
    re.IGNORECASE,
)
BOUNDARY_RE = re.compile(
    r"\b(?:do\s+not|don't)\s+have\s+(?:an?\s+|any\s+)?(?:additional\s+)?preference\b|"
    r"\bno\s+(?:additional\s+)?preference\b|"
    r"\b(?:do\s+not|don't)\s+care\b|\bno\s+opinion\b|"
    r"\b(?:use\s+your\s+judg(?:e)?ment|does\s+not\s+matter|doesn't\s+matter|"
    r"anything\s+is\s+fine|any\s+\w+\s+is\s+fine)\b",
    re.IGNORECASE,
)
SESSION_SCOPE_RE = re.compile(
    r"\b(?:whatever|anything)\s+i\s+buy\b|\bfor\s+anything\s+i\s+buy\b|"
    r"\bacross\s+(?:all|every)\s+product",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return " ".join(
        WORD_NORMALIZATIONS.get(token.lower(), token.lower())
        for token in TOKEN_RE.findall(text)
    )


def value_variants(value: str) -> tuple[str, ...]:
    variants = [
        value,
        *[
            source
            for source, target in WORD_NORMALIZATIONS.items()
            if target == value
        ],
    ]
    return tuple(dict.fromkeys(variants))


def _raw_value(segment: str, value: str) -> str:
    pattern = "|".join(
        re.escape(variant).replace(r"\ ", r"[\s-]+")
        for variant in sorted(value_variants(value), key=len, reverse=True)
    )
    match = re.search(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", segment, re.I)
    return match.group(0) if match else value


def _boundary_attribute(
    message: str,
    supported_values: Mapping[str, set[str]],
    last_asked_attribute: str | None,
) -> str | None:
    if not BOUNDARY_RE.search(message):
        return None
    normalized = normalize_text(message)
    for attribute in supported_values:
        readable = attribute.replace("_", " ")
        if re.search(rf"(?<![a-z0-9]){re.escape(readable)}(?![a-z0-9])", normalized):
            return attribute
    return last_asked_attribute


def _active_for_attribute(
    state: ConstraintState,
    attribute: str,
) -> list[Constraint]:
    return [
        constraint
        for constraint in state.constraints
        if constraint.status == "active" and constraint.attribute == attribute
    ]


def interpret_turn(
    user_message: str,
    *,
    turn: int,
    state: ConstraintState,
    supported_values: Mapping[str, set[str]],
    last_asked_attribute: str | None = None,
) -> TurnPlan:
    mutations: list = []
    boundary_attribute = _boundary_attribute(
        user_message,
        supported_values,
        last_asked_attribute,
    )
    if boundary_attribute is not None:
        mutations.append(DismissAttribute(boundary_attribute, user_message))

    segments = [
        segment.strip()
        for segment in re.split(
            r"\s*(?:,|;|\.|\bbut\b|"
            r"\band\s+(?=(?:i\s+)?(?:need|must|require|prefer|want)\b))\s*",
            user_message,
            flags=re.I,
        )
        if segment.strip()
    ]
    found: list[tuple[int, str, str, str, str, float, str, int]] = []
    for segment_index, segment in enumerate(segments):
        if BOUNDARY_RE.search(segment):
            continue
        if (
            turn > 1
            and not HARD_RE.search(segment)
            and not SOFT_RE.search(segment)
            and not OVERRIDE_RE.search(user_message)
        ):
            continue
        candidate = re.split(
            r"\b(?:rather\s+than|instead\s+of)\b",
            segment,
            maxsplit=1,
            flags=re.I,
        )[0]
        normalized = normalize_text(candidate)
        classification = "soft" if SOFT_RE.search(segment) else "hard"
        confidence = 0.95 if (SOFT_RE.search(segment) or HARD_RE.search(segment)) else 0.85
        scope = "session" if SESSION_SCOPE_RE.search(segment) else "product_intent"
        for attribute, values in supported_values.items():
            for value in values:
                match = re.search(
                    rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])",
                    normalized,
                )
                if match:
                    found.append((
                        segment_index,
                        attribute,
                        value,
                        classification,
                        _raw_value(candidate, value),
                        confidence,
                        scope,
                        match.start(),
                    ))

    grouped: dict[tuple[int, str], list[tuple]] = defaultdict(list)
    for item in found:
        grouped[(item[0], item[1])].append(item)

    additions: list[AddConstraint] = []
    for (segment_index, attribute), items in sorted(grouped.items()):
        items.sort(key=lambda item: item[7])
        classifications = {item[3] for item in items}
        if len(classifications) != 1:
            continue
        values = tuple(dict.fromkeys(item[2] for item in items))
        segment = segments[segment_index]
        match_rule = "any" if len(values) > 1 and re.search(r"\bor\b", segment, re.I) else "all"
        additions.append(AddConstraint(
            attribute=attribute,
            values=values,
            match_rule=match_rule,
            classification=items[0][3],
            scope=items[0][6],
            raw_phrase=" or ".join(item[4] for item in items)
            if match_rule == "any"
            else " and ".join(item[4] for item in items),
            confidence=items[0][5],
        ))

    active_category = _active_for_attribute(state, "category")
    incoming_category = next(
        (item for item in additions if item.attribute == "category"),
        None,
    )
    active_product_constraints = [
        constraint
        for constraint in state.constraints
        if constraint.status == "active" and constraint.scope == "product_intent"
    ]
    replaces_product_intent = bool(
        incoming_category
        and active_product_constraints
        and (
            not active_category
            or set(incoming_category.values) != set(active_category[-1].values)
        )
        and OVERRIDE_RE.search(user_message)
    )
    if replaces_product_intent:
        mutations.append(ReplaceProductIntent(
            product_intent_id=state.active_product_intent_id,
            raw_phrase=user_message,
        ))

    explicit_constraint_override = bool(
        re.search(
            r"\b(?:ignore|forget)\s+(?:my\s+)?(?:earlier|previous)\s+preference\b",
            user_message,
            re.I,
        )
    )
    override_target = next(
        (
            constraint
            for constraint in reversed(state.constraints)
            if constraint.status == "active" and constraint.classification == "soft"
        ),
        None,
    ) if explicit_constraint_override else None
    if explicit_constraint_override and override_target is None:
        override_target = next(
            (
                constraint
                for constraint in reversed(state.constraints)
                if constraint.status == "active" and constraint.attribute != "category"
            ),
            None,
        )
    used_override_target = False

    for addition in additions:
        active = _active_for_attribute(state, addition.attribute)
        if replaces_product_intent and addition.scope == "product_intent":
            mutations.append(addition)
            continue
        if not active:
            if override_target is not None and not used_override_target:
                mutations.append(ReplaceConstraint(
                    constraint_id=override_target.constraint_id,
                    **addition.__dict__,
                ))
                used_override_target = True
                continue
            mutation_type = (
                ReintroduceConstraint
                if addition.attribute in state.dismissed_attributes
                else AddConstraint
            )
            mutations.append(mutation_type(**addition.__dict__))
            continue
        if any(
            constraint.values == addition.values
            and constraint.classification == addition.classification
            and constraint.scope == addition.scope
            for constraint in active
        ):
            continue
        if OVERRIDE_RE.search(user_message):
            mutations.append(ReplaceConstraint(
                constraint_id=active[-1].constraint_id,
                **addition.__dict__,
            ))
            if (
                override_target is not None
                and active[-1].constraint_id == override_target.constraint_id
            ):
                used_override_target = True

    return TurnPlan(
        expected_state_revision=state.revision,
        source_turn=turn,
        mutations=tuple(mutations),
    )
