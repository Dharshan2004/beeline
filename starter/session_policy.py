from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal, Mapping, Sequence

from starter.constraint_state import ConstraintState


SessionMode = Literal["buying", "browsing", "uncertain"]
SESSION_MODES: tuple[SessionMode, ...] = ("buying", "browsing", "uncertain")
SESSION_POLICY_VERSION = "shopping-session-policy-v1"
ALLOWED_ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)

_BROWSING_RE = re.compile(
    r"\b(?:still\s+explor(?:e|ing)|just\s+(?:looking|browsing)|browsing|"
    r"exploring|show\s+me\s+(?:some\s+)?(?:ideas|options)|open\s+to\s+options)\b",
    re.IGNORECASE,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:not\s+sure|unsure|uncertain|haven't\s+decided|have\s+not\s+decided|"
    r"don't\s+know\s+what|do\s+not\s+know\s+what|not\s+quite\s+right)\b",
    re.IGNORECASE,
)
_BUYING_RE = re.compile(
    r"\b(?:must|need|require|requirement|have\s+to|has\s+to|ready\s+to\s+buy|"
    r"key\s+requirement|what\s+matters)\b",
    re.IGNORECASE,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:no|don't\s+have|do\s+not\s+have)\s+(?:an?\s+|any\s+|additional\s+)*"
    r"preference\b|\b(?:don't|do\s+not)\s+care\b|\buse\s+your\s+judg(?:e)?ment\b",
    re.IGNORECASE,
)

_MODE_PRIORITIES: dict[SessionMode, tuple[str, ...]] = {
    "buying": ALLOWED_ASK_ATTRIBUTES,
    "browsing": (
        "category", "use_case", "style", "feature", "material", "color",
        "size", "brand", "budget", "other",
    ),
    "uncertain": (
        "category", "use_case", "material", "feature", "color", "style",
        "size", "brand", "budget", "other",
    ),
}

_PROFILE_HINT_ATTRIBUTES = {
    "fit": "size",
    "comfort": "feature",
    "durability": "feature",
    "performance": "use_case",
    "weather": "use_case",
    "warmth": "material",
}

SESSION_POLICY_SHA256 = hashlib.sha256(json.dumps({
    "version": SESSION_POLICY_VERSION,
    "allowed_ask_attributes": ALLOWED_ASK_ATTRIBUTES,
    "mode_priorities": _MODE_PRIORITIES,
    "profile_hint_attributes": _PROFILE_HINT_ATTRIBUTES,
    "browsing_pattern": _BROWSING_RE.pattern,
    "uncertain_pattern": _UNCERTAIN_RE.pattern,
    "buying_pattern": _BUYING_RE.pattern,
    "boundary_pattern": _BOUNDARY_RE.pattern,
}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionModeDecision:
    mode: SessionMode
    reason: str


@dataclass(frozen=True)
class ClarificationCandidate:
    ask_attribute: str
    expected_value: float
    reason: str


def aggregate_profile_hints(user_profile: object) -> tuple[str, ...]:
    """Return safe aggregate tags without interpreting them as requirements."""

    if not isinstance(user_profile, Mapping):
        return ()
    raw_tags = user_profile.get("preference_tags", ())
    if not isinstance(raw_tags, (list, tuple)):
        return ()
    hints: list[str] = []
    for value in raw_tags:
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace(" ", "_")
        attribute = _PROFILE_HINT_ATTRIBUTES.get(normalized, normalized)
        if attribute in ALLOWED_ASK_ATTRIBUTES and attribute not in hints:
            hints.append(attribute)
    return tuple(hints)


def revise_session_mode(
    user_message: str,
    state: ConstraintState,
    previous_mode: SessionMode,
    *,
    proposed_mode: SessionMode | None = None,
) -> SessionModeDecision:
    """Re-evaluate Session Mode from current-turn evidence on every turn.

    Explicit current-turn language wins. A connected planner may decide genuinely
    ambiguous turns, while local Constraint State remains the deterministic
    fallback and aggregate profile hints never participate in the decision.
    """

    message = user_message.strip()
    if _UNCERTAIN_RE.search(message):
        return SessionModeDecision("uncertain", "explicit_uncertainty")
    if _BROWSING_RE.search(message):
        return SessionModeDecision("browsing", "explicit_browsing")
    if _BUYING_RE.search(message):
        return SessionModeDecision("buying", "explicit_purchase_requirement")
    if _BOUNDARY_RE.search(message):
        return SessionModeDecision(previous_mode, "boundary_preserves_mode")
    if proposed_mode in SESSION_MODES:
        return SessionModeDecision(proposed_mode, "connected_interpretation")

    active = state.active_constraints()
    if any(
        constraint.classification == "hard" and constraint.attribute != "category"
        for constraint in active
    ):
        return SessionModeDecision("buying", "active_hard_requirement")
    if active and previous_mode != "uncertain":
        return SessionModeDecision(previous_mode, "active_state_continuity")
    return SessionModeDecision("uncertain", "insufficient_purchase_evidence")


def clarification_candidates(
    mode: SessionMode,
    state: ConstraintState,
    supported_values: Mapping[str, set[str]],
    *,
    asked_attributes: Sequence[str] = (),
    profile_hints: Sequence[str] = (),
) -> tuple[ClarificationCandidate, ...]:
    """Rank attributes whose answer can still add information to this intent."""

    active_attributes = {
        constraint.attribute for constraint in state.active_constraints()
    }
    unavailable = {
        *active_attributes,
        *state.dismissed_attributes,
        *asked_attributes,
    }
    profile_hint_set = set(profile_hints)
    priorities = _MODE_PRIORITIES[mode]
    candidates: list[ClarificationCandidate] = []
    for rank, attribute in enumerate(priorities):
        values = supported_values.get(attribute, set())
        # With fewer than two catalog-supported answers, asking cannot split the
        # decision space and has no useful expected value.
        if attribute in unavailable or len(values) < 2:
            continue
        priority_value = float(len(priorities) - rank)
        diversity_value = min(len(values), 10) / 10.0
        profile_value = 1.25 if attribute in profile_hint_set else 0.0
        expected_value = priority_value + diversity_value + profile_value
        reasons = [f"{mode}_priority", "catalog_answer_diversity"]
        if profile_value:
            reasons.append("aggregate_profile_tiebreak")
        candidates.append(ClarificationCandidate(
            ask_attribute=attribute,
            expected_value=round(expected_value, 3),
            reason="+".join(reasons),
        ))
    candidates.sort(key=lambda item: (-item.expected_value, item.ask_attribute))
    return tuple(candidates)
