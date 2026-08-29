from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
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
CONSTRAINT_VALUES = {
    "category": (
        "shoe", "boot", "slipper", "sandal", "sneaker", "dress", "shirt",
        "top", "pants", "jeans", "jacket", "coat", "sock", "jewelry",
    ),
    "material": (
        "cotton", "polyester", "nylon", "leather", "wool", "spandex",
        "silk", "rayon", "fabric",
    ),
    "color": (
        "black", "white", "blue", "red", "pink", "green", "brown",
        "gray", "purple", "yellow", "orange",
    ),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work"),
    "feature": (
        "breathable", "lightweight", "waterproof", "water resistant",
        "insulated", "stretch", "pockets", "zipper",
    ),
}
EXPLICIT_REQUIREMENT_RE = re.compile(
    r"(?:key\s+requirement\s+is|requirement\s+is|what\s+matters\s+is|"
    r"what\s+i\s+need\s+is)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)
SOFT_PREFERENCE_RE = re.compile(
    r"\b(?:prefer|preference|ideally|would\s+like|nice\s+to\s+have)\b",
    re.IGNORECASE,
)
HARD_CONSTRAINT_RE = re.compile(
    r"\b(?:must|need|require|requirement|only|have\s+to|has\s+to)\b",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead|rather\s+than|no\s+longer|changed\s+my\s+mind|"
    r"switch(?:ing)?\s+to|"
    r"ignore\s+(?:my\s+)?(?:earlier|previous)|forget\s+(?:my\s+)?(?:earlier|previous))\b",
    re.IGNORECASE,
)
BROAD_OVERRIDE_RE = re.compile(
    r"\b(?:no\s+longer|changed\s+my\s+mind|"
    r"ignore\s+(?:my\s+)?(?:earlier|previous)|forget\s+(?:my\s+)?(?:earlier|previous))\b",
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
QUESTION_ORDER = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _normalized_text(text: str) -> str:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return " ".join(WORD_NORMALIZATIONS.get(token, token) for token in tokens)


def _contains_value(text: str, value: str) -> bool:
    return f" {_normalized_text(value)} " in f" {text} "


def _raw_value_pattern(normalized_value: str) -> str:
    variants = [
        normalized_value,
        *[
            source
            for source, target in WORD_NORMALIZATIONS.items()
            if target == normalized_value
        ],
    ]
    patterns = [
        re.escape(variant).replace(r"\ ", r"[\s-]+")
        for variant in sorted(variants, key=len, reverse=True)
    ]
    return "(?:" + "|".join(patterns) + ")"


def _constraint_query_terms(normalized_value: str) -> list[str]:
    variants = [
        normalized_value,
        *[
            source
            for source, target in WORD_NORMALIZATIONS.items()
            if target == normalized_value
        ],
    ]
    return list(dict.fromkeys(
        term
        for variant in variants
        for term in _terms(variant)
    ))


@dataclass(frozen=True)
class Constraint:
    """One validated, provenance-rich customer constraint."""

    attribute: str
    raw_phrase: str
    normalized_value: str
    classification: str
    source_turn: int
    confidence: float
    status: str = "active"

    def as_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "raw_phrase": self.raw_phrase,
            "normalized_value": self.normalized_value,
            "classification": self.classification,
            "source_turn": self.source_turn,
            "confidence": self.confidence,
            "status": self.status,
        }


@dataclass
class ConstraintState:
    constraints: list[Constraint] = field(default_factory=list)
    dismissed_attributes: dict[str, dict] = field(default_factory=dict)
    last_asked_attribute: str | None = None


class Agent:
    """Offline BM25 agent with validated, session-local Constraint State."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, ConstraintState] = {}
        self._product_text: dict[str, str] = {}
        self._supported_values: dict[str, set[str]] = {
            attribute: set() for attribute in CONSTRAINT_VALUES
        }
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                searchable = " ".join(
                    _text(product.get(field_name))
                    for field_name in (
                        "title", "categories", "features", "details", "store", "description"
                    )
                )
                normalized_searchable = _normalized_text(searchable)
                parent_asin = str(product["parent_asin"])
                existing_text = self._product_text.get(parent_asin, "")
                self._product_text[parent_asin] = (
                    f"{existing_text} {normalized_searchable}".strip()
                )
                for attribute, values in CONSTRAINT_VALUES.items():
                    for value in values:
                        if _contains_value(normalized_searchable, value):
                            self._supported_values[attribute].add(_normalized_text(value))
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = ConstraintState()

    def get_constraint_state(self, session_id: str) -> list[dict]:
        """Return a copy of the inspectable Constraint State for a session."""
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before reading constraint state")
        return [
            constraint.as_dict()
            for constraint in self._sessions[session_id].constraints
        ]

    def get_dismissed_attributes(self, session_id: str) -> list[dict]:
        """Return Boundary Response history without exposing mutable state."""
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before reading constraint state")
        return [
            dict(dismissal)
            for dismissal in self._sessions[session_id].dismissed_attributes.values()
        ]

    def _boundary_attribute(
        self,
        session_id: str,
        user_message: str,
    ) -> str | None:
        if not BOUNDARY_RE.search(user_message):
            return None
        normalized_message = _normalized_text(user_message)
        for attribute in QUESTION_ORDER:
            attribute_pattern = re.escape(attribute).replace(r"_", r"[\s_]+")
            if re.search(
                rf"(?<![a-z0-9]){attribute_pattern}(?![a-z0-9])",
                normalized_message,
            ):
                return attribute
        return self._sessions[session_id].last_asked_attribute

    def _dismiss_attribute(
        self,
        session_id: str,
        attribute: str,
        raw_phrase: str,
        turn: int,
    ) -> None:
        state = self._sessions[session_id]
        state.constraints = [
            replace(constraint, status="dismissed")
            if constraint.attribute == attribute and constraint.status == "active"
            else constraint
            for constraint in state.constraints
        ]
        state.dismissed_attributes[attribute] = {
            "attribute": attribute,
            "raw_phrase": raw_phrase,
            "source_turn": turn,
            "status": "dismissed",
        }

    def _extract_constraint(self, user_message: str, turn: int) -> Constraint | None:
        explicit_match = EXPLICIT_REQUIREMENT_RE.search(user_message)
        if (
            turn > 1
            and explicit_match is None
            and not SOFT_PREFERENCE_RE.search(user_message)
            and not HARD_CONSTRAINT_RE.search(user_message)
            and not OVERRIDE_RE.search(user_message)
        ):
            return None
        candidate_phrase = (
            explicit_match.group("value") if explicit_match else user_message
        )
        candidate_phrase = re.split(
            r"\b(?:rather\s+than|instead\s+of)\b",
            candidate_phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        normalized_candidate = _normalized_text(candidate_phrase)
        matches: list[tuple[int, int, str, str]] = []
        for attribute, supported_values in self._supported_values.items():
            for value in supported_values:
                value_match = re.search(
                    rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])",
                    normalized_candidate,
                )
                if value_match:
                    matches.append(
                        (value_match.start(), -len(value), attribute, value)
                    )
        if not matches:
            return None

        non_category_matches = [
            match for match in matches if match[2] != "category"
        ]
        eligible_matches = non_category_matches or matches
        selected = max(eligible_matches) if OVERRIDE_RE.search(user_message) else min(eligible_matches)
        _, _, attribute, normalized_value = selected
        raw_match = re.search(
            rf"(?<![a-z0-9]){_raw_value_pattern(normalized_value)}(?![a-z0-9])",
            candidate_phrase,
            re.IGNORECASE,
        )
        raw_phrase = raw_match.group(0) if raw_match else normalized_value
        if explicit_match is not None:
            classification = "hard"
            confidence = 0.99
        elif SOFT_PREFERENCE_RE.search(user_message):
            classification = "soft"
            confidence = 0.95
        else:
            classification = "hard"
            confidence = 0.95 if HARD_CONSTRAINT_RE.search(user_message) else 0.85
        return Constraint(
            attribute=attribute,
            raw_phrase=raw_phrase,
            normalized_value=normalized_value,
            classification=classification,
            source_turn=turn,
            confidence=confidence,
        )

    def _record_constraint(self, session_id: str, constraint: Constraint) -> None:
        state = self._sessions[session_id]
        state.dismissed_attributes.pop(constraint.attribute, None)
        active_same_attribute = [
            existing
            for existing in state.constraints
            if existing.attribute == constraint.attribute and existing.status == "active"
        ]
        if any(
            existing.normalized_value == constraint.normalized_value
            and existing.classification == constraint.classification
            for existing in active_same_attribute
        ):
            return
        if active_same_attribute:
            state.constraints = [
                replace(existing, status="superseded")
                if existing.attribute == constraint.attribute and existing.status == "active"
                else existing
                for existing in state.constraints
            ]
        state.constraints.append(constraint)

    def _supersede_prior_for_override(
        self,
        session_id: str,
        incoming: Constraint,
    ) -> None:
        state = self._sessions[session_id]
        active_other_constraints = [
            constraint
            for constraint in state.constraints
            if constraint.status == "active" and constraint.attribute != incoming.attribute
        ]
        soft_constraints = [
            constraint
            for constraint in active_other_constraints
            if constraint.classification == "soft"
        ]
        superseded = soft_constraints or active_other_constraints[-1:]
        superseded_ids = {id(constraint) for constraint in superseded}
        state.constraints = [
            replace(constraint, status="superseded")
            if id(constraint) in superseded_ids
            else constraint
            for constraint in state.constraints
        ]

    def _next_ask_attribute(self, session_id: str) -> str | None:
        state = self._sessions[session_id]
        active_attributes = {
            constraint.attribute
            for constraint in state.constraints
            if constraint.status == "active"
        }
        for attribute in QUESTION_ORDER:
            if not self._supported_values.get(attribute):
                continue
            if attribute in active_attributes or attribute in state.dismissed_attributes:
                continue
            state.last_asked_attribute = attribute
            return attribute
        state.last_asked_attribute = None
        return None

    def _matches_hard_constraints(
        self,
        parent_asin: str,
        constraints: list[Constraint],
    ) -> bool:
        product_text = self._product_text.get(parent_asin, "")
        return all(
            _contains_value(product_text, constraint.normalized_value)
            for constraint in constraints
            if constraint.status == "active" and constraint.classification == "hard"
        )

    def _soft_match_count(
        self,
        parent_asin: str,
        constraints: list[Constraint],
    ) -> int:
        product_text = self._product_text.get(parent_asin, "")
        return sum(
            _contains_value(product_text, constraint.normalized_value)
            for constraint in constraints
            if constraint.status == "active" and constraint.classification == "soft"
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
        boundary_attribute = self._boundary_attribute(session_id, user_message)
        if boundary_attribute is not None:
            self._dismiss_attribute(
                session_id,
                boundary_attribute,
                user_message,
                turn,
            )
        else:
            extracted_constraint = self._extract_constraint(user_message, turn)
            if extracted_constraint is not None:
                if BROAD_OVERRIDE_RE.search(user_message):
                    self._supersede_prior_for_override(
                        session_id,
                        extracted_constraint,
                    )
                self._record_constraint(session_id, extracted_constraint)
        constraints = self._sessions[session_id].constraints
        inactive_terms = {
            term
            for constraint in constraints
            if constraint.status != "active"
            for term in _constraint_query_terms(constraint.normalized_value)
        }
        constraint_terms = [
            term
            for constraint in constraints
            if constraint.status == "active"
            for term in _constraint_query_terms(constraint.normalized_value)
        ]
        message_terms = [
            term for term in _terms(user_message) if term not in inactive_terms
        ]
        unique_terms = list(dict.fromkeys([
            *message_terms,
            *constraint_terms,
        ]))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        recommendations: list[dict] = []
        if expression and top_k > 0:
            rows = self.connection.execute(
                "SELECT parent_asin, "
                "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS relevance "
                "FROM products WHERE products MATCH ? ORDER BY relevance",
                (expression,),
            )
            ranked_candidates: dict[str, tuple[int, float]] = {}
            for row in rows:
                parent_asin = str(row[0])
                if not self._matches_hard_constraints(parent_asin, constraints):
                    continue
                score = float(row[1])
                candidate = (
                    self._soft_match_count(parent_asin, constraints),
                    score,
                )
                existing = ranked_candidates.get(parent_asin)
                if existing is None or score < existing[1]:
                    ranked_candidates[parent_asin] = candidate
            if constraints and len(ranked_candidates) < top_k:
                for parent_asin in self._product_text:
                    if parent_asin in ranked_candidates:
                        continue
                    if not self._matches_hard_constraints(parent_asin, constraints):
                        continue
                    ranked_candidates[parent_asin] = (
                        self._soft_match_count(parent_asin, constraints),
                        float("inf"),
                    )
                    if len(ranked_candidates) >= top_k:
                        break
            ordered = sorted(
                ranked_candidates,
                key=lambda parent_asin: (
                    -ranked_candidates[parent_asin][0],
                    ranked_candidates[parent_asin][1],
                    parent_asin,
                ),
            )
            recommendations = [
                {"parent_asin": parent_asin}
                for parent_asin in ordered[:top_k]
            ]
        ask_attribute = self._next_ask_attribute(session_id)
        message = "Here are the closest matches I found."
        if ask_attribute is not None:
            readable_attribute = ask_attribute.replace("_", " ")
            message = (
                f"{message} Do you have a preference for {readable_attribute}?"
            )
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
