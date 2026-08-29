from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
CONSTRAINT_VALUES = {
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
    r"(?:key\s+requirement\s+is|requirement\s+is)\s*:\s*(?P<value>.+)$",
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
    return " ".join("gray" if token == "grey" else token for token in tokens)


def _contains_value(text: str, value: str) -> bool:
    return f" {_normalized_text(value)} " in f" {text} "


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

    def _extract_constraint(self, user_message: str, turn: int) -> Constraint | None:
        explicit_match = EXPLICIT_REQUIREMENT_RE.search(user_message)
        candidate_phrase = (
            explicit_match.group("value") if explicit_match else user_message
        )
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

        _, _, attribute, normalized_value = min(matches)
        raw_match = re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])",
            candidate_phrase,
            re.IGNORECASE,
        )
        raw_phrase = raw_match.group(0) if raw_match else normalized_value
        if SOFT_PREFERENCE_RE.search(user_message):
            classification = "soft"
            confidence = 0.95
        else:
            classification = "hard"
            confidence = 0.99 if explicit_match else (
                0.95 if HARD_CONSTRAINT_RE.search(user_message) else 0.85
            )
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
        for existing in state.constraints:
            if existing.attribute != constraint.attribute:
                continue
            # Slice 03 owns overrides. Repeated or conflicting values cannot
            # silently duplicate or replace an already active constraint here.
            return
        state.constraints.append(constraint)

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
        extracted_constraint = self._extract_constraint(user_message, turn)
        if extracted_constraint is not None:
            self._record_constraint(session_id, extracted_constraint)
        constraints = self._sessions[session_id].constraints
        constraint_terms = [
            term
            for constraint in constraints
            if constraint.status == "active"
            for term in _terms(constraint.normalized_value)
        ]
        unique_terms = list(dict.fromkeys([
            *_terms(user_message),
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
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
