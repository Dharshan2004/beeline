from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from starter.constraint_state import Constraint
from starter.turn_interpreter import normalize_text, value_variants


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
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


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def query_terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _contains_value(text: str, value: str) -> bool:
    return f" {normalize_text(value)} " in f" {text} "


def _constraint_matches(product_text: str, constraint: Constraint) -> bool:
    matches = [
        _contains_value(product_text, value)
        for value in constraint.values
    ]
    return any(matches) if constraint.match_rule == "any" else all(matches)


def _soft_constraint_score(product_text: str, constraint: Constraint) -> int:
    if not _constraint_matches(product_text, constraint):
        return 0
    return sum(
        _contains_value(product_text, value)
        for value in constraint.values
    )


@dataclass(frozen=True)
class RankedCandidate:
    parent_asin: str
    soft_score: int
    lexical_score: float


class CatalogRetrieval:
    """Embedded catalog retrieval with explicit eligibility and ranking phases."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.product_text: dict[str, str] = {}
        self.supported_values: dict[str, set[str]] = {
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
                fields = tuple(
                    _text(product.get(field_name))
                    for field_name in (
                        "title", "categories", "features", "details", "store", "description"
                    )
                )
                normalized = normalize_text(" ".join(fields))
                parent_asin = str(product["parent_asin"])
                self.product_text[parent_asin] = (
                    f"{self.product_text.get(parent_asin, '')} {normalized}".strip()
                )
                for attribute, values in CONSTRAINT_VALUES.items():
                    for value in values:
                        if _contains_value(normalized, value):
                            self.supported_values[attribute].add(normalize_text(value))
                batch.append((parent_asin, *fields))
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        self.connection.commit()

    def candidate_pool(
        self,
        user_message: str,
        constraints: list[Constraint],
    ) -> list[RankedCandidate]:
        return self._candidate_pool(user_message, constraints, backfill_limit=None)

    def _candidate_pool(
        self,
        user_message: str,
        constraints: list[Constraint],
        backfill_limit: int | None,
    ) -> list[RankedCandidate]:
        active = [item for item in constraints if item.status == "active"]
        inactive_terms = {
            term
            for constraint in constraints
            if constraint.status != "active"
            for value in constraint.values
            for variant in value_variants(value)
            for term in query_terms(variant)
        }
        base_terms = [
            term for term in query_terms(user_message) if term not in inactive_terms
        ]
        hard_constraints = [
            item for item in active if item.classification == "hard"
        ]
        soft_constraints = [
            item for item in active if item.classification == "soft"
        ]
        base_terms.extend(
            term
            for constraint in hard_constraints
            for value in constraint.values
            for variant in value_variants(value)
            for term in query_terms(variant)
        )
        soft_terms = [
            term
            for constraint in soft_constraints
            for value in constraint.values
            for variant in value_variants(value)
            for term in query_terms(variant)
        ]

        scores: dict[str, float] = {}
        for terms in (base_terms, soft_terms):
            for parent_asin, score in self._search(terms):
                previous = scores.get(parent_asin)
                if previous is None or score < previous:
                    scores[parent_asin] = score

        candidates: list[RankedCandidate] = []
        for parent_asin, lexical_score in scores.items():
            product_text = self.product_text.get(parent_asin, "")
            if not all(
                _constraint_matches(product_text, constraint)
                for constraint in hard_constraints
            ):
                continue
            soft_score = sum(
                _soft_constraint_score(product_text, constraint)
                for constraint in soft_constraints
            )
            candidates.append(RankedCandidate(
                parent_asin=parent_asin,
                soft_score=soft_score,
                lexical_score=lexical_score,
            ))
        if constraints:
            present = {item.parent_asin for item in candidates}
            for parent_asin, product_text in self.product_text.items():
                if backfill_limit is not None and len(candidates) >= backfill_limit:
                    break
                if parent_asin in present:
                    continue
                if not all(
                    _constraint_matches(product_text, constraint)
                    for constraint in hard_constraints
                ):
                    continue
                candidates.append(RankedCandidate(
                    parent_asin=parent_asin,
                    soft_score=sum(
                        _soft_constraint_score(product_text, constraint)
                        for constraint in soft_constraints
                    ),
                    lexical_score=float("inf"),
                ))
        return sorted(
            candidates,
            key=lambda item: (-item.soft_score, item.lexical_score, item.parent_asin),
        )

    def recommend(
        self,
        user_message: str,
        constraints: list[Constraint],
        top_k: int,
    ) -> list[str]:
        if top_k <= 0:
            return []
        return [
            item.parent_asin
            for item in self._candidate_pool(
                user_message,
                constraints,
                backfill_limit=top_k,
            )[:top_k]
        ]

    def recommend_with_dense(
        self,
        user_message: str,
        constraints: list[Constraint],
        dense_candidates: list[tuple[str, float]],
        top_k: int,
    ) -> list[str]:
        """Preserve dense order, enforce eligibility, then backfill lexically."""
        if top_k <= 0:
            return []
        hard_constraints = [
            item
            for item in constraints
            if item.status == "active" and item.classification == "hard"
        ]
        ranked: list[str] = []
        seen: set[str] = set()
        for parent_asin, _score in dense_candidates:
            identifier = str(parent_asin)
            product_text = self.product_text.get(identifier)
            if product_text is None or identifier in seen:
                continue
            if not all(
                _constraint_matches(product_text, item)
                for item in hard_constraints
            ):
                continue
            seen.add(identifier)
            ranked.append(identifier)
            if len(ranked) >= top_k:
                return ranked

        for identifier in self.recommend(user_message, constraints, top_k):
            if identifier in seen:
                continue
            ranked.append(identifier)
            if len(ranked) >= top_k:
                break
        return ranked

    def _search(self, terms: list[str]) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(terms))[:40]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin, "
            "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS relevance "
            "FROM products WHERE products MATCH ? ORDER BY relevance",
            (expression,),
        )
        best: dict[str, float] = {}
        for parent_asin, score in rows:
            identifier = str(parent_asin)
            relevance = float(score)
            if identifier not in best or relevance < best[identifier]:
                best[identifier] = relevance
        return list(best.items())
