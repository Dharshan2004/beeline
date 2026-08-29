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
        self.value_index: dict[str, dict[str, set[str]]] = {
            attribute: {} for attribute in CONSTRAINT_VALUES
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
                            normalized_value = normalize_text(value)
                            self.supported_values[attribute].add(normalized_value)
                            self.value_index[attribute].setdefault(
                                normalized_value,
                                set(),
                            ).add(parent_asin)
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

    def hybrid_route_scores(
        self,
        user_message: str,
        constraints: list[Constraint],
        dense_candidates: list[tuple[str, float]],
        route_limit: int = 100,
        enabled_routes: set[str] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Return independent, higher-is-better scores for each Retrieval Route."""
        if route_limit <= 0:
            return {"structured": [], "bm25": [], "dense": []}
        enabled = (
            {"structured", "bm25", "dense"}
            if enabled_routes is None
            else set(enabled_routes)
        )
        active = [item for item in constraints if item.status == "active"]
        hard = [item for item in active if item.classification == "hard"]

        structured: list[tuple[str, float]] = []
        if "structured" in enabled:
            structured_ids: set[str] = set()
            for constraint in active:
                structured_ids.update(self._constraint_identifiers(constraint))
            hard_eligible: set[str] | None = None
            for constraint in hard:
                matches = self._constraint_identifiers(constraint)
                hard_eligible = (
                    matches
                    if hard_eligible is None
                    else hard_eligible.intersection(matches)
                )
            if hard_eligible is not None:
                structured_ids.intersection_update(hard_eligible)
            structured_scores = {
                identifier: float(sum(
                    _soft_constraint_score(self.product_text[identifier], item)
                    * (2 if item.classification == "hard" else 1)
                    for item in active
                ))
                for identifier in structured_ids
                if self._eligible(identifier, hard)
            }
            if active and len(structured_scores) < route_limit:
                backfill_ids = (
                    sorted(hard_eligible)
                    if hard_eligible is not None
                    else self.product_text
                )
                for identifier in backfill_ids:
                    if identifier in structured_scores:
                        continue
                    structured_scores[identifier] = 0.0
                    if len(structured_scores) >= route_limit:
                        break
            structured = list(structured_scores.items())
            structured.sort(key=lambda item: (-item[1], item[0]))
            if len(structured) > route_limit:
                # Never split a tied score group: cutting ties by ASIN order would
                # make route membership, and therefore fused evidence, arbitrary.
                cutoff_score = structured[route_limit - 1][1]
                end = route_limit
                while end < len(structured) and structured[end][1] == cutoff_score:
                    end += 1
                structured = structured[:end]

        inactive_terms = {
            term
            for constraint in constraints
            if constraint.status != "active"
            for value in constraint.values
            for variant in value_variants(value)
            for term in query_terms(variant)
        }
        terms = [
            term for term in query_terms(user_message) if term not in inactive_terms
        ]
        terms.extend(
            term
            for constraint in hard
            for value in constraint.values
            for variant in value_variants(value)
            for term in query_terms(variant)
        )
        bm25 = []
        if "bm25" in enabled:
            bm25 = [
                (identifier, -score)
                for identifier, score in self._search(terms)
                if self._eligible(identifier, hard)
            ][:route_limit]

        dense: list[tuple[str, float]] = []
        dense_seen: set[str] = set()
        if "dense" in enabled:
            for parent_asin, score in dense_candidates:
                identifier = str(parent_asin)
                if identifier in dense_seen or not self._eligible(identifier, hard):
                    continue
                dense_seen.add(identifier)
                dense.append((identifier, float(score)))
                if len(dense) >= route_limit:
                    break
        return {
            "structured": structured,
            "bm25": bm25,
            "dense": dense,
        }

    def _constraint_identifiers(self, constraint: Constraint) -> set[str]:
        value_sets = [
            self.value_index.get(constraint.attribute, {}).get(value, set())
            for value in constraint.values
        ]
        if not value_sets:
            return set()
        if constraint.match_rule == "any":
            return set.union(*value_sets)
        return set.intersection(*value_sets)

    def _eligible(
        self,
        parent_asin: str,
        hard_constraints: list[Constraint],
    ) -> bool:
        product_text = self.product_text.get(parent_asin)
        return product_text is not None and all(
            _constraint_matches(product_text, item)
            for item in hard_constraints
        )

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
