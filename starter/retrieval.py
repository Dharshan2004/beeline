from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from retrieval.product_text import product_text
from starter.constraint_state import Constraint
from starter.turn_interpreter import normalize_text, value_variants


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# A cross-encoder truncates its pair at a few hundred tokens, so keeping the
# whole catalog rendering in memory would cost tens of megabytes that the model
# never reads. This budget comfortably covers the sequence length in
# 'retrieval.reranker' while bounding the resident cost of the map.
RERANK_TEXT_CHAR_LIMIT = 512
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


# A price amount only counts as budget evidence when the customer marks it as
# money (a currency sign or word, or a spending keyword nearby); bare numbers
# stay untouched so sizes and quantities never become price constraints.
_AMOUNT = r"\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars|bucks|usd)?"
_MONEY_MARKER_RE = re.compile(
    r"\$|\b(?:dollars|bucks|usd|budget|price|spend|spending|cost|costs|pay)\b",
    re.IGNORECASE,
)
_BUDGET_UPPER_RE = re.compile(
    rf"\b(?:under|below|less\s+than|at\s+most|no\s+more\s+than|up\s+to|"
    rf"max(?:imum)?(?:\s+of)?|within)\s+{_AMOUNT}",
    re.IGNORECASE,
)
_BUDGET_LOWER_RE = re.compile(
    rf"\b(?:over|above|at\s+least|more\s+than|minimum(?:\s+of)?)\s+{_AMOUNT}",
    re.IGNORECASE,
)
_BUDGET_RANGE_RE = re.compile(
    rf"\bbetween\s+{_AMOUNT}\s+and\s+{_AMOUNT}", re.IGNORECASE
)
_BUDGET_NEAR_RE = re.compile(
    rf"\b(?:around|about|roughly|approximately|near)\s+{_AMOUNT}",
    re.IGNORECASE,
)
_BUDGET_BARE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")


def parse_budget(text: str) -> tuple[float, float] | None:
    """Extract a (low, high) price range from free text, or None.

    Interprets ordinary spending language — "under $50", "between 20 and 40
    dollars", "around $60", a bare "$35" — with an approximate mention widened
    to a tolerance band, because customers naming a rough figure rarely mean
    it exactly.
    """
    if not text or not _MONEY_MARKER_RE.search(text):
        return None
    match = _BUDGET_RANGE_RE.search(text)
    if match:
        low, high = sorted((float(match.group(1)), float(match.group(2))))
        return (low, high)
    match = _BUDGET_UPPER_RE.search(text)
    if match:
        return (0.0, float(match.group(1)))
    match = _BUDGET_LOWER_RE.search(text)
    if match:
        return (float(match.group(1)), float("inf"))
    match = _BUDGET_NEAR_RE.search(text) or _BUDGET_BARE_RE.search(text)
    if match:
        amount = float(match.group(1))
        return (amount * 0.75, amount * 1.3)
    return None


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
        # Cased, field-labelled renderings for the cross-encoder. 'product_text'
        # above is normalized for lexical matching and reads poorly to a model
        # trained on natural passages.
        self.rerank_text: dict[str, str] = {}
        self.supported_values: dict[str, set[str]] = {
            attribute: set() for attribute in CONSTRAINT_VALUES
        }
        self.value_index: dict[str, dict[str, set[str]]] = {
            attribute: {} for attribute in CONSTRAINT_VALUES
        }
        self.price: dict[str, float] = {}
        self.rating_number: dict[str, int] = {}
        self.rating_average: dict[str, float] = {}
        self._build_index()

    # Bayesian-average popularity prior: a listing's rating pulled toward the
    # global mean by a pseudo-count, so a 5.0 with three reviews cannot outrank
    # a 4.6 with four thousand.
    BAYESIAN_PSEUDO_COUNT = 20.0
    BAYESIAN_GLOBAL_MEAN = 4.2

    def popularity_prior(self, parent_asin: str) -> float:
        count = float(self.rating_number.get(parent_asin, 0))
        average = self.rating_average.get(parent_asin, self.BAYESIAN_GLOBAL_MEAN)
        smoothed = (
            self.BAYESIAN_PSEUDO_COUNT * self.BAYESIAN_GLOBAL_MEAN
            + count * average
        ) / (self.BAYESIAN_PSEUDO_COUNT + count)
        # Review volume dominates among near-identical listings; the smoothed
        # rating separates equal-volume listings.
        return count + 10.0 * smoothed

    def within_budget(self, parent_asin: str, budget: tuple[float, float]) -> bool:
        """True when the product's price fits the range or is unknown.

        A missing price is treated as compatible so budget evidence reorders
        rather than eliminates; hard elimination on incomplete catalog data
        would drop valid products.
        """
        price = self.price.get(parent_asin)
        if price is None:
            return True
        low, high = budget
        return low <= price <= high

    def rerank_documents(self, candidates: list[str]) -> list[str]:
        """Render catalog-valid candidates for the local cross-encoder."""
        return [self.rerank_text[parent_asin] for parent_asin in candidates]

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
                raw_rating_number = product.get("rating_number")
                if isinstance(raw_rating_number, (int, float)):
                    self.rating_number[parent_asin] = int(raw_rating_number)
                raw_rating_average = product.get("average_rating")
                if isinstance(raw_rating_average, (int, float)):
                    self.rating_average[parent_asin] = float(raw_rating_average)
                raw_price = product.get("price")
                if isinstance(raw_price, (int, float)):
                    self.price[parent_asin] = float(raw_price)
                elif isinstance(raw_price, str):
                    price_match = re.search(r"\d+(?:\.\d+)?", raw_price)
                    if price_match:
                        self.price[parent_asin] = float(price_match.group(0))
                self.product_text[parent_asin] = (
                    f"{self.product_text.get(parent_asin, '')} {normalized}".strip()
                )
                if parent_asin not in self.rerank_text:
                    self.rerank_text[parent_asin] = product_text(
                        product,
                    )[:RERANK_TEXT_CHAR_LIMIT]
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
        dialog_text: str = "",
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
            structured = structured[:route_limit]

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
        # Accumulated dialog evidence: the customer's need is everything they
        # have said, so earlier disclosures stay in the lexical query. Terms
        # from superseded constraints are excluded, which keeps Intent
        # Overrides from dragging replaced requirements back in.
        terms.extend(
            term
            for term in query_terms(dialog_text)
            if term not in inactive_terms
        )
        bm25 = []
        if "bm25" in enabled:
            # Dual-query lexical evidence: the accumulated-dialog query
            # carries the whole session, while a fresh latest-message query
            # lets the newest statement rescue a session whose older wording
            # has drifted from the current need (topic-shift recovery). A
            # product keeps the better of its two scores.
            full_scores = dict(self._search(terms))
            recent_terms = [
                term
                for term in query_terms(user_message)
                if term not in inactive_terms
            ]
            recent_terms.extend(
                term
                for constraint in hard
                for value in constraint.values
                for variant in value_variants(value)
                for term in query_terms(variant)
            )
            if recent_terms != terms:
                for identifier, score in self._search(recent_terms):
                    previous = full_scores.get(identifier)
                    if previous is None or score < previous:
                        full_scores[identifier] = score
            bm25 = [
                (identifier, -score)
                for identifier, score in sorted(
                    full_scores.items(), key=lambda item: (item[1], item[0])
                )
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
