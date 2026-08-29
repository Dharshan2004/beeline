from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.constraint_state import Constraint
from starter.retrieval import CatalogRetrieval


def constraint(
    attribute: str,
    values: tuple[str, ...],
    classification: str,
    match_rule: str = "all",
) -> Constraint:
    return Constraint(
        constraint_id=f"constraint-{attribute}-{classification}",
        product_intent_id="intent-1",
        scope="product_intent",
        attribute=attribute,
        values=values,
        match_rule=match_rule,
        classification=classification,
        raw_phrase=" ".join(values),
        source_turn=1,
        confidence=0.95,
    )


class CatalogRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"

    def write_catalog(self, rows: list[dict]) -> None:
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_soft_preference_augments_without_removing_base_candidates(self) -> None:
        self.write_catalog([
            {"parent_asin": "BASE", "title": "Everyday walking shoe", "features": ["black"]},
            *[
                {
                    "parent_asin": f"BLUE_{index:02d}",
                    "title": "Decorative accessory",
                    "features": ["blue"],
                }
                for index in range(12)
            ],
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        base_pool = retrieval.candidate_pool("walking shoe", [])
        preferred_pool = retrieval.candidate_pool(
            "walking shoe",
            [constraint("color", ("blue",), "soft")],
        )

        self.assertEqual([item.parent_asin for item in base_pool], ["BASE"])
        self.assertTrue(
            {item.parent_asin for item in base_pool}
            <= {item.parent_asin for item in preferred_pool}
        )
        self.assertEqual(len(preferred_pool), 13)

    def test_soft_match_orders_ahead_of_tied_nonmatch(self) -> None:
        self.write_catalog([
            {"parent_asin": "A_BLACK", "title": "Everyday shoe", "features": ["black"]},
            {"parent_asin": "B_BLUE", "title": "Everyday shoe", "features": ["blue"]},
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        ranked = retrieval.recommend(
            "everyday shoe",
            [constraint("color", ("blue",), "soft")],
            top_k=10,
        )

        self.assertEqual(ranked, ["B_BLUE", "A_BLACK"])

    def test_hard_mismatch_is_ineligible_despite_lexical_score(self) -> None:
        self.write_catalog([
            {"parent_asin": "A_LEATHER", "title": "Walking walking walking shoe", "features": ["leather"]},
            {"parent_asin": "B_COTTON", "title": "Walking shoe", "features": ["cotton"]},
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        ranked = retrieval.recommend(
            "cotton walking shoe",
            [constraint("material", ("cotton",), "hard")],
            top_k=10,
        )

        self.assertEqual(ranked, ["B_COTTON"])

    def test_soft_all_values_gives_no_partial_match_boost(self) -> None:
        self.write_catalog([
            {"parent_asin": "A_RED", "title": "Everyday shoe", "features": ["red"]},
            {"parent_asin": "B_BOTH", "title": "Everyday shoe", "features": ["red", "blue"]},
            {"parent_asin": "C_NONE", "title": "Everyday shoe", "features": ["black"]},
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        pool = retrieval.candidate_pool(
            "everyday shoe",
            [constraint("color", ("red", "blue"), "soft", match_rule="all")],
        )
        scores = {item.parent_asin: item.soft_score for item in pool}

        self.assertEqual(scores["B_BOTH"], 2)
        self.assertEqual(scores["A_RED"], 0)
        self.assertEqual(scores["C_NONE"], 0)


if __name__ == "__main__":
    unittest.main()
