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

    def test_hybrid_route_scores_keep_structured_bm25_and_dense_independent(self) -> None:
        self.write_catalog([
            {"parent_asin": "STRUCTURED", "title": "Cotton gala footwear"},
            {"parent_asin": "BM25", "title": "Formal ceremony dress"},
            {"parent_asin": "DENSE", "title": "Evening accessory"},
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        scores = retrieval.hybrid_route_scores(
            "ceremony",
            [constraint("material", ("cotton",), "soft")],
            [("DENSE", 0.95), ("UNKNOWN", 0.99)],
        )

        self.assertIn("STRUCTURED", dict(scores["structured"]))
        self.assertIn("BM25", dict(scores["bm25"]))
        self.assertEqual(scores["dense"], [("DENSE", 0.95)])

    def test_soft_preference_does_not_remove_a_base_bm25_candidate(self) -> None:
        self.write_catalog([
            {"parent_asin": "BASE", "title": "Everyday walking shoe"},
            {"parent_asin": "SOFT", "title": "Blue blue blue accessory"},
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        base = retrieval.hybrid_route_scores("walking shoe", [], [], route_limit=1)
        preferred = retrieval.hybrid_route_scores(
            "walking shoe",
            [constraint("color", ("blue",), "soft")],
            [],
            route_limit=1,
        )

        self.assertLessEqual(
            {identifier for identifier, _score in base["bm25"]},
            {identifier for identifier, _score in preferred["bm25"]},
        )

    def test_structured_route_strictly_caps_a_tied_score_group(self) -> None:
        self.write_catalog([
            {
                "parent_asin": f"A{index:02d}",
                "title": "Cotton slipper house shoe",
            }
            for index in range(7)
        ] + [
            {"parent_asin": "ZZ_TARGET", "title": "Cotton slipper house shoe"},
        ])
        retrieval = CatalogRetrieval(self.catalog_path)

        scores = retrieval.hybrid_route_scores(
            "cotton slipper",
            [constraint("material", ("cotton",), "hard")],
            [],
            route_limit=5,
        )

        self.assertEqual(
            [identifier for identifier, _score in scores["structured"]],
            ["A00", "A01", "A02", "A03", "A04"],
        )

    def test_structured_tie_cap_is_independent_of_catalog_file_order(self) -> None:
        rows = [
            {
                "parent_asin": f"M{index:02d}",
                "title": "Cotton slipper house shoe",
            }
            for index in range(7)
        ]

        def structured_for(catalog_rows: list[dict]) -> list[str]:
            self.write_catalog(catalog_rows)
            retrieval = CatalogRetrieval(self.catalog_path)
            scores = retrieval.hybrid_route_scores(
                "cotton slipper",
                [constraint("material", ("cotton",), "hard")],
                [],
                route_limit=4,
            )
            return [identifier for identifier, _score in scores["structured"]]

        self.assertEqual(structured_for(rows), ["M00", "M01", "M02", "M03"])
        self.assertEqual(structured_for(list(reversed(rows))), ["M00", "M01", "M02", "M03"])


if __name__ == "__main__":
    unittest.main()
