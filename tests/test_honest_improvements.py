"""Tests for dialog accumulation, information-value asks, and budget parsing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.retrieval import parse_budget


def write_catalog(root: Path, rows: list[dict]) -> Path:
    path = root / "catalog.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class RecordingDenseRoute:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        self.queries.append(query)
        return []

    def metrics(self) -> dict:
        return {"status": "disabled"}


class BudgetParsingTest(unittest.TestCase):
    def test_common_spending_phrases_parse_to_ranges(self) -> None:
        self.assertEqual(parse_budget("keep it under $50"), (0.0, 50.0))
        self.assertEqual(parse_budget("no more than 80 dollars"), (0.0, 80.0))
        self.assertEqual(
            parse_budget("something between $20 and $40 please"), (20.0, 40.0)
        )
        low, high = parse_budget("around $60 is my budget")
        self.assertLess(low, 60.0)
        self.assertGreater(high, 60.0)
        minimum, maximum = parse_budget("I want to spend at least $100")
        self.assertEqual(minimum, 100.0)
        self.assertEqual(maximum, float("inf"))

    def test_bare_numbers_without_money_markers_never_trigger(self) -> None:
        self.assertIsNone(parse_budget("I wear size 10 wide"))
        self.assertIsNone(parse_budget("a pack of 12 socks"))
        self.assertIsNone(parse_budget("under 30 minutes of running"))

    def test_latest_money_statement_is_found_in_free_text(self) -> None:
        self.assertEqual(parse_budget("my price cap is up to $75 today"), (0.0, 75.0))
        band = parse_budget("$35 works for me")
        self.assertIsNotNone(band)
        self.assertLessEqual(band[0], 35.0)
        self.assertGreaterEqual(band[1], 35.0)


class DialogAccumulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.catalog_path = write_catalog(Path(self._directory.name), [
            {
                "parent_asin": "HIKING_WATERPROOF",
                "title": "Trail boot",
                "features": ["waterproof membrane", "for hiking"],
                "categories": ["Clothing", "Boots"],
            },
            {
                "parent_asin": "PLAIN",
                "title": "Plain city boot",
                "features": ["everyday wear"],
                "categories": ["Clothing", "Boots"],
            },
        ])

    def test_dense_query_carries_prior_messages(self) -> None:
        route = RecordingDenseRoute()
        agent = Agent(self.catalog_path, dense_route=route)
        self.addCleanup(agent.close)
        agent.reset("session", {})
        agent.respond("session", "I need boots for hiking.", 1, 10)
        agent.respond("session", "They must be waterproof.", 2, 10)
        self.assertIn("hiking", route.queries[1])
        self.assertIn("waterproof", route.queries[1])

    def test_earlier_disclosures_keep_influencing_ranking(self) -> None:
        agent = Agent(self.catalog_path)
        self.addCleanup(agent.close)
        agent.reset("session", {})
        agent.respond("session", "I want something waterproof for hiking.", 1, 10)
        response = agent.respond("session", "Show me your best options.", 2, 10)
        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(identifiers[0], "HIKING_WATERPROOF")


class InformationValueAskTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        # Material is uniform across the pool; color splits it evenly, so the
        # informative question is color even though material precedes it in
        # the stable question order.
        self.catalog_path = write_catalog(Path(self._directory.name), [
            {"parent_asin": "A", "title": "Cotton shoe in black"},
            {"parent_asin": "B", "title": "Cotton shoe in blue"},
            {"parent_asin": "C", "title": "Cotton shoe in red"},
            {"parent_asin": "D", "title": "Cotton shoe in green"},
        ])

    def test_asks_the_attribute_that_splits_the_pool(self) -> None:
        agent = Agent(self.catalog_path)
        self.addCleanup(agent.close)
        agent.reset("session", {})
        response = agent.respond("session", "I need a cotton shoe.", 1, 10)
        self.assertEqual(response["ask_attribute"], "color")

    def test_never_repeats_an_asked_attribute(self) -> None:
        agent = Agent(self.catalog_path)
        self.addCleanup(agent.close)
        agent.reset("session", {})
        asked = []
        for turn in range(1, 6):
            response = agent.respond("session", "Show me shoes.", turn, 10)
            if response["ask_attribute"] is not None:
                asked.append(response["ask_attribute"])
        self.assertEqual(len(asked), len(set(asked)))

    def test_dismissed_attribute_is_not_asked_again(self) -> None:
        agent = Agent(self.catalog_path)
        self.addCleanup(agent.close)
        agent.reset("session", {})
        first = agent.respond("session", "I need a cotton shoe.", 1, 10)
        self.assertEqual(first["ask_attribute"], "color")
        agent.respond(
            "session",
            "I don't have a preference for color; please use your judgment.",
            2,
            10,
        )
        for turn in range(3, 6):
            response = agent.respond("session", "Show me more shoes.", turn, 10)
            self.assertNotEqual(response["ask_attribute"], "color")


class BudgetReorderingTest(unittest.TestCase):
    def test_budget_moves_affordable_products_ahead_without_eliminating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = write_catalog(Path(directory), [
                {
                    "parent_asin": "EXPENSIVE",
                    "title": "Leather boot premium",
                    "price": 250.0,
                    "categories": ["Clothing", "Boots"],
                },
                {
                    "parent_asin": "AFFORDABLE",
                    "title": "Leather boot value",
                    "price": 40.0,
                    "categories": ["Clothing", "Boots"],
                },
                {
                    "parent_asin": "UNPRICED",
                    "title": "Leather boot unknown",
                    "categories": ["Clothing", "Boots"],
                },
            ])
            agent = Agent(catalog_path)
            try:
                agent.reset("session", {})
                response = agent.respond(
                    "session", "Leather boots under $60 please.", 1, 10
                )
                identifiers = [
                    item["parent_asin"] for item in response["recommendations"]
                ]
                self.assertIn("EXPENSIVE", identifiers)
                self.assertLess(
                    identifiers.index("AFFORDABLE"),
                    identifiers.index("EXPENSIVE"),
                )
                self.assertLess(
                    identifiers.index("UNPRICED"),
                    identifiers.index("EXPENSIVE"),
                )
            finally:
                agent.close()


if __name__ == "__main__":
    unittest.main()
