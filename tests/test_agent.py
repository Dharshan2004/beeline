from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import catalog_index, evaluate
from starter.agent import Agent


class AgentContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"

    def write_catalog(self, rows: list[dict]) -> None:
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_recommendations_are_unique_and_preserve_rank_order(self) -> None:
        rows = [
            {"parent_asin": "A", "title": "Blue running shoe"}
            for _ in range(10)
        ]
        rows.append({
            "parent_asin": "B",
            "title": "Blue running shoe with many decorative details and accessories",
        })
        self.write_catalog(rows)
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        response = agent.respond("session", "blue running shoe", 1, 2)

        recommendations = [
            item["parent_asin"] for item in response["recommendations"]
        ]
        self.assertEqual(recommendations, ["A", "B"])

    def test_respond_requires_reset_for_each_session(self) -> None:
        self.write_catalog([{"parent_asin": "A", "title": "Blue shoe"}])
        agent = Agent(self.catalog_path)
        agent.reset("ready", {})

        with self.assertRaisesRegex(
            RuntimeError,
            "reset must be called before respond",
        ):
            agent.respond("not-ready", "blue shoe", 1, 10)

        response = agent.respond("ready", "blue shoe", 1, 10)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])

    def test_empty_input_and_empty_candidate_pool_return_deterministic_responses(self) -> None:
        self.write_catalog([{"parent_asin": "A", "title": "Blue shoe"}])
        agent = Agent(self.catalog_path)
        agent.reset("empty", {})
        agent.reset("miss", {})

        empty_response = agent.respond("empty", "", 1, 10)
        missed_response = agent.respond("miss", "unfindableword", 1, 10)

        self.assertEqual(empty_response, missed_response)
        self.assert_valid_response(empty_response, {"A"})
        self.assertEqual(empty_response["recommendations"], [])

    def test_agent_completes_representative_offline_evaluation(self) -> None:
        catalog_rows = [
            {
                "parent_asin": "A",
                "title": "Blue cotton running shoe",
                "features": ["cotton"],
                "details": {"department": "womens"},
                "description": ["running shoe"],
                "categories": ["Clothing", "Shoes"],
                "store": "Example",
                "price": 49.0,
            },
            {
                "parent_asin": "B",
                "title": "Black leather winter boot",
                "features": ["leather"],
                "details": {"department": "womens"},
                "description": ["winter boot"],
                "categories": ["Clothing", "Boots"],
                "store": "Example",
                "price": 89.0,
            },
        ]
        self.write_catalog(catalog_rows)
        samples = [{
            "sample_id": "offline_0001",
            "scenario_type": "buying",
            "user_profile": {"summary": "Prefers practical shoes"},
            "ground_truth": {"parent_asin": "A"},
        }]
        catalog_ids, categories, products = catalog_index(self.catalog_path)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network access is disabled"),
            ),
        ):
            result = evaluate(
                Agent(self.catalog_path),
                samples,
                catalog_ids,
                categories,
                products,
            )

        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(result["sessions"][0]["first_hit_turn"], 1)
        self.assertEqual(result["reported_token_usage"], {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })

    def assert_valid_response(self, response: dict, catalog_ids: set[str]) -> None:
        allowed_attributes = {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        }
        self.assertEqual(
            set(response),
            {"message", "ask_attribute", "recommendations", "usage"},
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], allowed_attributes)
        recommendations = response["recommendations"]
        self.assertLessEqual(len(recommendations), 10)
        identifiers = [item["parent_asin"] for item in recommendations]
        self.assertEqual(identifiers, list(dict.fromkeys(identifiers)))
        self.assertTrue(set(identifiers).issubset(catalog_ids))
        self.assertEqual(set(response["usage"]), {"prompt_tokens", "completion_tokens"})
        for token_count in response["usage"].values():
            self.assertIsInstance(token_count, int)
            self.assertGreaterEqual(token_count, 0)


if __name__ == "__main__":
    unittest.main()
