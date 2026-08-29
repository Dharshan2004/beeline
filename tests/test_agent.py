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

    def test_hard_constraint_has_complete_provenance_and_persists(self) -> None:
        self.write_catalog([
            {
                "parent_asin": "A",
                "title": "Everyday walking shoe",
                "features": ["cotton"],
            },
            {
                "parent_asin": "B",
                "title": "Everyday walking shoe",
                "features": ["leather"],
            },
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        first_response = agent.respond(
            "session",
            "I need Cotton walking shoes.",
            1,
            10,
        )
        second_response = agent.respond(
            "session",
            "Show me the walking shoes again.",
            2,
            10,
        )

        expected_recommendations = [{"parent_asin": "A"}]
        self.assertEqual(first_response["recommendations"], expected_recommendations)
        self.assertEqual(second_response["recommendations"], expected_recommendations)
        self.assertEqual(agent.get_constraint_state("session"), [{
            "attribute": "material",
            "raw_phrase": "Cotton",
            "normalized_value": "cotton",
            "classification": "hard",
            "source_turn": 1,
            "confidence": 0.95,
            "status": "active",
        }])

    def test_soft_preference_changes_order_without_excluding_candidates(self) -> None:
        self.write_catalog([
            {
                "parent_asin": "A_BLACK",
                "title": "Everyday walking shoe",
                "features": ["black"],
            },
            {
                "parent_asin": "B_BLUE",
                "title": "Everyday walking shoe",
                "features": ["blue"],
            },
        ])
        agent = Agent(self.catalog_path)
        agent.reset("with-preference", {})
        agent.reset("without-preference", {})

        initial_preference = agent.respond(
            "with-preference",
            "I prefer blue.",
            1,
            10,
        )
        preferred = agent.respond(
            "with-preference",
            "Show me an everyday walking shoe.",
            2,
            10,
        )
        baseline = agent.respond(
            "without-preference",
            "Show me an everyday walking shoe.",
            1,
            10,
        )

        self.assertEqual(
            [item["parent_asin"] for item in baseline["recommendations"]],
            ["A_BLACK", "B_BLUE"],
        )
        self.assertEqual(
            [item["parent_asin"] for item in initial_preference["recommendations"]],
            ["B_BLUE", "A_BLACK"],
        )
        self.assertEqual(
            [item["parent_asin"] for item in preferred["recommendations"]],
            ["B_BLUE", "A_BLACK"],
        )

    def test_unsupported_value_does_not_corrupt_constraint_state(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Cotton walking shoe"},
            {"parent_asin": "B", "title": "Leather walking shoe"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})
        agent.respond("session", "I need cotton.", 1, 10)
        original_state = agent.get_constraint_state("session")

        response = agent.respond("session", "I need unobtainium.", 2, 10)

        self.assertEqual(agent.get_constraint_state("session"), original_state)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])

    def test_repeated_category_overrides_preserve_history(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Everyday shoes"},
            {"parent_asin": "B", "title": "Everyday boots"},
            {"parent_asin": "C", "title": "Everyday slippers"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        shoes = agent.respond("session", "I need shoes.", 1, 10)
        boots = agent.respond(
            "session",
            "Actually, I need boots instead.",
            2,
            10,
        )
        slippers = agent.respond("session", "I need slippers now.", 3, 10)

        self.assertEqual(shoes["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(boots["recommendations"], [{"parent_asin": "B"}])
        self.assertEqual(slippers["recommendations"], [{"parent_asin": "C"}])
        state = agent.get_constraint_state("session")
        self.assertEqual(
            [constraint["normalized_value"] for constraint in state],
            ["shoe", "boot", "slipper"],
        )
        self.assertEqual(
            [constraint["raw_phrase"] for constraint in state],
            ["shoes", "boots", "slippers"],
        )
        self.assertEqual(
            [constraint["source_turn"] for constraint in state],
            [1, 2, 3],
        )
        self.assertEqual(
            [constraint["status"] for constraint in state],
            ["superseded", "superseded", "active"],
        )

    def test_override_supersedes_an_earlier_preference_on_the_same_turn(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Blue walking shoe"},
            {"parent_asin": "B", "title": "Red cotton walking shoe"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})
        agent.respond("session", "I prefer blue.", 1, 10)

        response = agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            2,
            10,
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "B"}])
        state = agent.get_constraint_state("session")
        self.assertEqual(
            [(item["normalized_value"], item["status"]) for item in state],
            [("blue", "superseded"), ("cotton", "active")],
        )

    def test_boundary_response_is_respected_until_explicit_reintroduction(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Blue cotton shoes"},
            {"parent_asin": "B", "title": "Black leather shoes"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        first = agent.respond("session", "I need shoes.", 1, 10)
        boundary = agent.respond(
            "session",
            "I don't have a preference for material; please use your judgment.",
            2,
            10,
        )
        later = agent.respond("session", "Show me more shoes.", 3, 10)

        self.assertEqual(first["ask_attribute"], "material")
        self.assertEqual(boundary["ask_attribute"], "color")
        self.assertNotEqual(later["ask_attribute"], "material")
        self.assertEqual(agent.get_dismissed_attributes("session"), [{
            "attribute": "material",
            "raw_phrase": (
                "I don't have a preference for material; please use your judgment."
            ),
            "source_turn": 2,
            "status": "dismissed",
        }])

        reintroduced = agent.respond(
            "session",
            "Actually, I need cotton.",
            4,
            10,
        )

        self.assertEqual(reintroduced["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(agent.get_dismissed_attributes("session"), [])
        self.assertIn(
            ("cotton", "active"),
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
        )

    def test_boundary_dismisses_but_retains_an_active_constraint(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Cotton walking shoe"},
            {"parent_asin": "B", "title": "Leather walking shoe"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})
        agent.respond("session", "I need cotton.", 1, 10)

        response = agent.respond(
            "session",
            "I have no preference for material now.",
            2,
            10,
        )

        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A", "B"],
        )
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [("cotton", "dismissed")],
        )

    def test_intent_override_evaluator_converts_only_after_new_intent(self) -> None:
        self.write_catalog([
            {
                "parent_asin": "OLD",
                "title": "Blue walking shoe",
                "categories": ["Clothing", "Shoes"],
            },
            {
                "parent_asin": "TARGET",
                "title": "Red cotton walking shoe",
                "features": ["cotton"],
                "categories": ["Clothing", "Shoes"],
            },
        ])
        samples = [{
            "sample_id": "override_0001",
            "scenario_type": "intent_override",
            "user_profile": {"summary": "x"},
            "ground_truth": {"parent_asin": "TARGET"},
            "intent_card": {
                "target_category": "walking shoe",
                "hard_constraints": ["cotton"],
                "soft_preferences": ["blue"],
            },
            "behavior": {
                "scenario_type": "intent_override",
                "override": {
                    "turn": 2,
                    "old_value": "blue",
                    "new_value": "cotton",
                    "message": (
                        "Actually, ignore my earlier preference. "
                        "What I need is: cotton."
                    ),
                },
            },
        }]
        catalog_ids, categories, products = catalog_index(self.catalog_path)
        agent = Agent(self.catalog_path)

        result = evaluate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
        )

        self.assertEqual(result["hit_rate_at_10"], 1.0)
        self.assertEqual(result["sessions"][0]["first_hit_turn"], 2)
        session_id = next(iter(agent._sessions))
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state(session_id)
            ],
            [("blue", "superseded"), ("cotton", "active")],
        )

    def test_boundary_evaluator_records_dismissed_attributes(self) -> None:
        target = {
            "parent_asin": "ZZ_TARGET",
            "title": "Generic shoe",
            "features": ["cotton", "blue"],
            "categories": ["Clothing", "Shoes"],
        }
        decoys = [
            {
                "parent_asin": f"DECOY_{index:02d}",
                "title": "Generic shoe",
                "features": ["leather", "black"],
                "categories": ["Clothing", "Shoes"],
            }
            for index in range(12)
        ]
        self.write_catalog([target, *decoys])
        samples = [{
            "sample_id": "boundary_0001",
            "scenario_type": "boundary",
            "user_profile": {"summary": "x"},
            "ground_truth": {"parent_asin": "ZZ_TARGET"},
        }]
        catalog_ids, categories, products = catalog_index(self.catalog_path)
        agent = Agent(self.catalog_path)

        evaluate(agent, samples, catalog_ids, categories, products)

        session_id = next(iter(agent._sessions))
        dismissed = {
            item["attribute"]
            for item in agent.get_dismissed_attributes(session_id)
        }
        self.assertEqual(dismissed, {"material"})

    def test_constraint_improves_small_catalog_evaluator_ranking(self) -> None:
        class ConstraintBlindAgent(Agent):
            def _extract_constraint(self, user_message: str, turn: int):
                return None

            def respond(
                self,
                session_id: str,
                user_message: str,
                turn: int,
                top_k: int,
            ) -> dict:
                return super().respond(
                    session_id,
                    user_message.replace("cotton", ""),
                    turn,
                    top_k,
                )

        target = {
            "parent_asin": "TARGET",
            "title": "Plain footwear",
            "features": ["cotton"],
            "categories": ["Clothing", "Shoes"],
        }
        decoys = [
            {
                "parent_asin": f"DECOY_{index:02d}",
                "title": "Shoes key requirement",
                "features": ["leather"],
                "categories": ["Clothing", "Shoes"],
            }
            for index in range(12)
        ]
        self.write_catalog([target, *decoys])
        samples = [{
            "sample_id": "constraint_0001",
            "scenario_type": "buying",
            "user_profile": {"summary": "Prefers practical shoes"},
            "ground_truth": {"parent_asin": "TARGET"},
        }]
        catalog_ids, categories, products = catalog_index(self.catalog_path)

        constrained_result = evaluate(
            Agent(self.catalog_path),
            samples,
            catalog_ids,
            categories,
            products,
        )
        baseline_result = evaluate(
            ConstraintBlindAgent(self.catalog_path),
            samples,
            catalog_ids,
            categories,
            products,
        )

        self.assertEqual(constrained_result["hit_rate_at_10"], 1.0)
        self.assertEqual(baseline_result["hit_rate_at_10"], 0.0)
        self.assertGreater(constrained_result["mrr"], baseline_result["mrr"])

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
