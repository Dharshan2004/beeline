from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

from evaluator.local_evaluator import (
    build_evaluation_agent,
    catalog_index,
    evaluate,
    load_evaluation_samples,
    load_openai_evaluation_settings,
    metric_summary,
    normalize_recommendations,
    validate_connected_evaluation_scope,
)


class EchoTargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        asin = "A"
        if "B" in user_message:
            asin = "B"
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": asin}]}


class EvaluatorTest(unittest.TestCase):
    def test_openai_evaluation_settings_use_the_frozen_model_configuration(self) -> None:
        settings = load_openai_evaluation_settings(
            "config/openai_phase_a_benchmark.json",
            "lower_cost",
        )

        self.assertEqual(settings.model, "gpt-5.6-luna")
        self.assertEqual(settings.pricing.input_per_million_usd, 0.2)
        self.assertEqual(settings.pricing.output_per_million_usd, 1.2)
        self.assertEqual(settings.budget_limit_usd, 10.0)

    def test_connected_evaluation_builds_agent_with_openai_planning(self) -> None:
        settings = load_openai_evaluation_settings(
            "config/openai_phase_a_benchmark.json",
            "lower_cost",
        )
        provider = object()

        with (
            patch(
                "evaluator.local_evaluator.OpenAIPlanningProvider",
                return_value=provider,
            ) as provider_type,
            patch("evaluator.local_evaluator.Agent") as agent_type,
        ):
            build_evaluation_agent("data/catalog.jsonl", settings)

        provider_type.assert_called_once()
        provider_arguments = provider_type.call_args.kwargs
        self.assertEqual(provider_arguments["model"], "gpt-5.6-luna")
        self.assertEqual(
            provider_arguments["pricing"].output_per_million_usd,
            1.2,
        )
        agent_type.assert_called_once_with(
            "data/catalog.jsonl",
            planning_provider=provider,
        )

    def test_connected_evaluation_subset_is_deterministic_and_development_only(self) -> None:
        first = load_evaluation_samples(
            "data/public_set.jsonl",
            development_only=True,
            session_count=20,
        )
        second = load_evaluation_samples(
            "data/public_set.jsonl",
            development_only=True,
            session_count=20,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(
            {sample["scenario_type"] for sample in first},
            {"buying", "browsing", "intent_override", "boundary"},
        )

    def test_connected_full_public_evaluation_requires_explicit_acknowledgement(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit evaluation scope"):
            validate_connected_evaluation_scope(
                connected=True,
                development_only=False,
                full_exposed_public_set=False,
            )

        self.assertEqual(
            validate_connected_evaluation_scope(
                connected=True,
                development_only=False,
                full_exposed_public_set=True,
            ),
            "full_exposed_public_set",
        )

        with self.assertRaisesRegex(ValueError, "all 200 sessions"):
            load_evaluation_samples(
                "data/public_set.jsonl",
                development_only=False,
                full_exposed_public_set=True,
                session_count=20,
            )

    def test_evaluate_reports_connected_planning_fallbacks(self) -> None:
        class FallbackAgent(EchoTargetAgent):
            def __init__(self) -> None:
                self.history: list[dict] = []

            def respond(
                self,
                session_id: str,
                user_message: str,
                turn: int,
                top_k: int,
            ) -> dict:
                self.history.append({
                    "source": "fallback",
                    "fallback_reason": "provider_timeout",
                })
                return super().respond(session_id, user_message, turn, top_k)

            def get_planning_history(self, session_id: str) -> list[dict]:
                return list(self.history)

        samples = [{
            "sample_id": "diagnostics",
            "scenario_type": "buying",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "intent_card": {"hard_constraints": [], "soft_preferences": []},
            "behavior": {"scenario_type": "buying"},
        }]
        result = evaluate(
            FallbackAgent(),
            samples,
            {"A"},
            {"A": []},
            {"A": {"parent_asin": "A"}},
        )

        self.assertEqual(result["execution_diagnostics"], {
            "response_exception_count": 0,
            "invalid_response_count": 0,
            "connected_plan_turn_count": 0,
            "fallback_plan_turn_count": 1,
            "fallback_reason_counts": {"provider_timeout": 1},
        })

    def test_normalization_preserves_first_valid_unique_order(self) -> None:
        payload = [
            {"parent_asin": "A"}, {"parent_asin": "bad"}, {"parent_asin": "A"},
            "B", {"parent_asin": "C"},
        ]
        self.assertEqual(normalize_recommendations(payload, {"A", "B", "C"}), ["A", "B", "C"])

    def test_metric_summary_assigns_turn_11_to_miss(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": .5, "first_hit_turn": 2},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        self.assertEqual(metric_summary(sessions), {
            "sample_count": 2,
            "hit_rate_at_10": .5,
            "mrr": .25,
            "mttc": 6.5,
        })

    def test_evaluate_derives_hidden_fields_when_public_set_omits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_rows = [
                {
                    "parent_asin": "A",
                    "title": "Blue running shoe",
                    "features": ["cotton"],
                    "details": {"department": "womens"},
                    "description": ["walking shoe"],
                    "categories": ["Clothing", "Shoes"],
                    "store": "Example",
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black winter boot",
                    "features": ["leather"],
                    "details": {"department": "womens"},
                    "description": ["winter boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "Example",
                    "average_rating": 4.4,
                    "rating_number": 12,
                    "price": 89.0,
                },
            ]
            catalog_path.write_text("".join(json.dumps(row) + "\n" for row in catalog_rows), encoding="utf-8")
            catalog_ids, categories, products = catalog_index(catalog_path)
            samples = [{
                "sample_id": "public_v2_0001",
                "scenario_type": "buying",
                "user_profile": {"summary": "x"},
                "ground_truth": {"parent_asin": "A"},
            }]
            result = evaluate(EchoTargetAgent(), samples, catalog_ids, categories, products)
            self.assertEqual(result["hit_rate_at_10"], 1.0)

    def test_evaluate_records_end_to_end_agent_turn_latency(self) -> None:
        samples = [{
            "sample_id": "timed",
            "scenario_type": "browsing",
            "user_profile": {},
            "ground_truth": {"parent_asin": "A"},
            "intent_card": {"hard_constraints": [], "soft_preferences": []},
            "behavior": {"scenario_type": "browsing"},
        }]
        products = {"A": {"parent_asin": "A"}}

        with patch(
            "evaluator.local_evaluator.time.perf_counter",
            side_effect=[10.0, 10.125],
        ):
            result = evaluate(EchoTargetAgent(), samples, {"A"}, {"A": []}, products)

        self.assertEqual(result["turn_latency"], {
            "turn_count": 1,
            "p50_seconds": 0.125,
            "p95_seconds": 0.125,
            "mean_seconds": 0.125,
        })

if __name__ == "__main__":
    unittest.main()
