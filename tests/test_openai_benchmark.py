from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from starter.planning import ProviderResponse
from tools.benchmark_openai_planning import (
    BenchmarkConfigurationError,
    ModelBenchmarkConfig,
    assert_development_only_samples,
    build_planning_fixtures,
    compare_providers,
    load_benchmark_config,
)
from tools.dataset_split import FROZEN_HOLDOUT_SAMPLE_IDS


class RecordingProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("provider unavailable")
        return ProviderResponse(
            output={
                "expected_state_revision": request.state_snapshot["revision"],
                "source_turn": request.turn,
                "mutations": [],
                "retrieval_tools": list(request.allowed_tools),
                "clarification": None,
            },
            prompt_tokens=100,
            completion_tokens=20,
        )


class OpenAIPlanningBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog = Path(self.directory.name) / "catalog.jsonl"
        self.catalog.write_text(
            json.dumps({
                "parent_asin": "TARGET",
                "title": "Everyday product",
                "categories": ["Clothing", "Accessories"],
            }) + "\n",
            encoding="utf-8",
        )
        self.samples = [{
            "sample_id": "public_0001",
            "scenario_type": "browsing",
            "user_profile": {},
            "ground_truth": {"parent_asin": "TARGET"},
            "intent_card": {
                "target_category": "Everyday product",
                "hard_constraints": [],
                "soft_preferences": [],
            },
            "behavior": {"scenario_type": "browsing"},
        }]

    def model(self, role: str, model: str, input_price: float) -> ModelBenchmarkConfig:
        return ModelBenchmarkConfig(
            role=role,
            model=model,
            input_per_million_usd=input_price,
            output_per_million_usd=input_price * 5,
        )

    def test_locked_holdout_identifiers_are_rejected_before_fixture_building(self) -> None:
        locked_id = sorted(FROZEN_HOLDOUT_SAMPLE_IDS)[0]
        samples = [{**self.samples[0], "sample_id": locked_id}]

        with self.assertRaisesRegex(BenchmarkConfigurationError, "Locked Holdout"):
            assert_development_only_samples(samples)

    def test_both_models_receive_the_exact_same_canonical_requests(self) -> None:
        fixtures = build_planning_fixtures(self.samples, self.catalog)
        reference = RecordingProvider()
        lower_cost = RecordingProvider()

        report = compare_providers(
            fixtures=fixtures,
            catalog_path=self.catalog,
            providers={
                "quality_reference": reference,
                "lower_cost": lower_cost,
            },
            models={
                "quality_reference": self.model("quality_reference", "reference", 4),
                "lower_cost": self.model("lower_cost", "cheap", 0.2),
            },
            configuration_sha256="a" * 64,
        )

        self.assertGreater(len(fixtures), 0)
        self.assertEqual(reference.requests, lower_cost.requests)
        self.assertEqual(
            report["models"]["quality_reference"]["input_corpus_sha256"],
            report["models"]["lower_cost"]["input_corpus_sha256"],
        )
        self.assertEqual(report["input_parity"], {
            "identical": True,
            "fixture_count": len(fixtures),
            "corpus_sha256": report["models"]["quality_reference"]["input_corpus_sha256"],
        })
        metrics = report["models"]["quality_reference"]
        self.assertEqual(metrics["aggregate"]["failure_rate"], 0.0)
        self.assertEqual(metrics["aggregate"]["state_accuracy"], 1.0)
        self.assertEqual(metrics["aggregate"]["tool_decision_accuracy"], 1.0)
        self.assertEqual(metrics["token_usage"], {
            "prompt_tokens": len(fixtures) * 100,
            "completion_tokens": len(fixtures) * 20,
            "total_tokens": len(fixtures) * 120,
        })
        self.assertIn("p50_seconds", metrics["latency"])
        self.assertIn("p95_seconds", metrics["latency"])
        self.assertIn("browsing", metrics["scenario_metrics"])
        self.assertEqual(report["selection_status"], "provisional_no_default_activation")

        comparison = report["provisional_comparison"]
        self.assertEqual(comparison["lower_cost_within_tolerance"], True)
        self.assertEqual(comparison["default_activation_allowed"], False)

    def test_failures_are_counted_per_scenario_without_stopping_the_pair(self) -> None:
        fixtures = build_planning_fixtures(self.samples, self.catalog)

        report = compare_providers(
            fixtures=fixtures,
            catalog_path=self.catalog,
            providers={
                "quality_reference": RecordingProvider(),
                "lower_cost": RecordingProvider(fail=True),
            },
            models={
                "quality_reference": self.model("quality_reference", "reference", 4),
                "lower_cost": self.model("lower_cost", "cheap", 0.2),
            },
            configuration_sha256="b" * 64,
        )

        cheap = report["models"]["lower_cost"]
        self.assertEqual(cheap["aggregate"]["failure_rate"], 1.0)
        self.assertEqual(cheap["scenario_metrics"]["browsing"]["failure_rate"], 1.0)
        self.assertEqual(cheap["failure_causes"], {"RuntimeError": len(fixtures)})
        self.assertEqual(cheap["estimated_cost_usd"], 0.0)

    def test_checked_in_configuration_has_two_distinct_roles_and_budget_gates(self) -> None:
        config_path = Path(__file__).parents[1] / "config" / "openai_phase_a_benchmark.json"

        configuration = load_benchmark_config(config_path)

        self.assertEqual(set(configuration.models), {"quality_reference", "lower_cost"})
        self.assertNotEqual(
            configuration.models["quality_reference"].model,
            configuration.models["lower_cost"].model,
        )
        self.assertEqual(configuration.budget_limit_usd, 10.0)
        self.assertEqual(configuration.warning_usd, 40.0)
        self.assertEqual(configuration.review_boundary_usd, 50.0)
        self.assertEqual(configuration.absolute_stop_usd, 600.0)
        self.assertEqual(configuration.dataset_split, "public-split-v1-development-only")
        self.assertEqual(len(configuration.sha256), 64)


if __name__ == "__main__":
    unittest.main()
