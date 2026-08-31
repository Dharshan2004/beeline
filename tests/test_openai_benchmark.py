from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from starter.constraint_state import (
    AddConstraint,
    DismissAttribute,
    ReplaceConstraint,
    ReplaceProductIntent,
    TurnPlan,
)
from starter.planning import ProviderResponse
from tools.benchmark_openai_planning import (
    BenchmarkConfigurationError,
    ModelBenchmarkConfig,
    _mutation_signature,
    _replacement_signature,
    assert_development_only_samples,
    build_planning_fixtures,
    compare_providers,
    load_benchmark_config,
    validate_benchmark_report,
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


class MeteredFailingProvider:
    def __init__(self) -> None:
        self.spent_usd = 0.0

    def plan(self, request):
        self.spent_usd += 0.25
        raise TimeoutError("request timed out after submission")

    def metrics(self):
        return {"budget": {"spent_usd": self.spent_usd}}


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
        self.assertEqual(metrics["aggregate"]["constraint_decision_accuracy"], 1.0)
        self.assertEqual(metrics["route_selection_rates"], {
            "bm25": 1.0,
            "dense": 1.0,
            "structured": 1.0,
        })
        self.assertEqual(metrics["token_usage"], {
            "prompt_tokens": len(fixtures) * 100,
            "completion_tokens": len(fixtures) * 20,
            "total_tokens": len(fixtures) * 120,
        })
        self.assertIn("p50_seconds", metrics["latency"])
        self.assertIn("p95_seconds", metrics["latency"])
        self.assertIn("browsing", metrics["scenario_metrics"])
        self.assertEqual(report["selection_status"], "provisional_no_default_activation")
        self.assertEqual(
            report["planning_contract"]["transport_schema"],
            {
                "version": "openai-anyof-v1",
                "sha256": "fc76794ea220f56a227a92fe63e7b26b1ebeed329b5de56bcfd1dc27773bf46f",
            },
        )
        validate_benchmark_report(report)

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

    def test_failed_call_budget_charges_are_attributed_to_the_model(self) -> None:
        fixtures = build_planning_fixtures(self.samples, self.catalog)

        report = compare_providers(
            fixtures=fixtures,
            catalog_path=self.catalog,
            providers={
                "quality_reference": RecordingProvider(),
                "lower_cost": MeteredFailingProvider(),
            },
            models={
                "quality_reference": self.model("quality_reference", "reference", 4),
                "lower_cost": self.model("lower_cost", "cheap", 0.2),
            },
            configuration_sha256="d" * 64,
        )

        cheap = report["models"]["lower_cost"]
        self.assertEqual(cheap["estimated_cost_usd"], len(fixtures) * 0.25)
        self.assertTrue(
            cheap["cost_accounting"][
                "includes_pessimistic_failed_call_reservations"
            ]
        )

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

    def test_report_schema_validation_rejects_missing_aggregate_metrics(self) -> None:
        fixtures = build_planning_fixtures(self.samples, self.catalog)
        report = compare_providers(
            fixtures=fixtures,
            catalog_path=self.catalog,
            providers={
                "quality_reference": RecordingProvider(),
                "lower_cost": RecordingProvider(),
            },
            models={
                "quality_reference": self.model("quality_reference", "reference", 4),
                "lower_cost": self.model("lower_cost", "cheap", 0.2),
            },
            configuration_sha256="c" * 64,
        )
        del report["models"]["lower_cost"]["aggregate"]["state_accuracy"]

        with self.assertRaisesRegex(BenchmarkConfigurationError, "state_accuracy"):
            validate_benchmark_report(report)

        schema = Path(__file__).parents[1] / "docs" / "openai_model_benchmark_report.schema.json"
        self.assertTrue(schema.is_file())

    def test_report_schema_validation_rejects_wrong_metric_types(self) -> None:
        fixtures = build_planning_fixtures(self.samples, self.catalog)
        report = compare_providers(
            fixtures=fixtures,
            catalog_path=self.catalog,
            providers={
                "quality_reference": RecordingProvider(),
                "lower_cost": RecordingProvider(),
            },
            models={
                "quality_reference": self.model("quality_reference", "reference", 4),
                "lower_cost": self.model("lower_cost", "cheap", 0.2),
            },
            configuration_sha256="e" * 64,
        )
        report["models"]["lower_cost"]["scenario_metrics"] = "invalid"

        with self.assertRaisesRegex(BenchmarkConfigurationError, "scenario_metrics"):
            validate_benchmark_report(report)

    def test_mutation_metrics_distinguish_kind_and_replacement_target(self) -> None:
        common = {
            "attribute": "color",
            "values": ("black",),
            "match_rule": "any",
            "classification": "hard",
            "scope": "product_intent",
            "raw_phrase": "black",
            "confidence": 1.0,
        }
        expected = TurnPlan(0, 1, (
            AddConstraint(**common),
            DismissAttribute(attribute="material", raw_phrase="any material"),
            ReplaceConstraint(constraint_id="c-1", **common),
            ReplaceProductIntent(product_intent_id="intent-2", raw_phrase="boots"),
        ))
        wrong_target = TurnPlan(0, 1, (
            AddConstraint(**common),
            DismissAttribute(attribute="material", raw_phrase="any material"),
            ReplaceConstraint(constraint_id="c-2", **common),
            ReplaceProductIntent(product_intent_id="intent-3", raw_phrase="boots"),
        ))

        self.assertNotEqual(
            _mutation_signature(expected),
            _mutation_signature(wrong_target),
        )
        self.assertNotEqual(
            _replacement_signature(expected),
            _replacement_signature(wrong_target),
        )


if __name__ == "__main__":
    unittest.main()
