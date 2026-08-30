from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from tools.validate_fusion_policy import (
    plateau_neighbor_count,
    scenario_hit_guardrail,
    scenario_stratified_folds,
    select_stable_candidate,
    validate_live_evidence,
    validate_training_provenance,
    validated_local_weights,
)


class ValidateFusionPolicyTest(unittest.TestCase):
    def test_scenario_folds_are_deterministic_disjoint_and_representative(self) -> None:
        records = [
            {
                "sample_id": f"{scenario}-{index}",
                "scenario_type": scenario,
                "turn": 1,
            }
            for scenario in ("buying", "browsing")
            for index in range(4)
        ]

        folds = scenario_stratified_folds(records, fold_count=2, seed=7)

        self.assertEqual(folds, scenario_stratified_folds(records, fold_count=2, seed=7))
        ids = [
            {record["sample_id"] for record in fold}
            for fold in folds
        ]
        self.assertFalse(ids[0].intersection(ids[1]))
        self.assertEqual(ids[0].union(ids[1]), {record["sample_id"] for record in records})
        for fold in folds:
            self.assertEqual(
                {scenario: sum(row["scenario_type"] == scenario for row in fold)
                 for scenario in ("buying", "browsing")},
                {"buying": 2, "browsing": 2},
            )

    def test_scenario_guardrail_allows_at_most_five_point_drop(self) -> None:
        baseline = {
            "buying": {"metrics": {"hit_rate_at_10": 0.60}},
            "browsing": {"metrics": {"hit_rate_at_10": 0.70}},
        }
        passing = {
            "buying": {"metrics": {"hit_rate_at_10": 0.55}},
            "browsing": {"metrics": {"hit_rate_at_10": 0.75}},
        }
        failing = {
            **passing,
            "buying": {"metrics": {"hit_rate_at_10": 0.549}},
        }

        self.assertTrue(scenario_hit_guardrail(passing, baseline)["passed"])
        self.assertFalse(scenario_hit_guardrail(failing, baseline)["passed"])

    def test_plateau_support_uses_one_simplex_grid_move(self) -> None:
        candidates = [
            {"weights": {"structured": 0.0, "bm25": 0.68, "dense": 0.32}},
            {"weights": {"structured": 0.02, "bm25": 0.66, "dense": 0.32}},
            {"weights": {"structured": 0.12, "bm25": 0.56, "dense": 0.32}},
        ]

        self.assertEqual(plateau_neighbor_count(candidates[0], candidates, step=0.02), 1)
        self.assertEqual(plateau_neighbor_count(candidates[2], candidates, step=0.02), 0)

    def test_local_grid_validation_rejects_a_missing_candidate(self) -> None:
        training = json.loads(
            Path("docs/fusion_policy_training.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(validated_local_weights(training)), 91)
        training["search"]["local_candidates"].pop()
        with self.assertRaisesRegex(ValueError, "grid is incomplete"):
            validated_local_weights(training)

    def test_training_provenance_rejects_stale_configuration(self) -> None:
        training = json.loads(
            Path("docs/fusion_policy_training.json").read_text(encoding="utf-8")
        )
        artifact = json.loads(
            Path("docs/fusion_training_dataset.json").read_text(encoding="utf-8")
        )

        validate_training_provenance(training, artifact)
        training["provenance"]["configuration_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_training_provenance(training, artifact)

    def test_stable_selection_prefers_supported_plateau_then_fold_floor(self) -> None:
        fragile = {
            "weights": {"structured": 0.0, "bm25": 0.68, "dense": 0.32},
            "plateau_neighbor_count": 1,
            "fold_stability": {
                "minimum_session_pool_recall": 0.8,
                "session_pool_recall_pstdev": 0.01,
                "minimum_technical_score": 0.5,
            },
            "full": {"metrics": {"recommended_technical_score": 0.6, "hit_rate_at_10": 0.7, "mrr": 0.5}},
        }
        supported = {
            "weights": {"structured": 0.02, "bm25": 0.66, "dense": 0.32},
            "plateau_neighbor_count": 3,
            "fold_stability": {
                "minimum_session_pool_recall": 0.75,
                "session_pool_recall_pstdev": 0.02,
                "minimum_technical_score": 0.48,
            },
            "full": {"metrics": {"recommended_technical_score": 0.55, "hit_rate_at_10": 0.65, "mrr": 0.4}},
        }

        self.assertIs(select_stable_candidate([fragile, supported]), supported)

    def test_live_evidence_matches_freeze_and_rejects_stale_weights(self) -> None:
        runtime = json.loads(
            Path("docs/fusion_policy_live_evaluation.json").read_text(encoding="utf-8")
        )
        freeze = json.loads(
            Path("docs/fusion_policy_freeze.json").read_text(encoding="utf-8")
        )
        artifact = json.loads(
            Path("docs/fusion_training_dataset.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            Path("docs/fusion_policy_training.json").read_text(encoding="utf-8")
        )

        evidence = validate_live_evidence(
            runtime,
            freeze["selected"],
            artifact,
            training,
            freeze["frozen_configuration"],
        )
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["rerank_query_count"], 892)

        mutations = [
            lambda value: value["configuration"]["fusion_and_retrieval"]["weights"].update(bm25=0.63),
            lambda value: value["configuration"]["fusion_and_retrieval"].update(normalizer="stale"),
            lambda value: value["configuration"]["catalog"].update(sha256="stale"),
            lambda value: value["configuration"]["planning"].update(prompt_sha256="stale"),
            lambda value: value["configuration"]["dense_index_and_model"]["manifest"]["embedding_model"].update(fingerprint_sha256="stale"),
            lambda value: value["configuration"]["dense_index_and_model"]["manifest"]["vector_store"].update(directory_sha256="stale"),
            lambda value: value["configuration"]["fusion_and_retrieval"].update(reranker_directory_sha256="stale"),
            lambda value: value["runtime"]["dense_route"].update(status="disabled", disabled_reason="test"),
        ]
        for mutate in mutations:
            stale = deepcopy(runtime)
            mutate(stale)
            with self.assertRaisesRegex(ValueError, "live activated policy"):
                validate_live_evidence(
                    stale,
                    freeze["selected"],
                    artifact,
                    training,
                    freeze["frozen_configuration"],
                )

        stale_scenario = deepcopy(runtime)
        stale_scenario["metrics"]["scenario_metrics"]["buying"]["mrr"] = 0.0
        with self.assertRaisesRegex(ValueError, "live scenario metrics"):
            validate_live_evidence(
                stale_scenario,
                freeze["selected"],
                artifact,
                training,
                freeze["frozen_configuration"],
            )


if __name__ == "__main__":
    unittest.main()
