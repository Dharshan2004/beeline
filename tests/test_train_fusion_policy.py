from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.train_fusion_policy import (
    evaluate_weighted_policy,
    load_fused30_baseline,
    rank_weighted_pool,
    select_best,
    simplex_weights,
    train_fusion_policy,
)


def _record(
    sample_id: str,
    scenario: str,
    target: str,
    structured: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    scores: dict[str, float],
) -> dict:
    def candidates(values: list[tuple[str, float]]) -> list[dict]:
        return [
            {"parent_asin": identifier, "raw_score": score, "normalized_score": score}
            for identifier, score in values
        ]

    union = sorted(scores)
    return {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "turn": 1,
        "target": target,
        "hit_eligible": True,
        "route_candidates": {
            "structured": candidates(structured),
            "bm25": candidates(bm25),
            "dense": candidates(dense),
        },
        "reranker_scores": [
            {"parent_asin": identifier, "score": scores[identifier]}
            for identifier in union
        ],
    }


class TrainFusionPolicyTest(unittest.TestCase):
    def records(self) -> list[dict]:
        return [
            _record(
                "one",
                "buying",
                "TARGET-A",
                [("TARGET-A", 1.0), ("OTHER", 0.0)],
                [("OTHER", 1.0), ("TARGET-A", 0.0)],
                [],
                {"TARGET-A": 3.0, "OTHER": 0.0},
            ),
            _record(
                "two",
                "browsing",
                "TARGET-B",
                [],
                [("OTHER", 1.0)],
                [("TARGET-B", 1.0), ("OTHER", 0.0)],
                {"TARGET-B": 2.0, "OTHER": 0.0},
            ),
        ]

    def test_simplex_grid_is_exact_complete_and_deterministic(self) -> None:
        weights = simplex_weights(10)

        self.assertEqual(len(weights), 66)
        self.assertEqual(weights, simplex_weights(10))
        self.assertIn({"structured": 0.0, "bm25": 0.0, "dense": 1.0}, weights)
        self.assertIn({"structured": 0.2, "bm25": 0.5, "dense": 0.3}, weights)
        for candidate in weights:
            self.assertTrue(all(value >= 0.0 for value in candidate.values()))
            self.assertAlmostEqual(sum(candidate.values()), 1.0)

    def test_weighted_ranking_uses_cached_normalized_scores(self) -> None:
        record = self.records()[0]

        structured = rank_weighted_pool(
            record,
            {"structured": 1.0, "bm25": 0.0, "dense": 0.0},
            depth=1,
        )
        bm25 = rank_weighted_pool(
            record,
            {"structured": 0.0, "bm25": 1.0, "dense": 0.0},
            depth=1,
        )

        self.assertEqual(structured, ["TARGET-A"])
        self.assertEqual(bm25, ["OTHER"])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            rank_weighted_pool(
                record,
                {"structured": 1.0, "bm25": 1.0, "dense": 0.0},
                depth=1,
            )

    def test_evaluation_reports_pool_rerank_and_scenario_metrics(self) -> None:
        result = evaluate_weighted_policy(
            self.records(),
            {"structured": 0.5, "bm25": 0.0, "dense": 0.5},
            depth=1,
        )

        self.assertEqual(result["pool_recall"]["session_pool_recall"], 1.0)
        self.assertEqual(result["metrics"]["hit_rate_at_10"], 1.0)
        self.assertEqual(set(result["scenario_metrics"]), {"browsing", "buying"})
        self.assertEqual(result["recall_to_hit_conversion"], 1.0)

    def test_selection_prioritizes_pool_recall_before_final_metrics(self) -> None:
        higher_score = {
            "weights": {"structured": 1.0, "bm25": 0.0, "dense": 0.0},
            "pool_recall": {"session_pool_recall": 0.5, "turn_pool_recall": 0.5},
            "metrics": {
                "recommended_technical_score": 0.9,
                "hit_rate_at_10": 0.9,
                "mrr": 0.9,
            },
        }
        higher_recall = {
            "weights": {"structured": 0.0, "bm25": 0.0, "dense": 1.0},
            "pool_recall": {"session_pool_recall": 1.0, "turn_pool_recall": 1.0},
            "metrics": {
                "recommended_technical_score": 0.1,
                "hit_rate_at_10": 0.1,
                "mrr": 0.1,
            },
        }

        self.assertIs(select_best([higher_score, higher_recall]), higher_recall)

    def test_fused30_control_loads_the_complete_slice7_trajectory(self) -> None:
        baseline = load_fused30_baseline("docs/reranker_benchmark.json")

        self.assertEqual(baseline["provenance"]["turn_count"], 1009)
        self.assertEqual(baseline["pool_recall"]["session_pool_recall"], 0.725)
        self.assertEqual(
            baseline["metrics"]["recommended_technical_score"], 0.467096
        )
        self.assertFalse(baseline["reranked"])

    def test_fused30_control_rejects_inconsistent_aggregate_metrics(self) -> None:
        report = json.loads(
            Path("docs/reranker_benchmark.json").read_text(encoding="utf-8")
        )
        row = next(
            item for item in report["rows"]
            if item["identity"] == "none (fused-30 baseline)"
        )
        row["mrr"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control row"):
                load_fused30_baseline(path)

    def test_training_is_model_free_and_records_required_baselines(self) -> None:
        manifest = {
            "artifact_sha256": "artifact",
            "configuration_sha256": "configuration",
            "identities_sha256": "identities",
        }
        fused30 = {
            "policy": "fixed-hybrid-v1",
            "depth": 30,
            "reranked": False,
            "pool_recall": {
                "session_pool_recall": 0.5,
                "turn_pool_recall": 0.5,
            },
            "metrics": {"recommended_technical_score": 0.4},
        }
        with patch(
            "retrieval.reranker.CrossEncoderReranker.__init__",
            side_effect=AssertionError("model must not load"),
        ):
            first = train_fusion_policy(
                self.records(), manifest, fused30,
                coarse_units=2, local_units=10, depth=1
            )
            second = train_fusion_policy(
                self.records(), manifest, fused30,
                coarse_units=2, local_units=10, depth=1
            )

        self.assertEqual(first, second)
        self.assertEqual(first["provenance"]["artifact_sha256"], "artifact")
        self.assertEqual(
            set(first["baselines"]),
            {
                "fused_30",
                "current_fixed",
                "rrf",
                "single_structured",
                "single_bm25",
                "single_dense",
            },
        )
        self.assertFalse(first["baselines"]["fused_30"]["reranked"])
        self.assertIn("depth_30", first["winner_depth_comparison"])
        self.assertIn("full_union", first["winner_depth_comparison"])


if __name__ == "__main__":
    unittest.main()
