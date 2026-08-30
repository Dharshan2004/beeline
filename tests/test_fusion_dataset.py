from __future__ import annotations

import json
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.build_fusion_dataset import (
    ARTIFACT_VERSION,
    FusionDatasetError,
    FROZEN_CATALOG_SHA256,
    FROZEN_DENSE_STORE_SHA256,
    FROZEN_EMBEDDING_FINGERPRINT,
    FROZEN_RERANKER_DIRECTORY_SHA256,
    load_artifact,
    replay_records,
    validate_training_records,
    validate_current_identities,
    write_artifact,
)
from starter.planning import PLANNING_PROMPT_SHA256, PLANNING_PROMPT_VERSION
from starter.replacement_evidence import (
    REPLACEMENT_EVIDENCE_SHA256,
    REPLACEMENT_EVIDENCE_VERSION,
)


class FusionTrainingDatasetTest(unittest.TestCase):
    def test_artifact_uses_planning_contract_v2(self) -> None:
        self.assertEqual(ARTIFACT_VERSION, "fusion-training-v2")

    def records(self) -> list[dict]:
        return [
            {
                "sample_id": "dev-a",
                "scenario_type": "buying",
                "turn": 1,
                "target": "TARGET-A",
                "hit_eligible": True,
                "query": "query a",
                "planning": {
                    "source": "fallback",
                    "state_revision": 1,
                    "retrieval_tools": ["structured", "bm25", "dense"],
                },
                "route_candidates": {
                    "structured": [
                        {"parent_asin": "TARGET-A", "raw_score": 2.0, "normalized_score": 1.0},
                        {"parent_asin": "OTHER", "raw_score": 0.0, "normalized_score": 0.0},
                    ],
                    "bm25": [],
                    "dense": [],
                },
                "candidate_pool": ["TARGET-A", "OTHER"],
                "frozen_candidate_pool": ["TARGET-A", "OTHER"],
                "reranker_scores": [
                    {"parent_asin": "TARGET-A", "score": 2.5, "is_target": True},
                    {"parent_asin": "OTHER", "score": 0.1, "is_target": False},
                ],
                "response_pool": ["TARGET-A", "OTHER"],
            },
            {
                "sample_id": "dev-b",
                "scenario_type": "browsing",
                "turn": 1,
                "target": "TARGET-B",
                "hit_eligible": True,
                "query": "query b",
                "planning": {
                    "source": "fallback",
                    "state_revision": 1,
                    "retrieval_tools": ["structured", "bm25", "dense"],
                },
                "route_candidates": {
                    "structured": [],
                    "bm25": [{"parent_asin": "OTHER", "raw_score": 1.0, "normalized_score": 1.0}],
                    "dense": [],
                },
                "candidate_pool": ["OTHER"],
                "frozen_candidate_pool": ["OTHER"],
                "reranker_scores": [
                    {"parent_asin": "OTHER", "score": 0.3, "is_target": False},
                ],
                "response_pool": ["OTHER"],
            },
        ]

    def test_validation_enforces_scenario_counts_and_excludes_holdout(self) -> None:
        records = self.records()

        validate_training_records(
            records,
            expected_scenarios={"buying": 1, "browsing": 1},
            expected_session_count=2,
            holdout_ids={"locked"},
        )

        records[1]["sample_id"] = "locked"
        with self.assertRaisesRegex(FusionDatasetError, "locked holdout"):
            validate_training_records(
                records,
                expected_scenarios={"buying": 1, "browsing": 1},
                expected_session_count=2,
                holdout_ids={"locked"},
            )

    def test_incomplete_or_stale_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.jsonl"
            manifest = write_artifact(
                self.records(),
                path,
                configuration={"policy": "fixed-v1"},
                expected_scenarios={"buying": 1, "browsing": 1},
                expected_session_count=2,
                holdout_ids=set(),
            )
            path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FusionDatasetError, "checksum"):
                load_artifact(
                    path,
                    expected_scenarios={"buying": 1, "browsing": 1},
                    expected_session_count=2,
                    holdout_ids=set(),
                    enforce_current_identities=False,
                )
            self.assertEqual(manifest["session_count"], 2)

    def test_incomplete_route_and_score_rows_are_rejected(self) -> None:
        records = self.records()
        del records[0]["route_candidates"]["structured"][0]["normalized_score"]

        with self.assertRaisesRegex(FusionDatasetError, "route candidate"):
            validate_training_records(
                records,
                expected_scenarios={"buying": 1, "browsing": 1},
                expected_session_count=2,
                holdout_ids=set(),
            )

    def test_truncated_route_union_is_rejected_even_when_scores_align(self) -> None:
        records = self.records()
        records[0]["candidate_pool"] = ["TARGET-A"]
        records[0]["frozen_candidate_pool"] = ["TARGET-A"]
        records[0]["reranker_scores"] = records[0]["reranker_scores"][:1]
        records[0]["response_pool"] = ["TARGET-A"]

        with self.assertRaisesRegex(FusionDatasetError, "complete route union"):
            validate_training_records(
                records,
                expected_scenarios={"buying": 1, "browsing": 1},
                expected_session_count=2,
                holdout_ids=set(),
            )

    def test_replay_reproduces_metrics_without_loading_models(self) -> None:
        records = self.records()
        records[0]["response_pool"] = ["OTHER", "TARGET-A"]
        with patch(
            "retrieval.reranker.CrossEncoderReranker.__init__",
            side_effect=AssertionError("model must not load"),
        ):
            metrics = replay_records(records)

        self.assertEqual(metrics["session_count"], 2)
        self.assertEqual(metrics["hit_rate_at_10"], 0.5)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["recommended_technical_score"], 0.5)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FusionDatasetError, "reconstruct response"):
                write_artifact(
                    records,
                    Path(directory) / "invalid.jsonl",
                    configuration={"policy": "fixed-v1"},
                    expected_scenarios={"buying": 1, "browsing": 1},
                    expected_session_count=2,
                    holdout_ids=set(),
                )

    def test_current_identity_validation_rejects_each_stale_asset(self) -> None:
        revision = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
        configuration = {
            "catalog": {"sha256": FROZEN_CATALOG_SHA256},
            "planning": {
                "prompt_version": PLANNING_PROMPT_VERSION,
                "prompt_sha256": PLANNING_PROMPT_SHA256,
                "replacement_evidence_version": REPLACEMENT_EVIDENCE_VERSION,
                "replacement_evidence_sha256": REPLACEMENT_EVIDENCE_SHA256,
                "provider": None,
                "connected_model_version": None,
            },
            "fusion_and_retrieval": {
                "policy_version": "fixed-hybrid-v1",
                "fused_candidate_depth": 50,
                "reranker_revision": revision,
                "weights": {"structured": 0.15, "bm25": 0.55, "dense": 0.3},
            },
        }
        identities = {
            "catalog_sha256": FROZEN_CATALOG_SHA256,
            "reranker": {
                "identity": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "revision": revision,
                "directory_sha256": FROZEN_RERANKER_DIRECTORY_SHA256,
            },
            "dense_index_and_model": {
                "status": "available",
                "manifest": {
                    "catalog": {"file_sha256": FROZEN_CATALOG_SHA256},
                    "embedding_model": {"fingerprint_sha256": FROZEN_EMBEDDING_FINGERPRINT},
                    "vector_store": {"directory_sha256": FROZEN_DENSE_STORE_SHA256},
                },
            },
        }
        validate_current_identities(configuration, identities)

        mutations = [
            ("configuration", lambda value: value["catalog"].update(sha256="stale")),
            (
                "configuration",
                lambda value: value["fusion_and_retrieval"].update(
                    fused_candidate_depth=49
                ),
            ),
            (
                "configuration",
                lambda value: value["fusion_and_retrieval"]["weights"].update(
                    dense=0.29
                ),
            ),
            (
                "configuration",
                lambda value: value["planning"].update(
                    replacement_evidence_sha256="stale"
                ),
            ),
            (
                "identities",
                lambda value: value["dense_index_and_model"]["manifest"]["catalog"].update(
                    file_sha256="stale"
                ),
            ),
            (
                "identities",
                lambda value: value["dense_index_and_model"]["manifest"]["embedding_model"].update(
                    fingerprint_sha256="stale"
                ),
            ),
            (
                "identities",
                lambda value: value["dense_index_and_model"]["manifest"]["vector_store"].update(
                    directory_sha256="stale"
                ),
            ),
            ("identities", lambda value: value["reranker"].update(revision="stale")),
            (
                "identities",
                lambda value: value["reranker"].update(directory_sha256="stale"),
            ),
        ]
        for target_name, mutate in mutations:
            stale_configuration = deepcopy(configuration)
            stale_identities = deepcopy(identities)
            target = (
                stale_configuration
                if target_name == "configuration"
                else stale_identities
            )
            mutate(target)
            with self.assertRaisesRegex(FusionDatasetError, "stale"):
                validate_current_identities(stale_configuration, stale_identities)


if __name__ == "__main__":
    unittest.main()
