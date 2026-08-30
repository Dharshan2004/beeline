from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.reranker import (
    DEFAULT_RERANKER_DIR,
    DEFAULT_RERANKER_IDENTITY,
    FROZEN_RERANK_DEPTH,
    RERANKER_CANDIDATES,
    CrossEncoderReranker,
    RerankerCandidate,
    RerankerUnavailable,
    order_by_scores,
)
from tools.benchmark_reranker import (
    _model_provenance,
    baseline_rows,
    build_cache,
    pool_recall,
    session_metrics,
    score_cache,
    subset_records,
    summarize,
)
from tools.dataset_split import (
    FROZEN_HOLDOUT_SAMPLE_IDS,
    HOLDOUT_SCENARIO_COUNTS,
    development_samples,
    holdout_sample_ids,
    holdout_samples,
    load_frozen_development_samples,
    stratified_subset,
)


# transformers lazily imports a model's modeling module on first use, and in this
# environment a second lazy import re-registers a torch mega-cache "precompile"
# artifact and raises. It is a dependency defect, not an Agent defect: it also
# breaks the untouched suite on main, and it only appears once several models
# have been loaded in one long-lived process. Tests that need a real model skip
# on it rather than reporting a failure the Agent did not cause.
MEGA_CACHE_DEFECT = "mega-cache artifact factory"


def skip_on_mega_cache_defect(case: unittest.TestCase, error: Exception) -> None:
    if MEGA_CACHE_DEFECT in str(error):
        case.skipTest(f"transformers/torch lazy-import defect: {error}")
    raise error


class OrderByScoresTest(unittest.TestCase):
    def test_higher_scores_come_first(self) -> None:
        self.assertEqual(
            order_by_scores(["A", "B", "C"], [0.1, 9.0, 3.0]),
            ["B", "C", "A"],
        )

    def test_ties_keep_the_fused_ordering(self) -> None:
        # A cross-encoder that cannot separate two products must leave the
        # fused decision standing rather than fall back to identifier order.
        self.assertEqual(
            order_by_scores(["Z", "A", "M"], [1.0, 1.0, 1.0]),
            ["Z", "A", "M"],
        )

    def test_empty_candidates_are_handled(self) -> None:
        self.assertEqual(order_by_scores([], []), [])


class CrossEncoderRerankerTest(unittest.TestCase):
    def test_missing_model_directory_is_a_clear_failure(self) -> None:
        with self.assertRaises(RerankerUnavailable):
            CrossEncoderReranker(Path("models") / "not-a-bundled-model")


class CrossEncoderRerankerIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        DEFAULT_RERANKER_DIR.is_dir(),
        "the bundled cross-encoder is not installed",
    )
    def test_bundled_cross_encoder_loads_and_scores_without_downloading(self) -> None:
        try:
            reranker = CrossEncoderReranker(DEFAULT_RERANKER_DIR)
        except RerankerUnavailable as error:
            skip_on_mega_cache_defect(self, error)

        scores = reranker.score(
            "waterproof hiking boots for wet trails",
            [
                "title: Black leather waterproof hiking boot | features: sealed seams",
                "title: Silk evening scarf | features: lightweight print",
            ],
        )

        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])

class DatasetSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            {"sample_id": f"public_{index:04d}", "scenario_type": scenario}
            for index, scenario in enumerate(
                ["buying"] * 80 + ["browsing"] * 80
                + ["intent_override"] * 30 + ["boundary"] * 10,
                start=1,
            )
        ]

    def test_the_holdout_matches_the_official_scenario_distribution(self) -> None:
        holdout = holdout_samples(self.samples)

        counts: dict[str, int] = {}
        for sample in holdout:
            counts[sample["scenario_type"]] = counts.get(sample["scenario_type"], 0) + 1

        self.assertEqual(len(holdout), 40)
        self.assertEqual(counts, HOLDOUT_SCENARIO_COUNTS)

    def test_frozen_loader_never_deserializes_holdout_payloads(self) -> None:
        samples = load_frozen_development_samples("data/public_set.jsonl")

        self.assertEqual(len(samples), 160)
        self.assertFalse(
            {sample["sample_id"] for sample in samples} & FROZEN_HOLDOUT_SAMPLE_IDS
        )

    def test_development_and_holdout_partition_the_public_set(self) -> None:
        development = development_samples(self.samples)
        holdout = holdout_samples(self.samples)

        self.assertEqual(len(development), 160)
        self.assertEqual(
            {sample["sample_id"] for sample in development}
            & {sample["sample_id"] for sample in holdout},
            set(),
        )
        self.assertEqual(
            len(development) + len(holdout),
            len(self.samples),
        )

    def test_the_split_does_not_depend_on_file_order(self) -> None:
        shuffled = list(reversed(self.samples))

        self.assertEqual(
            holdout_sample_ids(self.samples),
            holdout_sample_ids(shuffled),
        )

    def test_a_stratified_subset_keeps_the_scenario_mix_and_is_reproducible(self) -> None:
        first = stratified_subset(development_samples(self.samples), 40, seed=7)
        second = stratified_subset(development_samples(self.samples), 40, seed=7)

        counts: dict[str, int] = {}
        for sample in first:
            counts[sample["scenario_type"]] = counts.get(sample["scenario_type"], 0) + 1

        self.assertEqual(len(first), 40)
        self.assertEqual(
            [sample["sample_id"] for sample in first],
            [sample["sample_id"] for sample in second],
        )
        self.assertEqual(counts, {"buying": 16, "browsing": 16, "intent_override": 6, "boundary": 2})

    def test_a_subset_never_reaches_into_the_holdout(self) -> None:
        locked = holdout_sample_ids(self.samples)

        subset = stratified_subset(development_samples(self.samples), 60, seed=7)

        self.assertEqual({sample["sample_id"] for sample in subset} & locked, set())


class BenchmarkMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "sample_id": "public_0001",
                "scenario_type": "buying",
                "turn": 1,
                "query": "q1",
                "target": "T1",
                "pools": {
                    "1": ["A"],
                    "2": ["A", "B"],
                    "3": ["A", "B", "T1"],
                    "30": ["A", "B", "T1"],
                },
                "response_pool": ["A", "B", "T1"],
            },
            {
                "sample_id": "public_0002",
                "scenario_type": "browsing",
                "turn": 1,
                "query": "q2",
                "target": "T2",
                "pools": {
                    "1": ["A"],
                    "2": ["A", "B"],
                    "3": ["A", "B"],
                    "30": ["A", "B"],
                },
                "response_pool": ["A", "B"],
            },
            {
                "sample_id": "public_0002",
                "scenario_type": "browsing",
                "turn": 2,
                "query": "q3",
                "target": "T2",
                "pools": {
                    "1": ["A"],
                    "2": ["A", "T2"],
                    "3": ["A", "T2"],
                    "30": ["A", "T2"],
                },
                "response_pool": ["A", "T2"],
            },
        ]

    def test_pool_recall_separates_turn_and_session_level_evidence(self) -> None:
        deep = pool_recall(self.records, 3)
        shallow = pool_recall(self.records, 1)

        self.assertEqual(deep["turn_pool_recall"], round(2 / 3, 6))
        self.assertEqual(deep["session_pool_recall"], 1.0)
        self.assertEqual(shallow["session_pool_recall"], 0.0)

    def test_truncating_the_pool_can_only_lose_recall(self) -> None:
        self.assertLessEqual(
            pool_recall(self.records, 2)["session_pool_recall"],
            pool_recall(self.records, 3)["session_pool_recall"],
        )

    def test_a_session_converts_on_its_earliest_hitting_turn(self) -> None:
        ranked = {
            ("public_0001", 1): ["A", "T1"],
            ("public_0002", 1): ["A", "B"],
            ("public_0002", 2): ["T2", "A"],
        }

        metrics = session_metrics(self.records, ranked)

        self.assertEqual(metrics["session_count"], 2)
        self.assertEqual(metrics["hit_rate_at_10"], 1.0)
        # Ranks 2 and 1 across the two sessions.
        self.assertEqual(metrics["mrr"], round((0.5 + 1.0) / 2, 6))

    def test_intent_override_target_is_eligible_only_after_the_override(self) -> None:
        records = [
            {
                "sample_id": "override",
                "scenario_type": "intent_override",
                "turn": 1,
                "target": "TARGET",
                "hit_eligible": False,
            },
            {
                "sample_id": "override",
                "scenario_type": "intent_override",
                "turn": 3,
                "target": "TARGET",
                "hit_eligible": True,
            },
        ]
        ranked = {
            ("override", 1): ["TARGET", "A"],
            ("override", 3): ["A", "TARGET"],
        }

        metrics = session_metrics(records, ranked)

        self.assertEqual(metrics["hit_rate_at_10"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["mttc"], 3.0)

    def test_a_session_that_never_hits_contributes_zero(self) -> None:
        ranked = {
            ("public_0001", 1): ["A"],
            ("public_0002", 1): ["A"],
            ("public_0002", 2): ["A"],
        }

        metrics = session_metrics(self.records, ranked)

        self.assertEqual(metrics["hit_rate_at_10"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_a_session_subset_keeps_whole_sessions(self) -> None:
        subset = subset_records(self.records, 1)

        self.assertEqual(len({record["sample_id"] for record in subset}), 1)
        self.assertTrue(
            all(record["sample_id"] == subset[0]["sample_id"] for record in subset)
        )

    def test_the_fused_baseline_row_is_computed_without_a_model(self) -> None:
        row = baseline_rows(self.records)

        self.assertFalse(row["reranked"])
        self.assertEqual(row["depth"], 30)
        self.assertEqual(row["hit_rate_at_10"], 1.0)
        self.assertEqual(row["turn_latency_p95_ms"], 0.0)


class BenchmarkCacheTest(unittest.TestCase):
    def test_score_stage_rejects_general_network_connections(self) -> None:
        attempts = {
            "connect": lambda connection: connection.connect(("127.0.0.1", 9)),
            "connect_ex": lambda connection: connection.connect_ex(("127.0.0.1", 9)),
            "sendto": lambda connection: connection.sendto(
                b"blocked", ("127.0.0.1", 9)
            ),
        }
        for name, attempt in attempts.items():
            with self.subTest(name=name):
                def attempt_connection(_arguments):
                    with socket.socket() as connection:
                        attempt(connection)

                with patch(
                    "tools.benchmark_reranker._score_cache_offline",
                    side_effect=attempt_connection,
                ):
                    with self.assertRaisesRegex(RuntimeError, "network is disabled"):
                        score_cache(object())

    def test_cache_refuses_to_measure_without_the_dense_route(self) -> None:
        class DisabledDenseAgent:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def get_dense_route_metrics(self) -> dict:
                return {"status": "disabled", "disabled_reason": "missing index"}

        class _Arguments:
            dataset = "unused.jsonl"
            catalog = "unused.jsonl"
            output = "unused.jsonl"
            sessions = 0

        with (
            patch("tools.benchmark_reranker.load_frozen_development_samples", return_value=[]),
            patch("tools.benchmark_reranker.catalog_index", return_value=(set(), {}, {})),
            patch("tools.benchmark_reranker.Agent", DisabledDenseAgent),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the dense Retrieval Route"):
                build_cache(_Arguments())


class SummaryTest(unittest.TestCase):
    def summarize_declared_reports(self, arguments):
        identities = {
            json.loads(Path(path).read_text(encoding="utf-8"))["identity"]
            for path in arguments.reports
        }
        with patch(
            "tools.benchmark_reranker.RERANKER_CANDIDATES",
            {identity: object() for identity in identities},
        ):
            return summarize(arguments)

    def test_model_provenance_requires_the_declared_immutable_revision(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "weights.bin").write_bytes(b"weights")
        (root / "FETCHED.json").write_text(json.dumps({
            "identity": "candidate/model",
            "revision": "abc123",
        }), encoding="utf-8")

        manifest = {
            "candidate/model": RerankerCandidate("candidate", "abc123"),
            "different/model": RerankerCandidate("different", "abc123"),
        }
        with patch("tools.benchmark_reranker.RERANKER_CANDIDATES", manifest):
            provenance = _model_provenance(root, "candidate/model")

            self.assertEqual(provenance["revision"], "abc123")
            self.assertGreater(provenance["model_bytes"], len(b"weights"))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _model_provenance(root, "different/model")

    def test_the_winner_respects_the_runtime_budget(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        meta = root / "cache.jsonl.meta.json"
        meta.write_text(json.dumps({
            "session_count": 160,
            "turn_count": 1000,
            "baseline_wall_seconds": 100.0,
            "baseline_peak_rss_bytes": 1,
            "cache_sha256": "digest",
            "baseline_metrics": {"hit_rate_at_10": 0.50, "mrr": 0.3},
            "pool_recall_by_depth": {"30": {
                "session_pool_recall": 0.7, "turn_pool_recall": 0.5,
            }},
        }), encoding="utf-8")
        report = root / "model.json"
        report.write_text(json.dumps({
            "identity": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            "cache_sha256": "digest", "session_count": 160,
            "turn_count": 1000, "depths": [30, 50, 100, 150, 200, 250, 300],
            "model_bytes": 10,
            "peak_rss_bytes": 20,
            "baseline_row": {
                "identity": "none (fused-30 baseline)",
                "depth": 30,
                "reranked": False,
                "session_pool_recall": 0.7,
                "turn_pool_recall": 0.5,
                "hit_rate_at_10": 0.50,
                "mrr": 0.3,
                "session_count": 160,
                "recall_to_hit_conversion": 0.71,
                "turn_latency_p50_ms": 0.0,
                "turn_latency_p95_ms": 0.0,
                "turn_latency_mean_ms": 0.0,
                "added_seconds_total": 0.0,
            },
            "rows": [
                {
                    "identity": "cross-encoder/ms-marco-MiniLM-L-2-v2",
                    "depth": 50, "reranked": True,
                    "session_pool_recall": 0.8, "turn_pool_recall": 0.6,
                    "hit_rate_at_10": 0.60, "mrr": 0.4, "session_count": 160,
                    "recall_to_hit_conversion": 0.75,
                    "turn_latency_p50_ms": 10.0, "turn_latency_p95_ms": 20.0,
                    "turn_latency_mean_ms": 10.0, "added_seconds_total": 50.0,
                },
                {
                    "identity": "cross-encoder/ms-marco-MiniLM-L-2-v2",
                    "depth": 200, "reranked": True,
                    "session_pool_recall": 0.9, "turn_pool_recall": 0.8,
                    "hit_rate_at_10": 0.70, "mrr": 0.5, "session_count": 160,
                    "recall_to_hit_conversion": 0.78,
                    "turn_latency_p50_ms": 10.0, "turn_latency_p95_ms": 8000.0,
                    "turn_latency_mean_ms": 10.0, "added_seconds_total": 50.0,
                },
            ],
        }), encoding="utf-8")

        class _Arguments:
            reports = [str(report)]
            cache_meta = str(meta)
            full_run_sessions = 200
            runtime_budget_seconds = 400.0
            p95_budget_ms = 1500.0

        summary = self.summarize_declared_reports(_Arguments())

        # Depth 200 fits the aggregate budget but violates the independent
        # per-turn tail-latency gate.
        self.assertEqual(summary["selected"]["depth"], 50)
        self.assertEqual(summary["pool_recall_ceiling"], 0.9)
        self.assertEqual(
            summary["pool_recall_lost_to_truncation"],
            round(0.9 - 0.8, 6),
        )

        mismatched = root / "mismatched.json"
        mismatched_report = json.loads(report.read_text(encoding="utf-8"))
        mismatched_report["identity"] = "second-model"
        mismatched_report["cache_sha256"] = "different"
        mismatched.write_text(json.dumps(mismatched_report), encoding="utf-8")
        _Arguments.reports = [str(report), str(mismatched)]
        with self.assertRaisesRegex(RuntimeError, "identical cache/depth manifest"):
            self.summarize_declared_reports(_Arguments())

    def test_the_winner_prioritizes_depth_after_quality_floors(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        meta = root / "cache.jsonl.meta.json"
        meta.write_text(json.dumps({
            "session_count": 160,
            "turn_count": 1000,
            "baseline_wall_seconds": 100.0,
            "baseline_peak_rss_bytes": 1,
            "cache_sha256": "digest",
            "baseline_metrics": {
                "hit_rate_at_10": 0.50, "mrr": 0.3, "mttc": 6.0,
                "efficiency": 0.5, "recommended_technical_score": 0.44,
            },
            "pool_recall_by_depth": {"30": {
                "session_pool_recall": 0.7, "turn_pool_recall": 0.5,
            }},
        }), encoding="utf-8")
        baseline = {
            "identity": "none (fused-30 baseline)", "depth": 30,
            "reranked": False, "session_pool_recall": 0.7,
            "turn_pool_recall": 0.5, "hit_rate_at_10": 0.50,
            "mrr": 0.3, "mttc": 6.0, "efficiency": 0.5,
            "recommended_technical_score": 0.44, "session_count": 160,
            "recall_to_hit_conversion": 0.71, "turn_latency_p50_ms": 0.0,
            "turn_latency_p95_ms": 0.0, "turn_latency_mean_ms": 0.0,
            "added_seconds_total": 0.0,
        }
        shallow = {
            **baseline, "identity": "model", "depth": 50, "reranked": True,
            "session_pool_recall": 0.8, "hit_rate_at_10": 0.70,
            "mrr": 0.6, "recommended_technical_score": 0.70,
            "turn_latency_p50_ms": 10.0, "turn_latency_p95_ms": 20.0,
            "turn_latency_mean_ms": 10.0,
        }
        deep = {
            **shallow, "depth": 100, "session_pool_recall": 0.9,
            "hit_rate_at_10": 0.60, "mrr": 0.5,
            "recommended_technical_score": 0.60,
        }
        report = root / "model.json"
        report.write_text(json.dumps({
            "identity": "model", "model_bytes": 10, "peak_rss_bytes": 20,
            "cache_sha256": "digest", "session_count": 160,
            "turn_count": 1000, "depths": [30, 50, 100, 150, 200, 250, 300],
            "baseline_row": baseline, "rows": [shallow, deep],
        }), encoding="utf-8")

        class _Arguments:
            reports = [str(report)]
            cache_meta = str(meta)
            full_run_sessions = 200
            runtime_budget_seconds = 900.0
            p95_budget_ms = 1500.0

        summary = self.summarize_declared_reports(_Arguments())

        self.assertEqual(summary["selected"]["depth"], 100)

    def test_no_winner_does_not_relax_the_quality_floor(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        meta = root / "cache.jsonl.meta.json"
        meta.write_text(json.dumps({
            "session_count": 160, "turn_count": 1000,
            "baseline_wall_seconds": 100.0, "baseline_peak_rss_bytes": 1,
            "cache_sha256": "digest",
            "baseline_metrics": {
                "hit_rate_at_10": 0.60, "mrr": 0.4,
                "recommended_technical_score": 0.60,
            },
            "pool_recall_by_depth": {"30": {
                "session_pool_recall": 0.7, "turn_pool_recall": 0.5,
            }},
        }), encoding="utf-8")
        baseline = {
            "identity": "none", "depth": 30, "reranked": False,
            "session_pool_recall": 0.7, "turn_pool_recall": 0.5,
            "hit_rate_at_10": 0.60, "mrr": 0.4,
            "recommended_technical_score": 0.60, "session_count": 160,
            "recall_to_hit_conversion": 0.8, "turn_latency_p50_ms": 0.0,
            "turn_latency_p95_ms": 0.0, "turn_latency_mean_ms": 0.0,
            "added_seconds_total": 0.0,
        }
        regressing = {
            **baseline, "identity": "model", "depth": 300, "reranked": True,
            "session_pool_recall": 0.95, "hit_rate_at_10": 0.59,
            "recommended_technical_score": 0.61,
            "turn_latency_p50_ms": 10.0, "turn_latency_p95_ms": 20.0,
            "turn_latency_mean_ms": 10.0,
        }
        report = root / "model.json"
        report.write_text(json.dumps({
            "identity": "model", "model_bytes": 10, "peak_rss_bytes": 20,
            "cache_sha256": "digest", "session_count": 160,
            "turn_count": 1000, "depths": [30, 50, 100, 150, 200, 250, 300],
            "baseline_row": baseline, "rows": [regressing],
        }), encoding="utf-8")

        class _Arguments:
            reports = [str(report)]
            cache_meta = str(meta)
            full_run_sessions = 200
            runtime_budget_seconds = 900.0
            p95_budget_ms = 1500.0

        summary = self.summarize_declared_reports(_Arguments())

        self.assertIsNone(summary["selected"])
        self.assertEqual(summary["decision"], "no_reranker")

    def test_the_candidate_models_are_declared_for_reproduction(self) -> None:
        self.assertIn("cross-encoder/ms-marco-MiniLM-L-2-v2", RERANKER_CANDIDATES)
        self.assertGreaterEqual(len(RERANKER_CANDIDATES), 2)

    def test_the_benchmark_winner_is_frozen_for_slice_8(self) -> None:
        self.assertEqual(
            DEFAULT_RERANKER_IDENTITY,
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.assertEqual(FROZEN_RERANK_DEPTH, 50)


if __name__ == "__main__":
    unittest.main()
