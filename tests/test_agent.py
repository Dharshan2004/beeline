from __future__ import annotations

import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import catalog_index, evaluate
from retrieval.dense_index import BuildConfig, build
from retrieval.dense_route import DenseRetrievalRoute
from retrieval.embedder import DEFAULT_MODEL_DIR
from retrieval.fusion import FixedFusionPolicy
from retrieval.reranker import (
    DEFAULT_RERANKER_DIR,
    LocalRerankerWorker,
    UnavailableReranker,
)
from starter.agent import Agent
from starter.constraint_state import AddConstraint, TurnPlan


class AgentContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        reranker_patch = patch(
            "starter.agent.build_live_reranker",
            return_value=UnavailableReranker("disabled_for_non_reranker_test"),
        )
        reranker_patch.start()
        self.addCleanup(reranker_patch.stop)

    def write_catalog(self, rows: list[dict]) -> None:
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    class RecordingDenseRoute:
        def __init__(self, candidates: list[tuple[str, float]]) -> None:
            self.candidates = candidates
            self.queries: list[tuple[str, int]] = []
            self.query_count = 0
            self.last_candidate_count = 0

        def search(self, query: str, limit: int) -> list[tuple[str, float]]:
            self.queries.append((query, limit))
            self.query_count += 1
            result = self.candidates[:limit]
            self.last_candidate_count = len(result)
            return result

        def metrics(self) -> dict:
            return {
                "status": "available",
                "load_seconds": 0.125,
                "query_count": self.query_count,
                "last_query_seconds": 0.004,
                "last_candidate_count": self.last_candidate_count,
            }

    class FailingDenseRoute:
        def search(self, query: str, limit: int) -> list[tuple[str, float]]:
            raise RuntimeError("dense query failed")

        def metrics(self) -> dict:
            return {"status": "available", "query_count": 0}

    class RecordingReranker:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.calls: list[tuple[str, list[str], list[str]]] = []

        def rerank(self, query, candidates, documents):
            self.calls.append((query, list(candidates), list(documents)))
            if self.fail:
                raise RuntimeError("reranker failed")
            return [candidates[-1], *candidates[:-1]] if candidates else []

        def metrics(self) -> dict:
            return {"status": "available"}

        def close(self) -> None:
            pass

    def test_semantic_paraphrase_uses_ordered_dense_candidates_live(self) -> None:
        self.write_catalog([
            {
                "parent_asin": "SCARF",
                "title": "Red wool winter scarf",
                "description": ["Soft cold-weather accessory"],
            },
            {"parent_asin": "BOOTS", "title": "Black leather ankle boots"},
        ])
        literal_agent = Agent(
            self.catalog_path,
            dense_route=self.RecordingDenseRoute([]),
        )
        literal_agent.reset("literal-session", {})
        literal_response = literal_agent.respond(
            "literal-session",
            "Something cozy to wrap around my neck",
            1,
            2,
        )
        route = self.RecordingDenseRoute([("SCARF", 0.91), ("BOOTS", 0.42)])
        agent = Agent(self.catalog_path, dense_route=route)
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "Something cozy to wrap around my neck",
            1,
            2,
        )

        self.assertEqual(literal_response["recommendations"], [])
        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "SCARF"}, {"parent_asin": "BOOTS"}],
        )
        self.assertEqual(
            route.queries,
            [("Something cozy to wrap around my neck", 100)],
        )

    def test_dense_candidates_are_catalog_valid_unique_and_keep_route_order(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "First product"},
            {"parent_asin": "B", "title": "Second product"},
        ])
        route = self.RecordingDenseRoute([
            ("UNKNOWN", 0.99),
            ("B", 0.90),
            ("B", 0.85),
            ("A", 0.80),
        ])
        agent = Agent(self.catalog_path, dense_route=route)
        agent.reset("session", {})

        response = agent.respond("session", "semantic wording", 1, 10)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "B"}, {"parent_asin": "A"}],
        )

    def test_fixed_hybrid_fusion_combines_all_routes_through_agent(self) -> None:
        self.write_catalog([
            {"parent_asin": "STRUCTURED", "title": "Cotton gala footwear"},
            {"parent_asin": "BM25", "title": "Formal ceremony dress"},
            {"parent_asin": "DENSE", "title": "Evening accessory"},
        ])
        route = self.RecordingDenseRoute([("DENSE", 0.95)])
        agent = Agent(
            self.catalog_path,
            dense_route=route,
            fusion_policy=FixedFusionPolicy(
                weights={"structured": 0.4, "bm25": 0.3, "dense": 0.3},
            ),
        )
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I prefer cotton for a ceremony.",
            1,
            10,
        )

        self.assertEqual(
            {item["parent_asin"] for item in response["recommendations"]},
            {"STRUCTURED", "BM25", "DENSE"},
        )
        self.assertEqual(agent.get_retrieval_configuration(), {
            "policy_version": "pool-aware-global-v2",
            "normalizer": "per-route-min-max-v1",
            "route_depth": 100,
            "fused_candidate_depth": 50,
            "reranker_identity": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_depth": 50,
            "reranker_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            "reranker_directory_sha256": None,
            "rerank_deadline_seconds": 1.5,
            "reranker_status": "disabled",
            "reranker_enabled": False,
            "weights": {"structured": 0.4, "bm25": 0.3, "dense": 0.3},
        })

    def test_live_reranker_receives_the_deep_pool_and_changes_official_order(self) -> None:
        self.write_catalog([
            {"parent_asin": f"A{index:03d}", "title": f"Product {index}"}
            for index in range(60)
        ])
        route = self.RecordingDenseRoute([
            (f"A{index:03d}", float(60 - index)) for index in range(60)
        ])
        reranker = self.RecordingReranker()
        agent = Agent(self.catalog_path, dense_route=route, reranker=reranker)
        agent.reset("session", {})

        response = agent.respond("session", "semantic wording", 1, 10)

        query, candidates, documents = reranker.calls[0]
        self.assertEqual(len(candidates), 50)
        self.assertEqual(candidates[-1], "A049")
        self.assertEqual(len(documents), 50)
        self.assertEqual(response["recommendations"][0], {"parent_asin": "A049"})
        self.assertEqual(len(response["recommendations"]), 10)

    @unittest.skipUnless(
        DEFAULT_RERANKER_DIR.is_dir(),
        "the bundled cross-encoder is not installed",
    )
    def test_bundled_reranker_changes_order_through_official_agent_interface(self) -> None:
        class FixedPoolPolicy:
            version = "fixed-test-pool-v1"
            candidate_limit = 50

            def rank(self, route_scores, candidate_limit=None):
                return ["SCARF", "BOOTS"]

        self.write_catalog([
            {
                "parent_asin": "SCARF",
                "title": "Silk evening scarf",
                "features": ["lightweight print"],
            },
            {
                "parent_asin": "BOOTS",
                "title": "Black leather waterproof hiking boot",
                "features": ["sealed seams"],
            },
        ])
        reranker = LocalRerankerWorker(DEFAULT_RERANKER_DIR)
        agent = Agent(
            self.catalog_path,
            dense_route=self.RecordingDenseRoute([]),
            fusion_policy=FixedPoolPolicy(),
            reranker=reranker,
        )
        self.addCleanup(agent.close)
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "waterproof hiking boots for wet trails",
            1,
            2,
        )

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "BOOTS"}, {"parent_asin": "SCARF"}],
        )

    def test_live_reranker_failure_preserves_fused_order_through_agent(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "First product"},
            {"parent_asin": "B", "title": "Second product"},
        ])
        route = self.RecordingDenseRoute([("A", 1.0), ("B", 0.5)])
        reranker = self.RecordingReranker(fail=True)
        agent = Agent(self.catalog_path, dense_route=route, reranker=reranker)
        agent.reset("session", {})

        response = agent.respond("session", "semantic wording", 1, 10)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
        )

    def test_invalid_reranker_identifiers_cannot_escape_the_catalog(self) -> None:
        class InvalidReranker(self.RecordingReranker):
            def rerank(self, query, candidates, documents):
                return ["UNKNOWN", *candidates[:-1]]

        self.write_catalog([
            {"parent_asin": "A", "title": "First product"},
            {"parent_asin": "B", "title": "Second product"},
        ])
        route = self.RecordingDenseRoute([("A", 1.0), ("B", 0.5)])
        agent = Agent(
            self.catalog_path,
            dense_route=route,
            reranker=InvalidReranker(),
        )
        agent.reset("session", {})

        response = agent.respond("session", "semantic wording", 1, 10)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "A"}, {"parent_asin": "B"}],
        )

    def test_dense_query_failure_falls_back_to_deterministic_local_retrieval(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Blue running shoe"},
            {"parent_asin": "B", "title": "Red wool scarf"},
        ])
        agent = Agent(self.catalog_path, dense_route=self.FailingDenseRoute())
        agent.reset("session", {})

        first = agent.respond("session", "blue running shoe", 1, 10)
        second = agent.respond("session", "blue running shoe", 2, 10)

        expected = [{"parent_asin": "A"}]
        self.assertEqual(first["recommendations"], expected)
        self.assertEqual(second["recommendations"], expected)

    def test_dense_route_metrics_expose_load_latency_and_candidate_count(self) -> None:
        self.write_catalog([{"parent_asin": "A", "title": "Product"}])
        route = self.RecordingDenseRoute([("A", 0.90)])
        agent = Agent(self.catalog_path, dense_route=route)
        agent.reset("session", {})

        agent.respond("session", "semantic wording", 1, 10)

        self.assertEqual(agent.get_dense_route_metrics(), {
            "status": "available",
            "load_seconds": 0.125,
            "query_count": 1,
            "last_query_seconds": 0.004,
            "last_candidate_count": 1,
        })

    def test_runtime_configuration_identifies_every_scored_component(self) -> None:
        self.write_catalog([{"parent_asin": "A", "title": "Product"}])
        agent = Agent(
            self.catalog_path,
            dense_route=self.RecordingDenseRoute([]),
        )

        configuration = agent.get_runtime_configuration()

        self.assertEqual(configuration["version"], "shopping-agent-runtime-v1")
        self.assertEqual(len(configuration["catalog"]["sha256"]), 64)
        self.assertIn("dense_index_and_model", configuration)
        self.assertEqual(
            configuration["planning"]["prompt_version"],
            "shopping-turn-planner-v2",
        )
        self.assertEqual(
            configuration["planning"]["replacement_evidence_version"],
            "explicit-replacement-evidence-v1",
        )
        self.assertEqual(
            len(configuration["planning"]["replacement_evidence_sha256"]),
            64,
        )
        self.assertIn("fusion_and_retrieval", configuration)
        self.assertEqual(configuration["feature_flags"]["local_reranking"], False)
        self.assertEqual(configuration["reranker"]["status"], "disabled")
        self.assertEqual(configuration["cost_limits_usd"]["absolute_stop"], 600)

    def test_candidate_tracing_records_each_requested_depth_independently(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Blue running shoe"},
            {"parent_asin": "B", "title": "Blue walking shoe"},
            {"parent_asin": "C", "title": "Blue casual shoe"},
        ])
        agent = Agent(
            self.catalog_path,
            dense_route=self.RecordingDenseRoute([]),
            trace_pool_depths=(1, 2, 3),
        )
        agent.reset("session", {})

        agent.respond("session", "blue shoe", 1, 10)

        trace = agent.get_candidate_traces()["session"][0]
        self.assertEqual(set(trace["pools"]), {"1", "2", "3"})
        self.assertEqual(set(trace["route_candidates"]), {"structured", "bm25", "dense"})
        self.assertEqual(trace["planning"]["source"], "fallback")
        self.assertEqual(trace["planning"]["state_revision"], 1)
        self.assertEqual(len(trace["pools"]["1"]), 1)
        self.assertEqual(len(trace["pools"]["2"]), 2)
        self.assertEqual(len(trace["pools"]["3"]), 3)

    def test_dense_query_includes_accumulated_active_constraint_state(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Cotton walking shoe"},
            {"parent_asin": "B", "title": "Leather walking shoe"},
        ])
        route = self.RecordingDenseRoute([("A", 0.9)])
        agent = Agent(self.catalog_path, dense_route=route)
        agent.reset("session", {})
        agent.respond("session", "I need cotton.", 1, 10)

        agent.respond("session", "Show me options.", 2, 10)

        second_query, depth = route.queries[1]
        self.assertIn("material: cotton", second_query)
        self.assertEqual(depth, 100)

    def test_missing_dense_assets_disable_route_without_network_or_invalid_output(self) -> None:
        self.write_catalog([{"parent_asin": "A", "title": "Blue running shoe"}])
        absent_model = Path(self.directory.name) / "absent-model"
        absent_artifact = Path(self.directory.name) / "absent-artifact"
        with (
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network access is disabled"),
            ),
            patch.object(
                socket.socket,
                "bind",
                side_effect=AssertionError("listening ports are not permitted"),
            ),
        ):
            route = DenseRetrievalRoute(
                self.catalog_path,
                artifact_dir=absent_artifact,
                model_dir=absent_model,
            )
            agent = Agent(self.catalog_path, dense_route=route)
            agent.reset("session", {})
            response = agent.respond("session", "blue running shoe", 1, 10)

        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        metrics = agent.get_dense_route_metrics()
        self.assertEqual(metrics["status"], "disabled")
        self.assertIn("ModelUnavailable", metrics["disabled_reason"])
        self.assertEqual(metrics["query_count"], 0)
        self.assertEqual(metrics["last_candidate_count"], 0)

    @unittest.skipUnless(
        DEFAULT_MODEL_DIR.is_dir()
        and importlib.util.find_spec("qdrant_client") is not None,
        "bundled dense runtime assets are not installed",
    )
    def test_real_local_dense_route_recovers_paraphrase_through_agent(self) -> None:
        self.write_catalog([
            {
                "parent_asin": "SCARF",
                "title": "Red wool winter scarf",
                "description": ["A soft scarf for cold weather"],
            },
            {
                "parent_asin": "BOOTS",
                "title": "Black leather ankle boots",
                "description": ["Durable footwear for wet trails"],
            },
        ])
        artifact = Path(self.directory.name) / "dense"
        build(BuildConfig(
            catalog_path=self.catalog_path,
            artifact_dir=artifact,
            model_dir=DEFAULT_MODEL_DIR,
            batch_size=2,
        ))
        with (
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network access is disabled"),
            ),
            patch.object(
                socket.socket,
                "bind",
                side_effect=AssertionError("listening ports are not permitted"),
            ),
        ):
            route = DenseRetrievalRoute(
                self.catalog_path,
                artifact_dir=artifact,
                model_dir=DEFAULT_MODEL_DIR,
            )
            agent = Agent(self.catalog_path, dense_route=route)
            agent.reset("session", {})
            response = agent.respond(
                "session",
                "Something warm to wrap around my neck in winter",
                1,
                2,
            )
            route.close()

        self.assertEqual(response["recommendations"][0], {"parent_asin": "SCARF"})
        self.assertEqual(agent.get_dense_route_metrics()["last_candidate_count"], 2)

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
        state = agent.get_constraint_state("session")
        self.assertEqual(
            [(item["attribute"], item["normalized_value"]) for item in state],
            [("category", "shoe"), ("material", "cotton")],
        )
        material = state[1]
        self.assertEqual(material["raw_phrase"], "Cotton")
        self.assertEqual(material["classification"], "hard")
        self.assertEqual(material["source_turn"], 1)
        self.assertEqual(material["confidence"], 0.95)
        self.assertEqual(material["status"], "active")
        self.assertEqual(material["scope"], "product_intent")
        self.assertEqual(material["match_rule"], "all")
        self.assertTrue(material["constraint_id"])

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
        slippers = agent.respond(
            "session",
            "Actually, I need slippers instead of boots.",
            3,
            10,
        )

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

    def test_compound_boundary_and_requirement_apply_on_the_same_turn(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Blue cotton walking shoe"},
            {"parent_asin": "B", "title": "Red leather walking shoe"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})
        agent.respond("session", "I prefer blue.", 1, 10)

        response = agent.respond(
            "session",
            "I don't care about color, but I need cotton.",
            2,
            10,
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(agent.get_constraint_revision("session"), 2)
        self.assertEqual(
            [
                (item["attribute"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [("color", "dismissed"), ("material", "active")],
        )

    def test_mixed_soft_and_hard_requirements_are_both_active(self) -> None:
        self.write_catalog([
            {"parent_asin": "A", "title": "Blue cotton walking shoe"},
            {"parent_asin": "B", "title": "Blue leather walking shoe"},
            {"parent_asin": "C", "title": "Red cotton walking shoe"},
        ])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I prefer blue, but must have cotton.",
            1,
            10,
        )

        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["A", "C"],
        )
        self.assertEqual(
            [
                (item["attribute"], item["classification"])
                for item in agent.get_constraint_state("session")
            ],
            [("color", "soft"), ("material", "hard")],
        )

    def test_invalid_fallback_plan_preserves_state_and_still_recommends(self) -> None:
        class InvalidPlanAgent(Agent):
            def _interpret_turn(self, user_message, turn, state):
                return TurnPlan(
                    expected_state_revision=state.revision,
                    source_turn=turn,
                    mutations=(
                        AddConstraint(
                            attribute="unknown",
                            values=("invalid",),
                            match_rule="all",
                            classification="hard",
                            scope="product_intent",
                            raw_phrase="invalid",
                            confidence=0.9,
                        ),
                    ),
                )

        self.write_catalog([{"parent_asin": "A", "title": "Blue shoe"}])
        agent = InvalidPlanAgent(self.catalog_path)
        agent.reset("session", {})

        response = agent.respond("session", "blue shoe", 1, 10)

        self.assertEqual(agent.get_constraint_revision("session"), 1)
        self.assertEqual(
            [item["attribute"] for item in agent.get_constraint_state("session")],
            ["category", "color"],
        )
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])

    def test_clarification_tracking_does_not_change_constraint_revision(self) -> None:
        self.write_catalog([{"parent_asin": "A", "title": "Blue shoe"}])
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        agent.respond("session", "unrecognized phrase", 1, 10)

        self.assertEqual(agent.get_constraint_revision("session"), 0)

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
            [("shoe", "active"), ("blue", "superseded"), ("cotton", "active")],
        )

    def test_evaluator_handles_repeated_product_intent_overrides(self) -> None:
        class ScriptedOverrideAgent(Agent):
            def respond(self, session_id, user_message, turn, top_k):
                scripted_messages = {
                    1: "I need shoes.",
                    2: "Actually, I need boots instead of shoes.",
                    3: "Actually, I need slippers instead of boots.",
                }
                return super().respond(
                    session_id,
                    scripted_messages.get(turn, user_message),
                    turn,
                    top_k,
                )

        self.write_catalog([
            {"parent_asin": "A", "title": "Everyday shoes", "categories": ["Shoes"]},
            {"parent_asin": "B", "title": "Everyday boots", "categories": ["Boots"]},
            {"parent_asin": "C", "title": "Everyday slippers", "categories": ["Slippers"]},
        ])
        samples = [{
            "sample_id": "repeated_override_0001",
            "scenario_type": "intent_override",
            "user_profile": {},
            "ground_truth": {"parent_asin": "C"},
            "intent_card": {
                "target_category": "slippers",
                "hard_constraints": ["slipper"],
                "soft_preferences": ["shoe"],
            },
            "behavior": {
                "scenario_type": "intent_override",
                "override": {
                    "turn": 2,
                    "old_value": "shoe",
                    "new_value": "boot",
                    "message": "Actually, boots instead of shoes.",
                },
            },
        }]
        catalog_ids, categories, products = catalog_index(self.catalog_path)
        agent = ScriptedOverrideAgent(self.catalog_path)

        result = evaluate(agent, samples, catalog_ids, categories, products)

        self.assertEqual(result["sessions"][0]["first_hit_turn"], 3)
        session_id = next(iter(agent._sessions))
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state(session_id)
            ],
            [
                ("shoe", "superseded"),
                ("boot", "superseded"),
                ("slipper", "active"),
            ],
        )

    def test_evaluator_falls_back_from_a_contradictory_turn_plan(self) -> None:
        class ContradictoryPlanAgent(Agent):
            def _interpret_turn(self, user_message, turn, state):
                return TurnPlan(
                    expected_state_revision=state.revision,
                    source_turn=turn,
                    mutations=(
                        AddConstraint(
                            attribute="color",
                            values=("blue",),
                            match_rule="all",
                            classification="hard",
                            scope="product_intent",
                            raw_phrase="blue",
                            confidence=0.9,
                        ),
                        AddConstraint(
                            attribute="color",
                            values=("red",),
                            match_rule="all",
                            classification="hard",
                            scope="product_intent",
                            raw_phrase="red",
                            confidence=0.9,
                        ),
                    ),
                )

        self.write_catalog([
            {
                "parent_asin": "TARGET",
                "title": "Blue walking shoe",
                "features": ["blue"],
                "categories": ["Shoes"],
            },
            {
                "parent_asin": "DECOY",
                "title": "Red walking shoe",
                "features": ["red"],
                "categories": ["Shoes"],
            },
        ])
        samples = [{
            "sample_id": "contradictory_plan_0001",
            "scenario_type": "buying",
            "user_profile": {},
            "ground_truth": {"parent_asin": "TARGET"},
            "intent_card": {
                "target_category": "walking shoe",
                "hard_constraints": ["blue"],
                "soft_preferences": [],
            },
            "behavior": {"scenario_type": "buying"},
        }]
        catalog_ids, categories, products = catalog_index(self.catalog_path)
        agent = ContradictoryPlanAgent(self.catalog_path)

        result = evaluate(agent, samples, catalog_ids, categories, products)

        self.assertEqual(result["sessions"][0]["first_hit_turn"], 1)
        session_id = next(iter(agent._sessions))
        self.assertEqual(agent.get_constraint_revision(session_id), 1)
        self.assertEqual(
            [item["normalized_value"] for item in agent.get_constraint_state(session_id)],
            ["shoe", "blue"],
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
