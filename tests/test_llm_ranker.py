"""Offline tests for the optional LLM semantic-ranking stage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starter.agent import Agent
from starter.llm_ranker import OpenAISemanticRanker
from starter.openai_planning import DevelopmentBudget, ModelPricing


def build_ranker(**overrides) -> OpenAISemanticRanker:
    return OpenAISemanticRanker(
        model="test-model",
        pricing=ModelPricing(
            input_per_million_usd=0.2, output_per_million_usd=1.2
        ),
        budget=overrides.pop(
            "budget",
            DevelopmentBudget(
                limit_usd=10.0,
                warning_usd=40.0,
                review_boundary_usd=50.0,
                absolute_stop_usd=600.0,
            ),
        ),
        api_key="test-key",
        **overrides,
    )


class FakeResponses:
    def __init__(self, payload):
        self._payload = payload

    def create(self, **kwargs):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)

    def with_options(self, **kwargs):
        return self


def fake_response(text: str):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        status="completed",
        output_text=text,
    )


CANDIDATES = [("A", "product a"), ("B", "product b"), ("C", "product c")]


class SemanticRankerTest(unittest.TestCase):
    def test_valid_ranking_reorders_and_keeps_omissions_in_local_order(self) -> None:
        ranker = build_ranker()
        ranker._client = FakeClient(fake_response(json.dumps({"ranking": [3, 1]})))
        result = ranker.rank(["I want c"], "", CANDIDATES)
        self.assertIsNotNone(result)
        ordered, prompt_tokens, completion_tokens = result
        self.assertEqual(ordered, ["C", "A", "B"])
        self.assertEqual(prompt_tokens, 100)
        self.assertEqual(completion_tokens, 20)

    def test_out_of_range_and_duplicate_indices_are_ignored(self) -> None:
        ranker = build_ranker()
        ranker._client = FakeClient(
            fake_response(json.dumps({"ranking": [2, 2, 9, 0, 1]}))
        )
        ordered, _, _ = ranker.rank(["hello"], "", CANDIDATES)
        self.assertEqual(ordered, ["B", "A", "C"])

    def test_provider_failure_fails_open_to_none(self) -> None:
        ranker = build_ranker()
        ranker._client = FakeClient(TimeoutError("slow"))
        self.assertIsNone(ranker.rank(["hello"], "", CANDIDATES))
        self.assertEqual(ranker.metrics()["failure_count"], 1)

    def test_malformed_output_fails_open_to_none(self) -> None:
        ranker = build_ranker()
        ranker._client = FakeClient(fake_response("not json"))
        self.assertIsNone(ranker.rank(["hello"], "", CANDIDATES))

    def test_exhausted_budget_prevents_the_call(self) -> None:
        budget = DevelopmentBudget(
            limit_usd=0.000001,
            warning_usd=40.0,
            review_boundary_usd=50.0,
            absolute_stop_usd=600.0,
        )
        ranker = build_ranker(budget=budget)
        ranker._client = FakeClient(fake_response(json.dumps({"ranking": [1]})))
        self.assertIsNone(ranker.rank(["hello"], "", CANDIDATES))
        self.assertEqual(ranker.metrics()["call_count"], 0)

    def test_single_candidate_needs_no_call(self) -> None:
        ranker = build_ranker()
        ranker._client = FakeClient(fake_response(json.dumps({"ranking": [1]})))
        self.assertIsNone(ranker.rank(["hello"], "", CANDIDATES[:1]))
        self.assertEqual(ranker.metrics()["call_count"], 0)


class RecordingSemanticRanker:
    max_candidates = 20

    def __init__(self, ordered: list[str]) -> None:
        self.ordered = ordered
        self.calls: list[tuple[list[str], str]] = []

    def rank(self, dialog_messages, constraint_summary, candidates):
        self.calls.append((list(dialog_messages), constraint_summary))
        available = [parent_asin for parent_asin, _ in candidates]
        head = [item for item in self.ordered if item in available]
        head.extend(item for item in available if item not in head)
        return head, 111, 22

    def configuration(self) -> dict:
        return {"stage": "semantic_ranking", "model": "recording"}


class AgentSemanticRankingTest(unittest.TestCase):
    def test_agent_applies_ranking_and_reports_token_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text("".join(
                json.dumps(row) + "\n" for row in [
                    {"parent_asin": "X1", "title": "Blue cotton shoe"},
                    {"parent_asin": "X2", "title": "Blue cotton shoe deluxe"},
                    {"parent_asin": "X3", "title": "Blue cotton shoe premium"},
                ]
            ), encoding="utf-8")
            ranker = RecordingSemanticRanker(["X3", "X2", "X1"])
            agent = Agent(catalog_path, semantic_ranker=ranker)
            try:
                agent.reset("session", {})
                response = agent.respond("session", "blue cotton shoe", 1, 10)
            finally:
                agent.close()
            identifiers = [
                item["parent_asin"] for item in response["recommendations"]
            ]
            self.assertEqual(identifiers[0], "X3")
            self.assertEqual(response["usage"]["prompt_tokens"], 111)
            self.assertEqual(response["usage"]["completion_tokens"], 22)
            self.assertEqual(len(ranker.calls), 1)
            self.assertEqual(ranker.calls[0][0], ["blue cotton shoe"])




class ScriptedRanker:
    max_candidates = 12

    def __init__(self, order_map, fail=False):
        self.order_map = order_map
        self.fail = fail
        self.calls = 0

    def rank(self, dialog_messages, constraint_summary, candidates):
        self.calls += 1
        if self.fail:
            return None
        available = [parent_asin for parent_asin, _ in candidates]
        ordered = [a for a in self.order_map if a in available]
        ordered.extend(a for a in available if a not in ordered)
        return ordered, 10, 2

    def configuration(self):
        return {"stage": "scripted"}

    def metrics(self):
        return {"calls": self.calls}


class TournamentRankerTest(unittest.TestCase):
    def _candidates(self, count):
        return [(f"P{i:02d}", f"product {i}") for i in range(count)]

    def test_finalists_advance_and_final_orders_them(self) -> None:
        from starter.llm_ranker import TournamentSemanticRanker

        chunk = ScriptedRanker(order_map=[f"P{i:02d}" for i in range(24)])
        final = ScriptedRanker(order_map=["P13", "P01", "P12", "P00"])
        ranker = TournamentSemanticRanker(
            chunk, final, chunk_size=12, chunk_count=2, finalists_per_chunk=2
        )
        result = ranker.rank(["hi"], "", self._candidates(24))
        self.assertIsNotNone(result)
        ordered, prompt_tokens, completion_tokens = result
        self.assertEqual(ordered[:4], ["P13", "P01", "P12", "P00"])
        self.assertEqual(len(ordered), 24)
        self.assertEqual(len(set(ordered)), 24)
        self.assertEqual(chunk.calls, 2)
        self.assertEqual(final.calls, 1)
        self.assertEqual(prompt_tokens, 30)
        self.assertEqual(completion_tokens, 6)

    def test_failed_chunk_keeps_local_order_and_still_competes(self) -> None:
        from starter.llm_ranker import TournamentSemanticRanker

        chunk = ScriptedRanker(order_map=[], fail=True)
        final = ScriptedRanker(order_map=["P12", "P00"])
        ranker = TournamentSemanticRanker(
            chunk, final, chunk_size=12, chunk_count=2, finalists_per_chunk=2
        )
        result = ranker.rank(["hi"], "", self._candidates(24))
        self.assertIsNotNone(result)
        ordered, _, _ = result
        self.assertEqual(ordered[0], "P12")
        self.assertEqual(len(set(ordered)), 24)

    def test_everything_failing_returns_none(self) -> None:
        from starter.llm_ranker import TournamentSemanticRanker

        chunk = ScriptedRanker(order_map=[], fail=True)
        final = ScriptedRanker(order_map=[], fail=True)
        ranker = TournamentSemanticRanker(chunk, final)
        self.assertIsNone(ranker.rank(["hi"], "", self._candidates(24)))


if __name__ == "__main__":
    unittest.main()
