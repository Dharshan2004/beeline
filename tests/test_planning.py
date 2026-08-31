from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.reranker import UnavailableReranker
from starter.agent import Agent
from starter.constraint_state import ConstraintState
from starter.planning import (
    MissingCredentialsError,
    PlanningRequest,
    PlanningToolError,
    ProviderResponse,
    decode_plan,
)


def plan_payload(
    *,
    revision: int = 0,
    turn: int = 1,
    mutations: list[dict] | None = None,
    tools: list[str] | None = None,
    clarification: dict | None = None,
) -> dict:
    return {
        "expected_state_revision": revision,
        "source_turn": turn,
        "mutations": mutations or [],
        "retrieval_tools": tools or ["structured", "bm25", "dense"],
        "clarification": clarification,
    }


def add_constraint(
    value: str,
    *,
    attribute: str = "material",
    classification: str = "hard",
    scope: str = "product_intent",
) -> dict:
    return {
        "type": "add_constraint",
        "attribute": attribute,
        "values": [value],
        "match_rule": "all",
        "classification": classification,
        "scope": scope,
        "raw_phrase": value,
        "confidence": 0.97,
    }


def replace_product_intent(
    *,
    product_intent_id: str = "intent-1",
    raw_phrase: str,
) -> dict:
    return {
        "type": "replace_product_intent",
        "product_intent_id": product_intent_id,
        "raw_phrase": raw_phrase,
    }


def replace_constraint(
    constraint_id: str,
    value: str,
    *,
    attribute: str,
    raw_phrase: str,
) -> dict:
    return {
        **add_constraint(value, attribute=attribute),
        "type": "replace_constraint",
        "constraint_id": constraint_id,
        "raw_phrase": raw_phrase,
    }


class ScriptedProvider:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[PlanningRequest] = []

    def plan(self, request: PlanningRequest):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class CallbackProvider:
    def __init__(self) -> None:
        self.requests: list[PlanningRequest] = []

    def plan(self, request: PlanningRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            return plan_payload(
                mutations=[add_constraint("cotton")],
                tools=["structured"],
            )
        return plan_payload(
            revision=1,
            turn=2,
            tools=["bm25"],
        )


class PlanningLoopTest(unittest.TestCase):
    class RecordingDenseRoute:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, limit: int) -> list[tuple[str, float]]:
            self.calls += 1
            return []

        def metrics(self) -> dict:
            return {"status": "available", "query_count": self.calls}

    class RecordingReranker:
        identity = "test-reranker"

        def __init__(self) -> None:
            self.calls = 0

        def rerank(self, query, candidates, documents):
            self.calls += 1
            return list(reversed(candidates))

        def metrics(self) -> dict:
            return {"status": "available"}

        def close(self) -> None:
            pass

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        reranker_patch = patch(
            "starter.agent.build_live_reranker",
            return_value=UnavailableReranker("disabled_for_planning_test"),
        )
        reranker_patch.start()
        self.addCleanup(reranker_patch.stop)
        self.write_catalog([
            {
                "parent_asin": "A",
                "title": "Blue cotton walking shoe",
                "features": ["cotton", "blue"],
            },
            {
                "parent_asin": "B",
                "title": "Red leather walking shoe",
                "features": ["leather", "red"],
            },
            {
                "parent_asin": "C",
                "title": "Plain house slipper",
                "categories": ["Slippers"],
            },
            {
                "parent_asin": "D",
                "title": "Plain outdoor boot",
                "categories": ["Boots"],
            },
        ])

    def write_catalog(self, rows: list[dict]) -> None:
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_connected_plan_updates_state_selects_tools_and_clarifies(self) -> None:
        provider = ScriptedProvider([
            ProviderResponse(
                output=plan_payload(
                    mutations=[add_constraint("cotton")],
                    tools=["structured"],
                    clarification={
                        "ask_attribute": "color",
                        "message": "Which color would you prefer?",
                    },
                ),
                prompt_tokens=21,
                completion_tokens=8,
            ),
        ])
        dense_route = self.RecordingDenseRoute()
        agent = Agent(
            self.catalog_path,
            planning_provider=provider,
            dense_route=dense_route,
        )
        agent.reset("session", {})

        with patch.object(
            agent.retrieval,
            "_search",
            side_effect=AssertionError("BM25 was not selected"),
        ):
            response = agent.respond(
                "session",
                "Natural-fiber walking shoes",
                1,
                10,
            )

        self.assertEqual(dense_route.calls, 0)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(response["ask_attribute"], "color")
        self.assertIn("Which color would you prefer?", response["message"])
        self.assertEqual(response["usage"], {
            "prompt_tokens": 21,
            "completion_tokens": 8,
        })
        self.assertEqual(agent.get_constraint_revision("session"), 1)
        self.assertEqual(
            [item["normalized_value"] for item in agent.get_constraint_state("session")],
            ["cotton"],
        )
        self.assertEqual(agent.get_planning_history("session")[0], {
            "turn": 1,
            "user_message": "Natural-fiber walking shoes",
            "state_revision": 1,
            "source": "connected",
            "attempts": 1,
            "retrieval_tools": ["structured"],
            "ask_attribute": "color",
            "fallback_reason": None,
            "errors": [],
        })

    def test_connected_tool_selection_cannot_bypass_fixed_post_fusion_reranking(self) -> None:
        provider = ScriptedProvider([plan_payload(tools=["structured"])])
        reranker = self.RecordingReranker()
        agent = Agent(
            self.catalog_path,
            planning_provider=provider,
            dense_route=self.RecordingDenseRoute(),
            reranker=reranker,
        )
        agent.reset("session", {})

        agent.respond("session", "walking shoe", 1, 10)

        self.assertEqual(reranker.calls, 1)

    def test_shell_web_code_and_catalog_mutation_tools_are_never_allowed(self) -> None:
        supported_values = {
            "material": {"cotton"},
            "color": {"blue"},
        }
        for forbidden_tool in (
            "shell",
            "web",
            "python",
            "catalog_mutation",
            "local_rerank",
        ):
            with self.subTest(tool=forbidden_tool):
                with self.assertRaisesRegex(
                    PlanningToolError,
                    "unapproved retrieval tools",
                ):
                    decode_plan(
                        plan_payload(tools=[forbidden_tool]),
                        user_message="walking shoes",
                        turn=1,
                        state=ConstraintState(),
                        supported_values=supported_values,
                    )

    def test_unknown_tools_retry_once_then_take_over_offline(self) -> None:
        provider = ScriptedProvider([
            plan_payload(tools=["shell"]),
            plan_payload(tools=["web"]),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        response = agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(
            [item["normalized_value"] for item in agent.get_constraint_state("session")],
            ["cotton"],
        )
        diagnostic = agent.get_planning_history("session")[0]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "unapproved_tool")
        self.assertEqual(diagnostic["attempts"], 2)

    def test_ambiguous_product_mention_cannot_replace_product_intent(self) -> None:
        ambiguous_message = "What about slippers?"
        ambiguous_plan = plan_payload(
            revision=1,
            turn=2,
            mutations=[
                replace_product_intent(raw_phrase=ambiguous_message),
                add_constraint("slipper", attribute="category"),
            ],
            tools=["structured"],
        )
        provider = ScriptedProvider([
            plan_payload(
                mutations=[add_constraint("shoe", attribute="category")],
                tools=["structured"],
            ),
            ambiguous_plan,
            dict(ambiguous_plan),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "I need shoes.", 1, 10)

        response = agent.respond("session", ambiguous_message, 2, 10)

        self.assertEqual(len(provider.requests), 3)
        self.assertNotIn({"parent_asin": "C"}, response["recommendations"])
        self.assertEqual(agent.get_constraint_revision("session"), 1)
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [("shoe", "active")],
        )
        diagnostic = agent.get_planning_history("session")[1]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "rejected_state_change")
        self.assertEqual(diagnostic["attempts"], 2)

    def test_connected_plan_cannot_ignore_explicit_product_intent_replacement(self) -> None:
        provider = ScriptedProvider([
            plan_payload(
                mutations=[
                    add_constraint("shoe", attribute="category"),
                    add_constraint(
                        "blue",
                        attribute="color",
                        classification="soft",
                        scope="session",
                    ),
                ],
                tools=["structured"],
            ),
            plan_payload(revision=1, turn=2, tools=["structured"]),
            plan_payload(revision=1, turn=2, tools=["structured"]),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "Whatever I buy, I prefer blue shoes.", 1, 10)

        response = agent.respond(
            "session",
            "Actually, slippers instead of shoes.",
            2,
            10,
        )

        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(response["recommendations"], [{"parent_asin": "C"}])
        self.assertEqual(
            [
                (item["normalized_value"], item["scope"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [
                ("shoe", "product_intent", "superseded"),
                ("blue", "session", "active"),
                ("slipper", "product_intent", "active"),
            ],
        )
        diagnostic = agent.get_planning_history("session")[1]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "rejected_state_change")

    def test_connected_product_intent_replacement_must_match_requested_successor(self) -> None:
        message = "Actually, slippers instead of shoes."
        wrong_successor = plan_payload(
            revision=1,
            turn=2,
            mutations=[
                replace_product_intent(raw_phrase=message),
                add_constraint("boot", attribute="category"),
            ],
            tools=["structured"],
        )
        provider = ScriptedProvider([
            plan_payload(
                mutations=[add_constraint("shoe", attribute="category")],
                tools=["structured"],
            ),
            wrong_successor,
            dict(wrong_successor),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "I need shoes.", 1, 10)

        response = agent.respond("session", message, 2, 10)

        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(response["recommendations"], [{"parent_asin": "C"}])
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [("shoe", "superseded"), ("slipper", "active")],
        )
        diagnostic = agent.get_planning_history("session")[1]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "rejected_state_change")

    def test_connected_product_intent_replacement_preserves_session_constraint(self) -> None:
        message = "Actually, slippers instead of shoes."
        provider = ScriptedProvider([
            plan_payload(
                mutations=[
                    add_constraint("shoe", attribute="category"),
                    add_constraint(
                        "blue",
                        attribute="color",
                        classification="soft",
                        scope="session",
                    ),
                ],
                tools=["structured"],
            ),
            plan_payload(
                revision=1,
                turn=2,
                mutations=[
                    replace_product_intent(raw_phrase=message),
                    add_constraint("slipper", attribute="category"),
                ],
                tools=["structured"],
            ),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "Whatever I buy, I prefer blue shoes.", 1, 10)

        response = agent.respond("session", message, 2, 10)

        self.assertEqual(response["recommendations"], [{"parent_asin": "C"}])
        self.assertEqual(
            [
                (item["normalized_value"], item["scope"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [
                ("shoe", "product_intent", "superseded"),
                ("blue", "session", "active"),
                ("slipper", "product_intent", "active"),
            ],
        )
        self.assertEqual(agent.get_planning_history("session")[1]["source"], "connected")

    def test_connected_attribute_correction_preserves_product_intent(self) -> None:
        message = "Actually, ignore my earlier preference. What I need is cotton."
        provider = ScriptedProvider([
            plan_payload(
                mutations=[
                    add_constraint(
                        "blue",
                        attribute="color",
                        classification="soft",
                    ),
                ],
                tools=["structured"],
            ),
            plan_payload(
                revision=1,
                turn=2,
                mutations=[
                    replace_constraint(
                        "c1",
                        "cotton",
                        attribute="material",
                        raw_phrase=message,
                    ),
                ],
                tools=["structured"],
            ),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "I prefer blue.", 1, 10)
        original_intent = agent.get_constraint_state("session")[0]["product_intent_id"]

        response = agent.respond("session", message, 2, 10)

        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(
            agent.get_constraint_state("session")[1]["product_intent_id"],
            original_intent,
        )
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [("blue", "superseded"), ("cotton", "active")],
        )

    def test_connected_preference_override_must_target_same_attribute(self) -> None:
        message = "Actually, ignore my previous preference. I need cotton."
        wrong_target = plan_payload(
            revision=1,
            turn=2,
            mutations=[
                replace_constraint(
                    "c2",
                    "cotton",
                    attribute="material",
                    raw_phrase=message,
                ),
            ],
            tools=["structured"],
        )
        provider = ScriptedProvider([
            plan_payload(
                mutations=[
                    add_constraint(
                        "leather",
                        attribute="material",
                        classification="soft",
                    ),
                    add_constraint(
                        "blue",
                        attribute="color",
                        classification="soft",
                    ),
                ],
                tools=["structured"],
            ),
            wrong_target,
            dict(wrong_target),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "I prefer leather, but I prefer blue.", 1, 10)

        agent.respond("session", message, 2, 10)

        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [
                ("leather", "superseded"),
                ("blue", "active"),
                ("cotton", "active"),
            ],
        )
        diagnostic = agent.get_planning_history("session")[1]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "rejected_state_change")

    def test_connected_repeated_product_intent_replacements_preserve_audit_history(self) -> None:
        boots_message = "Actually, boots instead of shoes."
        slippers_message = "Actually, slippers instead of boots."
        provider = ScriptedProvider([
            plan_payload(
                mutations=[add_constraint("shoe", attribute="category")],
                tools=["structured"],
            ),
            plan_payload(
                revision=1,
                turn=2,
                mutations=[
                    replace_product_intent(raw_phrase=boots_message),
                    add_constraint("boot", attribute="category"),
                ],
                tools=["structured"],
            ),
            plan_payload(
                revision=2,
                turn=3,
                mutations=[
                    replace_product_intent(
                        product_intent_id="intent-2",
                        raw_phrase=slippers_message,
                    ),
                    add_constraint("slipper", attribute="category"),
                ],
                tools=["structured"],
            ),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})
        agent.respond("session", "I need shoes.", 1, 10)
        boots = agent.respond("session", boots_message, 2, 10)

        slippers = agent.respond("session", slippers_message, 3, 10)

        self.assertEqual(boots["recommendations"], [{"parent_asin": "D"}])
        self.assertEqual(slippers["recommendations"], [{"parent_asin": "C"}])
        self.assertEqual(
            [
                (item["normalized_value"], item["status"])
                for item in agent.get_constraint_state("session")
            ],
            [
                ("shoe", "superseded"),
                ("boot", "superseded"),
                ("slipper", "active"),
            ],
        )
        self.assertEqual(
            [event["from_product_intent_id"] for event in agent.get_transition_history("session")],
            ["intent-1", "intent-2"],
        )

    def test_invalid_schema_is_retried_with_feedback_then_succeeds(self) -> None:
        invalid = plan_payload()
        del invalid["clarification"]
        provider = ScriptedProvider([
            ProviderResponse(invalid, prompt_tokens=5, completion_tokens=2),
            ProviderResponse(
                plan_payload(tools=["bm25"]),
                prompt_tokens=6,
                completion_tokens=3,
            ),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        response = agent.respond("session", "walking shoe", 1, 10)

        self.assertEqual(len(provider.requests), 2)
        self.assertIsNone(provider.requests[0].validation_error)
        self.assertIn("PlanningSchemaError", provider.requests[1].validation_error)
        self.assertEqual(response["usage"], {
            "prompt_tokens": 11,
            "completion_tokens": 5,
        })
        self.assertEqual(agent.get_planning_history("session")[0]["source"], "connected")

    def test_repeated_invalid_schema_takes_over_offline(self) -> None:
        invalid = plan_payload()
        del invalid["clarification"]
        provider = ScriptedProvider([invalid, dict(invalid)])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        response = agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        diagnostic = agent.get_planning_history("session")[0]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "invalid_schema")

    def test_rejected_state_change_is_retried_against_original_snapshot(self) -> None:
        provider = ScriptedProvider([
            plan_payload(mutations=[add_constraint("unobtainium")]),
            plan_payload(mutations=[add_constraint("cotton")]),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.requests[0].state_snapshot["revision"], 0)
        self.assertEqual(provider.requests[1].state_snapshot["revision"], 0)
        self.assertEqual(provider.requests[1].state_snapshot["constraints"], [])
        self.assertIn("PlanningStateError", provider.requests[1].validation_error)
        self.assertEqual(agent.get_constraint_revision("session"), 1)
        self.assertEqual(
            [item["normalized_value"] for item in agent.get_constraint_state("session")],
            ["cotton"],
        )

    def test_repeated_state_rejection_takes_over_without_partial_mutation(self) -> None:
        provider = ScriptedProvider([
            plan_payload(mutations=[add_constraint("unobtainium")]),
            plan_payload(mutations=[add_constraint("unobtainium")]),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(agent.get_constraint_revision("session"), 1)
        self.assertEqual(
            [item["normalized_value"] for item in agent.get_constraint_state("session")],
            ["cotton"],
        )
        diagnostic = agent.get_planning_history("session")[0]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "rejected_state_change")

    def test_timeouts_retry_once_then_take_over(self) -> None:
        provider = ScriptedProvider([
            TimeoutError("first deadline"),
            TimeoutError("second deadline"),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        response = agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        diagnostic = agent.get_planning_history("session")[0]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "timeout")

    def test_missing_credentials_take_over_without_pointless_retry(self) -> None:
        provider = ScriptedProvider([
            MissingCredentialsError("OPENAI_API_KEY is not configured"),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        response = agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        diagnostic = agent.get_planning_history("session")[0]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "missing_credentials")
        self.assertEqual(diagnostic["attempts"], 1)

    def test_each_call_receives_local_state_and_recent_history(self) -> None:
        provider = CallbackProvider()
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        agent.respond("session", "I need cotton.", 1, 10)
        response = agent.respond("session", "Show me options.", 2, 10)

        self.assertEqual(len(provider.requests), 2)
        second = provider.requests[1]
        self.assertEqual(second.state_snapshot["revision"], 1)
        self.assertEqual(
            second.state_snapshot["constraints"][0]["normalized_value"],
            "cotton",
        )
        self.assertEqual(len(second.recent_history), 1)
        self.assertEqual(second.recent_history[0]["user_message"], "I need cotton.")
        self.assertEqual(second.prompt_version, "shopping-turn-planner-v2")
        self.assertIn("Local Constraint State is", second.instructions)
        self.assertFalse(hasattr(second, "conversation_id"))
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])


if __name__ == "__main__":
    unittest.main()
