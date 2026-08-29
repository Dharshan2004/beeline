from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
) -> dict:
    return {
        "type": "add_constraint",
        "attribute": attribute,
        "values": [value],
        "match_rule": "all",
        "classification": classification,
        "scope": "product_intent",
        "raw_phrase": value,
        "confidence": 0.97,
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

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
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
                    tools=["structured", "local_rerank"],
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
            "retrieval_tools": ["structured", "local_rerank"],
            "ask_attribute": "color",
            "fallback_reason": None,
            "errors": [],
        })

    def test_shell_web_code_and_catalog_mutation_tools_are_never_allowed(self) -> None:
        supported_values = {
            "material": {"cotton"},
            "color": {"blue"},
        }
        for forbidden_tool in ("shell", "web", "python", "catalog_mutation"):
            with self.subTest(tool=forbidden_tool):
                with self.assertRaisesRegex(
                    PlanningToolError,
                    "unapproved retrieval tools",
                ):
                    decode_plan(
                        plan_payload(tools=[forbidden_tool]),
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
        self.assertEqual(second.prompt_version, "shopping-turn-planner-v1")
        self.assertIn("Local Constraint State is", second.instructions)
        self.assertFalse(hasattr(second, "conversation_id"))
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])


if __name__ == "__main__":
    unittest.main()
