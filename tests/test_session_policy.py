from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.reranker import UnavailableReranker
from starter.agent import Agent
from starter.constraint_state import AddConstraint, ConstraintState, TurnPlan
from starter.planning import PlanningRequest
from starter.session_policy import (
    ALLOWED_ASK_ATTRIBUTES,
    clarification_candidates,
)


def plan_payload(
    *,
    revision: int,
    turn: int,
    session_mode: str,
    clarification: dict | None,
) -> dict:
    return {
        "expected_state_revision": revision,
        "source_turn": turn,
        "mutations": [],
        "retrieval_tools": ["structured", "bm25", "dense"],
        "session_mode": session_mode,
        "clarification": clarification,
    }


class ScriptedProvider:
    def __init__(self, outcomes: list[dict]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[PlanningRequest] = []

    def plan(self, request: PlanningRequest) -> dict:
        self.requests.append(request)
        return self.outcomes.pop(0)


class SessionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        self.catalog_path.write_text("".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "parent_asin": "A",
                    "title": "Blue cotton running shoe",
                    "features": ["waterproof", "lightweight"],
                },
                {
                    "parent_asin": "B",
                    "title": "Red leather hiking shoe",
                    "features": ["insulated", "pockets"],
                },
                {
                    "parent_asin": "C",
                    "title": "Black wool winter boot",
                    "features": ["waterproof", "zipper"],
                },
                {
                    "parent_asin": "D",
                    "title": "White nylon outdoor boot",
                    "features": ["breathable", "stretch"],
                },
            )
        ), encoding="utf-8")
        reranker_patch = patch(
            "starter.agent.build_live_reranker",
            return_value=UnavailableReranker("disabled_for_session_policy_test"),
        )
        reranker_patch.start()
        self.addCleanup(reranker_patch.stop)

    def test_session_mode_is_revised_on_every_turn_as_evidence_changes(self) -> None:
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        browsing = agent.respond(
            "session", "I'm looking for shoes, but I'm still exploring.", 1, 10,
        )
        buying = agent.respond("session", "I need cotton.", 2, 10)
        uncertain = agent.respond(
            "session", "I'm not sure what I need yet.", 3, 10,
        )

        self.assertTrue(browsing["recommendations"])
        self.assertTrue(buying["recommendations"])
        self.assertTrue(uncertain["recommendations"])
        self.assertEqual(
            [item["to"] for item in agent.get_session_mode_history("session")],
            ["browsing", "buying", "uncertain"],
        )
        self.assertEqual(agent.get_session_mode("session"), "uncertain")

    def test_clarifications_are_allowed_useful_ranked_and_not_repeated(self) -> None:
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        responses = [
            agent.respond(
                "session", "I'm looking for shoes, but I'm still exploring.", 1, 10,
            ),
            agent.respond("session", "Show me some options.", 2, 10),
            agent.respond("session", "Show me more options.", 3, 10),
        ]

        asked = [response["ask_attribute"] for response in responses]
        self.assertEqual(len(asked), len(set(asked)))
        for response in responses:
            self.assertIn(response["ask_attribute"], ALLOWED_ASK_ATTRIBUTES)
            self.assertTrue(response["recommendations"])
            self.assertIn("preference", response["message"].lower())

    def test_boundary_attribute_is_never_asked_again(self) -> None:
        agent = Agent(self.catalog_path)
        agent.reset("session", {})

        first = agent.respond("session", "I need shoes.", 1, 10)
        dismissed = first["ask_attribute"]
        boundary = agent.respond(
            "session",
            f"I don't have a preference for {dismissed}; please use your judgment.",
            2,
            10,
        )
        later = agent.respond("session", "Show me more.", 3, 10)

        self.assertEqual(dismissed, "material")
        self.assertNotEqual(boundary["ask_attribute"], dismissed)
        self.assertNotEqual(later["ask_attribute"], dismissed)
        self.assertEqual(
            {item["attribute"] for item in agent.get_dismissed_attributes("session")},
            {dismissed},
        )

    def test_connected_planner_retries_a_repeated_low_value_question(self) -> None:
        provider = ScriptedProvider([
            plan_payload(
                revision=0,
                turn=1,
                session_mode="uncertain",
                clarification={
                    "ask_attribute": "material",
                    "message": "Which material do you prefer?",
                },
            ),
            plan_payload(
                revision=0,
                turn=2,
                session_mode="uncertain",
                clarification={
                    "ask_attribute": "material",
                    "message": "Which material do you prefer?",
                },
            ),
            plan_payload(
                revision=0,
                turn=2,
                session_mode="uncertain",
                clarification={
                    "ask_attribute": "color",
                    "message": "Which color do you prefer?",
                },
            ),
        ])
        agent = Agent(self.catalog_path, planning_provider=provider)
        agent.reset("session", {})

        first = agent.respond("session", "Show me a shoe.", 1, 10)
        second = agent.respond("session", "Show me more shoe options.", 2, 10)

        self.assertEqual(first["ask_attribute"], "material")
        self.assertEqual(second["ask_attribute"], "color")
        self.assertEqual(len(provider.requests), 3)
        self.assertIn("previously asked", provider.requests[-1].validation_error)

    def test_profile_hints_only_tiebreak_eligible_questions(self) -> None:
        state = ConstraintState()
        state.apply(TurnPlan(
            expected_state_revision=0,
            source_turn=1,
            mutations=(AddConstraint(
                attribute="category",
                values=("shoe",),
                match_rule="all",
                classification="hard",
                scope="product_intent",
                raw_phrase="shoes",
                confidence=0.95,
            ),),
        ), {
            "category": {"shoe", "boot"},
            "material": {"cotton", "leather"},
            "color": {"blue", "red"},
        })
        supported = {
            "category": {"shoe", "boot"},
            "material": {"cotton", "leather"},
            "color": {"blue", "red"},
        }

        ordinary = clarification_candidates("buying", state, supported)
        personalized = clarification_candidates(
            "buying", state, supported, profile_hints=("color",),
        )

        self.assertEqual(ordinary[0].ask_attribute, "material")
        self.assertEqual(personalized[0].ask_attribute, "color")

    def test_explicit_current_requirement_outranks_aggregate_profile(self) -> None:
        agent = Agent(self.catalog_path)
        agent.reset("session", {
            "preference_tags": ["material"],
            "summary": "Prior purchases often involved cotton.",
        })

        response = agent.respond(
            "session", "I need a red leather hiking shoe.", 1, 10,
        )

        self.assertEqual(response["recommendations"], [{"parent_asin": "B"}])
        active = {
            item["attribute"]: item["normalized_value"]
            for item in agent.get_constraint_state("session")
            if item["status"] == "active"
        }
        self.assertEqual(active["material"], "leather")
        self.assertEqual(active["color"], "red")

    def test_scenario_fixtures_have_distinct_deterministic_allowed_behavior(self) -> None:
        def run_fixture(name: str) -> tuple:
            agent = Agent(self.catalog_path)
            agent.reset(name, {})
            if name == "buying":
                response = agent.respond(name, "I need shoes.", 1, 10)
            elif name == "browsing":
                response = agent.respond(
                    name, "I'm looking for shoes, but I'm still exploring.", 1, 10,
                )
            elif name == "intent_override":
                agent.respond(name, "I'm looking at shoes.", 1, 10)
                response = agent.respond(
                    name, "Actually, I need boots instead of shoes.", 2, 10,
                )
            else:
                first = agent.respond(
                    name, "I'm looking for shoes, but I'm still exploring.", 1, 10,
                )
                agent.respond(
                    name,
                    f"I don't have a preference for {first['ask_attribute']}.",
                    2,
                    10,
                )
                response = agent.respond(name, "Show me more options.", 3, 10)
            return (
                agent.get_session_mode(name),
                response["ask_attribute"],
                tuple(item["attribute"] for item in agent.get_dismissed_attributes(name)),
                agent.get_constraint_state(name)[-1]["product_intent_id"],
            )

        first_run = {
            name: run_fixture(name)
            for name in ("buying", "browsing", "intent_override", "boundary")
        }
        second_run = {
            name: run_fixture(name)
            for name in ("buying", "browsing", "intent_override", "boundary")
        }

        self.assertEqual(first_run, second_run)
        self.assertEqual(first_run["buying"][:2], ("buying", "material"))
        self.assertEqual(first_run["browsing"][:2], ("browsing", "use_case"))
        self.assertEqual(first_run["intent_override"][3], "intent-2")
        self.assertEqual(first_run["boundary"][2], ("use_case",))
        self.assertEqual(len(set(first_run.values())), 4)


if __name__ == "__main__":
    unittest.main()
