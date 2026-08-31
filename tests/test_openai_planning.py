from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from starter.openai_planning import (
    BudgetExceededError,
    DevelopmentBudget,
    ModelPricing,
    OpenAIPlanningProvider,
)
from starter.agent import Agent
from retrieval.reranker import UnavailableReranker
from starter.planning import (
    MissingCredentialsError,
    PlanningRequest,
    PlanningSchemaError,
)


def planning_request() -> PlanningRequest:
    return PlanningRequest(
        session_id="session-a",
        turn=2,
        user_message="Actually, I need blue boots instead.",
        state_snapshot={"revision": 1, "constraints": []},
        recent_history=({"turn": 1, "user_message": "I need shoes."},),
        supported_values={"category": ("boot", "shoe"), "color": ("blue",)},
        allowed_tools=("structured", "bm25", "dense"),
        prompt_version="shopping-turn-planner-v2",
        instructions="Return the validated Turn Plan.",
        response_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        },
        validation_error="previous output was invalid",
    )


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float):
        self.timeouts.append(timeout)
        return self


class APITimeoutError(Exception):
    pass


class EmptyDenseRoute:
    configured = False

    def search(self, query: str, limit: int):
        return []

    def metrics(self) -> dict:
        return {"status": "disabled"}


class OpenAIPlanningProviderTest(unittest.TestCase):
    def test_missing_credentials_fail_before_client_construction(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                MissingCredentialsError,
                "OPENAI_API_KEY",
            ):
                OpenAIPlanningProvider(model="gpt-test")

    def test_plan_sends_a_stateless_strict_structured_request(self) -> None:
        response = SimpleNamespace(
            output_text=json.dumps({"value": "ok"}),
            usage=SimpleNamespace(input_tokens=123, output_tokens=17),
        )
        client = FakeClient(response)
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            timeout_seconds=4.5,
            max_output_tokens=900,
            reasoning_effort="low",
            pricing=ModelPricing(input_per_million_usd=1, output_per_million_usd=2),
            budget=DevelopmentBudget(limit_usd=1),
        )

        result = provider.plan(planning_request())

        self.assertEqual(result.output, {"value": "ok"})
        self.assertEqual(result.prompt_tokens, 123)
        self.assertEqual(result.completion_tokens, 17)
        self.assertEqual(client.timeouts, [4.5])
        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-test")
        self.assertEqual(call["instructions"], "Return the validated Turn Plan.")
        self.assertEqual(call["store"], False)
        self.assertEqual(call["tools"], [])
        self.assertEqual(call["max_output_tokens"], 900)
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(call["text"]["format"], {
            "type": "json_schema",
            "name": "shopping_turn_plan",
            "schema": planning_request().response_schema,
            "strict": True,
        })
        payload = json.loads(call["input"])
        self.assertEqual(payload["prompt_version"], "shopping-turn-planner-v2")
        self.assertEqual(payload["allowed_tools"], ["structured", "bm25", "dense"])
        self.assertEqual(payload["validation_error"], "previous output was invalid")
        self.assertNotIn("api_key", payload)
        self.assertAlmostEqual(provider.budget.spent_usd, 0.000157)

    def test_transport_schema_converts_one_of_without_mutating_contract(self) -> None:
        request = replace(
            planning_request(),
            response_schema={
                "type": "object",
                "properties": {
                    "kind": {"const": "example"},
                    "mode": {"enum": ["one", "two"]},
                    "value": {
                        "oneOf": [
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                            {"type": "null"},
                        ],
                    },
                },
                "required": ["kind", "mode", "value"],
                "additionalProperties": False,
            },
        )
        original_schema = json.loads(json.dumps(request.response_schema))
        client = FakeClient(SimpleNamespace(
            output_text='{"kind":"example","mode":"one","value":null}',
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        ))
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            budget=DevelopmentBudget(limit_usd=1),
        )

        provider.plan(request)

        transport_schema = client.responses.calls[0]["text"]["format"]["schema"]
        self.assertNotIn("oneOf", transport_schema["properties"]["value"])
        self.assertIn("anyOf", transport_schema["properties"]["value"])
        self.assertNotIn(
            "uniqueItems",
            transport_schema["properties"]["value"]["anyOf"][0],
        )
        self.assertEqual(transport_schema["properties"]["kind"]["type"], "string")
        self.assertEqual(transport_schema["properties"]["mode"]["type"], "string")
        self.assertEqual(request.response_schema, original_schema)

    def test_invalid_json_is_rejected_at_the_adapter_boundary(self) -> None:
        client = FakeClient(SimpleNamespace(
            output_text="not json",
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        ))
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            pricing=ModelPricing(input_per_million_usd=1, output_per_million_usd=2),
            budget=DevelopmentBudget(limit_usd=1),
        )

        with self.assertRaisesRegex(PlanningSchemaError, "valid JSON"):
            provider.plan(planning_request())
        self.assertEqual(provider.budget.spent_usd, 0.000007)

    def test_incomplete_response_is_rejected_after_usage_is_counted(self) -> None:
        client = FakeClient(SimpleNamespace(
            status="incomplete",
            output_text='{"value":"partial"}',
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        ))
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            pricing=ModelPricing(input_per_million_usd=1, output_per_million_usd=2),
            budget=DevelopmentBudget(limit_usd=1),
        )

        with self.assertRaisesRegex(PlanningSchemaError, "incomplete"):
            provider.plan(planning_request())
        self.assertEqual(provider.budget.spent_usd, 0.000007)

    def test_provider_timeout_maps_to_the_planning_timeout_contract(self) -> None:
        client = FakeClient(TimeoutError("request expired"))
        budget = DevelopmentBudget(limit_usd=1)
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            max_input_tokens_estimate=100,
            max_output_tokens=100,
            pricing=ModelPricing(input_per_million_usd=1, output_per_million_usd=1),
            budget=budget,
        )

        with self.assertRaisesRegex(TimeoutError, "OpenAI planning request"):
            provider.plan(planning_request())
        self.assertGreater(budget.spent_usd, 0)
        self.assertEqual(provider.metrics()["submitted_call_count"], 1)
        self.assertEqual(provider.metrics()["pessimistically_accounted_call_count"], 1)

    def test_sdk_timeout_maps_to_the_planning_timeout_contract(self) -> None:
        client = FakeClient(APITimeoutError("SDK deadline"))
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
        )

        with self.assertRaisesRegex(TimeoutError, "OpenAI planning request"):
            provider.plan(planning_request())

    def test_runtime_configuration_records_model_pricing_and_budget(self) -> None:
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=FakeClient(SimpleNamespace()),
            timeout_seconds=7,
            max_output_tokens=800,
            reasoning_effort="medium",
            pricing=ModelPricing(input_per_million_usd=3, output_per_million_usd=9),
            budget=DevelopmentBudget(limit_usd=10),
        )
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                json.dumps({"parent_asin": "A", "title": "Product"}) + "\n",
                encoding="utf-8",
            )
            agent = Agent(
                catalog,
                dense_route=EmptyDenseRoute(),
                planning_provider=provider,
                reranker=UnavailableReranker("disabled_for_test"),
            )

            configuration = agent.get_runtime_configuration()

        self.assertEqual(configuration["planning"]["provider"], "OpenAIPlanningProvider")
        self.assertEqual(configuration["planning"]["connected_model_version"], "gpt-test")
        self.assertEqual(configuration["planning"]["provider_configuration"], {
            "api": "responses-v1",
            "model": "gpt-test",
            "reasoning_effort": "medium",
            "timeout_seconds": 7.0,
            "max_output_tokens": 800,
            "structured_outputs": True,
            "transport_schema": {
                "version": "openai-anyof-v1",
                "sha256": "fc76794ea220f56a227a92fe63e7b26b1ebeed329b5de56bcfd1dc27773bf46f",
            },
            "store": False,
            "pricing_usd_per_million_tokens": {"input": 3, "output": 9},
            "budget": {
                "spent_usd": 0.0,
                "limit_usd": 10,
                "warning_usd": 40.0,
                "review_boundary_usd": 50.0,
                "absolute_stop_usd": 600.0,
                "warning_reached": False,
                "review_approved": False,
            },
        })
        self.assertEqual(
            configuration["cost_limits_usd"]["enforcement_status"],
            "enforced_by_connected_provider",
        )

    def test_budget_stops_before_a_call_that_could_cross_the_limit(self) -> None:
        budget = DevelopmentBudget(limit_usd=0.0001)
        client = FakeClient(SimpleNamespace(
            output_text='{"value":"ok"}',
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        ))
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            max_output_tokens=1000,
            pricing=ModelPricing(input_per_million_usd=1, output_per_million_usd=1),
            budget=budget,
        )

        with self.assertRaisesRegex(BudgetExceededError, "budget"):
            provider.plan(planning_request())
        self.assertEqual(client.responses.calls, [])
        self.assertEqual(budget.spent_usd, 0)

    def test_budget_requires_explicit_review_approval_above_fifty_dollars(self) -> None:
        with self.assertRaisesRegex(ValueError, "review boundary"):
            DevelopmentBudget(limit_usd=51)

        approved = DevelopmentBudget(limit_usd=51, review_approved=True)
        self.assertEqual(approved.limit_usd, 51)

    def test_budget_exhaustion_takes_over_offline_without_retrying(self) -> None:
        client = FakeClient(SimpleNamespace(
            output_text='{"value":"unused"}',
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        ))
        provider = OpenAIPlanningProvider(
            model="gpt-test",
            api_key="secret",
            client=client,
            pricing=ModelPricing(input_per_million_usd=1, output_per_million_usd=1),
            budget=DevelopmentBudget(limit_usd=0.0001),
        )
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                json.dumps({
                    "parent_asin": "A",
                    "title": "Cotton walking shoe",
                }) + "\n",
                encoding="utf-8",
            )
            agent = Agent(
                catalog,
                dense_route=EmptyDenseRoute(),
                planning_provider=provider,
                reranker=UnavailableReranker("disabled_for_test"),
            )
            agent.reset("session", {})

            response = agent.respond("session", "I need cotton.", 1, 10)

        self.assertEqual(client.responses.calls, [])
        self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
        diagnostic = agent.get_planning_history("session")[0]
        self.assertEqual(diagnostic["source"], "fallback")
        self.assertEqual(diagnostic["fallback_reason"], "budget_exhausted")
        self.assertEqual(diagnostic["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
