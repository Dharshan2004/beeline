from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent
from starter.telemetry import (
    LangfuseSink,
    MemorySink,
    NullSink,
    Tracer,
    is_denied_key,
    sanitize,
)


CATALOG = [
    {
        "parent_asin": "B001",
        "title": "Blue cotton running shoe",
        "features": ["cotton upper", "blue"],
        "categories": ["Shoes"],
    },
    {
        "parent_asin": "B002",
        "title": "Black leather boot",
        "features": ["leather", "black"],
        "categories": ["Shoes"],
    },
    {
        "parent_asin": "B003",
        "title": "Blue nylon sandal",
        "features": ["nylon", "blue"],
        "categories": ["Shoes"],
    },
]
USER_PROFILE = {
    "customer_id": "hidden-customer-9",
    "purchase_history": ["B002"],
    "api_key": "lf-secret-value",
}


class FailingSink:
    """Sink that reproduces a telemetry outage at export time."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.export_calls = 0

    def export(self, traces: list[dict]) -> None:
        self.export_calls += 1
        raise self.error

    def flush(self) -> None:
        return None

    def metrics(self) -> dict:
        return {"status": "available", "disabled_reason": None}


class FakeSpan:
    def __init__(self, name, metadata, level, status_message) -> None:
        self.name = name
        self.metadata = metadata
        self.level = level
        self.status_message = status_message
        self.children: list["FakeSpan"] = []
        self.ended = False

    def span(self, name, metadata, level, status_message) -> "FakeSpan":
        child = FakeSpan(name, metadata, level, status_message)
        self.children.append(child)
        return child

    def end(self) -> None:
        self.ended = True


class FakeTrace(FakeSpan):
    def __init__(self, name, session_id, metadata, tags) -> None:
        super().__init__(name, metadata, "DEFAULT", None)
        self.session_id = session_id
        self.tags = tags


class FakeLangfuseClient:
    def __init__(self) -> None:
        self.traces: list[FakeTrace] = []
        self.flush_count = 0

    def trace(self, name, session_id, metadata, tags) -> FakeTrace:
        created = FakeTrace(name, session_id, metadata, tags)
        self.traces.append(created)
        return created

    def flush(self) -> None:
        self.flush_count += 1


class TelemetryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.catalog_path = Path(self.directory.name) / "catalog.jsonl"
        self.catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in CATALOG),
            encoding="utf-8",
        )

    class DisabledDenseRoute:
        def search(self, query: str, limit: int) -> list[tuple[str, float]]:
            return []

        def metrics(self) -> dict:
            return {
                "status": "disabled",
                "disabled_reason": "FileNotFoundError: index missing",
            }

    def build_agent(self, tracer: Tracer) -> Agent:
        return Agent(
            self.catalog_path,
            dense_route=self.DisabledDenseRoute(),
            tracer=tracer,
        )

    def traced_agent(self) -> tuple[Agent, MemorySink, Tracer]:
        sink = MemorySink()
        tracer = Tracer(sink, register_atexit=False)
        return self.build_agent(tracer), sink, tracer

    @staticmethod
    def observation(trace: dict, name: str) -> dict:
        def walk(observations):
            for observation in observations:
                if observation["name"] == name:
                    return observation
                found = walk(observation["observations"])
                if found is not None:
                    return found
            return None

        result = walk(trace["observations"])
        assert result is not None, f"missing observation: {name}"
        return result


class TraceContentTest(TelemetryTestCase):
    def test_turns_are_grouped_by_session_with_nested_operation_timing(self) -> None:
        agent, sink, tracer = self.traced_agent()
        agent.reset("session-a", USER_PROFILE)
        agent.reset("session-b", USER_PROFILE)
        agent.respond("session-a", "blue cotton running shoe", 1, 10)
        agent.respond("session-b", "black leather boot", 1, 10)
        agent.respond("session-a", "I prefer cotton", 2, 10)
        tracer.flush()

        grouped: dict[str, list[int]] = {}
        for trace in sink.traces:
            grouped.setdefault(trace["session_id"], []).append(trace["turn"])
        self.assertEqual(grouped, {"session-a": [1, 2], "session-b": [1]})

        first = sink.traces[0]
        self.assertEqual(first["name"], "shopping-turn")
        self.assertGreaterEqual(first["latency_ms"], 0.0)
        names = [observation["name"] for observation in first["observations"]]
        self.assertEqual(
            names,
            [
                "interpretation",
                "state_validation",
                "retrieval",
                "fusion",
                "reranking",
                "clarification",
                "response",
            ],
        )
        retrieval = self.observation(first, "retrieval")
        self.assertEqual(
            [child["name"] for child in retrieval["observations"]],
            ["retrieval.dense"],
        )
        for observation in first["observations"]:
            self.assertGreaterEqual(observation["latency_ms"], 0.0)
            self.assertLessEqual(observation["latency_ms"], first["latency_ms"] + 1.0)

    def test_configuration_identity_accompanies_every_turn(self) -> None:
        agent, sink, tracer = self.traced_agent()
        agent.reset("session", USER_PROFILE)
        agent.respond("session", "blue cotton running shoe", 1, 10)
        tracer.flush()

        configuration = sink.traces[0]["metadata"]["configuration"]
        self.assertEqual(
            configuration["retrieval"]["policy_version"],
            agent.fusion_policy.version,
        )
        self.assertEqual(configuration["retrieval"]["route_depth"], 100)
        self.assertEqual(
            configuration["planning_prompt_version"],
            "shopping-turn-planner-v1",
        )
        self.assertEqual(configuration["planning_source"], "local")
        self.assertEqual(configuration["dense_route_status"], "disabled")
        self.assertEqual(configuration["catalog_product_count"], len(CATALOG))

    def test_traces_record_decisions_counts_usage_and_reason_codes(self) -> None:
        agent, sink, tracer = self.traced_agent()
        agent.reset("session", USER_PROFILE)
        agent.respond("session", "blue cotton running shoe", 1, 10)
        tracer.flush()

        trace = sink.traces[0]
        interpretation = self.observation(trace, "interpretation")["metadata"]
        self.assertEqual(interpretation["source"], "fallback")
        self.assertEqual(interpretation["fallback_reason"], "missing_credentials")
        self.assertEqual(interpretation["reason_codes"], [])
        self.assertEqual(
            interpretation["retrieval_tools"],
            ["structured", "bm25", "dense"],
        )

        validation = self.observation(trace, "state_validation")["metadata"]
        self.assertEqual(validation["applied_plan"], "turn_plan")
        self.assertEqual(validation["revision_before"], 0)
        self.assertGreaterEqual(validation["revision_after"], 1)
        decisions = validation["constraint_decisions"]
        self.assertTrue(decisions)
        for decision in decisions:
            self.assertIn(decision["classification"], ("hard", "soft"))
            self.assertGreaterEqual(decision["confidence"], 0.0)
            self.assertLessEqual(decision["confidence"], 1.0)

        retrieval = self.observation(trace, "retrieval")["metadata"]
        self.assertEqual(
            sorted(retrieval["candidate_counts"]),
            ["bm25", "dense", "structured"],
        )
        self.assertEqual(retrieval["candidate_counts"]["dense"], 0)

        dense = self.observation(trace, "retrieval.dense")["metadata"]
        self.assertTrue(dense["requested"])
        self.assertEqual(dense["route_status"], "disabled")

        fusion = self.observation(trace, "fusion")["metadata"]
        self.assertEqual(fusion["policy_version"], agent.fusion_policy.version)
        self.assertGreaterEqual(fusion["fused_candidate_count"], 1)

        response = self.observation(trace, "response")["metadata"]
        self.assertEqual(response["requested_top_k"], 10)
        self.assertEqual(
            response["usage"],
            {"prompt_tokens": 0, "completion_tokens": 0},
        )
        self.assertEqual(trace["metadata"]["source"], "fallback")

    def test_failure_causes_are_classified_without_changing_the_response(self) -> None:
        class ExplodingDenseRoute:
            def search(self, query: str, limit: int) -> list[tuple[str, float]]:
                raise ConnectionRefusedError("dense route unavailable")

            def metrics(self) -> dict:
                return {"status": "disabled", "disabled_reason": "refused"}

        sink = MemorySink()
        tracer = Tracer(sink, register_atexit=False)
        agent = Agent(
            self.catalog_path,
            dense_route=ExplodingDenseRoute(),
            tracer=tracer,
        )
        agent.reset("session", USER_PROFILE)
        response = agent.respond("session", "blue cotton running shoe", 1, 10)
        tracer.flush()

        self.assertTrue(response["recommendations"])
        dense = self.observation(sink.traces[0], "retrieval.dense")
        self.assertEqual(dense["status"], "error")
        self.assertEqual(dense["failure_cause"], "ConnectionRefusedError")


class TraceRedactionTest(TelemetryTestCase):
    def test_denied_keys_cover_secrets_profiles_and_private_reasoning(self) -> None:
        for key in (
            "api_key",
            "Secret_Key",
            "user_profile",
            "raw_phrase",
            "chain_of_thought",
            "authorization",
            "openai_api_key",
        ):
            self.assertTrue(is_denied_key(key), key)
        for key in ("prompt_tokens", "completion_tokens", "candidate_counts"):
            self.assertFalse(is_denied_key(key), key)

    def test_sanitize_drops_denied_keys_and_bounds_payload_size(self) -> None:
        payload = sanitize({
            "api_key": "lf-secret",
            "nested": {"raw_phrase": "I want blue", "count": 2},
            "usage": {"prompt_tokens": 4},
            "long": "x" * 500,
            "wide": list(range(50)),
            "unsupported": object(),
        })
        self.assertNotIn("api_key", payload)
        self.assertEqual(payload["nested"], {"count": 2})
        self.assertEqual(payload["usage"], {"prompt_tokens": 4})
        self.assertEqual(len(payload["long"]), 200)
        self.assertEqual(len(payload["wide"]), 20)
        self.assertIsNone(payload["unsupported"])

    def test_exported_traces_exclude_profiles_messages_and_catalog_records(
        self,
    ) -> None:
        agent, sink, tracer = self.traced_agent()
        agent.reset("session", USER_PROFILE)
        agent.respond("session", "blue cotton running shoe for hiking", 1, 10)
        agent.respond("session", "actually I want a black leather boot", 2, 10)
        tracer.flush()

        exported = json.dumps(sink.traces)
        for secret in (
            "lf-secret-value",
            "hidden-customer-9",
            "purchase_history",
            "for hiking",
            "actually I want",
        ):
            self.assertNotIn(secret, exported)
        for catalog_field in ("cotton upper", "Blue cotton running shoe"):
            self.assertNotIn(catalog_field, exported)
        # Structured operational evidence is still present.
        self.assertIn("candidate_counts", exported)
        self.assertIn("fused_candidate_count", exported)


class FailOpenTest(TelemetryTestCase):
    def baseline_transcript(self, agent: Agent, session_id: str) -> list[dict]:
        agent.reset(session_id, USER_PROFILE)
        return [
            agent.respond(session_id, "blue cotton running shoe", 1, 10),
            agent.respond(session_id, "I don't have a preference for size", 2, 10),
            agent.respond(session_id, "actually I want leather", 3, 10),
        ]

    def test_missing_credentials_disable_telemetry_without_changing_responses(
        self,
    ) -> None:
        disabled = Tracer.from_environment({}, register_atexit=False)
        self.assertFalse(disabled.enabled)
        self.assertEqual(
            disabled.metrics()["sink"]["disabled_reason"],
            "missing_credentials",
        )

        traced_agent, _, tracer = self.traced_agent()
        untraced_agent = self.build_agent(disabled)
        self.assertEqual(
            self.baseline_transcript(traced_agent, "session"),
            self.baseline_transcript(untraced_agent, "session"),
        )
        self.assertFalse(disabled.flush())
        self.assertEqual(disabled.metrics()["submitted_traces"], 0)

    def test_switched_off_telemetry_is_disabled_even_with_credentials(self) -> None:
        tracer = Tracer.from_environment(
            {
                "SHOPPING_AGENT_TELEMETRY": "0",
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk",
            },
            register_atexit=False,
        )
        self.assertFalse(tracer.enabled)
        self.assertEqual(
            tracer.metrics()["sink"]["disabled_reason"],
            "telemetry_switched_off",
        )

    def test_connection_refusal_and_timeout_do_not_change_responses(self) -> None:
        reference_agent, _, _ = self.traced_agent()
        expected = self.baseline_transcript(reference_agent, "session")
        for error in (
            ConnectionRefusedError("langfuse refused the connection"),
            TimeoutError("langfuse export timed out"),
            RuntimeError("langfuse queue is full"),
        ):
            with self.subTest(error=type(error).__name__):
                sink = FailingSink(error)
                tracer = Tracer(sink, register_atexit=False)
                agent = self.build_agent(tracer)
                self.assertEqual(
                    self.baseline_transcript(agent, "session"),
                    expected,
                )
                self.assertFalse(tracer.flush())
                metrics = tracer.metrics()
                self.assertEqual(metrics["export_failures"], 1)
                self.assertEqual(metrics["last_failure_cause"], type(error).__name__)
                self.assertEqual(metrics["exported_traces"], 0)

    def test_full_buffer_drops_oldest_traces_without_failing_a_turn(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink, buffer_limit=2, register_atexit=False)
        agent = self.build_agent(tracer)
        self.baseline_transcript(agent, "session")
        tracer.flush()

        self.assertEqual(len(sink.traces), 2)
        self.assertEqual([trace["turn"] for trace in sink.traces], [2, 3])
        metrics = tracer.metrics()
        self.assertEqual(metrics["submitted_traces"], 3)
        self.assertEqual(metrics["dropped_traces"], 1)

    def test_export_failure_clears_the_buffer_and_later_flushes_succeed(self) -> None:
        sink = FailingSink(ConnectionRefusedError("refused"))
        tracer = Tracer(sink, register_atexit=False)
        agent = self.build_agent(tracer)
        self.baseline_transcript(agent, "session")
        self.assertFalse(tracer.flush())
        self.assertEqual(tracer.metrics()["buffered_traces"], 0)
        self.assertTrue(tracer.flush())
        self.assertEqual(sink.export_calls, 1)

    def test_offline_agent_traces_without_network_access(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink, register_atexit=False)
        agent = self.build_agent(tracer)
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
            responses = self.baseline_transcript(agent, "session")
            tracer.flush()
        self.assertTrue(responses[0]["recommendations"])
        for item in responses:
            self.assertIsInstance(item["message"], str)
            self.assertIsInstance(item["recommendations"], list)
        self.assertEqual(len(sink.traces), 3)


class FlushBoundaryTest(TelemetryTestCase):
    def test_turns_buffer_traces_and_export_only_happens_on_flush(self) -> None:
        agent, sink, tracer = self.traced_agent()
        agent.reset("session", USER_PROFILE)
        agent.respond("session", "blue cotton running shoe", 1, 10)
        agent.respond("session", "I prefer leather", 2, 10)

        self.assertEqual(sink.traces, [])
        self.assertEqual(sink.flush_count, 0)
        self.assertEqual(tracer.metrics()["buffered_traces"], 2)

        self.assertTrue(agent.flush_telemetry())
        self.assertEqual(len(sink.traces), 2)
        self.assertEqual(sink.flush_count, 1)
        self.assertEqual(agent.get_telemetry_metrics()["buffered_traces"], 0)
        self.assertEqual(agent.get_telemetry_metrics()["exported_traces"], 2)

    def test_enabled_tracer_registers_a_process_completion_flush(self) -> None:
        registered: list[object] = []
        with patch("atexit.register", side_effect=registered.append):
            tracer = Tracer(MemorySink())
        self.assertTrue(tracer.metrics()["atexit_registered"])
        self.assertEqual(registered, [tracer.flush])

    def test_disabled_tracer_registers_nothing(self) -> None:
        registered: list[object] = []
        with patch("atexit.register", side_effect=registered.append):
            tracer = Tracer(NullSink("missing_credentials"))
        self.assertEqual(registered, [])
        self.assertFalse(tracer.metrics()["atexit_registered"])


class LangfuseSinkTest(TelemetryTestCase):
    def test_client_is_built_lazily_and_traces_keep_session_grouping(self) -> None:
        client = FakeLangfuseClient()
        sink = LangfuseSink(client_factory=lambda: client)
        tracer = Tracer(sink, register_atexit=False)
        agent = self.build_agent(tracer)
        agent.reset("session", USER_PROFILE)
        agent.respond("session", "blue cotton running shoe", 1, 10)

        self.assertEqual(client.traces, [])
        self.assertEqual(sink.status, "pending")

        self.assertTrue(tracer.flush())
        self.assertEqual(sink.status, "available")
        self.assertEqual(len(client.traces), 1)
        exported = client.traces[0]
        self.assertEqual(exported.session_id, "session")
        self.assertEqual(exported.name, "shopping-turn")
        self.assertEqual(
            [child.name for child in exported.children],
            [
                "interpretation",
                "state_validation",
                "retrieval",
                "fusion",
                "reranking",
                "clarification",
                "response",
            ],
        )
        retrieval = next(
            child for child in exported.children if child.name == "retrieval"
        )
        self.assertEqual(
            [child.name for child in retrieval.children],
            ["retrieval.dense"],
        )
        self.assertTrue(all(child.ended for child in exported.children))
        self.assertEqual(client.flush_count, 1)

    def test_missing_credentials_never_build_a_client(self) -> None:
        sink = LangfuseSink(public_key=None, secret_key=None)
        self.assertEqual(sink.status, "disabled")
        self.assertEqual(sink.metrics()["disabled_reason"], "missing_credentials")
        with self.assertRaises(RuntimeError):
            sink.export([{"name": "shopping-turn", "observations": []}])
        sink.flush()
        self.assertEqual(sink.metrics()["exported_traces"], 0)

    def test_absent_client_dependency_disables_the_sink_after_one_attempt(self) -> None:
        attempts: list[int] = []

        def factory():
            attempts.append(1)
            raise ImportError("no module named langfuse")

        sink = LangfuseSink(client_factory=factory)
        tracer = Tracer(sink, register_atexit=False)
        agent = self.build_agent(tracer)
        agent.reset("session", USER_PROFILE)
        agent.respond("session", "blue cotton running shoe", 1, 10)

        self.assertFalse(tracer.flush())
        self.assertEqual(tracer.metrics()["last_failure_cause"], "ImportError")
        agent.respond("session", "I prefer leather", 2, 10)
        self.assertFalse(tracer.flush())
        self.assertEqual(len(attempts), 1)
        self.assertEqual(sink.metrics()["disabled_reason"], "dependency_unavailable")
        self.assertEqual(tracer.metrics()["exported_traces"], 0)

    def test_client_construction_failure_surfaces_as_a_flush_failure(self) -> None:
        def factory():
            raise ConnectionRefusedError("langfuse host is unreachable")

        sink = LangfuseSink(client_factory=factory)
        tracer = Tracer(sink, register_atexit=False)
        agent = self.build_agent(tracer)
        agent.reset("session", USER_PROFILE)
        response = agent.respond("session", "blue cotton running shoe", 1, 10)

        self.assertTrue(response["recommendations"])
        self.assertFalse(tracer.flush())
        self.assertEqual(
            tracer.metrics()["last_failure_cause"],
            "ConnectionRefusedError",
        )


if __name__ == "__main__":
    unittest.main()
