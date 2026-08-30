from __future__ import annotations

import atexit
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Protocol


TRACE_NAME = "shopping-turn"
DEFAULT_BUFFER_LIMIT = 512
MAX_STRING_LENGTH = 200
MAX_SEQUENCE_LENGTH = 20
MAX_MAPPING_KEYS = 40
MAX_DEPTH = 4

# Exact field names that may carry credentials, the raw user profile, full
# catalog records, or private chain-of-thought. They are never exported.
DENIED_KEYS = frozenset({
    "api_key",
    "authorization",
    "catalog",
    "catalog_record",
    "catalog_records",
    "chain_of_thought",
    "credentials",
    "instructions",
    "message",
    "password",
    "products",
    "profile",
    "prompt",
    "public_key",
    "raw_phrase",
    "rationale",
    "reasoning",
    "secret_key",
    "thoughts",
    "user_message",
    "user_profile",
})
# Substrings that mark a secret regardless of the surrounding field name. The
# usage counters `prompt_tokens` and `completion_tokens` deliberately do not
# match any of them.
DENIED_KEY_SUBSTRINGS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "bearer",
    "chain_of_thought",
    "credential",
    "password",
    "secret",
)


def is_denied_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in DENIED_KEYS:
        return True
    return any(marker in lowered for marker in DENIED_KEY_SUBSTRINGS)


def sanitize(value: object, *, depth: int = 0) -> object:
    """Reduce a payload to structured operational evidence that is safe to export."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]
    if depth >= MAX_DEPTH:
        return None
    if isinstance(value, Mapping):
        result: dict = {}
        for key, item in value.items():
            if not isinstance(key, str) or is_denied_key(key):
                continue
            result[key] = sanitize(item, depth=depth + 1)
            if len(result) >= MAX_MAPPING_KEYS:
                break
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = (
            sorted(value, key=repr)
            if isinstance(value, (set, frozenset))
            else list(value)
        )
        return [sanitize(item, depth=depth + 1) for item in items[:MAX_SEQUENCE_LENGTH]]
    return None


class TraceSink(Protocol):
    def export(self, traces: list[dict]) -> None: ...

    def flush(self) -> None: ...

    def metrics(self) -> dict: ...


class NullSink:
    """Sink used when telemetry is disabled; it never touches the network."""

    def __init__(self, reason: str = "disabled") -> None:
        self.reason = reason

    def export(self, traces: list[dict]) -> None:
        return None

    def flush(self) -> None:
        return None

    def metrics(self) -> dict:
        return {"status": "disabled", "disabled_reason": self.reason}


class MemorySink:
    """In-process sink used by tests and offline inspection."""

    def __init__(self) -> None:
        self.traces: list[dict] = []
        self.flush_count = 0

    def export(self, traces: list[dict]) -> None:
        self.traces.extend(traces)

    def flush(self) -> None:
        self.flush_count += 1

    def metrics(self) -> dict:
        return {
            "status": "available",
            "disabled_reason": None,
            "exported_traces": len(self.traces),
        }


class LangfuseSink:
    """Fail-open Langfuse client wrapper.

    The client is created lazily on the first export so that a missing
    dependency, missing credentials, or a refused connection can only be
    observed outside the latency-critical response path.
    """

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        client_factory=None,
    ) -> None:
        self._public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self._secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self._host = host or os.environ.get("LANGFUSE_HOST")
        self._client_factory = client_factory
        self._client = None
        self._status = "pending"
        self._disabled_reason: str | None = None
        self._exported_traces = 0
        if client_factory is None and not (self._public_key and self._secret_key):
            self._disable("missing_credentials")

    @property
    def status(self) -> str:
        return self._status

    def _disable(self, reason: str) -> None:
        self._status = "disabled"
        self._disabled_reason = reason

    def _build_client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from langfuse import Langfuse  # noqa: PLC0415 - optional runtime dependency

        options = {"public_key": self._public_key, "secret_key": self._secret_key}
        if self._host:
            options["host"] = self._host
        return Langfuse(**options)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if self._status == "disabled":
            raise RuntimeError(f"langfuse sink is disabled: {self._disabled_reason}")
        try:
            self._client = self._build_client()
        except ImportError:
            # The optional client is absent; stop retrying on every flush.
            self._disable("dependency_unavailable")
            raise
        self._status = "available"
        return self._client

    def export(self, traces: list[dict]) -> None:
        client = self._ensure_client()
        for record in traces:
            self._export_trace(client, record)
            self._exported_traces += 1
        return None

    def _export_trace(self, client, record: dict) -> None:
        trace = client.trace(
            name=record.get("name", TRACE_NAME),
            session_id=record.get("session_id"),
            metadata=record.get("metadata"),
            tags=record.get("tags"),
        )
        for observation in record.get("observations", ()):
            self._export_observation(trace, observation)

    def _export_observation(self, parent, observation: dict) -> None:
        span = parent.span(
            name=observation.get("name"),
            metadata=observation.get("metadata"),
            level="ERROR" if observation.get("status") == "error" else "DEFAULT",
            status_message=observation.get("failure_cause"),
        )
        for child in observation.get("observations", ()):
            self._export_observation(span, child)
        end = getattr(span, "end", None)
        if callable(end):
            end()

    def flush(self) -> None:
        if self._client is None:
            return None
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            flush()
        return None

    def metrics(self) -> dict:
        return {
            "status": self._status,
            "disabled_reason": self._disabled_reason,
            "exported_traces": self._exported_traces,
            "host": self._host,
        }


@dataclass
class _Observation:
    name: str
    metadata: dict = field(default_factory=dict)
    children: list["_Observation"] = field(default_factory=list)
    status: str = "ok"
    failure_cause: str | None = None
    started: float = 0.0
    latency_ms: float | None = None

    def record(self, **fields: object) -> None:
        self.metadata.update(fields)

    def fail(self, cause: str) -> None:
        self.status = "error"
        self.failure_cause = str(cause)[:MAX_STRING_LENGTH]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "failure_cause": self.failure_cause,
            "latency_ms": self.latency_ms,
            "metadata": sanitize(self.metadata),
            "observations": [child.as_dict() for child in self.children],
        }


class NoOpOperation:
    """Operation handle returned when tracing is disabled."""

    def record(self, **fields: object) -> None:
        return None

    def fail(self, cause: str) -> None:
        return None


@contextmanager
def _noop_operation() -> Iterator[NoOpOperation]:
    yield NoOpOperation()


class NoOpTurnTrace:
    """Turn handle returned when tracing is disabled or already failing."""

    def record(self, **fields: object) -> None:
        return None

    def fail(self, cause: str) -> None:
        return None

    def operation(self, name: str):
        return _noop_operation()

    def close(self) -> None:
        return None

    def __enter__(self) -> "NoOpTurnTrace":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


class TurnTrace:
    """One evaluation turn, holding nested operation timings for a session."""

    def __init__(self, tracer: "Tracer", session_id: str, turn: int) -> None:
        self._tracer = tracer
        self._session_id = session_id
        self._turn = turn
        self._clock = tracer.clock
        self._root = _Observation(name=TRACE_NAME)
        self._root.started = self._clock()
        self._stack: list[_Observation] = [self._root]
        self._closed = False

    def record(self, **fields: object) -> None:
        try:
            self._stack[-1].record(**fields)
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            self._tracer.note_failure("record_failed")

    def fail(self, cause: str) -> None:
        try:
            self._stack[-1].fail(cause)
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            self._tracer.note_failure("record_failed")

    @contextmanager
    def operation(self, name: str) -> Iterator[_Observation]:
        try:
            observation = _Observation(name=name)
            observation.started = self._clock()
            self._stack[-1].children.append(observation)
            self._stack.append(observation)
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            self._tracer.note_failure("operation_failed")
            yield NoOpOperation()  # type: ignore[misc]
            return
        try:
            yield observation
        except BaseException as error:
            observation.fail(type(error).__name__)
            raise
        finally:
            observation.latency_ms = (self._clock() - observation.started) * 1000.0
            if self._stack and self._stack[-1] is observation:
                self._stack.pop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.latency_ms = (self._clock() - self._root.started) * 1000.0
            metadata = {
                "turn": self._turn,
                "session_id": self._session_id,
                "configuration": self._tracer.session_configuration(self._session_id),
            }
            metadata.update(sanitize(self._root.metadata))  # type: ignore[arg-type]
            self._tracer.submit({
                "name": TRACE_NAME,
                "session_id": self._session_id,
                "turn": self._turn,
                "status": self._root.status,
                "failure_cause": self._root.failure_cause,
                "latency_ms": self._root.latency_ms,
                "metadata": metadata,
                "tags": ["shopping-agent"],
                "observations": [child.as_dict() for child in self._root.children],
            })
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            self._tracer.note_failure("submit_failed")

    def __enter__(self) -> "TurnTrace":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self._root.fail(exc_type.__name__)
        self.close()
        return False


class Tracer:
    """Buffer structured turn traces and export them outside the response path."""

    def __init__(
        self,
        sink: TraceSink | None = None,
        *,
        enabled: bool = True,
        buffer_limit: int = DEFAULT_BUFFER_LIMIT,
        clock=time.perf_counter,
        register_atexit: bool = True,
    ) -> None:
        self.clock = clock
        self._sink: TraceSink = sink if sink is not None else NullSink()
        self._enabled = bool(enabled) and not isinstance(self._sink, NullSink)
        self._buffer: deque[dict] = deque(maxlen=max(1, buffer_limit))
        self._configurations: dict[str, dict] = {}
        self._submitted = 0
        self._dropped = 0
        self._exported = 0
        self._export_failures = 0
        self._flush_count = 0
        self._last_failure_cause: str | None = None
        self._atexit_registered = False
        if self._enabled and register_atexit:
            try:
                atexit.register(self.flush)
                self._atexit_registered = True
            except Exception:  # noqa: BLE001 - telemetry must never change behavior
                self.note_failure("atexit_registration_failed")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs,
    ) -> "Tracer":
        env = os.environ if environ is None else environ
        switch = str(env.get("SHOPPING_AGENT_TELEMETRY", "1")).strip().lower()
        if switch in ("0", "false", "off", "no"):
            return cls(NullSink("telemetry_switched_off"), enabled=False, **kwargs)
        if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
            return cls(NullSink("missing_credentials"), enabled=False, **kwargs)
        try:
            sink: TraceSink = LangfuseSink(
                public_key=env.get("LANGFUSE_PUBLIC_KEY"),
                secret_key=env.get("LANGFUSE_SECRET_KEY"),
                host=env.get("LANGFUSE_HOST"),
            )
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            return cls(NullSink("sink_construction_failed"), enabled=False, **kwargs)
        return cls(sink, **kwargs)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_session(
        self,
        session_id: str,
        configuration: Mapping | None = None,
    ) -> None:
        if not self._enabled:
            return None
        try:
            self._configurations[session_id] = sanitize(dict(configuration or {}))
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            self.note_failure("session_start_failed")
        return None

    def session_configuration(self, session_id: str) -> dict:
        return dict(self._configurations.get(session_id, {}))

    def turn(self, session_id: str, turn: int):
        if not self._enabled:
            return NoOpTurnTrace()
        try:
            return TurnTrace(self, session_id, turn)
        except Exception:  # noqa: BLE001 - telemetry must never change behavior
            self.note_failure("turn_start_failed")
            return NoOpTurnTrace()

    def submit(self, record: dict) -> None:
        if len(self._buffer) == self._buffer.maxlen:
            self._dropped += 1
        self._buffer.append(record)
        self._submitted += 1

    def note_failure(self, cause: str) -> None:
        self._last_failure_cause = cause

    def buffered_traces(self) -> list[dict]:
        return list(self._buffer)

    def flush(self) -> bool:
        """Export buffered traces. Never raises and never blocks a turn."""
        if not self._enabled:
            return False
        pending = list(self._buffer)
        self._buffer.clear()
        self._flush_count += 1
        try:
            if pending:
                self._sink.export(pending)
            self._sink.flush()
        except Exception as error:  # noqa: BLE001 - export must fail open
            self._export_failures += 1
            self.note_failure(type(error).__name__)
            return False
        self._exported += len(pending)
        return True

    def metrics(self) -> dict:
        try:
            sink_metrics = dict(self._sink.metrics())
        except Exception as error:  # noqa: BLE001 - metrics must fail open
            sink_metrics = {
                "status": "unknown",
                "disabled_reason": type(error).__name__,
            }
        return {
            "enabled": self._enabled,
            "buffered_traces": len(self._buffer),
            "submitted_traces": self._submitted,
            "exported_traces": self._exported,
            "dropped_traces": self._dropped,
            "export_failures": self._export_failures,
            "flush_count": self._flush_count,
            "last_failure_cause": self._last_failure_cause,
            "atexit_registered": self._atexit_registered,
            "sink": sink_metrics,
        }
