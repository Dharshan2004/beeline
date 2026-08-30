from __future__ import annotations

import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class RerankerCandidate:
    directory: str
    revision: str


# The one immutable candidate manifest used by fetch, score, and summarize.
RERANKER_CANDIDATES = MappingProxyType({
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": RerankerCandidate(
        "cross-encoder__ms-marco-TinyBERT-L-2-v2",
        "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc",
    ),
    "cross-encoder/ms-marco-MiniLM-L-2-v2": RerankerCandidate(
        "cross-encoder__ms-marco-MiniLM-L-2-v2",
        "1b5cd67b15209f24824c50370e0397743aa9b787",
    ),
    "cross-encoder/ms-marco-MiniLM-L-6-v2": RerankerCandidate(
        "cross-encoder__ms-marco-MiniLM-L-6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ),
})

DEFAULT_RERANKER_IDENTITY = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_DIR = (
    Path("models") / RERANKER_CANDIDATES[DEFAULT_RERANKER_IDENTITY].directory
)
FROZEN_RERANK_DEPTH = 50

# A query plus one product rendering. Cross-encoder cost is dominated by
# sequence length, and 128 tokens already covers title, store, categories, and
# the leading features, which is where the discriminating evidence lives.
MAX_SEQUENCE_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_RERANK_DEADLINE_SECONDS = 1.5
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0


class RerankerUnavailable(RuntimeError):
    """The bundled cross-encoder is missing; the scoring path never downloads one."""


class Reranker(Protocol):
    identity: str
    configured: bool

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        documents: Sequence[str],
    ) -> list[str]: ...

    def metrics(self) -> dict: ...

    def close(self) -> None: ...


class CrossEncoderReranker:
    """Scores benchmark (query, product) pairs with a local cross-encoder.

    Padding is fixed to ``max_sequence_length`` so a product's score does not
    depend on which products happened to share its batch. That is what makes a
    benchmark row reproducible and a live turn deterministic.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_RERANKER_DIR,
        *,
        identity: str = DEFAULT_RERANKER_IDENTITY,
        torch_threads: int = 8,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.identity = identity
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        if not self.model_dir.is_dir():
            raise RerankerUnavailable(
                f"No cross-encoder at {self.model_dir}. Fetch it once with "
                "'python -m tools.fetch_model --identity <id> --destination <dir>'; "
                "the scoring path must not download models."
            )

        # Belt and braces: even a corrupted local cache must not trigger a fetch.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        torch.set_num_threads(torch_threads)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir), local_files_only=True, dtype=torch.float32
            )
        except Exception as error:  # noqa: BLE001 - surfaced as a clear startup failure
            raise RerankerUnavailable(
                f"Could not load the cross-encoder at {self.model_dir}: {error}"
            ) from error
        self.model.eval()

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """Return one benchmark relevance score per document, higher is better."""
        if not documents:
            return []
        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = list(documents[start : start + self.batch_size])
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            scores.extend(float(value) for value in logits[:, 0].tolist())
        return scores


def order_by_scores(
    candidates: Sequence[str],
    scores: Sequence[float],
) -> list[str]:
    """Order candidates by descending score, breaking ties by input position.

    The input order is the fused Candidate Pool ordering, so a cross-encoder
    that cannot separate two products leaves the fused decision standing.
    """
    if len(candidates) != len(scores):
        raise ValueError("reranker must return exactly one score per candidate")
    numeric_scores = [float(score) for score in scores]
    if any(not isfinite(score) for score in numeric_scores):
        raise ValueError("reranker scores must be finite numbers")
    ranked = sorted(
        zip(range(len(candidates)), candidates, numeric_scores),
        key=lambda item: (-item[2], item[0]),
    )
    return [parent_asin for _position, parent_asin, _score in ranked]


def _percentile(ordered_values: Sequence[float], fraction: float) -> float:
    if not ordered_values:
        return 0.0
    index = max(0, ceil(len(ordered_values) * fraction) - 1)
    return float(ordered_values[index])


def _model_settings(
    model_dir: Path,
    identity: str,
    torch_threads: int,
    batch_size: int,
    max_sequence_length: int,
) -> dict[str, Any]:
    return {
        "model_dir": str(model_dir),
        "identity": identity,
        "torch_threads": torch_threads,
        "batch_size": batch_size,
        "max_sequence_length": max_sequence_length,
    }


def _local_worker_main(connection, settings: dict[str, Any]) -> None:
    """Load once, then serve score requests until the parent closes the worker."""
    try:
        scorer = CrossEncoderReranker(**settings)
        connection.send({"type": "ready", "pid": os.getpid()})
        while True:
            request = connection.recv()
            if request.get("type") == "close":
                return
            if request.get("type") != "score":
                raise ValueError("unknown worker request")
            scores = scorer.score(request["query"], request["documents"])
            connection.send({
                "type": "scores",
                "request_id": request["request_id"],
                "scores": scores,
            })
    except EOFError:
        return
    except Exception as error:  # noqa: BLE001 - parent must receive fail-open evidence
        try:
            message_type = "startup_error" if "scorer" not in locals() else "worker_error"
            connection.send({"type": message_type, "error": repr(error)})
        except Exception:  # noqa: BLE001 - the parent also detects a closed pipe
            pass
    finally:
        connection.close()


class UnavailableReranker:
    """No-op reranker that exposes why the optional local route is disabled."""

    identity = DEFAULT_RERANKER_IDENTITY

    def __init__(self, reason: str, *, configured: bool = False) -> None:
        self.reason = reason
        self.configured = configured

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        documents: Sequence[str],
    ) -> list[str]:
        return list(candidates)

    def metrics(self) -> dict:
        return {
            "status": "disabled",
            "identity": self.identity,
            "revision": RERANKER_CANDIDATES[self.identity].revision,
            "depth": FROZEN_RERANK_DEPTH,
            "deadline_seconds": DEFAULT_RERANK_DEADLINE_SECONDS,
            "failure_cause": self.reason,
            "query_count": 0,
            "attempt_count": 0,
            "last_latency_seconds": 0.0,
            "latency_p50_seconds": 0.0,
            "latency_p95_seconds": 0.0,
        }

    def close(self) -> None:
        pass


def verify_frozen_reranker_model(
    model_dir: str | Path,
    identity: str = DEFAULT_RERANKER_IDENTITY,
) -> str:
    """Verify a local reranker's declared immutable identity and revision."""
    directory = Path(model_dir)
    candidate = RERANKER_CANDIDATES.get(identity)
    if candidate is None:
        raise RerankerUnavailable(f"unknown reranker identity {identity!r}")
    if not directory.is_dir():
        raise RerankerUnavailable(
            f"No cross-encoder at {directory}; the scoring path must not download models."
        )
    provenance_path = directory / "FETCHED.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RerankerUnavailable(
            f"Could not verify frozen reranker provenance at {provenance_path}: {error}"
        ) from error
    if (
        provenance.get("identity") != identity
        or provenance.get("revision") != candidate.revision
    ):
        raise RerankerUnavailable(
            "Bundled reranker provenance does not match the frozen identity and revision"
        )
    return candidate.revision


class LocalRerankerWorker:
    """Persistent, cancellable process for the frozen local cross-encoder.

    The parent owns the absolute deadline. Any lifecycle or protocol failure
    terminates the process and permanently returns the fused ordering, avoiding
    repeated model-load cost during an evaluator run.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_RERANKER_DIR,
        *,
        identity: str = DEFAULT_RERANKER_IDENTITY,
        deadline_seconds: float = DEFAULT_RERANK_DEADLINE_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        torch_threads: int = 8,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        context=None,
        worker_target: Callable = _local_worker_main,
    ) -> None:
        if deadline_seconds <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("reranker deadlines must be positive")
        self.model_dir = Path(model_dir)
        self.configured = True
        self.identity = identity
        self.revision = self._verify_frozen_model()
        self.deadline_seconds = deadline_seconds
        self._status = "starting"
        self._failure_cause: str | None = None
        self._failure_detail: str | None = None
        self._query_count = 0
        self._attempt_count = 0
        self._latencies_seconds: list[float] = []
        self._last_latency_seconds = 0.0
        self._request_id = 0
        self._worker_pid: int | None = None
        process_context = context or multiprocessing.get_context()
        parent_connection, child_connection = process_context.Pipe()
        self._connection = parent_connection
        settings = _model_settings(
            self.model_dir,
            identity,
            torch_threads,
            batch_size,
            max_sequence_length,
        )
        self._process = process_context.Process(
            target=worker_target,
            args=(child_connection, settings),
            name="shopping-agent-reranker",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        if not self._connection.poll(startup_timeout_seconds):
            self._disable("startup_timeout", "worker did not load before its startup deadline")
            raise RerankerUnavailable(self._failure_detail or "reranker startup timed out")
        try:
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            self._disable("startup_failure", repr(error))
            raise RerankerUnavailable(f"reranker worker exited during startup: {error}") from error
        if not isinstance(response, dict) or response.get("type") != "ready":
            detail = response.get("error") if isinstance(response, dict) else repr(response)
            self._disable("startup_failure", str(detail))
            raise RerankerUnavailable(f"reranker worker startup failed: {detail}")
        self._worker_pid = int(response.get("pid", self._process.pid))
        self._status = "available"

    def _verify_frozen_model(self) -> str:
        return verify_frozen_reranker_model(self.model_dir, self.identity)

    def _terminate(self, *, join_timeout: float = 0.0) -> None:
        process = getattr(self, "_process", None)
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=join_timeout)
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def _disable(self, cause: str, detail: str) -> None:
        self._status = "disabled"
        self._failure_cause = cause
        self._failure_detail = detail
        self._terminate()

    def _record_latency(self, started: float) -> None:
        self._last_latency_seconds = time.monotonic() - started
        self._latencies_seconds.append(self._last_latency_seconds)

    def rerank(
        self,
        query: str,
        candidates: Sequence[str],
        documents: Sequence[str],
    ) -> list[str]:
        fallback = list(candidates)[:FROZEN_RERANK_DEPTH]
        if self._status != "available" or not fallback:
            return fallback
        capped_documents = list(documents)[:FROZEN_RERANK_DEPTH]
        if len(capped_documents) != len(fallback):
            self._disable("malformed_input", "candidate and document counts differ")
            return fallback
        self._request_id += 1
        request_id = self._request_id
        started = time.monotonic()
        self._attempt_count += 1
        try:
            self._connection.send({
                "type": "score",
                "request_id": request_id,
                "query": str(query),
                "documents": capped_documents,
            })
            remaining_seconds = self.deadline_seconds - (time.monotonic() - started)
            if remaining_seconds <= 0 or not self._connection.poll(remaining_seconds):
                self._record_latency(started)
                self._disable("deadline_exceeded", "reranking exceeded the absolute deadline")
                return fallback
            response = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError) as error:
            self._record_latency(started)
            self._disable("worker_crash", repr(error))
            return fallback
        self._record_latency(started)
        if isinstance(response, dict) and response.get("type") == "worker_error":
            self._disable("worker_failure", str(response.get("error")))
            return fallback
        try:
            if (
                not isinstance(response, dict)
                or response.get("type") != "scores"
                or response.get("request_id") != request_id
            ):
                raise ValueError(f"unexpected worker response: {response!r}")
            ranked = order_by_scores(fallback, response.get("scores", ()))
        except (TypeError, ValueError) as error:
            self._disable("malformed_output", str(error))
            return fallback
        self._query_count += 1
        return ranked

    def metrics(self) -> dict:
        ordered_latencies = sorted(self._latencies_seconds)
        p50 = _percentile(ordered_latencies, 0.50)
        p95 = _percentile(ordered_latencies, 0.95)
        return {
            "status": self._status,
            "identity": self.identity,
            "revision": self.revision,
            "depth": FROZEN_RERANK_DEPTH,
            "deadline_seconds": self.deadline_seconds,
            "worker_pid": self._worker_pid,
            "query_count": self._query_count,
            "attempt_count": self._attempt_count,
            "last_latency_seconds": self._last_latency_seconds,
            "latency_p50_seconds": p50,
            "latency_p95_seconds": p95,
            "failure_cause": self._failure_cause,
            "failure_detail": self._failure_detail,
        }

    def close(self) -> None:
        if self._status == "available":
            try:
                self._connection.send({"type": "close"})
                self._process.join(timeout=1.0)
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._terminate(join_timeout=1.0)
        if self._status == "available":
            self._status = "closed"


def build_live_reranker(
    model_dir: str | Path = DEFAULT_RERANKER_DIR,
) -> Reranker:
    """Build the frozen worker, or a deterministic no-op when it cannot start."""
    try:
        return LocalRerankerWorker(model_dir)
    except Exception as error:  # noqa: BLE001 - optional route must fail open
        return UnavailableReranker(
            f"{type(error).__name__}: {error}",
            configured=True,
        )
