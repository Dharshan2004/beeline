from __future__ import annotations

from copy import deepcopy
import time
from pathlib import Path

from retrieval.dense_index import DEFAULT_ARTIFACT_DIR, DenseIndex
from retrieval.embedder import DEFAULT_MODEL_DIR, LocalEmbedder


class DenseRetrievalRoute:
    """Optional embedded dense Retrieval Route used by the live Shopping Agent."""

    def __init__(
        self,
        catalog_path: str | Path,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
    ) -> None:
        started = time.perf_counter()
        self.configured = True
        self._embedder: LocalEmbedder | None = None
        self._index: DenseIndex | None = None
        self._status = "disabled"
        self._disabled_reason: str | None = None
        self._query_count = 0
        self._last_query_seconds: float | None = None
        self._last_candidate_count = 0

        try:
            embedder = LocalEmbedder(model_dir)
            index = DenseIndex(artifact_dir, catalog_path, model_dir)
        except Exception as error:  # noqa: BLE001 - this route must fail open
            self._disable(error)
        else:
            self._embedder = embedder
            self._index = index
            self._status = "available"
        self._load_seconds = time.perf_counter() - started

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self._status != "available" or limit <= 0:
            return []
        started = time.perf_counter()
        self._query_count += 1
        try:
            assert self._embedder is not None
            assert self._index is not None
            candidates = self._index.search(
                self._embedder.embed_query(query),
                limit=limit,
            )
        except Exception as error:  # noqa: BLE001 - preserve the Agent contract
            self._disable(error)
            candidates = []
        self._last_query_seconds = time.perf_counter() - started
        self._last_candidate_count = len(candidates)
        return candidates

    def _disable(self, error: Exception) -> None:
        self._status = "disabled"
        self._disabled_reason = f"{type(error).__name__}: {error}"

    def metrics(self) -> dict:
        return {
            "status": self._status,
            "disabled_reason": self._disabled_reason,
            "load_seconds": self._load_seconds,
            "query_count": self._query_count,
            "last_query_seconds": self._last_query_seconds,
            "last_candidate_count": self._last_candidate_count,
        }

    def configuration(self) -> dict:
        """Return the verified index/model identity used by this route."""
        return {
            "status": self._status,
            "disabled_reason": self._disabled_reason,
            "manifest": (
                deepcopy(self._index.manifest)
                if self._index is not None
                else None
            ),
        }

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None
