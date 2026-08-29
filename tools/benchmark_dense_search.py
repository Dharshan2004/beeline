"""Measure local dense-search latency against a completed artifact."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from retrieval.dense_index import DEFAULT_ARTIFACT_DIR, DenseIndex
from retrieval.embedder import DEFAULT_MODEL_DIR, LocalEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--query", default="comfortable shoes for everyday walking")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    arguments = parser.parse_args()

    vector = LocalEmbedder(arguments.model_dir).embed_query(arguments.query)
    with DenseIndex(
        arguments.artifact_dir, arguments.catalog, arguments.model_dir
    ) as index:
        for _ in range(5):
            index.search(vector, limit=arguments.limit)
        durations = []
        for _ in range(arguments.iterations):
            started = time.perf_counter()
            index.search(vector, limit=arguments.limit)
            durations.append((time.perf_counter() - started) * 1000)

    ordered = sorted(durations)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    print(
        json.dumps(
            {
                "iterations": len(durations),
                "limit": arguments.limit,
                "median_ms": round(statistics.median(durations), 3),
                "p95_ms": round(p95, 3),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
