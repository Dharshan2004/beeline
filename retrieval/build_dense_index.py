"""Build the versioned dense index artifact for the frozen catalog.

    python -m retrieval.build_dense_index --catalog data/catalog.jsonl

The artifact is written ahead of time and loaded once at agent startup; no turn
ever pays indexing cost and the scoring path never downloads a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.dense_index import (
    DEFAULT_ARTIFACT_DIR,
    BuildConfig,
    DenseIndex,
    build,
)
from retrieval.embedder import DEFAULT_MODEL_DIR, DEFAULT_MODEL_IDENTITY, MAX_SEQUENCE_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-identity", default=DEFAULT_MODEL_IDENTITY)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument(
        "--keep-embeddings",
        action="store_true",
        help="also write embeddings.npy, roughly doubling artifact size",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="reopen the finished artifact and report load time and memory",
    )
    arguments = parser.parse_args()

    manifest = build(
        BuildConfig(
            catalog_path=Path(arguments.catalog),
            artifact_dir=Path(arguments.artifact_dir),
            model_dir=Path(arguments.model_dir),
            model_identity=arguments.model_identity,
            batch_size=arguments.batch_size,
            torch_threads=arguments.torch_threads,
            max_sequence_length=arguments.max_sequence_length,
            keep_embeddings=arguments.keep_embeddings,
        )
    )

    report = {
        "artifact_dir": arguments.artifact_dir,
        "catalog": manifest["catalog"],
        "embedding_model": manifest["embedding_model"],
        "embedding_checksum": manifest["embedding_checksum"],
        "build_metrics": manifest["metrics"],
    }
    if arguments.verify_load:
        with DenseIndex(
            arguments.artifact_dir, arguments.catalog, arguments.model_dir
        ) as index:
            report["load_metrics"] = index.load_metrics
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
