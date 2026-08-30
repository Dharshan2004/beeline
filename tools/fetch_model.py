"""One-time download of a bundled embedding or cross-encoder model.

Run this during development or image build only. The scoring path loads the
model from disk and never downloads. The resolved immutable revision is written
to ``FETCHED.json`` for benchmark and release reproducibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.embedder import DEFAULT_MODEL_DIR, DEFAULT_MODEL_IDENTITY

ALLOWED_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", default=DEFAULT_MODEL_IDENTITY)
    parser.add_argument("--destination", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--revision", default=None, help="pin a specific commit sha")
    arguments = parser.parse_args()

    from huggingface_hub import snapshot_download

    destination = Path(arguments.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=arguments.identity,
        revision=arguments.revision,
        local_dir=str(destination),
        allow_patterns=ALLOWED_PATTERNS,
    )

    from huggingface_hub import HfApi

    revision = arguments.revision or HfApi().model_info(arguments.identity).sha
    (destination / "FETCHED.json").write_text(
        json.dumps({"identity": arguments.identity, "revision": revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"identity": arguments.identity, "revision": revision, "path": path}, indent=2))


if __name__ == "__main__":
    main()
