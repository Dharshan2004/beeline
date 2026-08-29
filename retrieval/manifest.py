from __future__ import annotations

import hashlib
import json
from pathlib import Path

from retrieval.product_text import product_text


ARTIFACT_VERSION = 1
MANIFEST_NAME = "manifest.json"
_CHUNK = 1024 * 1024

# Weight files are large and are covered by the tokenizer/config hashes plus the
# recorded revision; hashing every shard on each load would dominate startup, so
# the model fingerprint covers the files that actually change behavior.
MODEL_FINGERPRINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


class DenseIndexMismatch(RuntimeError):
    """The artifact on disk does not match the catalog or model it was built from."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_content_sha256(products: list[dict]) -> str:
    """Checksum the catalog's meaning rather than its bytes.

    Two catalog files that differ only in line endings, key order, or row order
    describe the same products and must produce the same vectors, so they share
    a content checksum.
    """
    digest = hashlib.sha256()
    for parent_asin, text in sorted(
        (str(product["parent_asin"]), product_text(product)) for product in products
    ):
        digest.update(parent_asin.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def model_fingerprint(model_dir: str | Path) -> str:
    directory = Path(model_dir)
    digest = hashlib.sha256()
    for name in MODEL_FINGERPRINT_FILES:
        candidate = directory / name
        if not candidate.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def embedding_checksum(vectors: object) -> str:
    """Checksum the embedding matrix itself.

    Determinism is asserted here rather than over the vector store directory,
    because embedded Qdrant is free to lay its storage out differently between
    otherwise identical builds.
    """
    import numpy

    array = numpy.ascontiguousarray(numpy.asarray(vectors, dtype=numpy.float32))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def read_manifest(artifact_dir: str | Path) -> dict:
    path = Path(artifact_dir) / MANIFEST_NAME
    if not path.is_file():
        raise DenseIndexMismatch(
            f"No dense index manifest at {path}. Build one with "
            "'python -m retrieval.build_dense_index' before running a scored session."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(artifact_dir: str | Path, manifest: dict) -> Path:
    path = Path(artifact_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_manifest(
    manifest: dict,
    catalog_path: str | Path,
    model_dir: str | Path,
    *,
    embedding_dimensions: int | None = None,
) -> None:
    """Fail loudly when the artifact no longer describes the catalog or model in use.

    Called before the artifact serves any turn, so an incompatible index is a
    clear startup error rather than a silent quality regression mid-session.
    """
    problems: list[str] = []

    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        problems.append(
            f"artifact_version is {manifest.get('artifact_version')!r}, "
            f"this build of the agent reads {ARTIFACT_VERSION!r}"
        )

    catalog = manifest.get("catalog") or {}
    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        problems.append(f"catalog file {catalog_path} does not exist")
    else:
        actual_file = file_sha256(catalog_path)
        if actual_file != catalog.get("file_sha256"):
            # The bytes differ; the products may still be identical, so fall back
            # to the content checksum before rejecting a merely reformatted file.
            products = [
                json.loads(line)
                for line in catalog_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            actual_content = catalog_content_sha256(products)
            if actual_content != catalog.get("content_sha256"):
                problems.append(
                    "catalog checksum mismatch: artifact was built from "
                    f"content {str(catalog.get('content_sha256'))[:12]}… "
                    f"but {catalog_path} has content {actual_content[:12]}…"
                )

    model = manifest.get("embedding_model") or {}
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        problems.append(
            f"embedding model directory {model_dir} does not exist; "
            "fetch it once with 'python -m tools.fetch_model'"
        )
    else:
        actual_model = model_fingerprint(model_dir)
        if actual_model != model.get("fingerprint_sha256"):
            problems.append(
                f"embedding model mismatch: artifact was built with "
                f"{model.get('identity')!r} fingerprint {str(model.get('fingerprint_sha256'))[:12]}… "
                f"but {model_dir} fingerprints as {actual_model[:12]}…"
            )

    if embedding_dimensions is not None and model.get("dimensions") != embedding_dimensions:
        problems.append(
            f"embedding dimensions mismatch: artifact records {model.get('dimensions')!r}, "
            f"loaded vectors are {embedding_dimensions!r}"
        )

    if problems:
        raise DenseIndexMismatch(
            "The dense index artifact does not match the current catalog or model:\n  - "
            + "\n  - ".join(problems)
            + "\nRebuild it with 'python -m retrieval.build_dense_index'."
        )
