from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from retrieval.embedder import (
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_IDENTITY,
    MAX_SEQUENCE_LENGTH,
    NORMALIZE,
    POOLING,
    QUERY_PREFIX,
    LocalEmbedder,
)
from retrieval.manifest import (
    ARTIFACT_VERSION,
    catalog_content_sha256,
    directory_size_bytes,
    embedding_checksum,
    file_sha256,
    model_fingerprint,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from retrieval.product_text import TEXT_FIELDS, TEXT_TEMPLATE_VERSION, product_text
from retrieval.resources import peak_rss_bytes


COLLECTION_NAME = "catalog_dense"
VECTOR_STORE_DIRECTORY = "qdrant"
IDS_NAME = "ids.json"
DEFAULT_ARTIFACT_DIR = Path("artifacts") / "dense"


@dataclass(frozen=True)
class BuildConfig:
    catalog_path: Path
    artifact_dir: Path
    model_dir: Path = DEFAULT_MODEL_DIR
    model_identity: str = DEFAULT_MODEL_IDENTITY
    batch_size: int = 64
    torch_threads: int = 8
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    keep_embeddings: bool = False


def load_catalog(catalog_path: str | Path) -> list[dict]:
    """Read the catalog into a stable order.

    Sorting by parent_asin makes a point id a function of the catalog's content
    rather than of its line order, so a reordered or re-exported catalog file
    rebuilds to the same index.
    """
    products: dict[str, dict] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            products[str(product["parent_asin"])] = product
    return [products[key] for key in sorted(products)]


def build(config: BuildConfig) -> dict:
    """Build the versioned dense artifact and return its manifest."""
    import numpy

    started = time.perf_counter()
    catalog_path = Path(config.catalog_path)
    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    products = load_catalog(catalog_path)
    identifiers = [str(product["parent_asin"]) for product in products]
    texts = [product_text(product) for product in products]

    embedder = LocalEmbedder(
        config.model_dir,
        identity=config.model_identity,
        torch_threads=config.torch_threads,
        max_sequence_length=config.max_sequence_length,
    )
    embed_started = time.perf_counter()
    vectors = embedder.embed(texts, batch_size=config.batch_size)
    embed_seconds = time.perf_counter() - embed_started

    (artifact_dir / IDS_NAME).write_text(
        json.dumps(identifiers) + "\n", encoding="utf-8"
    )
    if config.keep_embeddings:
        numpy.save(artifact_dir / "embeddings.npy", vectors)

    _write_vector_store(artifact_dir, identifiers, vectors)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "catalog": {
            "file_name": catalog_path.name,
            "file_sha256": file_sha256(catalog_path),
            "content_sha256": catalog_content_sha256(products),
            "product_count": len(products),
        },
        "embedding_model": {
            "identity": config.model_identity,
            "revision": _recorded_revision(config.model_dir),
            "fingerprint_sha256": model_fingerprint(config.model_dir),
            "dimensions": embedder.dimensions,
            "pooling": POOLING,
            "normalize": NORMALIZE,
            "max_sequence_length": config.max_sequence_length,
            "query_prefix": QUERY_PREFIX,
        },
        "build_config": {
            "text_template_version": TEXT_TEMPLATE_VERSION,
            "text_fields": list(TEXT_FIELDS),
            "batch_size": config.batch_size,
            "torch_threads": config.torch_threads,
            "padding": "max_length",
            "dtype": "float32",
        },
        "vector_store": {
            "engine": "qdrant-local",
            "collection": COLLECTION_NAME,
            "distance": "cosine",
            "point_id_scheme": "sorted_parent_asin_row_index",
            "directory": VECTOR_STORE_DIRECTORY,
        },
        "embedding_checksum": embedding_checksum(vectors),
        "metrics": {
            "build_seconds": round(time.perf_counter() - started, 3),
            "embed_seconds": round(embed_seconds, 3),
            "products_per_second": (
                round(len(products) / embed_seconds, 2) if embed_seconds else None
            ),
            "artifact_bytes": directory_size_bytes(artifact_dir),
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    # Written last, so a build interrupted partway leaves no manifest and the
    # incomplete artifact refuses to load rather than serving short results.
    write_manifest(artifact_dir, manifest)
    return manifest


def _write_vector_store(artifact_dir: Path, identifiers: list[str], vectors: object) -> None:
    from qdrant_client import QdrantClient, models

    store_path = artifact_dir / VECTOR_STORE_DIRECTORY
    if store_path.exists():
        shutil.rmtree(store_path)
    client = QdrantClient(path=str(store_path))
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=int(vectors.shape[1]), distance=models.Distance.COSINE
            ),
        )
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=row,
                    vector=vectors[row].tolist(),
                    payload={"parent_asin": parent_asin},
                )
                for row, parent_asin in enumerate(identifiers)
            ],
            wait=True,
        )
    finally:
        client.close()


def _recorded_revision(model_dir: str | Path) -> str | None:
    fetched = Path(model_dir) / "FETCHED.json"
    if not fetched.is_file():
        return None
    return json.loads(fetched.read_text(encoding="utf-8")).get("revision")


class DenseIndex:
    """The dense Retrieval Route's loaded artifact.

    Opening an index verifies it against the catalog and model actually present,
    so an artifact built for a different catalog fails at startup rather than
    quietly scoring against products that are no longer in the catalog.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        catalog_path: str | Path,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        verify: bool = True,
    ) -> None:
        started = time.perf_counter()
        self.artifact_dir = Path(artifact_dir)
        self.manifest = read_manifest(self.artifact_dir)
        if verify:
            verify_manifest(self.manifest, catalog_path, model_dir)

        from qdrant_client import QdrantClient

        # Local mode: a directory on disk. No port is bound and no separate
        # vector service has to be running.
        self.client = QdrantClient(path=str(self.artifact_dir / VECTOR_STORE_DIRECTORY))
        self.identifiers: list[str] = json.loads(
            (self.artifact_dir / IDS_NAME).read_text(encoding="utf-8")
        )
        self.dimensions = int(self.manifest["embedding_model"]["dimensions"])
        self.load_metrics = {
            "load_seconds": round(time.perf_counter() - started, 3),
            "artifact_bytes": directory_size_bytes(self.artifact_dir),
            "peak_rss_bytes": peak_rss_bytes(),
            "product_count": len(self.identifiers),
        }

    def search(self, vector: object, limit: int = 100) -> list[tuple[str, float]]:
        points = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(vector),
            limit=limit,
            with_payload=True,
        ).points
        return [(str(point.payload["parent_asin"]), float(point.score)) for point in points]

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "DenseIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
