from __future__ import annotations

import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from retrieval.dense_index import (
    BuildConfig,
    DenseIndex,
    IDS_NAME,
    VECTOR_STORE_DIRECTORY,
    _write_vector_store,
    _verify_artifact_files,
    _verify_loaded_collection,
    build,
    load_catalog,
)
from retrieval.embedder import DEFAULT_MODEL_DIR, ModelUnavailable
from retrieval.manifest import (
    ARTIFACT_VERSION,
    DenseIndexMismatch,
    directory_sha256,
    file_sha256,
    model_fingerprint,
    verify_manifest,
    write_manifest,
)
from retrieval.product_text import product_text


MODEL_DIR = DEFAULT_MODEL_DIR
model_available = MODEL_DIR.is_dir()
requires_model = unittest.skipUnless(
    model_available,
    f"bundled embedding model missing at {MODEL_DIR}; run 'python -m tools.fetch_model'",
)
requires_qdrant = unittest.skipUnless(
    importlib.util.find_spec("qdrant_client") is not None,
    "qdrant-client is not installed",
)


CATALOG_ROWS = [
    {
        "parent_asin": "B0000000C1",
        "title": "Womens black leather ankle boot",
        "categories": ["Clothing", "Shoes", "Boots"],
        "features": ["genuine leather", "side zip"],
        "details": {"department": "womens", "material": "leather"},
        "description": ["A durable ankle boot for winter wear."],
        "store": "Northline",
        "price": 89.0,
    },
    {
        "parent_asin": "B0000000A1",
        "title": "Blue cotton running shoe",
        "categories": ["Clothing", "Shoes", "Athletic"],
        "features": ["breathable cotton upper", "cushioned sole"],
        "details": {"department": "womens", "material": "cotton"},
        "description": ["A lightweight running shoe for daily training."],
        "store": "Example",
        "price": 49.0,
    },
    {
        "parent_asin": "B0000000B1",
        "title": "Red wool winter scarf",
        "categories": ["Clothing", "Accessories", "Scarves"],
        "features": ["merino wool"],
        "details": {"department": "unisex", "material": "wool"},
        "description": ["A soft scarf for cold weather."],
        "store": "Example",
        "price": 25.0,
    },
]


def write_catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class ProductTextTest(unittest.TestCase):
    def test_representation_is_independent_of_key_and_whitespace_variation(self) -> None:
        first = {
            "parent_asin": "A",
            "title": "Blue  running\n shoe",
            "details": {"material": "cotton", "department": "womens"},
        }
        second = {
            "parent_asin": "A",
            "details": {"department": "womens", "material": "cotton"},
            "title": "Blue running shoe",
        }

        self.assertEqual(product_text(first), product_text(second))

    def test_representation_excludes_empty_fields(self) -> None:
        text = product_text({"parent_asin": "A", "title": "Shoe", "features": [], "store": None})

        self.assertEqual(text, "title: Shoe")


class CatalogOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_catalog_loads_in_parent_asin_order_regardless_of_file_order(self) -> None:
        forward = self.root / "forward.jsonl"
        reversed_file = self.root / "reversed.jsonl"
        write_catalog(forward, CATALOG_ROWS)
        write_catalog(reversed_file, list(reversed(CATALOG_ROWS)))

        self.assertEqual(
            [row["parent_asin"] for row in load_catalog(forward)],
            ["B0000000A1", "B0000000B1", "B0000000C1"],
        )
        self.assertEqual(load_catalog(forward), load_catalog(reversed_file))


class MissingModelTest(unittest.TestCase):
    def test_absent_model_directory_reports_how_to_fetch_it_and_does_not_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            write_catalog(catalog, CATALOG_ROWS)

            with self.assertRaisesRegex(ModelUnavailable, "tools.fetch_model"):
                build(
                    BuildConfig(
                        catalog_path=catalog,
                        artifact_dir=root / "artifact",
                        model_dir=root / "absent-model",
                    )
                )


class IntegrityVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_model_fingerprint_detects_weight_and_revision_changes(self) -> None:
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"hidden_size": 2}', encoding="utf-8")
        (model / "FETCHED.json").write_text(
            '{"identity": "example/model", "revision": "first"}', encoding="utf-8"
        )
        weights = model / "model.safetensors"
        weights.write_bytes(b"first weights")
        original = model_fingerprint(model)

        weights.write_bytes(b"second weights")
        self.assertNotEqual(model_fingerprint(model), original)

        weights.write_bytes(b"first weights")
        (model / "FETCHED.json").write_text(
            '{"identity": "example/model", "revision": "second"}', encoding="utf-8"
        )
        self.assertNotEqual(model_fingerprint(model), original)

    def test_manifest_model_identity_must_match_fetched_provenance(self) -> None:
        catalog = self.root / "catalog.jsonl"
        write_catalog(catalog, CATALOG_ROWS[:1])
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"test weights")
        (model / "FETCHED.json").write_text(
            '{"identity": "expected/model", "revision": "abc123"}',
            encoding="utf-8",
        )
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "catalog": {"file_sha256": file_sha256(catalog)},
            "embedding_model": {
                "identity": "wrong/model",
                "revision": "abc123",
                "fingerprint_sha256": model_fingerprint(model),
            },
        }

        with self.assertRaisesRegex(DenseIndexMismatch, "identity mismatch"):
            verify_manifest(manifest, catalog, model)

    def test_artifact_file_verification_rejects_changed_identifier_map(self) -> None:
        artifact = self.root / "artifact"
        store = artifact / "qdrant"
        store.mkdir(parents=True)
        ids = artifact / "ids.json"
        ids.write_text('["A"]\n', encoding="utf-8")
        (store / "storage.db").write_bytes(b"vector store")
        manifest = {
            "vector_store": {
                "directory": "qdrant",
                "directory_sha256": directory_sha256(store),
                "ids_file": "ids.json",
                "ids_sha256": file_sha256(ids),
            }
        }
        _verify_artifact_files(manifest, artifact)

        ids.write_text('["B"]\n', encoding="utf-8")
        with self.assertRaisesRegex(DenseIndexMismatch, "identifier map checksum"):
            _verify_artifact_files(manifest, artifact)

    def test_artifact_file_verification_rejects_changed_vector_store(self) -> None:
        artifact = self.root / "artifact"
        store = artifact / "qdrant"
        store.mkdir(parents=True)
        ids = artifact / "ids.json"
        ids.write_text('["A"]\n', encoding="utf-8")
        stored_vector = store / "storage.db"
        stored_vector.write_bytes(b"first")
        manifest = {
            "vector_store": {
                "directory": "qdrant",
                "directory_sha256": directory_sha256(store),
                "ids_file": "ids.json",
                "ids_sha256": file_sha256(ids),
            }
        }

        stored_vector.write_bytes(b"corrupt")
        with self.assertRaisesRegex(DenseIndexMismatch, "vector store checksum"):
            _verify_artifact_files(manifest, artifact)

    def test_loaded_collection_must_match_manifest_shape_and_count(self) -> None:
        client = SimpleNamespace(
            get_collection=lambda _: SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=128))
                ),
                points_count=2,
            )
        )
        manifest = {
            "embedding_model": {"dimensions": 384},
            "catalog": {"product_count": 3},
        }

        with self.assertRaisesRegex(
            DenseIndexMismatch,
            "(?s)embedding dimensions mismatch.*vector count mismatch",
        ):
            _verify_loaded_collection(client, manifest, ["A", "B"])

    def test_failed_staged_build_preserves_previous_artifact(self) -> None:
        artifact = self.root / "artifact"
        artifact.mkdir()
        previous_manifest = artifact / "manifest.json"
        previous_manifest.write_text('{"artifact_version": 1}\n', encoding="utf-8")
        config = BuildConfig(
            catalog_path=self.root / "catalog.jsonl",
            artifact_dir=artifact,
            model_dir=self.root / "model",
        )

        def fail_after_partial_write(_: BuildConfig, staging_dir: Path) -> dict:
            (staging_dir / "partial").write_text("incomplete", encoding="utf-8")
            raise RuntimeError("simulated build interruption")

        with (
            patch("retrieval.dense_index._build_into", side_effect=fail_after_partial_write),
            self.assertRaisesRegex(RuntimeError, "simulated build interruption"),
        ):
            build(config)

        self.assertEqual(
            previous_manifest.read_text(encoding="utf-8"),
            '{"artifact_version": 1}\n',
        )
        self.assertEqual(list(self.root.glob(".artifact.build-*")), [])

    @requires_qdrant
    def test_real_qdrant_artifact_remains_verifiable_across_reloads(self) -> None:
        import numpy

        catalog = self.root / "catalog.jsonl"
        write_catalog(catalog, CATALOG_ROWS[:2])
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"hidden_size": 3}', encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"test weights")
        artifact = self.root / "artifact"
        artifact.mkdir()
        identifiers = ["B0000000A1", "B0000000C1"]
        (artifact / IDS_NAME).write_text(json.dumps(identifiers) + "\n", encoding="utf-8")
        _write_vector_store(
            artifact,
            identifiers,
            numpy.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=numpy.float32),
        )
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "catalog": {
                "file_sha256": file_sha256(catalog),
                "content_sha256": "unused because file checksum matches",
                "product_count": 2,
            },
            "embedding_model": {
                "fingerprint_sha256": model_fingerprint(model),
                "dimensions": 3,
            },
            "vector_store": {
                "directory": VECTOR_STORE_DIRECTORY,
                "directory_sha256": directory_sha256(
                    artifact / VECTOR_STORE_DIRECTORY
                ),
                "ids_file": IDS_NAME,
                "ids_sha256": file_sha256(artifact / IDS_NAME),
            },
        }
        write_manifest(artifact, manifest)

        with DenseIndex(artifact, catalog, model) as index:
            self.assertEqual(index.dimensions, 3)
        with DenseIndex(artifact, catalog, model) as index:
            self.assertEqual(index.identifiers, identifiers)


@requires_model
class DenseIndexBuildTest(unittest.TestCase):
    """Covers the Slice 04 acceptance criteria against a small fixed catalog."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.catalog = cls.root / "catalog.jsonl"
        write_catalog(cls.catalog, CATALOG_ROWS)
        cls.artifact = cls.root / "artifact"
        cls.manifest = build(
            BuildConfig(
                catalog_path=cls.catalog,
                artifact_dir=cls.artifact,
                model_dir=MODEL_DIR,
                batch_size=2,
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_build_is_deterministic_for_a_fixed_catalog_and_configuration(self) -> None:
        rebuilt = build(
            BuildConfig(
                catalog_path=self.catalog,
                artifact_dir=self.root / "rebuild",
                model_dir=MODEL_DIR,
                # A different batch size must not change a single vector, because
                # padding is fixed rather than per-batch.
                batch_size=3,
            )
        )

        self.assertEqual(rebuilt["embedding_checksum"], self.manifest["embedding_checksum"])
        self.assertEqual(rebuilt["catalog"], self.manifest["catalog"])

    def test_thread_count_does_not_change_the_vectors(self) -> None:
        """Build machines differ in core count; the artifact must not."""
        rebuilt = build(
            BuildConfig(
                catalog_path=self.catalog,
                artifact_dir=self.root / "single-thread",
                model_dir=MODEL_DIR,
                torch_threads=1,
            )
        )

        self.assertEqual(rebuilt["embedding_checksum"], self.manifest["embedding_checksum"])

    def test_reordered_catalog_file_rebuilds_to_the_same_vectors(self) -> None:
        shuffled = self.root / "shuffled.jsonl"
        write_catalog(shuffled, list(reversed(CATALOG_ROWS)))

        rebuilt = build(
            BuildConfig(
                catalog_path=shuffled,
                artifact_dir=self.root / "shuffled-artifact",
                model_dir=MODEL_DIR,
            )
        )

        self.assertEqual(rebuilt["embedding_checksum"], self.manifest["embedding_checksum"])
        self.assertEqual(
            rebuilt["catalog"]["content_sha256"], self.manifest["catalog"]["content_sha256"]
        )

    def test_manifest_records_catalog_model_dimensions_and_build_configuration(self) -> None:
        manifest = json.loads((self.artifact / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["artifact_version"], ARTIFACT_VERSION)
        self.assertEqual(manifest["catalog"]["product_count"], len(CATALOG_ROWS))
        self.assertEqual(len(manifest["catalog"]["file_sha256"]), 64)
        self.assertEqual(len(manifest["catalog"]["content_sha256"]), 64)
        self.assertEqual(
            manifest["embedding_model"]["identity"],
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertEqual(manifest["embedding_model"]["dimensions"], 384)
        self.assertEqual(len(manifest["embedding_model"]["fingerprint_sha256"]), 64)
        self.assertIn("text_template_version", manifest["build_config"])
        self.assertEqual(manifest["vector_store"]["engine"], "qdrant-local")

    def test_build_reports_time_size_and_peak_memory(self) -> None:
        metrics = self.manifest["metrics"]

        self.assertGreater(metrics["build_seconds"], 0)
        self.assertGreater(metrics["artifact_bytes_before_manifest"], 0)
        self.assertGreater(metrics["peak_rss_bytes"], 0)

    def test_artifact_loads_offline_without_network_access(self) -> None:
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
            with DenseIndex(self.artifact, self.catalog, MODEL_DIR) as index:
                self.assertEqual(index.dimensions, 384)
                self.assertEqual(len(index.identifiers), len(CATALOG_ROWS))
                self.assertGreater(index.load_metrics["load_seconds"], 0)
                self.assertGreater(index.load_metrics["peak_rss_bytes"], 0)

    def test_semantically_similar_query_ranks_the_matching_product_first(self) -> None:
        from retrieval.embedder import LocalEmbedder

        embedder = LocalEmbedder(MODEL_DIR)
        vector = embedder.embed_query("something warm to wrap around my neck in winter")

        with DenseIndex(self.artifact, self.catalog, MODEL_DIR) as index:
            ranked = index.search(vector, limit=3)

        self.assertEqual(ranked[0][0], "B0000000B1")

    def test_changed_catalog_is_rejected_before_the_index_is_used(self) -> None:
        altered = self.root / "altered.jsonl"
        write_catalog(altered, CATALOG_ROWS + [{"parent_asin": "B0000000D1", "title": "Green hat"}])

        with self.assertRaisesRegex(DenseIndexMismatch, "catalog checksum mismatch"):
            DenseIndex(self.artifact, altered, MODEL_DIR)

    def test_changed_model_is_rejected_before_the_index_is_used(self) -> None:
        substitute = self.root / "substitute-model"
        substitute.mkdir(exist_ok=True)
        (substitute / "config.json").write_text('{"hidden_size": 384}', encoding="utf-8")

        with self.assertRaisesRegex(DenseIndexMismatch, "embedding model mismatch"):
            DenseIndex(self.artifact, self.catalog, substitute)

    def test_absent_manifest_reports_how_to_build_one(self) -> None:
        with self.assertRaisesRegex(DenseIndexMismatch, "build_dense_index"):
            DenseIndex(self.root / "never-built", self.catalog, MODEL_DIR)


if __name__ == "__main__":
    unittest.main()
