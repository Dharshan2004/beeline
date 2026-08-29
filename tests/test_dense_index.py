from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retrieval.dense_index import BuildConfig, DenseIndex, build, load_catalog
from retrieval.embedder import DEFAULT_MODEL_DIR, ModelUnavailable
from retrieval.manifest import ARTIFACT_VERSION, DenseIndexMismatch
from retrieval.product_text import product_text


MODEL_DIR = DEFAULT_MODEL_DIR
model_available = MODEL_DIR.is_dir()
requires_model = unittest.skipUnless(
    model_available,
    f"bundled embedding model missing at {MODEL_DIR}; run 'python -m tools.fetch_model'",
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
        self.assertEqual(manifest["embedding_model"]["identity"], "BAAI/bge-small-en-v1.5")
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
