"""Benchmark one local embedding candidate on a deterministic public-set proxy.

This is a development-only selection tool. Run each candidate in a fresh process
so peak memory measurements are comparable. It never downloads models.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from evaluator.local_evaluator import (
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from retrieval.product_text import product_text
from retrieval.resources import peak_rss_bytes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pooling", choices=("cls", "mean"), required=True)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--public-set", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--distractors", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    return parser.parse_args()


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_products(path: Path) -> dict[str, dict]:
    return {
        str(product["parent_asin"]): product
        for product in load_jsonl(path)
    }


def candidate_pool(
    products: dict[str, dict], target_ids: list[str], distractor_count: int
) -> list[str]:
    targets = set(target_ids)
    distractors = sorted(set(products) - targets)
    selected = random.Random(20260829).sample(
        distractors, min(distractor_count, len(distractors))
    )
    return sorted(targets.union(selected))


def benchmark(arguments: argparse.Namespace) -> dict:
    import numpy
    import torch
    from transformers import AutoModel, AutoTokenizer

    products = load_products(arguments.catalog)
    samples = load_jsonl(arguments.public_set)
    target_ids = [str(sample["ground_truth"]["parent_asin"]) for sample in samples]
    pool_ids = candidate_pool(products, target_ids, arguments.distractors)
    passages = [product_text(products[parent_asin]) for parent_asin in pool_ids]
    queries: list[str] = []
    for sample in samples:
        card, behavior = materialize_hidden_fields(sample, products)
        enriched = {**sample, "intent_card": card, "behavior": behavior}
        category = coarse_category(products[str(sample["ground_truth"]["parent_asin"])].get("categories") or [])
        queries.append(
            arguments.query_prefix + initial_message(enriched, category, set())
        )

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(
        str(arguments.model_dir), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(arguments.model_dir), local_files_only=True, dtype=torch.float32
    )
    model.eval()

    def embed(texts: list[str]) -> object:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), arguments.batch_size):
                encoded = tokenizer(
                    texts[start : start + arguments.batch_size],
                    padding="max_length",
                    truncation=True,
                    max_length=arguments.max_sequence_length,
                    return_tensors="pt",
                )
                hidden = model(**encoded).last_hidden_state
                if arguments.pooling == "cls":
                    vectors = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                chunks.append(vectors.to(torch.float32).numpy())
        return numpy.ascontiguousarray(numpy.concatenate(chunks, axis=0))

    passage_started = time.perf_counter()
    passage_vectors = embed(passages)
    passage_seconds = time.perf_counter() - passage_started
    query_started = time.perf_counter()
    query_vectors = embed(queries)
    query_seconds = time.perf_counter() - query_started

    ranks: list[int] = []
    id_to_row = {parent_asin: row for row, parent_asin in enumerate(pool_ids)}
    for query_vector, target_id in zip(query_vectors, target_ids, strict=True):
        scores = passage_vectors @ query_vector
        target_score = scores[id_to_row[target_id]]
        ranks.append(1 + int(numpy.count_nonzero(scores > target_score)))

    return {
        "identity": arguments.identity,
        "pooling": arguments.pooling,
        "query_prefix": arguments.query_prefix,
        "sample_count": len(samples),
        "candidate_pool_size": len(pool_ids),
        "distractor_seed": 20260829,
        "hit_rate_at_10": round(sum(rank <= 10 for rank in ranks) / len(ranks), 6),
        "mrr": round(statistics.fmean(1 / rank for rank in ranks), 6),
        "passage_embedding_seconds": round(passage_seconds, 3),
        "passages_per_second": round(len(passages) / passage_seconds, 2),
        "query_embedding_ms_mean": round(query_seconds * 1000 / len(queries), 2),
        "peak_rss_bytes": peak_rss_bytes(),
        "model_bytes": directory_size_bytes(arguments.model_dir),
        "max_sequence_length": arguments.max_sequence_length,
        "batch_size": arguments.batch_size,
    }


def main() -> None:
    print(json.dumps(benchmark(parse_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
