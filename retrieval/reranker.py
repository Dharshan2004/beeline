from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


# Candidate cross-encoders compared in Slice 07. Each is small enough to bundle
# and runs on CPU without a runtime download.
RERANKER_CANDIDATES: dict[str, str] = {
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": "cross-encoder__ms-marco-TinyBERT-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-2-v2": "cross-encoder__ms-marco-MiniLM-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "cross-encoder__ms-marco-MiniLM-L-6-v2",
}

DEFAULT_RERANKER_IDENTITY = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_DIR = Path("models") / RERANKER_CANDIDATES[DEFAULT_RERANKER_IDENTITY]
FROZEN_RERANK_DEPTH = 50

# A query plus one product rendering. Cross-encoder cost is dominated by
# sequence length, and 128 tokens already covers title, store, categories, and
# the leading features, which is where the discriminating evidence lives.
MAX_SEQUENCE_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


class RerankerUnavailable(RuntimeError):
    """The bundled cross-encoder is missing; the scoring path never downloads one."""


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
    ranked = sorted(
        zip(range(len(candidates)), candidates, scores),
        key=lambda item: (-item[2], item[0]),
    )
    return [parent_asin for _position, parent_asin, _score in ranked]
