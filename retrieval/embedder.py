from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_IDENTITY = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_DIR = Path("models") / "BAAI__bge-small-en-v1.5"

# bge-small-en-v1.5 reads the [CLS] token and expects L2-normalized vectors.
POOLING = "cls"
NORMALIZE = True
MAX_SEQUENCE_LENGTH = 256

# Asymmetric retrieval: passages are embedded bare, queries carry this prefix.
# Recorded in the manifest so the live Retrieval Route cannot drift from the
# convention the index was built under.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ModelUnavailable(RuntimeError):
    """The bundled embedding model is missing; the scoring path never downloads one."""


class LocalEmbedder:
    """Embeds text with a bundled local model and never reaches the network.

    Padding is fixed to ``MAX_SEQUENCE_LENGTH`` rather than to the longest item in
    each batch, so a product's vector does not depend on which products happened
    to share its batch. That is what makes a rebuild reproducible.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        identity: str = DEFAULT_MODEL_IDENTITY,
        torch_threads: int = 8,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.identity = identity
        self.torch_threads = torch_threads
        self.max_sequence_length = max_sequence_length
        if not self.model_dir.is_dir():
            raise ModelUnavailable(
                f"No embedding model at {self.model_dir}. Fetch it once with "
                "'python -m tools.fetch_model'; the scoring path must not download models."
            )

        # Belt and braces: even a corrupted local cache must not trigger a fetch.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        torch.set_num_threads(torch_threads)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                str(self.model_dir), local_files_only=True, dtype=torch.float32
            )
        except Exception as error:  # noqa: BLE001 - surfaced as a clear startup failure
            raise ModelUnavailable(
                f"Could not load the embedding model at {self.model_dir}: {error}"
            ) from error
        self.model.eval()
        self.dimensions = int(self.model.config.hidden_size)

    def embed(self, texts: list[str], batch_size: int = 64) -> "object":
        import numpy

        torch = self._torch
        if not texts:
            return numpy.zeros((0, self.dimensions), dtype=numpy.float32)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_sequence_length,
                    return_tensors="pt",
                )
                output = self.model(**encoded).last_hidden_state[:, 0]
                if NORMALIZE:
                    output = torch.nn.functional.normalize(output, p=2, dim=1)
                chunks.append(output.to(torch.float32).numpy())
        return numpy.ascontiguousarray(numpy.concatenate(chunks, axis=0))

    def embed_query(self, text: str) -> "object":
        return self.embed([QUERY_PREFIX + text])[0]
