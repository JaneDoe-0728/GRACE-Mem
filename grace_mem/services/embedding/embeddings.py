"""The single text embedding model shared by every vector store in the system.

One model, loaded once, is a deliberate constraint rather than an oversight:
summaries, entities and relationships all land in the same Chroma instance and
are compared against one another during fusion, so they must live in the same
vector space. A per-collection choice of encoder would make those cross-store
similarity scores meaningless.

The module-level `embedder` singleton at the bottom is imported directly across
the codebase; instantiating it here means the weights are paid for once per
process instead of once per caller.
"""

import os
from collections.abc import Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from grace_mem.utils.paths import resolve_project_root

_REPO_ROOT = resolve_project_root()
load_dotenv(_REPO_ROOT / ".env")

class HFTextEmbedding:
    """Encode text to L2-normalized vectors with Qwen3-Embedding-0.6B.

    Vectors are 1024-dimensional and normalized at encode time, which lets
    every downstream consumer treat a dot product as cosine similarity and skip
    its own normalization step.

    Example:
        >>> embedder = HFTextEmbedding(device="cuda")
        >>> vecs = embedder.embed(["Bonjour", "Hello"])
        >>> vecs.shape
        (2, 1024)
    """

    MODEL_PATH = _REPO_ROOT / "models" / "embedding_models" / "qwen3-0.6b"

    def __init__(self, device: str | None = None, batch_size: int = 16, max_length: int = 512):
        """Load the encoder, resolving the device explicit > env > best-available.

        Args:
            device: Torch device string. None defers to EMBEDDING_DEVICE, then
                to the best accelerator present.
            batch_size: Sequences per forward pass. Lower it before lowering
                max_length if the GPU is tight -- truncation loses information,
                smaller batches only cost time.
            max_length: Token cap per input.
        """
        env_device = os.getenv("EMBEDDING_DEVICE")
        if device is None and env_device:
            device = env_device

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        path = str(self.MODEL_PATH)
        if os.path.isdir(path):
            print(f"[HFTextEmbedding] using local embedding model: {path}  (device={device})")
        else:
            print(
                f"[HFTextEmbedding] {path} not found locally, falling back to "
                f"HF Hub: Qwen/Qwen3-Embedding-0.6B  (device={device})"
            )
            path = "Qwen/Qwen3-Embedding-0.6B"

        self.device = device
        self.model = self._load_model(path, device)
        self.batch_size = batch_size
        self.max_length = max_length

    def _load_model(self, path: str, device: str) -> SentenceTransformer:
        """Load the encoder, degrading to CPU rather than dying on GPU OOM.

        Two exception types are caught because the OOM surfaces differently
        depending on where allocation fails: torch raises OutOfMemoryError from
        the allocator, but a failure inside a kernel arrives as a generic
        RuntimeError whose message is the only way to tell it apart.
        """
        try:
            return SentenceTransformer(path, device=device)
        except torch.OutOfMemoryError:
            if device == "cuda":
                print("[HFTextEmbedding] CUDA OOM during model load, fallback to CPU.")
                torch.cuda.empty_cache()
                self.device = "cpu"
                return SentenceTransformer(path, device="cpu")
            raise
        except RuntimeError as exc:
            if device == "cuda" and "out of memory" in str(exc).lower():
                print("[HFTextEmbedding] CUDA runtime OOM during model load, fallback to CPU.")
                torch.cuda.empty_cache()
                self.device = "cpu"
                return SentenceTransformer(path, device="cpu")
            raise

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an np.ndarray of shape (n, dim), already normalized."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)
    
embedder = HFTextEmbedding()
