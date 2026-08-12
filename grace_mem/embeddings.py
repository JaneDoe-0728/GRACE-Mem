from typing import List, Sequence
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

class HFTextEmbedding:
    """
    單一多語 embedding：BGE-M3
    - 向量維度：1024
    - 使用方式：
        embedder = HFTextEmbedding(device="cuda")
        vecs = embedder.embed(["你好", "Hello"])
    """

    MODEL_PATH = _REPO_ROOT / "models" / "embedding_models" / "qwen3-0.6b"
    # MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "embedding_models" / "bge-m3"

    def __init__(self, device: str | None = None, batch_size: int = 16, max_length: int = 512):
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
            print(f"[HFTextEmbedding] 使用本地embeddding model：{path}  (device={device})")
        else:
            print(f"[HFTextEmbedding] 本地找不到 {path}，改用 HF Hub：BAAI/bge-m3  (device={device})")
            path = "Qwen/Qwen3-Embedding-0.6B"

        self.device = device
        self.model = self._load_model(path, device)
        self.batch_size = batch_size
        self.max_length = max_length

    def _load_model(self, path: str, device: str) -> SentenceTransformer:
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
        """回傳 np.ndarray，shape=(n, dim)，已經 normalize"""
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
