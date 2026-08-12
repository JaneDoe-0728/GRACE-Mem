"""缺口描述向量補搜(sufficiency 修復臂的第二把武器)。

grep 修復臂的空手率 ~87%,根因是 paraphrase gap:verifier 指出的缺失資訊
往往不是 corpus 裡的字面 span。這裡把「question + missing 描述」embed 後
直接查該題的 summaries VDB(:u/:a split,rebuild_split_summaries.py 產物),
撈回 grep 搆不到的語意近鄰,交給 agent 確認。

VDB client 與 embedder 都是 lazy 全域快取(embedder ~2-3GB GPU,只載一次)。
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

_lock = threading.Lock()
_vdb_cache: dict[str, object] = {}
_embed_fn = None


def _get_embed():
    global _embed_fn
    if _embed_fn is None:
        from grace_mem.embeddings import embedder  # lazy: 載入 qwen3-0.6b(GPU)
        _embed_fn = embedder.embed
    return _embed_fn


def _get_vdb(artifact_dir: Path):
    key = str(artifact_dir)
    with _lock:
        if key not in _vdb_cache:
            from grace_mem.storage.chroma_vdb import SummariesVDB
            _vdb_cache[key] = SummariesVDB(
                dim=1024,
                path=str(Path(artifact_dir) / "summaries_chroma"),
                collection_name="summaries",
            )
        return _vdb_cache[key]


def vector_gap_candidates(
    artifact_dir: str | Path,
    query_text: str,
    *,
    exclude: set[str],
    topn: int = 6,
    min_score: float = 0.30,
) -> list[tuple[str, float]]:
    """回傳 [(sid, score)],已排除 exclude 內的 sid。任何失敗回空 list。"""
    try:
        artifact_dir = Path(artifact_dir)
        if not (artifact_dir / "summaries_chroma").exists():
            return []
        vec = _get_embed()([query_text])
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 2:
            vec = vec[0]
        hits = _get_vdb(artifact_dir).search(vec, top_k=topn * 3, threshold=min_score)
        out: list[tuple[str, float]] = []
        for meta, score in hits:
            sid = str(meta.get("id") or "").strip()
            if not sid or sid in exclude:
                continue
            out.append((sid, float(score)))
            if len(out) >= topn:
                break
        return out
    except Exception:
        return []
