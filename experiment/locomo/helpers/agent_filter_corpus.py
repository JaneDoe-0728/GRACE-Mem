"""The Agent Filter corpus for a LoCoMo sample.

LongMem hands Agent Filter a per-question script_data CSV. LoCoMo has no such
file: its retrievable unit is a chunk cut out of locomo10.json, so the corpus has
to be built from the raw sample. The splitting reproduces ingest exactly -- drop
empty turns, then group by ``pos // chunk_turns`` -- because the sids the agent
selects have to name the same units retrieval scored.

Shared by the live pipeline (locomo/pipeline/worker.py) and the replay entry
point (common/post_retrieval/locomo.py), which is the only reason it is not a private
function of either.
"""
from __future__ import annotations

from grace_mem.agent_filter.corpus import Corpus, Turn


def build_chunk_corpus(sample: dict, sample_idx: int, n: int, unit: str = "chunk") -> Corpus:
    """Chunk-level corpus: the splitting logic matches ingest exactly (filter empty
    turns, then pos//N)."""
    conv = sample.get("conversation", {}) or {}
    turns: list[Turn] = []
    for key, sess_turns in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(sess_turns, list):
            continue
        sess = int(key.split("_", 1)[1])
        date = str(conv.get(f"session_{sess}_date_time", "") or "")
        pos = 0
        chunks: dict[int, list[str]] = {}
        for t in sess_turns:
            speaker = str(t.get("speaker", "")).strip()
            text = str(t.get("text", "")).strip()
            caption = str(t.get("blip_caption", "")).strip()
            if not speaker and not text and not caption:
                continue
            line = f"{speaker}: {text}"
            if caption:
                line += f" (Image: {caption})"
            chunks.setdefault(pos // n, []).append(line)
            pos += 1
        if unit == "turn":
            # Turn granularity: every kept turn is its own unit (recall that
            # session-level precision ceilings out around 0.06, so only a finer unit
            # leaves room for localization). The sid carries the chunk for tracing.
            for ci in sorted(chunks):
                for off, line in enumerate(chunks[ci]):
                    turns.append(Turn(
                        sid=f"{sample_idx}__{sess}:{ci}t{off}",
                        session_id=f"{sample_idx}__{sess}",
                        turn_index=ci * 100 + off,
                        pos=ci * 100 + off,
                        role="dialogue",
                        date=date,
                        text=line,
                    ))
        else:
            for ci in sorted(chunks):
                turns.append(Turn(
                    sid=f"{sample_idx}__{sess}:{ci}",
                    session_id=f"{sample_idx}__{sess}",
                    turn_index=ci,
                    pos=ci,
                    role="dialogue",
                    date=date,
                    text="\n".join(chunks[ci]),
                ))
    return Corpus(turns)
