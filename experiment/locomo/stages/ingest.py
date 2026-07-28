# locamal_ingest.py
# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys
import argparse

import pandas as pd

# ======= project import (keep consistent with your repo layout) =======
sys.path.append(str(Path(__file__).resolve().parents[2]))
from locomo.helpers.dataset import build_session_records_from_json, normalize_dataset_name, resolve_dataset_path  # noqa: E402
from locomo.utils.io import load_jsonl_records  # noqa: E402


# ========= Config: edit here =========
INPUT_JSONL = "data/locomo_by_session.jsonl"   # one session per line (a session dict)
MAKE_SESSION_UID = True             # session_id = f"{sample_index}__{session_id}"
PREV_K = 2
ENTITY_SIM_TOPK = 4
ENTITY_SIM_THRESHOLD = 0.5

DIALOGUE_JOINER = "\n"              # keep each utterance on its own line
PUT_SPEAKER_PREFIX = True           # keep "Caroline: ..." in text if present

# Turns per summary chunk. 0 (default) = original behaviour: one summary per whole
# session. When >0, each session is split into consecutive windows of this many
# turns, each becoming its own summary/ingest unit (message_id = chunk index). This
# gives a finer summary-retrieval pool so direct-vector + rerank have room to work.
CHUNK_TURNS = int(os.environ.get("LOCOMO_CHUNK_TURNS", "0") or 0)


def _iter_dialogue_chunks(dialogue: List[str]):
    """Yield (message_id, chunk_lines). CHUNK_TURNS<=0 → whole session as one chunk."""
    if CHUNK_TURNS and CHUNK_TURNS > 0:
        for start in range(0, len(dialogue), CHUNK_TURNS):
            yield start // CHUNK_TURNS, dialogue[start:start + CHUNK_TURNS]
    else:
        yield 0, dialogue

def load_sessions(
    *,
    dataset: str,
    sessions_jsonl: str | Path | None = None,
    dataset_json: str | Path | None = None,
) -> List[Dict[str, Any]]:
    sessions_path = resolve_dataset_path(
        dataset=dataset,
        kind="sessions_jsonl",
        explicit_path=sessions_jsonl,
        required=False,
    )
    if sessions_path and sessions_path.exists():
        return load_jsonl_records(sessions_path)

    dataset_path = resolve_dataset_path(
        dataset=dataset,
        kind="qa_json",
        explicit_path=dataset_json,
    )
    return build_session_records_from_json(dataset_path)


def _session_uid(sample_index: Any, session_id: Any, make_uid: bool) -> str:
    return f"{sample_index}__{session_id}" if make_uid else str(session_id)


def _build_dialogue_text(dialogue: List[str]) -> str:
    """Join dialogue lines, optionally stripping speaker prefixes per PUT_SPEAKER_PREFIX."""
    if PUT_SPEAKER_PREFIX:
        return DIALOGUE_JOINER.join(dialogue)
    stripped = []
    for line in dialogue:
        if ":" in line:
            stripped.append(line.split(":", 1)[1].lstrip())
        else:
            stripped.append(line)
    return DIALOGUE_JOINER.join(stripped)


def sessions_to_one_turn_df(
    sessions: List[Dict[str, Any]],
    *,
    make_session_uid: bool = True,
    sample_filter: Optional[int] = None,
) -> pd.DataFrame:
    """
    每個 session -> 一個 turn：
      - session_id: 唯一（建議 sample_index__session_id）
      - message_id: 固定 0
      - dialogue_datetime: date_time
      - user_text: 整段 dialogue (A/B 兩人對話原樣串起來)
      - assistant_text: 空字串
    """
    rows: List[Dict[str, Any]] = []
    for s in sessions:
        sample_index = s.get("sample_index")
        if sample_filter is not None and int(sample_index) != sample_filter:
            continue
        sess_id = s.get("session_id")
        dialogue = ["" if x is None else str(x) for x in (s.get("dialogue", []) or [])]
        for message_id, chunk in _iter_dialogue_chunks(dialogue):
            rows.append(
                {
                    "session_id": _session_uid(sample_index, sess_id, make_session_uid),
                    "message_id": message_id,
                    "dialogue_datetime": str(s.get("date_time", "")).strip(),
                    "user_text": _build_dialogue_text(chunk),
                    "assistant_text": "",
                    "sample_index": sample_index,
                    "orig_session_id": sess_id,
                    "speaker_a": s.get("speaker_a", ""),
                    "speaker_b": s.get("speaker_b", ""),
                }
            )

    return pd.DataFrame(rows)


def session_records_to_df(
    records: List[Dict[str, Any]],
    *,
    conv_id: str,
) -> pd.DataFrame:
    """Build a one-turn-per-session DataFrame from a list of session record dicts.

    Session UIDs are ``<conv_id>__<session_id>`` so they are unique per conversation
    and do not collide across different source conversations.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        sess_id = rec.get("session_id")
        dialogue = ["" if x is None else str(x) for x in (rec.get("dialogue", []) or [])]
        for message_id, chunk in _iter_dialogue_chunks(dialogue):
            rows.append(
                {
                    "session_id": f"{conv_id}__{sess_id}",
                    "message_id": message_id,
                    "dialogue_datetime": str(rec.get("date_time", "")).strip(),
                    "user_text": _build_dialogue_text(chunk),
                    "assistant_text": "",
                    "sample_index": conv_id,
                    "orig_session_id": sess_id,
                    "speaker_a": rec.get("speaker_a", ""),
                    "speaker_b": rec.get("speaker_b", ""),
                }
            )
    return pd.DataFrame(rows)


def ingest_by_session_one_turn(
    ingestor,
    df: pd.DataFrame,
    *,
    prev_k: Optional[int] = None,
    entity_sim_topk: Optional[int] = None,
    entity_sim_threshold: Optional[float] = None,
) -> dict:
    need_cols = {"session_id", "message_id", "user_text", "assistant_text", "dialogue_datetime"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位: {sorted(miss)}")

    report = defaultdict(list)

    for r in df.to_dict("records"):
        sid = str(r["session_id"])
        res = ingestor.summarize_and_ingest_turn(
            session_id=sid,
            message_id=int(r["message_id"]),
            user_text=str(r["user_text"]),
            assistant_text=str(r["assistant_text"]),
            prev_k=prev_k,
            entity_sim_topk=entity_sim_topk,
            entity_sim_threshold=entity_sim_threshold,
            dialogue_datetime=str(r["dialogue_datetime"]),
        )
        report[sid].append({"message_id": r["message_id"], "result": res})

    return dict(report)


class IngestStage:
    """Class interface for standalone or embedded ingest runs.

    Free functions (load_sessions, sessions_to_one_turn_df, ingest_by_session_one_turn)
    remain as the stable API called by stage_adapter.py.
    """

    def __init__(
        self,
        *,
        ingestor,
        dataset: str,
        dataset_json=None,
        sessions_jsonl=None,
        sample_index: Optional[int] = None,
        prev_k: Optional[int] = None,
        entity_sim_topk: Optional[int] = None,
        entity_sim_threshold: Optional[float] = None,
        make_session_uid: bool = True,
    ) -> None:
        self.ingestor = ingestor
        self.dataset = dataset
        self.dataset_json = dataset_json
        self.sessions_jsonl = sessions_jsonl
        self.sample_index = sample_index
        self.prev_k = prev_k
        self.entity_sim_topk = entity_sim_topk
        self.entity_sim_threshold = entity_sim_threshold
        self.make_session_uid = make_session_uid

    def run(self) -> dict:
        sessions = load_sessions(
            dataset=self.dataset,
            sessions_jsonl=self.sessions_jsonl,
            dataset_json=self.dataset_json,
        )
        df = sessions_to_one_turn_df(
            sessions,
            make_session_uid=self.make_session_uid,
            sample_filter=self.sample_index,
        )
        if df.empty:
            raise RuntimeError(f"No sessions found for sample_index={self.sample_index}")
        return ingest_by_session_one_turn(
            self.ingestor,
            df,
            prev_k=self.prev_k,
            entity_sim_topk=self.entity_sim_topk,
            entity_sim_threshold=self.entity_sim_threshold,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest by-session conversational JSONL into KG/VDB")
    parser.add_argument("--dataset", choices=["locomo", "locomo-plus"], default="locomo")
    parser.add_argument("--sessions-jsonl", default=None, help="Defaults are resolved from --dataset when available")
    parser.add_argument("--dataset-json", default=None, help="Fallback source used to derive sessions when JSONL is absent")
    parser.add_argument("--sample-index", type=int, default=3)
    parser.add_argument("--prev-k", type=int, default=PREV_K)
    parser.add_argument("--entity-sim-topk", type=int, default=ENTITY_SIM_TOPK)
    parser.add_argument("--entity-sim-threshold", type=float, default=ENTITY_SIM_THRESHOLD)
    parser.add_argument("--no-session-uid", action="store_true")
    args = parser.parse_args()

    from KG.pipeline.factory import build_pipeline
    _pipeline = build_pipeline()
    ingestor = _pipeline["ingestor"]

    dataset = normalize_dataset_name(args.dataset)
    sessions = load_sessions(
        dataset=dataset,
        sessions_jsonl=args.sessions_jsonl,
        dataset_json=args.dataset_json,
    )
    df = sessions_to_one_turn_df(
        sessions,
        make_session_uid=not args.no_session_uid,
        sample_filter=args.sample_index,
    )
    if df.empty:
        raise SystemExit(f"No sessions found for sample_index={args.sample_index}")

    print(f"[INFO] dataset={dataset} sessions(lines)={len(sessions)}")
    print(df[["session_id", "dialogue_datetime"]].head(10).to_string(index=False))

    report = ingest_by_session_one_turn(
        ingestor,
        df,
        prev_k=args.prev_k,
        entity_sim_topk=args.entity_sim_topk,
        entity_sim_threshold=args.entity_sim_threshold,
    )
    print(f"[DONE] sessions_ingested={len(report)}")


if __name__ == "__main__":
    main()
