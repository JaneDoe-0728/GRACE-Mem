# locamal_ingest.py
# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import argparse

import pandas as pd

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.helpers.dataset import build_session_records_from_json, normalize_dataset_name, resolve_dataset_path
from experiment.locomo.utils.io import load_jsonl_records
from experiment.experiment_config import INGEST_PARAMS


# ========= Config: edit here =========
INPUT_JSONL = "data/locomo_by_session.jsonl"   # one session per line (a session dict)
MAKE_SESSION_UID = True             # session_id = f"{sample_index}__{session_id}"
PREV_K = 2
ENTITY_SIM_TOPK = 4
ENTITY_SIM_THRESHOLD = 0.5

DIALOGUE_JOINER = "\n"              # keep each utterance on its own line
PUT_SPEAKER_PREFIX = True           # keep "Caroline: ..." in text if present

# Turns per summary chunk. Single source of truth is INGEST_PARAMS["chunk_turns"] in
# experiment/experiment_config.py; it is threaded down as an explicit argument (never
# read from the environment) so the orchestrator, each worker subprocess and the
# snapshot builder cannot silently disagree about the chunk size.
CHUNK_TURNS = int(INGEST_PARAMS.get("chunk_turns", 8) or 0)


def _iter_dialogue_chunks(dialogue: List[str], chunk_turns: Optional[int] = None):
    """Yield (message_id, chunk_lines).

    chunk_turns > 0 → consecutive windows of that many turns, message_id = chunk index.
    chunk_turns <= 0 → the whole session as one chunk (message_id = 0), i.e. the
    pre-chunking behaviour. None falls back to the configured default.

    Empty dialogue is the one place the two modes differ: chunked mode yields nothing
    (no summary is written for a session with no turns), while chunk_turns<=0 yields a
    single empty chunk. That asymmetry is deliberate — chunk_turns<=0 exists to
    reproduce pre-chunking runs byte for byte, so its behaviour is frozen.
    """
    n = CHUNK_TURNS if chunk_turns is None else int(chunk_turns)
    if n > 0:
        for start in range(0, len(dialogue), n):
            yield start // n, dialogue[start:start + n]
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
    chunk_turns: Optional[int] = None,
) -> pd.DataFrame:
    """
    每個 session -> 一個或多個 chunk，每個 chunk 一個 turn：
      - session_id: 唯一（建議 sample_index__session_id）
      - message_id: chunk 索引（chunk_turns<=0 時固定 0，即整個 session 一塊）
      - dialogue_datetime: date_time
      - user_text: 該 chunk 的 dialogue (A/B 兩人對話原樣串起來)
      - assistant_text: 空字串
    """
    rows: List[Dict[str, Any]] = []
    for s in sessions:
        sample_index = s.get("sample_index")
        if sample_filter is not None and int(sample_index) != sample_filter:
            continue
        sess_id = s.get("session_id")
        dialogue = ["" if x is None else str(x) for x in (s.get("dialogue", []) or [])]
        for message_id, chunk in _iter_dialogue_chunks(dialogue, chunk_turns):
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
    chunk_turns: Optional[int] = None,
) -> pd.DataFrame:
    """Build a one-turn-per-chunk DataFrame from a list of session record dicts.

    Session UIDs are ``<conv_id>__<session_id>`` so they are unique per conversation
    and do not collide across different source conversations. ``chunk_turns`` must
    match the value used by the run that produced the artifacts being restored,
    otherwise the resulting summary_ids will not line up.
    """
    rows: List[Dict[str, Any]] = []
    for rec in records:
        sess_id = rec.get("session_id")
        dialogue = ["" if x is None else str(x) for x in (rec.get("dialogue", []) or [])]
        for message_id, chunk in _iter_dialogue_chunks(dialogue, chunk_turns):
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
        chunk_turns: Optional[int] = None,
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
        self.chunk_turns = chunk_turns

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
            chunk_turns=self.chunk_turns,
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
    parser.add_argument("--chunk-turns", type=int, default=CHUNK_TURNS,
                        help="Turns per ingest chunk (0 = whole session as one chunk)")
    parser.add_argument("--no-session-uid", action="store_true")
    args = parser.parse_args()

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
        chunk_turns=args.chunk_turns,
    )
    if df.empty:
        raise SystemExit(f"No sessions found for sample_index={args.sample_index}")

    print(f"[INFO] dataset={dataset} sessions(lines)={len(sessions)}")
    print(df[["session_id", "dialogue_datetime"]].head(10).to_string(index=False))

    from grace_mem.pipeline.factory import build_pipeline

    with build_pipeline() as runtime:
        report = ingest_by_session_one_turn(
            runtime.ingestor,
            df,
            prev_k=args.prev_k,
            entity_sim_topk=args.entity_sim_topk,
            entity_sim_threshold=args.entity_sim_threshold,
        )
    print(f"[DONE] sessions_ingested={len(report)}")


if __name__ == "__main__":
    main()
