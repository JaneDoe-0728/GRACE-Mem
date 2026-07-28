from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd


class IngestStage:
    """Conversation ingestion stage for LongMem datasets."""

    def normalize_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        need_cols = {"session_id", "turn_index", "role", "content", "dialogue_datetime"}
        miss = need_cols - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns: {sorted(miss)}")

        out = df.copy()
        out["session_id"] = out["session_id"].astype(str)
        out["turn_index"] = pd.to_numeric(out["turn_index"], errors="coerce").fillna(0).astype(int)
        out["role"] = out["role"].astype(str).str.lower().str.strip()
        out["content"] = out["content"].astype(str)
        out["dialogue_datetime"] = out["dialogue_datetime"].astype(str).str.strip()
        return out[["session_id", "turn_index", "role", "content", "dialogue_datetime"]]

    def ingest_by_turn_pairs(
        self,
        ingestor,
        df: pd.DataFrame,
        *,
        prev_k: Optional[int] = None,
        entity_sim_topk: Optional[int] = None,
        entity_sim_threshold: Optional[float] = None,
        ignore_trailing_user_without_reply: bool = True,
    ) -> dict:
        data = self.normalize_sessions(df)
        report = defaultdict(list)

        for sid, group in data.groupby("session_id", sort=False):
            rows = group.sort_values("turn_index").to_dict("records")
            dialogue_datetime = group["dialogue_datetime"].iloc[0]

            pending_user = None
            for row in rows:
                if row["role"] == "user":
                    if pending_user is None:
                        pending_user = {"turn_index": row["turn_index"], "content": row["content"]}
                    else:
                        if not ignore_trailing_user_without_reply:
                            result = ingestor.summarize_and_ingest_turn(
                                session_id=sid,
                                message_id=pending_user["turn_index"],
                                user_text=pending_user["content"],
                                assistant_text="",
                                prev_k=prev_k,
                                entity_sim_topk=entity_sim_topk,
                                entity_sim_threshold=entity_sim_threshold,
                                dialogue_datetime=dialogue_datetime,
                            )
                            report[sid].append({"pair": (pending_user["turn_index"], None), "result": result})
                        pending_user = {"turn_index": row["turn_index"], "content": row["content"]}
                elif row["role"] == "assistant":
                    if pending_user is not None:
                        result = ingestor.summarize_and_ingest_turn(
                            session_id=sid,
                            message_id=row["turn_index"],
                            user_text=pending_user["content"],
                            assistant_text=row["content"],
                            prev_k=prev_k,
                            entity_sim_topk=entity_sim_topk,
                            entity_sim_threshold=entity_sim_threshold,
                            dialogue_datetime=dialogue_datetime,
                        )
                        report[sid].append({"pair": (pending_user["turn_index"], row["turn_index"]), "result": result})
                        pending_user = None
                    else:
                        result = ingestor.summarize_and_ingest_turn(
                            session_id=sid,
                            message_id=row["turn_index"],
                            user_text="",
                            assistant_text=row["content"],
                            prev_k=prev_k,
                            entity_sim_topk=entity_sim_topk,
                            entity_sim_threshold=entity_sim_threshold,
                            dialogue_datetime=dialogue_datetime,
                        )
                        report[sid].append({"pair": (None, row["turn_index"]), "result": result})

            if pending_user is not None and not ignore_trailing_user_without_reply:
                result = ingestor.summarize_and_ingest_turn(
                    session_id=sid,
                    message_id=pending_user["turn_index"],
                    user_text=pending_user["content"],
                    assistant_text="",
                    prev_k=prev_k,
                    entity_sim_topk=entity_sim_topk,
                    entity_sim_threshold=entity_sim_threshold,
                    dialogue_datetime=dialogue_datetime,
                )
                report[sid].append({"pair": (pending_user["turn_index"], None), "result": result})

        return dict(report)

    def ingest_by_session(
        self,
        ingestor,
        df: pd.DataFrame,
        *,
        prev_k: Optional[int] = None,
        entity_sim_topk: Optional[int] = None,
        entity_sim_threshold: Optional[float] = None,
    ) -> dict:
        data = self.normalize_sessions(df)
        results = {}

        for sid, group in data.groupby("session_id", sort=False):
            dialogue_lines = []
            for _, row in group.iterrows():
                dialogue_lines.append(f"{row['role']}: {row['content'].strip()}")

            dialogue_text = "\n".join(dialogue_lines)
            message_id = int(group["turn_index"].max())
            dialogue_datetime = group["dialogue_datetime"].iloc[0]

            result = ingestor.summarize_and_ingest_turn(
                session_id=sid,
                message_id=message_id,
                user_text=dialogue_text,
                assistant_text="",
                prev_k=prev_k,
                entity_sim_topk=entity_sim_topk,
                entity_sim_threshold=entity_sim_threshold,
                dialogue_datetime=dialogue_datetime,
            )

            results[sid] = {
                "message_id": message_id,
                "turns": len(group),
                "result": result,
            }

        return results
