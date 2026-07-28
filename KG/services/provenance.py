# services/provenance.py
class Provenance:
    """
    負責追蹤與合併實體/關係的來源資訊 (provenance)，
    主要處理事件記錄的格式統一與合併，方便後續追溯來源。
    """
    @staticmethod
    def prov_to_events(prov: dict) -> list[dict]:
        """
        將不同格式的 provenance 字典轉換為統一的事件列表：
        - 支援 prov["events"] 已是列表的情況
        - 或支援 summary_ids / session_id / message_id / dialogue_datetime 的簡單格式
        - 輸出標準化事件：{ts, summary_id, session_id, message_id, dialogue_datetime}
        """
        if not prov: return []
        if isinstance(prov.get("events"), list):
            evs = []
            for e in prov["events"]:
                sess = e.get("session_id") or e.get("session_id ") or e.get("sess_id")
                msg  = e.get("message_id") or e.get("msg_id")
                dt = e.get("dialogue_datetime")
                evs.append({
                    "ts": e.get("ts", 0),
                    "summary_id": e.get("summary_id"),
                    "session_id": None if sess is None else str(sess),
                    "message_id": None if msg is None else str(msg),
                    "dialogue_datetime": dt
                })
            return evs
        summary_ids = prov.get("summary_ids") or []
        if isinstance(summary_ids, str): summary_ids = [summary_ids]
        sess = prov.get("session_id")
        msg = prov.get("message_id")
        dt = prov.get("dialogue_datetime")  # NEW: Extract dialogue_datetime
        sess = None if sess is None else str(sess)
        msg = None if msg is None else str(msg)
        evs = []
        for idx, sid in enumerate(summary_ids):
            s_sess, s_msg = sess, msg
            if (s_sess is None or s_msg is None) and isinstance(sid, str) and ":" in sid:
                try: p_sess, p_msg = sid.split(":", 1); s_sess = s_sess or str(p_sess); s_msg = s_msg or str(p_msg)
                except Exception: pass
            evs.append({
                "ts": idx,
                "summary_id": sid,
                "session_id": s_sess,
                "message_id": s_msg,
                "dialogue_datetime": dt  # NEW: Add dialogue_datetime to event
            })
        return evs

    @staticmethod
    def merge_prov(old: dict | None, new: dict | None, max_events: int = 50) -> dict:
        """
        合併舊與新的 provenance：
        - 去重（同 session_id/message_id/summary_id）
        - 依照 ts 排序，僅保留最近 max_events 筆
        - 輸出格式：{"events": [...]}，方便附加到 entity/relationship meta
        """
        def _to_events(x: dict | None) -> list[dict]:
            """Convert one provenance blob into normalized events before merging."""
            return Provenance.prov_to_events(x or {})
        def k(e: dict) -> tuple[str | None, str | None, str | None]:
            """Build the deduplication key used when merging provenance events."""
            return (e.get("session_id"), e.get("message_id"), e.get("summary_id"))
        merged = {k(e): e for e in _to_events(old)}
        for e in _to_events(new): merged[k(e)] = e
        events = sorted(merged.values(), key=lambda e: e.get("ts", 0))  # [-max_events:]
        return {"events": events}
