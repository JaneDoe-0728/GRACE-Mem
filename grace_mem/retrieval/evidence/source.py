"""
RawContextLookup — load raw conversation turn text from script_data CSV files.

Scans all CSVs under a given directory tree, builds an in-memory index keyed by
(session_id, turn_index), and reconstructs the pre-compression turn text that the
Compressor would have produced for each summary_id.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path


class RawContextLookup:
    """
    Lazy-loaded index of raw conversation turns from script_data CSV files.

    Usage:
        lookup = RawContextLookup("/path/to/script_data")
        text = lookup.get(session_id="b10f3828_1", message_id=2)
        # → "User: ...\nAssistant: ..."
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)
        # session_id → list of {turn_index, role, content} sorted ascending by turn_index
        self._index: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        """Scan all CSV files and build the index (called once, thread-safe)."""
        index: dict[str, list[dict]] = {}
        for csv_path in self._data_dir.rglob("*.csv"):
            try:
                with open(csv_path, newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sid = (row.get("session_id") or "").strip()
                        role = (row.get("role") or "").strip().lower()
                        content = (row.get("content") or "").strip()
                        try:
                            turn_index = int(row.get("turn_index", 0))
                        except (ValueError, TypeError):
                            continue
                        if not sid or not role:
                            continue
                        if sid not in index:
                            index[sid] = []
                        index[sid].append({"turn_index": turn_index, "role": role, "content": content})
            except Exception:
                continue

        # Sort each session's turns by turn_index
        for turns in index.values():
            turns.sort(key=lambda r: r["turn_index"])

        self._index = index
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load()

    def get(self, session_id: str, message_id: int) -> str | None:
        """
        Reconstruct the raw curr_text for a given (session_id, message_id).

        message_id matches the assistant's turn_index for a user+assistant pair,
        or the user's turn_index for an orphaned user turn.

        Returns None if session_id or message_id is not found.
        """
        self._ensure_loaded()
        turns = self._index.get(str(session_id))
        if not turns:
            return None

        # Build a map turn_index → row for this session
        by_index = {r["turn_index"]: r for r in turns}
        row = by_index.get(int(message_id))
        if row is None:
            return None

        if row["role"] == "assistant":
            assistant_text = row["content"]
            # Find the most recent user turn before this assistant turn
            user_text = ""
            for r in reversed(turns):
                if r["turn_index"] < int(message_id) and r["role"] == "user":
                    user_text = r["content"]
                    break
            if not user_text:
                return assistant_text.strip() or None
            return f"User: {user_text.strip()}\nAssistant: {assistant_text.strip()}"
        else:
            # Orphaned user turn
            user_text = row["content"]
            return user_text.strip() or None

    def get_user_text(self, session_id: str, message_id: int) -> str | None:
        """Return only the raw user turn text preceding the assistant at message_id."""
        self._ensure_loaded()
        turns = self._index.get(str(session_id))
        if not turns:
            return None
        by_index = {r["turn_index"]: r for r in turns}
        row = by_index.get(int(message_id))
        if row is None:
            return None
        if row["role"] == "assistant":
            for r in reversed(turns):
                if r["turn_index"] < int(message_id) and r["role"] == "user":
                    return r["content"].strip() or None
            return None
        return row["content"].strip() or None

    def get_assistant_text(self, session_id: str, message_id: int) -> str | None:
        """Return only the raw assistant turn text at message_id."""
        self._ensure_loaded()
        turns = self._index.get(str(session_id))
        if not turns:
            return None
        by_index = {r["turn_index"]: r for r in turns}
        row = by_index.get(int(message_id))
        if row is None or row["role"] != "assistant":
            return None
        return row["content"].strip() or None
