"""Per-question raw-turn corpus for the grep agent.

每題的 haystack 就是它自己的 script_data CSV。這裡把 CSV 載成 in-memory corpus,
提供 grep / read_window 兩個工具的實作。sid 採 split-embed 慣例:

    user turn      (turn_index t) -> {session}:{t+1}:u
    assistant turn (turn_index t) -> {session}:{t}:a

也接受無後綴的 pair sid `{session}:{p}`(= user :u + assistant :a 兩個 turn)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Turn:
    sid: str            # split sid, e.g. "answer_abc:6:u"
    session_id: str
    turn_index: int
    pos: int            # position within its session (0-based, in CSV order)
    role: str           # "user" / "assistant"
    date: str
    text: str


def _snippet(text: str, match: re.Match | None, width: int = 90) -> str:
    """截取 match 附近 ±width 字元;無 match 時取開頭。"""
    text = " ".join(str(text).split())
    if match is None:
        core = text[: 2 * width]
        return core + ("…" if len(text) > 2 * width else "")
    lo = max(0, match.start() - width)
    hi = min(len(text), match.end() + width)
    return ("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "")


class Corpus:
    def __init__(self, turns: list[Turn]):
        self.turns = turns
        self.by_sid: dict[str, Turn] = {t.sid: t for t in turns}
        self._sessions: dict[str, list[Turn]] = {}
        for t in turns:
            self._sessions.setdefault(t.session_id, []).append(t)

    # ── sid resolution ──────────────────────────────────────────────────
    def resolve(self, sid: str) -> list[Turn]:
        """回傳 sid 對應的 turn(s)。支援 :u/:a split sid 與無後綴 pair sid;
        LLM 常把 session 前綴脫落(answer_555dfb94 → 555dfb94),做模糊補救。"""
        sid = sid.strip()
        if sid in self.by_sid:
            return [self.by_sid[sid]]
        out = [t for suf in (":u", ":a") if (t := self.by_sid.get(sid + suf))]
        if out:
            return out
        # LoCoMo chunk sid → turn 展開:turn 粒度 corpus 的 sid 是 {chunk}t{off},
        # context seed 是 chunk 級({sample}__{sess}:{ci})→ 展開成該 chunk 全部 turn。
        out = [t for t in self.turns if t.sid.startswith(sid + "t")]
        if out:
            return out
        # 模糊補救:session 部分用尾端比對(唯一命中且與原 sid 不同才遞迴,避免自旋)
        sess, _, rest = sid.partition(":")
        if sess and rest:
            matches = {t.session_id for t in self.turns
                       if t.session_id != sess
                       and (t.session_id.endswith("_" + sess) or t.session_id.endswith(sess))}
            if len(matches) == 1:
                fixed = f"{matches.pop()}:{rest}"
                if fixed != sid:
                    return self.resolve(fixed)
        return []

    def normalize_sids(self, sids: list[str]) -> list[str]:
        """展開成存在於 corpus 的 split sid,保序去重。"""
        out: list[str] = []
        seen: set[str] = set()
        for s in sids:
            for t in self.resolve(s):
                if t.sid not in seen:
                    seen.add(t.sid)
                    out.append(t.sid)
        return out

    # ── tools ───────────────────────────────────────────────────────────
    def grep(self, pattern: str, *, max_lines: int = 30, max_chars: int = 8000) -> str:
        """Case-insensitive regex over raw turn text;非法 regex 退回 literal。
        引號幾乎不會是字面目標,先剝掉;整句 pattern 0 命中時退回
        「所有詞都出現在同一 turn」的 AND 搜尋(LLM 很愛丟整句)。"""
        pattern = pattern.replace('"', " ").replace("“", " ").replace("”", " ").strip()
        try:
            pat = re.compile(pattern, re.IGNORECASE)
        except re.error:
            pat = re.compile(re.escape(pattern), re.IGNORECASE)

        def _scan(p: re.Pattern) -> list[tuple["Turn", re.Match, str]]:
            found = []
            for t in self.turns:
                # 日期戳也納入搜尋範圍(GREP 2023/03 可找出三月的 turn)
                haystack = f"[{t.date}] {t.text}" if t.date else t.text
                m = p.search(haystack)
                if m:
                    found.append((t, m, haystack))
            return found

        hits = _scan(pat)
        and_note = ""
        if not hits:
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]{2,}", pattern)]
            if len(words) >= 2:
                word_pats = [re.compile(rf"(?<![A-Za-z0-9]){re.escape(w)}", re.IGNORECASE) for w in words]
                for t in self.turns:
                    haystack = f"[{t.date}] {t.text}" if t.date else t.text
                    ms = [wp.search(haystack) for wp in word_pats]
                    if all(ms):
                        hits.append((t, ms[0], haystack))
                if hits:
                    and_note = f" (exact phrase not found; showing turns containing ALL words: {' '.join(words)})"

        if not hits:
            return (f"grep {pattern!r}: 0 matches. "
                    "Hint: multi-word patterns match as an exact phrase — try ONE rare word, "
                    "or word1.*word2.")

        lines = [f"grep {pattern!r}: {len(hits)} matching turns"
                 + (f" (showing first {max_lines})" if len(hits) > max_lines else "")
                 + and_note]
        for t, m, haystack in hits[:max_lines]:
            lines.append(f"[sid={t.sid}] [{t.date}] {t.role}: {_snippet(haystack, m)}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n…(truncated; narrow your pattern)"
        return out

    def read_window(self, sid: str, k: int = 2, *, max_chars: int = 8000) -> str:
        """顯示 sid 所在 session 中前後 ±k 個 turn 的原文。
        目標 turn 給全文(答案 span 可能埋很深),鄰居 turn 截短。"""
        targets = self.resolve(sid)
        if not targets:
            return f"read {sid!r}: sid not found in this corpus."
        t0 = targets[0]
        sess = self._sessions[t0.session_id]
        lo, hi = max(0, t0.pos - k), min(len(sess), t0.pos + k + 1)
        target_sids = {x.sid for x in targets}
        lines = [f"context around [sid={t0.sid}] (session {t0.session_id}):"]
        for t in sess[lo:hi]:
            is_target = t.sid in target_sids
            marker = ">>" if is_target else "  "
            body = " ".join(t.text.split())
            limit = 3000 if is_target else 400
            if len(body) > limit:
                body = body[:limit] + "…"
            lines.append(f"{marker} [sid={t.sid}] [{t.date}] {t.role}: {body}")
        out = "\n".join(lines)
        if len(out) > max_chars:
            out = out[:max_chars] + "\n…(truncated)"
        return out

    def display_entry(self, sid: str, *, max_chars: int = 4000) -> str | None:
        """組出 Evidence Summary 一行的 snippet 文字(raw turn text)。

        max_chars 不能太小:LongMem 的答案是字面 span,可能埋在長 assistant turn
        的深處,截斷 = 直接毀掉證據(oracle 實測 600 字會讓 assistant 類掉到 64%)。"""
        targets = self.resolve(sid)
        if not targets:
            return None
        parts = []
        for t in targets:
            body = " ".join(t.text.split())
            parts.append(f"{'User' if t.role == 'user' else 'Assistant'} : {body}")
        text = " \n ".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text


def load_corpus(csv_path: str | Path) -> Corpus:
    df = pd.read_csv(csv_path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    turns: list[Turn] = []
    pos_counter: dict[str, int] = {}
    for _, r in df.iterrows():
        content = r.get("content")
        if pd.isna(content) or not str(content).strip():
            continue
        session = str(r["session_id"]).strip()
        turn_idx = int(r["turn_index"])
        role = str(r["role"]).strip().lower()
        pair = turn_idx + 1 if role == "user" else turn_idx
        suffix = "u" if role == "user" else "a"
        pos = pos_counter.get(session, 0)
        pos_counter[session] = pos + 1
        turns.append(Turn(
            sid=f"{session}:{pair}:{suffix}",
            session_id=session,
            turn_index=turn_idx,
            pos=pos,
            role=role,
            date=str(r.get("dialogue_datetime", "") or ""),
            text=str(content),
        ))
    return Corpus(turns)
