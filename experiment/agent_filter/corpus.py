"""Per-question raw-turn corpus for the grep agent.

Each question's haystack is its own script_data CSV. This module loads that CSV
into an in-memory corpus and implements the two tools, grep and read_window.
sids follow the split-embed convention:

    user turn      (turn_index t) -> {session}:{t+1}:u
    assistant turn (turn_index t) -> {session}:{t}:a

A suffix-less pair sid `{session}:{p}` is also accepted (= the user :u turn plus
the assistant :a turn).
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
    """Take +/-width characters around the match, or the beginning when there is
    no match."""
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
        """Return the turn(s) for a sid. Handles :u/:a split sids and suffix-less
        pair sids, with a fuzzy fallback because the LLM often drops the session
        prefix (answer_555dfb94 -> 555dfb94)."""
        sid = sid.strip()
        if sid in self.by_sid:
            return [self.by_sid[sid]]
        out = [t for suf in (":u", ":a") if (t := self.by_sid.get(sid + suf))]
        if out:
            return out
        # Expanding a LoCoMo chunk sid into turns: a turn-granularity corpus uses
        # sids of the form {chunk}t{off}, while the context seed is chunk-level
        # ({sample}__{sess}:{ci}), so expand it to every turn in that chunk.
        out = [t for t in self.turns if t.sid.startswith(sid + "t")]
        if out:
            return out
        # Fuzzy fallback: match the session part by suffix. Recurse only on a unique
        # match that differs from the original sid, so this cannot spin.
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
        """Expand to the split sids that exist in the corpus, order-preserving and
        deduplicated."""
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
        """Case-insensitive regex over the raw turn text; an invalid regex falls
        back to a literal search.
        Quotes are almost never the literal target, so they are stripped first. When
        a whole-sentence pattern matches nothing, fall back to an AND search --
        every word appearing somewhere in the same turn -- because the LLM loves to
        throw an entire sentence at it."""
        pattern = pattern.replace('"', " ").replace("“", " ").replace("”", " ").strip()
        try:
            pat = re.compile(pattern, re.IGNORECASE)
        except re.error:
            pat = re.compile(re.escape(pattern), re.IGNORECASE)

        def _scan(p: re.Pattern) -> list[tuple["Turn", re.Match, str]]:
            found = []
            for t in self.turns:
                # Date stamps are searchable too (GREP 2023/03 finds March's turns)
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
        """Show the raw text of the +/-k turns around the sid within its session.
        The target turn is given in full, since the answer span can be buried deep;
        the neighbouring turns are truncated."""
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
        """Build the snippet text for one Evidence Summary line, from raw turn text.

        max_chars must not be small: LongMem answers are literal spans that can sit
        deep inside a long assistant turn, so truncating destroys the evidence
        outright (measured against the oracle, 600 characters dropped the assistant
        category to 64%)."""
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
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
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
