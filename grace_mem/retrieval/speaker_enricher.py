"""Enrich Evidence Summary blocks with speaker hints from raw dialogue context."""
from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SID_RE = re.compile(r"\[sid=([^\]]+)\]")
SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ._'’-]{0,60}|user|assistant)\s*:\s*(.*)$", re.IGNORECASE)
EVIDENCE_BULLET_RE = re.compile(
    r"^(?P<prefix>\s*• .*?\[sid=(?P<sid>[^\]]+)\].*?\[score=[^\]]+\](?:\[speakers=[^\]]+\])?\s*)(?P<snippet>.*)$"
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SpeakerTurn:
    """One raw utterance with a speaker label."""

    speaker: str
    text: str


def tokenize(text: str) -> set[str]:
    """Return coarse tokens for fuzzy matching compressed summaries to raw turns."""
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text or "")}


def parse_speaker_turns(raw_text: str) -> list[SpeakerTurn]:
    """Parse ``Name: utterance`` lines from raw dialogue text.

    Continuation lines are attached to the previous speaker. Lines without a
    speaker prefix are ignored unless they continue a previous speaker turn.
    """
    turns: list[SpeakerTurn] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_lines
        if current_speaker and current_lines:
            text = "\n".join(line for line in current_lines if line).strip()
            if text:
                turns.append(SpeakerTurn(current_speaker, text))
        current_speaker = None
        current_lines = []

    for line in (raw_text or "").splitlines():
        match = SPEAKER_LINE_RE.match(line)
        if match:
            flush()
            current_speaker = match.group(1).strip()
            current_lines = [match.group(2).strip()]
        elif current_speaker:
            current_lines.append(line.strip())

    flush()
    return turns


def infer_speakers(
    summary_text: str,
    raw_text: str,
    *,
    min_score: float = 0.12,
    max_speakers: int = 4,
) -> list[str]:
    """Infer which raw dialogue speakers are represented in a compressed summary.

    The score is based on token overlap between the compressed summary and each
    raw utterance. This intentionally uses a simple deterministic heuristic so
    the enrichment can run offline without another LLM call.
    """
    summary_tokens = tokenize(summary_text)
    if not summary_tokens:
        return []

    best_by_speaker: dict[str, float] = {}
    for turn in parse_speaker_turns(raw_text):
        turn_tokens = tokenize(turn.text)
        if not turn_tokens:
            continue
        overlap = len(summary_tokens & turn_tokens)
        denom = max(1, min(len(summary_tokens), len(turn_tokens)))
        score = overlap / denom
        if turn.speaker.lower() in (summary_text or "").lower():
            score = max(score, min_score)
        if score >= min_score:
            best_by_speaker[turn.speaker] = max(best_by_speaker.get(turn.speaker, 0.0), score)

    ranked = sorted(best_by_speaker.items(), key=lambda item: (-item[1], item[0].lower()))
    return [speaker for speaker, _ in ranked[:max_speakers]]


def _ordered_speakers(turns: Iterable[SpeakerTurn]) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        key = turn.speaker.lower()
        if key not in seen:
            speakers.append(turn.speaker)
            seen.add(key)
    return speakers


def _speaker_from_line(line: str, known_speakers: list[str]) -> tuple[str | None, str]:
    """Return explicit leading speaker and line text without the speaker marker."""
    stripped = line.strip()
    colon_match = SPEAKER_LINE_RE.match(stripped)
    if colon_match:
        speaker = colon_match.group(1).strip()
        return speaker, colon_match.group(2).strip()

    for speaker in sorted(known_speakers, key=len, reverse=True):
        # The fullwidth colon is corpus data, not prose: transcripts carry it as a
        # speaker separator, so the class must keep matching it.  # allow-cjk
        match = re.match(rf"^\s*{re.escape(speaker)}(?:\s*[,!:：]|[\s]+)(?P<rest>.*)$", stripped, re.IGNORECASE)
        if match:
            return speaker, match.group("rest").strip()

    return None, stripped


def _best_overlap_speaker(line: str, raw_turns: list[SpeakerTurn], min_score: float) -> str | None:
    line_tokens = tokenize(line)
    if not line_tokens:
        return None
    best_speaker = None
    best_score = 0.0
    for turn in raw_turns:
        turn_tokens = tokenize(turn.text)
        if not turn_tokens:
            continue
        overlap = len(line_tokens & turn_tokens)
        denom = max(1, min(len(line_tokens), len(turn_tokens)))
        score = overlap / denom
        if score > best_score:
            best_score = score
            best_speaker = turn.speaker
    return best_speaker if best_score >= min_score else None


def _alternate_speaker(previous_speaker: str | None, ordered_speakers: list[str]) -> str | None:
    if not previous_speaker or len(ordered_speakers) != 2:
        return None
    if previous_speaker.lower() == ordered_speakers[0].lower():
        return ordered_speakers[1]
    if previous_speaker.lower() == ordered_speakers[1].lower():
        return ordered_speakers[0]
    return None


def annotate_summary_with_speakers(
    summary_text: str,
    raw_text: str,
    *,
    min_score: float = 0.3,
) -> str:
    """Prefix each summary utterance line with ``[Speaker]`` when possible.

    The primary signal is an explicit speaker marker in either the summary or raw
    dialogue. If compression removed the marker, the function falls back to raw
    token overlap, then to two-person dialogue alternation.
    """
    raw_turns = parse_speaker_turns(raw_text)
    known_speakers = _ordered_speakers(raw_turns) or _ordered_speakers(parse_speaker_turns(summary_text))
    if not known_speakers:
        return summary_text
    raw_is_summary = (raw_text or "").strip() == (summary_text or "").strip()

    annotated: list[str] = []
    previous_speaker: str | None = None
    for raw_line in (summary_text or "").splitlines():
        if not raw_line.strip() or re.match(r"^\s*\[[^\]]+\]\s+", raw_line):
            annotated.append(raw_line)
            continue

        explicit_speaker, cleaned_line = _speaker_from_line(raw_line, known_speakers)
        speaker = explicit_speaker
        if speaker is None and not raw_is_summary:
            speaker = _best_overlap_speaker(cleaned_line, raw_turns, min_score)
        if speaker is None:
            speaker = _alternate_speaker(previous_speaker, known_speakers)

        if speaker:
            previous_speaker = speaker
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            annotated.append(f"{indent}[{speaker}] {cleaned_line}")
        else:
            annotated.append(raw_line)

    return "\n".join(annotated)


def _insert_speakers_tag(line: str, speakers: list[str]) -> str:
    if not speakers or "[speakers=" in line:
        return line
    tag = f"[speakers={', '.join(speakers)}]"
    score_pos = line.find("[score=")
    if score_pos >= 0:
        score_end = line.find("]", score_pos)
        if score_end >= 0:
            return f"{line[:score_end + 1]}{tag}{line[score_end + 1:]}"
    sid_match = SID_RE.search(line)
    if sid_match:
        return f"{line[:sid_match.end()]}{tag}{line[sid_match.end():]}"
    return f"{tag} {line}"


def enrich_evidence_summary(
    evidence_text: str,
    raw_by_sid: Mapping[str, str],
    *,
    min_score: float = 0.12,
    max_speakers: int = 4,
) -> str:
    """Add speaker tags to Evidence Summary bullet blocks."""
    def raw_for_sid(sid: str) -> str | None:
        raw_text = raw_by_sid.get(sid)
        if raw_text is None and ":" in sid:
            raw_text = raw_by_sid.get(sid.rsplit(":", 1)[0])
        return raw_text

    def flush_block(block_lines: list[str]) -> None:
        if not block_lines:
            return
        first = block_lines[0]
        match = EVIDENCE_BULLET_RE.match(first)
        if not match:
            enriched_lines.extend(block_lines)
            return

        sid = match.group("sid").strip()
        raw_text = raw_for_sid(sid)
        if not raw_text:
            enriched_lines.extend(block_lines)
            return

        snippet_lines = [match.group("snippet")]
        snippet_lines.extend(block_lines[1:])
        snippet = "\n".join(snippet_lines)
        annotated_snippet = annotate_summary_with_speakers(snippet, raw_text, min_score=max(min_score, 0.3))
        speakers = infer_speakers(snippet, raw_text, min_score=min_score, max_speakers=max_speakers)
        prefix = _insert_speakers_tag(match.group("prefix").rstrip(), speakers)
        enriched_lines.append(f"{prefix} {annotated_snippet}" if annotated_snippet else prefix)

    enriched_lines: list[str] = []
    current_block: list[str] = []
    for line in (evidence_text or "").splitlines():
        if EVIDENCE_BULLET_RE.match(line):
            flush_block(current_block)
            current_block = [line]
            continue

        if current_block:
            if line.startswith("  • "):
                flush_block(current_block)
                current_block = [line]
            else:
                current_block.append(line)
            continue

        sid_match = SID_RE.search(line)
        if sid_match is None:
            enriched_lines.append(line)
            continue

        sid = sid_match.group(1).strip()
        raw_text = raw_for_sid(sid)
        if raw_text:
            speakers = infer_speakers(line, raw_text, min_score=min_score, max_speakers=max_speakers)
            enriched_lines.append(_insert_speakers_tag(line, speakers))
        else:
            enriched_lines.append(line)

    flush_block(current_block)

    return "\n".join(enriched_lines)


def build_raw_map_from_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Build ``summary_id -> raw dialogue`` map from LongMem-style CSV rows.

    Expected columns are ``session_id``, ``turn_index``, ``role``, and
    ``content``. User/assistant pairs are keyed by the assistant turn index,
    matching the ingestion path that stores summary IDs as ``session_id:message_id``.
    A session-level key is also emitted for by-session ingestion.
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        sid = str(row.get("session_id", "")).strip()
        if not sid:
            continue
        grouped.setdefault(sid, []).append(dict(row))

    raw_by_sid: dict[str, str] = {}
    for sid, items in grouped.items():
        normalized: list[dict[str, Any]] = []
        for row in items:
            try:
                turn_index = int(float(str(row.get("turn_index", "0")).strip() or "0"))
            except ValueError:
                turn_index = 0
            normalized.append(
                {
                    "turn_index": turn_index,
                    "role": str(row.get("role", "")).strip().lower(),
                    "content": str(row.get("content", "")).strip(),
                }
            )

        normalized.sort(key=lambda row: row["turn_index"])
        session_lines = [f"{row['role']}: {row['content']}" for row in normalized if row["role"] and row["content"]]
        if session_lines:
            raw_by_sid[sid] = "\n".join(session_lines)
            raw_by_sid[f"{sid}:{normalized[-1]['turn_index']}"] = raw_by_sid[sid]

        pending_user: dict[str, object] | None = None
        for row in normalized:
            role = str(row["role"])
            if role == "user":
                pending_user = row
            elif role == "assistant":
                lines = []
                if pending_user and pending_user.get("content"):
                    lines.append(f"user: {pending_user['content']}")
                if row.get("content"):
                    lines.append(f"assistant: {row['content']}")
                if lines:
                    raw_by_sid[f"{sid}:{row['turn_index']}"] = "\n".join(lines)
                pending_user = None

    return raw_by_sid


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def enrich_context_csv(
    input_csv: Path,
    output_csv: Path,
    raw_by_sid: Mapping[str, str],
    *,
    context_column: str,
    min_score: float,
    max_speakers: int,
) -> int:
    rows = read_csv_rows(input_csv)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return 0
    if context_column not in rows[0]:
        raise ValueError(f"Context column not found: {context_column}")

    for row in rows:
        row[context_column] = enrich_evidence_summary(
            row.get(context_column, ""),
            raw_by_sid,
            min_score=min_score,
            max_speakers=max_speakers,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_raw_map(paths: Iterable[Path]) -> dict[str, str]:
    raw_by_sid: dict[str, str] = {}
    for path in paths:
        raw_by_sid.update(build_raw_map_from_rows(read_csv_rows(path)))
    return raw_by_sid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add [speakers=...] tags to Evidence Summary lines by comparing them with raw dialogue CSVs."
    )
    parser.add_argument("--raw-csv", action="append", type=Path, required=True, help="Raw dialogue CSV path.")
    parser.add_argument("--input-text", type=Path, help="Text file containing an Evidence Summary block.")
    parser.add_argument("--output-text", type=Path, help="Where to write enriched text.")
    parser.add_argument("--input-csv", type=Path, help="CSV containing retrieved context/evidence.")
    parser.add_argument("--output-csv", type=Path, help="Where to write enriched CSV.")
    parser.add_argument("--context-column", default="Retrieved_Context", help="CSV column to enrich.")
    parser.add_argument("--min-score", type=float, default=0.12, help="Minimum token-overlap score.")
    parser.add_argument("--max-speakers", type=int, default=4, help="Maximum speakers per evidence line.")
    args = parser.parse_args()

    raw_by_sid = load_raw_map(args.raw_csv)
    if not raw_by_sid:
        raise SystemExit("No raw dialogue rows were loaded.")

    if args.input_csv:
        if not args.output_csv:
            raise SystemExit("--output-csv is required with --input-csv")
        count = enrich_context_csv(
            args.input_csv,
            args.output_csv,
            raw_by_sid,
            context_column=args.context_column,
            min_score=args.min_score,
            max_speakers=args.max_speakers,
        )
        print(f"Enriched {count} CSV rows -> {args.output_csv}")

    if args.input_text:
        text = args.input_text.read_text(encoding="utf-8")
        enriched = enrich_evidence_summary(
            text,
            raw_by_sid,
            min_score=args.min_score,
            max_speakers=args.max_speakers,
        )
        if args.output_text:
            args.output_text.parent.mkdir(parents=True, exist_ok=True)
            args.output_text.write_text(enriched, encoding="utf-8")
            print(f"Enriched text -> {args.output_text}")
        else:
            print(enriched)

    if not args.input_csv and not args.input_text:
        raise SystemExit("Provide --input-text or --input-csv.")


if __name__ == "__main__":
    main()
