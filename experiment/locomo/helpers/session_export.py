"""
Convert dataset samples into by-session conversational records.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.helpers.dataset import (
    get_sample_conversation,
    get_sample_speakers,
    load_raw_samples,
    normalize_dataset_name,
    resolve_dataset_path,
)
from experiment.locomo.utils.io import append_jsonl_record, append_text, ensure_dir, remove_if_exists

SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
SESSION_DT_RE = re.compile(r"^session_(\d+)_date_time$")


def extract_sessions(conv: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sessions: Dict[str, Dict[str, Any]] = {}
    for key, value in conv.items():
        match = SESSION_KEY_RE.match(key)
        if match and isinstance(value, list):
            session_id = match.group(1)
            sessions[session_id] = {"turns": value, "date_time": None}
    for key, value in conv.items():
        match = SESSION_DT_RE.match(key)
        if match and match.group(1) in sessions:
            sessions[match.group(1)]["date_time"] = value
    return sessions


def build_lines(turns: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip().replace("\n", " ")
        caption = str(turn.get("blip_caption", "")).strip()
        if not speaker and not text and not caption:
            continue
        suffix = f" (Image: {caption})" if caption else ""
        if not text and caption:
            lines.append(f"{speaker}:{suffix}")
        else:
            lines.append(f"{speaker}: {text}{suffix}")
    return lines


def convert(in_path: Path, out_jsonl: Path, out_txt: Path) -> None:
    samples = load_raw_samples(in_path)
    ensure_dir(out_jsonl.parent)
    ensure_dir(out_txt.parent)
    remove_if_exists(out_jsonl)
    remove_if_exists(out_txt)

    for sample_index, sample in enumerate(samples):
        conv = get_sample_conversation(sample)
        speaker_a, speaker_b = get_sample_speakers(conv)
        append_text(out_txt, f"### Sample {sample_index}\n")
        append_text(out_txt, f"Speakers: {speaker_a or '?'} & {speaker_b or '?'}\n\n")

        sessions = extract_sessions(conv)
        for session_id in sorted(sessions.keys(), key=lambda value: int(value)):
            session = sessions[session_id]
            lines = build_lines(session["turns"])
            record = {
                "sample_index": sample_index,
                "session_id": int(session_id),
                "date_time": session.get("date_time"),
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "dialogue": lines,
            }
            append_jsonl_record(out_jsonl, record)

            header = f"Session {session_id}"
            if session.get("date_time"):
                header += f" ({session['date_time']})"
            append_text(out_txt, header + ":\n")
            for line in lines:
                append_text(out_txt, f"  {line}\n")
            append_text(out_txt, "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dataset JSON to by-session conversational records")
    parser.add_argument("--dataset", choices=["locomo"], default="locomo")
    parser.add_argument("-i", "--input", type=Path, default=None, help="Defaults are resolved from --dataset")
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--out-txt", type=Path, default=None)
    args = parser.parse_args()

    try:
        dataset = normalize_dataset_name(args.dataset)
        input_path = resolve_dataset_path(dataset=dataset, kind="qa_json", explicit_path=args.input)
        output_stem = dataset.replace("-", "_")
        out_jsonl = args.out_jsonl or (input_path.parent / f"{output_stem}_by_session.jsonl")
        out_txt = args.out_txt or (input_path.parent / f"{output_stem}_by_session.txt")
        convert(input_path, out_jsonl, out_txt)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Wrote: {out_jsonl}")
    print(f"[OK] Wrote: {out_txt}")


if __name__ == "__main__":
    main()
