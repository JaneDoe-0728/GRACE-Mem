"""Append a traceable Agent Filter run record to EXPERIMENT_LOG.md.

This deliberately records metadata only. It never starts retrieval or changes
an experiment output directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


LOG_PATH = Path(__file__).with_name("EXPERIMENT_LOG.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("longmem", "locomo"))
    parser.add_argument("--model", default="")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", default="filter_fetch")
    parser.add_argument("--answer-model", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--question-range", default="all")
    parser.add_argument("--output", default="")
    parser.add_argument("--status", choices=("planned", "running", "done", "failed"), default="planned")
    parser.add_argument("--command", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--force", action="store_true", help="append another record with the same tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = LOG_PATH.read_text(encoding="utf-8")
    if not args.force and f"- `run_tag`: `{args.run_tag}`" in existing:
        raise SystemExit(f"run tag already exists: {args.run_tag}; use --force only for an intentional duplicate")

    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    fields = [
        ("run_tag", args.run_tag),
        ("description", args.description),
        ("benchmark/model", f"{args.benchmark} / {args.model}".rstrip(" /")),
        ("source", args.source),
        ("mode", args.mode),
        ("answer model", args.answer_model),
        ("judge model", args.judge_model),
        ("question range", args.question_range),
        ("command", args.command),
        ("output", args.output),
        ("status", args.status),
        ("result", ""),
        ("error analysis", ""),
        ("notes", args.notes),
    ]
    lines = [f"\n### {now} · {args.run_tag}\n", ""]
    lines.extend(f"- {key}: {value}\n" for key, value in fields)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)
    print(f"recorded {args.run_tag} in {LOG_PATH}")


if __name__ == "__main__":
    main()
