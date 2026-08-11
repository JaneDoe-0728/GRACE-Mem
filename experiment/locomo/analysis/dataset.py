from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.helpers.dataset import category_to_label
from experiment.locomo.utils.io import load_json_records


@dataclass(frozen=True)
class ConversationStats:
    conv_id: str
    session_count: int
    turn_count: int
    qa_count: int
    category_counts: Counter[str]

def session_sort_key(key: str) -> tuple[int, str]:
    try:
        return (int(key.split("_")[1]), key)
    except (IndexError, ValueError):
        return (10**9, key)


def extract_sessions(sample: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        return []

    sessions: list[tuple[str, list[Any]]] = []
    for key in sorted(conversation.keys(), key=session_sort_key):
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        turns = conversation.get(key)
        if isinstance(turns, list):
            sessions.append((key, turns))
    return sessions


def extract_qa_items(sample: dict[str, Any]) -> list[dict[str, Any]]:
    qa_items = sample.get("qa")
    if not isinstance(qa_items, list):
        return []
    return [item for item in qa_items if isinstance(item, dict)]


def conversation_id(sample: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "conversation_id", "id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"sample_{index}"


def analyze_samples(samples: list[dict[str, Any]]) -> tuple[list[ConversationStats], Counter[str], dict[str, int]]:
    conv_stats: list[ConversationStats] = []
    overall_categories: Counter[str] = Counter()
    irregular = {
        "missing_conversation": 0,
        "missing_qa_list": 0,
        "empty_sessions": 0,
        "empty_qa": 0,
    }

    for index, sample in enumerate(samples):
        conv_id = conversation_id(sample, index)
        sessions = extract_sessions(sample)
        qa_items = extract_qa_items(sample)

        if not isinstance(sample.get("conversation"), dict):
            irregular["missing_conversation"] += 1
        if not isinstance(sample.get("qa"), list):
            irregular["missing_qa_list"] += 1
        if not sessions:
            irregular["empty_sessions"] += 1
        if not qa_items:
            irregular["empty_qa"] += 1

        turn_count = 0
        for _, turns in sessions:
            turn_count += sum(1 for turn in turns if isinstance(turn, dict) or turn is not None)

        category_counts: Counter[str] = Counter()
        for qa in qa_items:
            label = category_to_label(qa.get("category"))
            category_counts[label] += 1
            overall_categories[label] += 1

        conv_stats.append(
            ConversationStats(
                conv_id=conv_id,
                session_count=len(sessions),
                turn_count=turn_count,
                qa_count=len(qa_items),
                category_counts=category_counts,
            )
        )

    return conv_stats, overall_categories, irregular


def summarize_numeric(values: Iterable[int]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"min": 0, "max": 0, "avg": 0.0, "total": 0}
    return {
        "min": min(values),
        "max": max(values),
        "avg": mean(values),
        "total": sum(values),
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_float(value: float) -> str:
    return f"{value:.2f}"


def build_report(dataset_path: Path) -> str:
    samples = load_json_records(dataset_path)
    conv_stats, overall_categories, irregular = analyze_samples(samples)

    session_summary = summarize_numeric(stat.session_count for stat in conv_stats)
    qa_summary = summarize_numeric(stat.qa_count for stat in conv_stats)
    turn_summary = summarize_numeric(stat.turn_count for stat in conv_stats)

    total_conversations = len(conv_stats)
    total_sessions = int(session_summary["total"])
    total_qas = int(qa_summary["total"])

    basic_rows = [
        ["Conversations", str(total_conversations)],
        ["Sessions", str(total_sessions)],
        ["QA items", str(total_qas)],
    ]

    per_conv_rows = [
        ["Sessions per conversation", str(int(session_summary["min"])), str(int(session_summary["max"])), format_float(session_summary["avg"])],
        ["QA items per conversation", str(int(qa_summary["min"])), str(int(qa_summary["max"])), format_float(qa_summary["avg"])],
    ]

    length_rows = [
        ["Sessions", format_float(session_summary["avg"]), str(int(session_summary["min"])), str(int(session_summary["max"]))],
        ["Turns", format_float(turn_summary["avg"]), str(int(turn_summary["min"])), str(int(turn_summary["max"]))],
    ]

    total_categories = sum(overall_categories.values())
    overall_cat_rows: list[list[str]] = []
    for label, count in sorted(overall_categories.items(), key=lambda item: (-item[1], item[0])):
        share = (count / total_categories * 100.0) if total_categories else 0.0
        overall_cat_rows.append([label, str(count), format_float(share)])

    category_headers = sorted(overall_categories.keys())
    per_conv_cat_rows: list[list[str]] = []
    for stat in sorted(conv_stats, key=lambda item: item.conv_id):
        row = [stat.conv_id, str(stat.session_count), str(stat.turn_count), str(stat.qa_count)]
        for label in category_headers:
            row.append(str(stat.category_counts.get(label, 0)))
        per_conv_cat_rows.append(row)

    findings: list[str] = []
    if session_summary["min"] == session_summary["max"]:
        findings.append(
            f"Sessions are perfectly balanced across conversations: every conversation has {int(session_summary['min'])} sessions."
        )
    else:
        findings.append(
            "Session counts vary across conversations, indicating uneven conversation lengths."
        )

    if qa_summary["max"] >= 1.5 * qa_summary["avg"]:
        findings.append(
            f"QA coverage is uneven: the busiest conversation has {int(qa_summary['max'])} QA items versus an average of {format_float(qa_summary['avg'])}."
        )
    if qa_summary["min"] <= 0.6 * qa_summary["avg"]:
        findings.append(
            f"The sparsest conversation has only {int(qa_summary['min'])} QA items, well below the average of {format_float(qa_summary['avg'])}."
        )

    if total_categories:
        dominant_label, dominant_count = max(overall_categories.items(), key=lambda item: item[1])
        rare_label, rare_count = min(overall_categories.items(), key=lambda item: item[1])
        dominant_share = dominant_count / total_categories * 100.0
        rare_share = rare_count / total_categories * 100.0
        if dominant_share >= 35.0:
            findings.append(
                f"Category balance is skewed toward {dominant_label} ({dominant_count} items, {format_float(dominant_share)}%)."
            )
        if rare_share <= 8.0:
            findings.append(
                f"{rare_label} is underrepresented ({rare_count} items, {format_float(rare_share)}%)."
            )

    if any(irregular.values()):
        irregular_bits = [f"{key}={value}" for key, value in irregular.items() if value]
        findings.append("Irregular or missing fields detected: " + ", ".join(irregular_bits) + ".")
    else:
        findings.append("No missing conversation or QA containers were detected in the dataset.")

    parts = [
        f"# LoCoMo Statistics\n",
        f"Dataset: `{dataset_path}`\n",
        "## Basic Counts\n",
        markdown_table(["Metric", "Value"], basic_rows),
        "\n## Per-Conversation Statistics\n",
        markdown_table(["Measure", "Min", "Max", "Avg"], per_conv_rows),
        "\n## Conversation Length\n",
        markdown_table(["Measure", "Avg", "Min", "Max"], length_rows),
        "\n## Overall QA Category Distribution\n",
        markdown_table(["Category", "Count", "Share %"], overall_cat_rows),
        "\n## QA Category Distribution Per Conversation\n",
        markdown_table(["Conversation", "Sessions", "Turns", "QA items", *category_headers], per_conv_cat_rows),
        "\n## Key Findings\n",
        "\n".join(f"- {item}" for item in findings),
        "",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze conversation/session/QA statistics for locomo10.json")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "locomo10.json",
        help="Path to locomo10.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(build_report(args.dataset))


if __name__ == "__main__":
    main()
