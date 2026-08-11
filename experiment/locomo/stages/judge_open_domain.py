import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.helpers.dataset import (
    category_to_label,
    find_evidence_turns_from_sample,
    load_qa_items,
    load_raw_samples,
    normalize_dataset_name,
    resolve_dataset_path,
)
from experiment.locomo.helpers.llm import build_open_domain_plus_messages, build_open_domain_standard_messages, llm_post

INPUT_CSV = "data/4o-open-domain.csv"
OUTPUT_CSV = "data/4o-open-domain_judged.csv"

def _parse_label(text: str) -> float | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "label" in data:
        label = str(data["label"]).strip().lower()
        if label == "correct":
            return 1
        if label == "partial":
            return 0.5
        if label == "wrong":
            return 0

    t = text.strip().lower()
    if "correct" in t and "wrong" not in t:
        return 1
    if "partial" in t and "correct" not in t and "wrong" not in t:
        return 0.5
    if "wrong" in t and "correct" not in t:
        return 0
    return None


def _build_dia_index(conversation: dict) -> dict:
    dia_index = {}
    for key, turns in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(turns, list):
            continue
        for turn in turns:
            dia_id = turn.get("dia_id")
            if dia_id:
                dia_index[dia_id] = turn
    return dia_index


def _find_evidence_turns(dataset: list, question: str, sample: str | None) -> list[str]:
    q_norm = question.strip()
    candidates = dataset
    if sample and sample.startswith("sample_"):
        try:
            idx = int(sample.split("_", 1)[1])
        except ValueError:
            idx = None
        if idx is not None and 0 <= idx < len(dataset):
            candidates = [dataset[idx]]

    # Prefer exact match within sample, then case-insensitive across all samples.
    def _match(sample_item, use_casefold: bool) -> dict | None:
        for qa in sample_item.get("qa", []):
            q = str(qa.get("question", "")).strip()
            if use_casefold:
                if q.casefold() == q_norm.casefold():
                    return qa
            else:
                if q == q_norm:
                    return qa
        return None

    found = None
    for item in candidates:
        found = _match(item, use_casefold=False)
        if found:
            conv = item.get("conversation", {})
            break
    else:
        conv = None

    if not found:
        for item in dataset:
            found = _match(item, use_casefold=True)
            if found:
                conv = item.get("conversation", {})
                break

    if not found:
        for item in dataset:
            turns = find_evidence_turns_from_sample(item, question)
            if turns:
                return turns

    if not found or not conv:
        return []

    dia_index = _build_dia_index(conv)
    evidence_turns = []
    for dia_id in found.get("evidence", []):
        turn = dia_index.get(dia_id)
        if not turn:
            continue
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if speaker and text:
            evidence_turns.append(f"{speaker}: {text}")
    return evidence_turns


def judge_single(
    question: str,
    gold: str,
    gen: str,
    evidence_turns: str,
    *,
    dataset: str,
    category: str | None = None,
) -> float:
    if dataset == "locomo-plus":
        messages = build_open_domain_plus_messages(
            label=category_to_label(category),
            category=category,
            gold=gold,
            pred=gen,
            evidence=evidence_turns,
        )
    else:
        messages = build_open_domain_standard_messages(
            question=question,
            gold=gold,
            gen=gen,
            evidence_turns=evidence_turns,
        )
    text = llm_post(messages, temperature=0, max_tokens=256).strip()
    label_val = _parse_label(text)
    if label_val is not None:
        return label_val
    print("[WARN] LLM judge response unparsable; returning 0. Raw:", text)
    return 0


def compute_avg_correctness(df: pd.DataFrame) -> dict:
    stats = {"avg_correctness": None, "avg_correctness_percent": None}
    scored = pd.to_numeric(df.get("correctness"), errors="coerce").dropna()
    if not scored.empty:
        avg = float(scored.mean())
        stats["avg_correctness"] = round(avg, 6)
        stats["avg_correctness_percent"] = round(avg * 100.0, 2)
    return stats


def _load_category_map(dataset_json: str) -> dict[str, str]:
    samples = load_raw_samples(dataset_json)
    q_to_cat: dict[str, str] = {}
    for sample_index in range(len(samples)):
        try:
            qa_items = load_qa_items(dataset_json, sample_index=sample_index)
        except Exception:
            continue
        for item in qa_items:
            question = str(item.get("question", "")).strip()
            if question:
                q_to_cat[question] = str(item.get("category", "")).strip()
    return q_to_cat


def llm_as_judge_open_domain(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV, *, dataset_json: str, dataset: str):
    df = pd.read_csv(input_csv)
    locomo_data = load_raw_samples(dataset_json)

    # For locomo-plus merged CSVs the `category` column is already present; build the
    # cross-sample lookup only as a fallback for older CSVs that lack the column.
    has_category_col = "category" in df.columns
    if dataset == "locomo-plus" and not has_category_col:
        q_to_cat = _load_category_map(dataset_json)
    else:
        q_to_cat: dict[str, str] = {}

    # For locomo-plus merged CSVs `gold_evidence_source` already contains the
    # evidence text; only run the heavier `_find_evidence_turns` lookup when it
    # is absent (older single-sample CSVs).
    has_evidence_col = "gold_evidence_source" in df.columns

    q_col = next((c for c in df.columns if c.lower() == "question"), None)
    g_col = next((c for c in df.columns if c.lower() in ["answer", "gold_answer"]), None)
    gen_col = next((c for c in df.columns if c.lower() in ["generated_answer", "model_answer", "gpt_answer"]), None)

    if not all([q_col, g_col, gen_col]):
        raise ValueError("Missing required columns: question, answer/gold_answer, generated_answer/model_answer")

    if "correctness" not in df.columns:
        df["correctness"] = ""
    if "evidence_turns" not in df.columns:
        df["evidence_turns"] = ""

    for i, row in df.iterrows():
        q = str(row[q_col]).strip()
        gold = str(row[g_col]).strip()
        gen = str(row[gen_col]).strip()
        if not gen:
            df.at[i, "correctness"] = ""
            continue

        # Resolve evidence text
        if has_evidence_col:
            evidence_turns = str(row.get("gold_evidence_source", "")).strip()
        else:
            sample = str(row.get("sample", "")).strip() if "sample" in df.columns else ""
            evidence_turns_list = _find_evidence_turns(locomo_data, q, sample)
            evidence_turns = "\n".join(evidence_turns_list)
            if not evidence_turns_list:
                print(f"[WARN] No evidence turns found for row {i}: {q[:80]}...")
        df.at[i, "evidence_turns"] = evidence_turns

        # Resolve category
        if has_category_col:
            category = str(row.get("category", "")).strip() or None
        else:
            category = q_to_cat.get(q)

        print(f"Judging row {i}: {q[:50]}...")
        val = judge_single(
            q,
            gold,
            gen,
            evidence_turns,
            dataset=dataset,
            category=category,
        )
        df.at[i, "correctness"] = val

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Saved to {output_csv}")

    stats = compute_avg_correctness(df)
    if stats["avg_correctness"] is not None:
        print(f"📊 Avg correctness: {stats['avg_correctness']:.4f} ({stats['avg_correctness_percent']:.2f}%)")
    else:
        print("📊 Avg correctness: N/A (no scored rows)")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM judge correctness for open-domain QA (avg only)")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--dataset", choices=["locomo", "locomo-plus"], default="locomo")
    parser.add_argument("--dataset-json", default=None, help="Defaults are resolved from --dataset")
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(
        dataset=dataset,
        kind="qa_json",
        explicit_path=args.dataset_json,
    )

    llm_as_judge_open_domain(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        dataset_json=str(dataset_json),
        dataset=dataset,
    )
