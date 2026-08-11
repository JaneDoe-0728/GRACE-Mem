import os
import pandas as pd
import re
from pathlib import Path
import sys
import argparse
import nltk
from tqdm import tqdm

# Silence HuggingFace transformers generation-flag warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.helpers.dataset import category_to_label, load_qa_items, normalize_dataset_name, resolve_dataset_path
from experiment.locomo.helpers.llm import build_judge_plus_messages, build_judge_standard_messages, llm_post
from experiment.judge import (
    normalize_temporal_gold as _normalize_temporal_gold,
    parse_locomo_verdict as _parse_label,
)

INPUT_CSV = "data/sample0_eval__20260205_111338_judge.csv"
OUTPUT_CSV = "data/sample0_eval__20260205_111338_judgev2.csv"
LOCOMO_JSON = "data/locomo10.json"

CATEGORY_MAP = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
    6: "Cognitive",
}

def compute_correctness_stats(df: pd.DataFrame, *, exclude_adversarial: bool = True) -> dict:
    stats = {
        "avg_correctness": None,
        "avg_correctness_percent": None,
        "by_category": {},
        "avg_f1": None,
        "avg_bleu1": None,
    }

    def ensure_category_stats(label: str) -> dict:
        return stats["by_category"].setdefault(
            str(label),
            {
                "avg_correctness": None,
                "avg_correctness_percent": None,
                "avg_f1": None,
                "avg_bleu1": None,
            },
        )

    mask = pd.Series(True, index=df.index)
    if exclude_adversarial:
        if "category_label" in df.columns:
            mask &= df["category_label"].astype(str).str.strip().str.lower() != "adversarial"
        elif "category" in df.columns:
            mask &= ~df["category"].astype(str).str.strip().isin(["5", "adversarial", "Adversarial"])

    scored = pd.to_numeric(df.get("correctness"), errors="coerce")[mask].dropna()
    if not scored.empty:
        avg = float(scored.mean())
        stats["avg_correctness"] = round(avg, 6)
        stats["avg_correctness_percent"] = round(avg * 100.0, 2)

    if "category_label" in df.columns:
        cat_df = df.copy()
        cat_df["correctness_num"] = pd.to_numeric(cat_df["correctness"], errors="coerce")
        cat_df["f1_num"] = pd.to_numeric(cat_df.get("f1"), errors="coerce")
        cat_df["bleu1_num"] = pd.to_numeric(cat_df.get("bleu1"), errors="coerce")
        grouped = (
            cat_df.dropna(subset=["correctness_num"])
            .groupby("category_label")["correctness_num"]
            .mean()
            .sort_index()
        )
        for label, val in grouped.items():
            category_stats = ensure_category_stats(str(label))
            category_stats["avg_correctness"] = round(float(val), 6)
            category_stats["avg_correctness_percent"] = round(float(val) * 100.0, 2)
        f1_grouped = (
            cat_df.dropna(subset=["f1_num"])
            .groupby("category_label")["f1_num"]
            .mean()
            .sort_index()
        )
        for label, val in f1_grouped.items():
            ensure_category_stats(str(label))["avg_f1"] = round(float(val), 6)
        bleu_grouped = (
            cat_df.dropna(subset=["bleu1_num"])
            .groupby("category_label")["bleu1_num"]
            .mean()
            .sort_index()
        )
        for label, val in bleu_grouped.items():
            ensure_category_stats(str(label))["avg_bleu1"] = round(float(val), 6)

    f1_scored = pd.to_numeric(df.get("f1"), errors="coerce")[mask].dropna()
    if not f1_scored.empty:
        stats["avg_f1"] = round(float(f1_scored.mean()), 6)
    bleu_scored = pd.to_numeric(df.get("bleu1"), errors="coerce")[mask].dropna()
    if not bleu_scored.empty:
        stats["avg_bleu1"] = round(float(bleu_scored.mean()), 6)
    return stats

def judge_single(
    question: str,
    gold: str,
    gen: str,
    *,
    dataset: str,
    category: str | None = None,
    evidence: str = "",
) -> float:
    """
    單題評分：回傳 CORRECT/WRONG JSON；我們映射為 1/0。
    """
    if dataset == "locomo-plus":
        messages = build_judge_plus_messages(
            label=category_to_label(category),
            category=category,
            gold=gold,
            pred=gen,
            evidence=evidence,
        )
    else:
        gold_hint = _normalize_temporal_gold(gold)
        gold_for_judge = (
            f"{gold}\n[Normalized: {gold_hint}]" if gold_hint else gold
        )
        messages = build_judge_standard_messages(question=question, gold=gold_for_judge, gen=gen)
    # gpt-oss-20b is a reasoning model: it emits reasoning tokens before the
    # verdict. 256 tokens was exhausted mid-reasoning → empty completions
    # (finish_reason='length'), leaving ~70% of rows unjudged. Give it room.
    text = llm_post(messages, temperature=0, max_tokens=2048).strip()
    label_val = _parse_label(text)
    if label_val is not None:
        return label_val

    print("[WARN] LLM judge response unparsable; returning 0. Raw:", text)
    return 0

def simple_tokenize(text: str) -> list[str]:
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def safe_bleu_tokenize(text: str) -> list[str]:
    normalized = str(text).strip().lower()
    if not normalized:
        return []
    try:
        return nltk.word_tokenize(normalized)
    except LookupError:
        # Fall back when punkt/punkt_tab is unavailable in the runtime env.
        return simple_tokenize(normalized)

def compute_f1_and_bleu1(gold: str, pred: str) -> tuple[float, float]:
    if not pred or not gold:
        return 0.0, 0.0

    pred_str = str(pred).strip()
    gold_str = str(gold).strip()

    pred_tokens = set(simple_tokenize(pred_str))
    gold_tokens = set(simple_tokenize(gold_str))
    if not pred_tokens or not gold_tokens:
        f1 = 0.0
    else:
        common_tokens = pred_tokens & gold_tokens
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gold_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    pred_bleu_tokens = safe_bleu_tokenize(pred_str)
    gold_bleu_tokens = [safe_bleu_tokenize(gold_str)]
    if not pred_bleu_tokens or not gold_bleu_tokens[0]:
        return f1, 0.0
    smooth = SmoothingFunction().method1
    try:
        bleu1 = sentence_bleu(gold_bleu_tokens, pred_bleu_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    except Exception:
        bleu1 = 0.0
    return f1, bleu1

def load_category_map(dataset_json_path: str, sample_index: int) -> dict:
    qa_list = load_qa_items(dataset_json_path, sample_index=sample_index)
    q_to_cat = {}
    for item in qa_list:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        q_to_cat[q] = item.get("category")
    return q_to_cat


def _infer_sample_index(input_csv: str) -> int | None:
    match = re.search(r"sample(\d+)", Path(input_csv).name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def llm_as_judge_singlemode(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    *,
    sample_index: int | None = None,
    dataset_json: str = LOCOMO_JSON,
    dataset: str = "locomo",
    exclude_adversarial: bool = True,
):
    # Resume from existing output if present (incremental mode).
    output_path = Path(output_csv)
    if output_path.exists():
        df = pd.read_csv(output_csv)
        print(f"[resume] Loading existing output ({len(df)} rows): {output_csv}")
    else:
        df = pd.read_csv(input_csv)

    # 標準化欄位
    q_col = next((c for c in df.columns if c.lower() == "question"), None)
    g_col = next((c for c in df.columns if c.lower() in ["answer", "gold_answer"]), None)
    gen_col = next((c for c in df.columns if c.lower() in ["generated_answer", "model_answer", "gpt_answer"]), None)

    if not all([q_col, g_col, gen_col]):
        raise ValueError("找不到必要欄位 (question, answer/gold_answer, generated_answer/model_answer)")

    if "correctness" not in df.columns:
        df["correctness"] = ""
    if "f1" not in df.columns:
        df["f1"] = ""
    if "bleu1" not in df.columns:
        df["bleu1"] = ""

    if sample_index is None:
        sample_index = _infer_sample_index(str(input_csv))

    if sample_index is not None:
        q_to_cat = load_category_map(dataset_json, sample_index)
        df["category"] = df[q_col].apply(lambda x: q_to_cat.get(str(x).strip(), None))
        df["category_label"] = df["category"].apply(category_to_label)
        category_col = "category"
    else:
        category_col = next((c for c in df.columns if c.lower() == "category"), None)
        if category_col:
            df["category_label"] = df[category_col].apply(category_to_label)

    total = len(df)
    already_scored = int(df["correctness"].apply(lambda x: pd.notna(x) and str(x).strip() != "").sum())
    pbar = tqdm(df.iterrows(), total=total, desc="Judging", unit="q",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for i, row in pbar:
        q = str(row[q_col]).strip()
        gold = str(row[g_col]).strip()
        gen = str(row[gen_col]).strip()
        if not gen:
            df.at[i, "correctness"] = ""
            df.at[i, "f1"] = ""
            df.at[i, "bleu1"] = ""
            continue

        existing_correctness = row.get("correctness", "")
        has_existing_correctness = (
            pd.notna(existing_correctness)
            and str(existing_correctness).strip() != ""
        )
        if has_existing_correctness:
            # Skip LLM judge if correctness already present.
            pass
        else:
            cat = str(row.get("category_label") or row.get("category") or "")
            pbar.set_postfix(cat=cat[:12], q=q[:30])
            val = judge_single(
                q,
                gold,
                gen,
                dataset=dataset,
                category=row.get("category"),
                evidence=str(row.get("gold_evidence_source", "")).strip(),
            )
            df.at[i, "correctness"] = val
        f1, bleu1 = compute_f1_and_bleu1(gold, gen)
        df.at[i, "f1"] = round(f1, 6)
        df.at[i, "bleu1"] = round(bleu1, 6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Saved to {output_csv}")

    stats = compute_correctness_stats(df, exclude_adversarial=exclude_adversarial)
    if stats["avg_correctness"] is not None:
        print(f"📊 Avg correctness: {stats['avg_correctness']:.4f} ({stats['avg_correctness_percent']:.2f}%)")
    else:
        print("📊 Avg correctness: N/A (no scored rows)")

    if stats["avg_f1"] is not None:
        print(f"📊 Avg F1: {stats['avg_f1']:.4f}")
    else:
        print("📊 Avg F1: N/A (no scored rows)")

    if stats["avg_bleu1"] is not None:
        print(f"📊 Avg BLEU-1: {stats['avg_bleu1']:.4f}")
    else:
        print("📊 Avg BLEU-1: N/A (no scored rows)")

    if stats["by_category"]:
        print("📊 Correctness by category:")
        for label, val in stats["by_category"].items():
            parts = []
            correctness = val.get("avg_correctness")
            correctness_pct = val.get("avg_correctness_percent")
            if correctness is not None and correctness_pct is not None:
                parts.append(f"correctness={correctness:.4f} ({correctness_pct:.2f}%)")
            else:
                parts.append("correctness=N/A")
            if val.get("avg_f1") is not None:
                parts.append(f"f1={val['avg_f1']:.4f}")
            if val.get("avg_bleu1") is not None:
                parts.append(f"bleu1={val['avg_bleu1']:.4f}")
            print(f"  - {label}: {', '.join(parts)}")
    return stats

class JudgeStage:
    """Class interface for standalone or embedded judge runs."""

    def __init__(
        self,
        *,
        input_csv,
        output_csv,
        dataset_json,
        dataset: str,
        sample_index=None,
        exclude_adversarial: bool = True,
    ) -> None:
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.dataset_json = dataset_json
        self.dataset = dataset
        self.sample_index = sample_index
        self.exclude_adversarial = exclude_adversarial

    def run(self) -> dict:
        return llm_as_judge_singlemode(
            input_csv=str(self.input_csv),
            output_csv=str(self.output_csv),
            sample_index=self.sample_index,
            dataset_json=str(self.dataset_json),
            dataset=self.dataset,
            exclude_adversarial=self.exclude_adversarial,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM judge correctness with optional category stats")
    parser.add_argument("--input-csv", default=INPUT_CSV)
    parser.add_argument("--output-csv", default=OUTPUT_CSV)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--dataset", choices=["locomo", "locomo-plus"], default="locomo")
    parser.add_argument("--dataset-json", default=None, help="Defaults are resolved from --dataset")
    parser.add_argument("--adv", action="store_true", help="Include adversarial rows in summary stats")
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(
        dataset=dataset,
        kind="qa_json",
        explicit_path=args.dataset_json,
    )

    llm_as_judge_singlemode(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        sample_index=args.sample_index,
        dataset_json=str(dataset_json),
        dataset=dataset,
        exclude_adversarial=not args.adv,
    )
