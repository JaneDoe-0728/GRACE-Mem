"""Judge stage: score generated answers and aggregate by category.

Three metrics, kept together because they disagree in informative ways. The LLM
judge decides semantic equivalence, which is the number that matters. F1 and
BLEU-1 are lexical and cheap, and they exist as a sanity check on the judge: a
run where judge accuracy moved but lexical overlap did not usually means the
judge changed its mind, not that the system improved.

Adversarial questions are excluded from the headline average by default. They
are unanswerable by construction, so scoring them together with answerable ones
conflates "found the wrong evidence" with "correctly declined" -- they are
reported separately instead.

Stats are computed per category as well as overall, since an aggregate can hide
a regression in one question type behind gains in another.
"""

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

from experiment.locomo.helpers.dataset import (
    category_to_label,
    find_evidence_turns_from_sample,
    load_qa_items,
    load_raw_samples,
    normalize_dataset_name,
    resolve_dataset_path,
)
from experiment.locomo.helpers.llm import (
    build_judge_standard_messages,
    build_open_domain_standard_messages,
    llm_post,
)
from experiment.common.evaluation.judge import (
    normalize_temporal_gold as _normalize_temporal_gold,
    parse_locomo_verdict as _parse_label,
)

INPUT_CSV = "data/sample0_eval__20260205_111338_judge.csv"
OUTPUT_CSV = "data/sample0_eval__20260205_111338_judgev2.csv"
OPEN_DOMAIN_INPUT_CSV = "data/4o-open-domain.csv"
OPEN_DOMAIN_OUTPUT_CSV = "data/4o-open-domain_judged.csv"
LOCOMO_JSON = "data/locomo10.json"

CATEGORY_MAP = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}

def compute_correctness_stats(df: pd.DataFrame, *, exclude_adversarial: bool = True) -> dict:
    """Aggregate judged rows into overall and per-category accuracy, plus F1/BLEU.

    Per-category alongside the overall figure because an aggregate hides a
    regression in one question type behind a gain in another.

    Args:
        exclude_adversarial: Adversarial questions are unanswerable by
            construction; averaging them with answerable ones conflates finding
            the wrong evidence with correctly declining to answer.

    Returns:
        Stats with None -- not 0 -- where there was nothing to average, so an
        empty category is distinguishable from one that scored zero.
    """
    stats = {
        "avg_correctness": None,
        "avg_correctness_percent": None,
        "by_category": {},
        "avg_f1": None,
        "avg_bleu1": None,
    }

    def ensure_category_stats(label: str) -> dict:
        """Return the stats bucket for a category, creating it on first sight."""
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
    mode: str = "standard",
) -> float:
    """
    Score a single question: the judge returns CORRECT/WRONG as JSON, which we map
    to 1/0.
    """
    if mode == "open-domain":
        messages = build_open_domain_standard_messages(
            question=question,
            gold=gold,
            gen=gen,
            evidence_turns=evidence,
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
    """Lowercase and split on non-alphanumerics, for the F1 overlap metric.

    Deliberately cruder than the BLEU tokenizer: F1 here is a bag-of-words
    overlap check on the LLM judge, and punctuation or casing differences should
    not register as disagreement.
    """
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
    """Tokenize for BLEU, falling back to a simple split if NLTK data is missing.

    The punkt data is a separate download, and a scoring run must not fail
    because it is absent. The fallback shifts BLEU values slightly, so compare
    BLEU only within a run, not across machines.
    """
    normalized = str(text).strip().lower()
    if not normalized:
        return []
    try:
        return nltk.word_tokenize(normalized)
    except LookupError:
        # Fall back when punkt/punkt_tab is unavailable in the runtime env.
        return simple_tokenize(normalized)

def compute_f1_and_bleu1(gold: str, pred: str) -> tuple[float, float]:
    """Compute token-overlap F1 and BLEU-1 between gold and prediction.

    Both are lexical, and neither is the headline metric -- the LLM judge is.
    They exist as a cross-check on it: a run where judged accuracy moved but
    lexical overlap did not usually means the judge changed its mind rather than
    the system improving.

    Unigram BLEU specifically, with smoothing, because gold answers are short
    phrases where higher-order n-gram precision is mostly zero and would swamp
    the signal.

    Returns:
        (f1, bleu1), both 0.0 when either side is empty.
    """
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
    """Load question -> category for one sample, for the per-category breakdown.

    The eval CSVs do not carry categories; they live only in the source dataset,
    and this is the join back to it.
    """
    qa_list = load_qa_items(dataset_json_path, sample_index=sample_index)
    q_to_cat = {}
    for item in qa_list:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        q_to_cat[q] = item.get("category")
    return q_to_cat




def _build_dia_index(conversation: dict) -> dict:
    """Index a conversation's turns by their D{session}:{turn} evidence id."""
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
    """Locate a question's gold evidence turns in the dataset.

    Searches the named sample first and falls back to scanning all of them,
    because the sample label is absent or malformed in older outputs. The
    fallback can match an identically worded question in a different sample --
    acceptable, since this feeds diagnostics rather than scoring.
    """
    q_norm = question.strip()
    candidates = dataset
    if sample and sample.startswith("sample_"):
        try:
            idx = int(sample.split("_", 1)[1])
        except ValueError:
            idx = None
        if idx is not None and 0 <= idx < len(dataset):
            candidates = [dataset[idx]]

    def _match(sample_item, use_casefold: bool) -> dict | None:
        for qa in sample_item.get("qa", []):
            q = str(qa.get("question", "")).strip()
            matched = q.casefold() == q_norm.casefold() if use_casefold else q == q_norm
            if matched:
                return qa
        return None

    found = None
    conv = None
    for item in candidates:
        found = _match(item, use_casefold=False)
        if found:
            conv = item.get("conversation", {})
            break

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


def _infer_sample_index(input_csv: str) -> int | None:
    """Recover a sample index from a path or filename, or None."""
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
    """Judge one answer with the standard LoCoMo rubric.

    Category selects the prompt template: a temporal question and an open-domain
    one fail in different ways, and grading both by one rubric scores at least
    one of them on the wrong criterion.
    """
    output_path = Path(output_csv)
    if output_path.exists():
        df = pd.read_csv(output_csv)
        print(f"[resume] Loading existing output ({len(df)} rows): {output_csv}")
    else:
        df = pd.read_csv(input_csv)

    # Normalize the column names
    q_col = next((c for c in df.columns if c.lower() == "question"), None)
    g_col = next((c for c in df.columns if c.lower() in ["answer", "gold_answer"]), None)
    gen_col = next((c for c in df.columns if c.lower() in ["generated_answer", "model_answer", "gpt_answer"]), None)

    if not all([q_col, g_col, gen_col]):
        raise ValueError("required columns not found (question, answer/gold_answer, generated_answer/model_answer)")

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


def llm_as_judge_open_domain(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    *,
    dataset_json: str,
    dataset: str,
):
    """Judge one open-domain answer, where gold is a reference rather than an oracle.

    Separate from the standard path because the grading rule genuinely differs:
    these questions admit several correct answers, and the standard rubric marks
    correct answers wrong for not matching the reference string.
    """
    df = pd.read_csv(input_csv)
    locomo_data = load_raw_samples(dataset_json)

    has_category_col = "category" in df.columns
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

        if has_evidence_col:
            evidence_turns = str(row.get("gold_evidence_source", "")).strip()
        else:
            sample = str(row.get("sample", "")).strip() if "sample" in df.columns else ""
            evidence_turns_list = _find_evidence_turns(locomo_data, q, sample)
            evidence_turns = "\n".join(evidence_turns_list)
            if not evidence_turns_list:
                print(f"[WARN] No evidence turns found for row {i}: {q[:80]}...")
        df.at[i, "evidence_turns"] = evidence_turns

        if has_category_col:
            category = str(row.get("category", "")).strip() or None
        else:
            category = None
        print(f"Judging row {i}: {q[:50]}...")
        val = judge_single(
            q,
            gold,
            gen,
            dataset=dataset,
            category=category,
            evidence=evidence_turns,
            mode="open-domain",
        )
        df.at[i, "correctness"] = val

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Saved to {output_csv}")

    stats = compute_correctness_stats(df, exclude_adversarial=False)
    if stats["avg_correctness"] is not None:
        print(f"📊 Avg correctness: {stats['avg_correctness']:.4f} ({stats['avg_correctness_percent']:.2f}%)")
    else:
        print("📊 Avg correctness: N/A (no scored rows)")
    return {
        "avg_correctness": stats["avg_correctness"],
        "avg_correctness_percent": stats["avg_correctness_percent"],
    }

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
        mode: str = "standard",
    ) -> None:
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.dataset_json = dataset_json
        self.dataset = dataset
        self.sample_index = sample_index
        self.exclude_adversarial = exclude_adversarial
        self.mode = mode

    def run(self) -> dict:
        if self.mode == "open-domain":
            return llm_as_judge_open_domain(
                input_csv=str(self.input_csv),
                output_csv=str(self.output_csv),
                dataset_json=str(self.dataset_json),
                dataset=self.dataset,
            )
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
    parser.add_argument("--mode", choices=["standard", "open-domain"], default="standard")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--dataset", choices=["locomo"], default="locomo")
    parser.add_argument("--dataset-json", default=None, help="Defaults are resolved from --dataset")
    parser.add_argument("--adv", action="store_true", help="Include adversarial rows in summary stats")
    args = parser.parse_args()

    dataset = normalize_dataset_name(args.dataset)
    dataset_json = resolve_dataset_path(
        dataset=dataset,
        kind="qa_json",
        explicit_path=args.dataset_json,
    )

    input_csv = args.input_csv or (OPEN_DOMAIN_INPUT_CSV if args.mode == "open-domain" else INPUT_CSV)
    output_csv = args.output_csv or (OPEN_DOMAIN_OUTPUT_CSV if args.mode == "open-domain" else OUTPUT_CSV)

    if args.mode == "open-domain":
        llm_as_judge_open_domain(
            input_csv=input_csv,
            output_csv=output_csv,
            dataset_json=str(dataset_json),
            dataset=dataset,
        )
    else:
        llm_as_judge_singlemode(
            input_csv=input_csv,
            output_csv=output_csv,
            sample_index=args.sample_index,
            dataset_json=str(dataset_json),
            dataset=dataset,
            exclude_adversarial=not args.adv,
        )
