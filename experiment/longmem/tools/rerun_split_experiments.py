"""Re-run the 8 split-embed retrieval experiments with the *current* reranker
code (grace_mem/utils/reranker.py), reproducing each run's original config.

Config source of truth is experiment/experiment_config.py. Only the 5 swept
knobs differ between experiments; this driver edits ONLY those lines (regex,
in place) so REPRODUCIBILITY_PARAMS and everything else are preserved. The
original file is backed up and restored at the end.

Per experiment: write config -> run retrieval+generation (watchdog, stage
qa_eval, reusing ingested artifacts, NO inline judge) into a NEW run dir
(<orig>-rr2) -> score with the new category-aware judge into `correctness`.

Usage:
    python -m experiment.longmem.tools.rerun_split_experiments --smoke      # 1 dataset, rerank16
    python -m experiment.longmem.tools.rerun_split_experiments --only rerank16
    python -m experiment.longmem.tools.rerun_split_experiments              # all 8, full
    python -m experiment.longmem.tools.rerun_split_experiments --judge-only # re-judge existing rr2 dirs
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from grace_mem.llm import LLMClient
from experiment.common.paths import REPO_ROOT
from experiment.common.evaluation.judge import (
    LONGMEM_CATEGORIES,
    SKIP_LONGMEM_FILES,
    JudgeEngine,
    find_column,
)

CONFIG_PATH = REPO_ROOT / "experiment" / "experiment_config.py"
OUTPUT_DIR = REPO_ROOT / "experiment" / "longmem" / "output"
# Overridable so the same driver can point at different ingest artifacts
# (e.g. oss-120b) and tag its output distinctly without editing this file.
ARTIFACT_DIR = os.environ.get("LONGMEM_ARTIFACT_DIR", "experiment/longmem/output/oss-20b-0427")
SUFFIX = os.environ.get("LONGMEM_RUN_SUFFIX", "-rr2")

# new_tag suffix is appended; overrides are the 5 swept knobs (confirmed w/ user).
EXPERIMENTS: list[tuple[str, dict]] = [
    ("split-embed",   dict(summary_topk_per_item=16, summary_direct_vector_topn=0,  summary_direct_vector_min_score=0.0,  summary_rerank_topk=0,  summary_rerank_cosine_only=False)),
    # sweep-topk16 skipped per user (identical config to split-embed).
    ("sweep-topk24",  dict(summary_topk_per_item=24, summary_direct_vector_topn=0,  summary_direct_vector_min_score=0.0,  summary_rerank_topk=0,  summary_rerank_cosine_only=False)),
    ("sweep-topk32",  dict(summary_topk_per_item=32, summary_direct_vector_topn=0,  summary_direct_vector_min_score=0.0,  summary_rerank_topk=0,  summary_rerank_cosine_only=False)),
    ("extraslot-t50", dict(summary_topk_per_item=16, summary_direct_vector_topn=50, summary_direct_vector_min_score=0.50, summary_rerank_topk=0,  summary_rerank_cosine_only=False)),
    ("extraslot-t40", dict(summary_topk_per_item=16, summary_direct_vector_topn=50, summary_direct_vector_min_score=0.40, summary_rerank_topk=0,  summary_rerank_cosine_only=False)),
    ("extraslot-t35", dict(summary_topk_per_item=16, summary_direct_vector_topn=50, summary_direct_vector_min_score=0.35, summary_rerank_topk=0,  summary_rerank_cosine_only=False)),
    ("rerank16",      dict(summary_topk_per_item=16, summary_direct_vector_topn=50, summary_direct_vector_min_score=0.35, summary_rerank_topk=16, summary_rerank_cosine_only=False)),
]


def set_config_knobs(overrides: dict) -> None:
    """Surgically replace `key=value,` lines for the swept knobs; leave all else intact."""
    text = CONFIG_PATH.read_text()
    for key, value in overrides.items():
        pattern = rf"(?m)^(\s*){re.escape(key)}\s*=.*$"
        repl = rf"\g<1>{key}={value!r},"
        new_text, n = re.subn(pattern, repl, text)
        if n != 1:
            raise RuntimeError(f"expected exactly 1 match for {key}, got {n}")
        text = new_text
    CONFIG_PATH.write_text(text)


def run_retrieval(run_tag: str, *, smoke: bool) -> None:
    cmd = [
        sys.executable, "-m", "experiment.longmem.pipeline.watchdog",
        "--run-tag", run_tag,
        "--artifact-dir", ARTIFACT_DIR,
        "--output-root", f"experiment/longmem/output/{run_tag}",
        "--stage", "qa_eval",
        "--no-judge",
    ]
    if smoke:
        cmd += ["--type", "single_session_user", "--num", "1", "--max-restarts", "1"]
    print(f"  CMD: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def judge_dir(run_tag: str, *, llm) -> tuple[int, int]:
    base = OUTPUT_DIR / run_tag
    judged = skipped = 0
    engine = JudgeEngine(llm, "longmem")
    for cat_sub, category in LONGMEM_CATEGORIES.items():
        cat_dir = base / cat_sub
        if not cat_dir.exists():
            continue
        for path in sorted(cat_dir.glob("*.csv")):
            if path.name in SKIP_LONGMEM_FILES:
                continue
            df = pd.read_csv(path, encoding="utf-8-sig")
            q = find_column(df, ["question"]); g = find_column(df, ["answer", "gold_answer"])
            gen = find_column(df, ["Generated_Answer", "generated_answer", "model_answer"])
            if not all([q, g, gen]):
                continue
            if "correctness" not in df.columns:
                df["correctness"] = ""
            changed = False
            for i, row in df.iterrows():
                _ex = str(row.get("correctness", "")).strip()
                try:
                    _done = float(_ex) in (0.0, 1.0)
                except ValueError:
                    _done = False
                if _done:
                    skipped += 1
                    continue
                question = str(row[q]).strip(); gold = str(row[g]).strip(); generated = str(row[gen]).strip()
                if not question or not generated:
                    continue
                df.at[i, "correctness"] = engine.judge(
                    question=question,
                    gold=gold,
                    generated=generated,
                    category=category,
                    is_abstention=path.stem.endswith("_abs"),
                )
                judged += 1
                changed = True
            if changed:
                df.to_csv(path, index=False)
    return judged, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 dataset, rerank16 config only")
    ap.add_argument("--only", nargs="*", default=None, help="subset of orig dir names")
    ap.add_argument("--judge-only", action="store_true", help="skip retrieval; just judge existing rr2 dirs")
    args = ap.parse_args()

    exps = EXPERIMENTS
    if args.smoke:
        exps = [e for e in EXPERIMENTS if e[0] == "rerank16"]
    elif args.only:
        exps = [e for e in EXPERIMENTS if e[0] in set(args.only)]

    backup = CONFIG_PATH.read_text()
    # Judge on the dedicated judge endpoint (JUDGE_LLM_API/JUDGE_MODEL_NAME) so the
    # judge model stays independent of the answer model (e.g. answer=oss-120b on .34,
    # judge=oss-20b on .63). Falls back to LLM_API/MODEL_NAME when JUDGE_* is unset.
    llm = LLMClient(
        base_url=os.getenv("JUDGE_LLM_API") or None,
        model_name=os.getenv("JUDGE_MODEL_NAME") or None,
    )
    print(f"[JUDGE] model={llm.model_name} base_url={llm._base_url}", flush=True)
    try:
        for orig, overrides in exps:
            run_tag = ("smoke-rr16" if args.smoke else orig + SUFFIX)
            print(f"\n{'='*60}\nEXPERIMENT: {orig} -> {run_tag}\n  knobs: {overrides}\n{'='*60}", flush=True)
            if not args.judge_only:
                set_config_knobs(overrides)
                run_retrieval(run_tag, smoke=args.smoke)
            j, s = judge_dir(run_tag, llm=llm)
            print(f"  [JUDGE] judged={j} skipped={s}", flush=True)
    finally:
        CONFIG_PATH.write_text(backup)
        print("\nRestored experiment_config.py", flush=True)


if __name__ == "__main__":
    main()
