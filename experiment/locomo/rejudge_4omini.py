"""Re-judge existing LoCoMo eval results with gpt-4o-mini, WITHOUT touching the
original gpt-oss-20b judge output.

For each run-tag, reads every sample's existing *_judge.csv, re-judges each row
with gpt-4o-mini (reusing the LoCoMo standard judge prompt + temporal
normalization), and writes:
  - sample_<n>/<sampleN>_eval_<tag>_judge_4omini.csv   (original cols + correctness_4omini)
  - <run-tag>/_correctness_aggregate_4omini.json        (overall + per-category)

The original *_judge.csv, correctness column, and _correctness_aggregate.json are
never modified. Unlike the pipeline judge stage, this does NOT skip a sample just
because its judge.csv exists — it re-judges into a separate column/file. Rows whose
correctness_4omini is already 0/1 are skipped (resumable). 429 rate limits are
retried with exponential backoff.

Usage:
    python experiment/locomo/rejudge_4omini.py locomo-rr16 locomo-n8
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from requests.exceptions import HTTPError, RequestException

from KG.llm import LLMClient
from experiment.locomo.helpers.llm import build_judge_standard_messages
from experiment.locomo.stages.judge import (
    _normalize_temporal_gold,
    _parse_label,
    compute_correctness_stats,
)

OUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
NEW_COL = "correctness_4omini"


def _openai_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env = _ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.search(r'OPENAI_API_KEY="?(sk-[^"\s]+)', line)
            if m:
                return m.group(1)
    return None


# 3 票去噪用的溫度序列(與 LongMem codex rejudge_output_dirs.py 同口徑)。
_VOTE_TEMPS = (0.0, 0.3, 0.6)


def judge_4omini(llm, *, question: str, gold: str, gen: str, votes: int = 1) -> int:
    """4o-mini LoCoMo standard-rubric judge。

    votes=1(預設)= 原本的單票(temp 0),行為完全不變。
    votes>1 = 多數決去噪(temps 0/0.3/0.6 循環),平手偏「對」。
    LoCoMo 無 abs 題,故不做棄答分流。
    """
    gold_hint = _normalize_temporal_gold(gold)
    gold_for_judge = f"{gold}\n[Normalized: {gold_hint}]" if gold_hint else gold
    messages = build_judge_standard_messages(question=question, gold=gold_for_judge, gen=gen)

    def _one(temp: float) -> int:
        delay = 2.0
        for attempt in range(8):
            try:
                resp = llm.chat(messages=messages, temperature=temp, max_tokens=256)
                val = _parse_label((resp.choices[0].message.content or "").strip())
                return int(val) if val is not None else 0
            except HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
            except RequestException:
                # transient network errors (ReadTimeout / ConnectionError / etc.)
                if attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
        return 0

    if votes <= 1:
        return _one(0.0)
    temps = [_VOTE_TEMPS[i % len(_VOTE_TEMPS)] for i in range(votes)]
    tally = sum(_one(t) for t in temps)
    return 1 if tally * 2 >= votes else 0  # majority (ties -> correct)


def rejudge_sample(judge_csv: Path, out_csv: Path, *, llm) -> tuple[int, int]:
    # Resume from out_csv if it exists, else start from the original judge csv.
    src = out_csv if out_csv.exists() else judge_csv
    df = pd.read_csv(src, encoding="utf-8-sig")
    if NEW_COL not in df.columns:
        df[NEW_COL] = ""

    judged = skipped = 0
    for i, row in df.iterrows():
        ex = str(row.get(NEW_COL, "")).strip()
        try:
            if float(ex) in (0.0, 1.0):
                skipped += 1
                continue
        except ValueError:
            pass
        q = str(row.get("question", "")).strip()
        gold = str(row.get("gold_answer", "")).strip()
        gen = str(row.get("model_answer", "")).strip()
        if not q or not gen:
            df.at[i, NEW_COL] = ""
            continue
        df.at[i, NEW_COL] = judge_4omini(llm, question=q, gold=gold, gen=gen)
        judged += 1
        if judged % 25 == 0:
            df.to_csv(out_csv, index=False)  # periodic checkpoint
    df.to_csv(out_csv, index=False)
    return judged, skipped


def main():
    run_tags = sys.argv[1:] or ["locomo-rr16", "locomo-n8"]
    llm = LLMClient(base_url="https://api.openai.com/v1", model_name="gpt-4o-mini", api_key=_openai_key())
    print(f"[JUDGE] model={llm.model_name} base={llm._base_url}")

    for tag in run_tags:
        base = OUT / tag
        if not base.exists():
            print(f"[MISS] {tag}: not found")
            continue
        print(f"\n=== {tag} ===")
        frames = []
        for sdir in sorted(base.glob("sample_*")):
            jc = next(iter(sdir.glob("*_judge.csv")), None)
            if jc is None or "_judge_4omini" in jc.name:
                continue
            out_csv = jc.with_name(jc.name.replace("_judge.csv", "_judge_4omini.csv"))
            j, s = rejudge_sample(jc, out_csv, llm=llm)
            print(f"  {sdir.name}: judged={j} skipped={s} -> {out_csv.name}")
            d = pd.read_csv(out_csv, encoding="utf-8-sig")
            d2 = d.copy()
            d2["correctness"] = pd.to_numeric(d2[NEW_COL], errors="coerce")
            frames.append(d2)
        if frames:
            alldf = pd.concat(frames, ignore_index=True)
            stats = compute_correctness_stats(alldf, exclude_adversarial=True)
            agg_path = base / "_correctness_aggregate_4omini.json"
            agg_path.write_text(json.dumps({"root": str(base), "judge_model": "gpt-4o-mini", "overall": stats}, indent=2))
            print(f"  [AGG] overall {stats.get('avg_correctness_percent')}% -> {agg_path.name}")


if __name__ == "__main__":
    main()
