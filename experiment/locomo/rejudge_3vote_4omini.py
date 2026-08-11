"""LoCoMo codex 式 3 票去噪 rejudge(LoCoMo 無 abs 題)。

流程(對齊 LongMem docs/JUDGING.md §三,但適配 LoCoMo 每檔多列的佈局):
  - 讀 sample_*/sample*_eval_<tag>_judge_4omini.csv(已有單票 correctness_4omini)。
  - 建 correctness_3vote 欄:
      單票判「對」(correctness_4omini==1)→ carry 1(省 API,不重判)
      單票判「錯」(==0 / 空)              → 3 票多數決重判(temps 0/0.3/0.6)
  - LoCoMo 沒有 abs 題,故無棄答分流(全部同一 general rubric)。
  - retry 已在 judge_4omini(429/5xx/ReadTimeout backoff)→ 可放大 --workers。
  - resumable:correctness_3vote 已是 0/1 的列跳過。就地寫回。

判分:排除 Adversarial(category_label),印 overall + 分類別(correctness_3vote)。

Usage:
    python experiment/locomo/rejudge_3vote_4omini.py --run-tag locomo-n8 --workers 48
    python experiment/locomo/rejudge_3vote_4omini.py --run-tag locomo-n8 --samples 0-9 --votes 3
"""
from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.locomo.rejudge_4omini import _openai_key, judge_4omini

OUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
SRC_COL = "correctness_4omini"
VOTE_COL = "correctness_3vote"

_tls = threading.local()


def _llm():
    if getattr(_tls, "c", None) is None:
        _tls.c = LLMClient(base_url="https://api.openai.com/v1",
                           model_name="gpt-4o-mini", api_key=_openai_key())
    return _tls.c


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _as01(x) -> int | None:
    s = str(x).strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def rejudge_sample(path: Path, *, votes: int, workers: int) -> tuple[int, int]:
    """Returns (judged, carried). 就地寫回 correctness_3vote。"""
    df = pd.read_csv(path)
    q_col = _find_col(df, ["question"])
    g_col = _find_col(df, ["gold_answer", "answer", "gold"])
    a_col = _find_col(df, ["model_answer", "Generated_Answer", "generated_answer"])
    if not all([q_col, g_col, a_col]) or SRC_COL not in df.columns:
        print(f"  [SKIP] {path.name}: missing cols")
        return 0, 0
    if VOTE_COL not in df.columns:
        df[VOTE_COL] = ""

    jobs = []          # (row_index) 要 3 票重判的錯題
    carried = 0
    for i, row in df.iterrows():
        if _as01(row.get(VOTE_COL)) in (0, 1):        # resumable
            continue
        if _as01(row.get(SRC_COL)) == 1:              # carry 判對題
            df.at[i, VOTE_COL] = 1
            carried += 1
        else:
            jobs.append(i)                            # 錯題 → 3 票

    def _do(i):
        r = df.loc[i]
        return i, judge_4omini(_llm(), question=str(r[q_col]).strip(),
                               gold=str(r[g_col]).strip(), gen=str(r[a_col]).strip(),
                               votes=votes)

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_do, i) for i in jobs]):
                i, v = fut.result()
                df.at[i, VOTE_COL] = v

    df.to_csv(path, index=False)
    return len(jobs), carried


def score(run_tag: str, sample_ids: list[int]) -> None:
    import collections
    per = collections.defaultdict(lambda: [0, 0])
    tot = [0, 0]
    for si in sample_ids:
        p = OUT / run_tag / f"sample_{si}" / f"sample{si}_eval_{run_tag}_judge_4omini.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        lab_col = _find_col(df, ["category_label"])
        for _, r in df.iterrows():
            label = str(r.get(lab_col, "")).strip() if lab_col else ""
            if label.lower() == "adversarial":
                continue
            v = _as01(r.get(VOTE_COL))
            if v is None:
                continue
            tot[0] += v; tot[1] += 1
            per[label][0] += v; per[label][1] += 1
    print(f"\n=== {run_tag}  |  {VOTE_COL}(排除 Adversarial)===")
    for label in sorted(per):
        c, n = per[label]
        print(f"  {label:<16}{c}/{n}  {100*c/n:.1f}%")
    if tot[1]:
        print(f"  {'OVERALL':<16}{tot[0]}/{tot[1]}  {100*tot[0]/tot[1]:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="locomo-n8")
    ap.add_argument("--samples", default="0-9")
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48, help="row-level 併發(retry 已強,可放大)")
    ap.add_argument("--score-only", action="store_true", help="不判分,只重算分數")
    args = ap.parse_args()

    ids: list[int] = []
    for part in args.samples.split(","):
        if "-" in part:
            a, b = part.split("-"); ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))

    if not args.score_only:
        tot_j = tot_c = 0
        for si in ids:
            p = OUT / args.run_tag / f"sample_{si}" / f"sample{si}_eval_{args.run_tag}_judge_4omini.csv"
            if not p.exists():
                print(f"[MISS] sample_{si}: {p.name} not found")
                continue
            j, c = rejudge_sample(p, votes=args.votes, workers=args.workers)
            tot_j += j; tot_c += c
            print(f"sample_{si}: carried={c} rejudged(votes={args.votes})={j}")
        print(f"\nDone. carried={tot_c} rejudged={tot_j} col={VOTE_COL}")

    score(args.run_tag, ids)


if __name__ == "__main__":
    main()
