"""Grep 可達率分析(Step 0 of grep-agent plan)。

問題:vector+rerank 漏掉的 gold,grep agent 補得回來嗎?
本腳本不跑任何模型,直接量「gold turn 的 raw text 與 question 的字面重疊」,
給出 grep agent(V2 filter+fetch)的理論天花板。

每個 gold turn 量四種可達性(由弱到強):
  R_any     gold turn 含 >=1 個 question 內容詞(word-boundary, case-insensitive)
  R_useful  存在一個 question 內容詞 k:k 命中 gold turn,且 k 在整個 haystack
            的命中 turn 數 <= --max-df(grep 結果放得進 capped output,agent 看得完)
  R_pair    question 的某兩個內容詞同時出現在該 gold turn(AND 過濾 = rg|rg,高精準)
  R_ans     gold turn 含 >=1 個 answer 內容詞(oracle 上限:agent 不知道答案,
            但答案本身是字面 span 時,好 query 常能收斂到它)

Usage:
    python -m experiment.agent_filter.grep_reachability
    python -m experiment.agent_filter.grep_reachability --max-df 40 --per-question-csv /tmp/reach.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"

CATEGORIES = [
    "single_session_user",
    "single_session_assistant",
    "multi_session",
    "single_session_preference",
    "temporal_reasoning",
    "knowledge_update",
]

# 精簡英文 stopword + 問句功能詞。目的不是語言學完備,而是把「不能當 grep anchor 的詞」剔掉。
_STOPWORDS = frozenset("""
a an the and or but if then else so of in on at to from by with without for as is are was were be
been being am do does did done have has had having will would shall should can could may might must
i me my mine you your yours he him his she her hers it its we us our ours they them their theirs
this that these those there here what which who whom whose when where why how whats
not no nor only own same too very just also than more most much many some any all both each few
about into over under again further once during before after above below up down out off between
tell say said asked ask know remember mentioned mention talk talked told
""".split())

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def content_words(text: str) -> list[str]:
    """依序回傳去重後的內容詞(小寫);數字保留,單字母丟掉。"""
    seen: set[str] = set()
    out: list[str] = []
    for w in _WORD_RE.findall(str(text).lower()):
        if len(w) < 2 and not w.isdigit():
            continue
        if w in _STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _kw_re(word: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9'-])", re.IGNORECASE)


def analyze_question(src_csv: Path, *, max_df: int) -> dict | None:
    df = pd.read_csv(src_csv)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    if "has_answer" not in df.columns or "content" not in df.columns:
        return None
    gold_mask = df["has_answer"] == True  # noqa: E712
    if not gold_mask.any():
        return None

    question = str(df["question"].dropna().iloc[0])
    answer = str(df["answer"].dropna().iloc[0]) if "answer" in df.columns and df["answer"].notna().any() else ""
    turns = df["content"].fillna("").astype(str).tolist()
    gold_idx = set(df.index[gold_mask].tolist())

    q_words = content_words(question)
    a_words = [w for w in content_words(answer) if w not in set(q_words)]

    # 每個 question keyword 在 haystack 的命中 turn 集(document frequency)
    q_hits: dict[str, set[int]] = {}
    for w in q_words:
        pat = _kw_re(w)
        q_hits[w] = {i for i, t in enumerate(turns) if pat.search(t)}
    a_pats = [_kw_re(w) for w in a_words]

    per_gold = []
    for gi in sorted(gold_idx):
        text = turns[gi]
        matched = [w for w, hits in q_hits.items() if gi in hits]
        r_any = bool(matched)
        r_useful = any(len(q_hits[w]) <= max_df for w in matched)
        r_pair = len(matched) >= 2
        r_ans = any(p.search(text) for p in a_pats)
        per_gold.append(dict(r_any=r_any, r_useful=r_useful, r_pair=r_pair, r_ans=r_ans))

    n = len(per_gold)
    return dict(
        name=src_csv.stem,
        n_gold=n,
        n_turns=len(turns),
        any=sum(g["r_any"] for g in per_gold),
        useful=sum(g["r_useful"] for g in per_gold),
        pair=sum(g["r_pair"] for g in per_gold),
        ans=sum(g["r_ans"] for g in per_gold),
        all_any=all(g["r_any"] for g in per_gold),
        all_useful=all(g["r_useful"] for g in per_gold),
        all_useful_or_ans=all(g["r_useful"] or g["r_ans"] for g in per_gold),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-df", type=int, default=40,
                    help="keyword 命中 turn 數上限,超過視為 grep 結果太吵(default 40)")
    ap.add_argument("--per-question-csv", default=None, help="輸出每題明細 CSV 路徑")
    args = ap.parse_args()

    agg: dict[str, dict] = defaultdict(lambda: dict(
        n_q=0, n_gold=0, any=0, useful=0, pair=0, ans=0,
        all_any=0, all_useful=0, all_useful_or_ans=0))
    rows = []

    for cat in CATEGORIES:
        cdir = DATA_ROOT / cat
        if not cdir.exists():
            continue
        for p in sorted(cdir.glob("*.csv")):
            r = analyze_question(p, max_df=args.max_df)
            if r is None:
                continue
            d = agg[cat]
            d["n_q"] += 1
            d["n_gold"] += r["n_gold"]
            for k in ("any", "useful", "pair", "ans"):
                d[k] += r[k]
            for k in ("all_any", "all_useful", "all_useful_or_ans"):
                d[k] += r[k]
            rows.append(dict(category=cat, **r))

    def pct(a, b):
        return f"{100*a/b:5.1f}%" if b else "  n/a"

    print(f"\n=== grep 可達率(max_df={args.max_df})===")
    header = (f"{'category':<28} {'#q':>4} {'#gold':>6} | "
              f"{'R_any':>7} {'R_useful':>8} {'R_pair':>7} {'R_ans':>7} | "
              f"{'全any':>7} {'全useful':>8} {'全use|ans':>9}")
    print(header)
    print("-" * len(header))
    tot = dict(n_q=0, n_gold=0, any=0, useful=0, pair=0, ans=0, all_any=0, all_useful=0, all_useful_or_ans=0)
    for cat in CATEGORIES:
        d = agg.get(cat)
        if not d:
            continue
        for k in tot:
            tot[k] += d[k]
        print(f"{cat:<28} {d['n_q']:>4} {d['n_gold']:>6} | "
              f"{pct(d['any'], d['n_gold']):>7} {pct(d['useful'], d['n_gold']):>8} "
              f"{pct(d['pair'], d['n_gold']):>7} {pct(d['ans'], d['n_gold']):>7} | "
              f"{pct(d['all_any'], d['n_q']):>7} {pct(d['all_useful'], d['n_q']):>8} "
              f"{pct(d['all_useful_or_ans'], d['n_q']):>9}")
    print("-" * len(header))
    print(f"{'TOTAL':<28} {tot['n_q']:>4} {tot['n_gold']:>6} | "
          f"{pct(tot['any'], tot['n_gold']):>7} {pct(tot['useful'], tot['n_gold']):>8} "
          f"{pct(tot['pair'], tot['n_gold']):>7} {pct(tot['ans'], tot['n_gold']):>7} | "
          f"{pct(tot['all_any'], tot['n_q']):>7} {pct(tot['all_useful'], tot['n_q']):>8} "
          f"{pct(tot['all_useful_or_ans'], tot['n_q']):>9}")

    if args.per_question_csv:
        pd.DataFrame(rows).to_csv(args.per_question_csv, index=False)
        print(f"\nper-question 明細 → {args.per_question_csv}")


if __name__ == "__main__":
    main()
