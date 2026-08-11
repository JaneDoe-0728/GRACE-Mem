"""Self-consistency vote merge (LoCoMo): 三票(fo2 / fo2r / vote3)等價分群取
多數,輸出投票後的逐題判分。

- 等價分群:gpt-4o-mini 一題一 call,判三個答案哪些給出同一實質答案。
- 多數代表:優先取多數群中最低票序(fo2 > fo2r > vote3);三方分裂保留 fo2。
- 判分重用:代表是 fo2/fo2r 時直接用其既有 correctness_4omini;只有代表是
  vote3 新答案時才需新判(--judge 開啟)。

Usage:
    python experiment/locomo/vote_merge.py --out /tmp/vote_result.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from KG.llm import LLMClient
from experiment.locomo.rejudge_4omini import _openai_key, judge_4omini

OUT = _ROOT / "experiment" / "locomo" / "output" / "standard"

CLUSTER_SYSTEM = """You get a QUESTION and three candidate ANSWERS (1,2,3) from the same system.
Decide which answers give the SAME substantive answer to the question (same
entity/date/value — wording may differ). Reply ONLY with JSON:
{"groups": [[...],[...]]}   e.g. {"groups": [[1,3],[2]]} or {"groups": [[1,2,3]]}"""


def load_4o(run):
    frames = []
    for f in sorted((OUT / run).glob("sample_*/*judge_4omini.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["sample"] = f.parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["c4o"] = pd.to_numeric(df["correctness_4omini"], errors="coerce")
    df["key"] = df["sample"] + "||" + df["question"].astype(str).str.strip()
    return df.drop_duplicates("key").set_index("key")


def load_raw(run):
    frames = []
    for f in sorted((OUT / run).glob("sample_*/*_eval_*.csv")):
        if "_judge" in f.name:
            continue
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["sample"] = f.parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["sample"] + "||" + df["question"].astype(str).str.strip()
    return df.drop_duplicates("key").set_index("key")


def cluster(llm, question: str, answers: list[str]) -> list[list[int]]:
    body = "\n\n".join(f"ANSWER {i+1}: {a[:600]}" for i, a in enumerate(answers))
    try:
        resp = llm.chat(messages=[
            {"role": "system", "content": CLUSTER_SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\n\n{body}"},
        ], temperature=0.0, max_tokens=200)
        text = (resp.choices[0].message.content or "")
        m = re.search(r"\{.*\}", text, re.S)
        groups = json.loads(m.group(0))["groups"]
        seen = set()
        out = []
        for g in groups:
            g = [int(x) for x in g if int(x) in (1, 2, 3) and int(x) not in seen]
            seen.update(g)
            if g:
                out.append(g)
        for x in (1, 2, 3):
            if x not in seen:
                out.append([x])
        return out
    except Exception:
        return [[1], [2], [3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=3, default=["locomo-n8-120b-fo2", "locomo-n8-120b-fo2r", "vote3"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    a = load_4o(args.runs[0])
    b = load_4o(args.runs[1])
    c = load_raw(args.runs[2])
    keys = [k for k in c.index if k in a.index and k in b.index]
    if args.limit:
        keys = keys[: args.limit]
    print(f"voting over {len(keys)} questions")

    llm = LLMClient(base_url="https://api.openai.com/v1", model_name="gpt-4o-mini",
                    api_key=_openai_key())

    rows = []
    n_judged = 0
    for n, k in enumerate(keys, 1):
        q = k.split("||", 1)[1]
        ans = [str(a.loc[k, "model_answer"]), str(b.loc[k, "model_answer"]),
               str(c.loc[k, "model_answer"])]
        groups = cluster(llm, q, ans)
        maj = max(groups, key=len)
        rep = min(maj)  # 1-based
        if len(maj) == 1:
            rep = 1  # 三方分裂 → 保留 fo2
        if rep == 1:
            corr = a.loc[k, "c4o"]
        elif rep == 2:
            corr = b.loc[k, "c4o"]
        else:
            corr = judge_4omini(llm, question=q,
                                gold=str(a.loc[k, "gold_answer"]),
                                gen=ans[2])
            n_judged += 1
        rows.append({"key": k, "rep": rep, "corr": corr,
                     "category_label": a.loc[k].get("category_label")})
        if n % 100 == 0:
            print(f"  {n}/{len(keys)} (new judgments: {n_judged})", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    print(f"done. voted acc on slice: {100*pd.to_numeric(df.corr_ if hasattr(df,'corr_') else df['corr']).mean():.2f}%  new judgments={n_judged}")


if __name__ == "__main__":
    main()
