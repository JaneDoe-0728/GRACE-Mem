"""Self-consistency vote merge (LoCoMo): cluster the three votes (fo2 / fo2r /
vote3) by equivalence, take the majority, and emit the post-vote per-question
scores.

- Equivalence clustering: one gpt-4o-mini call per question, deciding which of
  the three answers say substantially the same thing.
- Majority representative: prefer the lowest vote order within the majority
  cluster (fo2 > fo2r > vote3); a three-way split keeps fo2.
- Score reuse: when the representative is fo2 or fo2r, its existing
  correctness_4omini is used as is. A fresh judgement is only needed when the
  representative is a new vote3 answer (enabled with --judge).

Usage:
    python -m experiment.locomo.analysis.vote_merge --out /tmp/vote_result.csv
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from experiment.common.paths import REPO_ROOT

OUT = REPO_ROOT / "experiment" / "locomo" / "output" / "standard"

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
    """Group differing answers across runs into clusters of equivalent ones.

    Runs phrase the same answer differently, so a plain string comparison
    reports disagreement that is not there. Clustering first makes the majority
    vote count meanings rather than spellings.
    """
    body = "\n\n".join(f"ANSWER {i+1}: {a[:600]}" for i, a in enumerate(answers))
    try:
        resp = llm.chat(messages=[
            {"role": "system", "content": CLUSTER_SYSTEM},
            {"role": "user", "content": f"QUESTION: {question}\n\n{body}"},
        ], temperature=0.0, max_tokens=200)
        text = (resp.choices[0].message.content or "")
        m = re.search(r"\{.*\}", text, re.DOTALL)
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
    from experiment.common.evaluation.judge import JudgeEngine, openai_api_key
    from grace_mem.adapters.llm import LLMClient

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
                    api_key=openai_api_key())
    judge = JudgeEngine(llm, "locomo")

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
            rep = 1  # a three-way split keeps fo2
        if rep == 1:
            corr = a.loc[k, "c4o"]
        elif rep == 2:
            corr = b.loc[k, "c4o"]
        else:
            corr = judge.judge(
                question=q,
                gold=str(a.loc[k, "gold_answer"]),
                generated=ans[2],
            )
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
