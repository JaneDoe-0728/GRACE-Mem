"""Answer Tribunal v1 -- the first experiment in verification-native memory.

The paradigm is inverted: retrieval does not feed generation, it puts candidate
answers on trial.
  candidates = the differing answers several existing runs gave the same question
               (system diversity is a free candidate pool)
  trial      = lexical retrieval using the content words of the candidate answer
               plus the question, feeding the supporting evidence to a verifier
               that rules SUPPORTED / PARTIAL / UNSUPPORTED
  verdict    = most-supported wins; on a tie, majority vote, then the default run
               as the backstop

How this differs from self-consistency: it does not rely on how often an answer
was voted for but on **evidence binding** -- answers with a supporting span live,
answers without one die -- which turns "is this answer right" from a generation
problem into a retrieval and verification one.

Usage:
    python -m experiment.longmem.analysis.agent_filter_tribunal \
        --runs rr16-grep-split rr32-base-split rr32-grep rr16-grep-120bans rr16-base-split \
        --default-run rr16-grep-split --workers 2
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from rank_bm25 import BM25Okapi

from grace_mem.llm import LLMClient
from experiment.agent_filter.corpus import load_corpus

DATA = _ROOT / "experiment" / "longmem" / "script_data"
OUTPUT = _ROOT / "experiment" / "longmem" / "output"

TOK = re.compile(r"[a-z0-9][a-z0-9'-]*")
STOP = set("a an the and or but of in on at to from by with for as is are was were be been am "
           "do does did have has had i me my you your he she it we they this that what when where "
           "who whom how why not no yes if then so there here also just about".split())

VERIFIER_SYSTEM = """You are an evidence auditor. Given a QUESTION, a CANDIDATE ANSWER, and
EVIDENCE excerpts from the user's conversation history, judge whether the evidence
supports the candidate answer.

Reply with EXACTLY one word on the first line:
SUPPORTED   — the evidence contains the facts that make this answer correct
PARTIAL     — the evidence is consistent with the answer but incomplete
UNSUPPORTED — the evidence contradicts the answer, or contains nothing confirming it

Judge ONLY against the evidence shown. An answer claiming "not enough information"
is SUPPORTED only if the evidence indeed lacks the asked-for fact."""

VERDICT_SCORE = {"SUPPORTED": 2, "PARTIAL": 1, "UNSUPPORTED": 0}

# v2: relative adjudication -- shared evidence, all candidates side by side, and
# the incumbent is only overturned on high confidence
COMPARATIVE_SYSTEM = """You are an evidence judge. Given a QUESTION and several CANDIDATE answers
(labeled A, B, C, ...), plus EVIDENCE excerpts from the user's conversation history,
decide which candidate the evidence best supports.

Rules:
- Judge ONLY from the evidence shown; do not use outside knowledge.
- A candidate saying "not enough information" is best ONLY if the evidence truly
  lacks the asked-for fact.
- Candidate {incumbent} is the incumbent (came from the strongest system). Prefer it
  unless the evidence CLEARLY favors another candidate.

Reply with exactly two lines:
BEST: <letter>
CONFIDENCE: HIGH or LOW"""

_tls = threading.local()


def _llm() -> LLMClient:
    if getattr(_tls, "llm", None) is None:
        _tls.llm = LLMClient(timeout=300.0)
    return _tls.llm


def toks(t: str) -> list[str]:
    return [w for w in TOK.findall(str(t).lower()) if w not in STOP]


def load_run(run: str) -> dict:
    out = {}
    for f in glob.glob(str(OUTPUT / run / "*" / "*.csv")):
        p = Path(f)
        if p.stem in ("all_answers", "progress") or p.name.endswith(".lock"):
            continue
        try:
            df = pd.read_csv(f)
            r = df.iloc[0]
            out[(p.parent.name, p.stem)] = dict(
                ok=float(r.get("correctness_4o")) == 1.0,
                answer=str(r.get("Generated_Answer") or "").strip(),
                question=str(r.get("question") or "").strip(),
                qdate=str(r.get("question_date") or "").strip(),
            )
        except Exception:
            continue
    return out


def norm_answer(a: str) -> str:
    return re.sub(r"\W+", " ", a.lower()).strip()[:200]


def evidence_for(corpus, question: str, candidate: str, k: int = 6) -> str:
    docs = [toks(t.text) for t in corpus.turns]
    bm = BM25Okapi(docs)
    q = toks(question) + toks(candidate)
    scores = bm.get_scores(q)
    order = sorted(range(len(corpus.turns)), key=lambda i: -scores[i])[:k]
    lines = []
    for i in order:
        t = corpus.turns[i]
        body = " ".join(t.text.split())
        if len(body) > 800:
            body = body[:800] + "…"
        lines.append(f"[{t.date}] {t.role}: {body}")
    return "\n".join(lines)


def verify(question: str, qdate: str, candidate: str, evidence: str) -> int:
    msgs = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content":
            f"QUESTION: {question}\n"
            + (f"QUESTION DATE: {qdate}\n" if qdate else "")
            + f"\nCANDIDATE ANSWER: {candidate}\n\nEVIDENCE:\n{evidence}\n\nVerdict:"},
    ]
    resp = _llm().chat(messages=msgs, temperature=0.0, max_tokens=200)
    reply = (resp.choices[0].message.content or "").strip().upper()
    for word, sc in VERDICT_SCORE.items():
        if word in reply.split("\n")[0] or word in reply[:80]:
            return sc
    return 0


def arbitrate_comparative(key, cands, default_answer):
    """v2: shared evidence, side-by-side comparison, and an incumbent bias.
    cands=[(ans, n_runs, ok, q, qd), ...]"""
    cat, name = key
    corpus = load_corpus(DATA / cat / f"{name}.csv")
    question, qdate = cands[0][3], cands[0][4]
    # Shared evidence: the union of every candidate's words, to remove confirmation bias
    union_query = question + " " + " ".join(a for a, *_ in cands)
    ev = evidence_for(corpus, union_query, "", k=8)
    letters = "ABCDEFG"
    incumbent_idx = next((i for i, (a, *_r) in enumerate(cands)
                          if norm_answer(a) == norm_answer(default_answer)), 0)
    blocks = "\n\n".join(f"CANDIDATE {letters[i]}: {a}" for i, (a, *_r) in enumerate(cands))
    msgs = [
        {"role": "system", "content": COMPARATIVE_SYSTEM.format(incumbent=letters[incumbent_idx])},
        {"role": "user", "content":
            f"QUESTION: {question}\n" + (f"QUESTION DATE: {qdate}\n" if qdate else "")
            + f"\n{blocks}\n\nEVIDENCE:\n{ev}\n\nWhich candidate does the evidence best support?"},
    ]
    resp = _llm().chat(messages=msgs, temperature=0.0, max_tokens=200)
    reply = (resp.choices[0].message.content or "").upper()
    m = re.search(r"BEST\s*[::]\s*([A-G])", reply)
    hi = "HIGH" in reply
    pick = incumbent_idx
    if m:
        idx = letters.index(m.group(1))
        if idx < len(cands) and (hi or idx == incumbent_idx):
            pick = idx
    return cands[pick][2], {"pick": pick, "incumbent": incumbent_idx,
                            "high_conf": hi, "n_cands": len(cands)}


def arbitrate_one(key, cands, default_answer):
    """cands: list of (answer, n_runs, ok_flag). Returns (picked_ok, detail)."""
    cat, name = key
    src = DATA / cat / f"{name}.csv"
    corpus = load_corpus(src)
    question = cands[0][3]
    qdate = cands[0][4]
    scored = []
    for ans, n_runs, ok, q, qd in cands:
        ev = evidence_for(corpus, question, ans)
        sc = verify(question, qdate, ans, ev)
        scored.append((sc, n_runs, ans, ok))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    top = scored[0]
    # Nothing survived -> fall back to the default run's answer
    if top[0] == 0:
        picked_ok = next((ok for _, _, a, ok in scored if norm_answer(a) == norm_answer(default_answer)),
                         scored[0][3])
        return picked_ok, {"verdicts": [(s, n) for s, n, _, _ in scored], "fallback": True}
    return top[3], {"verdicts": [(s, n) for s, n, _, _ in scored], "fallback": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--default-run", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", choices=["absolute", "comparative"], default="comparative")
    ap.add_argument("--out", default="/tmp/tribunal_results.jsonl")
    args = ap.parse_args()

    data = {r: load_run(r) for r in args.runs}
    common = set.intersection(*[set(d) for d in data.values()])
    default = data[args.default_run]

    agree, disputes = [], []
    for k in sorted(common):
        verdicts = {data[r][k]["ok"] for r in args.runs}
        if len(verdicts) == 1:
            agree.append((k, data[args.default_run][k]["ok"]))
        else:
            # Candidates = the differing answers, carrying how many runs backed each
            # and what the judge said
            by_ans = {}
            for r in args.runs:
                d = data[r][k]
                key_a = norm_answer(d["answer"])
                if key_a not in by_ans:
                    by_ans[key_a] = [d["answer"], 0, d["ok"], d["question"], d["qdate"]]
                by_ans[key_a][1] += 1
            disputes.append((k, list(by_ans.values())))

    print(f"common={len(common)} agree={len(agree)} disputes={len(disputes)}", flush=True)

    fn = arbitrate_comparative if args.mode == "comparative" else arbitrate_one
    picked = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn, k, cands, default[k]["answer"]): k
                for k, cands in disputes}
        done = 0
        with open(args.out, "w") as fh:
            for fut in as_completed(futs):
                k = futs[fut]
                done += 1
                try:
                    ok, detail = fut.result()
                except Exception as e:  # noqa: BLE001
                    ok, detail = default[k]["ok"], {"error": str(e)[:200]}
                picked[k] = ok
                fh.write(json.dumps({"key": list(k), "picked_ok": ok, **detail}) + "\n")
                if done % 20 == 0 or done == len(disputes):
                    print(f"({done}/{len(disputes)})", flush=True)

    agree_ok = sum(ok for _, ok in agree)
    disp_ok = sum(picked.values())
    total_ok = agree_ok + disp_ok
    n = len(agree) + len(disputes)
    base_disp_ok = sum(default[k]["ok"] for k, _ in disputes)
    print(f"\nOverall accuracy after arbitration: {total_ok}/{n} = {100*total_ok/n:.1f}%")
    print(f"  where runs agreed: {agree_ok}/{len(agree)}")
    print(f"  where runs disagreed: arbitrated {disp_ok}/{len(disputes)} vs {args.default_run} {base_disp_ok}/{len(disputes)}")
    print(f"  reference: {args.default_run} overall = {100*(agree_ok+base_disp_ok)/n:.1f}%  ceiling = any candidate correct")


if __name__ == "__main__":
    main()
