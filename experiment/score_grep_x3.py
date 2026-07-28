#!/usr/bin/env python3
"""Score 3 grep-replay runs: per-category + overall accuracy, with 3-run mean±SD.

Auto-detects LongMem vs LoCoMo layout, and auto-picks the correctness column.

Usage:
    python experiment/score_grep_x3.py <tag_or_path1> <tag2> <tag3> [--col COL]

    # LoCoMo（自動判別、自動選 correctness 欄）
    uv run python experiment/score_grep_x3.py locomo-n8-120b-grep-r1 locomo-n8-120b-grep-r2 locomo-n8-120b-grep-r3

    # LongMem（可指定欄位，例如比 20b judge vs 4o-mini judge）
    uv run python experiment/score_grep_x3.py rr16-20b-grep-r{1,2,3} --col correctness_4omini
    uv run python experiment/score_grep_x3.py rr16-20b-grep-r{1,2,3} --col correctness_20b

    # 也可直接給完整路徑
    uv run python experiment/score_grep_x3.py experiment/locomo/output/standard/locomo-n4-20b-grep-r{1,2,3}
    
  <tag_or_path>: either a full run dir, or a bare run-tag resolved under
    experiment/longmem/output/<tag>  (LongMem)  or
    experiment/locomo/output/standard/<tag>  (LoCoMo).

  --col: correctness column to score. Default = auto (first present of
    correctness_4omini, correctness_20b, correctness_20b63, correctness_20b92,
    correctness_new, correctness).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics as st
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONGMEM_OUT = os.path.join(_ROOT, "experiment", "longmem", "output")
LOCOMO_OUT = os.path.join(_ROOT, "experiment", "locomo", "output", "standard")
LOCOMO_JSON = os.path.join(_ROOT, "experiment", "locomo", "data", "locomo10.json")

LM_CATS = ["single_session_user", "single_session_assistant", "single_session_preference",
           "multi_session", "knowledge_update", "temporal_reasoning"]
LOCOMO_CAT = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial"}
LOCOMO_ORDER = ["Single-hop", "Multi-hop", "Temporal", "Open-domain"]
COL_CANDIDATES = ["correctness_4omini", "correctness_20b", "correctness_20b63",
                  "correctness_20b92", "correctness_new", "correctness"]


def _to01(v):
    v = str(v).strip()
    if not v or v.lower() in ("nan", "none"):
        return None
    try:
        return 1 if float(v) >= 0.5 else 0
    except ValueError:
        return 1 if v.lower() in ("1", "true", "correct", "yes") else 0


def _resolve(tag_or_path):
    if os.path.isdir(tag_or_path):
        return os.path.abspath(tag_or_path)
    for root in (LONGMEM_OUT, LOCOMO_OUT):
        p = os.path.join(root, tag_or_path)
        if os.path.isdir(p):
            return p
    sys.exit(f"[error] run not found: {tag_or_path}")


def _detect(run_dir):
    if glob.glob(os.path.join(run_dir, "sample_*")):
        return "locomo"
    if any(os.path.isdir(os.path.join(run_dir, c)) for c in LM_CATS):
        return "longmem"
    sys.exit(f"[error] cannot detect benchmark layout for {run_dir}")


def _pick_col(run_dir, kind, forced):
    # sample one row and see which correctness col is present + populated
    if kind == "longmem":
        files = [f for c in LM_CATS for f in glob.glob(os.path.join(run_dir, c, "*.csv"))
                 if not os.path.basename(f).startswith(("all_answers", "progress"))]
    else:
        files = glob.glob(os.path.join(run_dir, "sample_*", "*_judge*.csv"))
    cols = set()
    for f in files[:5]:
        try:
            with open(f, encoding="utf-8-sig") as fh:
                cols |= set(next(csv.reader(fh)))
        except Exception:
            pass
    avail = sorted(c for c in cols if "correct" in c)
    if forced:
        if forced not in cols:
            sys.exit(f"[error] --col {forced} not found in {os.path.basename(run_dir)}.\n"
                     f"        available correctness columns: {avail}\n"
                     f"        (note: 120b grep was judged on .63 -> 'correctness_20b63'; "
                     f"20b grep on .92 -> 'correctness_20b')")
        return forced
    for c in COL_CANDIDATES:
        if c in cols:
            return c
    sys.exit(f"[error] no correctness column in {run_dir}; have: {avail}")


def _score_longmem(run_dir, col):
    per = {c: [] for c in LM_CATS}
    for c in LM_CATS:
        for f in glob.glob(os.path.join(run_dir, c, "*.csv")):
            if os.path.basename(f).startswith(("all_answers", "progress")):
                continue
            try:
                with open(f, encoding="utf-8-sig") as fh:
                    row = next(csv.DictReader(fh), None)
            except Exception:
                continue
            if not row or col not in row:
                continue
            r = _to01(row.get(col))
            if r is not None:
                per[c].append(r)
    return per


_LOCOMO_QCAT = None


def _locomo_qcat():
    global _LOCOMO_QCAT
    if _LOCOMO_QCAT is None:
        data = json.load(open(LOCOMO_JSON))
        _LOCOMO_QCAT = {}
        for si, sample in enumerate(data):
            for qa in sample.get("qa", []):
                _LOCOMO_QCAT[(si, str(qa.get("question", "")).strip())] = qa.get("category")
    return _LOCOMO_QCAT


def _score_locomo(run_dir, col):
    import pandas as pd
    qcat = _locomo_qcat()
    per = {}
    for si in range(10):
        for f in glob.glob(os.path.join(run_dir, f"sample_{si}", "*_judge*.csv")):
            df = pd.read_csv(f)
            if col not in df.columns:
                continue
            for _, r in df.iterrows():
                v = _to01(r.get(col))
                if v is None:
                    continue
                cat = LOCOMO_CAT.get(qcat.get((si, str(r.get("question", "")).strip())), "?")
                if cat == "Adversarial":
                    continue
                per.setdefault(cat, []).append(v)
    return per


# ─────────────────────────── agent filter 統計 ───────────────────────────
# grep trace 落點:LongMem = <cat>/_grep_agent_traces.jsonl(完整 trace);
# LoCoMo = sample_*/_grep_traces.jsonl(舊 run 精簡:只有 fallback/kept/added;
# 新 run 已改為完整 trace)。欄位缺就跳過該指標,不會報錯。
_AGENT_TRACE_GLOBS = {
    "longmem": [os.path.join("*", "_grep_agent_traces.jsonl")],
    "locomo": [os.path.join("sample_*", "_grep_traces.jsonl"),
               os.path.join("sample_*", "_grep_agent_traces.jsonl")],
}

# (key, 顯示名, 格式)——只印至少一個 run 有值的列
AGENT_METRICS = [
    ("fallback_pct",  "fallback 率(%)",       "{:7.2f}"),
    ("total_sec",     "agent 時間/題 (s)",     "{:7.3f}"),
    ("total_sum",     "agent 時間 總計 (s)",   "{:7.1f}"),
    ("llm_total_sec", "└ 其中 LLM/題 (s)",     "{:7.3f}"),
    ("n_llm_calls",   "LLM 呼叫/題",           "{:7.2f}"),
    ("tool_calls",    "tool calls/題",         "{:7.2f}"),
    ("main_loop_sec", "main_loop/題 (s)",      "{:7.3f}"),
    ("verify_sec",    "verify/題 (s)",         "{:7.3f}"),
    ("verify_pct",    "verify 觸發率(%)",      "{:7.2f}"),
    ("kept",          "kept/題",               "{:7.2f}"),
    ("added",         "added/題",              "{:7.2f}"),
    ("dropped",       "dropped/題",            "{:7.2f}"),
]


def _len_or_int(v):
    if isinstance(v, list):
        return len(v)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    return None


def _agent_traces(run_dir, kind):
    recs = []
    for g in _AGENT_TRACE_GLOBS.get(kind, []):
        for f in glob.glob(os.path.join(run_dir, g)):
            try:
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def _agent_stats(run_dir, kind):
    """回傳單一 run 的 agent 指標 dict(缺的指標為 None)。無 trace 檔回 None。"""
    recs = _agent_traces(run_dir, kind)
    if not recs:
        return None

    def _tim(r, k):
        t = r.get("timing")
        return t.get(k) if isinstance(t, dict) else None

    def _mean(vals):
        xs = [v for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None

    def _sum(vals):
        xs = [v for v in vals if v is not None]
        return sum(xs) if xs else None

    n = len(recs)
    totals = [_tim(r, "total_sec") for r in recs]
    suff = [r.get("sufficiency") for r in recs]
    return {
        "n": n,
        "fallback_pct": 100.0 * sum(1 for r in recs if r.get("fallback")) / n,
        "fallback_reasons": collections.Counter(
            r.get("fallback") for r in recs if r.get("fallback")),
        "total_sec": _mean(totals),
        "total_sum": _sum(totals),
        "llm_total_sec": _mean([_tim(r, "llm_total_sec") for r in recs]),
        "n_llm_calls": _mean([_tim(r, "n_llm_calls") for r in recs]),
        "main_loop_sec": _mean([_tim(r, "main_loop_sec") for r in recs]),
        "verify_sec": _mean([_tim(r, "verify_sec") for r in recs]),
        # verify 觸發率:sufficiency 有內容且任一輪判 insufficient
        "verify_pct": 100.0 * sum(
            1 for s in suff if isinstance(s, list) and any(
                not x.get("sufficient", True) for x in s)) / n,
        "tool_calls": _mean([len(r["commands"]) if isinstance(r.get("commands"), list) else None
                             for r in recs]),
        "kept": _mean([_len_or_int(r.get("kept")) for r in recs]),
        "added": _mean([_len_or_int(r.get("added")) for r in recs]),
        "dropped": _mean([_len_or_int(r.get("dropped")) for r in recs]),
    }


def _report_agent(dirs, kind, ms_fn):
    stats = [_agent_stats(d, kind) for d in dirs]
    if not any(stats):
        return
    print("\n── agent filter(harness;時間僅 refine_context,不含答題/judge)──")
    hdr = " ".join(f"{'r'+str(i+1):>9s}" for i in range(len(dirs)))
    print(f"{'metric':22s} {hdr}   mean±SD")
    for key, label, fmt in AGENT_METRICS:
        vals = [(s.get(key) if s else None) for s in stats]
        if not any(v is not None for v in vals):
            continue
        cells = " ".join((fmt.format(v) if v is not None else f"{'--':>7s}") for v in vals)
        ms = ms_fn([v for v in vals if v is not None])
        mss = f"   {ms[0]:.3f}±{ms[1]:.3f}" if ms else ""
        print(f"{label:22s} {cells}{mss}")
    print("n traces per run:", [s["n"] if s else 0 for s in stats])

    # fallback reason 分佈(所有 run 加總,由多到少;標出最多的)
    agg = collections.Counter()
    for s in stats:
        if s:
            agg.update(s.get("fallback_reasons") or {})
    if agg:
        tot = sum(agg.values())
        print(f"\n  fallback reasons(共 {tot} 次,最多:{agg.most_common(1)[0][0]}):")
        for reason, cnt in agg.most_common():
            print(f"    {reason:16s} {cnt:5d}  ({100.0 * cnt / tot:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="3 run-tags or paths (any number works)")
    ap.add_argument("--col", default=None, help="correctness column (default: auto)")
    ap.add_argument("--no-agent", action="store_true",
                    help="不輸出 agent filter(fallback/時間/agent 數據)那一段")
    args = ap.parse_args()

    dirs = [_resolve(r) for r in args.runs]
    kinds = [_detect(d) for d in dirs]
    if len(set(kinds)) != 1:
        sys.exit(f"[error] mixed benchmarks: {list(zip(args.runs, kinds))}")
    kind = kinds[0]
    col = _pick_col(dirs[0], kind, args.col)

    per_runs = [(_score_longmem if kind == "longmem" else _score_locomo)(d, col) for d in dirs]
    cats = LM_CATS if kind == "longmem" else (
        [c for c in LOCOMO_ORDER if any(c in p for p in per_runs)])

    def _ms(vals):
        fin = [v for v in vals if v == v]  # drop NaN
        if not fin:
            return None
        return st.mean(fin), (st.pstdev(fin) if len(fin) > 1 else 0.0)

    print(f"benchmark={kind}  col={col}  runs={len(dirs)}")
    hdr = " ".join(f"{'r'+str(i+1):>8s}" for i in range(len(dirs)))
    print(f"{'category':26s} {hdr}   mean±SD")
    for c in cats:
        cells = [100 * sum(p[c]) / len(p[c]) if p.get(c) else float("nan") for p in per_runs]
        ms = _ms(cells)
        cellstr = " ".join((f"{x:7.1f}%" if x == x else f"{'--':>8s}") for x in cells)
        print(f"{c:26s} " + cellstr + (f"   {ms[0]:5.1f}±{ms[1]:.2f}" if ms else "   n/a"))
    ov = [100 * sum(x for c in p for x in p[c]) / max(sum(len(p[c]) for c in p), 1) for p in per_runs]
    ms = _ms(ov)
    print("-" * (26 + 9 * len(dirs) + 14))
    ovstr = " ".join((f"{x:7.2f}%" if x == x else f"{'--':>8s}") for x in ov)
    print(f"{'OVERALL':26s} " + ovstr + (f"   {ms[0]:5.2f}±{ms[1]:.2f}" if ms else "   n/a"))
    ns = [sum(len(p[c]) for c in p) for p in per_runs]
    print("n per run:", ns)

    if not args.no_agent:
        _report_agent(dirs, kind, _ms)


if __name__ == "__main__":
    main()
