"""統一實驗報表:輸入 benchmark + run_tag(實驗資料夾),輸出
  - OVERALL 正確率 + 各分類正確率
  - fb% / kept / added / dropped(agent 定性)

LongMem:正確率讀 `correctness_new`,分類 = 目錄名(6 類)。
LoCoMo :正確率讀 `correctness_3vote`,分類由 `locomo10.json` 的 category
         join question(1=multi-hop 2=temporal 3=open-domain 4=single-hop,5=adversarial 排除)。

裁決/hint run(本身無 agent trace):
  - 正確率算「全題集」——base 有、run 沒有的題(fallback)沿用 base 正確率(需 --base)。
  - fb%/kept/added/dropped 取自 --base(這類工具不重跑 agent);裁決 run 另印裁決後 kept。

用法:
  python experiment/report_run.py longmem longmem_ff_v1
  python experiment/report_run.py longmem longmem_20B_ff_adj --base longmem_ff_v1
  python experiment/report_run.py locomo  locomo_20b_ff
  python experiment/report_run.py locomo  locomo_4b_hyp --base locomo_4b_ff
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
LM_OUT = _ROOT / "experiment" / "longmem" / "output"
LC_OUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
LC_DATA = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
LM_CATS = ["single_session_user", "single_session_assistant", "multi_session",
           "single_session_preference", "temporal_reasoning", "knowledge_update"]
LC_CATN = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_SKIP = ("all_answers", "progress")


def _as01(x):
    try:
        return int(float(str(x).strip()))
    except (ValueError, TypeError):
        return None


def _num(v):  # LongMem 存 list、LoCoMo 存 int/list,都轉數量
    return len(v) if isinstance(v, (list, tuple, set)) else (float(v) if isinstance(v, (int, float)) else 0)


def _agent_metrics(bench: str, tag: str) -> dict | None:
    """fb%/kept/added/dropped——讀 trace,依 sample 去重取最後一筆。"""
    if bench == "longmem":
        pat = str(LM_OUT / tag / "*" / "_grep_agent_traces.jsonl")
    else:
        pat = str(LC_OUT / tag / "sample_*" / "_grep_traces.jsonl")
    latest = {}
    for f in glob.glob(pat):
        for line in open(f):
            if not line.strip():
                continue
            d = json.loads(line)
            latest[(f, d.get("sample") or d.get("question", "")[:60])] = d
    rows = list(latest.values())
    if not rows:
        return None
    n = len(rows)
    fb = sum(1 for r in rows if r.get("fallback"))

    def avg(field):
        vs = [_num(r[field]) for r in rows if r.get(field) is not None]
        return sum(vs) / len(vs) if vs else 0.0
    return {"n": n, "fb%": 100 * fb / n, "kept": avg("kept"),
            "added": avg("added"), "dropped": avg("dropped")}


def _lm_scores(tag: str) -> dict:
    """(cat, stem) -> 0/1(correctness_new)。"""
    m = {}
    for f in glob.glob(str(LM_OUT / tag / "*" / "*.csv")):
        if any(s in f for s in _SKIP):
            continue
        v = _as01(pd.read_csv(f).iloc[0].get("correctness_new"))
        if v is not None:
            m[(f.split("/")[-2], f.split("/")[-1])] = v
    return m


def _lc_scores(tag: str, q2c: dict) -> dict:
    """(category_int, question) -> 0/1(correctness_3vote)。"""
    m = {}
    for f in glob.glob(str(LC_OUT / tag / "sample_*" / "*_judge_4omini.csv")):
        d = pd.read_csv(f)
        if "correctness_3vote" not in d.columns:
            continue
        for _, r in d.iterrows():
            q = str(r.get("question", "")).strip()
            v = _as01(r.get("correctness_3vote"))
            if v is not None:
                m[(q2c.get(q), q)] = v
    return m


def _print_percat(per: dict, names: dict, order):
    tot = [0, 0]
    print(f"  {'category':<26}{'acc':>10}")
    for c in order:
        k = per.get(c)
        if k and k[1]:
            tot[0] += k[0]; tot[1] += k[1]
            print(f"  {names[c]:<26}{k[0]}/{k[1]:<5} {100*k[0]/k[1]:5.1f}%")
    if tot[1]:
        print(f"  {'OVERALL':<26}{tot[0]}/{tot[1]:<5} {100*tot[0]/tot[1]:5.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["longmem", "locomo"])
    ap.add_argument("run_tag")
    ap.add_argument("--base", default=None, help="裁決/hint run:base run(繼承四項數據 + fallback 補全題集)")
    args = ap.parse_args()
    print(f"===== {args.benchmark} / {args.run_tag}"
          + (f"  (base={args.base})" if args.base else "") + " =====")

    if args.benchmark == "longmem":
        run = _lm_scores(args.run_tag)
        keyset = run
        if args.base:  # 全題集:fallback 用 base
            base = _lm_scores(args.base)
            merged = {k: run.get(k, base[k]) for k in base}
            keyset = merged
        per = defaultdict(lambda: [0, 0])
        for (cat, _stem), v in keyset.items():
            per[cat][0] += v; per[cat][1] += 1
        _print_percat({c: per[c] for c in LM_CATS}, {c: c for c in LM_CATS}, LM_CATS)
    else:
        q2c = {str(qa.get("question", "")).strip(): qa.get("category")
               for s in json.loads(LC_DATA.read_text()) for qa in s.get("qa", [])}
        run = _lc_scores(args.run_tag, q2c)
        per = defaultdict(lambda: [0, 0])
        for (cat, _q), v in run.items():
            if cat in (5, None):
                continue
            per[cat][0] += v; per[cat][1] += 1
        _print_percat(per, LC_CATN, [1, 2, 3, 4])

    # 四項數據:run 自己有 trace 就用,否則取 base
    m = _agent_metrics(args.benchmark, args.run_tag)
    src = args.run_tag
    if m is None and args.base:
        m = _agent_metrics(args.benchmark, args.base); src = f"base={args.base}"
    if m:
        print(f"  [agent 數據 from {src}] fb%={m['fb%']:.1f} kept={m['kept']:.1f} "
              f"added={m['added']:.2f} dropped={m['dropped']:.1f}  (n={m['n']})")
    else:
        print("  [agent 數據] 無 trace(裁決/hint run 請加 --base)")

    # 裁決/hint run 專屬:裁決後 kept、hypothesis 覆蓋率
    csv_pat = (str(LM_OUT / args.run_tag / "*" / "*.csv") if args.benchmark == "longmem"
               else str(LC_OUT / args.run_tag / "sample_*" / f"sample*_eval_{args.run_tag}.csv"))
    cov_t = cov_h = 0
    bf = ak = fa = na = 0
    for f in glob.glob(csv_pat):
        if any(s in f for s in _SKIP):
            continue
        d = pd.read_csv(f)
        if "hypothesis" in d.columns:
            for h in d["hypothesis"]:
                cov_t += 1
                if pd.notna(h) and str(h).strip().upper() not in ("", "NONE", "NAN"):
                    cov_h += 1
        if "n_final_adj" in d.columns:
            for _, r in d.iterrows():
                if pd.notna(r.get("n_final_adj")):
                    bf += _num(r.get("n_base_final", 0)); ak += _num(r.get("n_adj_kept", 0))
                    fa += _num(r.get("n_final_adj", 0)); na += 1
    if na:
        print(f"  [裁決] base_final={bf/na:.1f} 補回={ak/na:.1f} → kept(裁決後)={fa/na:.1f}")
    if cov_t:
        print(f"  [hypothesis 覆蓋率] {cov_h}/{cov_t} = {100*cov_h/cov_t:.1f}%")


if __name__ == "__main__":
    main()
