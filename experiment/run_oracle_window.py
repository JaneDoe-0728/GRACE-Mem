#!/usr/bin/env python3
"""LoCoMo / LongMem oracle:gold evidence **± N turns** 版本(修正窄證據問題)。

背景:兩個 benchmark 的 gold evidence 都只標「最小必要 turn」,導致代名詞失去
先行詞、問題預設的前提不在證據裡。實測窄證據 oracle 反而輸給實際檢索 baseline。
本腳本把每個 gold turn 的前後 N 個 turn(同 session)一併納入 context。

展開機制(各 benchmark 不同):
  - LoCoMo : patch _format_locomo_evidence —— ev_ids(D<s>:<t>)依 turn 編號 ±N。
  - LongMem: patch _format_longmem_evidence —— per-question CSV 中 has_answer 的 row,
             在同 session_id 內依 turn_index ±N 補齊鄰居後,交還原 formatter。

judge 一律沿用 category-aware 版(prompts/judge.py 的 per-category rubric),
與 rejudge_output_dirs 同口徑;.pyc 內建的舊 judge_single 簽名已不相容,故覆寫。

用法:
    uv run python experiment/run_oracle_window.py --benchmark both --window 2 --workers 32
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DIA = re.compile(r"^D(\d+):(\d+)$")


def _load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.search(r'OPENAI_API_KEY="?(sk-[^"\s]+)', line)
        if m:
            return m.group(1)
    sys.exit("[error] 找不到 OPENAI_API_KEY")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="both", choices=["locomo", "longmem", "both"])
    ap.add_argument("--window", type=int, default=2, help="每個 gold turn 前後各取 N 個 turn")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--photo", default="both", choices=["both", "yes", "no"],
                    help="僅 LoCoMo 適用")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--answer-api", default="https://api.openai.com/v1")
    ap.add_argument("--answer-model", default="gpt-4o-mini")
    ap.add_argument("--judge-api", default="https://api.openai.com/v1")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    args = ap.parse_args()

    os.environ["OPENAI_API_KEY"] = _load_openai_key()
    os.environ["LLM_API"] = args.answer_api
    os.environ["MODEL_NAME"] = args.answer_model
    os.environ["JUDGE_LLM_API"] = args.judge_api
    os.environ["JUDGE_MODEL_NAME"] = args.judge_model

    sys.path.insert(0, str(ROOT / "experiment"))
    sys.path.insert(0, str(ROOT))

    # LongMem 的 category-aware judge(.pyc 內建簽名已不相容,見 run_oracle_4omini)
    import experiment.longmem.stage_adapter as _sa
    from experiment.longmem.prompts import build_judge_messages as _bjm
    from experiment.longmem.rejudge_output_dirs import _parse_correct as _pc

    def _cat_judge(*, llm, question, gold, generated, category=None):
        msgs = _bjm(question=question, gold=gold, generated=generated, category=category)
        resp = llm.chat(messages=msgs, temperature=0.0, max_tokens=256)
        return _pc((resp.choices[0].message.content or "").strip())

    _sa.judge_single = _cat_judge

    spec = importlib.util.spec_from_file_location(
        "oracle_gold_eval", ROOT / "experiment" / "oracle_gold_eval.pyc")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    W = args.window

    # ── LoCoMo:ev_ids ±W(同 session,依 turn 編號) ──
    orig_loco = oracle._format_locomo_evidence

    def _expand_loco(ev_ids, dia_index):
        out, seen = [], set()
        for sid in ev_ids or []:
            m = _DIA.match(str(sid).strip())
            if not m:
                if sid not in seen:
                    seen.add(sid); out.append(sid)
                continue
            s, t = int(m.group(1)), int(m.group(2))
            for k in range(t - W, t + W + 1):
                cand = f"D{s}:{k}"
                if k >= 1 and cand in dia_index and cand not in seen:
                    seen.add(cand); out.append(cand)
        return out

    oracle._format_locomo_evidence = lambda ev, di: orig_loco(_expand_loco(ev, di), di)

    # ── LongMem:has_answer 的 row,同 session_id 內 turn_index ±W 補齊 ──
    orig_lm = oracle._format_longmem_evidence

    def _truthy(v):
        return str(v).strip().lower() in ("true", "1")

    def _patched_lm(df):
        d = df.copy().reset_index(drop=True)
        if "has_answer" not in d.columns:
            return orig_lm(df)
        ha = d["has_answer"].map(_truthy)
        tidx = d["turn_index"] if "turn_index" in d.columns else d.index
        sid = d["session_id"] if "session_id" in d.columns else 0
        gold = d.index[ha].tolist()
        keep = set(gold)
        for g in gold:
            gs = sid.iloc[g] if hasattr(sid, "iloc") else sid
            gt = int(tidx.iloc[g]) if hasattr(tidx, "iloc") else g
            for j in d.index:
                sj = sid.iloc[j] if hasattr(sid, "iloc") else sid
                tj = int(tidx.iloc[j]) if hasattr(tidx, "iloc") else j
                if sj == gs and abs(tj - gt) <= W:
                    keep.add(j)
        d.loc[sorted(keep), "has_answer"] = True
        return orig_lm(d)

    oracle._format_longmem_evidence = _patched_lm

    # ── 組 arms ──
    arms = []  # (label, benchmark, photo, out)
    if args.benchmark in ("both", "locomo"):
        if args.photo in ("both", "no"):
            arms.append((f"LoCoMo ±{W}(無照片)", "locomo", False,
                         ROOT / f"experiment/oracle_win{W}_locomo_nophoto{args.suffix}"))
        if args.photo in ("both", "yes"):
            arms.append((f"LoCoMo ±{W}(有照片)", "locomo", True,
                         ROOT / f"experiment/oracle_win{W}_locomo_photo{args.suffix}"))
    if args.benchmark in ("both", "longmem"):
        arms.append((f"LongMem ±{W}", "longmem", True,
                     ROOT / f"experiment/oracle_win{W}_longmem{args.suffix}"))

    for label, bench, photo, out in arms:
        print("\n" + "=" * 72, flush=True)
        print(f"### {label} | 答題={args.answer_model} @ {args.answer_api}", flush=True)
        print(f"### judge={args.judge_model} | window=±{W} | workers={args.workers} -> {out}", flush=True)
        print("=" * 72, flush=True)
        oracle._INCLUDE_PHOTO = photo
        out.mkdir(parents=True, exist_ok=True)
        runner = oracle.run_locomo if bench == "locomo" else oracle.run_longmem
        runner(args.limit, args.workers, out, 0)


if __name__ == "__main__":
    main()
