#!/usr/bin/env python3
"""跑 gpt-4o-mini oracle:LoCoMo(有/無照片敘述)+ LongMem。

oracle_gold_eval 的 `_INCLUDE_PHOTO` 是模組層級常數(原作者靠改原始碼切換),
這裡改成 runtime 覆寫,一次跑完三個 arm 不必動原檔。
答題與 judge 都用 gpt-4o-mini;OPENAI_API_KEY 從 .env 讀。

用法:
    uv run python experiment/run_oracle_4omini.py --workers 32
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.search(r'OPENAI_API_KEY="?(sk-[^"\s]+)', line)
        if m:
            return m.group(1)
    sys.exit("[error] 找不到 OPENAI_API_KEY(env 或 .env 都沒有)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32, help="並發數(4o-mini 端點吃得住)")
    ap.add_argument("--limit", type=int, default=None, help="每個 arm 的題數上限(校準用)")
    ap.add_argument("--only", default="all", choices=["all", "locomo", "longmem"])
    ap.add_argument("--suffix", default="", help="輸出目錄後綴(重複跑取平均用,如 _r1)")
    ap.add_argument("--answer-api", default="https://api.openai.com/v1", help="答題端點")
    ap.add_argument("--answer-model", default="gpt-4o-mini", help="答題模型")
    ap.add_argument("--judge-api", default="https://api.openai.com/v1", help="judge 端點")
    ap.add_argument("--judge-model", default="gpt-4o-mini", help="judge 模型")
    ap.add_argument("--out-prefix", default="oracle", help="輸出目錄前綴(如 oracle_20b)")
    ap.add_argument("--photo", default="both", choices=["both", "yes", "no"],
                    help="LoCoMo 照片敘述:both=兩個都跑 / yes / no")
    args = ap.parse_args()

    # 答題與 judge 都指向 gpt-4o-mini —— 必須在 import oracle 模組前設好
    os.environ["OPENAI_API_KEY"] = _load_openai_key()
    os.environ["LLM_API"] = args.answer_api
    os.environ["MODEL_NAME"] = args.answer_model
    os.environ["JUDGE_LLM_API"] = args.judge_api
    os.environ["JUDGE_MODEL_NAME"] = args.judge_model

    sys.path.insert(0, str(ROOT / "experiment"))
    sys.path.insert(0, str(ROOT))

    # oracle(.pyc)是對「舊版 category-aware judge_single」編譯的,但本 branch 的
    # stage_adapter.judge_single 已改成不吃 category 的退化版 → TypeError。
    # 這裡把它換回 category-aware(prompts/judge.py 的 per-category rubric),
    # 與 rejudge_output_dirs 同口徑;直接丟掉 category 會讓 LongMem 分數失真。
    import experiment.longmem.stage_adapter as _sa
    from experiment.judge import JudgeEngine

    def _category_aware_judge(*, llm, question, gold, generated, category=None):
        return JudgeEngine(llm, "longmem").judge(
            question=question,
            gold=gold,
            generated=generated,
            category=category,
        )

    _sa.judge_single = _category_aware_judge

    spec = importlib.util.spec_from_file_location(
        "oracle_gold_eval", ROOT / "experiment" / "oracle_gold_eval.pyc")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    sfx, pfx = args.suffix, args.out_prefix
    arms = []
    if args.only in ("all", "locomo"):
        # 依 --photo 決定跑哪些;順序照使用者指定(no 先跑)
        if args.photo in ("both", "no"):
            arms.append(("LoCoMo(無照片敘述)", "locomo", False,
                         ROOT / f"experiment/{pfx}_locomo_nophoto{sfx}"))
        if args.photo in ("both", "yes"):
            arms.append(("LoCoMo(有照片敘述)", "locomo", True,
                         ROOT / f"experiment/{pfx}_locomo_photo{sfx}"))
    if args.only in ("all", "longmem"):
        # LongMem 證據來自 script_data CSV,沒有照片,_INCLUDE_PHOTO 不影響
        arms += [("LongMem", "longmem", True,
                  ROOT / f"experiment/{pfx}_longmem{sfx}")]

    for label, bench, photo, out in arms:
        print("\n" + "=" * 72, flush=True)
        print(f"### {label} | 答題={args.answer_model} @ {args.answer_api}", flush=True)
        print(f"###   judge={args.judge_model} @ {args.judge_api} | workers={args.workers}", flush=True)
        print(f"### _INCLUDE_PHOTO={photo}  ->  {out}", flush=True)
        print("=" * 72, flush=True)
        oracle._INCLUDE_PHOTO = photo
        out.mkdir(parents=True, exist_ok=True)
        runner = oracle.run_locomo if bench == "locomo" else oracle.run_longmem
        runner(args.limit, args.workers, out, 0)


if __name__ == "__main__":
    main()
