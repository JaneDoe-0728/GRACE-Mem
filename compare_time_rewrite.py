"""
compare_time_rewrite.py — 比較兩個 LoCoMo run（baseline vs time-rewrite），
量化 time-rewrite 的影響到底落在「檢索(gold recall)」還是「答題」。

輸入：兩個 run 的路徑（standard/ 下的資料夾名，或絕對路徑）。
    argv1 = baseline run（無 time-rewrite）
    argv2 = tr run      （有 time-rewrite）

用法：
    python compare_time_rewrite.py <baseline_run> <tr_run> \
        [--chunk-turns auto|0|N] [--out-json PATH] [--n-examples 10] \
        [--max-context-chars 0]

產出：
    證據1  兩個 run 的整體 gold-recall 四項指標 + Δ
    證據2  退步/進步題分析 + jaccard + 一份 JSON（題目、rewrite 前後檢索文字、
           baseline 答 / TR 答 / 正解、檢索 sid、jaccard、gold recall）
    證據3  同四項指標的 per-category 版本 + Δ
    證據4  N 個「同證據但答案翻掉」的具體案例（預設 10 題）

重用 locomo_gold_recall_metrics.py 的 gold-sid / retrieved-sid 計算，確保與既有
gold recall 口徑一致（chunk 粒度、sample-scoped）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import locomo_gold_recall_metrics as glm  # 重用既有 helper（import 不會觸發其 main()）

_ROOT = Path(__file__).resolve().parent
_RUN_ROOT = _ROOT / "experiment" / "locomo" / "output" / "standard"


# --------------------------------------------------------------------------- #
# jaccard                                                                      #
# --------------------------------------------------------------------------- #
JACCARD_DOC = (
    "Jaccard(A,B) = |A∩B| / |A∪B|，A/B 是兩個 run 對同一題檢索到的 summary 片段 "
    "(sid) 集合。1.0 = 兩邊檢索到完全相同的證據；越低代表檢索內容差異越大。"
    "用途：判斷『翻題(答案對錯改變)』的來源——高 jaccard + 翻題 => 拿到同樣證據卻答不同"
    "(答題端造成)；低 jaccard => 檢索內容變了(檢索端造成)。"
)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


# --------------------------------------------------------------------------- #
# 逐題評分：回傳 {(sample_idx, question): record}                              #
# --------------------------------------------------------------------------- #
def score_run(run_dir: Path, turn_index, q_evidence, n: int) -> dict:
    df = glm._load_judge_df(run_dir)
    recs: dict = {}
    for _, row in df.iterrows():
        sample_idx = glm._sample_index(row.get("sample"))
        if sample_idx is None:
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        try:
            correct = float(row.get("correctness")) == 1.0
        except (TypeError, ValueError):
            correct = False

        units = q_evidence.get(sample_idx, {}).get(question)
        if units is not None:
            gold = glm._gold_sids(units, sample_idx, n, turn_index.get(sample_idx, {}))
        else:  # 沒對到 json → 退回 CSV gold_evidence_source（僅 session 粒度）
            gold = glm._gold_from_csv_field(row.get("gold_evidence_source"), sample_idx)

        retrieved = glm._retrieved_sids(
            row.get("selected_evidence_ids"), row.get("retrieved_context"), sample_idx, n
        )
        recs[(sample_idx, question)] = dict(
            sample_idx=sample_idx,
            question=question,
            category=str(row.get("category_label") or row.get("category") or "?").strip(),
            correct=correct,
            gold=gold,
            retrieved=retrieved,
            hit=gold & retrieved,
            gold_answer=("" if row.get("gold_answer") is None else str(row.get("gold_answer"))),
            model_answer=("" if row.get("model_answer") is None else str(row.get("model_answer"))),
            retrieved_context=str(row.get("retrieved_context") or ""),
        )
    return recs


# --------------------------------------------------------------------------- #
# 聚合四項指標（整體 + per-category）                                          #
# --------------------------------------------------------------------------- #
def _blank() -> dict:
    return dict(n_q=0, n_correct=0, gt=0, gh=0, qg=0, ah=0, ahc=0)


def aggregate(recs: dict):
    overall = _blank()
    per_cat: dict = {}
    for r in recs.values():
        d = per_cat.setdefault(r["category"], _blank())
        for tgt in (overall, d):
            tgt["n_q"] += 1
            if r["correct"]:
                tgt["n_correct"] += 1
        if not r["gold"]:
            continue
        for tgt in (overall, d):
            tgt["gt"] += len(r["gold"])
            tgt["gh"] += len(r["hit"])
            tgt["qg"] += 1
            if r["hit"] == r["gold"]:
                tgt["ah"] += 1
                if r["correct"]:
                    tgt["ahc"] += 1
    return overall, per_cat


def _pct(a: int, b: int) -> float:
    return 100 * a / b if b else 0.0


def _fmt(a: int, b: int) -> str:
    return f"{a}/{b} = {_pct(a, b):5.1f}%" if b else f"{a}/{b} = n/a"


METRIC_ROWS = [
    ("整體正確率        ", "n_correct", "n_q"),
    ("Gold返回率(檢索)  ", "gh", "gt"),
    ("整題gold全中率    ", "ah", "qg"),
    ("gold全中的正確率  ", "ahc", "ah"),
]


def print_metric_block(title: str, base: dict, tr: dict) -> None:
    print(f"\n{title}")
    print(f"  {'指標':18s} {'baseline':>16s} {'time-rewrite':>16s} {'Δ(pp)':>8s}")
    for label, num, den in METRIC_ROWS:
        pb, pt = _pct(base[num], base[den]), _pct(tr[num], tr[den])
        print(f"  {label} {_fmt(base[num], base[den]):>16s} {_fmt(tr[num], tr[den]):>16s} {pt - pb:+8.1f}")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def _resolve(run: str) -> Path:
    p = Path(run)
    if p.exists():
        return p
    cand = _RUN_ROOT / run
    if cand.exists():
        return cand
    sys.exit(f"run 不存在：{run}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs time-rewrite LoCoMo runs")
    ap.add_argument("baseline_run", help="無 time-rewrite 的 run（名稱或絕對路徑）")
    ap.add_argument("tr_run", help="有 time-rewrite 的 run（名稱或絕對路徑）")
    ap.add_argument("--chunk-turns", default="auto", help="0=session, N>0=chunk, auto=推斷(預設)")
    ap.add_argument("--out-json", default=None, help="證據2 JSON 輸出路徑（預設 tr_run/_time_rewrite_flips.json）")
    ap.add_argument("--n-examples", type=int, default=10, help="證據4 印出的具體案例數（預設 10）")
    ap.add_argument("--max-context-chars", type=int, default=0, help="JSON 內檢索文字截斷長度，0=不截斷(預設)")
    args = ap.parse_args()

    base_dir, tr_dir = _resolve(args.baseline_run), _resolve(args.tr_run)

    # 資料集 → gold evidence + turn index
    data = json.loads(glm.DATASET_JSON.read_text(encoding="utf-8"))
    turn_index = glm._build_turn_index(data)
    q_evidence = glm._build_question_evidence(data)

    # chunk 粒度（用 tr run 推斷；兩個 run 都是同樣 N=8）
    if args.chunk_turns == "auto":
        n = glm._infer_chunk_turns(tr_dir, turn_index)
    else:
        n = int(args.chunk_turns)

    base_recs = score_run(base_dir, turn_index, q_evidence, n)
    tr_recs = score_run(tr_dir, turn_index, q_evidence, n)

    base_ov, base_pc = aggregate(base_recs)
    tr_ov, tr_pc = aggregate(tr_recs)

    print("=" * 74)
    print(f"baseline = {base_dir.name}   |   time-rewrite = {tr_dir.name}   |   granularity = "
          f"{'chunk N=' + str(n) if n > 0 else 'session'}")
    print("=" * 74)

    # ---------------- 證據 1：整體 gold recall ---------------- #
    print("\n########## 證據1：整體 gold-recall 指標（檢索是否變少）##########")
    print_metric_block("[Overall]", base_ov, tr_ov)
    print("\n  解讀：若『Gold返回率』Δ≈0 → time-rewrite 沒有影響檢索/給 LLM 的 gold recall。")

    # ---------------- 證據 3：per-category ---------------- #
    print("\n########## 證據3：per-category 指標（拆解檢索 vs 答題）##########")
    for cat in sorted(set(base_pc) | set(tr_pc)):
        print_metric_block(f"[{cat}]", base_pc.get(cat, _blank()), tr_pc.get(cat, _blank()))
    print("\n  解讀：每類的『Gold返回率』持平但『gold全中的正確率』有升有降 → 差異在答題端，不在檢索端。")

    # ---------------- 證據 2：翻題 + jaccard + JSON ---------------- #
    common = set(base_recs) & set(tr_recs)
    regressions, gains = [], []
    for k in common:
        rb, rt = base_recs[k], tr_recs[k]
        if rb["correct"] and not rt["correct"]:
            flip = "regression"
        elif not rb["correct"] and rt["correct"]:
            flip = "gain"
        else:
            continue
        j = jaccard(rb["retrieved"], rt["retrieved"])

        def _ctx(s: str) -> str:
            return s if args.max_context_chars <= 0 else s[: args.max_context_chars]

        entry = dict(
            flip_type=flip,
            sample=k[0],
            question=k[1],
            category=rb["category"],
            gold_answer=rb["gold_answer"],
            baseline_answer=rb["model_answer"],
            tr_answer=rt["model_answer"],
            baseline_retrieved_context=_ctx(rb["retrieved_context"]),
            tr_retrieved_context=_ctx(rt["retrieved_context"]),
            baseline_retrieved_sids=sorted(rb["retrieved"]),
            tr_retrieved_sids=sorted(rt["retrieved"]),
            gold_sids=sorted(rb["gold"]),
            retrieved_sid_jaccard=round(j, 4),
            baseline_gold_recall=f"{len(rb['hit'])}/{len(rb['gold'])}",
            tr_gold_recall=f"{len(rt['hit'])}/{len(rt['gold'])}",
        )
        (regressions if flip == "regression" else gains).append(entry)

    def _mean_j(items):
        return round(sum(e["retrieved_sid_jaccard"] for e in items) / len(items), 4) if items else None

    # net by category
    net_by_cat: dict = {}
    for e in regressions:
        net_by_cat[e["category"]] = net_by_cat.get(e["category"], 0) - 1
    for e in gains:
        net_by_cat[e["category"]] = net_by_cat.get(e["category"], 0) + 1

    print("\n########## 證據2：翻題分析（退步/進步）+ jaccard ##########")
    print(f"  退步(baseline對→TR錯) = {len(regressions)}   "
          f"進步(baseline錯→TR對) = {len(gains)}   淨 = {len(gains) - len(regressions):+d}")
    print(f"  退步題平均 jaccard = {_mean_j(regressions)}   進步題平均 jaccard = {_mean_j(gains)}")
    print(f"  net by category: {dict(sorted(net_by_cat.items(), key=lambda x: x[1]))}")
    print(f"  jaccard 說明：{JACCARD_DOC}")

    out_json = Path(args.out_json) if args.out_json else (tr_dir / "_time_rewrite_flips.json")
    payload = dict(
        _meta=dict(
            baseline_run=base_dir.name,
            tr_run=tr_dir.name,
            chunk_turns=n,
            jaccard_explanation=JACCARD_DOC,
            field_notes={
                "baseline_retrieved_context": "rewrite 前(baseline)給 LLM 的檢索文字",
                "tr_retrieved_context": "rewrite 後(time-rewrite)給 LLM 的檢索文字",
                "*_retrieved_sids": "各自檢索到的 summary 片段 id（chunk 粒度）",
                "*_gold_recall": "gold 命中數/總 gold 數（該題）",
            },
            counts=dict(regressions=len(regressions), gains=len(gains), net=len(gains) - len(regressions)),
            jaccard_mean=dict(regressions=_mean_j(regressions), gains=_mean_j(gains)),
            net_by_category=net_by_cat,
        ),
        regressions=sorted(regressions, key=lambda e: -e["retrieved_sid_jaccard"]),
        gains=sorted(gains, key=lambda e: -e["retrieved_sid_jaccard"]),
    )
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✅ 證據2 JSON 已寫出 → {out_json}")
    print("     (regressions/gains 依 jaccard 由高到低排序；高 jaccard=同證據不同答案)")

    # ---------------- 證據 4：具體案例（同證據、答案翻掉）---------------- #
    print(f"\n########## 證據4：{args.n_examples} 個『同證據但答案翻掉』的退步案例 ##########")
    print("(依 jaccard 由高到低取樣：jaccard 越高代表兩 run 檢索證據越一致，翻題純屬答題差異)")
    top = sorted(regressions, key=lambda e: -e["retrieved_sid_jaccard"])[: args.n_examples]
    for i, e in enumerate(top, 1):
        print(f"\n--- 案例 {i} [{e['category']}]  jaccard={e['retrieved_sid_jaccard']}  "
              f"gold_recall baseline={e['baseline_gold_recall']} / TR={e['tr_gold_recall']} ---")
        print(f"  Q   : {e['question']}")
        print(f"  正解: {e['gold_answer'][:120]}")
        print(f"  base(對): {e['baseline_answer'][:120]}")
        print(f"  TR (錯): {e['tr_answer'][:120]}")


if __name__ == "__main__":
    main()
