"""把 grep-agent 的 _grep_agent_traces.jsonl 編成一份 self-contained HTML viewer。

replay_run.py 每題把 trace append 到 output/<run-tag>/<cat>/_grep_agent_traces.jsonl
(含 sample/category/question/gold/answer/commands/kept/dropped/fallback/agent_ms…)。
本腳本補上兩類「推導」欄位,再把資料內嵌進 template.html:

  * correctness   — 從 output/<run-tag>/<cat>/<sample>.csv 的 correctness 欄 join
                    (replay --no-judge 時為空 → null,viewer 顯示「未評分」)
  * gold_sids / seed_recall / final_recall
                    — 用 script_data 的 has_answer=True turn 當 gold,套 corpus 的
                      sid 對應(user t → {sess}:{t+1}:u,assistant t → {sess}:{t}:a)。
                      seed_recall 對 seed_sids;final_recall 對 context_sids
                      (fallback 時 context 不變 → 等於 seed_recall)。

用法:
    python -m experiment.agent_filter.trace_viewer.build --run-tag rr2-grep
    # 產出 output/rr2-grep/trace_viewer.html(雙擊即開)
    #   + output/rr2-grep/agent_traces.enriched.jsonl(補完欄位的原始資料)
    # 之後有 judge 分數,重跑一次即可把 correctness 帶進來。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"
DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"
TEMPLATE = Path(__file__).with_name("template.html")

CATEGORIES = [
    "single_session_user", "single_session_assistant", "multi_session",
    "single_session_preference", "temporal_reasoning", "knowledge_update",
]

_gold_cache: dict[tuple[str, str], set[str]] = {}
_corr_cache: dict[tuple[str, str], object] = {}
_corpus_cache: dict[tuple[str, str], object] = {}


def _session_of(sid: str) -> str:
    return sid.split(":", 1)[0]


def _corpus(data_root: Path, cat: str, stem: str):
    key = (cat, stem)
    if key not in _corpus_cache:
        from experiment.agent_filter.corpus import load_corpus
        p = data_root / cat / f"{stem}.csv"
        try:
            _corpus_cache[key] = load_corpus(p) if p.exists() else None
        except Exception:  # noqa: BLE001
            _corpus_cache[key] = None
    return _corpus_cache[key]


def sid_texts(data_root: Path, cat: str, stem: str, sids: set[str]) -> dict[str, str]:
    """relevant sid → raw turn 文字(前端點 SID 反查用),截 300 字。"""
    corp = _corpus(data_root, cat, stem)
    if corp is None:
        return {}
    out: dict[str, str] = {}
    for s in sids:
        try:
            turns = corp.resolve(s)
        except Exception:  # noqa: BLE001
            turns = []
        if turns:
            txt = " ".join(str(t.text) for t in turns).strip()
            out[s] = (txt[:300] + "…") if len(txt) > 300 else txt
    return out


def gold_sids(data_root: Path, cat: str, stem: str) -> set[str]:
    key = (cat, stem)
    if key in _gold_cache:
        return _gold_cache[key]
    p = data_root / cat / f"{stem}.csv"
    out: set[str] = set()
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.lstrip("﻿") for c in df.columns]
        if {"has_answer", "role", "session_id", "turn_index"} <= set(df.columns):
            for _, r in df.iterrows():
                if str(r.get("has_answer")).strip().lower() not in ("true", "1", "yes"):
                    continue
                role = str(r["role"]).strip().lower()
                sess = str(r["session_id"]).strip()
                try:
                    t = int(r["turn_index"])
                except (TypeError, ValueError):
                    continue
                pair = t + 1 if role == "user" else t
                out.add(f"{sess}:{pair}:{'u' if role == 'user' else 'a'}")
    _gold_cache[key] = out
    return out


# judge 分數可能落在不同欄位(rejudge_output_dirs.py 預設寫 correctness_new;
# 也支援 4o-mini / 自訂欄位)。依序取第一個有值的。
_CORR_COLS = ("correctness", "correctness_new", "correctness_4o",
              "correctness_4omini", "correctness_v2", "correctness_normalized")
_JUDGE_COLS = {
    "4o-mini": ("correctness_new", "correctness_4o", "correctness_4omini"),
    "oss-20b": ("correctness_20b",),
}


def _coerce_corr(c):
    if c is None or (isinstance(c, float) and pd.isna(c)) or str(c).strip() == "":
        return None
    s = str(c).strip().lower()
    if s in ("1", "1.0", "yes", "true", "correct"):
        return 1.0
    if s in ("0", "0.0", "no", "false", "incorrect"):
        return 0.0
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def correctness(out_root: Path, run_tag: str, cat: str, stem: str):
    """從 agent run 的答題輸出 CSV 取 judge 分數(未 judge → None → 前端顯示未評分)。"""
    key = (cat, stem)
    if key in _corr_cache:
        return _corr_cache[key]
    p = out_root / run_tag / cat / f"{stem}.csv"
    val = None
    if p.exists():
        try:
            df = pd.read_csv(p)
            if len(df):
                for col in _CORR_COLS:
                    if col in df.columns:
                        v = _coerce_corr(df.iloc[0][col])
                        if v is not None:
                            val = v
                            break
        except Exception:  # noqa: BLE001
            val = None
    _corr_cache[key] = val
    return val


def judge_scores(out_root: Path, run_tag: str, cat: str, stem: str) -> dict[str, float]:
    """Read all explicitly named judge columns so the viewer shows judge provenance."""
    p = out_root / run_tag / cat / f"{stem}.csv"
    scores: dict[str, float] = {}
    if not p.exists():
        return scores
    try:
        df = pd.read_csv(p)
        if not len(df):
            return scores
        row = df.iloc[0]
        for model, columns in _JUDGE_COLS.items():
            for col in columns:
                if col in df.columns:
                    value = _coerce_corr(row[col])
                    if value is not None:
                        scores[model] = value
                        break
    except Exception:  # noqa: BLE001
        return {}
    return scores


def enrich(rec: dict, out_root: Path, data_root: Path, run_tag: str) -> dict:
    cat = rec.get("category") or ""
    stem = rec.get("sample") or ""
    g = gold_sids(data_root, cat, stem)
    rec["gold_sids"] = sorted(g)
    seed = set(rec.get("seed_sids") or [])
    ctx = set(rec.get("context_sids") or rec.get("final_sids") or []) or seed
    rec["seed_recall"] = round(len(seed & g) / len(g), 3) if g else None
    rec["final_recall"] = round(len(ctx & g) / len(g), 3) if g else None
    # session 級 recall(分母寬窄問題:strict gold turn recall 可能很低但答案仍對)
    gsess = {_session_of(s) for s in g}
    csess = {_session_of(s) for s in ctx}
    rec["gold_sessions"] = sorted(gsess)
    rec["session_recall"] = round(len(gsess & csess) / len(gsess), 3) if gsess else None
    if rec.get("correctness") in (None, ""):
        rec["correctness"] = correctness(out_root, run_tag, cat, stem)
    scores = judge_scores(out_root, run_tag, cat, stem)
    if scores:
        rec["judge_scores"] = scores
        rec["judge_model"] = "4o-mini" if "4o-mini" in scores else next(iter(scores))
    # relevant sid → raw turn 文字(點 SID 反查)
    relevant = seed | ctx | g | set(rec.get("added") or []) | set(rec.get("dropped") or [])
    rec["sid_text"] = sid_texts(data_root, cat, stem, relevant)
    return rec


def collect(out_root: Path, data_root: Path, run_tag: str) -> list[dict]:
    run_dir = out_root / run_tag
    rows: list[dict] = []
    for cat in CATEGORIES:
        tp = run_dir / cat / "_grep_agent_traces.jsonl"
        if not tp.exists():
            continue
        for line in tp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec.setdefault("category", cat)
            rows.append(enrich(rec, out_root, data_root, run_tag))
    return rows


def build_html(rows: list[dict], run_name: str) -> str:
    # </ → <\/ 讓 result/reply 內若含 </script> 也不會提前結束 script tag(JSON 仍合法)
    ndjson = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("__NDJSON_DATA__", ndjson).replace("__RUN_NAME__", run_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tag", required=True, help="agent run 的 output 目錄名(含 _grep_agent_traces.jsonl)")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT), help="run 目錄的根(預設 experiment/longmem/output)")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="script_data 根(算 gold recall 用)")
    ap.add_argument("--out", default=None, help="輸出 HTML 路徑(預設 <run>/trace_viewer.html)")
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    rows = collect(out_root, data_root, args.run_tag)
    if not rows:
        raise SystemExit(f"找不到任何 trace:{out_root / args.run_tag}/*/_grep_agent_traces.jsonl")

    run_dir = out_root / args.run_tag
    html_path = Path(args.out) if args.out else run_dir / "trace_viewer.html"
    jsonl_path = run_dir / "agent_traces.enriched.jsonl"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(rows, args.run_tag), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    scored = [r for r in rows if r.get("correctness") is not None]
    fb = sum(1 for r in rows if r.get("fallback"))
    print(f"{len(rows)} traces → {html_path}")
    print(f"  fallback={fb}  scored={len(scored)}  (enriched jsonl → {jsonl_path.name})")


if __name__ == "__main__":
    main()
