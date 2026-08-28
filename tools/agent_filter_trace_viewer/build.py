"""Compile the grep agent's _grep_agent_traces.jsonl into a self-contained HTML
viewer.

For each question replay_run.py appends a trace to
output/<run-tag>/<cat>/_grep_agent_traces.jsonl, carrying sample, category,
question, gold, answer, commands, kept, dropped, fallback, agent_ms and so on.
This script adds two kinds of *derived* field, then embeds the data into
template.html:

  * correctness   -- joined from the correctness column of
                     output/<run-tag>/<cat>/<sample>.csv (empty under
                     replay --no-judge -> null, which the viewer shows as
                     "unscored")
  * gold_sids / seed_recall / final_recall
                  -- taking script_data's has_answer=True turns as gold and
                     applying the corpus sid mapping (user t -> {sess}:{t+1}:u,
                     assistant t -> {sess}:{t}:a).
                     seed_recall is measured against seed_sids, final_recall
                     against context_sids (on a fallback the context is unchanged,
                     so the two are equal).

Usage:
    python -m tools.agent_filter_trace_viewer.build --run-tag rr2-grep
    # produces output/rr2-grep/trace_viewer.html (just double-click it)
    #   + output/rr2-grep/agent_traces.enriched.jsonl (the raw data with the
    #     derived fields filled in)
    # Once judge scores exist, rerunning this pulls correctness in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"
DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"
TEMPLATE = Path(__file__).with_name("template.html")

QUESTION_CATEGORIES = [
    "single_session_user", "single_session_assistant", "multi_session",
    "single_session_preference", "temporal_reasoning", "knowledge_update",
]

_gold_cache: dict[tuple[str, str], set[str]] = {}
_corr_cache: dict[tuple[str, str], object] = {}
_corpus_cache: dict[tuple[str, str], object] = {}


def _session_of(sid: str) -> str:
    return sid.split(":", 1)[0]


def _corpus(data_root: Path, cat: str, stem: str):
    """Load and memoize one corpus, caching None when it is unavailable.

    A missing or unparseable corpus is cached as None rather than retried:
    the viewer resolves sids for many rows over the same corpus, and without
    the negative cache each one would re-attempt the same failing read.
    """
    key = (cat, stem)
    if key not in _corpus_cache:
        from experiment.agent_filter.corpus import load_corpus
        p = data_root / cat / f"{stem}.csv"
        try:
            _corpus_cache[key] = load_corpus(p) if p.exists() else None
        except Exception:
            _corpus_cache[key] = None
    return _corpus_cache[key]


def sid_texts(data_root: Path, cat: str, stem: str, sids: set[str]) -> dict[str, str]:
    """relevant sid -> raw turn text, truncated to 300 characters, for the front
    end's click-a-SID lookup."""
    corp = _corpus(data_root, cat, stem)
    if corp is None:
        return {}
    out: dict[str, str] = {}
    for s in sids:
        try:
            turns = corp.resolve(s)
        except Exception:
            turns = []
        if turns:
            txt = " ".join(str(t.text) for t in turns).strip()
            out[s] = (txt[:300] + "…") if len(txt) > 300 else txt
    return out


def gold_sids(data_root: Path, cat: str, stem: str) -> set[str]:
    """Read the gold-evidence sids for one question, as the viewer's ground truth.

    The CSV marks gold at turn granularity, but sids address speaker-turn
    pairs, so a user turn is mapped to the following index and an assistant
    turn to its own. Getting that off by one silently marks the wrong turns
    gold and makes every trace look worse than it is.

    A BOM is stripped from the column names: these CSVs are frequently opened
    in Excel, which prepends one and would break the column check below.

    Returns:
        Sids as "session:pair:role", or an empty set if the file is absent or
        lacks the required columns.
    """
    key = (cat, stem)
    if key in _gold_cache:
        return _gold_cache[key]
    p = data_root / cat / f"{stem}.csv"
    out: set[str] = set()
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
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


# The judge score may land in any of several columns (rejudge_output_dirs.py
# writes correctness_new by default, and 4o-mini or custom columns are supported
# too). Take the first one that has a value, in order.
_CORR_COLS = ("correctness", "correctness_new", "correctness_4o",
              "correctness_4omini", "correctness_v2", "correctness_normalized")
_JUDGE_COLS = {
    "4o-mini": ("correctness_new", "correctness_4o", "correctness_4omini"),
    "oss-20b": ("correctness_20b",),
}


def _coerce_corr(c):
    """Parse a judge correctness cell into 1.0, 0.0, or None.

    The column has been written by several judges over time in whatever
    spelling each used -- "1", "yes", "true", "correct" -- so the accepted
    forms are enumerated rather than guessed at.

    Returns None for blank or unrecognised values, which keeps an unjudged row
    out of the averages instead of counting it as incorrect.
    """
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
    """Read the judge score from the agent run's answer output CSV (not yet judged
    -> None -> the front end shows it as unscored)."""
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
        except Exception:
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
    except Exception:
        return {}
    return scores


def enrich(rec: dict, out_root: Path, data_root: Path, run_tag: str) -> dict:
    """Attach recall metrics, judge scores, and sid text to one trace record.

    Three recalls are computed because they answer different questions and
    routinely disagree. Seed recall says what retrieval handed the agent;
    final recall says what survived it -- the difference is the agent's actual
    contribution. Session recall widens the denominator to whole sessions,
    which matters because strict turn recall can look poor while the answer is
    right: the gold turn was missed but a neighbouring turn in the same session
    carried the fact.

    Mutates and returns `rec`.
    """
    cat = rec.get("category") or ""
    stem = rec.get("sample") or ""
    g = gold_sids(data_root, cat, stem)
    rec["gold_sids"] = sorted(g)
    seed = set(rec.get("seed_sids") or [])
    ctx = set(rec.get("context_sids") or rec.get("final_sids") or []) or seed
    rec["seed_recall"] = round(len(seed & g) / len(g), 3) if g else None
    rec["final_recall"] = round(len(ctx & g) / len(g), 3) if g else None
    # Session-level recall (a question of how wide the denominator is: strict gold
    # turn recall can be very low while the answer is still right)
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
    # relevant sid -> raw turn text (for the click-a-SID lookup)
    relevant = seed | ctx | g | set(rec.get("added") or []) | set(rec.get("dropped") or [])
    rec["sid_text"] = sid_texts(data_root, cat, stem, relevant)
    return rec


def collect(out_root: Path, data_root: Path, run_tag: str) -> list[dict]:
    """Load and enrich every agent trace across all categories of one run.

    Unparseable lines are skipped: traces are appended live during a run, so
    the last line is frequently a partial write and failing on it would make
    the viewer unusable mid-run. Categories with no trace file are simply
    absent.
    """
    run_dir = out_root / run_tag
    rows: list[dict] = []
    for cat in QUESTION_CATEGORIES:
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
    """Inline the traces into the HTML template as NDJSON.

    Self-contained output by design -- the result is a single file that opens
    from disk with no server and no sidecar data file to keep alongside it.
    """
    # </ becomes <\/ so that a </script> inside result/reply cannot end the script
    # tag early. The JSON stays valid either way.
    ndjson = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("__NDJSON_DATA__", ndjson).replace("__RUN_NAME__", run_name)


def main() -> None:
    """Build the standalone trace-viewer HTML for one run."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tag", required=True, help="name of the agent run's output directory, the one holding _grep_agent_traces.jsonl")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT), help="root holding the run directories (default experiment/longmem/output)")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="script_data root, used to compute gold recall")
    ap.add_argument("--out", default=None, help="output HTML path (default <run>/trace_viewer.html)")
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    rows = collect(out_root, data_root, args.run_tag)
    if not rows:
        raise SystemExit(f"no traces found: {out_root / args.run_tag}/*/_grep_agent_traces.jsonl")

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
