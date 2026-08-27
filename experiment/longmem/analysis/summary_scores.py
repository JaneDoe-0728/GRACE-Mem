"""Score distribution of retrieved summaries vs gold summaries (split-embed run).

Parses every [sid=...][score=...] entry in the Evidence Summary block of each
question's Retrieved_Context, labels each as gold / non-gold (side-level), and
reports the score distribution for:
    - ALL retrieved summaries
    - GOLD summaries that were retrieved

Usage:
    python -m experiment.longmem.analysis.summary_scores --run split-embed
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

DATA_ROOT = _ROOT / "experiment" / "longmem" / "script_data"
OUTPUT_ROOT = _ROOT / "experiment" / "longmem" / "output"
CATEGORIES = [
    "single_session_user", "single_session_assistant", "multi_session",
    "single_session_preference", "temporal_reasoning", "knowledge_update",
]
_ENTRY_RE = re.compile(r"\[sid=([^\]]+)\]\[score=([-0-9.eE]+)\]")
_HEADER = "### Evidence Summary"


def _is_main(p: Path) -> bool:
    s = p.stem
    bad = ("_abs", "_replay_fact", "_replay_fact_user_only", "_gold_summary")
    return not any(s.endswith(x) for x in bad) and s != "all_answers"


def _gold_sids(src: Path) -> set[str]:
    """Read the gold-evidence sids for one question CSV."""
    df = pd.read_csv(src)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    if "has_answer" not in df.columns:
        return set()
    out = set()
    for _, r in df[df["has_answer"] == True].iterrows():
        t = int(r["turn_index"]); role = str(r["role"]).strip().lower()
        mid = t + 1 if role == "user" else t
        out.add(f"{str(r['session_id']).strip()}:{mid}:{'u' if role == 'user' else 'a'}")
    return out


def _evidence_entries(ctx) -> list[tuple[str, float]]:
    """Parse the retrieved evidence entries out of a stored context."""
    if not isinstance(ctx, str):
        return []
    i = ctx.find(_HEADER)
    block = ctx[i:] if i != -1 else ctx
    out = []
    for sid, sc in _ENTRY_RE.findall(block):
        try:
            out.append((sid.strip(), float(sc)))
        except ValueError:
            pass
    return out


def _stats(xs: list[float]) -> str:
    """Summarize a score distribution: count, mean, and quantiles.

    Quantiles rather than mean alone, because these distributions are routinely
    bimodal -- a handful of strong hits and a long tail of near-zero scores --
    and a mean over that describes neither group.
    """
    if not xs:
        return "n=0"
    xs = sorted(xs)
    n = len(xs)
    def q(p):
        return xs[min(n - 1, int(p * n))]
    mean = sum(xs) / n
    return (f"n={n}  mean={mean:.3f}  min={xs[0]:.3f}  p25={q(.25):.3f}  "
            f"median={q(.5):.3f}  p75={q(.75):.3f}  p90={q(.9):.3f}  max={xs[-1]:.3f}")


def _hist(xs: list[float], lo: float, hi: float, bins: int = 14, width: int = 40) -> str:
    """Render a score distribution as a text histogram, for terminal reading."""
    if not xs:
        return "  (empty)"
    step = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        b = int((x - lo) / step) if step else 0
        b = max(0, min(bins - 1, b))
        counts[b] += 1
    mx = max(counts) or 1
    lines = []
    for b in range(bins):
        a = lo + b * step
        bar = "█" * round(width * counts[b] / mx)
        lines.append(f"  [{a:5.2f},{a+step:5.2f})  {counts[b]:5d} | {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exclude-ku", action="store_true", help="exclude knowledge_update")
    args = ap.parse_args()
    run_root = OUTPUT_ROOT / args.run

    all_scores: list[float] = []
    gold_scores: list[float] = []
    nongold_scores: list[float] = []

    for cat in CATEGORIES:
        if args.exclude_ku and cat == "knowledge_update":
            continue
        cdir = run_root / cat
        if not cdir.exists():
            continue
        for p in sorted(cdir.glob("*.csv")):
            if not _is_main(p):
                continue
            df = pd.read_csv(p)
            if "Retrieved_Context" not in df.columns:
                continue
            entries = _evidence_entries(df.iloc[0]["Retrieved_Context"])
            src = DATA_ROOT / cat / f"{p.stem}.csv"
            gold = _gold_sids(src) if src.exists() else set()
            for sid, sc in entries:
                all_scores.append(sc)
                (gold_scores if sid in gold else nongold_scores).append(sc)

    lo = min(all_scores) if all_scores else 0.0
    hi = max(all_scores) if all_scores else 1.0

    print(f"\n=== {args.run}{' (ex-KU)' if args.exclude_ku else ''} — retrieved summary score distribution ===\n")
    print("ALL retrieved summaries :", _stats(all_scores))
    print("  GOLD (retrieved)      :", _stats(gold_scores))
    print("  NON-GOLD              :", _stats(nongold_scores))
    print(f"\n[ALL retrieved]  range [{lo:.2f}, {hi:.2f}]")
    print(_hist(all_scores, lo, hi))
    print("\n[GOLD retrieved]")
    print(_hist(gold_scores, lo, hi))
    print("\n[NON-GOLD]")
    print(_hist(nongold_scores, lo, hi))


if __name__ == "__main__":
    main()
