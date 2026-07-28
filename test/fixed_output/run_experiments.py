"""
Fixed-Output Experiment Runner

Run one or more EXP-F01 … EXP-F08 experiments.

Usage:
    python test/fixed_output/run_experiments.py [EXP_IDs] [options]

EXP_IDs:
    Comma-separated experiment numbers, e.g.:  1,3,5
    Ranges supported:                          1-4
    Keyword "all":                             all
    Default (no arg):                          all

Options:
    --tag TAG     Run-tag written into the report path  (default: timestamp)
    --stop-on-fail
                  Abort the sequence after the first FAIL
    --dry-run     Print selected experiments and exit

Examples:
    python test/fixed_output/run_experiments.py all
    python test/fixed_output/run_experiments.py 1,3
    python test/fixed_output/run_experiments.py 1-4 --tag baseline
    python test/fixed_output/run_experiments.py 5,6,7 --stop-on-fail
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]

_EXPERIMENTS = {
    #  num: (exp_id,   script,                          description,                   needs_llm)
    1: ("EXP-F01", SCRIPT_DIR / "f01_rng.py",         "RNG / DataLoader determinism",  False),
    2: ("EXP-F02", SCRIPT_DIR / "f02_llm.py",         "LLM API determinism",           True),
    3: ("EXP-F03", SCRIPT_DIR / "f03_embedder.py",    "Embedder determinism",          False),
    4: ("EXP-F04", SCRIPT_DIR / "f04_reranker.py",    "Reranker determinism",          False),
    5: ("EXP-F05", SCRIPT_DIR / "f05_ingest.py",      "Ingest reproducibility",        True),
    6: ("EXP-F06", SCRIPT_DIR / "f06_retrieval.py",   "Retrieval determinism",         False),
    7: ("EXP-F07", SCRIPT_DIR / "f07_e2e.py",         "End-to-end reproducibility",   True),
    8: ("EXP-F08", SCRIPT_DIR / "f08_concurrency.py", "Concurrency stress",            True),
}

_STATUS_COLOR = {
    "PASS": "\033[32m",   # green
    "WARN": "\033[33m",   # yellow
    "FAIL": "\033[31m",   # red
    "SKIP": "\033[36m",   # cyan
    "ERR":  "\033[35m",   # magenta
}
_RESET = "\033[0m"
_STATUS_RE = re.compile(r'"status"\s*:\s*"(PASS|WARN|FAIL|SKIP)"')


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_ids(raw: str) -> List[int]:
    """Parse '1,3,5-7' into [1, 3, 5, 6, 7]."""
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    # deduplicate, preserve order
    seen: set[int] = set()
    result: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments.py",
        description="Run Fixed-Output experiments EXP-F01 … EXP-F08.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "experiments",
        nargs="?",
        default="all",
        help="Experiment IDs: '1,3', '1-4', or 'all'  (default: all)",
    )
    p.add_argument("--tag",       default=None,                      help="Run-tag for report paths")
    p.add_argument("--llm-url",   default="http://localhost:1234/v1", dest="llm_url",   help="LM Studio base URL (F02/F07/F08)")
    p.add_argument("--llm-model", default="openai/gpt-oss-20b",               dest="llm_model", help="LLM model name (F02/F07/F08)")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Abort after first FAIL result")
    p.add_argument("--dry-run", action="store_true",
                   help="Print selected experiments and exit")
    return p


# ── Experiment runner ─────────────────────────────────────────────────────────

def _run_one(
    exp_id: str,
    script: Path,
    tag: Optional[str],
    *,
    needs_llm: bool = False,
    llm_url: str = "",
    llm_model: str = "",
) -> str:
    """Run one experiment script; return status string."""
    cmd = [sys.executable, str(script)]
    if tag:
        cmd.extend(["--tag", tag])
    if needs_llm:
        cmd.extend(["--llm-url", llm_url, "--llm-model", llm_model])

    print(f"\n{'='*60}")
    print(f"  {exp_id}  →  {script.name}" + (f"  [tag={tag}]" if tag else ""))
    print(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)

    # Prefer the experiment's own reported status when available.
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    matches = _STATUS_RE.findall(output)
    parsed_status = matches[-1] if matches else None

    if parsed_status is not None:
        status = parsed_status
    elif result.returncode == 0:
        status = "PASS"
    elif result.returncode == 1:
        status = "FAIL"
    else:
        status = "ERR"

    color  = _STATUS_COLOR.get(status, "")
    timing = f"  ({elapsed:.1f}s)"
    print(f"\n{color}[{exp_id}] {status}{_RESET}{timing}")
    return status


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_argparser()
    args   = parser.parse_args()

    # Resolve experiment IDs
    if args.experiments.lower() == "all":
        selected = list(_EXPERIMENTS.keys())
    else:
        try:
            selected = _parse_ids(args.experiments)
        except ValueError as exc:
            print(f"ERROR: invalid experiment IDs — {exc}", file=sys.stderr)
            return 2

    unknown = [i for i in selected if i not in _EXPERIMENTS]
    if unknown:
        print(f"ERROR: unknown experiment numbers: {unknown}", file=sys.stderr)
        print(f"Valid numbers: {sorted(_EXPERIMENTS.keys())}", file=sys.stderr)
        return 2

    # Print plan
    print(f"\nFixed-Output Experiment Runner")
    print(f"{'─'*40}")
    for num in selected:
        exp_id, script, desc, needs_llm = _EXPERIMENTS[num]
        llm_marker = "  [LLM]" if needs_llm else ""
        print(f"  {num:2d}.  {exp_id}  —  {desc}{llm_marker}")
    print(f"{'─'*40}")
    if args.tag:
        print(f"  run-tag   : {args.tag}")
    if args.llm_url:
        print(f"  llm-url   : {args.llm_url}")
    if args.llm_model:
        print(f"  llm-model : {args.llm_model}")
    if args.stop_on_fail:
        print(f"  mode      : stop-on-fail")
    print()

    if args.dry_run:
        print("[dry-run] exiting without running.")
        return 0

    # Execute
    summary: List[tuple[str, str]] = []
    overall_ok = True

    for num in selected:
        exp_id, script, _, needs_llm = _EXPERIMENTS[num]
        status = _run_one(
            exp_id, script, args.tag,
            needs_llm=needs_llm, llm_url=args.llm_url, llm_model=args.llm_model,
        )
        summary.append((exp_id, status))

        if status in ("FAIL", "ERR"):
            overall_ok = False
            if args.stop_on_fail:
                print(f"\n[runner] stop-on-fail triggered by {exp_id}.", file=sys.stderr)
                break

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'─'*60}")
    for exp_id, status in summary:
        color = _STATUS_COLOR.get(status, "")
        print(f"  {exp_id:<10} {color}{status}{_RESET}")
    ran_ids  = {s[0] for s in summary}
    skipped  = [_EXPERIMENTS[n][0] for n in selected if _EXPERIMENTS[n][0] not in ran_ids]
    for exp_id in skipped:
        print(f"  {exp_id:<10} (not reached)")
    print(f"{'='*60}\n")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
