"""Argument parsing for the LoCoMo runner, and the worker command it builds.

Kept separate from the runner so the argument surface can be tested and
inspected without importing the pipeline, and because both the orchestrator and
each worker subprocess parse the same arguments -- `build_worker_command`
round-trips a parsed config back into the argv the child will re-parse.

That round-trip is the constraint to respect when adding a flag: an option the
orchestrator accepts but does not forward silently applies to the parent only,
and the sample it was meant to affect runs without it.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

MODULE_DIR = Path(__file__).resolve().parent
if __package__ in (None, ""):
    repo_root = MODULE_DIR.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiment.locomo.models import RunConfig, SamplePlan

PIPELINE_MODULE = "experiment.locomo.pipeline.runner"

try:
    from experiment.experiment_config import INGEST_PARAMS
except Exception:
    INGEST_PARAMS = {
        "prev_k": 2,
        "entity_sim_topk": 4,
        "entity_sim_threshold": 0.5,
    }


VALID_STAGES = ("ingest", "qa_eval", "judge")
DEFAULT_STAGES = VALID_STAGES
RETRIEVAL_ONLY_STAGES = tuple(stage for stage in VALID_STAGES if stage != "ingest")


def build_parser() -> argparse.ArgumentParser:
    """Define the LoCoMo runner's full argument surface.

    One parser serves both the orchestrator and the worker subprocess, since the
    child re-parses the argv `build_worker_command` produces. A flag added here
    but not forwarded there applies to the parent only.
    """
    parser = argparse.ArgumentParser(
        description="Conversational multisample ingest/eval/judge pipeline"
    )
    parser.add_argument("--dataset", choices=["locomo"], default="locomo")
    parser.add_argument("--sessions-jsonl", default=None)
    parser.add_argument("--dataset-json", default=None)
    parser.add_argument(
        "--source-json",
        default=None,
        help="Optional source conversation JSON used by snapshot tooling. "
        "Defaults to the standard LoCoMo dataset path.",
    )
    parser.add_argument("--prev-k", type=int, default=INGEST_PARAMS.get("prev_k", 2))
    parser.add_argument("--entity-sim-topk", type=int, default=INGEST_PARAMS.get("entity_sim_topk", 4))
    parser.add_argument("--entity-sim-threshold", type=float, default=INGEST_PARAMS.get("entity_sim_threshold", 0.5))
    parser.add_argument(
        "--chunk-turns",
        type=int,
        default=INGEST_PARAMS.get("chunk_turns", 8),
        help="Turns per ingest chunk. Each session is split into consecutive windows "
        "of this many turns, each becoming its own summary (message_id = chunk index). "
        "0 = one summary per whole session. Must match the run that produced any "
        "artifacts reused via --artifact-dir.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive re-search retrieval",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.70,
        help="Confidence threshold for adaptive re-search (default 0.70)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="(locomo only) Base run directory of a previous run whose artifacts "
        "should be used instead of re-ingesting. Per-sample artifacts are "
        "resolved as <artifact-dir>/sample_<id>/artifacts/. "
        "Each sample dir should contain ChromaDB artifacts and optionally "
        "graph_export.json for FalkorDB restore.",
    )
    parser.add_argument("--sample-ids", help="e.g. 0,2,5-7")
    parser.add_argument(
        "--stage",
        dest="stages",
        nargs="+",
        choices=VALID_STAGES,
        default=None,
        metavar="STAGE",
        help=(
            "Stages to run. Default: full pipeline "
            "(ingest qa_eval judge). "
            "Examples: --stage ingest qa_eval, --stage judge"
        ),
    )
    parser.add_argument("--out-root", default="experiment/locomo/output")
    parser.add_argument("--run-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--post-refresh-sleep", type=float, default=7.0)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip judge and correctness aggregation",
    )
    parser.add_argument(
        "--adv",
        action="store_true",
        help="Include adversarial questions; default runs skip them across eval, judge, and aggregate",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="",
        help="Retrieval mode override. 'gold_summary_only' skips KG retrieval and answers "
        "from gold session summaries only; 'gold_raw_text_only' answers from raw conversation "
        "turns only; 'replay_summary_raw_text_from_run' reuses selected summary ids from a prior "
        "run, swaps summary text for raw session text, and answers without re-running retrieval.",
    )
    parser.add_argument(
        "--replay-run-dir",
        default=None,
        help="Base run directory to replay retrieval results from when "
        "--retrieval-mode=replay_summary_raw_text_from_run. "
        "Expected layout: <run>/sample_<id>/logs/error_analysis_retrieval_summary.jsonl",
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="Base run directory of a baseline KG run to load gold session summaries from "
        "when --retrieval-mode=gold_summary_only. "
        "Expected layout: <run>/sample_<id>/artifacts/summaries_meta.jsonl. "
        "When omitted, summaries are read from the dataset JSON's session_summary field.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-snapshots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sample-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--eval-csv", help=argparse.SUPPRESS)
    parser.add_argument("--judge-csv", help=argparse.SUPPRESS)
    parser.add_argument("--stats-json", help=argparse.SUPPRESS)
    parser.add_argument("--run-root", help=argparse.SUPPRESS)
    parser.add_argument("--conv-id", help=argparse.SUPPRESS)
    parser.add_argument("--up-to-session", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_sample_ids(raw: str) -> list[int]:
    """Expand a sample spec like "0,2,5-7" into [0, 2, 5, 6, 7].

    Ranges are inclusive at both ends, matching how sample ids are written in
    the run logs.
    """
    out = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                start = int(a.strip())
                end = int(b.strip())
                for i in range(min(start, end), max(start, end) + 1):
                    out.add(i)
            continue
        if part.isdigit():
            out.add(int(part))
    return sorted(out)


def resolve_stages(
    stages: Sequence[str] | None,
    *,
    no_judge: bool = False,
    artifact_dir: str | Path | None = None,
) -> list[str]:
    """Work out which stages will run, from the include and skip flags.

    Centralised because the orchestrator and the worker must agree: if the
    parent thinks judging is on and the child does not, the run finishes with no
    judge output and no error.
    """
    retrieval_only = artifact_dir is not None
    resolved = list(stages) if stages else list(RETRIEVAL_ONLY_STAGES if retrieval_only else DEFAULT_STAGES)

    deduped: list[str] = []
    for stage in resolved:
        if stage in VALID_STAGES and stage not in deduped:
            deduped.append(stage)

    if retrieval_only:
        deduped = [stage for stage in deduped if stage != "ingest"]

    if no_judge:
        deduped = [stage for stage in deduped if stage != "judge"]

    return deduped


def build_worker_command(*, args, config: RunConfig, plan: SamplePlan) -> list[str]:
    """Render a sample's plan back into the argv its worker will parse.

    The round trip that makes subprocess isolation work. Every setting the child
    needs has to appear here -- an option the parent accepted but does not
    forward silently applies to the parent alone, and the sample runs without
    it.
    """
    cmd = [
        sys.executable,
        "-m",
        PIPELINE_MODULE,
        "--worker",
        "--sample-index",
        str(plan.sample_index),
        "--dataset",
        config.dataset,
        "--prev-k",
        str(args.prev_k),
        "--entity-sim-topk",
        str(args.entity_sim_topk),
        "--entity-sim-threshold",
        str(args.entity_sim_threshold),
        "--chunk-turns",
        str(args.chunk_turns),
        "--eval-csv",
        str(plan.worker_paths.eval_csv),
        "--judge-csv",
        str(plan.worker_paths.judge_csv),
        "--stats-json",
        str(plan.worker_paths.stats_json),
        "--tau",
        str(args.tau),
        "--run-root",
        str(config.run_root),
    ]
    if getattr(args, "retrieval_mode", ""):
        cmd.extend(["--retrieval-mode", args.retrieval_mode])
    if getattr(args, "stages", None):
        cmd.extend(["--stage", *args.stages])
    if args.adaptive:
        cmd.append("--adaptive")
    if args.adv:
        cmd.append("--adv")
    if args.no_judge:
        cmd.append("--no-judge")
    if args.artifact_dir is not None:
        cmd.extend(["--artifact-dir", str(args.artifact_dir)])
    if args.replay_run_dir is not None:
        cmd.extend(["--replay-run-dir", str(args.replay_run_dir)])
    if getattr(args, "baseline_run_dir", None) is not None:
        cmd.extend(["--baseline-run-dir", str(args.baseline_run_dir)])
    if args.source_json:
        cmd.extend(["--source-json", str(args.source_json)])
    if args.sessions_jsonl is not None:
        cmd.extend(["--sessions-jsonl", str(args.sessions_jsonl)])
    elif config.sessions_jsonl_path is not None:
        cmd.extend(["--sessions-jsonl", str(config.sessions_jsonl_path)])
    if args.dataset_json is not None:
        cmd.extend(["--dataset-json", str(args.dataset_json)])
    else:
        cmd.extend(["--dataset-json", str(config.dataset_json_path)])
    return cmd
