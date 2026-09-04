"""Generate answers from gold evidence turns for LoCoMo or LongMemEval.

Oracle generation is intentionally separate from standardized judging. Pass
``--judge`` to run the shared judge after generation, or invoke
``experiment/common/evaluation/judge.py`` later.

Examples:
    uv run python experiment/common/evaluation/oracle.py locomo oracle-locomo --samples 0-9
    uv run python experiment/common/evaluation/oracle.py locomo oracle-window2 --window 2 --include-photo
    uv run python experiment/common/evaluation/oracle.py longmem oracle-longmem --workers 8
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from experiment.agent_filter.corpus import Corpus, load_corpus
from experiment.common.evaluation.judge import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LONGMEM_CATEGORIES,
    openai_api_key,
)
from experiment.locomo.cli import parse_sample_ids
from experiment.locomo.helpers.dataset import (
    is_adversarial_item,
    load_qa_items_from_sample,
    normalize_qa_item,
)
from experiment.longmem.stages.qa_eval import QAEvalStage
from grace_mem.services.llm import LLMClient

LOCOMO_DATA = _ROOT / "experiment" / "locomo" / "data" / "locomo10.json"
LOCOMO_OUTPUT = _ROOT / "experiment" / "locomo" / "output" / "standard"
LONGMEM_DATA = _ROOT / "experiment" / "longmem" / "script_data"
LONGMEM_OUTPUT = _ROOT / "experiment" / "longmem" / "output"
EVIDENCE_ID = re.compile(r"D(\d+):(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class OracleConfig:
    """Settings for an oracle run: answer with gold evidence, skip retrieval.

    The oracle establishes the ceiling. Feeding the model exactly the gold
    evidence answers "how much of the remaining error is retrieval's fault?" --
    whatever the oracle still gets wrong is answer-generation error that better
    retrieval cannot fix.

    Attributes:
        window: Turns of context included on each side of a gold turn. Non-zero
            because a gold turn read alone often loses its referent, and a
            ceiling measured on unusable evidence is not a ceiling.
    """

    benchmark: str
    run_tag: str
    window: int
    answer_base_url: str
    answer_model: str
    include_photo: bool = False


_thread_local = threading.local()


def _answer_client(config: OracleConfig) -> LLMClient:
    """Return this thread's client, rebuilding it if the endpoint changed.

    Thread-local because LLMClient is not thread-safe and the oracle answers
    questions in parallel. Keyed on (base_url, model) so a sweep that varies
    the answering model does not keep reusing a client pointed at the previous
    one.

    The timeout is far above the default: oracle prompts carry whole sessions
    of gold evidence and are correspondingly slow.
    """
    key = (config.answer_base_url, config.answer_model)
    if getattr(_thread_local, "client_key", None) != key:
        api_key = (
            openai_api_key()
            if config.answer_base_url.rstrip("/").endswith("openai.com/v1")
            else None
        )
        _thread_local.client = LLMClient(
            base_url=config.answer_base_url,
            model_name=config.answer_model,
            api_key=api_key,
            timeout=300.0,
        )
        _thread_local.client_key = key
    return _thread_local.client


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def expand_longmem_sids(corpus: Corpus, gold_sids: Sequence[str], window: int) -> list[str]:
    """Expand gold split-sids by corpus position within the same session."""
    selected: set[str] = set()
    for sid in gold_sids:
        for target in corpus.resolve(sid):
            for turn in corpus.turns:
                if turn.session_id == target.session_id and abs(turn.pos - target.pos) <= window:
                    selected.add(turn.sid)
    return [turn.sid for turn in corpus.turns if turn.sid in selected]


def longmem_gold_sids(frame: pd.DataFrame) -> list[str]:
    """Read the gold-evidence sids from a LongMemEval question CSV.

    The CSV marks gold per turn, while sids address user/assistant pairs, so a
    user turn maps to the following index and an assistant turn to its own.
    That off-by-one is the thing to get right: reversed, the oracle is fed the
    turns adjacent to the evidence rather than the evidence.

    Returns [] when the frame has no has_answer column, which is how a file
    without gold annotation is skipped rather than treated as having none.
    """
    if "has_answer" not in frame.columns:
        return []
    result: list[str] = []
    for _, row in frame[frame["has_answer"].map(_truthy)].iterrows():
        session = str(row["session_id"]).strip()
        turn_index = int(row["turn_index"])
        role = str(row["role"]).strip().lower()
        pair = turn_index + 1 if role == "user" else turn_index
        result.append(f"{session}:{pair}:{'u' if role == 'user' else 'a'}")
    return result


def build_longmem_context(csv_path: Path, window: int) -> tuple[str, list[str]]:
    """Render the gold evidence for one question as the oracle's context.

    Sids are emitted inline so the answer can be traced back to specific turns,
    and dates are included because many questions are temporal and unanchored
    evidence cannot answer them.

    Returns:
        (rendered context, the sids it covers) -- the sid list is what recall
        is measured against.
    """
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    frame.columns = [column.lstrip("\ufeff") for column in frame.columns]
    corpus = load_corpus(csv_path)
    sids = expand_longmem_sids(corpus, longmem_gold_sids(frame), window)
    lines = ["### Gold Evidence"]
    for sid in sids:
        turn = corpus.resolve(sid)[0]
        body = corpus.display_entry(sid) or ""
        date = f"[{turn.date}]" if turn.date else ""
        lines.append(f"{date}[sid={sid}] {body}")
    return "\n".join(lines), sids


def _locomo_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a LoCoMo sample's sessions into one ordered turn list.

    Sessions are visited in numeric order so turn positions match the
    conversation; the evidence expansion below addresses turns by position, and
    lexical session ordering would shift every one of them.
    """
    conversation = sample.get("conversation", {}) or {}
    turns: list[dict[str, Any]] = []
    for key, values in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if not match or not isinstance(values, list):
            continue
        session = int(match.group(1))
        date = str(conversation.get(f"session_{session}_date_time", "") or "")
        for position, raw in enumerate(values, start=1):
            dia_id = str(raw.get("dia_id", f"D{session}:{position}"))
            id_match = EVIDENCE_ID.search(dia_id)
            turn_number = int(id_match.group(2)) if id_match else position
            turns.append(
                {
                    "dia_id": f"D{session}:{turn_number}",
                    "session": session,
                    "turn": turn_number,
                    "speaker": str(raw.get("speaker", "?")),
                    "text": str(raw.get("text", "")),
                    "caption": str(raw.get("blip_caption", "") or ""),
                    "date": date,
                }
            )
    return turns


def expand_locomo_evidence(
    sample: dict[str, Any],
    evidence: Sequence[str],
    window: int,
) -> list[dict[str, Any]]:
    """Widen gold evidence ids to include neighbouring turns.

    A gold turn read alone often loses its referent -- "yes, that one" needs the
    turn that named it -- so a ceiling measured on isolated turns understates
    what perfect retrieval could achieve. Expansion is bounded to the same
    session, since adjacency across a session boundary is not adjacency in the
    conversation.
    """
    turns = _locomo_turns(sample)
    targets = {
        (int(match.group(1)), int(match.group(2)))
        for value in evidence
        for match in EVIDENCE_ID.finditer(str(value))
    }
    selected = {
        turn["dia_id"]
        for turn in turns
        if any(
            turn["session"] == session and abs(turn["turn"] - number) <= window
            for session, number in targets
        )
    }
    return [turn for turn in turns if turn["dia_id"] in selected]


def build_locomo_context(
    sample: dict[str, Any],
    evidence: Sequence[str],
    *,
    window: int,
    include_photo: bool,
) -> tuple[str, list[str]]:
    """Render a LoCoMo question's expanded gold evidence as the oracle's context.

    Speaker and date are kept on each turn: many questions are temporal or turn
    on who said something, and stripped of both the evidence cannot answer them.
    """
    turns = expand_locomo_evidence(sample, evidence, window)
    lines = ["### Gold Evidence"]
    for turn in turns:
        caption = f" [Image: {turn['caption']}]" if include_photo and turn["caption"] else ""
        date = f"[{turn['date']}]" if turn["date"] else ""
        lines.append(
            f"{date}[sid={turn['dia_id']}] {turn['speaker']}: {turn['text']}{caption}"
        )
    return "\n".join(lines), [turn["dia_id"] for turn in turns]


def _ask(config: OracleConfig, question: str, context: str, question_date: str | None = None) -> str:
    """Ask the answering model one question against a fixed context.

    The single place the oracle calls a model, so retrieval is provably not in
    the loop -- whatever it gets wrong is answer-generation error.
    """
    stage = QAEvalStage()
    rewritten = stage.rewrite_temporal_question(question, query_time=question_date)
    return stage.ask_llm(
        _answer_client(config),
        question=rewritten,
        context=context,
        question_date=question_date,
    )


def _first_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = frame[column].dropna()
    return str(values.iloc[0]).strip() if not values.empty else ""


def _process_longmem_file(
    source: Path,
    output: Path,
    config: OracleConfig,
) -> str:
    """Run the oracle over one LongMemEval question file.

    Returns:
        A result record, or None when the file carries no gold annotation and so
        has no evidence to feed the oracle.
    """
    frame = pd.read_csv(source, encoding="utf-8-sig")
    frame.columns = [column.lstrip("\ufeff") for column in frame.columns]
    question = _first_text(frame, "question")
    gold = _first_text(frame, "answer")
    question_date = _first_text(frame, "question_date") or None
    context, sids = build_longmem_context(source, config.window)
    if not sids:
        return f"[SKIP] {source.stem}: no gold evidence"
    answer = _ask(config, question, context, question_date)
    QAEvalStage().single_result_frame(
        question=question,
        question_date=question_date,
        context=context,
        answer=answer,
        gold=gold,
    ).to_csv(output, index=False, encoding="utf-8-sig")
    return f"[OK] {source.parent.name}/{source.stem}: evidence={len(sids)}"


def run_longmem(args: argparse.Namespace) -> Path:
    """Run the oracle over a LongMemEval run and report its ceiling.

    Whatever the oracle gets wrong given perfect evidence is answer-generation
    error that no retrieval improvement can recover.
    """
    config = OracleConfig(
        benchmark="longmem",
        run_tag=args.run_tag,
        window=args.window,
        answer_base_url=args.answer_base_url,
        answer_model=args.answer_model,
    )
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    run_dir = output_root / args.run_tag
    jobs: list[tuple[Path, Path]] = []
    for directory in LONGMEM_CATEGORIES:
        if args.category and directory != args.category:
            continue
        output_dir = run_dir / directory
        output_dir.mkdir(parents=True, exist_ok=True)
        picked = 0
        for source in sorted((data_root / directory).glob("*.csv")):
            if args.name and source.stem != args.name:
                continue
            output = output_dir / source.name
            if output.exists() and not args.force:
                continue
            jobs.append((source, output))
            picked += 1
            if args.limit and picked >= args.limit:
                break

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_longmem_file, source, output, config): source
            for source, output in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                message = future.result()
            except Exception as exc:
                message = f"[ERR] {futures[future]}: {exc}"
            print(f"({index}/{len(jobs)}) {message}", flush=True)
    _write_metadata(run_dir, config, args)
    return run_dir


def _process_locomo_sample(
    sample_index: int,
    sample: dict[str, Any],
    run_dir: Path,
    config: OracleConfig,
    *,
    include_adversarial: bool,
    limit: int,
) -> str:
    """Run the oracle over every question in one LoCoMo sample.

    The sample's conversation is flattened once and reused across its questions,
    since re-flattening per question is the dominant cost for a long
    conversation.
    """
    rows: list[dict[str, object]] = []
    for raw_item in load_qa_items_from_sample(sample):
        item = normalize_qa_item(raw_item)
        if not include_adversarial and is_adversarial_item(item):
            continue
        context, sids = build_locomo_context(
            sample,
            item["evidence"],
            window=config.window,
            include_photo=config.include_photo,
        )
        if not sids:
            continue
        answer = _ask(config, item["question"], context)
        rows.append(
            {
                "question": item["question"],
                "gold_answer": item["answer"],
                "model_answer": answer,
                "category": item["category"],
                "category_label": item["category_label"],
                "retrieved_context": context,
                "selected_evidence_ids": json.dumps(sids),
            }
        )
        if limit and len(rows) >= limit:
            break
    sample_dir = run_dir / f"sample_{sample_index}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    output = sample_dir / f"sample{sample_index}_eval_{config.run_tag}.csv"
    pd.DataFrame(
        rows,
        columns=(
            "question",
            "gold_answer",
            "model_answer",
            "category",
            "category_label",
            "retrieved_context",
            "selected_evidence_ids",
        ),
    ).to_csv(output, index=False, encoding="utf-8-sig")
    return f"[OK] sample_{sample_index}: questions={len(rows)}"


def run_locomo(args: argparse.Namespace) -> Path:
    """Run the oracle over a LoCoMo run and report its ceiling."""
    dataset_path = Path(args.dataset_json)
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    sample_ids = parse_sample_ids(args.samples)
    if not sample_ids:
        raise ValueError("--samples did not resolve to any sample IDs")
    invalid_ids = [sample_index for sample_index in sample_ids if not 0 <= sample_index < len(samples)]
    if invalid_ids:
        raise ValueError(f"Sample IDs out of range: {invalid_ids}")
    config = OracleConfig(
        benchmark="locomo",
        run_tag=args.run_tag,
        window=args.window,
        answer_base_url=args.answer_base_url,
        answer_model=args.answer_model,
        include_photo=args.include_photo,
    )
    run_dir = Path(args.output_root) / args.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        sample_index
        for sample_index in sample_ids
        if args.force
        or not (
            run_dir
            / f"sample_{sample_index}"
            / f"sample{sample_index}_eval_{config.run_tag}.csv"
        ).exists()
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_locomo_sample,
                sample_index,
                samples[sample_index],
                run_dir,
                config,
                include_adversarial=args.include_adversarial,
                limit=args.limit,
            ): sample_index
            for sample_index in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                message = future.result()
            except Exception as exc:
                message = f"[ERR] sample_{futures[future]}: {exc}"
            print(f"({index}/{len(jobs)}) {message}", flush=True)
    _write_metadata(run_dir, config, args)
    return run_dir


def _write_metadata(run_dir: Path, config: OracleConfig, args: argparse.Namespace) -> None:
    """Record the oracle run's configuration alongside its results."""
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "entrypoint": "experiment.common.evaluation.oracle",
        "config": asdict(config),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "handler"
        },
    }
    (run_dir / "oracle_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_judge(args: argparse.Namespace) -> None:
    """Judge the oracle's answers using the same judge as a normal run.

    The same judge on purpose: an oracle scored by a different rubric would not
    be comparable with the runs it is meant to bound.
    """
    command = [
        sys.executable,
        "-m",
        "experiment.common.evaluation.judge",
        args.benchmark,
        args.run_tag,
        "--output-root",
        str(args.output_root),
        "--workers",
        str(args.workers),
        "--judge-base-url",
        args.judge_base_url,
        "--judge-model",
        args.judge_model,
    ]
    if args.benchmark == "locomo":
        command.extend(["--samples", args.samples])
        if args.include_adversarial:
            command.append("--include-adversarial")
    subprocess.run(command, cwd=_ROOT, check=True)


def _add_common(parser: argparse.ArgumentParser, *, output_root: Path) -> None:
    parser.add_argument("run_tag")
    parser.add_argument("--output-root", default=str(output_root))
    parser.add_argument("--window", type=int, default=0, help="Turns before/after each gold turn")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--answer-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--answer-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge", action="store_true", help="Run standardized judge after generation")
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    locomo = subparsers.add_parser("locomo", help="LoCoMo gold-turn oracle")
    _add_common(locomo, output_root=LOCOMO_OUTPUT)
    locomo.add_argument("--dataset-json", default=str(LOCOMO_DATA))
    locomo.add_argument("--samples", default="0-9")
    locomo.add_argument("--include-photo", action="store_true")
    locomo.add_argument("--include-adversarial", action="store_true")
    locomo.add_argument("--limit", type=int, default=0, help="Maximum questions per sample")
    locomo.set_defaults(handler=run_locomo)

    longmem = subparsers.add_parser("longmem", help="LongMemEval gold-turn oracle")
    _add_common(longmem, output_root=LONGMEM_OUTPUT)
    longmem.add_argument("--data-root", default=str(LONGMEM_DATA))
    longmem.add_argument("--category", choices=tuple(LONGMEM_CATEGORIES), default=None)
    longmem.add_argument("--name", default=None)
    longmem.add_argument("--limit", type=int, default=0, help="Maximum questions per category")
    longmem.set_defaults(handler=run_longmem)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.window < 0:
        raise SystemExit("--window must be non-negative")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    args.output_root = Path(args.output_root)
    args.handler(args)
    if args.judge:
        _run_judge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
