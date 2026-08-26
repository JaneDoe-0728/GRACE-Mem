"""Re-run selected parts of a finished LongMemEval run.

Entry point for the iterate-on-one-thing workflow: change a judge prompt or a
retrieval parameter, recompute only what that affects, and merge the results
back into the existing tables so the comparison stays valid.

`gc` is imported and used deliberately -- a rerun walks many datasets in one
process, each loading a pipeline and its models, and without explicit
collection between them the process grows until it is killed.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from experiment.experiment_config import INGEST_PARAMS, RETRIEVAL_PARAMS, RERANKER_PARAMS

# LongMem-only: must match the value used when these artifacts were ingested.
USE_SPLIT_SUMMARY = bool(INGEST_PARAMS.get("use_split_summary", True))
from experiment.longmem.pipeline.aggregate import update_all_answers_csv, update_progress_rows
from experiment.longmem.helpers.args import add_data_args, add_rerun_args, add_run_args, resolve_stages
from experiment.longmem.helpers.datasets import get_question_info, select_dataset_names
from experiment.longmem.helpers.rerun_support import (
    failed_datasets,
    retrieval_datasets,
    retrieval_datasets_from_artifacts,
    resolve_artifact_dir,
    rerun_accuracy,
    setup_retrieval_loggers,
)
from experiment.longmem.artifacts.snapshot import restore_graph_from_cache
from experiment.longmem.stages.judge import JudgeStage
from experiment.longmem.stages.qa_eval import QAEvalStage
from experiment.longmem.utils.io import append_type_subdir, ensure_dir, read_csv_frame
from grace_mem.utils.error_analysis import (
    append_analysis_record,
    append_pretty_block,
    build_bridge_label,
    build_top_miss_snapshot,
    coerce_float,
    derive_anomaly_flags,
    derive_drop_reasons,
    derive_failure_type,
    render_failure_digest,
)


class LongMemRerun:
    """Drives a partial re-run over a completed LongMemEval run.

    Holds the resolved targets and the pipeline components across datasets, so
    weights are loaded once for the whole sweep rather than per dataset.

    The consequence is that this object is long-lived and accumulates: it calls
    `gc` explicitly between datasets, because otherwise a sweep over many
    categories grows until the process is killed.
    """
    def __init__(self, *, llm, graph) -> None:
        self.llm = llm
        self.graph = graph
        self.qa_stage = QAEvalStage()
        self.judge_stage = JudgeStage()
        self._closed = False

    @classmethod
    def from_env(cls) -> "LongMemRerun":
        """Create a rerun runtime and roll back partially opened resources."""
        from grace_mem.graph.falkordb import graph_from_env
        from grace_mem.llm import LLMClient

        llm = LLMClient()
        graph = None
        try:
            graph = graph_from_env()
            opened_graph = graph.open()
            return cls(llm=llm, graph=opened_graph)
        except BaseException:
            if graph is not None:
                try:
                    graph.close()
                except Exception:
                    pass
            try:
                llm.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        """Release shared graph and LLM transports once."""
        if self._closed:
            return
        cleanup_error: Exception | None = None
        try:
            self.graph.close()
        except Exception as exc:
            cleanup_error = exc
        try:
            self.llm.close()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        self._closed = True
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self) -> "LongMemRerun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def rerun_dataset(
        self,
        *,
        dataset_name: str,
        output_dir: Path,
        data_folder: Path | None,
        log_dir: Path,
        artifact_dir: Path | None = None,
        no_judge: bool = False,
        stages: set[str] | None = None,
    ) -> dict:
        """Re-run the requested stages for one dataset and merge the results back.

        Merging rather than replacing keeps the comparison apples-to-apples: only
        the recomputed columns change, and everything else stays as the original run
        left it.
        """
        selected_stages = set(stages or {"qa_eval", "judge"})
        run_qa = "qa_eval" in selected_stages
        run_judge = "judge" in selected_stages and not no_judge

        # artifacts_dir: use artifact_dir when provided (retrieval-only mode)
        artifacts_root = artifact_dir if artifact_dir is not None else output_dir
        artifacts_dir = resolve_artifact_dir(artifacts_root, dataset_name)
        output_csv = output_dir / f"{dataset_name}.csv"
        question_csv = output_csv
        if artifact_dir is not None and not question_csv.exists():
            question_csv = artifacts_root / f"{dataset_name}.csv"

        if artifacts_dir is None:
            raise FileNotFoundError(f"artifacts dir not found for dataset: {dataset_name}")

        from grace_mem.llm import token_tracker
        from grace_mem.pipeline.retriever import Retriever, RetrieverConfig
        from grace_mem.storage import VDBManager
        from grace_mem.embeddings import embedder

        mgr = None
        retriever = None
        try:
            mgr = VDBManager(artifacts_dir)
            mgr.initialize()
            restore_graph_from_cache(self.graph, mgr.cache)
            setup_retrieval_loggers(dataset_name, log_dir)

            # Mirror the processor.py coupling: INGEST_PARAMS["use_split_summary"]
            # decides whether these artifacts have :u/:a entries, so retrieval must
            # follow the same flag (shared config keeps True for LoCoMo).
            # Ablation I: KG_ABLATION_NO_SPLIT=1 -> force single-entry pair mode
            # (paired with a pair-entry summaries_chroma; every other mechanism on
            # the split path is unchanged).
            import os as _os
            _no_split = _os.getenv("KG_ABLATION_NO_SPLIT", "0").lower() not in ("0", "", "false")
            _longmem_reranker_params = {
                **RERANKER_PARAMS,
                "split_single_entry_raw": _no_split or not USE_SPLIT_SUMMARY,
            }
            retriever = Retriever(
                llm=self.llm,
                graph=self.graph,
                mgr=mgr,
                embed=embedder.embed,
                cache=mgr.cache,
                config=RetrieverConfig(**_longmem_reranker_params),
            )

            question, question_date, gold = get_question_info(dataset_name, data_folder, question_csv)
            rewritten_q = self.qa_stage.rewrite_temporal_question(question, query_time=question_date)

            if run_qa:
                token_tracker.set_context(dataset=dataset_name, stage="retrieve", log_dir=log_dir)
                context = self.qa_stage.build_context(
                    retriever,
                    question=rewritten_q,
                    retrieval_params=RETRIEVAL_PARAMS,
                    query_time=question_date,
                )
                from experiment.agent_filter.harness import maybe_refine_context
                context = maybe_refine_context(
                    question=rewritten_q,
                    context=context,
                    csv_path=(data_folder / f"{dataset_name}.csv") if data_folder else None,
                    llm=self.llm,
                    question_date=question_date,
                    category=data_folder.name if data_folder else None,
                    log_dir=log_dir,
                    artifact_dir=artifacts_dir,
                )
                has_context = "(no KG context)" not in context and context.strip()

                # Write retrieval_summary + drop_reasons for error analysis
                trace = getattr(retriever, "last_retrieval_trace", None) or {}
                if trace:
                    selected_evidence = trace.get("selected_evidence") or []
                    retrieval_record = {
                        "request_id": trace.get("request_id"),
                        "question": question,
                        "stop_reason": trace.get("stop_reason"),
                        "branches": trace.get("branches", {}),
                        "pass2_triggered": bool(trace.get("pass2_triggered", False)),
                        "conf_pass1": trace.get("conf_pass1"),
                        "conf_pass2": trace.get("conf_pass2"),
                        "conf_final": trace.get("conf_final"),
                        "tau_confidence": trace.get("tau_confidence"),
                        "pass1_entity_ids": trace.get("pass1_entity_ids", []),
                        "final_entity_count": trace.get("final_entity_count", 0),
                        "final_relationship_count": trace.get("final_relationship_count", 0),
                        "selected_evidence_count": trace.get("selected_evidence_count", 0),
                        "selected_evidence_ids": [
                            item.get("summary_id") for item in selected_evidence if item.get("summary_id")
                        ],
                        "has_temporal_evidence": bool(trace.get("has_temporal_evidence", False)),
                    }
                    append_analysis_record(log_dir, "retrieval_summary", retrieval_record)
                    for drop_reason in derive_drop_reasons(retrieval_record):
                        append_analysis_record(log_dir, "drop_reasons", drop_reason)
                else:
                    retrieval_record = {}

                token_tracker.set_context(dataset=dataset_name, stage="qa", log_dir=log_dir)
                answer = self.qa_stage.ask_llm(
                    self.llm,
                    question=rewritten_q,
                    context=context,
                    question_date=question_date,
                )
            else:
                if not output_csv.exists():
                    raise FileNotFoundError(
                        f"Output CSV not found for stage selection without qa_eval: {output_csv}"
                    )
                existing_df = read_csv_frame(output_csv)
                row = existing_df.iloc[0].to_dict() if len(existing_df) else {}
                context = str(row.get("Retrieved_Context", ""))
                answer = str(row.get("Generated_Answer", ""))
                has_context = "(no KG context)" not in context and context.strip()
                retrieval_record = {}

            correctness = ""
            if run_judge and gold:
                token_tracker.set_context(dataset=dataset_name, stage="judge", log_dir=log_dir)
                correctness = str(
                    self.judge_stage.judge_single(
                        self.llm,
                        question=question,
                        gold=gold,
                        generated=answer,
                    )
                )

            # Write anomaly flags, failure verdict, and digest for error analysis
            correctness_float = coerce_float(correctness)
            summary = {**retrieval_record, "question": question, "answer": answer}
            anomaly_flags = derive_anomaly_flags(summary=summary, correctness=correctness_float)
            if anomaly_flags:
                append_analysis_record(log_dir, "anomaly_flags", {
                    "request_id": retrieval_record.get("request_id"),
                    "question": question,
                    "flags": anomaly_flags,
                })
            if correctness_float is not None and correctness_float < 1:
                failure_type = derive_failure_type(summary=summary, correctness=correctness_float)
                bridge_label = build_bridge_label(summary=summary, correctness=correctness_float)
                top_miss = build_top_miss_snapshot(log_dir=log_dir, request_id=retrieval_record.get("request_id"))
                append_analysis_record(log_dir, "failure_verdict", {
                    "request_id": retrieval_record.get("request_id"),
                    "question": question,
                    "failure_type": failure_type,
                    "correctness": correctness_float,
                    "anomaly_flags": anomaly_flags,
                    "stop_reason": retrieval_record.get("stop_reason"),
                    "selected_evidence_count": retrieval_record.get("selected_evidence_count", 0),
                })
                append_analysis_record(log_dir, "top_miss", {
                    "request_id": retrieval_record.get("request_id"),
                    "question": question,
                    "top_miss": top_miss,
                })
                append_analysis_record(log_dir, "evidence_bridge", {
                    "request_id": retrieval_record.get("request_id"),
                    "question": question,
                    "bridge_label": bridge_label,
                })
            digest = render_failure_digest(
                sample_index=0,
                ingest_records=[],
                failures=[{
                    "question": question,
                    "failure_type": derive_failure_type(summary=summary, correctness=correctness_float) if correctness_float is not None and correctness_float < 1 else "correct",
                    "correctness": correctness_float,
                    "anomaly_flags": anomaly_flags,
                    "request_id": retrieval_record.get("request_id"),
                    "stop_reason": retrieval_record.get("stop_reason"),
                    "selected_evidence_count": retrieval_record.get("selected_evidence_count", 0),
                    "top_miss": build_top_miss_snapshot(log_dir=log_dir, request_id=retrieval_record.get("request_id")) if correctness_float is not None and correctness_float < 1 else [],
                }] if correctness_float is not None else [],
            )
            append_pretty_block(log_dir, "error_analysis_failure_digest.log", digest)

            result_df = self.qa_stage.single_result_frame(
                question=question,
                question_date=question_date,
                context=context,
                answer=answer,
                gold=gold,
                correctness=correctness,
            )
            result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

            return {
                "dataset": dataset_name,
                "has_context": has_context,
                "question": question,
                "question_date": question_date or "",
                "context": context,
                "answer": answer,
                "gold": gold,
                "correctness": correctness,
                "output_path": str(output_csv),
            }
        finally:
            try:
                self.graph.clear_all()
            except Exception:
                pass
            if mgr is not None:
                mgr.close(clear_cache=True)
            del retriever, mgr
            gc.collect()


def resolve_rerun_targets(
    *,
    data_root: Path | None,
    artifact_root: Path | None,
    output_root: Path,
    result_root: Path | None,
    type_names: list[str] | None,
) -> list[tuple[str | None, Path | None, Path | None, Path, Path | None]]:
    """Decide which datasets this rerun will cover."""
    if type_names:
        return [
            (
                name,
                append_type_subdir(data_root, name) if data_root is not None else None,
                append_type_subdir(artifact_root, name) if artifact_root is not None else None,
                append_type_subdir(output_root, name),
                append_type_subdir(result_root, name) if result_root is not None else None,
            )
            for name in type_names
        ]

    if artifact_root is None:
        return [(None, data_root, None, output_root, result_root)]

    category_dirs: list[Path] = []
    for subdir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        try:
            if retrieval_datasets_from_artifacts(subdir, None):
                category_dirs.append(subdir)
        except Exception:
            continue

    if not category_dirs:
        return [(None, data_root, artifact_root, output_root, result_root)]

    targets: list[tuple[str | None, Path | None, Path | None, Path, Path | None]] = []
    for category_dir in category_dirs:
        category = category_dir.name
        if data_root is None:
            category_data = None
        else:
            candidate_data = data_root / category
            category_data = candidate_data if candidate_data.exists() else data_root
        targets.append(
            (
                category,
                category_data,
                category_dir,
                output_root / category,
                result_root / category if result_root is not None else None,
            )
        )
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongMem retrieval rerun tool")
    add_data_args(parser)
    add_run_args(parser)
    add_rerun_args(parser)
    parser.add_argument("--type", nargs="+", default=None, metavar="TYPE", help="One or more subfolders to append to --data-folder")
    # output-root is required for standalone rerun
    parser.set_defaults()
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output_root is None:
        parser.error("--output-root is required")

    output_root = Path(args.output_root)
    data_root = Path(args.data_folder) if args.data_folder else None
    artifact_root = Path(args.artifact_dir) if args.artifact_dir else None
    result_root = Path(args.result_dir) if args.result_dir else None
    selected_stages = resolve_stages(
        args.stages,
        no_judge=args.no_judge,
        artifact_dir=str(artifact_root) if artifact_root is not None else None,
    )
    dataset_selector = args.dataset_id
    targets = resolve_rerun_targets(
        data_root=data_root,
        artifact_root=artifact_root,
        output_root=output_root,
        result_root=result_root,
        type_names=args.type,
    )
    resolved_targets: list[tuple[str | None, Path | None, Path | None, Path, Path | None, list[str]]] = []
    total = 0
    for category, data_folder, artifact_dir, output_dir, result_dir in targets:
        ensure_dir(output_dir)
        if artifact_dir is not None:
            to_rerun = retrieval_datasets_from_artifacts(artifact_dir, args.datasets)
        else:
            to_rerun = retrieval_datasets(output_dir, args.datasets, force=args.force)
            to_rerun = [name for name in to_rerun if (output_dir / f"artifacts_{name}").exists()]

        to_rerun = select_dataset_names(
            to_rerun,
            dataset_selector,
            scope_label=f"rerun target '{(artifact_dir or output_dir).name}'",
        )
        if args.num is not None:
            to_rerun = to_rerun[: args.num]
        resolved_targets.append((category, data_folder, artifact_dir, output_dir, result_dir, to_rerun))
        total += len(to_rerun)

    print(f"Datasets to rerun: {total}")
    for category, _, _, _, _, to_rerun in resolved_targets:
        if category is not None and len(resolved_targets) > 1:
            print(f"[{category}] {len(to_rerun)}")
        for name in to_rerun:
            print(f"  - {name}")
    if total == 0:
        print("Nothing to do.")
        return

    results = []
    success_results: dict[Path, list[dict]] = {}
    error_results: list[dict] = []
    current = 0
    with LongMemRerun.from_env() as runner:
        for category, data_folder, artifact_dir, output_dir, result_dir, to_rerun in resolved_targets:
            if category is not None and len(resolved_targets) > 1:
                print(f"\n=== Category: {category} ===")
            for dataset_name in to_rerun:
                current += 1
                print(f"\n{'#' * 60}")
                print(f"# [{current}/{total}] {dataset_name}")
                print(f"{'#' * 60}")

                dataset_log_dir = output_dir / f"logs_{dataset_name}"
                ensure_dir(dataset_log_dir)

                try:
                    result = runner.rerun_dataset(
                        dataset_name=dataset_name,
                        output_dir=output_dir,
                        data_folder=data_folder,
                        log_dir=dataset_log_dir,
                        artifact_dir=artifact_dir,
                        no_judge=args.no_judge,
                        stages=set(selected_stages),
                    )
                    results.append(result)
                    success_results.setdefault(output_dir, []).append(result)
                    print(f"{dataset_name} | correctness={result['correctness']}")
                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    print(f"{dataset_name}: {exc}")
                    error_result = {"dataset": dataset_name, "error": str(exc)}
                    results.append(error_result)
                    error_results.append(error_result)
                    try:
                        runner.graph.clear_all()
                    except Exception:
                        pass

            success = success_results.get(output_dir, [])
            if success:
                progress_filename = "progress.csv" if artifact_dir is not None else "progress_rerun.csv"
                update_progress_rows(output_dir, success, filename=progress_filename)
                if artifact_dir is not None:
                    update_all_answers_csv(output_dir, success)
                if result_dir is not None:
                    update_progress_rows(result_dir, success, filename=progress_filename)
            if result_dir is not None:
                ensure_dir(result_dir)

    if error_results:
        print(f"Errors: {len(error_results)}")
        for row in error_results:
            print(f"  {row['dataset']}: {row['error']}")

    success = [row for row in results if "error" not in row]
    correct, judged = rerun_accuracy(success)
    if judged:
        print(f"Accuracy: {correct}/{judged} = {correct / judged:.1%}")


if __name__ == "__main__":
    main()
