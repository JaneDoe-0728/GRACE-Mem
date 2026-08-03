"""
Multi-Dataset Iterative Processing Script
==========================================
Process multiple CSV datasets iteratively:
1. Ingest sessions from CSV → build separate VDB for each dataset
2. Answer questions using QA pipeline
3. Clear graph database (reusable)
4. Keep separate VDBs (multi-VDB architecture)

Each dataset gets its own:
- Artifacts directory with VDB files
- Cache files
- Output CSV with answers

Shared across datasets:
- Graph database (cleared between datasets)
- LLM client
- Embedder
"""

import gc
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Set
import sys
import traceback
from datetime import datetime

_HERE = Path(__file__).resolve().parent        # experiment/longmem/
_EXP = _HERE.parent                            # experiment/
_ROOT = _EXP.parent                            # repo root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP))

from KG.storage import CacheStore, VDBManager
from KG.pipeline.ingestor import Ingestor
from KG.pipeline.retriever import Retriever, RetrieverConfig
from experiment_config import INGEST_PARAMS, RERANKER_PARAMS

# LongMem-only: whether summaries are rebuilt into :u/:a entry pairs after ingest.
# Drives both the rebuild step and the matching retrieval setting.
USE_SPLIT_SUMMARY = bool(INGEST_PARAMS.get("use_split_summary", True))
from KG.llm import LLMClient, token_tracker
from KG.graph.falkordb import graph_from_env
from embeddings import embedder
from KG.services import EntityManager, RelationshipManager, Provenance
from KG.utils.logger_config import make_module_jlog
from experiment.longmem import decision
from experiment.longmem.aggregate import update_all_answers_csv
from experiment.longmem.helpers.checkpoints import (
    checkpoint_path as shared_checkpoint_path,
    load_checkpoint as shared_load_checkpoint,
    save_checkpoint as shared_save_checkpoint,
)
from experiment.longmem.helpers.rerun_support import cleanup_retrieval_loggers
from experiment.longmem.helpers.progress import (
    append_stuck_history as shared_append_stuck_history,
    init_progress_rows as shared_init_progress_rows,
    load_progress as shared_load_progress,
    progress_path as shared_progress_path,
    save_progress_row as shared_save_progress_row,
)
from experiment.longmem.models import DatasetConfig
from experiment.longmem.snapshot import restore_graph_from_cache
from experiment.longmem.stage_adapter import (
    rewrite_temporal_question as shared_rewrite_temporal_question,
    single_result_frame,
)
from experiment.longmem.stages import IngestStage, JudgeStage, QAEvalStage, UploadStage
from experiment.longmem.utils.io import append_jsonl, ensure_dir, read_csv_frame, write_csv_frame
from KG.utils.error_analysis import (
    append_analysis_record,
    append_pretty_block,
    build_top_miss_snapshot,
    coerce_float,
    compact_json,
    derive_anomaly_flags,
    derive_drop_reasons,
    derive_failure_type,
    build_bridge_label,
    render_failure_digest,
)


logger = logging.getLogger(__name__)


class MultiDatasetProcessor:
    """Process multiple datasets with separate VDBs but shared graph"""

    def __init__(
        self,
        base_output_dir: str = "./experiment/longmem/output/default/multi_session",
        *,
        upload_nocodb: bool = False,
        noco_table_name: str | None = None,
    ):
        self.base_output_dir = Path(base_output_dir)
        ensure_dir(self.base_output_dir)
        self.upload_nocodb = upload_nocodb
        self.noco_table_name = noco_table_name

        # Shared runtime components are loaded lazily so upload-only runs
        # do not require graph / embedding initialization.
        self.llm = None
        self.graph = None
        self.embedder = None

        # Split-summary rebuild helpers. Expensive to construct (raw-context index +
        # llmlingua model) and stateless across datasets, so they are built once on
        # first use and deliberately NOT reset by _cleanup_current_dataset().
        self._split_lookup = None
        self._split_compressor = None

        # Dataset-specific components (reinitialized per dataset)
        self.current_mgr: Optional[VDBManager] = None
        self.current_ingestor: Optional[Ingestor] = None
        self.current_retriever: Optional[Retriever] = None
        self.current_ent: Optional[EntityManager] = None
        self.current_rel: Optional[RelationshipManager] = None
        self.ingest_stage = IngestStage()
        self.qa_stage = QAEvalStage()
        self.judge_stage = JudgeStage()
        self.upload_stage = UploadStage()

    def _ensure_runtime_components(self) -> None:
        if self.llm is None:
            self.llm = LLMClient()
        if self.graph is None:
            self.graph = graph_from_env().open()
        if self.embedder is None:
            self.embedder = embedder

    @staticmethod
    def parse_and_rewrite_question(question: str, query_time: str | None = None) -> str:
        return shared_rewrite_temporal_question(question, query_time=query_time)

    def _normalize_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.ingest_stage.normalize_sessions(df)

    def _session_failure_log_path(self, config: DatasetConfig) -> Path:
        return self.base_output_dir / f"session_failures_{config.name}.jsonl"

    def _record_session_failure(
        self,
        config: DatasetConfig,
        *,
        session_id: str,
        mode: str,
        turn_count: int,
        error: Exception,
    ) -> None:
        append_jsonl(
            self._session_failure_log_path(config),
            {
                "dataset": config.name,
                "session_id": session_id,
                "mode": mode,
                "turn_count": turn_count,
                "error_type": type(error).__name__,
                "error": str(error),
                "logged_at": datetime.utcnow().isoformat() + "Z",
            },
        )

    def _ingest_by_turn_pairs(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
    ) -> Dict:
        """Ingest data as user-assistant turn pairs"""
        data = self._normalize_sessions(df)
        report = {}
        checkpoint = self._load_checkpoint(config)
        processed = set(str(s) for s in checkpoint.get("processed_session_ids", []))
        total_sessions = len(data.groupby("session_id"))

        print(f"\n[INGEST] Processing {len(data.groupby('session_id'))} sessions as turn pairs...")

        for sid_idx, (sid, g) in enumerate(data.groupby("session_id", sort=False), start=1):
            sid = str(sid)
            if config.resume and sid in processed:
                print(f"  [{sid_idx}] Session {sid} - skipped (checkpoint)")
                continue
            print(f"  [{sid_idx}] Session {sid} - {len(g)} turns")
            try:
                report[sid] = self.ingest_stage.ingest_by_turn_pairs(
                    self.current_ingestor,
                    g,
                    prev_k=config.prev_k,
                    entity_sim_topk=config.entity_sim_topk,
                    entity_sim_threshold=config.entity_sim_threshold,
                ).get(sid, [])
            except Exception as exc:
                print(f"  [{sid_idx}] Session {sid} - failed, skipping: {exc}")
                logger.exception("Session ingest failed for dataset %s session %s", config.name, sid)
                self._record_session_failure(
                    config,
                    session_id=sid,
                    mode="turn_pairs",
                    turn_count=len(g),
                    error=exc,
                )
                report[sid] = {
                    "skipped": True,
                    "error": str(exc),
                    "turn_count": len(g),
                }
                processed.add(sid)
                self._save_checkpoint(
                    config,
                    processed,
                    total_sessions=total_sessions,
                    stage="ingest_in_progress",
                )
                if self.current_mgr:
                    self.current_mgr.flush_persist()
                continue

            processed.add(sid)
            if config.checkpoint_every_n_sessions > 0 and len(processed) % config.checkpoint_every_n_sessions == 0:
                self._save_checkpoint(
                    config,
                    processed,
                    total_sessions=total_sessions,
                    stage="ingest_in_progress",
                )
                if self.current_mgr:
                    self.current_mgr.flush_persist()

        # Final checkpoint after ingestion
        self._save_checkpoint(
            config,
            processed,
            total_sessions=total_sessions,
            stage="ingest_complete",
        )
        if self.current_mgr:
            self.current_mgr.flush_persist()

        return dict(report)

    def _ingest_by_session(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
    ) -> Dict:
        """Ingest entire session as one turn"""
        data = self._normalize_sessions(df)
        results = {}
        checkpoint = self._load_checkpoint(config)
        processed = set(str(s) for s in checkpoint.get("processed_session_ids", []))
        total_sessions = len(data.groupby("session_id"))

        print(f"\n[INGEST] Processing {len(data.groupby('session_id'))} sessions as whole conversations...")

        for sid_idx, (sid, g) in enumerate(data.groupby("session_id", sort=False), start=1):
            sid = str(sid)
            if config.resume and sid in processed:
                print(f"  [{sid_idx}] Session {sid} - skipped (checkpoint)")
                continue
            print(f"  [{sid_idx}] Session {sid} - {len(g)} turns combined")
            try:
                results[sid] = self.ingest_stage.ingest_by_session(
                    self.current_ingestor,
                    g,
                    prev_k=config.prev_k,
                    entity_sim_topk=config.entity_sim_topk,
                    entity_sim_threshold=config.entity_sim_threshold,
                ).get(sid, {})
            except Exception as exc:
                print(f"  [{sid_idx}] Session {sid} - failed, skipping: {exc}")
                logger.exception("Session ingest failed for dataset %s session %s", config.name, sid)
                self._record_session_failure(
                    config,
                    session_id=sid,
                    mode="session",
                    turn_count=len(g),
                    error=exc,
                )
                results[sid] = {
                    "skipped": True,
                    "error": str(exc),
                    "turn_count": len(g),
                }
                processed.add(sid)
                self._save_checkpoint(
                    config,
                    processed,
                    total_sessions=total_sessions,
                    stage="ingest_in_progress",
                )
                if self.current_mgr:
                    self.current_mgr.flush_persist()
                continue

            processed.add(sid)
            if config.checkpoint_every_n_sessions > 0 and len(processed) % config.checkpoint_every_n_sessions == 0:
                self._save_checkpoint(
                    config,
                    processed,
                    total_sessions=total_sessions,
                    stage="ingest_in_progress",
                )
                if self.current_mgr:
                    self.current_mgr.flush_persist()

        self._save_checkpoint(
            config,
            processed,
            total_sessions=total_sessions,
            stage="ingest_complete",
        )
        if self.current_mgr:
            self.current_mgr.flush_persist()

        return results

    def _maybe_rebuild_split_summaries(self) -> None:
        """Rebuild this dataset's summaries_chroma into :u/:a entry pairs.

        Runs right after ingest when INGEST_PARAMS["use_split_summary"] is True, so the
        artifacts on disk match what the retriever is configured to look for
        (split_single_entry_raw=False). Without this step retrieval would query
        {sid}:u / {sid}:a entries that the Ingestor never writes, and the whole
        provenance channel would silently drop to zero candidates.

        rebuild_artifact() is idempotent — it skips an artifact dir that already has a
        summaries_chroma_bak backup — so reruns and resumed runs are safe.
        """
        if not USE_SPLIT_SUMMARY:
            return
        if self.current_mgr is None:
            return

        artifact_dir = Path(self.current_mgr.ART)
        try:
            from experiment.longmem.rebuild_split_summaries import (
                SCRIPT_DATA_DIR,
                get_compressor,
                rebuild_artifact,
            )

            if self._split_lookup is None:
                from KG.utils.raw_context_lookup import RawContextLookup

                print(f"[SPLIT] Loading raw context from {SCRIPT_DATA_DIR} ...")
                self._split_lookup = RawContextLookup(SCRIPT_DATA_DIR)
                self._split_lookup._ensure_loaded()
            if self._split_compressor is None:
                print("[SPLIT] Loading llmlingua compressor ...")
                self._split_compressor = get_compressor()

            result = rebuild_artifact(
                artifact_dir, self._split_lookup, self._split_compressor, False
            )
            status = result.get("status")
            print(f"[SPLIT] {artifact_dir.name}: {status}"
                  + (f" — {result.get('reason')}" if result.get("reason") else ""))
            if status not in ("ok", "skip"):
                print(f"[SPLIT][WARN] unexpected rebuild status for {artifact_dir}: {result}")
        except Exception as exc:
            # Do not kill the run: report loudly and let the caller decide. Retrieval
            # will fall back to the direct-vector channel only.
            print(f"[SPLIT][ERROR] split-summary rebuild failed for {artifact_dir}: {exc}")
            traceback.print_exc()

    def _build_context(self, question: str, config: DatasetConfig, query_time: str | None = None) -> str:
        retrieval_params = {
            "ent_topk": config.ent_topk,
            "rel_topk": config.rel_topk,
            "ent_threshold": config.ent_threshold,
            "rel_threshold": config.rel_threshold,
            "filter_ent_topk": config.filter_ent_topk,
            "filter_rel_topk": config.filter_rel_topk,
            "filter_ent_threshold": config.filter_ent_threshold,
            "filter_rel_threshold": config.filter_rel_threshold,
            "summary_topk_per_item": config.summary_topk_per_item,
            "summary_vec_threshold": config.summary_vec_threshold,
        }
        return self.qa_stage.build_context(
            self.current_retriever,
            question=question,
            retrieval_params=retrieval_params,
            query_time=query_time,
        )

    def _ask_llm(self, question: str, context: str, question_date: str | None = None) -> str:
        return self.qa_stage.ask_llm(
            llm=self.llm,
            question=question,
            context=context,
            question_date=question_date,
        )

    def _maybe_refine_with_grep_agent(
        self,
        *,
        question: str,
        context: str,
        config: DatasetConfig,
        question_date: str | None,
    ) -> str:
        from experiment.longmem.agent_filter.harness import maybe_refine_context

        return maybe_refine_context(
            question=question,
            context=context,
            csv_path=config.csv_path,
            llm=self.llm,
            question_date=question_date,
            log_dir=self.current_log_dir,
            artifact_dir=getattr(config, "artifacts_dir", None),
        )


    def _answer_questions(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        output_path: Path,
    ) -> pd.DataFrame:
        """
        Process question and generate answers using base retrieval method.

        Returns a DataFrame containing ONE row with columns:
        - question: The question asked
        - question_date: Date when question was asked (if available)
        - Retrieved_Context: Context retrieved from KG
        - Generated_Answer: Answer generated by LLM
        - answer: Original answer from input (if exists)
        """
        if config.question_column not in df.columns:
            raise ValueError(f"CSV missing '{config.question_column}' column")

        # Get the unique question (assumes all rows have same question)
        questions = df[config.question_column].dropna().unique()

        if len(questions) == 0:
            raise ValueError(f"No questions found in column '{config.question_column}'")

        if len(questions) > 1:
            print(f"\n[WARNING] Found {len(questions)} different questions, using first one")

        # Get question and its date
        q = str(questions[0]).strip()
        print(f"\n[QA] Processing question: {q[:80]}{'...' if len(q) > 80 else ''}")

        # Get question date if available
        question_date = None
        date_columns = ["question_date", "dialogue_datetime", "date", "timestamp"]
        for col in date_columns:
            if col in df.columns and not df[col].dropna().empty:
                question_date = str(df[col].dropna().iloc[0]).strip()
                break

        if question_date:
            print(f"[QA] Question Date: {question_date}")

        # Parse and rewrite temporal expressions ONCE
        rewritten_q = self.parse_and_rewrite_question(q, query_time=question_date)

        # Get original answer if it exists
        original_answer = ""
        if "answer" in df.columns and not df["answer"].dropna().empty:
            original_answer = str(df["answer"].dropna().iloc[0])

        # ========== BASE RETRIEVER ==========
        print(f"\n[QA] Running retriever...")
        ctx_base = self._build_context(rewritten_q, config, query_time=question_date)
        ctx_base = self._maybe_refine_with_grep_agent(
            question=rewritten_q,
            context=ctx_base,
            config=config,
            question_date=question_date,
        )
        ans_base = self._ask_llm(rewritten_q, ctx_base, question_date=question_date)
        print(f"[QA] Answer: {ans_base[:80]}{'...' if len(ans_base) > 80 else ''}")

        # Write retrieval_summary + drop_reasons for error analysis
        trace = getattr(self.current_retriever, "last_retrieval_trace", None) or {}
        if trace and self.current_log_dir:
            selected_evidence = trace.get("selected_evidence") or []
            summary_record = {
                "request_id": trace.get("request_id"),
                "question": q,
                "low_level_keywords": trace.get("low_level_keywords", []),
                "high_level_keywords": trace.get("high_level_keywords", []),
                "stop_reason": trace.get("stop_reason"),
                "branches": trace.get("branches", {}),
                "pass2_triggered": bool(trace.get("pass2_triggered", False)),
                "rewritten_query": trace.get("rewritten_query"),
                "conf_pass1": trace.get("conf_pass1"),
                "conf_pass2": trace.get("conf_pass2"),
                "conf_final": trace.get("conf_final"),
                "tau_confidence": trace.get("tau_confidence"),
                "pass1_entity_ids": trace.get("pass1_entity_ids", []),
                "pass2_entity_ids": trace.get("pass2_entity_ids", []),
                "pass1_relation_ids": trace.get("pass1_relation_ids", []),
                "pass2_relation_ids": trace.get("pass2_relation_ids", []),
                "final_entity_count": trace.get("final_entity_count", 0),
                "final_relationship_count": trace.get("final_relationship_count", 0),
                "final_entity_names": trace.get("final_entity_names", []),
                "final_relationship_names": trace.get("final_relationship_names", []),
                "selected_evidence_count": trace.get("selected_evidence_count", 0),
                "selected_evidence_ids": [item.get("summary_id") for item in selected_evidence if item.get("summary_id")],
                "selected_evidence_preview": [item.get("preview") for item in selected_evidence[:3]],
                "has_temporal_evidence": bool(trace.get("has_temporal_evidence", False)),
                "answer": ans_base,
            }
            append_analysis_record(self.current_log_dir, "retrieval_summary", summary_record)
            for drop_reason in derive_drop_reasons(summary_record):
                append_analysis_record(self.current_log_dir, "drop_reasons", drop_reason)

        result_df = single_result_frame(
            question=q,
            question_date=question_date,
            context=ctx_base,
            answer=ans_base,
            gold=original_answer,
        )
        write_csv_frame(result_df, output_path)
        print(f"[QA] Results saved to: {output_path}")

        # Update checkpoint stage after QA
        if config.resume:
            checkpoint = self._load_checkpoint(config)
            processed = set(str(s) for s in checkpoint.get("processed_session_ids", []))
            total_sessions = checkpoint.get("total_sessions", None)
            self._save_checkpoint(
                config,
                processed,
                total_sessions=total_sessions,
                stage="qa_complete",
            )

        return result_df

    def _setup_dataset(self, config: DatasetConfig):
        """Initialize VDB manager and pipeline components for a dataset"""
        self._ensure_runtime_components()

        # Determine artifacts directory
        if config.artifacts_dir is None:
            artifacts_dir = self.base_output_dir / f"artifacts_{config.name}"
        else:
            artifacts_dir = Path(config.artifacts_dir)

        ensure_dir(artifacts_dir)

        # Determine log directory (per-dataset)
        log_dir = self.base_output_dir / f"logs_{config.name}"
        ensure_dir(log_dir)
        self.current_log_dir = log_dir

        print(f"\n{'='*60}")
        print(f"DATASET: {config.name}")
        print(f"Artifacts: {artifacts_dir}")
        print(f"Logs: {log_dir}")
        print(f"{'='*60}")

        # Create new VDB manager for this dataset
        self.current_mgr = VDBManager(artifacts_dir)
        is_fresh = self.current_mgr.initialize()
        print(f"[INIT] VDB Manager initialized (fresh: {is_fresh})")

        # Create entity and relationship managers
        self.current_ent = EntityManager(
            embedder=self.embedder,
            mgr=self.current_mgr,
            provenance=Provenance,
            GLOBAL_CACHE=self.current_mgr.cache,
            processed_ent_map=self.current_mgr.cache["entities"],
            processed_ent_full_map=self.current_mgr.cache["entities_full"],
        )

        self.current_rel = RelationshipManager(
            embedder=self.embedder,
            mgr=self.current_mgr,
            provenance=Provenance,
            GLOBAL_CACHE=self.current_mgr.cache,
            processed_rel_map=self.current_mgr.cache["relationships"],
            processed_rel_full_map=self.current_mgr.cache["relationships_full"],
        )

        # Create per-dataset loggers (override module-level loggers)
        ingestor_jlog = make_module_jlog(
            name=f"KG.Ingestor.{config.name}",
            filename="kg_ingestor.jsonl",
            log_dir=str(log_dir),
        )
        retriever_jlog = make_module_jlog(
            name=f"KG.Retriever.{config.name}",
            filename="kg_retriever.jsonl",
            log_dir=str(log_dir),
        )

        # Create ingestor and retriever
        self.current_ingestor = Ingestor(
            llm=self.llm,
            graph=self.graph,
            mgr=self.current_mgr,
            ent_svc=self.current_ent,
            rel_svc=self.current_rel,
        )

        # ── Benchmark split: LongMem vs LoCoMo retrieval semantics ──────────────
        # The shared experiment_config.py sets split_single_entry_raw=True for the
        # LoCoMo rerank16 flow (one entry per summary_id, no :u/:a suffix, fed the
        # raw turn text). LongMem can additionally split each turn into :u (user raw)
        # and :a (assistant compressed) entries, which are built by the
        # _maybe_rebuild_split_summaries() step right after ingest.
        # Both sides are driven by the SAME flag so the artifact layout and the
        # retrieval config can never disagree: use_split_summary=True means the
        # rebuild runs AND retrieval looks for :u/:a; False means neither.
        _longmem_reranker_params = {
            **RERANKER_PARAMS,
            "split_single_entry_raw": not USE_SPLIT_SUMMARY,
        }
        self.current_retriever = Retriever(
            llm=self.llm,
            graph=self.graph,
            mgr=self.current_mgr,
            embed=self.embedder.embed,
            cache=self.current_mgr.cache,
            config=RetrieverConfig(**_longmem_reranker_params),
        )

        # Monkey-patch the _jlog functions to use dataset-specific loggers
        # This overrides the module-level _jlog defined at import time
        import KG.pipeline.ingestor as ingestor_module
        import KG.pipeline.retriever as retriever_module
        import KG.pipeline.ingest_steps.sync as sync_step_module
        import KG.graph.falkordb as falkordb_module
        import KG.pipeline.retrieval_steps.search as search_module
        import KG.pipeline.retrieval_steps.filtering as filtering_module
        import KG.pipeline.retrieval_steps.temporal as temporal_module
        import KG.pipeline.retrieval_steps.evidence as evidence_module
        ingestor_module._jlog = ingestor_jlog
        sync_step_module._jlog = ingestor_jlog  # sync events co-located in kg_ingestor.jsonl
        falkordb_module._jlog = make_module_jlog(
            name=f"KG.Graph.{config.name}",
            filename="kg_ingestor.jsonl",
            log_dir=str(log_dir),
        )
        retriever_module._jlog = retriever_jlog
        search_module._jlog = make_module_jlog(
            name=f"KG.Retrieval.Search.{config.name}",
            filename="kg_retrieval_search.jsonl",
            log_dir=str(log_dir),
        )
        filtering_module._jlog = make_module_jlog(
            name=f"KG.Retrieval.Filtering.{config.name}",
            filename="kg_retrieval_filtering.jsonl",
            log_dir=str(log_dir),
        )
        temporal_module._jlog = make_module_jlog(
            name=f"KG.Retrieval.Temporal.{config.name}",
            filename="kg_retrieval_temporal.jsonl",
            log_dir=str(log_dir),
        )
        evidence_module._jlog = make_module_jlog(
            name=f"KG.Retrieval.Evidence.{config.name}",
            filename="kg_retrieval_evidence.jsonl",
            log_dir=str(log_dir),
        )

        print(f"[INIT] Ingestor and Retriever initialized with per-dataset logs")

    def _teardown_dataset(self, config: DatasetConfig):
        """Clean up after processing dataset"""
        log_dir = getattr(self, "current_log_dir", None)

        # Persist VDB changes
        if self.current_mgr:
            print(f"\n[CLEANUP] Persisting VDB changes...")
            self.current_mgr.flush_persist()

            # Close ChromaDB clients to release SQLite file descriptors, then null refs
            mgr = self.current_mgr
            for vdb in (mgr._entities_vdb, mgr._relationships_vdb, mgr._summaries_vdb):
                if vdb is not None:
                    try:
                        vdb.close()
                    except Exception:
                        pass
            mgr._entities_vdb = None
            mgr._relationships_vdb = None
            mgr._summaries_vdb = None
            mgr._entities_bm25 = None
            CacheStore.clear(mgr.cache)

        # 然後才清掉圖
        if self.graph is not None:
            print(f"[CLEANUP] Clearing graph database...")
            self.graph.clear_all()

        # Note: Logger monkey-patches will be overwritten by next dataset's setup
        # Each dataset gets its own log directory, so no conflict

        # Reset current components and force GC
        self.current_mgr = None
        self.current_ingestor = None
        self.current_retriever = None
        self.current_ent = None
        self.current_rel = None
        if log_dir is not None:
            cleanup_retrieval_loggers(Path(log_dir))
        gc.collect()
        print(f"[CLEANUP] Memory released (gc.collect done)")

    def _checkpoint_path(self, config: DatasetConfig) -> Path:
        return shared_checkpoint_path(self.base_output_dir, config)

    def _output_path(self, config: DatasetConfig) -> Path:
        if config.output_path is None:
            return self.base_output_dir / f"{config.name}.csv"
        return Path(config.output_path)

    @staticmethod
    def _read_answer_data(output_path: Path) -> Dict:
        if not output_path.exists():
            raise FileNotFoundError(f"Output CSV not found: {output_path}")
        result_df = read_csv_frame(output_path)
        if result_df.empty:
            raise ValueError(f"Output CSV is empty: {output_path}")
        return result_df.iloc[0].to_dict()

    def _persist_correctness(self, output_path: Path, correctness: str) -> None:
        if not output_path.exists():
            return
        out_df = read_csv_frame(output_path)
        out_df["correctness"] = correctness
        write_csv_frame(out_df, output_path)

    def _load_checkpoint(self, config: DatasetConfig) -> Dict:
        return shared_load_checkpoint(self.base_output_dir, config)

    def _save_checkpoint(
        self,
        config: DatasetConfig,
        processed: Set[str],
        total_sessions: Optional[int] = None,
        stage: str = "ingest_in_progress",
    ):
        shared_save_checkpoint(
            self.base_output_dir,
            config,
            processed,
            total_sessions=total_sessions,
            stage=stage,
        )

    def process_dataset(
        self,
        config: DatasetConfig,
        *,
        run_ingest: bool = True,
        run_qa: bool = True,
    ) -> Dict:
        """
        Process a single dataset:
        1. Setup VDB manager
        2. Ingest sessions
        3. Answer questions with retriever
        4. Teardown (persist VDB, clear graph)
        """
        # Handle legacy skipped_by_watchdog checkpoints (written by older watchdog versions).
        # Reset stage so this run re-ingests from the saved breakpoint.
        checkpoint = self._load_checkpoint(config)
        if decision.should_reset_legacy_skipped_stage(checkpoint):
            processed = len(checkpoint.get("processed_session_ids", []))
            total = checkpoint.get("total_sessions", "?")
            print(
                f"\n[RESUME] Dataset {config.name} has legacy skipped_by_watchdog stage "
                f"(processed {processed}/{total} sessions) — resetting to resume."
            )
            self._append_stuck_history(config.name, processed, total)
            self._save_checkpoint(
                config,
                set(str(s) for s in checkpoint.get("processed_session_ids", [])),
                total_sessions=checkpoint.get("total_sessions"),
                stage="ingest_in_progress",
            )

        output_path = self._output_path(config)
        should_setup_runtime = run_ingest or run_qa
        df: Optional[pd.DataFrame] = None

        if should_setup_runtime:
            self._setup_dataset(config)

            # Load data
            print(f"\n[LOAD] Reading CSV: {config.csv_path}")
            df = read_csv_frame(Path(config.csv_path))
            print(f"[LOAD] Loaded {len(df)} rows")

        try:
            if not should_setup_runtime:
                answer_data = self._read_answer_data(output_path) if output_path.exists() else {}
                return {
                    "dataset": config.name,
                    "ingest_results": {},
                    "output_path": str(output_path),
                    "num_questions": 1 if answer_data else 0,
                    "artifacts_dir": str(self.base_output_dir / f"artifacts_{config.name}"),
                    "answer_data": answer_data,
                    "resume_skipped": True,
                }

            # If output exists, treat dataset as completed and skip ingestion/QA.
            if run_ingest and run_qa and decision.should_treat_output_as_complete(output_path):
                print(f"[RESUME] Dataset {config.name} already completed (output exists), skipping.")
                answer_data = self._read_answer_data(output_path)
                return {
                    "dataset": config.name,
                    "ingest_results": {},
                    "output_path": str(output_path),
                    "num_questions": 1,
                    "artifacts_dir": str(self.current_mgr.ART),
                    "answer_data": answer_data,
                    "resume_skipped": True,
                }

            ingest_results = {}
            if run_ingest:
                token_tracker.set_context(dataset=config.name, stage="ingest", log_dir=self.current_log_dir)
                if config.ingest_mode == "turn_pairs":
                    ingest_results = self._ingest_by_turn_pairs(df, config)
                elif config.ingest_mode == "session":
                    ingest_results = self._ingest_by_session(df, config)
                else:
                    raise ValueError(f"Unknown ingest_mode: {config.ingest_mode}")
                print(f"\n[INGEST] Completed! Processed {len(ingest_results)} sessions")
                self._maybe_rebuild_split_summaries()
            else:
                print(f"\n[INGEST] Skipped for dataset {config.name}")

            if run_qa and not run_ingest:
                print(f"[RESTORE] Restoring graph from cached artifacts for dataset {config.name}...")
                restore_graph_from_cache(self.graph, self.current_mgr.cache)

            if run_qa:
                # If output exists after ingestion (e.g., from a previous run), skip recompute.
                if run_ingest and decision.should_treat_output_as_complete(output_path):
                    print(f"[RESUME] QA already complete for {config.name} (output exists), skipping QA.")
                    answer_data = self._read_answer_data(output_path)
                    result_df = pd.DataFrame([answer_data])
                else:
                    token_tracker.set_context(dataset=config.name, stage="qa", log_dir=self.current_log_dir)
                    result_df = self._answer_questions(df, config, output_path)
            else:
                answer_data = self._read_answer_data(output_path) if output_path.exists() else {}
                result_df = pd.DataFrame([answer_data]) if answer_data else pd.DataFrame()

            # Extract answer data for merged CSVs (as dict for easy handling)
            answer_data = result_df.iloc[0].to_dict() if not result_df.empty else {}

            return {
                "dataset": config.name,
                "ingest_results": ingest_results,
                "output_path": str(output_path),
                "num_questions": len(result_df),
                "artifacts_dir": str(self.current_mgr.ART),
                "answer_data": answer_data,  # Store for merged CSV
                "resume_skipped": False,
            }
        except KeyboardInterrupt:
            # 按 Ctrl+C：印個訊息，直接往外拋，不做 teardown
            print(f"\n[INTERRUPT] Dataset {config.name} interrupted by user. Skipping teardown.")
            raise

        finally:
            # Always cleanup
            import sys as _sys
            exc_type, _, _ = _sys.exc_info()
            if should_setup_runtime and (exc_type is None or exc_type is not KeyboardInterrupt):
                self._teardown_dataset(config)

    # =====================================================================
    # Error analysis
    # =====================================================================

    def _emit_error_analysis_bundle(
        self,
        config: DatasetConfig,
        *,
        answer_data: dict,
        correctness: str,
    ) -> None:
        log_dir = self.base_output_dir / f"logs_{config.name}"
        ensure_dir(log_dir)

        correctness_float = coerce_float(correctness)
        question = str(answer_data.get("question", "")).strip()

        # Build summary from the retrieval_summary record already written to the log.
        # We read it back so the bundle has the same data the retriever recorded.
        retrieval_summary_path = log_dir / "error_analysis_retrieval_summary.jsonl"
        retrieval_record: dict = {}
        if retrieval_summary_path.exists():
            import json as _json
            try:
                lines = retrieval_summary_path.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        obj = _json.loads(line)
                        if str(obj.get("question", "")).strip() == question:
                            retrieval_record = obj
                            break
            except Exception:
                pass

        summary = {
            "question": question,
            "request_id": retrieval_record.get("request_id", ""),
            "stop_reason": retrieval_record.get("stop_reason"),
            "conf_final": retrieval_record.get("conf_final"),
            "tau_confidence": retrieval_record.get("tau_confidence"),
            "selected_evidence_count": int(retrieval_record.get("selected_evidence_count") or 0),
            "final_entity_count": int(retrieval_record.get("final_entity_count") or 0),
            "final_relationship_count": int(retrieval_record.get("final_relationship_count") or 0),
            "pass1_entity_ids": retrieval_record.get("pass1_entity_ids", []),
            "has_temporal_evidence": bool(retrieval_record.get("has_temporal_evidence", False)),
            "coverage_percent": None,
            "ingest_entities_added": 0,
            "selected_evidence_preview": retrieval_record.get("selected_evidence_preview", []),
            "gold_answer": str(answer_data.get("answer", "")),
            "model_answer": str(answer_data.get("Generated_Answer", "")),
        }

        top_miss = build_top_miss_snapshot(log_dir=log_dir, request_id=str(summary["request_id"]))
        anomaly_flags = derive_anomaly_flags(summary=summary, correctness=correctness_float)
        if anomaly_flags:
            append_analysis_record(
                log_dir,
                "anomaly_flags",
                {
                    "dataset": config.name,
                    "question": question,
                    "request_id": summary["request_id"],
                    "correctness": correctness_float,
                    "flags": anomaly_flags,
                },
            )

        if correctness_float is not None and correctness_float < 1:
            failure_type = derive_failure_type(summary=summary, correctness=correctness_float)
            bridge_label = build_bridge_label(summary=summary, correctness=correctness_float)
            verdict_record = {
                "scope": "question",
                "dataset": config.name,
                "question": question,
                "request_id": summary["request_id"],
                "failure_type": failure_type,
                "correctness": correctness_float,
                "stop_reason": summary["stop_reason"],
                "selected_evidence_count": summary["selected_evidence_count"],
                "top_miss_count": len(top_miss),
                "gold_answer": summary["gold_answer"],
                "model_answer": summary["model_answer"],
            }
            append_analysis_record(log_dir, "failure_verdict", verdict_record)
            append_analysis_record(
                log_dir,
                "top_miss",
                {
                    "dataset": config.name,
                    "question": question,
                    "request_id": summary["request_id"],
                    "candidates": top_miss,
                },
            )
            append_analysis_record(
                log_dir,
                "evidence_bridge",
                {
                    "dataset": config.name,
                    "question": question,
                    "request_id": summary["request_id"],
                    "correctness": correctness_float,
                    "bridge_label": bridge_label,
                    "failure_type": failure_type,
                    "selected_evidence_count": summary["selected_evidence_count"],
                    "selected_evidence_preview": summary["selected_evidence_preview"],
                },
            )

        digest = render_failure_digest(
            sample_index=0,
            ingest_records=[],
            failures=[{**verdict_record, "anomaly_flags": anomaly_flags, "top_miss": top_miss}]
            if correctness_float is not None and correctness_float < 1
            else [],
        )
        append_pretty_block(log_dir, "error_analysis_failure_digest.log", digest)

    # =====================================================================
    # Judge integration
    # =====================================================================

    def _judge_single(self, question: str, gold: str, gen: str) -> int:
        return self.judge_stage.judge_single(
            llm=self.llm,
            question=question,
            gold=gold,
            generated=gen,
        )

    # =====================================================================
    # Progress tracking
    # =====================================================================

    def _progress_path(self) -> Path:
        return shared_progress_path(self.base_output_dir)

    def _load_progress(self) -> pd.DataFrame:
        return shared_load_progress(self.base_output_dir)

    def _save_progress_row(self, dataset: str, status: str, correctness: str = "",
                           question: str = "", gold_answer: str = "", generated_answer: str = "",
                           upload: bool = True):
        row = {
            "dataset": dataset,
            "status": status,
            "correctness": correctness,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
        }
        shared_save_progress_row(
            self.base_output_dir,
            **row,
        )

        try:
            if self.upload_nocodb and upload:
                table_name = self.noco_table_name or self.base_output_dir.name
                self.upload_stage.upsert_progress_row(table_name=table_name, row=row)
        except Exception as exc:
            logger.warning("NocoDB upsert skipped: %s", exc)

    def _init_progress_rows(self, configs: List[DatasetConfig]):
        shared_init_progress_rows(self.base_output_dir, [cfg.name for cfg in configs])

    def _append_stuck_history(self, dataset: str, processed: int, total) -> None:
        shared_append_stuck_history(
            self.base_output_dir,
            dataset=dataset,
            processed=processed,
            total=total,
        )

    # =====================================================================
    # Main loop
    # =====================================================================

    def process_all(self, configs: List[DatasetConfig], run_judge: bool = True, stages: Optional[Set[str]] = None) -> List[Dict]:
        """Process multiple datasets sequentially and create one merged CSV.

        Args:
            configs: List of dataset configurations.
            run_judge: If True, run LLM-as-judge after each QA and record
                       correctness in progress.csv and the individual output CSV.
        """
        selected_stages = set(stages or {"ingest", "qa_eval", "judge", "upload"})
        run_ingest = "ingest" in selected_stages
        run_qa = "qa_eval" in selected_stages
        run_upload = "upload" in selected_stages
        run_judge = run_judge and "judge" in selected_stages

        results = []
        merged_csv_path = self.base_output_dir / "all_answers.csv"
        merged_done = set()
        if merged_csv_path.exists():
            try:
                existing = read_csv_frame(merged_csv_path)
                if "dataset" in existing.columns:
                    merged_done = set(existing["dataset"].astype(str).tolist())
            except Exception:
                merged_done = set()

        # Initialise progress tracker with not_started rows for new datasets
        self._init_progress_rows(configs)

        print(f"\n{'#'*60}")
        print(f"# Multi-Dataset Processing")
        print(f"# Total datasets: {len(configs)}")
        print(f"# Output directory: {self.base_output_dir}")
        print(f"# Progress tracker: {self._progress_path()}")
        print(f"# Stages: {', '.join(sorted(selected_stages))}")
        print(f"# Judge enabled: {run_judge}")
        print(f"# NocoDB upload: {self.upload_nocodb and run_upload}")
        print(f"{'#'*60}")

        for i, config in enumerate(configs, start=1):
            print(f"\n\n{'='*60}")
            print(f"Processing dataset {i}/{len(configs)}: {config.name}")
            print(f"{'='*60}")

            try:
                result = self.process_dataset(
                    config,
                    run_ingest=run_ingest,
                    run_qa=run_qa,
                )
                results.append(result)

                answer_data = result.get("answer_data", {})
                output_path = Path(result.get("output_path", self._output_path(config)))
                if not answer_data and output_path.exists():
                    answer_data = self._read_answer_data(output_path)
                    result["answer_data"] = answer_data

                progress_df = self._load_progress()
                progress_row = progress_df[progress_df["dataset"].astype(str) == config.name]
                progress_data = progress_row.iloc[0].to_dict() if not progress_row.empty else {}

                question = str(answer_data.get("question", ""))
                gold = str(answer_data.get("answer", ""))
                generated = str(answer_data.get("Generated_Answer", ""))

                # Run judge if requested and we have the necessary fields
                correctness = str(answer_data.get("correctness", "")).strip()
                already_judged = correctness in ("0", "1")
                if run_judge and question and gold and generated and not already_judged:
                    print(f"\n[JUDGE] Evaluating answer for dataset {config.name}...")
                    judge_log_dir = getattr(self, "current_log_dir", self.base_output_dir / f"logs_{config.name}")
                    ensure_dir(judge_log_dir)
                    token_tracker.set_context(dataset=config.name, stage="judge", log_dir=judge_log_dir)
                    correctness = str(self._judge_single(question, gold, generated))
                    print(f"[JUDGE] Result: {'correct' if correctness == '1' else 'incorrect'} ({correctness})")
                    self._persist_correctness(output_path, correctness)
                    self._emit_error_analysis_bundle(config, answer_data=answer_data, correctness=correctness)
                elif already_judged:
                    print(f"[JUDGE] Skipping dataset {config.name} (already judged: {correctness})")
                    try:
                        self._persist_correctness(output_path, correctness)
                    except Exception as e:
                        print(f"[JUDGE] Warning: could not update output CSV: {e}")
                    self._emit_error_analysis_bundle(config, answer_data=answer_data, correctness=correctness)
                elif "judge" in selected_stages:
                    print(f"[JUDGE] Skipping dataset {config.name} (missing question/gold/generated output).")

                # Update progress tracker
                if run_judge and correctness != "":
                    status = "judged"
                elif run_qa:
                    status = "qa_complete"
                elif run_ingest:
                    status = "ingest_complete"
                else:
                    status = str(progress_data.get("status", "") or ("qa_complete" if generated else "not_started"))

                if run_ingest or run_qa or run_judge:
                    self._save_progress_row(
                        dataset=config.name,
                        status=status,
                        correctness=correctness,
                        question=question,
                        gold_answer=gold,
                        generated_answer=generated,
                        upload=False,
                    )
                else:
                    correctness = str(progress_data.get("correctness", correctness))
                    question = str(progress_data.get("question", question))
                    gold = str(progress_data.get("gold_answer", gold))
                    generated = str(progress_data.get("generated_answer", generated))

                if answer_data and (run_qa or run_judge):
                    answer_row = answer_data.copy()
                    answer_row["dataset"] = config.name
                    if correctness != "":
                        answer_row["correctness"] = correctness
                    update_all_answers_csv(
                        self.base_output_dir,
                        [
                            {
                                "dataset": config.name,
                                "question": answer_row.get("question", ""),
                                "question_date": answer_row.get("question_date", ""),
                                "gold": answer_row.get("answer", ""),
                                "answer": answer_row.get("Generated_Answer", ""),
                                "correctness": answer_row.get("correctness", ""),
                                "context": answer_row.get("Retrieved_Context", ""),
                            }
                        ],
                    )
                    merged_done.add(config.name)

                if run_upload and self.upload_nocodb:
                    try:
                        table_name = self.noco_table_name or self.base_output_dir.name
                        print(f"[UPLOAD] Upserting dataset {config.name} into NocoDB table '{table_name}'")
                        self.upload_stage.upsert_progress_row(
                            table_name=table_name,
                            row={
                                "dataset": config.name,
                                "status": status,
                                "correctness": correctness,
                                "question": question,
                                "gold_answer": gold,
                                "generated_answer": generated,
                            },
                        )
                    except Exception as exc:
                        logger.warning("NocoDB upsert skipped: %s", exc)

                print(f"\n✅ Dataset {config.name} completed successfully!")

            except Exception as e:
                print(f"\n❌ Dataset {config.name} failed with error: {e}")
                traceback.print_exc()
                results.append({
                    "dataset": config.name,
                    "error": str(e),
                })
                self._save_progress_row(dataset=config.name, status="error")

        # Summary
        print(f"\n\n{'#'*60}")
        print(f"# Processing Complete!")
        print(f"{'#'*60}")

        successful = [r for r in results if "error" not in r]

        for i, res in enumerate(results, start=1):
            if "error" in res:
                print(f"  [{i}] {res['dataset']}: FAILED - {res['error']}")
            else:
                print(f"  [{i}] {res['dataset']}: SUCCESS")
                print(f"      Output: {res['output_path']}")

        if successful:
            print(f"\n📊 Merged CSV: {merged_csv_path}")
            print(f"📋 Progress:   {self._progress_path()}")
            print(f"   Total: {len(successful)} datasets successfully processed")
            if run_upload and self.upload_nocodb:
                try:
                    table_name = self.noco_table_name or self.base_output_dir.name
                    print(f"[UPLOAD] Upserting accuracy summary into NocoDB table '{table_name}'")
                    self.upload_stage.upsert_accuracy_summary(
                        table_name=table_name,
                        progress_df=self._load_progress(),
                    )
                except Exception as exc:
                    logger.warning("NocoDB accuracy summary upsert skipped: %s", exc)

        return results
    
