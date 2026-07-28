# Experiment guide

This document covers the full flow (ingest → retrieve → answer → judge) for both benchmarks:

1. **LongMemEval** (`longmem/watchdog.py`)
2. **LoCoMo** (`locomo/pipeline.py`)

> Prerequisite: complete the Setup in the root [readme.md](../readme.md) first (start FalkorDB, set the LLM endpoint in `.env`, run `setup_env.sh`). All ingest / retrieve / reranker parameters live in [experiment_config.py](experiment_config.py) (single source of truth).

---

## Part A: LongMemEval (watchdog)

### Quickstart

Run from the repo root:

```bash
# Full flow: ingest → retrieve → answer → judge
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run \
  --type single_session_user

# Only specific dataset ids / index range
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run \
  --type temporal_reasoning \
  --dataset-id 2ebe6c92,0,3-5

# Only specific stages (e.g. ingest + qa_eval)
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run \
  --type temporal_reasoning \
  --stage ingest qa_eval

# Retrieval-only: reuse existing artifacts, skip ingest
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run-rerun \
  --type temporal_reasoning \
  --artifact-dir ./experiment/longmem/output/my-ingest-run \
  --output-root ./experiment/longmem/output/my-run-rerun
```

### How it runs (simplified)

- `watchdog.py` checks whether each dataset's output is complete
- If incomplete, it runs `run_batch.py` → `processor.py`
- Default data root is `./experiment/longmem/script_data/`; it actually reads `./experiment/longmem/script_data/<type>/`
- Default output goes to `experiment/longmem/output/<run_tag>/<category>/`
- For each dataset, in order:
  - read CSV → ingest into KG/VDB
  - retrieve and answer with the LLM
  - judge
  - write the answer CSV + checkpoint
- On failure or incomplete state, the watchdog sleeps and reruns until done or a limit is hit

Common parameters:

- `--run-tag`: output folder name for this run → `experiment/longmem/output/<run_tag>/<type>/`
- `--type`: LongMem question category, e.g. `single_session_user`, `single_session_assistant`, `single_session_preference`, `multi_session`, `temporal_reasoning`
- `--data-folder`: LongMem data root, default `./experiment/longmem/script_data/`; the actual path is composed as `<data-folder>/<type>`
- `--file-pattern`: glob for data discovery, default `*.csv`
- `--child` / `--child-file`: use manifest mode to select data; `--type` becomes a category filter in this mode
- `--dataset-id`: supports dataset id, in-folder lexical index, and ranges, e.g. `abc123,0,3-5`
- `--num`: limit this run to the first N resolved datasets
- `--stage ingest qa_eval judge`: which stages to run; defaults to the full flow
- `--no-judge`: skip judge; equivalent to removing `judge` from the stage list
- `--artifact-dir`: switch to retrieval-only / rerun mode, reusing existing `artifacts_<dataset>/`
- `--force`: rerun even if the context looks usable

---

## Part B: LoCoMo (pipeline.py)

### Quickstart

Sample ids are required:

```bash
# Standard LoCoMo full flow (ingest → retrieve → answer → judge)
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-9 \
  --run-tag your_exp_name

# Retrieval-only: reuse an existing run's artifacts, skip ingest
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-9 \
  --run-tag <exp_name> \
  --artifact-dir <artifact_dir>

# Only specific stages (e.g. skip ingest, do qa + judge)
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-9 \
  --run-tag <exp_name> \
  --stage qa_eval judge

# Include adversarial questions
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-3 \
  --run-tag <exp_name> \
  --adv
```

Module layout:

- `experiment/locomo/pipeline.py`: orchestrator and worker entry point
- `experiment/locomo/workers.py`: per-sample worker flow
- `experiment/locomo/snapshot.py`: snapshot build / restore
- `experiment/locomo/stages/ingest.py`: LoCoMo ingest stage
- `experiment/locomo/stages/qa_eval.py`: RAG eval / answer stage
- `experiment/locomo/stages/judge.py`: judge stage
- `experiment/locomo/prompts/`: judge / open-domain prompt templates
- `experiment/locomo/aggregate.py`: run-level correctness aggregation
- `experiment/locomo/utils/`: shared io / log / graph helpers

Common parameters:

- `--dataset {locomo,locomo-plus}`: default `locomo`
- `--sessions-jsonl`: default `experiment/locomo/data/locomo_by_session.jsonl`
- `--dataset-json`: for `locomo`, default `experiment/locomo/data/locomo10.json`
- `--out-root`: default base `experiment/locomo/output`
- `--run-tag`: defaults to a timestamp
- `--artifact-dir`: reuse an existing run's `sample_<id>/artifacts/`, skip re-ingest
- `--adv`: by default adversarial questions are **not** processed; add this to include them in eval / judge / aggregate
- `--stage ingest qa_eval judge`: which stages to run; defaults to the full flow
- `--retrieval-mode`: supports `gold_summary_only`, `gold_raw_text_only`, `replay_summary_raw_text_from_run`
- `--replay-run-dir`: source run for `--retrieval-mode=replay_summary_raw_text_from_run`
- `--baseline-run-dir`: baseline summary source run for `--retrieval-mode=gold_summary_only`
- `--no-judge`: run only ingest + eval, no judge / aggregate
- `--adaptive` / `--tau`: enable adaptive re-search retrieval

For `locomo-plus`:

- `--dataset-json` defaults to `experiment/locomo/data/unified_input_samples_v2.json`
- `--sessions-jsonl` defaults to `experiment/locomo/data/locomo_plus_by_session.jsonl`
- If no ready-made by-session JSONL exists, the program parses sessions from `input_prompt` before ingesting

### How it runs (simplified)

`pipeline.py` is an "orchestrator + subprocess" design:

- **Orchestrator**: parse `--sample-ids` → launch one clean Python subprocess per sample → back up artifacts/logs on success, then call `refresh_system.py` to clean the environment
- **Worker (per sample)**:
  1. **Ingest**: read JSONL → convert to single-turn data → ingest into KG/VDB
  2. **Eval**: load LoCoMo questions → RAG answer → write eval CSV
  3. **Judge**: call the LLM judge → produce correctness stats and a judge CSV

Default output paths:

- `locomo` → `experiment/locomo/output/standard/<run_tag>/`
- `locomo-plus` → `experiment/locomo/output/plus/<run_tag>/`

### After all samples finish: aggregate correctness

Once judging finishes, results are aggregated automatically, producing:

- `experiment/locomo/output/<variant>/<run_tag>/_correctness_aggregate.json`
- `experiment/locomo/output/<variant>/<run_tag>/_judge_merged.csv`

To rerun / debug aggregation on its own:

```bash
uv run python experiment/locomo/aggregate.py \
  --dataset locomo \
  --run-root experiment/locomo/output/standard/<run_tag>
```

- Reads `_judge_merged.csv` and `_correctness_aggregate.json` under the run root first
- Produces overall and per-sample correctness / f1 / bleu1, plus category breakdowns
- Skips adversarial questions by default; add `--adv` on the pipeline to include them

---

## Where to change parameters (ingest / retrieve / reranker)

**Single entry point**: edit only [experiment_config.py](experiment_config.py). Both the LongMem and LoCoMo pipelines read `INGEST_PARAMS` / `RETRIEVAL_PARAMS` / `RERANKER_PARAMS` from here; change one place and it applies everywhere.

```python
# experiment/experiment_config.py
REPRODUCIBILITY_PARAMS = dict(seed=42, deterministic=True)
INGEST_PARAMS   = dict(ingest_mode="turn_pairs", prev_k=2, ...)
RETRIEVAL_PARAMS = dict(ent_topk=20, filter_ent_topk=15, ...)
RERANKER_PARAMS  = dict(use_reranker=True, reranker_topk=10, ...)
```

- **Ingest params** (`prev_k`, `entity_sim_topk`, `entity_sim_threshold`): edit `INGEST_PARAMS`; for LoCoMo you can also override on `pipeline.py` with `--prev-k`, `--entity-sim-topk`, `--entity-sim-threshold`.
- **Retrieval params** (`ent_topk`, `rel_topk`, `*_threshold`, `filter_*`, `summary_*`): edit `RETRIEVAL_PARAMS`; read by `rag_answer()` in `qa_eval.py`.
- **Reranker params** (`use_reranker`, `reranker_threshold`, `reranker_topk`): edit `RERANKER_PARAMS`; applied when building the retriever.

`REPRODUCIBILITY_PARAMS` sets seeds/determinism across Python / NumPy / PyTorch / CUDA and any supported LLM seed. This improves reproducibility but does not guarantee bit-level identical output under multi-threading, GPU-nondeterministic ops, or backend server-side batching.

---

## FAQ

- **Why does the LongMemEval watchdog keep rerunning?** Some dataset is not complete, or the checkpoint hasn't reached `qa_complete`.
- **Why is every LoCoMo sample a new subprocess?** To avoid KG/LLM state leaking between samples, so each sample runs in a clean environment.

---

## Before you start

- Confirm FalkorDB is up: open `http://localhost:3000/` in a browser; it should connect.
- If using a local LM Studio, set a large enough context length (16000+ for larger models) and **reload the model** after changing it.
- To switch to a cloud API (e.g. gpt-4o-mini), uncomment `LLM_API`/`MODEL_NAME`/`OPENAI_API_KEY` in `.env` and fill in your key.
- If a run didn't finish cleanly and you need to rerun, first clear the graph data and existing VDB: run `refresh_system.py`.
- After starting, check the first session's log to confirm ingest / VDB / graph writes succeeded — **don't wait until the whole batch finishes** to check.
