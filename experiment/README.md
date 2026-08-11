# Experiment Guide

GRACE-Mem provides end-to-end runners for LongMemEval and LoCoMo. Both use the
same core ingest/retrieval packages and the shared settings in
[`experiment_config.py`](experiment_config.py), while keeping benchmark data,
artifacts, graph state, logs, and evaluation output isolated.

Complete the root [Quick Start](../README.md#quick-start) before using this guide.
The benchmark datasets are not included in the repository.

## Execution Model

Both runners expose the same ordered stage vocabulary:

| Stage | Responsibility |
|---|---|
| `ingest` | Build summaries, entity/relationship indexes, and graph state |
| `qa_eval` | Retrieve evidence and generate answers |
| `judge` | Grade generated answers against benchmark gold answers |

The default is `ingest qa_eval judge`. Use `--stage` to select an explicit
subset or `--no-judge` to omit judge behavior. When `--artifact-dir` is
provided, the runners switch to retrieval-only behavior and reuse existing
artifacts instead of ingesting again.

## Data Layout

### LoCoMo

Place the official conversation/QA file under `experiment/locomo/data/`:

| File | Required | Purpose |
|---|---|---|
| `locomo10.json` | yes | Primary LoCoMo conversations and questions; `locomo.json` is also accepted |
| `locomo_by_session.jsonl` | no | Session-oriented input; derived automatically when absent |

For `--dataset locomo-plus`, the default filenames are
`unified_input_samples_v2.json` and `locomo_plus_by_session.jsonl`. Override
dataset paths with `--dataset-json` and `--sessions-jsonl`.

### LongMemEval

LongMem expects one preprocessed CSV per question under
`experiment/longmem/script_data/<category>/`. The raw LongMemEval release is not
read directly, and this repository does not include a converter.

| Column | Required | Purpose |
|---|---|---|
| `session_id` | yes | Groups turns into a session |
| `turn_index` | yes | Orders turns within the session |
| `role` | yes | Identifies `user` and `assistant` rows |
| `content` | yes | Contains the turn text |
| `dialogue_datetime` | yes | Anchors relative-time normalization |
| `question` | yes | Question for this CSV; configurable with `question_column` |
| `answer` | no | Gold answer used by the judge |
| `question_date` | no | Query time; falls back to available dialogue/date fields |

Supported category directories include `single_session_user`,
`single_session_assistant`, `single_session_preference`, `multi_session`,
`temporal_reasoning`, and `knowledge_update`.

## LongMemEval

Entrypoint: [`longmem/watchdog.py`](longmem/watchdog.py)

### Commands

Full category run:

```bash
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run \
  --type single_session_user
```

Select dataset IDs, lexical indexes, or index ranges:

```bash
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run \
  --type temporal_reasoning \
  --dataset-id 2ebe6c92,0,3-5
```

Run only ingest and answer generation:

```bash
uv run python experiment/longmem/watchdog.py \
  --run-tag my-run \
  --type temporal_reasoning \
  --stage ingest qa_eval
```

Reuse an existing artifact run:

```bash
uv run python experiment/longmem/watchdog.py \
  --run-tag my-rerun \
  --type temporal_reasoning \
  --artifact-dir experiment/longmem/output/my-ingest-run \
  --output-root experiment/longmem/output/my-rerun
```

### Selection and Paths

| Option | Meaning |
|---|---|
| `--type TYPE [TYPE ...]` | One or more category directories |
| `--data-folder` | Data root; defaults to `experiment/longmem/script_data/` |
| `--file-pattern` | Question CSV glob; defaults to `*.csv` |
| `--dataset-id` | Comma-separated IDs/indexes/ranges such as `abc123,0,3-5` |
| `--num N` | Limit the resolved dataset list to the first N entries |
| `--child --child-file PATH` | Select datasets from a child manifest |
| `--artifact-dir` | Root containing reusable `artifacts_<dataset>/` directories |
| `--force` | Reprocess work that completion checks would otherwise skip |

In batch mode, the watchdog launches `run_batch.py`, which delegates each
question CSV to `MultiDatasetProcessor`. It records completion state and restarts
incomplete work up to `--max-restarts`. In retrieval-only mode it runs
`LongMemRerun` in-process, restores graph state from the artifact cache, and
closes dataset-local vector clients after every question.

Default output:

```text
experiment/longmem/output/<run-tag>/
  run_metadata.json
  _watchdog/
  <category>/
    artifacts_<dataset>/
    logs_<dataset>/
    checkpoint_<dataset>.json
    <dataset>.csv
    progress.csv
```

## LoCoMo

Entrypoint: [`locomo/pipeline.py`](locomo/pipeline.py)

`--sample-ids` is required for normal runs.

### Commands

Full run:

```bash
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-9 \
  --run-tag my-run
```

Reuse artifacts from an existing LoCoMo run:

```bash
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-9 \
  --run-tag my-rerun \
  --artifact-dir experiment/locomo/output/standard/my-ingest-run
```

Run selected stages or include adversarial questions:

```bash
uv run python experiment/locomo/pipeline.py \
  --dataset locomo \
  --sample-ids 0-3 \
  --run-tag my-run \
  --stage qa_eval judge \
  --adv
```

### Selection and Paths

| Option | Meaning |
|---|---|
| `--dataset {locomo,locomo-plus}` | Select the dataset adapter |
| `--sample-ids` | Sample selector such as `0,2,5-7` |
| `--chunk-turns` | Turns per ingest chunk; `0` keeps one whole-session summary |
| `--artifact-dir` | Existing LoCoMo run root used instead of ingest |
| `--adaptive --tau` | Enable confidence-triggered adaptive re-search |
| `--adv` | Include adversarial questions; excluded by default |
| `--out-root` | Output base; defaults to `experiment/locomo/output` |
| `--retrieval-mode` | Run gold/replay retrieval ablations |

The orchestrator launches one clean subprocess per sample. A worker ingests
session chunks, answers that sample's questions, runs the judge, and writes its
artifacts/logs below `sample_<id>/`. Process isolation prevents graph and model
state from leaking between samples.

Default output:

```text
experiment/locomo/output/
  standard/<run-tag>/        # --dataset locomo
    sample_<id>/
      artifacts/
      logs/
      *_eval*.csv
      *_judge*.csv
    _correctness_aggregate.json
    _judge_merged.csv
  plus/<run-tag>/            # --dataset locomo-plus
```

Judged LoCoMo runs aggregate automatically. To rebuild aggregate output:

```bash
uv run python experiment/locomo/aggregate.py \
  --dataset locomo \
  --root experiment/locomo/output/standard/<run-tag>
```

Aggregation reports correctness, F1, BLEU-1, and category breakdowns. It excludes
adversarial questions unless `--include-adversarial` is supplied.

For paper scoring, use the shared post-hoc judge after answer generation:

```bash
uv run python experiment/judge.py locomo <run-tag> --samples 0-9
uv run python experiment/judge.py longmem <run-tag>
uv run python experiment/score.py <run-tag>
```

The exact carry/rejudge rule, LongMemEval abstention handling, result columns,
and aggregate files are defined in the [evaluation protocol](../EVALUATION.md).
The same document defines the gold-evidence oracle and its context-window rule.

## Shared Configuration

Edit [`experiment_config.py`](experiment_config.py) for experiment-wide defaults:

| Group | Main responsibility |
|---|---|
| `REPRODUCIBILITY_PARAMS` | Seed and deterministic execution settings |
| `INGEST_PARAMS` | Previous context, entity matching, chunking, and split summaries |
| `RETRIEVAL_PARAMS` | Search/filter top-k values and similarity thresholds |
| `RERANKER_PARAMS` | Graph filtering, reranking, evidence selection, and SA-RAG |
| `GREP_AGENT_PARAMS` | Optional post-retrieval evidence-refinement behavior |

CLI overrides exist for run-specific selectors and a small number of ingest or
adaptive-retrieval settings. Do not duplicate shared defaults in benchmark code.

### Artifact Compatibility

Artifacts encode the ingest layout used to create them. Keep these settings
aligned when running retrieval-only experiments:

- **LoCoMo `chunk_turns`**: default `8`. A session is split into consecutive
  windows, and each window receives its own summary ID. Reuse artifacts with the
  same value; `0` reproduces the older whole-session layout.
- **LongMem `use_split_summary`**: default `True`. After ingest, LongMem rebuilds
  each summary entry into `:u` (user raw) and `:a` (assistant compressed)
  candidates. The same setting configures retrieval, so do not override
  `split_single_entry_raw` independently.
- **LoCoMo split summaries**: LoCoMo always uses one summary entry per ingest
  chunk and ignores the LongMem-only `use_split_summary` setting.

Changing candidate-pool granularity can invalidate previously tuned
`summary_direct_vector_topn` and `summary_rerank_topk` values; re-sweep evidence
parameters when changing chunk/layout behavior.

## Offline Analysis

Offline diagnostics are separated from benchmark execution and live under each
benchmark's `analysis` package. The legacy script paths remain compatibility
wrappers, but new automation should use the canonical modules below.

| Purpose | Command |
|---|---|
| LoCoMo gold recall | `python -m experiment.locomo.analysis.gold_recall --help` |
| LoCoMo dataset statistics | `python -m experiment.locomo.analysis.dataset --help` |
| LoCoMo turn filtering | `python -m experiment.locomo.analysis.turn_filter --help` |
| LongMem gold recall | `python -m experiment.longmem.analysis.gold_recall --help` |
| LongMem judge flips | `python -m experiment.longmem.analysis.judge_flips --help` |
| LongMem summary scores | `python -m experiment.longmem.analysis.summary_scores --help` |
| LongMem fact replay | `python -m experiment.longmem.analysis.fact_replay --help` |

Agent Filter reachability, resampling, and tribunal studies are LongMem analysis
modules prefixed with `agent_filter_`. The trace viewer and live smoke probe live
under `tools/` because they inspect artifacts or require configured services.

## Recovery and Diagnostics

- Inspect available flags directly with `uv run python <entrypoint> --help`.
- Check the first dataset/sample logs before starting a large run.
- LongMem watchdog status is written under `<run-root>/_watchdog/`.
- Reusing artifacts avoids ingest cost but still requires matching data and
  artifact layouts.
- `--no-judge` is useful when judge credentials are unavailable; answer CSVs are
  still produced by `qa_eval`.
- If a process was interrupted, use the normal command again first. Completion
  checks and checkpoints are designed to resume work without rebuilding complete
  datasets.
- Use `refresh_system.py` only when intentionally clearing active graph/model
  state between manual experiments; it is not a substitute for matching artifact
  configuration.
