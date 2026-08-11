# GRACE-Mem

GRACE-Mem is a research framework for graph-based long-term conversational
memory, with reproducible pipelines for LoCoMo and LongMemEval experiments.

[Overview](#overview) | [Architecture](#architecture) | [Quick Start](#quick-start) |
[Benchmarks](#benchmarks) | [Agent Filter](#agent-filter) |
[Documentation](#documentation)

## Overview

GRACE-Mem addresses long-context conversational memory: given many dialogue
turns across sessions, it builds a persistent memory that can later answer
questions with evidence rather than relying only on the prompt window. The
repository is designed for research experiments, benchmark reproduction, and
retrieval diagnostics rather than as a hosted memory service.

The memory contains compressed dialogue records, extracted entities,
relationships, temporal information, provenance links back to source turns, and
vector indexes for semantic retrieval. The graph representation is useful
because conversational facts are often distributed across turns: a later
question may require resolving entities, following relationships, and grounding
the final context in the turns where the evidence appeared.

Retrieval combines query analysis, dense and lexical candidate discovery,
relationship retrieval, graph expansion, filtering, reranking, and
provenance-backed evidence selection. The selected evidence is then used for
answer generation and optional benchmark judging.

Agent Filter is an optional post-retrieval evidence-refinement layer. It starts
from an existing retrieved context and uses GREP, READ, and VECTOR tools over the
question corpus to verify seed evidence, recover missed facts, and fall back to
the original context if refinement fails.

## Architecture

![GRACE-Mem architecture](docs/architecture/flow.png)

The repository has three layers:

- **Core GRACE-Mem (`KG/`)**: ingestion, retrieval, graph synchronization,
  vector and lexical storage, LLM calls, provenance handling, and runtime
  helpers.
- **Benchmark layer (`experiment/`)**: LoCoMo and LongMemEval orchestration,
  shared judging/scoring utilities, artifact management, and run metadata.
- **Optional analysis and refinement**: Agent Filter, offline retrieval
  diagnostics, gold-recall analysis, trace inspection, and replay tools.

Dependency direction is intentionally one way:

```text
benchmarks / analysis / tools
        -> experiment orchestration
        -> KG pipeline facades
        -> KG services, storage, graph, LLM utilities
```

The core `KG/` package does not depend on benchmark-specific LoCoMo or LongMem
code.

## Workflow

### Ingestion

The benchmark and core pipeline code support the following memory-building flow:

```text
dialogue turns
  -> temporal normalization
  -> dialogue compression / summary representation
  -> entity extraction
  -> entity reconciliation
  -> relationship extraction
  -> vector and BM25 storage
  -> FalkorDB graph synchronization
```

Each stored item keeps provenance so retrieved answers can be traced back to
source dialogue context. Benchmark runs write self-contained artifact sets that
can be reused for retrieval-only reruns when the artifact layout matches the
configuration used to create them.

### Retrieval

The retrieval path is:

```text
question + query time
  -> query analysis
  -> hybrid entity retrieval
  -> relationship retrieval
  -> graph expansion
  -> compressed-record evidence retrieval
  -> filtering and reranking
  -> optional Agent Filter
  -> answer generation
```

The retriever combines graph-linked evidence with direct compressed-record
retrieval. Reranking and evidence selection are configured in
[`experiment/experiment_config.py`](experiment/experiment_config.py) for
benchmark runs.

## Key Features

- Graph-based conversational memory over entities, relationships, summaries, and
  provenance.
- Hybrid dense and BM25 candidate retrieval for entities and stored evidence.
- FalkorDB-backed graph expansion with Chroma-backed vector stores.
- Two-stage filtering/reranking and provenance-aware evidence construction.
- Optional Agent Filter for post-retrieval evidence verification and recovery.
- LoCoMo and LongMemEval benchmark runners with staged ingest, QA, and judge
  execution.
- Reusable artifact directories, run metadata, checkpoints, and offline
  diagnostic tools.

## Installation

### Requirements

| Requirement | Purpose |
|---|---|
| Python `>=3.10,<3.14` | Project runtime |
| [uv](https://docs.astral.sh/uv/) | Dependency and virtual environment management |
| Docker | Local FalkorDB via `docker-compose.yml` |
| OpenAI-compatible endpoint | Ingestion, retrieval-time LLM calls, answering, and judging |
| Local model storage | Qwen embedding and reranker weights downloaded by setup |

### Setup

```bash
git clone https://github.com/JaneDoe-0728/GRACE-Mem.git
cd GRACE-Mem
cp .env.example .env
uv sync
```

Edit `.env` before running the system. If you want the repository to start
FalkorDB and download the local embedding/reranker models for you, run:

```bash
bash tools/setup_env.sh
```

The setup script runs `uv sync`, starts the primary `falkordb` container,
downloads `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B` into
`models/`, and verifies the expected files.

To manage FalkorDB manually:

```bash
docker compose up -d falkordb
docker compose logs -f falkordb
docker compose down
```

The bundled container uses Redis port `6379` and exposes the browser UI on
`http://localhost:3000`.

## Configuration

Configuration is split between `.env` for runtime endpoints and
`experiment/experiment_config.py` for benchmark parameters.

### LLM

Set these in `.env`:

| Variable | Purpose |
|---|---|
| `LLM_API` | OpenAI-compatible base URL used by the KG pipeline |
| `MODEL_NAME` | Model served by `LLM_API` |
| `JUDGE_LLM_API` | OpenAI-compatible base URL used by benchmark judging |
| `JUDGE_MODEL_NAME` | Judge model name |

The defaults in `.env.example` point to a local OpenAI-compatible server and are
placeholders until you run such a server.

### FalkorDB

The bundled Docker service works with the default values:

| Variable | Purpose |
|---|---|
| `NEO4J_URI` | FalkorDB Redis URI |
| `FALKORDB_PASSWORD` | Password used by the Docker service |
| `GRAPH_NAME` | Graph key/name used by the wrapper |

Despite the `NEO4J_*` variable names, the active graph adapter is FalkorDB.

### Embedding and Reranker

`KG/embeddings.py` loads the embedding model from
`models/embedding_models/qwen3-0.6b` when present and falls back to the
Hugging Face model id otherwise. The reranker uses
`models/reranker/qwen3-reranker-0.6b`.

Use this helper to download both:

```bash
uv run python tools/download_models.py
```

### Experiment Configuration

Benchmark defaults live in
[`experiment/experiment_config.py`](experiment/experiment_config.py):

- `REPRODUCIBILITY_PARAMS`: seed and deterministic execution controls.
- `INGEST_PARAMS`: ingest mode, entity matching, LoCoMo chunking, and LongMem
  split-summary behavior.
- `RETRIEVAL_PARAMS`: initial search and post-filter thresholds/top-k values.
- `RERANKER_PARAMS`: graph filtering, reranking, evidence selection, and
  spreading-activation settings.
- `GREP_AGENT_PARAMS`: Agent Filter behavior.

Use CLI flags for run selectors such as sample IDs, categories, stage selection,
artifact reuse, and output roots.

## Quick Start

After setup and `.env` configuration, this minimal example ingests one dialogue
turn and retrieves context for a question:

```python
from KG.pipeline.factory import build_pipeline

with build_pipeline() as runtime:
    runtime.ingestor.summarize_and_ingest_turn(
        session_id=1,
        message_id=1,
        user_text="I attended an AI workshop yesterday.",
        assistant_text="What did you learn there?",
        dialogue_datetime="2023/02/18 (Sat) 08:08",
    )

    context = runtime.retriever.build_kg_context(
        question="Which workshop did the user attend?",
        query_time="2023/02/18 (Sat) 08:08",
    )
    print(context)
```

Run it from the repository root with FalkorDB and your LLM endpoint available.
For benchmark execution, use the LoCoMo and LongMemEval entrypoints below.

## Usage

### Core API

`build_pipeline()` is the intended high-level API for direct use. It creates the
ingestor, retriever, graph client, vector-store manager, and LLM client, and it
closes owned resources when the context exits.

```python
from KG.pipeline.factory import build_pipeline

with build_pipeline() as runtime:
    ingestor = runtime.ingestor
    retriever = runtime.retriever
```

### Agent Filter

![Agent Filter flow](docs/architecture/agent_flow_v2.png)

Agent Filter consumes an existing benchmark run that already has a
`Retrieved_Context` column. It does not rerun graph/vector retrieval; it refines
the retrieved evidence with tool actions and writes a new run.

LongMemEval replay:

```bash
uv run python -m experiment.agent_filter.replay_run \
  --source-run <existing-run> \
  --run-tag <agent-filter-run> \
  --workers 4
```

LoCoMo replay:

```bash
uv run python -m experiment.agent_filter.locomo_replay \
  --source-run <existing-run> \
  --run-tag <agent-filter-run> \
  --chunk-turns 8 \
  --samples 0-9 \
  --workers 4 \
  --granularity turn
```

For LongMem VECTOR support, set `LONGMEM_ARTIFACT_ROOT` or pass
`--artifact-root`. LoCoMo finds the summary VDB under each sample artifact
directory. See [experiment/agent_filter/README.md](experiment/agent_filter/README.md)
for details.

## Benchmarks

GRACE-Mem keeps benchmark orchestration separate from the core `KG/` package.
Both benchmark runners use the same stage vocabulary:

```text
ingest -> qa_eval -> judge
```

Use `--stage` to select a subset, `--no-judge` to skip judging, and
`--artifact-dir` to reuse an existing artifact run for retrieval-only evaluation.
Benchmark datasets are not included in this repository.

### LongMemEval

LongMemEval expects one preprocessed CSV per question under
`experiment/longmem/script_data/<category>/`. The repository does not include a
raw LongMemEval-to-CSV converter.

Minimal category run:

```bash
uv run python experiment/longmem/pipeline/watchdog.py \
  --run-tag my-run \
  --type temporal_reasoning
```

Default output:

```text
experiment/longmem/output/<run-tag>/<category>/
```

Supported category directories and required CSV columns are documented in
[experiment/README.md](experiment/README.md#longmemeval).

### LoCoMo

LoCoMo expects the official data file under `experiment/locomo/data/`, usually
`locomo10.json` or `locomo.json`. Normal runs require explicit sample selection.

Minimal run:

```bash
uv run python experiment/locomo/pipeline/runner.py \
  --dataset locomo \
  --sample-ids 0-9 \
  --run-tag my-run
```

Default output:

```text
experiment/locomo/output/standard/<run-tag>/
```

For dataset layout, reruns, stage selection, aggregation, and scoring commands,
see [experiment/README.md](experiment/README.md).

## Analysis & Diagnostics

Offline diagnostics are research utilities for existing artifacts. They are not
required for the main runtime and generally avoid rerunning the full ingest or
answer-generation pipeline.

Useful entrypoints include:

| Purpose | Command |
|---|---|
| LoCoMo gold evidence recall | `python -m experiment.locomo.analysis.gold_recall --help` |
| LoCoMo dataset statistics | `python -m experiment.locomo.analysis.dataset --help` |
| LoCoMo aggregate outputs | `python -m experiment.locomo.analysis.aggregate --help` |
| LoCoMo turn filtering | `python -m experiment.locomo.analysis.turn_filter --help` |
| LongMem gold evidence recall | `python -m experiment.longmem.analysis.gold_recall --help` |
| LongMem judge flips | `python -m experiment.longmem.analysis.judge_flips --help` |
| LongMem summary scores | `python -m experiment.longmem.analysis.summary_scores --help` |
| LongMem fact replay | `python -m experiment.longmem.analysis.fact_replay --help` |

Agent Filter reachability, resampling, and tribunal studies live in
`experiment.longmem.analysis` with the `agent_filter_*` prefix. The trace viewer
and live smoke probe live under `tools/`.

## Repository Structure

```text
GRACE-Mem/
├── KG/                         # core ingestion, retrieval, graph, storage, LLM utilities
├── experiment/
│   ├── common/                 # shared evaluation, scoring, reproducibility helpers
│   ├── locomo/                 # LoCoMo runner, stages, artifacts, analysis
│   ├── longmem/                # LongMemEval runner, stages, artifacts, analysis
│   ├── agent_filter/           # post-retrieval evidence refinement
│   └── noco/                   # optional NocoDB upload helpers
├── docs/
│   └── architecture/           # README architecture figures
├── tools/                      # setup, model download, refresh, and audit helpers
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

Benchmark datasets, generated artifacts, model weights, logs, and local
development-only files are intentionally gitignored.

## Reproducibility

The benchmark layer centralizes run settings in
`experiment/experiment_config.py` and writes run metadata through
`experiment/common/run_metadata.py`. Benchmark runs preserve artifacts,
checkpoints, logs, answer CSVs, judge CSVs, and aggregate outputs under their run
directories.

Deterministic reproduction is limited by external dependencies: the configured
LLM endpoint, judge model, embedding/reranker versions, hardware, and benchmark
data preprocessing can all affect results. Reuse artifacts with matching ingest
layout and retrieval configuration when comparing reruns.

## Documentation

| Document | Purpose |
|---|---|
| [Experiment guide](experiment/README.md) | Data layout, benchmark commands, stages, artifact reuse, and scoring |
| [Evaluation protocol](EVALUATION.md) | Judge model behavior, voting rules, output columns, and scoring |
| [Agent Filter guide](experiment/agent_filter/README.md) | Evidence-refinement workflow, VECTOR setup, and trace output |
| [.env example](.env.example) | Runtime endpoint and graph configuration |

## Current Limitations

- Benchmark datasets and generated artifacts are not distributed with the
  repository.
- LongMemEval requires the documented preprocessed CSV layout.
- End-to-end results depend on external LLM and judge endpoints.
- No repository license file is currently present.
