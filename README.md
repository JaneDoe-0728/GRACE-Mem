<div align="center">

# GRACE-Mem

### Graph Retrieval with Agentic Corpus Evidence for Long-Term Conversational Memory

**Structured memory, graph-aware retrieval, traceable evidence.**

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-2ea44f)](LICENSE)
[![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![Docker](https://img.shields.io/badge/FalkorDB-Docker-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Benchmarks](https://img.shields.io/badge/benchmarks-LoCoMo%20%7C%20LongMemEval-555)](experiment/README.md)

[Quick Start](#quick-start) | [Architecture](#architecture) |
[Benchmarks](#benchmarks) | [Documentation](#documentation)

</div>

## Overview

GRACE-Mem is an open-source long-term conversational memory framework that
converts dialogue into structured, retrievable memory. It is research software
for memory experiments and retrieval diagnostics, not a hosted memory service.

Instead of treating memory as a flat vector store, GRACE-Mem combines a
knowledge graph, dense and lexical retrieval, temporal information, and source
provenance to recover evidence across conversations. It also provides
reproducibility-oriented LoCoMo and LongMemEval pipelines, optional agentic
evidence refinement, and offline retrieval diagnostics.

## Why GRACE-Mem?

<table>
<tr>
<th width="33%" align="left" valign="top">Graph-Structured Memory</th>
<th width="33%" align="left" valign="top">Evidence-Centric Retrieval</th>
<th width="33%" align="left" valign="top">Reproducible Evaluation</th>
</tr>
<tr>
<td width="33%" valign="top">Connect entities, relationships, and temporal facts across dialogue sessions instead of treating each memory as an isolated chunk.</td>
<td width="33%" valign="top">Combine vector search, BM25, graph expansion, reranking, and provenance-aware evidence reconstruction in one retrieval path.</td>
<td width="33%" valign="top">Run staged LoCoMo and LongMemEval experiments with pinned datasets, reusable artifacts, run metadata, shared judging, and offline diagnostics.</td>
</tr>
</table>

## Architecture

![GRACE-Mem architecture](docs/architecture/flow.png)

GRACE-Mem has two primary execution paths.

### Memory Construction

```text
dialogue turns
  -> temporal normalization
  -> compression / summary representation
  -> entity and relationship extraction
  -> entity reconciliation
  -> vector and BM25 storage
  -> FalkorDB graph synchronization
```

Stored evidence retains provenance back to source dialogue. Benchmark runs keep
artifacts and metadata so retrieval-only reruns can reuse a compatible ingest.

### Memory Retrieval

```text
question + query time
  -> query analysis
  -> hybrid entity and relationship retrieval
  -> graph expansion and direct evidence retrieval
  -> filtering and reranking
  -> optional Agent Filter
  -> answer generation
```

The Evidence Curation Agent shown in the retrieval panel is optional; the
standard pipeline can send reranked evidence directly to answer generation.

The repository keeps a one-way dependency direction:

```text
benchmarks / analysis / tools
        -> experiment orchestration
        -> grace_mem pipeline facades
        -> grace_mem services, storage, graph, and LLM utilities
```

- **Core (`grace_mem/`)**: ingestion, retrieval, graph synchronization, storage,
  LLM access, temporal handling, and provenance.
- **Benchmarks (`experiment/`)**: LoCoMo and LongMemEval orchestration, shared
  judging/scoring, artifacts, and run metadata.
- **Optional tools**: Agent Filter, replay utilities, and offline diagnostics.

The core `grace_mem/` package does not depend on benchmark-specific code.

## Quick Start

### 1. Set Up GRACE-Mem

The recommended local setup uses the included FalkorDB Docker service:

```bash
git clone https://github.com/JaneDoe-0728/GRACE-Mem.git
cd GRACE-Mem
cp .env.example .env
# Edit .env and configure the LLM and judge endpoints.
bash tools/setup_env.sh
```

The setup script installs locked dependencies, starts FalkorDB, downloads the
pinned embedding and reranker snapshots, and verifies the database and model
files.

### 2. Ingest and Retrieve One Memory

With FalkorDB and the configured LLM endpoint running:

```python
from grace_mem.pipeline.factory import build_pipeline

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

This example:

- stores one conversation turn;
- extracts structured memory and temporal information;
- synchronizes graph and vector indexes;
- retrieves provenance-linked evidence for a later question.

`build_pipeline()` owns the ingestor, retriever, graph client, vector stores,
and LLM client. Using it as a context manager closes owned resources.

## Benchmarks

GRACE-Mem ships with reproducibility-oriented pipelines for two long-term
memory benchmarks.

| Benchmark | Focus | Pipeline |
|---|---|---|
| **[LoCoMo](https://github.com/snap-research/locomo)** | Long conversational memory across sessions | ingest -> retrieve -> answer -> judge |
| **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** | Long-term memory and temporal reasoning | ingest -> retrieve -> answer -> judge |

Datasets are not bundled. Download pinned source revisions, verify SHA-256
checksums, and convert LongMemEval into runner-ready CSVs with:

```bash
uv run python -m tools.download_datasets --dataset all
```

Source JSON files and generated CSVs remain gitignored. Re-running the command
verifies existing files and skips a complete matching conversion.

### LoCoMo

The downloader writes `experiment/locomo/data/locomo10.json`. Run selected
samples with:

```bash
uv run python -m experiment.locomo.pipeline.runner \
  --sample-ids 0-9 \
  --run-tag my-run
```

### LongMemEval

The downloader writes one question CSV under
`experiment/longmem/script_data/<category>/`. Run a category with:

```bash
uv run python -m experiment.longmem.pipeline.watchdog \
  --run-tag my-run \
  --type temporal_reasoning
```

The default is the cleaned `S` variant. The downloader also supports `M` and
`oracle` with `--longmem-variant`. Pinned revisions, checksums, generated CSV
schema, output layout, and artifact compatibility rules are in the
[experiment guide](experiment/README.md).

### Evaluation

Use the shared post-hoc evaluation modules on completed runs:

```bash
uv run python -m experiment.common.evaluation.judge locomo <run-tag> --samples 0-9
uv run python -m experiment.common.evaluation.judge longmem <run-tag>
uv run python -m experiment.common.evaluation.score <run-tag>
```

See [EVALUATION.md](EVALUATION.md) for voting, abstention, output-column, and
oracle rules.

## Agent Filter

![Illustrative Agent Filter trace](docs/architecture/agent_flow_v2.png)

Agent Filter is an optional post-retrieval evidence-refinement layer. It starts
from an existing run's `Retrieved_Context`: GREP and READ inspect the question
corpus, while optional VECTOR performs a separate semantic search over the
existing summary VDB. It does not rerun the full KG retrieval pipeline.

The IDs and evidence counts in the figure illustrate one trace; they are not
fixed pipeline invariants. Execution failures preserve the original context,
but successful refinement is not a guarantee that answer quality improves.

```bash
# LongMemEval
uv run python -m experiment.agent_filter.replay.longmem \
  --source-run <existing-run> --run-tag <agent-run> --workers 4

# LoCoMo
uv run python -m experiment.agent_filter.replay.locomo \
  --source-run <existing-run> --run-tag <agent-run> \
  --chunk-turns 8 --samples 0-9 --workers 4 --granularity turn
```

LongMem VECTOR discovery uses `LONGMEM_ARTIFACT_ROOT` or `--artifact-root`.
LoCoMo discovers its summary VDB under each source sample. See the
[Agent Filter guide](experiment/agent_filter/README.md) for defaults,
adjudication scope, scoring, and trace inspection.

## Analysis and Diagnostics

GRACE-Mem includes post-run tools for investigating retrieval behavior without
changing the core runtime. Typical analyses include:

- gold-evidence and supplemental recall;
- retrieval failure and judge-flip analysis;
- filter, reranking, and evidence-selection ablations;
- dataset statistics and artifact inspection.

Most offline diagnostics operate directly on saved benchmark artifacts, so
expensive ingestion or retrieval stages do not need to be rerun. Some tools call
an LLM or launch benchmark subprocesses; check each module's `--help` and the
runtime matrix in the [experiment guide](experiment/README.md#offline-analysis).

## Installation

### Requirements

- Python `3.10` through `3.13`.
- [uv](https://docs.astral.sh/uv/) for dependency management.
- An OpenAI-compatible endpoint for LLM-backed pipeline stages.
- FalkorDB, either from the included Docker Compose service or an external URI.
- Disk space for the Qwen embedding and reranker weights.

The lockfile currently selects PyTorch CUDA 12.8 wheels. A compatible NVIDIA
runtime is the documented setup; CPU-only installations must select an
appropriate PyTorch source before `uv sync`.

The recommended Docker installation is shown in [Quick Start](#quick-start).
Docker is not required when an existing FalkorDB instance is available:

```bash
git clone https://github.com/JaneDoe-0728/GRACE-Mem.git
cd GRACE-Mem
cp .env.example .env
# Set NEO4J_URI and the endpoint variables in .env.
uv sync
uv run python tools/download_models.py
```

The bundled database listens on port `6379`; its browser UI is available at
`http://localhost:3000`.

## Configuration

Runtime endpoints belong in `.env`; experiment defaults belong in
[`experiment/experiment_config.py`](experiment/experiment_config.py).

### LLM

| Variable | Purpose |
|---|---|
| `LLM_API` | OpenAI-compatible base URL used by the KG and answer pipeline |
| `MODEL_NAME` | Model served by `LLM_API` |
| `JUDGE_LLM_API` | Base URL used by benchmark judging |
| `JUDGE_MODEL_NAME` | Judge model name |
| `GREP_AGENT_LLM_API` | Optional Agent Filter endpoint override |
| `GREP_AGENT_MODEL_NAME` | Optional Agent Filter model override |

The values in `.env.example` are local placeholders, not hosted services.

### FalkorDB

| Variable | Purpose |
|---|---|
| `NEO4J_URI` | FalkorDB Redis URI |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Graph connection credentials |
| `FALKORDB_PASSWORD` | Password used by the bundled container |
| `GRAPH_NAME` | FalkorDB graph key |

The historical `NEO4J_*` names are retained even though the active adapter is
FalkorDB.

### Embedding and Reranker

`tools/download_models.py` installs pinned snapshots of
`Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B` under `models/`.
`grace_mem/embeddings.py` uses the local embedding path when available and
otherwise falls back to the Hugging Face model ID.

### Experiment Configuration

`experiment/experiment_config.py` centralizes reproducibility, ingest,
retrieval, reranker, and Agent Filter defaults. CLI flags are intended for run
selection, stage selection, artifact reuse, and output paths. See the
[experiment guide](experiment/README.md) for the supported interface.

## Repository Structure

<details>
<summary><strong>Expand repository layout</strong></summary>

```text
GRACE-Mem/
├── grace_mem/                  # core memory, retrieval, storage, graph, and LLM code
├── experiment/
│   ├── common/                 # shared evaluation and run helpers
│   ├── locomo/                 # LoCoMo pipeline and analysis
│   ├── longmem/                # LongMemEval pipeline and analysis
│   └── agent_filter/           # optional evidence refinement
├── docs/architecture/          # architecture figures
├── tools/                      # setup, dataset, model, trace, and maintenance tools
├── LICENSE
├── EVALUATION.md
├── pyproject.toml
└── uv.lock
```

</details>

Datasets, generated artifacts, model weights, logs, secrets, and local test
suites are intentionally excluded from version control.

## Reproducibility

The benchmark layer centralizes settings and writes run metadata alongside
artifacts, checkpoints, logs, answers, and judge output. Dataset downloads,
embedding weights, and reranker weights use immutable revisions and checksums.
The LongMem converter writes a source manifest and one deterministic CSV per
question.

Exact results can still vary with the answer/judge endpoint and model revision,
hardware, experiment configuration, and external service behavior. This
repository currently provides no canonical paper configuration or
expected-score table; do not interpret a successful run as an exact reproduction
of an unpublished reference score.

## Documentation

| Document | Purpose |
|---|---|
| [Experiment guide](experiment/README.md) | Data layout, commands, artifacts, and analysis requirements |
| [Evaluation protocol](EVALUATION.md) | Judge, voting, scoring, abstention, and oracle behavior |
| [Agent Filter guide](experiment/agent_filter/README.md) | Evidence refinement, VECTOR, adjudication, and traces |
| [.env example](.env.example) | Runtime endpoint and graph configuration |

## License

GRACE-Mem is released under the [MIT License](LICENSE).
