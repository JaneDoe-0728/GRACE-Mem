# Agent Filter

Agent Filter is an optional post-retrieval evidence-refinement layer. It starts
from an existing benchmark row's `Retrieved_Context`, lets an LLM inspect the
question corpus with `GREP` and `READ`, and can use `VECTOR` to discover
semantically related summary records. It then rebuilds the answer context from
the selected evidence.

The current retrieval configuration keeps up to 16 reranked summaries by
default, and Agent Filter has its own maximum of 16 evidence IDs. These are
configuration defaults, not invariants for every run.

- Agent Filter does **not** rerun the full KG entity, relationship, graph, and
  reranker pipeline.
- `VECTOR`, when available, performs a separate embedding search over the
  existing `summaries_chroma` store. `GREP` and `READ` operate on the corpus.
- Parsing, tool, and execution failures preserve the original context. This
  protects availability; it does not guarantee that successful refinement
  improves answer quality.

## Implementation

| Purpose | File |
|---|---|
| Orchestration: prepare, search, verify, finalize | [`harness.py`](harness.py) |
| Configuration read from `GREP_AGENT_PARAMS` | [`config.py`](config.py) |
| The per-question corpus, and the GREP/READ tools | [`corpus.py`](corpus.py) |
| Command parsing across every reply format | [`protocol.py`](protocol.py) |
| Reading and rebuilding the answer context | [`context.py`](context.py) |
| The search loop and its tools | [`loop.py`](loop.py) |
| Sufficiency verification and its repair round | [`verification.py`](verification.py) |
| Answer-blind adjudication | [`adjudication.py`](adjudication.py) |
| Evidence selection policy | [`finalization.py`](finalization.py) |
| Semantic search over the summaries VDB | [`vector_search.py`](vector_search.py) |
| Prompts | [`prompting/`](prompting) |
| Optional mechanisms (dated fact ledger) | [`extensions/`](extensions) |
| LongMemEval replay | [`replay/longmem.py`](replay/longmem.py) |
| LoCoMo replay | [`replay/locomo.py`](replay/locomo.py) |
| Shared defaults | [`../experiment_config.py`](../experiment_config.py) |

`GREP_AGENT_PARAMS` is the source of truth for algorithm defaults such as mode,
call caps, evidence caps, VECTOR thresholds, graph context, and adjudication.
Run selection and filesystem inputs are still controlled by CLI flags, and
endpoint/artifact discovery can use environment variables.

## Prerequisites

1. Complete the root [installation](../../README.md#installation) and endpoint
   [configuration](../../README.md#configuration).
2. Produce or obtain a benchmark run whose answer CSV contains
   `Retrieved_Context`.
3. For LongMem VECTOR support, retain the matching ingest artifacts and set
   `LONGMEM_ARTIFACT_ROOT` or pass `--artifact-root`.

Relevant environment variables:

| Variable | Purpose |
|---|---|
| `LLM_API` / `MODEL_NAME` | Answer model and fallback Agent Filter endpoint |
| `GREP_AGENT_LLM_API` / `GREP_AGENT_MODEL_NAME` | Optional Agent Filter endpoint override |
| `JUDGE_LLM_API` / `JUDGE_MODEL_NAME` | Post-hoc judge |
| `LONGMEM_ARTIFACT_ROOT` | LongMem root containing per-question summary VDBs |

## LongMemEval Replay

```bash
uv run python -m experiment.agent_filter.replay.longmem \
  --source-run <existing-retrieval-run> \
  --run-tag <agent-filter-run> \
  --workers 4
```

Useful selectors:

- `--category <category>` and `--limit N` restrict a debugging run.
- `--names-file <path>` reads a `category,stem` allowlist.
- `--artifact-root <path>` enables VECTOR from matching summary artifacts.
- `--force` permits replacing output for the same run tag.

The command reads the source answer CSVs, refines their contexts, generates new
answers, and writes a separate run under
`experiment/longmem/output/<agent-filter-run>/`.

## LoCoMo Replay

```bash
uv run python -m experiment.agent_filter.replay.locomo \
  --source-run <existing-retrieval-run> \
  --run-tag <agent-filter-run> \
  --chunk-turns 8 \
  --samples 0-9 \
  --workers 4 \
  --granularity turn
```

LoCoMo locates each summary VDB below the source sample's `artifacts/`
directory and reports whether VECTOR is available. `--granularity chunk` exposes
chunk IDs to the agent; `--granularity turn` exposes individual turns.

## VECTOR Discovery

For LongMem, the expected layout below `--artifact-root` is:

```text
<artifact-root>/<category>/artifacts_<question-stem>/summaries_chroma/
```

An empty artifact root disables VECTOR. A missing or mismatched
`summaries_chroma` directory also reports `VECTOR OFF`; GREP and READ remain
available. VECTOR results are discovery candidates and are subject to the
harness's evidence-selection behavior.

## Adjudication

After the agent proposes FINAL evidence, the optional answer-blind adjudicator
can reconsider dropped seed evidence and add relevant items back. It never sees
the generated answer and only adds evidence.

With the current defaults, adjudication is enabled for these LongMem categories:

- `single_session_preference`
- `multi_session`
- `temporal_reasoning`
- `knowledge_update`

It is not globally always-on: `grep_agent_adjudicate=0` disables it, and
`grep_agent_adjudicate_categories` controls its scope. LoCoMo replay passes no
LongMem category, so the current category allowlist does not enable adjudication
there.

## Outputs and Evaluation

Trace files are written alongside each run:

- LongMem: `_grep_agent_traces.jsonl`
- LoCoMo: `_grep_traces.jsonl`

For LongMem traces, build a self-contained HTML viewer with:

```bash
uv run python -m tools.agent_filter_trace_viewer.build \
  --run-tag <agent-filter-run>
```

Judge and score the new outputs with the shared evaluation CLIs:

```bash
uv run python -m experiment.common.evaluation.judge longmem <agent-filter-run>
uv run python -m experiment.common.evaluation.judge locomo <agent-filter-run> --samples 0-9
uv run python -m experiment.common.evaluation.score <agent-filter-run> --agent
```

Use the same source run, question set, answer model, judge model, and evaluation
settings when comparing a baseline with Agent Filter.

## LoCoMo Trace Accounting

LoCoMo seed IDs are chunk-level, while turn-granularity FINAL IDs can include a
turn suffix such as `t2`. `replay/locomo.py` normalizes FINAL IDs to their chunk
prefix before writing kept/added/dropped trace accounting. LongMem seed and
FINAL IDs already use the same ID space.
