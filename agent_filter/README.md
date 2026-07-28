# Agent Filter — post-retrieval evidence refinement

Agent Filter is a layer that runs after vector retrieval + reranking. Once retrieval has picked the top-16 candidate summaries for a question, the agent uses terminal-style tools (`GREP` / `READ` / `VECTOR`) directly over the question's raw corpus to **verify the candidates and recover missing literal evidence**, then rebuilds the answer context from the selected evidence.

- **No re-retrieval**: Agent Filter starts from an existing run's `Retrieved_Context` and never touches embedding / reranker / graph retrieval.
- **Fail-safe**: if any step fails, it falls back to the un-refined original context, so answering never gets worse.
- **Single source of truth**: all pipeline parameters are hardcoded in `GREP_AGENT_PARAMS` in [`experiment/experiment_config.py`](../experiment/experiment_config.py); nothing is toggled via environment variables or CLI flags.

Implementation locations:

| Purpose | File |
|---|---|
| Core harness (filter / fetch / VECTOR / adjudicate) | [`experiment/longmem/agent_filter/harness.py`](../experiment/longmem/agent_filter/harness.py) |
| LongMem entry point | [`experiment/longmem/agent_filter/replay_run.py`](../experiment/longmem/agent_filter/replay_run.py) |
| LoCoMo entry point | [`experiment/locomo/grep_replay.py`](../experiment/locomo/grep_replay.py) |
| Parameters | [`experiment/experiment_config.py`](../experiment/experiment_config.py) |

---

## 1. Prerequisites

1. **Retrieval already done**: you need an existing run whose CSV contains a `Retrieved_Context` column (the top-16 candidates). Agent Filter only operates on top of that.
2. **Configure the endpoint**: copy `.env.example` to `.env` and fill in your OpenAI-compatible endpoint and model:

   ```bash
   cp .env.example .env
   ```

   Relevant variables:

   | Variable | Purpose |
   |---|---|
   | `LLM_API` / `MODEL_NAME` | LLM for answering and retrieval |
   | `GREP_AGENT_LLM_API` / `GREP_AGENT_MODEL_NAME` | model that drives the filter agent (falls back to the KG LLM above if unset) |
   | `JUDGE_LLM_API` / `JUDGE_MODEL_NAME` | LLM for judging |
   | `LONGMEM_ARTIFACT_ROOT` | root dir of the summary VDB for LongMem's `VECTOR` tool (see §4) |

---

## 2. Run LongMem

```bash
python -m experiment.longmem.agent_filter.replay_run \
  --source-run <existing retrieval run> \
  --run-tag   <output name> \
  --workers 4
```

- Reads only `--source-run`'s `Retrieved_Context` and reruns from the agent layer.
- Does not overwrite existing output; add `--force` to rerun the same tag.
- Small-sample debugging: `--limit N` (per-category cap), `--category <cat>`, `--names-file <list>`.

## 3. Run LoCoMo

```bash
python experiment/locomo/grep_replay.py \
  --source-run <existing retrieval run> \
  --run-tag   <output name> \
  --chunk-turns 8 --samples 0-9 --workers 4 \
  --granularity turn
```

- LoCoMo's summary VDB lives in the source folder under `sample_<N>/artifacts/`; `grep_replay.py` picks it up automatically and prints `VECTOR ON/OFF` per sample.

---

## 4. The VECTOR tool and artifact-root (LongMem)

`VECTOR` lets the agent run its own semantic search over the question's summary VDB (`summaries_chroma`) to recover paraphrased evidence that literal GREP can't reach. This VDB is built during ingestion and is **not** part of the retrieval source.

- Point at it via the `LONGMEM_ARTIFACT_ROOT` environment variable or the `--artifact-root <dir>` flag (per-question layout: `<root>/<cat>/artifacts_<stem>/summaries_chroma`).
- Empty = VECTOR disabled. On startup it prints `artifact-root = … (summaries_chroma: N → VECTOR ON/OFF)`; `OFF` means the path is wrong or not mounted.
- LoCoMo does not need this setting (the VDB travels with the source folder).

---

## 5. Adjudicate — a built-in step

After FINAL, an answer-blind independent call (which cannot see the answer the agent already inferred) judges each dropped seed one by one as KEEP/DROP, and **recovers** (add-only, never remove) evidence relevant to the question topic. This is a **built-in, always-on** step in the pipeline and needs no flag.

The whole pipeline's behavior (mode, call-count cap, adjudicate, graph context, VECTOR thresholds, etc.) is fixed and hardcoded in `GREP_AGENT_PARAMS` in [`experiment/experiment_config.py`](../experiment/experiment_config.py); change that file to adjust.

---

## 6. Output

- LongMem: `experiment/longmem/output/<run-tag>/`
- LoCoMo: `experiment/locomo/output/standard/<run-tag>/`
- Step-by-step agent trace: `_grep_agent_traces.jsonl` / `_grep_traces.jsonl` under each run directory (keep it to reconstruct the agent's per-step search decisions).

For a fair comparison, the baseline and Agent Filter arms must use the **same source, question set, answer model, and judge settings**, otherwise the results are not directly comparable.

---

## 7. Note: LoCoMo's kept/added/dropped convention

LoCoMo seeds are fixed-16 chunks (e.g. `0__4:1`), but the agent's `final_sids` are turn-level (e.g. `0__4:1t2`). Before computing kept/added/dropped, you must truncate `final` back to the chunk prefix (`re.sub(r't\d+$','',s)`) and then compare against the 16 seeds, to preserve the `kept + dropped ≡ 16` invariant. `grep_replay.py` already applies this conversion when writing traces. (LongMem's seed and final share the same sid space, so it is naturally conserved and has no such issue.)

---

## Related analysis

- Cross-model five-config ladder: [`docs/analysis/five-config-ladder-cross-model.md`](docs/analysis/five-config-ladder-cross-model.md)
- LongMem question case studies: [`docs/analysis/longmem-v2/`](docs/analysis/longmem-v2/)
