# GRACE-Mem

**G**raph **R**etrieval with **A**gentic **C**orpus **E**vidence — a verifiable long-term conversational memory framework, shipping with complete experiment pipelines for the **LongMemEval** and **LoCoMo** benchmarks (ingest → retrieve → answer → judge), plus a post-retrieval agentic evidence-refinement layer (**Agent Filter**).

At its core the repo builds and queries a Knowledge Graph (KG):
1. **KG Context Retrieve** — given a user query, find the relevant entities and relations and assemble the semantic context needed to answer.
2. **KG Ingest** — extract entities and relations from dialogue, and update the vector store and FalkorDB.

---

## Requirements

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — manages dependencies and the virtualenv.
- **Python 3.10–3.13** — `uv` fetches a matching interpreter automatically.
- **An OpenAI-compatible LLM endpoint** — see "Configure the LLM endpoint" below.
- **Docker** (optional) — only needed to run the bundled local FalkorDB; if you already have a FalkorDB, just point `NEO4J_URI` at it.

## Setup

### 1. Get the source

```bash
git clone <REPO_URL>
```

Everything, including `noco-db-uploader/`, is vendored directly in the repo — there
are no git submodules to initialize.

### 2. Configure the LLM endpoint

The pipeline connects to any **OpenAI-compatible** endpoint via `LLM_API` / `MODEL_NAME` in `.env` — no IPs are hardcoded anywhere. You can either:

- **Run a local server** (recommended, works offline): use [LM Studio](https://lmstudio.ai/), [vLLM](https://docs.vllm.ai/), or [Ollama](https://ollama.com/) to load a model (e.g. `gpt-oss-20b`). Each exposes an OpenAI-compatible endpoint such as `http://localhost:1234/v1` (LM Studio). Verify the endpoint is serving and that the model id matches `MODEL_NAME` exactly:

  ```bash
  curl http://localhost:1234/v1/models
  ```

- **Or use a cloud API** (e.g. OpenAI): set `LLM_API` to `https://api.openai.com/v1`, `MODEL_NAME` to the model name, and fill in `OPENAI_API_KEY`.

### 3. Configure `.env`

```bash
cp .env.example .env
# Edit the LLM section (LLM_API / MODEL_NAME) to point at the endpoint from step 2;
# the FalkorDB section needs no changes if you use the bundled docker-compose.
```

Field descriptions live in the comments of `.env.example` (LLM, Judge, Agent filter, FalkorDB sections).

### 4. One-shot environment setup

```bash
bash setup_env.sh
```

Runs, in order: `uv sync` (install deps) → `docker compose up -d falkordb` (start FalkorDB) → `download_model.py` (download embedding / reranker models, skipped if present) → verify FalkorDB is reachable and model files are in place. It uses plain `docker` when your user can reach the daemon and falls back to `sudo` only if needed.

### 5. Get the benchmark data

**The benchmark datasets are not included in this repo** — both data directories are
gitignored, so a fresh clone has none of them and the commands under "Running
experiments" will fail with a missing-file error until you populate them. Download each
benchmark from its official release and lay the files out as below.

#### LoCoMo → `experiment/locomo/data/`

| File | Required | Notes |
|---|---|---|
| `locomo10.json` | yes | the LoCoMo QA/conversation file; `locomo.json` is also accepted |
| `locomo_by_session.jsonl` | no | one session per line. If absent it is derived from `locomo10.json` automatically |

For `--dataset locomo-plus` the expected names are `unified_input_samples_v2.json` and
`locomo_plus_by_session.jsonl` instead. Override any of these per run with
`--dataset-json` / `--sessions-jsonl`.

#### LongMemEval → `experiment/longmem/script_data/<type>/`

`<type>` is the question category you pass to `--type`, e.g. `single_session_user`,
`single_session_assistant`, `single_session_preference`, `multi_session`,
`temporal_reasoning`. Each category directory holds one **CSV per question** (discovered
with `--file-pattern`, default `*.csv`).

This is a **preprocessed** layout, not the raw LongMemEval release, and **no conversion
script ships with this repo** — you have to produce these CSVs yourself. Each file needs
these columns:

| Column | Purpose |
|---|---|
| `session_id` | groups turns into a session |
| `turn_index` | ordering within the session; user/assistant turns are paired in this order |
| `role` | `user` or `assistant` |
| `content` | the turn text |
| `dialogue_datetime` | when the turn happened; drives relative-time resolution |
| `question` | the question for this file (rename via `question_column`) |
| `answer` | optional gold answer, used by the judge |
| `question_date` | optional; falls back to `dialogue_datetime` / `date` / `timestamp` |

---

## Running experiments

Once the environment is ready (Setup above), the full benchmark flow (ingest → retrieve → answer → judge) is documented in **[experiment/readme.md](experiment/readme.md)**:

- **LongMemEval** (Part A): `experiment/longmem/watchdog.py` runs the whole flow.
- **LoCoMo** (Part B): `experiment/locomo/pipeline.py`, e.g.:

  ```bash
  uv run python experiment/locomo/pipeline.py \
    --dataset locomo --sample-ids 0-9 --run-tag my-run
  ```

All retrieval/ingest parameters live in [experiment/experiment_config.py](experiment/experiment_config.py) (single source of truth).

**Agent Filter** (the post-retrieval evidence-refinement layer) has its own guide, **[agent_filter/README.md](agent_filter/README.md)** — it runs `replay_run.py` / `grep_replay.py` on top of an existing retrieval run.

---

## Module layout

```bash
KG/
├── pipeline/
│   ├── factory.py           # Build and return the retriever / ingestor / graph / mgr bundle for callers
│   ├── retriever.py         # KG context assembly: query → keywords → entities/relationships → context/evidence
│   ├── ingestor.py          # KG ingest flow: compress summary → two-stage extraction → upsert VDB → sync FalkorDB
│   └── retrieval_steps/     # Modular Retriever sub-components (imported by retriever.py)
│       ├── search.py        # Hybrid search (VDB + BM25) over entities and relationships
│       ├── filtering.py     # Subgraph assembly, secondary filtering, and reranker recovery
│       ├── temporal.py      # Temporal relevance (Weibull decay)
│       └── evidence.py      # Evidence expansion: fetch summary spans by provenance
├── graph/
│   └── falkordb.py          # FalkorDB connection, schema, upsert, subgraph queries
├── llm/
│   ├── client.py            # LLM API wrapper (keyword extraction, extraction calls, etc.)
│   └── prompts/             # Prompt package (keyword / extraction / entity_ops sub-modules)
├── services/
│   ├── entity_manager.py    # Entity normalization, similarity search, ADD/UPDATE, embedding, caching
│   ├── relationship_manager.py # Relationship alignment, dedup, merge, embedding
│   └── provenance.py        # Unified tracking and merging of entity/relationship provenance (summary/session/message)
├── storage/
│   ├── chroma_manager.py    # VDBManager: unified entry point for VDB / BM25 / cache with async persistence
│   ├── chroma_vdb.py        # ChromaDB vector store wrapper (entities / relationships / summaries)
│   ├── bm25.py              # Dual BM25 index over entity name/desc
│   ├── cache.py             # Load/save cache for entities / relationships
│   └── paths.py             # Resolve the working artifacts dir (honors KG_ARTIFACTS_DIR)
├── utils/
│   ├── common.py            # Data models (Entity, Relationship, ExtractionResult), ID generation, tokenization
│   ├── reranker.py          # LLM reranker (Qwen3-Reranker-0.6B) for relevance scoring
│   ├── temporal/            # Shared temporal core (parsing / resolution / normalization)
│   ├── query_time_parser.py # Compatibility layer over utils/temporal
│   └── logger_config.py     # Logging utilities (JSONL event logs, timers)
```

> The tree lists the modules this document describes, not every file in the package.

---

## FalkorDB (Docker)

The project root ships a `docker-compose.yml` defining **four** FalkorDB services —
`falkordb` (6379 / UI 3000) plus `falkordb-2` … `falkordb-4` (6380-6382 / UI 3001-3003),
so several experiment runs can be kept on separate graphs at once.

Start just the one the default `.env` points at:

```bash
sudo docker compose up -d falkordb
```

Or start all four (this is what `setup_env.sh` does):

```bash
sudo docker compose up -d
```

Confirm it is running:
```bash
sudo docker ps | grep falkordb
sudo docker exec falkordb redis-cli -a falkordb ping   # should return PONG
```

(The `docker exec` form needs no `redis-cli` on the host. If you have one installed,
`redis-cli -p 6379 -a falkordb ping` works too.)

View logs:
```bash
sudo docker logs falkordb
```

Stop:
```bash
sudo docker compose down
```

Admin UI: `http://localhost:3000`

## FalkorDB credentials
Host: localhost  
Port: 6379  
Username:  
Password: falkordb  
TLS: off

## Core flows

### 1. KG context retrieval (`Retriever`)
Takes a user query as input and outputs structured KG context text.

**Main steps:**
1. **Keyword extraction**: the LLM produces high-level (global) and low-level (local) keywords.
2. **Hybrid search**:
   - **Entities**: vector similarity (VDB) + BM25 (dual name/desc index)
   - **Relationships**: vector similarity search
3. **Subgraph assembly and filtering**:
   - Fetch the entity and relationship subgraph (currently a union strategy)
   - Secondary filtering by query vector similarity (`filter_ent_threshold`, `filter_rel_threshold`)
   - Take top-K (`filter_ent_topk`, `filter_rel_topk`)
4. **Reranker**:
   - Optionally enable the reranker (`use_reranker=True`)
   - Rerank filtered-out entities/relationships and recover relevant items by `reranker_threshold`
5. **Evidence retrieval**:
   - Fetch the corresponding spans from summaries tracked via provenance
   - Candidates are scored and selected at **VDB-entry level** (`use_split_embeddings=True`,
     the default — the same path the benchmark pipelines take)
   - Use LLMlingua-compressed summary text as evidence
   - Supports full summary or fallback to raw turn
6. **Temporal relevance**:
   - Supports LiCoMemory-style Weibull time decay (currently commented out)
   - Enable time-aware retrieval via the `query_time` parameter

**Config parameters (`RetrieverConfig`):**
- `ent_topk`, `ent_threshold`: top-K and similarity threshold for initial entity search
- `rel_topk`, `rel_threshold`: top-K and threshold for initial relationship search
- `filter_ent_topk`, `filter_ent_threshold`: post-filter entity top-K and threshold
- `filter_rel_topk`, `filter_rel_threshold`: post-filter relationship top-K and threshold
- `use_reranker`, `reranker_threshold`, `reranker_topk`: reranker settings
- `summary_topk_per_item`, `summary_vec_threshold`: evidence retrieval settings
- `use_split_embeddings` (default `True`), `split_single_entry_raw` (default `True`): evidence
  selection granularity — see [Two artifact layouts](#two-artifact-layouts) below
- `use_full_summary`, `fallback_to_raw`: summary text handling on the legacy turn-level
  evidence path (`use_split_embeddings=False`)

**Example:**
```python
from KG.pipeline.factory import build_pipeline
_pipeline = build_pipeline()
retriever = _pipeline["retriever"]

kg_context = retriever.build_kg_context(
    question="When did Caroline attend the support group?",
    ent_topk=5,
    filter_ent_topk=3,
    query_time="2023/02/18 (Sat) 08:08"  # optional
)
print(kg_context)
```

### 2. KG ingest (`Ingestor`)
Extracts new knowledge from dialogue and updates the KG.

**Main steps:**
1. **Summary generation (via LLMlingua compression)**:
   - Compress the current dialogue text with `llmlingua-2` (compression rate = 0.6)
   - No longer generates summaries with an LLM; compresses the dialogue directly
   - The compressed text ("summary") is written to the summaries VDB
2. **Two-stage entity and relationship extraction**:
   - **Stage 1**: extract entities with the `entity_extraction_only` prompt
   - **Stage 2**: extract relationships over the already-extracted entities with `relationship_extraction_only`
   - **Note**: extraction input is the *raw dialogue*, not the summary; "extraction after summarization" refers only to execution order, not to using the summary as the extraction source
   - **Note**: the raw dialogue is first rewritten by `detect_and_parse_time_expressions`, then fed into extraction
   - Supports automatic retry (`max_retries=2`)
3. **Entity dedup and decision**:
   - Find similar entities via hybrid search (VDB + BM25)
   - Call the LLM's `generate_entity_ops` to decide ADD or UPDATE
   - Execute via `EntityManager.apply_ops`
4. **Relationship alignment and merge**:
   - Align relationship endpoints with `input2resolved`
   - Add or merge relationship descriptions and keywords
5. **Sync to FalkorDB**:
   - Call `graph.sync_entities()` / `graph.sync_relationships()`
   - Persist the knowledge and update the graph

**Config parameters (`IngestorConfig`):**
- `summary_embed_dim`: summary vector dimension (default 1024)
- `similar_entity_top_k`: top-K for similar-entity search (default 3)
- `entity_sim_threshold`: entity similarity threshold (default 0.7)
- `summary_context_prev_k_default`: previous-K turns of context (deprecated, replaced by compression)
- `llm_tuple_delim`, `llm_record_delim`, `llm_completion_delim`: delimiters for parsing LLM output

**Example:**
```python
from KG.pipeline.factory import build_pipeline
_pipeline = build_pipeline()
ingestor = _pipeline["ingestor"]

results = ingestor.summarize_and_ingest_turn(
    session_id=1,
    message_id=42,
    user_text="I went to an AI workshop yesterday.",
    assistant_text="That's great! What did you learn?",
    prev_k=2,  # unused, kept for compatibility
    dialogue_datetime="2023/02/18 (Sat) 08:08"  # optional
)
```

### 3. Two artifact layouts

The summaries vector store can hold one of two things, and **retrieval must be told which
one it is looking at**. Getting this wrong fails silently: retrieval looks up entry ids
that were never written, every provenance candidate scores `None`, and the evidence block
quietly degrades to direct-vector hits only — no error is raised.

| | **Single entry** (default) | **`:u` / `:a` pairs** |
|---|---|---|
| Who writes it | `Ingestor.summarize_and_ingest_turn` | `experiment/longmem/rebuild_split_summaries.py`, run **after** ingest |
| Entry ids | one per turn: `<session_id>:<message_id>` | two per turn: `…:u` (user raw) and `…:a` (assistant compressed) |
| Retrieval setting | `split_single_entry_raw=True` | `split_single_entry_raw=False` |
| Used by | the `build_pipeline()` default, and **LoCoMo** | **LongMem** only |

**`Ingestor` never produces `:u` / `:a` entries.** They exist only as an optional
post-processing pass, and only for LongMem. A single entry already carries *both* the
LLMlingua-compressed summary and the raw turn text (`add_summary(..., raw_text=...)`), so
the split is not what gives you access to the raw text — its only effect is that the user
turn and the assistant turn compete as two independent retrieval candidates.

For the experiment pipelines, one flag controls both halves:

```python
# experiment/experiment_config.py
INGEST_PARAMS = dict(
    use_split_summary=True,   # LongMem only
    ...
)
```

`use_split_summary=True` (the default) makes the LongMem pipeline run the rebuild
automatically after ingest **and** sets `split_single_entry_raw=False` for retrieval;
`False` skips the rebuild and keeps the single-entry layout. Because one flag drives both,
the artifacts and the retrieval config cannot drift apart. LoCoMo ignores the flag and
always uses the single-entry layout.

If you use `build_pipeline()` directly (outside the experiment pipelines), you get the
single-entry layout and need no rebuild.

## Component interactions
```mathematica
[User Query]
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Retriever (base retrieval)                                  │
│  → keyword extraction → Hybrid Search → subgraph assembly    │
│  → filter + Reranker → build Context + Evidence              │
└─────────────────────────────────────────────────────────────┘
    or
┌─────────────────────────────────────────────────────────────┐
│  Ingestor (knowledge write)                                  │
│  → LLMlingua summary compression → two-stage LLM extraction  │
│    (entities → relationships)                                │
│  → entity dedup decision (LLM ADD/UPDATE) → relation merge   │
│  → update VectorDB (entities/relationships/summaries) + FalkorDB │
└─────────────────────────────────────────────────────────────┘
```

## Responsibilities and key functions (Quick API at a glance)

### pipeline (factory and public API) (`pipeline/factory.py`)
**Wire dependencies → build services → return a dict of pipeline objects:**
1. **Create shared resources**: `LLMClient()`, `graph = graph_from_env().open()`, `GLOBAL_CACHE = MGR.cache`
2. **Instantiate services**:
  - `EntityManager(embedder, MGR, Provenance, GLOBAL_CACHE, processed_*_map...)`
  - `RelationshipManager(embedder, MGR, Provenance, GLOBAL_CACHE, processed_*_map...)`
3. **Return the object dict**:
  - `retriever = Retriever(llm, graph, mgr=MGR, embed=embedder.embed, cache=GLOBAL_CACHE, config=None)`
  - `ingestor = Ingestor(llm, graph, mgr=MGR, ent_svc=ent, rel_svc=rel, config=None)`
  - `return {"retriever": retriever, "ingestor": ingestor, "graph": graph, "mgr": MGR}`

**Main methods for a caller (e.g. server.py):**
- `_pipeline = build_pipeline(); retriever = _pipeline["retriever"]`
  Query → Keywords → Hybrid Search → filter + Reranker → Context + Evidence (string)
- `_pipeline = build_pipeline(); ingestor = _pipeline["ingestor"]`
  LLMlingua summary compression → two-stage extraction → entity dedup decision → upsert VDB → `sync_entities/relationships` to FalkorDB
- `retriever.build_kg_context(question, ent_topk=5, filter_ent_topk=3, query_time=None, ...)`
  Query → Keywords → Hybrid Search → filter + Reranker → Context + Evidence (string)
- `ingestor.summarize_and_ingest_turn(session_id, message_id, user_text, assistant_text, prev_k=2, dialogue_datetime=None, ...)`
  LLMlingua summary compression → two-stage extraction → entity dedup decision → upsert VDB → `sync_entities/relationships` to FalkorDB

---

### Retriever (KG retrieval) (`pipeline/retriever.py`)
Turns **Query → Keywords → Entities/Relationships → Context/Evidence**.

> The Retriever has been refactored into a modular architecture; core logic is split into `pipeline/retrieval_steps/` sub-modules (`search`, `filtering`, `temporal`, `evidence`), with `retriever.py` as the assembly entry point.

**Main methods:**
- `generate_query_keywords(question, request_id=None)`: LLM extraction of high/low-level keywords (falls back to the query as the low-level keyword on failure).
- `assemble_context_from_query(question, low_level_keywords, high_level_keywords, ...)`:
  - Delegates to `search.py` for **vector + BM25 hybrid search** over entities/relationships
  - Delegates to `filtering.py` for subgraph assembly, secondary filtering (`filter_ent/rel_threshold`), and reranker recovery
  - Delegates to `temporal.py` to apply temporal relevance (LiCoMemory-style, currently optional)
  - Returns `(context_entities, context_relationships, context_text, query_vec)`
- `self.evidence_builder.build_evidence_block(context_entities, context_relationships, ...)`
  (defined on `EvidenceBuilder` in `evidence.py`, not on `Retriever`; `build_kg_context` calls it):
  - Fetches **summary spans** as evidence by provenance
  - Two stages: score first → global sort for top-K → fetch text
  - Supports full summary or fallback to raw turn
- `build_kg_context(question, *, ent_topk, rel_topk, filter_ent_topk, ...)`:
  - **Main entry point**; returns the **final KG context (with evidence)**
  - All parameters are optional; uses `RetrieverConfig` defaults when omitted

**Sub-modules (`pipeline/retrieval_steps/`):**
- `search.py` (`EntityRelationshipSearcher`): vector + BM25 hybrid entity search, keyword-vector relationship search
- `filtering.py` (`ContextFilter`): subgraph union, query-vector secondary filtering, reranker recovery
- `temporal.py` (`TemporalRelevanceCalculator`): Weibull time-decay scoring
- `evidence.py` (`EvidenceBuilder`): fetch and rank summary spans by provenance

---

### Ingestor (KG write) (`pipeline/ingestor.py`)
Turns **dialogue** into **entities/relationships** and writes them to the VDB and FalkorDB.

**Main methods:**
- `summarize_turn(session_id, message_id, user_text, assistant_text, prev_k, request_id, dialogue_datetime=None)`:
  - Compress dialogue text with **LLMlingua-2** (compression rate = 0.6)
  - Returns `(summary_id, summary_text)`
  - No longer uses an LLM to generate summaries; uses the compressor
- `extract_entities_only(prompt_vars, prompt_template, request_id, ...)`:
  - **Stage 1**: extract entities only
  - Supports automatic retry (`max_retries=2`)
- `extract_relationships_only(prompt_vars, prompt_template, extracted_entities, request_id, ...)`:
  - **Stage 2**: extract relationships over the already-extracted entities
  - Validates relationship endpoints against `valid_entity_names`
  - Returns `(success: bool, relationships_list or error_msg)`
- `apply_extraction_and_sync(result, provenance, request_id, ...)`:
  - Find similar entities → LLM decides ADD/UPDATE → apply ops → align relationships → sync FalkorDB
  - Returns `{"entity_idx": ..., "relationship_metas": ...}`
- `ingest_turn(prompt_vars, prompt_templates, provenance, request_id, ...)`:
  - **Two-stage extraction**: entities → relationships → apply and sync
  - Uses the `entity_extraction_only` and `relationship_extraction_only` prompts
- `summarize_and_ingest_turn(session_id, message_id, user_text, assistant_text, prev_k, ...)`:
  - **Main entry point**: compress summary → two-stage extraction → write to VDB & FalkorDB (end to end)
  - Supports a fallback: if summarization fails, extract directly from the dialogue text

---

## LLM prompts (overview)
The main prompts and their tasks (actual content lives in `KG/llm/prompts/`):

- `keyword_extraction_PROMPT` (`KG/llm/prompts/keyword/extraction.py`)
  - Used by the **Retriever**: extract high- and low-level keywords from the user query/dialogue for hybrid retrieval (VDB + BM25).
- `entity_extraction_only` (`KG/llm/prompts/extraction/two_step.py`)
  - Used by the **Ingestor**: extract entities only (including Event/Date/Time/Timespan) and resolve relative time via `dialogue_datetime`.
- `relationship_extraction_only` (`KG/llm/prompts/extraction/two_step.py`)
  - Used by the **Ingestor**: build relationships over the extracted entity list; adding new entities is forbidden.
- `ENTITY_OPS_RULES_V2` (`KG/llm/prompts/entity_ops/rules.py`)
  - Used by the **EntityManager**: rules and merge strategy for ADD / UPDATE decisions.
- `ENTITY_OPS_FEW_SHOT` (`KG/llm/prompts/entity_ops/examples.py`)
  - Provides examples to make the LLM's entity-ops output format and decisions more consistent.
  - `EntityManager` concatenates the two into one prompt (`ENTITY_OPS_RULES_V2` + `ENTITY_OPS_FEW_SHOT` + the entity block).

---


### EntityManager (entities) (`services/entity_manager.py`)
Normalizes input entities, finds similar ones, performs ADD/UPDATE, and writes to the vector store / BM25.
- `normalize_entities(entities)`: convert to `{entity_name, entity_type, entity_description}`.
- `find_similar_for_hybrid(entities, top_k=5, threshold=0.6)`: merge similar-entity candidates, **BM25 first, then vector**.
- `apply_ops(ops_results, provenance)`: write according to the LLM's ADD/UPDATE decisions; returns  
  `entity_idx` ((name,type)→meta) and `input2resolved` ((input_name,input_type)→meta).

---

### RelationshipManager (relationships) (`services/relationship_manager.py`)
Aligns extracted relationships to entities, dedups, merges descriptions/keywords, writes to the vector store.
- `upsert_from_extraction(result, provenance, input2resolved, sync_to_graph=False, ...)`:  
  align endpoints via `input2resolved` → **add or merge** → write to VDB (optionally sync FalkorDB).

---

### Provenance (source tracking) (`services/provenance.py`)
Unifies sources (summary/session/message) into an event list that can be merged.
Both are `@staticmethod`s on the `Provenance` class — call them as `Provenance.prov_to_events(...)`.
- `prov_to_events(prov)`: different formats → a **standardized event list**.
- `merge_prov(old, new, max_events=50)`: **dedup/sort/truncate**, output `{"events":[...]}`.

---

### VDBManager (index/cache management) (`storage/chroma_manager.py`)
Centrally manages ChromaDB and BM25 for Entities/Relationships/Summaries, plus caches.
- `get_entities_vdb(dim)` / `get_relationships_vdb(dim)` / `get_summaries_vdb(dim)`: get / lazily init an index.
- `get_entities_bm25(load_if_empty=True)`: get the entity BM25 (dual name/desc index).
- `persist_async()`: background-save **VDB + BM25 + cache**.
- `reset_all(delete_files=True)`: reset all indexes and cache files.

---

### SimpleChromaVDB (ChromaDB wrapper, `storage/chroma_vdb.py`)
A thin vector-store wrapper over a ChromaDB collection. It takes **pre-computed
vectors**, not raw text — embedding happens in the caller (`embeddings.embedder`).
IDs are not passed separately: each row's id is read from its metadata dict.

Base class (`SimpleChromaVDB(dim, path, collection_name)`):
- `add(vectors: np.ndarray, metadatas: list[dict])`: append rows; the id of each row comes from `metadatas[i]["id"]`.
- `upsert(vectors, metadatas)`: same as `add`, overwriting rows whose id already exists.
- `search(query_vec: np.ndarray, top_k=5, threshold=None)`: cosine similarity search; returns `[(meta, score), ...]`, dropping anything below `threshold`.
- `batch_search(query_vecs, top_k=5, threshold=None)`: one result list per query vector.
- `compare_by_id(mid, query_vec, threshold=0.0)`: score a *known* row against the query; returns `(meta, score)` or `None`.
- `compare_by_id_raw(mid, query_vec, ...)`: same comparison returning the bare score, with no threshold applied.
- `update(ids, vectors=None, metadatas=None)`: patch vectors and/or metadata in place.
- `delete(ids: list[str])`: delete rows by id.
- `rebuild(all_vectors, all_metadatas)`: drop and recreate the collection from scratch.
- `save()` / `load()` / `close()`: persistence and connection lifecycle.
- `size`: row count (**property**, not a method).
- `export_metadatas_jsonl(output_path)`: dump all metadata to JSONL; returns the row count.

Subclasses: `EntitiesVDB`, `RelationshipsVDB` (base behaviour only) and
`SummariesVDB`, which adds the summary-specific helpers:
- `add_summary(session_id, message_id, summary_text, dialogue_datetime=None, raw_text=None)`: write one entry per turn; returns the `summary_id` (`"<session_id>:<message_id>"`). This is what the Ingestor calls.
- `add_split_turns(session_id, message_id, user_text, assistant_summary, dialogue_datetime=None)`: write the `:u` / `:a` entry pair used by the split-embedding evidence path (built offline by `experiment/longmem/rebuild_split_summaries.py`).
- `get_text_by_entry_id(entry_id)` / `get_raw_turn_text_by_id(summary_id)` / `get_summary_text_by_id(summary_id)`: fetch stored text by id, in decreasing order of preference.
- `get_summaries_by_ids(summary_ids, max_len=3000, top_n=10)`: batch fetch summary texts.
- `get_recent_summaries(session_id, k=2, text_only=True)`: most recent summaries of a session.

---

### EntitiesBM25 (`storage/bm25.py`)
- `add(tokens_name, tokens_desc, meta)`: incremental add (rebuilds IDF internally).
- `get_scores(q_tokens)`: return name/desc scores from both paths.
- `metas`: meta array aligned with the VDB order.

---

### CacheStore (`storage/cache.py`)
- `load()` / `save()` / `clear()` / `reset()`: in-memory + on-disk cache management for entities/relationships.
- `build_id_to_meta_maps(cache)`: quickly build **id→meta** lookups (one each for entities/relationships).

---

### LLMClient (LLM call wrapper) (`llm/client.py`)
Unified management of LLM chat, extraction, and entity-ops decisions.
- `chat(messages, temperature=0.6, max_tokens=2048)`: regular non-streaming chat.
- `stream_chat(messages, temperature=0.6, max_tokens=2048)`: streaming chat reply.
- `generate_llm_extract(prompt, max_tokens=1024, temperature=0)`:
  - General extraction tasks (entities, relationships)
  - Returns `(output_text, latency_sec)`
- `generate_llm_keyword(prompt, max_tokens=512, temperature=0)`:
  - Extract keywords in a JSON Schema format
  - Returns `(json_string, latency_sec)`
- `generate_entity_ops(new_entities, similar_map)`:
  - Generate an ADD/UPDATE decision instruction block
  - Returns a structured operations list

**Note:** summary generation now uses the LLMlingua-2 compressor; the LLM's `generate_llm_summary` is no longer used.

---

### Graph (FalkorDB wrapper) (`graph/falkordb.py`)
Handles FalkorDB connection, schema creation, data sync, and subgraph queries.
- `open()` / `close()`: open or close the connection (supports `with`).
- `init_schema()`: create unique constraints (`Entity.id`, `KG_REL.id`).
- `sync_entities(entity_idx)`: batch-upsert entity nodes.
- `sync_relationships(relationship_metas)`: batch-upsert relationship edges.
- `get_node_subgraph(entity_ids)`: query nodes and their adjacent relationships.
- `get_edge_subgraph(rel_ids)`: query relationships and their source/target nodes.
- `clear_all()`: wipe the whole graph (for testing).
- `graph_from_env(entity_label="Entity", rel_type="KG_REL")`: build a Graph object from environment variables (requires `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`).

---

### Utils (`utils/`)

#### Data models and helpers (`utils/common.py`)
Defines core data structures and helper functions.
- **Data models**:
  - `EntityType`: entity-type enum (Person, Event, Date, Time, Location, Organization, Product, etc.)
  - `Entity(BaseModel)`: entity structure (entity_name, entity_type, entity_description)
  - `Relationship(BaseModel)`: relationship structure (source_entity, target_entity, relationship_description, relationship_keywords)
  - `ExtractionResult(BaseModel)`: extraction result (entities, relationships)
  - `KeywordExtractionResult(BaseModel)`: keyword-extraction result (high_level_keywords, low_level_keywords)
- **ID generation** (lowercased, non-alphanumerics collapsed to `_`):
  - `canonical_entity_id(name, etype)`: entity ID, format `<type>_<name>` — e.g. `("AI Workshop", "Event")` → `event_ai_workshop`
  - `canonical_rel_id(src_id, tgt_id)`: relationship ID, format `<src_id>_<tgt_id>`
- **Tokenization and normalization**:
  - `tokenize_en(text)`: English tokenization
- **Pickle utilities**:
  - `pickle_dump(path, obj)` / `pickle_load(path, default=None)`

#### Reranker (`utils/reranker.py`)
An LLM-based pointwise reranker using Qwen3-Reranker-0.6B.
- `LLMPointwiseReranker(model_name=None, device=None)`:
  - Initialize the reranker (defaults to the local Qwen3-Reranker-0.6B)
  - `rank_pairs(query, texts, threshold=None, doc_type="entity")`: score and sort query-text pairs
  - Returns `[(idx, score), ...]`, dropping items below `threshold` when one is given
  - `rerank(query, texts, ...)`: the batched variant used by the split-evidence path
- `get_reranker()`: get the global reranker singleton (lazy initialization)

#### Time parser (`utils/query_time_parser.py`)
The shared temporal core lives in `utils/temporal/` (the canonical implementation); `utils/query_time_parser.py` is a compatibility layer.
- `utils/temporal/`
  - English-only, deterministic-first temporal parsing / resolution
  - Returns structured temporal constraints, preserving the original phrase, range, granularity, status, and confidence
  - Currently shared by ingest and LongMem query rewrite
- `parse_query_time(query_time_str)`
  - Parse the project's reference-time format (e.g. `"YYYY/MM/DD (Weekday) HH:MM"`)
- `detect_and_parse_time_expressions(query, query_time, rewrite_query=True)`
  - Detect high-confidence English time expressions via the shared temporal core
  - Compat return `(rewritten_query, info_dict)`
  - `info_dict` additionally includes structured temporal constraints

Retrieval-time temporal constraints for LoCoMo remain future work; V1 does not enable LoCoMo query rewrite by default.

#### Logging (`utils/logger_config.py`)
Provides structured logging and timing.
- `setup_logger(name, log_dir, level)`: set up a human-readable logger (for the server)
- `make_module_jlog(name, filename)`: create a module-specific JSONL event-log function
  - Returns a `_jlog(event, request_id, **payload)` function
  - Each call writes one JSON line to the given file (for performance tracing and debugging)
- `_StepTimer()`: a simple timer; call `.sec()` to get elapsed seconds
