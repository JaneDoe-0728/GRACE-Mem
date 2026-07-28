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

### 1. Get the source (with submodule)

The repo uses a `noco-db-uploader` submodule:

```bash
git clone --recurse-submodules <REPO_URL>
# If already cloned without the submodule, or the parent updated the pointer:
git submodule update --init --recursive
```

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

Runs, in order: `uv sync` (install deps) → `docker compose up -d` (start FalkorDB) → `download_model.py` (download embedding / reranker models, skipped if present) → verify FalkorDB is reachable and model files are in place.

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
│   ├── client.py            # LLM API wrapper (keyword extraction, entity-ops decisions, etc.)
│   └── prompts.py           # Manages and stores the prompts used to call the LLM
├── services/
│   ├── EntityManager.py     # Entity normalization, similarity search, ADD/UPDATE, embedding, caching
│   ├── RelationshipManager.py # Relationship alignment, dedup, merge, embedding
│   └── provenance.py        # Unified tracking and merging of entity/relationship provenance (summary/session/message)
├── storage/
│   ├── manager.py           # Unified entry point for VDB / BM25 / cache with async persistence
│   ├── chroma_manager.py    # ChromaDB manager (collection lifecycle / embedding binding)
│   ├── chroma_vdb.py        # ChromaDB vector store wrapper (entities / relationships / summaries)
│   ├── bm25.py              # Dual BM25 index over entity name/desc
│   └── cache.py             # Load/save cache for entities / relationships
├── utils/
│   ├── utils.py             # Data models (Entity, Relationship, ExtractionResult), ID generation, tokenization
│   ├── reranker.py          # LLM reranker (Qwen3-Reranker-0.6B) for relevance scoring
│   ├── query_time_parser.py # Time-expression parsing (relative time → absolute date)
│   └── logger_config.py     # Logging utilities (JSONL event logs, timers)
```

---

## FalkorDB (Docker)

The project root ships a `docker-compose.yml`; start it directly:

```bash
sudo docker compose up -d
```

Confirm it is running:
```bash
sudo docker ps | grep falkordb
redis-cli -p 6379 -a falkordb ping   # should return PONG
```

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
- `use_full_summary`, `fallback_to_raw`: summary text handling

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
- `build_evidence_block(context_entities, context_relationships, ...)`:
  - Delegates to `evidence.py` to fetch **summary spans** as evidence by provenance
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
- `ENTITY_OPS_RULES` / `ENTITY_OPS_RULES_V2` (`KG/llm/prompts/entity_ops/rules.py`)
  - Used by the **EntityManager**: rules and merge strategy for ADD / UPDATE decisions.
- `ENTITY_OPS_ONE_SHOT` / `ENTITY_OPS_FEW_SHOT` (`KG/llm/prompts/entity_ops/examples.py`)
  - Provide examples to make the LLM's entity-ops output format and decisions more consistent.

---


### EntityManager (entities) (`services/EntityManager.py`)
Normalizes input entities, finds similar ones, performs ADD/UPDATE, and writes to the vector store / BM25.
- `normalize_entities(entities)`: convert to `{entity_name, entity_type, entity_description}`.
- `find_similar_for_hybrid(entities, top_k=5, threshold=0.6)`: merge similar-entity candidates, **BM25 first, then vector**.
- `apply_ops(ops_results, provenance)`: write according to the LLM's ADD/UPDATE decisions; returns  
  `entity_idx` ((name,type)→meta) and `input2resolved` ((input_name,input_type)→meta).

---

### RelationshipManager (relationships) (`services/RelationshipManager.py`)
Aligns extracted relationships to entities, dedups, merges descriptions/keywords, writes to the vector store.
- `upsert_from_extraction(result, provenance, input2resolved, sync_to_graph=False, ...)`:  
  align endpoints via `input2resolved` → **add or merge** → write to VDB (optionally sync FalkorDB).

---

### Provenance (source tracking) (`services/provenance.py`)
Unifies sources (summary/session/message) into an event list that can be merged.
- `prov_to_events(prov)`: different formats → a **standardized event list**.
- `merge_prov(old, new, max_events=50)`: **dedup/sort/truncate**, output `{"events":[...]}`.

---

### VDBManager (index/cache management) (`storage/manager.py`)
Centrally manages ChromaDB and BM25 for Entities/Relationships/Summaries, plus caches.
- `get_entities_vdb(dim)` / `get_relationships_vdb(dim)` / `get_summaries_vdb(dim)`: get / lazily init an index.
- `get_entities_bm25(load_if_empty=True)`: get the entity BM25 (dual name/desc index).
- `persist_async()`: background-save **VDB + BM25 + cache**.
- `reset_all(delete_files=True)`: reset all indexes and cache files.

---

### ChromaVDB (ChromaDB, `storage/chroma_vdb.py`)
- `add(texts, metas, ids)`: add text, metadata, and optional IDs.
- `query(query_texts, n_results, where={}, where_document={})`: similarity query with conditional filtering.
- `get_by_ids(ids)`: fetch items by ID.
- `delete_by_ids(ids)`: delete items by ID.

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
  - Includes `_validate_entity_ops_output` validation
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

#### Data models and helpers (`utils/utils.py`)
Defines core data structures and helper functions.
- **Data models**:
  - `EntityType`: entity-type enum (Person, Event, Date, Time, Location, Organization, Product, etc.)
  - `Entity(BaseModel)`: entity structure (entity_name, entity_type, entity_description)
  - `Relationship(BaseModel)`: relationship structure (source_entity, target_entity, relationship_description, relationship_keywords)
  - `ExtractionResult(BaseModel)`: extraction result (entities, relationships)
  - `KeywordExtractionResult(BaseModel)`: keyword-extraction result (high_level_keywords, low_level_keywords)
- **ID generation**:
  - `canonical_entity_id(name, etype)`: generate an entity ID (format: `type::name`)
  - `canonical_rel_id(src_id, tgt_id)`: generate a relationship ID (format: `src::tgt`)
- **Tokenization and normalization**:
  - `tokenize_en(text)`: English tokenization
  - `tokenize_zh(text)`: Chinese tokenization
- **Pickle utilities**:
  - `pickle_dump(path, obj)` / `pickle_load(path, default=None)`

#### Reranker (`utils/reranker.py`)
An LLM-based pointwise reranker using Qwen3-Reranker-0.6B.
- `LLMPointwiseReranker(model_name=None, device=None)`:
  - Initialize the reranker (defaults to the local Qwen3-Reranker-0.6B)
  - `rank_pairs(query, texts, threshold=-3.0)`: score and sort query-text pairs
  - Returns `[(idx, score), ...]`, dropping items below the threshold
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
