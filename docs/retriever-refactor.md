# Refactoring `grace_mem/`

How the memory system's package layout and its two largest modules get taken
apart, and how each step is shown not to change behaviour.

Companion to [package-structure.md](package-structure.md), which defines the
target directory layout and the rules behind it, and to
[ubiquitous-language.md](ubiquitous-language.md), which names things. This file
covers the part neither of them does: decomposing `retriever.py`, and proving
the decomposition is faithful without a live FalkorDB or LLM.

**Status: done.** See *Progress* at the end.

---

## Why `retriever.py`, measured

Not because it is 2112 lines. Because it has too many reasons to change, and
that is measurable: it holds nine retrieval-step modules at arm's length while
doing the work itself.

| Method | Lines | Calls into `retrieval_steps/` | Reads `self.cfg.<knob>` |
| --- | ---: | ---: | ---: |
| `assemble_context_from_query` | 780 | 12 | 34 |
| `build_kg_context` | 366 | 2 | 39 |
| `_adaptive_research` | 223 | 0 | 5 |

Twelve delegations across 780 lines is not thin orchestration that grew fat. It
is a method that reads thirty-four configuration knobs and decides everything
itself, next to a `retrieval_steps/` package that was built to hold exactly
those decisions.

`assemble_context_from_query` breaks along seams its own comments already mark:

| Stage | Lines | Goes to |
| --- | ---: | --- |
| 0) Embed query | 3 | stays |
| 1) Search entities and relationships | **311** | `candidates.py` |
| 2) Compute intersection | 30 | stays |
| 3) Filter candidates — dispatch on `filter_method` | **185** | `filter_policy.py` |
| 4) Reranker recovery / reranker-only selection | 83 | stays |
| 4b) Temporal containment boost | 38 | stays |
| 5) Render context text | 31 | `rendering.py` |

Stages 1 and 3 are 64% of it. They are the work.

`build_kg_context`'s 366 lines split the same way: keywords (16), adaptive (22),
two ablation blocks (29), HyDE (60), narrowing (16), and a 91-line combine.

---

## Target layout

Feature-first, layer-inside. Adjusted from the proposal against what the code
actually contains -- seven entries were dropped or renamed because nothing
existed to put in them, and twenty-six real files the proposal omitted are
placed.

```
grace_mem/
├── domain/                       data concepts; imports nothing from grace_mem
│   ├── entities.py               Entity, EntityType, canonical_entity_id
│   ├── relationships.py          Relationship, canonical_rel_id
│   ├── extraction.py             ExtractionResult, KeywordExtractionResult
│   └── provenance.py             Provenance -- retrieval uses it too
│
├── ingestion/                    Turn -> Graph + VDB + Cache
│   ├── pipeline.py               was pipeline/ingestor.py
│   ├── parsing.py                the parser half of utils/common.py
│   ├── prompts/                  config.py, extraction/, entity_ops/
│   ├── steps/                    compress.py, sync.py
│   ├── extractors/
│   │   ├── entity_extractor.py   split from ingest_steps/extract.py
│   │   └── relationship_extractor.py
│   └── managers/                 entity_manager.py, relationship_manager.py
│
├── retrieval/                    Query -> Evidence
│   ├── pipeline.py         ★     orchestration only                 target ~450
│   ├── config.py                 Search/Filter/Evidence/AdaptiveConfig
│   ├── trace.py                  pure trace builders
│   ├── keyword_cache.py          KeywordCache
│   ├── candidates.py       ★     stage 1 -> CandidateSet                  ~330
│   ├── filter_policy.py    ★     stage 3, the filter_method dispatch      ~200
│   ├── keywords.py         ★     keyword extraction                       ~150
│   ├── query_rewrite.py    ★     relative-time rewriting                   ~90
│   ├── hyde.py             ★     hypothetical-summary vectors             ~100
│   ├── rendering.py        ★     context text and evidence combine        ~170
│   ├── ablation.py         ★     the one KG_ABLATION_* registry            ~60
│   ├── evidence.py               EvidenceBuilder
│   ├── reranker.py               was utils/reranker.py
│   ├── speaker_enricher.py       was utils/evidence_speaker_enricher.py
│   ├── raw_turn_lookup.py        was utils/raw_context_lookup.py
│   ├── prompts/                  keyword/, adaptive/, hyde.py
│   └── steps/                    search, filtering, spreading_activation,
│                                 narrowing, pagerank, temporal_relevance,
│                                 adaptive
│
├── temporal/                     time expressions -> ResolvedTimeRange
│   └── types, classifier, normalizer, resolver, patterns, query_time_parser
│
├── adapters/                     one concrete implementation per technology
│   ├── embedding/embeddings.py
│   ├── graph/falkordb.py
│   ├── vector_store/             chroma_vdb.py, chroma_manager.py
│   ├── sparse_index/bm25.py      lexical, not a vector store
│   ├── cache/cache.py
│   └── llm/                      client.py, token_tracking.py
│
├── runtime/                      process- and environment-scoped services
│   └── logging_setup, reproducibility, paths, analysis_log
│
├── text.py                       tokenize_en -- both capabilities need it
└── bootstrap.py                  was pipeline/factory.py
```

★ = new module carved out of `retriever.py`.

**No top-level `utils/`.** Each of its seven files has a real home, which is
what made it a dumping ground.

### Dropped from the proposed tree, with the reason

| Proposed | Why not |
| --- | --- |
| `domain/conversation.py` | Nothing to put in it. `Session` is declared nowhere in the repo, `Turn` is declared in `experiment/agent_filter/corpus.py` -- outside `grace_mem` -- and `Speaker`/`SpeakerTurn` serve evidence assembly in retrieval. They stay glossary terms without being a module. |
| `domain/summaries.py` | There is no `Summary` type. Summaries are metadata dicts in a vector store. Giving them a type is a new domain model, not a refactor. |
| `ports/` | Zero `Protocol` or `ABC` exist in `grace_mem`, and there is exactly one implementation of each external technology. Four interface files with one implementation each is the over-layering the structure rules warn against. Revisit when a second implementation appears, or when a test needs a fake and monkeypatching the concrete class stops being tolerable. |
| `adapters/llm/openai_client.py` | The file is `client.py`. Renaming during a move breaks the "this is only a move" property that makes move commits reviewable. |
| `adapters/cache/pickle_cache.py` | `pickle.py` would shadow the stdlib module it imports. Kept as `cache/cache.py`. |
| `runtime/logging.py` | Collides with stdlib `logging`. Works under absolute imports, but every reader has to think twice. Named `logging_setup.py`. |
| `adapters/vector_store/chroma.py` | There are two files, not one: `chroma_vdb.py` and `chroma_manager.py`. |

---

## Verifying no behaviour changed

`retriever.py` sits at 23% coverage, and the uncovered ranges include all of
`assemble_context_from_query` and all of `build_kg_context`. The existing suite
cannot see the code this plan takes apart. A green run proves nothing here.

### Why not EXP-F06

`tests/fixed_output/f06_retrieval.py` is a retrieval determinism harness and
would work as an oracle, but it needs a live FalkorDB, a live LLM endpoint, the
downloaded models, and 100 LLM calls per run. More to the point, the endpoint it
would depend on is itself nondeterministic -- `keyword_cache.py` exists because
of that. Depending on something nondeterministic to demonstrate determinism is
the wrong shape.

### Offline characterization tests instead

The pattern is already established in this repo by
`tests/agent_filter_fakes.py`, whose docstring states the intent exactly:

> Both are faked here so the tests pin the harness's own behaviour -- command
> parsing, the tool loop, and evidence selection -- without a live endpoint.

Retrieval's external boundary is smaller than the agent filter's. Everything
`assemble_context_from_query` reaches outside itself is eleven calls across five
components:

```
self.searcher         embed_query, search_entities_hybrid, search_relationships_by_vec
self.evidence_filter  compute_subgraph_intersection, filter_by_similarity,
                      filter_by_rrf, compute_cosine_scores, rerank_filter,
                      rerank_and_recover
self.ppr_engine       run_ppr                      (twice)
self.sa_engine        run
self.graph            get_node_subgraph, get_edge_subgraph
```

plus `self.cache`, a dict, and `self.cfg`. None of them needs FalkorDB, an LLM
or a model.

**Files:**

```
tests/retrieval_fakes.py           doubles for the five components
tests/test_retrieval_pipeline.py   the characterization tests
tests/fixtures/retrieval_*.json    the snapshots, force-added like the guards
```

Construction uses the `_retriever_without_init()` pattern already used by
`tests/test_adaptive_trace.py`: `object.__new__(Retriever)`, then set the
component attributes directly, bypassing an `__init__` that would want real
dependencies.

**Workflow, per commit:**

1. Snapshot before the refactor: entity ids, relationship ids, context text and
   the trace, for each of the five `filter_method` values.
2. Refactor.
3. Re-run. The snapshots must match byte for byte.

**What this proves.** Given the same inputs and the same faked component
responses, the refactored code produces identical output, across every branch of
the `filter_method` dispatch -- which is the part a reader cannot check by eye.

**What it does not prove.** That the fakes return the shape the real FalkorDB
and Chroma return. The honest mitigation is to record the fake data from one
real run rather than hand-writing it -- one run that needs the live services,
after which every replay is offline. Hand-written fakes still catch the errors a
decomposition actually makes: a reordered stage, a dropped argument, an inverted
branch condition. They cannot catch a wrong belief about the real data shape.

---

## Deleting the dormant summary scorer

`retrieval_steps/summary_scoring.py` is 835 lines that the default
configuration cannot reach. `summary_filter_mode` defaults to `"semantic"`,
which takes the `else` branch in `evidence.py` -- a plain cosine sort. The
`_GRAPH_WEIGHTED_MODES` and `_RRF_MODES` branches are never entered, and no
experiment script sets the flag to a graph mode. The paper uses semantic only.

It is deleted rather than moved. The blast radius:

| Also removed | Where |
| --- | --- |
| The two graph branches | `evidence.py` lines 365-412, and the `scoring_weights` parameter |
| `ScoringWeights` construction | `retriever.py`, one block |
| Three exports | `retrieval_steps/__init__.py` |
| Eleven config fields | `EvidenceConfig` and `experiment_config.py`: `summary_filter_mode`, six weights, three enable flags, `summary_rrf_k` |
| `tests/test_summary_scoring.py` | 674 lines testing only the deleted module |

`summary_filter_mode` appears in `run_metadata.json` for historical runs. Those
files stay readable -- nothing reads that key back -- but the key stops being
written.

---

## Phases

Phase 1 needs no oracle: every extraction is a function with defined inputs and
outputs that reads no retrieval state, so an AST comparison per function proves
the move is faithful. That is the method already used for `trace.py`.

Phase 2 does need the oracle. Those 496 lines read 34 config knobs, call the
graph, and dispatch five ways; an AST comparison shows the code was carried over
intact, not that splitting it into two functions preserves behaviour.

| Phase | Content | Oracle |
| --- | --- | --- |
| **0** | `tests/retrieval_fakes.py` + snapshots | — |
| **1a** | `rendering.py` -- `_render_context_text` + combine | AST |
| **1b** | `query_rewrite.py` -- `_maybe_rewrite_retrieval_question` | AST |
| **1c** | `ablation.py` -- the scattered `KG_ABLATION_*` reads | AST |
| **1d** | `hyde.py` -- `generate_hyde_vector` + the blend | AST |
| **1e** | `keywords.py` -- `generate_query_keywords` | AST |
| **1f** | Delete `summary_scoring.py` and its eleven config fields | suite |
| **2a** | `candidates.py` -- stage 1, 311 lines, introduces `CandidateSet` | snapshot |
| **2b** | `filter_policy.py` -- stage 3, 185 lines | snapshot |
| **2c** | `steps/adaptive.py` absorbs `_adaptive_research` + `_additive_merge` | snapshot |
| **3** | Directory moves and file renames, with the package-structure work | suite |
| **4** | `ingestion/extractors/` split; then `evidence.py`, `resolver.py`, `ingestion/pipeline.py` | separate |

`CandidateSet` is the one new domain type this plan introduces -- it replaces
the six loose variables stage 1 currently threads into stages 2 and 3. It goes
in the glossary when it lands.

---

## Not in scope, and what that means for `ingestor.py`

`retriever.py` is decomposed but does not disappear: it becomes
`retrieval/pipeline.py` at roughly 450 lines, holding the stage order and
nothing else.

`ingestor.py` is only moved and renamed by this plan -- `ingestion/pipeline.py`,
1074 lines, unchanged inside. It has the same shape as `retriever.py` did:

```
 316  _repair_temporal_entities        a module-level function
 597  class Ingestor
 260    summarize_and_ingest_turn
  97    ingest_turn
  88    _log_ingest_delta
```

with **five** delegation calls in the whole file. It is the same God Object,
smaller, and it deserves the same treatment -- but on its own branch, with its
own characterization tests. Phase 4 is where that starts.

Also untouched: `evidence.py` (970), `temporal/resolver.py` (1322),
`steps/filtering.py` (894). Each is P1 or P2 on its own.

---

## Progress

All five phases landed. `retriever.py` went from 2496 lines to
`retrieval/pipeline.py` at 1836, and its 780-line method to 330.

| Phase | | |
| --- | --- | --- |
| 0 | ✅ | `retrieval_fakes.py`, five snapshots, injection-verified |
| 1 | ✅ | config groups, keyword cache, trace, rendering, query rewrite, ablation, HyDE, keywords; `summary_scoring.py` deleted |
| 2 | ✅ | `_search_candidates` + `CandidateSet`, `_filter_candidates`, `additive_merge` |
| 3 | ✅ | domain, retrieval, ingestion, temporal, adapters, runtime, bootstrap |
| 4 | ✅ | `extractors/` split |

Three findings the plan did not anticipate:

**An output-only snapshot proved nothing.** The reranker recovery unions
its result with the filter's, so a deliberately broken filter top-k left
the returned entity list identical. The doubles record the whole
conversation with the collaborators, not just what came back, and that
is what makes the snapshots protective.

**Stage 1 stayed a method.** It has fourteen free variables and needs
twelve things from `self`, eight of them name-resolution helpers. A
module function would take about twenty parameters. All nineteen uses of
those helpers are logging, so the logic is separable from its
instrumentation -- that separation is what a later step should do.

**`_adaptive_research` cannot leave the class.** It calls
`assemble_context_from_query` to run the second pass; moving it out
means passing the Retriever in, which is a circular dependency wearing a
parameter's clothes.

Two things nearly broke silently and were caught by something other than
the tests: a bare `return` inside stage 1 that used to end the whole
query (caught by mypy, on disagreeing return types), and
`_DEFAULT_ARTIFACTS_DIR` following `paths.py` out of `storage/` and
orphaning 123 MB of Chroma state (caught by checking the resolved path
against the old one).

Still untouched, each its own job: `evidence.py` (900),
`temporal/resolver.py` (1322), `ingestion/pipeline.py` (1074),
`retrieval/steps/filtering.py` (894).
