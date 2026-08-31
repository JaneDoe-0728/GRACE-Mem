# Package Structure

The target directory layout for GRACE-Mem, and the rules that decide where new
code goes.

This document answers **"where does it live?"**.
[ubiquitous-language.md](ubiquitous-language.md) answers **"what is it called?"**.
Every directory name below is a term defined there; introducing a directory
whose name is not in the glossary means adding the term first.

**Status: implemented.** The three branches below have landed; `grace_mem/` now
has this shape. Five things came out differently from the plan, each recorded in
place: `grace_mem/text.py`, `Provenance` in `domain/`, `sparse_index/` instead of
`vector_store/bm25.py`, `paths.py` moving one commit early, and `extractors/`
not being created.

---

## Organising principle

**Feature-first, layer-inside.** The top level of `grace_mem/` groups by domain
capability; inside each capability, files are grouped by responsibility.

The alternative — grouping by technical role at the top level (`models/`,
`services/`, `repositories/`) — scatters one capability across the tree. The
current layout is a partial version of that failure: **Ingest** today lives in
five places.

```
grace_mem/pipeline/ingest_steps/extract.py    the LLM call
grace_mem/services/entity_manager.py          resolve and persist
grace_mem/llm/prompts/extraction/             the prompts
grace_mem/storage/cache.py                    skipping re-extraction
grace_mem/utils/common.py:215-788             parsing the LLM's reply
```

The current top level also mixes classification principles at one level:
`graph/` (a backend), `llm/` (a vendor), `pipeline/` (an execution shape),
`services/` (a code role), `storage/` (infrastructure), `utils/` (nothing).
These are not siblings in any meaningful sense.

---

## Target: `grace_mem/`

```
grace_mem/
├── domain/                     what GRACE-Mem stores
│   ├── entities.py             Entity, EntityType, canonical_entity_id
│   ├── relationships.py        Relationship, canonical_rel_id
│   ├── extraction.py           ExtractionResult, KeywordExtractionResult
│   └── provenance.py           Provenance
│
├── ingestion/                  Turns -> Graph + VDB + Cache
│   ├── pipeline.py             the ingestion pipeline, end to end
│   ├── parsing.py              LLM reply -> ExtractionResult
│   ├── prompts/
│   │   ├── config.py           EXTRA_KWARGS: delimiters + entity_types
│   │   ├── extraction/         two_step.py
│   │   └── entity_ops/         rules.py, examples.py
│   ├── steps/
│   │   ├── compress.py
│   │   ├── extract.py          EntityExtractor + RelationshipExtractor
│   │   └── sync.py
│   └── managers/
│       ├── entity_manager.py
│       └── relationship_manager.py
│
├── retrieval/                  Query -> Evidence
│   ├── pipeline.py             the retrieval pipeline, end to end
│   ├── evidence.py             assembling the Evidence block
│   ├── reranker.py
│   ├── evidence_speaker_enricher.py
│   ├── raw_context_lookup.py
│   ├── prompts/
│   │   ├── adaptive/           multihop.py, rewrite.py
│   │   ├── keyword/            extraction.py
│   │   └── hyde_prompting.py
│   └── steps/
│       ├── search.py
│       ├── filtering.py
│       ├── spreading_activation.py
│       ├── narrowing.py
│       ├── pagerank.py
│       ├── summary_scoring.py
│       ├── adaptive.py
│       └── temporal.py
│
├── temporal/                   time expressions -> ResolvedTimeRange
│   ├── types.py                TimeContext, ResolvedTimeRange, TimeCategory, …
│   ├── classifier.py           TemporalMatch
│   ├── normalizer.py
│   ├── resolver.py
│   ├── patterns.py
│   └── query_time_parser.py
│
├── adapters/                   concrete external technology
│   ├── embedding/
│   │   └── embeddings.py       the one shared text embedding model
│   ├── graph/
│   │   └── falkordb.py
│   ├── vector_store/
│   │   ├── chroma_vdb.py
│   │   └── chroma_manager.py   VDBManager
│   ├── sparse_index/
│   │   └── bm25.py             EntitiesBM25 -- lexical, not a vector store
│   ├── cache/
│   │   └── cache.py            CacheStore
│   └── llm/
│       ├── client.py           LLMClient
│       └── token_tracking.py
│
├── runtime/                    process- and environment-scoped services
│   ├── logger_config.py
│   ├── reproducibility.py
│   ├── paths.py                resolve_artifacts_dir: reads KG_ARTIFACTS_DIR
│   └── analysis_log.py         append_analysis_record / append_pretty_block
│                               + the closed _JSONL_FILES artifact set
│
├── text.py                     tokenize_en: shared by both capabilities
└── bootstrap.py                composition root -- today's pipeline/factory.py
```

### Deviations from this tree, and why

**`grace_mem/text.py`.** `tokenize_en`, `is_date_token` and the stopword
set were in `utils/common.py`. Retrieval tokenizes for BM25, ingestion tokenizes
entity names; neither owns them and they carry no domain meaning. Putting them in
either capability would have made the other import across the boundary, so they
sit at the top level as a single module — no directory, per Rule 6.

**`Provenance` went to `domain/`, not `ingestion/`.** It imports nothing at all,
and retrieval uses it in three modules. Sending it to `ingestion/` would have
been the same boundary violation; sending it to `domain/` deleted a
`retrieval -> services` edge that should never have existed.

**`adapters/sparse_index/bm25.py`, not `adapters/vector_store/bm25.py`.** Its own
docstring calls it "the lexical half of the hybrid entity search" — a sparse
index whose rankings are fused with the dense ones by RRF, not a vector store.

**No `ingestion/extractors/`.** `EntityExtractor` and `RelationshipExtractor`
share one file, `ingestion/steps/extract.py`. Splitting one file into two is a
refactor, not a move, and belongs in a commit that is not claiming to be only a
move. An earlier draft of the tree above showed the split; it has been removed
so the tree states the target rather than a wish.

**No `domain/conversation.py`.** An earlier draft placed Turn, Session, Speaker
and the sid helpers there, taken from the glossary's vocabulary rather than from
the code. Checked: `Session` is not declared anywhere, `Turn` is declared in
`experiment/agent_filter/corpus.py` -- outside `grace_mem` entirely -- and
`Speaker`/`SpeakerTurn` live in `retrieval/evidence_speaker_enricher.py`, where
they serve evidence assembly. There is nothing to put in the module, so it is
gone from the tree. Those four remain glossary terms; they are simply not a
grace_mem module.

**`storage/paths.py` moved with the adapters, not with the runtime commit.** It
was the last file left in `storage/`, and leaving a one-file package behind for
one commit serves nobody. That move needed care: `_DEFAULT_ART_DIR` was
`Path(__file__).parent / "artifacts"`, so relocating the file would have silently
moved the default artifacts directory and orphaned every existing artifact tree.
It is now anchored on the package root and pinned.

### Responsibilities

| Directory | Owns | Never contains |
| --- | --- | --- |
| `domain/` | The data concepts and their identity rules. Plain models. | I/O, LLM calls, framework imports |
| `ingestion/` | Everything that turns a **Turn** into stored graph state. | Retrieval logic, scoring |
| `retrieval/` | Everything that turns a **Query** into an **Evidence** block. | Writes to the graph |

**A capability is not a **Stage**.** `Stage` is a benchmark term with exactly three
values — `ingest`, `qa_eval`, `judge` — and it is CLI-addressable. `retrieval/` is
a capability the `qa_eval` stage calls; it is not a stage and has no CLI value.
`ingestion/` is the capability the `ingest` stage calls. Say "the ingestion
pipeline", never "the ingestion stage".
| `temporal/` | Detecting, classifying, and resolving time expressions. | Anything benchmark-aware |
| `adapters/` | The one concrete implementation per external technology. | Domain decisions, business rules |
| `runtime/` | Services scoped to the process and its environment: logging config, RNG seeding, env-var-driven path resolution. | Domain rules. Naming a domain concept is fine; deciding anything about one is not |
| `bootstrap.py` | Wiring: select adapters by config, construct dependencies, validate required settings, decide object lifecycle. The only file that knows every adapter by name. | Domain behaviour, ingestion logic, or retrieval logic |

### Dependency direction

```
domain/          depends on nothing in grace_mem
   ↑
temporal/        depends on domain/
   ↑
ingestion/       depends on domain/, temporal/, adapters/
retrieval/       depends on domain/, temporal/, adapters/
   ↑
bootstrap.py     depends on everything
```

`runtime/paths.py` is the edge case worth stating: **Artifact** is a domain term,
but `resolve_artifacts_dir` decides nothing about artifacts — it reads
`KG_ARTIFACTS_DIR` from the environment so two concurrent processes do not
interleave their Chroma writes. That is a process-isolation concern, not a domain
one. Naming a domain concept does not make a module domain logic; deciding
something about one does.

Two rules follow, and they are the ones worth enforcing in review:

- **`domain/` imports nothing from the rest of `grace_mem/`.** If a domain model
  needs a vector store, the model is wrong.
- **`ingestion/` and `retrieval/` never import each other.** They share through
  `domain/` and `adapters/`. Today they already respect this; the layout should
  make breaking it obvious.

### Why there is no `ports/` (decided)

A hexagonal layout would put `Protocol` definitions in `ports/` and
implementations in `adapters/`. Deliberately not doing that yet:

- There is exactly **one** implementation of each external technology: FalkorDB,
  Chroma, OpenAI.
- `grace_mem/` currently declares **zero** `Protocol` or `ABC`.

Four interface files each with one implementation is the over-layering that
Rule 6 below warns against. `adapters/` alone still earns its place — it says
*why* `falkordb.py` and `chroma_vdb.py` are siblings.

This is a decision, not a deferral: **do not introduce `Protocol` or `ports/`
as part of this restructure.** Revisit only when a second implementation of any
adapter appears, or when a test needs a fake and monkeypatching the concrete
class stops being tolerable. The informal precedent already exists —
`EntityOpsProcessor` takes a `generate_fn` rather than an LLM client, which is a
port in everything but name, and that pattern is the cheaper answer while there
is one implementation of everything.

---

## Target: `experiment/`

`experiment/` is not a feature tree. It is a benchmark harness, so it groups by
**benchmark first**, and the two benchmarks are kept as parallel as their
execution models genuinely allow.

```
experiment/
├── common/                     used by BOTH benchmarks, identically
│   ├── evaluation/             JudgeEngine, Score, Verdict parsing
│   ├── error_analysis.py       derive_*, build_*, render_*, coerce_*
│   ├── artifacts/
│   └── runtime/
│
├── locomo/
│   ├── cli.py
│   ├── config.py               RunConfig, WorkerPaths, SamplePlan
│   ├── pipeline/
│   │   ├── runner.py           orchestrator: one Worker per Sample
│   │   ├── worker.py           one Sample, own process
│   │   └── stages/
│   │       ├── ingest.py
│   │       ├── qa_eval.py
│   │       └── judge.py
│   ├── analysis/
│   └── artifacts/
│
├── longmem/
│   ├── cli.py
│   ├── config.py               DatasetConfig
│   ├── pipeline/
│   │   ├── runner.py           orchestrator: one Dataset at a time, in-process
│   │   ├── batch.py
│   │   ├── watchdog.py         entry point: python -m …pipeline.watchdog
│   │   ├── rerun.py            RerunTarget
│   │   ├── decision.py
│   │   ├── aggregate.py
│   │   └── stages/
│   │       ├── ingest.py
│   │       ├── qa_eval.py
│   │       └── judge.py
│   ├── analysis/
│   └── artifacts/
│
└── agent_filter/               a study, not a benchmark; left as-is
```

### How far the symmetry goes

Symmetric, and should stay so:

| | LoCoMo | LongMem |
| --- | --- | --- |
| CLI | `cli.py` | `cli.py` |
| Config types | `config.py` | `config.py` |
| Stages | `pipeline/stages/` | `pipeline/stages/` |
| Orchestrator | `pipeline/runner.py` | `pipeline/runner.py` |
| Analysis | `analysis/` | `analysis/` |
| Artifacts | `artifacts/` | `artifacts/` |

**Asymmetric on purpose — do not "fix" this:**

- LoCoMo isolates each **Sample** in its own **Worker** process, because a sample
  builds a graph and loads model weights, and a crash must not take the run with
  it. `KG_ARTIFACTS_DIR` is a real per-sample boundary because it is set in the
  child's environment.
- LongMem runs **Datasets** in-process with separate VDBs but a shared graph, and
  has a **watchdog / rerun / decision** loop LoCoMo has no equivalent of.

So LongMem has no `worker.py`, and adding one would be an execution-model change,
not a file move. The one thing that *should* converge is the name: the LongMem
orchestrator is currently `MultiDatasetProcessor` and should become a **Runner**
per the glossary.

---

## Naming rules

**1. Directories are named for a domain capability.**
Use `ingestion/`, `retrieval/`, `temporal/`, `evaluation/`.
Not `helpers/`, `misc/`, `components/`, `processors/`, `managers/` at the top
level — those say what shape the code is, not what it does.

**2. Siblings sit at the same level of abstraction.**
`ingestion/ retrieval/ temporal/` are all capabilities. `ingestion/ falkordb/
utils/` is a process, a vendor, and a shrug.

**3. `utils/` is not a destination.**
If only **Retrieval** uses it, it belongs in `retrieval/`. Only genuinely
domain-free code (hashing, serialization, retry) may live in a `utils/`, and
the target above has no top-level `utils/` at all — everything currently there
has a real home.

**4. `common/` requires two real consumers.**
Both benchmarks must use it, with the same behaviour, without branching. If a
file in `common/` contains `if benchmark == "locomo": … elif "longmem": …`, it
is not a shared abstraction; it is two implementations sharing a file.

**5. Directory names are concepts; file names are roles.**
`retrieval/pipeline.py`, `retrieval/steps/search.py` — not
`pipelines/retrieval_pipeline/`, which puts the role back on the outside and
grows a layer-first tree all over again.

**6. A subpackage needs a boundary, not a headcount.**
Create one when it establishes a real namespace, dependency, ownership, or
public-API boundary. File count is evidence, not the test: `adapters/graph/`
holding only `falkordb.py` is justified because the directory is the statement
*"this is where a graph backend goes, and there is currently one"* — the boundary
exists whether or not a second file does.

What fails the test is a directory that adds a path segment and nothing else.
`ports/` would be exactly that today: four interface files whose only consumer is
their single implementation, drawing no boundary that `adapters/` does not
already draw.

(PEP 423 advises against deep nesting, but its status is Deferred — it is a
reference, not a rule this project is bound by.)

**7. Domain directories are named from the glossary; architectural directories
are named here.**

The two vocabularies are separate on purpose, and the split follows the
ubiquitous-language skill's own rule: *domain terms only — skip generic
programming concepts unless they carry domain-specific meaning*. Forcing
`adapters` or `bootstrap` into a domain glossary would dilute it with words that
say nothing about what GRACE-Mem *is*.

**Domain capability names — defined in [ubiquitous-language.md](ubiquitous-language.md):**

| Directory | Term |
| --- | --- |
| `ingestion/` | **Ingestion** — needs adding; the noun form of the **Ingest** stage |
| `retrieval/` | **Retrieval** — defined |
| `temporal/` | **Temporal** — needs adding |
| `steps/` | **Step** — defined, and genuinely domain-specific here: it exists to contrast with **Stage** |
| `prompts/`, `extractors/`, `managers/` | **Extractor**, **Manager** — defined |
| `evidence.py`, `narrowing.py`, `filtering.py` | defined |

**Architecture names — defined in this file, and nowhere else:**

| Directory | Meaning |
| --- | --- |
| `domain/` | The layer holding data concepts with no infrastructure dependency |
| `adapters/` | The single concrete implementation of one external technology |
| `runtime/` | Services scoped to the process and its environment |
| `bootstrap.py` | The composition root |
| `pipeline.py` | A capability's end-to-end entry point |

Adding a **domain** directory requires a glossary entry first, in the same PR or
an earlier one. Adding an **architectural** directory requires a row in the table
above and a reason it draws a boundary (Rule 6).

---

## Explicit non-goals

**This restructure does not make large files smaller.** These are the real
navigability problem and moving them does not touch it:

| File | Size | Target location |
| --- | ---: | --- |
| `pipeline/retriever.py` | 105.5 KB | `retrieval/pipeline.py` — still 105.5 KB |
| `pipeline/ingestor.py` | 50.8 KB | `ingestion/pipeline.py` |
| `utils/temporal/resolver.py` | 48.3 KB | `temporal/resolver.py` |
| `pipeline/retrieval_steps/evidence.py` | 41.1 KB | `retrieval/evidence.py` |

Splitting them is separate work with a different risk profile, and mixing it
into a move PR destroys the "this is only a move" property that makes these PRs
reviewable. Do the moves first; the feature boundaries will make the right split
points more obvious afterwards.

**It also does not change behaviour, CLI surface, or artifact layout.** See the
invariants below.

**And it does not rename files.** Every filename in the target tree is the
current one; only directories change. A move PR that also renames is no longer
reviewable as a move. Two consequences worth stating: `chroma_vdb.py` keeps its
name even though its directory now says `vector_store`, and the cache adapter is
`adapters/cache/cache.py` — not `pickle.py`, which would shadow the stdlib module
it imports.

---

## Invariants every migration commit must preserve

- **Documented entry points keep working.** These appear in `README.md` /
  `EVALUATION.md` and are the contract:
  ```
  python -m experiment.locomo.pipeline.runner
  python -m experiment.longmem.pipeline.watchdog
  python -m experiment.common.evaluation.judge
  python -m experiment.common.evaluation.score
  python -m experiment.agent_filter.locomo_replay
  python -m experiment.agent_filter.replay_run
  python -m tools.download_datasets
  ```
- **CLI flags and stage names are untouched.** `ingest`, `qa_eval`, `judge` stay
  spelled exactly that way — see glossary ambiguity #4.
- **Artifact paths and file names are untouched.** Anything under an artifacts
  directory, and every CSV column name.
- **Structured log event names are untouched** unless the PR is explicitly about
  them. `experiment/locomo/analysis/flips.py` and
  `experiment/longmem/analysis/fact_replay.py` match event strings literally, and
  historical logs cannot be rewritten.
- **Re-export barrels are dissolved, not preserved.** `grace_mem/llm/prompts/__init__.py`
  re-exports across what will become two features; splitting it is part of the
  move, not a follow-up.
- **String-based imports are updated by hand.** A mechanical rename misses these:
  - `experiment/locomo/helpers/__init__.py` — `import_module`
  - `experiment/common/reproducibility.py` — `importlib.import_module("experiment.experiment_config")`
- **Pickles are safe.** `CacheStore` and the BM25 store pickle plain dicts and
  lists, not class instances, so no module path is embedded in a cache file. No
  compatibility shim is needed. *(Verify this still holds before each PR.)*
- **ruff, mypy, pytest green at every commit**, not merely at the tip of the branch.

---

## Resolved decisions

These were open; they are settled. Recorded with the reasoning so a later PR
does not reopen them by accident.

**1. `error_analysis.py` moves to `experiment/common/` — but it splits first.**

The file already moved once, in the wrong direction. The shim it left behind,
`experiment/locomo/utils/error_analysis.py`, says so outright: *"These functions
used to live here. They moved to `grace_mem.utils.error_analysis` once the
LongMem runner needed them too."* Needing it in both benchmarks is the definition
of `experiment/common/`; that home did not exist at the time, so it went into
`grace_mem/` instead.

A whole-file move is blocked by one import: `grace_mem/pipeline/ingestor.py:59`
takes `append_analysis_record`. Sending the file to `experiment/common/` whole
would make the core depend on the harness, inverting the dependency rule above.

The API surface splits cleanly — the core needs **one** symbol; the harness needs
the other fourteen:

| Stays in `grace_mem/runtime/analysis_log.py` | Moves to `experiment/common/error_analysis.py` |
| --- | --- |
| `append_analysis_record` — the only symbol `ingestor.py` imports | `build_top_miss_snapshot`, `read_reranker_rows` |
| `append_pretty_block`, `timestamp_now` | `derive_drop_reasons`, `derive_anomaly_flags`, `derive_failure_type` |
| `_JSONL_FILES` and the private file helpers | `build_bridge_label`, `render_failure_digest` |
| | `extract_context_session_ids`, `is_temporal_question` |
| | `coerce_float`, `coerce_bool`, `compact_json` |

`_JSONL_FILES` stays with the writers on purpose: it is a *closed* set of
artifact names, and the file's own docstring notes the analysis scripts read
those files by name, so a typo would create an orphan nothing reads. One writer,
one name table, one place.

Once LoCoMo imports from `experiment/common/` directly, the 1.2 KB shim at
`experiment/locomo/utils/error_analysis.py` has no remaining callers and is
deleted in the same PR.

**2. The prompt re-export barrel is removed, not preserved.**

`grace_mem/llm/prompts/__init__.py` flatly re-exports nine names from four
subpackages; its docstring says the point is backward compatibility with an
older flat `prompts.py`. Splitting `extraction/` and `entity_ops/` into
`ingestion/` while `adaptive/`, `keyword/` and `hyde_prompting.py` go to
`retrieval/` makes the barrel unrepresentable, and that is the correct outcome —
it is precisely what allowed prompts for two capabilities to share a namespace.
Every `from grace_mem.llm.prompts import X` becomes a direct import.

**3. `EntityType` becomes the single source of truth for entity types.**

`llm/prompts/config.py` hardcodes the twelve type names as a comma-joined string
that `domain/entities.py` will declare as enum members. `EXTRA_KWARGS["entity_types"]`
is generated from the enum instead.

**This is verified to be a zero-diff change.** The enum's declaration order and
the hardcoded string are already identical:

```
Person, Event, Date, Time, Timespan, Location,
Organization, Product, Service, Activity, Topic, Concept
```

Python enums preserve declaration order, so `", ".join(t.value for t in EntityType)`
reproduces the current string byte for byte. The prompt text does not change, so
neither does model behaviour, so results stay comparable with historical runs.
**The PR must assert this equality in a test** — that assertion is what stops a
future reordering of the enum from silently rewriting every extraction prompt.

**4. `retriever.py` and `ingestor.py` are not split.**

Confirmed as out of scope. See Explicit non-goals above: they move to
`retrieval/pipeline.py` and `ingestion/pipeline.py` at their current size.

**5. No `Protocol` or `ports/`.** See *Why there is no `ports/`* above.

---

## Migration plan

**Three branches, many commits each** — not nine branches, and not one branch
holding everything. The grouping is by *kind of change*, because that is what
determines how a PR is reviewed:

| Branch | Kind of change | Reviewed by asking |
| --- | --- | --- |
| `refactor/package-structure-preparation` | Behaviour-adjacent edits and new guards | "Is this logic still correct?" |
| `refactor/terminology` | Renames inside a stable tree | "Is this only a rename?" |
| `refactor/grace-mem-package-structure` | Directory moves | "Is this only a move?" |

Mixing them destroys the question. A move PR that also renames cannot be
reviewed as a move; a rename PR that also changes logic cannot be reviewed as a
rename.

### Landing order: preparation → terminology → moves

Terminology goes **before** the moves, not after. The renames and the moves touch
the same files, and doing them in this order means no file is ever renamed into a
location it is about to leave. Concretely: if `ContextFilter` still has that name
when it moves, the new tree briefly contains `retrieval/steps/filtering.py`
declaring `ContextFilter` right beside `retrieval/evidence.py` — a contradiction
introduced by the move itself.

The exception is the terminology work confined to `experiment/`
(`QuestionCategoryScore`, `DatasetRunner`, `probe_*`): the moves never touch those
files, so those commits are order-independent and can ship whenever.

---

### Branch 1 — `refactor/package-structure-preparation`

Not directory moves. These change behaviour or add guards, and must be separable
from the moves for exactly that reason.

| Commit | Scope |
| --- | --- |
| `refactor(domain): derive prompt entity types from EntityType` | `EXTRA_KWARGS["entity_types"]` generated from the enum, plus the byte-equality test |
| `refactor(analysis): separate runtime logging from error analysis` | Writers → `grace_mem/runtime/analysis_log.py`; the fourteen analysis symbols → `experiment/common/error_analysis.py`; delete the LoCoMo shim |
| `test(architecture): guard the target package boundaries` | Extend `tests/test_architecture.py` |

**On the third commit.** `tests/test_architecture.py` already exists and already
does most of this work — it is the right home, not a new file:

- `test_core_to_experiment_dependencies_are_explicitly_bounded` already asserts
  no `grace_mem.*` module imports `experiment.*`. That is precisely the invariant
  the `error_analysis` split must not break, and it is already green.
- `test_internal_import_graph_has_no_cycles` will catch a move that creates one.
- `tools/import_graph.py` already builds the graph the new assertions need.

What to add now, on the current tree, so it guards every later commit:

```
test_ingestion_and_retrieval_do_not_import_each_other()
```

Verified green today: `pipeline/ingest_steps/` and `pipeline/retrieval_steps/`
have no imports of each other. Write it against the current paths and update the
paths in Branch 3.

What **cannot** be added yet: `test_domain_imports_nothing_from_grace_mem()`
needs `domain/` to exist, so it lands with the commit that creates it.

---

### Branch 2 — `refactor/terminology`

Resolves the *Flagged ambiguities* in
[ubiquitous-language.md](ubiquitous-language.md). Renames only; the tree does not
change shape.

| Commit | Resolves | Risk |
| --- | --- | --- |
| `refactor(analysis): rename numbered steps to probes` | #8 | None — one module, no artifact exposure |
| `refactor(evaluation): qualify question category scores` | #2 | Low — check CSV column names first |
| `refactor(longmem): rename MultiDatasetProcessor to DatasetRunner` | #3 | Medium — also renames `processor.py` → `runner.py`; check the LongMem CLI |
| `refactor(storage): spell out entity and relationship names` | #5 | None, but ~798 sites. Machine-applied, reviewed by sampling |
| `refactor(retrieval): rename ContextFilter to EvidenceFilter` | #1 | Low — 6 references to the class; `ctx_*` lives in 2 files |

Deliberately **not** in this branch:

- **#4, the `qa_eval` stage rename.** Not doing it at all: CLI value, artifact
  directory name, and a column prefix in every historical result CSV.
- **#3b, the three `*context*` functions** (`assemble_context_from_query`,
  `build_kg_context`, `_render_context_text`). Their names are mirrored in
  structured log event strings that `locomo/analysis/flips.py` and
  `longmem/analysis/fact_replay.py` match literally. Renaming the function orphans
  the event name; renaming both makes historical logs unreadable. Needs its own
  decision, not a commit in a rename sweep.

---

### Branch 3 — `refactor/grace-mem-package-structure`

Moves only. One shared purpose: turn `grace_mem/` into the feature-first tree
above. Landing them as one PR is right — a half-moved tree is worse than either
end state.

| Commit | Scope |
| --- | --- |
| `refactor(domain): move core models into domain package` | The model half of `utils/common.py` → `domain/`. Adds `test_domain_imports_nothing_from_grace_mem` |
| `refactor(retrieval): move retrieval code into feature package` | `retrieval_steps/` + `retriever.py` + the three retrieval-only `utils/` files + the retrieval prompts |
| `refactor(ingestion): move ingestion code into feature package` | `ingest_steps/` + `services/` + the ingestion prompts + the parser half of `utils/common.py`; **removes the prompts barrel** |
| `refactor(temporal): move temporal code into feature package` | `utils/temporal/` → `temporal/` |
| `refactor(adapters): group external technology implementations` | `graph/` + `storage/` + `llm/client.py` + `embeddings.py` → `adapters/` |
| `refactor(runtime): move process-scoped services` | `logger_config.py`, `paths.py`, `reproducibility.py` → `runtime/`; `factory.py` → `bootstrap.py` |
| `refactor(cleanup): remove empty legacy packages` | Delete the emptied `utils/`, `pipeline/`, `services/`, `llm/` |

**Domain goes first**, before ingestion, because the ingestion commit imports
from `domain/entities.py`.

**The prompts barrel cannot be split across two commits.** The retrieval commit
takes `adaptive/`, `keyword/` and `hyde_prompting.py`; the ingestion commit takes
`extraction/` and `entity_ops/` and deletes
`grace_mem/llm/prompts/__init__.py`. Between them the barrel is half empty. Either
have the retrieval commit leave the barrel re-exporting from the new location, or
merge the two commits. Do not leave a commit where
`from grace_mem.llm.prompts import X` fails for some X.

**`tests/test_architecture.py:31-34` breaks during this branch.** It hardcodes
`grace_mem.llm.prompts.config` and `grace_mem.utils.temporal.*` as exact module
paths. Update it in the same commit that moves each — the prompts assertions with
ingestion, the temporal assertions with temporal. A grep for the old package path
across `tests/` is part of every move commit.

---

### Per-commit gate

Every commit, not every PR:

```bash
ruff check grace_mem experiment
mypy grace_mem                    # CI's scope; add experiment locally if you want it stricter
pytest -q -m "not integration"
git diff --staged --stat
```

And each commit must independently satisfy:

- One step, stated in the subject line.
- Imports resolve; every documented `python -m` entry point still runs.
- No temporarily broken state — if splitting two steps guarantees a broken commit
  in between, they are one commit.
- No unrelated reformatting mixed in.
- A move commit changes no behaviour; a rename commit changes no behaviour.
- The message says the architectural purpose, not just what moved.

`experiment/` restructuring is in none of these branches. It is lower value than
`grace_mem/` (the benchmarks are already mostly symmetric) and higher risk
(documented `python -m` entry points). Sequence it after all three land.

---

## Relationship to the glossary

```
docs/ubiquitous-language.md      what things are called
            ↓
docs/package-structure.md        where they live          ← this file
            ↓
refactor/* PRs                   moving and renaming
```

A rename in the glossary backlog and a move in the migration order above may
touch the same file. Do the rename first where they overlap: renaming inside a
stable tree is easier to review than renaming while moving.
