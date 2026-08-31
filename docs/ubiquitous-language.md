# Ubiquitous Language

The canonical vocabulary for GRACE-Mem. One term per concept; everything else is
an alias to avoid. Specs, PRDs, commit messages, CLI flags, and identifiers use
these words and no others.

> Traditional Chinese edition: [ubiquitous-language.zh-TW.md](ubiquitous-language.zh-TW.md).
> Both files must be updated together.
> Where code lives: [package-structure.md](package-structure.md).

**Scope boundary.** This file holds *domain* terms only. Architectural vocabulary —
`domain`, `adapters`, `runtime`, `bootstrap`, `pipeline` as a filename — is defined
in [package-structure.md](package-structure.md) and deliberately kept out of here;
those words say nothing about what GRACE-Mem is. A word earns a place below only
if it carries GRACE-Mem-specific meaning: **Step** qualifies because it exists to
contrast with **Stage**; **Adapter** does not.

Scope: `grace_mem/` (the memory system), `experiment/` (the benchmark harness),
`tools/` (dev utilities). Regenerate the candidate list with:

```bash
python3 .claude/skills/uncle-dev-ubiquitous-language/scripts/scan_terms.py --top 40
```

---

## Knowledge graph core

The objects GRACE-Mem stores. All live in `grace_mem/utils/common.py`.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Entity** | One node in the knowledge graph: a named thing with a type and an embedded description. | `ent`, node, item, concept |
| **Relationship** | One directed edge between two entities, addressed by entity *name* rather than id. | `rel`, relation, edge, link, fact, triple |
| **EntityType** | The classification an entity carries: Activity, Concept, Date, Event, Location, Organization, Person, Product, Service, Time, Timespan, Topic. | kind, class, label |
| **ExtractionResult** | Everything extracted from a single turn — the entities and relationships an LLM call produced, before resolution. | extraction, extracted, payload |
| **Provenance** | The record of which turn(s) an entity or relationship came from. | source, origin, trace |
| **Summary** | A compressed restatement of one or more turns, stored alongside the graph and independently retrievable. | digest, abstract, compressed context |

**Not domain terms here.** *Fact* and *memory* appear under 100 times combined and
never as a declared type. Do not introduce them: what a caller might call a
"fact" is a **Relationship**, and what they might call a "memory" is the whole
graph plus its summaries.

## Conversation source

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Turn** | One utterance by one speaker, the atomic unit ingestion consumes. | message, utterance, line, exchange |
| **Session** | An ordered sequence of turns sharing a timestamp frame. | conversation, dialogue, thread |
| **Speaker** | The participant a turn is attributed to. | user, role, actor, author |
| **sid** | The stable turn identifier, `"session:pair:role"` (e.g. `answer_abc:6:u`). The token retrieval, gold annotation, and the agent filter all address turns by. | turn_id, uid, key |

## Ingestion

`ingest` is a **stage**. `compress`, `extract`, and `sync` are its **steps**
(`grace_mem/pipeline/ingest_steps/`), not stages of their own.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Ingestion** | The capability that turns **Turns** into stored graph state. Names the code that does the work, and the `ingestion/` package. | ingest (that is the stage), indexing, loading |
| **Ingest** | The benchmark **Stage** that invokes **Ingestion** for a whole run. A CLI value; see **Stage**. | ingestion (that is the capability), index, load, import, build |
| **Compress** | The step that shortens a turn's text before extraction. | summarize (reserve for **Summary**), shrink, prune |
| **Extract** | The step that asks the LLM for the entities and relationships in a turn, producing an **ExtractionResult**. | parse, mine, derive |
| **Sync** | The step that resolves extracted names against existing nodes and writes the result to graph, vector store, and cache. | persist, save, upsert, merge, commit |
| **Extractor** | A component that performs **Extract** for one object kind (`EntityExtractor`, `RelationshipExtractor`). | miner, parser |
| **Manager** | A component that resolves, merges, and persists one object kind (`EntityManager`, `RelationshipManager`). Owns writes; holds no state of its own. | service, handler, repository, DAO |

## Retrieval

`grace_mem/pipeline/retrieval_steps/`.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Retrieval** | The stage that turns a question into an evidence block. | search (reserve for the vector/BM25 step), lookup, recall |
| **Query** | The question as reformulated for search, after rewriting and embedding. | prompt, request, input |
| **Evidence** | The assembled block of entities, relationships, provenance, and summaries handed to the answering LLM. | context (see *Flagged ambiguities*), passages, snippets, retrieved docs |
| **Filtering** | The step that narrows and reranks retrieved entities and relationships. | pruning, selection, cleanup |
| **Narrowing** | The step that reduces an evidence block to its question-relevant snippets by keyword and entity overlap. | filtering (distinct step), compression, agent filter |
| **Spreading activation** | The step that expands from seed entities across graph edges. | traversal, walk, expansion |
| **SummaryScore** | The scoring breakdown for one summary candidate. | rank, weight, relevance |
| **Retriever** | The component that runs the retrieval stage end to end. | searcher, engine, reader |

## Temporal

`grace_mem/utils/temporal/`. The vocabulary here is already precise — treat it as fixed.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Temporal** | The capability that detects, classifies, and resolves time expressions in turns and questions. Names the `temporal/` package. | time handling, datetime, chrono |
| **TimeContext** | The reference frame a relative expression resolves against, anchored on the turn's own timestamp. | context (see *Flagged ambiguities*), frame, now, clock |
| **TemporalMatch** | One detected time expression — its text, span, and category — before resolution. | hit, mention, candidate |
| **TimeCategory** | The classification of a temporal expression (`RELATIVE_DAY`, `SEASON_POINT`, `MONTH_WEEK_RANGE`, …). | type, kind, pattern |
| **TimeGranularity** | The coarseness of a resolved range: DAY, WEEK, WEEKEND, MONTH, SEASON, YEAR, TIME, RANGE. | precision, resolution (see below), scale |
| **ResolvedTimeRange** | One temporal expression fully resolved to a start/end range, with provenance. Always a range, even for a point in time. | timestamp, date, interval |
| **ResolutionStatus** | How completely an expression resolved: RESOLVED, PARTIALLY_RESOLVED, AMBIGUOUS, UNRESOLVED, INVALID. Graded, never collapsed to a boolean. | state, success, valid |
| **TemporalConstraint** | A `ResolvedTimeRange` plus the operator relating a question to it ("in July" vs "before July"). | filter, predicate, range |
| **Anchor** | The configured clock time a vague daypart ("morning") maps to. | default, base, pivot |

## Experiment harness

Two benchmarks — **LoCoMo** (`experiment/locomo/`) and **LongMem**
(`experiment/longmem/`) — run the same three stages through different orchestration.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Run** | One end-to-end execution of a benchmark under one `RunConfig`. | job, session (reserve for conversations), trial, experiment |
| **Stage** | One of exactly three top-level phases: `ingest`, `qa_eval`, `judge`. Selected by CLI flag; the unit a run can be resumed from. | step (reserve for pipeline internals), phase, task |
| **Step** | One unit inside a grace_mem pipeline (`ingest_steps/`, `retrieval_steps/`). Never CLI-addressable. | stage, substage, module |
| **Probe** | One numbered diagnostic check over a finished run's logs (`step2_ingest` … `step9_evidence`). Reads artifacts; runs no pipeline. | step (see *Flagged ambiguities* #8), check, test, assertion |
| **Sample** | One LoCoMo conversation instance, addressed by integer `sample_index`. LoCoMo's unit of parallelism. | record, item, case, instance, example |
| **Dataset** | One LongMem data folder plus its config. LongMem's unit of parallelism, the counterpart to LoCoMo's **Sample**. | data, corpus, collection |
| **Runner** | The orchestrator that plans the units of work and spawns one **Worker** per unit. | processor, driver, executor, controller |
| **Worker** | The subprocess that runs one **Sample** or **Dataset** end to end. Communicates only through files. | job, task, child, thread |
| **Artifact** | Any file a stage writes for a later stage or for analysis: CSVs, stats JSON, snapshots, error bundles. | output, result, dump, export |
| **Snapshot** | A saved graph + vector-store state a run can resume ingestion from. | checkpoint, backup, cache (see **Cache**) |
| **RunConfig** | The immutable, hashable configuration parsed from the CLI for one run. | args, options, settings, params |

## Evaluation

`experiment/common/evaluation/`.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **QAEval** | The stage that retrieves evidence and generates an answer for every question in a run. It does **not** score. | eval, evaluation, inference, answering |
| **Judge** | The stage that decides whether a generated answer matches the gold answer, using an LLM. | eval, grade, score, assess |
| **JudgeEngine** | The benchmark-aware prompt, retry, and voting policy the judge stage applies. | judge (the stage), grader, scorer |
| **Verdict** | The judge's binary decision for one question. | score (reserve for numbers), result, correctness, grade |
| **Score** | A numeric measure computed from verdicts or text overlap (accuracy, F1, BLEU-1). | metric, rating, verdict |
| **Gold** | The reference answer a generated answer is compared against. | truth, ground truth, expected, label |
| **Category** | The question class a LoCoMo/LongMem question belongs to, used to break accuracy down. | type, bucket, label, tag |
| **Flip** | A question whose verdict changed between two runs; the signal a raw accuracy delta hides. | regression, diff, change |

## Storage

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Graph** | The FalkorDB knowledge graph holding entities and relationships for one run. | KG, db, store, neo4j |
| **VDB** | The vector store holding entity and relationship embeddings. Spelled `VDB` in identifiers, "vector store" in prose. | vector db, chroma, index, embedding store |
| **Cache** | The on-disk pickle of extraction results that lets a re-run skip LLM calls. | snapshot, store, memo |
| **Artifacts directory** | The per-run, per-sample directory every store derives its paths from (`KG_ARTIFACTS_DIR`). | output dir, workdir, run dir |

---

## Relationships

- A **Session** contains ordered **Turns**; a **Turn** is identified by its **sid**.
- **Ingest** consumes **Turns** and produces **Entities**, **Relationships**, and **Summaries** in the **Graph**, **VDB**, and **Cache**.
- **Ingest** is composed of the **Compress**, **Extract**, and **Sync** steps, in that order.
- **Extract** produces an **ExtractionResult**; **Sync** resolves it into the stores.
- A **Relationship** names its endpoints by **Entity** name; **Sync** is where an edge naming an unextracted entity is dropped.
- **Retrieval** turns a **Query** into an **Evidence** block, via the **Filtering**, **Spreading activation**, and **Narrowing** steps.
- **QAEval** invokes **Retrieval**, then generates an answer; **Judge** compares that answer to **Gold** and emits a **Verdict**.
- **Scores** are aggregated from **Verdicts**, optionally broken down by **Category**.
- A **Run** executes **Stages** in the order `ingest → qa_eval → judge`.
- A **Runner** spawns one **Worker** per **Sample** (LoCoMo) or per **Dataset** (LongMem).
- A **Worker** communicates with its **Runner** only through **Artifacts**.
- A **TemporalMatch** resolves into a **ResolvedTimeRange** against a **TimeContext**, carrying a **ResolutionStatus**.

---

## Example dialogue

> **Dev:** When a **Worker** finishes the `ingest` **Stage**, what has it actually written?
>
> **Domain expert:** Every **Turn** in its **Sample** has gone through **Compress**, **Extract**, and **Sync**. So the **Graph** holds the **Entities** and **Relationships**, the **VDB** holds their embeddings, and the **Cache** holds the raw **ExtractionResults** so a re-run can skip the LLM.
>
> **Dev:** And the **Summaries** — are those **Entities**?
>
> **Domain expert:** No. A **Summary** is a compressed restatement of turns, retrievable on its own. It is scored separately during **Retrieval**; it never becomes a node.
>
> **Dev:** During `qa_eval` we produce a **Score**, then?
>
> **Domain expert:** No — `qa_eval` retrieves **Evidence** and generates an answer, nothing more. The `judge` **Stage** compares that answer to **Gold** and emits a **Verdict**. A **Score** is only computed by aggregating **Verdicts** afterwards. Three different things, three different stages.
>
> **Dev:** So "the eval said 62%" is wrong phrasing.
>
> **Domain expert:** Right. The **Judge** produced the **Verdicts**; the accuracy **Score** over them was 62%.

---

## Flagged ambiguities

**1. "Context" means two unrelated things.** `ContextFilter`
([filtering.py:38](../grace_mem/pipeline/retrieval_steps/filtering.py#L38)) operates on
retrieved entities and relationships; `TimeContext`
([types.py:97](../grace_mem/utils/temporal/types.py#L97)) is a temporal reference frame.
Nothing connects them.
*Recommendation:* keep **TimeContext** as-is — it is the precise term for a
reference frame. Retire "context" everywhere else in favour of **Evidence**;
rename `ContextFilter` → `EvidenceFilter`.

**2. "Category" means two unrelated things.** `TimeCategory` classifies a temporal
expression; `CategoryScore` ([score.py:68](../experiment/common/evaluation/score.py#L68))
breaks accuracy down by question class.
*Recommendation:* both keep their prefix and are never called bare "Category".
The evaluation side should be **QuestionCategory** to make the prefix explicit;
`CATEGORIES` / `LONGMEM_CATEGORIES` become `QUESTION_CATEGORIES`.

**3. "Runner" and "Processor" are the same concept under two names.**
`experiment/locomo/pipeline/runner.py` and `MultiDatasetProcessor`
([processor.py:93](../experiment/longmem/pipeline/processor.py#L93)) both plan units of
work and drive per-unit execution. The names diverged because the benchmarks
were written separately.
*Recommendation:* **Runner** is canonical. Rename `MultiDatasetProcessor` →
`DatasetRunner`. Keep `EntityOpsProcessor` — "Processor" there means "adjudicates
a batch", a different and legitimate role.

**4. "qa_eval" does not evaluate.** The stage retrieves and generates; judging
happens in `judge` and scoring happens after that. The name has made every
discussion of "the eval results" ambiguous between generated answers, verdicts,
and accuracy numbers.
*Recommendation:* the concept is **QAEval** and the definition above pins it to
"retrieve + generate". **Renaming the stage is deliberately out of scope** —
`qa_eval` is a CLI value, an artifact directory name, and a column prefix in
every historical result CSV. Fix the prose, keep the identifier.

**5. `ent` / `rel` abbreviations rival the full words in usage.** `rel` 676 uses vs
`relationship` 565; `ent` 389 vs `entity` 1400.
*Recommendation:* full words are canonical. `ENT_FILE` → `ENTITY_CACHE_FILE`,
`REL_FILE` → `RELATIONSHIP_CACHE_FILE`. Safe: the on-disk names are already
`entities_cache.pkl` / `relationships_cache.pkl`
([cache.py:30-31](../grace_mem/storage/cache.py#L30-L31)), so no artifact changes.

**6. "Turn" is declared twice at different granularity.** `SpeakerTurn`
([evidence_speaker_enricher.py:21](../grace_mem/utils/evidence_speaker_enricher.py#L21))
is speaker + text; `Turn` ([corpus.py:23](../experiment/agent_filter/corpus.py#L23)) is
sid-addressable with a position.
*Recommendation:* one concept, two projections — not an ambiguity. Keep both, but
`SpeakerTurn` should be understood as "a **Turn** with only its speaker and text
populated". Do not introduce a third spelling.

**7. Other abbreviation pairs.** Each was checked against the code with an AST
pass separating identifiers from string literals. The outcome differs per pair,
and two of the original rulings were simply wrong.

| Pair | Ruling | State |
| --- | --- | --- |
| `ctx` → **Evidence** / **TimeContext** | per #1 | **Done.** `ctx_dataset`, `ctx_stage` and `ctx_base` stay: they are a token-tracking context and a prompt context, a third and fourth sense of the word, neither of which is Evidence |
| `art` → **Artifact** | spell out | **Done.** 44 identifiers, no artifact schema, nothing documented |
| `vdb` | **VDB** *is* the canonical spelling in identifiers | **Already conforming.** Nothing to change; the original entry implied otherwise |
| `stat` → `stats` | — | **Withdrawn.** All 25 occurrences are `Path.stat()`. The scanner's `stat`/`statuse` pair was an artifact of its own stemming, not a real synonym |
| `data` vs `dataset` | `data` only in path constants | **Already conforming.** `DATA_ROOT`, `SCRIPT_DATA_DIR`, `LOCOMO_DATA`, `DATA_JSON` are path constants; `graph_data` and `export_data` name a payload, not the **Dataset** domain term |
| `meta` → **metadata** | spell out | **Not done.** 21 of 42 names are frozen: `"metas"` is a key inside the BM25 pickle, `"meta"` is a key in `cases/<id>.json`, and `entity_meta`/`rel_meta` are parameters. The safe 21 interleave with them — the same half-pair problem as #5 |
| `vec` → **vector** | for values | **Not done.** `summary_vec_threshold`, `entity_vec_threshold` and `relationship_vec_threshold` are config keys *and* `DatasetConfig` fields and cannot move; `query_vec` is a parameter crossing seven modules. Renaming the rest leaves the two halves disagreeing |
| `qa` outside `qa_eval` | remove | **Not done.** `"qa_json"` is a dataset-kind lookup key shared by the worker, the judge and the path resolver; only 5 names are actually free |
| `eval` | spell out per #4 | **Not done**, and mostly moot — 123 of the occurrences are `qa_eval` itself, which #4 freezes |

The pattern across `meta`, `vec`, `qa` and #5 is the same: a frozen list has to be
decided first — which config keys may change, what compatibility the stored keys
need — before any of them can be swept coherently. That is a design question, not
a rename.

**8. "step" has a third meaning nobody declared.**
`experiment/longmem/helpers/analysis_cases.py` defines `step2_ingest`,
`step3_has_answer` … `step9_evidence` — numbered probes that walk a finished
run's logs and artifacts. They are neither a CLI **Stage** nor a pipeline
**Step**: they run no pipeline and only read what a run left behind.
*Recommendation:* the concept is a **Probe**, defined above. Rename
`stepN_<thing>` → `probe_<thing>`; the numbers encode a reading order that the
call site already imposes, and they make the functions impossible to reorder
without renumbering.

*Correction, found while implementing this.* An earlier version of this entry
said the names "appear in no artifact". They do: `analyze_one` returns a dict
keyed `"step2_ingest"` … `"step9_evidence"`, `collect_cases` writes it to
`cases/<id>.json`, and the summary tooling reads those keys back. **The
functions were renamed; the dict keys were not**, for the same reason `qa_eval`
keeps its name.

---

## Dismissed suspects

Reported by the scanner, checked, and deliberately not glossary entries:

- **Graph** in `tools/gen_dep_graph.py` — a source-dependency graph for dev
  tooling, unrelated to the knowledge graph. `tools/` is outside the domain.
- **Dataset** in `tools/download_datasets.py` (`DatasetFile`) — a pinned download
  with a checksum, not a LongMem run unit. Same reason.
- **Entity** / **Relationship** declared across `pipeline`, `services`, `utils` —
  one concept with layered roles (`*Extractor`, `*Manager`, the model itself),
  which the role-suffix vocabulary already distinguishes. Not ambiguity.
- **IngestStage** / **QAEvalStage** / **JudgeStage** duplicated across
  `locomo/stages/` and `longmem/stages/` — parallel implementations of the same
  stage concept for two benchmarks. A naming collision by design, not by drift.
  (Whether they should share code is a separate question this glossary does not answer.)
- **Judge** as `JudgeEngine` vs `JudgeStage` — engine is the policy, stage is the
  harness phase that applies it. Both defined above.
- **Manager** as a suffix on `EntityManager`, `RelationshipManager`, `VDBManager` —
  consistently "owns and persists one thing". Vague, but uniform; not worth a rename.
- `filter`/`filtering`, `token`/`tokenize`, `final`/`finalize`, `day`/`daypart` —
  verb/noun pairs, not synonyms.

---

## Rename backlog

All of these land as commits on one branch, `refactor/terminology`, ordered by
risk. See the migration plan in
[package-structure.md](package-structure.md) for how that branch sits relative to
the package moves — it goes **before** them, so nothing is renamed into a
directory it is about to leave.

| Order | Commit | Resolves | Risk |
| --- | --- | --- | --- |
| 1 | `refactor(analysis): rename numbered steps to probes` — `stepN_*` → `probe_*` in `longmem/helpers/analysis_cases.py` | #8 | None — one module, no artifact exposure |
| 2 | `refactor(evaluation): qualify question category scores` — `CategoryScore` → `QuestionCategoryScore`, `CATEGORIES` → `QUESTION_CATEGORIES` | #2 | Low — check CSV column names first |
| 3 | `refactor(longmem): rename MultiDatasetProcessor to DatasetRunner` — also `processor.py` → `runner.py` | #3 | Medium — check the LongMem CLI |
| 4 | `refactor(storage): spell out the cache file constants` — `ENT_FILE` → `ENTITY_CACHE_FILE`, `REL_FILE` → `RELATIONSHIP_CACHE_FILE` | #5, in part | None — two constants and their locals; the disk names were already spelled out |
| 5 | `refactor(retrieval): rename ContextFilter to EvidenceFilter` — plus `ctx_*` → `evidence_*` | #1 | Low — 6 references to the class; `ctx_*` lives in 2 files |

**The wider `ent_`/`rel_` sweep is not in this branch, and the risk this file
recorded for it — "None, internal identifiers only" — was wrong.** An AST pass
over `grace_mem`, `experiment` and `tools` classified 97 abbreviated names and
found 38 unrenameable:

- `rel_id`, `rel_desc`, `rel_keywords`, `rel_strength` are **FalkorDB graph
  property names**, read straight out of Cypher records; so is the `KG_REL`
  relationship type label. Renaming them stops existing graphs from being read.
- `ent_topk`, `rel_topk`, `ent_threshold`, `rel_threshold` are one name wearing
  four hats — a config key in `experiment_config.py`, a `DatasetConfig` field, a
  keyword argument, and a parameter — and they reach `run_metadata.json`.

Only 59 came out safe, and they interleave with the frozen ones: `ent_id2meta`
is renameable while `rel_id2meta` is not, because the latter is a keyword
argument. Renaming half a pair leaves the vocabulary *less* consistent than
leaving it alone. A coherent sweep needs the frozen names decided first — which
config keys may change, and what compatibility the graph properties need. That
is a design question, not a rename.

**Also not in this branch:**

- **#4, `qa_eval`.** Not doing it at all — CLI value, artifact directory name, and
  a column prefix in every historical result CSV.
- **#3b, the three `*context*` functions** — `assemble_context_from_query`,
  `build_kg_context`, `_render_context_text`. Their names are mirrored in
  structured log event strings (`"build_kg_context_start"`) that
  `experiment/locomo/analysis/flips.py` and
  `experiment/longmem/analysis/fact_replay.py` match literally. Renaming the
  function orphans the event name; renaming both makes historical logs
  unreadable. Needs its own decision, not a commit in a rename sweep.

Per commit, verify: rename only (no behaviour change) · imports updated · CLI
flags and artifact paths unchanged · ruff, mypy, pytest green · every identifier
touched matches this file.
