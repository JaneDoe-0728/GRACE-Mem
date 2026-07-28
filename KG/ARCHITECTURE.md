# KG Architecture

`KG/` is the core knowledge-memory package. The current refactor goal is to keep each subpackage responsible for one kind of work and to keep imports stable at the package boundary.

## Package Roles

### `KG/llm`
- Owns model-facing logic.
- Public API: `from KG.llm import LLMClient, token_tracker`
- Main responsibilities:
  - OpenAI-compatible client setup
  - token usage tracking
  - entity operation prompting / parsing
  - prompt assets under `KG/llm/prompts/`

Use this package when the code is about prompting, model calls, retries, or output parsing. Avoid mixing storage or graph mutation logic into `llm`.

### `KG/services`
- Owns domain mutation and orchestration logic.
- Public API: `from KG.services import EntityManager, RelationshipManager, Provenance`
- Main responsibilities:
  - normalize extracted entities / relationships
  - deduplicate or merge graph facts
  - attach provenance
  - prepare writes for storage and graph sync

Use this package when the code is about knowledge semantics and update rules. Avoid putting persistence-path decisions or prompt-building here.

### `KG/storage`
- Owns persistence and artifact-backed indexes.
- Public API: `from KG.storage import MGR, VDBManager, CacheStore, build_id_to_meta_maps`
- Main responsibilities:
  - Chroma / vector DB lifecycle
  - BM25 persistence
  - cache load/save/reset
  - artifact path resolution

`KG/storage/artifacts` is the canonical runtime artifact directory. Top-level `KG/artifacts` is legacy-only and should not be used by new code.

### `KG/pipeline`
- Owns end-to-end assembly and workflow entry points.
- Main responsibilities:
  - build shared runtime objects in `factory.py`
  - ingest turns into memory
  - retrieve context from memory
  - delegate detailed retrieval behavior to `retrieval_steps/`

This package should orchestrate other packages, not become the home for low-level utility logic.

### `KG/graph`
- Owns graph database adapters and graph sync behavior.
- Main responsibilities:
  - graph connection setup
  - schema initialization
  - graph read/write sync helpers

### `KG/utils`
- Owns cross-cutting helpers that do not clearly belong to `llm`, `services`, `storage`, or `graph`.
- Examples:
  - logging helpers
  - time parsing
  - reranking helpers
  - shared parsing / utility functions

Keep `utils` small. If a helper is mainly storage-specific or llm-specific, prefer moving it back into that package.

`KG/utils/temporal/*` is now the canonical home for shared temporal parsing and
resolution. `KG/utils/query_time_parser.py` remains a compatibility wrapper for
older call sites while migration is in progress.

## Runtime Data

### Canonical artifacts path
- Canonical path: `KG/storage/artifacts`
- Legacy path: `KG/artifacts`

The storage manager now migrates missing legacy payload from `KG/artifacts` into `KG/storage/artifacts` when needed. New code should never target `KG/artifacts` directly.

### What lives in artifacts
- vector DB files and directories
- cache pickle files
- BM25 index files
- metadata jsonl exports
- graph export snapshots used by some experiment flows

Artifacts are generated runtime state, not source code.

## Import Rules

Prefer package-level imports for the public surface:

```python
from KG.llm import LLMClient, token_tracker
from KG.services import EntityManager, RelationshipManager, Provenance
from KG.storage import MGR, VDBManager, CacheStore, build_id_to_meta_maps
```

Use deeper module imports only for internal implementation details inside the owning package.

## Practical Boundaries

- If the change touches prompts or model output shape, start in `KG/llm`.
- If the change decides how entities or relationships are merged, start in `KG/services`.
- If the change touches cache files, vector DBs, or artifact paths, start in `KG/storage`.
- If the change mostly wires components together, start in `KG/pipeline`.
- If the change touches graph schema or sync, start in `KG/graph`.

## Current Cleanup Status

- `KG/services` snake_case module rename is complete.
- Package-level public APIs for `KG.llm`, `KG.services`, and `KG.storage` are in place.
- `KG/storage/artifacts` is canonical.
- `KG/llm/prompts.py.backup` has been removed.

The main remaining cleanup risk is old local data still sitting under `KG/artifacts`, not active code structure.
