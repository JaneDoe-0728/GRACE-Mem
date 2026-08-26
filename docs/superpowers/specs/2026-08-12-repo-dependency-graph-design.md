# Repo Dependency Graph → draw.io

**Date:** 2026-08-12
**Status:** Approved design, not yet implemented
**Revision:** 3 — drawio-skill 2.1.0 installed and its behaviour verified; the
revision-2 assumptions are now replaced by observed facts

## Goal

Produce a package-level dependency graph of GRACE-Mem, rendered as an editable
draw.io diagram, so the architecture can be explained and defended from evidence
rather than from memory. The graph must be regenerable so it does not silently go
stale as the code changes, and every edge on it must be traceable to a line of
source.

## Source of truth

The canonical architecture description is:

```
GRACE-Mem source code
  + ROLLUP          (which directories collapse)
  + MODULE_ALIAS    (how bare package roots are named)
  + LAYER_MAP       (which semantic layer each node belongs to)
  + SUBPROCESS_EDGES (dependencies that exist without an import)
```

`deps.drawio`, `deps.mmd`, `deps.json`, `deps.svg` and `deps.png` are all
**generated artifacts**. None of them is authoritative. Editing `deps.drawio` by
hand in draw.io is not a way to change the architecture record — the next
generator run overwrites it. This distinction matters and is called out in the
file header comment of every generated artifact.

## Why not an off-the-shelf extractor

`pydeps`, `grimp`, and `drawio-skill`'s own `scripts/pyimports.py` all resolve
module-level imports only. A probe over this repo (173 files, `ast`-based) found
**46 of 138** package-level edges exist *only* because of imports written inside
a function body. The affected edges are not incidental — they are the most
important ones in the codebase:

| Edge | Total | Function-local |
|---|---|---|
| `experiment/locomo/pipeline` → `experiment/locomo/stages` | 9 | **9** |
| `experiment/locomo/pipeline` → `grace_mem/llm` | 8 | **8** |
| `experiment/locomo/pipeline` → `experiment/locomo/helpers` | 13 | 11 |
| `experiment/longmem/helpers` → `grace_mem/pipeline/retrieval_steps` | 4 | **4** |

A top-level-only scan renders the worker→stages and worker→engine relationships
as absent. The graph would be confidently wrong. Hence a repository-specific
`ast` walk.

## Diagram tooling

Dependency extraction and graph semantics are owned by `tools/gen_dep_graph.py`.
Neither draw.io MCP nor drawio-skill is a source of architectural truth.

### drawio-skill (`Agents365-ai/365-skills`, v2.1.0)

Installed at `~/.claude/plugins/cache/365-skills/drawio/2.1.0/skills/drawio-skill`.
Behaviour below was verified by reading the scripts and running `autolayout.py`,
not taken from the README.

**`autolayout.py` consumes graph JSON, not `.drawio`.** Its input schema is:

```jsonc
{ "direction": "TB",                       // or "LR"
  "nodes": [ { "id": "...", "label": "...", "style": "...",
               "width": 120, "height": 60,
               "group": "Engine/Foundation",   // hierarchical path
               "groupLabel": "...", "link": "..." } ],
  "edges": [ { "source": "...", "target": "...",
               "label": "...", "style": "..." } ] }
```

This is a **simplification**, not an obstacle: we never write mxGraph XML at all.
`autolayout.py` runs `dot`, fills in geometry, draws the group containers, and
emits the `.drawio`.

Consequences for this design:

- **Groups are native.** `node.group` takes a hierarchical path (`"Engine/Foundation"`),
  parsed by `group_tree()` into nested Graphviz `subgraph cluster_N` blocks with
  `margin=24`, then drawn as bounding boxes with palette colours per top-level
  group. Our `LAYER_MAP` therefore emits directly as `group` strings; no swimlane
  construction on our side. Verified: an 8-node fixture with 4 groups produced 14
  `group_*` cells including correct nesting.
- **Per-edge `style` is supported** (`autolayout.py:316`,
  `edge.get("style", EDGE_STYLE)`). Dashed subprocess edges, `strokeWidth` scaled
  by `source_file_count`, and faint sink edges are all expressible. Verified in
  the fixture output.
- **Tooltips are not supported anywhere in the skill.** No script emits a
  `tooltip` attribute. See *Evidence*.
- **`node.link` is supported** via `UserObject`, so nodes can be made clickable.

**`restyle.py` is explicitly built for externally-produced diagrams.** Its
docstring: *"Re-theme an EXISTING .drawio with a style preset — layout and shapes
untouched … the post-processor for diagrams that already exist."* It remaps fills
and strokes to a preset palette and leaves edge routing and geometry alone.

Its generic Python import extractor (`pyimports.py`) is **not** used, for the
reason given above.

**Prerequisites:** Graphviz (`dot`) — installed, 2.43.0. draw.io desktop CLI —
required only for SVG/PNG/PDF export.

### jgraph/drawio-mcp

Optional preview surface. The generated Mermaid may be passed to `create_diagram`
for quick interactive inspection. It does not participate in extraction, layout,
or export, and is not required to regenerate any committed artifact.

## Scope

**In:** `grace_mem/`, `experiment/`, `tools/`
**Out:** `tests/` (60 files, reverse dependency direction, would connect to every
node), `noco-db-uploader/` (independent subproject), `.venv/`, `models/`, `logs/`,
`vdb_cache/`, `__pycache__/`

## Architecture

Extraction and emission use the standard library only (`ast`, `pathlib`,
`collections`, `json`, `xml.etree.ElementTree`). No new project dependencies.
Graphviz and the draw.io CLI are needed only for the layout/export stage, which
runs after the generator.

```
tools/gen_dep_graph.py
  scan()                walk *.py under the in-scope roots; collect Import /
                        ImportFrom, resolving relative imports; record file:line
                        for each occurrence and whether it is nested inside a
                        FunctionDef / AsyncFunctionDef
  rollup()              apply ROLLUP, then MODULE_ALIAS: 40 raw nodes → 29
  add_subprocess_edges()  merge the hand-declared SUBPROCESS_EDGES registry
  assign_layers()       apply the explicit LAYER_MAP.  Layer membership is
                        semantic and intentionally hand-maintained.
  order_within_layers() order nodes within each layer by descending fan-in,
                        lexical path as deterministic tie-breaker
  emit_json()        → docs/architecture/deps.json    (canonical, full evidence)
  emit_mermaid()     → docs/architecture/deps.mmd
  emit_layout_json() → docs/architecture/deps.layout.json
                       (autolayout.py's input schema: nodes with group paths
                        and style strings, edges with style strings)
```

Each function takes data and returns data; only `main()` touches the filesystem.
That keeps `scan`, `rollup`, `assign_layers`, and `order_within_layers`
unit-testable without fixtures on disk.

**This script never writes mxGraph XML.** `deps.drawio` is produced by
`autolayout.py` from `deps.layout.json`. That removes roughly 150 lines of XML
emission and all coordinate maths from our side, and it removes an entire class
of malformed-XML bug.

## Data model

```python
Node = str          # canonical package path after rollup + alias

NodeMeta = {
    "file_count":  int,
    "total_lines": int,
    "layer":       str,     # LAYER_MAP key
    "is_sink":     bool,
    "real_paths":  list[str],   # directories folded into this node
}

EdgeMeta = {
    "import_count":      int,        # total import statements
    "source_file_count": int,        # distinct .py files containing them
    "nested_count":      int,        # of those, how many are function-local
    "source_refs":       list[str],  # "path/to/file.py:123", sorted
}

SubprocEdge = {
    "src": Node, "dst": Node, "label": str, "source_ref": str,
}

Graph = {
    "nodes":            dict[Node, NodeMeta],
    "import_edges":     dict[tuple[Node, Node], EdgeMeta],
    "subprocess_edges": list[SubprocEdge],
}
```

`import_count` and `source_file_count` are deliberately separate. Given:

```python
# a.py
from grace_mem.storage import MGR
from grace_mem.storage import VDBManager
```

`import_count == 2` but `source_file_count == 1`. **Edge thickness uses
`source_file_count`**, because five different modules depending on `storage` is
stronger architectural coupling than one module importing it five times.

## Rollup and aliasing

The raw scan yields 40 package nodes. Two transformations reduce this to **29**.

**ROLLUP** folds leaf directories that exist for file organisation rather than
architectural meaning:

```python
ROLLUP = {
    "grace_mem/llm/prompts":            "grace_mem/llm",
    "grace_mem/llm/prompts/adaptive":   "grace_mem/llm",
    "grace_mem/llm/prompts/entity_ops": "grace_mem/llm",
    "grace_mem/llm/prompts/extraction": "grace_mem/llm",
    "grace_mem/llm/prompts/keyword":    "grace_mem/llm",
    "tools/manual":                     "tools",
    "tools/agent_filter_trace_viewer":  "tools",
    "experiment/locomo/utils":      "experiment/locomo/support",
    "experiment/locomo/prompts":    "experiment/locomo/support",
    "experiment/locomo/artifacts":  "experiment/locomo/support",
    "experiment/longmem/utils":     "experiment/longmem/support",
    "experiment/longmem/prompts":   "experiment/longmem/support",
    "experiment/longmem/artifacts": "experiment/longmem/support",
}
```

**MODULE_ALIAS** renames bare package roots. A node labelled `grace_mem` reads as
"the whole grace_mem package", but it is in fact only `grace_mem/embeddings.py`.
All four bare roots have this problem:

```python
MODULE_ALIAS = {
    # actual contents (excluding __init__.py)      → displayed as
    "grace_mem":          "grace_mem/embeddings",   # embeddings.py
    "experiment":         "experiment/config",      # experiment_config.py
    "experiment/locomo":  "experiment/locomo/cli",  # cli.py, models.py
    "experiment/longmem": "experiment/longmem/models",  # models.py
}
```

These are virtual architectural nodes; they need not correspond to a directory,
because the diagram depicts package architecture, not a directory tree. The
`real_paths` field in `NodeMeta` records what each node actually covers, and
`experiment/locomo/cli` carries the subtitle `cli.py, models.py` so the two-file
case is not hidden.

`grace_mem/utils/temporal` is deliberately **not** rolled into `grace_mem/utils`.
It is a 1326-line temporal resolver and a distinct capability of the system;
collapsing it would hide the thing most worth pointing at.

Self-edges created by rollup or aliasing are dropped.

## Edges

Two visually distinct edge types.

### Import edges

Solid; stroke width `1 + min(source_file_count, 8) / 3`. Derived entirely from
the AST scan.

**Relative imports are resolved, not skipped.** An earlier revision of this spec
claimed the repo uses absolute imports throughout; that was false. The repo
contains 34 relative imports (28 under `grace_mem/`, 6 under `experiment/`).
Resolution walks `ImportFrom.level` up from the containing package and maps the
result through the same rollup pipeline.

For the current codebase this changes nothing on the diagram: 30 of the 34 are
intra-package `__init__.py` re-exports, and the remaining 4 are
`grace_mem/llm/prompts/__init__.py` importing its own subpackages — all of which
ROLLUP folds into `grace_mem/llm`. Every one becomes a self-edge and is dropped.
The handling is correct rather than merely convenient, and it protects against a
future cross-package relative import being silently lost. A relative import that
resolves outside the scanned roots emits a warning rather than being discarded
in silence.

### Subprocess edges

Dashed, hand-declared, each carrying a source reference:

```python
SUBPROCESS_EDGES = [
    # runner.py:200 — subprocess.run(build_worker_command(...))
    ("experiment/locomo/pipeline", "experiment/locomo/pipeline",
     "self-spawn: --worker", "experiment/locomo/pipeline/runner.py:200"),
    # watchdog.py:510 — subprocess.Popen([py, "-m", BATCH_MODULE])
    ("experiment/longmem/pipeline", "experiment/longmem/pipeline",
     "spawn batch (MDQA_* env)", "experiment/longmem/pipeline/watchdog.py:510"),
    # run_hooks.py:95 — subprocess.run([python, "-c", "from tools.refresh_system
    # import refresh_system; refresh_system()"]).  LoCoMo never imports tools/
    # in-process; it shells out.  No AST edge exists, so it must be declared.
    ("experiment/locomo/helpers", "tools",
     "refresh_system() via python -c",
     "experiment/locomo/helpers/run_hooks.py:95"),
]
```

The first two are self-loops: `runner` and `watchdog` each spawn a module inside
their own package. These render as a loop badge on the node rather than an edge
crossing the canvas.

The third is exactly the failure mode this design exists to avoid. LoCoMo reaches
`tools/refresh_system` by spawning `python -c "..."` from `helpers/run_hooks.py` —
there is no import anywhere, so *every* import-based tool, including our own AST
scanner, sees no relationship at all. LongMem reaches the same module differently:
`watchdog.py:1055` does a real function-local
`from tools.refresh_system import refresh_system`, which the scanner picks up
automatically and which therefore must **not** be duplicated here. Two entry
points, same dependency, two mechanisms, only one mechanically discoverable.

This registry is hand-maintained and will drift if new subprocess call sites
appear. Accepted: there are three, they are architectural, they change rarely.
See *Risks*.

## Evidence

Every edge carries `source_refs` — not only subprocess edges. This is what makes
the Goal's "defended from evidence" claim real: any line on the diagram can be
resolved to specific `file.py:line` locations.

`deps.json` carries this completely and untruncated. That alone satisfies the
goal for anyone at a terminal.

On the diagram itself it is harder: **drawio-skill emits no `tooltip` attribute
anywhere**, so hovering an edge shows nothing. Since the diagram is the thing
shown in a meeting, and "which line proves this?" is exactly the question that
gets asked there, a small post-pass closes the gap:

`tools/annotate_evidence.py` reads the `.drawio` produced by `autolayout.py` and
rewrites each edge `mxCell` as a `UserObject` carrying a `tooltip`:

```
imports: 8   source files: 3   (all function-local)
experiment/locomo/pipeline/worker.py:698
experiment/locomo/pipeline/worker.py:736
experiment/locomo/pipeline/runner.py:159
```

Truncated to the first 10 refs with a `+N more` marker; `deps.json` has the rest.
This is ~30 lines of `ElementTree`, it touches only edge cells, and it runs
before `restyle.py` (which only remaps vertex fills and strokes, so the two do
not collide).

## Noise reduction

`grace_mem/utils` has the highest fan-in in the repo — nearly every package
depends on it, and drawing all of those at full weight turns the centre of the
diagram into a star. Utility nodes are marked `is_sink=True`; their **inbound
edges render faint and thin** (`strokeColor=#B0B0B0;opacity=40`) and their node
label gains a `used by: N` badge.

No edge is deleted. The information stays on the diagram, in the tooltips, and in
`deps.json`; only its visual weight is reduced.

Sink set: `grace_mem/utils`, `experiment/locomo/support`,
`experiment/longmem/support`.

## Semantic layers

`LAYER_MAP` is explicit and hand-maintained. Inferring layers from the import
graph would place `grace_mem/utils` at the bottom by fan-in — correct — but would
scatter the `experiment/` nodes according to accidents of import direction rather
than according to what they mean.

All 29 nodes are assigned; this table is exhaustive.

| Group | Count | Nodes |
|---|---|---|
| Setup & peripheral | 1 | `tools` |
| Benchmark orchestration — entry | 2 | `experiment/locomo/pipeline`, `experiment/longmem/pipeline` |
| Benchmark orchestration — stages | 9 | `experiment/locomo/{stages,helpers,support,cli}`, `experiment/longmem/{stages,helpers,support,models,tools}` |
| Shared evaluation & config | 3 | `experiment/config`, `experiment/common`, `experiment/common/evaluation` |
| Optional runtime | 1 | `experiment/agent_filter` |
| Diagnostics (post-run) | 2 | `experiment/locomo/analysis`, `experiment/longmem/analysis` |
| Engine — facade | 2 | `grace_mem/pipeline`, `grace_mem/embeddings` |
| Engine — steps | 2 | `grace_mem/pipeline/{ingest_steps,retrieval_steps}` |
| Engine — foundation | 7 | `grace_mem/{services,storage,graph,llm,utils,runtime}`, `grace_mem/utils/temporal` |

Two groups are deliberately **off the main vertical stack**, drawn as a side
column:

- **Diagnostics.** `*/analysis` is post-run tooling, not a benchmark execution
  stage. Placing it inline between pipeline and engine would imply that a
  benchmark run passes through it. It does not.
- **Optional runtime.** `experiment/agent_filter` participates in answering only
  when enabled (`processor._maybe_refine_with_grep_agent`); drawing it inline
  would imply it is always on.

`experiment/longmem/tools` stays in orchestration: it is dataset preparation and
rerun tooling that feeds the pipeline, not post-run diagnostics.

`LAYER_MAP` emits directly as `autolayout.py`'s hierarchical `group` paths — we
build no containers ourselves:

```python
LAYER_MAP = {
    "tools":                        "Peripheral",
    "experiment/locomo/pipeline":   "Orchestration/Entry",
    "experiment/longmem/pipeline":  "Orchestration/Entry",
    "experiment/locomo/stages":     "Orchestration/Stages",
    # ...
    "experiment/locomo/analysis":   "Diagnostics",
    "experiment/agent_filter":      "Optional runtime",
    "grace_mem/pipeline":           "Engine/Facade",
    "grace_mem/pipeline/retrieval_steps": "Engine/Steps",
    "grace_mem/utils":              "Engine/Foundation",
    # ...
}
```

`group_tree()` parses the `/` nesting into nested Graphviz clusters, and each
top-level group (`Engine`, `Orchestration`, …) gets its own palette colour
automatically. Verified against a fixture: 4 groups with nesting produced correct
containers.

## Determinism

Revision 2 assumed that handing geometry to `autolayout.py` would cost
byte-identical `.drawio` output. **Measured: it does not.** Running
`autolayout.py` twice over the same 8-node fixture produced byte-identical files
(`cmp` clean). `dot` is deterministic for a fixed input, fixed version, and fixed
direction, and `--tune`'s TB/LR choice is itself a deterministic scoring pass.

So the diff contract survives on both artifacts. Two caveats keep `deps.json` as
the *primary* diff target:

- a Graphviz version bump can shift coordinates, whereas `deps.json` is immune;
- `deps.drawio` carries geometry noise that obscures semantic change in a diff.

**`deps.json` contract:**

```
nodes sorted by canonical path
edges sorted by (source, target, edge_type)
source_refs sorted lexically within each edge
node id  = "node:" + canonical_path        (never uuid4)
edge id  = f"edge:{src}->{dst}:{edge_type}"
no timestamps, no hostnames, no absolute paths
json.dump(..., sort_keys=True, indent=2, ensure_ascii=False)
```

Two runs over unchanged source produce byte-identical `deps.json`. That makes
`git diff docs/architecture/deps.json` the signal that the architecture moved.
`deps.mmd` and `deps.layout.json` inherit the same ordering.

Node and edge IDs are stable across all formats, so a node in `deps.json` can be
located in `deps.drawio` by id.

The Graphviz version is recorded in `deps.json` under `meta.graphviz_version`
(currently `2.43.0`) so an unexplained visual change can be attributed.

## Pipeline

```
GRACE-Mem source
      │
      ▼
tools/gen_dep_graph.py  ── canonical Graph (in memory)   [ours, stdlib only]
      │
      ├──────────────┬──────────────────────┐
      ▼              ▼                      ▼
 deps.json       deps.mmd            deps.layout.json
 (diff target)   (preview)                  │
      │              │                      ▼
      │              │        autolayout.py            [drawio-skill]
      │              │                      │
      │              │                      ▼
      │              │              deps.drawio  (geometry + groups)
      │              │                      │
      │              │        annotate_evidence.py     [ours, ~30 lines]
      │              │                      │  edge tooltips from deps.json
      │              │                      ▼
      │              │        restyle.py --preset ...  [drawio-skill]
      │              │                      │
      │              │        draw.io CLI export
      │              │                      │
      │              │              ┌───────┴───────┐
      │              │              ▼               ▼
      │              │          deps.svg        deps.png
      │              ▼
      │       drawio-mcp (optional inline preview)
      ▼
 CI / --check
```

## Outputs

Under `docs/architecture/`:

| File | Role | Deterministic | Committed |
|---|---|---|---|
| `deps.json` | canonical graph + full evidence; **the diff target** | yes | yes |
| `deps.mmd` | Mermaid `flowchart TD`, quick preview | yes | yes |
| `deps.drawio` | editable diagram, laid out | yes, per Graphviz version | yes |
| `deps.svg` | rendered, for docs/README embedding | — | yes |
| `deps.png` | rendered, for slides | — | yes |
| `deps.layout.json` | intermediate handoff to `autolayout.py` | yes | no (gitignored) |

All derive from the same in-memory `Graph`, so they cannot disagree about nodes
or edges.

## Usage

```bash
# 1. extract  (ours, stdlib only — no graphviz, no draw.io needed)
uv run python -m tools.gen_dep_graph
#    → deps.json, deps.mmd, deps.layout.json

# 2. lay out  (drawio-skill; needs graphviz)
SKILL=~/.claude/plugins/cache/365-skills/drawio/2.1.0/skills/drawio-skill
python3 $SKILL/scripts/autolayout.py docs/architecture/deps.layout.json \
        -o docs/architecture/deps.drawio

# 3. annotate + theme
uv run python -m tools.annotate_evidence docs/architecture/deps.drawio
python3 $SKILL/scripts/restyle.py docs/architecture/deps.drawio --preset default

# 4. export  (needs draw.io CLI)
drawio -x -f svg -o docs/architecture/deps.svg docs/architecture/deps.drawio
drawio -x -f png -s 2 -o docs/architecture/deps.png docs/architecture/deps.drawio
```

Steps 2–4 get wrapped in `tools/build_dep_graph.sh` so the whole thing is one
command. Step 1 stands alone and is the only step with tests.

`--check` compares only `deps.json`. This makes CI wiring possible later; not
wired up as part of this work.

## Testing

`tests/test_gen_dep_graph.py`:

1. **Nested import detection** — an inline fixture with a function-local `import`
   produces an edge. This is the property off-the-shelf tools fail; it gets a
   dedicated test.
2. **Relative import resolution** — `from .foo import x` in `pkg/a.py` resolves to
   `pkg/foo`, and an out-of-scope relative import warns rather than vanishing.
3. **Rollup** — `grace_mem/llm/prompts/keyword` maps to `grace_mem/llm`; the
   resulting self-edge is dropped.
4. **Aliasing** — the bare root `grace_mem` becomes `grace_mem/embeddings`, and
   `real_paths` records what it covers.
5. **`grace_mem/utils/temporal` survives rollup** — guards against someone
   "tidying" the ROLLUP table later.
6. **Weight semantics** — a fixture file with two imports of the same target
   yields `import_count == 2`, `source_file_count == 1`.
7. **`deps.json` determinism** — two runs produce identical bytes; no `uuid`,
   timestamp, or absolute path appears in the output.
8. **Well-formed XML** — `deps.drawio` parses via `ElementTree`, and every edge's
   `source`/`target` resolves to a declared node id.
9. **LAYER_MAP completeness** — every node produced by `rollup()` over the real
   repo has a layer. Fails loudly when someone adds a package, which is the
   intended prompt to update the table.
10. **Evidence completeness** — every import edge in `deps.json` has at least one
    `source_ref`, and each ref points at an existing file.
11. **`deps.layout.json` matches autolayout's schema** — a fixture pinning the
    v2.1.0 field set (`nodes[].id/label/style/group`, `edges[].source/target/
    style`), so a skill upgrade that changes the contract fails here rather than
    producing a silently wrong diagram.
12. **Tooltip post-pass** — `annotate_evidence.py` turns edge `mxCell`s into
    `UserObject`s with a `tooltip`, leaves vertex cells untouched, and the result
    still parses.

Tests 1–7, 9, 10 and 11 need neither Graphviz nor draw.io. Tests 8 and 12 run
against a small fixture put through `autolayout.py`, so they are skipped when
`dot` is absent.

## Risks

- **`SUBPROCESS_EDGES` drifts.** Mitigated by a companion check: the script greps
  for `subprocess.run(` / `subprocess.Popen(` under the in-scope roots and warns
  if the count exceeds the number of declared edges. A warning, not a failure —
  the point is to prompt a human, not to block.
- **`LAYER_MAP` drifts.** Test 9 fails on unassigned nodes.
- **Graphviz version changes shift the layout.** The only remaining source of
  non-determinism, and it is out of our control. `deps.json` is the diff target
  and is immune; `meta.graphviz_version` records the version used.
- **drawio-skill's input schema changes on upgrade.** `deps.layout.json` is
  written to `autolayout.py`'s v2.1.0 schema. A breaking change there breaks step
  2 of the build, not the extraction. Pin the observed schema in a test fixture
  so an upgrade fails loudly.
- **Someone hand-edits `deps.drawio`.** Their changes are lost on regeneration.
  Mitigated by a header comment in the file and the *Source of truth* section.
- **Dynamic imports via `importlib` are invisible.** The repo does not currently
  use `importlib.import_module` for first-party modules; if that changes the
  graph will under-report. Not worth solving until it happens.
- **`__getattr__`-based lazy exports** (`grace_mem/storage/__init__.py`,
  `grace_mem/services/__init__.py`) mean `from grace_mem.storage import MGR`
  produces an edge to `grace_mem/storage`, not to `storage/chroma_manager`. At
  package granularity that is the correct answer. It would matter at module
  granularity.

## Verification log

Revision 2 listed three assumptions about drawio-skill taken from its README.
The skill is now installed (v2.1.0) and all three were checked directly.

| Revision-2 assumption | Verdict | Consequence |
|---|---|---|
| `autolayout.py` accepts an externally-produced `.drawio` | **False** — it takes graph JSON and *emits* `.drawio` | Better than assumed. We drop `emit_drawio()` entirely and write `deps.layout.json` instead; ~150 lines of XML emission and all coordinate maths leave our codebase |
| `autolayout.py` honours pre-existing swimlane containers | **Reframed** — no containers needed; `node.group` takes a hierarchical path and it builds nested Graphviz clusters itself | `LAYER_MAP` emits straight to `group` strings. Verified: 4 groups with nesting → 14 correct `group_*` cells |
| `restyle.py` applies to a diagram it did not generate | **True**, and explicitly designed for it | Used as planned |

Two further facts, neither assumed in revision 2:

| Finding | Consequence |
|---|---|
| Per-edge `style` **is** supported (`autolayout.py:316`) | Dashed subprocess edges, `strokeWidth` by `source_file_count`, faint sink edges all work with no post-pass |
| No script in the skill emits a `tooltip` attribute | Evidence-on-hover needs `tools/annotate_evidence.py` (~30 lines). See *Evidence* |

And the determinism measurement that reverses a revision-2 trade-off:

| Measurement | Result |
|---|---|
| `autolayout.py` run twice over one fixture, `cmp` | **Byte-identical.** The byte-identical `.drawio` given up in revision 2 was never actually lost |

Reproduction of the determinism check is kept as test 11.

Environment as verified: Graphviz 2.43.0, drawio-skill 2.1.0, draw.io desktop
31.1.8 (downloaded; install pending).

## Out of scope

- Module-level (per-file) graph. A per-file view is a separate diagram with
  different readability constraints.
- Test coverage graph.
- Data-flow / artifact-handoff edges (`artifacts/`, eval CSVs, checkpoints).
  Considered and rejected: three edge types over 29 nodes is unreadable. These
  belong on separate pages — *LoCoMo execution flow*, *LongMem execution flow*,
  *Artifact lifecycle* — if wanted later.
- CI integration of `--check`.
- Live editing via a browser-driving MCP server.
