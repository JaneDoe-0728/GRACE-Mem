# Python dependency graphs

These diagrams are generated from Python imports by
[`pydeps`](https://github.com/thebjorn/pydeps) and rendered by Graphviz.

Read an arrow as **importer → dependency**. Blue boxes are nodes that belong to
a dependency cycle at the aggregation level shown; the other nodes are regular
packages or modules. A cycle in a rolled-up package view does not necessarily
mean that the individual Python modules form an import-time cycle.

## Recommended viewing order

1. [`repo-overview.svg`](repo-overview.svg) — the three top-level project areas.
2. [`grace-mem.svg`](grace-mem.svg) — the runtime engine, rolled up to its
   immediate packages.
3. [`experiment-overview.svg`](experiment-overview.svg) — shared experiment
   code and the two benchmark families.
4. [`experiment-locomo.svg`](experiment-locomo.svg) and
   [`experiment-longmem.svg`](experiment-longmem.svg) — one additional package
   level inside each benchmark.
5. [`tools.svg`](tools.svg) — dependencies among tool modules.

PNG copies are included for quick previews. SVG is preferable for inspection
because labels stay sharp while zooming. Every view also has a `.dot` source.

`repo-modules.json` is the uncoalesced internal dependency analysis emitted by
pydeps. Use it when a rolled-up edge needs to be traced back to a module or file.

Python standard-library and third-party modules are intentionally omitted from
the visual views. Including them makes application structure much harder to
see; the declared third-party library list remains in `pyproject.toml`.

## Rebuild

From the repository root:

```bash
python -m tools.build_pydeps_graph
```

The builder uses an installed `pydeps` command when available. Otherwise it
runs pinned `pydeps==3.0.7` in an isolated `uvx` environment, so rebuilding does
not add pydeps to the application's runtime dependencies. Graphviz `dot` must
be on `PATH`.
