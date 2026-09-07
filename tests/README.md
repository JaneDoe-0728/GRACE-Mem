# Test Suite

The default suite is deterministic and does not require API credentials, a live
FalkorDB instance, or downloaded model weights:

```bash
uv run pytest -q
```

`pyproject.toml` enables strict configuration and marker validation.

## Layout

| Directory | What it holds |
|---|---|
| `contracts/` | The claims the repository makes to a reader: the package dependency rules (`test_package_boundaries.py`) and whether the READMEs still describe the code (`test_readme_claims.py`) |
| `engine/` | `grace_mem` behaviour: retrieval, evidence assembly, ingestion, durable writes |
| `agent_filter/` | The agent loop, its reply protocol, and its configuration |
| `benchmark/` | The `experiment` layer: judge rubric selection, snapshot resume |
| `support/` | Fakes and the import-graph helper, plus `snapshots/` |

`support/snapshots/` holds golden files, one directory per module. Each records
both the sequence of calls the code made to its stores and the text it rendered,
so a change to either shows up as a diff. They are generated, not written by
hand: `KG_UPDATE_EVIDENCE_SNAPSHOTS=1 uv run pytest` rewrites them, and the
diff is what you review.

Paths are derived from `support/paths.py` rather than from each test's own
`parents[N]`, so a test can move between these directories without its file
lookups caring.

## Result Categories

- `passed`: automated regression tests executed successfully.
- `skipped`: optional integration behavior whose declared prerequisite is not
  available in the current environment.
- `xfailed`: a known temporal parser limitation recorded as an expected failure.
  An unexpected pass is reported by pytest so the expectation can be removed.

## What is in version control

This directory is excluded by default and a deterministic subset is
force-tracked; that subset is what a fresh clone runs and what CI runs on every
pull request. The larger local suite -- including probes that need a live
endpoint, FalkorDB, or downloaded weights -- stays out of version control, so a
green run here is narrower than a green run on a maintainer's machine.

Automated contracts must collect and either pass, skip on an explicit runtime
prerequisite, or carry a narrow `xfail` with a reason. Do not hide a regression
with collection exclusions.
