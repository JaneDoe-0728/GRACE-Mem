# Test Suite

The default suite is deterministic and does not require API credentials, a live
FalkorDB instance, or downloaded model weights:

```bash
uv run pytest -q
```

`pyproject.toml` enables strict configuration and marker validation. Package
dependency rules, cycle detection, canonical imports, and import-time
`sys.path` behavior are enforced in `test_architecture.py`.

## Result Categories

- `passed`: automated regression tests executed successfully.
- `skipped`: optional integration behavior whose declared prerequisite is not
  available in the current environment.
- `xfailed`: a known temporal parser limitation recorded as an expected failure.
  An unexpected pass is reported by pytest so the expectation can be removed.

## Manual Probes

Historical scripts that directly call live APIs, local endpoints, or large
models are excluded from the public regression suite. The maintained Agent
Filter smoke probe lives outside `tests/` and must be run explicitly after its
services are configured:

```bash
uv run python -m experiment.agent_filter.smoke
```

Automated contracts must collect and either pass, skip on an explicit runtime
prerequisite, or carry a narrow `xfail` with a reason. Do not hide a regression
with collection exclusions.
