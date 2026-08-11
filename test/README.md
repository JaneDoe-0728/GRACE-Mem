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

Nine historical `test_*.py` files call live APIs, local model endpoints, or
large reranker models directly. They are scripts, not pytest tests, and are
listed explicitly as `MANUAL_SCRIPT_NAMES` in `conftest.py`. Run one directly
only after configuring the service or model it requires, for example:

```bash
uv run python test/test_inference_keyword.py
```

Do not add an automated test to `collect_ignore` to work around a missing module
or failing behavior. Automated contracts must collect and either pass, skip on
an explicit runtime prerequisite, or carry a narrow `xfail` with a reason.
