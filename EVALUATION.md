# Evaluation Protocol

GRACE-Mem uses one post-hoc judge entrypoint for LoCoMo and LongMemEval:

```bash
uv run python experiment/common/evaluation/judge.py --help
```

The benchmark pipelines can run a configured inline judge and write a
`correctness` column. Paper results should instead use the standardized post-hoc
protocol below so answer generation and judging remain independent.

## Judge Configuration

The default judge is `gpt-4o-mini` through `https://api.openai.com/v1`. Set
`OPENAI_API_KEY` in the environment or `.env`. A compatible endpoint and model
can be selected explicitly:

```bash
--judge-base-url <openai-compatible-url> --judge-model <model-name>
```

Record any override with reported results. Scores produced by different judge
models or rubrics are not directly comparable.

## Voting Rule

The default protocol is `--votes 3`:

1. Judge every answer once at temperature `0.0`.
2. Carry a correct first verdict forward without more API calls.
3. Rejudge an incorrect first verdict with three independent votes at
   temperatures `0.0`, `0.3`, and `0.6`.
4. Use the three-vote majority as the final verdict.

This is an error-focused rejudging protocol, not full-dataset three-vote
judging. Use `--votes 1` only when a single-vote comparison is required.

Calls retry transient HTTP `429` and `5xx` responses and network failures with
bounded exponential backoff. Completed `0`/`1` cells are skipped, so an
interrupted run can resume with the same command.

## LoCoMo

Run the standardized judge after answer generation:

```bash
uv run python experiment/common/evaluation/judge.py locomo <run-tag> \
  --samples 0-9 \
  --workers 8
```

Input is discovered under
`experiment/locomo/output/standard/<run-tag>/sample_<id>/`. The command accepts
raw `*_eval_*.csv`, legacy `*_judge.csv`, or resumable
`*_judge_4omini.csv` files.

The judge uses the LoCoMo standard rubric and adds an absolute-date hint for
supported relative temporal gold answers. It writes:

- `correctness_4omini`: first-pass verdict;
- `correctness_3vote`: final carry/rejudge verdict;
- `_correctness_aggregate_correctness_3vote.json`: run-level score.

Adversarial questions are excluded from the aggregate by default. Pass
`--include-adversarial` to include them.

## LongMemEval

Run the standardized judge on a completed LongMemEval output directory:

```bash
uv run python experiment/common/evaluation/judge.py longmem <run-tag> \
  --workers 8
```

The command discovers one-question CSVs under the six category directories in
`experiment/longmem/output/<run-tag>/`. General questions use the category-aware
LongMemEval rubric and write `correctness_3vote`.

Files whose stem ends in `_abs` are abstention questions. They use the dedicated
abstention rubric, always receive one temperature-`0.0` vote, and write
`correctness_absrubric`. They are never carried from a general-rubric verdict or
sent through majority voting.

`_correctness_aggregate_judge.json` reports the final score by combining:

- `correctness_3vote` for general questions;
- `correctness_absrubric` for `_abs` questions.

The aggregate also records the selected columns, overall count, and per-category
counts. Supplying `--column <name>` intentionally replaces this protocol with a
single custom result column and is recorded as a custom score.

## Scoring Existing Runs

Use the shared scorer for either benchmark. It applies the final result columns,
prints overall and per-category accuracy, and also reports F1 and BLEU-1:

```bash
uv run python experiment/common/evaluation/score.py <run-tag>
```

Multiple runs produce an overall mean and population standard deviation:

```bash
uv run python experiment/common/evaluation/score.py <run-r1> <run-r2> <run-r3>
```

Pass `--agent` to include fallback/kept/added/dropped metrics from Agent Filter
traces, `--column <name>` for a deliberate legacy/custom comparison, or
`--json <path>` for machine-readable output.

## Gold-Evidence Oracle

The shared oracle answers from annotated gold turns without running retrieval:

```bash
uv run python experiment/common/evaluation/oracle.py locomo oracle-locomo --samples 0-9
uv run python experiment/common/evaluation/oracle.py longmem oracle-longmem
```

`--window 0` uses only annotated turns. `--window N` includes N neighboring
turns on each side within the same session. LoCoMo image captions are excluded
unless `--include-photo` is supplied. Use a different run tag for each window or
photo condition.

Oracle generation writes the normal benchmark output layout. Pass `--judge` to
run the standardized judge immediately, or run `experiment/common/evaluation/judge.py` separately.
Every oracle output includes `oracle_config.json` with the resolved settings.

## Reproducibility Checklist

For every reported run, retain:

- `run_metadata.json` and the resolved experiment configuration;
- answer and judged CSV files;
- the judge aggregate JSON;
- answer model, judge model, endpoint type, and commit hash;
- benchmark dataset version and exact question subset.

Do not compare runs unless they use the same question set, answer-generation
settings, judge model, voting rule, and adversarial-question policy.
