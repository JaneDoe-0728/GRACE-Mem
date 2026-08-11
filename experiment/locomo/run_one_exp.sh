#!/usr/bin/env bash
# run_one_experiment.sh — run the full locomo10 benchmark (samples 0-9)
#
# Usage:
#   bash experiment/locomo/run_one_experiment.sh <run-tag> [options]
#
# Required:
#   <run-tag>          Unique name for this run (e.g. "oss-20b-0430")
#
# Options:
#   --artifact-dir DIR  Reuse ingest artifacts from a previous run's output dir
#                       (skips re-ingest; resolves per-sample as DIR/sample_<n>/artifacts/)
#   --adaptive          Enable adaptive two-pass retrieval
#   --tau FLOAT         Confidence threshold for adaptive re-search (default: 0.70)
#   --sample-ids RANGE  Override sample range (default: 0-9, e.g. "0,2,5-7")
#   --adv               Include adversarial questions (excluded by default)
#
# Examples:
#   bash experiment/locomo/run_one_experiment.sh oss-20b-0430
#   bash experiment/locomo/run_one_experiment.sh oss-20b-0430 --artifact-dir experiment/locomo/output/standard/oss-20b-0429
#   bash experiment/locomo/run_one_experiment.sh oss-20b-ada-0430 --adaptive --tau 0.70
#
# Output:
#   experiment/locomo/output/standard/<run-tag>/
#     sample_<n>/           per-sample eval + judge CSVs and retrieval logs
#     _correctness_aggregate.json  overall and per-category correctness summary

set -euo pipefail

# ── timeout: 18 hours (120b ingest+QA is slower than 20b) ────────────────────
export TIMEOUT_SECONDS=64800

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
while [[ ! -f "${REPO_ROOT}/pyproject.toml" && "${REPO_ROOT}" != "/" ]]; do
    REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
done

# ── force-load .env from repo root ───────────────────────────────────────────

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
    echo "[run_one_exp] loaded .env from ${REPO_ROOT}/.env"
else
    echo "[run_one_exp] WARNING: .env not found at ${REPO_ROOT}/.env" >&2
fi

# ── help ─────────────────────────────────────────────────────────────────────

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    sed -n '2,/^set -/p' "$0" | grep -v '^set ' | sed 's/^# \{0,1\}//'
    exit 0
fi

# ── preflight check ──────────────────────────────────────────────────────────

preflight() {
    local fail=0

    # FalkorDB — prefer docker exec (works without redis-cli on host)
    local pong
    pong=$(docker exec falkordb redis-cli -a falkordb ping 2>/dev/null \
           || redis-cli -p 6379 -a falkordb ping 2>/dev/null \
           || true)
    if [[ "${pong}" != *"PONG"* ]]; then
        echo "PREFLIGHT FAIL: FalkorDB not reachable. Run: sudo docker compose up -d" >&2
        fail=1
    fi

    # Embedding model
    if [[ ! -f "${REPO_ROOT}/models/embedding_models/qwen3-0.6b/config.json" ]]; then
        echo "PREFLIGHT FAIL: embedding model missing. Run: bash tools/setup_env.sh" >&2
        fail=1
    fi

    # Reranker model
    if [[ ! -f "${REPO_ROOT}/models/reranker/qwen3-reranker-0.6b/config.json" ]]; then
        echo "PREFLIGHT FAIL: reranker model missing. Run: bash tools/setup_env.sh" >&2
        fail=1
    fi

    [[ "${fail}" -eq 0 ]] || exit 1
}

preflight

# ── argument parsing ────────────────────────────────────────────────────────

RUN_TAG="$1"
shift

ARTIFACT_DIR=""
ADAPTIVE=0
TAU="0.70"
SAMPLE_IDS="0-9"
ADV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact-dir)  ARTIFACT_DIR="$2"; shift 2 ;;
        --adaptive)      ADAPTIVE=1; shift ;;
        --tau)           TAU="$2"; shift 2 ;;
        --sample-ids)    SAMPLE_IDS="$2"; shift 2 ;;
        --adv)           ADV=1; shift ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── build command ────────────────────────────────────────────────────────────

CMD=(
    uv run python experiment/locomo/pipeline/runner.py
    --dataset locomo
    --sample-ids "${SAMPLE_IDS}"
    --run-tag "${RUN_TAG}"
)

[[ -n "${ARTIFACT_DIR}" ]] && CMD+=(--artifact-dir "${ARTIFACT_DIR}")
[[ "${ADAPTIVE}" -eq 1 ]]  && CMD+=(--adaptive --tau "${TAU}")
[[ "${ADV}"      -eq 1 ]]  && CMD+=(--adv)

OUT_DIR="experiment/locomo/output/standard/${RUN_TAG}"

# ── run ──────────────────────────────────────────────────────────────────────

echo "========================================"
echo "run-tag:      ${RUN_TAG}"
echo "sample-ids:   ${SAMPLE_IDS}"
echo "adaptive:     ${ADAPTIVE}$([[ "${ADAPTIVE}" -eq 1 ]] && echo "  tau=${TAU}")"
echo "artifact-dir: ${ARTIFACT_DIR:-<fresh ingest>}"
echo "output:       ${OUT_DIR}"
echo "========================================"
echo ""

timeout "${TIMEOUT_SECONDS}" "${CMD[@]}" || { rc=$?; [[ $rc -eq 137 ]] && echo "TIMEOUT: exceeded ${TIMEOUT_SECONDS}s" >&2; exit $rc; }

echo ""
echo "========================================"
echo "Done: ${RUN_TAG}"
echo "Results: ${OUT_DIR}/_correctness_aggregate.json"
echo "========================================"
