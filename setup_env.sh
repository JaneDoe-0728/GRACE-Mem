#!/usr/bin/env bash
# setup_env.sh — one-time environment setup before running experiments
#
# Run this once per machine (or after pulling new deps/models):
#   bash setup_env.sh
#
# Steps:
#   1. uv sync          — install Python dependencies
#   2. docker compose   — start FalkorDB container
#   3. download_model   — pull embedding + reranker models (skips if present)
#   4. verify           — confirm FalkorDB reachable and model files exist

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
while [[ ! -f "${REPO_ROOT}/pyproject.toml" && "${REPO_ROOT}" != "/" ]]; do
    REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
done

# ── 1. Python deps ────────────────────────────────────────────────────────────

echo "[1/4] uv sync"
cd "${REPO_ROOT}"
uv sync
echo "      OK"

# ── 2. FalkorDB ───────────────────────────────────────────────────────────────

echo "[2/4] docker compose up -d (FalkorDB)"
sudo docker compose up -d --force-recreate
echo "      OK"

# ── 3. Download models ────────────────────────────────────────────────────────

echo "[3/4] download models (no-op if already present)"
uv run python download_model.py
echo "      OK"

# ── 4. Verify ─────────────────────────────────────────────────────────────────

echo "[4/4] verifying..."

# FalkorDB — prefer docker exec (works without redis-cli on host)
PONG=$(sudo docker exec falkordb redis-cli -a falkordb ping 2>/dev/null \
       || redis-cli -p 6379 -a falkordb ping 2>/dev/null \
       || true)
if [[ "${PONG}" != *"PONG"* ]]; then
    echo "  FAIL: FalkorDB not reachable (got: '${PONG}')" >&2
    echo "        Try: sudo docker compose up -d" >&2
    exit 1
fi
echo "  FalkorDB: OK"

# Embedding model
EMB_DIR="${REPO_ROOT}/models/embedding_models/qwen3-0.6b"
if [[ ! -f "${EMB_DIR}/config.json" ]]; then
    echo "  FAIL: embedding model not found at ${EMB_DIR}" >&2
    exit 1
fi
echo "  Embedding model: OK"

# Reranker model
RNK_DIR="${REPO_ROOT}/models/reranker/qwen3-reranker-0.6b"
if [[ ! -f "${RNK_DIR}/config.json" ]]; then
    echo "  FAIL: reranker model not found at ${RNK_DIR}" >&2
    exit 1
fi
echo "  Reranker model: OK"

echo ""
echo "Environment ready. See experiment/readme.md to run the benchmarks."
