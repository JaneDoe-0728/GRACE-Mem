"""
EXP-F01 — Local Reproducibility Smoke Test

Verifies that activate_reproducibility(seed=42, deterministic=True) produces
identical pseudorandom sequences across re-seeded trials within and across
processes.

Usage:
    python test/exp_f01_rng.py [run-tag]

Writes:
    test/fixed_output/results/<run-tag>/EXP-F01.json
"""
from __future__ import annotations

import json
import random
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── repo root on path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment.reproducibility import activate_reproducibility, build_dataloader_seed_components
from shared import (
    canonical_json,
    finalize_report,
    hash_obj,
    make_base_report,
    sha256_hex,
    write_report,
)

EXP_ID      = "EXP-F01"
SEED        = 42
REPEAT      = 5          # spec requires ≥ 5
SAMPLE_SIZE = 100        # spec: 100 floats per RNG source


# ── RNG samplers ─────────────────────────────────────────────────────────────

def _sample_rng() -> Dict[str, Any]:
    activate_reproducibility(seed=SEED, deterministic=True)
    payload: Dict[str, Any] = {
        "python_rng": [random.random() for _ in range(SAMPLE_SIZE)],
        "numpy_rng":  np.random.rand(SAMPLE_SIZE).tolist(),
        "torch_cpu":  torch.rand(SAMPLE_SIZE).tolist(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.rand(SAMPLE_SIZE, device="cuda").cpu().tolist()
    return payload


class _FixedDataset(Dataset):
    """24-item dataset that draws from all three RNGs per item."""
    def __len__(self) -> int:
        return 24

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "idx":    int(idx),
            "python": float(random.random()),
            "numpy":  float(np.random.rand()),
            "torch":  float(torch.rand(()).item()),
        }


def _collect_dataloader(num_workers: int) -> List[Dict[str, Any]]:
    worker_init_fn, generator = build_dataloader_seed_components(SEED)
    loader = DataLoader(
        _FixedDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )
    out: List[Dict[str, Any]] = []
    for batch in loader:
        for i in range(len(batch["idx"])):
            out.append({
                "idx":    int(batch["idx"][i]),
                "python": float(batch["python"][i]),
                "numpy":  float(batch["numpy"][i]),
                "torch":  float(batch["torch"][i]),
            })
    return out


def _sample_dataloader(warnings: List[str]) -> str:
    for num_workers in (2, 0):
        try:
            activate_reproducibility(seed=SEED, deterministic=True)
            batches = _collect_dataloader(num_workers)
            return sha256_hex(canonical_json(batches))
        except Exception as exc:
            if num_workers == 2:
                warnings.append(f"DataLoader num_workers=2 failed, retrying with 0: {exc}")
            else:
                warnings.append(f"DataLoader test failed: {exc}")
    return "ERROR"


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(trial_id: int, warnings: List[str]) -> Dict[str, Any]:
    rng_samples = _sample_rng()
    hashes: Dict[str, str] = {
        "python_rng_hash": sha256_hex(canonical_json(rng_samples["python_rng"])),
        "numpy_rng_hash":  sha256_hex(canonical_json(rng_samples["numpy_rng"])),
        "torch_cpu_hash":  sha256_hex(canonical_json(rng_samples["torch_cpu"])),
    }
    if "torch_cuda" in rng_samples:
        hashes["torch_cuda_hash"] = sha256_hex(canonical_json(rng_samples["torch_cuda"]))

    # DataLoader re-seeds internally, so we just run it from fresh seed
    hashes["dataloader_hash"] = _sample_dataloader(warnings)

    return {"trial_id": trial_id, "artifact_hashes": hashes}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run-tag for report path")
    args = parser.parse_args()

    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        config_snapshot={"seed": SEED, "sample_size": SAMPLE_SIZE},
    )
    warnings: List[str] = []

    for i in range(REPEAT):
        trial = run_trial(i, warnings)
        report["trials"].append(trial)

    report["warnings"] = warnings

    # All hash keys are primary for F01 (spec: unique_hash_counts == 1 for all)
    primary_keys = list(report["trials"][0]["artifact_hashes"].keys()) if report["trials"] else []
    finalize_report(report, primary_keys=primary_keys)

    path = write_report(report, EXP_ID, args.tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
