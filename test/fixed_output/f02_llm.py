"""
EXP-F02 — LLM API Determinism

Empirically tests whether LM Studio returns identical token sequences for the
same prompt and parameters.  Three sub-cases:
  F02-a: temperature=0.0, isolated   → must PASS
  F02-b: temperature=0.6, isolated   → WARN acceptable
  F02-c: temperature=0.6, shared     → WARN acceptable

Usage:
    python test/exp_f02_llm.py [run-tag]

Env vars (required):
    LLM_BASE_URL  e.g. http://localhost:1234/v1
    LLM_MODEL     e.g. gpt-oss-20b
    LLM_API_KEY   (optional, default "dummy")

Writes:
    test/fixed_output/results/<run-tag>/EXP-F02.json
"""
from __future__ import annotations

import json
import sys
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import (
    compute_unique_hash_counts,
    finalize_report,
    lm_studio_chat,
    make_base_report,
    probe_lm_studio,
    sha256_hex,
    write_report,
)

EXP_ID  = "EXP-F02"
SEED    = 42
REPEAT  = 20
TOP_P   = 0.95
MAX_TOK = 512

# 5 fixed prompts covering required task types
FIXED_PROMPTS: List[Dict[str, Any]] = [
    {
        "prompt_id": "keyword_extraction",
        "messages": [
            {"role": "system", "content": "Extract 3-5 keywords from the user message. Return only a JSON list."},
            {"role": "user",   "content": "Alice visited Paris last summer and bought a vintage Chanel bag at the flea market near Montmartre."},
        ],
    },
    {
        "prompt_id": "entity_extraction",
        "messages": [
            {"role": "system", "content": "Extract all named entities (person, location, organization, product). Return JSON: [{\"name\": ..., \"type\": ...}]."},
            {"role": "user",   "content": "Bob works at Google DeepMind in London and collaborates with Dr. Sara Chen from MIT."},
        ],
    },
    {
        "prompt_id": "relationship_extraction",
        "messages": [
            {"role": "system", "content": "Extract relationships as JSON: [{\"source\": ..., \"predicate\": ..., \"target\": ...}]."},
            {"role": "user",   "content": "Marie Curie discovered polonium. She worked at the University of Paris. Pierre Curie was her husband."},
        ],
    },
    {
        "prompt_id": "entity_op_decision",
        "messages": [
            {"role": "system", "content": "Decide whether to ADD or UPDATE each entity. Return JSON: [{\"name\": ..., \"op\": \"ADD\" or \"UPDATE\"}]."},
            {"role": "user",   "content": "Existing: [{\"name\": \"Alice\", \"description\": \"Software engineer\"}]. New: [{\"name\": \"Alice\", \"description\": \"Senior software engineer at Anthropic\"}]"},
        ],
    },
    {
        "prompt_id": "qa_answering",
        "messages": [
            {"role": "system", "content": "Answer the question concisely based on the context."},
            {"role": "user",   "content": "Context: Tom moved to Seattle in 2019 to join Amazon as a data scientist.\nQuestion: Where does Tom work?"},
        ],
    },
]


def run_case(
    case_id: str,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    *,
    warnings: List[str],
    failure_diagnosis: List[str],
) -> Dict[str, Any]:
    """Run all prompts × REPEAT calls for one case.  Return per-case sub-report."""
    prompts_report = []
    for pspec in FIXED_PROMPTS:
        pid = pspec["prompt_id"]
        text_hashes: List[str] = []
        errors: List[str] = []
        for rep in range(REPEAT):
            try:
                text = lm_studio_chat(
                    pspec["messages"],
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    top_p=TOP_P,
                    max_tokens=MAX_TOK,
                    seed=SEED,
                    api_key=api_key,
                )
                text_hashes.append(sha256_hex(text))
                time.sleep(0.05)
            except Exception as exc:
                errors.append(str(exc))
                text_hashes.append("ERROR")

        unique_count = len(set(text_hashes) - {"ERROR"})
        is_identical = unique_count == 1

        if not is_identical:
            msg = f"{case_id}/{pid}: {unique_count} distinct outputs across {REPEAT} repeats"
            if case_id == "F02-a":
                failure_diagnosis.append(f"FAIL {msg}")
            else:
                warnings.append(f"WARN {msg}")

        prompts_report.append({
            "prompt_id":              pid,
            "repeat_count":           REPEAT,
            "response_text_unique":   unique_count,
            "response_text_identical": is_identical,
            "response_text_hashes":   text_hashes,
            "errors":                 errors,
        })

    return {"case_id": case_id, "temperature": temperature, "prompts": prompts_report}


def compute_case_status(case: Dict[str, Any], case_id: str) -> str:
    for p in case["prompts"]:
        if p["response_text_unique"] > 1:
            if case_id == "F02-a":
                return "FAIL"
            return "WARN"
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag",       default=None,                      help="Run-tag for report path")
    parser.add_argument("--llm-url",   default="http://localhost:1234/v1", dest="llm_url",   help="LM Studio base URL")
    parser.add_argument("--llm-model", default="gpt-oss-20b",               dest="llm_model", help="LLM model name")
    parser.add_argument("--llm-api-key", default="dummy",                  dest="llm_api_key", help="API key")
    args = parser.parse_args()

    base_url = args.llm_url
    model    = args.llm_model
    api_key  = args.llm_api_key

    report = make_base_report(
        EXP_ID,
        repeat_count=REPEAT,
        temperature=0.0,
        top_p=TOP_P,
        max_tokens=MAX_TOK,
        config_snapshot={
            "base_url": base_url,
            "model":    model,
            "prompts":  [p["prompt_id"] for p in FIXED_PROMPTS],
        },
        llm_url=base_url,
        llm_model=model,
    )
    warnings: List[str]           = []
    failure_diagnosis: List[str]  = []

    if not probe_lm_studio(base_url):
        report["status"] = "SKIP"
        report["warnings"].append(f"LM Studio unreachable at {base_url}; set LLM_BASE_URL to enable")
        path = write_report(report, EXP_ID, args.tag)
        print(json.dumps(report, indent=2))
        print(f"\n[{EXP_ID}] SKIP  report → {path}", file=sys.stderr)
        return 0

    cases_report = []

    # F02-a: temp=0.0 isolated
    case_a = run_case("F02-a", base_url, model, api_key, 0.0,
                      warnings=warnings, failure_diagnosis=failure_diagnosis)
    case_a["status"] = compute_case_status(case_a, "F02-a")
    cases_report.append(case_a)

    # F02-b: temp=0.6 isolated
    case_b = run_case("F02-b", base_url, model, api_key, 0.6,
                      warnings=warnings, failure_diagnosis=failure_diagnosis)
    case_b["status"] = compute_case_status(case_b, "F02-b")
    cases_report.append(case_b)

    # F02-c: temp=0.6 shared load
    # "shared" is environmental; we document it but cannot enforce it from here
    warnings.append("F02-c runs sequentially; 'shared load' condition must be arranged externally")
    case_c = run_case("F02-c", base_url, model, api_key, 0.6,
                      warnings=warnings, failure_diagnosis=failure_diagnosis)
    case_c["status"] = compute_case_status(case_c, "F02-c")
    cases_report.append(case_c)

    report["cases"] = cases_report
    report["warnings"] = warnings
    report["failure_diagnosis"] = failure_diagnosis

    # Overall status: F02-a dictates pass/fail; b/c can only WARN
    if any(c["status"] == "FAIL" for c in cases_report):
        report["status"] = "FAIL"
    elif any(c["status"] == "WARN" for c in cases_report):
        report["status"] = "WARN"
    else:
        report["status"] = "PASS"

    # Synthetic trial records for consistent schema (one trial = one prompt × case)
    for case in cases_report:
        for p in case["prompts"]:
            report["trials"].append({
                "trial_id": f"{case['case_id']}/{p['prompt_id']}",
                "artifact_hashes": {
                    "response_text_hash": p["response_text_hashes"][0] if p["response_text_hashes"] else "MISSING",
                },
                "unique_count": p["response_text_unique"],
            })

    path = write_report(report, EXP_ID, args.tag)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[{EXP_ID}] status={report['status']}  report → {path}", file=sys.stderr)
    return 0 if report["status"] in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
