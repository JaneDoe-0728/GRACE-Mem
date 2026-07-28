"""
Direct keyword-inference probe for the configured LLM backend.

Prints:
- model / endpoint info
- exact prompt sent to the model
- OpenAI-compatible request payload
- raw model output
- optional JSON parse result

Usage:
    cd /path/to/gigabyte_kg
    uv run test/test_inference_keyword.py
    uv run test/test_inference_keyword.py --question "What did Caroline research?"
    uv run test/test_inference_keyword.py --max-tokens 512 --temperature 0
    uv run test/test_inference_keyword.py --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from KG.llm.client import LLMClient
from KG.llm.prompts.keyword.extraction import keyword_extraction_PROMPT
from KG.utils.utils import KeywordExtractionResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect keyword-extraction LLM IO.")
    parser.add_argument(
        "--question",
        default="When did Caroline go to the LGBTQ support group?",
        help="Question to send to the keyword extractor.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="max_tokens for the LLM call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature for the LLM call.",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Only print raw output, do not attempt JSON parsing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to the OpenAI-compatible backend.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    llm = LLMClient()
    prompt = keyword_extraction_PROMPT.format(query=args.question)
    payload = {
        "model": llm.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "response_format": {"type": "text"},
    }

    print("=== MODEL ===")
    print(llm.model_name)
    print("\n=== QUESTION ===")
    print(args.question)
    print("\n=== PROMPT_SENT_TO_LLM ===")
    print(prompt)
    print("\n=== REQUEST_PAYLOAD ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n=== RAW_RESPONSE ===")
    t0 = time.perf_counter()
    try:
        response = llm.client.chat.completions.create(**payload)
    except Exception as exc:
        print(f"elapsed_sec = {time.perf_counter() - t0}")
        print(f"ERROR_TYPE = {type(exc).__name__}")
        print(f"ERROR = {exc}")
        return 1

    elapsed = time.perf_counter() - t0
    text = response.choices[0].message.content or ""

    print(f"elapsed_sec = {elapsed}")
    print(f"usage = {getattr(response, 'usage', None)}")
    print(f"repr = {text!r}")
    print("\n=== RESPONSE_TEXT ===")
    print(text)

    if args.skip_parse:
        return 0

    print("\n=== PARSE_RESULT ===")
    try:
        parsed = KeywordExtractionResult.model_validate_json(text)
        print(json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"PARSE_ERROR_TYPE = {type(exc).__name__}")
        print(f"PARSE_ERROR = {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
