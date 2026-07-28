# compress_and_extract_with_baseline.py
# -*- coding: utf-8 -*-
"""
以 llmlingua 測試未壓縮 (baseline) 與多個壓縮率下的抽取結果。
會印 raw 輸出 + 解析結果，並存成 JSON。

執行：
    python compress_and_extract_with_baseline.py
"""

import os, json, time
from typing import Dict, Any, List, Tuple

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from KG.llm import LLMClient
from KG.llm.prompts import EXTRA_KWARGS, entity_relation_extraction_PROMPT_longmem1
from KG.utils.utils import parse_delimited_extraction

# ===== 可調參數 =====
# DIALOGUE = """
# User: I'm looking for some new hiking boots. Do you have any recommendations for waterproof boots that are good for multi-day back…rom a 3-day solo camping trip to Big Sur in early April and my current boots did okay, but I think I need something better.
# Assistant: Big Sur in April sounds amazing! I'm glad you got to experience it. Now, let's get you into some new boots that'll keep your… recommend trying on several options and taking them for a spin around the block to ensure a comfortable fit. Happy trails!
# """
# DIALOGUE = """
# User: I'm also wondering if you can recommend a good fertilizer for my snake plant and succulent? I've been using 
# a general-purpose fertilizer, but I'm not sure if they need something more specific.

# Assistant: Fertilizing your plants can make a big difference in their health and growth.

# For your Snake Plant (Sansevieria Trifasciata) and Succulents, you'll want to use a fertilizer that's formulated for their specific needs. Here are some recommendations:

# **Snake Plant:**

# * Balanced fertilizer with a slightly acidic pH: Snake plants prefer a slightly acidic to neutral soil pH (around 6.0-7.0). Look for a fertilizer with a balanced N-P-K ratio (e.g., 20-20-20) that's specifically formulated for indoor plants or tropical plants.
# * Water-soluble fertilizer: Snake plants don't require a lot of fertilizer, so a water-soluble fertilizer is a good choice. You can dilute it to half the recommended strength to avoid burning the roots.
# * Organic options: Consider using an organic fertilizer like fish emulsion or compost tea, which will provide a slow release of nutrients.

# Some good options for Snake Plants:

# * Miracle-Gro Indoor Plant Food (20-20-20)
# * Espoma Organic Indoor Plant Fertilizer (2-2-2)
# * Schultz All Purpose Plant Food (20-20-20)

# **Succulents:**

# * Cactus and Succulent-specific fertilizer: Succulents have different nutrient requirements than other plants. Look for a fertilizer specifically formulated for cacti and succulents, which will typically have a higher phosphorus content (e.g., 10-20-10) to promote root growth and flowering.
# * Low-nitrogen fertilizer: Succulents don't require a lot of nitrogen, so a low-nitrogen fertilizer is best. This will help prevent over-fertilization, which can lead to weak growth and poor root development.
# * Water-soluble fertilizer: Like snake plants, succulents benefit from water-soluble fertilizers that can be diluted to half the recommended strength.

# Some good options for Succulents:

# * Miracle-Gro Cactus, Palm & Citrus Food (10-20-10)
# * Espoma Organic Cactus! Fertilizer (5-2-6)
# * Schultz Cactus and Succulent Food (10-20-10)

# **General tips:**

# * Always read the label: Make sure the fertilizer you choose is specifically formulated for indoor plants, tropical plants, or cacti and succulents.
# * Dilute to half strength: To avoid burning your plants' roots, dilute the fertilizer to half the recommended strength.
# * Fertilize sparingly: Feed your plants during the growing season (spring and summer) and skip fertilizing during the dormant season (fall and winter).
# * Soil quality matters: Remember that fertilizer is only as good as the soil it's applied to. Make sure your potting mix is well-draining and of good quality.
# """
DIALOGUE = """new hiking boots. recommendations waterproof boots multi - day 3 - day camping trip Big Sur early April current boots, need better. 
 Big Sur April!., new boots options spin comfortable fit.! """
 
# 這些 rate 會在 baseline（未壓縮）之後依序測試
COMPRESSED_RATES = [0.8, 0.6, 0.4]

LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
USE_LLMLINGUA2 = True
SHOW_PROMPT = True # 是否印出送給 LLM 的實際 prompt
SAVE_JSON = True      # 是否輸出 JSON 檔案

# 分隔符（需與你的 parse_delimited_extraction 一致）
TUPLE_DELIM = "<|>"
RECORD_DELIM = "##"
COMPLETION_DELIM = "<|COMPLETE|>"

# =====================

def build_compressor(model_name: str, use_llmlingua2: bool):
    from llmlingua import PromptCompressor
    return PromptCompressor(model_name=model_name, use_llmlingua2=use_llmlingua2)

def compress_once(compressor, text: str, rate: float) -> Dict[str, Any]:
    """
    使用 llmlingua 進行壓縮；回傳壓縮統計與文本。
    """
    t0 = time.time()
    res = compressor.compress_prompt_llmlingua2(
        text,
        rate=rate,
        force_tokens=['\n', '.', '!', '?', ','],
        chunk_end_tokens=['.', '\n'],
        return_word_label=False,
        drop_consecutive=True,
    )
    t1 = time.time()
    compressed = res.get("compressed_prompt", "").strip() or text
    return {
        "compressed_text": compressed,
        "report_rate": res.get("rate", None),
        "elapsed_sec": t1 - t0,
        "orig_len": len(text),
        "compressed_len": len(compressed),
    }

def run_extraction(
    llm: LLMClient,
    summary_text: str,
    raw_conversation: str,
    prompt_templates: Dict[str, str],
) -> List[Tuple[str, bool, Any]]:
    """
    執行抽取並回傳 [(prompt_name, ok, payload)]
    payload = { "latency_sec": float, "parsed": ExtractionResult, "raw": str }
    """
    # ✅ 改成你想要的方式：先展開 EXTRA_KWARGS，再放 summary_text / raw_conversation
    variables = {
        **EXTRA_KWARGS,          # 展開額外參數（共用 system_prompt、delimiter等）
        "summary_text": summary_text,   # 確保模板中的 {summary_text} 能吃到壓縮後文字
        "dialogue_datetime":"2023/04/29 (Sat) 20:02",
        # "dialogue_datetime":"2023/05/27 (Sat) 14:40"

    }

    results = []
    for name, tmpl in prompt_templates.items():
        prompt = tmpl.format(**variables)
        if SHOW_PROMPT:
            print(f"\n{'='*60}\n[PROMPT - {name}]\n{prompt}\n{'='*60}")

        js, sec = llm.generate_llm_extract(prompt)
        parsed = parse_delimited_extraction(js, TUPLE_DELIM, RECORD_DELIM, COMPLETION_DELIM)
        results.append((name, True, {"latency_sec": sec, "parsed": parsed, "raw": js}))
    return results

def print_and_maybe_save(tag: str, cstats: Dict[str, Any], results: List[Tuple[str, bool, Any]]):
    """
    印出壓縮摘要、raw、解析結果；並視設定輸出 JSON。
    tag：顯示與檔名用，例如 '1.0'（未壓縮）、'0.7' 等。
    """
    print("-" * 80)
    print(f"[Rate = {tag}]")
    print(f"  Compression: orig_len={cstats['orig_len']}, "
          f"compressed_len={cstats['compressed_len']}, "
          f"elapsed={cstats['elapsed_sec']:.2f}s, "
          f"report_rate={cstats.get('report_rate')}")

    print("\n  ── Text Used For Extraction ──\n")
    print(cstats["compressed_text"])
    print("\n───────────────────────────────────────────────\n")

    out_results = []
    for name, ok, payload in results:
        parsed = payload["parsed"]
        ents = parsed.entities or []
        rels = parsed.relationships or []
        print(f"  [{name}] entities={len(ents)}, relationships={len(rels)}, "
              f"llm_latency={payload.get('latency_sec', 0):.2f}s")

        print("    ─ Raw LLM Output ─")
        print(payload["raw"].strip())

        print("    ─ Parsed Entities ─")
        for e in ents:
            print(f"      - {e.entity_name} ({e.entity_type}) — {e.entity_description}")

        print("    ─ Parsed Relationships ─")
        for r in rels:
            print(f"      - {r.source_entity} → {r.target_entity} - {r.relationship_description}")
        print("")

        out_results.append({
            "prompt_name": name,
            "entities": [e.__dict__ for e in ents],
            "relationships": [r.__dict__ for r in rels],
            "raw_output": payload["raw"],
        })

    if SAVE_JSON:
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/extract_rate_{tag}.json"
        data = {
            "rate": tag,
            "compression": cstats,
            "results": out_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  → 已輸出結果到 {out_path}\n")

def main():
    print("\n=== Baseline + Compression Extraction Test ===")
    print(f"Dialogue length = {len(DIALOGUE)} chars\n")

    # LLM 與模板
    llm = LLMClient()
    templates = entity_relation_extraction_PROMPT_longmem1

    # 先跑「未壓縮 baseline」
    baseline_cstats = {
        "compressed_text": DIALOGUE,
        "report_rate": 1.0,
        "elapsed_sec": 0.0,
        "orig_len": len(DIALOGUE),
        "compressed_len": len(DIALOGUE),
    }
    baseline_results = run_extraction(
        llm=llm,
        summary_text=DIALOGUE,
        raw_conversation=DIALOGUE,
        prompt_templates=templates,
    )
    print_and_maybe_save(tag="1.0", cstats=baseline_cstats, results=baseline_results)

    # # 再跑各種壓縮率
    # compressor = build_compressor(LLMLINGUA_MODEL, USE_LLMLINGUA2)
    # for rate in COMPRESSED_RATES:
    #     cstats = compress_once(compressor, DIALOGUE, rate)
    #     results = run_extraction(
    #         llm=llm,
    #         summary_text=cstats["compressed_text"],
    #         raw_conversation=DIALOGUE,
    #         prompt_templates=templates,
    #     )
    #     print_and_maybe_save(tag=str(rate), cstats=cstats, results=results)

    # print("\n✅ Done.\n")

if __name__ == "__main__":
    main()
