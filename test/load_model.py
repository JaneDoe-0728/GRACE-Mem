# load_model.py
from __future__ import annotations
from typing import Dict, Optional, Generator
import os
import threading

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# --------- 型號登錄（不會真的下載，跑 load_model 才會觸發） ---------
# 你可以把 model_id 改成本地路徑（例如 "/models/Qwen2.5-3B-Instruct"）
MODEL_CATALOG: Dict[str, Dict] = {
    # 中文
    "zh-small": {
        "name": "Qwen2.5-3B-Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": True,
        "quant": None,
        "lang": "zh",
        "size": "small",
    },
    "zh-medium": {
        "name": "Qwen3-8B",
        "model_id": "Qwen/Qwen3-8B",
        "trust_remote_code": True,
        "quant": None,
        "lang": "zh",
        "size": "medium",
    },
    "zh-large": {
        "name": "Qwen2.5-72B-Instruct-GPTQ-Int4",
        "model_id": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        "trust_remote_code": True,
        "quant": "gptq-int4",
        "lang": "zh",
        "size": "large",
    },
    # 英文
    "en-small": {
        "name": "Llama-3.2-3B-Instruct",
        "model_id": "models/Llama-3.2-3B-Instruct",  # 本地路徑
        "trust_remote_code": False,
        "quant": None,
        "lang": "en",
        "size": "small",
    },
    "en-medium": {
        "name": "gpt-oss-20b",
        "model_id": "openai/gpt-oss-20b",
        "trust_remote_code": False,
        "quant": None,
        "lang": "en",
        "size": "medium",
    },
    "en-large": {
        "name": "gpt-oss-120b",
        "model_id": "openai/gpt-oss-120b",
        "trust_remote_code": False,
        "quant": None,
        "lang": "en",
        "size": "large",
    },
}

def pick_dtype() -> torch.dtype:
    # 盡量用 bf16（A100/H100 等），否則 fp16；CPU 則 fp32
    if torch.cuda.is_available():
        # 有些卡不支援 bf16，就用 fp16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class ModelManagerLocal:
    """
    專注本地推論：
    - load_model(model_key): 切換/載入指定 HF 權重（或本地路徑）
    - generate_stream(prompt): 使用單一字串 prompt，串流輸出 tokens
    - 不組裝對話訊息；不依賴 LangChain
    """
    def __init__(self):
        self.current_key: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.device_map = "auto" if (torch.cuda.is_available() or torch.backends.mps.is_available()) else None
        self.dtype = pick_dtype()

        # 生成預設（可在呼叫 generate_stream 時覆寫）
        self.gen_defaults = {
            "max_new_tokens": int(os.getenv("GEN_MAX_TOKENS", "512")),
            "temperature": float(os.getenv("GEN_TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("GEN_TOP_P", "0.9")),
            "do_sample": True,
            "repetition_penalty": 1.05,
        }

    # 列出可用鍵名與對應模型
    def list_models(self) -> Dict[str, Dict]:
        return MODEL_CATALOG

    def unload(self):
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    def load_model(
        self,
        model_key: str,
        *,
        lang: str | None = None,
        auto_correct_lang: bool = True   # 預設檢查到不一致就切換
    ):
        if model_key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model key: {model_key}")

        spec = MODEL_CATALOG[model_key]
        spec_lang = spec.get("lang")  # 'zh' or 'en'

        # --- 語言一致性檢查 ---
        if lang is not None:
            if spec_lang and lang != spec_lang:
                if auto_correct_lang:
                    # 依相同 size 自動尋找對應語言的模型鍵
                    size = spec.get("size", "medium")
                    corrected = next((k for k, s in MODEL_CATALOG.items()
                                    if s.get("lang")==lang and s.get("size")==size), None)
                    if corrected is None:
                        raise ValueError(f"語言不一致且無可自動更正的型號 (got lang={lang}, but {model_key} is {spec_lang}).")
                    model_key = corrected
                    spec = MODEL_CATALOG[model_key]
                    spec_lang = spec.get("lang")
                else:
                    raise ValueError(f"語言不一致：前端 lang={lang}，但 {model_key} 是 {spec_lang}。若要自動更正，請設 auto_correct_lang=True")

        # 如果已載入相同模型就略過
        if self.current_key == model_key and self.model is not None:
            return

        # 清理舊模型
        self.unload()

        model_id = spec["model_id"]
        trust_remote_code = spec.get("trust_remote_code", False)
        quant = spec.get("quant")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
            local_files_only=True,   # 本地優先/純離線
        )
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model
        if quant == "gptq-int4":
            try:
                from auto_gptq import AutoGPTQForCausalLM  # type: ignore
            except Exception as e:
                raise RuntimeError("此模型為 GPTQ-INT4，請先安裝 auto-gptq：pip install auto-gptq") from e
            self.model = AutoGPTQForCausalLM.from_pretrained(
                model_id,
                device_map=self.device_map or "auto",
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )

        if hasattr(self.model, "eval"):
            self.model.eval()

        self.current_key = model_key

    def generate_stream(
        self,
        prompt: str,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        """
        直接吃單一 prompt 字串，不做任何模板拼裝。
        以 TextIteratorStreamer 串流回傳。
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_model(model_key) first.")

        params = {**self.gen_defaults, **gen_kwargs}
        # 防止過長
        max_input = int(os.getenv("MAX_INPUT_TOKENS", "2048"))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif torch.backends.mps.is_available():
            # MPS 目前多數可自動搬上去；保守起見保持 CPU 也可
            pass

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            **params,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

        yield "[DONE]"
