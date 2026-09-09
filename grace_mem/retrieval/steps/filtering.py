"""Context filtering, intersection, and reranking -- and the reranker itself.

`EvidenceFilter` is the stage: it takes the candidate entities and
relationships stage 1 produced, computes the subgraph intersection, scores what
survives, and cuts to top-K above a threshold.

The pointwise reranker it scores with lives in the same module because it has
exactly one caller. It runs Qwen3-Reranker-0.6B in generative mode, judging
relevance by the yes/no logit gap under Qwen3 chat format with memory-QA
specific instructions per document type, and has two backends:

  API  (RERANKER_API set in .env): OpenAI-compatible chat/completions with
       logprobs. No local GPU required; parallelised with ThreadPoolExecutor.
  Local (default): loads the model on CUDA/CPU and scores via yes/no logit gap.

`get_reranker()` picks between them and memoises the choice for the process.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from grace_mem.services.cache.cache import build_id_to_meta_maps
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog

# Repo root. This module sits at grace_mem/retrieval/steps/, so the root is four
# levels up. Getting the count wrong does not raise: load_dotenv would simply
# find no file, RERANKER_API would read as unset, and every run would silently
# fall back to the local GPU backend.
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_jlog = make_module_jlog(name="grace_mem.Retrieval.Filtering", filename="kg_retrieval_filtering.jsonl")


# Default reranker batch size; overridable via env so it can be tuned to fit GPU
# memory (lower avoids the CUDA-OOM retry churn). Only affects grouping, not scores.
_DEFAULT_BATCH = int(os.environ.get("KG_RERANKER_BATCH_SIZE") or 8)

DocType = Literal["entity", "relationship"]

INSTRUCTIONS: dict[str, str] = {
    "entity": (
        "You are reranking candidate entities for long-term memory question answering. "
        "Judge whether the candidate entity is useful for retrieving evidence needed to answer the query. "
        "A relevant entity does not need to contain the final answer directly, but it should help locate, connect, "
        "or justify the answer-bearing memory.\n\n"

        "First infer the query's retrieval need (temporal, preference, multi-hop, or knowledge-update). "
        "Then apply the matching rule below:\n"
        "- Temporal: prefer entities that represent dates, time ranges, recency, event chronology, "
        "or before/after order relevant to the query.\n"
        "- Preference: prefer entities about stable likes, dislikes, favorites, habits, goals, "
        "choices, constraints, or personal attributes.\n"
        "- Multi-hop: prefer bridge entities that connect the query subject to the answer, "
        "even if the entity does not directly state the final answer.\n"
        "- Knowledge-update: prefer entities pointing to newer, corrected, or currently valid memories; "
        "penalize stale entities when the query asks for the latest or updated fact.\n\n"

        "Mark as relevant only if the entity can help retrieve or justify the answer. "
        "Mark as not relevant if it is only broadly related to the topic, shares a keyword, "
        "or is a generic background concept with no direct link to the evidence.\n\n"

        "Example — Query: \"When did Audrey make muffins for herself?\"\n"
        "Relevant: \"Week of April 3, 2023\" — anchors the answer time.\n"
        "Not relevant: \"Pastries\" — generic topic entity with no direct link to the muffin-making event or its date."
    ),

    "relationship": (
        "You are reranking candidate relationships for long-term memory question answering. "
        "Judge whether the candidate relationship expresses a fact, event, temporal link, preference, update, "
        "or bridge connection useful for answering the query. "
        "A relevant relationship may directly state the answer or provide a necessary reasoning step toward it.\n\n"

        "First infer the query's retrieval need (temporal, preference, multi-hop, or knowledge-update). "
        "Then apply the matching rule below:\n"
        "- Temporal: prefer relationships that state when something happened, link an event to a date, "
        "describe before/after order, recency, duration, or event chronology.\n"
        "- Preference: prefer relationships that state likes, dislikes, favorites, habits, goals, choices, "
        "constraints, or stable personal tendencies.\n"
        "- Multi-hop: prefer relationships that connect two useful entities in the reasoning chain "
        "(subject–event, event–date, person–preference, or bridge relations).\n"
        "- Knowledge-update: prefer relationships that indicate newer, corrected, replaced, or currently valid "
        "information; penalize stale or superseded facts when the query asks for the latest state.\n\n"

        "Mark as relevant only if the relationship helps answer, retrieve evidence for, or justify the answer. "
        "Mark as not relevant if it only shares broad topic words or expresses a generic association "
        "that does not narrow down the answer.\n\n"

        "Example — Query: \"When did Audrey make muffins for herself?\"\n"
        "Relevant: a relationship linking the muffin/pastry event to \"Week of April 3, 2023\" — supports the temporal answer.\n"
        "Not relevant: a relationship about pastries in general with no connection to the specific event or its date."
    ),
}

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\"."
)


# ---------------------------------------------------------------------------
# Shared prompt builder
# ---------------------------------------------------------------------------

def _build_chat_messages(query: str, doc: str, instruction: str) -> list:
    """Frame a (query, document) pair as the yes/no question the reranker scores."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<Instruct>: {instruction}\n\n"
                f"<Query>: {query}\n\n"
                f"<Document>: {doc}"
            ),
        },
    ]


def _build_local_prompt(query: str, doc: str, instruction: str) -> str:
    """Qwen3 chat-style raw-text prompt for local tokeniser scoring."""
    return (
        "<|im_start|>system\n"
        + _SYSTEM_PROMPT
        + "<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {instruction}\n\n"
        f"<Query>: {query}\n\n"
        f"<Document>: {doc}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


# ---------------------------------------------------------------------------
# API-based reranker (no local GPU)
# ---------------------------------------------------------------------------

class APIPointwiseReranker:
    """
    Reranker that calls an OpenAI-compatible /v1/chat/completions endpoint.
    Uses logprobs to compute yes/no log-probability gap; falls back to
    plain text (yes→+1, no→-1) if the server does not support logprobs.
    Concurrent scoring via ThreadPoolExecutor replaces GPU batching.
    """

    def __init__(self, api_base: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(
            base_url=api_base,
            api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        )
        self._model = model
        print(f"[Reranker] API mode: base={api_base!r}, model={model!r}")

    def _score_one(self, query: str, doc: str, instruction: str) -> float:
        """Score one pair from the logprobs of a single generated token.

        The relevance score is the yes/no logprob margin, not the sampled token: one
        token is generated with logprobs requested, and the gap between P(yes) and
        P(no) is read off the distribution. That yields a continuous, rankable score
        where the sampled token alone would give only a binary.

        `top_logprobs=20` is margin. "yes" and "no" are normally the top candidates,
        but tokenizers emit leading-space variants, so the lookup strips those and
        needs room to find both. A token absent from the list falls back to -100.0 --
        effectively impossible, without tripping over a true zero probability.
        """
        messages = _build_chat_messages(query, doc, instruction)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                temperature=0,
            )
            choice_logprobs = resp.choices[0].logprobs
            if choice_logprobs is None or not choice_logprobs.content:
                raise ValueError("Reranker API response did not include token logprobs")
            lps = choice_logprobs.content[0].top_logprobs
            yes_lp = next(
                (lp.logprob for lp in lps if lp.token.lower().strip("▁ ") == "yes"),
                -100.0,
            )
            no_lp = next(
                (lp.logprob for lp in lps if lp.token.lower().strip("▁ ") == "no"),
                -100.0,
            )
            return yes_lp - no_lp
        except Exception:
            # logprobs not supported — fall back to text output
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=1,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").lower().strip()
            return 1.0 if text.startswith("yes") else -1.0

    def _score_all(
        self, query: str, docs: list[str], doc_type: DocType, max_workers: int
    ) -> list[float]:
        instruction = INSTRUCTIONS[doc_type]
        scores: list[float] = [0.0] * len(docs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._score_one, query, doc, instruction): i
                for i, doc in enumerate(docs)
            }
            for fut in as_completed(future_to_idx):
                scores[future_to_idx[fut]] = fut.result()
        return scores

    def rerank(
        self,
        query: str,
        docs: list[str],
        batch_size: int = _DEFAULT_BATCH,
        doc_type: DocType = "entity",
    ) -> list[tuple[int, float]]:
        """Score documents against a query and return them ordered, best first."""
        scores = self._score_all(query, docs, doc_type=doc_type, max_workers=batch_size)
        results = list(enumerate(scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def rank_pairs(
        self,
        query: str,
        texts: list[str],
        threshold: float | None = None,
        doc_type: DocType = "entity",
    ) -> list[tuple[int, float]]:
        """Score explicit (query, document) pairs, best first, with a threshold.

        Returns (index, score) pairs ordered by score, not by input position -- it
        delegates to `rerank`. Both callers in the repo depend on that: they read
        the result as a ranking and take `passing[:top_k]`. Zip the indices, never
        the positions.
        """
        if not texts:
            return []
        results = self.rerank(query, texts, doc_type=doc_type)
        if threshold is not None:
            results = [(i, s) for i, s in results if s >= threshold]
        return results


# ---------------------------------------------------------------------------
# Local-GPU reranker
# ---------------------------------------------------------------------------

class LLMPointwiseReranker:
    """
    Point-wise reranking driven by an LLM prompt and its yes/no logits.
    Automatically retries with halved batch_size on CUDA OOM.
    """

    MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "reranker"
    MODEL_PATH = MODELS_DIR / "qwen3-reranker-0.6b"

    def _resolve_model_name(self, name: str) -> str:
        """Resolve RERANKER_MODEL_NAME to a loadable path or an HF repo id.

        1. an absolute path, or a relative one that exists -> use it as given
        2. a subdirectory name under models/reranker/ (e.g. "qwen3-reranker-4b")
           -> use the local copy
        3. anything else -> treat it as an HF repo id (e.g. "Qwen/Qwen3-Reranker-4B")
        """
        if os.path.isabs(name) or os.path.isdir(name):
            return name
        local = self.MODELS_DIR / name
        if local.is_dir():
            return str(local)
        print(f"[Reranker] {name} not found locally, loading it as an HF repo id")
        return name

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            if os.path.isdir(str(self.MODEL_PATH)):
                model_path = str(self.MODEL_PATH)
                print(f"[Reranker] Local: {model_path} (device={self.device})")
            else:
                model_path = "Qwen/Qwen3-Reranker-0.6B"
                print(f"[Reranker] Local model not found, using HF: {model_path} (device={self.device})")
        else:
            model_path = self._resolve_model_name(model_name)
            print(f"[Reranker] Local (custom): {model_path} (device={self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        model: Any = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(self.device)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def _score_one(self, query: str, doc: str, doc_type: DocType = "entity") -> float:
        """Score one pair locally, reading the yes/no margin from the model's logits.

        The same margin trick as the API reranker, computed from the logits directly
        rather than from a returned logprob list -- so both backends produce scores
        on a comparable scale.

        Falls back to `_score_one_cpu` on CUDA OOM: a long document can exceed
        memory even at batch size 1, and losing its score entirely would drop a
        candidate silently.
        """
        import torch
        instruction = INSTRUCTIONS[doc_type]
        prompt = _build_local_prompt(query, doc, instruction)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=1,
                return_dict_in_generate=True, output_scores=True, do_sample=False,
            )
        logits = out.scores[-1]
        yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        return logits[0, yes_id].item() - logits[0, no_id].item()

    def _score_batch(
        self, query: str, docs: list[str], batch_size: int = _DEFAULT_BATCH, doc_type: DocType = "entity"
    ) -> list[float]:
        """Batch scoring with automatic OOM retry (halves batch_size each attempt)."""
        import torch
        instruction = INSTRUCTIONS[doc_type]
        yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        all_scores: list[float] = []

        i = 0
        current_batch_size = batch_size
        while i < len(docs):
            batch_docs = docs[i:i + current_batch_size]
            batch_prompts = [_build_local_prompt(query, doc, instruction) for doc in batch_docs]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", truncation=True,
                max_length=self.tokenizer.model_max_length, padding=True,
            ).to(self.device)
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=1,
                        return_dict_in_generate=True, output_scores=True, do_sample=False,
                    )
                logits = out.scores[-1]
                batch_scores = (logits[:, yes_id] - logits[:, no_id]).cpu().tolist()
                all_scores.extend(batch_scores)
                i += current_batch_size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"[Reranker] CUDA OOM — retrying with batch_size={current_batch_size}")
                else:
                    # batch_size=1 still OOMs; score this item on CPU
                    print("[Reranker] CUDA OOM at batch_size=1 — falling back to CPU for this item")
                    score = self._score_one_cpu(query, batch_docs[0], doc_type)
                    all_scores.append(score)
                    i += 1

        return all_scores

    def _score_one_cpu(self, query: str, doc: str, doc_type: DocType) -> float:
        """Last-resort single-item scoring on CPU."""
        import torch
        instruction = INSTRUCTIONS[doc_type]
        prompt = _build_local_prompt(query, doc, instruction)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        with torch.no_grad():
            out = self.model.to("cpu").generate(
                **inputs, max_new_tokens=1,
                return_dict_in_generate=True, output_scores=True, do_sample=False,
            )
        self.model.to(self.device)
        logits = out.scores[-1]
        yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        return logits[0, yes_id].item() - logits[0, no_id].item()

    def rerank(
        self, query: str, docs: list[str], batch_size: int = _DEFAULT_BATCH, doc_type: DocType = "entity"
    ) -> list[tuple[int, float]]:
        """Score documents against a query and return them ordered, best first."""
        scores = self._score_batch(query, docs, batch_size=batch_size, doc_type=doc_type)
        results = list(enumerate(scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def rank_pairs(
        self,
        query: str,
        texts: list[str],
        threshold: float | None = None,
        doc_type: DocType = "entity",
    ) -> list[tuple[int, float]]:
        """Score explicit (query, document) pairs, best first, with a threshold.

        Ordered by score like the API reranker's, not by input position."""
        if not texts:
            return []
        results = self.rerank(query, texts, doc_type=doc_type)
        if threshold is not None:
            results = [(i, s) for i, s in results if s >= threshold]
        return results


# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

Reranker = APIPointwiseReranker | LLMPointwiseReranker

# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_reranker_instance: Reranker | None = None


def get_reranker(
    model_name: str | None = None,
    device: str | None = None,
    force_reload: bool = False,
) -> Reranker:
    """
    Return the global reranker instance (lazy-loaded).

    Selection order:
      1. RERANKER_API in .env → APIPointwiseReranker (no local GPU)
      2. Otherwise            → LLMPointwiseReranker (local CUDA/CPU)

    Model name comes from RERANKER_MODEL_NAME (.env) for both backends:
      - API   backend: defaults to "qwen3-reranker-0.6b".
      - Local backend: resolved as local subfolder / abs path / HF repo id;
        unset → default local folder qwen3-reranker-0.6b (fallback HF 0.6B).
    """
    global _reranker_instance
    if _reranker_instance is not None and not force_reload:
        return _reranker_instance

    api_base = os.getenv("RERANKER_API", "").strip()
    if api_base:
        api_model = os.getenv("RERANKER_MODEL_NAME", "qwen3-reranker-0.6b")
        _reranker_instance = APIPointwiseReranker(api_base=api_base, model=api_model)
    else:
        if device is None:
            device = os.getenv("RERANKER_DEVICE", "").strip() or None
        if model_name is None:
            model_name = os.getenv("RERANKER_MODEL_NAME", "").strip() or None
        _reranker_instance = LLMPointwiseReranker(model_name=model_name, device=device)

    return _reranker_instance

# ===========================================================================
# The filtering stage
# ===========================================================================

_RERANKER_SCORE_CSV = os.path.join("logs", "reranker_scores.csv")
_RERANKER_CSV_HEADER = [
    "ts_ms", "request_id",
    "item_type", "item_id", "name",
    "score", "rank",
    "above_threshold", "selected",
    "threshold", "top_k",
    "drop_reason",  # "": selected | "topk_cutoff": passed threshold but rank > top_k | "below_threshold": score < threshold
]


def _append_reranker_scores(
    rows: list[dict],
) -> None:
    """Append scored candidate rows to logs/reranker_scores.csv."""
    os.makedirs("logs", exist_ok=True)
    write_header = not os.path.exists(_RERANKER_SCORE_CSV)
    with open(_RERANKER_SCORE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RERANKER_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


class EvidenceFilter:
    """
    Handles filtering and reranking of context entities and relationships.
    """

    def __init__(self, vector_db_manager: Any, cache: dict[str, Any]) -> None:
        """
        Args:
            vector_db_manager: Manager for accessing vector databases
            cache: Global cache with entity/relationship metadata
        """
        self.vector_db_manager = vector_db_manager
        self.cache = cache

    def _entity_names_from_ids(self, entity_ids: set[str] | list[str]) -> list[str]:
        """Resolve entity IDs into readable names."""
        ent_id2meta, _ = build_id_to_meta_maps(self.cache)
        names: list[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            meta = ent_id2meta.get(entity_id, {}) or {}
            name = meta.get("name") or entity_id
            if name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def _relationship_names_from_ids(self, relationship_ids: set[str] | list[str]) -> list[str]:
        """Resolve relationship IDs into readable source->target labels."""
        ent_id2meta, rel_id2meta = build_id_to_meta_maps(self.cache)
        labels: list[str] = []
        seen: set[str] = set()
        for relationship_id in relationship_ids:
            meta = rel_id2meta.get(relationship_id, {}) or {}
            if not meta:
                label = relationship_id
            else:
                source_id = meta.get("source_id")
                target_id = meta.get("target_id")
                source_meta = ent_id2meta.get(source_id, {}) if isinstance(source_id, str) else {}
                target_meta = ent_id2meta.get(target_id, {}) if isinstance(target_id, str) else {}
                src_name = (
                    meta.get("source_entity")
                    or source_meta.get("name")
                    or source_id
                    or "?"
                )
                tgt_name = (
                    meta.get("target_entity")
                    or target_meta.get("name")
                    or target_id
                    or "?"
                )
                desc = (meta.get("description") or "").strip()
                label = f"{src_name} -> {tgt_name}" if not desc else f"{src_name} -> {tgt_name} | {desc}"
            if label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    def compute_subgraph_intersection(
        self,
        node_subgraph: dict,
        edge_subgraph: list,
        use_union: bool = True,
        request_id: str | None = None,
    ) -> tuple[set[str], set[str]]:
        """
        Compute intersection (or union) of entity/relationship IDs from local and global subgraphs.

        Args:
            node_subgraph: Node-based subgraph from entity search
            edge_subgraph: Edge-based subgraph from relationship search
            use_union: If True, use union; if False, use intersection
            request_id: Request ID for logging

        Returns:
            (entity_ids, relationship_ids) tuple
        """
        _jlog(
            "subgraph_merge_start",
            request_id,
            step="2.5",
            node_subgraph_nodes=len(node_subgraph or {}),
            edge_subgraph_edges=len(edge_subgraph or []),
            operation="union" if use_union or not edge_subgraph else "intersection",
        )

        # Collect local (entity-based) subgraph IDs
        local_entity_set = set(node_subgraph.keys()) | {
            nb["neighbor_id"]
            for b in node_subgraph.values()
            for nb in (b.get("neighbors") or [])
        }
        local_rel_set = {
            nb["rel_id"]
            for b in node_subgraph.values()
            for nb in (b.get("neighbors") or [])
            if "rel_id" in nb
        }

        # Collect global (relationship-based) subgraph IDs
        global_entity_set = (
            ({e["source_id"] for e in edge_subgraph} | {e["target_id"] for e in edge_subgraph})
            if edge_subgraph
            else set()
        )
        global_rel_set = ({e["rel_id"] for e in edge_subgraph}) if edge_subgraph else set()

        # Intersection or union
        if use_union or not edge_subgraph:
            intersect_entity_ids = local_entity_set | global_entity_set
            intersect_rel_ids = local_rel_set | global_rel_set
        else:
            intersect_entity_ids = local_entity_set & global_entity_set
            intersect_rel_ids = local_rel_set & global_rel_set

        _jlog(
            "intersection_done",
            request_id,
            step="2.5",
            local_entities=len(local_entity_set),
            local_rels=len(local_rel_set),
            global_entities=len(global_entity_set),
            global_rels=len(global_rel_set),
            intersect_entities=len(intersect_entity_ids),
            intersect_rels=len(intersect_rel_ids),
            use_union=use_union,
            operation="union" if use_union or not edge_subgraph else "intersection",
            sample_entity_ids=sorted(intersect_entity_ids)[:20],
            sample_relationship_ids=sorted(intersect_rel_ids)[:20],
            sample_entity_names=self._entity_names_from_ids(sorted(intersect_entity_ids)[:20]),
            sample_relationship_names=self._relationship_names_from_ids(sorted(intersect_rel_ids)[:20]),
        )

        return intersect_entity_ids, intersect_rel_ids





    def rerank_filter(
        self,
        question: str,
        entity_ids: Collection[str],
        relationship_ids: Collection[str],
        entity_top_k: int,
        relationship_top_k: int,
        threshold: float,
        request_id: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Primary reranker filter: score all candidates, keep top-K above threshold.

        Returns ordered lists (reranker rank order, deduplicated) rather than
        sets so the final Retrieved_Context ordering is deterministic across runs.

        Takes ordered collections, not sets: `rank_pairs` sorts stably, so ties --
        every doc on the reranker's no-logprobs fallback -- keep the caller's order,
        and the top-K cut below is only deterministic if that order is.
        """
        timer_total = _StepTimer()
        _jlog(
            "rerank_filter_start",
            request_id,
            step="2.7",
            entity_count=len(entity_ids),
            relationship_count=len(relationship_ids),
            entity_top_k=entity_top_k,
            relationship_top_k=relationship_top_k,
            threshold=threshold,
        )

        reranker = get_reranker()
        entity_id2meta, relationship_id2meta = build_id_to_meta_maps(self.cache)
        ts_ms = int(time.time() * 1000)
        csv_rows: list[dict] = []

        # Score all entities
        entity_texts = []
        entity_ids_list = []
        for entity_id in entity_ids:
            meta = entity_id2meta.get(entity_id, {})
            if not meta:
                continue
            name = meta.get("name", "").strip()
            type_val = meta.get("type", "").strip()
            description = meta.get("description", "").strip()
            text = f"{name} [type={type_val}] {description}".strip() if name else f"{type_val} {description}".strip()
            entity_texts.append(text)
            entity_ids_list.append(entity_id)

        # Ordered list preserving reranker rank order; a set tracks membership
        # for dedup so the final context ordering is deterministic across runs.
        selected_entity_ids: list[str] = []
        _seen_entity_ids: set[str] = set()
        if entity_texts:
            # Fetch all scores without threshold so every candidate is logged.
            all_entity_results = reranker.rank_pairs(query=question, texts=entity_texts, threshold=None, doc_type="entity")
            # Apply threshold then top-K to determine selection.
            passing = [(i, s) for i, s in all_entity_results if s >= threshold]
            for idx, score in passing[:entity_top_k]:
                eid = entity_ids_list[idx]
                if eid not in _seen_entity_ids:
                    _seen_entity_ids.add(eid)
                    selected_entity_ids.append(eid)
            _jlog(
                "rerank_filter_entities_done",
                request_id,
                step="2.7",
                selected_count=len(selected_entity_ids),
                sample_selected=[
                    {"name": entity_id2meta.get(entity_ids_list[idx], {}).get("name", entity_ids_list[idx]), "score": score}
                    for idx, score in passing[:entity_top_k]
                ],
            )
            selected_top_k_ids = {entity_ids_list[i] for i, _ in passing[:entity_top_k]}
            for rank, (idx, score) in enumerate(all_entity_results, start=1):
                eid = entity_ids_list[idx]
                meta = entity_id2meta.get(eid, {})
                above = score >= threshold
                sel = eid in selected_top_k_ids
                if sel:
                    drop_reason = ""
                elif above:
                    drop_reason = "topk_cutoff"
                else:
                    drop_reason = "below_threshold"
                csv_rows.append({
                    "ts_ms": ts_ms,
                    "request_id": request_id,
                    "item_type": "entity",
                    "item_id": eid,
                    "name": meta.get("name", eid),
                    "score": round(score, 6),
                    "rank": rank,
                    "above_threshold": above,
                    "selected": sel,
                    "threshold": threshold,
                    "top_k": entity_top_k,
                    "drop_reason": drop_reason,
                })

        # Score all relationships
        relationship_texts = []
        relationship_ids_list = []
        for relationship_id in relationship_ids:
            meta = relationship_id2meta.get(relationship_id, {})
            if not meta:
                continue
            source_entity = meta.get("source_entity", "")
            target_entity = meta.get("target_entity", "")
            description = meta.get("description", "")
            keywords = meta.get("keywords", "")
            text = f"{source_entity} -> {target_entity} | {description} (keywords: {keywords})"
            relationship_texts.append(text)
            relationship_ids_list.append(relationship_id)

        selected_relationship_ids: list[str] = []
        _seen_relationship_ids: set[str] = set()
        if relationship_texts:
            all_rel_results = reranker.rank_pairs(query=question, texts=relationship_texts, threshold=None, doc_type="relationship")
            passing = [(i, s) for i, s in all_rel_results if s >= threshold]
            for idx, score in passing[:relationship_top_k]:
                rid = relationship_ids_list[idx]
                if rid not in _seen_relationship_ids:
                    _seen_relationship_ids.add(rid)
                    selected_relationship_ids.append(rid)
            _jlog(
                "rerank_filter_relationships_done",
                request_id,
                step="2.7",
                selected_count=len(selected_relationship_ids),
                sample_selected=[
                    {"name": self._relationship_names_from_ids([relationship_ids_list[idx]])[0], "score": score}
                    for idx, score in passing[:relationship_top_k]
                ],
            )
            selected_top_k_rel_ids = {relationship_ids_list[i] for i, _ in passing[:relationship_top_k]}
            for rank, (idx, score) in enumerate(all_rel_results, start=1):
                rid = relationship_ids_list[idx]
                meta = relationship_id2meta.get(rid, {})
                name = f"{meta.get('source_entity', '')} -> {meta.get('target_entity', '')}"
                above = score >= threshold
                sel = rid in selected_top_k_rel_ids
                if sel:
                    drop_reason = ""
                elif above:
                    drop_reason = "topk_cutoff"
                else:
                    drop_reason = "below_threshold"
                csv_rows.append({
                    "ts_ms": ts_ms,
                    "request_id": request_id,
                    "item_type": "relationship",
                    "item_id": rid,
                    "name": name,
                    "score": round(score, 6),
                    "rank": rank,
                    "above_threshold": above,
                    "selected": sel,
                    "threshold": threshold,
                    "top_k": relationship_top_k,
                    "drop_reason": drop_reason,
                })

        if csv_rows:
            _append_reranker_scores(csv_rows)

        _jlog(
            "rerank_filter_complete",
            request_id,
            step="2.7",
            final_entity_count=len(selected_entity_ids),
            final_relationship_count=len(selected_relationship_ids),
            total_elapsed_sec=timer_total.sec(),
        )

        return selected_entity_ids, selected_relationship_ids
