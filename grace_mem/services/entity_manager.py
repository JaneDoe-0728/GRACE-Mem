"""Entity identity resolution: deciding when two mentions are one entity.

This is the hard problem in building a KG from conversation. The same person is
"Mel", "Melanie", and "my sister" across a corpus; two different people share a
first name. Get it wrong in one direction and the graph fragments into
duplicates that each hold part of the answer; wrong in the other and unrelated
facts merge onto one node and the model answers with someone else's history.

The decision is made in two stages, cheap filter then expensive judge. Vector
search over existing entities proposes a handful of candidates
(`similar_entity_top_k`, above `entity_sim_threshold`), and an LLM call decides
whether the new mention is one of them (UPDATE) or genuinely new (ADD). The
filter exists because the judge cannot be run against every entity in the
graph; the judge exists because cosine similarity alone merges "John Smith" with
"Jane Smith".

Everything else here is about making that LLM call survivable in bulk. Entities
are adjudicated concurrently through a thread pool, rate-limited to stay inside
the backend's budget, retried on failure, and -- when retries are exhausted --
resolved to a fallback ADD. The bias is deliberate and asymmetric: a wrongly
split entity leaves both halves retrievable, while a wrong merge destroys
information irreversibly.
"""

import time
import logging
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from grace_mem.utils.common import (
    tokenize_en,
    canonical_entity_id,
    _entity_key,
    _parse_entity_ops_block,
    canonicalize_entity_type_label,
)
from grace_mem.llm.prompts.entity_ops import ENTITY_OPS_RULES_V2, ENTITY_OPS_FEW_SHOT
from grace_mem.llm.token_tracking import token_tracker
from grace_mem.storage import build_id_to_meta_maps
from grace_mem.utils.logger_config import make_module_jlog
import numpy as np


@dataclass
class EntityOpsConfig:
    """Concurrency and retry budget for entity adjudication.

    Attributes:
        max_workers: Concurrent adjudication calls. Must stay under
            rate_limit_per_minute in practice, or workers spend their time
            blocked in the limiter instead of in flight.
        max_tokens: Ceiling on one adjudication reply. Small because the reply
            is a verdict plus a merged description, not prose.
        max_retries: Attempts before falling back to ADD.
        timeout: Seconds to wait on one entity's result before treating it as
            failed. Generous relative to a single call, since a queued request
            may sit behind the rate limiter first.
        rate_limit_per_minute: Calls per minute across all workers.
    """

    max_workers: int = 10
    max_tokens: int = 800
    max_retries: int = 3
    timeout: int = 120
    rate_limit_per_minute: int = 100


class RateLimiter:
    """Thread-safe sliding-window limiter over a shared call budget.

    Sliding rather than fixed-window: a fixed window lets a burst at the end of
    one minute and another at the start of the next exceed the nominal rate
    over any 60-second span, which is what the backend actually measures.
    """

    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        """Initialize the call budget and tracking state for rate limiting."""
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Block until this caller is within budget, then claim a slot.

        Sleeps while holding the lock, which serializes all workers behind one
        sleeper. That is intentional: once the budget is spent, every other
        worker would have to wait anyway, and releasing the lock would let them
        wake and re-check in a spin.
        """
        with self.lock:
            now = time.time()
            self.calls = [c for c in self.calls if now - c < self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    print(f"⏳ Rate limit reached, sleeping {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    self.calls = []
            self.calls.append(time.time())


def _classify_entity_action(action: Any) -> Optional[str]:
    """Return the normalized action type when the input contains ADD or UPDATE."""
    action_text = str(action or "").strip().upper()
    if "UPDATE" in action_text:
        return "UPDATE"
    if "ADD" in action_text:
        return "ADD"
    return None


def _normalize_entity_action(action: Optional[str], target_id: Any, valid_ids: set[Any]) -> Tuple[str, Any]:
    """Force an adjudication verdict into a safe (action, target) pair.

    The model can return an UPDATE naming an id that was never a candidate --
    hallucinated, or copied from an earlier example. Trusting it would merge
    the new entity onto an arbitrary existing node.

    So an unusable target is repaired: with exactly one candidate available the
    intent is unambiguous and that candidate is used; with several there is no
    way to pick, and the action degrades to ADD. Degrading to ADD is always
    safe -- it can leave a duplicate node, but it never fuses two real entities.

    Returns:
        (action, target_id), where target_id is always None for ADD.
    """
    if not action:
        return "ADD", None
    if action == "UPDATE":
        if not target_id or target_id not in valid_ids:
            target_id = next(iter(valid_ids)) if len(valid_ids) == 1 else None
            action = "UPDATE" if target_id else "ADD"
    if action == "ADD":
        target_id = None
    return action, target_id


class EntityOpsProcessor:
    """Adjudicates a batch of extracted entities against existing graph nodes.

    Takes `generate_fn` rather than an LLM client so the ablation that skips
    adjudication can substitute a deterministic function without touching the
    concurrency and retry machinery around it.
    """

    def __init__(self, generate_fn: Callable[..., Any], config: EntityOpsConfig = None) -> None:
        """Configure the batch entity-op generator and its rate limiter."""
        self.generate_fn = generate_fn
        self.config = config or EntityOpsConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)

    def process_batch(self, entities: List["EntityInput"], similar_map: "SimilarMap") -> "EntityOpsBatchResult":
        """Adjudicate every entity in the batch concurrently.

        Results are written back by input index, not appended as futures
        complete: the caller pairs `results[i]` with `entities[i]`, so
        completion order must not leak into the output.

        An entity whose adjudication fails outright still gets a result -- a
        fallback ADD -- rather than a hole in the list. One unresolved entity
        must not abort a turn's ingestion.

        Args:
            entities: Newly extracted entities awaiting a verdict.
            similar_map: Candidate existing entities per (name, type), from the
                upstream vector search.

        Returns:
            `{"results": [...]}` positionally aligned with `entities`.
        """
        if not entities:
            return {"results": []}

        print(f"\n🚀 Processing {len(entities)} entities (parallel={self.config.max_workers})...")

        # Captured here and re-applied inside each worker: the token tracker
        # keys its context to the current thread, so pool threads would
        # otherwise attribute their usage to "unknown" and drop this stage out
        # of the per-stage cost report.
        ctx_dataset, ctx_stage = token_tracker._get_context()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_with_retry, e, similar_map, ctx_dataset, ctx_stage): i
                for i, e in enumerate(entities)
            }
            results = [None] * len(entities)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=self.config.timeout)
                except Exception:
                    results[idx] = self._create_fallback_result(entities[idx])

        return {"results": results}

    def _process_with_retry(self, entity: "EntityInput", similar_map: "SimilarMap", ctx_dataset: str = "unknown", ctx_stage: str = "unknown") -> "EntityOp":
        """Retry one entity-op request while restoring the caller's token-tracker context."""
        token_tracker.set_context(dataset=ctx_dataset, stage=ctx_stage)
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                return self._process_single(entity, similar_map)
            except Exception:
                if attempt == self.config.max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"⚠️  Retry {attempt+1}/{self.config.max_retries} for {entity['entity_name']}, wait {wait}s")
                time.sleep(wait)

    def _process_single(self, entity: "EntityInput", similar_map: "SimilarMap") -> "EntityOp":
        """Run one entity through prompt construction, LLM generation, and validation."""
        name, type_val, desc = self._extract_entity_info(entity)
        candidates = similar_map.get((name, type_val), [])
        prompt = self._build_prompt(name, type_val, desc, candidates)
        raw_output, _ = self.generate_fn(prompt, max_tokens=self.config.max_tokens)
        parsed = _parse_entity_ops_block(raw_output or "")
        parsed_result = parsed.get("results", [{}])[0] if parsed.get("results") else {}
        result = self._validate_result(parsed_result, entity, candidates)
        print(f"🔍 Parsed result for '{name}':\n{result}\n")
        return result

    def _extract_entity_info(self, entity: "EntityInput") -> tuple[str, str, str]:
        """Read the normalized name, type, and description from an entity payload."""
        name = entity["entity_name"]
        type_val = entity["entity_type"].value if hasattr(entity["entity_type"], "value") else entity["entity_type"]
        type_val = canonicalize_entity_type_label(str(type_val))
        desc = entity.get("entity_description", "")
        return name, type_val, desc

    def _build_prompt(self, name: str, type_val: str, desc: str, candidates: List["VDBSearchHit"]) -> str:
        """Build the entity-op prompt including any similar-entity candidates."""
        lines = [f"[INPUT] name={name} | type={type_val} | desc={desc}"]
        if candidates:
            lines.append("Candidates:")
            valid_ids = []
            for meta, score in candidates:
                cid = meta.get("id", "?")
                valid_ids.append(cid)
                lines.append(
                    f"- id={cid} | name={meta.get('name')} | type={meta.get('type')} | "
                    f"score={score:.3f} | from={meta.get('_source', '?')} | desc={meta.get('description', '')}"
                )
            quoted_ids = ",".join([f'"{x}"' for x in valid_ids])
            lines.append(f"Valid target_existing_id choices: [{quoted_ids}]")
        else:
            lines.append("Candidates: (none)")
            lines.append("Valid target_existing_id choices: []  # No candidates -> only valid action is ADD")
        prompt_block = "\n".join(lines)
        print(f"📝 Built prompt for entity '{name}':\n{prompt_block}\n")
        return f"{ENTITY_OPS_RULES_V2}\n\n{ENTITY_OPS_FEW_SHOT}\n=== SINGLE ENTITY ===\n{prompt_block}"

    def _validate_result(self, parsed: "EntityOp", entity: "EntityInput", candidates: List["VDBSearchHit"]) -> "EntityOp":
        """Normalize raw LLM output into a safe ADD or UPDATE entity operation."""
        name, type_val, desc = self._extract_entity_info(entity)
        action = _classify_entity_action(parsed.get("action"))
        target_id = parsed.get("target_existing_id")
        canonical_name = parsed.get("canonical_name") or name
        canonical_type = canonicalize_entity_type_label(parsed.get("canonical_type") or type_val)
        merged_desc = (parsed.get("merged_description") or "").strip()
        valid_ids = {meta.get("id") for meta, _ in candidates}

        action, target_id = _normalize_entity_action(action, target_id, valid_ids)
        if not merged_desc:
            merged_desc = desc or f"{name} ({type_val})"
        return {
            "input_name": name,
            "input_type": type_val,
            "action": action,
            "target_existing_id": target_id,
            "canonical_name": canonical_name,
            "canonical_type": canonical_type,
            "merged_description": merged_desc,
            "entity_metadata": entity.get("entity_metadata"),
        }

    def _create_fallback_result(self, entity: "EntityInput") -> "EntityOp":
        """Create a default ADD operation when entity-op generation fails."""
        name, type_val, desc = self._extract_entity_info(entity)
        return {
            "input_name": name,
            "input_type": type_val,
            "action": "ADD",
            "target_existing_id": None,
            "canonical_name": name,
            "canonical_type": type_val,
            "merged_description": desc or f"{name} ({type_val})",
            "entity_metadata": entity.get("entity_metadata"),
        }

_jlog = make_module_jlog(name="grace_mem.EntityManager", filename="kg_ingestor.jsonl")
_log = logging.getLogger(__name__)

Meta = Dict[str, Any]
EntityInput = Dict[str, Any]
EntityOp = Dict[str, Any]
EntityOpsBatchResult = Dict[str, List[EntityOp]]
EntityLike = Union[EntityInput, Any]
KeyNameType = Tuple[str, str]
VDBSearchHit = Tuple[Meta, float]
KeyNameTypeDesc = Tuple[str, str, str]  # (name, type, desc)
SimilarMap = Dict[KeyNameType, List[VDBSearchHit]]
_TEMPORAL_ANCHOR_TYPES = {"date", "time", "timespan"}

class EntityManager:
    """
    Normalize incoming entities, look up similar ones, and write the added or
    updated entities into the vector store and the cache.
    """
    def __init__(
        self,
        *,
        embedder: Any,
        mgr: Any,
        provenance: Any,
        GLOBAL_CACHE: Dict[str, Any],
        processed_ent_map: Dict[str, Meta],
        processed_ent_full_map: Dict[KeyNameTypeDesc, Meta],
    ) -> None:
        """Store entity persistence dependencies and processed-entity caches."""
        self._embedder = embedder
        self._mgr = mgr
        self._prov = provenance
        self._GLOBAL_CACHE = GLOBAL_CACHE
        self._processed = processed_ent_map
        self._processed_full = processed_ent_full_map

    @staticmethod
    def _build_entity_repr(name: str, type_val: str, description: str) -> str:
        """
        Compose (name, type, desc) into text that carries the entity's meaning,
        for embedding.
        Example: 'Apple [type=Company] iPhone manufacturer'
        """
        name = (name or "").strip()
        type_val = (type_val or "").strip()
        description = (description or "").strip()
        if name:
            return f"{name} [type={type_val}] {description}".strip()
        return f"{type_val} {description}".strip()

    @staticmethod
    def _to_name_type_desc(e: EntityLike) -> KeyNameTypeDesc:
        """Extract normalized name, type, and description from a dict or object entity."""
        if isinstance(e, dict):
            name = e.get("entity_name") or e.get("name")
            type_val = e.get("entity_type") or e.get("type")
            desc = e.get("entity_description") or e.get("description") or ""
        else:
            name = getattr(e, "entity_name", None) or getattr(e, "name", None)
            type_val = getattr(e, "entity_type", None) or getattr(e, "type", None)
            desc = getattr(e, "entity_description", None) or getattr(e, "description", None) or ""
        if hasattr(type_val, "value"):
            type_str = type_val.value
        else:
            type_str = str(type_val) if type_val is not None else ""
        type_str = canonicalize_entity_type_label(type_str)
        return (name or "").strip(), (type_str or "").strip(), (desc or "").strip()

    def normalize_entities(self, entities: Iterable[EntityLike]) -> List[EntityInput]:
        """Convert mixed entity inputs into the canonical dict representation."""
        out: List[EntityInput] = []
        for e in entities or []:
            name, type_str, desc = self._to_name_type_desc(e)
            if not name:
                continue
            metadata = e.get("entity_metadata") if isinstance(e, dict) else getattr(e, "entity_metadata", None)
            out.append({"entity_name": name, "entity_type": type_str, "entity_description": desc, "entity_metadata": metadata})
        return out

    @staticmethod
    def _is_temporal_anchor_type(type_str: str) -> bool:
        """Return whether the entity type is a temporal anchor type."""
        return (type_str or "").strip().lower() in _TEMPORAL_ANCHOR_TYPES

    def _find_exact_temporal_candidates(self, name: str, type_str: str) -> List[VDBSearchHit]:
        """Return exact-name candidates for temporal anchors from in-memory caches.

        Temporal anchors are intentionally conservative: they should only merge
        when both entity type and canonical entity_name match exactly.
        """
        key = _entity_key(name, type_str)
        candidates: List[VDBSearchHit] = []
        seen_ids: set[str] = set()

        direct_sources = [
            self._processed.get(key),
            self._GLOBAL_CACHE.get("entities", {}).get(key),
        ]
        for meta in direct_sources:
            if not isinstance(meta, dict):
                continue
            eid = meta.get("id")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            meta_obj = dict(meta)
            meta_obj["_source"] = "exact_name"
            candidates.append((meta_obj, 1.0))

        if candidates:
            return candidates

        for meta in self._GLOBAL_CACHE.get("entities", {}).values():
            if not isinstance(meta, dict):
                continue
            if _entity_key(meta.get("name", ""), meta.get("type", "")) != key:
                continue
            eid = meta.get("id")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            meta_obj = dict(meta)
            meta_obj["_source"] = "exact_name"
            candidates.append((meta_obj, 1.0))

        return candidates

    
    def find_similar_for_hybrid(
            self, entities: Iterable[EntityLike], top_k: int = 5, threshold: float = 0.6,
        ) -> SimilarMap:
        """Find candidate existing entities by combining vector and BM25 retrieval."""

        normalized = self.normalize_entities(entities)
        if not normalized:
            return {}

        texts = [
            self._build_entity_repr(e["entity_name"], e["entity_type"], e.get("entity_description") or "")
            for e in normalized
        ]
        keys = [(e["entity_name"], e["entity_type"]) for e in normalized]

        # embed vectors
        vecs = self._embedder.embed(texts)
        ent_vdb = self._mgr.get_entities_vdb(vecs.shape[1])

        # BM25 init
        try:
            bm25 = self._mgr.get_entities_bm25(load_if_empty=True)
            metas = getattr(bm25, "metas", []) or []
        except Exception:
            bm25 = None
            metas = []

        out: SimilarMap = {}

        search_indexes: list[int] = []
        for i, e in enumerate(normalized):
            if self._is_temporal_anchor_type(e["entity_type"]):
                exact_candidates = self._find_exact_temporal_candidates(e["entity_name"], e["entity_type"])
                out[keys[i]] = exact_candidates
                continue
            search_indexes.append(i)

        if not search_indexes:
            return out

        # --- Batch vector search: one VDB query for all N entities ---
        _t_vec_start = time.time()
        _batch_vec_results: Optional[List] = None
        try:
            _batch_vec_results = ent_vdb.batch_search(vecs[search_indexes], top_k=top_k, threshold=threshold)
        except Exception as _batch_exc:
            _log.warning("[find_similar_for_hybrid] batch_search failed, falling back to per-entity search: %s", _batch_exc)
            _batch_vec_results = None
        _t_vec_end = time.time()
        _log.info(
            "[find_similar_for_hybrid] vector search: n_entities=%d mode=%s elapsed=%.4fs",
            len(search_indexes),
            "batch" if _batch_vec_results is not None else "per-entity",
            _t_vec_end - _t_vec_start,
        )

        for search_pos, i in enumerate(search_indexes):
            key = keys[i]
            name = normalized[i]["entity_name"]
            desc = normalized[i].get("entity_description", "") or ""

            # -------------------------------
            # BM25 branch (handled first)   # MODIFIED
            # -------------------------------
            bm25_list: List[Tuple[Dict[str, Any], float]] = []
            bm25_ids = set()

            if bm25 and metas:
                q_tokens = tokenize_en(f"{name} {desc}")

                if q_tokens:
                    name_scores, desc_scores = bm25.get_scores(q_tokens)
                    name_scores = np.asarray(name_scores, dtype=float)
                    desc_scores = np.asarray(desc_scores, dtype=float)

                    scores = np.maximum(name_scores, desc_scores)
                    max_score = float(scores.max()) if scores.size else 0.0
                    bm25_threshold = round(0.6 * max_score, 1)
                    strong_mask = (scores >= bm25_threshold) if max_score > 0 else np.zeros_like(scores, dtype=bool)
                    idxs = np.where(strong_mask)[0]

                    idxs = list(idxs)[::-1]

                    bm25_best: Dict[str, Tuple[float, int]] = {}

                    for idx in idxs:
                        meta_i = metas[idx]
                        if not meta_i:
                            continue
                        eid = meta_i.get("id")
                        if not eid:
                            continue

                        score = float(scores[idx])

                        if eid in bm25_best:
                            continue

                        bm25_best[eid] = (score, idx)

                    # Assemble (meta, score, from=bm25)
                    for eid, (score, best_idx) in bm25_best.items():
                        meta_obj = dict(metas[best_idx])
                        meta_obj["_source"] = "bm25"
                        bm25_list.append((meta_obj, score))
                        bm25_ids.add(eid)

            # -------------------------------
            # Vector search (after BM25)
            # -------------------------------
            if _batch_vec_results is not None:
                vec_hits = _batch_vec_results[search_pos] or []
            else:
                # Fallback: per-entity search (original behavior)
                _t_single_start = time.time()
                try:
                    vec_hits = ent_vdb.search(vecs[i], top_k=top_k, threshold=threshold) or []
                except Exception:
                    vec_hits = []
                _log.debug(
                    "[find_similar_for_hybrid] per-entity vector search entity=%d elapsed=%.4fs",
                    i, time.time() - _t_single_start,
                )

            vec_list: List[Tuple[Dict[str, Any], float]] = []
            vec_ids = set()

            for meta, sim in vec_hits:
                if not meta:
                    continue

                eid = meta.get("id") or meta.get("summary_id") or meta.get("name")
                if not eid or eid in vec_ids or eid in bm25_ids:
                    continue

                meta_obj = dict(meta)
                meta_obj["_source"] = "vector"
                vec_ids.add(eid)
                vec_list.append((meta_obj, float(sim)))

            # -------------------------------
            # Fuse the results: BM25 -> Vector
            # -------------------------------
            merged: List[Tuple[Dict[str, Any], float]] = []
            merged.extend(bm25_list)
            merged.extend(vec_list)

            out[key] = merged

        return out
    
    def apply_ops(self, ops_results: Dict[str, Any], provenance: Dict[str, Any] | None = None, *, request_id: str = "UNKNOWN") -> Tuple[Dict[KeyNameType, Meta], Dict[KeyNameType, Meta], Dict[str, int]]:
        """
        - read every action in ops_results['results']
        - ADD: build a new meta, embed it, write it to the VDB, refresh the
          processed cache
        - UPDATE: fetch the existing meta by target_existing_id, update its
          description/evidence, write it back
        Returns: (entity_idx, input2resolved, stats)
        """
        ent_id2meta, _ = build_id_to_meta_maps(self._GLOBAL_CACHE)
        added = updated = 0
        texts: List[str] = []
        metas: List[Meta] = []
        entity_idx: Dict[KeyNameType, Meta] = {}
        input2resolved: Dict[KeyNameType, Meta] = {}

        for r in (ops_results or {}).get("results", []):
            in_key: KeyNameType = (r.get("input_name", ""), r.get("input_type", ""))
            action = _classify_entity_action(r.get("action"))

            def _add(final_name: str, final_type: str, final_desc: str) -> None:
                """Stage a new canonical entity for cache and vector-store insertion."""
                nonlocal added
                final_type = canonicalize_entity_type_label(final_type)
                eid = canonical_entity_id(final_name, final_type)
                key_nt = _entity_key(final_name, final_type)
                meta = {"id": eid, "name": final_name, "type": final_type,
                        "description": final_desc, "prov": self._prov.merge_prov(None, provenance)} # provenance: entity related conversation
                temporal_meta = r.get("entity_metadata")
                if temporal_meta:
                    meta["temporal"] = temporal_meta.get("temporal", temporal_meta)
                texts.append(f"{final_name} [type={final_type}] {final_desc}")
                metas.append(meta)
                self._processed[key_nt] = meta
                self._processed_full[(final_name, final_type, final_desc)] = meta
                entity_idx[key_nt] = meta
                input2resolved[in_key] = meta
                added += 1
            
            # add operation
            if action == "ADD":
                _add(
                    (r.get("canonical_name") or "").strip(),
                    canonicalize_entity_type_label((r.get("canonical_type") or "").strip()),
                    (r.get("merged_description") or "").strip(),
                )

            # update operation
            elif action == "UPDATE":
                tgt_id = r.get("target_existing_id")
                existing = ent_id2meta.get(tgt_id) or {}
                if not existing:
                    _add(
                        (r.get("canonical_name") or "").strip() or (r.get("input_name") or "").strip(),
                        canonicalize_entity_type_label((r.get("canonical_type") or "").strip())
                        or canonicalize_entity_type_label((r.get("input_type") or "").strip()),
                        (r.get("merged_description") or "").strip(),
                    )
                    continue

                new_desc = (r.get("merged_description") or "").strip()
                updated_meta = {
                    "id": existing["id"],
                    "name": existing.get("name", ""),
                    "type": canonicalize_entity_type_label(existing.get("type", "")),
                    "description": new_desc,
                    "prov": self._prov.merge_prov(existing.get("prov"), provenance),
                }
                temporal_meta = r.get("entity_metadata") or existing.get("temporal")
                if temporal_meta:
                    updated_meta["temporal"] = temporal_meta.get("temporal", temporal_meta) if isinstance(temporal_meta, dict) else temporal_meta
                texts.append(f"{updated_meta['name']} [type={updated_meta['type']}] {new_desc}")
                metas.append(updated_meta)

                key_nt = _entity_key(updated_meta["name"], updated_meta["type"])
                self._processed[key_nt] = updated_meta
                self._processed_full[(updated_meta["name"], updated_meta["type"], new_desc)] = updated_meta
                entity_idx[key_nt] = updated_meta
                input2resolved[in_key] = updated_meta
                updated += 1

            else:
                _add(
                    (r.get("canonical_name") or r.get("input_name") or "").strip(),
                    canonicalize_entity_type_label((r.get("canonical_type") or r.get("input_type") or "").strip()),
                    (r.get("merged_description") or "").strip(),
                )

        if texts:
            vecs = self._embedder.embed(texts)
            ent_vdb = self._mgr.get_entities_vdb(vecs.shape[1])
            ent_vdb.add(vecs, metas)

            bm25 = self._mgr.get_entities_bm25(load_if_empty=True)
            _jlog("entity_bm25_before_add", request_id, bm25_size=bm25.size, metas_count=len(bm25.metas))
            for m in metas:
                name = (m.get("name") or "").strip()
                desc = (m.get("description") or "").strip()
                bm25.add(tokenize_en(name), tokenize_en(desc), m)
            _jlog("entity_bm25_after_add", request_id, bm25_size=bm25.size, metas_count=len(bm25.metas))

            self._mgr.persist_async()
            _jlog("entity_ops_upsert_done", request_id, upsert_count=len(texts), added=added, updated=updated)

        return entity_idx, input2resolved, {"added": added, "updated": updated}
