# services/entity_manager.py
import time
import logging
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from KG.utils.common import (
    tokenize_en,
    canonical_entity_id,
    _entity_key,
    _parse_entity_ops_block,
    canonicalize_entity_type_label,
)
from KG.llm.prompts.entity_ops import ENTITY_OPS_RULES_V2, ENTITY_OPS_FEW_SHOT
from KG.llm.token_tracking import token_tracker
from KG.storage import build_id_to_meta_maps
from KG.utils.logger_config import make_module_jlog
import numpy as np


@dataclass
class EntityOpsConfig:
    max_workers: int = 10
    max_tokens: int = 800
    max_retries: int = 3
    timeout: int = 120
    rate_limit_per_minute: int = 100


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        """Initialize the call budget and tracking state for rate limiting."""
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Pause until another call is allowed within the configured window."""
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
    """Resolve a parsed action into a safe final action/target pair."""
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
    def __init__(self, generate_fn: Callable[..., Any], config: EntityOpsConfig = None) -> None:
        """Configure the batch entity-op generator and its rate limiter."""
        self.generate_fn = generate_fn
        self.config = config or EntityOpsConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)

    def process_batch(self, entities: List["EntityInput"], similar_map: "SimilarMap") -> "EntityOpsBatchResult":
        """Generate entity operations for a batch and preserve input ordering."""
        if not entities:
            return {"results": []}

        print(f"\n🚀 Processing {len(entities)} entities (parallel={self.config.max_workers})...")

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

_jlog = make_module_jlog(name="KG.EntityManager", filename="kg_ingestor.jsonl")
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
    將輸入實體轉換為統一格式、尋找相似實體，將新增/更新的實體寫入向量庫與快取
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
        把 (name, type, desc) 組成能代表實體語義的文字，用於嵌入。
        例：'Apple [type=Company] iPhone 製造商'
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
            # BM25 部分（先處理）   # MODIFIED
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

                    # 組合 (meta, score, from=bm25)
                    for eid, (score, best_idx) in bm25_best.items():
                        meta_obj = dict(metas[best_idx])  # copy
                        meta_obj["_source"] = "bm25"      # NEW: 標記來源
                        bm25_list.append((meta_obj, score))
                        bm25_ids.add(eid)

            # -------------------------------
            # Vector 搜尋（BM25 之後）
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
                meta_obj["_source"] = "vector"          # NEW: 標記來源
                vec_ids.add(eid)
                vec_list.append((meta_obj, float(sim)))

            # -------------------------------
            # 融合結果：BM25 → Vector
            # -------------------------------
            merged: List[Tuple[Dict[str, Any], float]] = []
            merged.extend(bm25_list)
            merged.extend(vec_list)

            out[key] = merged

        return out
    
    def apply_ops(self, ops_results: Dict[str, Any], provenance: Dict[str, Any] | None = None, *, request_id: str = "UNKNOWN") -> Tuple[Dict[KeyNameType, Meta], Dict[KeyNameType, Meta], Dict[str, int]]:
        """
        - 讀取 ops_results['results'] 每筆 action
        - ADD：建立新 meta + 嵌入 + 寫入 VDB + 更新 processed 快取
        - UPDATE：以 target_existing_id 取出既有 meta，更新描述/證據，再寫回
        回傳： (entity_idx, input2resolved, stats)
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
                eid = canonical_entity_id(final_name, final_type) # format entity id
                key_nt = _entity_key(final_name, final_type) # key = name + type
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
