# services/relationship_manager.py
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from KG.utils.common import canonical_rel_id
from KG.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="KG.RelationshipManager", filename="kg_ingestor.jsonl")

Meta = Dict[str, object]
KeyNameType = Tuple[str, str]  # (input_name, input_type)
RelKeyST = Tuple[str, str]     # (source_id, target_id)
RelKeySTD = Tuple[str, str, str]  # (source_id, target_id, description)

class RelationshipManager:
    """
    將抽取到的關係對應到已解析好的實體，合併描述與關鍵詞後，存入向量資料庫與快取
    """
    def __init__(
        self,
        *,
        embedder: Any,                          # object: 需有 .embed(List[str]) -> np.ndarray
        mgr: Any,                               # object: 需有 .get_relationships_vdb(dim), .persist_async()
        provenance: Any,                        # object: 需有 .merge_prov(old, new)
        GLOBAL_CACHE: Dict[str, Any],           # object/state
        processed_rel_map: Dict[RelKeyST, Meta],
        processed_rel_full_map: Dict[RelKeySTD, Meta]
    ) -> None:
        """Store relationship persistence dependencies and processed caches."""
        self._embedder = embedder
        self._mgr = mgr
        self._prov = provenance
        self._GLOBAL_CACHE = GLOBAL_CACHE
        self._processed = processed_rel_map
        self._processed_full = processed_rel_full_map

    # ---- 只用 input2resolved 來對實體 ----
    @staticmethod
    def _resolve_via_input2resolved(
        name: Optional[str],
        input2resolved: Dict[KeyNameType, Meta] | None
    ) -> Optional[Meta]:
        """Resolve an extracted endpoint name to its canonical entity metadata."""
        if not name or not input2resolved:
            return None
        # 如果你的 input2resolved key 只放 name，可改成 for (in_name, _type) in input2resolved ...
        for (in_name, _), meta in input2resolved.items():
            if in_name == name:
                return meta
        return None
    
    @staticmethod
    def _check_mappings(relationships: Iterable[Any], input2resolved: Dict[KeyNameType, Meta] | None) -> set[str]:
        """
        確保每個關係的 source/target 名稱，都能在 input2resolved 找到映射。
        input2resolved keys: (input_name, input_type) -> meta
        這裡只用 input_name 做精準比對。
        """
        missing: set[str] = set()
        if not relationships:
            return missing
        names_in_map = {(in_name or "").strip()
                        for (in_name, _t) in (input2resolved or {}).keys()}
        for r in relationships:
            if (r.source_entity or "").strip() not in names_in_map:
                missing.add((r.source_entity or "").strip())
            if (r.target_entity or "").strip() not in names_in_map:
                missing.add((r.target_entity or "").strip())
        return missing

    def upsert_from_extraction(
        self,
        result: Any,                 # ExtractionResult (含 result.relationships)
        provenance: Optional[dict] = None,
        input2resolved: Dict[KeyNameType, Meta] | None = None,
        *,
        request_id: str = "UNKNOWN",
        sync_to_graph: bool = False,
        sync_fn: Optional[Callable[[List[Meta]], int]] = None           # callable: List[Meta] -> int
    ) -> List[Meta]:
        """
        先確認關係兩端的實體已對應到內部 ID
        若已存在 → 合併描述/關鍵詞並更新
        若不存在 → 新增關係並嵌入向量庫
        最後回傳成功處理的關係 meta 列表。
        """
        rels = getattr(result, "relationships", None)
        if not rels:
            return []
        
        missing = self._check_mappings(rels, input2resolved)
        if missing:
            _jlog("relationship_endpoint_mapping_missing", request_id, missing_names=sorted(missing))
            
        texts: List[str] = []
        metas: List[Meta] = []
        skipped = 0

        for r in rels:
            src_meta = self._resolve_via_input2resolved(r.source_entity, input2resolved) 
            tgt_meta = self._resolve_via_input2resolved(r.target_entity, input2resolved)

            if not src_meta or not tgt_meta:
                skipped += 1
                _jlog("relationship_skipped", request_id,
                      source_entity=r.source_entity, target_entity=r.target_entity,
                      missing_side="source" if not src_meta else "target")
                continue

            sid, tid = src_meta.get("id"), tgt_meta.get("id")
            src_type, tgt_type = src_meta.get("type"), tgt_meta.get("type")

            key_st: RelKeyST = (sid, tid)
            key_std: RelKeySTD = (sid, tid, r.relationship_description or "")

            # 去重（同 sid/tid/description）
            if key_std in self._processed_full:
                continue

            if key_st in self._processed:
                existing = self._processed[key_st]

                merged_desc = existing.get("description") or ""
                if r.relationship_description and r.relationship_description not in merged_desc:
                    merged_desc = f"{merged_desc}; {r.relationship_description}" if merged_desc else r.relationship_description #add ; between descriptions

                merged_kw = existing.get("keywords") or ""
                if r.relationship_keywords and r.relationship_keywords not in merged_kw:
                    merged_kw = f"{merged_kw}, {r.relationship_keywords}" if merged_kw else r.relationship_keywords

                rid = canonical_rel_id(sid, tid)
                meta = {
                    "id": rid,
                    "source_entity": src_meta.get("name"),
                    "target_entity": tgt_meta.get("name"),
                    "description": merged_desc,
                    "keywords": merged_kw,
                    "source_id": sid, "target_id": tid,
                    "source_type": src_type, "target_type": tgt_type,
                    "prov": self._prov.merge_prov(existing.get("prov"), provenance)
                }
                text = f"{meta['source_entity']} -> {meta['target_entity']} | {merged_desc} (keywords: {merged_kw})"
                texts.append(text); metas.append(meta)
                self._processed[key_st] = meta
                self._processed_full[key_std] = meta

            else:
                rid = canonical_rel_id(sid, tid)
                desc = r.relationship_description or ""
                kw = r.relationship_keywords or ""
                meta = {
                    "id": rid,
                    "source_entity": src_meta.get("name"),
                    "target_entity": tgt_meta.get("name"),
                    "description": desc,
                    "keywords": kw,
                    "source_id": sid, "target_id": tid,
                    "source_type": src_type, "target_type": tgt_type,
                    "prov": self._prov.merge_prov(None, provenance)
                }
                text = f"{meta['source_entity']} -> {meta['target_entity']} | {desc} (keywords: {kw})"
                texts.append(text); metas.append(meta)
                self._processed[key_st] = meta
                self._processed_full[key_std] = meta

        if texts:
            vecs: np.ndarray = self._embedder.embed(texts)  # shape = (n, dim), 已 normalize
            rel_vdb = self._mgr.get_relationships_vdb(vecs.shape[1])
            rel_vdb.add(vecs, metas)
            self._mgr.persist_async()
            _jlog("relationship_upsert_done", request_id, upsert_count=len(texts))

            if sync_to_graph and sync_fn:
                try:
                    ct = sync_fn(metas)
                    _jlog("relationship_neo4j_sync_done", request_id, synced_count=ct)
                except Exception as e:
                    _jlog("relationship_neo4j_sync_failed", request_id, error=str(e), error_type=type(e).__name__)

        if skipped:
            _jlog("relationship_skipped_total", request_id, skipped_count=skipped)

        return metas
