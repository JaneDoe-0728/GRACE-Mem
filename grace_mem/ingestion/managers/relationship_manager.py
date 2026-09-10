"""Relationship persistence: turning extracted edges into graph edges.

Runs strictly after entity adjudication, and that ordering is the module's
central constraint. Extraction names relationship endpoints by surface string
("Mel said this to her sister"), but adjudication may have merged, renamed, or
dropped those entities. So every endpoint has to be resolved through
`input2resolved` before an edge can be written, and an edge whose endpoint did
not survive is dropped rather than written dangling -- a graph edge pointing at
a node that does not exist breaks traversal for every query that reaches it.

Merging is looser here than for entities. Two edges between the same ordered
pair are treated as the same relationship and their descriptions concatenated,
with no LLM adjudication. The asymmetry is deliberate: entity identity is
ambiguous and expensive to get wrong, whereas an edge is already pinned by two
resolved endpoints, which leaves far less room for a wrong merge.
"""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from grace_mem.data_model.relationships import canonical_rel_id
from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.RelationshipManager", filename="kg_ingestor.jsonl")

Meta = dict[str, object]
KeyNameType = tuple[str, str]  # (input_name, input_type)
# The two relationship keyings are not redundant. RelKeyST identifies the edge
# and is what merging collapses onto; RelKeySTD additionally carries the
# description, which preserves each distinct statement about that edge for
# retrieval even after the edge itself has been merged.
RelKeyST = tuple[str, str]     # (source_id, target_id)
RelKeySTD = tuple[str, str, str]  # (source_id, target_id, description)

class RelationshipManager:
    """Resolve, merge, and persist extracted relationships.

    Holds no state of its own: the caches and the vector-store manager are
    injected and shared with the entity side, so that one turn's writes are
    visible to the next without a round trip through storage.
    """

    def __init__(
        self,
        *,
        embedder: Any,                          # must provide .embed(List[str]) -> np.ndarray
        mgr: Any,                               # must provide .get_relationships_vdb(dim), .persist_async()
        provenance: Any,                        # must provide .merge_prov(old, new)
        global_cache: dict[str, Any],
        processed_rel_map: dict[RelKeyST, Meta],
        processed_rel_full_map: dict[RelKeySTD, Meta]
    ) -> None:
        """Store the injected collaborators and the two processed-edge caches.

        Collaborators are typed `Any` and specified by the comments above
        because the experiment harness substitutes fakes for all three;
        requiring a concrete class here would force those fakes to inherit from
        implementations they do not otherwise use.
        """
        self._embedder = embedder
        self._mgr = mgr
        self._prov = provenance
        self._global_cache = global_cache
        self._processed = processed_rel_map
        self._processed_full = processed_rel_full_map

    # ---- resolve entities through input2resolved only ----
    @staticmethod
    def _resolve_via_input2resolved(
        name: str | None,
        input2resolved: dict[KeyNameType, Meta] | None
    ) -> Meta | None:
        """Look up the resolved entity behind an extracted endpoint name.

        Matches on name only, ignoring the type half of the key. Extraction
        labels the same entity with different types across turns ("Mel" as
        Person in one, Topic in another), so requiring both to agree would drop
        edges whose endpoint was resolved perfectly well.

        The consequence is that the first match wins when one name was
        extracted under two types. A linear scan rather than a dict lookup for
        the same reason -- the key is a pair and only half of it is being
        compared. Endpoint counts per turn are small enough that this does not
        matter.
        """
        if not name or not input2resolved:
            return None
        for (in_name, _), meta in input2resolved.items():
            if in_name == name:
                return meta
        return None
    
    @staticmethod
    def _check_mappings(relationships: Iterable[Any], input2resolved: dict[KeyNameType, Meta] | None) -> set[str]:
        """Report endpoint names that no resolved entity accounts for.

        Diagnostic only -- the caller logs the result and carries on, because
        the per-edge resolution below drops these anyway. It is reported
        separately because an edge silently vanishing is the expected symptom
        of an extraction problem upstream, and a name that never resolves is
        the evidence for it.

        Compares names exactly and ignores the type half of the key, matching
        `_resolve_via_input2resolved`; the two must agree or this reports
        misses that do not occur.
        """
        missing: set[str] = set()
        if not relationships:
            return missing
        names_in_map = {(in_name or "").strip()
                        for (in_name, _t) in (input2resolved or {})}
        for r in relationships:
            if (r.source_entity or "").strip() not in names_in_map:
                missing.add((r.source_entity or "").strip())
            if (r.target_entity or "").strip() not in names_in_map:
                missing.add((r.target_entity or "").strip())
        return missing

    def upsert_from_extraction(
        self,
        result: Any,                 # ExtractionResult (carries result.relationships)
        provenance: dict | None = None,
        input2resolved: dict[KeyNameType, Meta] | None = None,
        *,
        request_id: str = "UNKNOWN",
        sync_to_graph: bool = False,
        sync_fn: Callable[[list[Meta]], int] | None = None           # callable: List[Meta] -> int
    ) -> list[Meta]:
        """Resolve, merge, and persist every relationship in an extraction result.

        Per edge: both endpoints must resolve, or the edge is skipped and
        logged. An edge already present under the same (source, target) has its
        description and keywords merged into the existing record; a new one is
        embedded and written.

        Args:
            result: An ExtractionResult; only `.relationships` is read.
            provenance: Origin blob merged into each edge's existing
                provenance, so a re-stated relationship accumulates every turn
                that mentioned it.
            input2resolved: (name, type) -> resolved entity meta, produced by
                entity adjudication. Without it every edge is skipped.
            sync_to_graph: Whether to push to the graph backend as well as the
                vector store. Off by default so a caller can batch the graph
                write across turns, which is markedly cheaper than one write
                per edge.
            sync_fn: The graph writer, injected for the same substitutability
                reason as the constructor's collaborators.

        Returns:
            Metadata for the edges that were persisted. Skipped edges are
            absent, so a short return is normal and not an error.
        """
        rels = getattr(result, "relationships", None)
        if not rels:
            return []
        
        missing = self._check_mappings(rels, input2resolved)
        if missing:
            _jlog("relationship_endpoint_mapping_missing", request_id, missing_names=sorted(missing))
            
        texts: list[str] = []
        metas: list[Meta] = []
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

            sid_value, tid_value = src_meta.get("id"), tgt_meta.get("id")
            if not isinstance(sid_value, str) or not isinstance(tid_value, str):
                skipped += 1
                _jlog(
                    "relationship_skipped",
                    request_id,
                    source_entity=r.source_entity,
                    target_entity=r.target_entity,
                    missing_side="invalid_entity_id",
                )
                continue
            sid, tid = sid_value, tid_value
            src_type, tgt_type = src_meta.get("type"), tgt_meta.get("type")

            key_st: RelKeyST = (sid, tid)
            key_std: RelKeySTD = (sid, tid, r.relationship_description or "")

            # Deduplicate on (sid, tid, description)
            if key_std in self._processed_full:
                continue

            if key_st in self._processed:
                existing = self._processed[key_st]

                merged_desc = str(existing.get("description") or "")
                if r.relationship_description and r.relationship_description not in merged_desc:
                    merged_desc = f"{merged_desc}; {r.relationship_description}" if merged_desc else r.relationship_description
                merged_kw = str(existing.get("keywords") or "")
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
            vecs: np.ndarray = self._embedder.embed(texts)  # shape = (n, dim), already normalized
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
