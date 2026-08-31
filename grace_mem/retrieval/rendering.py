"""Turning retrieved entities and relationships into the text the LLM reads.

Pure formatting. It decides nothing about what was retrieved -- by the time
anything arrives here the candidate set is settled -- which is what lets it sit
outside the pipeline that produced it.

The cache is passed in rather than reached for: this needs the metadata behind
each id, not the retriever that happens to hold it.
"""

from typing import Any

from grace_mem.retrieval.steps.temporal_relevance import TemporalRelevanceCalculator
from grace_mem.storage import build_id_to_meta_maps
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")


def render_context_text(
    entities: list[dict],
    relationships: list[dict],
    cache: dict[str, Any],
    request_id: str | None = None,
) -> str:
    """Render entities and relationships into readable context text."""
    timer_render = _StepTimer()
    lines = []

    ent_id2meta, rel_id2meta = build_id_to_meta_maps(cache)

    temporal_types = {"Date", "Event", "Activity"}
    if entities:
        lines.append("=== Entities ===")
        for ent in entities:
            name = ent.get("name", "")
            ent_type = ent.get("type", "")
            desc = ent.get("desc", "")
            eid = ent.get("id", "")
            meta = ent_id2meta.get(eid, {}) or {}
            prov = meta.get("prov") or {}
            dt_str, _ = TemporalRelevanceCalculator.get_newest_dialogue_datetime(prov, request_id)
            temporal_tag = f" [mentioned_at:{dt_str}]" if dt_str and ent_type in temporal_types else ""
            temporal_meta = meta.get("temporal") or {}
            temporal_suffix = ""
            if temporal_meta:
                parts = []
                if temporal_meta.get("display_value"):
                    parts.append(f"display_value={temporal_meta['display_value']}")
                if temporal_meta.get("normalized_start"):
                    parts.append(f"normalized_start={temporal_meta['normalized_start']}")
                if temporal_meta.get("normalized_end"):
                    parts.append(f"normalized_end={temporal_meta['normalized_end']}")
                if temporal_meta.get("original_phrase"):
                    parts.append(f"original_phrase={temporal_meta['original_phrase']}")
                if temporal_meta.get("reference_time"):
                    parts.append(f"reference_time={temporal_meta['reference_time']}")
                if parts:
                    temporal_suffix = " [" + "; ".join(parts) + "]"
            lines.append(f"- {name} ({ent_type}): {desc}{temporal_suffix}{temporal_tag}")

    if relationships:
        lines.append("\n=== Relationships ===")
        for rel in relationships:
            src_name = rel.get("source_name", "")
            tgt_name = rel.get("target_name", "")
            rel_desc = rel.get("rel_desc", "")
            rid = rel.get("rel_id", "")
            rmeta = rel_id2meta.get(rid, {}) or {}
            rprov = rmeta.get("prov") or {}
            rdt_str, _ = TemporalRelevanceCalculator.get_newest_dialogue_datetime(rprov, request_id)
            rtype = rmeta.get("type", "")
            temporal_tag = f" [mentioned_at:{rdt_str}]" if rdt_str and rtype in temporal_types else ""
            lines.append(f"- {src_name} -> {tgt_name}: {rel_desc}{temporal_tag}")

    result = "\n".join(lines) if lines else ""
    _jlog(
        "context_text_rendered",
        request_id,
        step="2",
        entity_count=len(entities),
        relationship_count=len(relationships),
        line_count=len(lines),
        result_length=len(result),
        elapsed_sec=timer_render.sec(),
    )
    return result
