"""Building and rendering retrieval traces, separated from doing retrieval.

Every function here is pure: it takes what a retrieval stage produced and turns
it into a record or a block of text. None reads configuration, touches a store,
or decides anything the retrieval acts on. That is what makes them safe to lift
out of the Retriever, and what makes the stages easier to follow once the
reporting is no longer interleaved with them.

Two shapes of trace live here:

  * the stage waterfall -- what each narrowing step saw, kept and dropped,
    rendered as a snapshot record and as a human-readable block
  * the adaptive-pass comparison -- how pass 2's candidate set overlapped
    pass 1's, and what the confidence did

Two things deliberately stayed behind. Writing traces out needs the module
loggers and the instance field that remembers the last one. And the
subgraph-to-display-name helpers need `self.cache` to resolve an entity id into
a name -- three of the five do, so the family stays together in the Retriever
rather than being split across the boundary.
"""

from typing import Any


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate strings while preserving the original order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out

def build_stage_trace_snapshot(*,
    step: str,
    stage: str,
    entity_names: list[str],
    relationship_names: list[str],
    previous: dict[str, Any] | None = None,
    skipped: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build one readable stage snapshot with additions/removals from the previous stage."""
    current_entities = dedupe_preserve_order(entity_names)
    current_relationships = dedupe_preserve_order(relationship_names)
    prev_entities = (previous or {}).get("entity_names") or []
    prev_relationships = (previous or {}).get("relationship_names") or []
    prev_entity_set = set(prev_entities)
    prev_relationship_set = set(prev_relationships)
    current_entity_set = set(current_entities)
    current_relationship_set = set(current_relationships)

    return {
        "step": step,
        "stage": stage,
        "skipped": skipped,
        "reason": reason,
        "entity_count": len(current_entities),
        "relationship_count": len(current_relationships),
        "entity_names": current_entities,
        "relationship_names": current_relationships,
        "added_entity_names": [name for name in current_entities if name not in prev_entity_set],
        "removed_entity_names": [name for name in prev_entities if name not in current_entity_set],
        "added_relationship_names": [name for name in current_relationships if name not in prev_relationship_set],
        "removed_relationship_names": [name for name in prev_relationships if name not in current_relationship_set],
    }
def format_trace_names(names: list[str]) -> str:
    """Render a readable names list for the pretty waterfall trace."""
    if not names:
        return "-"
    return "; ".join(names)

def format_retrieval_stage_trace_text(*,
    request_id: str | None,
    question: str,
    low_level_keywords: list[str],
    high_level_keywords: list[str],
    local_branch: list[dict[str, Any]],
    global_branch: list[dict[str, Any]],
    merged_branch: list[dict[str, Any]],
    graph_override: bool,
    stop_reason: str | None,
    elapsed_sec: float | None,
) -> str:
    """Format one retrieval request as a readable waterfall trace block."""
    lines = [
        "=" * 80,
        f"request_id: {request_id or '-'}",
        f"question: {question}",
        f"low_level_keywords: {format_trace_names(low_level_keywords)}",
        f"high_level_keywords: {format_trace_names(high_level_keywords)}",
        f"graph_override: {graph_override}",
        f"stop_reason: {stop_reason or '-'}",
        f"elapsed_sec: {round(elapsed_sec, 4) if elapsed_sec is not None else '-'}",
        "",
    ]

    def append_branch(branch_name: str, stages: list[dict[str, Any]]) -> None:
        """Record one retrieval branch's state into the trace.

        What the differential analysis in `derive_drop_reasons` later diffs: each
        stage appends the entities and relationships still standing, and what
        disappeared between consecutive entries is what that stage dropped. A stage
        that does not append here is invisible to failure analysis, and its drops
        get attributed to the next stage that does.
        """
        lines.append(f"[{branch_name}]")
        if not stages:
            lines.append("  (no stages)")
            lines.append("")
            return

        for stage in stages:
            header = f"{stage['step']} {stage['stage']}"
            if stage.get("skipped"):
                header += f" (skipped: {stage.get('reason') or 'unknown'})"
            lines.append(header)
            lines.append(
                f"  entities[{stage['entity_count']}]: "
                f"{format_trace_names(stage.get('entity_names') or [])}"
            )
            lines.append(
                f"  relationships[{stage['relationship_count']}]: "
                f"{format_trace_names(stage.get('relationship_names') or [])}"
            )
            lines.append(
                f"  + entities: {format_trace_names(stage.get('added_entity_names') or [])}"
            )
            lines.append(
                f"  - entities: {format_trace_names(stage.get('removed_entity_names') or [])}"
            )
            lines.append(
                "  + relationships: "
                f"{format_trace_names(stage.get('added_relationship_names') or [])}"
            )
            lines.append(
                "  - relationships: "
                f"{format_trace_names(stage.get('removed_relationship_names') or [])}"
            )
            lines.append("")

    append_branch("local", local_branch)
    append_branch("global", global_branch)
    append_branch("merged", merged_branch)
    return "\n".join(lines).rstrip()
def compute_overlap_metrics(
    pass1_ids: list[str],
    pass2_ids: list[str],
) -> tuple[int, float | None]:
    """Return intersection size and Jaccard overlap for unique IDs."""
    pass1 = set(pass1_ids)
    pass2 = set(pass2_ids)
    union = pass1 | pass2
    if not union:
        return 0, None
    overlap_count = len(pass1 & pass2)
    return overlap_count, overlap_count / len(union)

def build_adaptive_trace(*,
    pass2_triggered: bool,
    pass1_entity_ids: list[str],
    pass1_relation_ids: list[str],
    pass2_entity_ids: list[str] | None = None,
    pass2_relation_ids: list[str] | None = None,
    conf_pass1: float | None = None,
    conf_pass2: float | None = None,
    conf_final: float | None = None,
    rewritten_query: str | None = None,
    adaptive_skip_reason: str | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Build a stable trace from pre-merge pass results."""
    entity_ids_2 = list(pass2_entity_ids or []) if pass2_triggered else []
    relation_ids_2 = list(pass2_relation_ids or []) if pass2_triggered else []
    if pass2_triggered:
        entity_overlap_count, entity_overlap_pct = compute_overlap_metrics(
            pass1_entity_ids,
            entity_ids_2,
        )
        relation_overlap_count, relation_overlap_pct = compute_overlap_metrics(
            pass1_relation_ids,
            relation_ids_2,
        )
    else:
        entity_overlap_count = relation_overlap_count = 0
        entity_overlap_pct = relation_overlap_pct = None

    trace = {
        "pass2_triggered": pass2_triggered,
        "conf_pass1": conf_pass1,
        "conf_pass2": conf_pass2,
        "conf_final": conf_final,
        "tau_confidence": getattr(config, "tau_confidence", None),
        "rewritten_query": rewritten_query,
        "adaptive_skip_reason": adaptive_skip_reason,
        "pass1_entity_ids": list(pass1_entity_ids),
        "pass2_entity_ids": entity_ids_2,
        "pass1_relation_ids": list(pass1_relation_ids),
        "pass2_relation_ids": relation_ids_2,
        "entity_overlap_count": entity_overlap_count,
        "entity_overlap_pct": entity_overlap_pct,
        "relation_overlap_count": relation_overlap_count,
        "relation_overlap_pct": relation_overlap_pct,
    }
    return trace
