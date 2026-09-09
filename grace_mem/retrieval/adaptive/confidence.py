"""Was the first pass good enough, and if not, what went wrong.

Two pure functions and no state. `compute_confidence` scores a finished pass --
the mean of the top-3 similarities across the filtered entities and
relationships -- and `diagnose_retrieval` turns that pass into a labelled
failure pattern (no anchors, anchors but no edges, an incomplete multi-hop
chain, weak coverage). The controller decides what to do about either; neither
function reads config, opens a client, or retrieves anything.
"""

from typing import Any

# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence(
    entity_ids: list[str],
    rel_ids: list[str],
    query_vec: Any,
    vdb_manager: Any,
) -> float:
    """
    Compute retrieval confidence as the mean of the top-3 similarity scores
    across the union of filtered entity and relationship IDs.

    Args:
        entity_ids:   IDs of filtered entities (from assemble_context_from_query).
        rel_ids:      IDs of filtered relationships.
        query_vec:    Query embedding vector (already normalised).
        vdb_manager:  VDB manager with get_entities_vdb() / get_relationships_vdb().

    Returns:
        float in [0, 1]. 0.0 if nothing was retrieved.
    """
    scores: list[float] = []

    ent_vdb = vdb_manager.get_entities_vdb(0)
    for eid in entity_ids:
        res = ent_vdb.compare_by_id(eid, query_vec, threshold=0.0)
        if res is not None:
            _, score = res
            scores.append(score)

    rel_vdb = vdb_manager.get_relationships_vdb(0)
    for rid in rel_ids:
        res = rel_vdb.compare_by_id(rid, query_vec, threshold=0.0)
        if res is not None:
            _, score = res
            scores.append(score)

    if not scores:
        return 0.0

    top3 = sorted(scores, reverse=True)[:3]
    return sum(top3) / len(top3)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis
# ──────────────────────────────────────────────────────────────────────────────

def diagnose_retrieval(
    entities: list[dict],
    rels: list[dict],
    conf: float,
) -> tuple[str, str]:
    """
    Produce a (pattern_label, human_readable_diagnosis) for the retrieval result.

    Patterns:
      - no_entities_found            → broaden / decompose
      - entities_no_relations        → single anchor, target the missing edge / predicate
      - multihop_chain_incomplete    → 2+ anchors found but no connecting relationships
      - multihop_weak_chain          → 2+ anchors + some rels found but confidence is low
      - weak_coverage                → single anchor with rels but low confidence

    Returns:
        (pattern, diagnosis_text)
    """
    n_ent = len(entities)
    n_rel = len(rels)
    ent_names = [e.get("name", "?") for e in entities[:5]]
    ent_types = list({e.get("type", "?") for e in entities[:5]})

    if n_ent == 0:
        pattern = "no_entities_found"
        diagnosis = (
            "No entities were matched in the knowledge graph. "
            "The query may use phrasing absent from the KG. "
            "Suggestion: broaden or decompose the query into simpler sub-concepts."
        )
    elif n_ent >= 2 and n_rel == 0:
        pattern = "multihop_chain_incomplete"
        diagnosis = (
            f"Multiple anchor entities found: {', '.join(ent_names)} (types: {ent_types}). "
            "No connecting relationships were retrieved between them. "
            "This is a multi-hop query: the chain linking these entities is missing. "
            "Suggestion: keep ALL found entity names as anchors and add explicit "
            "relationship/action keywords that connect them (e.g. 'worked together', "
            "'met at', 'related to'). Do NOT drop any anchor entity name."
        )
    elif n_ent >= 2 and n_rel > 0:
        rel_descs = [r.get("rel_desc", "?") for r in rels[:3]]
        pattern = "multihop_weak_chain"
        diagnosis = (
            f"Multiple anchor entities found: {', '.join(ent_names)} (types: {ent_types}). "
            f"Partial relationships found: {', '.join(rel_descs)}. "
            f"Confidence is low ({conf:.3f}), indicating missing hops in the chain. "
            "Suggestion: keep ALL found entity names as anchors and strengthen the "
            "intermediate link by adding more specific relationship keywords or the "
            "name of the intermediate entity if inferable from context. "
            "Do NOT drop any anchor entity name from the rewrite."
        )
    elif n_ent == 1 and n_rel == 0:
        pattern = "entities_no_relations"
        diagnosis = (
            f"Entities found: {', '.join(ent_names)} (types: {ent_types}). "
            "No relationships were retrieved. "
            "Suggestion: rewrite to explicitly target the missing edge or predicate "
            "from this entity (e.g. add action verbs or relationship keywords)."
        )
    else:
        rel_descs = [r.get("rel_desc", "?") for r in rels[:3]]
        pattern = "weak_coverage"
        diagnosis = (
            f"Entities found: {', '.join(ent_names)} (types: {ent_types}). "
            f"Relationships found: {', '.join(rel_descs)}. "
            f"Confidence is low ({conf:.3f}). "
            "Suggestion: add type hints, synonyms, or alternate phrasing to sharpen relevance."
        )

    return pattern, diagnosis
