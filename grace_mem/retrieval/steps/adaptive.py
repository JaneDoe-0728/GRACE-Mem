"""
Adaptive re-search: confidence scoring, retrieval diagnosis, and LLM-driven query rewriting.

CONFIDENCE SCORING:
  - conf = mean(top-3 similarity scores across filtered entities + relationships)
  - Scores come from compare_by_id() (cosine inner-product, range [0, 1]).
  - If fewer than 3 items exist, use whatever is available.
  - If nothing is found, conf = 0.0.

REWRITE TRIGGER:
  - Triggered when conf < tau_confidence (configurable, default 0.70).
  - Capped at 2 passes total; pass-2 uses relaxed thresholds.
  - Uses diagnosis-driven LLM rewrite, not generic paraphrase.
"""
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from grace_mem.retrieval.prompts.adaptive import (
    ADAPTIVE_REWRITE_SYSTEM,
    ADAPTIVE_REWRITE_SYSTEM_MULTIHOP,
)

_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

logger = logging.getLogger("grace_mem.Adaptive")

# ──────────────────────────────────────────────────────────────────────────────
# Adaptive LLM client
# ──────────────────────────────────────────────────────────────────────────────

from grace_mem.retrieval.rendering import render_context_text
from grace_mem.runtime.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retriever", filename="kg_retriever.jsonl")


def build_adaptive_llm_client() -> Any:
    """Build an LLMClient for adaptive re-search query rewriting."""
    from grace_mem.adapters.llm import LLMClient
    logger.debug("Building adaptive LLM client")
    return LLMClient()


def build_adaptive_graph() -> Any:
    """
    Build a Graph (FalkorDB) connection for adaptive re-search using NEO4J_URI.

    Raises EnvironmentError if NEO4J_URI is not set.
    """
    uri   = os.getenv("NEO4J_URI", "").strip()
    user  = os.getenv("NEO4J_USERNAME", "")
    pwd   = os.getenv("NEO4J_PASSWORD", "")
    gname = os.getenv("GRAPH_NAME", "memory")

    if not uri:
        raise OSError(
            "NEO4J_URI is not set. Set it in .env before using adaptive re-search."
        )

    from grace_mem.adapters.graph.falkordb import Graph, GraphConfig
    logger.debug("Building adaptive graph: uri=%s graph=%s", uri, gname)
    return Graph(GraphConfig(uri=uri, user=user, password=pwd, graph_name=gname)).open()


# ──────────────────────────────────────────────────────────────────────────────
# Confidence scoring
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


# ──────────────────────────────────────────────────────────────────────────────
# Query rewrite
# ──────────────────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = ADAPTIVE_REWRITE_SYSTEM

_REWRITE_SYSTEM_MULTIHOP = ADAPTIVE_REWRITE_SYSTEM_MULTIHOP


_MULTIHOP_PATTERNS = {"multihop_chain_incomplete", "multihop_weak_chain"}


def rewrite_query(
    question: str,
    entities: list[dict],
    rels: list[dict],
    conf: float,
    local_llm: Any,
) -> tuple[str, float]:
    """
    Use the local LLM to rewrite the query based on retrieval diagnosis.

    For multi-hop patterns, a dedicated system prompt enforces anchor preservation:
    all entity names found in pass-1 are injected as must-keep anchors so the
    rewrite only extends toward the missing link, never drops known entities.

    Args:
        question:   Original query.
        entities:   Entities found in pass-1.
        rels:       Relationships found in pass-1.
        conf:       Pass-1 confidence score.
        local_llm:  LLMClient built by build_adaptive_llm_client().

    Returns:
        (rewritten_query, latency_sec)
    """
    pattern, diagnosis = diagnose_retrieval(entities, rels, conf)
    ent_names = [e.get("name", "?") for e in entities] or ["none"]
    rel_descs = [r.get("rel_desc", "?") for r in rels] or ["none"]

    is_multihop = pattern in _MULTIHOP_PATTERNS
    system_prompt = _REWRITE_SYSTEM_MULTIHOP if is_multihop else _REWRITE_SYSTEM

    if is_multihop:
        anchor_line = (
            f"Anchor entities you MUST keep verbatim: {', '.join(ent_names)}\n"
            "Only add keywords for the missing connection — do NOT remove any anchor.\n\n"
        )
    else:
        anchor_line = ""

    user_prompt = (
        f"Original query: {question}\n\n"
        f"Failure pattern: {pattern}\n"
        f"Diagnosis: {diagnosis}\n\n"
        f"{anchor_line}"
        f"Retrieved entities: {', '.join(ent_names)}\n"
        f"Retrieved relationships: {', '.join(rel_descs)}\n"
        f"Confidence: {conf:.3f}\n\n"
        "Rewrite the query to fix the failure pattern. Return ONLY the rewritten query."
    )

    t0 = time.perf_counter()
    response = local_llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_tokens=64,
    )
    latency = time.perf_counter() - t0

    raw = (response.choices[0].message.content or "").strip()
    # Strip surrounding quotes if the model wrapped the output
    rewritten = raw.strip('"').strip("'").strip()
    if not rewritten:
        rewritten = question  # safe fallback

    # Deterministic fallback: weak local models (e.g. gpt-oss-20b) frequently
    # disobey the "do not paraphrase" instruction and echo the query verbatim,
    # which makes the caller skip pass-2 entirely (rewritten == question). When
    # that happens, synthesise a genuinely different query by appending the
    # pass-1 entity anchors and relationship keywords as extra retrieval signal,
    # so pass-2 reaches a different candidate pool instead of being a no-op.
    if _normalise(rewritten) == _normalise(question):
        expansion = _build_expansion(question, ent_names, rel_descs)
        if expansion:
            rewritten = expansion
            logger.debug("Query rewrite fell back to deterministic expansion: %r", rewritten)

    logger.debug(
        "Query rewrite: pattern=%s multihop=%s conf=%.3f latency=%.2fs\n  original : %r\n  rewritten: %r",
        pattern, is_multihop, conf, latency, question, rewritten,
    )
    return rewritten, latency


def _normalise(s: str) -> str:
    """Lowercase + collapse whitespace + strip trailing punctuation for echo detection."""
    return re.sub(r"\s+", " ", s.lower()).strip().rstrip("?.!")


def _build_expansion(question: str, ent_names: list[str], rel_descs: list[str]) -> str:
    """Deterministically widen the query with pass-1 anchors + relation keywords.

    Adds only terms NOT already present in the question (case-insensitive), so the
    result is guaranteed lexically different from the original whenever any anchor
    or relation term is novel. Returns "" if nothing new can be added.
    """
    q_lower = question.lower()
    extras: list[str] = []
    seen: set[str] = set()
    for term in list(ent_names) + list(rel_descs):
        if not term or term in ("?", "none"):
            continue
        t = term.strip()
        key = t.lower()
        if key in seen or key in q_lower:
            continue
        seen.add(key)
        extras.append(t)
    if not extras:
        return ""
    # Cap to keep the query retrieval-oriented and avoid embedding dilution.
    extras = extras[:8]
    return f"{question.rstrip('?.! ')} {' '.join(extras)}"


def additive_merge(
    *,
    vdb_manager,
    cache,
    cfg,
    entities_1: list[dict],
    rels_1: list[dict],
    entities_2: list[dict],
    rels_2: list[dict],
    request_id: str | None,
    conf_1: float,
    conf_2: float,
    query_vec: Any = None,
) -> tuple[list[dict], list[dict], str, float]:
    """
    Additive context merge: preserve all pass-1 results and append only
    pass-2 items whose IDs were not already retrieved in pass-1.

    This ensures the answer-bearing context from pass-1 is never displaced.
    Novel entities are filtered by their similarity to the original query_vec
    (threshold: cfg.novel_ent_threshold) to discard noise introduced by
    rewrite drift.  Novel rels are admitted unconditionally since they are
    anchored to already-filtered entities.

    Returns:
        (merged_entities, merged_rels, merged_text, conf_merged)
    """
    # Collect pass-1 IDs to identify novel pass-2 items
    ent_ids_1: set = {e["id"] for e in entities_1}
    rel_ids_1: set = {r["rel_id"] for r in rels_1}

    novel_ents_raw = [e for e in entities_2 if e["id"] not in ent_ids_1]
    novel_rels = [r for r in rels_2 if r["rel_id"] not in rel_ids_1]

    # Filter novel entities by similarity to the original question embedding
    if query_vec is not None and novel_ents_raw:
        ent_vdb = vdb_manager.get_entities_vdb(0)
        novel_ents = [
            e for e in novel_ents_raw
            if (res := ent_vdb.compare_by_id(e["id"], query_vec, threshold=0.0))
            is not None and res[1] >= cfg.novel_ent_threshold
        ]
    else:
        novel_ents = novel_ents_raw

    merged_entities = entities_1 + novel_ents
    merged_rels = rels_1 + novel_rels

    # Confidence unchanged from pass-1 (pass-1 context is fully preserved)
    conf_merged = conf_1

    _jlog(
        "adaptive_merge_rerank",
        request_id,
        step="2b",
        entities_pass1=len(entities_1),
        entities_pass2=len(entities_2),
        rels_pass1=len(rels_1),
        rels_pass2=len(rels_2),
        novel_entities_raw=len(novel_ents_raw),
        novel_entities=len(novel_ents),
        novel_rels=len(novel_rels),
        merged_entities=len(merged_entities),
        merged_rels=len(merged_rels),
        conf_1=conf_1,
        conf_2=conf_2,
        conf_merged=conf_merged,
        novel_ent_threshold=cfg.novel_ent_threshold,
    )

    merged_text = render_context_text(
        cache=cache,
        entities=merged_entities,
        relationships=merged_rels,
        request_id=request_id,
    )
    return merged_entities, merged_rels, merged_text, conf_merged
