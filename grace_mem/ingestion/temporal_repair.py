"""Fixing up the temporal entities an extraction produced.

The extraction prompt is given pre-resolved temporal hints -- "last Tuesday"
already turned into a date -- and asked to use them. Models routinely ignore
that and emit the literal phrase anyway, which lands in the graph as an entity
nobody can match a date against.

This repairs that after the fact: it finds the entities that are temporal
literals, replaces them with the resolved form, and rewrites the relationships
that pointed at the old name so no edge is left dangling.

Pure functions over the extraction result. They read no config, touch no store,
and know nothing about the Ingestor that calls them -- which is why they can
live outside it.
"""

from __future__ import annotations

import re

from grace_mem.data_model.entities import Entity, EntityType
from grace_mem.data_model.relationships import Relationship
from grace_mem.temporal import extract_temporal_hints, rewrite_temporal_text
from grace_mem.temporal.types import TimeContext
from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Ingestor", filename="kg_ingestor.jsonl")


def _repair_temporal_entities(
    entities: list[Entity],
    relationships: list[Relationship],
    temporal_hints: list[dict],
    tctx: TimeContext | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Normalize temporal entity names to resolved values before the KG write.

    A time entity is only useful in the graph if its name is absolute. Two
    mentions of "last Tuesday" from different weeks must not collapse into one
    node, and a query for a date range cannot match a relative phrase. So every
    Date/Time/Timespan name has to become a resolved value before it is written
    -- after that point the turn's reference time is gone and the phrase can
    never be resolved again.

    The repairs, all driven by `temporal_hints` extracted upstream:

    - Date/Timespan named by a relative phrase -> resolved ISO value.
    - Date/Timespan named with bracket marker syntax ("[TIMESPAN: 2022]") ->
      stripped. The extraction prompt uses those markers; they are scaffolding,
      not part of the name.
    - Date/Timespan whose name is an unambiguous suffix of an expected marker
      -> renamed to the full marker.
    - Event names carrying a relative phrase or embedded marker -> cleaned.
    - Descriptions rewritten the same way, so retrieval text agrees with the
      node name.
    - Relationship endpoints follow any rename, or the edge would point at a
      node that no longer exists under that name.
    - Expected marker entities the extractor missed -> injected, so a hint that
      was resolved does not vanish for want of the model mentioning it.

    Returns:
        The repaired (entities, relationships). Inputs are not mutated.
    """
    _temporal_types = (EntityType.Date, EntityType.Time, EntityType.Timespan)
    # A whole name that is a bracket marker: "[TIMESPAN: the weekend before 2023-07-15]"
    _bracket_re = re.compile(r'^\[(?:DATE|TIMESPAN|TIME):\s*(.*?)\]$', re.IGNORECASE)
    # The same markers embedded mid-string, which is how they show up in Event names.
    _embedded_bracket_re = re.compile(r'\s*\[(?:DATE|TIMESPAN|TIME):[^\]]*\]', re.IGNORECASE)

    hint_lookup: dict[str, dict] = {h["original"].lower(): h for h in temporal_hints}
    # Reverse index, resolved value -> hint. Needed because the model sometimes
    # does the resolution itself and emits "2023-05-29 to 2023-06-04" directly.
    # Without this the entity would look unmatched and lose its provenance back
    # to the original phrase ("last week"), which the temporal metadata records.
    resolved_lookup: dict[str, dict] = {}
    for h in temporal_hints:
        for key in (h.get("display_value"), h.get("resolved_to")):
            if key and key.lower() not in resolved_lookup:
                resolved_lookup[key.lower()] = h
    rename_map: dict[str, str] = {}

    # (entity_type_str, entity_name) → hint, for every expected marker entity
    marker_hint_for: dict[tuple[str, str], dict] = {}
    for h in temporal_hints:
        for m in (h.get("markers") or []):
            key = (m["entity_type"], m["entity_name"])
            if key not in marker_hint_for:
                marker_hint_for[key] = h

    def _temporal_meta_for(name: str) -> dict | None:
        """Build the temporal metadata block for one entity name, or None.

        Two sources, in priority order. A precomputed hint is preferred because
        it was resolved against the turn's own reference time. Failing that,
        the name is re-parsed against `tctx` -- which catches phrases the hint
        extractor missed but the entity extractor surfaced.

        Returns None when neither path yields a resolution, meaning the name is
        not temporal after all and should be left alone.
        """
        key = name.lower().strip()
        hint = hint_lookup.get(key) or resolved_lookup.get(key)
        if hint:
            return {
                "temporal": {
                    "display_value": hint.get("display_value") or hint.get("resolved_to"),
                    "normalized_time": hint.get("normalized_time"),
                    "normalized_start": hint.get("normalized_start"),
                    "normalized_end": hint.get("normalized_end"),
                    "granularity": hint.get("granularity"),
                    "original_phrase": hint.get("original"),
                    "reference_time": hint.get("reference_time"),
                    "status": hint.get("status"),
                    "confidence": hint.get("confidence"),
                }
            }
        if tctx is None:
            return None
        rewritten, meta = rewrite_temporal_text(name, tctx)
        constraints = meta.get("constraints") or []
        if not constraints:
            return None
        resolution = (constraints[0] or {}).get("resolution") or {}
        display = resolution.get("display_value") or rewritten
        return {
            "temporal": {
                "display_value": display,
                "normalized_time": resolution.get("normalized_time"),
                "normalized_start": resolution.get("normalized_start"),
                "normalized_end": resolution.get("normalized_end"),
                "granularity": resolution.get("granularity"),
                "original_phrase": constraints[0].get("original_text"),
                "reference_time": resolution.get("reference_time"),
                "status": resolution.get("status"),
                "confidence": resolution.get("confidence"),
            }
        }

    def _temporal_anchor_description(name: str, etype: EntityType, meta: dict | None) -> str:
        """Compose a description for a temporal node the extractor left bare.

        Injected marker entities arrive with no description, and a node with an
        empty description is effectively invisible to retrieval: the entity
        vector store embeds the description, so an empty one embeds to noise.
        Stating the granularity in words also gives dense search something to
        match a query like "that week" against.
        """
        temporal = (meta or {}).get("temporal") or {}
        granularity = temporal.get("granularity")
        if etype == EntityType.Date:
            return f"The calendar date {name}."
        if etype == EntityType.Time:
            return f"The clock time {name}."
        if granularity == "week":
            return f"The week-long timespan {name}."
        if granularity == "weekend":
            return f"The weekend timespan {name}."
        if granularity == "month":
            return f"The month-long timespan {name}."
        if granularity == "season":
            return f"The seasonal timespan {name}."
        if granularity == "year":
            return f"The year-long timespan {name}."
        return f"The timespan {name}."

    def _prefer_existing_temporal_description(
        description: str | None,
        *,
        name: str,
        etype: EntityType,
        meta: dict | None,
    ) -> str:
        """Keep the extractor's description, falling back to a generated anchor.

        Only fills a genuine gap. The extracted description says what the date
        meant in the conversation; the generated one only says what kind of
        date it is, so overwriting would trade specific text for generic.
        """
        if description and description.strip():
            return description.strip()
        return _temporal_anchor_description(name, etype, meta)

    repaired: list[Entity] = []
    covered_markers: set[tuple[str, str]] = set()

    for ent in entities:
        name = ent.entity_name
        etype = ent.entity_type
        desc = ent.entity_description
        meta = getattr(ent, "entity_metadata", None)
        if tctx is not None and desc and etype == EntityType.Event:
            desc, _ = rewrite_temporal_text(desc, tctx)
        name_lower = name.lower().strip()

        if etype in _temporal_types:
            original_name = name

            # Fix 1: strip bracket marker syntax like [TIMESPAN: 2022] → 2022
            _bracket = _bracket_re.match(name.strip())
            if _bracket:
                name = _bracket.group(1).strip()
                name_lower = name.lower()

            hint = hint_lookup.get(name_lower)
            resolved = hint.get("resolved_to") if hint else None
            # Fallback: re-resolve relative phrases (e.g. "Yesterday") via the parser.
            # Skip names that already contain an ISO date (YYYY-MM-DD) — re-parsing
            # those causes two bugs: date ranges get truncated to the first date, and
            # "night of YYYY-MM-DD" shifts one day due to daypart boundary semantics.
            _has_iso_date = bool(re.search(r'\d{4}-\d{2}-\d{2}', name))
            if resolved is None and tctx is not None and not _has_iso_date:
                sub_hints = extract_temporal_hints([name], tctx)
                if sub_hints:
                    hint = sub_hints[0]
                    resolved = hint.get("resolved_to")
            final_name = resolved or name

            # Fix 2: partial extract — if still unresolved, rename when name is an
            # unambiguous suffix of exactly one expected marker entity of the same type.
            # e.g. "before 2023-07-15" → "the weekend before 2023-07-15"
            if final_name == name:
                _candidates = [
                    mname
                    for (mtype, mname) in marker_hint_for
                    if mtype == etype.value and mname.endswith(name) and mname != name
                ]
                if len(_candidates) == 1:
                    final_name = _candidates[0]

            final_meta = meta or _temporal_meta_for(name) or _temporal_meta_for(final_name)
            final_desc = _prefer_existing_temporal_description(
                desc,
                name=final_name,
                etype=etype,
                meta=final_meta,
            )
            repaired.append(Entity(entity_name=final_name, entity_type=etype, entity_description=final_desc, entity_metadata=final_meta))
            if final_name != original_name:
                rename_map[original_name] = final_name
                _jlog("temporal_entity_name_repaired", "repair",
                      original_name=original_name, repaired_name=final_name, entity_type=etype.value)
            covered_markers.add((etype.value, final_name))
            continue

        elif etype == EntityType.Event:
            new_name = name
            # strip embedded bracket markers (e.g. "pottery workshop[TIMESPAN: 2022]")
            new_name = _embedded_bracket_re.sub("", new_name).strip(" ,.-")
            for original_phrase, hint in hint_lookup.items():
                pattern = re.compile(re.escape(original_phrase), re.IGNORECASE)
                if pattern.search(new_name):
                    stripped = pattern.sub("", new_name).strip(" ,.-")
                    if stripped:
                        new_name = stripped
            if new_name != name:
                repaired.append(Entity(entity_name=new_name, entity_type=etype, entity_description=desc, entity_metadata=meta))
                rename_map[name] = new_name
                _jlog("temporal_entity_name_repaired", "repair",
                      original_name=name, repaired_name=new_name, entity_type=etype.value)
                continue

        repaired.append(Entity(entity_name=name, entity_type=etype, entity_description=desc, entity_metadata=meta))

    # Fix 3: inject expected marker entities that the LLM omitted entirely
    for (mtype, mname), hint in marker_hint_for.items():
        if (mtype, mname) in covered_markers:
            continue
        try:
            etype = EntityType(mtype)
        except ValueError:
            continue
        inj_meta: dict = {
            "temporal": {
                "display_value": hint.get("display_value") or hint.get("resolved_to"),
                "normalized_time": hint.get("normalized_time"),
                "normalized_start": hint.get("normalized_start"),
                "normalized_end": hint.get("normalized_end"),
                "granularity": hint.get("granularity"),
                "original_phrase": hint.get("original"),
                "reference_time": hint.get("reference_time"),
                "status": hint.get("status"),
                "confidence": hint.get("confidence"),
            }
        }
        inj_desc = _temporal_anchor_description(mname, etype, inj_meta)
        repaired.append(Entity(entity_name=mname, entity_type=etype, entity_description=inj_desc, entity_metadata=inj_meta))
        _jlog("temporal_entity_injected", "repair",
              entity_name=mname, entity_type=mtype, original_phrase=hint.get("original"))

    # Fix 4: fallback hard-inject for hints whose markers list was empty (or whose
    # resolved name wasn't captured by Fix 3).  Uses granularity to pick entity type.
    _granularity_to_etype: dict[str, EntityType] = {
        "day": EntityType.Date,
        "time": EntityType.Time,
    }
    _covered_names_lower: set[str] = {e.entity_name.lower() for e in repaired}
    for hint in temporal_hints:
        candidate = hint.get("display_value") or hint.get("resolved_to")
        if not candidate:
            continue
        if candidate.lower() in _covered_names_lower:
            continue
        granularity = (hint.get("granularity") or "").lower()
        etype = _granularity_to_etype.get(granularity, EntityType.Timespan)
        # skip if normalized_time present but granularity says day — time wins
        if hint.get("normalized_time") and etype == EntityType.Date:
            etype = EntityType.Time
        fallback_meta: dict = {
            "temporal": {
                "display_value": hint.get("display_value") or hint.get("resolved_to"),
                "normalized_time": hint.get("normalized_time"),
                "normalized_start": hint.get("normalized_start"),
                "normalized_end": hint.get("normalized_end"),
                "granularity": hint.get("granularity"),
                "original_phrase": hint.get("original"),
                "reference_time": hint.get("reference_time"),
                "status": hint.get("status"),
                "confidence": hint.get("confidence"),
            }
        }
        inj_desc = _temporal_anchor_description(candidate, etype, fallback_meta)
        repaired.append(
            Entity(
                entity_name=candidate,
                entity_type=etype,
                entity_description=inj_desc,
                entity_metadata=fallback_meta,
            )
        )
        _covered_names_lower.add(candidate.lower())
        _jlog("temporal_entity_hint_fallback_injected", "repair",
              entity_name=candidate, entity_type=etype.value, original_phrase=hint.get("original"))

    repaired_relationships: list[Relationship] = []
    for rel in relationships:
        repaired_relationships.append(
            Relationship(
                source_entity=rename_map.get(rel.source_entity, rel.source_entity),
                target_entity=rename_map.get(rel.target_entity, rel.target_entity),
                relationship_description=rel.relationship_description,
                relationship_keywords=rel.relationship_keywords,
            )
        )

    return repaired, repaired_relationships


def _expected_temporal_marker_entities(temporal_hints: list[dict]) -> list[dict[str, str]]:
    """List the temporal entities the hints say the extractor should have found.

    Every resolved hint declares the marker entities it implies; comparing that
    list against what extraction actually returned is how `_repair_temporal_entities`
    knows what to inject. Deduplicated on (type, name) because several phrases
    in one turn commonly resolve to the same date -- "yesterday" and "the 14th"
    should yield one node, not two.

    Markers missing either field are skipped rather than defaulted: a nameless
    entity cannot be matched against extraction output or written to the graph.
    """
    expected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for hint in temporal_hints or []:
        for marker in hint.get("markers") or []:
            entity_type = (marker.get("entity_type") or "").strip()
            entity_name = (marker.get("entity_name") or "").strip()
            if not entity_type or not entity_name:
                continue
            key = (entity_type, entity_name)
            if key in seen:
                continue
            seen.add(key)
            expected.append(
                {
                    "entity_type": entity_type,
                    "entity_name": entity_name,
                    "original": hint.get("original", ""),
                    "marker_type": marker.get("marker_type", ""),
                }
            )
    return expected
