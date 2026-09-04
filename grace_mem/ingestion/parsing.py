"""Parsing the LLM's extraction reply into data models.

The extraction prompt asks for delimited text rather than JSON, so this is where
a model's reply becomes an `ExtractionResult`. Everything here is tolerant by
design: a malformed record is dropped, not raised, because one bad line in a
turn must not cost the whole turn.

The data models these functions produce live in `grace_mem.data_model`. The BM25
tokenizer that used to live here moved to `grace_mem.lexical`, which is below both
capabilities rather than inside one of them.
"""

import logging
import re
from typing import Any

from grace_mem.data_model.entities import Entity, EntityType
from grace_mem.data_model.extraction import ExtractionResult
from grace_mem.data_model.relationships import Relationship

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH_ERROR_PATTERNS = (
    "context_length_exceeded",
    "context length exceeded",
    "maximum context length",
    "maximum context length is",
    "maximum context size",
    "reduce the length of the messages",
    "too many tokens",
    "prompt is too long",
)

def is_context_length_exceeded_error(error: Any) -> bool:
    """Best-effort detection for LLM context-window overflow errors."""
    if error is None:
        return False

    text = str(error).lower()
    if any(pattern in text for pattern in _CONTEXT_LENGTH_ERROR_PATTERNS):
        return True

    code = getattr(error, "code", None)
    if isinstance(code, str) and code.lower() == "context_length_exceeded":
        return True

    body = getattr(error, "body", None)
    if body is not None:
        body_text = str(body).lower()
        if any(pattern in body_text for pattern in _CONTEXT_LENGTH_ERROR_PATTERNS):
            return True

    return False

def parse_delimited_extraction(raw: str, tuple_delim: str, record_delim: str, completion_token: str) -> ExtractionResult:
    """Parse the extractor's delimited text output into validated models.

    Never raises. Extraction runs once per turn across an entire corpus, so a
    parser that aborted on the first malformed record would discard a whole
    run's work over one bad line. Anything unparseable is dropped, counted, and
    logged; the caller gets whatever was valid.

    That tolerance is the reason to watch the logs: a systematically broken
    prompt shows up as a high skip count, not as a failure. An empty result is
    indistinguishable here from a turn that genuinely contained no facts.

    Records are dropped when the entity type is outside `EntityType`, a
    required field is missing, or -- for relationships -- an endpoint names an
    entity that was not extracted, which would otherwise create a dangling edge.

    Args:
        raw: The model's reply. Text after `completion_token` is discarded,
            since models routinely continue past it with commentary.
        tuple_delim: Field separator within one record.
        record_delim: Separator between records.
        completion_token: End-of-output marker.

    Returns:
        The valid entities and relationships. Empty on empty or wholly
        unparseable input.
    """
    if not raw:
        logger.warning("parse_delimited_extraction: Empty raw input")
        return ExtractionResult(entities=[], relationships=[])

    raw = raw.split(completion_token)[0]

    parts = [p.strip() for p in raw.split(record_delim)]
    ent_list, rel_list = [], []

    # Accumulated rather than raised, then logged as a summary at the end: one
    # log line per malformed record would swamp the run.
    parsing_errors: dict[str, list[Any]] = {
        "entity_errors": [],
        "relationship_errors": [],
        "skipped_lines": [],
    }

    def _clean(s: str) -> str:
        """Trim a raw extracted field and remove one pair of wrapping quotes."""
        s = (s or "").strip()
        # Drop one matched pair of surrounding quotes
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        return s

    for line in parts:
        if not line:
            continue

        line_stripped = line.strip()

        # Support the legacy format: ("entity"<|>...) / ("relationship"<|>...)
        if line_stripped.startswith("(") and line_stripped.endswith(")"):
            line_stripped = line_stripped[1:-1].strip()

        # Split the line into fields on tuple_delim
        cols = [c.strip() for c in line_stripped.split(tuple_delim)]
        if not cols:
            continue

        # The first column should be entity / relationship, possibly quoted
        kind = _clean(cols[0]).lower()

        # ---------------------
        # Parse an entity record
        # ---------------------
        if kind == "entity":
            # Needs at least 4 columns: entity, name, type, desc
            if len(cols) < 4:
                parsing_errors["entity_errors"].append({
                    "name": "MISSING",
                    "type": "MISSING",
                    "error": f"Too few columns for entity line: got {len(cols)}"
                })
                continue

            name = _clean(cols[1])
            etype = _clean(cols[2])

            # desc may itself contain tuple_delim; conservatively treat every
            # column from index 3 onward as part of the description
            if len(cols) > 4:
                desc_raw = tuple_delim.join(cols[3:])
            else:
                desc_raw = cols[3]
            desc = _clean(desc_raw)

            if name and etype and desc:
                try:
                    ent_list.append(
                        Entity(
                            entity_name=name,
                            entity_type=EntityType(etype),
                            entity_description=desc
                        )
                    )
                except ValueError as e:
                    # Invalid EntityType value
                    parsing_errors["entity_errors"].append({
                        "name": name,
                        "type": etype,
                        "error": f"Invalid entity type: {e}"
                    })
                    logger.warning(
                        f"Invalid entity type '{etype}' for entity '{name}'. "
                        f"Valid types: {[e.value for e in EntityType]}"
                    )
                except Exception as e:
                    parsing_errors["entity_errors"].append({
                        "name": name,
                        "type": etype,
                        "error": str(e)
                    })
                    logger.warning(f"Failed to parse entity '{name}': {e}")
            else:
                parsing_errors["entity_errors"].append({
                    "name": name or "MISSING",
                    "type": etype or "MISSING",
                    "error": "Missing required fields"
                })
            continue

        # -------------------------
        # Parse a relationship record
        # -------------------------
        if kind == "relationship":
            # Needs at least 5 columns: relationship, src, tgt, desc, keywords
            if len(cols) < 5:
                parsing_errors["relationship_errors"].append({
                    "src": "MISSING",
                    "tgt": "MISSING",
                    "error": f"Too few columns for relationship line: got {len(cols)}"
                })
                continue

            src = _clean(cols[1])
            tgt = _clean(cols[2])

            # The description may contain tuple_delim, so pin the ends instead:
            # last column is keywords, columns 3..-2 join back into description
            if len(cols) > 5:
                rdesc_raw = tuple_delim.join(cols[3:-1])
            else:
                rdesc_raw = cols[3]
            rdesc = _clean(rdesc_raw)

            rkeys = _clean(cols[-1])

            if src and tgt and rdesc and rkeys:
                rel_list.append(
                    Relationship(
                        source_entity=src,
                        target_entity=tgt,
                        relationship_description=rdesc,
                        relationship_keywords=rkeys
                    )
                )
            else:
                parsing_errors["relationship_errors"].append({
                    "src": src or "MISSING",
                    "tgt": tgt or "MISSING",
                    "error": "Missing required fields"
                })
            continue

        # Neither entity nor relationship: record it as a skipped line
        if line.strip():
            parsing_errors["skipped_lines"].append(line[:100])

    # Filter relationships to only include those with valid entities
    names = {e.entity_name for e in ent_list}
    orphaned_rels = [
        r for r in rel_list
        if r.source_entity not in names or r.target_entity not in names
    ]
    rel_list = [
        r for r in rel_list
        if r.source_entity in names and r.target_entity in names
    ]

    if orphaned_rels:
        logger.warning(f"Filtered {len(orphaned_rels)} orphaned relationships (entities not found)")
        for r in orphaned_rels:
            parsing_errors["relationship_errors"].append({
                "src": r.source_entity,
                "tgt": r.target_entity,
                "error": "Referenced entity not found in parsed entities"
            })

    # Log summary if there were issues
    if (
        parsing_errors["entity_errors"]
        or parsing_errors["relationship_errors"]
        or parsing_errors["skipped_lines"]
    ):
        logger.warning(
            f"Parse summary: {len(ent_list)} entities, {len(rel_list)} relationships. "
            f"Errors: {len(parsing_errors['entity_errors'])} entities, "
            f"{len(parsing_errors['relationship_errors'])} relationships, "
            f"{len(parsing_errors['skipped_lines'])} skipped lines"
        )
    return ExtractionResult(entities=ent_list, relationships=rel_list)


# Strip generation artefacts from the tail only, so real content is never damaged
_TAIL_GARBAGE_PATTERNS = [
    r"<\|.*?\|>$",   # <|COMPLETE|> / <|...|>
    r"<\|.*?$",      # broken token like <|COMP...
    r">+$",          # stray >
    r"\)+$",         # extra )))
    r"\s+$",         # trailing whitespace
]

def _strip_wrapping_quotes(s: str) -> str:
    """Remove a single pair of matching outer quotes from a field."""
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def _strip_outer_parens_tolerant(line: str) -> str:
    """Strip a leading or trailing parenthesis even if the pair is unbalanced."""
    # tolerant: strip '(' even if missing ')', and vice versa
    line = (line or "").strip()
    if line.startswith("("):
        line = line[1:].strip()
    if line.endswith(")"):
        line = line[:-1].strip()
    return line

def _truncate_on_forbidden_tokens(s: str, record_delim: str, completion_token: str) -> str:
    """
    Spec says these tokens must NOT appear inside any field.
    If LLM violates, we deterministically cut at the first occurrence.
    """
    if not s:
        return s
    cut_pos = None
    for tok in [record_delim, completion_token]:
        if tok:
            idx = s.find(tok)
            if idx != -1:
                cut_pos = idx if cut_pos is None else min(cut_pos, idx)
    return s[:cut_pos].strip() if cut_pos is not None else s

def clean_field(s: str, record_delim: str, completion_token: str) -> str:
    """Normalize one extracted field and strip trailing generation artifacts."""
    s = (s or "").strip()
    s = _strip_wrapping_quotes(s)
    s = _truncate_on_forbidden_tokens(s, record_delim, completion_token)

    # strip tail garbage only
    for pat in _TAIL_GARBAGE_PATTERNS:
        s = re.sub(pat, "", s).strip()

    return s

def coerce_entity_type(etype_raw: str) -> EntityType:
    """
    Make EntityType parsing case-insensitive and robust to minor formatting differences.
    """
    etype = (etype_raw or "").strip()
    if not etype:
        raise ValueError("empty entity_type")

    # First try the plain enum constructor
    try:
        return EntityType(etype)
    except Exception:
        pass

    # Then try a case-insensitive match on the enum values
    etype_l = etype.lower()
    for m in EntityType:
        try:
            if str(m.value).lower() == etype_l:
                return m
        except Exception:
            continue

    # The enum may also be addressed by member name (e.g. EntityType.PERSON)
    for m in EntityType:
        if str(m.name).lower() == etype_l:
            return m

    raise ValueError(f"unknown entity_type: {etype_raw}")


def canonicalize_entity_type_label(etype_raw: str) -> str:
    """Canonicalize an entity-type label for storage and downstream matching.

    - Known enum values collapse to the exact EntityType value, e.g. ``date`` -> ``Date``.
    - Unknown labels keep their wording but are normalized to sentence-style casing,
      e.g. ``organization`` -> ``Organization``.
    """
    etype = (etype_raw or "").strip()
    if not etype:
        return ""

    try:
        return coerce_entity_type(etype).value
    except ValueError:
        pass

    parts = re.split(r"([_\-\s]+)", etype)
    normalized_parts = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[_\-\s]+", part):
            normalized_parts.append(part)
        else:
            normalized_parts.append(part[:1].upper() + part[1:].lower())
    return "".join(normalized_parts).strip()

_FALLBACK_DELIMITERS = [",", "|", "\t"]

def _try_lenient_entity_parse(
    line: str, tuple_delim: str, record_delim: str, completion_token: str
) -> Entity | None:
    """
    Best-effort re-parse of a single malformed entity line.
    Returns an Entity on success, None if the line is unrecoverable.

    Strategy (applied in order):
    1. Too few columns with primary delimiter → retry with fallback delimiters.
    2. Missing "entity" kind prefix → treat 3-col lines as (name, type, desc).
    3. tuple_delim embedded in name/type → scan cols for a known EntityType value.
    4. Empty desc → use name as fallback description.
    5. Unknown entity_type → fall back to EntityType.Concept.
    """
    def _extract(cols: list[str], name_idx: int, type_idx: int, desc_start: int, delim: str) -> Entity | None:
        name = clean_field(cols[name_idx], record_delim, completion_token)
        etype_raw = clean_field(cols[type_idx], record_delim, completion_token)
        desc_raw = delim.join(cols[desc_start:]) if len(cols) > desc_start else ""
        desc = clean_field(desc_raw, record_delim, completion_token) or name
        if not name:
            return None
        try:
            etype = coerce_entity_type(etype_raw) if etype_raw else EntityType.Concept
        except ValueError:
            etype = EntityType.Concept
        return Entity(entity_name=name, entity_type=etype, entity_description=desc)

    # Try primary delimiter first, then fallbacks
    delimiters_to_try = [tuple_delim] + [d for d in _FALLBACK_DELIMITERS if d != tuple_delim]
    for delim in delimiters_to_try:
        if not delim:
            continue
        cols = [c.strip() for c in line.split(delim)]
        kind = clean_field(cols[0], record_delim, completion_token).lower()

        if kind == "entity" and len(cols) >= 4:
            # Check for embedded delimiter in name/type fields
            name_raw = clean_field(cols[1], record_delim, completion_token)
            type_raw = clean_field(cols[2], record_delim, completion_token)
            if delim and (delim in name_raw or delim in type_raw):
                # Scan all cols for a known EntityType to anchor the split
                known_type_idx = None
                for i, col in enumerate(cols):
                    try:
                        coerce_entity_type(col.strip())
                        known_type_idx = i
                        break
                    except (ValueError, Exception):
                        continue
                if known_type_idx and known_type_idx > 0:
                    name = clean_field(cols[known_type_idx - 1], record_delim, completion_token)
                    etype_raw = cols[known_type_idx].strip()
                    desc_raw = delim.join(cols[known_type_idx + 1:])
                    desc = clean_field(desc_raw, record_delim, completion_token) or name
                    if name:
                        try:
                            return Entity(entity_name=name, entity_type=coerce_entity_type(etype_raw), entity_description=desc)
                        except Exception:
                            pass
                continue
            ent = _extract(cols, 1, 2, 3, delim)
            if ent:
                return ent

        elif kind != "entity" and len(cols) >= 3:
            # No "entity" kind prefix — treat cols as (name, type, desc)
            ent = _extract(cols, 0, 1, 2, delim)
            if ent:
                return ent

    return None


def parse_entities_only(raw: str, tuple_delim: str, record_delim: str, completion_token: str) -> list[Entity]:
    """Parse only entity rows from the model output and drop malformed entries."""
    if not raw:
        logger.warning("parse_entities_only: Empty raw input")
        return []

    # Guard 1: cut at the completion token first (it sits outside the records)
    raw = raw.split(completion_token)[0] if completion_token else raw

    # Guard 2: split on the record delimiter, tolerating blank runs
    parts = [p.strip() for p in (raw.split(record_delim) if record_delim else raw.splitlines()) if p and p.strip()]

    ent_list: list[Entity] = []
    parsing_errors: list[str] = []

    for part in parts:
        line = _strip_outer_parens_tolerant(part)
        if not line:
            continue

        cols = [c.strip() for c in line.split(tuple_delim)] if tuple_delim else [line]
        parse_ok = False

        if len(cols) >= 4:
            kind = clean_field(cols[0], record_delim, completion_token).lower()
            if kind == "entity":
                name = clean_field(cols[1], record_delim, completion_token)
                etype_raw = clean_field(cols[2], record_delim, completion_token)
                desc_raw = tuple_delim.join(cols[3:]) if len(cols) > 3 else ""
                desc = clean_field(desc_raw, record_delim, completion_token)

                if tuple_delim and (tuple_delim in name or tuple_delim in etype_raw):
                    parsing_errors.append(f"Forbidden tuple_delim inside name/type: name='{name}', type='{etype_raw}'")
                elif not (name and etype_raw and desc):
                    parsing_errors.append(f"Missing fields: name={name}, type={etype_raw}, desc={desc[:40]}")
                else:
                    try:
                        ent_list.append(Entity(
                            entity_name=name,
                            entity_type=coerce_entity_type(etype_raw),
                            entity_description=desc
                        ))
                        parse_ok = True
                    except Exception as e:
                        parsing_errors.append(f"Failed to parse entity '{name}' type='{etype_raw}': {e}")
            else:
                parsing_errors.append(f"Skipped non-entity kind='{kind}': {line[:120]}")
        else:
            parsing_errors.append(f"Too few columns ({len(cols)}): {line[:120]}")

        if not parse_ok:
            recovered = _try_lenient_entity_parse(line, tuple_delim, record_delim, completion_token)
            if recovered:
                ent_list.append(recovered)
                parsing_errors.pop()  # remove the error we just logged since recovery succeeded
                logger.debug(f"Entity lenient-recovered: '{recovered.entity_name}' type={recovered.entity_type.value}")
            # else: stay discarded, error already appended

    if parsing_errors:
        logger.warning(f"Entity parsing: {len(ent_list)} parsed, {len(parsing_errors)} issues")

    return ent_list


def parse_relationships_only(
    raw: str,
    tuple_delim: str,
    record_delim: str,
    completion_token: str,
    valid_entity_names: set[str]
) -> list[Relationship]:
    """Parse only relationship rows and keep only ones whose endpoints are valid."""
    if not raw:
        logger.warning("parse_relationships_only: Empty raw input")
        return []

    raw = raw.split(completion_token)[0] if completion_token else raw
    parts = [p.strip() for p in (raw.split(record_delim) if record_delim else raw.splitlines()) if p and p.strip()]

    rel_list: list[Relationship] = []
    parsing_errors: list[str] = []
    orphaned_rels: list[tuple] = []

    for part in parts:
        line = _strip_outer_parens_tolerant(part)
        if not line:
            continue

        cols = [c.strip() for c in line.split(tuple_delim)] if tuple_delim else [line]
        if len(cols) < 5:
            parsing_errors.append(f"Too few columns ({len(cols)}): {line[:120]}")
            continue

        kind = clean_field(cols[0], record_delim, completion_token).lower()
        if kind != "relationship":
            continue

        # Fixed 5-column semantics: src, tgt, desc, keywords
        src = clean_field(cols[1], record_delim, completion_token)
        tgt = clean_field(cols[2], record_delim, completion_token)

        # desc may contain tuple_delim → join middle columns (3 .. -2)
        rdesc_raw = tuple_delim.join(cols[3:-1])
        rdesc = clean_field(rdesc_raw, record_delim, completion_token)

        rkeys = clean_field(cols[-1], record_delim, completion_token)
        rkeys = rkeys.lower().strip()

        # spec guard: tuple_delim must not be inside src/tgt
        if tuple_delim and (tuple_delim in src or tuple_delim in tgt):
            parsing_errors.append(f"Forbidden tuple_delim inside src/tgt: src='{src}', tgt='{tgt}'")
            continue

        if not (src and tgt and rdesc and rkeys):
            parsing_errors.append(f"Missing fields: src={src}, tgt={tgt}, desc={rdesc[:40]}, keys={rkeys}")
            continue

        # validate entity references
        if src not in valid_entity_names or tgt not in valid_entity_names:
            orphaned_rels.append((src, tgt))
            parsing_errors.append(f"Orphaned relationship: {src} -> {tgt} (entity not found)")
            continue

        try:
            rel_list.append(Relationship(
                source_entity=src,
                target_entity=tgt,
                relationship_description=rdesc,
                relationship_keywords=rkeys
            ))
        except Exception as e:
            parsing_errors.append(f"Failed to build relationship {src}->{tgt}: {e}")

    if orphaned_rels:
        logger.warning(f"Filtered {len(orphaned_rels)} orphaned relationships (entities not in valid set)")

    if parsing_errors:
        logger.warning(f"Relationship parsing: {len(rel_list)} parsed, {len(parsing_errors)} issues")

    return rel_list


def _parse_entity_ops_block(text: str) -> dict[str, list[dict[str, str | None]]]:
    """
    Parse the LLM output: '||'-separated fields wrapped in ===BEGIN=== / ===END===.
    Line format:
    input_name||input_type||action||target_existing_id_or_NULL||canonical_name||canonical_type||merged_description
    """
    m = re.search(r"===BEGIN===\s*(.*?)\s*===END===", text, flags=re.DOTALL)
    if not m:
        return {"results": []}
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    results = []
    for ln in lines:
        parts = ln.split("||")
        if len(parts) < 7:
            # Skip lines whose field count is incomplete
            continue
        input_name, input_type, action, target_id, cname, ctype, mdesc = parts[:7]
        results.append({
            "input_name": input_name,
            "input_type": input_type,
            "action": action,
            "target_existing_id": None if target_id == "NULL" else target_id,
            "canonical_name": cname,
            "canonical_type": ctype,
            "merged_description": mdesc
        })
    return {"results": results}
