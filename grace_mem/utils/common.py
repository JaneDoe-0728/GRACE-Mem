"""Shared data models, id conventions, and the extraction-output parser.

Three things live here because everything else depends on them and a home
further down the tree would create import cycles.

The data models (`Entity`, `Relationship`, `ExtractionResult`) are the contract
between extraction, storage, and retrieval. They are Pydantic models so the
same declaration serves two purposes: runtime validation of LLM output, and the
JSON Schema handed to the model via `response_format` -- the schema and the
parser cannot drift apart because they are generated from one source.

The id helpers define entity identity. `canonical_entity_id` is what makes the
graph converge: two extractions of "Dr. Smith" and "dr smith" must produce the
same id or the KG grows a duplicate node per mention. Identity is
(type, normalized name), so the same string under two types stays two entities
-- a "Paris" Location and a "Paris" Person are genuinely different.

`parse_delimited_extraction` recovers structure from the model's delimited text
output. It is long and forgiving on purpose: extraction runs once per turn over
a whole corpus, so a parser that raised on the first malformed record would
lose the run. Every tolerated deformation there is one observed in practice.
"""

from pathlib import Path
import pickle, logging
from typing import Any
from typing import Callable, Optional, List
import re, unicodedata
from pydantic import BaseModel
from enum import Enum

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

# ---------- file helpers ----------
def file_exists(*paths: str | Path) -> bool:
    """Return True only when every path exists."""
    return all(Path(p).exists() for p in paths)


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

# ---------- Pickle ----------
def pickle_dump(path: str | Path, obj: Any) -> None:
    """Serialize an object to pickle, logging instead of raising on failure."""
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        logger.error("Pickle dump failed: %s → %s", path, e)

def pickle_load(path: str | Path, default: Any = None) -> Any:
    """Load a pickle file and return a fallback value when it is unavailable."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        logger.error("Pickle load failed: %s → %s", path, e)
        return default

def load_vdb_if_exists(vdb_obj: Any, index_path: str | Path, meta_path: str | Path) -> None:
    """Call vdb.load() only when both the index and the meta file exist."""
    if file_exists(index_path, meta_path):
        vdb_obj.load()
        logger.info("Loaded VDB from %s / %s", index_path, meta_path)

# --- Identity: ids and cache keys ---------------------------------------

def _slugify(text: str) -> str:
    """Reduce an id fragment to a lowercase, separator-free key.

    `/` and `::` are replaced rather than kept because ids end up in graph node
    keys and filesystem paths, where both are structural characters.
    """
    return (text.strip().lower().replace(" ", "_").replace("/", "_").replace("::", "_"))

def canonical_entity_id(name: str, etype: str) -> str:
    """Derive an entity's stable id from its type and name.

    Deterministic rather than generated, which is what lets ingestion recognise
    a re-mention across turns and sessions without a lookup: the same person
    named the same way yields the same id in any process, in any run.
    """
    return _slugify(f"{etype}::{name}")

def canonical_rel_id(src_id: str, tgt_id: str) -> str:
    """Derive a relationship's stable id from its endpoints.

    Direction-sensitive: (a, b) and (b, a) are different edges, because
    "manages" does not mean the same thing reversed. Note the corollary --
    only one edge can exist per ordered pair, so a second relationship between
    the same two entities merges into the first rather than coexisting.
    """
    return _slugify(f"{src_id}::{tgt_id}")

def _norm_name(s: str) -> str:
    """Normalize an entity name for identity comparison.

    NFKC folds the compatibility variants that arrive from copied text --
    fullwidth forms, ligatures -- onto their ASCII equivalents, so a name that
    looks identical on screen compares identical here. Whitespace runs collapse
    for the same reason.
    """
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return re.sub(r"\s+", " ", s)

def _entity_key(name: str, etype: str) -> str:
    """Build the in-memory cache key for an entity.

    Distinct from `canonical_entity_id`: this one keeps the "::" separator and
    skips slugification, so it stays reversible for debugging. Do not persist
    it -- ids are the durable form.
    """
    return f"{_norm_name(name)}::{etype.lower()}"

# --- Data models ---------------------------------------------------------

class EntityType(str, Enum):
    """The closed set of entity types extraction may emit.

    Closed on purpose. A free-text type field let the model invent near
    synonyms ("person", "individual", "human") that split one real entity
    across several graph nodes. Inheriting from `str` keeps members usable
    directly as dict keys and in serialized metadata.

    Date/Time/Timespan are separate members rather than one "temporal" type
    because retrieval filters on granularity -- a query bounded to a day must
    not match a node spanning a year.
    """

    Person      = "Person"
    Event       = "Event"
    Date        = "Date"
    Time        = "Time"
    Timespan    = "Timespan"
    Location    = "Location"
    Organization= "Organization"
    Product     = "Product"
    Service     = "Service"
    Activity    = "Activity"
    Topic       = "Topic"
    Concept     = "Concept"

class Entity(BaseModel):
    """One node in the knowledge graph.

    Attributes:
        entity_description: Free text, and the field that is embedded for dense
            retrieval -- the name alone is too short to encode usefully. An
            entity with an empty description is effectively unsearchable.
        entity_metadata: Type-specific extras. Temporal entities carry their
            resolved value here; see `pipeline/ingestor.py`.
    """

    entity_name: str
    entity_type: EntityType
    entity_description: str
    entity_metadata: dict[str, Any] | None = None

class Relationship(BaseModel):
    """One directed edge between two entities.

    Endpoints are entity *names*, not ids, because extraction produces this
    model before ids are assigned. Resolving names to ids is the syncer's job,
    and it is where an edge naming an entity that was never extracted gets
    dropped.

    Attributes:
        relationship_keywords: Lexical anchors for BM25 edge search, stored as
            one delimited string rather than a list -- the graph backends
            accept only scalar property values.
    """

    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_keywords: str

class ExtractionResult(BaseModel):
    """Everything extracted from a single conversation turn.

    Both fields default to empty so a turn that yields nothing is an ordinary
    empty result rather than an error -- plenty of turns are pure
    back-channel ("sure", "ok") and contain no facts at all.
    """

    entities: List[Entity] = []
    relationships: List[Relationship] = []

class KeywordExtractionResult(BaseModel):
    """Query keywords, split by how they will be used in retrieval.

    See `llm/prompts/keyword/extraction.py` for what distinguishes the two
    lists and why one call produces both.
    """

    high_level_keywords: List[str] = []
    low_level_keywords: List[str] = []

# --- JSON Schemas passed to the LLM via response_format ------------------
#
# Generated from the models above so the schema the LLM is constrained by and
# the parser that validates its reply can never disagree.

SCHEMA = ExtractionResult.model_json_schema()
_raw_kw_schema = KeywordExtractionResult.model_json_schema()
# Both keyword lists are tightened to minItems=1 and marked required. Pydantic
# defaults them to empty, which the schema advertises as permission: given a
# hard query the model would return `{}` rather than attempt keywords, and an
# empty low_level list silently disables BM25 for that question. Constrained
# decoding enforces this, so the model must produce something.
for _field in ("high_level_keywords", "low_level_keywords"):
    _raw_kw_schema["properties"][_field]["minItems"] = 1
_raw_kw_schema["required"] = ["high_level_keywords", "low_level_keywords"]
SCHEMA_keyword = _raw_kw_schema

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
    parsing_errors = {"entity_errors": [], "relationship_errors": [], "skipped_lines": []}

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

import re
from typing import List, Set

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


def parse_entities_only(raw: str, tuple_delim: str, record_delim: str, completion_token: str) -> List[Entity]:
    """Parse only entity rows from the model output and drop malformed entries."""
    if not raw:
        logger.warning("parse_entities_only: Empty raw input")
        return []

    # Guard 1: cut at the completion token first (it sits outside the records)
    raw = raw.split(completion_token)[0] if completion_token else raw

    # Guard 2: split on the record delimiter, tolerating blank runs
    parts = [p.strip() for p in (raw.split(record_delim) if record_delim else raw.splitlines()) if p and p.strip()]

    ent_list: List[Entity] = []
    parsing_errors: List[str] = []

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
    valid_entity_names: Set[str]
) -> List[Relationship]:
    """Parse only relationship rows and keep only ones whose endpoints are valid."""
    if not raw:
        logger.warning("parse_relationships_only: Empty raw input")
        return []

    raw = raw.split(completion_token)[0] if completion_token else raw
    parts = [p.strip() for p in (raw.split(record_delim) if record_delim else raw.splitlines()) if p and p.strip()]

    rel_list: List[Relationship] = []
    parsing_errors: List[str] = []
    orphaned_rels: List[tuple] = []

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
    m = re.search(r"===BEGIN===\s*(.*?)\s*===END===", text, flags=re.S)
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

## BM25 tokenization

# Prefer NLTK's word_tokenize; fall back to a regex if the resources or the
# install are missing
import nltk
from nltk import word_tokenize, pos_tag

# Extend as needed
EN_STOPWORDS = {
    "the","a","an","in","on","at","for","to","from","of",
    "and","or","but","with","without","is","are","was","were",
    "be","been","being","it","its","they","them","this","that",
    "these","those","as","by","into","over","under","up","down",
}

# regex fallback
_WORD_RE = re.compile(r"[a-zA-Z]+")

# ===== date token patterns =====

# 1) ISO / full year formats
_DATE_YMD_RE = re.compile(
    r"^\d{4}([-/.])\d{1,2}\1\d{1,2}$" # matches: 2023-03-04, 2023/3/4, 2023.03.04
)

# 2) Short numeric dates (month/day)
_DATE_MD_RE = re.compile(
    r"^\d{1,2}/\d{1,2}$" # matches: 4/1, 04/01
)

def is_date_token(token: str) -> bool:
    """Return whether a token already looks like a date literal."""
    if not token:
        return False
    return bool(
        _DATE_YMD_RE.match(token)
        or _DATE_MD_RE.match(token)
    )

def tokenize_en(text: str) -> list[str]:
    """Tokenize English text for BM25 while preserving informative date tokens."""
    t = (text or "").strip()
    if not t:
        return []

    # Step 1: basic tokenize
    try:
        toks = word_tokenize(t)
    except Exception:
        toks = _WORD_RE.findall(t)
        
    toks = [w.lower() for w in toks if any(c.isalpha() for c in w) or is_date_token(w)]

    # Step 2: POS tagging
    try:
        tags = pos_tag(toks)
    except Exception:
        # Fallback: with no POS tagger, keep every token (nltk is installed here,
        # so this is effectively unreachable)
        return toks

    KEEP_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "FW"}

    filtered = []

    for token, tag in tags:
        # Dates are kept verbatim, bypassing the POS, stopword and length rules
        if is_date_token(token):
            filtered.append(token)
            continue

        # Length too short
        if len(token) < 3:
            continue

        # Stopwords / known noise
        if token in EN_STOPWORDS:
            continue

        # POS-based filtering
        if tag in KEEP_TAGS:
            filtered.append(token)
            continue

        # Fallback for unknown proper nouns / foreign words
        # 1) all-alphabetic & length >= 3 & not a stopword -> keep
        if token.isalpha():
            filtered.append(token)
            continue

    return filtered
