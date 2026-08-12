# utils.py
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

# ---------- 檔案相關 ----------
def file_exists(*paths: str | Path) -> bool:
    """所有路徑都存在才回傳 True"""
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
    """若 index+meta 均存在才呼叫 vdb.load()"""
    if file_exists(index_path, meta_path):
        vdb_obj.load()
        logger.info("Loaded VDB from %s / %s", index_path, meta_path)

## id相關

def _slugify(text: str) -> str:
    """Convert an identifier fragment into a simple lowercase storage key."""
    return (text.strip().lower().replace(" ", "_").replace("/", "_").replace("::", "_"))

def canonical_entity_id(name: str, etype: str) -> str:
    """Build the canonical entity ID from its type and name."""
    return _slugify(f"{etype}::{name}")

def canonical_rel_id(src_id: str, tgt_id: str) -> str:
    """Build the canonical relationship ID from source and target entity IDs."""
    return _slugify(f"{src_id}::{tgt_id}")

def _norm_name(s: str) -> str:
    """Normalize an entity name for case-insensitive cache-key comparisons."""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return re.sub(r"\s+", " ", s)

def _entity_key(name: str, etype: str) -> str:
    """Create the normalized cache key used for entity lookup and deduplication."""
    return f"{_norm_name(name)}::{etype.lower()}"

## schema 相關

# === 資料模型 ===
# class EntityType(str, Enum):
#     person="person"; organization="organization"; location="location"; geo="geo"
#     event="event"; animal="animal"; food="food"; product="product"
#     category="category"; unknown="unknown"

class EntityType(str, Enum):
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
    entity_name: str
    entity_type: EntityType
    entity_description: str
    entity_metadata: dict[str, Any] | None = None

class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_keywords: str

class ExtractionResult(BaseModel):
    entities: List[Entity] = []
    relationships: List[Relationship] = []

class KeywordExtractionResult(BaseModel):
    high_level_keywords: List[str] = []
    low_level_keywords: List[str] = []

# === JSON Schema（給 response_format 用）===
SCHEMA = ExtractionResult.model_json_schema()
_raw_kw_schema = KeywordExtractionResult.model_json_schema()
# Force LLM to produce at least one keyword per list (prevents empty {} shortcut)
for _field in ("high_level_keywords", "low_level_keywords"):
    _raw_kw_schema["properties"][_field]["minItems"] = 1
_raw_kw_schema["required"] = ["high_level_keywords", "low_level_keywords"]
SCHEMA_keyword = _raw_kw_schema

def parse_delimited_extraction(raw: str, tuple_delim: str, record_delim: str, completion_token: str) -> ExtractionResult:
    """
    Parse LLM extraction output with improved error handling and logging.

    Returns ExtractionResult which may have empty entities/relationships if:
    - LLM output is empty or malformed
    - Entity types don't match EntityType enum
    - Required fields are missing
    - Relationships reference non-existent entities
    """
    if not raw:
        logger.warning("parse_delimited_extraction: Empty raw input")
        return ExtractionResult(entities=[], relationships=[])

    # 只保留 completion_token 前面的內容
    raw = raw.split(completion_token)[0]

    # 用 record_delim 切成一條一條記錄
    parts = [p.strip() for p in raw.split(record_delim)]
    ent_list, rel_list = [], []

    # Track parsing errors for debugging
    parsing_errors = {"entity_errors": [], "relationship_errors": [], "skipped_lines": []}

    def _clean(s: str) -> str:
        """Trim a raw extracted field and remove one pair of wrapping quotes."""
        s = (s or "").strip()
        # 去掉外層成對的引號
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
        return s

    for line in parts:
        if not line:
            continue

        line_stripped = line.strip()

        # 支援舊格式：("entity"<|>...) / ("relationship"<|>...)
        if line_stripped.startswith("(") and line_stripped.endswith(")"):
            line_stripped = line_stripped[1:-1].strip()

        # 用 tuple_delim 切欄位
        cols = [c.strip() for c in line_stripped.split(tuple_delim)]
        if not cols:
            continue

        # 第一欄應該是 entity / relationship（可能有引號）
        kind = _clean(cols[0]).lower()

        # ---------------------
        # 解析 Entity 記錄
        # ---------------------
        if kind == "entity":
            # 至少要有 4 欄：entity, name, type, desc
            if len(cols) < 4:
                parsing_errors["entity_errors"].append({
                    "name": "MISSING",
                    "type": "MISSING",
                    "error": f"Too few columns for entity line: got {len(cols)}"
                })
                continue

            name = _clean(cols[1])
            etype = _clean(cols[2])

            # desc 可能本身包含 tuple_delim，保守作法：把第 3 欄之後都當成 description
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
        # 解析 Relationship 記錄
        # -------------------------
        if kind == "relationship":
            # 至少要有 5 欄：relationship, src, tgt, desc, keywords
            if len(cols) < 5:
                parsing_errors["relationship_errors"].append({
                    "src": "MISSING",
                    "tgt": "MISSING",
                    "error": f"Too few columns for relationship line: got {len(cols)}"
                })
                continue

            src = _clean(cols[1])
            tgt = _clean(cols[2])

            # 描述可能含 tuple_delim：用「最後一欄當 keywords，3~倒數第二欄 join 成 description」
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

        # 既不是 entity 也不是 relationship，當作 skipped line
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

# 只清「尾端」的 generation artefacts，避免誤傷內容
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

    # 先試原本 enum 建構
    try:
        return EntityType(etype)
    except Exception:
        pass

    # 再試 case-insensitive match on enum values
    etype_l = etype.lower()
    for m in EntityType:
        try:
            if str(m.value).lower() == etype_l:
                return m
        except Exception:
            continue

    # 若你的 Enum 也可能用 name 表示（例如 EntityType.PERSON）
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

    # 防線 1：先切 completion token（record 外部）
    raw = raw.split(completion_token)[0] if completion_token else raw

    # 防線 2：按 record delimiter 切（並容忍空白）
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

        # 🔒 固定 5 欄語意：src, tgt, desc, keywords
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
    解析 LLM 以 '||' 分隔、包在 ===BEGIN=== / ===END=== 的輸出。
    每行格式：
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
            # 跳過格式不完整的行
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

## BM25斷詞

# 盡量用 NLTK 的 word_tokenize；若缺資源或安裝問題，就 fallback 到 regex
import nltk
from nltk import word_tokenize, pos_tag

# 可自行擴增
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
        # Fallback：沒 POS tagger 時全保留 toks（但你有 nltk，基本不會發生）
        return toks

    KEEP_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "FW"}

    filtered = []

    for token, tag in tags:
        # ✅ 日期直接保留（不進 POS / stopword / length 規則）
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
        # 1) 全字母 & 長度>=3 & 非停用詞 → 保留
        if token.isalpha():
            filtered.append(token)
            continue

    return filtered
