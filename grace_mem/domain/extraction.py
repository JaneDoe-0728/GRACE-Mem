"""What one LLM call yields, and the schemas that shape it.

The JSON schemas are generated from the models rather than written alongside
them, so the schema the LLM is constrained by and the parser that validates its
reply can never disagree.
"""

from pydantic import BaseModel

from grace_mem.domain.entities import Entity
from grace_mem.domain.relationships import Relationship


class ExtractionResult(BaseModel):
    """Everything extracted from a single conversation turn.

    Both fields default to empty so a turn that yields nothing is an ordinary
    empty result rather than an error -- plenty of turns are pure
    back-channel ("sure", "ok") and contain no facts at all.
    """

    entities: list[Entity] = []
    relationships: list[Relationship] = []

class KeywordExtractionResult(BaseModel):
    """Query keywords, split by how they will be used in retrieval.

    See `llm/prompts/keyword/extraction.py` for what distinguishes the two
    lists and why one call produces both.
    """

    high_level_keywords: list[str] = []
    low_level_keywords: list[str] = []

# --- JSON Schemas passed to the LLM via response_format ------------------
#
# Generated from the models above so the schema the LLM is constrained by and
# the parser that validates its reply can never disagree.

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
