"""The Relationship: one directed edge between two entities.

Endpoints are entity *names*, not ids, because extraction produces this model
before ids are assigned.
"""

from pydantic import BaseModel

from grace_mem.domain.entities import _slugify


def canonical_rel_id(src_id: str, tgt_id: str) -> str:
    """Derive a relationship's stable id from its endpoints.

    Direction-sensitive: (a, b) and (b, a) are different edges, because
    "manages" does not mean the same thing reversed. Note the corollary --
    only one edge can exist per ordered pair, so a second relationship between
    the same two entities merges into the first rather than coexisting.
    """
    return _slugify(f"{src_id}::{tgt_id}")


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
