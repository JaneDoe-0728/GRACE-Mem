"""What stage 1 of retrieval produces.

Hybrid entity search, spreading activation and graph expansion between them
decide what can be retrieved at all. Their result used to leave the stage as
six loose local variables threaded into the stages that follow; naming it makes
the boundary between "finding candidates" and "narrowing them" something the
type system holds rather than something the reader has to reconstruct.

Two subgraphs and four score maps, because the stages downstream need both what
was found and how well each thing scored -- RRF fuses the four maps, and the
filter dispatch reads the subgraphs.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateSet:
    """Everything stage 1 hands to stage 2.

    Attributes:
        node_subgraph: entity id -> metadata, for every entity reachable from
            the search hits. The pool the filters narrow.
        edge_subgraph: the relationship records joining those entities.
        entity_emb_scores: entity id -> dense similarity to the query.
        entity_bm25_scores: entity id -> lexical score. Separate from the dense
            scores because RRF fuses the two rankings rather than the values,
            so they must not be averaged before it sees them.
        rel_emb_scores: relationship id -> dense similarity.
        rel_endpoint_scores: relationship id -> score inherited from its
            endpoints, for edges no vector search returned directly.
    """

    node_subgraph: dict[str, Any] = field(default_factory=dict)
    edge_subgraph: list[Any] = field(default_factory=list)
    entity_emb_scores: dict[str, float] = field(default_factory=dict)
    entity_bm25_scores: dict[str, float] = field(default_factory=dict)
    rel_emb_scores: dict[str, float] = field(default_factory=dict)
    rel_endpoint_scores: dict[str, float] = field(default_factory=dict)
