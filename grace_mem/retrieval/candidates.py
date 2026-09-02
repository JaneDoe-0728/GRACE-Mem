"""What stage 1 of retrieval produces.

Hybrid entity search, spreading activation and graph expansion between them
decide what can be retrieved at all. Their result used to leave the stage as
loose local variables threaded into the stages that follow; naming it makes the
boundary between "finding candidates" and "narrowing them" something the type
system holds rather than something the reader has to reconstruct.

Two subgraphs and nothing else. It also carried four per-source score maps for
RRF to fuse; RRF was deleted in 20ce40f and the reranker scores the pool itself,
so the maps were accumulated on every query and read by nobody.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateSet:
    """Everything stage 1 hands to stage 2.

    Attributes:
        node_subgraph: entity id -> metadata, for every entity reachable from
            the search hits. The pool the reranker narrows.
        edge_subgraph: the relationship records joining those entities.
    """

    node_subgraph: dict[str, Any] = field(default_factory=dict)
    edge_subgraph: list[Any] = field(default_factory=list)
