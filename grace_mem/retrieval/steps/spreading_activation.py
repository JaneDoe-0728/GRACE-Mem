"""
Spreading Activation guided retrieval — faithful to Algorithm 1 in the paper.

Two-phase execution
-------------------
Phase 1  Build adj_dict
    Fetch the N-hop neighbourhood of seed entities from the graph (level-by-level).
    For every discovered edge, compute:
        w_ij   = cos(query_vec, rel_embed(r_ij))   # query-guided edge relevance
        w'_ij  = (w_ij - c) / (1 - c)             # linear rescale, c ≈ 0.4
    Edges with w'_ij ≤ 0 are dropped.

Phase 2  BFS spreading activation  (Algorithm 1 verbatim)
    activation_value[seed]   = 1   for each seed entity
    activation_value[other]  = 0
    BFS from seeds; for each edge (node → target, weight):
        activation_value[target] = min(1, activation_value[target]
                                         + weight · activation_value[node])
    Activated set Ra = { e | activation_value[e] > τa }

Paper best values:  c = 0.4,  τa = 0.5
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from grace_mem.services.cache.cache import build_id_to_meta_maps
from grace_mem.utils.logger_config import _StepTimer, make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retrieval.SA", filename="kg_retrieval_sa.jsonl")


@dataclass
class SAConfig:
    """Parameters for spreading activation over the entity graph.

    See the module docstring for what the algorithm does. The two thresholds
    pull against each other: `max_hops` bounds how far activation travels and
    `tau_a` bounds how weak a node may be and still count as activated. Raising
    hops without raising tau_a floods the result with distant, barely-activated
    nodes.
    """
    max_hops: int = 2          # depth used to build the subgraph (Phase 1)
    rescale_c: float = 0.4     # c in w' = (w - c) / (1 - c)
    tau_a: float = 0.5         # activation threshold — only nodes above this are "activated"
    max_activated: int = 20    # deprecated: no longer caps results (kept for logging compatibility)


class SpreadingActivationEngine:
    """
    Runs spreading activation over the entity-relation graph.

    Edge weights = cos(query_vec, relation_embedding), rescaled with the
    paper's linear factor to prevent activation explosion.
    """

    def __init__(self, graph, vector_db_manager, config: SAConfig, cache: dict | None = None):
        self.graph = graph
        self.vdb = vector_db_manager
        self.config = config
        self.cache = cache or {}

    def _entity_name(self, entity_id: str) -> str:
        """Resolve one entity ID into a readable entity name."""
        ent_id2meta, _ = build_id_to_meta_maps(self.cache)
        meta = ent_id2meta.get(entity_id, {}) or {}
        return meta.get("name") or entity_id

    def _edge_label(self, source_id: str, rel_id: str | None, target_id: str) -> str:
        """Render one traversed edge into a readable label."""
        source_name = self._entity_name(source_id)
        target_name = self._entity_name(target_id)
        _, rel_id2meta = build_id_to_meta_maps(self.cache)
        meta = rel_id2meta.get(rel_id or "", {}) or {}
        desc = (meta.get("description") or meta.get("keywords") or "").strip()
        return f"{source_name} -> {target_name}" if not desc else f"{source_name} -> {target_name} | {desc}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        seed_entity_ids: list[str],
        query_vec: np.ndarray,
        request_id: str | None = None,
    ) -> dict[str, float]:
        """
        Run spreading activation from seed entities.

        Args:
            seed_entity_ids: Entity IDs found by initial search (all get activation = 1.0).
            query_vec:       Normalised query embedding (weights the edges).
            request_id:      Propagated to log entries.

        Returns:
            {entity_id -> activation_score} for activated entities (score > τa),
            sorted descending.
            Seeds that stay at 1.0 without being "re-activated" are also included.
        """
        if not seed_entity_ids:
            _jlog("sa_skipped", request_id, step="2.1.5", reason="no_seed_entities")
            return {}

        timer = _StepTimer()

        _jlog(
            "sa_start",
            request_id,
            step="2.1.5",
            seed_count=len(seed_entity_ids),
            sample_seed_ids=seed_entity_ids[:10],
            sample_seed_names=[self._entity_name(entity_id) for entity_id in seed_entity_ids[:10]],
            max_hops=self.config.max_hops,
            rescale_c=self.config.rescale_c,
            tau_a=self.config.tau_a,
            max_activated=self.config.max_activated,
        )

        # ── Phase 1: build in-memory adj_dict ────────────────────────────
        adj_dict: dict[str, list[tuple[str, float]]] = {}   # node → [(target, w'), ...]
        all_nodes: set[str] = set(seed_entity_ids)

        frontier = list(seed_entity_ids)
        _jlog(
            "sa_phase1_start",
            request_id,
            step="2.1.5",
            frontier_count=len(frontier),
        )
        phase1_edges_total = 0
        phase1_edges_kept = 0
        phase1_edges_pruned = 0
        for hop in range(self.config.max_hops):
            if not frontier:
                break

            timer_hop = _StepTimer()
            subgraph = self.graph.get_node_subgraph(frontier)
            next_frontier: list[str] = []
            hop_edges_total = 0
            hop_edges_kept = 0
            hop_edges_pruned = 0
            kept_edge_names: list[str] = []
            pruned_edge_names: list[str] = []

            for eid, node_data in subgraph.items():
                arcs: list[tuple[str, float]] = []

                for neighbor in (node_data.get("neighbors") or []):
                    neighbor_id = neighbor.get("neighbor_id")
                    rel_id = neighbor.get("rel_id")
                    if not neighbor_id:
                        continue

                    # Edge weight: query vs relation embedding, then rescale
                    hop_edges_total += 1
                    w = self._raw_edge_weight(rel_id, query_vec)
                    w_prime = (w - self.config.rescale_c) / (1.0 - self.config.rescale_c)
                    if w_prime <= 0.0:
                        hop_edges_pruned += 1
                        if len(pruned_edge_names) < 20:
                            pruned_edge_names.append(self._edge_label(eid, rel_id, neighbor_id))
                        continue  # prune weak / irrelevant edges

                    arcs.append((neighbor_id, w_prime))
                    hop_edges_kept += 1
                    if len(kept_edge_names) < 20:
                        kept_edge_names.append(self._edge_label(eid, rel_id, neighbor_id))

                    if neighbor_id not in all_nodes:
                        all_nodes.add(neighbor_id)
                        next_frontier.append(neighbor_id)

                # Merge arcs (a node may be reached via multiple frontier calls)
                if eid not in adj_dict:
                    adj_dict[eid] = arcs
                else:
                    adj_dict[eid].extend(arcs)

            frontier = next_frontier
            phase1_edges_total += hop_edges_total
            phase1_edges_kept += hop_edges_kept
            phase1_edges_pruned += hop_edges_pruned
            _jlog(
                "sa_phase1_hop",
                request_id,
                step="2.1.5",
                hop=hop + 1,
                frontier_count=len(subgraph),
                new_nodes=len(next_frontier),
                new_node_names=[self._entity_name(entity_id) for entity_id in next_frontier[:20]],
                total_nodes=len(all_nodes),
                edges_total=hop_edges_total,
                edges_kept=hop_edges_kept,
                edges_pruned=hop_edges_pruned,
                kept_edge_names=kept_edge_names,
                pruned_edge_names=pruned_edge_names,
                elapsed_sec=timer_hop.sec(),
            )

        _jlog(
            "sa_phase1_complete",
            request_id,
            step="2.1.5",
            total_nodes=len(all_nodes),
            total_node_names=[self._entity_name(entity_id) for entity_id in list(all_nodes)[:20]],
            total_edges=phase1_edges_total,
            kept_edges=phase1_edges_kept,
            pruned_edges=phase1_edges_pruned,
        )

        # ── Phase 2: BFS spreading activation (Algorithm 1) ──────────────
        activation_value: dict[str, float] = {e: 0.0 for e in all_nodes}
        for eid in seed_entity_ids:
            activation_value[eid] = 1.0

        visited: set[str] = set()
        queue: deque = deque(seed_entity_ids)
        _jlog(
            "sa_phase2_start",
            request_id,
            step="2.1.5",
            initial_queue_size=len(queue),
        )
        timer_phase2 = _StepTimer()
        propagation_count = 0

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            for target, weight in (adj_dict.get(node) or []):
                propagation_count += 1
                activation_value[target] = min(
                    1.0,
                    activation_value[target] + weight * activation_value[node],
                )
                if target not in visited:
                    queue.append(target)

        _jlog(
            "sa_phase2_complete",
            request_id,
            step="2.1.5",
            visited_count=len(visited),
            visited_names=[self._entity_name(entity_id) for entity_id in list(visited)[:20]],
            propagation_count=propagation_count,
            elapsed_sec=timer_phase2.sec(),
        )

        # ── Collect activated entities ────────────────────────────────────
        activated = {
            e: v
            for e, v in activation_value.items()
            if v > self.config.tau_a
        }

        # Sort by activation score (no cap — keep all activated nodes)
        result = dict(
            sorted(activated.items(), key=lambda x: x[1], reverse=True)
        )

        _jlog(
            "sa_complete",
            request_id,
            step="2.1.5",
            seed_count=len(seed_entity_ids),
            subgraph_size=len(all_nodes),
            activated_count=len(activated),
            returned_count=len(result),
            elapsed_sec=timer.sec(),
            top5=[(eid, round(sc, 4)) for eid, sc in list(result.items())[:5]],
            top5_names=[
                {"name": self._entity_name(eid), "score": round(sc, 4)}
                for eid, sc in list(result.items())[:5]
            ],
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_edge_weight(self, rel_id: str | None, query_vec: np.ndarray) -> float:
        """
        cos(query_vec, rel_embed(rel_id))  — before rescaling.
        Returns 0.0 if the relation is not found in the VDB.
        """
        if not rel_id:
            return 0.0
        try:
            rel_vdb = self.vdb.get_relationships_vdb(query_vec.shape[0])
            res = rel_vdb.compare_by_id(rel_id, query_vec, threshold=0.0)
            if res is not None:
                _, score = res
                return max(0.0, float(score))
        except Exception:
            pass
        return 0.0
