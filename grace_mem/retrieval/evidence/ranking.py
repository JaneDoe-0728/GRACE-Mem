"""Ordering the evidence candidates and taking the global top-K.

One decision, kept out of the builder because it is the only place the shape of
the final evidence set is chosen: everything before it scores candidates and
everything after it fetches text for whatever survived.
"""

from grace_mem.utils.logger_config import make_module_jlog

_jlog = make_module_jlog(name="grace_mem.Retrieval.Evidence", filename="kg_retrieval_evidence.jsonl")


def rank_and_cut(
        *,
        scored_events: list,
        summary_topk_global,
        summary_per_entity_min: int,
        request_id: str | None,
        ev_source: dict,
    ) -> list:
        """Stage 3: order the candidates and take the global top-K.

        The per-entity quota runs before the global cut, not after: without it a
        single well-matched entity can take every slot, and the question that
        needed two sources gets one source twice.
        """
        # ========== Stage 3: Rank and take global top-K ==========
        # "semantic" mode: sort by semantic score (baseline, unchanged behaviour).
        # Graph modes: rerank via graph-linked scoring formula.
        topk = summary_topk_global if isinstance(summary_topk_global, int) else None

        # Semantic mode: stable descending sort by cosine score
        scored_events.sort(key=lambda x: x[0], reverse=True)

        # Per-entity quota: guarantee at least summary_per_entity_min snippets
        # per source entity/relationship before applying global top-K.
        if summary_per_entity_min > 0 and topk is not None:
            # Step 1: collect the best `summary_per_entity_min` events per source
            source_counts: dict[str, int] = {}
            guaranteed: list[tuple[float, dict]] = []
            remainder: list[tuple[float, dict]] = []
            for score, ev in scored_events:
                src = ev_source.get(id(ev), "")
                if source_counts.get(src, 0) < summary_per_entity_min:
                    guaranteed.append((score, ev))
                    source_counts[src] = source_counts.get(src, 0) + 1
                else:
                    remainder.append((score, ev))
            # Step 2: fill remaining slots from remainder (already sorted by score)
            n_fill = max(0, topk - len(guaranteed))
            scored_events = guaranteed + remainder[:n_fill]
            _jlog(
                "evidence_per_entity_quota",
                request_id,
                step="3.3",
                per_entity_min=summary_per_entity_min,
                guaranteed_count=len(guaranteed),
                remainder_count=len(remainder),
                filled=min(n_fill, len(remainder)),
                total_selected=len(scored_events),
            )
        elif topk is not None:
            scored_events = scored_events[:topk]

        _jlog(
            "evidence_topk_selected",
            request_id,
            step="3.3",
            requested_topk=summary_topk_global,
            selected_count=len(scored_events),
            sample_top_candidates=[
                {
                    "rank": i + 1,
                    "score": score,
                    "summary_id": ev.get("summary_id"),
                    "session_id": ev.get("session_id"),
                    "message_id": ev.get("message_id"),
                }
                for i, (score, ev) in enumerate(scored_events[:topk] if topk is not None else scored_events)
            ],
        )


        return scored_events
