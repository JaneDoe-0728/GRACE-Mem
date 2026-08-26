"""Provenance tracking: where an entity or relationship came from.

Every entity and relationship in the KG carries a provenance blob recording the
turns it was extracted from. That is what makes an answer auditable -- given a
retrieved fact, the evidence stage walks provenance back to the original
session and message.

The complication this module absorbs is that provenance arrives in two shapes.
Extraction emits a flat form (a list of summary_ids plus one session/message
id), while anything already merged carries an explicit `events` list. Callers
should never branch on which: normalize through `prov_to_events` first, then
merge.
"""


class Provenance:
    """Normalize and merge provenance records into a single event form.

    Stateless -- both methods are static. It is a class rather than two module
    functions so the pair stays findable together as the provenance format
    evolves.
    """

    @staticmethod
    def prov_to_events(prov: dict) -> list[dict]:
        """Normalize any provenance shape into a flat event list.

        Accepts either an already-normalized `{"events": [...]}` blob or the
        flat `summary_ids` / `session_id` / `message_id` form that extraction
        produces, and emits events keyed
        {ts, summary_id, session_id, message_id, dialogue_datetime}.

        Two normalizations matter downstream. Ids are coerced to strings
        because the two input shapes disagree about int vs str, and a merge
        deduplicates on those ids -- an int 3 and a str "3" would survive as
        two distinct events. And when session/message are absent from the flat
        form, they are recovered by splitting a "session:message" summary_id,
        which is the only place that information exists for older records.

        Args:
            prov: Provenance blob in either shape. Falsy input yields [].

        Returns:
            Events in input order. `ts` is a positional index for the flat
            form, not a timestamp, so it orders events within one blob but
            carries no meaning across blobs.
        """
        if not prov: return []
        if isinstance(prov.get("events"), list):
            evs = []
            for e in prov["events"]:
                sess = e.get("session_id") or e.get("session_id ") or e.get("sess_id")
                msg  = e.get("message_id") or e.get("msg_id")
                dt = e.get("dialogue_datetime")
                evs.append({
                    "ts": e.get("ts", 0),
                    "summary_id": e.get("summary_id"),
                    "session_id": None if sess is None else str(sess),
                    "message_id": None if msg is None else str(msg),
                    "dialogue_datetime": dt
                })
            return evs
        summary_ids = prov.get("summary_ids") or []
        if isinstance(summary_ids, str): summary_ids = [summary_ids]
        sess = prov.get("session_id")
        msg = prov.get("message_id")
        dt = prov.get("dialogue_datetime")
        sess = None if sess is None else str(sess)
        msg = None if msg is None else str(msg)
        evs = []
        for idx, sid in enumerate(summary_ids):
            s_sess, s_msg = sess, msg
            # Recover the origin from the summary_id itself when the blob did
            # not carry one. Older records encode it as "<session>:<message>",
            # and without this the event is unusable for evidence lookup.
            if (s_sess is None or s_msg is None) and isinstance(sid, str) and ":" in sid:
                try: p_sess, p_msg = sid.split(":", 1); s_sess = s_sess or str(p_sess); s_msg = s_msg or str(p_msg)
                except Exception: pass
            evs.append({
                "ts": idx,
                "summary_id": sid,
                "session_id": s_sess,
                "message_id": s_msg,
                "dialogue_datetime": dt
            })
        return evs

    @staticmethod
    def merge_prov(old: dict | None, new: dict | None) -> dict:
        """Merge two provenance blobs, deduplicating on origin.

        Called whenever an entity is seen again in a later turn, so the merged
        result accumulates every turn that mentioned it. `new` wins on
        collision: a re-extraction of the same origin carries the fresher
        dialogue_datetime.

        Args:
            old: Existing provenance, or None for a first sighting.
            new: Incoming provenance.
        Returns:
            `{"events": [...]}` sorted by ts, ready to attach to an entity or
            relationship meta.
        """
        def _to_events(x: dict | None) -> list[dict]:
            """Convert one provenance blob into normalized events before merging."""
            return Provenance.prov_to_events(x or {})
        def k(e: dict) -> tuple[str | None, str | None, str | None]:
            """Build the deduplication key used when merging provenance events."""
            return (e.get("session_id"), e.get("message_id"), e.get("summary_id"))
        merged = {k(e): e for e in _to_events(old)}
        # Insert `new` second so it overwrites on key collision: same origin,
        # fresher record.
        for e in _to_events(new): merged[k(e)] = e
        events = sorted(merged.values(), key=lambda e: e.get("ts", 0))
        return {"events": events}
