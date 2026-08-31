"""Shared doubles for the ingestion characterization tests.

`Ingestor.summarize_and_ingest_turn` is 260 lines, and
`_repair_temporal_entities` beside it is 316 more, in a file that delegates
five times in total. It is the same shape retriever.py had, one size smaller.
These doubles let it be taken apart offline.

The boundary is five calls across four collaborators -- the compressor, the two
extractors, the syncer, and the vector manager -- all of which the Ingestor
builds in `__init__` from an LLM client, a graph and a manager. None of the
doubles needs FalkorDB, Chroma, an LLM or the models.

Reuses the CallLog from retrieval_fakes: the reason it exists there applies
here too, since a later step can mask an earlier one.
"""

from __future__ import annotations

from typing import Any

from grace_mem.domain.entities import Entity, EntityType
from grace_mem.domain.extraction import ExtractionResult
from grace_mem.domain.relationships import Relationship
from tests.retrieval_fakes import CallLog

# --------------------------------------------------------------------------- #
# The fixture turn                                                             #
# --------------------------------------------------------------------------- #
SESSION_ID = 1
MESSAGE_ID = 3
USER_TEXT = "I ran the Taipei marathon last Tuesday and bought new shoes after."
ASSISTANT_TEXT = "Congratulations. Which shoes did you pick?"
DIALOGUE_DATETIME = "2023/02/18 (Sat) 08:08"

SUMMARY_ID = f"{SESSION_ID}_{MESSAGE_ID}"
SUMMARY_TEXT = "The user ran the Taipei marathon and bought new running shoes."

#: Extraction results the doubles return. A temporal entity is included on
#: purpose: it is what _repair_temporal_entities exists to fix up, so a fixture
#: without one would leave 316 lines untouched.
EXTRACTED_ENTITIES = [
    Entity(entity_name="Taipei Marathon", entity_type=EntityType.Event,
           entity_description="A running event the user took part in."),
    Entity(entity_name="last Tuesday", entity_type=EntityType.Date,
           entity_description="The day the marathon happened."),
    Entity(entity_name="running shoes", entity_type=EntityType.Product,
           entity_description="Shoes bought after the marathon."),
]
EXTRACTED_RELATIONSHIPS = [
    Relationship(source_entity="Taipei Marathon", target_entity="last Tuesday",
                 relationship_description="The marathon took place on that day.",
                 relationship_keywords="when|date"),
    Relationship(source_entity="running shoes", target_entity="Taipei Marathon",
                 relationship_description="The shoes were bought after the marathon.",
                 relationship_keywords="after|purchase"),
]


class _Logged:
    def __init__(self, log: CallLog) -> None:
        self.log = log


class FakeCompressor(_Logged):
    """Turns one turn into the summary the graph is built from."""

    def summarize_turn(self, session_id, message_id, user_text, assistant_text,
                       request_id, dialogue_datetime=None, temporal_hints=None, tctx=None):
        self.log.record("compressor.summarize_turn", session_id=session_id,
                        message_id=message_id, dialogue_datetime=dialogue_datetime,
                        temporal_hints=temporal_hints, has_tctx=tctx is not None)
        return f"{session_id}_{message_id}", SUMMARY_TEXT


class FakeEntityExtractor(_Logged):
    """Signature matches the real EntityExtractor: (ok, payload)."""

    def extract(self, prompt_vars, prompt_template, request_id, *,
                tuple_delim=None, record_delim=None, completion_delim=None, max_retries=2):
        self.log.record("entity_extractor.extract",
                        prompt_var_keys=sorted(prompt_vars), tuple_delim=tuple_delim,
                        record_delim=record_delim, completion_delim=completion_delim)
        return True, list(EXTRACTED_ENTITIES)


class FakeRelationshipExtractor(_Logged):
    """Takes the extracted entities too: relationships are asked for in terms
    of the entities the first call found."""

    def extract(self, prompt_vars, prompt_template, extracted_entities, request_id, *,
                tuple_delim=None, record_delim=None, completion_delim=None, max_retries=2):
        self.log.record("relationship_extractor.extract",
                        prompt_var_keys=sorted(prompt_vars),
                        extracted_entity_names=[getattr(e, "entity_name", str(e))
                                                for e in extracted_entities],
                        tuple_delim=tuple_delim, record_delim=record_delim,
                        completion_delim=completion_delim)
        return True, list(EXTRACTED_RELATIONSHIPS)


class FakeSyncer(_Logged):
    """Resolves extracted names to ids and writes them to graph, VDB and cache."""

    def sync(self, result: ExtractionResult, provenance, request_id, *,
             entity_sim_topk=None, entity_sim_threshold=None, **kwargs):
        self.log.record("syncer.sync",
                        entity_names=[e.entity_name for e in result.entities],
                        relationship_pairs=[(r.source_entity, r.target_entity)
                                            for r in result.relationships],
                        provenance_keys=sorted(provenance or {}),
                        entity_sim_topk=entity_sim_topk,
                        entity_sim_threshold=entity_sim_threshold)
        return {
            # entity_idx maps a name to the entity's *metadata*, not its id --
            # _log_ingest_delta reads meta["name"] out of the values.
            "entity_idx": {
                e.entity_name: {"id": f"e{i}", "name": e.entity_name,
                                "entity_type": e.entity_type.value}
                for i, e in enumerate(result.entities, 1)
            },
            "entity_summary": {"added": len(result.entities), "updated": 0},
            "relationship_metas": [
                {"rel_id": f"r{i}", "source_id": f"e{i}", "target_id": f"e{i + 1}"}
                for i, _ in enumerate(result.relationships, 1)
            ],
        }


class FakeSummariesVDB(_Logged):
    def add_summary(self, session_id, message_id, summary_text,
                    dialogue_datetime=None, raw_text=None):
        self.log.record("summaries_vdb.add_summary", session_id=session_id,
                        message_id=message_id, dialogue_datetime=dialogue_datetime,
                        has_raw_text=raw_text is not None)
        return f"{session_id}_{message_id}"

    def add_split_turns(self, session_id, message_id, user_text,
                        assistant_summary, dialogue_datetime=None):
        self.log.record("summaries_vdb.add_split_turns", session_id=session_id,
                        message_id=message_id, dialogue_datetime=dialogue_datetime)


class FakeVectorDBManager(_Logged):
    def __init__(self, log: CallLog) -> None:
        super().__init__(log)
        self.summaries_vdb = FakeSummariesVDB(log)

    def get_summaries_vdb(self, dim=None):
        self.log.record("vector_db_manager.get_summaries_vdb", dim=dim)
        return self.summaries_vdb
