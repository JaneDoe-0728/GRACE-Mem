from __future__ import annotations

import os
from pathlib import Path

from experiment.longmem.utils.io import ensure_dir, read_csv_frame, read_jsonl_file, write_json_file


LONGMEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT_DATA_ROOT = LONGMEM_ROOT / "script_data"
DEFAULT_OUTPUT_ROOT = LONGMEM_ROOT / "output"
DEFAULT_ANALYSIS_ROOT = LONGMEM_ROOT / "error_analysis"


def scenario_alias(name: str) -> str:
    mapping = {
        "temporal": "temporal_reasoning",
        "temporal_reasoning": "temporal_reasoning",
        "multi_session": "multi_session",
        "single_session_user": "single_session_user",
        "single_session_assistant": "single_session_assistant",
        "single_session_preference": "single_session_preference",
        "knowledge_update": "knowledge_update",
    }
    return mapping.get(name, name)


def data_folder_for(type_name: str, base: Path | None = None) -> Path:
    root = base or DEFAULT_SCRIPT_DATA_ROOT
    return root / scenario_alias(type_name)


def output_dir_for(run_tag: str, type_name: str, base: Path | None = None) -> Path:
    root = base or DEFAULT_OUTPUT_ROOT
    return root / run_tag / scenario_alias(type_name)


def analysis_dir_for(run_tag: str, type_name: str, base: Path | None = None) -> Path:
    root = base or DEFAULT_ANALYSIS_ROOT
    return root / run_tag / scenario_alias(type_name)


def load_jsonl(path: Path) -> list[dict]:
    return read_jsonl_file(path)


def events_by_type(records: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for record in records:
        event = record.get("event", "")
        result.setdefault(event, []).append(record)
    return result


def step3_has_answer(data_folder: Path, dataset_id: str) -> dict:
    csv_path = data_folder / f"{dataset_id}.csv"
    if not csv_path.exists():
        return {"turns": [], "summary_ids": []}

    df = read_csv_frame(csv_path, encoding="utf-8-sig")
    if "has_answer" not in df.columns:
        return {"turns": [], "summary_ids": []}

    df["has_answer"] = df["has_answer"].map(lambda value: str(value).strip().lower() == "true")
    has_answer = df[df["has_answer"]].copy()

    turns = []
    for _, row in has_answer.iterrows():
        turn_index = int(row["turn_index"])
        message_id = turn_index + 1 if turn_index % 2 == 1 else turn_index
        session_id = str(row["session_id"])
        summary_id = f"{session_id}:{message_id}"
        turns.append(
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "role": str(row.get("role", "")),
                "content": str(row.get("content", "")),
                "message_id": message_id,
                "summary_id": summary_id,
            }
        )
    summary_ids = list({turn["summary_id"] for turn in turns})
    return {"turns": turns, "summary_ids": summary_ids}


def step2_ingest(log_dir: Path, target_turns: list[dict]) -> dict:
    path = log_dir / "kg_ingestor.jsonl"
    records = load_jsonl(path)
    if not records:
        return {"status": "no_log", "fail_events": [], "checked_request_ids": []}

    target_keys = {(turn["session_id"], turn["message_id"]) for turn in target_turns}
    request_ids: set[str] = set()
    for record in records:
        if record.get("event") == "summarize_and_ingest_turn_start":
            key = (record.get("session_id", ""), record.get("message_id"))
            if key in target_keys:
                request_ids.add(record["request_id"])

    fail_events = []
    for record in records:
        if record.get("request_id") not in request_ids:
            continue
        event = record.get("event", "")
        if any(token in event.lower() for token in ("fail", "error", "exception")):
            fail_events.append(event)
        elif record.get("success") is False:
            fail_events.append(event)

    return {
        "status": "fail" if fail_events else "pass",
        "fail_events": list(set(fail_events)),
        "checked_request_ids": list(request_ids),
    }


def step4_summaries(artifacts_dir: Path, target_summary_ids: list[str]) -> dict:
    records = load_jsonl(artifacts_dir / "summaries_meta.jsonl")
    target_set = set(target_summary_ids)
    summaries = [
        {"summary_id": record["summary_id"], "summary_text": record.get("summary_text", "")}
        for record in records
        if record.get("summary_id") in target_set
    ]
    return {"summaries": summaries, "count": len(summaries)}


def step5_entities(artifacts_dir: Path, target_summary_ids: list[str]) -> dict:
    records = load_jsonl(artifacts_dir / "entities_meta.jsonl")
    target_set = set(target_summary_ids)
    entity_map: dict[str, dict] = {}

    for record in records:
        prov_events = record.get("prov", {}).get("events", [])
        matched = any(event.get("summary_id") in target_set for event in prov_events)
        if not matched:
            continue
        entity_id = record.get("id", "")
        latest_ts = max((event.get("ts", 0) for event in prov_events), default=0)
        existing = entity_map.get(entity_id)
        if existing is None or latest_ts >= existing["_ts"]:
            entity_map[entity_id] = {
                "_ts": latest_ts,
                "type": record.get("type", ""),
                "id": entity_id,
                "name": record.get("name", ""),
                "description": record.get("description", ""),
            }

    all_desc: dict[str, list[str]] = {}
    for record in records:
        entity_id = record.get("id", "")
        if entity_id in entity_map:
            all_desc.setdefault(entity_id, []).append(record.get("description", ""))

    entities = []
    for entity_id, info in entity_map.items():
        unique_descs = list(dict.fromkeys(all_desc.get(entity_id, [])))
        new_des = unique_descs[-1] if len(unique_descs) > 1 else None
        entities.append(
            {
                "type": info["type"],
                "id": entity_id,
                "name": info["name"],
                "description": info["description"],
                "new_des": new_des,
            }
        )

    return {
        "entities": entities,
        "count": len(entities),
        "updated_count": sum(1 for entity in entities if entity["new_des"] is not None),
        "entity_ids": [entity["id"] for entity in entities],
    }


def step6_relationships(artifacts_dir: Path, target_summary_ids: list[str]) -> dict:
    records = load_jsonl(artifacts_dir / "relationships_meta.jsonl")
    target_set = set(target_summary_ids)
    seen_ids: set[str] = set()
    relationships = []
    for record in records:
        prov_events = record.get("prov", {}).get("events", [])
        matched = any(event.get("summary_id") in target_set for event in prov_events)
        if not matched:
            continue
        relationship_id = record.get("id", "")
        if relationship_id in seen_ids:
            continue
        seen_ids.add(relationship_id)
        relationships.append(
            {
                "id": relationship_id,
                "description": record.get("description", ""),
                "source_entity": record.get("source_entity", ""),
                "target_entity": record.get("target_entity", ""),
            }
        )
    return {"relationships": relationships, "count": len(relationships), "rel_ids": [row["id"] for row in relationships]}


def step7_search(log_dir: Path, target_entity_ids: list[str], target_rel_ids: list[str]) -> dict:
    records = load_jsonl(log_dir / "kg_retrieval_search.jsonl")
    grouped = events_by_type(records)
    target_entities = set(target_entity_ids)
    target_rels = set(target_rel_ids)

    vec_done = grouped.get("vector_search_done", [{}])[-1]
    rel_vec_done = grouped.get("rel_vector_search_done", [{}])[-1]
    bm25_all = grouped.get("bm25_all_done", [{}])[-1]

    entity_vector_sample = vec_done.get("sample_hits", [])
    rel_vector_sample = rel_vec_done.get("sample_any", [])
    bm25_sample: list[dict] = []
    for record in grouped.get("bm25_keyword_done", []):
        bm25_sample.extend(record.get("sample_hits", []))

    return {
        "entity_vector_hit_count": vec_done.get("hit_count", 0),
        "entity_vector_threshold": vec_done.get("threshold", None),
        "entity_vector_sample": entity_vector_sample,
        "rel_vector_sample": rel_vector_sample,
        "bm25_total_hit_pairs": bm25_all.get("total_hit_pairs", 0),
        "bm25_sample": bm25_sample,
        "target_entity_in_vector_sample": any(hit.get("id") in target_entities for hit in entity_vector_sample),
        "target_entity_in_bm25_sample": any(hit.get("id") in target_entities for hit in bm25_sample),
        "target_rel_in_vector_sample": any(hit.get("id") in target_rels for hit in rel_vector_sample),
    }


def step8_filtering(log_dir: Path, target_entity_ids: list[str], target_rel_ids: list[str]) -> dict:
    records = load_jsonl(log_dir / "kg_retrieval_filtering.jsonl")
    grouped = events_by_type(records)
    target_entities = set(target_entity_ids)
    target_rels = set(target_rel_ids)

    entity_results = [
        {
            "id": record["entity_id"],
            "passed": record.get("passed", None),
            "threshold": record.get("filter_entity_threshold", None),
        }
        for record in grouped.get("entity_compare_by_id", [])
        if record.get("entity_id") in target_entities
    ]
    rel_results = [
        {
            "id": record["relationship_id"],
            "passed": record.get("passed", None),
            "threshold": record.get("filter_relationship_threshold", None),
        }
        for record in grouped.get("relationship_compare_by_id", [])
        if record.get("relationship_id") in target_rels
    ]

    intersect = grouped.get("intersection_done", [{}])[-1]
    filtered = grouped.get("intersection_filtered", [{}])[-1]
    ent_reranker = grouped.get("reranker_entities_done", [{}])[-1]
    rel_reranker = grouped.get("reranker_relationships_done", [{}])[-1]
    complete = grouped.get("reranker_complete", [{}])[-1]

    return {
        "target_entity_results": entity_results,
        "target_rel_results": rel_results,
        "target_entity_filtered_out": any(not row["passed"] for row in entity_results if row["passed"] is not None),
        "target_rel_filtered_out": any(not row["passed"] for row in rel_results if row["passed"] is not None),
        "intersect_entities": intersect.get("intersect_entities", None),
        "intersect_rels": intersect.get("intersect_rels", None),
        "after_filter_entities": filtered.get("filtered_entity_count", None),
        "after_filter_rels": filtered.get("filtered_relationship_count", None),
        "reranker_entity_recovered": ent_reranker.get("recovered_count", 0),
        "reranker_rel_recovered": rel_reranker.get("recovered_count", 0),
        "sample_entity_recovered": ent_reranker.get("sample_recovered", []),
        "sample_rel_recovered": rel_reranker.get("sample_recovered", []),
        "final_entity_count": complete.get("final_entity_count", None),
        "final_rel_count": complete.get("final_relationship_count", None),
    }


def step9_evidence(log_dir: Path, target_entity_ids: list[str], target_rel_ids: list[str]) -> dict:
    records = load_jsonl(log_dir / "kg_retrieval_evidence.jsonl")
    grouped = events_by_type(records)

    iter_entity_ids = [record["entity_id"] for record in grouped.get("evidence_iter_entity", [])]
    iter_rel_ids = [record["relationship_id"] for record in grouped.get("evidence_iter_relationship", [])]
    target_entities = set(target_entity_ids)
    target_rels = set(target_rel_ids)
    found_entities = [entity_id for entity_id in iter_entity_ids if entity_id in target_entities]
    found_rels = [relationship_id for relationship_id in iter_rel_ids if relationship_id in target_rels]
    complete = grouped.get("build_evidence_complete", [])
    final_count = complete[-1].get("total_snippets", 0) if complete else 0

    return {
        "iter_entity_ids": iter_entity_ids,
        "iter_rel_ids": iter_rel_ids,
        "target_entity_ids": target_entity_ids,
        "target_rel_ids": target_rel_ids,
        "target_entity_found": found_entities,
        "target_rel_found": found_rels,
        "target_entity_in_evidence": len(found_entities) > 0,
        "target_rel_in_evidence": len(found_rels) > 0,
        "final_count": final_count,
    }


CUT_ITEMS_JUDGE_SYSTEM = (
    "You are an evaluation assistant. "
    "Given a list of knowledge graph entities/relationships that were cut from retrieval, "
    "and a question with its gold answer, determine whether any of the cut items contain "
    "information needed to answer the question.\n"
    "Reply with exactly one word: yes / no"
)


def llm_judge_cut_items(llm, question: str, gold_answer: str, cut_entities: list[str], cut_rels: list[str], entity_meta: dict, rel_meta: dict) -> str:
    lines = []
    for entity_id in cut_entities:
        meta = entity_meta.get(entity_id, {})
        lines.append(f"[Entity] {meta.get('name', entity_id)} ({meta.get('type', '')}): {meta.get('description', '')}")
    for rel_id in cut_rels:
        meta = rel_meta.get(rel_id, {})
        lines.append(
            f"[Relationship] {meta.get('source_entity', '')} -> {meta.get('target_entity', '')}: "
            f"{meta.get('description', '')}"
        )
    if not lines:
        return "no_cut"

    messages = [
        {"role": "system", "content": CUT_ITEMS_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Cut items:\n{os.linesep.join(lines)}\n\n"
                f"Question: {question}\n"
                f"Gold answer: {gold_answer}\n\n"
                "Do any cut items contain information needed to answer the question? Reply: yes / no"
            ),
        },
    ]
    response = llm.chat(messages=messages, temperature=0.0, max_tokens=10)
    text = (response.choices[0].message.content or "").strip().lower()
    return "yes" if "yes" in text else "no"


def build_error_analysis_llm():
    from dotenv import load_dotenv
    from openai import OpenAI

    repo_root = LONGMEM_ROOT.parent.parent
    load_dotenv(repo_root / ".env")
    api = os.getenv("ERROR_ANALYSIS_LLM_API")
    model = os.getenv("ERROR_ANALYSIS_MODEL_NAME")
    if not api or not model:
        raise ValueError("ERROR_ANALYSIS_LLM_API / ERROR_ANALYSIS_MODEL_NAME not set in .env")

    class _SimpleClient:
        def __init__(self, base_url: str, model_name: str):
            self._client = OpenAI(base_url=base_url, api_key="lm-studio")
            self.model = model_name

        def chat(self, messages, temperature=0.0, max_tokens=10):
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    return _SimpleClient(api, model)


def analyze_one(
    *,
    scenario: str,
    dataset_id: str,
    question: str,
    gold_answer: str,
    generated_answer: str,
    data_folder: Path,
    output_dir: Path,
    llm=None,
) -> dict:
    log_dir = output_dir / f"logs_{dataset_id}"
    artifacts_dir = output_dir / f"artifacts_{dataset_id}"

    s3 = step3_has_answer(data_folder, dataset_id)
    s2 = step2_ingest(log_dir, target_turns=s3["turns"])
    s4 = step4_summaries(artifacts_dir, target_summary_ids=s3["summary_ids"])
    s5 = step5_entities(artifacts_dir, target_summary_ids=s3["summary_ids"])
    s6 = step6_relationships(artifacts_dir, target_summary_ids=s3["summary_ids"])
    s7 = step7_search(log_dir, s5["entity_ids"], s6["rel_ids"])
    s8 = step8_filtering(log_dir, s5["entity_ids"], s6["rel_ids"])
    s9 = step9_evidence(log_dir, s5["entity_ids"], s6["rel_ids"])

    passed_thresh_ent = {row["id"] for row in s8.get("target_entity_results", []) if row.get("passed") is True}
    passed_thresh_rel = {row["id"] for row in s8.get("target_rel_results", []) if row.get("passed") is True}
    evidence_ent = set(s9.get("iter_entity_ids", []))
    evidence_rel = set(s9.get("iter_rel_ids", []))
    s8["target_entity_topk_cut"] = list(passed_thresh_ent - evidence_ent)
    s8["target_rel_topk_cut"] = list(passed_thresh_rel - evidence_rel)

    threshold_cut_ent = [row["id"] for row in s8.get("target_entity_results", []) if not row.get("passed")]
    threshold_cut_rel = [row["id"] for row in s8.get("target_rel_results", []) if not row.get("passed")]
    all_cut_ent = list(set(threshold_cut_ent) | set(s8["target_entity_topk_cut"]))
    all_cut_rel = list(set(threshold_cut_rel) | set(s8["target_rel_topk_cut"]))
    entity_meta = {entity["id"]: entity for entity in s5.get("entities", [])}
    rel_meta = {rel["id"]: rel for rel in s6.get("relationships", [])}

    if not (all_cut_ent or all_cut_rel):
        cut_judgment = "no_cut"
    elif llm is None:
        cut_judgment = "skipped"
    else:
        cut_judgment = llm_judge_cut_items(llm, question, gold_answer, all_cut_ent, all_cut_rel, entity_meta, rel_meta)
    s8["cut_contains_answer"] = cut_judgment

    return {
        "meta": {
            "dataset_id": dataset_id,
            "scenario": scenario,
            "question": question,
            "gold_answer": gold_answer,
            "generated_answer": generated_answer,
        },
        "step2_ingest": s2,
        "step3_turns": s3["turns"],
        "step3_summary_ids": s3["summary_ids"],
        "step4_summaries": s4["summaries"],
        "step5_entities": s5["entities"],
        "step6_relationships": s6["relationships"],
        "step7_search": s7,
        "step8_filtering": s8,
        "step9_evidence": s9,
    }


def collect_cases(
    *,
    output_dir: Path,
    data_folder: Path,
    analysis_dir: Path,
    scenario: str,
    dataset_id: str | None = None,
    no_llm: bool = False,
    no_overwrite: bool = False,
) -> int:
    cases_dir = analysis_dir / "cases"
    ensure_dir(cases_dir)

    llm = None
    if not no_llm:
        try:
            llm = build_error_analysis_llm()
            print(f"Error analysis LLM: {llm.model}")
        except Exception as exc:
            print(f"[WARN] Error analysis LLM init failed: {exc}. Step 8 cut judgment will be skipped.")

    progress_path = output_dir / "progress.csv"
    if not progress_path.exists():
        raise FileNotFoundError(f"progress.csv not found: {progress_path}")

    df = read_csv_frame(progress_path)
    failed = df[df["correctness"] == 0].copy()
    if dataset_id:
        failed = failed[failed["dataset"] == dataset_id]

    print(f"Scenario: {scenario}")
    print(f"Failed questions to analyse: {len(failed)}")

    written = 0
    for index, (_, row) in enumerate(failed.iterrows(), 1):
        current_id = str(row["dataset"])
        case_path = cases_dir / f"{current_id}.json"
        if case_path.exists() and no_overwrite:
            print(f"[{index}/{len(failed)}] {current_id} — skip (already exists)")
            continue

        print(f"[{index}/{len(failed)}] {current_id}")
        result = analyze_one(
            scenario=scenario,
            dataset_id=current_id,
            question=str(row.get("question", "")),
            gold_answer=str(row.get("gold_answer", "")),
            generated_answer=str(row.get("generated_answer", "")),
            data_folder=data_folder,
            output_dir=output_dir,
            llm=llm,
        )
        write_json_file(case_path, result)
        print(f"  → saved: {case_path}")
        written += 1

    print("\nDone. Run experiment/longmem/analysis/summarize.py to produce summary.csv.")
    return written
