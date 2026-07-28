from __future__ import annotations

from collections import Counter
from pathlib import Path

from experiment.longmem.utils.io import ensure_dir, glob_sorted, read_json_file, write_csv_dict_rows


def diagnose(row: dict) -> str:
    if row["step2_ingest"] == "fail":
        return "Ingest失敗"

    if row["step3_has_answer_count"] == 0:
        return "無has_answer標記"

    if row["step4_summary_count"] == 0:
        return "Summary未建立"

    if row["step5_entity_count"] == 0 and row["step6_rel_count"] == 0:
        return "Entity和Rel皆未抽出"

    if not row["step7_target_in_vector_sample"] and not row["step7_target_in_bm25_sample"]:
        return "未進候選池（向量和BM25皆未命中）"

    threshold_cut = row["step8_target_entity_filtered_out"] or row["step8_target_rel_filtered_out"]
    topk_cut = row["step8_target_entity_topk_cut"] or row["step8_target_rel_topk_cut"]

    if threshold_cut or topk_cut:
        recovered = (row["step8_reranker_entity_recovered"] or 0) + (row["step8_reranker_rel_recovered"] or 0)
        if threshold_cut and topk_cut:
            cause = "threshold+topk皆有截掉"
        elif threshold_cut:
            cause = "被threshold截掉"
        else:
            cause = "通過threshold但被topk截掉"
        cut_judgment = row.get("step8_cut_contains_answer", "skipped")
        if cut_judgment == "yes":
            cause += "（含答案資訊）"
        elif cut_judgment == "no":
            cause += "（不含答案資訊）"
        if recovered > 0:
            return f"{cause}，Reranker部分救回"
        return f"{cause}，Reranker未救回"

    if not row["step9_target_entity_in_evidence"] and not row["step9_target_rel_in_evidence"]:
        return "目標ID未進入Evidence"

    if row["step9_evidence_final_count"] == 0:
        return "Evidence為空"

    return "資料齊全但生成答案錯誤"


def flatten_case(data: dict) -> dict:
    meta = data.get("meta", {})
    s2 = data.get("step2_ingest", {})
    s3_turns = data.get("step3_turns", [])
    s4_summaries = data.get("step4_summaries", [])
    s5_entities = data.get("step5_entities", [])
    s6_rels = data.get("step6_relationships", [])
    s7 = data.get("step7_search", {})
    s8 = data.get("step8_filtering", {})
    s9 = data.get("step9_evidence", {})

    row = {
        "dataset_id": meta.get("dataset_id", ""),
        "scenario": meta.get("scenario", ""),
        "question": meta.get("question", ""),
        "gold_answer": meta.get("gold_answer", ""),
        "generated_answer": meta.get("generated_answer", ""),
        "step2_ingest": s2.get("status", "no_log"),
        "step2_fail_event": "|".join(s2.get("fail_events", [])) or None,
        "step3_has_answer_count": len(s3_turns),
        "step4_summary_found": len(s4_summaries) > 0,
        "step4_summary_count": len(s4_summaries),
        "step5_entity_count": len(s5_entities),
        "step5_entity_updated_count": sum(1 for entity in s5_entities if entity.get("new_des")),
        "step6_rel_count": len(s6_rels),
        "step7_entity_vector_hit_count": s7.get("entity_vector_hit_count", 0),
        "step7_bm25_total_hit_pairs": s7.get("bm25_total_hit_pairs", 0),
        "step7_target_in_vector_sample": s7.get("target_entity_in_vector_sample", False),
        "step7_target_in_bm25_sample": s7.get("target_entity_in_bm25_sample", False),
        "step7_target_rel_in_vector_sample": s7.get("target_rel_in_vector_sample", False),
        "step8_target_entity_filtered_out": s8.get("target_entity_filtered_out", False),
        "step8_target_rel_filtered_out": s8.get("target_rel_filtered_out", False),
        "step8_target_entity_topk_cut": bool(s8.get("target_entity_topk_cut")),
        "step8_target_entity_topk_cut_ids": "|".join(s8.get("target_entity_topk_cut") or []) or None,
        "step8_target_rel_topk_cut": bool(s8.get("target_rel_topk_cut")),
        "step8_target_rel_topk_cut_ids": "|".join(s8.get("target_rel_topk_cut") or []) or None,
        "step8_cut_contains_answer": s8.get("cut_contains_answer", "skipped"),
        "step8_intersect_entities": s8.get("intersect_entities", None),
        "step8_after_filter_entities": s8.get("after_filter_entities", None),
        "step8_reranker_entity_recovered": s8.get("reranker_entity_recovered", 0),
        "step8_reranker_rel_recovered": s8.get("reranker_rel_recovered", 0),
        "step8_final_entity_count": s8.get("final_entity_count", None),
        "step8_final_rel_count": s8.get("final_rel_count", None),
        "step9_target_entity_in_evidence": s9.get("target_entity_in_evidence", False),
        "step9_target_rel_in_evidence": s9.get("target_rel_in_evidence", False),
        "step9_evidence_final_count": s9.get("final_count", 0),
    }

    row["diagnosis"] = diagnose(row)
    return row


FIELDNAMES = [
    "dataset_id", "scenario", "question", "gold_answer", "generated_answer",
    "step2_ingest", "step2_fail_event",
    "step3_has_answer_count",
    "step4_summary_found", "step4_summary_count",
    "step5_entity_count", "step5_entity_updated_count",
    "step6_rel_count",
    "step7_entity_vector_hit_count", "step7_bm25_total_hit_pairs",
    "step7_target_in_vector_sample", "step7_target_in_bm25_sample", "step7_target_rel_in_vector_sample",
    "step8_target_entity_filtered_out", "step8_target_rel_filtered_out",
    "step8_target_entity_topk_cut", "step8_target_entity_topk_cut_ids",
    "step8_target_rel_topk_cut", "step8_target_rel_topk_cut_ids",
    "step8_cut_contains_answer",
    "step8_intersect_entities", "step8_after_filter_entities",
    "step8_reranker_entity_recovered", "step8_reranker_rel_recovered",
    "step8_final_entity_count", "step8_final_rel_count",
    "step9_target_entity_in_evidence", "step9_target_rel_in_evidence", "step9_evidence_final_count",
    "diagnosis",
]


def summarize_cases(cases_dir: Path, output_path: Path) -> int:
    ensure_dir(output_path.parent)
    case_files = glob_sorted(cases_dir, "*.json")

    rows = []
    for case_file in case_files:
        try:
            data = read_json_file(case_file)
            rows.append(flatten_case(data))
        except Exception as exc:
            print(f"  [WARN] {case_file.name}: {exc}")

    total = len(rows)
    counts = Counter(row["diagnosis"] for row in rows)
    stat_rows = [{}]
    for label, count in counts.most_common():
        pct = count / total * 100 if total > 0 else 0
        stat_rows.append({
            "dataset_id": "【統計】",
            "question": label,
            "gold_answer": count,
            "generated_answer": f"{pct:.1f}%",
        })
    stat_rows.append({
        "dataset_id": "【合計】",
        "gold_answer": total,
        "generated_answer": "100%",
    })

    write_csv_dict_rows(output_path, fieldnames=FIELDNAMES, rows=rows + stat_rows)

    print(f"Written {len(rows)} rows → {output_path}")
    for label, count in counts.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label:<40} {count}  ({pct:.1f}%)")
    return len(rows)
