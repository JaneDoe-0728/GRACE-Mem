import argparse
import re
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Sequence

MODULE_DIR = Path(__file__).resolve().parent
LOCOMO_ROOT = MODULE_DIR.parent
EXPERIMENT_ROOT = LOCOMO_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
for _path in (EXPERIMENT_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from locomo.aggregate import compute_summary_from_rows
from locomo.utils.io import load_csv_rows, load_json_object


LONG_TEXT_COLUMNS = {
    "question",
    "gold_answer",
    "gold_evidence_source",
    "model_answer",
    "retrieved_context",
    "rendered_evidence",
    "source",
    "note",
}
PRECISE_VALUE_COLUMNS = {
    "correctness",
    "correctness_percent",
    "f1",
    "bleu1",
    "macro_avg_by_category",
    "macro_avg_by_category_percent",
    "coverage_percent",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _noco_uploader_root() -> Path:
    return _repo_root() / "noco-db-uploader"


def _ensure_noco_import_path() -> None:
    uploader_root = _noco_uploader_root()
    if str(uploader_root) not in sys.path:
        sys.path.insert(0, str(uploader_root))


def _load_noco_env() -> None:
    from dotenv import load_dotenv

    repo_root = _repo_root()
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root / "noco-db-uploader" / ".env")
    os.environ.setdefault(
        "NOCO_TARGETS_PATH",
        str(repo_root / "experiment" / "noco" / "noco_targets.yaml"),
    )


def _load_noco_modules():
    _ensure_noco_import_path()
    config_mod = import_module("config")
    noco_client_mod = import_module("src.noco_client")
    return config_mod.NocoDBConfig, noco_client_mod.NocoDBClient


def _infer_uidt(column: str, value: Any) -> str:
    if column in LONG_TEXT_COLUMNS:
        return "LongText"
    if column in PRECISE_VALUE_COLUMNS:
        return "SingleLineText"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "Number"
    return "SingleLineText"


def _sanitize_run_tag(run_tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", run_tag.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "run"


def _is_adversarial_category(label: Any) -> bool:
    return str(label).strip().lower() == "adversarial"


def _row_is_adversarial(row: Dict[str, Any]) -> bool:
    for key in ("category_label", "category"):
        value = row.get(key)
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"adversarial", "5"}:
            return True
    return False


def _build_judge_rows(
    *,
    merged_judge_csv: Path,
    dataset: str,
    run_tag: str,
    include_adversarial: bool,
) -> List[Dict[str, Any]]:
    if not merged_judge_csv.exists():
        raise FileNotFoundError(f"Merged judge CSV not found: {merged_judge_csv}")

    rows: List[Dict[str, Any]] = []
    row_index = 0
    for row in load_csv_rows(merged_judge_csv):
        if not include_adversarial and _row_is_adversarial(row):
            continue
        row_index += 1
        out = {"run_tag": run_tag, "dataset": dataset, "row_index": row_index}
        out.update(row)
        rows.append(out)
    return rows


def _load_judge_csv_rows(merged_judge_csv: Path) -> List[Dict[str, Any]]:
    if not merged_judge_csv.exists():
        raise FileNotFoundError(f"Merged judge CSV not found: {merged_judge_csv}")
    return load_csv_rows(merged_judge_csv)


def _build_summary_overall_rows(
    *,
    aggregate_payload: Dict[str, Any],
    judge_rows: Sequence[Dict[str, Any]],
    dataset: str,
    run_tag: str,
    include_adversarial: bool,
) -> List[Dict[str, Any]]:
    aggregate_overall = aggregate_payload.get("overall", {})
    computed = compute_summary_from_rows(
        judge_rows,
        exclude_adversarial=not include_adversarial,
    )
    overall = computed["overall"]
    category_rows = list(computed["by_category"].items())

    rows: List[Dict[str, Any]] = [
        {
            "run_tag": run_tag,
            "dataset": dataset,
            "scope": "overall",
            "category": "overall",
            "correctness": overall.get("avg_correctness"),
            "correctness_percent": overall.get("avg_correctness_percent"),
            "f1": overall.get("avg_f1"),
            "bleu1": overall.get("avg_bleu1"),
            "count": overall.get("count"),
            "count_f1": overall.get("count_f1"),
            "count_bleu1": overall.get("count_bleu1"),
            "macro_avg_by_category": overall.get("macro_avg_by_category"),
            "macro_avg_by_category_percent": overall.get("macro_avg_by_category_percent"),
            "exclude_adversarial": overall.get("exclude_adversarial"),
            "note": aggregate_overall.get("note", ""),
        }
    ]

    for category, stats in category_rows:
        if not include_adversarial and _is_adversarial_category(category):
            continue
        rows.append(
            {
                "run_tag": run_tag,
                "dataset": dataset,
                "scope": "category",
                "category": category,
                "correctness": stats.get("avg_correctness"),
                "correctness_percent": stats.get("avg_correctness_percent"),
                "f1": stats.get("avg_f1"),
                "bleu1": stats.get("avg_bleu1"),
                "count": stats.get("count"),
                "count_f1": stats.get("count_f1"),
                "count_bleu1": stats.get("count_bleu1"),
                "macro_avg_by_category": "",
                "macro_avg_by_category_percent": "",
                "exclude_adversarial": not include_adversarial,
                "note": "",
            }
        )
    return rows


def _build_summary_sample_rows(
    *,
    aggregate_payload: Dict[str, Any],
    judge_rows: Sequence[Dict[str, Any]],
    dataset: str,
    run_tag: str,
    include_adversarial: bool,
) -> List[Dict[str, Any]]:
    per_sample = aggregate_payload.get("per_sample", {})
    rows_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for row in judge_rows:
        if not include_adversarial and _row_is_adversarial(row):
            continue
        sample_name = str(row.get("sample", "")).strip()
        if not sample_name:
            continue
        rows_by_sample.setdefault(sample_name, []).append(row)

    rows: List[Dict[str, Any]] = []
    for sample_name in sorted(set(per_sample.keys()) | set(rows_by_sample.keys())):
        source_stats = per_sample.get(sample_name, {})
        sample_rows = rows_by_sample.get(sample_name, [])
        computed = compute_summary_from_rows(
            sample_rows,
            exclude_adversarial=not include_adversarial,
        )["overall"]
        if not sample_rows and not include_adversarial:
            continue
        rows.append(
            {
                "run_tag": run_tag,
                "dataset": dataset,
                "sample": sample_name,
                "correctness": computed.get("avg_correctness"),
                "correctness_percent": computed.get("avg_correctness_percent"),
                "f1": computed.get("avg_f1"),
                "bleu1": computed.get("avg_bleu1"),
                "source": source_stats.get("source", ""),
            }
        )
    return rows


def upload_run_tables(
    *,
    aggregate_json: Path,
    dataset: str,
    run_root: Path,
    sample_ids: Sequence[int],
    no_judge: bool,
    include_adversarial: bool,
) -> Dict[str, Dict[str, Any]]:
    del sample_ids, no_judge

    _load_noco_env()
    NocoDBConfig, NocoDBClient = _load_noco_modules()

    config = NocoDBConfig.from_dataset(dataset)
    if not config.api_token:
        raise EnvironmentError("API_TOKEN is required for NocoDB upload")
    if not config.project_id:
        raise EnvironmentError("PROJECT_ID is required for NocoDB upload")

    print(
        f"[UPLOAD] target dataset={dataset} noco_url={config.noco_url} "
        f"project_id={config.project_id} source_id={config.source_id or 'auto'}"
    )
    client = NocoDBClient(config.noco_url, config.api_token)
    aggregate_payload = load_json_object(aggregate_json)
    run_tag = _sanitize_run_tag(run_root.name)
    merged_judge_csv = run_root / "_judge_merged.csv"
    judge_csv_rows = _load_judge_csv_rows(merged_judge_csv)

    judge_rows = _build_judge_rows(
        merged_judge_csv=merged_judge_csv,
        dataset=dataset,
        run_tag=run_tag,
        include_adversarial=include_adversarial,
    )
    summary_overall_rows = _build_summary_overall_rows(
        aggregate_payload=aggregate_payload,
        judge_rows=judge_csv_rows,
        dataset=dataset,
        run_tag=run_tag,
        include_adversarial=include_adversarial,
    )
    summary_sample_rows = _build_summary_sample_rows(
        aggregate_payload=aggregate_payload,
        judge_rows=judge_csv_rows,
        dataset=dataset,
        run_tag=run_tag,
        include_adversarial=include_adversarial,
    )
    replace_kwargs = dict(
        project_id=config.project_id,
        org=config.org,
        source_id=config.source_id,
        uidt_fn=_infer_uidt,
        str_columns=PRECISE_VALUE_COLUMNS,
    )
    return {
        "judge": client.replace_table_rows(
            **replace_kwargs,
            table_name=f"{run_tag}_judge",
            rows=judge_rows,
        ),
        "summary_overall": client.replace_table_rows(
            **replace_kwargs,
            table_name=f"{run_tag}_summary_overall",
            rows=summary_overall_rows,
        ),
        "summary_sample": client.replace_table_rows(
            **replace_kwargs,
            table_name=f"{run_tag}_summary_sample",
            rows=summary_sample_rows,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload existing LoCoMo aggregate artifacts to NocoDB")
    parser.add_argument("--dataset", choices=["locomo", "locomo-plus"], required=True)
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run directory containing _correctness_aggregate.json and _judge_merged.csv",
    )
    parser.add_argument(
        "--adv",
        action="store_true",
        help="Include adversarial rows in uploaded summary metrics; default upload excludes them",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise SystemExit(f"Run root not found: {run_root}")

    aggregate_json = run_root / "_correctness_aggregate.json"
    if not aggregate_json.exists():
        raise SystemExit(f"Aggregate JSON not found: {aggregate_json}")

    result = upload_run_tables(
        aggregate_json=aggregate_json,
        dataset=args.dataset,
        run_root=run_root,
        sample_ids=[],
        no_judge=False,
        include_adversarial=args.adv,
    )
    for label, table_result in result.items():
        print(
            f"[UPLOAD] {table_result['status']} {label} table to "
            f"{table_result['table']} (table_id={table_result['table_id']}, rows={table_result['row_count']})"
        )


if __name__ == "__main__":
    main()
