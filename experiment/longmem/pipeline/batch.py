"""
Practical Example: Process Multiple Datasets
============================================
This script automatically processes all CSV files in a folder with the same configuration.

Usage:
    python -m experiment.longmem.pipeline.batch
    python -m experiment.longmem.pipeline.batch --stage qa_eval judge
"""

import argparse
from pathlib import Path
import os
import sys

from experiment.longmem.pipeline.processor import MultiDatasetProcessor
from experiment.longmem.helpers.args import add_child_args, add_data_args, add_run_args, resolve_stages
from experiment.longmem.helpers.datasets import discover_csv_datasets, resolve_child_datasets, select_datasets
from experiment.longmem.models import DatasetConfig
from experiment.common.run_metadata import namespace_to_dict, write_run_metadata
from experiment.longmem.utils.io import write_json_file
from experiment.experiment_config import INGEST_PARAMS, RETRIEVAL_PARAMS


def _parse_type_filter(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def _cli_flag_provided(raw_argv: list[str], option: str) -> bool:
    return option in raw_argv or any(arg.startswith(f"{option}=") for arg in raw_argv)


def _resolve_arg(
    *,
    raw_argv: list[str],
    option: str,
    cli_value,
    env_name: str,
    default,
):
    if _cli_flag_provided(raw_argv, option):
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    return cli_value if cli_value not in (None, "") else default


def discover_dataset_configs(
    *,
    folder_path: str,
    file_pattern: str,
    ingest_params: dict,
    retrieval_params: dict,
) -> list[DatasetConfig]:
    datasets: list[DatasetConfig] = []
    for csv_file in discover_csv_datasets(folder_path, file_pattern):
        datasets.append(
            DatasetConfig.from_params(
                name=Path(csv_file).stem,
                csv_path=str(csv_file),
                ingest_params=ingest_params,
                retrieval_params=retrieval_params,
            )
        )
    return datasets


def _write_run_metadata(
    metadata_path: Path,
    *,
    args,
    run_tag: str | None,
    data_root: Path,
    output_root: Path,
    selected_stages: list[str],
    child_mode: bool,
    child_file: Path,
    dataset_selector: str | None,
    run_targets: list[tuple[str, list[Path]]],
) -> None:
    target_rows: list[dict[str, object]] = []
    subfolders = sorted([p for p in data_root.iterdir() if p.is_dir()]) if data_root.exists() else []
    has_subfolders = len(subfolders) > 0

    for target_name, csv_paths in run_targets:
        if child_mode:
            target_output_dir = output_root / target_name
        else:
            target_output_dir = output_root / target_name if has_subfolders else output_root
        target_rows.append(
            {
                "target_name": target_name,
                "output_dir": str(target_output_dir.resolve()),
                "dataset_count": len(csv_paths),
                "datasets": [csv_path.stem for csv_path in csv_paths],
            }
        )

    write_run_metadata(
        metadata_path,
        {
            "entrypoint": "longmem.run_batch",
            "run_tag": run_tag,
            "run_root": str(output_root.resolve()),
            "data_root": str(data_root.resolve()),
            "child_mode": child_mode,
            "child_file": str(child_file.resolve()),
            "dataset_selector": dataset_selector,
            "stages": list(selected_stages),
            "cli": {
                "argv": list(getattr(args, "raw_argv", [])),
                "resolved_args": namespace_to_dict(args),
                "resolved_config": {
                    "output_root": str(output_root.resolve()),
                    "data_folder": str(data_root.resolve()),
                    "file_pattern": args.file_pattern,
                    "child": child_mode,
                    "child_file": str(child_file.resolve()),
                    "type": args.type,
                    "run_tag": run_tag,
                    "run_judge": not args.no_judge,
                    "stages": list(selected_stages),
                    "dataset_id": dataset_selector,
                    "num": args.num,
                },
            },
            "targets": target_rows,
        },
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run LongMem batch processing")
    add_data_args(parser)
    add_child_args(parser)
    add_run_args(parser)
    args = parser.parse_args(argv)
    args.raw_argv = list(argv) if argv is not None else sys.argv[1:]

    # Resolve values: CLI args take precedence, env vars are fallback (for watchdog compat)
    data_folder = _resolve_arg(
        raw_argv=args.raw_argv,
        option="--data-folder",
        cli_value=args.data_folder,
        env_name="MDQA_DATA_FOLDER",
        default="./experiment/longmem/sample_question/t",
    )
    output_root = _resolve_arg(
        raw_argv=args.raw_argv,
        option="--output-root",
        cli_value=args.output_root,
        env_name="MDQA_OUTPUT_ROOT",
        default="./experiment/longmem/output/default/sample_test",
    )
    file_pattern = _resolve_arg(
        raw_argv=args.raw_argv,
        option="--file-pattern",
        cli_value=args.file_pattern,
        env_name="MDQA_FILE_PATTERN",
        default="*.csv",
    )
    child_mode = args.child or os.environ.get("MDQA_CHILD", "0").strip().lower() in ("1", "true", "yes")
    child_file = _resolve_arg(
        raw_argv=args.raw_argv,
        option="--child-file",
        cli_value=args.child_file,
        env_name="MDQA_CHILD_FILE",
        default="./experiment/longmem/script_data/child_qa.txt",
    )
    child_type_filter = _parse_type_filter(
        args.type if args.type is not None else os.environ.get("MDQA_CHILD_TYPE", "").strip() or None
    )
    run_tag = args.run_tag or os.environ.get("MDQA_RUN_TAG", "").strip()
    run_judge = not args.no_judge and os.environ.get("MDQA_RUN_JUDGE", "1").strip().lower() not in ("0", "false", "no")
    selected_stages = resolve_stages(
        args.stages,
        env_value=os.environ.get("MDQA_STAGES"),
        no_judge=not run_judge,
    )
    dataset_selector = args.dataset_id or os.environ.get("MDQA_DATASET_ID", "").strip() or None
    num = args.num if args.num is not None else (
        int(os.environ["MDQA_NUM_DATASETS"]) if os.environ.get("MDQA_NUM_DATASETS") else None
    )

    # ============================================================
    # Configuration: Shared settings for ALL datasets
    # ============================================================

    # ============================================================
    # Auto-discover datasets from folder (or subfolders)
    # ============================================================

    data_root = Path(data_folder)
    if not data_root.exists():
        raise ValueError(f"Folder not found: {data_folder}")

    if child_mode:
        child_groups = resolve_child_datasets(
            data_root,
            child_file,
            type_name=child_type_filter,
        )
        run_targets = []
        for category, csv_paths in sorted(child_groups.items()):
            resolved_paths = select_datasets(
                csv_paths,
                dataset_selector,
                scope_label=f"category '{category}'",
            )
            run_targets.append((category, resolved_paths))
    else:
        subfolders = sorted([p for p in data_root.iterdir() if p.is_dir()])
        has_subfolders = len(subfolders) > 0
        if has_subfolders and child_type_filter:
            allowed_types = set(child_type_filter)
            subfolders = [p for p in subfolders if p.name in allowed_types]
            if not subfolders:
                raise ValueError(f"--type '{', '.join(child_type_filter)}' not found under {data_root}")
        folders = subfolders if has_subfolders else [data_root]
        run_targets = []
        for folder in folders:
            csv_paths = discover_csv_datasets(str(folder), file_pattern)
            resolved_paths = select_datasets(
                csv_paths,
                dataset_selector,
                scope_label=f"folder '{folder.name}'",
            )
            run_targets.append((folder.name, resolved_paths))

    _write_run_metadata(
        Path(output_root).resolve() / "run_metadata.json",
        args=args,
        run_tag=run_tag or None,
        data_root=data_root,
        output_root=Path(output_root).resolve(),
        selected_stages=selected_stages,
        child_mode=child_mode,
        child_file=Path(child_file),
        dataset_selector=dataset_selector,
        run_targets=run_targets,
    )

    # ============================================================
    # Run Processing (one run per folder)
    # ============================================================

    for target_name, csv_paths in run_targets:
        if child_mode:
            datasets = [
                DatasetConfig.from_params(
                    name=csv_path.stem,
                    csv_path=str(csv_path),
                    ingest_params=INGEST_PARAMS,
                    retrieval_params=RETRIEVAL_PARAMS,
                )
                for csv_path in csv_paths
            ]
            output_dir = Path(output_root) / target_name
        else:
            datasets = [
                DatasetConfig.from_params(
                    name=csv_path.stem,
                    csv_path=str(csv_path),
                    ingest_params=INGEST_PARAMS,
                    retrieval_params=RETRIEVAL_PARAMS,
                )
                for csv_path in csv_paths
            ]
            subfolders = sorted([p for p in data_root.iterdir() if p.is_dir()])
            has_subfolders = len(subfolders) > 0
            output_dir = Path(output_root) / target_name if has_subfolders else Path(output_root)

        if num is not None:
            datasets = datasets[:num]

        with MultiDatasetProcessor(
            base_output_dir=str(output_dir),
        ) as processor:
            results = processor.process_all(
                datasets,
                run_judge=run_judge,
                stages=set(selected_stages),
            )

            # ============================================================
            # Post-Processing: Save Summary
            # ============================================================

            summary_path = processor.base_output_dir / "processing_summary_0204.json"
            write_json_file(summary_path, results)

            print(f"\n{'='*60}")
            print(f"All processing complete!")
            print(f"{'='*60}")
            print(f"Output directory: {processor.base_output_dir}")
            print(f"Summary file: {summary_path}")
            print(f"Stages: {', '.join(selected_stages)}")
            print(f"\nResults:")

            for i, res in enumerate(results, start=1):
                if "error" in res:
                    print(f"  [{i}] ❌ {res['dataset']}: FAILED")
                    print(f"      Error: {res['error']}")
                else:
                    print(f"  [{i}] ✅ {res['dataset']}: SUCCESS")
                    print(f"      Questions answered: {res['num_questions']}")
                    print(f"      Output CSV: {res['output_path']}")
                    print(f"      VDB artifacts: {res['artifacts_dir']}")

            print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
