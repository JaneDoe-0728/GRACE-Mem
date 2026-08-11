from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.longmem.helpers.analysis_cases import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCRIPT_DATA_ROOT,
    analysis_dir_for,
    collect_cases,
    data_folder_for,
    output_dir_for,
    scenario_alias,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LongMem error analysis collector")
    parser.add_argument("--run-tag", default=None, help="LongMem run tag")
    parser.add_argument("--type", default=None, help="LongMem category, e.g. temporal_reasoning")
    parser.add_argument("--scenario", default=None, help="Backward-compatible alias for --type")
    parser.add_argument("--output-root", default=None, help="Override LongMem output dir")
    parser.add_argument("--data-folder", default=None, help="Override LongMem script-data dir")
    parser.add_argument("--analysis-dir", default=None, help="Override analysis output dir")
    parser.add_argument("--output", default=None, help="Backward-compatible alias for analysis dir")
    parser.add_argument("--dataset_id", default=None, help="Process single dataset only")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM judgment in step 8")
    parser.add_argument("--no_overwrite", action="store_true", help="Skip existing case JSON files")
    args = parser.parse_args(argv)

    type_name = scenario_alias(args.type or args.scenario or "")
    if not type_name:
        parser.error("--type/--scenario is required")

    run_tag = args.run_tag or "default"
    output_dir = Path(args.output_root) if args.output_root else output_dir_for(run_tag, type_name, DEFAULT_OUTPUT_ROOT)
    data_folder = Path(args.data_folder) if args.data_folder else data_folder_for(type_name, DEFAULT_SCRIPT_DATA_ROOT)
    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
    elif args.output:
        output_base = Path(args.output)
        analysis_dir = output_base if output_base.name == type_name else output_base / type_name
    else:
        analysis_dir = analysis_dir_for(run_tag, type_name, DEFAULT_ANALYSIS_ROOT)

    collect_cases(
        output_dir=output_dir,
        data_folder=data_folder,
        analysis_dir=analysis_dir,
        scenario=type_name,
        dataset_id=args.dataset_id,
        no_llm=args.no_llm,
        no_overwrite=args.no_overwrite,
    )


if __name__ == "__main__":
    main()
