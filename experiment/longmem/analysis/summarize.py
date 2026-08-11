from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.longmem.helpers.analysis_cases import DEFAULT_ANALYSIS_ROOT, analysis_dir_for, scenario_alias
from experiment.longmem.helpers.analysis_summary import summarize_cases


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise LongMem error analysis cases into CSV")
    parser.add_argument("--run-tag", default=None, help="LongMem run tag")
    parser.add_argument("--type", default=None, help="LongMem category")
    parser.add_argument("--scenario", default=None, help="Backward-compatible alias for --type")
    parser.add_argument("--analysis-root", default=None, help="Base analysis root")
    parser.add_argument("--input", default=None, help="Directory containing case JSON files")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args(argv)

    type_name = scenario_alias(args.type or args.scenario or "")
    if args.input is None and not type_name:
        parser.error("--type/--scenario is required unless --input is provided")

    run_tag = args.run_tag or "default"
    analysis_root = Path(args.analysis_root) if args.analysis_root else DEFAULT_ANALYSIS_ROOT
    analysis_dir = analysis_dir_for(run_tag, type_name, analysis_root) if type_name else None
    input_dir = Path(args.input) if args.input else analysis_dir / "cases"
    output_path = Path(args.output) if args.output else analysis_dir / "summary.csv"
    summarize_cases(input_dir, output_path)


if __name__ == "__main__":
    main()
