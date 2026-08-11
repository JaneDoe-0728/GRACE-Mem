"""Download pinned LoCoMo and LongMemEval datasets and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.longmem.tools.convert_dataset import convert_longmem_dataset


LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LONGMEM_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"


@dataclass(frozen=True)
class DatasetFile:
    name: str
    url: str
    revision: str
    sha256: str
    size: int


LOCOMO = DatasetFile(
    name="locomo10.json",
    url=(
        "https://raw.githubusercontent.com/snap-research/locomo/"
        f"{LOCOMO_REVISION}/data/locomo10.json"
    ),
    revision=LOCOMO_REVISION,
    sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    size=2_805_274,
)

LONGMEM_FILES = {
    "s": DatasetFile(
        name="longmemeval_s_cleaned.json",
        url=(
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
            f"{LONGMEM_REVISION}/longmemeval_s_cleaned.json"
        ),
        revision=LONGMEM_REVISION,
        sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        size=277_383_467,
    ),
    "m": DatasetFile(
        name="longmemeval_m_cleaned.json",
        url=(
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
            f"{LONGMEM_REVISION}/longmemeval_m_cleaned.json"
        ),
        revision=LONGMEM_REVISION,
        sha256="9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f",
        size=2_737_100_077,
    ),
    "oracle": DatasetFile(
        name="longmemeval_oracle.json",
        url=(
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
            f"{LONGMEM_REVISION}/longmemeval_oracle.json"
        ),
        revision=LONGMEM_REVISION,
        sha256="821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
        size=15_388_478,
    ),
}


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: DatasetFile) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise ValueError(
            f"Dataset size mismatch for {path}: expected {spec.size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise ValueError(
            f"Dataset checksum mismatch for {path}: expected {spec.sha256}, "
            f"got {actual_sha256}"
        )


def download_file(spec: DatasetFile, destination: Path, *, force: bool = False) -> None:
    destination = destination.resolve()
    if destination.exists() and not force:
        verify_file(destination, spec)
        print(f"Verified existing {spec.name}: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    temp_path.unlink(missing_ok=True)
    request = urllib.request.Request(
        spec.url,
        headers={"User-Agent": "GRACE-Mem dataset downloader"},
    )

    print(f"Downloading {spec.name} ({spec.size / 1024 / 1024:.1f} MiB)...")
    downloaded = 0
    next_report = 64 * 1024 * 1024
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  {downloaded / 1024 / 1024:.0f} MiB downloaded")
                    next_report += 64 * 1024 * 1024

        verify_file(temp_path, spec)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Verified SHA-256 {spec.sha256}: {destination}")


def validate_locomo(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError(f"Expected 10 LoCoMo conversations in {path}")
    required = {"sample_id", "conversation", "qa"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(f"Invalid LoCoMo record at index {index} in {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download checksum-pinned benchmark data. LongMemEval is converted "
            "to one CSV per question unless --download-only is used."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "locomo", "longmem"),
        default="all",
    )
    parser.add_argument(
        "--longmem-variant",
        choices=tuple(LONGMEM_FILES),
        default="s",
        help="s (default, 277 MB), m (2.7 GB), or oracle (15 MB)",
    )
    parser.add_argument(
        "--locomo-output",
        type=Path,
        default=REPO_ROOT / "experiment" / "locomo" / "data" / "locomo10.json",
    )
    parser.add_argument(
        "--longmem-raw-dir",
        type=Path,
        default=REPO_ROOT / "experiment" / "longmem" / "data",
    )
    parser.add_argument(
        "--longmem-output-dir",
        type=Path,
        default=REPO_ROOT / "experiment" / "longmem" / "script_data",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify downloaded source files without downloading or converting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files and replace a previous LongMem conversion",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = {"locomo", "longmem"} if args.dataset == "all" else {args.dataset}

    if "locomo" in selected:
        if args.verify_only:
            verify_file(args.locomo_output, LOCOMO)
        else:
            download_file(LOCOMO, args.locomo_output, force=args.force)
            validate_locomo(args.locomo_output)
        print(f"LoCoMo ready: {args.locomo_output}")

    if "longmem" in selected:
        spec = LONGMEM_FILES[args.longmem_variant]
        raw_path = args.longmem_raw_dir / spec.name
        if args.verify_only:
            verify_file(raw_path, spec)
            print(f"LongMemEval source verified: {raw_path}")
            return

        download_file(spec, raw_path, force=args.force)
        if args.download_only:
            print(f"LongMemEval source ready: {raw_path}")
            return

        summary = convert_longmem_dataset(
            raw_path,
            args.longmem_output_dir,
            source_revision=spec.revision,
            source_sha256=spec.sha256,
            variant=args.longmem_variant,
            force=args.force,
        )
        action = "Verified existing conversion" if summary.skipped else "Converted"
        print(
            f"{action} LongMemEval: {summary.records} questions, {summary.rows} turns -> "
            f"{args.longmem_output_dir}"
        )


if __name__ == "__main__":
    main()
