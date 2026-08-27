"""Load and normalize the LoCoMo datasets into the shapes the pipeline expects.

The official dataset has appeared under more than one filename, so paths are
resolved by trying known candidates and question records are normalized before
anything downstream sees them.

Adversarial questions are handled explicitly rather than filtered at the edges.
They are unanswerable by construction, so scoring them alongside ordinary
questions conflates "retrieved the wrong thing" with "correctly found nothing";
`include_adversarial` keeps that decision in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from experiment.locomo.utils.io import load_json_records

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_FILE_CANDIDATES = {
    "qa_json": ("locomo10.json", "locomo.json"),
    "sessions_jsonl": ("locomo_by_session.jsonl", "locomo_by_session_v2.jsonl"),
}

CATEGORY_LABELS = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
    "multi-hop": "Multi-hop",
    "temporal": "Temporal",
    "open-domain": "Open-domain",
    "single-hop": "Single-hop",
    "adversarial": "Adversarial",
}


def resolve_dataset_path(
    *,
    kind: str,
    explicit_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    required: bool = True,
) -> Path | None:
    """Find a dataset file by trying the known filenames for that variant.

    Candidates rather than one fixed name because the released files have been
    named differently across revisions (locomo10.json, locomo.json), and a run
    should work against whichever copy is present.

    Raises:
        FileNotFoundError: When no candidate exists -- failing here beats
            proceeding with an empty dataset and reporting zero accuracy.
    """
    if explicit_path:
        path = Path(explicit_path)
        if required and not path.exists():
            raise FileNotFoundError(f"{kind} file not found: {path}")
        return path if path.exists() or not required else None

    base_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    candidates = DATASET_FILE_CANDIDATES.get(kind, ())
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate

    if required:
        pretty = ", ".join(str(base_dir / name) for name in candidates)
        raise FileNotFoundError(
            f"Could not find a default LoCoMo {kind} file. "
            f"Searched: {pretty}. Pass an explicit path instead."
        )
    return None


def load_raw_samples(path: str | Path) -> List[Dict[str, Any]]:
    return load_json_records(path)


def category_to_label(value: Any) -> str:
    """Map a numeric category code to its human-readable label.

    Unknown codes pass through as their own string rather than becoming
    "Unknown", so a category added upstream shows up in reports as an
    unrecognised code instead of silently merging into an existing bucket.
    """
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if text.isdigit():
        return CATEGORY_LABELS.get(int(text), "Unknown")
    return CATEGORY_LABELS.get(text.lower(), text.title() if text else "Unknown")


def is_adversarial_category(value: Any) -> bool:
    return category_to_label(value).strip().lower() == "adversarial"


def normalize_qa_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one question record into the shape the pipeline expects.

    The field names differ across dataset variants and across their revisions --
    a question is "question", "trigger", or "query"; a gold answer is "answer",
    "gold_answer", or "reference_answer" -- so each is resolved by trying the
    known aliases. Everything downstream can then assume one schema.

    Evidence is coerced to a list because a single-evidence question stores a
    bare string, and code iterating it would otherwise walk the characters.

    The original record is kept under "raw", so a field this normalization does
    not know about is still reachable.
    """
    question = item.get("question", item.get("trigger", item.get("query", "")))
    answer = item.get("answer", item.get("gold_answer", item.get("reference_answer", "")))
    adversarial_answer = item.get("adversarial_answer", "")
    evidence = item.get("evidence", [])
    category = item.get("category")

    if evidence in (None, ""):
        evidence_list: List[str] = []
    elif isinstance(evidence, list):
        evidence_list = [str(x).strip() for x in evidence if str(x).strip()]
    else:
        evidence_list = [str(evidence).strip()]

    return {
        "question": str(question).strip(),
        "answer": "" if answer is None else str(answer).strip(),
        "adversarial_answer": "" if adversarial_answer is None else str(adversarial_answer).strip(),
        "evidence": evidence_list,
        "category": category,
        "category_label": category_to_label(category),
        "raw": item,
    }


def is_adversarial_item(item: Dict[str, Any]) -> bool:
    if "category_label" in item:
        return str(item.get("category_label", "")).strip().lower() == "adversarial"
    return is_adversarial_category(item.get("category"))


def load_qa_items(path: str | Path, *, sample_index: int, include_adversarial: bool = True) -> List[Dict[str, Any]]:
    """Load one sample's questions, normalized, optionally dropping adversarial ones.

    Args:
        include_adversarial: Adversarial questions are unanswerable by
            construction. Excluding them keeps them out of an accuracy average
            that would otherwise conflate "retrieved the wrong evidence" with
            "correctly declined to answer".
    """
    samples = load_raw_samples(path)
    if sample_index < 0 or sample_index >= len(samples):
        raise ValueError(f"sample_index out of range: {sample_index} (available: 0-{len(samples) - 1})")
    sample = samples[sample_index]
    if "qa" in sample and isinstance(sample["qa"], list):
        items = [normalize_qa_item(item) for item in sample["qa"] if isinstance(item, dict)]
        if include_adversarial:
            return items
        return [item for item in items if not is_adversarial_item(item)]
    raise ValueError(
        "Dataset sample is missing a supported QA schema. "
        f"Available keys: {', '.join(sorted(sample.keys()))}"
    )


def get_sample_conversation(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return one standard LoCoMo sample's structured conversation."""
    if "conversation" in sample and isinstance(sample["conversation"], dict):
        return sample["conversation"]
    raise ValueError(
        "Dataset sample is missing a supported conversation object. "
        f"Available top-level keys: {', '.join(sorted(sample.keys()))}"
    )


def get_sample_speakers(conversation: Dict[str, Any]) -> tuple[str | None, str | None]:
    return conversation.get("speaker_a"), conversation.get("speaker_b")


def build_session_records_from_json(path: str | Path) -> List[Dict[str, Any]]:
    """Turn a raw sample into per-session records ready for ingestion.

    Sessions are emitted in numeric order, not the dict order of the source
    JSON. Ingestion is sequential and provenance is positional, so ingesting
    session 10 before session 2 puts turns in the graph in an order the
    conversation never had.
    """
    samples = load_raw_samples(path)
    records: List[Dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        conv = get_sample_conversation(sample)
        speaker_a, speaker_b = get_sample_speakers(conv)
        for key, turns in conv.items():
            if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(turns, list):
                continue
            session_id = int(key.split("_", 1)[1])
            dialogue = []
            for turn in turns:
                speaker = str(turn.get("speaker", "")).strip()
                text = str(turn.get("text", "")).strip().replace("\n", " ")
                caption = str(turn.get("blip_caption", "")).strip()
                if not speaker and not text and not caption:
                    continue
                suffix = f" (Image: {caption})" if caption else ""
                if not text and caption:
                    dialogue.append(f"{speaker}:{suffix}")
                else:
                    dialogue.append(f"{speaker}: {text}{suffix}")
            records.append(
                {
                    "sample_index": sample_index,
                    "session_id": session_id,
                    "date_time": conv.get(f"session_{session_id}_date_time"),
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "dialogue": dialogue,
                }
            )
    return records


def find_evidence_turns_from_sample(sample: Dict[str, Any], question: str) -> List[str]:
    """Resolve a question's evidence ids to the turns they name."""
    normalized = [normalize_qa_item(item) for item in load_qa_items_from_sample(sample)]
    question_norm = question.strip()
    for item in normalized:
        if item["question"] == question_norm or item["question"].casefold() == question_norm.casefold():
            return item["evidence"]
    return []


def load_qa_items_from_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize an already-loaded sample's questions, without re-reading the file."""
    if "qa" in sample and isinstance(sample["qa"], list):
        return [item for item in sample["qa"] if isinstance(item, dict)]
    return []
