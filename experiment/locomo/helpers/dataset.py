from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from locomo.utils.io import load_json_records

SUPPORTED_DATASETS = ("locomo", "locomo-plus")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_FILE_CANDIDATES = {
    "locomo": {
        "qa_json": ("locomo10.json", "locomo.json"),
        "sessions_jsonl": ("locomo_by_session.jsonl", "locomo_by_session_v2.jsonl"),
    },
    "locomo-plus": {
        "qa_json": ("unified_input_samples_v2.json", "locomo_plus.json", "locomo-plus.json"),
        "sessions_jsonl": ("locomo_plus_by_session.jsonl", "locomo-plus_by_session.jsonl"),
    },
}

CATEGORY_LABELS = {
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
    6: "Cognitive",
    "multi-hop": "Multi-hop",
    "temporal": "Temporal",
    "open-domain": "Open-domain",
    "single-hop": "Single-hop",
    "adversarial": "Adversarial",
    "cognitive": "Cognitive",
}

TURN_RE = re.compile(r'^(?P<speaker>.+?) said, (?P<body>.+)$')


def normalize_dataset_name(dataset: str | None) -> str:
    value = (dataset or "locomo").strip().lower()
    if value not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}. Supported values: {', '.join(SUPPORTED_DATASETS)}")
    return value


def default_output_stem(dataset: str) -> str:
    return normalize_dataset_name(dataset).replace("-", "_")


def default_output_variant_dir(dataset: str) -> str:
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "locomo":
        return "standard"
    if dataset_name == "locomo-plus":
        return "plus"
    return dataset_name.replace("-", "_")


def resolve_dataset_path(
    *,
    dataset: str,
    kind: str,
    explicit_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    required: bool = True,
) -> Path | None:
    dataset = normalize_dataset_name(dataset)
    if explicit_path:
        path = Path(explicit_path)
        if required and not path.exists():
            raise FileNotFoundError(f"{kind} file not found: {path}")
        return path if path.exists() or not required else None

    base_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    candidates = DATASET_FILE_CANDIDATES[dataset].get(kind, ())
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate

    if required:
        pretty = ", ".join(str(base_dir / name) for name in candidates)
        raise FileNotFoundError(
            f"Could not find a default {kind} file for dataset={dataset!r}. "
            f"Searched: {pretty}. Pass an explicit path instead."
        )
    return None


def load_raw_samples(path: str | Path) -> List[Dict[str, Any]]:
    return load_json_records(path)


def category_to_label(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if text.isdigit():
        return CATEGORY_LABELS.get(int(text), "Unknown")
    return CATEGORY_LABELS.get(text.lower(), text.title() if text else "Unknown")


def is_adversarial_category(value: Any) -> bool:
    return category_to_label(value).strip().lower() == "adversarial"


def normalize_qa_item(item: Dict[str, Any]) -> Dict[str, Any]:
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
    samples = load_raw_samples(path)
    if sample_index < 0 or sample_index >= len(samples):
        raise ValueError(f"sample_index out of range: {sample_index} (available: 0-{len(samples) - 1})")
    sample = samples[sample_index]
    if "qa" in sample and isinstance(sample["qa"], list):
        items = [normalize_qa_item(item) for item in sample["qa"] if isinstance(item, dict)]
        if include_adversarial:
            return items
        return [item for item in items if not is_adversarial_item(item)]
    if {"input_prompt", "trigger", "answer"} <= set(sample.keys()):
        item = normalize_qa_item(sample)
        if include_adversarial or not is_adversarial_item(item):
            return [item]
        return []
    raise ValueError(
        "Dataset sample is missing a supported QA schema. "
        f"Available keys: {', '.join(sorted(sample.keys()))}"
    )


def _parse_turn(line: str) -> Dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    match = TURN_RE.match(text)
    if not match:
        return {"speaker": "", "text": text}
    speaker = match.group("speaker").strip()
    body = match.group("body").strip()
    caption = ""
    shared_marker = '" and shared a '
    if shared_marker in body:
        body, caption = body.rsplit(shared_marker, 1)
        caption = caption.strip().rstrip(".")
    if body.startswith('"') and body.endswith('"'):
        body = body[1:-1]
    return {
        "speaker": speaker,
        "text": body.strip(),
        "blip_caption": caption,
    }


def build_conversation_from_input_prompt(prompt: str) -> Dict[str, Any]:
    conversation: Dict[str, Any] = {}
    speakers: List[str] = []
    segments = prompt.split("\n\nDATE: ")
    normalized_segments = []
    for idx, segment in enumerate(segments):
        if idx == 0 and segment.startswith("DATE: "):
            normalized_segments.append(segment[len("DATE: "):])
        elif idx > 0:
            normalized_segments.append(segment)

    for session_idx, segment in enumerate(normalized_segments, start=1):
        if "\nCONVERSATION:\n" not in segment:
            continue
        date_time, block = segment.split("\nCONVERSATION:\n", 1)
        block = block.split("\n\nQuestion:", 1)[0].strip()
        turns: List[Dict[str, Any]] = []
        for turn_idx, raw_line in enumerate(block.splitlines(), start=1):
            parsed = _parse_turn(raw_line)
            if not parsed:
                continue
            speaker = parsed.get("speaker", "")
            if speaker and speaker not in speakers:
                speakers.append(speaker)
            parsed["dia_id"] = f"D{session_idx}:{turn_idx}"
            turns.append(parsed)
        conversation[f"session_{session_idx}_date_time"] = date_time.strip()
        conversation[f"session_{session_idx}"] = turns

    conversation["speaker_a"] = speakers[0] if speakers else None
    conversation["speaker_b"] = speakers[1] if len(speakers) > 1 else None
    if not any(key.startswith("session_") and not key.endswith("_date_time") for key in conversation):
        block = prompt.split("\n\nQuestion:", 1)[0].strip()
        turns: List[Dict[str, Any]] = []
        for turn_idx, raw_line in enumerate(block.splitlines(), start=1):
            parsed = _parse_turn(raw_line)
            if not parsed:
                continue
            speaker = parsed.get("speaker", "")
            if speaker and speaker not in speakers:
                speakers.append(speaker)
            parsed["dia_id"] = f"D1:{turn_idx}"
            turns.append(parsed)
        if turns:
            conversation["session_1_date_time"] = None
            conversation["session_1"] = turns
            conversation["speaker_a"] = speakers[0] if speakers else None
            conversation["speaker_b"] = speakers[1] if len(speakers) > 1 else None
        else:
            raise ValueError("Could not parse any conversation turns from locomo-plus input_prompt")
    return conversation


def get_sample_conversation(sample: Dict[str, Any]) -> Dict[str, Any]:
    if "conversation" in sample and isinstance(sample["conversation"], dict):
        return sample["conversation"]
    if "input_prompt" in sample:
        return build_conversation_from_input_prompt(str(sample["input_prompt"]))
    raise ValueError(
        "Dataset sample is missing a supported conversation object. "
        f"Available top-level keys: {', '.join(sorted(sample.keys()))}"
    )


def get_sample_speakers(conversation: Dict[str, Any]) -> tuple[str | None, str | None]:
    return conversation.get("speaker_a"), conversation.get("speaker_b")


def build_session_records_from_json(path: str | Path) -> List[Dict[str, Any]]:
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


def index_source_conversations(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load locomo10.json and return a dict keyed by sample_id (e.g. 'conv-26')."""
    samples = load_raw_samples(path)
    return {str(s["sample_id"]): s for s in samples if "sample_id" in s}


def build_session_records_for_conv(
    conv_id: str,
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build session records for one locomo10.json sample (identified by conv_id).

    Returns a list sorted by session_id.
    """
    conv = get_sample_conversation(sample)
    return _session_records_from_conv_dict(conv_id, conv)


def build_session_records_for_conv_dict(
    conv_id: str,
    conv: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build session records from an already-parsed conversation dict.

    Used for cognitive instances where the conversation comes from parsing
    input_prompt rather than from locomo10.json.
    Returns a list sorted by session_id.
    """
    return _session_records_from_conv_dict(conv_id, conv)


def extract_injected_session_record(
    conv_id: str,
    input_prompt: str,
    injected_session_id: int,
    source_session_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a session record for the injected content of a cognitive item.

    Cognitive input_prompts have no DATE headers — all sessions are concatenated
    as a flat dialogue.  We count the turns already covered by snapshots
    (sessions 1..injected_session_id-1) and treat everything beyond that boundary
    as the injected session at ``injected_session_id``.
    """
    # Count dialogue turns covered by the snapshot (sessions before injection point)
    original_turn_count = sum(
        len(r["dialogue"])
        for r in source_session_records
        if r["session_id"] < injected_session_id
    )

    # Parse every speaker turn from the flat input_prompt
    all_turns: List[Dict[str, Any]] = []
    for line in input_prompt.splitlines():
        parsed = _parse_turn(line)
        if parsed and (parsed.get("speaker") or parsed.get("text")):
            all_turns.append(parsed)

    injected_turns = all_turns[original_turn_count:]
    if not injected_turns:
        raise ValueError(
            f"conv_id={conv_id!r}: no injected turns found after "
            f"{original_turn_count} original turns "
            f"(injected_session_id={injected_session_id})"
        )

    # Derive speakers from source records
    speakers: List[str] = []
    for r in source_session_records:
        for s in (r.get("speaker_a"), r.get("speaker_b")):
            if s and s not in speakers:
                speakers.append(s)

    dialogue = []
    for i, t in enumerate(injected_turns, start=1):
        spk = t.get("speaker", "")
        txt = t.get("text", "")
        cap = t.get("blip_caption", "")
        suffix = f" (Image: {cap})" if cap else ""
        if txt or cap:
            dialogue.append(f"{spk}: {txt}{suffix}" if spk else txt + suffix)

    return {
        "session_id": injected_session_id,
        "date_time": None,
        "speaker_a": speakers[0] if speakers else None,
        "speaker_b": speakers[1] if len(speakers) > 1 else None,
        "dialogue": dialogue,
    }


def _session_records_from_conv_dict(
    conv_id: str,
    conv: Dict[str, Any],
) -> List[Dict[str, Any]]:
    speaker_a, speaker_b = get_sample_speakers(conv)
    records: List[Dict[str, Any]] = []
    for key, turns in conv.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        if not isinstance(turns, list):
            continue
        session_id = int(key.split("_", 1)[1])
        dialogue: List[str] = []
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
                "session_id": session_id,
                "date_time": conv.get(f"session_{session_id}_date_time"),
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "dialogue": dialogue,
            }
        )
    return sorted(records, key=lambda r: r["session_id"])


def classify_locomo_plus_items(
    path: str | Path,
) -> Dict[str, List[int]]:
    """Return {conv_id: [sample_index, ...]} for all items in a locomo-plus QA JSON.

    Also returns a mapping of sample_index -> is_cognitive for ordering purposes.
    """
    samples = load_raw_samples(path)
    result: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        conv_id = sample.get("conversation_id")
        if conv_id is None:
            continue
        result.setdefault(str(conv_id), []).append(idx)
    return result


def is_cognitive_item(sample: Dict[str, Any]) -> bool:
    cat = str(sample.get("category", "")).strip().lower()
    return cat in ("cognitive", "6")


def find_evidence_turns_from_sample(sample: Dict[str, Any], question: str) -> List[str]:
    normalized = [normalize_qa_item(item) for item in load_qa_items_from_sample(sample)]
    question_norm = question.strip()
    for item in normalized:
        if item["question"] == question_norm or item["question"].casefold() == question_norm.casefold():
            return item["evidence"]
    return []


def load_qa_items_from_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "qa" in sample and isinstance(sample["qa"], list):
        return [item for item in sample["qa"] if isinstance(item, dict)]
    if {"input_prompt", "trigger", "answer"} <= set(sample.keys()):
        return [sample]
    return []
