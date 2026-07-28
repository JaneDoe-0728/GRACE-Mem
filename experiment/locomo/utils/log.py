from __future__ import annotations


def log_event(tag: str, message: str, **fields: object) -> None:
    suffix = ""
    if fields:
        rendered = ", ".join(
            f"{key}={value}"
            for key, value in fields.items()
            if value is not None
        )
        if rendered:
            suffix = f" | {rendered}"
    print(f"[{tag}] {message}{suffix}")
