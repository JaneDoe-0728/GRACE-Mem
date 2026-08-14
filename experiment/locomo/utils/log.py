"""Tagged console logging for the run scripts.

Deliberately `print` rather than `logging`: these lines interleave with worker
subprocess output on one terminal, and the logging module's buffering reorders
them relative to that output. The `[tag] message | k=v` shape is what makes the
combined stream greppable afterwards.
"""

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
