"""Two logging channels: prose for humans, JSONL events for analysis.

The split is the point. `setup_logger` produces the readable log you tail while
a run is going. `get_event_logger` and `_jlog` produce one JSON object per line,
which is what the analysis scripts consume -- retrieval traces, per-turn ingest
deltas, and failure verdicts are all queried after the fact by loading those
files into a dataframe. A prose log cannot support that, and a JSONL log is
unpleasant to read live, so both exist.

Loggers are memoized by (name, filename) so repeated setup calls cannot attach
duplicate handlers -- the failure that mode causes is every line appearing two
or three times, which corrupts any count derived from the file.

Everything rotates, because a full experiment sweep produces event logs large
enough to fill a disk.
"""

import logging, json, os, time
from logging.handlers import RotatingFileHandler
from typing import Optional, Any, Dict, Tuple

__all__ = [
    "setup_logger",          # human-readable: [ts][LEVEL] name: msg
    "get_event_logger",      # JSONL event log
    "close_event_loggers",   # close the event logger handlers
    "_jlog",                 # write one JSON event line
    "_StepTimer",
]

# ----------- Human-readable logger (for the server) -----------
def setup_logger(
    name: str = "server",
    log_dir: str = "logs",
    level: int = logging.INFO,
    *,
    rotate_bytes: int = 5_000_000,
    backup_count: int = 5,
    to_console: bool = True,
) -> logging.Logger:
    """Return a named logger writing human-readable lines to a rotating file.

    Idempotent: a logger that already has handlers is returned untouched, so
    modules can call this at import time without every import multiplying the
    output.

    Args:
        name: Logger name, and the log file's basename.
        rotate_bytes: Size at which the file rolls over.
        backup_count: Rotated files kept. With the default size that bounds one
            logger at roughly 30 MB.
        to_console: Also echo to stderr. Turn this off for per-turn trace logs,
            where console output would bury everything else.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_path, maxBytes=rotate_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

# ----------- JSONL event logger (for KG retrieval/tracing) -----------
_EVENT_LOGGERS: Dict[Tuple[str, str], logging.Logger] = {}

def get_event_logger(
    name: str = "grace_mem.Events",
    filename: str = "kg_events.jsonl",
    log_dir: str = "logs",
    level: int = logging.INFO,
    *,
    rotate_bytes: int = 10_000_000,
    backup_count: int = 5,
    also_stdout: bool = False,
) -> logging.Logger:
    """Return a logger that writes one JSON object per line.

    The formatter is bare `%(message)s` so nothing is prepended to the JSON --
    a timestamp or level prefix would make every line unparseable. Callers
    supply their own `ts_ms` inside the payload instead.

    Memoized on (name, absolute path) rather than name alone, because the same
    logger name is legitimately used against different files across datasets.

    Args:
        also_stdout: Echo events to stdout as well. Off by default; these are
            written per retrieval step and would flood a console.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(log_dir, filename))
    key = (name, path)

    if key in _EVENT_LOGGERS:
        return _EVENT_LOGGERS[key]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    fh = RotatingFileHandler(path, maxBytes=rotate_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    if also_stdout and not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
                               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    _EVENT_LOGGERS[key] = logger
    return logger

def _now_ms() -> int:
    """Return the current Unix time in milliseconds."""
    return int(time.time() * 1000)

class _StepTimer:
    """Wall-clock stopwatch for one pipeline step.

    Uses `perf_counter`, not `time.time`, so a clock adjustment mid-run cannot
    produce a negative or wildly wrong duration in the latency records.
    """

    def __init__(self) -> None:
        """Capture the start time for a short-lived timing measurement."""
        self.t0 = time.perf_counter()
    def sec(self) -> float:
        """Return elapsed wall-clock time in seconds since initialization."""
        return time.perf_counter() - self.t0

def make_module_jlog(
    *,
    name: str,
    filename: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    rotate_bytes: int = 10_000_000,
    backup_count: int = 5,
    also_stdout: bool = False,
) -> Any:
    """Build a `_jlog(event, request_id, **data)` bound to one event file.

    Modules assign the result to a module-level `_jlog` and call it directly,
    which keeps every call site free of logger plumbing while still letting
    each module route its events to its own file -- retrieval traces and ingest
    deltas are analysed separately and would be awkward interleaved.

    Every event carries `request_id`, and that is the field the analysis
    scripts join on to reassemble one question's path across files. Passing
    None makes an event impossible to correlate.
    """
    logger = get_event_logger(
        name=name,
        filename=filename,
        log_dir=log_dir,
        level=level,
        rotate_bytes=rotate_bytes,
        backup_count=backup_count,
        also_stdout=also_stdout,
    )
    def _jlog(event: str, request_id: Optional[str], **data: Any) -> None:
        """Write one structured JSON event to the bound module logger.

        `ensure_ascii=False` keeps non-ASCII conversation text readable in the
        log rather than escaped; the handler is opened as UTF-8 to match.
        Note that `**data` is splatted last and so can shadow ts_ms, event, or
        request_id -- avoid those key names in payloads.
        """
        payload = {"ts_ms": _now_ms(), "event": event, "request_id": request_id, **data}
        logger.info(json.dumps(payload, ensure_ascii=False))
    return _jlog


def close_event_loggers(
    *,
    name_prefix: str | None = None,
    log_dir: str | None = None,
) -> int:
    """Close and forget cached event loggers, releasing their file handles.

    A sequential sweep creates a fresh logger per dataset, and the memo table
    keeps every one alive. Without this the process accumulates open
    RotatingFileHandlers until it exhausts its file-descriptor limit -- a
    failure that surfaces far from its cause, partway through a long run.

    Both filters are optional and combine with AND; passing neither closes
    everything.

    Args:
        name_prefix: Close only loggers whose name starts with this.
        log_dir: Close only loggers writing into this directory.

    Returns:
        How many loggers were closed.
    """
    abs_log_dir = os.path.abspath(log_dir) if log_dir is not None else None
    closed = 0

    for key, logger in list(_EVENT_LOGGERS.items()):
        name, path = key
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        if abs_log_dir is not None and os.path.dirname(path) != abs_log_dir:
            continue

        for handler in list(logger.handlers):
            try:
                logger.removeHandler(handler)
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass

        _EVENT_LOGGERS.pop(key, None)
        closed += 1

    return closed
