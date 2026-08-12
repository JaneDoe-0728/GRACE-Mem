# utils/logger_config.py
import logging, json, os, time
from logging.handlers import RotatingFileHandler
from typing import Optional, Any, Dict, Tuple

__all__ = [
    "setup_logger",          # 人類可讀: [ts][LEVEL] name: msg
    "get_event_logger",      # JSONL 事件日誌
    "close_event_loggers",   # 關閉事件 logger handlers
    "_jlog",                 # 寫一行 JSON 事件
    "_StepTimer",
]

# ----------- Human-readable logger (server 用) -----------
def setup_logger(
    name: str = "server",
    log_dir: str = "logs",
    level: int = logging.INFO,
    *,
    rotate_bytes: int = 5_000_000,
    backup_count: int = 5,
    to_console: bool = True,
) -> logging.Logger:
    """
    與你 server 現有版本對齊：人類可讀格式 + 檔案輪替 +（可選）console。
    若重複呼叫，不會重複加 handler。
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

# ----------- JSONL event logger (KG 檢索/追蹤用) -----------
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
    """Create or reuse a rotating JSONL event logger for KG telemetry."""
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
    """
    回傳一個「已綁定到指定檔案」的 _jlog(event, request_id, **data)。
    模組把它指派給 _jlog 名稱即可，呼叫點完全不用改。
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
        """Write one structured JSON event to the bound module logger."""
        payload = {"ts_ms": _now_ms(), "event": event, "request_id": request_id, **data}
        logger.info(json.dumps(payload, ensure_ascii=False))
    return _jlog


def close_event_loggers(
    *,
    name_prefix: str | None = None,
    log_dir: str | None = None,
) -> int:
    """
    Close and deregister cached event loggers matching the given filters.

    This prevents per-dataset RotatingFileHandlers from accumulating across
    long sequential runs.
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


def reopen_event_loggers() -> None:
    """
    Reopen all event logger file handlers after the log directory has been
    deleted and recreated (e.g. by refresh_system).  Without this, handlers
    keep writing to unlinked inodes on Linux and data is silently lost.
    """
    for (_name, _path), logger in _EVENT_LOGGERS.items():
        for handler in logger.handlers:
            if isinstance(handler, (RotatingFileHandler, logging.FileHandler)):
                if handler.stream:
                    handler.stream.close()
                os.makedirs(os.path.dirname(handler.baseFilename), exist_ok=True)
                handler.stream = handler._open()
