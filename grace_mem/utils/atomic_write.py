"""Write a file so a reader never sees it half-written.

Every artifact this package persists is read back by something else: the next
run resumes from the cache, the analysis tooling loads the metadata exports, the
BM25 index is restored at startup. A plain `open(path, "wb")` truncates the
target first, so from that instant until the write completes the file on disk is
short -- and a process killed in that window leaves it short permanently. An
unpickle of a truncated file raises; a truncated JSONL export just loses its
tail, silently.

The fix is the usual one: write a temp file beside the target, flush it to the
platter, then `os.replace` it into place. `os.replace` is atomic within a
filesystem, which is why the temp file must be a sibling of the target rather
than in /tmp -- across filesystems it degrades to a copy, and the guarantee is
gone.

This does not make a *set* of files atomic. `_persist_all` writes four of them,
and a crash between two still leaves the cache and the indexes disagreeing.
That is handled by ordering (the cache is written last, so a lagging cache means
re-extraction rather than skipped work), not by this module.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def atomic_write(path: str | Path, mode: str = "wb", **open_kwargs: Any) -> Iterator[IO]:
    """Yield a handle to a temp file that replaces `path` on clean exit.

    The temp file is removed and the target left untouched if the body raises,
    so a failed write cannot destroy the previous good copy.

    Args:
        path: Final destination. Its parent directory is created if missing.
        mode: Any write mode accepted by `open`.
        **open_kwargs: Passed through to `open` (`encoding`, `newline`, ...).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Sibling, not /tmp: os.replace is only atomic within one filesystem.
    tmp = target.with_name(f"{target.name}.tmp")

    handle = open(tmp, mode, **open_kwargs)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
        handle.close()
        tmp.unlink(missing_ok=True)
        raise
    else:
        handle.close()
        os.replace(tmp, target)
