"""JSONL event log for Odyssey branch experiments.

One process writes one file; the run script gives each branch its own path so
the analysis pass can group by branch without parsing vLLM's stdout.

Events are appended, never rewritten, so a crashed run still yields whatever
rounds completed.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()
_HANDLE = None
_PATH: str | None = None


def path() -> str | None:
    return os.environ.get("ODYSSEY_EVENT_LOG") or None


def enabled() -> bool:
    return path() is not None


def _handle():
    """Open lazily: vLLM forks workers, and a handle opened in the parent would
    interleave badly across processes."""
    global _HANDLE, _PATH
    p = path()
    if p is None:
        return None
    if _HANDLE is None or _PATH != p:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        # Suffix with pid so forked workers never share a file descriptor.
        real = f"{p}.{os.getpid()}"
        _HANDLE = open(real, "a", buffering=1, encoding="utf-8")
        _PATH = p
    return _HANDLE


def emit(event: str, **fields: Any) -> None:
    if not enabled():
        return
    h = _handle()
    if h is None:
        return
    rec = {"event": event, "pid": os.getpid(), **fields}
    with _LOCK:
        h.write(json.dumps(rec, sort_keys=True, default=_default) + "\n")


def _default(o: Any) -> Any:
    # Torch scalars and numpy types show up in field values; keep them readable
    # rather than letting json.dumps raise mid-run.
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    return str(o)
