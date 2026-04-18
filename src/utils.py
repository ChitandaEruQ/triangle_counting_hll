"""Timing, memory, and metric helpers."""

import sys
import time
from typing import Union

try:
    import resource
except ImportError:
    resource = None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def peak_rss_mb() -> float:
    """
    Peak resident set size in MB.
    macOS: ru_maxrss is in bytes.
    Linux: ru_maxrss is in kilobytes.
    Windows: returns 0.0 when platform peak RSS is unavailable.
    """
    if resource is None:
        return 0.0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def relative_error(estimate: float, truth: int) -> float:
    """
    |estimate - truth| / truth.
    Returns inf if truth == 0 and estimate != 0; 0 if both are 0.
    """
    if truth == 0:
        return float('inf') if estimate != 0 else 0.0
    return abs(estimate - truth) / truth


def bias(estimate: float, truth: int) -> float:
    """estimate - truth (signed)."""
    return estimate - truth


# ---------------------------------------------------------------------------
# Bucket label for proxy values
# ---------------------------------------------------------------------------

_BREAKS = [0, 32, 64, 128, 256, 512]
_LABELS = ['[1,31]', '[32,63]', '[64,127]', '[128,255]', '[256,511]', '[512+]']


def proxy_label(proxy: int) -> str:
    """Return the bucket label for a proxy value."""
    for i, bp in enumerate(_BREAKS[1:], start=1):
        if proxy < bp:
            return _LABELS[i - 1]
    return _LABELS[-1]


# ---------------------------------------------------------------------------
# Simple context-manager timer
# ---------------------------------------------------------------------------

class Timer:
    """Usage:  with Timer() as t:  ...  then t.elapsed gives seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> 'Timer':
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
