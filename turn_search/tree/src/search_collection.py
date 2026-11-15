"""Explicit, process-local cyclic-GC deferral for a synchronous search.

Does not disable reference counting. Nested scopes share one outer collection.
Use only in a worker whose application code owns GC state; GC is process-wide.
"""
from contextlib import contextmanager
import gc
import threading
import time

_LOCK = threading.RLock()
_DEPTH = 0


@contextmanager
def deferred_gc():
    """Yield timing/count diagnostics; collect and restore state on outer exit.

The caller must include context exit in its wall-clock measurement. Reachable
trees (including cached reuse trees or exception tracebacks) are never cleared.
No RNG, model, tree, GC thresholds, or callback settings are changed.
"""
    global _DEPTH
    with _LOCK:
        outer = _DEPTH == 0
        original_enabled = gc.isenabled()
        stats = {'outermost': outer, 'collected': 0, 'cleanup_seconds': 0.0}
        if outer:
            stats['gc_counts_before'] = gc.get_count()
            gc.disable()
        _DEPTH += 1
        started = time.perf_counter()
        try:
            yield stats
        finally:
            _DEPTH -= 1
            stats['body_seconds'] = time.perf_counter() - started
            if outer:
                cleanup_started = time.perf_counter()
                try:
                    stats['collected'] = gc.collect()
                finally:
                    stats['cleanup_seconds'] = time.perf_counter() - cleanup_started
                    if original_enabled:
                        gc.enable()
                    else:
                        gc.disable()
                    stats['gc_counts_after'] = gc.get_count()
                    stats['restored_enabled'] = gc.isenabled()
            stats['total_seconds'] = time.perf_counter() - started


# Evaluated at call time so explicit caller overrides remain authoritative.
import os
from functools import wraps


def search_collection_scope(search):
    """Outer boundary: existing search cleanup completes before collection."""
    @wraps(search)
    def scoped(*args, **kwargs):
        if os.environ.get('WOTR_SEARCH_DEFER_GC', '0') != '1':
            return search(*args, **kwargs)
        with deferred_gc():
            return search(*args, **kwargs)
    return scoped
