# -*- coding: utf-8 -*-
"""Robust trace-path resolution (GUI wave, 2026-08-11).

Stored trace paths in data\\v33loop\\all_games.jsonl go stale in
three live ways: the stale-pre-run sweep moves in-flight traces to
stale_quarantine_* dirs, artifacts get cleared, and historical
quarantine-renamed batch dirs (batch_000_quarantined_stageA — the
enforcement mechanism itself was retired 2026-08-11, the renamed dirs
persist on disk) hold pre-rename paths.

Policy: a stored trace_path that exists is used as-is.  Otherwise the
BASENAME is searched across every known trace root (batch dirs,
quarantine renames, baseline_games, smoke traces, the live trace dir)
and the NEWEST match by mtime wins.  Callers must show
`missing_message()` in the GUI when nothing is found — never a silent
failure, never a stack trace.

Kept dependency-free (os/glob only) so the replay GUI and the
dashboard can both import it, and lab tests can exercise it against a
fake tree via explicit `roots=`.
"""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_roots():
    """Search roots, in priority order.  data\\v33loop is searched
    RECURSIVELY, so batch_*, batch_*_quarantined_*, stale_quarantine_*,
    baseline_games and smoke_gate_traces are all covered without
    naming them; td_v33_traces holds in-flight live traces."""
    return [os.path.join(ROOT, 'data', 'v33loop'),
            os.path.join(ROOT, 'data', 'td_v33_traces')]


def build_index(roots=None):
    """basename -> [(mtime, path), ...] across all roots.  Build once
    when resolving many paths (card finder, batch listings); a single
    resolve_trace() call builds its own throwaway candidate list."""
    idx = {}
    for root in (roots if roots is not None else default_roots()):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                try:
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                idx.setdefault(fn, []).append((mt, p))
    return idx


def _pick(cands, stored_parent):
    """Basenames REPEAT across batches (w0_g0_* exists in every
    batch), so a bare newest-wins can cross into the WRONG game.
    The true dir's name always starts with the stored parent name:
    exact match for live batch dirs, underscore-suffixed for
    historical quarantine renames still on disk
    (batch_000_quarantined_stageA) — bare startswith would also let
    batch_100 claim batch_1000 at 4-digit indices.  Traces stored
    under the LIVE trace dir (td_v33_traces) get swept to
    stale_quarantine_* dirs, so those are preferred for that parent;
    global newest is only the last fallback."""
    if not cands:
        return None
    sp = (stored_parent or '').lower()

    def _par(c):
        return os.path.basename(os.path.dirname(c[1])).lower()

    if sp:
        pref = [c for c in cands
                if _par(c) == sp or _par(c).startswith(sp + '_')]
        if pref:
            return max(pref)[1]
        if sp == 'td_v33_traces':
            swept = [c for c in cands
                     if _par(c).startswith('stale_quarantine')]
            if swept:
                return max(swept)[1]
    return max(cands)[1]


def resolve_trace(path, roots=None, index=None):
    """Resolve a possibly-stale trace path.

    Returns (resolved_path_or_None, searched_roots).  resolved_path
    is the stored path itself when it still exists; else the best
    same-basename file under the roots (same-parent-prefix first,
    then newest); else None.  When `index` is supplied it MUST have
    been built over the same `roots` — the roots value is what
    missing_message() reports as searched."""
    if roots is None:
        roots = default_roots()
    path = str(path or '')
    if path and os.path.exists(path):
        return path, roots
    base = os.path.basename(path)
    if not base:
        return None, roots
    stored_parent = os.path.basename(os.path.dirname(path))
    if index is not None:
        return _pick(index.get(base, []), stored_parent), roots
    cands = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(glob.escape(root), '**',
                                        glob.escape(base)),
                           recursive=True):
            try:
                cands.append((os.path.getmtime(p), p))
            except OSError:
                continue
    return _pick(cands, stored_parent), roots


def missing_message(path, roots):
    """The in-GUI error text when a trace is truly gone (names the
    searched roots per USER spec — never a silent failure)."""
    return ('Trace file not found.\n\n'
            'Stored path (stale):\n  %s\n\n'
            'Searched (by basename, newest wins):\n%s\n\n'
            'The trace was likely swept to a stale_quarantine_* dir '
            'and later deleted, or artifacts were cleared.'
            % (path, '\n'.join('  %s' % r for r in roots)))
