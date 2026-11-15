# -*- coding: utf-8 -*-
"""Card-usage index over self-play traces (session 29, USER spec).

Scans every trace under _trace_dirs() (the single source of scan
locations) for card plays (play_event rows) and combat-card plays
(combat_card_play rows), and writes data/card_usage_index.json:

  {'cards':  {deck_key: [card_id, ...]},        # full 96-card vocab
   'usage':  {card_id: [{'trace','step','turn','side','combat',
                         'label'}, ...]},
   'scanned': {trace_path: mtime},
   'per_trace': {trace_path: [hit, ...]}}       # incremental cache

'step' is the raw index into trace['steps'] - exactly what the GUI's
--step argument jumps to.  Incremental: unchanged traces (same mtime)
are not re-read.

Usage: python card_usage_index.py [--rebuild]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, 'data', 'card_usage_index.json')
if os.environ.get('WOTR_LOOP_DIR'):
    INDEX_PATH = os.path.join(os.path.abspath(os.environ['WOTR_LOOP_DIR']), 'card_usage_index.json')


def _trace_dirs():
    """Live + archived v33 trace locations (2026-08-02 dead-path
    audit; archive roots added after the run-7 archive emptied the
    live dirs and a rebuild wiped the whole index).  Every dir under
    ANY data\\v33loop* root (live loop, run archives, their
    baseline_games/batch_* subdirs) plus the live trace dir.
    Resolved at call time — dirs appear as the loop archives."""
    configured = os.environ.get('WOTR_LOOP_DIR')
    if configured:
        # The dedicated dashboard indexes archived batches only; pending
        # worker traces are previewed separately before batch validation.
        return [path for path in sorted(glob.glob(os.path.join(
            os.path.abspath(configured), 'batch_*'))) if os.path.isdir(path)]
    dirs = [os.path.join(ROOT, 'data', 'td_v33_traces')]
    for top in sorted(glob.glob(os.path.join(ROOT, 'data',
                                             'v33loop*'))):
        if not os.path.isdir(top):
            continue
        for dirpath, _subdirs, _files in os.walk(top):
            dirs.append(dirpath)
    return [d for d in dirs if os.path.isdir(d)]
DECKS = ('FP Character', 'FP Strategy', 'SH Character', 'SH Strategy')


def deck_of(card_id):
    cid = str(card_id)
    side = 'FP' if cid.startswith('FREE_PEOPLES') else 'SH'
    kind = 'Character' if '_Character_' in cid else 'Strategy'
    return '%s %s' % (side, kind)


def card_vocab():
    """All 96 card ids from the engine's card index, grouped by deck."""
    import ai
    out = {d: [] for d in DECKS}
    for cid in ai._get_card_idx_map():
        out[deck_of(cid)].append(str(cid))
    for d in out:
        out[d] = sorted(out[d])
    return out


def scan_trace(tp):
    """All card-play hits in one trace, with raw step indices.
    Returns None (NOT []) on a load failure so build() retries the
    file next run instead of caching 'zero plays' as fact."""
    import game_trace
    hits = []
    try:
        tr = game_trace.load_trace(tp)
    except Exception as e:
        print('  WARN: unreadable trace skipped: %s (%r)'
              % (tp, e), flush=True)
        return None
    steps = tr['steps']
    for i, s in enumerate(steps):
        at = s.get('action_type')
        if at not in ('play_event', 'combat_card_play'):
            continue
        p = s.get('params') or {}
        cid = p.get('card_id')
        if not cid:
            continue
        hits.append({
            'card': str(cid),
            'step': i,
            'turn': int(s.get('turn') or 0),
            'side': str(s.get('side') or '?'),
            'combat': at == 'combat_card_play',
        })
    return hits


def build(rebuild=False):
    old = {}
    if not rebuild and os.path.exists(INDEX_PATH):
        try:
            old = json.load(open(INDEX_PATH, encoding='utf-8'))
        except Exception:
            old = {}
    scanned = dict(old.get('scanned') or {})
    per_trace = dict(old.get('per_trace') or {})

    traces = []
    for d in _trace_dirs():
        traces += glob.glob(os.path.join(d, '*.trace.gz'))
    live = set()
    n_new = 0
    for tp in sorted(traces):
        tp = os.path.abspath(tp)
        live.add(tp)
        try:
            mt = os.path.getmtime(tp)
        except OSError:
            continue
        if scanned.get(tp) == mt and tp in per_trace:
            continue
        hits = scan_trace(tp)
        if hits is None:
            continue        # unreadable now — retry on the next run
        per_trace[tp] = hits
        scanned[tp] = mt
        n_new += 1
        if n_new % 25 == 0:
            print('  scanned %d new traces...' % n_new, flush=True)
    if not live and per_trace:
        # zero traces on disk while the stored index is non-empty:
        # almost certainly a scan-roots regression - refuse to destroy
        # the history
        print('REFUSING rebuild: 0 traces found under %d roots but '
              'the existing index holds %d traces — index left '
              'untouched' % (len(_trace_dirs()), len(per_trace)),
              flush=True)
        return old
    # drop deleted traces
    for tp in list(per_trace):
        if tp not in live:
            per_trace.pop(tp, None)
            scanned.pop(tp, None)

    usage = {}
    for tp, hits in per_trace.items():
        base = os.path.basename(tp)
        for h in hits:
            rec = dict(h)
            rec.pop('card', None)
            rec['trace'] = tp
            rec['label'] = '%s  T%d %s%s' % (
                base.replace('.trace.gz', ''), h['turn'], h['side'],
                ' [combat]' if h['combat'] else '')
            usage.setdefault(h['card'], []).append(rec)
    for cid in usage:
        usage[cid].sort(key=lambda r: (r['trace'], r['step']))

    out = {'cards': card_vocab(), 'usage': usage,
           'scanned': scanned, 'per_trace': per_trace}
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    n_cards_hit = sum(1 for c in out['cards']
                      for cid in out['cards'][c] if cid in usage)
    print('index: %d traces, %d/96 cards seen, %d total plays'
          % (len(per_trace), n_cards_hit,
             sum(len(v) for v in usage.values())), flush=True)
    return out


if __name__ == '__main__':
    build(rebuild='--rebuild' in sys.argv)
