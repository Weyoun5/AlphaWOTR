"""corpus_rarity_audit.py -- find self-play moves that are RARE in the corpus.

Standing diagnostic (USER directive 2026-07-28, after the trollshaws
root-cause): moves the net plays that humans essentially never played
are the best bug detectors -- every such cluster is either (a) a
label/encoding contamination (hobbit_place class), (b) an encoding
blind spot (SoMM region invisibility), (c) noise exploiting a flat
value region (trollshaws separations), or (d) legitimate exploration.
Classifying WHICH is the analysis; this tool finds the clusters.

Method: every CHOSEN action (corpus + self-play) is reduced to a
signature derived from its ACTION ENCODING, so both sides live in the
same key space by construction:
  coarse key = (category, subtype, characters, turn-bucket)
  fine key   = coarse + target-region
The corpus index counts signatures over all ~2.17M chosen samples;
the audit encodes each self-play trace step with the live
encode_action and reports fine-keys whose corpus count is below
--min-corpus (default 3), grouped, with exemplar (trace, step) pairs
for deep-dive.  Whole action TYPES absent from the corpus are
reported separately (absent-by-construction vs within-type rarity).

Usage:
  python corpus_rarity_audit.py --build-index [--corpus DIR]
  python corpus_rarity_audit.py --audit TRACES_DIR [--min-corpus 3]
                                [--top 40] [--max-traces N]

Index file: data/corpus_action_index.json.gz
ASCII only (cp1252 console).
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# WIRING 2026-08-11: default tracks the shipped generation (the v31
# dir was deleted in the cleanup)
DEFAULT_CORPUS = os.path.join(ROOT, 'data', 'bc_corpus_v37_rendered')
INDEX_PATH = os.path.join(ROOT, 'data', 'corpus_action_index.json.gz')

# encoding layout anchors (v32 generation; symbols re-derived at runtime
# from ai so a future subtype append cannot silently skew this tool)
_CAT_LO, _CAT_HI = 0, 14
_SUB_LO = 34


def _probe_offsets():
    """Empirically locate subtype width and char-block offset by
    encoding known actions (robust to future subtype appends)."""
    import ai
    import board
    import game_logic
    import game_state as gs
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    import numpy as np
    # a subtype-bearing action with no character: hunt_take_corruption
    a1 = ai.Action(die_idx=-1, action_type='hunt_take_corruption',
                   params={})
    e1 = np.asarray(ai.encode_action(a1, st, gs.Side.FREE_PEOPLES))
    sub_cols = [c for c in range(_SUB_LO, len(e1)) if e1[c] == 1.0]
    # a character-bearing action: choose_guide Strider
    a2 = ai.Action(die_idx=-1, action_type='choose_guide',
                   params={'companion': 'Strider'})
    e2 = np.asarray(ai.encode_action(a2, st, gs.Side.FREE_PEOPLES))
    ch_cols = [c for c in range(_SUB_LO, len(e2))
               if e2[c] == 1.0 and c not in sub_cols]
    if not sub_cols or not ch_cols:
        raise RuntimeError('offset probe failed: sub=%s ch=%s'
                           % (sub_cols, ch_cols))
    if sub_cols[0] != _SUB_LO:   # hunt_take_corruption = subtype idx 0
        raise RuntimeError('subtype block moved: %d' % sub_cols[0])
    char_off = ch_cols[0] - 1    # Strider = char idx 1
    n_sub = char_off - 4 - _SUB_LO
    return _SUB_LO, n_sub, char_off, 13


def _turn_bucket(t):
    t = int(t)
    if t <= 3:
        return t
    if t <= 6:
        return 4
    if t <= 9:
        return 7
    return 10


def _sig_from_enc(row, region, turn, sub_lo, n_sub, ch_off, n_ch):
    import numpy as np
    r = np.asarray(row)
    cat = int(np.argmax(r[_CAT_LO:_CAT_HI])) if r[_CAT_LO:_CAT_HI].max() == 1.0 else -1
    subs = np.where(r[sub_lo:sub_lo + n_sub] == 1.0)[0]
    sub = int(subs[0]) if len(subs) else -1
    chars = tuple(int(c) for c in
                  np.where(r[ch_off:ch_off + n_ch] == 1.0)[0])
    coarse = (cat, sub, chars, _turn_bucket(turn))
    fine = coarse + (int(region),)
    return coarse, fine


def _key_s(k):
    return '|'.join(str(x) for x in k)


def build_index(corpus_dir):
    import numpy as np
    sub_lo, n_sub, ch_off, n_ch = _probe_offsets()
    coarse = collections.Counter()
    fine = collections.Counter()
    files = sorted(glob.glob(os.path.join(corpus_dir, '*.npz')))
    n_rows = 0
    n_skip = 0
    for fi, p in enumerate(files):
        try:
            d = np.load(p, allow_pickle=True)
            if 'cand_encs' not in d.files:
                continue
            ce = d['cand_encs']
            cr = d['cand_regs']
            off = d['offsets']
            tn = d['turn']
        except Exception:
            n_skip += 1
            continue
        n = len(tn)
        hum = d['human'].astype('int64')
        if len(off) != n + 1:
            raise RuntimeError('malformed offsets in %s' % p)
        starts = off[:-1].astype('int64')
        # ragged pack: the HUMAN-CHOSEN candidate sits at
        # offsets[i] + human[i] (offsets[i] alone is just menu slot 0)
        chosen = starts + hum
        for i in range(n):
            c, f = _sig_from_enc(ce[chosen[i]], cr[chosen[i]], tn[i],
                                 sub_lo, n_sub, ch_off, n_ch)
            coarse[_key_s(c)] += 1
            fine[_key_s(f)] += 1
            n_rows += 1
        if fi % 1000 == 0:
            print('  index %d/%d files, %d rows' % (fi, len(files),
                                                    n_rows), flush=True)
    if not files or n_rows == 0:
        raise RuntimeError('no corpus rows indexed from %s '
                           '(%d files, %d skipped) -- an empty index '
                           'would flag every self-play step as rare'
                           % (corpus_dir, len(files), n_skip))
    if n_skip:
        print('  WARN: %d unreadable npz files skipped' % n_skip)
    meta = {'rows': n_rows, 'files': len(files),
            'layout': [sub_lo, n_sub, ch_off, n_ch]}
    with gzip.open(INDEX_PATH, 'wt', encoding='utf-8') as f:
        json.dump({'meta': meta, 'coarse': dict(coarse),
                   'fine': dict(fine)}, f)
    print('index written: %s (%d rows, %d fine keys, %d coarse keys)'
          % (INDEX_PATH, n_rows, len(fine), len(coarse)))


def audit(traces_dir, min_corpus=3, top=40, max_traces=None):
    import numpy as np
    import ai
    import game_trace
    with gzip.open(INDEX_PATH, 'rt', encoding='utf-8') as f:
        idx = json.load(f)
    sub_lo, n_sub, ch_off, n_ch = idx['meta']['layout']
    p_lo, p_nsub, p_choff, p_nch = _probe_offsets()
    if [p_lo, p_nsub, p_choff, p_nch] != [sub_lo, n_sub, ch_off, n_ch]:
        raise RuntimeError('index layout %s != live layout %s -- rebuild '
                           'the index (encoding generation changed)'
                           % (idx['meta']['layout'],
                              [p_lo, p_nsub, p_choff, p_nch]))
    coarse, fine = idx['coarse'], idx['fine']
    # (cat, sub) marginal: full-coarse absence conflates unusual
    # turn-bucket/character combos with genuinely absent TYPES
    catsub = collections.Counter()
    for k, v in coarse.items():
        catsub['|'.join(k.split('|')[:2])] += v
    from game_state import Side
    flags = collections.defaultdict(list)   # fine key -> exemplars
    type_absent = collections.Counter()
    enc_fail = collections.Counter()
    n_load_fail = 0
    n_steps = 0
    # engine annotation rows (RNG / forced events, not chosen actions)
    # carry a side and would otherwise audit as self-play moves
    ann_types = {'auto_declare_gateway', 'dice_roll', 'hunt_tile_draw',
                 'stronghold_tile_draw', 'random_draw', 'card_drawn',
                 'hunt_pool_recycled', 'wk_sorcerer_draw',
                 'onslaught_auto'}
    files = sorted(glob.glob(os.path.join(traces_dir, '*.trace.gz')))
    if max_traces:
        files = files[:max_traces]
    for p in files:
        try:
            tr = game_trace.load_trace(p)
        except Exception:
            n_load_fail += 1
            continue
        for si, s in enumerate(tr.get('steps', [])):
            at = s.get('action_type')
            if not at or str(at) == 'None':
                continue
            if str(at) in ann_types:
                continue
            st = s.get('state')
            side = s.get('side')
            if isinstance(side, str):
                _su = side.upper()
                side = (Side.FREE_PEOPLES
                        if _su in ('FP', 'FREE_PEOPLES', 'FREE PEOPLES')
                        or 'FREE' in _su
                        else Side.SHADOW
                        if _su in ('SH', 'SA') or 'SHADOW' in _su
                        else None)
            if st is None or side is None:
                continue
            try:
                a = ai.Action(die_idx=s.get('die_idx') if
                              isinstance(s.get('die_idx'), int) else -1,
                              action_type=str(at),
                              params=s.get('params') or {})
                enc = np.asarray(ai.encode_action(a, st, side))
                # SAME key space as the corpus index by construction:
                # the index's cand_regs come from
                # ai._get_action_region_index (rendered_extract), so a
                # hand-rolled params walk here would diverge
                reg = ai._get_action_region_index(a)
                c, f = _sig_from_enc(enc, reg, s.get('turn', 0),
                                     sub_lo, n_sub, ch_off, n_ch)
            except Exception:
                enc_fail[str(at)] += 1
                continue
            n_steps += 1
            ck, fk = _key_s(c), _key_s(f)
            if catsub.get('|'.join(ck.split('|')[:2]), 0) == 0:
                type_absent[str(at)] += 1
            if fine.get(fk, 0) < min_corpus:
                flags[fk].append({
                    'trace': os.path.basename(p), 'step': si,
                    'turn': s.get('turn'), 'type': str(at),
                    'params': {k: str(v)[:40] for k, v in
                               (s.get('params') or {}).items()},
                    'corpus_fine': fine.get(fk, 0),
                    'corpus_coarse': coarse.get(ck, 0)})
    print('audited %d steps across %d traces' % (n_steps, len(files)))
    if n_load_fail:
        print('WARN: %d traces failed to load' % n_load_fail)
    if enc_fail:
        print('WARN: %d steps failed to encode: %s'
              % (sum(enc_fail.values()), dict(enc_fail)))
    print('=== whole-class corpus-ABSENT action types (check emission '
          'coverage first -- may be absent-by-construction) ===')
    for t, n in type_absent.most_common():
        print('  %5d  %s' % (n, t))
    print('=== rare fine-keys (corpus < %d), by self-play frequency ==='
          % min_corpus)
    ranked_all = sorted(flags.items(), key=lambda kv: -len(kv[1]))
    ranked = ranked_all[:top]
    for fk, ex in ranked:
        e0 = ex[0]
        print('%4dx  corpus=%d/%d  %s %s (e.g. %s step %s turn %s)'
              % (len(ex), e0['corpus_fine'], e0['corpus_coarse'],
                 e0['type'], e0['params'], e0['trace'], e0['step'],
                 e0['turn']))
    out = os.path.join(ROOT, 'data', 'rarity_audit_flags.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in ranked_all}, f, indent=1)
    print('full exemplar list (%d keys; console showed top %d): %s'
          % (len(ranked_all), len(ranked), out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-index', action='store_true')
    ap.add_argument('--corpus', default=DEFAULT_CORPUS)
    ap.add_argument('--audit', metavar='TRACES_DIR')
    ap.add_argument('--min-corpus', type=int, default=3)
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--max-traces', type=int, default=None)
    a = ap.parse_args()
    if a.build_index:
        build_index(a.corpus)
        return
    if a.audit:
        audit(a.audit, a.min_corpus, a.top, a.max_traces)
        return
    ap.error('pick --build-index or --audit')


if __name__ == '__main__':
    main()
