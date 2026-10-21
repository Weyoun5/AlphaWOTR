"""Log-ordered verification-skeleton generator (rendered-log campaign).

    python src/rendered_skeleton.py <log-stem> [out.md]

Runs the extractor over ONE game with an emission spy, resolves a
LOG-LINE ANCHOR for every kept row, sorts each turn by anchor, and
writes a markdown skeleton table per turn (anchor / die / side / kind /
action / params / menu-size, with an empty Decision column for the
human adjudication pass).

WHY (USER report 2026-07-24, GAME19): tables transcribed in EMISSION
order scrambled ledger-class rows - eye allocations are PRE-ROLL
turn-start gestures, hand-limit discards follow the turn-start draws,
card-interior moves belong inside their play's window, reveals ride
their hunt.  The anchor rules here are the mechanical form of that
line audit:

  1. batch-anchored rows           -> the batch's start_line
  2. eye_allocate                  -> pre-roll hunt-box gestures
                                      ('... to The Hunt For The Ring'
                                      die-moves or 'allocates an eye')
  3. pass                          -> '<nick> pass' chat lines
  4. fellowship_move               -> FSP track INCREMENT lines
  5. reveal / declare placements   -> 'moves The Fellowship from A to B'
                                      (before the turn's first
                                      action-dice roll of EITHER side =
                                      declaration, after = reveal;
                                      declarations are turn-start-only)
  6. discard_card                  -> 'kills <X> Card in <side> Hand'
  7. choose_guide                  -> '... to Guide Of The Fellowship'
  8. elven_ring                    -> Elven Ring / Dark Ring counter
  9. any row with from/to params   -> the turn's matching rendered
                                      'from A ... to B' move line
 10. everything else               -> carry-forward from the previous
                                      anchored row (flagged '~')
"""
import io
import os
import re
import sys
import inspect
import functools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def build_skeleton(stem, out_path=None):
    import rendered_extract as rx
    import rendered_replay as rr

    # ---- emission spy with batch spans -------------------------------
    cur = {'batch': None}
    for _nm in dir(rx):
        _fn = getattr(rx, _nm)
        # every module function whose FIRST param is 'batch' is a
        # batch-scoped emitter (_emit*, _combat_target_sample,
        # process_batch); _emit_sample's first param is 'st' so the
        # spy target itself stays unwrapped
        if _nm.startswith('__') or not callable(_fn) \
                or not getattr(_fn, '__module__', '').endswith(
                    'rendered_extract'):
            continue
        try:
            _p0 = list(inspect.signature(_fn).parameters)[0]
        except (ValueError, TypeError, IndexError):
            continue
        if _p0 != 'batch':
            continue

        def _wrap(f):
            @functools.wraps(f)
            def g(batch, *a, **k):
                old = cur['batch']
                cur['batch'] = batch
                try:
                    return f(batch, *a, **k)
                finally:
                    cur['batch'] = old
            return g
        setattr(rx, _nm, _wrap(_fn))

    rows = []
    orig = rx._emit_sample

    def spy(st, side, legal, hidx, kind, turn, *a, **kw):
        s = orig(st, side, legal, hidx, kind, turn, *a, **kw)
        if s is not None:
            b = cur['batch'] if isinstance(cur['batch'], dict) else None
            act = legal[hidx]
            rows.append({
                'obj': s, 'turn': turn, 'kind': kind,
                'side': getattr(side, 'name', str(side))[:2],
                'atype': act.action_type, 'nopt': len(legal),
                'params': dict(act.params or {}),
                'blo': b.get('start_line') if b else None,
                'bhi': b.get('end_line') if b else None,
                'die': ('%s/%s' % (str(b.get('die_side', ''))[:2],
                                   b.get('die_face'))) if b else None})
        return s

    tgt = next(((p, s, m) for p, s, m in rx._game_iter('canary')
                if stem in s), None)
    if tgt is None:
        tgt = next(((p, s, m) for p, s, m in rx._game_iter('corpus')
                    if stem in s), None)
    if tgt is None:
        raise SystemExit('stem not found: %s' % stem)
    rx._emit_sample = spy
    try:
        samples, _line, _cov = rx.extract_one(*tgt)   # v32: 3-tuple
    finally:
        # extract_one raises _Quarantine on gate failures; the batch
        # wrappers above are transparent and may stay
        rx._emit_sample = orig
    kept_ids = {id(s) for s in (samples or [])}
    kept = [r for r in rows if id(r['obj']) in kept_ids]
    kept.sort(key=lambda r: r['turn'])          # stable, keeps order

    # ---- log scan: per-class anchor queues ----------------------------
    log_path = tgt[0] if isinstance(tgt[0], str) and \
        os.path.exists(str(tgt[0])) else None
    if log_path is None:
        raise SystemExit('rendered log not found: %r' % (tgt[0],))
    text = io.open(log_path, encoding='utf-8',
                   errors='replace').read().splitlines()
    tb = {}
    for i, l in enumerate(text, 1):
        m = re.search(r'<replay> Turn (\d+)', l)
        if m:
            tb.setdefault(int(m.group(1)), i)

    def turn_of(ln):
        t = min(tb) if tb else 1
        for k, v in sorted(tb.items()):
            if ln >= v:
                t = k
        return t

    from collections import defaultdict
    q = defaultdict(list)      # (turn, class) -> [line, ...]
    first_roll = {}            # turn -> first action-dice roll line
                               # (EITHER side: declarations are
                               # turn-start-only, any roll ends the
                               # declare window)
    for i, l in enumerate(text, 1):
        s, t = l.strip(), turn_of(i)
        # undo/redo echoes embed the original gesture text; the
        # extractor emits only the undo-resolved net stream, so
        # queueing them would shift anchor pairing
        if re.search(r'\b(un|re)does last action\b', s):
            continue
        if re.search(r'rolls\s+[A-Z]( [A-Z])+', s) and t not in first_roll:
            first_roll[t] = i
        if re.search(r'to The Hunt [Ff]or [Tt]he Ring', s) \
                and 'Action Die' in s and 'Free Action Die' not in s:
            q[(t, 'eye')].append(i)
        elif re.search(r'allocates an eye', s):
            q[(t, 'eye')].append(i)
        if re.match(r'%?<[^>]+>\s*pass\s*$', s, re.I):
            q[(t, 'pass')].append(i)
        m = re.search(r'moves FSP from Fellowship Track (\d+) to '
                      r'Fellowship Track (\d+)', s)
        if m and int(m.group(2)) > int(m.group(1)):
            q[(t, 'fel')].append(i)
        if re.search(r'moves The Fellowship\b', s):
            cls = 'declare' if i < first_roll.get(t, 10 ** 9) else 'reveal'
            q[(t, cls)].append(i)
        if re.search(r'kills .*Card in (Free Peoples|Shadow) Hand', s):
            q[(t, 'discard')].append(i)
        if re.search(r'to Guide Of The Fellowship', s):
            q[(t, 'guide')].append(i)
        if re.search(r'Elven Ring|Dark Ring', s):
            q[(t, 'ring')].append(i)

    CLASS = {'eye_allocate': 'eye', 'pass': 'pass',
             'fellowship_move': 'fel',
             'fellowship_reveal_place': 'reveal',
             'fellowship_declare': 'declare',
             'discard_card': 'discard', 'choose_guide': 'guide',
             'elven_ring': 'ring'}

    san_lines = [rr.sanitize(l) for l in text]

    def move_line_anchor(turn, frm, to):
        lo = tb.get(turn, 1)
        hi = tb.get(turn + 1, len(text) + 1)
        f9, t9 = rr.sanitize(str(frm)), rr.sanitize(str(to))
        if not f9 or not t9:
            return None
        for i in range(lo, min(hi, len(text) + 1)):
            sl = san_lines[i - 1]
            if ('from ' + f9) in sl and ('to ' + t9) in sl:
                return i
        return None

    used = defaultdict(int)
    prev_anchor = 0.0
    for r in kept:
        a = None
        exact = True
        if r['blo'] is not None:
            a = float(r['blo'])
        else:
            cls = CLASS.get(r['kind']) or CLASS.get(r['atype'])
            if cls is not None:
                lst = q.get((r['turn'], cls)) or []
                k = used[(r['turn'], cls)]
                if k < len(lst):
                    a = float(lst[k])
                    used[(r['turn'], cls)] += 1
            if a is None:
                p = r['params']
                frm = p.get('from') or p.get('from_region')
                to = p.get('to') or p.get('to_region') or p.get('region')
                if frm and to:
                    ml = move_line_anchor(r['turn'], frm, to)
                    if ml is not None:
                        a = float(ml)
            if a is None:
                a = prev_anchor + 0.001      # carry-forward
                exact = False
        r['anchor'] = a
        r['anchor_exact'] = exact
        prev_anchor = max(prev_anchor, a)
    kept.sort(key=lambda r: (r['turn'], r['anchor']))

    # ---- markdown skeleton --------------------------------------------
    P_KEYS = ('from', 'from_region', 'to', 'to_region', 'region',
              'card_id', 'companion',
              'nation', 'count', 'n_regular', 'n_elite', 'take_leaders',
              'include_wk', 'deck', 'minion', 'type', 'new_face',
              'old_face')
    out = ['# %s - verification SKELETON (log-ordered, auto-generated)'
           % stem, '',
           'Anchors resolved per rendered_skeleton.py rules; rows '
           'flagged ~ carry forward from the previous anchored row. '
           'The Decision/adjudication column is the HUMAN pass.', '']
    from collections import Counter
    per_turn = Counter(r['turn'] for r in kept)
    out.append('Per-turn rows: %s   TOTAL %d' % (
        dict(sorted(per_turn.items())), len(kept)))
    out.append('')
    cur_t = None
    n = 0
    for r in kept:
        if r['turn'] != cur_t:
            cur_t = r['turn']
            n = 0
            out += ['## T%d (%d rows)' % (cur_t, per_turn[cur_t]), '',
                    '| # | Anchor | Die | Side | Kind | Action | Params '
                    '| Menu | Decision (human) |',
                    '|---|--------|-----|------|------|--------|--------'
                    '|------|------------------|']
        n += 1
        if r['blo'] is not None:
            anch = '%d-%d' % (r['blo'], r['bhi'])
        else:
            anch = '%s%d' % ('' if r['anchor_exact'] else '~',
                             int(r['anchor']))
        prm = ' '.join('%s=%s' % (k, str(v)[:28])
                       for k, v in r['params'].items() if k in P_KEYS)
        out.append('| %d | %s | %s | %s | %s | %s | %s | %d |  |' % (
            n, anch, r['die'] or '-', r['side'].lower(), r['kind'],
            r['atype'], prm[:100], r['nopt']))
        if n == per_turn[cur_t]:
            out.append('')
    md = '\n'.join(out) + '\n'
    if out_path:
        io.open(out_path, 'w', encoding='utf-8').write(md)
        print('wrote', out_path, '(%d rows)' % len(kept))
    return kept, md


if __name__ == '__main__':
    _stem = sys.argv[1]
    _out = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                     'data', '%s_skeleton.md' % _stem)
    build_skeleton(_stem, _out)
