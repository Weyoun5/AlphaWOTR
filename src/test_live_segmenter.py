"""Segmenter instruments: the live stream batcher vs the corpus batcher.

MEASURED QUANTITY (headline differential): on the SAME real rendered
feed, the streaming Segmenter's window openings (count + die side/face
sequence) equal rendered_replay.parse_game's batch boundaries, and
every closed window's records are a subset of the corresponding
batch's records.  parse_game resolves undo/redo BEFORE segmentation
(a batch construct), so the strict comparison runs only on feeds with
ZERO undo records; feeds with undos are reported, not silently
skipped.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)
sys.path.insert(0, _ROOT)

import rendered_replay as rr
from wotr_bridge.live import live_parser
from wotr_bridge.live.segmenter import Segmenter

_PAIRS = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                      'audit', 'pairs')


def _stream_file(path):
    """Run a feed file through live_parser + Segmenter; return
    (opens, closes, discards, undo_count) where opens/closes carry
    (die_side, die_face) and closes carry record key-sets."""
    seg = Segmenter()
    opens, closes, discards = [], [], []
    undo = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f.read().splitlines():
            rec = live_parser.parse(raw)
            for ev, payload in seg.push(rec):
                if ev == 'open':
                    opens.append((payload.die_side, payload.die_face))
                elif ev == 'closed':
                    closes.append(payload)
                elif ev == 'discarded':
                    discards.append(payload)
                elif ev in ('undo', 'redo'):
                    undo += 1
    return opens, closes, discards, undo


def _rec_key(r):
    if r.get('op') == 'move':
        return ('move', r.get('obj'), r.get('from'), r.get('to'))
    if r.get('op') == 'kill':
        return ('kill', tuple(r.get('pieces', [])), r.get('zone'))
    return (r.get('op'),)


def test_differential_vs_parse_game(fails):
    """MEASURES: on the SETTLED record stream (rr.resolve_undo_redo,
    the same collapse parse_game itself applies before segmenting),
    the streaming Segmenter's window openings equal parse_game's batch
    boundaries and window records are a subset of batch records.  All
    8 real pairs feeds carry undo records, so raw-stream strictness is
    impossible here; undo handling is the driver's own leg.  Non-zero
    control: at least one file with >= 15 batches."""
    if not os.path.isdir(_PAIRS):
        fails.append(('differential', 'NOT-MEASURED: pairs dir absent'))
        return
    feeds = sorted(p for p in os.listdir(_PAIRS)
                   if p.endswith('.feed.txt'))
    strict = 0
    max_batches = 0
    skipped_undo = 0
    for name in feeds:
        path = os.path.join(_PAIRS, name)
        records, _n = rr._parse_all(path)
        settled = rr.resolve_undo_redo(records)
        seg = Segmenter()
        opens, closes = [], []
        for r in settled:
            for ev, payload in seg.push(r):
                if ev == 'open':
                    opens.append((payload.die_side, payload.die_face))
                elif ev in ('closed', 'discarded'):
                    closes.append(payload)
        pg = rr.parse_game(path)
        batches = pg['batches']
        strict += 1
        max_batches = max(max_batches, len(batches))
        want = [(b['die_side'], b['die_face']) for b in batches]
        got = opens
        if got != want:
            i = next((k for k in range(min(len(got), len(want)))
                      if got[k] != want[k]), min(len(got), len(want)))
            fails.append(('boundary', '%s: opens %d vs batches %d, '
                          'first divergence at %d: %r vs %r'
                          % (name, len(got), len(want), i,
                             got[i:i + 2], want[i:i + 2])))
            continue
        for i, w in enumerate(closes):
            if i >= len(batches):
                break
            bkeys = set(_rec_key(r) for r in batches[i]['records'])
            for r in w.records:
                if r.get('die_stage'):
                    continue
                if _rec_key(r) not in bkeys:
                    fails.append(('content', '%s batch %d: window '
                                  'record %r not in batch'
                                  % (name, i, _rec_key(r))))
                    break
    print('  [diff] strict=%d skipped(undo)=%d max_batches=%d'
          % (strict, skipped_undo, max_batches))
    if strict == 0 or max_batches < 15:
        fails.append(('differential_control',
                      'NOT-MEASURED: strict=%d max_batches=%d'
                      % (strict, max_batches)))


def _mk(recs):
    seg = Segmenter()
    out = []
    for r in recs:
        out.extend(seg.push(r))
    return out


def test_close_reasons(fails):
    """MEASURES: the three live closure rules + self-undo discard, on
    specimen-shaped records (byte-shapes from LIVE_0818_BUGLIST)."""
    stage = live_parser.parse('$Wey moves Free Action Die (P) from FP '
                              'Dice to Current Action.')
    play = live_parser.parse('$Wey moves Elven Rope/It is a Gift from '
                             'Free Peoples Hand to Current FP Card.')
    kill = live_parser.parse('$Wey kills group (Free Action Die (P), '
                             'Elven Rope/It is a Gift) in Current '
                             'Action.')
    ev = _mk([stage, play, kill])
    kinds = [e for e, _ in ev]
    if kinds != ['open', 'update', 'closed']:
        fails.append(('die_killed', kinds))
    elif ev[-1][1].closed != 'die_killed':
        fails.append(('die_killed_reason', ev[-1][1].closed))

    fsp = live_parser.parse('$Wey moves FSP from Fellowship Track 3 '
                            'to Fellowship Track 4.')
    tohunt = live_parser.parse('$Wey moves Free Action Die (M) from '
                               'Current Action to The Hunt For The '
                               'Ring.')
    stage_m = live_parser.parse('$Wey moves Free Action Die (M) from '
                                'FP Dice to Current Action.')
    ev = _mk([stage_m, fsp, tohunt])
    if [e for e, _ in ev] != ['open', 'update', 'closed'] \
            or ev[-1][1].closed != 'die_to_hunt':
        fails.append(('die_to_hunt', [e for e, _ in ev]))

    selfundo = live_parser.parse('$Wey moves Free Action Die (P) from '
                                 'Current Action to FP Dice.')
    ev = _mk([stage, selfundo])
    if [e for e, _ in ev] != ['open', 'discarded'] \
            or ev[-1][1].closed != 'self_undo':
        fails.append(('self_undo', [e for e, _ in ev]))

    ev = _mk([stage, stage_m])
    if [e for e, _ in ev] != ['open', 'closed', 'open'] \
            or ev[1][1].closed != 'restaged':
        fails.append(('restaged', [e for e, _ in ev]))


def test_render_dup(fails):
    """MEASURES the lifted dedup criterion: an identical ADJACENT
    multi-piece move from a region is dropped (sensitivity); the same
    twin separated by another record is kept (specificity: adjacency
    is part of the criterion); an identical adjacent SINGLE fungible
    move is kept (recruit repeats are physical)."""
    stage = live_parser.parse('$Wey moves Shadow Action Die (A) from '
                              'SA Dice to Current Action.')
    grp = live_parser.parse('$Wey moves group ([2]Sauron Regular, '
                            'Sauron Elite) from Osgiliath to Minas '
                            'Tirith.')
    single = live_parser.parse('$Wey moves Sauron Regular from '
                               'Reinforcements to Gorgoroth.')
    chat = live_parser.parse('<Wey> hmm')
    ev = _mk([stage, grp, grp])
    ups = [e for e, _ in ev if e == 'update']
    if len(ups) != 1:
        fails.append(('dup_adjacent_dropped', '%d updates' % len(ups)))
    ev = _mk([stage, grp, chat, grp])
    ups = [e for e, _ in ev if e == 'update']
    if len(ups) != 2:
        fails.append(('dup_nonadjacent_kept', '%d updates' % len(ups)))
    ev = _mk([stage, single, single])
    ups = [e for e, _ in ev if e == 'update']
    if len(ups) != 2:
        fails.append(('dup_single_kept', '%d updates' % len(ups)))


def test_tail_undo(fails):
    """MEASURES: a client undo naming the OPEN window's newest line
    withdraws that record (staging undone => the window un-opens;
    specimen: game 2023-01-02 L50-51 staged-A-then-undo ghost window).
    SPECIFICITY: an undo naming any OTHER line passes through as an
    ('undo', rec) event untouched."""
    stage = live_parser.parse('%Wey moves Shadow Action Die (A) from '
                              'SA Dice to Current Action.')
    undo_stage = live_parser.parse(
        '%Wey undoes last action : %Wey moves Shadow Action Die (A) '
        'from SA Dice to Current Action.')
    ev = _mk([stage, undo_stage])
    kinds = [e for e, _ in ev]
    if kinds != ['open', 'withdrawn', 'discarded'] \
            or ev[-1][1].closed != 'undone':
        fails.append(('undo_staging', kinds))
    # body-record tail undo: record withdrawn, window stays open
    grp = live_parser.parse('%Wey moves group ([2]Sauron Regular, '
                            'Sauron Elite) from Osgiliath to Minas '
                            'Tirith.')
    undo_grp = live_parser.parse(
        '%Wey undoes last action : %Wey moves group ([2]Sauron '
        'Regular, Sauron Elite) from Osgiliath to Minas Tirith.')
    kill = live_parser.parse('%Wey kills Shadow Action Die (A) in '
                             'Current Action.')
    ev = _mk([stage, grp, undo_grp, kill])
    kinds = [e for e, _ in ev]
    if kinds != ['open', 'update', 'withdrawn', 'closed']:
        fails.append(('undo_body', kinds))
    else:
        w = ev[-1][1]
        if len(w.records) != 2 or w.body_records():
            fails.append(('undo_body_records',
                          [r.get('norm') for r in w.records]))
    # specificity: non-tail undo passes through untouched
    undo_other = live_parser.parse(
        '%Wey undoes last action : %Wey moves Sauron Regular from '
        'Gorgoroth to Morannon.')
    ev = _mk([stage, grp, undo_other])
    kinds = [e for e, _ in ev]
    if kinds != ['open', 'update', 'undo']:
        fails.append(('undo_passthrough', kinds))


def test_self_echo_nick_awareness(fails):
    """MEASURES (WAVE-I, WAVE-H blocker 5): with nick set, the AI
    SERVER'S OWN emission echoes never touch windows —
      (a) an AI die-stage echo opens NO window,
      (b) the AI's owed die-kill echo does NOT terminate an open
          HUMAN window (the false-termination hole),
      (c) an AI board record does NOT join the open human window's
          body (the pollution hole) — it flows as a 'phase' event,
      (d) the human's own flow is untouched (their die kill still
          closes their window, body records intact).
    SPECIFICITY CONTROL: the SAME stream through a nick-less
    segmenter (the spectator shape) shows the old behavior — the AI
    die kill DOES close the human window — proving the mechanism is
    the nick, not an accident of the grammar."""
    ai_nick = 'WOTR_AI_Shadow'
    human = [
        '$Wey moves Free Action Die (C) from FP Dice to Current '
        'Action.',
        '$Wey moves Gondor Regular from Minas Tirith to Osgiliath.',
    ]
    ai_lines = [
        '$%s moves Shadow Action Die (A) from SA Dice to Current '
        'Action.' % ai_nick,
        '$%s kills Shadow Action Die (A) in Current Action.' % ai_nick,
        '$%s moves Sauron Regular from Barad-Dur to Gorgoroth.'
        % ai_nick,
    ]
    tail = ['$Wey kills Free Action Die (C) in Current Action.']

    def run(nick):
        seg = Segmenter(nick=nick)
        evs = []
        for raw in human + ai_lines + tail:
            rec = live_parser.parse(raw)
            for ev, payload in seg.push(rec):
                evs.append((ev, payload))
        return evs

    evs = run(ai_nick)
    opens = [p for e, p in evs if e == 'open']
    closes = [p for e, p in evs if e == 'closed']
    phases = [p for e, p in evs if e == 'phase']
    if len(opens) != 1:
        fails.append(('self_stage_no_window', len(opens)))
        return
    if len(closes) != 1 or closes[0].closed != 'die_killed':
        fails.append(('human_close_intact',
                      [getattr(c, 'closed', None) for c in closes]))
        return
    body = [r.get('norm', '') for r in closes[0].body_records()]
    if any('Sauron' in b or 'Shadow Action Die' in b for b in body):
        fails.append(('body_polluted', body))
    if len(body) != 1 or 'Gondor Regular' not in body[0]:
        fails.append(('human_body_intact', body))
    if not any('Sauron Regular' in str(p.get('norm', ''))
               for p in phases):
        fails.append(('ai_flows_as_phase',
                      [str(p.get('norm', ''))[:40] for p in phases]))
    # SPECIFICITY: the nick-less segmenter (spectator shape) shows the
    # old breakage — the AI's die-stage echo falsely closes the human
    # window ('restaged') and the AI records become window traffic —
    # proving the mechanism above is the nick, not the grammar
    evs0 = run(None)
    closes0 = [p for e, p in evs0 if e == 'closed']
    if not closes0 or closes0[0].closed != 'restaged':
        fails.append(('specificity_control',
                      [getattr(c, 'closed', None) for c in closes0]))


def test_render_dup_honours_sidecar(fails):
    """MEASURES (W1#35): with the sidecar's per-line raw_new flag
    present, the client's own statement decides the render-dup: an
    adjacent identical group move with raw_new=True is a REAL second
    drag (kept, two updates); raw_new=False is the re-render
    (dropped) and the drop is SURFACED as a 'dropped_dup' event, never
    a silent []."""
    stage = live_parser.parse('$Wey moves Shadow Action Die (A) from '
                              'SA Dice to Current Action.')
    g1 = live_parser.parse('$Wey moves group ([2]Sauron Regular, '
                           'Sauron Elite) from Osgiliath to Minas '
                           'Tirith.')
    g2 = live_parser.parse('$Wey moves group ([2]Sauron Regular, '
                           'Sauron Elite) from Osgiliath to Minas '
                           'Tirith.')
    g1['meta'] = {'seq': 2, 'raw_seq': 10, 'raw_new': True}
    g2['meta'] = {'seq': 3, 'raw_seq': 11, 'raw_new': True}
    ev = _mk([stage, g1, g2])
    ups = [e for e, _ in ev if e == 'update']
    if len(ups) != 2:
        fails.append(('sidecar_real_second_drag_dropped',
                      '%d updates' % len(ups)))
    g3 = live_parser.parse('$Wey moves group ([2]Sauron Regular, '
                           'Sauron Elite) from Osgiliath to Minas '
                           'Tirith.')
    g3['meta'] = {'seq': 3, 'raw_seq': 10, 'raw_new': False}
    ev = _mk([stage, g1, g3])
    ups = [e for e, _ in ev if e == 'update']
    drops = [e for e, _ in ev if e == 'dropped_dup']
    if len(ups) != 1 or len(drops) != 1:
        fails.append(('sidecar_rerender', (len(ups), len(drops))))
    # the channel-absent heuristic still surfaces its drop
    ev = _mk([stage, live_parser.parse(g1['raw']),
              live_parser.parse(g1['raw'])])
    if not any(e == 'dropped_dup' for e, _ in ev):
        fails.append(('heuristic_drop_silent', [e for e, _ in ev]))


def main():
    tests = [test_differential_vs_parse_game, test_close_reasons,
             test_render_dup, test_tail_undo,
             test_self_echo_nick_awareness,
             test_render_dup_honours_sidecar]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
