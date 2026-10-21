# -*- coding: ascii -*-
"""CLIENT-WAVE sidecar adoption instruments (WAVE-J).

MEASURED QUANTITIES per leg; the driver legs run the REAL ingest
(feed file + meta file -> tick) so a contract drift in either channel
fails the test instead of testing nothing.
"""
import json
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_state import Side

from wotr_bridge.live import sidecar as sc_mod
from test_live_driver import Rig, fresh_state


def _write_meta(path, recs):
    with open(path, 'a', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')


def test_reader(fails):
    """MEASURES: seq-keyed reading, truncation re-read, and the
    BLINDING invariant - 'redacted' pids are NEVER a value
    (meta_pids None; no bypass around the feed's blinding)."""
    import tempfile
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'f.meta.jsonl')
    r = sc_mod.SidecarReader(p)
    if r.poll() != 0 or r.present:
        fails.append(('reader_absent', (r.present,)))
    _write_meta(p, [
        {'seq': 1, 'gen': 1, 'nick': 'Wey', 'tags': [],
         'pids': [], 'ts': 1},
        {'seq': 2, 'gen': 1, 'nick': 'Wey',
         'tags': ['Controls.491'], 'pids': 'redacted',
         'side': 'FP', 'ts': 2},
        {'seq': 3, 'gen': 1, 'nick': 'Wey',
         'tags': ['Controls.345', 'Controls.346'],
         'pids': ['100', '101', '13', '11'],
         'segs': [[], ['100', '101'], ['13', '11']], 'ts': 3}])
    n = r.poll()
    if n != 3 or not r.present:
        fails.append(('reader_read', (n, r.present)))
    if sc_mod.meta_pids(r.rec(2)) is not None:
        fails.append(('reader_redacted',
                      'redacted pids surfaced as a value'))
    if sc_mod.roll_pids(r.rec(3)) != ['100', '101']:
        fails.append(('reader_roll_channel',
                      sc_mod.roll_pids(r.rec(3))))
    if sc_mod.meta_side(r.rec(2)) is not Side.FREE_PEOPLES:
        fails.append(('reader_side', sc_mod.meta_side(r.rec(2))))
    # truncation: shrink + rewrite -> re-read idempotently
    with open(p, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'seq': 1, 'gen': 2, 'nick': 'Wey',
                            'tags': [], 'pids': [], 'ts': 9}) + '\n')
    r.poll()
    if r.rec(1) is None or r.rec(1).get('gen') != 2:
        fails.append(('reader_truncate', r.rec(1)))


def test_roll_binding_from_meta(fails):
    """MEASURES (raw import #2 RETIRED under the sidecar): the
    letters roll binds die pids from the meta DIE CHANNEL (segs[1]),
    with NO raw request line at all; and a raw 345/346 request line
    is IGNORED while the channel is present (one answerer - the
    heuristic dies where the sidecar answers)."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='roll')
    drv = rig.drv
    meta_p = rig.path + '.meta.jsonl'
    drv.sidecar = sc_mod.SidecarReader(meta_p)
    # REVERSED pool order (cbt3: the order provably changes across
    # turns) - the binding must OVERWRITE the constructor pre-seed
    # (099->0...), or this leg measures the pre-seed, not the channel
    # (the first M16 run proved exactly that vacuity)
    pids = ['105', '104', '103', '102', '101', '100', '099']
    _write_meta(meta_p, [
        {'seq': 1, 'gen': 1, 'nick': 'Wey',
         'tags': ['Controls.345', 'Controls.346'],
         'pids': pids + ['12', '13', '11', '10', '12', '13', '11'],
         'segs': [[], pids,
                  ['12', '13', '11', '10', '12', '13', '11']],
         'ts': 1}])
    drv.tick([])             # sidecar becomes present
    rig.feed('$Wey rolls A C H M A C H .')
    ent0 = drv.view.die_id_map.get('105')
    ent6 = drv.view.die_id_map.get('099')
    if not ent0 or int(ent0[1]) != 0 or not ent6 \
            or int(ent6[1]) != 6:
        fails.append(('meta_bind',
                      'reversed order not adopted: 105->%r 099->%r '
                      '(pre-seed measured instead of the channel?)'
                      % (ent0, ent6)))
    # a LATE raw 345/346 request line (AiOpp resends the whole log
    # every tick): with the sidecar present it must be IGNORED - a
    # parked stale pid list would poison the NEXT adoption (the exact
    # hazard the retirement guard kills; a mid-tick raw line is
    # harmlessly overwritten, so THIS ordering is the measuring one)
    raw = ('LOG<~Controls.345~>200 201 202 203 204 205 206 '
           '<~Controls.346~>12 13 11 10 12 13 11 ABCDEF')
    drv.tick([raw])
    if drv._pending_roll_pids:
        fails.append(('raw_lane_retired',
                      'a late raw roll line parked pids while the '
                      'sidecar was present: %r'
                      % (drv._pending_roll_pids,)))


def test_no_die_channel_alarm(fails):
    """MEASURES (WAVE-K R6 / ledger S10): with the sidecar PRESENT, a
    letters-roll record that carries NO die channel (segs[1]) leaves
    the adoption unbound - this must be a LOUD alarm within a few
    ticks, never silence.  Control (specificity): the same roll WITH
    the channel binds and raises no such alarm."""
    def run(segs):
        st = fresh_state()
        rig = Rig(st, Side.SHADOW, phase='roll')
        drv = rig.drv
        meta_p = rig.path + '.meta.jsonl'
        drv.sidecar = sc_mod.SidecarReader(meta_p)
        _write_meta(meta_p, [
            {'seq': 1, 'gen': 1, 'nick': 'Wey',
             'tags': ['Controls.345', 'Controls.346'],
             'pids': [], 'segs': segs, 'ts': 1}])
        drv.tick([])             # sidecar becomes present
        rig.feed('$Wey rolls A C H M A C H .')
        for _ in range(4):
            drv.tick([])
        return drv
    drv = run([[], [], ['12', '13', '11', '10', '12', '13', '11']])
    hits = [a for a in drv.alarms if 'ROLL PID BINDING STARVED' in a]
    if not hits:
        fails.append(('no_channel_silent',
                      'sidecar present, no segs[1], no alarm: %r'
                      % (drv.alarms,)))
    pids = ['105', '104', '103', '102', '101', '100', '099']
    drv2 = run([[], pids, ['12', '13', '11', '10', '12', '13', '11']])
    if any('ROLL PID BINDING STARVED' in a for a in drv2.alarms):
        fails.append(('channel_specificity', drv2.alarms))


def test_pid_true_tracking(fails):
    """MEASURES (adoption item c): meta pids give pid-TRUE view
    bookkeeping - a move record updates piece_region for its exact
    pids; a kill removes them; a REDACTED record updates NOTHING
    (blinding fail-closed)."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    meta_p = rig.path + '.meta.jsonl'
    drv.sidecar = sc_mod.SidecarReader(meta_p)
    _write_meta(meta_p, [
        {'seq': 1, 'gen': 1, 'nick': 'Wey',
         'tags': ['Scribe.141'], 'pids': ['061', '062'],
         'segs': [[], ['061', '062']], 'side': 'FP', 'ts': 1},
        {'seq': 2, 'gen': 1, 'nick': 'Wey',
         'tags': ['Controls.296'], 'pids': ['061'],
         'segs': [[], ['061']], 'ts': 2},
        {'seq': 3, 'gen': 1, 'nick': 'Wey',
         'tags': ['Controls.491'], 'pids': 'redacted',
         'side': 'FP', 'ts': 3}])
    rig.feed('$Wey moves group (Dwarven Regular, Dwarven Elite) '
             'from Erebor to Iron Hills.')
    if drv.view.piece_region.get('061') != 'Iron Hills' \
            or drv.view.piece_region.get('062') != 'Iron Hills':
        fails.append(('meta_move_track',
                      {p: drv.view.piece_region.get(p)
                       for p in ('061', '062')}))
    rig.feed('$Wey kills Dwarven Regular in Iron Hills.')
    if drv.view.piece_region.get('061') != 'Dead':
        fails.append(('meta_kill_track',
                      '061 must be marked Dead (never popped - an '
                      'untracked pid falls back to its setup zone): %r'
                      % drv.view.piece_region.get('061')))
    before = dict(drv.view.piece_region)
    rig.feed('$Wey draws Free Character Card.')
    if drv.view.piece_region != before:
        fails.append(('meta_redacted_track',
                      'a redacted record changed the view'))


def test_gen_replay_fields(fails):
    """MEASURES (W1#19): the reader honours the agent's gen/replay
    fields - gen 1 with replay=true paragraphs is the normal first
    attach (history the driver must process); a LATER gen is a
    re-attach (a new chat document) reported as gen_change with the
    seq it began at; replay(seq) answers per record."""
    import tempfile
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'f.meta.jsonl')
    r = sc_mod.SidecarReader(p)
    _write_meta(p, [
        {'seq': 1, 'gen': 1, 'replay': True, 'tags': [], 'pids': [],
         'ts': 1},
        {'seq': 2, 'gen': 1, 'raw_seq': 3, 'raw_new': True,
         'nick': 'Wey', 'tags': [], 'pids': [], 'ts': 2}])
    r.poll()
    if r.gen != 1 or r.gen_change is not None:
        fails.append(('gen_first_attach', (r.gen, r.gen_change)))
    if r.replay(1) is not True or r.replay(2) is not False:
        fails.append(('replay_flag', (r.replay(1), r.replay(2))))
    _write_meta(p, [
        {'seq': 3, 'gen': 2, 'replay': True, 'tags': [], 'pids': [],
         'ts': 3},
        {'seq': 4, 'gen': 2, 'replay': True, 'nick': 'Wey',
         'tags': [], 'pids': [], 'ts': 4}])
    r.poll()
    if r.gen != 2 or r.gen_change != (3, 1, 2):
        fails.append(('gen_reattach', (r.gen, r.gen_change)))
    # absent fields: never a crash, never a phantom re-attach
    _write_meta(p, [{'seq': 5, 'tags': [], 'pids': [], 'ts': 5}])
    r.poll()
    if r.gen != 2 or r.replay(5) is not False:
        fails.append(('gen_absent_fields', (r.gen, r.replay(5))))


def test_attestation_sets_no_baseline(fails):
    """MEASURES (real-client leg fw1cbt 2026-08-22, NO PROGRESS): the
    agent's premain attestation record (seq 1, gen 0, attest=true)
    precedes its FIRST chat-document attach (gen 1).  That pair is the
    normal boot, never a re-attach; a later gen change still is.
    Specimen = the leg's own ai_feed.txt.meta.jsonl head."""
    import tempfile
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'f.meta.jsonl')
    _write_meta(p, [
        {'seq': 1, 'gen': 0, 'attest': True, 'tags': [], 'pids': [],
         'ts': 1},
        {'seq': 2, 'gen': 1, 'replay': True, 'tags': [], 'pids': [],
         'ts': 2},
        {'seq': 3, 'gen': 1, 'raw_seq': 28, 'raw_new': True,
         'nick': 'base', 'tags': [], 'pids': [], 'ts': 3},
    ])
    r = sc_mod.SidecarReader(p)
    r.poll()
    if r.gen_change is not None:
        fails.append(('boot_attach_is_not_reattach', r.gen_change))
    if r.gen != 1:
        fails.append(('baseline_gen', r.gen))
    # sensitivity: a REAL second attach still registers
    _write_meta(p, [
        {'seq': 4, 'gen': 2, 'replay': True, 'tags': [], 'pids': [],
         'ts': 4},
    ])
    r.poll()
    if r.gen_change != (4, 1, 2):
        fails.append(('real_reattach_missed', r.gen_change))


def main():
    tests = [test_reader, test_gen_replay_fields,
             test_roll_binding_from_meta,
             test_no_die_channel_alarm,
             test_pid_true_tracking, test_attestation_sets_no_baseline]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
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
