"""GUIDE DESYNC (USER live catch 2026-08-21): "strider is guide, it
says illegal move ... the engine doesn't see strider is guide, so the
resync + reset doesn't work".

Mechanism, reproduced before patching:
  * The Guide is chosen by the FP player, but setup breaks the
    Gandalf-the-Grey / Strider level-3 tie SILENTLY - change_guide
    takes best[0] and never opens pending_guide_change.
  * A human who guides with Strider is therefore desynced from turn 1.
  * ai.py:7334 grants "hide the Fellowship using ANY Action die" only
    when `'Strider' in guide.name`, so every such hide is refused as
    an illegal move.
  * resync could not repair it: resync_recover adopts the client board
    into the VIEW and leaves ENGINE state untouched by design (I1).

Every assertion below is read off the ENGINE, never off the view.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ai                                  # noqa: E402
import log_replay as lr                    # noqa: E402
from game_state import ActionDieFace, Side  # noqa: E402

from test_live_driver import Rig, fresh_state   # noqa: E402

GUIDE_ZONE = 'Guide of the Fellowship'


def pid_named(name):
    for pid, nm in lr.COMPANIONS.items():
        if nm == name or name in (lr.names_for_pid(pid) or ()):
            return pid
    raise AssertionError('no pid for %r' % name)


def muster_hide_legal(st):
    """Is a Muster-die hide on the menu?  This is the exact question
    the human's gesture asked and the stack answered 'illegal'."""
    for d in st.dice[Side.FREE_PEOPLES]:
        d.used = False
        d.current = ActionDieFace.MUSTER
    st.fellowship.hidden = False
    return any(a.action_type == 'fellowship_hide'
               for a in ai.get_legal_actions(st, Side.FREE_PEOPLES))


def rig_showing(guide_name):
    """A driver whose CLIENT dump shows `guide_name` in the Guide box."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.piece_region = {str(pid_named(guide_name)): GUIDE_ZONE}
    return rig, st


def test_repro_strider_guide_is_invisible_to_the_engine():
    st = fresh_state()
    assert st.fellowship.guide.name == 'Gandalf the Grey', \
        'setup no longer defaults to best[0]; retune this test'
    assert st.pending_guide_change is None, \
        'setup now ASKS for the guide - the desync class is gone'
    assert not muster_hide_legal(st), \
        'REPRO FAILED: Muster hide already legal without Strider'


def test_reconcile_adopts_the_humans_guide_and_unblocks_the_hide():
    rig, st = rig_showing('Strider')
    assert not muster_hide_legal(st)          # broken before
    note = rig.drv.reconcile_guide_from_client()
    assert note and 'Strider' in note, note
    assert st.fellowship.guide.name == 'Strider', \
        st.fellowship.guide.name
    assert muster_hide_legal(st), \
        'guide adopted but the any-die hide is still refused'


def test_control_agreeing_client_changes_nothing():
    """SENSITIVITY CONTROL: without this column a reconcile that
    ALWAYS rewrote the guide would pass the test above."""
    rig, st = rig_showing('Gandalf the Grey')
    assert rig.drv.reconcile_guide_from_client() is None
    assert st.fellowship.guide.name == 'Gandalf the Grey'
    assert not muster_hide_legal(st)


def test_control_illegal_pick_is_refused_not_forced():
    """Legolas is level 2 - not tied for top, so the rules forbid him.
    The engine must keep its guide and SAY so (I1: we ingest a
    decision, we do not overwrite engine state)."""
    rig, st = rig_showing('Legolas')
    note = rig.drv.reconcile_guide_from_client()
    assert note and 'refuse' in note.lower(), note
    assert st.fellowship.guide.name == 'Gandalf the Grey'
    assert any(k == 'guide_reconcile_refused'
               for k, _ in rig.drv.findings)


def test_control_empty_and_ambiguous_guide_box_leave_engine_alone():
    rig, st = rig_showing('Strider')
    rig.drv.view.piece_region = {}                       # empty
    assert rig.drv.reconcile_guide_from_client() is None
    rig.drv.view.piece_region = {                        # two occupants
        str(pid_named('Strider')): GUIDE_ZONE,
        str(pid_named('Gandalf the Grey')): GUIDE_ZONE}
    assert rig.drv.reconcile_guide_from_client() is None
    assert st.fellowship.guide.name == 'Gandalf the Grey'


def test_mutation_disabling_the_fix_restores_the_bug():
    """MUTATION TEST: break the reader and the green legs above must
    go RED.  A leg that passes with the fix disabled proves nothing."""
    rig, st = rig_showing('Strider')
    rig.drv.guide_from_client = lambda: None      # the mutation
    assert rig.drv.reconcile_guide_from_client() is None
    assert st.fellowship.guide.name == 'Gandalf the Grey'
    assert not muster_hide_legal(st), \
        'MUTATION NOT DETECTED: hide legal with the fix disabled'


def test_resync_recover_carries_the_guide_fix():
    """The USER pressed RESYNC and it did not help - so the lever, not
    just the helper, has to repair the guide.  The lever adopts ONLY
    a full client dump (adopt refuses a torn/partial read, W1#17), so
    the leg runs on the USER's real 428-piece dump fixture."""
    import os
    from wotr_bridge.live import client_board as cb
    fx = os.path.join(_ROOT, 'data', 'live_fixtures',
                      'client_board_20260821_strider_guide.txt')
    assert os.path.exists(fx), ('NOT-MEASURED: the real dump fixture is '
                                'missing: %s' % fx)
    dump = cb.read_board(fx)
    assert dump and len(dump) >= cb.MIN_DUMP_PIECES, \
        'the real dump must survive read_board as a FULL board'
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.piece_region = dump
    import shutil
    from bridge_evidence_fixture import snapshot_header
    path = os.path.join(rig.td, 'board.txt')
    shutil.copy2(fx, path)
    rig.drv.view.dump_path = path
    snapshot_header(rig.drv)
    items = rig.drv.resync_recover(drain_feed=False)
    notes = ' | '.join(str(v) for k, v in items if k == 'note')
    # the lever's OUTCOME is the measure: the engine guide is the one
    # the client shows (by the guide hook's note, or recomputed by the
    # companion adoption when the old guide had left the Fellowship)
    assert 'Strider' in notes or 'guide recomputed' in notes, notes
    assert st.fellowship.guide.name == 'Strider', \
        (st.fellowship.guide.name, notes)
    assert muster_hide_legal(st)


def test_real_client_dump_from_the_reported_game():
    """STRONGEST LEG: the USER's ACTUAL board dump (2026-08-21, the
    game they reported), not a fixture I invented.

    This caught a real hole.  The synthetic legs above set a readable
    zone name, but the client writes OPAQUE CODES - the raw dump says

        <~Game.1999~>   241   U

    and only log_replay.decode_location turns Game.1999 into
    'Guide Box'.  Had the codes not decoded, every green leg above
    would have been vacuous while the fix did nothing on real data.
    """
    import os
    from wotr_bridge.live import client_board as cb
    fx = os.path.join(_ROOT, 'data', 'live_fixtures',
                      'client_board_20260821_strider_guide.txt')
    assert os.path.exists(fx), ('NOT-MEASURED: the real dump fixture is '
                                'missing: %s' % fx)
    dump = cb.read_board(fx)
    assert dump, 'the real dump must survive read_board (428 pieces)'
    assert dump.get(str(pid_named('Strider'))) == 'Guide Box',         dump.get(str(pid_named('Strider')))

    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.piece_region = dump
    assert not muster_hide_legal(st)
    note = rig.drv.reconcile_guide_from_client()
    assert note and 'Strider' in note, note
    assert st.fellowship.guide.name == 'Strider'
    assert muster_hide_legal(st),         'real dump adopted but the any-die hide is still refused'


if __name__ == '__main__':
    fails = 0
    for nm, fn in sorted(globals().items()):
        if not nm.startswith('test_'):
            continue
        try:
            fn()
            print('PASS %s' % nm)
        except Exception as e:
            fails += 1
            print('FAIL %s: %r' % (nm, e))
    print('\n%s' % ('ALL PASS' if not fails else '%d FAILED' % fails))
    raise SystemExit(1 if fails else 0)
