"""Standalone ring counter waits for its selected physical die change.

RB21 and Almanac31-32: a ring changes one unused result. Counter-only
input remains incomplete; once paired, the same player still acts next.
The old waiver assertion is preserved in the ring audit baseline.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_state import Side

RING_LINE = '$Wey moves Nenya from Elven Ring 2 to Dark Ring 2.'


def _presses(items):
    return sum(1 for i in items if i[0] == 'line'
               and '<~Controls.819~>' in str(i[1]))


def test_standalone_ring_completes_after_face_on_own_turn(fails):
    """Counter alone is held; explicit C->H completes one ring use.
    Same-player continuation and second-use refusal remain required."""
    import ai
    import game_logic as gl
    import test_live_driver as TD
    st = TD.adopted_state()
    rig = TD.Rig(st, Side.SHADOW, phase='actions', seed=5)   # human=FP
    drv = rig.drv
    drv.last_actor = Side.SHADOW            # FP to act
    rings0 = st.fp_elven_rings
    items = rig.feed(RING_LINE)
    if drv.es.fp_elven_rings != rings0 or drv.n_commits_total:
        fails.append(('bare_ring_chose_face', drv.es.fp_elven_rings))
    items += rig.feed('$Wey changes Free Action Die (C) to H.')
    es = drv.es
    if es.fp_elven_rings != rings0 - 1:
        fails.append(('ring_not_spent', (rings0, es.fp_elven_rings)))
    from game_state import ActionDieFace
    if es.dice[Side.FREE_PEOPLES][0].current != ActionDieFace.MUSTER_ARMY:
        fails.append(('selected_face_not_changed', es.dice[Side.FREE_PEOPLES][0].current))
    if gl.ring_waiver_active(es, Side.FREE_PEOPLES):
        fails.append(('obsolete_waiver', 'ring must change the physical face'))
    if drv.last_actor is not Side.FREE_PEOPLES \
            or ai._next_acting_side(es, drv.last_actor) \
            is not Side.FREE_PEOPLES:
        fails.append(('alternation_flipped', drv.last_actor))
    if any(i[0] == 'chat' and i[1] == 'refusal' for i in items) \
            or _presses(items):
        fails.append(('ring_refused_on_own_turn', items))
    if drv.alarms or drv.uncommitted:
        fails.append(('ring_alarm', (drv.alarms, drv.uncommitted)))
    if not any(f[0] == 'ring_standalone' for f in drv.findings):
        fails.append(('no_finding', drv.findings[-3:]))
    # a second ring the same turn: refused, one undo press
    items = rig.feed('$Wey moves Vilya from Elven Ring 1 to Dark Ring 1.')
    if es.fp_elven_rings != drv.es.fp_elven_rings:
        fails.append(('second_ring_spent', drv.es.fp_elven_rings))
    if not any(i[0] == 'chat' and i[1] == 'refusal' for i in items):
        fails.append(('second_ring_not_refused', items))
    if _presses(items) != 1:
        fails.append(('second_ring_presses', _presses(items)))


def test_standalone_ring_out_of_turn_refused(fails):
    """MEASURES: the same walk during the AI's alternation -> refused
    ('not your action turn'), one undo press, engine untouched."""
    import test_live_driver as TD
    st = TD.adopted_state()
    rig = TD.Rig(st, Side.SHADOW, phase='actions', seed=5)   # human=FP
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES      # the AI (Shadow) to act
    rings0 = st.fp_elven_rings
    items = rig.feed(RING_LINE)
    if drv.es.fp_elven_rings != rings0:
        fails.append(('ring_spent_out_of_turn', drv.es.fp_elven_rings))
    if not any(i[0] == 'chat' and i[1] == 'refusal'
               and 'not your action turn' in str(i[2]) for i in items):
        fails.append(('no_turn_refusal', items))
    if _presses(items) != 1:
        fails.append(('presses', _presses(items)))
    if drv.pending_buf is not None:
        fails.append(('buffered', 'ring walk landed in the pending '
                      'buffer'))


def main():
    tests = [test_standalone_ring_completes_after_face_on_own_turn,
             test_standalone_ring_out_of_turn_refused]
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
