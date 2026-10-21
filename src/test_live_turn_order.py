"""SCENARIO_DEFECTS D2 pin (real-client reproduction
`build_ambiguous_sa_rec1`: a second human die window 1 s after the
first, while the engine's alternation named the AI, was MATCHED and
COMMITTED - the AI then took two actions in a row (SA, SA, FP, FP),
no alarm, no note).

Rulebook p.20: the players ALTERNATE actions (the side with fewer dice
may pass / the side with dice left continues once the other is out).
Sided ruling 5: an illegal gesture is refused with the client-native
undo + a notice.  ONE answerer (ai._next_acting_side over last_actor)
shared by the window lane and the two chat lanes.
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


def _undo_presses(items):
    return sum(1 for i in items if i[0] == 'line'
               and '<~Controls.819~>' in str(i[1]))


def test_second_window_out_of_turn_is_refused(fails):
    """MEASURES: human SHADOW (AI FP) commits one army window (the
    AI's reply owed); a second die window before the AI acts is
    REFUSED - 'it is not your action turn' + undo presses - never
    matched, last_actor untouched.  Control: with the AI out of dice
    the same second window is NOT refused for turn order (the engine
    says the side with dice continues)."""
    import test_live_driver as TD
    st = TD.adopted_state()
    a, lines = TD._army_window_lines(st)
    if a is None:
        fails.append(('turn_control', 'NOT-MEASURED: no army move'))
        return
    rig = TD.Rig(st, Side.FREE_PEOPLES, phase='actions', seed=5)
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES        # the AI just acted
    rig.feed(*lines)
    if drv.window_stats['matched'] != 1 or drv.last_actor is not Side.SHADOW:
        fails.append(('turn_control', 'NOT-MEASURED: first window not '
                      'committed: %r last_actor=%r'
                      % (drv.window_stats, drv.last_actor)))
        return
    m0 = drv.window_stats['matched']
    items = rig.feed('$Wey moves Shadow Action Die (C) from SA Dice to '
                     'Current Action.',
                     '$Wey kills Shadow Action Die (C) in Current Action.')
    if drv.window_stats['matched'] != m0:
        fails.append(('second_window_matched', drv.window_stats))
    if drv.last_actor is not Side.SHADOW:
        fails.append(('alternation_reseeded', drv.last_actor))
    if not any(i[0] == 'chat' and 'not your action turn' in str(i[2])
               for i in items):
        fails.append(('no_turn_refusal', items))
    if not any(f[0] == 'window_out_of_turn' for f in drv.findings):
        fails.append(('no_finding', drv.findings[-3:]))
    if _undo_presses(items) < 1:
        fails.append(('no_undo_press', items))
    if drv.holds:
        fails.append(('held_not_refused', drv.holds[-1][1].note))
    # control: the AI is out of dice -> the human continues (engine
    # answer), no turn-order refusal
    st2 = TD.adopted_state()
    for d in st2.dice[Side.FREE_PEOPLES]:
        d.used = True
    a2, lines2 = TD._army_window_lines(st2)
    rig2 = TD.Rig(st2, Side.FREE_PEOPLES, phase='actions', seed=5)
    drv2 = rig2.drv
    drv2.last_actor = Side.FREE_PEOPLES
    rig2.feed(*lines2)
    if drv2.window_stats['matched'] != 1:
        fails.append(('control_first', drv2.window_stats))
        return
    items = rig2.feed('$Wey moves Shadow Action Die (C) from SA Dice to '
                      'Current Action.',
                      '$Wey kills Shadow Action Die (C) in Current '
                      'Action.')
    if any(i[0] == 'chat' and 'not your action turn' in str(i[2])
           for i in items) \
            or any(f[0] == 'window_out_of_turn' for f in drv2.findings):
        fails.append(('control_refused', items))


def main():
    tests = [test_second_window_out_of_turn_is_refused]
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
