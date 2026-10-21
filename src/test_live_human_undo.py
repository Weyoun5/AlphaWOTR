"""SCENARIO_DEFECTS D-T2-2 pin (real-client reproduction
`undo_refusal_and_human_undo_r1b`: the human pressed Undo three times
40 ms after its die window committed; the AI's reply to that window
had already been booked (same poll cycle), so the ledger walk found
the AI's LINELESS entry on top, said 'undo reached a chat-decided
action' and changed nothing - the client retracted the window, the
engine kept it, resync could not repair it).

Doctrine pinned: the ENGINE FOLLOWS THE CLIENT.  A human window
retracted line by line beneath the AI's lineless replies pops to its
pre-state and every AI commit above it is RE-BASED (re-applied) onto
the restored state in order; the AI's reply stands (its lines are on
the client), the alternation is the AI's last action, the human is
to act.  A reply that cannot re-apply (illegal / needs randomness) is
a LOUD alarm naming it - never a silent divergence.
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


def test_undo_beneath_ai_reply_rebases(fails):
    """MEASURES: human SHADOW window (stage A / army move / kill)
    committed; the AI (FP) reply booked lineless on top; the three
    undo echoes (kill, move, stage) -> the human's move is gone from
    the engine, its die unused, the AI's move still applied, the AI's
    die spent, last_actor FP, no alarm, finding undo_rebased."""
    import ai
    import test_live_driver as TD
    from board import sanitize
    st = TD.adopted_state()
    a, lines = TD._army_window_lines(st)
    if a is None:
        fails.append(('undo_control', 'NOT-MEASURED: no army move'))
        return
    rig = TD.Rig(st, Side.FREE_PEOPLES, phase='actions', seed=5)
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES
    rig.feed(*lines)
    if drv.window_stats['matched'] != 1:
        fails.append(('undo_control', 'NOT-MEASURED: window not '
                      'committed %r' % (drv.window_stats,)))
        return
    post_h = drv.es
    # the AI's reply: a plain FP army move on the post-human state
    menu = ai.get_legal_actions(post_h, Side.FREE_PEOPLES)
    ai_act = next((x for x in menu if x.action_type == 'army_move'
                   and sanitize(str(x.params.get('from')))
                   != sanitize(str(a.params.get('to')))), None)
    if ai_act is None:
        fails.append(('undo_control', 'NOT-MEASURED: no FP army move'))
        return
    post_ai = ai.apply_action(post_h, ai_act)
    drv._snapshot([], 'ai:action', post=post_ai, action=ai_act)
    drv.es = post_ai
    drv.hands.es = post_ai
    drv.last_actor = Side.FREE_PEOPLES
    # the human's three presses, LIFO per line
    items = []
    for ln in reversed(lines):
        items += rig.feed('$Wey undoes last action : %s' % ln)
    es = drv.es
    if any('lineless' in str(x) for x in drv.alarms) \
            or any('chat-decided' in str(i) for i in items):
        fails.append(('undo_lineless_hold', (drv.alarms[-1:], items)))
    if es.dice[Side.SHADOW][a.die_idx].used:
        fails.append(('human_die_still_spent', a.die_idx))
    frm = sanitize(str(a.params.get('from')))
    to = sanitize(str(a.params.get('to')))
    n_from_pre = sum(1 for k, ar in st.armies.items()
                     if sanitize(k) == frm for _u in ar.units)
    n_from_now = sum(1 for k, ar in es.armies.items()
                     if sanitize(k) == frm for _u in ar.units)
    if n_from_now != n_from_pre:
        fails.append(('human_move_not_retracted', (frm, n_from_pre,
                                                   n_from_now)))
    if not es.dice[Side.FREE_PEOPLES][ai_act.die_idx].used:
        fails.append(('ai_die_not_spent', 'the AI reply was dropped'))
    ai_to = sanitize(str(ai_act.params.get('to')))
    n_ai_to = sum(1 for k, ar in es.armies.items()
                  if sanitize(k) == ai_to for _u in ar.units)
    n_ai_to_pre = sum(1 for k, ar in st.armies.items()
                      if sanitize(k) == ai_to for _u in ar.units)
    if n_ai_to <= n_ai_to_pre:
        fails.append(('ai_move_not_rebased', (ai_to, n_ai_to_pre,
                                              n_ai_to)))
    if drv.last_actor is not Side.FREE_PEOPLES:
        fails.append(('alternation', drv.last_actor))
    if not any(f[0] == 'undo_rebased' for f in drv.findings):
        fails.append(('no_rebase_finding', drv.findings[-3:]))
    if drv.alarms:
        fails.append(('alarms', drv.alarms))
    # the notice rides a VOCABULARY chat kind through the policy exit
    # (real-client leg e1 2026-08-22: a raw 'note' kind raised
    # ChatPolicyError as a server cycle error)
    from wotr_bridge.live import chat as _chat
    chats = [i for i in items if i[0] == 'chat']
    if not chats:
        fails.append(('rebase_notice_missing', items))
    for _c in chats:
        try:
            _chat.ChatOut().say(_c[1], _c[2])
        except Exception as _e:
            fails.append(('rebase_notice_policy', repr(_e)[:120]))
    # the human acts again: a window on the restored state commits
    a2, lines2 = TD._army_window_lines(es)
    if a2 is None:
        return
    m0 = drv.window_stats['matched']
    rig.feed(*lines2)
    if drv.window_stats['matched'] != m0 + 1:
        fails.append(('human_cannot_act_after_rebase',
                      (drv.window_stats, drv.alarms[-1:],
                       drv.findings[-2:])))


def test_undo_beneath_unrebasable_reply_is_loud(fails):
    """MEASURES: an AI reply that cannot re-apply on the restored state
    (a move OUT OF the region the human's retracted move had filled)
    raises a LOUD alarm naming the reply; the human window is still
    retracted (the client is the authority for its own pieces)."""
    import ai
    import test_live_driver as TD
    from board import sanitize
    st = TD.adopted_state()
    a, lines = TD._army_window_lines(st)
    if a is None:
        fails.append(('undo2_control', 'NOT-MEASURED: no army move'))
        return
    rig = TD.Rig(st, Side.FREE_PEOPLES, phase='actions', seed=5)
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES
    rig.feed(*lines)
    if drv.window_stats['matched'] != 1:
        fails.append(('undo2_control', 'NOT-MEASURED'))
        return
    post_h = drv.es

    class _Bad(object):
        action_type = 'army_move'
        die_idx = 0
        params = {'from': 'nowhere', 'to': 'nowhere'}
    post_ai = ai.clone_state(post_h)
    drv._snapshot([], 'ai:action', post=post_ai, action=_Bad())
    drv.es = post_ai
    drv.hands.es = post_ai
    for ln in reversed(lines):
        rig.feed('$Wey undoes last action : %s' % ln)
    if not any('RE-BASE' in str(x).upper() for x in drv.alarms):
        fails.append(('no_loud_rebase_failure', drv.alarms[-2:]))
    if drv.es.dice[Side.SHADOW][a.die_idx].used:
        fails.append(('human_window_kept', 'engine still holds the '
                      'retracted window'))


def main():
    tests = [test_undo_beneath_ai_reply_rebases,
             test_undo_beneath_unrebasable_reply_is_loud]
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
