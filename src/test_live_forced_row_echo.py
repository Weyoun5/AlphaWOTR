"""SCENARIO_DEFECTS D6 pin (real-client reproduction
`human_retreat_into_siege_r3`: the AI's own forced casualty row,
banked on the echo ledger at emission, was never claimed by the
client's echo of that very line and leaked into echo_open - the
human's mirror list - so the harness refused the whole batch and the
human's forced casualties went unmirrored; witness escalation).

Pinned: the AI's OWN kill line (its nick) that is fully covered by an
AI-side ledger row CONSUMES that row (the self-echo is the claim);
a human-side row stays open for the human's mirror; an AI line not
on the ledger is untouched routing.
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


def test_ai_forced_row_claimed_by_its_echo(fails):
    import test_live_driver as TD
    st = TD.adopted_state()
    rig = TD.Rig(st, Side.SHADOW, phase='actions', seed=5)   # AI = Shadow
    drv = rig.drv
    nick = drv.nick
    drv.echo_ledger.append({'kind': 'unit', 'side': 'shadow',
                            'region': 'lorien', 'tkey': 'sauron:regular',
                            'n': 1})
    drv.echo_ledger.append({'kind': 'unit', 'side': 'fp',
                            'region': 'lorien', 'tkey': 'elves:elite',
                            'n': 1})
    rig.feed('$%s kills Sauron Regular in Lorien.' % nick)
    open_ = [e for e in drv.echo_ledger if e['n'] > 0]
    if any(e['side'] == 'shadow' for e in open_):
        fails.append(('ai_row_unclaimed', open_))
    if not any(e['side'] == 'fp' and e['n'] == 1 for e in open_):
        fails.append(('human_row_lost', open_))
    if not any(f[0] == 'mirror_kill_echo' for f in drv.findings):
        fails.append(('no_echo_finding', drv.findings[-3:]))
    # the human's mirror of its own row still claims it
    rig.feed('$Wey kills Elven Elite in Lorien.')
    if [e for e in drv.echo_ledger if e['n'] > 0]:
        fails.append(('human_row_unclaimed', drv.echo_ledger))
    if drv.alarms:
        fails.append(('alarms', drv.alarms))
    # sensitivity: an AI kill line NOT on the ledger consumes nothing
    drv.echo_ledger.append({'kind': 'unit', 'side': 'fp',
                            'region': 'lorien', 'tkey': 'elves:regular',
                            'n': 1})
    rig.feed('$%s kills Sauron Elite in Lorien.' % nick)
    if not [e for e in drv.echo_ledger if e['n'] > 0]:
        fails.append(('unrelated_ai_line_consumed', 'ledger emptied'))


def main():
    tests = [test_ai_forced_row_claimed_by_its_echo]
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
