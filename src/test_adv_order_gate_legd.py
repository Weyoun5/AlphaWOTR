# -*- coding: ascii -*-
"""RED-TEAM-3, 2026-08-18.  THE ORDER GATE'S CAUSE-BEFORE-EFFECT LEG
MISJUDGES IN BOTH DIRECTIONS.

Leg D of streak/order_check.py was added with ORDER-1 (BRIDGE-8 batch
4) and states its own rule:

    "a card-draw directive emitted in tick T, with the AI's own card
     commit landing in the NEXT LINE-BEARING TICK, is the retry
     protocol rendering an effect ahead of its cause.  Adjacency alone
     is the discriminator."

Adjacency alone is not enough in either direction, and the leg's own
wording names both holes.

FALSE NEGATIVE - "LINE-BEARING".  The implementation compares TICK
INDICES (`0 < (c[1] - i) <= _WINDOW`, _WINDOW = 1) over a tick list
that includes ticks which emitted nothing.  Line-less ticks are normal
and documented in this very driver ("an empty tick (phase change /
game over) legitimately repeats the log length").  Put ONE empty tick
between the aborted step's directive and its retry and the exact live
inversion the leg was built from goes silent.

FALSE POSITIVE - "ITS OWN".  The rule is causal: the draw must be the
one THAT card earned.  The test is purely positional - any draw
directive one tick before any Controls.840 commit is flagged, whether
or not the commit is the same card, the same battle, or even the same
action.  A correctly ordered log (card commits first, THEN takes its
bonus draw) followed by a different card's commit in the next tick is
reported as an inversion.  The leg even carries a `_BATTLE` regex for
this discrimination - it is defined and never used.

WHY IT MATTERS: leg D is the offline gate for a defect that cost the
USER a whole session (a working Witch-king draw that READ as broken).
A false negative means the class can come back through the retry
protocol unseen; a false positive means the gate cries wolf on
correct emissions and gets ignored or, worse, gets "fixed" by
reordering something that was right.

RED: the FN leg and both FP legs fail today.  The CONTROL leg is the
anti-vacuity proof - it drives the real gate with the real live
specimen shape and requires it to FIRE, so a silence elsewhere is the
gate's and not this harness's.
"""
import os
import sys
import tempfile

MAIN = r'C:\Users\sarve\Downloads\WOTR_AI_training'
LAB = r'C:\Users\sarve\Downloads\WOTR_AI_training_lab\streak'
for p in (LAB, os.path.join(MAIN, 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import order_check                            # noqa: E402

FAIL = []


def check(name, cond, extra=''):
    print(('  PASS  ' if cond else '  FAIL  ') + name
          + (('\n          <- ' + str(extra)) if (extra and not cond)
             else ''))
    if not cond:
        FAIL.append(name)


H = '[LIVE_SHADOW] %d log lines -> %s (0.4s)'

# The live inversion exactly as reported: server 427 emits the draw
# directive alone, server 428 emits the commit that earned it.
CONTROL = [
    H % (427, 'shadow combat card play (directive)'),
    '    !CALL draw_sa_strat',
    H % (428, 'retry emits the commit'),
    '    $WOTR_AI_Shadow<~Controls.840~>195',
]

# SAME inversion, one line-less tick between.  Nothing else changes.
FN_EMPTY_TICK = [
    H % (427, 'shadow combat card play (directive)'),
    '    !CALL draw_sa_strat',
    H % (427, 'phase tick, emitted no lines'),
    H % (428, 'retry emits the commit'),
    '    $WOTR_AI_Shadow<~Controls.840~>195',
]

# CORRECT order: the card commits FIRST, then takes the draw it earned.
# The next tick is a different action committing a different card.
FP_CORRECT_THEN_OTHER = [
    H % (310, 'plays the palantir of orthanc'),
    '    $WOTR_AI_Shadow<~Controls.840~>101',
    '    <WOTR_AI> plays the palantir of orthanc',
    '    !CALL draw_sa_strat',
    H % (311, 'next action: shadow commits its combat card'),
    '    $WOTR_AI_Shadow<~Controls.840~>195',
]

# A plain turn-start draw whose next tick happens to be a commit.
FP_TURNSTART = [
    H % (100, 'turn start draws'),
    '    !CALL draw_sa_strat',
    H % (101, 'battle: shadow commits its combat card'),
    '    $WOTR_AI_Shadow<~Controls.840~>195',
]


def legd(lines):
    fd, p = tempfile.mkstemp(suffix='.log', prefix='rt3_orderD_')
    os.close(fd)
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    try:
        issues, _notes = order_check.check(p)
    finally:
        try:
            os.unlink(p)
        except Exception:
            pass
    return [s for s in issues if 'effect-before-cause' in s]


def main():
    print('CONTROL  anti-vacuity: the real inversion must FIRE')
    c = legd(CONTROL)
    check('leg D fires on the live specimen shape', len(c) == 1,
          'the harness is not driving the gate; every other leg below '
          'is meaningless.  findings=%r' % (c,))
    if len(c) != 1:
        print('\n%d failure(s)' % len(FAIL))
        return 1

    print('\nFALSE NEGATIVE  one line-less tick hides the inversion')
    fn = legd(FN_EMPTY_TICK)
    check('leg D still fires when an EMPTY tick intervenes',
          len(fn) == 1,
          'the same inversion, with one tick that emitted no lines '
          'between the directive and the retry, is MISSED.  The rule '
          'says "next line-bearing tick"; the code compares tick '
          'indices, and empty ticks are documented as normal in this '
          'driver.  findings=%d' % len(fn))

    print('\nFALSE POSITIVE  no causal discrimination')
    fp1 = legd(FP_CORRECT_THEN_OTHER)
    check('a correctly-ordered bonus draw is NOT flagged when the next '
          'tick commits a different card', not fp1,
          'the commit at 840 that follows is a DIFFERENT card, and the '
          'draw already had its cause emitted ahead of it in its own '
          'tick.  Flagged anyway: %s' % (fp1[0][:110] if fp1 else ''))
    fp2 = legd(FP_TURNSTART)
    check('a turn-start draw is NOT flagged merely because the next '
          'tick happens to carry a commit', not fp2,
          'positional adjacency alone; the leg carries a _BATTLE regex '
          'for exactly this discrimination and never uses it.  '
          'Flagged: %s' % (fp2[0][:110] if fp2 else ''))

    print('\n%d failure(s)' % len(FAIL))
    for x in FAIL:
        print('  - ' + x)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
