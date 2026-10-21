"""Regression: a companion SEPARATING to DEAD MARSHES must not be
parsed as a casualty.

DEFECT (found 2026-08-16, specimen
SPECIMEN_companion_guidebox_parser_auto72_game02 log line 565,
'245 Game.1999 (Guide Box) -> Game.1959 (Dead Marshes)'):

_handle_fellowship_companions classified the destination with a private
SUBSTRING test --

    death_areas = ['Card Play', 'Dead', 'Casualties', 'Current Action']
    is_dying = any(area in to_loc for area in death_areas)

-- and 'Dead' is a substring of 'Dead Marshes'.  The first arm of the
if/elif chain therefore won, and the separation arm behind it was
unreachable for that one region: the companion was popped from the
fellowship, NO army was created at the destination, and a phantom
companion_death / hunt_sacrifice_guide was appended to the corpus
action log.  Nothing downstream ever corrects it.

is_map_region() was rewritten to exact board membership for precisely
this class (its docstring names KHAND/'Hand', CARDOLAN/'Card',
DEAD MARSHES/'Dead') but this classifier was missed.  The fix routes it
through the single resolver LogReplayer._zone_is_companion_death, whose
governing invariant is: A MAP REGION IS NEVER A CASUALTY PILE.

Run:  python test_companion_dead_marshes.py
Anti-vacuity:  WOTR_MUTATE_FIX=1 python test_companion_dead_marshes.py
               (restores the pre-fix classifier; MUST go red)
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import board                      # noqa: E402  (chdir happens here)
import game_state as gs           # noqa: E402
import log_replay as lr           # noqa: E402
from log_replay import sanitize   # noqa: E402

# Optional integration specimen (the live-play capture the defect was
# found in).  Legs 1 and 3 are self-contained and catch the defect on
# their own, so a missing specimen SKIPS rather than fails.
SPECIMEN = (r'C:\Users\sarve\Downloads\WOTR_AI_training_lab\streak'
            r'\SPECIMEN_companion_guidebox_parser_auto72_game02'
            r'\ai_live_game.log')

FAILURES = []
SKIPPED = []

# ---- anti-vacuity harness ------------------------------------------
if os.environ.get('WOTR_MUTATE_FIX') == '1':
    _OLD_DEATH_AREAS = ['Card Play', 'Dead', 'Casualties',
                        'Current Action']

    def _prefix_classifier(self, to_loc):
        return any(area in to_loc for area in _OLD_DEATH_AREAS)

    lr.LogReplayer._zone_is_companion_death = _prefix_classifier
    print('*** MUTATION ACTIVE: pre-fix substring classifier restored ***')


def check(cond, msg):
    print('  %s  %s' % ('PASS' if cond else 'FAIL', msg))
    if not cond:
        FAILURES.append(msg)


# --------------------------------------------------------------------
def leg1_classifier():
    """UNIT: the resolver itself.  No I/O - this leg alone fails on the
    substring implementation."""
    print('LEG 1  companion-death resolver vs map regions')
    b = board.load_board()
    st = gs.setup_initial_game(b)
    rep = lr.LogReplayer(st, b, verbose=False)
    resolver = getattr(rep, '_zone_is_companion_death', None)
    check(resolver is not None,
          'log_replay exposes the single companion-death resolver')
    if resolver is None:
        return
    check(resolver('Dead Marshes') is False,
          "'Dead Marshes' (a MAP REGION) is not a death zone")
    for z in ('Dead', 'FP Dead', 'SA Dead', 'FP Casualties',
              'SA Casualties', 'FP Card Play', 'SA Card Play',
              'Current Action', 'Paths of the Dead',
              'The King of the Dead',
              'Horde from the East/Deadly Strife'):
        check(resolver(z) is True, "'%s' still classifies as death" % z)
    bad = [r for r in board.regions if resolver(r)]
    check(not bad, 'no map region classifies as a death zone (got %r)' % bad)


# --------------------------------------------------------------------
def leg2_specimen():
    """INTEGRATION: the real live-play capture."""
    print('LEG 2  specimen replay')
    if not os.path.exists(SPECIMEN):
        print('  SKIP  specimen not on disk (%s)' % SPECIMEN)
        SKIPPED.append('leg2')
        return
    b = board.load_board()
    st = gs.setup_initial_game(b)
    rep = lr.LogReplayer(st, b, verbose=False)
    rep.parse_and_replay(SPECIMEN)

    dm = [a for k, a in st.armies.items() if sanitize(k) == 'dead marshes']
    check(len(dm) == 1,
          'an army exists at dead marshes (engine truth), got %d' % len(dm))
    if dm:
        names = [getattr(l, 'name', '?') for l in dm[0].leaders]
        check(len(dm[0].units) == 0 and len(dm[0].leaders) == 1,
              'dead marshes army is ((), 1) - got units=%d leaders=%d %r'
              % (len(dm[0].units), len(dm[0].leaders), names))
        check('Meriadoc' in names,
              'the separated companion is Meriadoc, got %r' % names)

    # NOTE the deliberate scope: an accompanying hunt_sacrifice_random is
    # legitimate (a hunt tile was drawn one line earlier, and the
    # pre-existing sacrifice-by-separation ruling in _record_action
    # labels any separation under unresolved hunt damage that way while
    # leaving the companion ON THE BOARD).  What must never appear is a
    # record that KILLS him.
    phantom = [(t, p) for (t, p, _s) in rep.action_log
               if t in ('companion_death', 'hunt_sacrifice_guide')
               and sanitize(str(p.get('companion', ''))) == 'meriadoc']
    check(not phantom,
          'no phantom Meriadoc death/guide-sacrifice record, got %r'
          % phantom)

    sep = [p for (t, p, _s) in rep.action_log
           if t == 'companion_separate'
           and sanitize(str(p.get('companion', ''))) == 'meriadoc']
    check(len(sep) == 1,
          'exactly one companion_separate for Meriadoc, got %r' % sep)
    if sep:
        check(sep[0].get('to_region') == 'dead marshes',
              "the separate record targets 'dead marshes', got %r"
              % sep[0].get('to_region'))


# --------------------------------------------------------------------
# The same defect with NO guide box involved - the family is
# 'companion separates to Dead Marshes', not 'guide box'.
_SYNTH = ('$P<~Scribe.141~>245 <~Scribe.142~><~Game.1998~>'
          '<~Scribe.143~><~Game.1959~><~Scribe.144~> 000000\n')


def leg3_synthetic():
    print('LEG 3  synthetic Fellowship Box -> Dead Marshes (no guide box)')
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.log', prefix='wotr_dm_')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(_SYNTH)
    try:
        b = board.load_board()
        st = gs.setup_initial_game(b)
        rep = lr.LogReplayer(st, b, verbose=False)
        check(any(c.name == 'Meriadoc'
                  for c in st.fellowship.companions),
              'Meriadoc starts in the fellowship')
        rep.parse_and_replay(path)
        dm = [a for k, a in st.armies.items()
              if sanitize(k) == 'dead marshes']
        check(len(dm) == 1 and len(dm[0].leaders) == 1,
              'plain separation to dead marshes lands in an army')
        deaths = [t for (t, _p, _s) in rep.action_log
                  if t in ('companion_death', 'hunt_sacrifice_guide',
                           'hunt_sacrifice_random')]
        check(not deaths,
              'no death recorded for a plain separation, got %r' % deaths)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    leg1_classifier()
    leg2_specimen()
    leg3_synthetic()
    print()
    if FAILURES:
        print('FAIL - %d assertion(s)' % len(FAILURES))
        for f in FAILURES:
            print('   *', f)
        return 1
    print('PASS - all assertions%s'
          % (' (%d leg(s) skipped)' % len(SKIPPED) if SKIPPED else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
