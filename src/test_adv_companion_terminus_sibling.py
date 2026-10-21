"""Companion terminus under USER ruling 1.252 (2026-08-29), the
hunt-placed-hobbit enumerator (EDGE-3's "seventh site").

HISTORY: this file pinned W5 B-2 (amendment 1.185, "a companion may
pass THROUGH a Shadow-army region but may not END on one") for the
hunt-placed hobbit separation enumerator.  USER ruling 1.252, verbatim:
"Yes, companions can end their movement anywhere. shadow field armies
do not inhibit companion movement at all."  The W5 B-2 bar is REMOVED
from every companion-destination site (the `_companion_terminus_ok`
predicate is gone); 1.249's stop-on-entry at Shadow-controlled
STRONGHOLDS is unchanged.

Mechanism pinned here:
  RULE (1.252): a hunt-placed hobbit (Meriadoc/Peregrin placed as if
  separated, movement = progress + level) may be placed on a
  non-stronghold region holding a Shadow army.
  ENGINE: `_synthetic_decision_actions` for a `companion_move` pending
  tagged `hunt_placement` proposes every `_companion_reachable` region
  and `is_legal_companion_action` (the ONE validator) judges each; the
  validator no longer carries a Shadow-army terminus bar.
  NUMBERS: fellowship at Rivendell, a Sauron army placed on the first
  adjacent non-stronghold region, progress 1 -> that region IS offered.
  CONSEQUENCE: the learned placement surface matches the ruling; the
  live matcher (which judges from this menu) accepts the human's drag.

Net-free (no nets loaded).
"""
import os
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
if 'torch' not in sys.modules and 'ai' not in sys.modules:
    sys.modules['torch'] = types.ModuleType('torch')


def test_hunt_placed_hobbit_may_end_on_shadow_army():
    import board as bm
    import ai
    import game_logic as gl
    from game_state import (setup_initial_game, Side, Unit, Nation,
                            UnitKind)
    from board import sanitize as san

    b = bm.Board()
    st = setup_initial_game(b)
    cur = san(gl.fellowship_region(st))
    # a NON-stronghold region adjacent to the fellowship, with a
    # Shadow army -> a LEGAL terminus per ruling 1.252
    target = None
    for a in st.board.adjacent(cur):
        a = san(a)
        meta = bm.regions.get(a, {})
        if not meta.get('is_stronghold'):
            target = a
            break
    assert target, 'setup: no adjacent non-stronghold region'
    army = gl.get_or_create_army(st, target)
    army.add(Unit(Nation.SAURON, UnitKind.REGULAR))
    army.add(Unit(Nation.SAURON, UnitKind.REGULAR))
    st.fellowship.progress = 1

    ok, why = ai.is_legal_companion_action(
        st, ai.Action(-1, 'companion_separate',
                      {'companion': 'Meriadoc', 'to_region': target}),
        hunt_place_level=1)
    assert ok, ('ruling 1.252: a hunt-placed hobbit may END on the '
                'Shadow-army region %r, validator said: %s' % (target, why))

    acts = ai._synthetic_decision_actions(
        st, Side.FREE_PEOPLES, 'companion_move',
        {'hunt_placement': True, 'companion': 'Meriadoc'}, {})
    dests = sorted(san(str(a.params.get('to_region', '')))
                   for a in acts
                   if a.action_type == 'companion_separate')
    assert target in dests, (
        'ruling 1.252 (USER 2026-08-29): the hunt_placement enumerator '
        'must offer the Shadow-army non-stronghold region %r as a '
        'terminus - Shadow FIELD armies never inhibit companion '
        'movement.  Offered destinations: %s' % (target, dests))


if __name__ == '__main__':
    try:
        test_hunt_placed_hobbit_may_end_on_shadow_army()
        print('PASS test_hunt_placed_hobbit_may_end_on_shadow_army')
    except AssertionError as e:
        print('FAIL test_hunt_placed_hobbit_may_end_on_shadow_army: %s'
              % str(e).encode('ascii', 'replace').decode())
        sys.exit(1)
