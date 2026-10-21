"""SENSITIVITY CONTROLS for the L0 oracles (live_oracles.py).  Every
oracle is proven to FIRE on a broken input and to stay quiet on a
good one - an oracle that cannot fail is not an oracle (instrument
discipline: mutation-test the test).
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_SRC, _ROOT, os.path.join(_ROOT, 'wotr_bridge')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import game_logic                                          # noqa: E402
from game_state import Side, Nation, ActionDieFace          # noqa: E402
import live_oracles as lo                                  # noqa: E402
from test_live_driver import Rig, fresh_state              # noqa: E402


def _kinds(rows):
    return sorted(set(k for k, _ in rows))


def test_invariants_quiet_on_fresh_state(fails):
    st = fresh_state()
    rows = lo.state_invariant_violations(st)
    if rows:
        fails.append(('fresh_state_not_clean', rows[:5]))


def test_invariants_fire_on_card_in_two_zones(fails):
    st = fresh_state()
    cid = st.fp_strat_deck[0]
    st.hands[Side.FREE_PEOPLES].hand.append(cid)      # still in the deck
    if 'card_in_two_zones' not in _kinds(lo.state_invariant_violations(st)):
        fails.append(('card_dup_missed', True))


def test_hand_over_six_is_not_a_violation(fails):
    """The 6-card limit is a TURN-END rule, not a state invariant
    (BL-148: fpgame7 legally held 7 at a declare pending).  A hand of
    7 in a settled state must NOT be flagged."""
    st = fresh_state()
    for cid in list(st.fp_strat_deck[:7]):
        st.fp_strat_deck.remove(cid)
        st.hands[Side.FREE_PEOPLES].hand.append(cid)
    if 'hand_over_limit' in _kinds(lo.state_invariant_violations(st)):
        fails.append(('hand_limit_false_positive', True))


def test_invariants_fire_on_companion_duplicate(fails):
    st = fresh_state()
    comp = next(c for c in st.fellowship.companions
                if getattr(c, 'name', '') == 'Legolas')
    game_logic.get_or_create_army(st, 'Eastemnet').leaders.append(comp)
    if 'companion_duplicate' not in _kinds(lo.state_invariant_violations(st)):
        fails.append(('companion_dup_missed', True))


def test_invariants_fire_on_passive_cap(fails):
    st = fresh_state()
    st.political.boxes[Nation.GONDOR] = 3
    st.political.active[Nation.GONDOR] = False
    if 'political_passive_cap' not in _kinds(lo.state_invariant_violations(st)):
        fails.append(('passive_cap_missed', True))


def test_invariants_fire_on_army_key_twin(fails):
    st = fresh_state()
    game_logic.get_or_create_army(st, "Helm's Deep")
    key = next(k for k in st.armies if 'helm' in str(k).lower())
    twin = 'helm s deep' if str(key) != 'helm s deep' else "Helm's Deep"
    st.armies[twin] = st.armies[key]          # one army, two keys
    if 'army_key_duplicate' not in _kinds(lo.state_invariant_violations(st)):
        fails.append(('army_twin_missed', (key, twin)))


def test_invariants_fire_on_units_over_components(fails):
    st = fresh_state()
    from game_state import Unit, UnitKind
    army = game_logic.get_or_create_army(st, 'Minas Tirith')
    for _ in range(12):
        army.units.append(Unit(Nation.GONDOR, UnitKind.REGULAR))
    if 'units_over_components' not in _kinds(lo.state_invariant_violations(st)):
        fails.append(('units_over_missed', True))


def test_lane_clean_quiet_on_fresh_rig(fails):
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    rows = lo.lane_violations(rig.drv)
    if rows:
        fails.append(('fresh_rig_not_clean', rows[:5]))


def test_lane_clean_fires_on_alarm_hold_and_buffer(fails):
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    drv.alarms.append('probe alarm')
    if 'alarm' not in _kinds(lo.lane_violations(drv)):
        fails.append(('alarm_missed', True))
    drv.alarms[:] = []
    drv.holds.append(('probe', None))
    if 'open_hold' not in _kinds(lo.lane_violations(drv)):
        fails.append(('hold_missed', True))
    drv.holds[:] = []
    # a card duplicated in the engine state surfaces through the lane
    cid = drv.es.fp_strat_deck[0]
    drv.es.hands[Side.FREE_PEOPLES].hand.append(cid)
    if 'state_invariant:card_in_two_zones' not in _kinds(lo.lane_violations(drv)):
        fails.append(('state_via_lane_missed', True))


def test_lane_clean_fires_on_census_rows(fails):
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    rows = lo.lane_violations(rig.drv, census_rows=['leaders eastemnet: 2 vs 0'])
    if 'census' not in _kinds(rows):
        fails.append(('census_missed', rows))
    try:
        lo.assert_lane_clean(rig.drv, census_rows=['x'])
        fails.append(('assert_did_not_raise', True))
    except lo.OracleFailure as e:
        if e.oracle != 'assert_lane_clean':
            fails.append(('wrong_oracle_name', e.oracle))


def test_voiced(fails):
    if lo.voice_violations([('chat', 'refusal', 'no')], 'refused'):
        fails.append(('voiced_refusal_flagged', True))
    if not lo.voice_violations([('note', 'holding')], 'ambiguous'):
        fails.append(('silent_ambiguous_missed', True))
    if not lo.voice_violations([], 'hold'):
        fails.append(('silent_hold_missed', True))
    if lo.voice_violations([('note', 'ok')], 'match'):
        fails.append(('quiet_match_flagged', True))
    if not lo.voice_violations([('chat', 'refusal', 'x')], 'match'):
        fails.append(('refusal_on_match_missed', True))


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    n_fail = 0
    for t in tests:
        fails = []
        try:
            t(fails)
        except Exception as e:
            import traceback
            fails.append(('exception', repr(e), traceback.format_exc()[-400:]))
        print('  [%s] %s' % ('PASS' if not fails else 'FAIL', t.__name__))
        for f in fails:
            print('    FAIL: %r' % (f,))
        n_fail += bool(fails)
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), n_fail))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
