"""Corsairs of Umbar through the protocol adapter (no pytest needed).

    python -I protocol/test_corsairs.py [--src SRC] [--seeds N]

A referee reports the Corsairs' battle as fought from Umbar ("from":
"umbar", "region": "dol-amroth"); the engine fights it co-located (the
Army lands first).  The adapter's pending_combat must be the engine's:
attacker_region == the landing region, the attackers moved there, the
Shadow unable to cease, and the battle must close under random legal
actions without a victor_advance (Umbar is not adjacent to the coast).
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wotr_protocol_bot as P  # noqa: E402


def region(units, siege=False, control=None):
    r = {'units': [{'nation': n, 'unit': u, 'n': k} for n, u, k in units],
         'characters': [], 'siege': siege}
    if control:
        r['control'] = control
    return r


def view_of(regions):
    return {
        'regions': regions,
        'politics': {'GONDOR': {'step': 0, 'active': True},
                     'SOUTHRONS_EASTERLINGS': {'step': 0, 'active': True}},
        'battle': {'attacker': 'SHADOW', 'battle': 'field', 'from': 'umbar',
                   'region': 'dol-amroth', 'round': 1,
                   'attackers': [{'region': 'umbar',
                                  'nation': 'SOUTHRONS_EASTERLINGS',
                                  'unit': 'regular', 'n': 6}],
                   'event_card': 'corsairs-of-umbar',
                   'cards_revealed': False, 'step': 'cards'},
    }


class Bot(object):
    def __init__(self, names, side):
        self.names = names
        self.memo = P.GameMemo(side, 1)


def count(st, key):
    a = P.army_at(st, key)
    if a is None:
        return 0
    return sum(1 for u in a.units if P.ai._side_of_nation(u.nation) is P.SH)


def run(names, view, seed, relief):
    bot = Bot(names, P.SH)
    st = P.build_state(view, bot.memo, names)
    assert P.setup_combat_card(bot, st, {'kind': 'COMBAT_CARD'}, view)
    pc = st.pending_combat
    assert pc['attacker_region'] == pc['defender_region'] == 'dol amroth', pc
    assert pc['attacker_no_retreat'] is True, pc
    assert 'rearguard_units' not in pc, pc
    # six of the eight sailed; two stay in Umbar
    assert count(st, 'umbar') == 2, count(st, 'umbar')
    assert count(st, 'dol amroth') == 6 + (2 if relief else 0)
    if relief:
        assert pc['is_siege'] is False and pc.get('corsairs_relief_garrison')
    rng = random.Random(seed)
    for _ in range(300):
        pc = st.pending_combat
        if not pc:
            return
        assert pc.get('phase') != 'victor_advance', pc
        acts = P.ai.get_legal_actions(st, pc.get('deciding_side'))
        assert acts, pc.get('phase')
        assert not any(a.action_type == 'victor_advance' for a in acts)
        P.ai._execute_action(st, rng.choice(acts))
    raise AssertionError('the battle did not close')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=P.DEFAULT_SRC)
    ap.add_argument('--seeds', type=int, default=30)
    args = ap.parse_args()
    P.import_engine(args.src)
    names = P.Names()
    se = ('SOUTHRONS_EASTERLINGS', 'regular', 8)
    field = view_of({'umbar': region([se]),
                     'dol-amroth': region([('GONDOR', 'regular', 1)])})
    # A relief landing: Dol Amroth held by the Shadow (two Southrons
    # inside), besieged by Gondor; the garrison stays inside.
    relief = view_of({
        'umbar': region([se]),
        'dol-amroth': region([('GONDOR', 'regular', 2),
                              ('SOUTHRONS_EASTERLINGS', 'regular', 2)],
                             siege='SHADOW', control='SHADOW')})
    for label, view, rel in (('field', field, False),
                             ('relief', relief, True)):
        for seed in range(args.seeds):
            run(names, view, seed, rel)
        print('%s: %d battles closed, no advance from Umbar'
              % (label, args.seeds))
    print('OK')


if __name__ == '__main__':
    main()
