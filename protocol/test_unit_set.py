"""An Army move that names its units (unit_set) applied natively as by Python (no pytest needed).

    PYTHONHASHSEED=0 python -s protocol/test_unit_set.py [--src SRC] [--per-move N]

The seat's unit rule (wotr_bridge/core/ai_units.py) turns a counted army move
out of a mixed-nation army into the same move naming its units ('unit_set',
army_card_cargo keys).  Python's _execute_action moves exactly those units
(game_logic.move_army chosen_units); the native executor must do the same.
For eight recorded positions (protocol/testdata/unit_set_views.json: Action
decisions of AlphaWOTR v0.2.0 games, both sides) every split army move out of
a mixed army is applied with each unit_set its eligible pool can supply:
Python on a copy, natively from the packed state (apply_descriptor_random),
and the two resulting boards must be equal (every army's units and leaders in
order, the pending army-move state, the dice).  A unit_set Python refuses must
be refused natively too.  The same moves without unit_set are checked too.
"""
import argparse
import collections
import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), 'native_backend'), os.path.join(os.path.dirname(HERE), 'wotr_bridge')]
import wotr_protocol_bot as P  # noqa: E402


def board(ai, st):
    """The figures of every army (in roster order) and the move state, as names."""
    out = {}
    for k, a in sorted(st.armies.items()):
        units = [(u.nation.name, u.kind.name) for u in a.units]
        leaders = [type(l).__name__ + ':' + str(getattr(l, 'name', '') or getattr(getattr(l, 'nation', None), 'name', ''))
                   for l in a.leaders]
        if units or leaders:
            out[ai.sanitize(k)] = (units, leaders)
    dice = {side.name: [str(getattr(d, 'current', d)) for d in ds] for side, ds in st.dice.items()}
    return out, st.pending_army_move_die, st.pending_army_move_from, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(os.path.dirname(HERE), 'src'))
    ap.add_argument('--per-move', type=int, default=2)
    args = ap.parse_args()
    P.import_engine(args.src, True)
    ai = P.ai
    from core import native_search
    native_search.load(ai)   # the box's library (built by native_backend/live_search_oct19/build_portable.py)
    import snapshot
    import random_stream
    from army_move_participants import figure_sets
    names = P.Names()
    views = json.load(open(os.path.join(HERE, 'testdata', 'unit_set_views.json')))
    rng = random.Random(1)
    checked = refused = counted = 0
    for n, v in enumerate(views):
        side = P.side_in(v['side'])
        memo = P.GameMemo(side, 1)
        st = P.build_state(v['view'], memo, names)
        P.setup_action(None, st, v['decision'], v['view'])
        packed = snapshot.pack_snapshot(ai, st)
        for a in ai.get_legal_actions(st, side):
            p = a.params or {}
            if a.action_type != 'army_move' or p.get('n_regular') is None:
                continue
            army = P.army_at(st, p['from'])
            own = [u for u in army.units if ai._side_of_nation(u.nation) == side]
            nr, ne = int(p.get('n_regular') or 0), int(p.get('n_elite') or 0)
            if len({u.nation for u in own}) < 2 or not 0 < nr + ne < len(own):
                continue
            sets = [s for s in figure_sets(own, nr + ne, unit=True)
                    if collections.Counter(k for (_u, _n, k, c) in s for _ in range(c)) == collections.Counter(
                        ['REGULAR'] * nr + ['ELITE'] * ne)]
            rng.shuffle(sets)
            for us in [None] + sets[:args.per_move]:
                params = dict(p) if us is None else dict(p, unit_set=us)
                act = type(a)(a.die_idx, a.action_type, params)
                state = rng.getstate()
                w = P.deep_copy_state(st)
                random.setstate(state)
                try:
                    ai._execute_action(w, act)
                    want = board(ai, w)
                except Exception as e:
                    want = 'refused: %s' % type(e).__name__
                menu = snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai, [(a.die_idx, a.action_type, params)]))
                stream = random_stream.NativeRandom(state)
                try:
                    child, _adv, _ann = packed.apply_descriptor_random(ai, stream, side.value if isinstance(side.value, int) else (0 if side is P.FP else 1), menu, 0)
                    got = board(ai, child.unpack(ai))
                except Exception as e:
                    got = 'refused: %s' % type(e).__name__
                ok = (want == got) or (isinstance(want, str) and isinstance(got, str))
                if not ok:
                    print('view %d %s %s: Python %s\n  native %s' % (n, p['from'], us, want, got))
                    return 1
                if us is None:
                    counted += 1
                elif isinstance(want, str):
                    refused += 1
                else:
                    checked += 1
    print('PASS: %d unit_set moves applied as by Python (%d refused by both), %d counted moves the same' % (checked, refused, counted))
    return 0 if checked else 1


if __name__ == '__main__':
    sys.exit(main())
