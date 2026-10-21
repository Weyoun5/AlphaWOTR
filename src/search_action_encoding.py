"""Exact shared features for army-size variants in one encoding call.

All original rows remain. Unknown shapes and cached companion blocks use the
generic encoder. Only size, moved regulars/elites, units left and vacated flags
are filled separately; leader, location, die and card features stay inherited.
No template survives the current call, so later state/action mutations apply.
"""
import numpy as np
ALLOWED = frozenset(('from', 'to', 'n_regular', 'n_elite', 'take_leaders', 'side', 'n_nazgul', 'n_leaders', 'mouth_muster', 'card_followup'))
REQUIRED = frozenset(('from', 'to', 'n_regular', 'n_elite'))

def encode_army_batch(ai, actions, state, side, comp_memo, original):
    if len(actions) < 32:
        return original(actions, state, side, comp_memo)
    if any(('_lattice' in a.__dict__ for a in actions)):
        return original(actions, state, side, comp_memo)
    groups = {}
    selected = []
    owner = []
    variants = []
    for action in actions:
        p = action.params
        eligible = action.action_type == 'army_move' and type(p) is dict and (REQUIRED <= p.keys()) and (p.keys() <= ALLOWED) and (type(p['from']) is str) and (type(p['to']) is str) and (type(p['n_regular']) is int) and (0 <= p['n_regular'] <= 10) and (type(p['n_elite']) is int) and (0 <= p['n_elite'] <= 10) and (type(p.get('take_leaders', True)) is bool) and (type(p.get('n_nazgul')) in (int, type(None))) and (type(p.get('n_leaders')) in (int, type(None)))
        if eligible:
            key = (action.die_idx, p['from'], p['to'], p.get('take_leaders', True), p.get('n_nazgul'), p.get('n_leaders'))
            index = groups.get(key)
            if index is None:
                index = len(selected)
                groups[key] = index
                selected.append(action)
            variants.append((len(owner), index, p['n_regular'], p['n_elite'], p['from']))
            owner.append(index)
        else:
            owner.append(len(selected))
            selected.append(action)
    saved = len(actions) - len(selected)
    if saved < 16:
        return original(actions, state, side, comp_memo)
    encs, ridx = original(selected, state, side, comp_memo)
    output = encs[owner]
    regions = ridx[owner]
    source_sizes = {}
    for row, index, nr, ne, source in variants:
        if source not in source_sizes:
            army = ai._armies_lookup(state, source)
            source_sizes[source] = sum((ai._side_of_nation(u.nation) == side for u in army.units)) if army else None
    rows = np.fromiter((v[0] for v in variants), dtype=np.int64, count=len(variants))
    nr = np.fromiter((v[2] for v in variants), dtype=np.float64, count=len(variants))
    ne = np.fromiter((v[3] for v in variants), dtype=np.float64, count=len(variants))
    output[rows, 23] = (nr + ne) / 10.0 + ne / 100.0
    output[rows, ai._ACT_COMP_OFF] = nr / 10.0
    output[rows, ai._ACT_COMP_OFF + 1] = ne / 10.0
    sizes = np.fromiter((source_sizes[v[4]] if source_sizes[v[4]] is not None else -1 for v in variants), dtype=np.float64, count=len(variants))
    present = sizes >= 0
    left = np.maximum(0, sizes - nr - ne)
    output[rows[present], ai._ACT_LEAVE_OFF] = left[present] / 10.0
    output[rows[present], ai._ACT_EMPTIES_OFF] = left[present] == 0
    return (output, regions)


# Explicit native activation; Python remains the default.
_PYTHON_ENCODE_ARMY_BATCH = encode_army_batch
import sys as _sys
from action_encoding_backend import activate as _activate_backend
BACKEND = _activate_backend(_sys.modules[__name__])
