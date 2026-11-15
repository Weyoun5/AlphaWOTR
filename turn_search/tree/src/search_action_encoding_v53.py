"""Encoding generation v53: exact army-size variants in one encoding call.

R3-V53-FOLLOWONS (fix round 3; USER D5 2026-10-22 "implement all d5,
including retraining").  The v52 fast path (search_action_encoding.
encode_army_batch) groups the army_move rows of a menu that differ only in
n_regular / n_elite, encodes one template per group and fills the count
columns of the other rows.  Under v53 that helper was never called
(ai._encode_actions_batch_ordinary gated it to v52), so every army_move row
went through the generic per-row encoder, whose exact v53 cache key builds
the moved roster for every row and builds it again on a miss - about 3x
slower than v52 on army-move menus.

This module is the v53 counterpart.  It is a separate file on purpose:
search_action_encoding.py is pinned by sha256 in action_encoding_source.json
(the approved native action-encoding binding), so it stays byte-identical.

Exactness (pinned by src/test_round3_R3_V53_FOLLOWONS.py: batch == generic
per-row encoder == encode_action, all 199 columns, under v53 and v52):
  * the v52 prefix columns of a variant row are filled exactly as the v52
    fast path fills them (size, moved regulars / elites, units left, the
    vacated flag; every other prefix column is the template's);
  * the group key carries EVERY non-count parameter (the v53 roster reads
    'side', and nothing outside the key may differ inside a group);
  * the v53 tail of every grouped row is recomputed by
    encoding_v53.army_move_variant_tails - the executor's own selection
    (army_move_participants.selected) per row over a count-independent
    context built once per group;
  * rows naming their figures (leader_set / unit_set, E5-01) never join a
    group (ALLOWED excludes those keys): each is its own template.
Unknown shapes, small menus and lattice menus use the generic encoder.
"""
import numpy as np

# importing the v52 helper also activates the optional native action-encoding
# backend (WOTR_ACTION_ENCODING=native, action_encoding_backend.activate)
from search_action_encoding import ALLOWED, REQUIRED
from action_encoding_backend import bind_active

COUNT_KEYS = frozenset(('n_regular', 'n_elite'))


def _eligible(action):
    p = action.params
    return (action.action_type == 'army_move' and type(p) is dict
            and REQUIRED <= p.keys() and p.keys() <= ALLOWED
            and type(p['from']) is str and type(p['to']) is str
            and type(p['n_regular']) is int and 0 <= p['n_regular'] <= 10
            and type(p['n_elite']) is int and 0 <= p['n_elite'] <= 10
            and type(p.get('take_leaders', True)) is bool
            and type(p.get('n_nazgul')) in (int, type(None))
            and type(p.get('n_leaders')) in (int, type(None)))


def _group_key(action):
    """(die, every non-count parameter) or None when a value is unhashable."""
    p = action.params
    key = (action.die_idx, tuple(sorted((k, v) for k, v in p.items() if k not in COUNT_KEYS)))
    try:
        hash(key)
    except TypeError:
        return None
    return key


def encode_army_batch(ai, actions, state, side, comp_memo, original):
    if not getattr(ai, '_ACT_V53_ON', False):
        raise RuntimeError('search_action_encoding_v53 serves encoding generation v53 only (ACT_GEN=%r)'
                           % (getattr(ai, 'ACT_GEN', None),))
    # native backend selected: verify + bind its generation-agnostic per-row
    # helpers once (the compiled v52 key and army batch stay unbound)
    bind_active(ai)
    if len(actions) < 32:
        return original(actions, state, side, comp_memo)
    if any(('_lattice' in a.__dict__ for a in actions)):
        return original(actions, state, side, comp_memo)
    groups = {}
    selected = []
    owner = []
    variants = []
    for action in actions:
        key = _group_key(action) if _eligible(action) else None
        if key is not None:
            p = action.params
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
    # v52 prefix: the same arithmetic as search_action_encoding.encode_army_batch
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
    # v53 tail (E5-01 moved figures) of every grouped row, the template's
    # included (identical values), from the executor's own selection
    from encoding_v53 import army_move_variant_tails
    start = ai._V53_ACTION_TAIL_START
    members = {}
    for row, index, n_reg, n_eli, _source in variants:
        members.setdefault(index, []).append((row, n_reg, n_eli))
    for index, group in members.items():
        tails = army_move_variant_tails(ai, state, side, selected[index].params,
                                        [(n_reg, n_eli) for _row, n_reg, n_eli in group])
        output[[row for row, _r, _e in group], start:] = tails
    return (output, regions)
