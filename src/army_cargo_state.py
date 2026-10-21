"""Exact public cargo commitments and the unfinished category selection."""
import numpy as np
from board import sanitize
from army_card_cargo import decode
from card_cargo_observation import CATEGORIES

TAIL_DIM = 166


def counts(cargo):
    values = np.zeros(12, dtype=np.float32)
    for category, count in decode(cargo).items():
        values[CATEGORIES.index(category)] = count / 10.0
    return values


def encode_tail(state, region_index):
    from card_movement_context import exact_commitments
    if len(region_index) != 105:
        raise ValueError('Cargo state requires the canonical 105 regions')
    out = np.zeros(TAIL_DIM, dtype=np.float32)
    combat = getattr(state, 'pending_combat', None) or {}
    if combat.get('corsairs_relief_garrison'):
        out[:12] = counts(combat['corsairs_relief_garrison'])
        out[60] = -1.0
        out[61 + region_index[sanitize(combat['defender_region'])]] = 1.0
        return out
    for index, (_, _, cargo) in enumerate(exact_commitments(state)):
        out[index * 12:(index + 1) * 12] = counts(cargo)
    pending = getattr(state, 'pending_card_option', None) or {}
    options = pending.get('options') or []
    landing = [o.get('_corsairs_landing_pc') for o in options if '_corsairs_landing_pc' in o]
    if landing:
        if len(landing) != len(options) or any(pc != landing[0] for pc in landing):
            raise ValueError('Corsairs landing options disagree about pending battle')
        out[60] = -2.0
        out[61 + region_index[sanitize(landing[0]['defender_region'])]] = 1.0
        return out
    selections = [o for o in options if '_cargo_index' in o]
    if not selections:
        return out
    if len(selections) != len(options):
        raise ValueError('Mixed cargo-stage and route options')
    evidence = []
    for choice in selections:
        category = tuple(choice['_cargo_key'])
        before = decode(choice.get('cargo', ()))
        before.pop(category, None)
        route = choice['_cargo_continue']
        source = sanitize(route.get('from_region', 'umbar'))
        target = sanitize(route.get('to_region', route.get('target_region', '')))
        evidence.append((category, before, source, target))
    if any(item != evidence[0] for item in evidence):
        raise ValueError('Cargo options disagree on their committed prefix')
    category, before, source, target = evidence[0]
    if source == target or source not in region_index or target not in region_index:
        raise ValueError('Invalid active cargo route')
    for key, count in before.items():
        out[36 + CATEGORIES.index(key)] = count / 10.0
    out[48 + CATEGORIES.index(category)] = 1.0
    out[60] = 1.0
    out[61 + region_index[source]] = 1.0
    out[61 + region_index[target]] = -1.0
    return out
