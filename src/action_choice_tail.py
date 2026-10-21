"""Explicit card choices omitted by the historical 93-column action prefix.

The versioned tail preserves that prefix. Card-specific composition lanes are
conditioned on the card identity in the prefix and survive float16 packing.
"""
import numbers

import numpy as np

TAIL_FIELDS = (
    'second_region', 'third_region',
    'rage_north_total', 'rage_south_total', 'rage_destination',
    'explicit_troop_subset', 'regulars', 'elites',
    'target_fp_army', 'use_hunt_reduction', 'palantir_deck',
    'wizardstaff_prevent',
)
TAIL_DIM = len(TAIL_FIELDS)


def encode_tail(action, region_index):
    result = np.zeros(TAIL_DIM, dtype=np.float32)
    if action.action_type != 'card_option':
        return result
    choices = (action.params or {}).get('choices') or {}
    if not isinstance(choices, dict):
        raise ValueError('Card choices must be a mapping')

    def region(value):
        if value not in region_index:
            raise ValueError('Unknown card-choice region: %r' % value)
        return (int(region_index[value]) + 1) / len(region_index)

    def count(value, maximum):
        if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
                or not 0 <= value <= maximum):
            raise ValueError('Invalid card-choice troop count: %r' % value)
        return int(value)

    fields = [key for key in ('strongholds', 'settlements', 'regions') if key in choices]
    if len(fields) > 1:
        raise ValueError('Ambiguous multi-region card choice')
    if fields:
        selected = choices[fields[0]]
        if not isinstance(selected, (list, tuple)) or not 1 <= len(selected) <= 3:
            raise ValueError('Card-choice region list must contain one to three locations')
        locations = [region(value) for value in selected]
        result[:len(locations) - 1] = locations[1:]
    if 'recruit_counts' in choices:
        counts = choices['recruit_counts']
        if (len(fields) != 1 or not isinstance(counts, (list, tuple))
                or len(counts) != len(choices[fields[0]])
                or any(count(n, 3) == 0 for n in counts)
                or sum(counts) > 6
                or 'rage_split' in choices or 'dice_faces' in choices):
            raise ValueError('Invalid regional recruitment allocation')
        # These lanes are card-conditioned; the affected recruitment cards
        # never use the Rage split or Lidless Eye meanings.
        result[2:2 + len(counts)] = np.asarray(counts, dtype=np.float32) / 6.
    if 'rage_split' in choices:
        split = choices['rage_split']
        if not isinstance(split, dict):
            raise ValueError('Invalid Dunland split')
        north, south = (count(split.get(key), 4) for key in ('n_north', 'n_south'))
        if north + south > 4:
            raise ValueError('Invalid Dunland split total')
        result[2:4] = north / 4., south / 4.
        result[4] = region(choices.get('_rage_dest'))
        if any(key in split for key in ('n_north_elite', 'n_south_elite')):
            if 'n_regular' in choices or 'n_elite' in choices:
                raise ValueError('Mixed Rage and generic troop compositions')
            en = count(split.get('n_north_elite'), north)
            es = count(split.get('n_south_elite'), south)
            result[5:8] = 1., en / 4., es / 4.
    if 'dice_faces' in choices:
        if any(key in choices for key in ('rage_split', 'n_regular', 'n_elite',
                                          'strongholds', 'settlements', 'regions')):
            raise ValueError('Mixed Lidless Eye and troop/region choices')
        faces = choices['dice_faces']
        names = ('ARMY', 'CHARACTER', 'EVENT', 'MUSTER', 'MUSTER_ARMY')
        if (not isinstance(faces, (list, tuple)) or len(faces) > 3
                or any(face not in names for face in faces)
                or count(choices.get('dice_count'), 3) != len(faces)):
            raise ValueError('Invalid Lidless Eye die composition')
        result[:5] = [faces.count(name) / 3. for name in names]
    if choices.get('n_regular') is not None or choices.get('n_elite') is not None:
        # A missing subset means the whole eligible army; it cannot encode
        # like an explicitly selected zero. Public troop counts are <= ten.
        regulars = count(0 if choices.get('n_regular') is None else choices['n_regular'], 10)
        elites = count(0 if choices.get('n_elite') is None else choices['n_elite'], 10)
        result[5:8] = 1., regulars / 10., elites / 10.
    if 'target_fp_army' in choices:
        result[8] = region(choices['target_fp_army'])
    for field, column in (('use_hunt_reduction', 9), ('wizardstaff_prevent', 11)):
        if field in choices:
            value = choices[field]
            if not isinstance(value, (bool, numbers.Integral)) or value not in (0, 1):
                raise ValueError('Invalid binary card choice: %s' % field)
            result[column] = 1. if value else -1.
    if 'palantir_deck' in choices:
        deck = choices['palantir_deck']
        if deck not in ('character', 'strategy'):
            raise ValueError('Invalid Palantir deck choice')
        result[10] = -1. if deck == 'character' else 1.
    return result
