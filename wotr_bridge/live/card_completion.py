"""Physical completion boundaries for variable-size human card movements.

A legal prefix does not select the final size of an optional movement. The
card/die disposal or an explicit done reply closes that physical choice.
This is independent of the renderer's original-cargo provenance protocol.
"""
_SINGLE_ARMY_CARDS = frozenset((
    'SHADOW_Strategy_Shadows_Gather',
    'SHADOW_Strategy_Corsairs_of_Umbar',
))
_COMPANION_GROUP_CARDS = frozenset((
    'FREE_PEOPLES_Character_Gwaihir_the_Windlord',
    'FREE_PEOPLES_Character_We_Prove_Swifter',
    'FREE_PEOPLES_Character_I_Will_Go_Alone',
    'FREE_PEOPLES_Character_There_and_Back_Again',
))


def _companion_group_choice(choices):
    action = choices.get('companion_action', '')
    return bool(choices.get('destination') and
                ('companions' in choices or action == 'group'
                 or action.startswith(('sep_', 'move_'))))


def needs_boundary(state):
    from .emission_card import staged_army_card
    pending = getattr(state, 'pending_card_option', None) or {}
    options = pending.get('options') or ()
    if staged_army_card(state) is not None:
        return True
    if any('rage_split' in option for option in options):
        return True
    if pending.get('card_id') == 'FREE_PEOPLES_Character_Dead_Men_of_Dunharrow':
        return any('companions' in option or 'recruit_count' in option for option in options)
    if pending.get('card_id') in _COMPANION_GROUP_CARDS:
        return any(_companion_group_choice(option) for option in options)
    return (pending.get('card_id') in _SINGLE_ARMY_CARDS
            and not any('_corsairs_landing_pc' in option for option in options))


def choice_needs_boundary(state, choices):
    return (needs_boundary(state) and isinstance(choices, dict)
            and (_companion_group_choice(choices) or any(key in choices for key in
                    ('rage_split', '_cargo_index', '_tsl_done', '_tsim_done',
                     'companions', 'recruit_count'))))


def empty_close(state, window):
    """Only an explicit completed optional movement can express no movement."""
    if not getattr(window, '_staged_card_closed', False) or not needs_boundary(state):
        return False
    options = (getattr(state, 'pending_card_option', None) or {}).get('options') or ()
    return any(option.get('_tsl_done') or option.get('_tsim_done')
               or option.get('recruit_count') == 0
               or ('rage_split' in option and not any(option['rage_split'].values()))
               for option in options)


def closes_card(state, record):
    from .gesture_match import title_to_cid
    from .action_die_zones import temporary, zone as normalize_zone
    import rendered_replay as rr
    import re
    if record.get('op') == 'kill':
        zone = str(record.get('zone', '')).strip().lower()
        pieces = record.get('pieces') or ()
    elif record.get('op') == 'move' and normalize_zone(record.get('to', '')) in (
            'fp casualties', 'sa casualties'):
        zone = str(record.get('from', '')).strip().lower()
        obj = str(record.get('obj', ''))
        pieces = rr.split_group(obj[7:-1]) if obj.startswith('group (') and obj.endswith(')') else [obj]
    else:
        return False
    if not (
            temporary(zone) or re.fullmatch(r'board\d+', zone)):
        return False
    pending = getattr(state, 'pending_card_option', None) or {}
    # Group disposal and a card parked on the table express the same finish.
    # The identity of the paid card matters; its temporary display slot and
    # the number of other discarded pieces do not.
    return (needs_boundary(state) and any(
        title_to_cid(str(piece)) == pending.get('card_id') for piece in pieces))


def closed_choice_correction(state, window, result, pending_rng=False):
    """Explain a completed, unmatched human choice without choosing or undoing.

    A polling gap is never completion. Only an explicit boundary followed by
    a complete NONE verdict reaches this lane; HOLD, ambiguous choices and
    outstanding native randomness retain their existing handling. Reopen the
    boundary before correction so the next legal prefix cannot commit early.
    """
    if (window is None or not needs_boundary(state)
            or not getattr(window, '_staged_card_closed', False)
            or getattr(result, 'status', None) != 'none'
            or getattr(result, 'starved', False) or pending_rng):
        return None
    signature = tuple((record.get('op'), record.get('raw') or record.get('norm'))
                      for record in getattr(window, 'records', ()))
    window._staged_card_closed = False
    if getattr(window, '_card_correction_signature', None) == signature:
        return None
    window._card_correction_signature = signature
    from .chat import _register
    return _register('The completed card movement or recruitment does not match a legal '
            'choice. Use native Undo: undo the card or die discard first if you '
            'already removed it, then undo or correct the incorrect piece moves '
            'and say done.')
