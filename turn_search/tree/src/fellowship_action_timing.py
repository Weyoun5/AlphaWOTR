"""Physical payment die enters the Hunt Box only after its movement resolves."""

def pending_die(state):
    """Return the consumed physical die, validating any outstanding obligation."""
    index = getattr(state, '_fellowship_move_die_pending', None)
    if index is None:
        return None
    from game_state import Side
    dice = state.dice[Side.FREE_PEOPLES]
    if type(index) is not int or index < 0 or index >= len(dice):
        raise ValueError('Invalid pending Fellowship movement die index')
    die = dice[index]
    if die.side is not Side.FREE_PEOPLES or not die.used or die.current is None:
        raise ValueError('Pending Fellowship movement die must be consumed')
    if any(die is other for other in state.hunt_box):
        raise ValueError('Pending Fellowship movement die is already in the Hunt Box')
    return die

def queue(state, index):
    if getattr(state, '_fellowship_move_die_pending', None) is not None:
        raise ValueError('Previous Fellowship movement has not finished')
    if index is None:
        return
    state._fellowship_move_die_pending = index
    pending_die(state)

def finish_if_settled(state):
    die = pending_die(state)
    if die is None:
        return False
    if any(getattr(state, key, None) is not None for key in (
            'pending_combat', 'pending_hunt', 'pending_guide_change',
            'pending_card_option', 'pending_card_followup', 'pending_discard',
            'pending_army_move_die', 'pending_muster_die')):
        return False
    if any(getattr(state, key, None) for key in (
            '_pending_hunt_damage', '_hunt_reveal_pending', '_reveal_place_due', '_pending_reveal_strongholds', '_fellowship_route_event', '_hobbit_place_queue',
            '_overstack_regions', '_wwsat_owed', '_mithril_stash', '_wizardstaff_stash')):
        return False
    from persistent_effects import should_add_dice_to_hunt_box
    # The Last Battle may have left play during an interposed reveal or loss
    # of Aragorn. Evaluate the final state, after persistent-card upkeep.
    if should_add_dice_to_hunt_box(state):
        state.hunt_box.append(die)
    state._fellowship_move_die_pending = None
    return True

def observation_tail(state):
    # The die is already spent and its identity cannot affect later legal
    # choices. Its pending addition changes the next Hunt's dice count.
    return [1.0 if pending_die(state) is not None else 0.0]


def search_context(state):
    """Search must distinguish held tiles and unfinished physical payment."""
    stash = getattr(state, '_mithril_stash', None)
    context = None
    if stash:
        tile = stash['tile']
        tile_properties = tuple(getattr(tile, key, None) for key in (
            'damage', 'is_eye', 'reveal', 'is_special', 'special_name',
            'stops_movement', 'is_dice_roll', 'is_fellowship_special'))
        continuation = tuple(stash.get(key) for key in (
            'successes', 'mordor', 'mordor_track', 'skip_auto_sacrifice',
            'eye_no_effect', 'apply_kind', 'return_pending'))
        # Card-resume choices are JSON-like public action parameters.
        def freeze(value):
            if isinstance(value, dict):
                return tuple(sorted((str(k), freeze(v)) for k, v in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(v) for v in value)
            return value
        context = (tile_properties, continuation, freeze(stash.get('choices') or {}))
    wizard = getattr(state, '_wizardstaff_stash', None)
    wizard_context = None
    if wizard:
        wizard_context = (wizard.get('apply_kind'), wizard.get('successes'),
                          wizard.get('mordor'), wizard.get('mordor_track'),
                          wizard.get('eye_no_effect'), wizard.get('region'), wizard.get('reason'))
    seen = (tuple(sorted(getattr(state, '_sh_tile_drawn_regions', None) or ()))
            if getattr(state, '_reveal_place_due', False) else ())
    from fellowship_reveal_route import pending_draws
    return (getattr(state, '_fellowship_move_die_pending', None), context, wizard_context, seen, pending_draws(state))
