"""Public native character faces, independent of gesture or action history.

AiBoot's fifth dump column exports H.k(): bV/PID240 is Grey=0,
White=1, White shining=2; cE/PID241 is Strider=0, Aragorn=1. The
fourth column and dynamic army replacement types carry no such identity.
Manual adoption restores the visible character, never replays its action.
"""
_FACES = {'240': {0: 'Gandalf the Grey', 1: 'Gandalf the White',
                  2: 'Gandalf the White'},
          '241': {0: 'Strider', 1: 'Aragorn'}}
_PROMOTED = {'240': 'Gandalf the White', '241': 'Aragorn'}
_FLAGS = {'240': ('gandalf_the_white', 'in_play'),
          '241': ('aragorn_crowned', 'crowned')}
_FELLOWSHIP = {'fellowship of the ring', 'guide of the fellowship',
               'fellowship box', 'guide box'}
_DEAD = {'fp casualties', 'fp dead', 'dead'}


def public_name(pid, marker_states):
    value = (marker_states or {}).get(str(pid))
    # bools and permissively parsed strings are not native H.k() values.
    return _FACES.get(str(pid), {}).get(value) if type(value) is int else None


def _roster(state, pid):
    names = set(_FACES[pid].values())
    out = [(c, state.fellowship.companions) for c in state.fellowship.companions
           if getattr(c, 'name', None) in names]
    for army in state.armies.values():
        out.extend((c, army.leaders) for c in army.leaders
                   if getattr(c, 'name', None) in names)
    return out


def _public_records(state, dump, marker_states):
    import board
    for pid in _FACES:
        if pid not in dump or pid not in (marker_states or {}):
            continue  # Legacy dumps do not authorize guessing a promotion.
        name = public_name(pid, marker_states)
        zone = board.sanitize(str(dump[pid]))
        base = zone[:-11] if zone.endswith(' stronghold') else zone
        where = ('region' if base in board.regions else 'fellowship'
                 if zone in _FELLOWSHIP else 'dead' if zone in _DEAD else 'other')
        attr, flag = _FLAGS[pid]
        canonical = getattr(state, attr, None)
        known = bool(getattr(canonical, flag, False))
        roster = _roster(state, pid)
        error = None
        if name is None:
            error = 'character identity %s: unrecognized native face %r' % (
                pid, marker_states[pid])
        elif name != _PROMOTED[pid] and (known or any(
                c.name == _PROMOTED[pid] for c, _ in roster)):
            # These engine flags also prohibit a second entry after death.
            # A lower face cannot establish whether native undo reversed that
            # history or somebody merely flipped the surviving physical piece.
            error = ('character promotion history unresolved for %s: client=%s; '
                     'known promotion cannot be erased by a face flip' % (pid, name))
        elif name == _PROMOTED[pid] and where not in ('region', 'dead'):
            error = ('character promotion history unresolved for %s: %s in %s '
                     'does not establish an entered character' % (pid, name, zone))
        yield pid, name, where, base, canonical, flag, known, roster, error


def identity_rows(state, dump, marker_states):
    """Read-only differences that alias-based location census cannot see."""
    rows = []
    for pid, name, where, _base, _canonical, _flag, known, roster, error in \
            _public_records(state, dump, marker_states):
        if error:
            rows.append(error)
            continue
        if where in ('region', 'fellowship') and (
                len(roster) != 1 or roster[0][0].name != name):
            rows.append('character identity %s: engine=%s client=%s' % (
                pid, ', '.join(c.name for c, _ in roster) or 'absent', name))
        if name == _PROMOTED[pid] and not known:
            rows.append('character promotion %s: engine=not recorded client=%s' % (pid, name))
    return rows


def _new_companion(name):
    """The same fixed character attributes as the engine's constructors."""
    import game_state as gs
    if name == 'Gandalf the White':
        return gs.Companion(name=name, level=3, leadership=1,
                            bonus_movement=1, solo_hobbit_movement=True)
    if name == 'Aragorn':
        return gs.Companion(name=name, level=3, leadership=2,
                            captain_of_the_west=True)
    if name == 'Gandalf the Grey':
        return gs.Companion(name=name, level=3, leadership=1,
                            captain_of_the_west=True, guide_ability='palantir_draw')
    if name == 'Strider':
        return gs.Companion(name=name, level=3, leadership=1,
                            captain_of_the_west=True, guide_ability='hide_any_die',
                            activates_nations=[gs.Nation.THE_NORTH],
                            activation_regions=['dale', 'the shire'])
    raise ValueError('No known character attributes for %r' % name)


def adopt_public_characters(state, dump, marker_states):
    """Manual recovery only; caller owns the complete-board transaction.

    Restore exact visible identity and historical entry flags. Do not spend,
    add or roll dice, activate nations, or replay promotion/card effects.
    Required-companion table cards are rechecked as derived roster coherence.
    The ordinary next-turn bonus calculation uses the repaired roster/flags.
    """
    import game_logic as gl
    records = list(_public_records(state, dump, marker_states))
    for *_, error in records:
        if error:
            raise ValueError(error)
    out = []
    for pid, name, where, base, canonical, flag, known, roster, _ in records:
        if name == _PROMOTED[pid] and not known:
            if canonical is None:
                raise ValueError('Missing canonical promotion record for %s' % pid)
            setattr(canonical, flag, True)
            out.append('character promotion recorded from native face: %s' % name)
        if where not in ('region', 'fellowship'):
            continue  # Existing companion adopter handles visible casualties.
        if len(roster) == 1 and roster[0][0].name == name:
            continue  # Existing companion adopter handles location changes.
        companion = next((c for c, _ in roster if c.name == name), None)
        if companion is None:
            companion = _new_companion(name)
        for old, holders in roster:
            holders.remove(old)
        if where == 'fellowship':
            state.fellowship.companions.append(companion)
        else:
            gl.get_or_create_army(state, base).leaders.append(companion)
        out.append('character identity adopted from native face: %s at %s' % (name, base))
    if out and state.fellowship.guide not in state.fellowship.companions:
        gl.change_guide(state, allow_pending=True)
    if out:
        # The engine may have missed Grey's separation as well as promotion.
        # Reuse required-companion coherence, never casualty/event handlers:
        # Staff leaves play, but no Hunt casualty choice or draw is triggered.
        for title in gl.check_companion_persistent_discards(state):
            out.append('table card discarded (required companion left the '
                       'Fellowship): %s' % title)
    return out
