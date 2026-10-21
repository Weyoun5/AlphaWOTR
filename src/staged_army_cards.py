"""Simultaneous Shadow army selection and shared original-cargo execution."""
import event_effects as ee
from board import sanitize
from game_state import Side

CARD_ID = 'SHADOW_Strategy_The_Shadow_is_Moving'

def handler(state, card, choices):
    if not choices.get('_from_pending'):
        raise ValueError('Staged handler requires a pending card choice')
    if choices.get('_tsim_moved') or choices.get('_tsim_n'):
        raise ValueError('Legacy immediate movement cannot become a staged plan')
    history = choices.get('_tsim_moves', ())
    if not isinstance(history, (tuple, list)) or len(history) > 3:
        raise ValueError('Invalid staged TSiM history')
    moves = []
    for move in history:
        if not isinstance(move, (tuple, list)) or len(move) not in (2, 3):
            raise ValueError('Invalid staged TSiM movement')
        source, target = map(sanitize, move[:2])
        moves.append((source, target, move[2] if len(move) == 3 else None))
    if len({source for source, _, _ in moves}) != len(moves):
        raise ValueError('TSiM source committed more than once')
    all_legal = {(o['from_region'], o['to_region']) for o in ee._tsim_next_options(state, set(), 0)}
    if any((source, target) not in all_legal for source, target, _ in moves):
        raise ValueError('Committed movement not available at action start')
    done = choices.get('_tsim_done', False)
    if type(done) is not bool:
        raise ValueError('Invalid movement terminator')
    if not done:
        source, target = sanitize(choices['from_region']), sanitize(choices['to_region'])
        if (source, target) not in all_legal or source in {m[0] for m in moves}:
            raise ValueError('TSiM selection not available')
        moves.append((source, target, choices.get('cargo')))
    moves = tuple(sorted(moves))
    if not done and len(moves) < 4:
        options = ee._tsim_next_options(state, {m[0] for m in moves}, len(moves))
        if options:
            options = [{'from_region': o['from_region'], 'to_region': o['to_region'],
                        '_tsim_moves': moves} for o in options]
            options.append({'_tsim_done': True, '_tsim_moves': moves})
            state.pending_card_option = {'card_id': getattr(card, 'card_id', CARD_ID),
                'card_name': 'The Shadow is Moving', 'side': 'sh', 'decider': 'sh',
                'direct': True, 'options': options}
            return {'effects': [f'Selected {len(moves)} of up to four Army movements']}
    if not moves:
        state._card_army_move_witness = (CARD_ID, ())
        return {'effects': ['The Shadow is Moving: movement ends']}
    return execute_staged(state, moves, {'effects': []})


def execute_staged(state,mvs,result):
    """Move original cargo out of every source before merging destinations."""
    # SIMULTANEOUS movement (almanac A9): each source army moves once;
    # destinations may not hold enemy units; process all move-OUTs
    # first so move-in-then-move-out is order-independent, then merge.
    from combat import side_of_nation as _son, _check_settlement_capture
    import game_logic as _gl9
    from army_card_cargo import selected
    # Resolve exact cargo against the entire unchanged start position before
    # removing anything; malformed later cargo cannot partially execute a plan.
    exact = {source: selected(state, source, cargo)
             for source, _, cargo in mvs if isinstance(cargo, (tuple, list))}
    if len(mvs) > 4 or len({m[0] for m in mvs}) != len(mvs):
        raise ValueError('TSiM requires up to four distinct original armies')
    seen_src = set()
    staged = []
    # D1 (2026-09-10, USER "fix the python"): SIMULTANEOUS movement - every
    # destination is judged at the START position (before any move-out of
    # this card), exactly as the native etsm_execute validates the whole
    # plan against etsm_legal before any cargo moves.  Rulebook p.25
    # "Movement Restrictions": figures move once per action, "this includes
    # movement effectuated by Event cards"; the card moves "up to four
    # different Shadow Armies up to one region each".  The sequential
    # re-check refused a move INTO a besieged stronghold whose besiegers
    # left with an earlier move of the SAME plan (no Shadow units outside
    # the walls any more) - bruteforce/BRUTEFORCE_NOTES.md 4.2 D1.
    _dest_ok0 = {}
    for _frm0, _to0, _c0 in mvs:
        if _to0 not in _dest_ok0:
            _dest_ok0[_to0] = ee._tsim_dest_ok(state, _to0)
    for frm, to, _comp9 in mvs:
        if frm in seen_src:
            result["effects"].append(
                f"{frm} already moved this card - move skipped")
            continue
        if to not in {sanitize(x) for x in state.board.adjacent(frm)}:
            result["effects"].append(f"{to} not adjacent to {frm}")
            continue
        if not _dest_ok0[to]:
            # ONE predicate with the spec / chain surface: a besieged
            # region is joinable only as the BESIEGER (the walled-in FP
            # garrison does not block the join - GAME19); a besieged SH
            # garrison is unjoinable; otherwise no enemy units
            if to in getattr(state, 'besieged_strongholds', set()):
                result["effects"].append(f"{to} is besieged - move refused")
            else:
                result["effects"].append(
                    f"{to} holds an enemy army - entering requires an "
                    f"attack, move refused")
            continue
        # 1.211h / row 43: a besieged Shadow garrison cannot march out.
        if ee._besieged_garrison_source(state, frm, Side.SHADOW):
            result["effects"].append(
                f"{frm} is a besieged garrison - cannot move out")
            continue
        src_key = ee._find_army_key(state, frm)
        src = state.armies.get(src_key)
        if not src or not any(u.nation in ee.SHADOW_NATIONS
                              for u in src.units):
            result["effects"].append(f"No Shadow army in {frm}")
            continue
        mv_u, mv_l = [], []
        if frm in exact:
            mv_u, mv_l = exact[frm]
            unit_ids, leader_ids = set(map(id, mv_u)), set(map(id, mv_l))
            src.units = [u for u in src.units if id(u) not in unit_ids]
            src.leaders = [l for l in src.leaders if id(l) not in leader_ids]
        elif _comp9:
            # TEACHER pure-mode: move the LOGGED composition only
            import game_state as _gs9c
            _want9 = {
                _gs9c.UnitKind.REGULAR:
                    int(_comp9.get('regular', 0) or 0),
                _gs9c.UnitKind.ELITE:
                    int(_comp9.get('elite', 0) or 0)}
            for u in list(src.units):
                if _son(u.nation) == Side.SHADOW \
                        and _want9.get(u.kind, 0) > 0:
                    src.units.remove(u)
                    mv_u.append(u)
                    _want9[u.kind] -= 1
            _wn9 = int(_comp9.get('nazgul', 0) or 0)
            _wl9 = int(_comp9.get('leader', 0) or 0)
            for l in list(src.leaders):
                if _gl9._is_saruman(l):
                    continue
                if _gl9.is_nazgul_figure(l):
                    if _wn9 > 0:
                        src.leaders.remove(l)
                        mv_l.append(l)
                        _wn9 -= 1
                elif _gl9._leader_side(l) == Side.SHADOW and _wl9 > 0:
                    src.leaders.remove(l)
                    mv_l.append(l)
                    _wl9 -= 1
        else:
            for u in list(src.units):
                if _son(u.nation) == Side.SHADOW:
                    src.units.remove(u)
                    mv_u.append(u)
            for l in list(src.leaders):
                if _gl9._is_saruman(l):
                    continue
                if _gl9._leader_side(l) == Side.SHADOW:
                    src.leaders.remove(l)
                    mv_l.append(l)
        if not src.units and not src.leaders:
            state.armies.pop(src_key, None)
        seen_src.add(frm)
        staged.append((frm, to, mv_u, mv_l))
    for frm, to, mv_u, mv_l in staged:
        dst_key = next((k for k in state.armies
                        if sanitize(k) == to), None)
        if dst_key is None:
            state.armies[to] = ee.Army(region=state.board.region(to))
            dst_key = to
        dst = state.armies[dst_key]
        own = sum(1 for u in dst.units
                  if _son(u.nation) == Side.SHADOW)
        room = _gl9.MAX_ARMY_UNITS - own
        # Land the whole moving group; the caller offers owner overstack
        # choices after all movements finish.
        # No incoming group is trimmed before another original Army leaves.
        dst.units.extend(mv_u)
        dst.leaders.extend(mv_l)
        if len(mv_u) > max(0, room):
            _ovr9 = getattr(state, '_overstack_regions', None)
            if _ovr9 is None:
                _ovr9 = set()
                state._overstack_regions = _ovr9
            _ovr9.add(to)
            result["effects"].append(
                f"over-stack at {to} recorded (excess returns to "
                f"reinforcements)")
        _gl9._check_political_activation_for_entry(state, to,
                                                   Side.SHADOW)
        _gl9._check_companion_activates_nation(state, to, mv_l)
        if not any(_son(u.nation) != Side.SHADOW for u in dst.units):
            _check_settlement_capture(state, to, Side.SHADOW)
        result["effects"].append(f"Army moved from {frm} to {to}")
    _gl9.remove_orphaned_leaders(state)
    from card_movement_witness import row as _witness_row
    state._card_army_move_witness = (CARD_ID, tuple(_witness_row(source, target, units, leaders) for source, target, units, leaders in staged))
    return result
