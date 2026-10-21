"""Observed Balrog opportunities, separate from compulsory stronghold draws.

Routes are proved from physical waypoints and equivalent legal paths. A tile
face never establishes whether Shadow used Balrog: that requires card-use
evidence or positive progression past an unused opportunity.
"""
import copy
from collections import deque


class Unproven(ValueError):
    pass


def tile_items(batch):
    """Include declaration draws before the action batch's RNG slice."""
    import rendered_extract as rx
    import rendered_census as rc
    import game_state as gs
    tiles = {int(t['line']): t for t in rx._hunt_tile_items(batch)}
    for record in batch['records']:
        if record.get('op') != 'draw':
            continue
        value = str(record.get('what', '')).strip().lower()
        # Numeric and Eye captions are unambiguous. Named special Event-card
        # captions require the extractor's contextual tile classification.
        if value not in ('eye', '0', '0r', '1', '1r', '2', '2r', '3', '3r'):
            continue
        if rc._tile_obj(gs, value) is not None:
            tiles.setdefault(int(record['line']), {'line': int(record['line']), 'value': value})
    return [tiles[line] for line in sorted(tiles)]


def observed_path(state, batch, after_line, before_line, *, declaration=False):
    import rendered_replay as rr
    import fellowship_reveal_route as route
    import game_logic as gl
    records = [r for r in batch['records'] if after_line <= int(r['line']) < before_line]
    moves = [r for r in records if r.get('op') == 'move'
             and str(r.get('obj', '')).lower() == 'the fellowship'
             and rr.zone_kind(r.get('to', ''))[0] == 'region']
    if not moves:
        raise Unproven('balrog_route_placement_missing')
    start = gl.fellowship_region(state)
    previous = start
    waypoints = []
    actor = moves[0].get('player')
    for move in moves:
        if move.get('player') != actor or rr.sanitize(move['from']) != previous:
            raise Unproven('balrog_route_gestures_disconnected')
        previous = rr.sanitize(move['to'])
        waypoints.append(previous)
    hints = [r for r in records if r.get('op') == 'route_hint']
    if any(r.get('player') != actor for r in hints):
        raise Unproven('balrog_route_hint_actor')
    if len(hints) > 1:
        raise Unproven('balrog_route_multiple_hints')
    required = hints[0]['region'] if hints else None
    target = waypoints[-1]
    already = getattr(state, '_sh_tile_drawn_regions', None) or set()
    def add(signature, region):
        strongholds, moria = signature
        if route._stronghold(state, region) and region not in already and region not in strongholds:
            strongholds += (region,)
        return strongholds, moria or region == 'moria'
    signature = add(((), False), start)
    queue = deque([(start, (start,), 0, start == required, signature)])
    seen, found = set(), {}
    while queue:
        current, path, wp, passed, sig = queue.popleft()
        while wp < len(waypoints) and current == waypoints[wp]:
            wp += 1
        key = (current, wp, passed, sig)
        if key in seen:
            continue
        seen.add(key)
        if current == target and wp == len(waypoints) and (required is None or passed):
            found.setdefault(sig, path)
            if len(found) > 1:
                raise Unproven('balrog_route_consequences_ambiguous')
        if len(path) - 1 >= state.fellowship.progress:
            continue
        for neighbor in sorted(state.board.adjacent(current)):
            queue.append((neighbor, path + (neighbor,), wp,
                          passed or neighbor == required, add(sig, neighbor)))
    if not found:
        raise Unproven('balrog_route_not_legal')
    path = next(iter(found.values()))
    route.choose_path(state, target, path, declaration=declaration)
    return target, path


def choose_pending(state, batch, after_line, before_line=None):
    """Return legal menu/index and a pre-outcome public observation boundary."""
    import ai
    import game_state as gs
    import rendered_extract as rx
    option = getattr(state, 'pending_card_option', None) or {}
    if not any('balrog_declare_draw' in o for o in option.get('options', ())):
        raise Unproven('balrog_menu_not_open')
    upper = int(before_line if before_line is not None else batch['end_line'] + 1)
    records = [r for r in batch['records'] if after_line <= int(r['line']) < upper]
    tiles = [t for t in tile_items(batch) if after_line <= int(t['line']) < upper]
    first = int(tiles[0]['line']) if tiles else upper
    ledger = rx._TABLE
    if ledger is None or not ledger.on_at('BALROG_OF_MORIA', after_line):
        raise Unproven('balrog_table_presence_unproven')
    paid = ledger.killed_in('BALROG_OF_MORIA', after_line, upper)
    event = getattr(state, '_fellowship_route_event', None) or {}
    # In a declaration the first draw can only be Balrog. On a reveal,
    # require activation before that draw to distinguish reordered human play.
    activation = [r for r in records if int(r['line']) < first and
                  ((r.get('op') == 'move' and rx._title_to_ptype(r.get('obj', '')) == 'BALROG_OF_MORIA'
                    and rx._zone_is_table(r.get('from', ''))
                    and str(r.get('from','')).strip().lower() != str(r.get('to','')).strip().lower()
                    and str(r.get('to','')).strip().lower() in ('current sa card','current fp card','current action'))
                   or (r.get('op') == 'kill' and any(rx._title_to_ptype(p) == 'BALROG_OF_MORIA'
                                                                for p in r.get('pieces', ()))))]
    if paid:
        if not tiles:
            raise Unproven('balrog_payment_without_observed_draw')
        if event.get('kind') != 'declare' and not activation:
            raise Unproven('balrog_reveal_draw_source_unproven')
        selected = True
        boundary = min([first] + [int(r['line']) for r in activation])
    else:
        if not ledger.on_at('BALROG_OF_MORIA', upper):
            raise Unproven('balrog_unclosed_table_exit')
        # A compulsory draw closes a reveal opportunity; an observed action
        # die roll/allocation closes the declaration opportunity. EOF does not.
        closes = [r for r in records if r.get('op') in ('roll_dice', 'alloc_eye')]
        if tiles and event.get('kind') == 'reveal' and getattr(state, '_pending_reveal_strongholds', ()):
            boundary = first
        elif event.get('kind') == 'declare' and closes and not tiles:
            boundary = min(int(r['line']) for r in closes)
        else:
            raise Unproven('balrog_decline_progression_unproven')
        selected = False
    menu = ai.get_legal_actions(state, gs.Side.SHADOW)
    matches = [i for i, a in enumerate(menu) if a.action_type == 'card_option'
               and a.params.get('choices') == {'balrog_declare_draw': selected}]
    if len(matches) != 1:
        raise Unproven('balrog_observed_choice_not_legal')
    return menu, matches[0], boundary, {'kind': 'observed_balrog_opportunity',
        'route_kind': event.get('kind'), 'used': selected, 'boundary': boundary,
        'first_draw_line': first if tiles else None,
        'activation_lines': [int(r['line']) for r in activation]}


def emit_declaration(state, action, batch, stats):
    import ai
    import combat
    import fellowship_reveal_route as route
    import rendered_extract as rx
    from game_state import Side
    from rendered_hunt_observation import mark
    from unittest.mock import patch
    if route.balrog_card(state) is None:
        return None

    try:
        tiles = tile_items(batch)
        stages = [int(r['line']) for r in batch['records'] if r.get('die_stage')]
        upper = min(stages, default=int(batch['end_line']) + 1)
        first = min([int(t['line']) for t in tiles if int(t['line']) < upper] + [upper])
        target, path = observed_path(state, batch, batch['start_line'], first, declaration=True)
        if target != action.params['region']:
            raise Unproven('balrog_declaration_destination_mismatch')
        if 'moria' not in path:
            return None
        picked = copy.deepcopy(action)
        picked.params['path'] = list(path)
        # The engine's teacher bypass must never erase this optional window.
        with patch.object(combat, 'FINAL_HITS_HOOK', None):
            after = ai.apply_action(state, picked)
        last_move = max(int(r['line']) for r in batch['records'] if int(r['line']) < first
                        and r.get('op') == 'move' and str(r.get('obj', '')).lower() == 'the fellowship')
        menu, index, boundary, evidence = choose_pending(after, batch, last_move + 1, upper)
        mark(after, boundary, 'before_draw')
        row = rx._emit_sample(after, Side.SHADOW, menu, index, 'card_option', batch['turn'])
        row['reactive_evidence'] = dict(evidence, path=list(path))
        stats['kinds_emit']['balrog_use' if evidence['used'] else 'balrog_decline'] += 1
        return row
    except (Unproven, ValueError) as error:
        stats['drop_reasons'][str(error)] += 1
        return None

def apply_observed(state, action, batch, next_index, live_tiles):
    """Apply a proved Balrog choice, consuming only the recorded next draws.

    A harmless Eye may immediately resume the compulsory stronghold draw.
    Returning the exact number consumed keeps those two sources separate.
    """
    import ai
    import game_logic as gl
    import game_state as gs
    import rendered_extract as rx
    import rendered_census as census
    from unittest.mock import patch
    from contextlib import ExitStack
    boundary = int(live_tiles[next_index]['line']) if next_index < len(live_tiles) else batch['end_line']+1
    if rx._RX_HUNT_HISTORY is None:
        raise Unproven('balrog_public_pool_missing')
    pool, discard, reason = rx._RX_HUNT_HISTORY.lookup(boundary)
    if reason:
        raise Unproven('balrog_public_pool:' + reason)
    state = ai.clone_state(state)
    state.hunt_pool, state.hunt_discard = pool, discard
    consumed = 0
    all_tiles = tile_items(batch)
    def pick(current):
        nonlocal consumed
        index = next_index + consumed
        if index >= len(live_tiles):
            raise Unproven('balrog_next_draw_missing')
        item = live_tiles[index]
        prior = boundary - 1 if consumed == 0 else int(live_tiles[index-1]['line'])
        next_actual = next((t for t in all_tiles if int(t['line']) > prior), None)
        if next_actual is None or int(next_actual['line']) != int(item['line']):
            raise Unproven('balrog_redraw_interpose_requires_observed_driver')
        expected = census._tile_obj(gs, str(item['value']).lower())
        keys = ('damage','is_eye','reveal','is_special','special_name',
                'stops_movement','is_dice_roll','is_fellowship_special')
        candidates = [t for t in current.hunt_pool if expected is not None
                      and all(getattr(t,key)==getattr(expected,key) for key in keys)]
        if not candidates:
            raise Unproven('balrog_tile_unavailable')
        consumed += 1
        return candidates[0]
    def pop(current):
        tile=pick(current);current.hunt_pool.remove(tile);return tile
    def unexpected(*a,**kw):
        raise Unproven('balrog_intervening_chance_unproven')
    with ExitStack() as stack:
        for name,fn in [('pick_hunt_tile',pick),('pop_hunt_tile',pop),
                        ('pick_random_companion',unexpected),('pick_random_card',unexpected),
                        ('roll_d6_batch',unexpected)]:
            stack.enter_context(patch.object(gl,name,fn))
        after=ai.apply_action(state,action)
    return after,consumed
