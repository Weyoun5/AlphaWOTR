"""Timed public table state and literal evidence for reactive card choices.

Uses the parser's normalized records, including cancelled-gesture removal.
The engine creates the decision menus; future gestures identify only labels.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
import copy
import re
from functools import wraps


class Unproven(ValueError):
    pass


def _rng_neutral(emitter):
    """Additive probes must not shift another decision's replay RNG stream."""
    @wraps(emitter)
    def run(*args, **kwargs):
        import random
        import numpy as np
        before = random.getstate(), np.random.get_state()
        try:
            return emitter(*args, **kwargs)
        finally:
            random.setstate(before[0])
            np.random.set_state(before[1])
    return run


def table_slot(zone):
    return re.fullmatch(r'board\d+(?:\s*\(chosen\))?', str(zone).strip(), re.I) is not None


class Timeline:
    def __init__(self, game):
        import rendered_extract as rx
        import rendered_replay as rr
        import persistent_effects as pe
        import game_logic as gl
        from game_state import Side
        accepted_stages = {int(batch['start_line']) for batch in game.get('batches', ())
                           if batch.get('die_side') is not None}
        self.completed_action_lines = sorted({int(record['line'])
            for record in game.get('net_records', ())
            if record.get('op') == 'turn' or (record.get('die_stage')
                                            and int(record['line']) in accepted_stages)})
        events = []
        first = {}
        for batch in game['batches']:
            first.setdefault(batch['turn'], batch)
        # The same pre-action nomination evidence used by the declaration
        # emitter. A later physical discard cannot resurrect an expired card.
        for batch in first.values():
            pre = []
            for record in batch['records']:
                if record.get('die_stage'):
                    break
                pre.append(record)
            tiles = [int(r['line']) for r in pre if r.get('op') == 'draw'
                     and rr.parse_piece(str(r.get('what', '')))
                     and rr.parse_piece(str(r.get('what', '')))[0] == 'tile']
            boundary = min(tiles, default=10**12)
            moves = [r for r in pre if r.get('op') == 'move'
                     and str(r.get('obj', '')).lower() == 'the fellowship'
                     and r['line'] < boundary and rr.zone_kind(r.get('to', ''))[0] == 'region']
            snap, anchor = None, None
            if moves and (moves[0].get('decision_truth') or {}).get('fel', {}).get('hidden'):
                snap = copy.deepcopy(moves[0]['decision_truth'])
                snap['fel']['region'] = rr.zone_kind(moves[-1]['to'])[1]
                anchor = int(moves[-1]['line'])
            elif not moves and not tiles:
                before = batch['truth_before']['fel']
                if before.get('hidden') and int(before.get('progress', 0)) == 0:
                    for record in pre:
                        if record.get('op') != 'move' or str(record.get('obj', '')).lower() != 'corruption':
                            continue
                        origin = re.fullmatch(r'Fellowship Track (\d+)', str(record.get('from', '')))
                        target = re.fullmatch(r'Fellowship Track (\d+)', str(record.get('to', '')))
                        if origin and target and int(origin[1]) == int(target[1]) + 1:
                            snap, anchor = copy.deepcopy(batch['truth_before']), int(record['line'])
                            break
            if snap is not None:
                state = rx._recon_state(snap, batch['turn'])
                # check_fellowship_declared_discards reads current cards; install
                # the two affected types only to ask the engine about location.
                for title in ('Flocks of Crebain', 'Worn with Sorrow and Toil'):
                    state.persistent_cards.add_card(pe.create_persistent_card(rx._title_to_cid(title), title, Side.SHADOW))
                expired = pe.check_fellowship_declared_discards(state, gl.fellowship_region(state))
                if expired:
                    events.append((anchor, 'expire', None))
        for record in game.get('net_records', []):
            op = record.get('op')
            tokens = record.get('pieces', []) if op == 'kill' else rx._group_tokens(record.get('obj', ''))
            for title in tokens:
                cid = rx._title_to_cid(title)
                if not cid:
                    continue
                try:
                    card = pe.create_persistent_card(cid, str(title).split('/')[0],
                        Side.FREE_PEOPLES if cid.startswith('FREE_PEOPLES') else Side.SHADOW)
                except ValueError:
                    continue
                if (op == 'move' and table_slot(record.get('to', ''))
                        and not table_slot(record.get('from', ''))):
                    events.append((int(record['line']), 'on', card))
                elif op == 'kill' or (op == 'move' and
                        (table_slot(record.get('from', '')) or
                         rr.zone_kind(record.get('to', ''))[0] in ('cas', 'reinf', 'hand'))):
                    events.append((int(record['line']), 'off', card))
        self.lines, self.states = [], []
        current = {}
        for line, kind, card in sorted(events, key=lambda item: item[0]):
            if kind == 'on':
                current[card.card_id] = card
            elif kind == 'off':
                current.pop(card.card_id, None)
            else:
                current = {cid: value for cid, value in current.items()
                           if value.card_type.name not in ('FLOCKS_OF_CREBAIN', 'WORN_WITH_SORROW')}
            self.lines.append(line)
            self.states.append(dict(current))

    def cards_at(self, line):
        index = bisect_left(self.lines, int(line)) - 1
        return {} if index < 0 else self.states[index]

    def install(self, state, line, *, crebain=False, worn=False):
        from game_state import Side
        import game_logic as gl
        cards = self.cards_at(line)
        for card in cards.values():
            wanted = ((crebain and card.card_type.name == 'FLOCKS_OF_CREBAIN') or
                      (worn and (card.card_type.name == 'WORN_WITH_SORROW' or
                                 (card.side == Side.FREE_PEOPLES and gl.card_deck_of(card.card_id) == 'character'))))
            if wanted and state.persistent_cards.get_by_type(card.card_type) is None:
                state.persistent_cards.add_card(copy.deepcopy(card))


def crebain_choice(batch, before_line=None):
    """A complete pre-roll discard is use; a card retained through rolls is keep."""
    import rendered_extract as rx
    rolls = [int(r['line']) for r in batch.get('rng', []) if r.get('kind') == 'roll_num']
    if not rolls:
        raise Unproven('crebain_no_roll')
    first_roll = min(rolls)
    lower = int(before_line or batch.get('start_line') or 0)
    exits, kills = [], []
    for record in batch['records']:
        if int(record['line']) < lower:
            continue
        op = record.get('op')
        if op == 'move' and 'flocks of crebain' in str(record.get('obj', '')).lower():
            if table_slot(record.get('from', '')) and not table_slot(record.get('to', '')):
                exits.append(int(record['line']))
        elif op == 'kill' and any('flocks of crebain' in str(p).lower() for p in record.get('pieces', [])):
            kills.append(int(record['line']))
    if exits or kills:
        if not kills or min(exits + kills) >= first_roll:
            raise Unproven('crebain_late_or_unclosed_discard')
        return True, min(exits + kills)
    return False, first_roll


def emit_crebain(batch, stats, timeline):
    import ai
    import rendered_extract as rx
    import rendered_census as rc
    from game_state import Side
    if timeline is None or batch.get('die_side') != 'FREE_PEOPLES' or not rc._has_fsp_move(batch):
        return []
    state, side = rx._mk_state_with_die(batch, install_hands=True)
    if state is None or state.fellowship.mordor_entered:
        return []
    timeline.install(state, batch['start_line'], crebain=True)
    if not any(c.card_type.name == 'FLOCKS_OF_CREBAIN' for c in state.persistent_cards.sh_cards):
        return []
    try:
        use, line = crebain_choice(batch)
        rx._prepare_fellowship_dice(state, batch, pending=False)
        legal = ai.get_legal_actions(state, side)
        actions = [a for a in legal if a.action_type == 'fellowship_move' and a.die_idx == 0]
        if len(actions) != 1:
            raise Unproven('crebain_move_unmatched')
        pending = ai.apply_action(state, actions[0])
        menu = ai.get_legal_actions(pending, Side.SHADOW)
        choices = [i for i, a in enumerate(menu) if a.action_type == 'card_option'
                   and (a.params.get('choices') or {}).get('crebain_discard') is use]
        if len(choices) != 1:
            raise Unproven('crebain_menu_unmatched')
        from rendered_hunt_observation import mark
        # The executor has stopped before rolling or drawing. A generic
        # snapshot marker cannot certify that a pending menu precedes the
        # tile which will be drawn later in the same physical action.
        mark(pending, int(batch['start_line']), 'before_draw')
        stats['kinds_emit']['cardopt_crebain_' + ('use' if use else 'keep')] += 1
        return [rx._emit_sample(pending, Side.SHADOW, menu, choices[0], 'card_option', batch['turn'])]
    except Unproven as error:
        stats['drop_reasons'][str(error)] += 1
        return []


def recorded_hand_discard(batch, lower, upper):
    import rendered_extract as rx
    if rx._LEDGER is None or not rx._LEDGER.enabled:
        raise Unproven('worn_hand_identity_unproven')
    records = [r for r in batch['records'] if lower < int(r['line']) < upper
               and re.search(r'randomly discards a Free Character Card\.', str(r.get('body', '')))]
    if not records:
        raise Unproven('worn_no_random_payment_gesture')
    line = min(int(r['line']) for r in records)
    gone = Counter(rx._LEDGER.hand_at('fp', line)) - Counter(rx._LEDGER.hand_at('fp', line + 1))
    if sum(gone.values()) != 1:
        raise Unproven('worn_hand_identity_unproven')
    return line, next(iter(gone))


def prepare_worn_trigger(state, batch, lower, upper, timeline):
    """Install public table and owner hands immediately before a Hunt casualty.

    Callers must prove that this is a Fellowship Hunt casualty, including
    card-triggered Hunts. Ordinary battle eliminations are not eligible.
    The upper boundary identifies the end of this casualty's discard window;
    it must not be extended across another casualty or Hunt tile.
    """
    import rendered_extract as rx
    if timeline is None or rx._LEDGER is None or not rx._LEDGER.enabled:
        raise Unproven('worn_trigger_observation_unproven')
    if int(lower) >= int(upper):
        raise Unproven('worn_trigger_window_invalid')
    timeline.install(state, int(lower), worn=True)
    rx._install_ledger_hands(state, batch, at_line=int(lower))
    return int(lower), int(upper)


def worn_casualty_window(batch, victim_line, exits, tiles, timeline):
    """Bound one casualty's payment without treating EOF as a decline.

    A later casualty, Hunt tile, die stage or turn proves progression. Batch
    end alone does not. The exclusive upper endpoint includes a payment on
    the last record line, while stopping before the next casualty's window.
    """
    lower = int(victim_line)
    victims = [int(line) for line in exits.values()]
    if victims.count(lower) > 1:
        raise Unproven('worn_simultaneous_victims_unproven')
    boundaries = [line for line in victims if line > lower]
    boundaries.extend(int(tile['line']) for tile in tiles if int(tile['line']) > lower)
    closures = getattr(timeline, 'completed_action_lines', ())
    index = bisect_right(closures, lower)
    if index < len(closures):
        boundaries.append(closures[index])
    end = int(batch['end_line']) + 1
    upper = min([end] + boundaries)
    if lower >= upper:
        raise Unproven('worn_trigger_window_invalid')
    return lower, upper, upper in boundaries


def drain_worn(state, batch, stats, lower, upper, *, complete_window=False):
    """Resolve the engine-opened payment using its actual recorded source.

    Named discarded table cards are public; a random hand discard's identity
    is used only to restore the owner's hand after the decision is encoded.
    Silence proves decline only when the caller certifies a completed physical
    decision window. An incomplete table-card gesture never proves decline.
    """
    import ai
    import game_logic as gl
    import rendered_extract as rx
    from game_state import Side
    from unittest.mock import patch
    output = []
    for _ in range(9):
        pending = getattr(state, 'pending_card_option', None) or {}
        if pending.get('card_name') != '__wwsat_discard__':
            return state, output
        menu = ai.get_legal_actions(state, Side.SHADOW)
        candidates = []
        for record in batch['records']:
            line = int(record['line'])
            if not lower < line < upper:
                continue
            body = str(record.get('body', ''))
            if record.get('op') == 'other' and re.search(r'randomly discards a Free Character Card\.', body):
                candidates.append((line, {'wwsat_source': 'hand'}, None))
            elif record.get('op') == 'kill':
                for title in record.get('pieces', []):
                    cid = rx._title_to_cid(title)
                    if cid and any((a.params.get('choices') or {}).get('card_id') == cid for a in menu):
                        candidates.append((line, {'wwsat_source': 'table', 'card_id': cid}, cid))
        if not candidates:
            if not complete_window:
                raise Unproven('worn_no_payment_gesture')
            table_targets = {a.params.get('choices', {}).get('card_id')
                             for a in menu
                             if a.params.get('choices', {}).get('wwsat_source') == 'table'}
            for record in batch['records']:
                if (not lower < int(record['line']) < upper
                        or record.get('op') != 'move'
                        or not table_slot(record.get('from', ''))
                        or table_slot(record.get('to', ''))):
                    continue
                if any(rx._title_to_cid(title) in table_targets
                       for title in rx._group_tokens(record.get('obj', ''))):
                    raise Unproven('worn_unclosed_table_discard')
            candidates = [(upper, {'wwsat_source': 'decline'}, None)]
        line = min(item[0] for item in candidates)
        evidence = [item for item in candidates if item[0] == line]
        if len(evidence) != 1:
            raise Unproven('worn_multiple_payments')
        _, choice, discarded = evidence[0]
        matching = [i for i, a in enumerate(menu) if a.action_type == 'card_option'
                    and a.params.get('choices') == choice]
        if len(matching) != 1:
            raise Unproven('worn_payment_not_legal')
        if choice['wwsat_source'] == 'hand':
            _recorded_line, discarded = recorded_hand_discard(batch, lower, upper)
        sample = rx._emit_sample(state, Side.SHADOW, menu, matching[0], 'card_option', batch['turn'])
        def observed_card(cards, ctx=''):
            if discarded not in cards or ctx != 'WWSAT hand discard':
                raise Unproven('worn_wrong_random_context')
            return discarded
        with patch.object(gl, 'pick_random_card', observed_card):
            state = ai.apply_action(state, menu[matching[0]])
        output.append(sample)
        stats['kinds_emit']['cardopt_worn_' + choice['wwsat_source']] += 1
        lower = line
    raise Unproven('worn_unfinished_chain')


@_rng_neutral
def emit_foul_worn(batch, stats, timeline):
    """Reach Worn through the played Event and its recorded random casualty.

    Foul Thing's compulsory random victim is a chance outcome, never a
    learned sacrifice arm. A delayed physical Worn payment can label the
    earlier engine decision only if it still has a unique casualty and its
    FP hand/table payment sources have not changed. Later Shadow draws and
    a harmless reveal Eye are not copied into that earlier observation.
    """
    import ai
    import game_logic as gl
    import game_state as gs
    import rendered_census as census
    import rendered_extract as rx
    import rendered_replay as rr
    import numpy as np
    from unittest.mock import patch
    from rendered_hunt_observation import mark

    cid = rx._title_to_cid(rx._card_play_title(batch) or '')
    if cid != 'SHADOW_Character_Foul_Thing_from_the_Deep':
        return []
    if timeline is None or rx._LEDGER is None or not rx._LEDGER.enabled:
        stats['drop_reasons']['foul_worn_missing_history'] += 1
        return []
    try:
        tiles = rx._hunt_tile_items(batch)
        if not tiles:
            raise Unproven('foul_worn_missing_tile')
        tile_line = int(tiles[0]['line'])
        public = timeline.cards_at(tile_line)
        if not any(c.card_type.name == 'WORN_WITH_SORROW' for c in public.values()):
            return []
        # A named companion draw is the random pick, whereas a guide move
        # can be a voluntary reduction and cannot stand in for this outcome.
        victims = []
        next_tile = int(tiles[1]['line']) if len(tiles) > 1 else int(batch['end_line']) + 1
        for record in batch['records']:
            if record.get('op') != 'draw' or not tile_line < int(record['line']) < next_tile:
                continue
            piece = rr.parse_piece(str(record.get('what', '')))
            if piece and piece[0] == 'named' and str(piece[1]) in rx._comp_exit_lines(batch):
                victims.append((int(record['line']), rr.COMP_ALIAS.get(str(piece[1]).lower(), piece[1])))
        if len(victims) != 1:
            raise Unproven('foul_worn_random_victim_unproven')
        victim_line, victim = victims[0]
        exits = rx._comp_exit_lines(batch)
        if len(exits) != 1 or victim not in exits:
            raise Unproven('foul_worn_multiple_or_unproved_casualties')
        lower, upper, complete = worn_casualty_window(batch, victim_line, exits, tiles, timeline)
        # Prefer the ordinary immediate window. Extend only for an explicit
        # hand payment; silence after another tile never means "decline".
        try:
            payment_line, _ = recorded_hand_discard(batch, lower, upper)
        except Unproven as error:
            if str(error) != 'worn_no_random_payment_gesture':
                raise
            payment_line = None
        if payment_line is None:
            end = int(batch['end_line']) + 1
            later = [r for r in batch['records'] if upper <= int(r['line']) < end
                     and re.search(r'randomly discards a Free Character Card\.', str(r.get('body', '')))]
            if later:
                if len(later) != 1:
                    raise Unproven('foul_worn_multiple_late_payments')
                payment_line = int(later[0]['line'])
                if any(victim_line < line <= payment_line for line in timeline.completed_action_lines):
                    raise Unproven('foul_worn_payment_crosses_action')
                if batch['truth_before']['fel'].get('mordor'):
                    raise Unproven('foul_worn_late_mordor_payment')
                for item in tiles[1:]:
                    if int(item['line']) >= payment_line:
                        break
                    observed = census._tile_obj(gs, str(item['value']).lower())
                    if observed is None or not observed.is_eye:
                        raise Unproven('foul_worn_late_damage_window')
                if Counter(rx._LEDGER.hand_at('fp', victim_line)) != Counter(rx._LEDGER.hand_at('fp', payment_line)):
                    raise Unproven('foul_worn_late_hand_changed')
                fp_cards = lambda at: {key for key, card in timeline.cards_at(at).items()
                                      if card.side == gs.Side.FREE_PEOPLES}
                if fp_cards(victim_line) != fp_cards(payment_line):
                    raise Unproven('foul_worn_late_table_changed')
                upper, complete = payment_line + 1, False

        initial, side = rx._mk_state_with_die(batch, install_hands=True)
        if initial is None or side != gs.Side.SHADOW:
            raise Unproven('foul_worn_initial_state_unproven')
        timeline.install(initial, int(batch['start_line']), worn=True)
        # Keep both players' real remaining dice, then let the Event consume
        # its own die. The generic encoding-time batch snapshot is pre-play.
        pools = getattr(initial, '_rx_full_pools', None)
        if pools is not None:
            initial.dice[gs.Side.FREE_PEOPLES] = rx._install_full_pools(pools)[gs.Side.FREE_PEOPLES]
        initial._rx_full_pools = None
        if rx._RX_HUNT_HISTORY is not None:
            pool, discarded, refusal = rx._RX_HUNT_HISTORY.lookup(tile_line)
            if refusal:
                raise Unproven('foul_worn_pool:' + refusal)
            initial.hunt_pool, initial.hunt_discard = pool, discarded
        faces = [rx._effective_lead_letter(batch)]
        if rx._ring_counter_used(batch) == 'sh' and faces[0] == batch['die_face']:
            # Both Character and Event may fund this same physical Ring use.
            # They must reach identical decision observations and menus below;
            # this does not create an inferred Ring-face training target.
            faces = ['P', 'C']

        reached = []
        for face in faces:
            state = ai.clone_state(initial)
            rx._set_effective_lead_face(state, side, rx._letter2face()[face])
            plays = [a for a in ai.get_legal_actions(state, side)
                     if a.action_type == 'play_event' and a.die_idx == 0 and a.params.get('card_id') == cid]
            if len(plays) != 1:
                raise Unproven('foul_worn_event_not_legal')
            drawn, picked = [], []
            boundary = tile_line + 1
            def draw_tile(current):
                if drawn:
                    raise Unproven('foul_worn_unproved_intervening_draw')
                obj = census._tile_obj(gs, str(tiles[0]['value']).lower())
                keys = ('damage', 'is_eye', 'reveal', 'is_special', 'special_name',
                        'stops_movement', 'is_dice_roll', 'is_fellowship_special')
                matching = [tile for tile in current.hunt_pool
                            if obj is not None and all(getattr(tile,k) == getattr(obj,k) for k in keys)]
                if not matching:
                    raise Unproven('foul_worn_recorded_tile_not_in_pool')
                drawn.append(tile_line)
                return matching[0]
            def draw_victim(candidates, ctx=''):
                matching = [c for c in candidates if rr._named_eq(c.name, victim)]
                if ctx != 'Foul Thing victim' or picked or len(matching) != 1:
                    raise Unproven('foul_worn_random_context_unproven')
                picked.append(victim)
                return matching[0]
            def unexpected(*args, **kwargs):
                raise Unproven('foul_worn_unrecorded_randomness')
            with patch.object(gl, 'pick_hunt_tile', draw_tile), \
                    patch.object(gl, 'pick_random_companion', draw_victim), \
                    patch.object(gl, 'pick_random_card', unexpected), \
                    patch.object(gl, 'roll_d6_batch', unexpected):
                state = ai.apply_action(state, plays[0])
                for _ in range(12):
                    option = getattr(state, 'pending_card_option', None) or {}
                    if option.get('card_name') == '__wwsat_discard__':
                        break
                    pending = getattr(state, 'pending_hunt', None) or {}
                    menu = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
                    if pending.get('phase') == 'hobbit_place':
                        proof = []
                        for record in batch['records']:
                            if (record.get('op') != 'move' or not victim_line < int(record['line']) < upper
                                    or not rr._named_eq(str(record.get('obj', '')), victim)):
                                continue
                            zone = rr.zone_kind(record.get('to', ''))
                            if zone[0] == 'region':
                                proof.append((int(record['line']), rr.sanitize(zone[1])))
                        if len(proof) != 1:
                            raise Unproven('foul_worn_hobbit_placement_unproven')
                        boundary, destination = proof[0]
                        boundary += 1
                        matched = [a for a in menu if a.action_type == 'companion_separate'
                                   and rr.sanitize(a.params.get('to_region', '')) == destination]
                    elif getattr(state, 'pending_guide_change', None):
                        proof = [r for r in batch['records'] if r.get('op') == 'move'
                                 and victim_line < int(r['line']) < upper
                                 and str(r.get('to', '')).lower() == 'guide of the fellowship']
                        if len(proof) != 1:
                            raise Unproven('foul_worn_guide_unproven')
                        boundary = int(proof[0]['line']) + 1
                        matched = [a for a in menu if a.action_type == 'choose_guide'
                                   and rr._named_eq(str(a.params.get('companion', '')), str(proof[0]['obj']))]
                    else:
                        raise Unproven('foul_worn_intervening_choice_unproven')
                    if len(matched) != 1:
                        raise Unproven('foul_worn_intervening_action_not_legal')
                    state = ai.apply_action(state, matched[0])
                else:
                    raise Unproven('foul_worn_unfinished_chain')
            if drawn != [tile_line] or picked != [victim]:
                raise Unproven('foul_worn_chance_not_consumed')
            mark(state, boundary, 'after_draw')
            menu = ai.get_legal_actions(state, gs.Side.SHADOW)
            reached.append((state, menu))
        state, menu = reached[0]
        for alternate, other_menu in reached[1:]:
            if ([(a.action_type,a.params) for a in menu] != [(a.action_type,a.params) for a in other_menu]
                    or not np.array_equal(ai.encode_state(state, gs.Side.SHADOW), ai.encode_state(alternate, gs.Side.SHADOW))):
                raise Unproven('foul_worn_funding_changes_observation')
        # Foul's physical tail can defer this payment beyond a reveal tile.
        # Until that whole tail has a separate completion certificate, require
        # an explicit hand/table payment rather than inferring a decline.
        resumed_tiles = []
        def resume_tile(current):
            index = 1 + len(resumed_tiles)
            if index >= len(tiles):
                raise Unproven('foul_worn_missing_reveal_tile')
            item = tiles[index]
            reveal = [record for record in batch['records']
                      if record.get('op') == 'flip'
                      and str(record.get('obj', '')).lower() == 'the fellowship'
                      and victim_line < int(record['line']) < int(item['line'])]
            if (len(reveal) != 1 or rr.sanitize(str(reveal[0].get('region', '')))
                    != rr.sanitize(gl.fellowship_region(current))):
                raise Unproven('foul_worn_reveal_region_unproven')
            obj = census._tile_obj(gs, str(item['value']).lower())
            if obj is None or not obj.is_eye:
                raise Unproven('foul_worn_reveal_tail_not_harmless')
            matching = [tile for tile in current.hunt_pool if tile.is_eye and not tile.is_special]
            if not matching:
                raise Unproven('foul_worn_reveal_tile_not_in_pool')
            resumed_tiles.append(int(item['line']))
            return matching[0]
        with patch.object(gl, 'pick_hunt_tile', resume_tile), \
                patch.object(gl, 'pick_random_companion', unexpected), \
                patch.object(gl, 'roll_d6_batch', unexpected):
            _, rows = drain_worn(state, batch, stats, lower, upper, complete_window=False)
        for row in rows:
            row['reactive_evidence'] = {'source_card_id': cid, 'random_victim': victim,
                'tile_line': tile_line, 'victim_line': victim_line,
                'decision_boundary': boundary, 'payment_line': payment_line,
                'executed_tile_lines': [tile_line] + resumed_tiles,
                'kind': 'observed_payment_after_actual_event',
                'delayed_gesture': bool(payment_line and payment_line >= next_tile)}
        stats['detected']['foul_worn_actual_event'] += bool(rows)
        return rows
    except Unproven as error:
        stats['drop_reasons'][str(error)] += 1
        return []


@_rng_neutral
def emit_shadow_hunt_reaction(batch, stats):
    """Replay a Shadow Event through its evidenced FP protection choices.

    The FP die never moves for this reaction. Later choices are not guessed:
    This prefix includes Staff, Mithril and the resulting reduction choice;
    casualty/guide/reveal tails remain with their separate replay owners.
    """
    import ai
    import game_logic as gl
    import game_state as gs
    import rendered_census as census
    import rendered_extract as rx
    import rendered_replay as rr
    import numpy as np
    from unittest.mock import patch
    from rendered_hunt_observation import mark
    if str(batch.get('kind', '')).startswith('combat'):
        return []  # A printed combat face is not this card's Event.
    cid = rx._title_to_cid(rx._card_play_title(batch) or '')
    if cid not in ('SHADOW_Character_Foul_Thing_from_the_Deep',
                   "SHADOW_Character_Isildur's_Bane", 'SHADOW_Character_Orc_Patrol'):
        return []
    lo, hi = int(batch['start_line']), int(batch['end_line']) + 1
    if rx._TABLE is None or not any(rx._TABLE.on_at(kind, lo) for kind in rx._HUNT_PTYPES):
        return []
    try:
        initial, side = rx._mk_state_with_die(batch, install_hands=True)
        if initial is None or side != gs.Side.SHADOW:
            raise Unproven('shadow_hunt_reaction_state_unproven')
        pools = getattr(initial, '_rx_full_pools', None)
        if pools is not None:
            initial.dice[gs.Side.FREE_PEOPLES] = rx._install_full_pools(pools)[gs.Side.FREE_PEOPLES]
        initial._rx_full_pools = None
        if rx._RX_HUNT_HISTORY is not None:
            pool, discarded, refusal = rx._RX_HUNT_HISTORY.lookup(lo)
            if refusal:
                raise Unproven('shadow_hunt_reaction_pool:' + refusal)
            initial.hunt_pool, initial.hunt_discard = pool, discarded
        tiles = rx._hunt_tile_items(batch)
        faces = [rx._effective_lead_letter(batch)]
        if rx._ring_counter_used(batch) == 'sh' and faces[0] == batch['die_face']:
            faces = ['P', 'C']
        reached = []
        stops = []
        for face in faces:
            state = ai.clone_state(initial)
            rx._set_effective_lead_face(state, side, rx._letter2face()[face])
            plays = [a for a in ai.get_legal_actions(state, side) if a.action_type == 'play_event'
                     and a.die_idx == 0 and a.params.get('card_id') == cid]
            if len(plays) != 1:
                raise Unproven('shadow_hunt_reaction_event_not_legal')
            drawn = []
            def observed_tile(current):
                if len(drawn) >= len(tiles):
                    raise Unproven('shadow_hunt_reaction_tile_unproven')
                item = tiles[len(drawn)]
                obj = census._tile_obj(gs, str(item['value']).lower())
                keys = ('damage','is_eye','reveal','is_special','special_name',
                        'stops_movement','is_dice_roll','is_fellowship_special')
                matching = [tile for tile in current.hunt_pool if obj is not None
                            and all(getattr(tile,key) == getattr(obj,key) for key in keys)]
                if not matching:
                    raise Unproven('shadow_hunt_reaction_tile_not_in_pool')
                drawn.append(int(item['line']))
                return matching[0]
            def unexpected(*args, **kwargs):
                raise Unproven('shadow_hunt_reaction_later_chance_unproven')
            def observed_pop(current):
                tile=observed_tile(current)
                current.hunt_pool.remove(tile)
                return tile
            with patch.object(gl,'pick_hunt_tile',observed_tile), \
                    patch.object(gl,'pop_hunt_tile',observed_pop), \
                    patch.object(gl,'pick_random_companion',unexpected), \
                    patch.object(gl,'pick_random_card',unexpected), \
                    patch.object(gl,'roll_d6_batch',unexpected):
                state = ai.apply_action(state, plays[0])
            steps = []
            stop_reason = None
            for _ in range(12):
                if not getattr(state,'pending_card_option',None):
                    break
                try:
                    menu,index,key,boundary,timing = _shadow_reaction_decision(state,batch,tiles,drawn)
                except Unproven as error:
                    if steps:
                        stop_reason = str(error)
                        break  # Preserve only the already proved prefix.
                    raise
                mark(state,boundary,timing)
                steps.append((state,menu,index,key,boundary,timing))
                choice = menu[index].params['choices']
                if key=='use_hunt_reduction' or choice.get('wizardstaff_prevent'):
                    break
                # Only chance outcomes still recorded in this physical Event
                # may advance the direct protection chain. A later casualty
                # or unsupported choice does not erase a proved earlier row.
                try:
                    with patch.object(gl,'pick_hunt_tile',observed_tile), \
                            patch.object(gl,'pop_hunt_tile',observed_pop), \
                            patch.object(gl,'pick_random_companion',unexpected), \
                            patch.object(gl,'pick_random_card',unexpected), \
                            patch.object(gl,'roll_d6_batch',unexpected):
                        state=ai.apply_action(state,menu[index])
                except Unproven as error:
                    stop_reason = str(error)
                    break
            if not steps:
                raise Unproven('shadow_hunt_reaction_intervening_choice_unproven')
            reached.append(steps)
            stops.append(stop_reason)
        for alternative in reached[1:]:
            if len(alternative)!=len(reached[0]):
                raise Unproven('shadow_hunt_reaction_funding_changes_observation')
            for first,second in zip(reached[0],alternative):
                state,menu,index,*_=first
                other,other_menu,other_index,*_=second
                if (index!=other_index or [(a.action_type,a.params) for a in menu]!=[(a.action_type,a.params) for a in other_menu]
                        or not np.array_equal(ai.encode_state(state,gs.Side.FREE_PEOPLES),ai.encode_state(other,gs.Side.FREE_PEOPLES))):
                    raise Unproven('shadow_hunt_reaction_funding_changes_observation')
        if stops[0]:
            bucket = 'detected' if stops[0]=='shadow_hunt_reaction_later_chance_unproven' else 'drop_reasons'
            stats[bucket]['shadow_event_prefix_stop:' + stops[0]] += 1
        output=[]
        for state,menu,index,key,boundary,timing in reached[0]:
            sample=rx._emit_sample(state,gs.Side.FREE_PEOPLES,menu,index,'card_option',batch['turn'])
            sample['reactive_evidence']={'source_card_id':cid,'kind':'actual_shadow_event_protection_chain',
                                         'decision_boundary':boundary,'timing':timing,'choice_key':key}
            stats['kinds_emit']['shadow_event_'+key]+=1
            output.append(sample)
        return output
    except Unproven as error:
        stats['drop_reasons'][str(error)]+=1
        return []


def _shadow_reaction_decision(state,batch,tiles,drawn):
    import ai
    import game_state as gs
    import rendered_extract as rx
    option=getattr(state,'pending_card_option',None) or {}
    choices=option.get('options') or []
    keys=[key for key in ('wizardstaff_prevent','mithril_redraw','use_hunt_reduction')
          if any(key in choice for choice in choices)]
    if len(keys)!=1:
        raise Unproven('shadow_hunt_reaction_intervening_choice_unproven')
    key=keys[0]
    lo,hi=int(batch['start_line']),int(batch['end_line'])+1
    next_line=int(tiles[len(drawn)]['line']) if len(drawn)<len(tiles) else hi
    tile_line=drawn[-1] if drawn else lo
    if key=='wizardstaff_prevent':
        use=rx._TABLE.killed_in('WIZARD_STAFF',tile_line,next_line)
        if not use and next_line==hi:
            raise Unproven('shadow_hunt_staff_decision_unproven')
        selected={'wizardstaff_prevent':use}
        boundary,timing=(next_line if next_line!=hi else tile_line),'before_draw'
    elif key=='mithril_redraw':
        use=rx._TABLE.killed_in('MITHRIL_COAT',tile_line,next_line)
        if use and next_line==hi:
            raise Unproven('shadow_hunt_mithril_replacement_unproven')
        if not use and not rx._TABLE.on_at('MITHRIL_COAT',next_line):
            raise Unproven('shadow_hunt_mithril_unclosed_discard')
        if not use and not _shadow_hunt_resolution_proved(batch,tile_line,next_line):
            raise Unproven('shadow_hunt_mithril_keep_unproven')
        selected={'mithril_redraw':use}
        boundary,timing=tile_line+1,'after_draw'
    else:
        exits=[int(line) for line in rx._comp_exit_lines(batch).values() if int(line)>tile_line]
        payment_end=min([next_line]+exits)
        paid=[kind for kind in ('AXE_AND_BOW','HORN_OF_GONDOR')
              if rx._TABLE.killed_in(kind,tile_line,payment_end)]
        if len(paid)>1:
            raise Unproven('shadow_hunt_multiple_reduction_payments')
        if paid:
            selected={'use_hunt_reduction':1,'card_id':rx._TABLE.cid[paid[0]]}
        else:
            if not _shadow_hunt_resolution_proved(batch,tile_line,next_line):
                raise Unproven('shadow_hunt_reduction_decline_unproven')
            selected={'use_hunt_reduction':0}
        boundary,timing=tile_line+1,'after_draw'
    menu=ai.get_legal_actions(state,gs.Side.FREE_PEOPLES)
    matching=[i for i,a in enumerate(menu) if a.action_type=='card_option' and a.params.get('choices')==selected]
    if len(matching)!=1:
        raise Unproven('shadow_hunt_reaction_choice_not_legal')
    return menu,matching[0],key,boundary,timing


def _shadow_hunt_resolution_proved(batch, tile_line, upper=None):
    """Positive physical progress past a protection opportunity, not EOF."""
    import rendered_extract as rx
    import rendered_replay as rr
    upper=int(batch['end_line'])+1 if upper is None else int(upper)
    if any(tile_line<int(line)<upper for line in rx._comp_exit_lines(batch).values()):
        return True
    for record in batch['records']:
        if not tile_line<int(record['line'])<upper:
            continue
        if record.get('op')=='move' and str(record.get('obj','')).lower()=='corruption':
            return True
        if record.get('op')=='kill':
            for title in record.get('pieces', []):
                piece=rr.parse_piece(title)
                if piece and piece[0]=='die' and 'shadow action die' in str(title).lower():
                    return True
    return False


def _observed_reveal_route(state, batch, after_line, before_line):
    """Literal placement plus an optional declared waypoint, with legal edges.

    A canonical path only removes immaterial route detail: it may not silently
    choose between different Shadow-stronghold draws or a Balrog opportunity.
    """
    from collections import deque
    import game_logic as gl
    import fellowship_reveal_route as route
    import rendered_replay as rr
    records = [record for record in batch['records']
               if after_line < int(record['line']) < before_line]
    moves = [record for record in records if record.get('op') == 'move'
             and str(record.get('obj', '')).lower() == 'the fellowship'
             and rr.zone_kind(record.get('to', ''))[0] == 'region']
    if len(moves) != 1:
        raise Unproven('axebow_reveal_placement_unproven')
    move = moves[0]
    actors = {record.get('player') for record in batch['records'] if record.get('die_stage')}
    if len(actors) != 1 or move.get('player') not in actors:
        raise Unproven('axebow_reveal_actor_unproven')
    if rr.sanitize(move.get('from', '')) != gl.fellowship_region(state):
        raise Unproven('axebow_reveal_origin_unproven')
    target = rr.sanitize(move['to'])
    hints = [record for record in records if record.get('op') == 'route_hint']
    if (len(hints) > 1 or any(record.get('player') != move.get('player')
                              for record in hints)):
        raise Unproven('axebow_reveal_route_ambiguous')
    if route.balrog_card(state) is not None:
        raise Unproven('axebow_reveal_balrog_choice_unproven')
    if not hints:
        # Missing chat is not automatically missing information: every legal
        # route can have the same draw consequences. Prove that equivalence
        # rather than treating the engine's preferred path as the human's.
        start = gl.fellowship_region(state)
        already = getattr(state, '_sh_tile_drawn_regions', None) or set()
        signature = (start,) if route._stronghold(state, start) and start not in already else ()
        queue = deque([(start, (start,), signature)])
        visited, destinations = set(), {}
        while queue:
            current, path, signature = queue.popleft()
            key = (current, signature)
            if key in visited:
                continue
            visited.add(key)
            if current == target:
                destinations.setdefault(signature, path)
                if len(destinations) > 1:
                    raise Unproven('axebow_reveal_route_ambiguous')
            if len(path) - 1 >= state.fellowship.progress:
                continue
            for neighbor in sorted(state.board.adjacent(current)):
                extra = route._stronghold(state, neighbor) and neighbor not in already and neighbor not in signature
                queue.append((neighbor, path + (neighbor,), signature + ((neighbor,) if extra else ())))
        if len(destinations) != 1:
            raise Unproven('axebow_reveal_route_not_legal')
        path = next(iter(destinations.values()))
        route.choose_path(state, target, path)
        return target, path
    waypoint = hints[0]['region']
    start = gl.fellowship_region(state)
    allowed = {start, target, waypoint}
    # BFS by region, waypoint-seen and depth. Other paths with the same public
    # stronghold set produce the same observation; no learned choice is guessed.
    queue = deque([(start, (start,), start == waypoint)])
    seen = set()
    while queue:
        current, path, passed = queue.popleft()
        key = (current, passed)
        if key in seen:
            continue
        seen.add(key)
        if current == target and passed:
            route.choose_path(state, target, path)  # Independent engine legality.
            return target, path
        if len(path) - 1 >= state.fellowship.progress:
            continue
        for neighbor in sorted(state.board.adjacent(current)):
            if route._stronghold(state, neighbor) and neighbor not in allowed:
                continue
            queue.append((neighbor, path + (neighbor,), passed or neighbor == waypoint))
    raise Unproven('axebow_reveal_route_not_legal')


@_rng_neutral
def emit_movement_hunt_reduction(batch, stats):
    """Drive observed movement/draws to the real Axe/Horn reaction boundary."""
    from contextlib import ExitStack
    from unittest.mock import patch
    import ai
    import game_logic as gl
    import game_state as gs
    import rendered_census as census
    import rendered_extract as rx
    from rendered_hunt_observation import mark
    try:
        if not rx._is_paid_fellowship_move(batch):
            raise Unproven('axebow_funded_movement_unproven')
        state, side = rx._mk_state_with_die(batch, install_hands=True)
        if state is None or side != gs.Side.FREE_PEOPLES:
            raise Unproven('axebow_initial_state_unproven')
        if rx._RX_HUNT_HISTORY is None:
            raise Unproven('axebow_public_pool_unproven')
        pool, discarded, reason = rx._RX_HUNT_HISTORY.lookup(batch['start_line'])
        if reason:
            raise Unproven('axebow_public_pool:' + reason)
        state.hunt_pool, state.hunt_discard = pool, discarded
        rx._prepare_fellowship_dice(state, batch, pending=False)
        moves = [action for action in ai.get_legal_actions(state, side)
                 if action.action_type == 'fellowship_move' and action.die_idx == 0]
        if len(moves) != 1:
            raise Unproven('axebow_movement_not_legal')
        tiles = rx._hunt_tile_items(batch)
        draws = []
        rolls = [int(value) for record in batch['rng'] if record.get('kind') == 'roll_num'
                 for value in record['ints']]
        roll_index = 0

        def observed_roll(n, *args, **kwargs):
            nonlocal roll_index
            if roll_index + n > len(rolls):
                raise Unproven('axebow_roll_prefix_incomplete')
            result = rolls[roll_index:roll_index + n]
            roll_index += n
            return result

        def observed_tile(current):
            if len(draws) >= len(tiles):
                raise Unproven('axebow_reveal_draw_incomplete')
            item = tiles[len(draws)]
            expected = census._tile_obj(gs, str(item['value']).lower())
            keys = ('damage', 'is_eye', 'reveal', 'is_special', 'special_name',
                    'stops_movement', 'is_dice_roll', 'is_fellowship_special')
            matching = [tile for tile in current.hunt_pool if expected is not None
                        and all(getattr(tile, key) == getattr(expected, key) for key in keys)]
            if not matching:
                raise Unproven('axebow_observed_tile_not_in_pool')
            draws.append(int(item['line']))
            return matching[0]

        def observed_pop(current):
            tile = observed_tile(current)
            current.hunt_pool.remove(tile)
            return tile

        def unexpected(*args, **kwargs):
            raise Unproven('axebow_intervening_random_choice_unproven')

        with ExitStack() as stack:
            for name, function in [('roll_d6_batch', observed_roll), ('pick_hunt_tile', observed_tile),
                                   ('pop_hunt_tile', observed_pop), ('pick_random_companion', unexpected),
                                   ('pick_random_card', unexpected)]:
                stack.enter_context(patch.object(gl, name, function))
            state = ai.apply_action(state, moves[0])
            route_evidence = None
            for _ in range(16):
                if getattr(state, 'pending_card_option', None):
                    menu, index, key, boundary, timing = _shadow_reaction_decision(state, batch, tiles, draws)
                    if key == 'use_hunt_reduction':
                        mark(state, boundary, timing)
                        sample = rx._emit_sample(state, side, menu, index, 'card_option', batch['turn'])
                        choice = menu[index].params['choices']
                        sample['reactive_evidence'] = {
                            'kind': 'actual_movement_hunt_protection_chain',
                            'decision_boundary': boundary, 'timing': timing,
                            'draw_lines': list(draws), 'reveal_route': route_evidence}
                        stats['kinds_emit']['hunt_reduction'] += 1
                        stats['kinds_emit']['hunt_reduction_' + ('use' if choice['use_hunt_reduction'] else 'decline')] += 1
                        return sample
                    state = ai.apply_action(state, menu[index])
                    continue
                if (getattr(state, 'pending_hunt', None)
                        or getattr(state, '_pending_hunt_damage', 0)):
                    raise Unproven('axebow_intervening_hunt_choice_unproven')
                if not getattr(state, '_reveal_place_due', False):
                    raise Unproven('axebow_no_actual_reduction_window')
                if not draws or len(draws) >= len(tiles):
                    raise Unproven('axebow_reveal_draw_incomplete')
                target, path = _observed_reveal_route(state, batch, draws[-1], int(tiles[len(draws)]['line']))
                placements = [a for a in ai.get_legal_actions(state, side)
                              if a.action_type == 'fellowship_reveal_place' and a.params.get('region') == target]
                if len(placements) != 1:
                    raise Unproven('axebow_reveal_placement_not_legal')
                action = copy.deepcopy(placements[0])
                action.params['path'] = list(path)
                before = len(draws)
                state = ai.apply_action(state, action)
                if len(draws) == before and not getattr(state, 'pending_card_option', None):
                    raise Unproven('axebow_extra_tile_route_unproven')
                route_evidence = {'destination': target, 'path': list(path)}
        raise Unproven('axebow_protection_chain_limit')
    except (Unproven, ValueError) as error:
        stats['drop_reasons'][str(error)] += 1
        return None
