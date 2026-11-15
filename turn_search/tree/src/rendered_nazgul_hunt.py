"""Recorded Nazgul Event continuations, without synthesizing Hunt decisions."""
from contextlib import ExitStack
import copy
from unittest.mock import patch
import re


class Unproven(ValueError):
    pass


def funding_face(batch, default):
    """A literal owner declaration can resolve a Ring-funded Event's face."""
    import rendered_extract as rx
    if rx._ring_counter_used(batch) != 'sh':
        return default
    staged = next((record for record in batch['records'] if record.get('die_stage')), None)
    if staged is None:
        return default
    declarations = []
    for line, body, actor in rx._COMBAT_CHAT_INDEX:
        if not batch['start_line'] <= line <= batch['end_line'] or actor != staged.get('player'):
            continue
        match = re.fullmatch(r'\s*as\s+(?:a\s+)?([cpmha])\s*[.!]?\s*', body, re.I)
        if match:
            declarations.append(match[1].upper())
    if len(set(declarations)) == 1:
        return declarations[0]
    return default


class RecordedNazgulHunt:
    def __init__(self, batch, card_id, stats):
        import rendered_extract as rx
        self.batch, self.card_id, self.stats = batch, card_id, stats
        self.tiles = rx._hunt_tile_items(batch)
        self.draws = []
        self.used_victims = set()
        self.boundary = int(batch['start_line'])
        self.rolls = [int(value) for record in batch['rng'] if record.get('kind') == 'roll_num'
                      for value in record.get('ints', [])]
        self.roll_index = 0
        self.random_victim = None

    @staticmethod
    def for_card(batch, card_id, stats):
        if card_id not in ('SHADOW_Character_Nazgûl_Search', 'SHADOW_Character_Nazgûl_Strike!'):
            return None
        batch['_rx_nazgul_hunt_owned'] = True
        return RecordedNazgulHunt(batch, card_id, stats)

    def initialize(self, state):
        import rendered_extract as rx
        import game_state as gs
        from game_state import Side
        pools = getattr(state, '_rx_full_pools', None)
        if pools is not None:
            state.dice[Side.FREE_PEOPLES] = rx._install_full_pools(pools)[Side.FREE_PEOPLES]
        state._rx_full_pools = None
        if rx._RX_HUNT_HISTORY is None:
            raise Unproven('nazgul_hunt_public_pool_missing')
        pool, discarded, reason = rx._RX_HUNT_HISTORY.lookup(self.batch['start_line'])
        if reason:
            raise Unproven('nazgul_hunt_public_pool:' + reason)
        state.hunt_pool, state.hunt_discard = pool, discarded
        state.hunt_box = []
        state.eyes_allocated = int((rx._RX_EYEALLOC or {}).get(self.batch['turn'], 0))
        count = state.eyes_allocated + int(self.batch.get('_rx_shadow_rolled_eyes', 0))
        eyes = [d for d in state.dice[Side.SHADOW] if d.current is gs.ActionDieFace.EYE]
        if len(eyes) > count:
            raise Unproven('nazgul_hunt_eye_pool_conflict')
        for _ in range(count - len(eyes)):
            die = gs.ActionDie(gs.SH_FACES, Side.SHADOW)
            die.current = gs.ActionDieFace.EYE
            state.dice[Side.SHADOW].append(die)
            eyes.append(die)
        for die in eyes:
            die.used = True
        state.hunt_box.extend(eyes)
        for line, turn, face in rx._FP_HUNTBOX_FACES:
            if turn != self.batch['turn'] or line >= self.batch['start_line']:
                continue
            die = next((d for d in state.dice[Side.FREE_PEOPLES] if d.used and
                        d.current is rx._letter2face().get(face) and all(d is not b for b in state.hunt_box)), None)
            if die is None:
                raise Unproven('nazgul_hunt_prior_fp_die_missing')
            state.hunt_box.append(die)
        for conversion in self.batch.get('_rx_shadow_conversions', []):
            if conversion.get('already_refaced'):
                die = gs.ActionDie(gs.SH_FACES, Side.SHADOW)
                die.current, die.used = gs.ActionDieFace.EYE, True
                state.dice[Side.SHADOW].append(die)
                state.hunt_box.append(die)
                continue
            die = next((d for d in state.dice[Side.SHADOW] if d.used and
                        d.current is rx._letter2face().get(conversion['face']) and all(d is not b for b in state.hunt_box)), None)
            if die is None:
                raise Unproven('nazgul_hunt_converted_die_missing')
            die.current = gs.ActionDieFace.EYE
            state.hunt_box.append(die)
        state._rx_eyes_stamped = True

    def _roll(self, count, *args, **kwargs):
        if self.roll_index + count > len(self.rolls):
            raise Unproven('nazgul_hunt_recorded_rolls_incomplete')
        values = self.rolls[self.roll_index:self.roll_index + count]
        self.roll_index += count
        return values

    def _tile(self, state):
        import rendered_census as census
        import game_state as gs
        if len(self.draws) >= len(self.tiles):
            raise Unproven('nazgul_hunt_missing_recorded_tile')
        record = self.tiles[len(self.draws)]
        expected = census._tile_obj(gs, str(record['value']).lower())
        fields = ('damage', 'is_eye', 'reveal', 'is_special', 'special_name',
                  'stops_movement', 'is_dice_roll', 'is_fellowship_special')
        matching = [tile for tile in state.hunt_pool if expected is not None and
                    all(getattr(tile, name) == getattr(expected, name) for name in fields)]
        if not matching:
            raise Unproven('nazgul_hunt_recorded_tile_not_in_pool')
        self.draws.append(int(record['line']))
        self.boundary = int(record['line']) + 1
        return matching[0]

    def _pop(self, state):
        tile = self._tile(state)
        state.hunt_pool.remove(tile)
        return tile

    def apply(self, state, action):
        import ai
        import game_logic as gl
        def unexpected(*args, **kwargs):
            raise Unproven('nazgul_hunt_unrecorded_random_choice')
        def companion(candidates, *args, **kwargs):
            import rendered_replay as rr
            matches = [value for value in candidates if self.random_victim is not None
                       and rr._named_eq(value.name, self.random_victim)]
            if len(matches) != 1:
                raise Unproven('nazgul_hunt_random_victim_not_eligible')
            self.random_victim = None
            return matches[0]
        with ExitStack() as stack:
            for name, function in (('pick_hunt_tile', self._tile), ('pop_hunt_tile', self._pop),
                                   ('roll_d6_batch', self._roll),
                                   ('pick_random_companion', companion), ('pick_random_card', unexpected)):
                stack.enter_context(patch.object(gl, name, function))
            return ai.apply_action(state, action)

    def has_continuation(self, state):
        pending = getattr(state, 'pending_card_option', None) or {}
        return bool(getattr(state, 'pending_hunt', None) or getattr(state, '_reveal_place_due', False)
                    or any(any(key in option for key in ('wizardstaff_prevent', 'mithril_redraw', 'use_hunt_reduction', 'balrog_declare_draw'))
                           for option in pending.get('options', [])))

    def resume(self, state):
        import ai
        import rendered_extract as rx
        import rendered_replay as rr
        from game_state import Side
        from rendered_hunt_observation import mark
        from rendered_reactive_cards import _shadow_reaction_decision
        from rendered_balrog import observed_path as _observed_reveal_route
        rows = []
        try:
            for _ in range(20):
                if not self.has_continuation(state):
                    break
                menu = [] if getattr(state, 'pending_card_option', None) else ai.get_legal_actions(state, Side.FREE_PEOPLES)
                pending = getattr(state, 'pending_hunt', None) or {}
                route = None
                side = Side.FREE_PEOPLES
                if getattr(state, 'pending_card_option', None):
                    if any('balrog_declare_draw' in choice for choice in state.pending_card_option.get('options', ())):
                        from rendered_balrog import choose_pending
                        menu, index, boundary, evidence = choose_pending(state, self.batch, self.boundary)
                        key, timing, side = 'balrog_declare_draw', 'before_draw', Side.SHADOW
                    else:
                        menu, index, key, boundary, timing = _shadow_reaction_decision(
                            state, self.batch, self.tiles, self.draws)
                    phase = 'card_option'
                elif pending:
                    lo = self.draws[-1] if self.draws else self.boundary
                    hi = int(self.tiles[len(self.draws)]['line']) if len(self.draws) < len(self.tiles) else int(self.batch['end_line']) + 1
                    before = self.batch['truth_before']['fel']
                    after = self.batch['truth']['fel']
                    exited = [name for name in before.get('comps', []) if name not in after.get('comps', [])]
                    exits = rx._comp_exit_lines(self.batch)
                    seps = dict(rx._sep_moves(self.batch))
                    evidence = rx._hunt_window_ev(self.batch, lo, hi, exited, seps, exits)
                    victims = [name for name in evidence['left'] if name not in self.used_victims]
                    victims.sort(key=lambda name: exits.get(name, hi))
                    guide = state.fellowship.guide
                    placement = None
                    if pending.get('phase') == 'hunt_gollum':
                        if not evidence['flips'] and not evidence['corr'] and not victims:
                            raise Unproven('nazgul_hunt_gollum_progression_unproven')
                        key = 'hunt_gollum_reveal' if evidence['flips'] % 2 else 'hunt_gollum_no_reveal'
                    elif pending.get('phase') == 'hobbit_place':
                        name = pending.get('companion', '')
                        destinations = [destination for victim, destination in seps.items() if rr._named_eq(name, victim)]
                        if len(set(destinations)) != 1:
                            raise Unproven('nazgul_hunt_hobbit_placement_unproven')
                        key, placement = 'companion_separate', destinations[0]
                    elif pending.get('phase') != 'hunt_sacrifice':
                        raise Unproven('nazgul_hunt_pending:' + str(pending.get('phase')))
                    elif victims and (guide is None or not rr._named_eq(guide.name, victims[0])
                            or any(rr._named_eq(victims[0], name) for name in evidence['random_ev'])):
                        key = 'hunt_sacrifice_random'
                        self.random_victim = victims[0]
                        self.used_victims.add(victims[0])
                    elif victims and guide is not None and rr._named_eq(guide.name, victims[0]):
                        key = ('hunt_guide_separate' if getattr(guide, 'guide_ability', None) == 'sacrifice_reduce_hunt'
                               else 'hunt_sacrifice_guide')
                        self.used_victims.add(victims[0])
                    elif not victims and evidence['corr'] > 0:
                        key = 'hunt_take_corruption'
                    else:
                        raise Unproven('nazgul_hunt_casualty_evidence_unproven')
                    choices = [i for i, action in enumerate(menu) if action.action_type == key
                               and (placement is None or action.params.get('to_region') == placement)]
                    if len(choices) != 1:
                        raise Unproven('nazgul_hunt_casualty_not_legal')
                    phase = pending.get('phase')
                    index, boundary, timing = choices[0], self.boundary, 'after_draw'
                else:
                    lo = self.draws[-1] if self.draws else int(self.batch['start_line'])
                    hi = int(self.tiles[len(self.draws)]['line']) if len(self.draws) < len(self.tiles) else int(self.batch['end_line']) + 1
                    destination, path = _observed_reveal_route(state, self.batch, lo, hi)
                    choices = [i for i, action in enumerate(menu) if action.action_type == 'fellowship_reveal_place'
                               and action.params.get('region') == destination]
                    if len(choices) != 1:
                        raise Unproven('nazgul_hunt_reveal_not_legal')
                    index, boundary, timing, phase = choices[0], self.boundary, ('after_draw' if self.draws else 'before_draw'), 'fellowship_reveal_place'
                    key = phase
                    route = list(path)
                mark(state, boundary, timing)
                if len(menu) >= 2:
                    sample = rx._emit_sample(state, side, menu, index, phase, self.batch['turn'])
                    sample['card_hunt_evidence'] = dict(source_card_id=self.card_id, decision_boundary=boundary,
                        draw_lines=list(self.draws), choice=key, route=route)
                    rows.append(sample)
                    self.stats['kinds_emit']['nazgul_followon_' + key] += 1
                action = copy.deepcopy(menu[index])
                if route is not None:
                    action.params['path'] = route
                    self.boundary = max(int(record['line']) for record in self.batch['records']
                        if record.get('op') == 'move' and str(record.get('obj', '')).lower() == 'the fellowship'
                        and int(record['line']) < hi) + 1
                state = self.apply(state, action)
            else:
                raise Unproven('nazgul_hunt_continuation_limit')
        except (ValueError, RuntimeError) as error:
            self.stats['drop_reasons'][str(error)] += 1
        return state, rows
