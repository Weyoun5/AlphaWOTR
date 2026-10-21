"""House of the Stewards: independently authored acceptance specification.

Authored before implementation; the saved baseline fails this specification.
Oracle: FP Character23 / supplied Almanac v98 p4/16/46, RB7/16/22/27.
Boromir must be a board Companion in Gondor (Osgiliath is not Gondor).
Recruit one Gondor Regular OR Elite with Boromir, then draw two Strategy
cards. No Leader is recruited. A blocked/empty unit supply does not cancel
the draws. AL16 explicitly orders both Strategy draws, required hand-limit
discards, an OPTIONAL Gandalf Event-die Character draw, then another discard
if needed. AL46 expressly allows declining that bonus draw.

Expected alternatives/counts below are literal, not the legal generator's
own output. Fixtures use the real Boromir object removed from the Fellowship.
No other test modules/classes, model weights, or training corpora are used.
"""
import copy
import unittest

import ai
import board
import game_logic as gl
import game_state as gs

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
GONDOR, SAURON = gs.Nation.GONDOR, gs.Nation.SAURON
REGULAR, ELITE = gs.UnitKind.REGULAR, gs.UnitKind.ELITE
CID = 'FREE_PEOPLES_Character_House_of_the_Stewards'
TITLE = 'House of the Stewards'
S1 = 'FREE_PEOPLES_Strategy_Guards_of_the_Citadel'
S2 = 'FREE_PEOPLES_Strategy_The_Red_Arrow'
S3 = 'FREE_PEOPLES_Strategy_The_Last_Battle'
BONUS = 'FREE_PEOPLES_Character_Gwaihir_the_Windlord'
FILLERS = ['FREE_PEOPLES_Character_Athelas',
           "FREE_PEOPLES_Character_Bilbo's_Song",
           'FREE_PEOPLES_Character_Mithril_Coat_and_Sting',
           'FREE_PEOPLES_Character_Axe_and_Bow',
           'FREE_PEOPLES_Character_Horn_of_Gondor']


def army(state, region):
    return next((a for k, a in state.armies.items()
                 if board.sanitize(k) == board.sanitize(region)), None)


def figures(state, region='minas tirith'):
    a = army(state, region)
    if a is None:
        return (0, 0, 0)
    return (sum(u.nation == GONDOR and u.kind == REGULAR for u in a.units),
            sum(u.nation == GONDOR and u.kind == ELITE for u in a.units),
            sum(isinstance(l, gs.Leader) and l.nation == GONDOR for l in a.leaders))


def before_play(region='minas tirith', face=gs.ActionDieFace.CHARACTER,
                fillers=()):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    if region is not None:
        boromir = next(c for c in state.fellowship.companions if c.name == 'Boromir')
        state.fellowship.companions.remove(boromir)
        gl.get_or_create_army(state, region).leaders.append(boromir)
    state.hands[FP].hand = [CID] + list(fillers)
    for name in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck', 'sh_strat_deck'):
        deck = getattr(state, name)
        for card in state.hands[FP].hand:
            if card in deck:
                deck.remove(card)
    # A literal possible shuffled order, with known cards as chance outcomes.
    # The policy's unit choice must precede seeing either of these draws.
    state.fp_strat_deck = [c for c in state.fp_strat_deck if c not in (S1, S2, S3)]
    state.fp_strat_deck.extend([S3, S2, S1])
    state.fp_char_deck.remove(BONUS)
    state.fp_char_deck.append(BONUS)
    first = gs.ActionDie([face], FP)
    first.current = face
    unused = gs.ActionDie([gs.ActionDieFace.MUSTER], FP)
    unused.current = gs.ActionDieFace.MUSTER
    shadow = gs.ActionDie([gs.ActionDieFace.MUSTER], SH)
    shadow.current = gs.ActionDieFace.MUSTER
    state.dice = {FP: [first, unused], SH: [shadow]}
    state.enemy_controlled = {FP: set(), SH: set()}
    return state


def exhaust_unit_pool(state, *kinds):
    # RB7 physical Gondor component counts, read independently of engine caps.
    caps = {'regular': 15, 'elite': 5}
    for kind in kinds:
        wanted = REGULAR if kind == 'regular' else ELITE
        deployed = sum(u.nation == GONDOR and u.kind == wanted
                       for a in state.armies.values() for u in a.units)
        state.fp_casualties.setdefault(GONDOR, {})[kind] = caps[kind] - deployed


def limit_strategy_deck(state, remaining):
    kept = [S1] if remaining == 1 else []
    state.discard_face_up['fp_strategy'].extend(
        c for c in state.fp_strat_deck if c not in kept)
    state.fp_strat_deck = kept


class HouseOfTheStewardsChoices(unittest.TestCase):
    def play(self, state):
        self.assertTrue(any(a.action_type == 'play_event' and a.die_idx == 0
                            and a.params.get('card_id') == CID
                            for a in ai.get_legal_actions(state, FP)),
                        'Board Boromir in Gondor satisfies the printed condition')
        return ai.apply_action(state, ai.Action(0, 'play_event', {
            'card_id': CID, 'side': 'fp'}))

    def menu(self, state, region='minas tirith'):
        pending = getattr(state, 'pending_card_option', None)
        self.assertIsNotNone(pending, 'Both supplied kinds require a real player decision')
        self.assertEqual(pending['card_id'], CID)
        self.assertEqual(ai._pending_decision_side(state), FP)
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual(len(menu), 2)
        self.assertEqual({a.action_type for a in menu}, {'card_option'})
        self.assertEqual({a.params['choices'].get('unit_type') for a in menu},
                         {'regular', 'elite'})
        self.assertEqual({a.params['choices'].get('boromir_region') for a in menu},
                         {board.sanitize(region)})
        return menu

    def choose(self, state, kind, region='minas tirith'):
        action, = [a for a in self.menu(state, region)
                   if a.params['choices']['unit_type'] == kind]
        return ai.apply_action(state, action)

    def forced(self, state, kind=None):
        pending = getattr(state, 'pending_card_option', None)
        if pending is not None and pending.get('card_id') == CID:
            menu = ai.get_legal_actions(state, FP)
            self.assertEqual(len(menu), 1, 'Forced supply is not an optional decline')
            if kind is not None:
                self.assertEqual(menu[0].params['choices'].get('unit_type'), kind)
            state = ai.apply_action(state, menu[0])
        return state

    def paid(self, state):
        self.assertNotIn(CID, state.hands[FP].hand)
        self.assertEqual(state.discard_face_up['fp_character'].count(CID), 1)
        self.assertEqual([d.used for d in state.dice[FP]], [True, False])
        self.assertEqual([d.used for d in state.dice[SH]], [False])
        self.assertEqual(state.hunt_box, [])
        self.assertIsNone(getattr(state, 'pending_muster_die', None))

    def discard(self, state, card):
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual({a.action_type for a in menu}, {'discard_card'})
        action, = [a for a in menu if a.params['card_id'] == card]
        return ai.apply_action(state, action)

    def gandalf_choice(self, state, take):
        pending = getattr(state, 'pending_card_option', None)
        self.assertIsNotNone(pending, 'AL46: Gandalf bonus is an optional decision')
        self.assertEqual(pending['card_id'], '__gandalf_draw__')
        self.assertEqual(ai._pending_decision_side(state), FP)
        menu = ai.get_legal_actions(state, FP)
        self.assertEqual(len(menu), 2)
        self.assertEqual({a.action_type for a in menu}, {'card_option'})
        self.assertCountEqual([a.params['choices'] for a in menu],
                              [{'gandalf_draw': True}, {'gandalf_draw': False}])
        action, = [a for a in menu if a.params['choices']['gandalf_draw'] is take]
        return ai.apply_action(state, action)

    def test_public_choice_precedes_placement_and_both_unknown_strategy_draws(self):
        initial = before_play()
        self.assertEqual(figures(initial), (3, 1, 1))
        state = self.play(initial)
        self.menu(state)
        self.assertEqual(figures(state), (3, 1, 1))
        self.assertEqual(state.fp_strat_deck, initial.fp_strat_deck)
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        self.assertEqual(state.hands[FP].hand, [CID])

    def test_each_kind_recruits_one_unit_no_leader_then_draws_two_strategy(self):
        for kind, expected in [('regular', (4, 1, 1)), ('elite', (3, 2, 1))]:
            with self.subTest(unit_type=kind):
                initial = before_play()
                state = self.choose(self.play(initial), kind)
                self.assertEqual(figures(state), expected)
                self.assertEqual(state.hands[FP].hand, [S1, S2])
                self.assertEqual(state.fp_strat_deck, initial.fp_strat_deck[:-2])
                self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
                self.assertEqual(state.political.boxes, initial.political.boxes)
                self.assertEqual(state.political.active, initial.political.active)
                self.assertEqual([l.name for l in army(state, 'minas tirith').leaders
                                  if isinstance(l, gs.Companion)], ['Boromir'])
                self.assertIsNone(getattr(state, 'pending_card_option', None))
                self.paid(state)

    def test_existing_unit_kind_encoding_can_distinguish_the_new_house_choices(self):
        # Positive control: Guards already restored the reusable unit_type
        # scalar. House needs menu/extraction work, not a new feature dimension.
        state = before_play()
        encs = [ai.encode_action(ai.Action(-1, 'card_option', {
            'card_id': CID, 'side': 'fp', 'choices': {
                'boromir_region': 'minas tirith', 'unit_type': kind}}), state, FP)
                for kind in ('regular', 'elite')]
        self.assertNotEqual(encs[0].tobytes(), encs[1].tobytes())
        self.assertAlmostEqual(float(encs[0][ai._ACT_OPTKIND_OFF]), 0.34)
        self.assertAlmostEqual(float(encs[1][ai._ACT_OPTKIND_OFF]), 0.67)

    def test_one_unit_pool_exhausted_forces_available_kind_and_keeps_both_draws(self):
        for exhausted, chosen, expected in [
                ('elite', 'regular', (4, 1, 1)),
                ('regular', 'elite', (3, 2, 1))]:
            with self.subTest(exhausted=exhausted):
                initial = before_play()
                exhaust_unit_pool(initial, exhausted)
                state = self.forced(self.play(initial), chosen)
                self.assertEqual(figures(state), expected)
                self.assertEqual(state.hands[FP].hand, [S1, S2])
                self.paid(state)

    def test_no_unit_supply_still_draws_two_strategy_cards(self):
        initial = before_play()
        exhaust_unit_pool(initial, 'regular', 'elite')
        state = self.forced(self.play(initial))
        self.assertEqual(figures(state), (3, 1, 1))
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.paid(state)

    def test_short_strategy_deck_draws_only_available_cards_and_keeps_recruit(self):
        for available, drawn in [(0, []), (1, [S1])]:
            with self.subTest(available=available):
                initial = before_play()
                exhaust_unit_pool(initial, 'elite')
                limit_strategy_deck(initial, available)
                state = self.forced(self.play(initial), 'regular')
                self.assertEqual(figures(state), (4, 1, 1))
                self.assertEqual(state.hands[FP].hand, drawn)
                self.assertEqual(state.fp_strat_deck, [])
                self.paid(state)

    def test_boromir_alone_in_nonsettlement_erech_can_choose_either_kind(self):
        for kind, expected in [('regular', (1, 0, 0)), ('elite', (0, 1, 0))]:
            with self.subTest(unit_type=kind):
                initial = before_play('erech')
                self.assertEqual(figures(initial, 'erech'), (0, 0, 0))
                state = self.choose(self.play(initial), kind, 'erech')
                self.assertEqual(figures(state, 'erech'), expected)
                self.assertEqual(state.hands[FP].hand, [S1, S2])
                self.paid(state)

    def test_captured_empty_gondor_settlement_blocks_recruit_but_not_draws(self):
        initial = before_play('pelargir')
        # The original one-Regular garrison was eliminated before capture;
        # Boromir subsequently moved into the empty enemy-controlled city.
        army(initial, 'pelargir').units = []
        initial.fp_casualties[GONDOR] = {'regular': 1}
        initial.enemy_controlled[SH].add('pelargir')
        state = self.forced(self.play(initial))
        self.assertEqual(figures(state, 'pelargir'), (0, 0, 0))
        self.assertIn('pelargir', state.enemy_controlled[SH])
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.paid(state)

    def test_boromir_inside_garrison_or_with_besieger_has_both_unit_choices(self):
        for shadow_controls in (False, True):
            for kind, expected in [('regular', (4, 1, 1)), ('elite', (3, 2, 1))]:
                with self.subTest(shadow_controls=shadow_controls, unit_type=kind):
                    initial = before_play()
                    for nation in (GONDOR, SAURON):
                        initial.political.active[nation] = True
                        initial.political.boxes[nation] = 3
                    army(initial, 'minas tirith').units.extend(
                        [gs.Unit(SAURON, REGULAR) for _ in range(2)])
                    initial.besieged_strongholds.add('minas tirith')
                    if shadow_controls:
                        initial.enemy_controlled[SH].add('minas tirith')
                    state = self.choose(self.play(initial), kind)
                    self.assertEqual(figures(state), expected)
                    self.assertEqual('minas tirith' in state.enemy_controlled[SH],
                                     shadow_controls)
                    self.assertEqual(sum(u.nation == SAURON for u in
                                         army(state, 'minas tirith').units), 2)
                    self.assertEqual(state.hands[FP].hand, [S1, S2])
                    self.paid(state)

    def test_osgiliath_or_boromir_inside_fellowship_does_not_satisfy_condition(self):
        for region in ('osgiliath', None):
            with self.subTest(boromir_region=region):
                state = before_play(region)
                if region is None:
                    state.fellowship.region = state.board.region('Minas Tirith')
                self.assertFalse(any(a.action_type == 'play_event'
                                     and a.params.get('card_id') == CID
                                     for a in ai.get_legal_actions(state, FP)))

    def test_character_die_full_hand_discards_after_both_strategy_draws(self):
        initial = before_play(fillers=FILLERS)
        exhaust_unit_pool(initial, 'elite')
        state = self.forced(self.play(initial), 'regular')
        self.assertEqual(state.hands[FP].hand, FILLERS + [S1, S2])
        self.assertEqual(len(state.hands[FP].hand), 7)
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        self.assertEqual(ai._pending_decision_side(state), FP)
        state = self.discard(state, S1)
        self.assertEqual(state.hands[FP].hand, FILLERS + [S2])
        self.assertIsNone(getattr(state, 'pending_discard', None))
        self.paid(state)

    def test_gandalf_event_bonus_waits_until_first_hand_limit_discard(self):
        initial = before_play(face=gs.ActionDieFace.EVENT, fillers=FILLERS)
        exhaust_unit_pool(initial, 'elite')  # isolate timing from missing type menu
        self.assertEqual(initial.fellowship.guide.name, 'Gandalf the Grey')
        state = self.forced(self.play(initial), 'regular')
        self.assertEqual(state.hands[FP].hand, FILLERS + [S1, S2],
                         'AL16: the first discard must not see Gandalf\'s later Character draw')
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        state = self.discard(state, S1)
        self.assertEqual(state.hands[FP].hand, FILLERS + [S2])
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        state = self.gandalf_choice(state, True)
        self.assertEqual(state.hands[FP].hand, FILLERS + [S2, BONUS])
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck[:-1])
        self.assertEqual(len(state.hands[FP].hand), 7)
        state = self.discard(state, BONUS)
        self.assertEqual(state.hands[FP].hand, FILLERS + [S2])
        self.assertIsNone(getattr(state, 'pending_discard', None))
        self.paid(state)

    def test_gandalf_event_bonus_is_one_character_card_when_no_first_discard_needed(self):
        initial = before_play(face=gs.ActionDieFace.EVENT)
        exhaust_unit_pool(initial, 'elite')
        state = self.forced(self.play(initial), 'regular')
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        state = self.gandalf_choice(state, True)
        self.assertEqual(state.hands[FP].hand, [S1, S2, BONUS])
        self.assertEqual(state.fp_strat_deck, initial.fp_strat_deck[:-2])
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck[:-1])
        self.assertEqual(figures(state), (4, 1, 1))
        self.paid(state)

    def test_gandalf_bonus_can_be_declined_without_drawing_or_another_payment(self):
        initial = before_play(face=gs.ActionDieFace.EVENT)
        exhaust_unit_pool(initial, 'elite')
        state = self.forced(self.play(initial), 'regular')
        state = self.gandalf_choice(state, False)
        self.assertEqual(state.hands[FP].hand, [S1, S2])
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        self.assertEqual(figures(state), (4, 1, 1))
        self.assertIsNone(getattr(state, 'pending_card_option', None))
        self.assertIsNone(getattr(state, 'pending_discard', None))
        self.paid(state)

    def test_gandalf_decline_after_first_discard_does_not_open_second_discard(self):
        initial = before_play(face=gs.ActionDieFace.EVENT, fillers=FILLERS)
        exhaust_unit_pool(initial, 'elite')
        state = self.forced(self.play(initial), 'regular')
        self.assertEqual(state.hands[FP].hand, FILLERS + [S1, S2])
        state = self.discard(state, S1)
        state = self.gandalf_choice(state, False)
        self.assertEqual(state.hands[FP].hand, FILLERS + [S2])
        self.assertEqual(state.fp_char_deck, initial.fp_char_deck)
        self.assertIsNone(getattr(state, 'pending_card_option', None))
        self.assertIsNone(getattr(state, 'pending_discard', None))
        self.paid(state)

    def test_gandalf_draw_and_decline_need_distinct_action_encodings(self):
        state = before_play(face=gs.ActionDieFace.EVENT)
        encs = [ai.encode_action(ai.Action(-1, 'card_option', {
            'card_id': '__gandalf_draw__', 'side': 'fp',
            'source_card_id': CID, 'deck': 'character',
            'choices': {'gandalf_draw': take}}), state, FP)
                for take in (True, False)]
        self.assertNotEqual(encs[0].tobytes(), encs[1].tobytes())

    def test_invalid_public_choices_reject_before_card_die_or_draw_payment(self):
        for choice in ({}, {'boromir_region': 'minas tirith'},
                       {'boromir_region': 'pelargir', 'unit_type': 'regular'},
                       {'boromir_region': 'minas tirith', 'unit_type': 'leader'}):
            with self.subTest(choices=choice):
                state = before_play()
                decks = (list(state.fp_char_deck), list(state.fp_strat_deck))
                with self.assertRaises((ValueError, RuntimeError)):
                    ai._execute_action(state, ai.Action(0, 'play_event', {
                        'card_id': CID, 'side': 'fp', 'choices': choice}))
                self.assertEqual(figures(state), (3, 1, 1))
                self.assertEqual(state.hands[FP].hand, [CID])
                self.assertEqual([d.used for d in state.dice[FP]], [False, False])
                self.assertEqual((state.fp_char_deck, state.fp_strat_deck), decks)
                self.assertNotIn(CID, state.discard_face_up['fp_character'])

    def test_invalid_pending_choice_retains_paid_decision_and_no_draws(self):
        for choice in ({}, {'boromir_region': 'minas tirith'},
                       {'boromir_region': 'minas tirith', 'unit_type': 'leader'}):
            with self.subTest(choices=choice):
                state = self.play(before_play())
                pending = state.pending_card_option
                before = copy.deepcopy(state.discard_face_up)
                with self.assertRaises((ValueError, RuntimeError)):
                    ai._execute_action(state, ai.Action(-1, 'card_option', {
                        'card_id': CID, 'side': 'fp', 'choices': choice}))
                self.assertIs(state.pending_card_option, pending)
                self.assertEqual(figures(state), (3, 1, 1))
                self.assertEqual(state.hands[FP].hand, [CID])
                self.assertEqual(state.discard_face_up, before)
                self.assertEqual([d.used for d in state.dice[FP]], [True, False])

    def test_stale_supply_or_boromir_region_rejects_without_losing_pending(self):
        for change in ('supply', 'boromir_region'):
            with self.subTest(change=change):
                state = self.play(before_play())
                action, = [a for a in self.menu(state)
                           if a.params['choices']['unit_type'] == 'regular']
                pending = state.pending_card_option
                if change == 'supply':
                    exhaust_unit_pool(state, 'regular')
                else:
                    comp = next(l for l in army(state, 'minas tirith').leaders
                                if isinstance(l, gs.Companion) and l.name == 'Boromir')
                    army(state, 'minas tirith').leaders.remove(comp)
                    gl.get_or_create_army(state, 'erech').leaders.append(comp)
                with self.assertRaises((ValueError, RuntimeError)):
                    ai._execute_action(state, action)
                self.assertIs(state.pending_card_option, pending)
                self.assertEqual(figures(state), (3, 1, 1))
                self.assertEqual(state.hands[FP].hand, [CID])
                self.assertEqual([d.used for d in state.dice[FP]], [True, False])


if __name__ == '__main__':
    unittest.main(verbosity=2)
