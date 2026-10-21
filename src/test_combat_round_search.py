"""Live AI menu, branching and sampling regressions for observed combat choices."""
import copy
import pickle
import unittest
from unittest.mock import patch

from test_combat_continuation_literal import (
    live_battle, companion, card_id, Dice, cb, gs,
)
import ai
from combat_round_search import round_signature, requires_phase_sampling


def mighty_position():
    state, attacker, defender = live_battle(
        fp_regulars=4, fp_leaders=[companion('Aragorn', 3, 2),
                                   companion(), gs.Leader(gs.Nation.ROHAN)])
    state.pending_combat = {
        'phase': 'defender_combat_card',
        'attacker_region': attacker, 'defender_region': defender,
        'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
        'deciding_side': gs.Side.SHADOW, 'is_siege': False,
        'atk_combat_card': card_id('Mighty Attack'), 'def_combat_card': None,
        'rounds_fought': 0, 'shine_asked': True,
    }
    # Mighty is chosen after the attacker primary roll, before its re-roll.
    # The user-required attacker-first sequence exposes no defender dice yet.
    with patch.object(cb, 'roll_d6_batch', Dice([6, 1, 1, 1, 1])):
        ai._run_combat_round(state, state.pending_combat)
    return state


class LiveSearchContinuation(unittest.TestCase):
    def test_precombat_payment_keeps_victim_owner_but_names_attacking_card(self):
        # Durin's Bane hits first. FP pays; preserving its Elite lets Charge
        # hit next, when SH pays. The card context must follow the source of
        # each hit, while decision ownership follows the casualty recipient.
        state, attacker, defender = live_battle(fp_regulars=1, fp_elites=1,
                                               sh_regulars=3)
        state.pending_combat = dict(
            phase='defender_combat_card', attacker_region=attacker,
            defender_region=defender, attacker_side=gs.Side.FREE_PEOPLES,
            defender_side=gs.Side.SHADOW, deciding_side=gs.Side.SHADOW,
            is_siege=False, atk_combat_card=card_id('Charge'),
            def_combat_card=card_id("Durin's Bane"), rounds_fought=0,
            shine_asked=True)
        with patch.object(cb, 'roll_d6_batch', Dice([4, 1, 1])):
            ai._run_combat_round(state, state.pending_combat)
        pc = state.pending_combat
        self.assertEqual(pc['deciding_side'], gs.Side.FREE_PEOPLES)
        self.assertEqual(pc['effect_choice']['effect'], 'pre_combat_casualty')
        self.assertEqual(pc['effect_choice']['card_id'], card_id("Durin's Bane"))
        actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
        self.assertTrue(all(a.params['card_id'] == card_id("Durin's Bane") for a in actions))
        regular = next(a for a in actions if a.params['kind'] == 'regular')
        with patch.object(cb, 'roll_d6_batch', Dice([6])):
            state = ai.apply_action(state, regular)
        pc = state.pending_combat
        self.assertEqual(pc['deciding_side'], gs.Side.SHADOW)
        self.assertEqual(pc['effect_choice']['card_id'], card_id('Charge'))
        self.assertEqual(pc['rounds_fought'], 0)
        action = ai.get_legal_actions(state, gs.Side.SHADOW)[0]
        with patch.object(cb, 'roll_d6_batch', Dice([1], [1, 1])):
            done = ai.apply_action(state, action)
        self.assertEqual(done.pending_combat['rounds_fought'], 1)
        self.assertEqual(len(done.armies[attacker].units), 1)
        self.assertEqual(len(done.armies[defender].units), 2)

    def test_live_menu_then_clone_resume_preserves_prior_rolls(self):
        state = mighty_position()
        pc = state.pending_combat
        self.assertEqual(pc['effect_choice']['effect'], 'mighty_attack')
        self.assertEqual(pc['effect_choice']['stage'], 'round')
        self.assertEqual(pc['rounds_fought'], 0)
        observed = pc['_round_resume']['cursor']
        self.assertEqual(observed.combat.attacker_combat_rolls, [6, 1, 1, 1, 1])
        self.assertEqual(observed.combat.attacker_rerolls, [])
        self.assertEqual(observed.combat.defender_combat_rolls, [])
        self.assertNotIn(('combat', 'defender'), observed.observed_roll_sides)
        actions = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
        self.assertEqual({a.params['target'] for a in actions},
                         {'Aragorn', 'Meriadoc Brandybuck'})
        before = pickle.dumps(state)
        for name, count in [('Aragorn', 2), ('Meriadoc Brandybuck', 3)]:
            with self.subTest(payer=name):
                cloned = ai.clone_state(state)
                payload = cloned.pending_combat['_round_resume']
                self.assertIs(payload['cursor'].combat._live_state, cloned)
                self.assertEqual(round_signature(payload), round_signature(pc['_round_resume']))
                self.assertEqual(ai._state_sig(cloned), ai._state_sig(state))
                action = next(a for a in actions if a.params['target'] == name)
                # Mighty converts one miss. Paying Aragorn's 2 Leadership
                # leaves 2 re-rolls; paying Merry's 1 leaves 3. Only after
                # those dice do the three Shadow units roll their primary.
                dice = Dice([6] * count, [6, 6, 6])
                with patch.object(cb, 'roll_d6_batch', dice):
                    result = ai.apply_action(cloned, action)
                self.assertFalse(dice.batches)
                self.assertEqual(dice.calls, [
                    (count, str(gs.Side.FREE_PEOPLES), (6,) * count),
                    (3, str(gs.Side.SHADOW), (6, 6, 6))])
                self.assertIsNone((result.pending_combat or {}).get('_round_resume'))
                self.assertEqual(pickle.dumps(state), before)

    def test_same_board_different_observed_rolls_must_not_merge(self):
        state = mighty_position()
        cloned = ai.clone_state(state)
        c = cloned.pending_combat['_round_resume']['cursor'].combat
        c.attacker_combat_rolls = [6, 6, 1, 1, 1]
        c.attacker_hits = 2
        self.assertIsNotNone(ai._state_sig(state))
        self.assertNotEqual(ai._state_sig(state), ai._state_sig(cloned))
        self.assertNotEqual(ai._chance_sig(state), ai._chance_sig(cloned))

    def test_old_whole_round_kernel_is_not_used_across_new_choice(self):
        state = mighty_position()
        self.assertTrue(requires_phase_sampling(state, ai.clone_state(state)))
        self.assertTrue(requires_phase_sampling(gs.setup_initial_game(state.board), state))

    def test_sampled_phase_keeps_full_support_after_sixteen_visits(self):
        parent = ai.MCTSNode(None, gs.Side.FREE_PEOPLES)
        chance = ai.MCTSNode(None, gs.Side.FREE_PEOPLES, parent=parent)
        chance.round_sampling = True
        chance.outcomes = []
        # First sixteen outcomes are identical, then a previously unseen
        # result occurs. A sixteen-draw frozen bank can never discover it.
        draws = [0] * 16 + list(range(1, 25))
        with patch.object(ai, '_overlay_prepare_apply'), \
             patch.object(ai, '_apply_action_detect_rng', side_effect=[(x, True) for x in draws]) as run, \
             patch.object(ai, '_chance_sig', side_effect=lambda st: st), \
             patch.object(ai, '_CHANCE_K_MAX', 1):
            observed = [ai._select_outcome(chance).state for _ in draws]
        self.assertEqual(observed, draws)
        self.assertEqual(run.call_count, 40)
        self.assertEqual(len(chance.outcomes), 1)
        self.assertIsNone(chance.kernel_cover)


if __name__ == '__main__':
    unittest.main()
