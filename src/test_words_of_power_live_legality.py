"""Printed Nazgul prerequisite and compulsory available target, via card play.

User clarification: a Companion is not a prerequisite for playing Words of
Power. Its cancellation chooses a participating Companion whenever available.
"""
import os
os.environ['WOTR_ACT_GEN'] = 'v52'
import unittest
from unittest.mock import patch
import ai
import ai_policies
import combat as cb
from test_combat_continuation_literal import live_battle, companion, card_id, gs


def position(names=(), nazgul=True):
    state, attacker, defender = live_battle(
        fp_leaders=[companion(name) for name in names],
        sh_leaders=[gs.Nazgul(gs.Nation.SAURON)] if nazgul else [])
    cid = card_id('Words of Power')
    state.hands[gs.Side.SHADOW].hand = [cid]
    state.pending_combat = {
        'attacker_region': attacker, 'defender_region': defender,
        'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
        'phase': 'defender_combat_card', 'deciding_side': gs.Side.SHADOW,
        'rounds_fought': 0, 'is_siege': False, 'shine_asked': True,
        'atk_combat_card': None}
    return state, cid


class WordsOfPowerLiveLegality(unittest.TestCase):
    def play(self, state, cid):
        legal = ai.get_legal_actions(state, gs.Side.SHADOW)
        action = next(a for a in legal if a.action_type == 'combat_card_play'
                      and a.params.get('card_id') == cid)
        return ai.apply_action(state, action)

    def trace_resolution(self, call):
        cancellations = []
        rolls = []
        original = cb.apply_words_of_power

        def observe(combat, choices=None):
            original(combat, choices)
            cancellations.append(getattr(combat._wop_cancelled, 'name', None))

        def physical_roll(count, ctx='', side=None):
            rolls.append(count)
            return [1] * count

        with patch.object(cb, 'apply_words_of_power', side_effect=observe), \
                patch.object(cb, 'roll_d6_batch', side_effect=physical_roll):
            result = call()
        self.assertTrue(rolls, 'Card resolution must progress to actual combat rolls')
        self.assertEqual(len(cancellations), 1)
        return result, cancellations[0]

    def test_no_companion_is_legal_and_resolves_without_empty_choice(self):
        state, cid = position()
        result, cancelled = self.trace_resolution(lambda: self.play(state, cid))
        self.assertIsNone(cancelled)
        self.assertNotEqual((result.pending_combat or {}).get('phase'), 'combat_effect_choice')
        self.assertNotIn(cid, result.hands[gs.Side.SHADOW].hand)

    def test_one_companion_is_cancelled_without_a_decline_or_target_pause(self):
        state, cid = position(['Meriadoc'])
        result, cancelled = self.trace_resolution(lambda: self.play(state, cid))
        self.assertEqual(cancelled, 'Meriadoc')
        self.assertNotEqual((result.pending_combat or {}).get('phase'), 'combat_effect_choice')

    def test_multiple_companions_offer_only_real_targets_before_any_roll(self):
        state, cid = position(['Boromir', 'Meriadoc'])
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Target needed first')):
            pending = self.play(state, cid)
        self.assertEqual(pending.pending_combat['phase'], 'combat_effect_choice')
        legal = ai.get_legal_actions(pending, gs.Side.SHADOW)
        self.assertEqual({a.params.get('target') for a in legal}, {'Boromir', 'Meriadoc'})
        projected = ai_policies.project_ai_actions(pending, gs.Side.SHADOW, legal).actions
        self.assertTrue(projected)
        self.assertTrue(all(a.params.get('target') in {'Boromir', 'Meriadoc'} for a in projected))
        for action in legal:
            with self.subTest(target=action.params['target']):
                _, cancelled = self.trace_resolution(
                    lambda: ai.apply_action(ai.clone_state(pending), action))
                self.assertEqual(cancelled, action.params['target'])

    def test_companions_do_not_replace_the_printed_nazgul_prerequisite(self):
        state, cid = position(['Boromir'], nazgul=False)
        self.assertNotIn(cid, [a.params.get('card_id')
                              for a in ai.get_legal_actions(state, gs.Side.SHADOW)
                              if a.action_type == 'combat_card_play'])


if __name__ == '__main__':
    unittest.main()
