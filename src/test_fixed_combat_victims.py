"""Fixed AI victim priorities applied to real, observed combat choices."""
import pickle
import unittest
from collections import Counter, defaultdict
from unittest.mock import patch

from test_combat_continuation_literal import (
    live_battle, companion, minion, card_id, Dice, begin, cb, gs,
)
import ai
import ai_policies as policies
from trace_decisions import is_learning_decision


def pending_victim(card, hits=2):
    state, atk, dfn = live_battle(
        fp_regulars=2, sh_regulars=3,
        fp_leaders=[companion(), gs.Leader(gs.Nation.ROHAN)],
        sh_leaders=[minion('Saruman'), minion('Mouth of Sauron'),
                    gs.Nazgul(gs.Nation.SAURON), minion('Witch King')])
    state.pending_combat = {
        'phase': 'defender_combat_card', 'attacker_region': atk,
        'defender_region': dfn, 'attacker_side': gs.Side.FREE_PEOPLES,
        'defender_side': gs.Side.SHADOW, 'is_siege': False,
        'rounds_fought': 0,
    }
    with patch.object(cb, 'roll_d6_batch', Dice([1, 1], [6, 6, 6],
                                             [6, 6] if hits == 2 else [6, 1])):
        result = begin(state, atk, dfn, card)
    ai._accept_combat_round_result(state, state.pending_combat, result,
                                   card_id(card), None)
    return state


class FixedVictims(unittest.TestCase):
    def test_legal_priority_and_no_state_mutation(self):
        for card in ('Blade of Westernesse', 'Fateful Strike'):
            state = pending_victim(card)
            menu = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
            before = pickle.dumps(state)
            for expected in ('Witch King', 'Mouth of Sauron', 'Saruman', 'nazgul'):
                candidates = [a for a in menu if a.params['target'] == expected]
                if not candidates:
                    continue
                result = policies.project_ai_actions(state, gs.Side.FREE_PEOPLES, menu)
                self.assertEqual(result.actions, (candidates[0],))
                self.assertEqual(result.forced_reason, policies.COMBAT_VICTIM)
                menu = [a for a in menu if a.params['target'] != expected]
            self.assertEqual(pickle.dumps(state), before)

    def test_fateful_one_hit_cannot_select_a_minion(self):
        state = pending_victim('Fateful Strike', hits=1)
        policy = ai.get_ai_policy(state, gs.Side.FREE_PEOPLES)
        self.assertEqual([a.params['target'] for a in policy.actions], ['nazgul'])
        self.assertEqual(policy.forced_reason, policies.COMBAT_VICTIM)

    def test_human_can_still_choose_mouth_and_hit_cost_stays_correct(self):
        for card, army_hits in [('Blade of Westernesse', 1), ('Fateful Strike', 2)]:
            state = pending_victim(card)
            menu = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
            human = next(a for a in menu if a.params['target'] == 'Mouth of Sauron')
            with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Repeated dice')):
                result = ai.apply_action(state, human)
            leaders = result.armies['orthanc'].leaders
            self.assertIn('Witch King', [getattr(l, 'name', '') for l in leaders])
            self.assertNotIn('Mouth of Sauron', [getattr(l, 'name', '') for l in leaders])
            self.assertEqual(len(result.armies['orthanc'].units), 3 - army_hits)

    def test_other_combat_effects_remain_learned(self):
        from test_combat_round_search import mighty_position
        state = mighty_position()
        menu = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
        for effect in ('mighty_attack', 'heroic_death', 'black_breath',
                       'relentless_assault', 'dread_and_despair'):
            state.pending_combat['effect_choice']['effect'] = effect
            policy = policies.project_ai_actions(state, gs.Side.FREE_PEOPLES, menu)
            self.assertEqual(policy.actions, tuple(menu))
            self.assertIsNone(policy.forced_reason)

    def test_forced_trace_kept_but_not_a_learning_decision(self):
        row = {'kind': 'action', 'side': 'FP', 'policy_forced': True,
               'forced_policy_reason': policies.COMBAT_VICTIM}
        self.assertFalse(is_learning_decision(row))
        self.assertTrue(is_learning_decision({'kind': 'action', 'side': 'FP'}))

    def test_corpus_omits_both_priority_and_alternate_human_target_labels(self):
        import rendered_extract as rx
        state = pending_victim('Blade of Westernesse')
        menu = ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
        before = pickle.dumps(state)
        for target in ('Witch King', 'Mouth of Sauron'):
            index = next(i for i, a in enumerate(menu) if a.params['target'] == target)
            with patch.object(rx, '_RX_RINGS_CUR', None), patch.object(rx, '_RX_LINE_CUR', None), \
                 patch.object(rx, '_RX_EYEALLOC', None), patch.object(rx, '_RX_EYEROLL', None):
                sample = rx._emit_sample(state, gs.Side.FREE_PEOPLES, menu,
                                         index, 'combat_effect_choice', state.turn)
            self.assertEqual(sample['ai_policy_exclusion'], policies.COMBAT_VICTIM)
            self.assertEqual(rx._filter_ai_policy_samples([sample], defaultdict(Counter)), [])
        self.assertEqual(pickle.dumps(state), before)


if __name__ == '__main__':
    unittest.main()
