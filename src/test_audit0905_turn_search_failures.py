"""Model failures must stop corrected search, including turn-start phases."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import torch
import ai
import board
import game_logic
import game_state as gs
from reward_contract import PROFILE, WINLOSS, LEGACY

if os.environ.get('WOTR_AUDIT_PHASE_BASELINE'):
    baseline = Path(os.environ['WOTR_AUDIT_PHASE_BASELINE'])
    exec(compile(baseline.read_text(encoding='utf-8'), str(baseline), 'exec'), vars(ai))


class FailingNet:
    def eval(self):
        return self

    def forward_backbone(self, _):
        raise RuntimeError('literal phase model failure')


class NonfiniteNet(FailingNet):
    def forward_backbone(self, _):
        return torch.zeros((1, 1)), None

    def value_head(self, _):
        return torch.tensor([[float('nan')]])


def initial():
    state = gs.setup_initial_game(board.load_board())
    game_logic.initialize_decks(state)
    return state


class TurnSearchFailureTests(unittest.TestCase):
    def test_phase_prior_model_exception_propagates_in_both_corrected_contracts(self):
        for contract in (PROFILE, WINLOSS):
            with self.subTest(contract=contract), patch.dict(
                    os.environ, {'WOTR_REWARD_CONTRACT': contract}):
                with self.assertRaisesRegex(RuntimeError, 'literal phase model failure'):
                    ai._greedy_prior_action(initial(), gs.Side.FREE_PEOPLES,
                                            FailingNet(), [ai.Action(-1, 'declare_decline')])

    def test_phase_prior_nonfinite_value_is_not_a_plausible_decline(self):
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}):
            with self.assertRaisesRegex(ValueError, 'Non-finite search value'):
                ai._greedy_prior_action(initial(), gs.Side.FREE_PEOPLES,
                                        NonfiniteNet(), [ai.Action(-1, 'declare_decline')])

    def test_full_turn_transition_does_not_roll_after_model_failure(self):
        state = initial()
        for dice in state.dice.values():
            for die in dice:
                die.used = True
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}), \
                patch.object(ai, '_SEARCH_NETS', (FailingNet(), FailingNet())), \
                patch.object(game_logic, 'roll_action_dice',
                             wraps=game_logic.roll_action_dice) as roll:
            with self.assertRaisesRegex(RuntimeError, 'literal phase model failure'):
                ai._advance_turn_for_search(state)
            roll.assert_not_called()
        self.assertFalse(getattr(state, '_declare_phase', False))
        self.assertFalse(getattr(state, '_eye_allocation_phase', False))

    def test_explicit_missing_model_and_legacy_fallback_remain_available(self):
        action = ai.Action(-1, 'declare_decline')
        for contract, net in ((PROFILE, None), (LEGACY, FailingNet())):
            with self.subTest(contract=contract), patch.dict(
                    os.environ, {'WOTR_REWARD_CONTRACT': contract}):
                self.assertIs(ai._greedy_prior_action(initial(), gs.Side.FREE_PEOPLES,
                                                      net, [action]), action)

    def test_failed_required_discard_is_not_cleared_to_continue_the_turn(self):
        state = initial()
        state.hands[gs.Side.FREE_PEOPLES].hand = [state.fp_char_deck.pop() for _ in range(5)]
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}), \
                patch.object(ai, '_SEARCH_NETS', (None, None)), \
                patch.object(ai, '_execute_action_inner',
                             side_effect=RuntimeError('literal discard execution failure')), \
                patch.object(game_logic, 'roll_action_dice',
                             wraps=game_logic.roll_action_dice) as roll:
            with self.assertRaisesRegex(RuntimeError, 'literal discard execution failure'):
                ai._advance_turn_for_search(state)
            roll.assert_not_called()
        self.assertIsNotNone(state.pending_discard)

    def test_required_discard_with_no_legal_choice_is_a_search_failure(self):
        state = initial()
        state.hands[gs.Side.FREE_PEOPLES].hand = [state.fp_char_deck.pop() for _ in range(5)]
        actions = ai.get_ai_actions
        def missing_discard(target, side):
            return [] if getattr(target, 'pending_discard', None) else actions(target, side)
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}), \
                patch.object(ai, '_SEARCH_NETS', (None, None)), \
                patch.object(ai, 'get_ai_actions', side_effect=missing_discard), \
                patch.object(game_logic, 'roll_action_dice',
                             wraps=game_logic.roll_action_dice) as roll:
            with self.assertRaisesRegex(RuntimeError, 'Required discard has no legal choice'):
                ai._advance_turn_for_search(state)
            roll.assert_not_called()
        self.assertIsNotNone(state.pending_discard)

    def test_failed_declaration_is_not_silently_treated_as_decline(self):
        state = initial()
        state.fellowship.progress = 1
        execute = ai._execute_action_inner
        def choose(_state, _side, _net, actions):
            return next((a for a in actions if a.action_type == 'fellowship_declare'),
                        actions[0] if actions else None)
        def fail_declare(target, action):
            if action.action_type == 'fellowship_declare':
                raise RuntimeError('literal declaration execution failure')
            return execute(target, action)
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}), \
                patch.object(ai, '_SEARCH_NETS', (None, None)), \
                patch.object(ai, '_greedy_prior_action', side_effect=choose), \
                patch.object(ai, '_execute_action_inner', side_effect=fail_declare), \
                patch.object(game_logic, 'roll_action_dice',
                             wraps=game_logic.roll_action_dice) as roll:
            with self.assertRaisesRegex(RuntimeError, 'literal declaration execution failure'):
                ai._advance_turn_for_search(state)
            roll.assert_not_called()
        self.assertFalse(getattr(state, '_declare_phase', False))

    def test_failed_hunt_allocation_cannot_become_zero_eyes(self):
        state = initial()
        def choose(_state, _side, _net, actions):
            return next((a for a in actions if a.action_type == 'eye_allocate'
                         and a.params.get('count') == 1), actions[0] if actions else None)
        with patch.dict(os.environ, {'WOTR_REWARD_CONTRACT': PROFILE}), \
                patch.object(ai, '_SEARCH_NETS', (None, None)), \
                patch.object(ai, '_greedy_prior_action', side_effect=choose), \
                patch.object(game_logic, 'allocate_eyes',
                             side_effect=RuntimeError('literal Hunt allocation failure')), \
                patch.object(game_logic, 'roll_action_dice',
                             wraps=game_logic.roll_action_dice) as roll:
            with self.assertRaisesRegex(RuntimeError, 'literal Hunt allocation failure'):
                ai._advance_turn_for_search(state)
            roll.assert_not_called()
        self.assertFalse(getattr(state, '_eye_allocation_phase', False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
