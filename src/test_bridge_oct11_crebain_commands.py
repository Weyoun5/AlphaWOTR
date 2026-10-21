"""Human Flocks replies identify real pending options without numbered picks."""
from pathlib import Path
import pickle
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import board
import cards
import event_effects
import game_logic as gl
import game_state as gs
from wotr_bridge.live import chat, decision_prompts


class CrebainCommands(unittest.TestCase):
    def setUp(self):
        state = gs.setup_initial_game(board.load_board())
        gl.initialize_decks(state)
        gl.adopt_rolled_faces(state, [(gs.Side.FREE_PEOPLES, 0, gs.ActionDieFace.CHARACTER)])
        gl.allocate_eyes(state, 1, indices=[0])
        card = next(c for c in cards.get_cards().values() if c.card_name == 'Flocks of Crebain')
        event_effects._effect_flocks_of_crebain(state, card, {})
        move = next(a for a in ai.get_legal_actions(state, gs.Side.FREE_PEOPLES)
                    if a.action_type == 'fellowship_move')
        self.state = ai.apply_action(state, move)
        self.dp = ai.describe_pending(self.state)
        self.assertEqual(self.dp['kind'], 'card_option')
        self.assertIs(self.dp['side'], gs.Side.SHADOW)
        self.actions = self.dp['actions']

    def test_plain_named_replies_select_only_the_intended_legal_option(self):
        before = pickle.dumps(self.state)
        for discard, phrases in ((False, ('keep', 'keep flocks', 'keep flocks of crebain')),
                                 (True, ('discard flocks', 'discard flocks of crebain'))):
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    action = chat.match_choice(phrase, self.actions)
                    self.assertIsNotNone(action)
                    self.assertIs(action.params['choices']['crebain_discard'], discard)
                    self.assertEqual(action.params['card_id'], self.state.pending_card_option['card_id'])
        self.assertEqual(pickle.dumps(self.state), before)

    def test_unrelated_or_unavailable_option_is_not_invented(self):
        keep = [a for a in self.actions if not a.params['choices']['crebain_discard']]
        self.assertIsNone(chat.match_choice('discard flocks', keep))
        for phrase in ('discard wizard staff', 'keep nazgul', 'press', 'none'):
            self.assertIsNone(chat.match_choice(phrase, self.actions), phrase)
        unrelated = [ai.Action(-1, 'card_option', {'card_id': '__other__',
                       'choices': {'crebain_discard': True}})]
        self.assertIsNone(chat.match_choice('discard flocks', unrelated))

    def test_instructions_and_panel_labels_identify_both_options(self):
        text = decision_prompts.pending_instructions(self.dp, self.state)
        self.assertIn('keep flocks', text.lower())
        self.assertIn('discard flocks', text.lower())
        self.assertNotIn('pick', text.lower())
        labels = [decision_prompts.action_description(a) for a in self.actions]
        self.assertEqual(len(set(labels)), 2)
        self.assertTrue(all('Flocks of Crebain' in label for label in labels))


if __name__ == '__main__':
    unittest.main()
