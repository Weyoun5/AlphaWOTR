"""Actual Lengthens checkpoint and distinct engine-generated continuation menus."""
from pathlib import Path
import copy
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import card_options
import event_effects as ee
import game_logic as gl
import game_state as gs
from live.decision_prompts import pending_instructions
from live.live_server import LiveServer
from test_live_driver import fresh_state

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
CAPTURE = ROOT/'audit/bridge_oct11/card_completion_targets/shadow_lengthens_passive_r2/illegal_before_done'


class CardChainPrompts(unittest.TestCase):
    def server(self, state, human):
        folder = tempfile.TemporaryDirectory(prefix='card_prompt_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        server = LiveServer(folder.name, SH if human is FP else FP, load_nets=False, seed=7)
        server.driver.es = state
        for model in server.driver.hand_models.values():
            model.es = state
        server.driver.phase = 'actions'
        server.driver.last_actor = human
        server.driver.enforce_human = True
        return server

    def prompt(self, state, human):
        server = self.server(state, human)
        pending = copy.deepcopy(state.pending_card_option)
        dp = ai.describe_pending(state)
        self.assertIs(dp['side'], human)
        before = tuple((a.action_type, repr(a.params)) for a in dp['actions'])
        items = server.driver._maybe_announce_pending()
        self.assertEqual(len(items), 1)
        rendered = server._dispatch(items[0])
        self.assertEqual(len(rendered), 1)
        self.assertEqual(server.driver._maybe_announce_pending(), [])
        self.assertEqual(state.pending_card_option, pending)
        self.assertEqual(tuple((a.action_type, repr(a.params))
                              for a in ai.describe_pending(state)['actions']), before)
        self.assertEqual(server.driver.n_commits_total, 0)
        self.assertIsNone(server.rand.pending())
        self.assertFalse(server.driver.alarms)
        return rendered[0].lower(), pending_instructions(dp, state).lower()

    def companions(self):
        state = fresh_state()
        state._declare_phase = False
        state.armies = {}
        army = gl.get_or_create_army(state, 'bree')
        for name in ('Gimli', 'Legolas'):
            companion = next(c for c in state.fellowship.companions if c.name == name)
            state.fellowship.companions.remove(companion)
            army.leaders.append(companion)
        return state

    def test_actual_lengthens_army_menu_never_prompts_nazgul(self):
        state = pickle.loads((CAPTURE/'ai_live_state.pkl').read_bytes())
        self.assertTrue(any(option.get('_tsl_done') for option in state.pending_card_option['options']))
        spoken, status = self.prompt(state, SH)
        self.assertIn('group movement', spoken)
        self.assertIn('then say done', spoken)
        self.assertIn('keep the action die staged', spoken)
        self.assertNotIn('nazgul', spoken)
        self.assertIn(status, spoken)

    def test_nazgul_chain_still_names_nazgul(self):
        state = fresh_state()
        state._declare_phase = False
        state.armies = {}
        gl.get_or_create_army(state, 'barad dur').leaders.append(gs.Nazgul(gs.Nation.SAURON))
        card = ee._card_by_name('The Ringwraiths Are Abroad')
        self.assertTrue(ee._seed_card_nazgul_chain(state, card))
        spoken, status = self.prompt(state, SH)
        self.assertIn('nazgul', spoken)
        self.assertIn('say done', spoken)
        self.assertIn('nazgul', status)
        self.assertNotIn('companions', spoken)

    def test_companion_die_chain_still_names_companions(self):
        state = self.companions()
        options = ee._companion_chain_options(state)
        self.assertTrue(options)
        state.pending_card_option = dict(card_id='__companion_moves__', card_name='__companion_moves__',
            side='fp', decider='fp', direct=True, options=options+[{'_comp_done': True}])
        spoken, status = self.prompt(state, FP)
        self.assertIn('companions', spoken)
        self.assertIn('say done', spoken)
        self.assertIn('companions', status)
        self.assertNotIn('nazgul', spoken)

    def test_mazarbul_ffm_chain_names_companions(self):
        state = self.companions()
        card = ee._card_by_name('Book of Mazarbul')
        options = card_options._companion_move_card(state, FP)
        self.assertTrue(any(option.get('_ffm_done') for option in options))
        state.pending_card_option = dict(card_id=card.card_id, card_name=card.card_name,
            side='fp', decider='fp', direct=True, options=options)
        spoken, status = self.prompt(state, FP)
        self.assertIn('companions', spoken)
        self.assertIn('say done', spoken)
        self.assertIn('companions', status)

    def test_ents_continuation_offers_card_play_or_done_without_piece_movement(self):
        state = fresh_state()
        state._declare_phase = False
        state.armies = {}
        gl.get_or_create_army(state, 'fangorn').leaders.append(
            gs.Companion(name='Gandalf the White', level=3, leadership=1))
        state.fellowship.corruption = 2
        state.hands[FP].hand = [ee._card_by_name("Bilbo's Song").card_id]
        card = ee._card_by_name('The Ents Awake: Treebeard')
        # Enter the actual post-casualty continuation; there is no fresh roll.
        ee._effect_ents_awake_common(state, card, {'_from_pending': True, '_ents_cas_resumed': True})
        self.assertTrue(any(option.get('_ents_chain_done') for option in state.pending_card_option['options']))
        self.assertTrue(any(option.get('ents_chain_play') for option in state.pending_card_option['options']))
        spoken, status = self.prompt(state, FP)
        for text in (spoken, status):
            self.assertIn('another', text)
            self.assertIn('character event', text)
            self.assertIn('say done', text)
            self.assertNotIn('move', text)


if __name__ == '__main__':
    unittest.main()
