"""Siege capacity returns living units; it is not hit allocation."""
from pathlib import Path
from collections import Counter
import pickle
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import ai
import board
import game_logic as gl
import game_state as gs
from wotr_bridge.live import chat,decision_prompts
from wotr_bridge.live.live_server import LiveServer


def trim_state(side, count=6):
    state=gs.setup_initial_game(board.load_board())
    region='morannon' if side is gs.Side.FREE_PEOPLES else 'lorien'
    nation=gs.Nation.GONDOR if side is gs.Side.FREE_PEOPLES else gs.Nation.SAURON
    opponent=gs.Side.SHADOW if side is gs.Side.FREE_PEOPLES else gs.Side.FREE_PEOPLES
    enemy_nation=gs.Nation.SAURON if side is gs.Side.FREE_PEOPLES else gs.Nation.ELVES
    state.armies={region:gs.Army(region,
        [gs.Unit(nation,gs.UnitKind.REGULAR) for _ in range(count-1)]
        +[gs.Unit(nation,gs.UnitKind.ELITE),gs.Unit(enemy_nation,gs.UnitKind.REGULAR)],[])}
    state.enemy_controlled[side].add(region)
    state.besieged_strongholds.add(region)
    state.pending_combat={'phase':'siege_trim','trim_region':region,
        'defender_side':side,'attacker_side':opponent,'deciding_side':side,
        'attacker_region':region,'defender_region':region,'is_siege':True}
    return state,region,nation


class SiegeTrimPromptTests(unittest.TestCase):
    def assert_return_text(self,text,destination):
        self.assertIn('alive to '+destination,text)
        self.assertNotIn('lost units',text)
        self.assertNotIn('Regular replacement',text)
        self.assertNotIn('reduce an Elite',text)

    def test_exact_real_trim_actions_return_each_kind_without_casualties(self):
        for side,destination in ((gs.Side.FREE_PEOPLES,'FP Reinforcements'),
                                 (gs.Side.SHADOW,'SA Reinforcements')):
            state,region,nation=trim_state(side)
            dp=ai.describe_pending(state,side)
            self.assertEqual(dp['kind'],'combat:siege_trim')
            self.assertEqual({(a.action_type,a.params['excess'],a.params['elites_back'])
                              for a in dp['actions']},{('siege_trim',1,0),('siege_trim',1,1)})
            before=pickle.dumps(state)
            self.assert_return_text(decision_prompts.pending_instructions(dp,state),destination)
            self.assertIn('leaving 5 army units',decision_prompts.pending_instructions(dp,state))
            self.assertEqual(pickle.dumps(state),before)
            for action in dp['actions']:
                with self.subTest(side=side,elites=action.params['elites_back']):
                    post=ai.apply_action(state,action)
                    ours=[u for u in post.armies[region].units if u.nation==nation]
                    kinds=Counter(u.kind for u in ours)
                    self.assertEqual(len(ours),5)
                    self.assertEqual(kinds[gs.UnitKind.ELITE],1-action.params['elites_back'])
                    self.assertEqual(post.fp_casualties,state.fp_casualties)
                    returned='elite' if action.params['elites_back'] else 'regular'
                    self.assertEqual(ai.figures_available(post,nation,returned),
                                     ai.figures_available(state,nation,returned)+1)
                    self.assertIsNone(post.pending_combat)
                    self.assertEqual(len(post.armies[region].units),6)  # Enemy untouched.
                    self.assertIn('alive to reinforcements',decision_prompts.action_description(action))

    def test_live_prompt_status_and_refusal_use_same_reserve_instruction(self):
        for side,destination in ((gs.Side.FREE_PEOPLES,'FP Reinforcements'),
                                 (gs.Side.SHADOW,'SA Reinforcements')):
            with tempfile.TemporaryDirectory(prefix='trim_prompt_',dir=ROOT) as folder:
                opposite=gs.Side.SHADOW if side is gs.Side.FREE_PEOPLES else gs.Side.FREE_PEOPLES
                server=LiveServer(folder,opposite,load_nets=False,seed=931)
                state,_,_=trim_state(side)
                server.driver.es=state;server.driver.phase='actions'
                server.driver.last_actor=opposite
                dp=ai.describe_pending(state,side)
                before=pickle.dumps(state)
                prompt=server.driver._pending_words(dp)
                self.assert_return_text(prompt,destination)
                self.assert_return_text(server._waiting_on(),destination)
                self.assert_return_text(server.driver.stall_text(),destination)
                refused=server.driver._on_chat({'op':'chat','nick':'Human','body':'banana'})
                self.assertTrue(any(item[:2]==('chat','refusal') for item in refused),refused)
                self.assert_return_text(' '.join(str(item[-1]) for item in refused),destination)
                self.assertEqual(pickle.dumps(state),before)
                self.assertEqual(server.driver.n_commits_total,0)

    def test_reminder_without_menu_and_multiple_excess_keep_capacity_wording(self):
        self.assert_return_text(chat.pending_reminder('combat:siege_trim'),'your reinforcements')
        state,_,_=trim_state(gs.Side.FREE_PEOPLES,count=7)
        dp=ai.describe_pending(state,gs.Side.FREE_PEOPLES)
        text=decision_prompts.pending_instructions(dp,state)
        self.assertIn('2 excess army units',text)
        self.assert_return_text(text,'FP Reinforcements')

    def test_real_hit_allocation_still_explains_elite_replacement(self):
        text=decision_prompts.pending_instructions({'kind':'combat:casualty_allocation'})
        self.assertIn('Regular replacement',text)
        self.assertNotIn('alive to',text)


if __name__=='__main__':unittest.main(verbosity=2)
