"""Literal army-card split gestures and a Corsairs battle opened by done."""
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import board
import event_effects as ee
import game_logic as gl
import game_state as gs
from test_live_driver import fresh_state
from test_bridge_oct12_dead_men_prefix_live import Rig

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


class CargoCompletionLive(unittest.TestCase):
    def position(self, corsairs=False):
        state = fresh_state()
        state.armies = {}
        state._declare_phase = False
        state.dice_rolled = True
        for side, pool in state.dice.items():
            for die in pool:
                die.used = True
                die.current = gs.ActionDieFace.ARMY if side is SH else gs.ActionDieFace.CHARACTER
        state.dice[SH][0].used = False
        for nation in (gs.Nation.SAURON, gs.Nation.SOUTHRON_EASTERLINGS):
            state.political.boxes[nation] = 3
            state.political.active[nation] = True
        source = 'umbar' if corsairs else 'gorgoroth'
        target = 'pelargir' if corsairs else 'nurn'
        nation = gs.Nation.SOUTHRON_EASTERLINGS if corsairs else gs.Nation.SAURON
        gl.get_or_create_army(state, source).units = [gs.Unit(nation, gs.UnitKind.REGULAR) for _ in range(3)]
        destination_nation = gs.Nation.GONDOR if corsairs else nation
        gl.get_or_create_army(state, target).units = [gs.Unit(destination_nation, gs.UnitKind.REGULAR)]
        name = 'Corsairs of Umbar' if corsairs else 'Shadows Gather'
        card = ee._card_by_name(name)
        state.hands[SH].hand = [card.card_id]
        self.assertTrue(any(a.action_type == 'play_event' and a.params['card_id'] == card.card_id
                            for a in ai.get_legal_actions(state, SH)))
        rig = Rig(state, FP, phase='actions')
        self.addCleanup(shutil.rmtree, rig.td)
        rig.drv.enforce_human = True
        rig.drv.last_actor = FP
        rig.feed('$Wey moves Shadow Action Die (A) from SA Dice to Current Action.',
                 '$Wey moves %s from Shadow Hand to Current SA Card.' % name)
        self.assertIsNotNone(rig.drv.es.pending_card_option)
        self.assertIsNotNone(rig.drv.seg.window)
        from live.decision_prompts import pending_instructions
        prompt = pending_instructions(ai.describe_pending(rig.drv.es), rig.drv.es)
        self.assertIn('then say done', prompt)
        self.assertIn('keep the action die staged', prompt)
        token = 'S&E Regular' if corsairs else 'Sauron Regular'
        line = '$Wey moves %s from %s to %s.' % (token, source.title(), target.title())
        return rig, source, target, nation, line

    @staticmethod
    def count(rig, region, nation):
        return sum(u.nation is nation for key, army in rig.drv.es.armies.items()
                   if board.sanitize(key) == region for u in army.units)

    def test_shadows_gather_two_piece_split_waits_for_done(self):
        rig, source, target, nation, line = self.position()
        before = rig.drv.n_commits_total
        rig.feed(line)
        self.assertEqual(rig.drv.n_commits_total, before)
        self.assertEqual(self.count(rig, source, nation), 3)
        rig.feed(line)
        self.assertEqual(rig.drv.n_commits_total, before)
        rig.feed('<Wey> done')
        self.assertEqual(self.count(rig, source, nation), 1)
        self.assertEqual(self.count(rig, target, nation), 3)
        self.assertIsNone(ai.describe_pending(rig.drv.es))
        self.assertIsNotNone(rig.drv.seg.window)
        self.assertFalse(rig.drv.alarms)

    def test_shadows_gather_one_piece_remains_legal(self):
        rig, source, target, nation, line = self.position()
        rig.feed(line, '<Wey> done')
        self.assertEqual(self.count(rig, source, nation), 2)
        self.assertEqual(self.count(rig, target, nation), 2)
        self.assertFalse(rig.drv.alarms)

    def test_corsairs_done_opens_battle_with_action_die_still_staged(self):
        rig, source, target, nation, line = self.position(corsairs=True)
        rig.drv.view.piece_region['260'] = 'Political Track 2'
        political_before = dict(rig.drv.es.political.boxes)
        before = rig.drv.n_commits_total
        rig.feed(line)
        self.assertEqual(rig.drv.n_commits_total, before)
        self.assertIsNone(rig.drv.es.pending_combat)
        rig.feed(line)
        self.assertEqual(rig.drv.n_commits_total, before)
        output = rig.feed('<Wey> done')
        self.assertEqual(self.count(rig, source, nation), 1)
        self.assertEqual(self.count(rig, target, nation), 2)
        self.assertIsNotNone(rig.drv.es.pending_combat)
        self.assertEqual(rig.drv.es.pending_combat['defender_region'], target)
        self.assertIsNotNone(rig.drv.seg.window)
        self.assertIsNone(rig.rand.pending())
        self.assertFalse(rig.drv.alarms)
        expected_politics = dict(political_before)
        expected_politics[gs.Nation.GONDOR] += 1
        self.assertEqual(rig.drv.es.political.boxes, expected_politics)
        political_lines = [item[1] for item in output
                           if item[0] == 'line' and 'HighlightLabel.' in item[1]]
        self.assertEqual(political_lines, [
            '$WOTR_AI<~HighlightLabel.96~>260<~HighlightLabel.97~><~Game.2001~>'
            '<~HighlightLabel.98~><~Game.2002~>'])

    def test_corsairs_wrong_voiced_political_step_is_not_ignored(self):
        rig, source, target, nation, line = self.position(corsairs=True)
        before = rig.drv.n_commits_total
        rig.feed(line, line,
                 '$Wey moves Gondor from Political Track 2 to Political Track 0.',
                 '<Wey> done')
        self.assertEqual(rig.drv.n_commits_total, before)
        self.assertEqual(self.count(rig, source, nation), 3)
        self.assertIsNone(rig.drv.es.pending_combat)
        self.assertIsNotNone(rig.drv.pending_buf)
        self.assertIsNone(rig.rand.pending())


if __name__ == '__main__':
    unittest.main()
