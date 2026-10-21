"""Native Recover routes through server while both new-turn pools are unrolled."""
from collections import defaultdict
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import board
import emitter
import game_state as gs
from live import client_board
from live.live_server import LiveServer


class BonusRolloverGuidance(unittest.TestCase):
    def rollover(self, human, already_taken):
        folder = tempfile.TemporaryDirectory(prefix='bonus_rollover_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)
        feed = path / 'ai_feed.txt'
        feed.write_text('', encoding='utf8')
        ai_side = gs.Side.SHADOW if human is gs.Side.FREE_PEOPLES else gs.Side.FREE_PEOPLES
        server = LiveServer(str(path), ai_side, load_nets=False, seed=47)
        drv = server.driver
        drv.phase = 'actions'
        drv.enforce_human = True
        if human is gs.Side.SHADOW:
            drv.es.saruman.in_play = True
        else:
            drv.es.gandalf_the_white.in_play = True
            grey = next(c for c in drv.es.fellowship.companions if c.name == 'Gandalf the Grey')
            drv.es.fellowship.companions.remove(grey)
            if drv.es.fellowship.guide is grey:
                drv.es.fellowship.guide = drv.es.fellowship.companions[0]
            grey.name = 'Gandalf the White'
            next(army for region, army in drv.es.armies.items()
                 if 'grey havens' in board.sanitize(region)).leaders.append(grey)
        counts = {side: len(pool) for side, pool in drv.es.dice.items()}
        for pool in drv.es.dice.values():
            for die in pool:
                die.current, die.used = gs.ActionDieFace.CHARACTER, True
        dump = client_board.read_board(str(ROOT / 'data/live_fixtures/client_board_20260821_strider_guide.txt'))
        for side, pids in ((gs.Side.FREE_PEOPLES, emitter._FP_DIE_IDS),
                           (gs.Side.SHADOW, emitter._SH_DIE_IDS)):
            label = 'FP' if side is gs.Side.FREE_PEOPLES else 'SA'
            for index, pid in enumerate(pids):
                dump[pid] = label + (' Dice' if side is human and index < counts[side]
                                     else ' Casualties' if index < counts[side] else ' Reinforcements')
        pids = emitter._FP_DIE_IDS if human is gs.Side.FREE_PEOPLES else emitter._SH_DIE_IDS
        bonus = pids[counts[human]]
        label = 'FP' if human is gs.Side.FREE_PEOPLES else 'SA'
        if already_taken:
            dump[bonus] = label + ' Dice'
        groups = defaultdict(list)
        for pid, zone in dump.items():
            groups[emitter.zone_code_for(board.sanitize(zone))].append(pid)
        rows = ['#SEQ dump=1 feed_paragraphs=0 ts=1200 session=1000']
        for index, (code, group) in enumerate(groups.items()):
            rows.append('%d\t<~%s~>\t%s\t%s\t%s' % (index, code, ' '.join(group),
                ' '.join('-' for _ in group), ' '.join('-' for _ in group)))
        (path / 'ai_client_board.txt').write_text('\n'.join(rows)+'\n', encoding='utf8')
        count = counts[human] + int(already_taken)
        name = 'Free' if human is gs.Side.FREE_PEOPLES else 'Shadow'
        feed.write_text('$Wey moves group ([%d]%s Action Die (C)) from %s Casualties to %s Dice.\n'
                        % (count, name, label, label), encoding='utf8')
        previous_turn = drv.es.turn
        lines = server.handle({'lines': ()})
        self.assertEqual(drv.es.turn, previous_turn + 1, (lines, drv.alarms))
        self.assertEqual(drv.phase, 'draws')
        self.assertEqual(len(drv.es.dice[human]), counts[human] + 1)
        self.assertEqual(drv.view.die_id_map[bonus], (human, counts[human]))
        self.assertTrue(all(die.current is None and not die.used
                            for pool in drv.es.dice.values() for die in pool))
        self.assertEqual(drv.turn_draws, {})
        self.assertEqual(drv.es.eyes_allocated, 0)
        self.assertFalse(drv.es.dice_rolled)
        self.assertEqual(server._live_asks, {})
        self.assertFalse(any('roll_action_dice' in line for line in lines), lines)
        self.assertFalse(drv.alarms)
        text = ' '.join(lines).lower()
        self.assertIn('wait for the normal pool roll after hunt allocation', text)
        self.assertNotIn('roll it alone', text)
        self.assertNotIn('roll all', text)
        if already_taken:
            self.assertIn('already on the table', text)
            self.assertNotIn('reinforcements to', text)
        else:
            self.assertIn('from %s reinforcements to %s dice' % (label.lower(), label.lower()), text)
        # Idle handling cannot grant or request another roll on this boundary.
        repeat = server.handle({'lines': ()})
        self.assertFalse(any('gained' in line or 'roll_action_dice' in line for line in repeat), repeat)
        self.assertEqual(len(drv.es.dice[human]), counts[human] + 1)

    def test_shadow_new_saruman_die_waits_for_turn_start(self):
        self.rollover(gs.Side.SHADOW, False)

    def test_shadow_already_taken_die_waits_for_turn_start(self):
        self.rollover(gs.Side.SHADOW, True)

    def test_fp_new_white_die_waits_for_turn_start(self):
        self.rollover(gs.Side.FREE_PEOPLES, False)

    def test_fp_already_taken_die_waits_for_turn_start(self):
        self.rollover(gs.Side.FREE_PEOPLES, True)


if __name__ == '__main__':
    unittest.main()
