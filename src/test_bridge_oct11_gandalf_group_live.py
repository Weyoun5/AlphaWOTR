"""Gandalf's physical bonus die must not become a second action payment."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import pickle
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import game_state as gs
import test_live_driver as td
from wotr_bridge.live import client_board, scenario, sidecar
import test_bridge_oct11_combat_recovery as recovery_fixtures

HERE = ROOT/'audit/bridge_oct11/player'
STAGE = '$Wey moves Free Action Die (W) from FP Dice to Current Action.'
ARRIVE = '$Wey moves Gandalf the Grey from FP Casualties to The Grey Havens.'
FLIP = '$Wey flips Gandalf the White in The Grey Havens.'
BONUS = '$Wey moves Free Action Die (W) from FP Reinforcements to Current Action.'
CLOSE = '$Wey kills group ([2]Free Action Die (W)) in Current Action.'


class GandalfGroupedLive(unittest.TestCase):
    write_dump = recovery_fixtures.LiveCombatRecovery.write_dump

    def setUp(self):
        spec = importlib.util.spec_from_file_location('oct11_gandalf_cold_setup', HERE/'cold_setup.py')
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        fixture = HERE/'scenarios/gandalf_group_die'
        stub = scenario.replay_prefix(str(fixture/'setup_feed.txt'), 17, 'SHADOW',
                         meta_path=str(fixture/'setup_feed.meta.jsonl'), seed=42)
        from test_live_witnesses import build_dump_from_state
        base_dump = build_dump_from_state(stub.es)
        manifest = json.loads((HERE/'gandalf_group_die_setup.json').read_text())
        setup.apply_position(stub, manifest)
        self.rig = td.Rig(stub.es, gs.Side.SHADOW, phase='actions')
        self.addCleanup(shutil.rmtree, self.rig.td)
        self.driver = self.rig.drv
        self.rand = self.rig.rand
        self.driver.last_actor = gs.Side.SHADOW
        self.driver.view.die_id_map = dict(stub.view.die_id_map)
        self.feed = Path(self.rig.path)
        self.feed.write_text('#BLIND ai_side=SHADOW\n', encoding='utf-8')
        self.meta = Path(str(self.feed)+'.meta.jsonl')
        self.meta.write_text('', encoding='utf-8')
        self.driver.sidecar = sidecar.SidecarReader(str(self.meta))
        self.seq = 1
        self.assertEqual(self.driver.view.die_id_map['211'], (gs.Side.FREE_PEOPLES, 1))
        self.assertTrue(any(a.action_type == 'transform_gandalf' for a in
                           ai.get_legal_actions(self.driver.es, gs.Side.FREE_PEOPLES)))
        self.dump = client_board.read_board(str(ROOT/'data/live_fixtures/client_board_20260821_strider_guide.txt'))
        self.assertEqual(len(self.dump), 428)
        for pid in self.dump:
            nation, kind = client_board.unit_type(pid)
            if kind in ('regular', 'elite', 'leader', 'nazgul'):
                side = gs.Side.SHADOW if kind == 'nazgul' else ai._side_of_nation(nation)
                self.dump[pid] = 'SA Reinforcements' if side is gs.Side.SHADOW else 'FP Reinforcements'
        for pid, zone in base_dump.items():
            if pid in self.dump and zone != 'Spare':
                self.dump[pid] = zone
        self.dump.update(manifest['expected_board'])
        self.dump['099'] = 'The Hunt For The Ring'
        self.marker_states = {'240': 0, '241': 0, '247': 0}
        self.dump_path = self.feed.with_name('board.txt')

    def physical(self, text, pids, *, segs=None):
        self.seq += 1
        meta = dict(seq=self.seq, gen=1, attest=True, nick='Wey', pids=pids,
                    tags=['Scribe.141', 'Scribe.142', 'Scribe.143', 'Scribe.144'])
        if segs is not None:
            meta.update(tags=['Controls.345', 'Controls.346'], segs=segs)
        with self.meta.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(meta)+'\n')
        return self.rig.feed(text)

    def promote(self, bonus_pid='358'):
        self.physical(STAGE, ['211'])
        self.physical(ARRIVE, ['240'])
        self.physical(FLIP, ['240'])
        self.physical(BONUS, [bonus_pid])
        return self.physical(CLOSE, ['211', bonus_pid])

    def test_group_kill_after_early_bonus_commits_one_legal_transform(self):
        before = [(d.used, d.current) for d in self.driver.es.dice[gs.Side.FREE_PEOPLES]]
        items = self.promote()
        self.assertEqual(self.driver.n_commits_total, 1,
                         (items, self.driver.alarms, self.driver.findings[-6:], self.driver.stall_description()))
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertEqual(len(self.driver.es.dice[gs.Side.FREE_PEOPLES]), 4)
        after = [(d.used, d.current) for d in self.driver.es.dice[gs.Side.FREE_PEOPLES]]
        before[1] = (True, before[1][1])
        self.assertEqual(after, before)
        self.assertIsNone(self.driver.seg.window)
        self.assertIsNone(self.driver.deferred)
        self.assertFalse(self.driver.alarms)

    def test_group_close_then_native_resync_preserves_known_rolls_and_payment(self):
        self.promote()
        self.dump.update({'240': 'Grey Havens', '211': 'FP Dead', '358': 'FP Dead'})
        self.marker_states['240'] = 1
        self.write_dump(feed_position=self.seq - 1)
        self.driver.view.dump_path = str(self.dump_path)
        dice = {side: [(d.used, d.current) for d in pool]
                for side, pool in self.driver.es.dice.items()}
        rng = pickle.dumps(vars(self.rand))
        actor, turn = self.driver.last_actor, self.driver.es.turn
        for _ in range(2):
            items, report = self.driver.reset_and_resync()
            self.assertTrue(self.driver._last_resync_adoption['complete'], (items, report))
            self.assertEqual({side: [(d.used, d.current) for d in pool]
                             for side, pool in self.driver.es.dice.items()}, dice)
            self.assertEqual(pickle.dumps(vars(self.rand)), rng)
            self.assertEqual((self.driver.last_actor, self.driver.es.turn), (actor, turn))
            self.assertEqual(self.driver.n_commits_total, 1)
            self.assertTrue(self.driver.es.gandalf_the_white.in_play)

    def test_native_resync_recovers_promotion_when_group_close_feed_is_missing(self):
        self.physical(STAGE, ['211'])
        self.physical(ARRIVE, ['240'])
        self.physical(FLIP, ['240'])
        self.physical(BONUS, ['358'])
        self.assertEqual(self.driver.n_commits_total, 0)
        self.dump.update({'240': 'Grey Havens', '211': 'FP Dead', '358': 'FP Dead'})
        self.marker_states['240'] = 1
        self.write_dump(feed_position=self.seq - 1)
        self.driver.view.dump_path = str(self.dump_path)
        faces = [d.current for d in self.driver.es.dice[gs.Side.FREE_PEOPLES]]
        rng = pickle.dumps(vars(self.rand))
        items, report = self.driver.reset_and_resync()
        self.assertTrue(self.driver._last_resync_adoption['complete'], (items, report))
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertEqual([d.current for d in self.driver.es.dice[gs.Side.FREE_PEOPLES]], faces)
        self.assertEqual([d.used for d in self.driver.es.dice[gs.Side.FREE_PEOPLES]],
                         [False, True, False, False])
        self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.assertIsNone(self.driver.seg.window)

    def recover_and_roll(self, bonus_pid):
        self.promote(bonus_pid)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.dump.update({'240': 'Grey Havens', '211': 'FP Dead', bonus_pid: 'FP Dead'})
        self.marker_states['240'] = 1
        # Isolate the next legal turn boundary; remaining actions are not
        # part of this promotion/physical-recovery contract.
        for side, dice in self.driver.es.dice.items():
            for die in dice:
                die.used = True
            for pid, binding in self.driver.view.die_id_map.items():
                if binding[0] is side:
                    self.dump[pid] = 'FP Dead' if side is gs.Side.FREE_PEOPLES else 'SA Dead'
        turn = self.driver.es.turn
        fp_pids = ['210', '211', '212', '213', bonus_pid]
        for pid in fp_pids:
            self.dump[pid] = 'FP Dice'
        self.write_dump(feed_position=self.seq)
        self.driver.view.dump_path = str(self.dump_path)
        items = self.physical('$Wey moves group (Free Action Die (P), [2]Free Action Die (W), '
                    'Free Action Die (H), Free Action Die (C)) from FP Casualties to FP Dice.', fp_pids)
        self.assertEqual(self.driver.es.turn, turn + 1, (items, self.driver.alarms))
        self.assertEqual(len(self.driver.es.dice[gs.Side.FREE_PEOPLES]), 5)
        self.assertEqual(self.driver.phase, 'draws')
        grants = [item for item in items if item[0] == 'bonus_dice']
        self.assertEqual(len(grants), 1, items)
        from wotr_bridge.live.live_server import LiveServer
        server = SimpleNamespace(driver=self.driver, ai_side=gs.Side.SHADOW,
                human_side=gs.Side.FREE_PEOPLES, nick=self.driver.nick, notes=[],
                _say=lambda kind, text: (kind, text))
        lines = LiveServer._bonus_dice_lines(server, grants[0][1])
        self.assertTrue(any('already' in str(line).lower() for line in lines), lines)
        self.assertFalse(any('drag' in str(line).lower() for line in lines), lines)
        # A second physical recovery leg is the same native Recover press.
        more = self.physical('$Wey moves Shadow Action Die (E) from The Hunt For The Ring to SA Dice.', ['099'])
        self.assertEqual(self.driver.es.turn, turn + 1)
        self.assertEqual(len(self.driver.es.dice[gs.Side.FREE_PEOPLES]), 5)
        self.assertFalse(any(item[0] == 'bonus_dice' for item in more))
        # The draw/declaration/eye choices are separate from this contract.
        self.driver.phase = 'eyes'
        self.physical('$Wey rolls W C P H A .', fp_pids,
                      segs=[[], fp_pids, ['0', '1', '2', '4', '3']])
        self.assertEqual([d.current.name for d in self.driver.es.dice[gs.Side.FREE_PEOPLES]],
                         ['WILL_OF_WEST', 'CHARACTER', 'EVENT', 'MUSTER_ARMY', 'ARMY'])
        self.assertEqual(self.driver.view.die_id_map[bonus_pid], (gs.Side.FREE_PEOPLES, 4))
        self.assertEqual(len({pid for pid, bind in self.driver.view.die_id_map.items()
                              if bind[0] is gs.Side.FREE_PEOPLES}), 5)
        self.assertFalse(self.driver.alarms)

    def test_early_bonus_358_is_recovered_and_rolled_once(self):
        self.recover_and_roll('358')

    def test_either_native_bonus_piece_can_be_recovered_and_rolled_once(self):
        self.recover_and_roll('359')


if __name__ == '__main__':
    unittest.main()
