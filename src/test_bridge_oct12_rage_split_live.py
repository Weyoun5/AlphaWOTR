"""Captured game01 Rage gestures; no scripted AI or native client contact."""
from collections import defaultdict
from pathlib import Path
import json
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import board
import emitter
import game_state as gs
from live import client_board, client_protocol, live_parser
from live.live_server import LiveServer

GAME = ROOT/'audit/bridge_oct11/player/full_games/game01'
BEFORE = GAME/'turn02_human01'
FAILED = GAME/'failure_rage_split'


class RageSplitLive(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='rage_native_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        self.feed = self.path/'ai_feed.txt'
        self.feed.write_text('', encoding='utf8')
        self.meta_path = self.path/'ai_feed.txt.meta.jsonl'
        self.meta_path.write_text('', encoding='utf8')
        self.server = LiveServer(str(self.path), gs.Side.FREE_PEOPLES, load_nets=False, seed=42)
        self.driver = self.server.driver
        state = pickle.loads((BEFORE/'ai_live_state.pkl').read_bytes())
        self.driver.es = state
        for model in self.driver.hand_models.values():
            model.es = state
        self.driver.phase = 'actions'
        self.driver.last_actor = gs.Side.FREE_PEOPLES
        self.driver.enforce_human = True
        status = json.loads((BEFORE/'ai_bridge_status.json').read_text())
        for die in status['human_menu']['dice']:
            self.driver.view.bind_die(die['pid'], gs.Side.SHADOW, die['idx'])
        self.dump = client_board.read_board(str(BEFORE/'ai_client_board.txt'))
        self.markers = client_protocol.read_marker_states(str(BEFORE/'ai_client_board.txt'))
        self.lines = (FAILED/'ai_feed.txt').read_text(encoding='utf8').splitlines()
        self.metadata = {row['seq']: row for row in
                         map(json.loads, (FAILED/'ai_feed.txt.meta.jsonl').read_text().splitlines())}
        self.seq = 0
        self.undo_stack = []
        self.write_dump()
        self.driver.view.refresh_dump()
        self.assertIsNone(ai.describe_pending(state))
        self.assertEqual(self.regulars('gap of rohan'), 0)
        self.assertEqual(self.regulars('north dunland'), 1)
        self.assertEqual(self.regulars('south dunland'), 1)

    def regulars(self, region):
        return sum(u.nation is gs.Nation.ISENGARD and u.kind is gs.UnitKind.REGULAR
                   for key, army in self.driver.es.armies.items() if board.sanitize(key) == region
                   for u in army.units)

    def write_dump(self):
        groups = defaultdict(list)
        for pid, zone in self.dump.items():
            groups[emitter.zone_code_for(board.sanitize(zone))].append(pid)
        rows = ['#SEQ dump=%d feed_paragraphs=%d ts=1200 session=1000' % (self.seq+1, self.seq-1)]
        for index, (code, pids) in enumerate(groups.items()):
            rows.append('%d\t<~%s~>\t%s\t%s\t%s' % (index, code, ' '.join(pids),
                ' '.join('-' for _ in pids), ' '.join(str(self.markers.get(pid, '-')) for pid in pids)))
        (self.path/'ai_client_board.txt').write_text('\n'.join(rows)+'\n', encoding='utf8')

    def native(self, *indices, driver_only=False):
        for index in indices:
            line = self.lines[index-1]
            meta = dict(self.metadata[index])
            self.seq += 1
            meta['seq'] = self.seq
            record = live_parser.parse(line)
            pids = meta.get('pids')
            if index == 95:  # Public combined title is revealed; native sidecar redacts hand exit.
                pids = ['185']
            if record.get('op') == 'undo':
                self.dump.update(self.undo_stack.pop())
            elif record.get('op') in ('move', 'kill') and isinstance(pids, list) and pids:
                self.undo_stack.append({pid: self.dump[pid] for pid in pids})
                zone = record.get('to') if record['op'] == 'move' else 'SA Casualties'
                self.dump.update({pid: zone for pid in pids})
            with self.feed.open('a', encoding='utf8') as stream:
                stream.write(line+'\n')
            with self.meta_path.open('a', encoding='utf8') as stream:
                stream.write(json.dumps(meta)+'\n')
        self.write_dump()
        return self.driver.tick() if driver_only else self.server.handle({'lines': ()})

    def recruit(self):
        self.native(93, 94, 95, 96, 97, 98, 99, 100)
        self.assertEqual(self.regulars('gap of rohan'), 2)
        pending = ai.describe_pending(self.driver.es)
        self.assertEqual(pending['kind'], 'card_option')
        self.assertTrue(all('rage_split' in action.params['choices'] for action in pending['actions']))
        from live.decision_prompts import pending_instructions
        prompt = pending_instructions(pending, self.driver.es)
        self.assertIn('then say done', prompt)
        self.assertIn('keep the action die staged', prompt)

    def test_two_moves_stay_open_until_physical_close(self):
        self.recruit()
        before = self.driver.n_commits_total
        self.native(103)
        self.assertIsNotNone(ai.describe_pending(self.driver.es))
        self.assertEqual(self.driver.n_commits_total, before)
        self.assertEqual(self.regulars('gap of rohan'), 2)
        self.native(104, 105, 106, 107)
        self.assertEqual(self.driver.n_commits_total, before + 1)
        self.assertIsNotNone(self.driver.seg.window)  # card close settles choice, die still owed
        self.assertEqual(self.driver.last_actor, gs.Side.SHADOW)
        self.native(108, driver_only=True)
        self.assertEqual(self.regulars('gap of rohan'), 4)
        self.assertEqual(self.regulars('north dunland'), 0)
        self.assertEqual(self.regulars('south dunland'), 0)
        self.assertIsNone(ai.describe_pending(self.driver.es))
        self.assertFalse(self.driver.alarms)

    def assert_native_census(self):
        rows = client_board.diff_engine_vs_dump(self.driver.es, self.dump,
                   marker_states=self.markers, ai_module=ai)
        self.assertFalse([row for row in rows if not client_board.is_staging_row(row)], rows)

    def test_one_optional_move_is_valid_when_card_closes(self):
        self.recruit()
        self.native(103)
        self.native(107)
        self.assertEqual(self.regulars('gap of rohan'), 3)
        self.assertEqual(self.regulars('south dunland'), 1)
        self.assertIsNone(ai.describe_pending(self.driver.es))
        self.assert_native_census()
        self.native(108, driver_only=True)
        self.assertFalse(self.driver.alarms)

    def test_declining_all_optional_moves_requires_card_close(self):
        self.recruit()
        before = self.driver.n_commits_total
        self.server.handle({'lines': ()})
        self.assertEqual(self.driver.n_commits_total, before)
        self.assertIsNotNone(ai.describe_pending(self.driver.es))
        self.native(107)
        self.assertEqual(self.driver.n_commits_total, before + 1)
        self.assertEqual(self.regulars('gap of rohan'), 2)
        self.assertEqual(self.regulars('north dunland'), 1)
        self.assertEqual(self.regulars('south dunland'), 1)
        self.assertIsNone(ai.describe_pending(self.driver.es))
        self.assert_native_census()
        self.native(108, driver_only=True)
        self.assertFalse(self.driver.alarms)

    def test_die_close_also_settles_the_complete_two_move_choice(self):
        self.recruit()
        self.native(103)
        self.native(104)
        self.assertIsNotNone(ai.describe_pending(self.driver.es))
        self.native(108, driver_only=True)
        self.assertEqual(self.regulars('gap of rohan'), 4)
        self.assertIsNone(ai.describe_pending(self.driver.es))
        self.assert_native_census()

    def test_explicit_done_closes_one_optional_move_without_killing_die(self):
        self.recruit()
        self.native(103)
        with self.feed.open('a', encoding='utf8') as stream:
            stream.write('<Wey> done\n')
        self.seq += 1
        with self.meta_path.open('a', encoding='utf8') as stream:
            stream.write(json.dumps({'seq': self.seq, 'gen': 1, 'nick': 'Wey', 'pids': []})+'\n')
        self.write_dump()
        self.server.handle({'lines': ()})
        self.assertEqual(self.regulars('gap of rohan'), 3)
        self.assertIsNone(ai.describe_pending(self.driver.es))
        self.assertIsNotNone(self.driver.seg.window)
        self.assert_native_census()

    def test_wrong_native_move_undo_then_both_moves_same_poll(self):
        self.recruit()
        self.native(101)
        self.native(102)
        self.native(103, 104, 105, 106, 107)
        self.native(108, driver_only=True)
        self.assertEqual(self.regulars('gap of rohan'), 4)
        self.assertEqual(self.regulars('north dunland'), 0)
        self.assertEqual(self.regulars('south dunland'), 0)
        rows = client_board.diff_engine_vs_dump(self.driver.es, self.dump,
                   marker_states=self.markers, ai_module=ai)
        self.assertFalse([row for row in rows if not client_board.is_staging_row(row)], rows)
        self.assertFalse(self.driver.alarms)


if __name__ == '__main__':
    unittest.main()
