"""Independent player-feed acceptance tests, with immutable captured inputs.

No engine action generator or emitter constructs the inputs or expected board
outcomes.  Each complete gesture is delivered at every possible line boundary,
including one line per tick.  A held gesture is a failure, never a coverage skip.

Run with Python's unittest runner, or execute this file.  To test another tree:
set WOTR_AUDIT_TARGET to that tree before starting a fresh Python process.  The
fixture remains in this checkout; all production imports come from the target.
This exercises the real Python ingest path, not a running Java GUI.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

SOURCE_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = Path(os.environ.get('WOTR_AUDIT_TARGET', SOURCE_ROOT)).resolve()
FIXTURE = SOURCE_ROOT / 'wotr_bridge/tests/fixtures/audit0904_player_gestures.json'
# Prevent the test script's src directory from supplying missing release code.
if TARGET_ROOT != SOURCE_ROOT:
    sys.dont_write_bytecode = True
    sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(SOURCE_ROOT)]
sys.path[:0] = [str(TARGET_ROOT / 'src'), str(TARGET_ROOT / 'wotr_bridge'), str(TARGET_ROOT)]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import board
import game_logic
import game_state
from game_state import Side, ActionDieFace
from live.driver import LiveDriver
from live.randomness import ClientRandomness


def captured(name):
    return json.loads(FIXTURE.read_text(encoding='utf-8'))[name]['lines']


def batches(lines, mask):
    pending = []
    for i, line in enumerate(lines):
        pending.append(line)
        if i == len(lines) - 1 or mask & (1 << i):
            yield pending
            pending = []


def initial(face=ActionDieFace.CHARACTER):
    st = game_state.setup_initial_game(board.load_board())
    game_logic.initialize_decks(st)
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, face),
        (Side.FREE_PEOPLES, 1, ActionDieFace.CHARACTER),
        (Side.SHADOW, 0, ActionDieFace.ARMY),
    ])
    return st


def army_counts(st, region):
    """Read raw state objects; do not use the bridge's board projection."""
    army = next((a for k, a in st.armies.items()
                 if board.sanitize(k) == board.sanitize(region)), None)
    if army is None:
        return Counter()
    counts = Counter((u.nation.name, u.kind.name) for u in army.units)
    for leader in army.leaders:
        counts[(leader.nation.name, 'LEADER')] += 1
    return counts


class PlayerGestureTests(unittest.TestCase):
    def rig(self, st):
        directory = tempfile.TemporaryDirectory(prefix='wotr-audit0904-')
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'ai_feed.txt'
        path.write_bytes(b'')
        rand = ClientRandomness(seed=7)
        rand.install()
        drv = LiveDriver(st, Side.SHADOW, str(path), randomness=rand)
        drv.phase = 'actions'
        return drv, path, rand

    @staticmethod
    def feed(drv, path, lines):
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines) + '\n')
        return drv.tick()

    def assert_settled(self, drv):
        self.assertEqual(drv.alarms, [])
        self.assertIsNone(drv.seg.window)
        self.assertIsNone(drv.deferred)
        self.assertEqual(drv.holds, [])
        self.assertEqual(drv.window_stats['matched'], 1)
        self.assertEqual(drv.window_stats['none'], 0)
        self.assertEqual(drv.window_stats['ambiguous'], 0)
        self.assertIs(drv.last_actor, Side.FREE_PEOPLES)

    def assert_movement(self, drv, progress, hunt_count):
        self.assert_settled(drv)
        self.assertEqual(drv.es.fellowship.progress, progress)
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertTrue(drv.es.dice[Side.FREE_PEOPLES][0].used)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][1].used)
        self.assertFalse(drv.es.dice[Side.SHADOW][0].used)
        self.assertEqual(len(drv.es.hunt_box), hunt_count)

    def test_first_move_track_then_direct_hunt_payment(self):
        lines = captured('shortcut_track_first')
        for mask in range(2):
            with self.subTest(partition=mask):
                drv, path, _ = self.rig(initial())
                for group in batches(lines, mask):
                    out = self.feed(drv, path, group)
                self.assert_movement(drv, 1, 1)
                self.assertTrue(any(i[0] == 'ai_decide' for i in out))

    def test_second_move_direct_payment_then_track(self):
        lines = captured('shortcut_die_first_second_move')
        for mask in range(2):
            with self.subTest(partition=mask):
                st = initial()
                # Explicit precondition from this recorded second move;
                # no hunt eyes were allocated or rolled in that game.
                st.fellowship.progress = 1
                drv, path, _ = self.rig(st)
                for group in batches(lines, mask):
                    out = self.feed(drv, path, group)
                self.assert_movement(drv, 2, 1)
                self.assertTrue(any(i[0] == 'ai_decide' for i in out))

    def test_shortcut_waits_for_recorded_hunt_then_exposes_damage_choice(self):
        import ai
        lines = captured('shortcut_hunt_hit')
        for mask in range(8):
            with self.subTest(partition=mask):
                st = initial()
                game_logic.allocate_eyes(st, 3, indices=[2, 3, 4])
                drv, path, rand = self.rig(st)
                for group in batches(lines, mask):
                    self.feed(drv, path, group)
                self.assert_settled(drv)
                self.assertEqual(drv.es.fellowship.progress, 1)
                self.assertEqual(drv.es.fellowship.corruption, 0)
                pending = ai.describe_pending(drv.es)
                self.assertIsNotNone(pending)
                self.assertEqual(pending['kind'], 'hunt')
                self.assertIs(pending['side'], Side.FREE_PEOPLES)
                self.assertEqual(drv.es.pending_hunt['damage'], 3)
                self.assertEqual({a.action_type for a in pending['actions']}, {
                    'hunt_take_corruption', 'hunt_sacrifice_guide',
                    'hunt_sacrifice_random'})
                self.assertFalse(any(rand.unconsumed().values()))

    def test_shortcut_without_hunt_result_never_guesses(self):
        st = initial()
        game_logic.allocate_eyes(st, 3, indices=[2, 3, 4])
        drv, path, _ = self.rig(st)
        out = self.feed(drv, path, captured('shortcut_hunt_hit')[:2])
        self.assertEqual(drv.es.fellowship.progress, 0)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)
        self.assertIsNotNone(drv.deferred)
        self.assertTrue(any(i[0] == 'rng_ask' and i[1].kind == 'd6' for i in out))
        self.assertFalse(any(i[0] in ('ai_decide', 'line') for i in out))

    def test_prefix_without_die_does_not_commit(self):
        drv, path, _ = self.rig(initial())
        out = self.feed(drv, path, captured('shortcut_track_first')[:1])
        self.assertEqual(drv.es.fellowship.progress, 0)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)
        self.assertIsNotNone(drv.seg.window)
        self.assertEqual(drv.window_stats['matched'], 0)
        self.assertFalse(any(i[0] == 'ai_decide' for i in out))

    def test_player_can_undo_unpaid_track_prefix(self):
        drv, path, _ = self.rig(initial())
        line = captured('shortcut_track_first')[0]
        self.feed(drv, path, [line])
        self.feed(drv, path, ['%Space_Surgeon undoes last action : ' + line])
        self.assertIsNone(drv.seg.window)
        self.assertEqual(drv.es.fellowship.progress, 0)
        self.assertEqual(drv.alarms, [])
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)

    def test_committed_shortcut_undo_restores_die_and_progress(self):
        for name in ('shortcut_track_first', 'shortcut_die_first_second_move'):
            with self.subTest(capture=name):
                st = initial()
                before = 1 if name.endswith('second_move') else 0
                st.fellowship.progress = before
                drv, path, _ = self.rig(st)
                lines = captured(name)
                self.feed(drv, path, lines)
                self.assert_movement(drv, before + 1, 1)
                for line in reversed(lines):
                    self.feed(drv, path, ['%Space_Surgeon undoes last action : ' + line])
                self.assertEqual(drv.es.fellowship.progress, before)
                self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)
                self.assertEqual(len(drv.es.hunt_box), 0)
                self.assertEqual(drv.alarms, [])

    def test_partial_byte_writes_do_not_commit_until_final_newline(self):
        drv, path, _ = self.rig(initial())
        payload = ('\n'.join(captured('shortcut_track_first')) + '\n').encode('utf-8')
        for byte in payload[:-1]:
            with path.open('ab') as stream:
                stream.write(bytes([byte]))
            drv.tick()
        self.assertEqual(drv.es.fellowship.progress, 0)
        with path.open('ab') as stream:
            stream.write(payload[-1:])
        drv.tick()
        self.assert_movement(drv, 1, 1)

    def test_hunt_pending_track_drag_is_not_a_new_action_prefix(self):
        st = initial()
        st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
        drv, path, _ = self.rig(st)
        self.feed(drv, path, captured('shortcut_track_first')[:1])
        self.assertIsNone(drv.seg.window)
        self.assertEqual(drv.window_stats['matched'], 0)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)
        self.assertIsNotNone(drv.es.pending_hunt)

    def test_already_applied_track_echo_is_not_an_unpaid_prefix(self):
        st = initial()
        st.fellowship.progress = 1
        drv, path, _ = self.rig(st)
        self.feed(drv, path, captured('shortcut_track_first')[:1])
        self.assertIsNone(drv.seg.window)
        self.assertEqual(drv.es.fellowship.progress, 1)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)

    def test_eomer_recruit_counts_card_and_die_for_every_partition(self):
        lines = captured('eomer_recruit')
        for mask in range(8):
            with self.subTest(partition=mask):
                st = initial(ActionDieFace.MUSTER)
                cid = 'FREE_PEOPLES_Strategy_\u00c9omer_son_of_\u00c9omund'
                self.assertIn(cid, st.fp_strat_deck)
                st.fp_strat_deck.remove(cid)
                st.hands[Side.FREE_PEOPLES].hand[:] = [cid]
                drv, path, _ = self.rig(st)
                out = []
                for group in batches(lines, mask):
                    out.extend(self.feed(drv, path, group))
                self.assert_settled(drv)
                self.assertEqual(army_counts(drv.es, "Helm's Deep"), Counter({
                    ('ROHAN', 'REGULAR'): 1, ('ROHAN', 'ELITE'): 1,
                    ('ROHAN', 'LEADER'): 1}))
                self.assertEqual(drv.es.hands[Side.FREE_PEOPLES].hand, [])
                self.assertTrue(drv.es.dice[Side.FREE_PEOPLES][0].used)
                self.assertEqual(drv.echo_ledger, [])
                self.assertFalse(any(i[0] == 'chat' and 'owed' in str(i[2])
                                     for i in out if len(i) > 2))

    def test_two_real_army_drags_spend_one_hybrid_die(self):
        lines = captured('two_army_move')
        for mask in range(8):
            with self.subTest(partition=mask):
                drv, path, _ = self.rig(initial(ActionDieFace.MUSTER_ARMY))
                for group in batches(lines, mask):
                    self.feed(drv, path, group)
                self.assert_settled(drv)
                self.assertEqual(army_counts(drv.es, 'Carrock'), Counter())
                self.assertEqual(army_counts(drv.es, 'Edoras'), Counter())
                self.assertEqual(army_counts(drv.es, 'Old Forest Road'),
                                 Counter({('THE_NORTH', 'REGULAR'): 1}))
                self.assertEqual(army_counts(drv.es, 'Westemnet'), Counter({
                    ('ROHAN', 'REGULAR'): 1, ('ROHAN', 'ELITE'): 1}))
                self.assertTrue(drv.es.dice[Side.FREE_PEOPLES][0].used)
                self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][1].used)

    def test_illegal_shortcut_cannot_spend_muster_as_character(self):
        drv, path, _ = self.rig(initial(ActionDieFace.MUSTER))
        lines = [line.replace('Die (C)', 'Die (M)')
                 for line in captured('shortcut_track_first')]
        out = self.feed(drv, path, lines)
        self.assertEqual(drv.es.fellowship.progress, 0)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][0].used)
        self.assertEqual(drv.window_stats['matched'], 0)
        self.assertTrue(any(i[0] == 'chat' for i in out))

    def test_second_human_move_before_ai_reply_is_refused(self):
        drv, path, _ = self.rig(initial())
        self.feed(drv, path, captured('shortcut_track_first'))
        out = self.feed(drv, path, captured('shortcut_die_first_second_move'))
        self.assertEqual(drv.es.fellowship.progress, 1)
        self.assertFalse(drv.es.dice[Side.FREE_PEOPLES][1].used)
        self.assertEqual(drv.window_stats['matched'], 1)
        self.assertTrue(any(i[0] == 'chat' and 'not your action turn' in str(i[2])
                            for i in out if len(i) > 2))


if __name__ == '__main__':
    print('Production target:', TARGET_ROOT)
    unittest.main(verbosity=2)
