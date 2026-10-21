"""Ordinary reveal placement from the held 2026-10-11 native Flocks game."""
from pathlib import Path
import pickle
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import ai
import game_logic as gl
import game_state as gs
from live import client_board, client_protocol, live_parser
from live.driver import LiveDriver, PhaseWindow
from live.randomness import ClientRandomness
from live.live_server import LiveServer

CAPTURE = ROOT / 'audit/bridge_oct11/player/Flocks_humanFP_physical/reveal_place_failure'
FLIP = '$WOTR_AI_Shadow flips The Fellowship in Rivendell. Previously on Fellowship Track 1.'
MOVE = '$Wey moves The Fellowship from Rivendell to Fords of Bruinen.'


class NativeRevealPlacementTests(unittest.TestCase):
    def rig(self, scenario_runtime=False):
        folder = tempfile.TemporaryDirectory(prefix='native_reveal_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)
        feed = path / 'ai_feed.txt'
        feed.write_text('', encoding='utf8')
        dump_path = path / 'ai_client_board.txt'
        dump_path.write_bytes((CAPTURE / 'ai_client_board.txt').read_bytes())
        state = pickle.loads((CAPTURE / 'engine_state.pkl').read_bytes())
        if scenario_runtime:
            # The native target starts through ScenarioDriver's real prefix
            # lane. Its immutable replay module survives the live flip.
            scenario = SimpleNamespace(ai_side_name='SHADOW', name='captured checkpoint',
                                       cut=0, prefix=[])
            server = LiveServer(str(path), gs.Side.SHADOW, seed=43,
                                load_nets=False, scenario=scenario)
            self.runtime_server = server
            driver = server.driver
            driver.es = state
            for model in driver.hand_models.values():
                model.es = state
            self.assertTrue(driver.live)
            self.assertFalse(driver.spectator)
            self.assertEqual(driver._rr.__name__, 'rendered_replay')
        else:
            rand = ClientRandomness(human_side=gs.Side.FREE_PEOPLES, seed=43)
            rand.install()
            driver = LiveDriver(state, gs.Side.SHADOW, str(feed), randomness=rand,
                                dump_path=str(dump_path), nick='WOTR_AI_Shadow')
        driver.phase = 'actions'
        driver.last_actor = gs.Side.FREE_PEOPLES
        driver.view.refresh_dump()
        self.assertEqual(ai.describe_pending(state)['kind'], 'reveal_place')
        self.assertEqual(state.fellowship.progress, 1)
        self.assertFalse(state.fellowship.hidden)
        self.assertIn(FLIP, (CAPTURE / 'ai_feed.txt').read_text(encoding='utf8'))
        self.assertIn(MOVE, (CAPTURE / 'ai_feed.txt').read_text(encoding='utf8'))
        return driver, feed, dump_path

    def route(self, driver, feed, *lines):
        with feed.open('a', encoding='utf8') as stream:
            stream.write('\n'.join(lines) + '\n')
        return driver.tick()

    def alter_piece(self, path, pid, *, zone=None, marker=None, remove=False):
        lines = path.read_text(encoding='utf8').splitlines()
        replacement = None
        for index, line in enumerate(lines):
            cols = line.split('\t')
            if len(cols) < 5 or pid not in cols[2].split():
                continue
            tokens = [col.split() for col in cols[2:5]]
            at = tokens[0].index(pid)
            values = [row.pop(at) for row in tokens]
            lines[index] = '\t'.join(cols[:2] + [' '.join(row) for row in tokens])
            if not remove:
                replacement = '\t'.join(['900', zone or cols[1], pid,
                                          values[1], str(marker) if marker is not None else values[2]])
            break
        else:
            self.fail('Missing native piece ' + pid)
        if replacement is not None:
            lines.append(replacement)
        path.write_text('\n'.join(lines) + '\n', encoding='utf8')

    def test_actual_own_reveal_then_human_placement_commits_once(self):
        driver, feed, _ = self.rig()
        self.route(driver, feed, FLIP)
        self.assertEqual(driver.es.fellowship.progress, 1)
        output = self.route(driver, feed, MOVE)
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen', output)
        self.assertEqual(driver.es.fellowship.progress, 0)
        self.assertFalse(driver.es.fellowship.hidden)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.pending_buf)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(driver.commits[-1][1], [live_parser.parse(MOVE)['norm']])
        self.assertFalse(any(k == 'progress_sweep_emitted' for k, _ in driver.findings))
        self.assertFalse(any(item[0] == 'rng_ask' for item in output))
        driver.tick()
        self.assertEqual(driver.n_commits_total, 1)

    def test_missing_or_wrong_reset_evidence_holds_original_legal_budget(self):
        for defect in ('no own echo', 'wrong origin', 'wrong previous track',
                       'other actor', 'missing progress', 'wrong progress',
                       'missing face', 'hidden face'):
            with self.subTest(defect=defect):
                driver, feed, dump = self.rig()
                flip = FLIP
                if defect == 'wrong origin':
                    flip = FLIP.replace('Rivendell', 'Trollshaws')
                elif defect == 'wrong previous track':
                    flip = FLIP.replace('Track 1', 'Track 2')
                elif defect == 'other actor':
                    flip = FLIP.replace('WOTR_AI_Shadow', 'WOTR_AI_Shadow_other')
                elif defect == 'missing progress':
                    self.alter_piece(dump, '248', remove=True)
                elif defect == 'wrong progress':
                    self.alter_piece(dump, '248', zone='<~Game.2015~>')
                elif defect == 'missing face':
                    self.alter_piece(dump, '247', marker='-')
                elif defect == 'hidden face':
                    self.alter_piece(dump, '247', marker=0)
                if defect != 'no own echo':
                    self.route(driver, feed, flip)
                before = pickle.dumps(driver.es)
                output = self.route(driver, feed, MOVE)
                self.assertEqual(pickle.dumps(driver.es), before)
                self.assertEqual(driver.es.fellowship.progress, 1)
                self.assertEqual(driver.n_commits_total, 0)
                self.assertIsNotNone(driver.pending_buf)
                self.assertFalse(any(item[0] == 'rng_ask' for item in output))

    def test_public_reset_never_invents_a_destination_or_extends_reach(self):
        for destination in (None, 'Moria'):
            with self.subTest(destination=destination):
                driver, feed, _ = self.rig()
                before = pickle.dumps(driver.es)
                self.route(driver, feed, FLIP)
                if destination is not None:
                    self.route(driver, feed, MOVE.replace('Fords of Bruinen', destination))
                self.assertEqual(pickle.dumps(driver.es), before)
                self.assertEqual(driver.n_commits_total, 0)
                self.assertEqual({a.params['region'] for a in ai.describe_pending(driver.es)['actions']},
                                 {'fords of bruinen', 'trollshaws'})

    def test_already_human_reset_or_reveal_is_not_counted_twice(self):
        cases = [
            (FLIP, '$Wey moves FSP from Fellowship Track 1 to Fellowship Track 0.', MOVE),
            (MOVE, '$Wey flips The Fellowship in Fords of Bruinen. Previously on Fellowship Track 1.'),
        ]
        for lines in cases:
            with self.subTest(lines=lines):
                driver, feed, _ = self.rig()
                output = self.route(driver, feed, *lines)
                self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen', output)
                self.assertEqual(driver.es.fellowship.progress, 0)
                self.assertEqual(driver.n_commits_total, 1)
                self.assertIsNone(driver.pending_buf)
                self.assertFalse(any(k == 'progress_sweep_emitted' for k, _ in driver.findings))

    def test_stale_own_echo_cannot_pay_a_later_reveal(self):
        driver, feed, _ = self.rig()
        self.route(driver, feed, FLIP)
        driver.es.turn += 1
        before = pickle.dumps(driver.es)
        self.route(driver, feed, MOVE)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(driver.n_commits_total, 0)

    def test_captured_held_placement_recovers_through_real_resync_once(self):
        driver, feed, _ = self.rig(scenario_runtime=True)
        # Reconstruct only the captured transport checkpoint: all native
        # paragraphs read, one original human destination still unmatched.
        # No own-echo witness is installed by this older held driver.
        feed.write_bytes((CAPTURE / 'ai_feed.txt').read_bytes())
        Path(driver.sidecar.path).write_bytes((CAPTURE / 'ai_feed.txt.meta.jsonl').read_bytes())
        driver.sidecar.poll()
        driver.feed.poll()
        driver._feed_seq = len(driver.feed.lines)
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [live_parser.parse(MOVE)]
        before_dice = [(die.current, die.used) for side in gs.Side for die in driver.es.dice[side]]
        before_tiles = repr(driver.es.hunt_discard)
        items, report = driver.reset_and_resync()
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen', (items, report))
        self.assertEqual(driver.es.fellowship.progress, 0)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.pending_buf)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual([(die.current, die.used) for side in gs.Side for die in driver.es.dice[side]], before_dice)
        self.assertEqual(repr(driver.es.hunt_discard), before_tiles)
        self.assertFalse(any(item[0] == 'rng_ask' for item in items))
        driver.reset_and_resync()
        self.assertEqual(driver.n_commits_total, 1)

    def test_real_scenario_recovery_sweep_failure_rolls_back_then_retries(self):
        driver, feed, _ = self.rig(scenario_runtime=True)
        feed.write_bytes((CAPTURE / 'ai_feed.txt').read_bytes())
        driver.feed.poll()
        driver._feed_seq = len(driver.feed.lines)
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [live_parser.parse(MOVE)]
        original_state, original_buffer, parser_module = driver.es, driver.pending_buf, driver._rr
        before_state = pickle.dumps(original_state)
        before_random = pickle.dumps(vars(driver.rand))
        hand_models = dict(driver.hand_models)
        before_hands = {side: dict(vars(model)) for side, model in hand_models.items()}
        def fail_after_apply(*args, **kwargs):
            driver.hands._mint('character')
            raise ValueError('native placement renderer failed')
        with patch.object(driver, 'post_commit_sweep', side_effect=fail_after_apply):
            items, report = driver.reset_and_resync()
        self.assertTrue(any('native placement renderer failed' in str(item) for item in items), (items, report))
        self.assertIs(driver.es, original_state)
        self.assertEqual(pickle.dumps(driver.es), before_state)
        self.assertIs(driver._rr, parser_module)
        self.assertIs(driver.pending_buf, original_buffer)
        self.assertEqual(driver.pending_buf.records, [live_parser.parse(MOVE)])
        self.assertEqual(pickle.dumps(vars(driver.rand)), before_random)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertEqual(driver.commits, [])
        self.assertEqual(driver._feed_seq, len(driver.feed.lines))
        for side, model in hand_models.items():
            self.assertIs(driver.hand_models[side], model)
            self.assertEqual(vars(model), before_hands[side])
        driver.reset_and_resync()
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen')
        self.assertEqual(driver.n_commits_total, 1)

    def test_real_panel_reset_control_recovers_without_starting_ai_action(self):
        from live.transport import FileBridge
        driver, feed, _ = self.rig(scenario_runtime=True)
        server = self.runtime_server
        feed.write_bytes((CAPTURE / 'ai_feed.txt').read_bytes())
        driver.feed.poll()
        driver._feed_seq = len(driver.feed.lines)
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [live_parser.parse(MOVE)]
        before_tiles = repr(driver.es.hunt_discard)
        before_dice = [(die.current, die.used) for side in gs.Side for die in driver.es.dice[side]]
        server.on_control({'reset_resync': 2000000000.0, 'enforce_human': False})
        # Observe the real server's scheduling boundary, while the driver,
        # legal matcher, census, transaction and control path remain real.
        with patch.object(server, '_run_open_step', wraps=server._run_open_step) as run_step:
            output = server.handle({'lines': ()})
            run_step.assert_not_called()
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen', output)
        self.assertEqual(driver.es.fellowship.progress, 0)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertIsNone(driver.pending_buf)
        self.assertIs(driver.last_actor, gs.Side.FREE_PEOPLES)
        self.assertEqual(repr(driver.es.hunt_discard), before_tiles)
        self.assertEqual([(die.current, die.used) for side in gs.Side for die in driver.es.dice[side]], before_dice)
        self.assertFalse(any(line.startswith('!CALL') for line in output), output)
        bridge = FileBridge(str(feed.parent), 'captured-reveal-resync', status_extra=server.status_menu)
        bridge.write_status(state='idle')
        status = json.loads((feed.parent / 'ai_bridge_status.json').read_text())
        self.assertTrue(status['recovery']['complete'], status['recovery'])

    def test_actual_button_checkpoint_never_reemits_already_zero_progress(self):
        physical = ROOT / 'audit/bridge_oct11/reveal_resync_checkpoint/fresh_physical/button_before'
        driver, feed, dump = self.rig(scenario_runtime=True)
        server = self.runtime_server
        driver.es = pickle.loads((physical / 'ai_live_state.pkl').read_bytes())
        for model in driver.hand_models.values():
            model.es = driver.es
        dump.write_bytes((physical / 'ai_client_board.txt').read_bytes())
        feed.write_bytes((physical / 'ai_feed.txt').read_bytes())
        Path(driver.sidecar.path).write_bytes((physical / 'ai_feed.txt.meta.jsonl').read_bytes())
        driver.sidecar.poll()
        driver.feed.poll()
        driver._feed_seq = len(driver.feed.lines)
        driver.view.refresh_dump()
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [live_parser.parse(MOVE)]
        driver._native_reveal_reset = None
        self.assertEqual(client_board.read_board(str(dump))['248'], 'Track 0')
        self.assertEqual(driver.es.fellowship.progress, 1)
        before_tiles = repr(driver.es.hunt_discard)
        before_cards = pickle.dumps(driver.es.discard_face_up)
        server.on_control({'reset_resync': 2000000010.0, 'enforce_human': False})
        with patch.object(server, '_run_open_step', wraps=server._run_open_step) as run_step:
            output = server.handle({'lines': ()})
            run_step.assert_not_called()
        self.assertEqual(driver.n_commits_total, 1, output)
        self.assertEqual(gl.fellowship_region(driver.es), 'fords of bruinen')
        self.assertEqual(driver.es.fellowship.progress, 0)
        self.assertIsNone(driver.pending_buf)
        self.assertIsNone(driver._native_reveal_reset)
        self.assertFalse(any(key == 'progress_sweep_emitted' for key, _ in driver.findings), output)
        self.assertFalse(any('248' in line and '<~Controls.790~>' in line for line in output), output)
        self.assertFalse(any(line.startswith('!CALL') for line in output), output)
        self.assertNotIn('248', getattr(driver, '_track_owed', {}))
        self.assertEqual(repr(driver.es.hunt_discard), before_tiles)
        self.assertEqual(pickle.dumps(driver.es.discard_face_up), before_cards)
        server.on_control({'reset_resync': 2000000011.0, 'enforce_human': False})
        with patch.object(server, '_run_open_step', wraps=server._run_open_step) as run_step:
            repeated = server.handle({'lines': ()})
            run_step.assert_not_called()
        self.assertEqual(driver.n_commits_total, 1, repeated)
        self.assertFalse(any('248' in line and '<~Controls.790~>' in line for line in repeated), repeated)
        self.assertFalse(any(line.startswith('!CALL') for line in repeated), repeated)


if __name__ == '__main__':
    unittest.main()
