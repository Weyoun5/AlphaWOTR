"""Literal Elven Ring acceptance: RB21 and Almanac31-32 timing.

An Elven Ring changes one selected unused die; it is not a generic payment
waiver. The player may spend another die or pass while keeping that change.
Human fixtures below are literal client records, never emitter round trips.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get('BRIDGE_TEST_PROJECT', ROOT))
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'wotr_bridge'), str(PROJECT)]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import ai
import board
import game_logic as gl
import game_state as gs
from live.driver import LiveDriver
from live.randomness import ClientRandomness
from live.sidecar import SidecarReader

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
F = gs.ActionDieFace
RING = '$Wey moves Nenya from Elven Ring 2 to Dark Ring 2.'
CHANGE = '$Wey changes Free Action Die (C) to H.'
ARMY = [
    '$Wey moves Free Action Die (H) from FP Dice to Current Action.',
    '$Wey moves Gondor Regular from Minas Tirith to Osgiliath.',
    '$Wey kills Free Action Die (H) in Current Action.',
]


def before_ring(fp_faces=(F.CHARACTER, F.MUSTER), sh_faces=(F.ARMY, F.CHARACTER, F.MUSTER)):
    state = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(state)
    state.dice = {}
    for side, faces in ((FP, fp_faces), (SH, sh_faces)):
        state.dice[side] = []
        for face in faces:
            die = gs.ActionDie([face], side)
            die.current = face
            state.dice[side].append(die)
    return state


def face_rows(state, side=FP):
    return [(d.current, d.used) for d in state.dice[side]]


class ElvenRingAcceptance(unittest.TestCase):
    diagnostics = []

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get('WOTR_AUDIT_RING_LOG')
        if path:
            Path(path).write_text(json.dumps(cls.diagnostics, indent=2, default=str) + '\n', encoding='utf-8')

    def rig(self, state=None, human=FP):
        temp = tempfile.TemporaryDirectory(prefix='wotr-elven-ring-')
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        rand = ClientRandomness(seed=31)
        rand.install()
        driver = LiveDriver(state or before_ring(), SH if human == FP else FP,
                            str(path), randomness=rand, nick='WOTR_AI')
        driver.phase = 'actions'
        driver.sidecar = SidecarReader(str(path) + '.meta.jsonl')
        driver.last_actor = driver.ai_side
        driver.turn_draws = {(s, deck): 1 for s in (FP, SH) for deck in ('character', 'strategy')}
        return driver, path

    def feed(self, driver, path, *lines, pids=None):
        if lines:
            if pids is not None:
                seq = len(path.read_text(encoding='utf-8').splitlines()) + 1
                with Path(str(path) + '.meta.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'seq': seq, 'pids': pids, 'raw_new': True}) + '\n')
            with path.open('a', encoding='utf-8') as stream:
                stream.write('\n'.join(lines) + '\n')
        driver.sidecar.poll()
        out = driver.tick()
        self.diagnostics.append(copy.deepcopy(dict(test=self.id(), lines=lines, output=out,
            fp_dice=face_rows(driver.es), sh_dice=face_rows(driver.es, SH),
            rings=(driver.es.fp_elven_rings, driver.es.sh_elven_rings),
            commits=driver.n_commits_total, alarms=driver.alarms,
            findings=driver.findings, uncommitted=driver.uncommitted)))
        return out

    def test_engine_changes_only_target_and_preserves_unused_status(self):
        state = before_ring()
        gl.use_elven_ring(state, FP, 0, F.MUSTER_ARMY)
        self.assertEqual(face_rows(state), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertEqual((state.fp_elven_rings, state.sh_elven_rings), (2, 1))
        self.assertTrue(state.fp_ring_used_this_turn)
        self.assertFalse(gl.ring_waiver_active(state, FP))

    def test_will_change_survives_spending_another_will_die(self):
        state = before_ring((F.WILL_OF_WEST, F.WILL_OF_WEST))
        changed = ai.apply_action(state, ai.Action(-1, 'elven_ring', {
            'side': 'fp', 'target_die_idx': 0, 'new_face': F.MUSTER_ARMY.value}))
        self.assertIs(ai._next_acting_side(changed, FP), FP)
        paid = ai.apply_action(changed, ai.Action(1, 'advance_political', {'nation': gs.Nation.GONDOR}))
        self.assertEqual(face_rows(paid), [(F.MUSTER_ARMY, False), (F.WILL_OF_WEST, True)])
        self.assertEqual((paid.fp_elven_rings, paid.sh_elven_rings), (2, 1))

    def test_change_survives_passing_without_spending_a_die(self):
        state = before_ring()
        changed = ai.apply_action(state, ai.Action(-1, 'elven_ring', {
            'side': 'fp', 'target_die_idx': 0, 'new_face': F.MUSTER_ARMY.value}))
        self.assertIn('pass', {a.action_type for a in ai.get_legal_actions(changed, FP)})
        passed = ai.apply_action(changed, ai.Action(-1, 'pass', {}))
        self.assertEqual(face_rows(passed), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertIs(ai._next_acting_side(passed, FP), SH)

    def test_second_use_and_forbidden_faces_are_atomic(self):
        for side, faces, target in [(FP, (F.CHARACTER,), F.WILL_OF_WEST),
                                    (FP, (F.CHARACTER,), F.EYE),
                                    (SH, (F.EYE,), F.CHARACTER)]:
            with self.subTest(side=side, target=target):
                state = before_ring(faces if side == FP else (F.CHARACTER,),
                                    faces if side == SH else (F.MUSTER,))
                state.sh_elven_rings = 1
                prior = (face_rows(state, side), state.fp_elven_rings, state.sh_elven_rings)
                with self.assertRaises(RuntimeError):
                    gl.use_elven_ring(state, side, 0, target)
                self.assertEqual((face_rows(state, side), state.fp_elven_rings, state.sh_elven_rings), prior)
        state = before_ring()
        gl.use_elven_ring(state, FP, 0, F.MUSTER_ARMY)
        prior = face_rows(state)
        with self.assertRaises(RuntimeError):
            gl.use_elven_ring(state, FP, 1, F.CHARACTER)
        self.assertEqual(face_rows(state), prior)
        self.assertEqual((state.fp_elven_rings, state.sh_elven_rings), (2, 1))

    def test_bare_counter_waits_without_inventing_face_and_can_be_undone(self):
        driver, path = self.rig()
        output = self.feed(driver, path, RING)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (3, 0))
        self.assertTrue(any(i[0] == 'chat' and 'die' in str(i).lower() and 'action' in str(i).lower() for i in output))
        self.assertIsNone(driver.ai_action_dp())
        self.feed(driver, path, '$Wey undoes last action : ' + RING.lstrip('$'))
        self.assertEqual(driver.n_commits_total, 0)
        self.feed(driver, path, RING, CHANGE)
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(driver.alarms, [])

    def test_literal_ring_and_changed_die_complete_one_army_action(self):
        for split in (False, True):
            with self.subTest(split=split):
                driver, path = self.rig()
                lines = [RING, CHANGE] + ARMY
                for part in ([[line] for line in lines] if split else [lines]):
                    self.feed(driver, path, *part)
                mt = next(a for key, a in driver.es.armies.items() if board.sanitize(key) == board.sanitize('Minas Tirith'))
                osg = next(a for key, a in driver.es.armies.items() if board.sanitize(key) == board.sanitize('Osgiliath'))
                self.assertEqual(sum(u.kind == gs.UnitKind.REGULAR for u in mt.units), 2)
                self.assertEqual(sum(u.kind == gs.UnitKind.REGULAR for u in osg.units), 3)
                self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, True), (F.MUSTER, False)])
                self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
                self.assertEqual(driver.n_commits_total, 2)
                self.assertEqual(driver.alarms, [])
                self.assertEqual(driver.uncommitted, [])
                self.assertIsNone(driver.seg.window)
                self.assertIsNone(driver.pending_buf)

    def test_same_face_dice_use_sidecar_target_identity(self):
        driver, path = self.rig(before_ring((F.CHARACTER, F.CHARACTER)))
        self.feed(driver, path, RING)
        self.feed(driver, path, CHANGE, pids=['211'])
        self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.MUSTER_ARMY, False)])
        self.assertEqual(driver.n_commits_total, 1)

    def test_out_of_turn_ring_remains_explicitly_refused(self):
        driver, path = self.rig()
        driver.last_actor = FP
        output = self.feed(driver, path, RING)
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (3, 0))
        self.assertTrue(any(i[0] == 'chat' and i[1] == 'refusal' for i in output))
        self.assertEqual(sum(i[0] == 'line' and 'Controls.819' in str(i[1]) for i in output), 1)

    def test_partial_counter_cannot_be_bypassed_with_pass(self):
        driver, path = self.rig()
        self.feed(driver, path, RING)
        self.feed(driver, path, '<Wey> pass')
        self.assertEqual(driver.n_commits_total, 0)
        self.assertIs(driver.last_actor, SH)
        self.assertIsNone(driver.ai_action_dp())
        self.feed(driver, path, CHANGE)
        self.feed(driver, path, '<Wey> pass')
        self.assertEqual(driver.n_commits_total, 2)
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertIsNotNone(driver.ai_action_dp())

    def test_human_second_ring_and_shadow_with_no_ring_are_refused(self):
        driver, path = self.rig()
        self.feed(driver, path, RING, CHANGE)
        output = self.feed(driver, path, '$Wey moves Vilya from Elven Ring 1 to Dark Ring 1.')
        self.assertTrue(any(i[0] == 'chat' and i[1] == 'refusal' for i in output))
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
        shadow, other_path = self.rig(human=SH)
        output = self.feed(shadow, other_path, '$Wey kills Nenya in Dark Ring 2.')
        self.assertTrue(any(i[0] == 'chat' and i[1] == 'refusal' for i in output))
        self.assertEqual(shadow.n_commits_total, 0)
        self.assertEqual((shadow.es.fp_elven_rings, shadow.es.sh_elven_rings), (3, 0))

    def test_changed_face_before_counter_commits_only_when_payment_arrives(self):
        driver, path = self.rig()
        self.feed(driver, path, CHANGE)
        self.assertEqual(driver.n_commits_total, 0)
        self.feed(driver, path, RING)
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertEqual(driver.n_commits_total, 1)
        # ay_0 dispatch snapshots moves, but not 'changes'. Undoing the
        # later payment leaves the earlier physical face as a held prefix.
        self.feed(driver, path, '$Wey undoes last action : ' + RING.lstrip('$'))
        self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.MUSTER, False)])
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (3, 0))
        self.assertIsNone(driver.ai_action_dp())
        self.feed(driver, path, RING)
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertEqual(driver.alarms, [])

    def test_staged_die_then_ring_face_and_paid_army_move(self):
        driver, path = self.rig()
        self.feed(driver, path,
            '$Wey moves Free Action Die (C) from FP Dice to Current Action.',
            RING, CHANGE, *ARMY[1:])
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, True), (F.MUSTER, False)])
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
        self.assertEqual(driver.alarms, [])
        self.assertEqual(driver.uncommitted, [])
        self.assertIsNone(driver.pending_buf)

    def test_counter_then_staged_die_and_change_keeps_no_orphan_ring(self):
        driver, path = self.rig()
        self.feed(driver, path, RING,
            '$Wey moves Free Action Die (C) from FP Dice to Current Action.',
            CHANGE, *ARMY[1:])
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, True), (F.MUSTER, False)])
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
        self.assertFalse(getattr(driver, '_ring_gesture_records', None))
        self.assertEqual(driver.alarms, [])
        self.assertEqual(driver.uncommitted, [])

    def test_shadow_disposes_actual_held_ring_and_changes_unused_die(self):
        for kill in (False, True):
            with self.subTest(kill=kill):
                state = before_ring()
                state.fp_elven_rings, state.sh_elven_rings = 2, 1
                driver, path = self.rig(state, human=SH)
                self.feed(driver, path,
                    '$Wey kills Nenya in Dark Ring 2.' if kill else
                    '$Wey moves Nenya from Dark Ring 2 to SA Casualties.',
                    '$Wey changes Shadow Action Die (A) to H.')
                self.assertEqual(face_rows(driver.es, SH), [(F.MUSTER_ARMY, False), (F.CHARACTER, False), (F.MUSTER, False)])
                self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 0))
                self.assertEqual(driver.n_commits_total, 1)
                self.assertEqual(driver.alarms, [])

    def test_invalid_face_then_native_undo_keeps_partial_payment_recoverable(self):
        driver, path = self.rig()
        self.feed(driver, path, RING)
        wrong = '$Wey changes Free Action Die (C) to W.'
        output = self.feed(driver, path, wrong)
        self.assertTrue(any(i[0] == 'chat' and i[1] == 'refusal' for i in output))
        self.assertEqual(driver.n_commits_total, 0)
        self.assertEqual(sum(i[0] == 'line' and 'Controls.819' in str(i[1]) for i in output), 1)
        self.feed(driver, path, '$Wey undoes last action : ' + RING.lstrip('$'))
        self.feed(driver, path, RING, CHANGE)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        self.assertEqual(driver.alarms, [])

    def test_ambiguous_old_face_without_pid_does_not_choose_first_die(self):
        driver, path = self.rig(before_ring((F.CHARACTER, F.CHARACTER)))
        self.feed(driver, path, RING, CHANGE)
        self.assertEqual(driver.n_commits_total, 0)
        self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.CHARACTER, False)])
        self.assertIsNone(driver.ai_action_dp())
        self.feed(driver, path, '$Wey undoes last action : ' + RING.lstrip('$'))
        self.feed(driver, path, RING)
        self.feed(driver, path, CHANGE, pids=['211'])
        self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.MUSTER_ARMY, False)])
        self.assertEqual(driver.n_commits_total, 1)

    def test_legacy_active_waiver_requires_replay_and_fp_army_target_is_not_offered(self):
        state = before_ring()
        self.assertNotIn(F.ARMY.value, {a.params['new_face'] for a in ai.get_legal_actions(state, FP) if a.action_type == 'elven_ring'})
        with self.assertRaises(RuntimeError):
            gl.use_elven_ring(state, FP, 0, F.ARMY)
        self.assertEqual((state.fp_elven_rings, state.sh_elven_rings), (3, 0))
        state.fp_ring_waiver = True
        with self.assertRaisesRegex(RuntimeError, 'resync or replay'):
            ai.get_legal_actions(state, FP)

    def test_shadow_eye_changes_face_banks_exactly_once_and_uses_ring_once(self):
        state = before_ring()
        state.sh_elven_rings = 1
        gl.use_elven_ring(state, SH, 0, F.EYE)
        self.assertEqual(face_rows(state, SH), [(F.EYE, True), (F.CHARACTER, False), (F.MUSTER, False)])
        self.assertEqual(state.hunt_box, [state.dice[SH][0]])
        self.assertEqual(state.sh_elven_rings, 0)
        with self.assertRaises(RuntimeError):
            gl.use_elven_ring(state, SH, 1, F.EYE)
        self.assertEqual(len(state.hunt_box), 1)

    def test_staged_same_face_pid_controls_both_ring_target_and_die_payment(self):
        driver, path = self.rig(before_ring((F.CHARACTER, F.CHARACTER)))
        self.feed(driver, path, '$Wey moves Free Action Die (C) from FP Dice to Current Action.', pids=['211'])
        self.feed(driver, path, RING, pids=['250'])
        self.feed(driver, path, CHANGE, pids=['211'])
        self.feed(driver, path, *ARMY[1:])
        self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.MUSTER_ARMY, True)])
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
        self.assertEqual(driver.alarms, [])

    def test_full_window_native_undo_has_no_separate_change_snapshot(self):
        driver, path = self.rig()
        stage = '$Wey moves Free Action Die (C) from FP Dice to Current Action.'
        self.feed(driver, path, stage, RING, CHANGE, *ARMY[1:])
        for line in (ARMY[2], ARMY[1], RING, stage):
            self.feed(driver, path, '$Wey undoes last action : ' + line.lstrip('$'))
        self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.MUSTER, False)])
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (3, 0))
        self.assertEqual(driver.alarms, [])

    def test_returning_staged_die_to_pool_retains_paid_ring_face(self):
        for prefix in (False, True):
            with self.subTest(prefix=prefix):
                driver, path = self.rig()
                stage = '$Wey moves Free Action Die (C) from FP Dice to Current Action.'
                returned = '$Wey moves Free Action Die (H) from Current Action to FP Dice.'
                lines = [RING, stage, CHANGE, returned] if prefix else [stage, RING, CHANGE, returned]
                self.feed(driver, path, *lines)
                self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
                self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
                self.assertEqual(driver.n_commits_total, 1)
                self.assertIsNone(driver.ai_action_dp())
                self.assertEqual(driver.alarms, [])
                for line in reversed([line for line in lines if line != CHANGE]):
                    self.feed(driver, path, '$Wey undoes last action : ' + line.lstrip('$'))
                self.assertEqual(face_rows(driver.es), [(F.CHARACTER, False), (F.MUSTER, False)])
                self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (3, 0))
                self.assertEqual(driver.alarms, [])

    def test_shadow_staged_ring_disposal_and_return_keeps_changed_unused_die(self):
        state = before_ring()
        state.fp_elven_rings, state.sh_elven_rings = 2, 1
        driver, path = self.rig(state, human=SH)
        self.feed(driver, path,
            '$Wey moves Shadow Action Die (A) from SA Dice to Current Action.',
            '$Wey kills Nenya in Dark Ring 2.',
            '$Wey changes Shadow Action Die (A) to H.',
            '$Wey moves Shadow Action Die (H) from Current Action to SA Dice.')
        self.assertEqual(face_rows(driver.es, SH), [(F.MUSTER_ARMY, False), (F.CHARACTER, False), (F.MUSTER, False)])
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 0))
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(driver.alarms, [])

    def test_actual_ai_entry_emits_change_and_returned_echoes_never_spend_again(self):
        import emitter
        import log_replay
        from live.live_server import LiveServer
        temporary = tempfile.TemporaryDirectory(prefix='wotr-ai-ring-')
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        server = LiveServer(temporary.name, FP, load_nets=False, seed=19)
        driver = server.driver
        driver.es = before_ring()
        driver.hands.es = driver.es
        driver.phase = 'actions'
        driver.last_actor = SH
        positions = json.loads((ROOT / 'wotr_bridge/initial_positions.json').read_text())
        driver.view.piece_region = {pid: log_replay.decode_location(zone) for pid, zone in positions.items()}
        for pid in ('210', '211'):
            driver.view.piece_region[pid] = 'FP Dice'
        selected = ai.Action(-1, 'elven_ring', {'side': 'fp', 'target_die_idx': 0,
            'new_face': F.MUSTER_ARMY.value, 'old_face': F.CHARACTER.value})
        output = server._ai_decide({'side': FP, 'kind': 'action', 'actions': [selected]})
        changes = [emitter.classify_line(emitter.strip_tail(line)) for line in output]
        self.assertTrue(any(op == 'die_change' and fields['die'] == '210' and fields['face'] == 'H' for op, fields in changes))
        self.assertEqual(face_rows(driver.es), [(F.MUSTER_ARMY, False), (F.MUSTER, False)])
        count = driver.n_commits_total
        with path.open('a', encoding='utf-8') as stream:
            stream.write('$%s moves Vilya from Vilya Slot to Used Ring 1.\n' % driver.nick)
            stream.write('$%s changes Free Action Die (C) to H.\n' % driver.nick)
        driver.tick()
        self.assertEqual(driver.n_commits_total, count)
        self.assertEqual((driver.es.fp_elven_rings, driver.es.sh_elven_rings), (2, 1))
        self.assertEqual(driver.alarms, [])
        self.assertIsNotNone(driver.ai_action_dp(), 'A ring does not hand the turn to the opponent')

    def test_explicit_incompatible_checkpoint_observation_version_is_rejected(self):
        import torch
        temporary = tempfile.TemporaryDirectory(prefix='wotr-ring-contract-')
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'incompatible.pt'
        torch.save({'observation_contract': 'obsolete_count_only_v0',
                    'action_encoding_contract': __import__('trace_decisions').ACTION_ENCODING_CONTRACT,
                    'model_state': {}}, path)
        with self.assertRaisesRegex(ValueError, 'observation contract'):
            ai.load_value_net(str(path), 'fp')

    def test_ai_emits_real_ring_inventory_and_exact_target_face(self):
        import emitter
        import log_replay
        for side in (FP, SH):
            with self.subTest(side=side):
                state = before_ring()
                if side == SH:
                    state.fp_elven_rings, state.sh_elven_rings = 2, 1
                driver, path = self.rig(state, human=SH if side == FP else FP)
                positions = json.loads((ROOT / 'wotr_bridge/initial_positions.json').read_text())
                driver.view.piece_region = {pid: log_replay.decode_location(zone) for pid, zone in positions.items()}
                if side == SH:
                    driver.view.piece_region['250'] = 'Used Ring 2'
                ops = emitter.plan_elven_ring(driver.view, 'WOTR_AI', side.name, {
                    'target_die_idx': 0, 'new_face': F.MUSTER_ARMY.value})
                changes = [fields for op, fields in ops if op == 'die_change']
                self.assertEqual(changes, [{'nick': 'WOTR_AI', 'die': '210' if side == FP else '099', 'face': 'H'}])
                if side == FP:
                    ring, = [fields for op, fields in ops if op == 'marker_366']
                    self.assertEqual((ring['piece'], ring['frm'], ring['to']), ('249', 'Game.2008', 'Game.2011'))
                else:
                    disposal, = [fields for op, fields in ops if op == 'kill']
                    self.assertEqual(disposal['pieces'], ['250'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
