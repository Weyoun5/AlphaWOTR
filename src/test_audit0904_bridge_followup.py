"""Recorded Hunt replay plus explicitly labeled legal repair interactions.

fpgame25 is an automated agent-seat capture from the actual Java client,
not an expert human game. Its last drag is incomplete: tile 2 cannot be resolved as +1 with
all companions still present. Neither the old pickle nor a regenerated
engine snapshot supplies the expected answer. See the independent rule
oracle in audit/2026-09-04/bridge_pass/fpgame25_hunt_rules.md.
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import ai
import board
import game_logic as gl
import game_state as gs
from live.driver import LiveDriver
from live.randomness import ClientRandomness

if os.environ.get('WOTR_AUDIT_BRIDGE_BASELINE'):
    path = Path(os.environ['WOTR_AUDIT_BRIDGE_BASELINE'])
    spec = importlib.util.spec_from_file_location('live._audit_before_hunt', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    LiveDriver = module.LiveDriver

FIXTURE = ROOT / 'wotr_bridge/tests/fixtures/fpgame25_hunt_capture.json'
CAPTURE = json.loads(FIXTURE.read_text(encoding='utf-8'))
LINES = CAPTURE['lines']
PARTIAL = '$Wey moves Corruption from Fellowship Track 0 to Fellowship Track 1.'
COMPLETE = '$Wey moves Corruption from Fellowship Track 1 to Fellowship Track 2.'


def initial_action_phase():
    st = gs.setup_initial_game(board.load_board())
    gl.initialize_decks(st)
    # Exactly the allocated eye and face order in captured feed 14–18.
    gl.allocate_eyes(st, 1, indices=[0])
    faces = dict(zip('EMCPH', (gs.ActionDieFace.EYE, gs.ActionDieFace.MUSTER,
                             gs.ActionDieFace.CHARACTER, gs.ActionDieFace.EVENT,
                             gs.ActionDieFace.MUSTER_ARMY)))
    gl.adopt_rolled_faces(st,
        [(gs.Side.SHADOW, i, faces[f]) for i, f in enumerate('EMECPH', 1)] +
        [(gs.Side.FREE_PEOPLES, i, faces[f]) for i, f in enumerate('PPMC')])
    return st


class RecordedHuntRecoveryTests(unittest.TestCase):
    def rig(self, state=None):
        td = tempfile.TemporaryDirectory(prefix='wotr-hunt-replay-')
        self.addCleanup(td.cleanup)
        path = Path(td.name) / 'ai_feed.txt'
        path.write_text('', encoding='utf-8')
        rand = ClientRandomness(seed=7)
        rand.install()
        drv = LiveDriver(initial_action_phase() if state is None else state,
                         gs.Side.SHADOW, str(path), randomness=rand,
                         nick='WOTR_AI_Shadow')
        drv.phase = 'actions'
        return drv, path, rand

    @staticmethod
    def feed(drv, path, lines):
        with path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines) + '\n')
        return drv.tick()

    def replay_to_hunt(self, *, split=True):
        drv, path, rand = self.rig()
        # Replay the native identity evidence alongside the archived text.
        # Both P dice have the same face; the real client moved PID 210.
        metadata = [json.loads(line) for line in
                    FIXTURE.with_name('fpgame25_feed.meta.jsonl').read_text().splitlines()]
        native_roll = next(row for row in metadata if row['seq'] == 18)
        drv.view.die_id_map.update({pid: (gs.Side.FREE_PEOPLES, index)
                                   for index, pid in enumerate(native_roll['segs'][1])})
        excerpt = [dict(row, seq=row['seq'] - 18) for row in metadata
                   if 19 <= row['seq'] <= 31]
        Path(str(path) + '.meta.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in excerpt) + '\n', encoding='utf-8')
        from live.sidecar import SidecarReader
        drv.sidecar = SidecarReader(str(path) + '.meta.jsonl')
        for group in ([line] for line in LINES[:4]) if split else [LINES[:4]]:
            self.feed(drv, path, group)
        self.assertEqual(drv.es.fellowship.progress, 1)
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertIsNone(drv.es.pending_hunt)
        # The archive's actual AI reply is fixed input here. Apply that
        # recorded political action, then ingest its real echo. This test
        # does not claim to test neural move selection or Java injection.
        reply = ai.Action(2, 'advance_political', {'nation': gs.Nation.ISENGARD})
        drv.es = ai.apply_action(drv.es, reply)
        drv.last_actor = gs.Side.SHADOW
        for group in ([line] for line in LINES[4:-1]) if split else [LINES[4:-1]]:
            self.feed(drv, path, group)
        self.assertEqual(drv.es.fellowship.progress, 2)
        self.assertEqual(drv.es.pending_hunt, {'damage': 2, 'phase': 'hunt_sacrifice'})
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertEqual(len(drv.es.fellowship.companions), 7)
        self.assertEqual(drv.es.fellowship.guide.name, 'Gandalf the Grey')
        self.assertEqual(drv.es.political.boxes[gs.Nation.ISENGARD], 3)
        self.assertEqual((drv.es.fp_elven_rings, drv.es.sh_elven_rings), (2, 1))
        self.assertTrue(drv.es.dice[gs.Side.FREE_PEOPLES][0].used)
        self.assertTrue(drv.es.dice[gs.Side.FREE_PEOPLES][3].used)
        self.assertFalse(drv.es.dice[gs.Side.FREE_PEOPLES][1].used)
        self.assertFalse(drv.es.dice[gs.Side.FREE_PEOPLES][2].used)
        self.assertEqual(drv.alarms, [])
        self.assertIsNone(drv.deferred)
        self.assertEqual(drv.holds, [])
        self.assertFalse(any(rand.unconsumed().values()))
        return drv, path, rand

    def assert_settled_at_two(self, drv, rand):
        self.assertEqual(drv.es.fellowship.corruption, 2)
        self.assertEqual(drv.es.fellowship.progress, 2)
        self.assertEqual(len(drv.es.fellowship.companions), 7)
        self.assertIsNone(drv.es.pending_hunt)
        self.assertIsNone(drv.pending_buf)
        self.assertIsNone(drv.deferred)
        self.assertIsNone(drv.chat_deferred)
        self.assertEqual(drv.alarms, [])
        self.assertEqual(drv.holds, [])
        self.assertFalse(any(rand.unconsumed().values()))

    def test_archived_client_one_is_incomplete_and_exact_remainder_is_explained(self):
        raw = FIXTURE.with_name('fpgame25_feed.txt').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CAPTURE['sha256'])
        self.assertEqual(raw.decode('utf-8').splitlines()[18:32], LINES)
        self.assertEqual(LINES[-1], PARTIAL)
        # The unchanged client export independently locates marker 252
        # at track 1. This is evidence of the partial physical position,
        # not permission to replace the rules' two-point result with one.
        dump = FIXTURE.with_name('fpgame25_client_board.txt').read_text()
        track_one = next(line.split('\t') for line in dump.splitlines()
                         if '<~Game.2015~>' in line)
        self.assertIn('252', track_one[2].split())
        for split in (False, True):
            with self.subTest(split=split):
                drv, path, rand = self.replay_to_hunt(split=split)
                before = drv.n_commits_total
                out = self.feed(drv, path, [PARTIAL])
                text = ' '.join(str(i[2]) for i in out if i[0] == 'chat')
                self.assertIn('2 damage', text)
                self.assertIn('1 more corruption', text)
                self.assertIn('from 1 to 2', text)
                self.assertIn('say corruption', text)
                self.assertIn('undo the partial marker move first', text)
                self.assertIn('say guide or random', text)
                self.assertEqual(drv.es.fellowship.corruption, 0)
                self.assertEqual(drv.es.pending_hunt['damage'], 2)
                self.assertEqual(drv.n_commits_total, before)
                self.assertEqual(len(drv.es.fellowship.companions), 7)
                self.assertFalse(any(i[0] in ('line', 'ai_decide', 'rng_ask') for i in out))

    def test_legal_physical_completion_commits_once(self):
        drv, path, rand = self.replay_to_hunt()
        self.feed(drv, path, [PARTIAL])
        before = drv.n_commits_total
        out = self.feed(drv, path, [COMPLETE])
        self.assert_settled_at_two(drv, rand)
        self.assertEqual(drv.n_commits_total, before + 1)
        self.assertFalse(any(i[0] == 'line' and '252' in i[1] for i in out))
        for _ in range(3):
            drv.tick()
        self.assert_settled_at_two(drv, rand)
        self.assertEqual(drv.n_commits_total, before + 1)

    def test_chat_completion_moves_from_actual_partial_position_and_clears_prefix(self):
        drv, path, rand = self.replay_to_hunt()
        self.feed(drv, path, [PARTIAL])
        before = drv.n_commits_total
        out = self.feed(drv, path, ['<Wey> corruption'])
        self.assert_settled_at_two(drv, rand)
        self.assertEqual(drv.n_commits_total, before + 1)
        moves = [i[1] for i in out if i[0] == 'line' and '252' in i[1]]
        self.assertEqual(moves, [
            '$WOTR_AI_Shadow<~Scribe.141~>252 <~Scribe.142~><~Game.2015~>'
            '<~Scribe.143~><~Game.2016~><~Scribe.144~>'])
        self.feed(drv, path, [
            '$WOTR_AI_Shadow moves Corruption from Fellowship Track 1 to Fellowship Track 2.'])
        for _ in range(3):
            drv.tick()
        self.assert_settled_at_two(drv, rand)
        self.assertEqual(drv.n_commits_total, before + 1)
        self.assertFalse(any(k == 'pending_buffer_orphaned' for k, _ in drv.findings))
        # A later Hunt cannot consume this earlier one-point prefix.
        drv.es.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
        drv._poke_pending_buffer()
        self.assertEqual(drv.es.fellowship.corruption, 2)
        self.assertEqual(drv.es.pending_hunt['damage'], 2)
        self.assertEqual(drv.n_commits_total, before + 1)

    def test_complete_batch_does_not_print_a_stale_partial_prompt(self):
        drv, path, rand = self.replay_to_hunt()
        out = self.feed(drv, path, [PARTIAL, COMPLETE])
        self.assert_settled_at_two(drv, rand)
        self.assertFalse(any(i[0] == 'chat' and 'full track total' in i[2] for i in out))

    def test_idle_ticks_do_not_repeat_the_partial_prompt_or_choose_for_player(self):
        drv, path, _ = self.replay_to_hunt()
        self.feed(drv, path, [PARTIAL])
        before = drv.n_commits_total
        for _ in range(55):
            out = drv.tick()
            self.assertFalse(any(i[0] == 'chat' and 'full track total' in i[2] for i in out))
        self.assertEqual(drv.n_commits_total, before)
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertEqual(len(drv.es.fellowship.companions), 7)
        self.assertIsNone(drv._repair)
        self.assertFalse(any(k == 'pending_buffer_stall' for k, _ in drv.findings))

    def test_multi_step_partial_chat_uses_absolute_total_once(self):
        st = initial_action_phase()
        st.fellowship.corruption = 2
        st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
        drv, path, _ = self.rig(st)
        self.feed(drv, path, [
            '$Wey moves Corruption from Fellowship Track 2 to Fellowship Track 3.',
            '$Wey moves Corruption from Fellowship Track 3 to Fellowship Track 4.'])
        out = self.feed(drv, path, ['<Wey> corruption'])
        self.assertEqual(drv.es.fellowship.corruption, 5)
        self.assertIsNone(drv.pending_buf)
        self.assertEqual(drv.alarms, [])
        moves = [i[1] for i in out if i[0] == 'line' and '252' in i[1]]
        self.assertEqual(len(moves), 1)
        self.assertIn('<~Game.2018~><~Scribe.143~><~Game.2019~>', moves[0])

    def test_player_can_undo_partial_drag_without_resolving_the_hunt(self):
        drv, path, _ = self.replay_to_hunt()
        self.feed(drv, path, [PARTIAL])
        before = drv.n_commits_total
        self.feed(drv, path, ['$Wey undoes last action : ' + PARTIAL.lstrip('$')])
        self.assertFalse(drv.pending_buf and drv.pending_buf.records)
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertEqual(drv.es.pending_hunt['damage'], 2)
        self.assertEqual(drv.n_commits_total, before)
        self.assertEqual(drv.alarms, [])
        self.assertEqual(drv.holds, [])
        self.feed(drv, path, [
            '$Wey moves Corruption from Fellowship Track 0 to Fellowship Track 2.'])
        self.assertEqual(drv.es.fellowship.corruption, 2)
        self.assertIsNone(drv.es.pending_hunt)
        self.assertEqual(drv.n_commits_total, before + 1)
        self.assertEqual(drv.alarms, [])
        self.assertEqual(drv.holds, [])
        self.assertFalse(any(k == 'pending_buffer_orphaned' for k, _ in drv.findings))

    def test_valid_guide_casualty_plus_corruption_still_matches_in_either_order(self):
        # A separate legal interaction control: after the other
        # companions are gone, Boromir can guide and absorb two of three
        # damage. The remaining one is corruption. No random choice is
        # invented; the player explicitly removes the named guide.
        for corruption_first in (False, True):
            with self.subTest(corruption_first=corruption_first):
                st = initial_action_phase()
                st.fellowship.companions[:] = [c for c in st.fellowship.companions
                                               if c.name in ('Boromir', 'Meriadoc', 'Peregrin')]
                st.fellowship.guide = next(c for c in st.fellowship.companions if c.name == 'Boromir')
                st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
                drv, path, _ = self.rig(st)
                kill = '$Wey moves Boromir from Guide Of The Fellowship to FP Casualties.'
                lines = [PARTIAL, kill] if corruption_first else [kill, PARTIAL]
                # The two remaining Hobbits tie: explicitly choose one.
                lines.append('$Wey moves Meriadoc from Fellowship Box to Guide Of The Fellowship.')
                for line in lines:
                    self.feed(drv, path, [line])
                self.assertEqual(drv.es.fellowship.corruption, 1)
                self.assertNotIn('Boromir', [c.name for c in drv.es.fellowship.companions])
                self.assertEqual(drv.es.fellowship.guide.name, 'Meriadoc')
                self.assertIsNone(drv.es.pending_hunt)
                self.assertEqual(drv.alarms, [])

    def test_undo_then_explicit_guide_does_not_apply_partial_corruption(self):
        drv, path, _ = self.replay_to_hunt()
        self.feed(drv, path, [PARTIAL])
        self.feed(drv, path, ['$Wey undoes last action : ' + PARTIAL.lstrip('$')])
        before = drv.n_commits_total
        self.feed(drv, path, ['<Wey> guide'])
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertNotIn('Gandalf the Grey', [c.name for c in drv.es.fellowship.companions])
        self.assertEqual(drv.es.fellowship.guide.name, 'Strider')
        self.assertIsNone(drv.es.pending_hunt)
        self.assertIsNone(drv.pending_buf)
        self.assertEqual(drv.n_commits_total, before + 1)
        self.assertEqual(drv.alarms, [])
        self.assertFalse(any(k == 'pending_buffer_orphaned' for k, _ in drv.findings))

    def test_partial_corruption_does_not_invent_a_guide_casualty(self):
        # Without a subsequent guide tie, the old speculative matcher
        # silently killed Boromir from this corruption-only prefix.
        st = initial_action_phase()
        st.fellowship.companions[:] = [c for c in st.fellowship.companions
                                       if c.name in ('Boromir', 'Meriadoc')]
        st.fellowship.guide = next(c for c in st.fellowship.companions if c.name == 'Boromir')
        st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
        drv, path, _ = self.rig(st)
        before = drv.n_commits_total
        out = self.feed(drv, path, [PARTIAL])
        self.assertEqual(drv.es.fellowship.corruption, 0)
        self.assertEqual(drv.es.fellowship.guide.name, 'Boromir')
        self.assertEqual([c.name for c in drv.es.fellowship.companions], ['Boromir', 'Meriadoc'])
        self.assertEqual(drv.es.pending_hunt['damage'], 3)
        self.assertEqual(drv.n_commits_total, before)
        self.assertFalse(any(i[0] in ('rng_ask', 'line', 'ai_decide') for i in out))
        self.feed(drv, path, [
            '$Wey moves Boromir from Guide Of The Fellowship to FP Casualties.'])
        self.assertEqual(drv.es.fellowship.corruption, 1)
        self.assertEqual([c.name for c in drv.es.fellowship.companions], ['Meriadoc'])
        self.assertIsNone(drv.es.pending_hunt)

    def test_terminal_corruption_cap_uses_track_twelve(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                st = initial_action_phase()
                st.fellowship.corruption = 10
                st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
                drv, path, _ = self.rig(st)
                if partial:
                    out = self.feed(drv, path, [
                        '$Wey moves Corruption from Fellowship Track 10 to Fellowship Track 11.'])
                    text = ' '.join(i[2] for i in out if i[0] == 'chat')
                    self.assertIn('3 damage', text)
                    self.assertIn('from 11 to 12', text)
                    self.assertNotIn('to 13', text)
                    out = self.feed(drv, path, ['<Wey> corruption'])
                    moves = [i[1] for i in out if i[0] == 'line' and '252' in i[1]]
                    self.assertEqual(len(moves), 1)
                    self.assertIn('<~Game.2025~><~Scribe.143~><~Game.2026~>', moves[0])
                else:
                    self.feed(drv, path, [
                        '$Wey moves Corruption from Fellowship Track 10 to Fellowship Track 12.'])
                self.assertEqual(drv.es.fellowship.corruption, 12)
                self.assertIsNone(drv.es.pending_hunt)
                self.assertIsNone(drv.pending_buf)
                self.assertEqual(drv.alarms, [])
                self.assertEqual(drv.holds, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
