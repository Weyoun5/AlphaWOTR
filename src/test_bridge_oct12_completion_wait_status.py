"""Saved actual Rage Undo checkpoint: an open completion is a human wait."""
from pathlib import Path
import json
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import game_state as gs
from live import live_parser
from live.live_server import LiveServer
from live.segmenter import Window
import test_bridge_oct12_dead_men_prefix_live as dead

CAPTURE = ROOT/'audit/bridge_oct11/card_completion_targets/rage_completion_undo_r1'


class CompletionWaitStatus(unittest.TestCase):
    def captured(self):
        folder = tempfile.TemporaryDirectory(prefix='completion_wait_', dir=ROOT)
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)
        feed = path/'ai_feed.txt'
        feed.write_text('', encoding='utf8')
        saved = CAPTURE/'after_committed_undo'
        server = LiveServer(str(path), gs.Side.FREE_PEOPLES, load_nets=False, seed=42)
        driver = server.driver
        driver.es = pickle.loads((saved/'ai_live_state.pkl').read_bytes())
        for model in driver.hand_models.values():
            model.es = driver.es
        driver.phase = 'actions'
        driver.last_actor = gs.Side.SHADOW
        driver.enforce_human = True
        status = json.loads((saved/'ai_bridge_status.json').read_text())
        driver.n_commits_total = status['progress']['n_commits']
        for die in status['human_menu']['dice']:
            driver.view.bind_die(die['pid'], gs.Side.SHADOW, die['idx'])
        # Restore only the actual runtime evidence that an engine-only pickle
        # omits: the still-paid staged die/card and retained North movement.
        lines = (saved/'ai_feed.txt').read_text(encoding='utf8').splitlines()
        stage = live_parser.parse(next(line for line in lines
            if 'Shadow Action Die (M) from SA Dice to Current Action' in line))
        card = live_parser.parse(next(line for line in lines
            if 'Rage of the Dunlendings/Relentless Assault from Shadow Hand' in line))
        north = live_parser.parse(next(line for line in lines
            if 'moves Isengard Regular from North Dunland to Gap of Rohan.' in line))
        driver.seg.window = Window(*stage['die_stage'], stage)
        driver.seg.window.records.extend((card, north))
        driver._staged_event_window = driver.seg.window
        driver.pending_buf = driver._new_phase_window()
        driver.pending_buf.records = [north]
        self.assertTrue(driver._staged_card_pending())
        self.assertEqual(len(driver.es.armies['gap of rohan'].units), 2)

        def board(checkpoint):
            text = (CAPTURE/checkpoint/'ai_client_board.txt').read_text(encoding='utf8')
            count = len(feed.read_text(encoding='utf8').splitlines())
            header = '#SEQ dump=1 feed_paragraphs=%d ts=1200 session=1000' % (count-1)
            (path/'ai_client_board.txt').write_text(header+'\n'+text.split('\n', 1)[1], encoding='utf8')

        def physical(line, checkpoint='after_committed_undo'):
            with feed.open('a', encoding='utf8') as stream:
                stream.write(line+'\n')
            board(checkpoint)
            return server.handle({'lines': ()})

        board('after_committed_undo')
        driver.view.refresh_dump()
        driver._try_pending()
        return server, path, physical

    def test_captured_undo_wait_has_no_stop_signal_or_repair_then_done_commits(self):
        server, path, physical = self.captured()
        driver = server.driver
        before = driver.n_commits_total
        state = driver.es
        records = list(driver.pending_buf.records)
        for _ in range(55):
            self.assertFalse(any('!CALL' in line or 'roll ' in line
                                 for line in server.handle({'lines': ()})))
        # Make the existing elapsed-feed diagnostic observable without sleeping.
        server._stalled = 60
        server._mirror_last = (-1, 0)
        server._write_mirror_snapshot()
        live = json.loads((path/'ai_live_status.json').read_text())
        self.assertEqual(live['stop_signals'], 0, live['mismatches'])
        self.assertNotIn('stalled', server._waiting_on())
        self.assertIn('say done', server._waiting_on())
        self.assertIsNone(driver._repair)
        self.assertFalse(any(kind in ('pending_buffer_stall', 'repair_not_offered') for kind, _ in driver.findings))
        self.assertIs(driver.es, state)
        self.assertEqual(driver.pending_buf.records, records)
        self.assertEqual(driver.n_commits_total, before)
        self.assertFalse(driver.alarms)
        physical('$Wey moves Isengard Regular from South Dunland to Gap of Rohan.', 'after_south_before_done')
        self.assertEqual(driver.n_commits_total, before)
        physical('<Wey> done', 'reclosed_after_done')
        self.assertEqual(driver.n_commits_total, before+1)
        self.assertEqual(len(driver.es.armies['gap of rohan'].units), 4)
        self.assertIsNone(ai.describe_pending(driver.es))
        self.assertFalse(any(kind == 'pending_buffer_stall' for kind, _ in driver.findings))

    def test_invalid_completed_choice_gets_once_correction_and_no_commit(self):
        server, path, physical = self.captured()
        before = server.driver.n_commits_total
        # Exact parsed illegal movement, not an engine-generated gesture.
        wrong = live_parser.parse('$Wey moves Isengard Regular from North Dunland to Rivendell.')
        server.driver.pending_buf.records = [wrong]
        server.driver.seg.window.records[-1] = wrong
        output = physical('<Wey> done')
        self.assertEqual(sum('does not match a legal choice' in line for line in output), 1)
        self.assertFalse(server.driver.pending_buf._staged_card_closed)
        for _ in range(30):
            output.extend(server.handle({'lines': ()}))
        output.extend(physical('<Wey> done'))
        self.assertEqual(sum('does not match a legal choice' in line for line in output), 1)
        self.assertEqual(server.driver.n_commits_total, before)
        self.assertEqual(len(server.driver.es.armies['gap of rohan'].units), 2)
        self.assertIsNotNone(ai.describe_pending(server.driver.es))
        self.assertIsNone(server.driver._repair)
        self.assertFalse(any(kind == 'pending_buffer_stall' for kind, _ in server.driver.findings))

    def test_noncompletion_pending_buffer_keeps_stall_diagnostic(self):
        fixture = dead.DeadMenPrefixes()
        rig = fixture.position(recruit=True)
        self.addCleanup(fixture.doCleanups)
        # A casualty schema is outside the optional-group completion contract;
        # unrelated human recruitment must remain unmatched and diagnostic.
        rig.drv.es.pending_card_option = dict(card_id=dead.CID, card_name='Dead Men of Dunharrow',
            side='fp', decider='fp', direct=True, card_casualty=True,
            options=[{'casualties': {'regular': 1}}])
        rig.feed('$Wey moves Gondor Regular from FP Reinforcements to Erech.')
        for _ in range(30):
            rig.drv.tick()
        self.assertFalse(rig.drv._staged_card_pending())
        self.assertTrue(any(kind == 'pending_buffer_stall' for kind, _ in rig.drv.findings))
        self.assertEqual(rig.drv.n_commits_total, 0)


if __name__ == '__main__':
    unittest.main()
