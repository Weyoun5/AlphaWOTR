"""A committed but unrenderable AI consequence remains a physical barrier."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
from pathlib import Path
import json
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import game_state as gs
from live.live_server import LiveServer
from live import client_board
import emitter
from test_bridge_oct12_forced_casualty_rows import great_host_position
from bridge_evidence_fixture import publish, request_header, stamp_acks

FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW


class ForcedRowsHold(unittest.TestCase):
    CAPTURE = ROOT/'audit/bridge_oct11/player/full_games/clean_r2_game01/failure_folde_great_host'
    def fixture(self):
        folder = tempfile.TemporaryDirectory(prefix='forced_rows_hold_')
        self.addCleanup(folder.cleanup)
        server = LiveServer(folder.name, FP, load_nets=False, seed=19)
        pre, post, chosen, view = great_host_position()
        pre._declare_phase = False
        pre.dice_rolled = True
        for pool in pre.dice.values():
            for die in pool:
                die.current = gs.ActionDieFace.ARMY
                die.used = True
        driver = server.driver
        driver.es = pre
        for model in driver.hand_models.values():
            model.es = pre
        driver.phase = 'actions'
        driver.last_actor = SH
        driver.enforce_human = True
        driver.view.piece_region = dict(view.piece_region)
        driver.view.piece_region.update({'288': 'FP Reinforcements',
                                         '289': 'FP Reinforcements', '105': 'Current Action'})
        server.own_net = object()
        server.own_search = SimpleNamespace(choose=lambda *args, **kw: (chosen, None))
        feed = Path(folder.name)/'ai_feed.txt'
        feed.write_text('', encoding='utf8')
        ack = Path(folder.name)/'ai_inject_ack.jsonl'
        ack.write_text('', encoding='utf8')
        driver.ack_path = str(ack)
        return server, feed, ack

    def cycle(self, server, feed, *lines):
        if lines:
            with feed.open('a', encoding='utf8') as stream:
                stream.write('\n'.join(lines)+'\n')
        with patch.object(server.cards, 'harvest', return_value=[]):
            header = request_header(server.driver)
            return server.handle({'lines': ('#SEQ session=%d req=%d' %
                                            (header['session'], header['req']),)})

    def hold(self):
        server, feed, ack = self.fixture()
        output = self.cycle(server, feed)
        chosen = [line for line in output if '<~Controls.296~>' in line]
        self.assertEqual(len(chosen), 1, output)
        self.assertIn('287', chosen[0])
        self.assertEqual(server.driver.n_commits_total, 1)
        self.assertEqual(server.driver.es.pending_combat['phase'], 'attacker_continue')
        # The selected Elite kill was actually delivered; it cannot be dropped
        # or replayed merely because the later mandatory downgrade held.
        record = stamp_acks(server.driver, [dict(raw=chosen[0], ok=True,
            paragraphs_before=0, paragraphs_after=1)])[0]
        ack.write_text(json.dumps(record)+'\n', encoding='utf8')
        server.driver._consume_inject_acks()
        self.assertFalse(server.driver.echo_expect)
        return server, feed, ack, output

    def test_hold_blocks_human_press_and_rng_after_selected_output_ack(self):
        server, feed, ack, output = self.hold()
        driver = server.driver
        self.assertIn('board correction', server._waiting_on().lower())
        self.assertNotIn('say press', server._waiting_on())
        before = ai.clone_state(driver.es)
        for lines in ((), ('<Wey> press',),
                      ('$Wey rolls 5 Combat Dice: 6 6 6 6 6.',),
                      ('<Wey> nc',), ('<Wey> stop',)):
            more = self.cycle(server, feed, *lines)
            self.assertFalse(any(line.startswith('!CALL') or '<~' in line for line in more), more)
            self.assertEqual(driver.n_commits_total, 1)
            self.assertEqual(driver.es.pending_combat, before.pending_combat)
            self.assertEqual(driver.es.dice[SH][0].used, before.dice[SH][0].used)
            self.assertIsNone(server.rand.pending())
        self.assertTrue(driver.echo_pending())
        self.assertEqual(sum('board correction' in line.lower() for line in output), 1)

    def captured_hold(self):
        """Saved public failure plus its trusted local engine checkpoint.

        Never print or rebind private hand contents. Only the known unpaid
        native Great Host downgrade is installed as the retained obligation.
        """
        server, feed, ack = self.fixture()
        driver = server.driver
        feed.write_text('#BLIND ai_side=FREE_PEOPLES\n', encoding='utf8')
        driver.feed.poll()
        driver._feed_seq = len(driver.feed.lines)
        driver.es = pickle.loads((self.CAPTURE/'ai_live_state.pkl').read_bytes())
        for model in driver.hand_models.values():
            model.es = driver.es
        driver.n_commits_total = 1
        driver.owed_die_kills = [('SHADOW', 0)]
        rows = {'units': {'rohan:elite': {'minus': [('folde', 1)], 'plus': []},
                          'rohan:regular': {'minus': [], 'plus': [('folde', 1)]}},
                'naz': {'minus': [], 'plus': []}, 'named': {'leaves': {}, 'joins': {}},
                'cas': {'rohan:elite': 1, 'rohan:regular': -1}, 'unexplained': [],
                'downgrade_sources': [('rohan', 'cas')]}
        dump_path = feed.parent/'ai_client_board.txt'
        driver.view.dump_path = str(dump_path)
        raw = (self.CAPTURE/'ai_client_board.txt').read_text(encoding='utf8')
        self.native_rows = raw.splitlines()[1:]
        self.write_capture_dump(driver)
        driver.view.refresh_dump()
        # Induce an actual emitter HOLD without replacing the emitter. The
        # subsequent stable native dump restores the real available Elites.
        driver.view.piece_region.update({'288': 'FP Reinforcements', '289': 'FP Reinforcements'})
        output = driver._render_post_commit_ai_rows(
            driver.es, driver.es, rows, False, 'captured Great Host')
        self.assertTrue(driver.forced_ai_rows_pending(), output)
        driver.view.refresh_dump()
        return server, feed, ack

    def write_capture_dump(self, driver, updates=None, *, feed_position=0):
        updates = updates or {}
        rows = []
        for line in self.native_rows:
            parts = line.split('\t')
            if len(parts) < 3:
                rows.append(line)
                continue
            pids = parts[2].split()
            for index, pid in enumerate(pids):
                code = updates.get(pid, parts[1].strip('<~>'))
                if code is None:
                    continue
                flags = parts[3].split()[index] if len(parts) > 3 else '-'
                face = parts[4].split()[index] if len(parts) > 4 else '-'
                rows.append('0\t<~%s~>\t%s\t%s\t%s' % (code, pid, flags, face))
        Path(driver.view.dump_path).write_text(
            '#SEQ dump=1 feed_paragraphs=%d raw_lines=0 ts=1 session=9911\n' % feed_position
            + '\n'.join(rows)+'\n', encoding='utf8')

    def test_captured_resync_retries_exact_residual_without_replaying_selected_casualty(self):
        server, feed, ack = self.captured_hold()
        driver = server.driver
        before = pickle.dumps(driver.es)
        items, report = driver.reset_and_resync()
        commands = [row[1] for row in items if row[0] == 'line']
        publish(driver, commands)
        self.assertEqual(len(commands), 2, (items, report))
        self.assertNotIn('287 ', '\n'.join(commands))
        ops = [emitter.classify_line(emitter.strip_tail(line)) for line in commands]
        self.assertEqual(ops[0][0], 'hand_exit_b')
        self.assertEqual(ops[1][1]['frm'], 'Game.2039')
        self.assertEqual(ops[1][1]['to'], emitter.zone_code_for('folde'))
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertTrue(driver.echo_pending())
        repeated, _ = driver.reset_and_resync()
        self.assertFalse(any(row[0] == 'line' for row in repeated), repeated)
        updates = {}
        acks = []
        for line, (op, fields) in zip(commands, ops):
            for pid in fields.get('pieces') or [fields['piece']]:
                updates[pid] = 'Game.2039' if op == 'hand_exit_b' else fields['to']
            acks.append(dict(raw=line, ok=True, paragraphs_before=len(acks), paragraphs_after=len(acks)+1))
        self.write_capture_dump(driver, updates)
        acks = stamp_acks(driver, acks)
        ack.write_text(''.join(json.dumps(row)+'\n' for row in acks), encoding='utf8')
        driver._consume_inject_acks()
        settled, report = driver.reset_and_resync()
        self.assertTrue(driver._last_resync_adoption['complete'], (settled, report))
        self.assertFalse(driver.forced_ai_rows_pending())
        self.assertFalse(any(row[0] == 'line' for row in settled), settled)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(driver.owed_die_kills, [('SHADOW', 0)])

    def test_captured_resync_refuses_incomplete_or_stale_measurement(self):
        for changes, position in (({'288': None}, 0), ({}, 99)):
            with self.subTest(changes=changes, position=position):
                server, _, _ = self.captured_hold()
                driver = server.driver
                self.write_capture_dump(driver, changes, feed_position=position)
                before = pickle.dumps(driver.es)
                items, report = driver.reset_and_resync()
                self.assertFalse(any(row[0] == 'line' for row in items), (items, report))
                self.assertTrue(driver.forced_ai_rows_pending())
                self.assertFalse(driver._last_resync_adoption['complete'])
                self.assertEqual(pickle.dumps(driver.es), before)

    def test_measured_already_fixed_board_clears_hold_without_another_casualty(self):
        server, _, _ = self.captured_hold()
        driver = server.driver
        before = pickle.dumps(driver.es)
        self.write_capture_dump(driver, {'288': 'Game.2039', '290': emitter.zone_code_for('folde')})
        items, report = driver.reset_and_resync()
        self.assertTrue(driver._last_resync_adoption['complete'], (items, report))
        self.assertFalse(driver.forced_ai_rows_pending())
        self.assertFalse(any(row[0] == 'line' for row in items), items)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(driver.owed_die_kills, [('SHADOW', 0)])

    def test_partial_or_unrelated_drift_cannot_replay_whole_retained_tail(self):
        for updates in ({'288': 'Game.2039'}, {'263': 'Game.2000'}):
            with self.subTest(updates=updates):
                server, _, _ = self.captured_hold()
                driver = server.driver
                before = pickle.dumps(driver.es)
                self.write_capture_dump(driver, updates)
                items, report = driver.reset_and_resync()
                self.assertTrue(driver.forced_ai_rows_pending(), report)
                self.assertFalse(any(row[0] == 'line' for row in items), items)
                self.assertFalse(driver._last_resync_adoption['complete'])
                self.assertEqual(pickle.dumps(driver.es), before)
                self.assertEqual(driver.n_commits_total, 1)

    def test_ai_owned_continuation_is_also_held_without_another_search(self):
        server, _, _, _ = self.hold()
        driver = server.driver
        server.ai_side = driver.ai_side = SH
        server.human_side = driver.human_side = FP
        pending = ai.describe_pending(driver.es)
        self.assertIs(pending['side'], SH)
        self.assertIsNone(driver.ai_action_dp())
        with patch.object(server.own_search, 'choose', side_effect=AssertionError('new search')):
            self.assertEqual(server._ai_decide(pending), [])
        self.assertEqual(driver.n_commits_total, 1)
        self.assertIsNone(server.rand.pending())

    def test_repair_cannot_adopt_stale_native_units_over_paid_losses(self):
        server, _, _ = self.captured_hold()
        driver = server.driver
        before = pickle.dumps(driver.es)
        direct = driver._adopt_board_now('manual')
        self.assertTrue(driver.forced_ai_rows_pending())
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertFalse(any(row[0] == 'line' for row in direct), direct)
        driver._repair = {'kind': 'census', 'phase': 'offered'}
        items = driver._repair_begin()
        self.assertEqual(sum(row[0] == 'line' for row in items), 2, items)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(driver.n_commits_total, 1)

    def test_retry_publication_failure_preserves_obligation_and_paid_state(self):
        server, _, _ = self.captured_hold()
        driver = server.driver
        before = pickle.dumps(driver.es)
        with patch.object(driver, 'expect_echo', side_effect=RuntimeError('isolated publication failure')):
            items, report = driver.reset_and_resync()
        self.assertTrue(driver.forced_ai_rows_pending(), report)
        self.assertFalse(any(row[0] == 'line' for row in items), items)
        self.assertEqual(driver.echo_ledger, [])
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(driver.n_commits_total, 1)
        self.assertEqual(driver.owed_die_kills, [('SHADOW', 0)])

    def published_failure(self, *, fungible=False):
        server, feed, ack = self.captured_hold()
        driver = server.driver
        if fungible:
            # Two interchangeable casualties exist before the replacement;
            # the engine's committed state retains precisely one of them.
            driver.es.fp_casualties[gs.Nation.ROHAN]['regular'] = 1
            self.write_capture_dump(driver, {'291': 'Game.2039'})
        before = pickle.dumps(driver.es)
        items, _ = driver.reset_and_resync()
        commands = [item[1] for item in items if item[0] == 'line']
        publish(driver, commands)
        self.assertEqual(len(commands), 2)
        self.assertIn('290 ', commands[1])
        self.write_capture_dump(driver, {'288': 'Game.2039', **({'291': 'Game.2039'} if fungible else {})})
        records = stamp_acks(driver, [dict(raw=commands[0], ok=True,
            paragraphs_before=0, paragraphs_after=1), dict(
            raw=commands[1], ok=False, error='isolated missing placement',
            paragraphs_before=1, paragraphs_after=1)])
        ack.write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf8')
        driver._consume_inject_acks()
        return server, feed, ack, before, commands

    def manual_replacement(self, server, feed, *, pid='290', updates=None, feed_position=1):
        line = '$Wey moves Rohan Regular from FP Casualties to Folde.'
        with feed.open('a', encoding='utf8') as stream:
            stream.write(line+'\n')
        Path(str(feed)+'.meta.jsonl').write_text(json.dumps(dict(
            seq=2, nick='Wey', raw=line, pids=[pid], side='FP', gen=1,
            replay=False, tags=['Scribe.141', 'Scribe.142', 'Game.2039',
                                'Scribe.143', 'Game.1933', 'Scribe.144'],
            segs=[[pid], [], [], [], [], []]))+'\n', encoding='utf8')
        self.write_capture_dump(server.driver, updates or {
            '288': 'Game.2039', pid: emitter.zone_code_for('folde')},
            feed_position=feed_position)
        server.on_control({'resync': 1, 'enforce_human': True})
        return self.cycle(server, feed)

    def test_failed_published_replacement_manual_correction_is_not_another_battle_choice(self):
        for fungible in (False, True):
            with self.subTest(fungible=fungible):
                server, feed, ack, before, commands = self.published_failure(fungible=fungible)
                driver = server.driver
                self.assertTrue(driver.forced_ai_rows_pending())
                pid = '291' if fungible else '290'
                output = self.manual_replacement(server, feed, pid=pid)
                self.assertEqual(pickle.dumps(driver.es), before)
                self.assertEqual(driver.n_commits_total, 1)
                self.assertTrue(driver._last_resync_adoption['complete'], output)
                self.assertFalse(driver.echo_expect)
                self.assertEqual(driver.echo_ledger, [])
                self.assertFalse(driver.forced_ai_rows_pending())
                self.assertFalse(driver.pending_buf)
                self.assertIsNone(server.rand.pending())
                self.assertFalse(any('<~' in line or 'draw a ' in line or 'roll ' in line for line in output), output)
                self.assertFalse(any(kind == 'battle_resolved_by_hand' for kind, _ in driver.findings))
                self.assertFalse(any('EMISSION REFUSED' in alarm for alarm in driver.alarms))
                self.assertEqual(driver.owed_die_kills, [('SHADOW', 0)])
                server.on_control({'resync': 2})
                repeated = self.cycle(server, feed)
                self.assertFalse(any('<~' in line or '!CALL' in line for line in repeated), repeated)
                self.assertEqual(pickle.dumps(driver.es), before)

    def test_published_repair_successful_acks_release_only_after_the_complete_tail(self):
        server, feed, ack = self.captured_hold()
        driver = server.driver
        before = pickle.dumps(driver.es)
        items, _ = driver.reset_and_resync()
        commands = [item[1] for item in items if item[0] == 'line']
        publish(driver, commands)
        self.assertTrue(driver.forced_ai_rows_pending())
        rows = [dict(raw=line, ok=True, paragraphs_before=i, paragraphs_after=i+1)
                for i, line in enumerate(commands)]
        rows = stamp_acks(driver, rows)
        ack.write_text(json.dumps(rows[0])+'\n', encoding='utf8')
        driver._consume_inject_acks()
        self.assertTrue(driver.forced_ai_rows_pending())
        self.assertIn('board correction', server._waiting_on().lower())
        ack.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf8')
        driver._consume_inject_acks()
        self.assertFalse(driver.forced_ai_rows_pending())
        self.assertFalse(driver.echo_pending())
        self.assertEqual(driver.echo_ledger, [])
        self.assertIn('say press', server._waiting_on())
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_failed_published_tail_rejects_unproved_or_unrelated_completion(self):
        variants = ('wrong_destination', 'missing', 'stale', 'unrelated_echo')
        for variant in variants:
            with self.subTest(variant=variant):
                server, feed, ack, before, commands = self.published_failure()
                driver = server.driver
                updates = {'288': 'Game.2039', '290': emitter.zone_code_for('folde')}
                if variant == 'wrong_destination':
                    updates['290'] = emitter.zone_code_for('edoras')
                elif variant == 'missing':
                    updates['289'] = None
                elif variant == 'unrelated_echo':
                    driver.echo_expect['105'] = dict(zone='Dead', deadline=100,
                        line='$unrelated-die-command', expired=False)
                output = self.manual_replacement(server, feed, updates=updates,
                    feed_position=99 if variant == 'stale' else 1)
                self.assertFalse(driver._last_resync_adoption['complete'], output)
                self.assertTrue(driver.forced_ai_rows_pending())
                self.assertIn('290', driver.echo_expect)
                self.assertEqual(pickle.dumps(driver.es), before)
                self.assertEqual(driver.n_commits_total, 1)
                self.assertIsNone(server.rand.pending())
                self.assertFalse(any('<~' in line or '!CALL' in line or 'draw a ' in line for line in output), output)

    def test_successful_ack_does_not_clear_an_unrelated_echo_or_equal_ledger_row(self):
        server, _, ack = self.captured_hold()
        driver = server.driver
        items, _ = driver.reset_and_resync()
        commands = [item[1] for item in items if item[0] == 'line']
        publish(driver, commands)
        unrelated = dict(driver.echo_ledger[0])
        unrelated.pop('echo_id')
        driver.echo_ledger.append(unrelated)
        # Normal ledger claiming can copy rows; identity must survive that.
        driver.echo_ledger[:] = [dict(row) for row in driver.echo_ledger]
        driver.echo_expect['105'] = dict(zone='Dead', deadline=100,
            line='$unrelated-die-command', expired=False)
        records = stamp_acks(driver, [dict(raw=line, ok=True,
            paragraphs_before=i, paragraphs_after=i+1) for i, line in enumerate(commands)])
        ack.write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf8')
        driver._consume_inject_acks()
        self.assertFalse(driver.forced_ai_rows_pending())
        self.assertEqual(driver.echo_ledger, [unrelated])
        self.assertEqual(list(driver.echo_expect), ['105'])

    def test_manual_completion_clears_only_its_owned_expired_output_alarm(self):
        server, feed, ack, before, _ = self.published_failure()
        driver = server.driver
        unrelated = 'EMISSION UNACKNOWLEDGED: unrelated historical output'
        driver.alarms.append(unrelated)
        driver._tick_n = max(row['deadline'] for row in driver.echo_expect.values())
        driver._check_echo_expiry()
        self.assertEqual(sum('EMISSION UNACKNOWLEDGED' in row
                             for row in driver.alarms), 2)
        self.manual_replacement(server, feed)
        self.assertTrue(driver._last_resync_adoption['complete'])
        self.assertFalse(driver.echo_pending())
        self.assertEqual([row for row in driver.alarms
                          if 'EMISSION UNACKNOWLEDGED' in row], [unrelated])
        self.assertEqual(pickle.dumps(driver.es), before)

    def test_mixed_expiry_alarm_survives_confirmation_of_only_the_retained_tail(self):
        server, _, ack = self.captured_hold()
        driver = server.driver
        items, _ = driver.reset_and_resync()
        commands = [item[1] for item in items if item[0] == 'line']
        publish(driver, commands)
        driver.echo_expect['105'] = dict(zone='Dead', deadline=0,
            line='$unrelated-die-command', expired=False)
        driver._tick_n = max(row['deadline'] for row in driver.echo_expect.values())
        driver._check_echo_expiry()
        expired = [row for row in driver.alarms if 'EMISSION UNACKNOWLEDGED' in row]
        self.assertEqual(len(expired), 1)
        records = stamp_acks(driver, [dict(raw=line, ok=True,
            paragraphs_before=i, paragraphs_after=i+1) for i, line in enumerate(commands)])
        ack.write_text(''.join(json.dumps(row)+'\n' for row in records), encoding='utf8')
        driver._consume_inject_acks()
        self.assertFalse(driver.forced_ai_rows_pending())
        self.assertEqual(list(driver.echo_expect), ['105'])
        self.assertTrue(all(row in driver.alarms for row in expired))


if __name__ == '__main__':
    unittest.main(verbosity=2)
