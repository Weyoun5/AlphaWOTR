"""Real-menu declaration completion after correcting the native Fellowship pose."""
from pathlib import Path
import copy
import itertools
import os
import pickle
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('WOTR_NATIVE', '0')
import ai
import board
import game_logic as gl
import test_bridge_oct11_combat_recovery as fixture
from live import choice_recovery, repair_evidence
from live.randomness import NeedClientCall


class DeclarationCorrection(unittest.TestCase):
    write_dump = fixture.LiveCombatRecovery.write_dump
    companion_pid = fixture.LiveNoncombatRecovery.companion_pid

    def setUp(self):
        fixture.LiveNoncombatRecovery.setUp(self)
        state = self.driver.es
        state.turn = 4
        state.fellowship.companions = [c for c in state.fellowship.companions
                                      if c.name not in ('Gandalf the Grey', 'Strider')]
        gl.change_guide(state, chosen='Legolas')
        for name in ('Gandalf the Grey', 'Strider'):
            self.dump[self.companion_pid(name)] = 'FP Dead'
        for companion in state.fellowship.companions:
            self.dump[self.companion_pid(companion.name)] = (
                'Guide Box' if companion.name == 'Legolas' else 'Fellowship Box')
        state.fellowship.region = state.board.region('eastern brown lands')
        state.fellowship.hidden = False
        state.fellowship.progress = 0
        state.fellowship.corruption = 5
        self.dump.update({'247': 'Eastern Brown Lands', '248': 'Track 0', '252': 'Track 5'})
        self.marker_states = {'247': 1}
        self.driver.phase = 'declare'
        self.driver._ensure_declare_armed()
        self.write_dump()
        self.assertEqual([a.action_type for a in ai.describe_pending(state)['actions']], ['pass'])

    def push(self, *lines):
        with self.feed.open('a', encoding='utf-8') as stream:
            stream.write(''.join(line + '\n' for line in lines))
        self.write_dump()
        return self.driver.tick()

    def buffer_correction(self):
        self.push('$Player flips The Fellowship in Eastern Brown Lands.')
        self.assertEqual(self.driver.phase, 'declare')
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(len(self.driver.pending_buf.records), 1)

    def assert_retained(self, operation=None):
        state = pickle.dumps(self.driver.es)
        random = pickle.dumps(vars(self.rand))
        old = self.driver.pending_buf
        records = copy.deepcopy(old.records)
        if operation is None:
            operation = lambda: self.driver._on_chat({'body': 'fp ok', 'nick': 'Player'})
        output = operation()
        self.assertEqual(self.driver.phase, 'declare', output)
        self.assertEqual(self.driver.n_commits_total, 0, output)
        self.assertIs(self.driver.pending_buf, old, output)
        self.assertEqual(old.records, records)
        self.assertEqual(pickle.dumps(self.driver.es), state)
        self.assertEqual(pickle.dumps(vars(self.rand)), random)
        return output

    def test_real_flip_then_fp_ok_uses_unchanged_endpoint_and_legal_pass_once(self):
        self.buffer_correction()
        before = pickle.dumps((self.driver.es.fellowship, self.driver.es.hands,
                               self.driver.es.dice, self.driver.es.turn))
        random = pickle.dumps((self.rand._feed, self.rand._feed_pos))
        output = self.push('<Player> fp ok')
        self.assertEqual(self.driver.phase, 'eyes', output)
        self.assertIsNone(self.driver.pending_buf)
        self.assertEqual([row[2] for row in self.driver.commits],
                         ['declare-decline', 'fellowship-guide-keep'])
        self.assertEqual(pickle.dumps((self.driver.es.fellowship, self.driver.es.hands,
                                     self.driver.es.dice, self.driver.es.turn)), before)
        self.assertEqual(pickle.dumps((self.rand._feed, self.rand._feed_pos)), random)
        self.assertEqual(self.rand._journal, [])
        self.assertIsNone(self.rand.pending())
        self.assertFalse(any(row[:2] == ('chat', 'refusal') for row in output), output)
        self.push('<Player> fp ok')
        self.assertEqual(self.driver.n_commits_total, 2)

    def test_pawn_progress_and_face_corrections_use_final_pose(self):
        self.push('$Player flips The Fellowship in Eastern Brown Lands.',
                  '$Player moves The Fellowship from Dagorlad to Eastern Brown Lands.',
                  '$Player moves FSP from Fellowship Track 1 to Fellowship Track 0.')
        self.assertEqual(len(self.driver.pending_buf.records), 3)
        output = self.push('<Player> fp ok')
        self.assertEqual(self.driver.phase, 'eyes', output)
        self.assertEqual(board.sanitize(str(gl.fellowship_region(self.driver.es))), 'eastern brown lands')
        self.assertEqual(self.driver.es.fellowship.progress, 0)
        self.assertFalse(self.driver.es.fellowship.hidden)

    def test_all_correction_orders_and_prefix_retries_wait_for_explicit_fp_ok(self):
        lines = ('$Player moves The Fellowship from Dagorlad to Eastern Brown Lands.',
                 '$Player moves FSP from Fellowship Track 1 to Fellowship Track 0.',
                 '$Player flips The Fellowship in Eastern Brown Lands.')
        for order in itertools.permutations(lines):
            for separate in (False, True):
                with self.subTest(order=order, separate=separate):
                    case = DeclarationCorrection()
                    case.setUp()
                    try:
                        before = pickle.dumps(case.driver.es)
                        for batch in ([order] if not separate else [(row,) for row in order]):
                            case.push(*batch)
                            self.assertEqual(case.driver.phase, 'declare')
                            self.assertEqual(case.driver.n_commits_total, 0)
                            self.assertEqual(pickle.dumps(case.driver.es), before)
                        self.assertFalse(case.driver._try_pending_prefixes())
                        self.assertEqual(case.driver.n_commits_total, 0)
                        self.assertEqual(pickle.dumps(case.driver.es), before)
                        output = case.push('<Player> fp ok')
                        self.assertEqual(case.driver.phase, 'eyes', output)
                        self.assertIsNone(case.driver.pending_buf)
                        self.assertEqual([row[2] for row in case.driver.commits],
                                         ['declare-decline', 'fellowship-guide-keep'])
                    finally:
                        case.doCleanups()

    def test_same_region_real_declaration_and_guide_nomination_still_commit(self):
        self.driver.es.fellowship.hidden = True
        self.driver.es.fellowship.progress = 1
        self.marker_states = {'247': 0}
        output = self.push('$Player moves FSP from Fellowship Track 1 to Fellowship Track 0.')
        self.assertEqual(self.driver.es.fellowship.progress, 0, output)
        self.assertEqual(self.driver.phase, 'guide', output)
        self.assertEqual(self.driver.commits[0][2], 'pending:declare')
        self.dump[self.companion_pid('Legolas')] = 'Fellowship Box'
        self.dump[self.companion_pid('Gimli')] = 'Guide Box'
        output = self.push('$Player moves Gimli from Fellowship Of The Ring to Guide Of The Fellowship.')
        self.assertEqual(self.driver.phase, 'eyes', output)
        self.assertEqual(self.driver.es.fellowship.guide.name, 'Gimli')

    def test_explicit_guide_nomination_still_declines_declaration(self):
        self.dump[self.companion_pid('Legolas')] = 'Fellowship Box'
        self.dump[self.companion_pid('Gimli')] = 'Guide Box'
        output = self.push('$Player moves Gimli from Fellowship Of The Ring to Guide Of The Fellowship.')
        self.assertEqual(self.driver.phase, 'eyes', output)
        self.assertEqual(self.driver.es.fellowship.guide.name, 'Gimli')
        self.assertEqual(self.driver.commits[0][2], 'declare-decline')

    def test_recovery_predicate_does_not_retire_or_close_and_can_validate_empty_buffer(self):
        self.buffer_correction()
        self.assert_retained(lambda: self.driver._recover_fellowship_choice())
        state = pickle.dumps(self.driver.es)
        random = pickle.dumps(vars(self.rand))
        self.assertTrue(choice_recovery.unchanged_declaration_correction(
            self.driver, self.dump, self.marker_states))
        self.assertIsNotNone(self.driver.pending_buf)
        self.driver.pending_buf = None
        self.assertFalse(choice_recovery.unchanged_declaration_correction(
            self.driver, self.dump, self.marker_states))
        self.assertTrue(choice_recovery.unchanged_declaration_correction(
            self.driver, self.dump, self.marker_states, allow_empty=True))
        self.assertEqual(self.driver.phase, 'declare')
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(pickle.dumps(self.driver.es), state)
        self.assertEqual(pickle.dumps(vars(self.rand)), random)
        self.assertEqual(ai.describe_pending(self.driver.es)['kind'], 'declare')

    def test_changed_or_unknown_pose_and_companion_locations_keep_correction(self):
        self.buffer_correction()
        good = dict(self.dump)
        for pid, zone in (('247', 'Dagorlad'), ('247', 'Spare'), ('247', 'Current Action'), ('248', 'Track 1'),
                          ('248', 'Spare'), ('252', 'Track 4'), ('252', 'Spare'),
                          (self.companion_pid('Legolas'), 'Fellowship Box'),
                          (self.companion_pid('Gimli'), 'FP Dead')):
            with self.subTest(pid=pid, zone=zone):
                self.dump = dict(good)
                self.dump[pid] = zone
                self.write_dump()
                self.assert_retained()
        self.dump = good
        del self.dump['247']
        self.write_dump()
        self.assert_retained()
        self.dump = dict(good, **{'247': 'Eastern Brown Lands'})
        for markers in ({}, {'247': 0}):
            with self.subTest(markers=markers):
                self.marker_states = markers
                self.write_dump()
                self.assert_retained()

    def test_missing_or_stale_board_and_unrouted_feed_keep_correction(self):
        self.buffer_correction()
        self.write_dump(complete=False)
        self.assert_retained()
        self.write_dump(feed_position=0)
        self.assert_retained()
        self.write_dump()
        self.dump_path.unlink()
        self.assert_retained()
        self.write_dump()
        self.driver.feed._buf = 'partial native record'
        self.assert_retained()
        self.driver.feed._buf = ''
        self.driver._feed_seq -= 1
        self.assert_retained()

    def test_card_companion_casualty_and_other_records_cannot_be_retired(self):
        self.buffer_correction()
        original = list(self.driver.pending_buf.records)
        for record in (dict(op='kill', pieces=['Gimli'], zone='Fellowship Box'),
                       dict(op='move', obj='Gimli', **{'from': 'Fellowship Box', 'to': 'Guide Box'}),
                       dict(op='move', obj='Corruption', **{'from': 'Track 4', 'to': 'Track 5'}),
                       dict(op='draw', deck='Character'),
                       dict(op='kill', pieces=['Event Card'], zone='FP Hand')):
            with self.subTest(record=record):
                self.driver.pending_buf.records = original + [record]
                self.assert_retained()

    def test_random_results_and_pending_random_request_remain_untouched(self):
        self.buffer_correction()
        for kind, payload in (('d6', [4]), ('tile', '1'),
                              ('companion', ['Gimli']), ('card_pick', ['opaque']),
                              ('card', ('FREE_PEOPLES', 'character', 'opaque'))):
            with self.subTest(kind=kind):
                self.rand.reset_feed()
                self.rand.feed(kind, payload)
                self.assert_retained()
        self.rand.reset_feed()
        self.rand._pending = NeedClientCall('draws', items=[('FREE_PEOPLES', 'character')])
        self.assert_retained()

    def test_pending_card_guide_hunt_draw_and_discard_obligations_are_preserved(self):
        self.buffer_correction()
        for name, value in (('pending_card_option', {'side': 'fp', 'card_name': 'retained'}),
                            ('pending_card_followup', {'side': 'fp', 'card_name': 'retained'}),
                            ('pending_guide_change', {'candidates': ['Legolas', 'Gimli']}),
                            ('pending_hunt', {'damage': 2, 'phase': 'hunt_sacrifice', 'casualty_taken': False}),
                            ('_pending_hunt_damage', 1), ('_hunt_reveal_pending', 1),
                            ('_reveal_place_due', True)):
            with self.subTest(name=name):
                setattr(self.driver.es, name, value)
                self.assert_retained()
                setattr(self.driver.es, name, None)
        hand = self.driver.es.hands[self.driver.human_side]
        hand.hand = list(self.driver.es.fp_char_deck[:7])
        gl.open_hand_limit_discard(self.driver.es)
        self.assertIsNotNone(self.driver.es.pending_discard)
        self.assert_retained()
        self.driver._turn_start_draws_open = True
        self.assert_retained()

    def test_consumed_random_result_or_recorded_pick_is_still_active_work(self):
        self.buffer_correction()
        self.rand.feed('d6', [4])
        self.assertEqual(self.rand._consume_d6(1), [4])
        self.assertFalse(any(self.rand.unconsumed().values()))
        self.assert_retained()
        self.rand.reset_feed()
        self.rand.begin_attempt()
        menu = ai.describe_pending(self.driver.es)['actions']
        self.rand.journal_pick(menu, lambda actions: actions)
        self.assertTrue(self.rand._journal)
        self.assert_retained()

    def test_changed_legal_declaration_is_committed_before_unchanged_correction_lane(self):
        state = self.driver.es
        state.fellowship.region = state.board.region('rivendell')
        state.fellowship.hidden = True
        state.fellowship.progress = 4
        self.dump.update({'247': 'Dimrill Dale', '248': 'Track 0'})
        self.marker_states = {'247': 0}
        self.write_dump()
        self.assertTrue(any(a.action_type == 'fellowship_declare'
                            for a in ai.describe_pending(state)['actions']))
        output = self.driver._on_chat({'body': 'fp ok', 'nick': 'Player'})
        self.assertEqual(self.driver.phase, 'eyes', output)
        self.assertEqual(board.sanitize(str(gl.fellowship_region(self.driver.es))), 'dimrill dale')
        self.assertEqual(self.driver.es.fellowship.progress, 0)
        self.assertTrue(self.driver.es.fellowship.hidden)
        self.assertEqual(self.driver.commits[0][2], 'pending:declare')

    def test_changing_export_during_retirement_rolls_back_original_buffer(self):
        self.buffer_correction()
        real_same = repair_evidence.same_dump
        calls = []
        def changing(driver, evidence):
            calls.append(1)
            return len(calls) < 3 and real_same(driver, evidence)
        with patch.object(repair_evidence, 'same_dump', side_effect=changing):
            self.assert_retained()
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
