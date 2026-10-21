"""Retained-battle recovery uses the legal matcher; no physical-game claim."""
from pathlib import Path
import contextlib
import os
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'wotr_bridge')]
os.environ.setdefault('WOTR_NATIVE', '0')

import ai
import board
import combat
import game_state as gs
from live import combat_recovery as recovery
from live.driver import PhaseWindow


def state_and_driver(phase='casualty_allocation', hits=1):
    state = gs.setup_initial_game(board.load_board())
    state.armies = {
        'fords of isen': gs.Army('fords of isen', [
            gs.Unit(gs.Nation.ROHAN, gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.ROHAN, gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.ROHAN, gs.UnitKind.ELITE)], []),
        'orthanc': gs.Army('orthanc', [
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR),
            gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.REGULAR)], []),
    }
    state.pending_combat = {
        'phase': phase, 'attacker_region': 'fords of isen',
        'defender_region': 'orthanc', 'attacker_side': gs.Side.FREE_PEOPLES,
        'defender_side': gs.Side.SHADOW, 'deciding_side': gs.Side.FREE_PEOPLES,
        'is_siege': False, 'rounds_fought': 1, 'alloc_stage': 'attacker',
        'atk_hits': hits, 'def_hits': 0,
        'atk_card_committed': True, 'def_card_committed': True,
        '_recovery_test_rolls': (6, 4, 1),
    }
    dynamic = {
        'recover_r1': (gs.Nation.ROHAN, gs.UnitKind.REGULAR),
        'recover_r2': (gs.Nation.ROHAN, gs.UnitKind.REGULAR),
        'recover_r3': (gs.Nation.ROHAN, gs.UnitKind.REGULAR),
        'recover_e1': (gs.Nation.ROHAN, gs.UnitKind.ELITE),
        'recover_s1': (gs.Nation.ISENGARD, gs.UnitKind.REGULAR),
        'recover_s2': (gs.Nation.ISENGARD, gs.UnitKind.REGULAR),
    }
    dump = {'recover_r1': 'Fords of Isen', 'recover_r2': 'Fords of Isen',
            'recover_r3': 'FP Reinforcements', 'recover_e1': 'Fords of Isen',
            'recover_s1': 'Orthanc', 'recover_s2': 'Orthanc'}
    driver = SimpleNamespace(es=state, human_side=gs.Side.FREE_PEOPLES,
                             pending_buf=None, view=SimpleNamespace(dynamic_types=dynamic),
                             rctl=None, _attempt_pin=lambda: contextlib.nullcontext,
                             _new_phase_window=PhaseWindow)
    return state, driver, dump


class CombatRecovery(unittest.TestCase):
    def prove(self, driver, dump):
        before = pickle.dumps(driver.es)
        buf = driver.pending_buf
        with patch.object(combat, 'roll_d6_batch', side_effect=AssertionError('repeated combat roll')):
            result = recovery.plan(driver, dump)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertIs(driver.pending_buf, buf)
        return result

    def test_lost_regular_removal_charges_existing_hit_once(self):
        state, driver, dump = state_and_driver()
        state.turn = 7
        state.dice[gs.Side.FREE_PEOPLES][0].used = True
        before_hands = pickle.dumps(state.hands)
        dump['recover_r1'] = 'FP Dead'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual(result.added, 1)
        self.assertEqual([a.action_type for a in result.match.chain], ['casualty_allocate'])
        self.assertEqual(result.match.chain[0].params['operation'], 'remove_regular')
        post = result.match.post
        self.assertEqual(len(post.armies['fords of isen'].units), 2)
        self.assertEqual(post.fp_casualties[gs.Nation.ROHAN]['regular'], 1)
        self.assertEqual(post.turn, 7)
        self.assertTrue(post.dice[gs.Side.FREE_PEOPLES][0].used)
        self.assertEqual(pickle.dumps(post.hands), before_hands)
        self.assertEqual(post.pending_combat['_recovery_test_rolls'], (6, 4, 1))
        driver.es = post
        again = self.prove(driver, dump)
        self.assertEqual(again.status, 'hold')

    def test_visible_elite_replacement_recovers_downgrade(self):
        _state, driver, dump = state_and_driver()
        dump['recover_e1'] = 'FP Dead'
        dump['recover_r3'] = 'Fords of Isen'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual(result.match.chain[0].params['operation'], 'reduce_elite')
        self.assertEqual(result.match.post.fp_casualties[gs.Nation.ROHAN]['elite'], 1)

    def test_partial_native_downgrade_retains_record_and_adds_lost_replacement(self):
        _state, driver, dump = state_and_driver()
        driver.pending_buf = PhaseWindow()
        native = {'op': 'kill', 'pieces': ['Rohan Elite'], 'zone': 'Fords of Isen',
                  'norm': '$Human kills Rohan Elite in Fords of Isen.',
                  'meta': {'tags': ['Controls.304']}}
        driver.pending_buf.records = [native]
        dump['recover_e1'] = 'FP Dead'
        dump['recover_r3'] = 'Fords of Isen'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertIs(result.window.records[0], native)
        self.assertEqual(len(driver.pending_buf.records), 1)
        self.assertEqual(result.added, 1)

    def test_two_hit_whole_elite_removal_recovers_from_complete_visible_loss(self):
        _state, driver, dump = state_and_driver(hits=2)
        dump['recover_e1'] = 'FP Dead'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual([a.params['operation'] for a in result.match.chain], ['remove_elite'])
        self.assertEqual(result.match.post.fp_casualties[gs.Nation.ROHAN]['elite'], 1)
        self.assertEqual(result.match.post.fp_casualties[gs.Nation.ROHAN]['regular'], 0)

    def test_one_hit_elite_removal_without_regular_replacement_stays_pending(self):
        _state, driver, dump = state_and_driver(hits=1)
        dump['recover_e1'] = 'FP Dead'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'hold')

    def test_native_downgrade_intent_cannot_be_recovered_as_whole_elite_removal(self):
        _state, driver, dump = state_and_driver(hits=2)
        driver.pending_buf = PhaseWindow()
        driver.pending_buf.records = [dict(op='kill', pieces=['Rohan Elite'],
            zone='Fords of Isen', norm='$Human kills Rohan Elite in Fords of Isen.',
            meta={'tags': ['Controls.304']})]
        dump['recover_e1'] = 'FP Dead'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'hold')

    def test_whole_elite_removed_to_reserve_cannot_be_a_permanent_fp_casualty(self):
        _state, driver, dump = state_and_driver(hits=2)
        dump['recover_e1'] = 'FP Reinforcements'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'hold')

    def test_two_nations_whole_elites_pay_four_hits_without_order_invention(self):
        state, driver, dump = state_and_driver(hits=4)
        state.armies['fords of isen'].units.append(gs.Unit(gs.Nation.ELVES, gs.UnitKind.ELITE))
        driver.view.dynamic_types['recover_elf'] = (gs.Nation.ELVES, gs.UnitKind.ELITE)
        dump['recover_elf'] = dump['recover_e1'] = 'FP Dead'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual(len(result.match.post.armies['fords of isen'].units), 2)
        self.assertEqual(result.match.post.fp_casualties[gs.Nation.ROHAN]['elite'], 1)
        self.assertEqual(result.match.post.fp_casualties[gs.Nation.ELVES]['elite'], 1)

    def test_shadow_whole_elite_loss_can_return_to_its_reinforcements(self):
        state, driver, dump = state_and_driver()
        state.armies['orthanc'].units[0] = gs.Unit(gs.Nation.ISENGARD, gs.UnitKind.ELITE)
        driver.view.dynamic_types['recover_s1'] = (gs.Nation.ISENGARD, gs.UnitKind.ELITE)
        driver.human_side = gs.Side.SHADOW
        state.pending_combat.update(deciding_side=gs.Side.SHADOW,
                                    alloc_stage='defender', atk_hits=0, def_hits=2)
        dump['recover_s1'] = 'SA Reinforcements'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual(len(result.match.post.armies['orthanc'].units), 1)
        self.assertEqual(result.match.post.armies['orthanc'].units[0].kind, gs.UnitKind.REGULAR)

    def test_wrong_extra_casualty_does_not_expand_hit_budget(self):
        _state, driver, dump = state_and_driver(hits=1)
        dump['recover_r1'] = dump['recover_r2'] = 'FP Dead'
        self.assertEqual(self.prove(driver, dump).status, 'hold')

    def test_opponent_board_change_is_not_a_human_choice(self):
        _state, driver, dump = state_and_driver()
        dump['recover_r1'] = 'FP Dead'
        dump['recover_s1'] = 'SA Reinforcements'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'hold')
        self.assertIn('another player', result.note)

    def test_parked_unit_is_not_invented_as_a_casualty(self):
        _state, driver, dump = state_and_driver()
        dump['recover_r1'] = 'Current FP Card'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'hold')
        self.assertIn('parked', result.note)

    def test_fp_returned_to_reserve_cannot_be_charged_as_permanent_casualty(self):
        _state, driver, dump = state_and_driver()
        dump['recover_r1'] = 'FP Reinforcements'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'hold')
        self.assertIn('casualty and reserve boxes', result.note)

    def test_lost_stronghold_entry_uses_legal_retreat(self):
        state, driver, dump = state_and_driver(phase='defender_siege_choice')
        driver.human_side = gs.Side.SHADOW
        state.pending_combat['deciding_side'] = gs.Side.SHADOW
        dump['recover_s1'] = dump['recover_s2'] = 'Orthanc Stronghold'
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual([a.action_type for a in result.match.chain], ['retreat_into_siege'])

    def test_declared_onslaught_payment_recovers_without_redeclaring_or_rolling(self):
        from test_onslaught_payments import position, choose
        from live.randomness import ClientRandomness
        from live.driver import _RandCtl
        _state, driver, dump = state_and_driver()
        driver.es = position(regulars=2, elites=0)
        driver.human_side = gs.Side.SHADOW
        driver.es = ai.apply_action(driver.es, choose(driver.es, count=1))
        # Match the fixture's three FP Regulars and the two Shadow Regulars.
        driver.view.dynamic_types['recover_e1'] = (gs.Nation.ROHAN, gs.UnitKind.REGULAR)
        dump['recover_s1'] = 'SA Reinforcements'
        before = pickle.dumps(driver.es)
        rand = ClientRandomness(human_side=gs.Side.SHADOW, seed=1701)
        rand.install()
        driver.rctl = _RandCtl(rand)
        driver._attempt_pin = lambda: rand.pin
        rng_before = pickle.dumps(vars(rand))
        result = recovery.plan(driver, dump)
        self.assertEqual(result.status, 'pending', result.note)
        self.assertEqual(pickle.dumps(driver.es), before)
        self.assertEqual(pickle.dumps(vars(rand)), rng_before)
        self.assertEqual(result.match.proven_starve_call.kw,
                         dict(need=1, ctx='post-casualty Onslaught', side=gs.Side.SHADOW, role='defender'))
        self.assertEqual(result.match.proven_starve_chain[0].params['kind'], 'regular')
        self.assertEqual(driver.es.pending_combat['_onslaught_payment']['count'], 1)
        self.assertEqual(driver.es.pending_combat['_onslaught_payment']['remaining'], 1)
        self.assertEqual(len(result.window.records), 1)

    def test_equal_board_does_not_invent_stand_or_card_choice(self):
        for phase in ('defender_retreat', 'attacker_combat_card'):
            with self.subTest(phase=phase):
                _state, driver, dump = state_and_driver(phase=phase)
                self.assertEqual(self.prove(driver, dump).status, 'hold')

    def test_lost_retreat_gesture_uses_current_legal_destination(self):
        state, driver, dump = state_and_driver(phase='defender_retreat')
        driver.human_side = gs.Side.SHADOW
        state.pending_combat['deciding_side'] = gs.Side.SHADOW
        options = [a for a in ai.describe_pending(state)['actions']
                   if a.action_type == 'combat_retreat']
        self.assertTrue(options)
        destination = options[0].params.get('to_region')
        self.assertIsNotNone(destination, options[0].params)
        dump['recover_s1'] = dump['recover_s2'] = destination
        result = self.prove(driver, dump)
        self.assertEqual(result.status, 'match', result.note)
        self.assertEqual([a.action_type for a in result.match.chain], ['combat_retreat'])
        self.assertEqual(result.match.chain[0].params, options[0].params)


class LiveCombatRecovery(unittest.TestCase):
    """Actual files, FeedReader, LiveDriver, matcher, commit, and census."""

    def setUp(self):
        from live import client_board
        from live.driver import LiveDriver
        from live.randomness import ClientRandomness
        from test_live_witnesses import build_dump_from_state
        self.folder = tempfile.TemporaryDirectory(prefix='combat_recovery_', dir=ROOT)
        self.addCleanup(self.folder.cleanup)
        folder = Path(self.folder.name)
        self.assertTrue(folder.resolve().is_relative_to(ROOT.resolve()))
        self.feed = folder / 'feed.txt'
        self.feed.write_text('#BLIND ai_side=SHADOW\n', encoding='utf-8')
        self.dump_path = folder / 'board.txt'
        state, _driver, _dump = state_and_driver()
        import game_logic
        game_logic.initialize_decks(state)
        self.rand = ClientRandomness(human_side=gs.Side.FREE_PEOPLES, seed=194)
        self.rand.install()
        self.driver = LiveDriver(state, gs.Side.SHADOW, str(self.feed),
                                 randomness=self.rand, dump_path=str(self.dump_path))
        self.driver.phase = 'actions'
        self.driver.last_actor = gs.Side.SHADOW
        # The real native fixture supplies the complete 428-PID roster. Put
        # its figures into the literal five-unit position used by these tests.
        self.dump = client_board.read_board(str(ROOT / 'data/live_fixtures/client_board_20260821_strider_guide.txt'))
        self.assertEqual(len(self.dump), 428)
        for pid in self.dump:
            nation, kind = client_board.unit_type(pid)
            if kind in ('regular', 'elite', 'leader', 'nazgul'):
                side = gs.Side.SHADOW if kind == 'nazgul' else ai._side_of_nation(nation)
                self.dump[pid] = 'SA Reinforcements' if side is gs.Side.SHADOW else 'FP Reinforcements'
        for pid, zone in build_dump_from_state(state).items():
            if pid in self.dump and zone != 'Spare':
                self.dump[pid] = zone
        self.fp_regulars = [pid for pid, zone in self.dump.items()
                            if board.sanitize(zone) == 'fords of isen'
                            and client_board.unit_type(pid) == (gs.Nation.ROHAN, 'regular')]
        self.shadow_regulars = [pid for pid, zone in self.dump.items()
                                if board.sanitize(zone) == 'orthanc'
                                and client_board.unit_type(pid) == (gs.Nation.ISENGARD, 'regular')]
        self.assertEqual(len(self.fp_regulars), 2)
        self.assertEqual(len(self.shadow_regulars), 2)
        self.write_dump()

    def write_dump(self, *, feed_position=None, complete=True):
        import emitter
        from collections import defaultdict
        groups = defaultdict(list)
        for pid, zone in self.dump.items():
            groups[emitter.zone_code_for(board.sanitize(zone))].append(pid)
        feed_position = len(self.feed.read_text(encoding='utf-8').splitlines()) - 1 if feed_position is None else feed_position
        raw_lines = list(getattr(self.driver, '_raw_request_lines', ()))
        self.driver._raw_request_lines = raw_lines
        lines = ['#SEQ dump=1 feed_paragraphs=%d raw_lines=%d session=9911' %
                 (feed_position, len(raw_lines))]
        for index, (code, pids) in enumerate(groups.items()):
            markers = getattr(self, 'marker_states', {})
            lines.append('%d\t<~%s~>\t%s\t%s\t%s' % (index, code, ' '.join(pids),
                ' '.join('-' for _ in pids), ' '.join(str(markers.get(pid, '-')) for pid in pids)))
        self.dump_path.write_text('\n'.join(lines if complete else lines[:3]) + '\n', encoding='utf-8')

    def lose_regular(self):
        self.dump[self.fp_regulars[0]] = 'FP Dead'
        self.write_dump()

    def assert_single_casualty(self):
        self.assertEqual(len(self.driver.es.armies['fords of isen'].units), 2)
        self.assertEqual(self.driver.es.fp_casualties[gs.Nation.ROHAN]['regular'], 1)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertTrue(self.driver._last_reset_preserved_combat)
        self.assertIs(self.driver.last_actor, gs.Side.SHADOW)
        self.assertFalse(self.driver._last_resync_adoption['complete'])

    def test_reset_recovers_lost_casualty_through_real_commit_and_census(self):
        self.lose_regular()
        items, report = self.driver.reset_and_resync()
        self.assert_single_casualty()
        self.assertEqual(self.driver.commits[-1][2], 'pending:combat-resync')
        self.assertTrue(any('recovered 1 missing' in row for row in report), report)
        self.assertIsNone(self.driver.pending_buf)
        self.assertFalse(any('units fords of isen' in row for row in report), report)
        self.driver.reset_and_resync()
        self.assert_single_casualty()

    def test_full_native_census_recovers_two_hit_elite_removal_once(self):
        from live import client_board
        self.driver.es.pending_combat['atk_hits'] = 2
        elite = next(pid for pid, zone in self.dump.items()
                     if board.sanitize(zone) == 'fords of isen'
                     and client_board.unit_type(pid) == (gs.Nation.ROHAN, 'elite'))
        self.dump[elite] = 'FP Dead'
        self.write_dump()
        self.driver.reset_and_resync()
        self.assertEqual(len(self.driver.es.armies['fords of isen'].units), 2)
        self.assertEqual(self.driver.es.fp_casualties[gs.Nation.ROHAN]['elite'], 1)
        self.assertEqual(self.driver.es.fp_casualties[gs.Nation.ROHAN]['regular'], 0)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1)

    def test_available_feed_gesture_routes_once_before_board_reconstruction(self):
        self.lose_regular()
        with self.feed.open('a', encoding='utf-8') as stream:
            stream.write('$Human kills Rohan Regular in Fords of Isen.\n')
        self.write_dump()
        _items, report = self.driver.reset_and_resync()
        self.assert_single_casualty()
        self.assertNotEqual(self.driver.commits[-1][2], 'pending:combat-resync')
        self.assertEqual(self.driver._feed_seq, 2)
        self.assertEqual(len(self.driver.feed.lines), 2)
        self.driver.reset_and_resync()
        self.assert_single_casualty()

    def test_stale_or_ahead_dump_cannot_duplicate_a_late_gesture(self):
        self.lose_regular()
        self.write_dump(feed_position=1)
        before = pickle.dumps(self.driver.es)
        _items, report = self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertTrue(any('ahead' in row for row in report), report)
        with self.feed.open('a', encoding='utf-8') as stream:
            stream.write('$Human kills Rohan Regular in Fords of Isen.\n')
        self.driver.reset_and_resync()
        self.assert_single_casualty()

    def test_another_client_session_halts_without_touching_the_battle(self):
        from live.driver import DriverHalt
        self.lose_regular()
        self.driver.session_nonce = 8123
        before = pickle.dumps(self.driver.es)
        with self.assertRaises(DriverHalt):
            self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertFalse(self.driver._combat_recovery_active)

    def test_prefetched_and_torn_reads_hold_all_gameplay(self):
        self.lose_regular()
        before = pickle.dumps(self.driver.es)
        self.driver.feed.poll()  # the outer tick has not yet routed its batch
        _items, report = self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertTrue(any('prefetched' in row for row in report), report)
        self.driver._feed_seq = len(self.driver.feed.lines)
        self.write_dump(complete=False)
        _items, report = self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertTrue(any('incomplete or changing' in row for row in report), report)

    def test_failed_recovered_commit_retains_engine_buffer_and_randomness(self):
        self.lose_regular()
        original = self.driver.es
        self.driver.pending_buf = PhaseWindow()
        buffer = self.driver.pending_buf
        self.rand.begin_attempt()
        self.rand._feed['d6'] = [('FREE_PEOPLES', 6), ('SHADOW', 1)]
        self.rand._feed_pos['d6'] = {0}
        before = pickle.dumps(vars(self.rand))
        with patch.object(ai, 'apply_action', side_effect=ValueError('commit failure after legal matching')):
            _items, report = self.driver.reset_and_resync()
        self.assertIs(self.driver.es, original)
        self.assertIs(self.driver.pending_buf, buffer)
        self.assertEqual(pickle.dumps(vars(self.rand)), before)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertFalse(self.driver._combat_recovery_active)
        self.assertTrue(any('not committed' in row for row in report), report)

    def test_failed_existing_buffer_commit_retains_arrived_evidence_and_rolls(self):
        self.lose_regular()
        self.driver.pending_buf = PhaseWindow()
        self.driver.pending_buf.records = [{'op': 'kill', 'pieces': ['Rohan Regular'],
                                           'zone': 'Fords of Isen', 'norm': 'existing casualty'}]
        buffer = self.driver.pending_buf
        self.rand._feed['d6'] = [('FREE_PEOPLES', 6)]
        before = pickle.dumps(vars(self.rand))
        with patch.object(ai, 'apply_action', side_effect=ValueError('commit failed')):
            _items, report = self.driver.reset_and_resync()
        self.assertIs(self.driver.pending_buf, buffer)
        self.assertEqual(len(buffer.records), 1)
        self.assertEqual(pickle.dumps(vars(self.rand)), before)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertTrue(any('could not be committed' in row for row in report), report)

    def test_lost_retreat_reaches_live_commit_without_a_new_die_or_roll(self):
        self.driver.human_side = gs.Side.SHADOW
        self.driver.ai_side = gs.Side.FREE_PEOPLES
        self.driver.es.pending_combat.update(phase='defender_retreat', deciding_side=gs.Side.SHADOW)
        for pid in self.shadow_regulars:
            self.dump[pid] = 'Gap of Rohan'
        self.write_dump()
        _items, report = self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1, report)
        self.assertNotIn('orthanc', self.driver.es.armies)
        army = next(army for key, army in self.driver.es.armies.items()
                    if board.sanitize(key) == 'gap of rohan')
        self.assertEqual(len(army.units), 2)
        self.assertEqual(self.driver.es.pending_combat['phase'], 'victor_advance')

    def test_feed_arriving_during_board_match_is_routed_on_next_attempt(self):
        self.lose_regular()
        original = self.driver.es
        real_plan = recovery.plan
        def arrival(driver, dump):
            result = real_plan(driver, dump)
            with self.feed.open('a', encoding='utf-8') as stream:
                stream.write('$Human kills Rohan Regular in Fords of Isen.\n')
            return result
        with patch.object(recovery, 'plan', side_effect=arrival):
            _items, report = self.driver.reset_and_resync()
        self.assertIs(self.driver.es, original)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertTrue(any('changed while matching' in row for row in report), report)
        self.write_dump()
        self.driver.reset_and_resync()
        self.assert_single_casualty()

    def test_exception_after_engine_apply_rolls_back_ledger_and_randomness(self):
        self.lose_regular()
        original = self.driver.es
        before_rng = pickle.dumps(vars(self.rand))
        before_echo = pickle.dumps(self.driver.echo_ledger)
        hand_models = dict(self.driver.hand_models)
        before_hands = {side: dict(vars(model)) for side, model in hand_models.items()}
        def failed_sweep(*args, **kwargs):
            for model in self.driver.hand_models.values():
                model._mint('character')
            raise ValueError('physical renderer failed')
        with patch.object(self.driver, 'post_commit_sweep', side_effect=failed_sweep):
            _items, report = self.driver.reset_and_resync()
        self.assertIs(self.driver.es, original)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(self.driver.commits, [])
        self.assertEqual(pickle.dumps(self.rand.__dict__), before_rng)
        self.assertEqual(pickle.dumps(self.driver.echo_ledger), before_echo)
        self.assertIs(self.driver.hands.es, original)
        for side, model in hand_models.items():
            self.assertIs(self.driver.hand_models[side], model)
            self.assertEqual(vars(model), before_hands[side])
        self.assertIs(self.driver.view.table_seats, self.driver.table_seats)
        self.assertTrue(any('physical renderer failed' in row for row in report), report)

    def test_routed_feed_commit_sweep_failure_retains_native_evidence_and_state(self):
        self.lose_regular()
        with self.feed.open('a', encoding='utf-8') as stream:
            stream.write('$Human kills Rohan Regular in Fords of Isen.\n')
        self.write_dump()
        original = self.driver.es
        before = pickle.dumps(original)
        rng = pickle.dumps(vars(self.rand))
        hands = {side: dict(vars(model)) for side, model in self.driver.hand_models.items()}
        def failed_sweep(*args, **kwargs):
            self.driver.hands._mint('character')
            raise ValueError('native commit renderer failed')
        with patch.object(self.driver, 'post_commit_sweep', side_effect=failed_sweep):
            _items, report = self.driver.reset_and_resync()
        self.assertIs(self.driver.es, original)
        self.assertEqual(pickle.dumps(original), before)
        self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertEqual(self.driver.commits, [])
        self.assertEqual(len(self.driver.pending_buf.records), 1, report)
        self.assertIn('kills Rohan Regular', self.driver.pending_buf.records[0]['raw'])
        for side, model in self.driver.hand_models.items():
            self.assertEqual(vars(model), hands[side])
        self.assertFalse(getattr(self.driver, '_resync_commit_transaction', False))
        self.driver.reset_and_resync()
        self.assert_single_casualty()

    def test_consumed_round_roll_context_waits_for_casualty_without_false_stall(self):
        from live.live_parser import parse
        self.driver.pending_buf = PhaseWindow()
        record = parse('<game> Human rolls 6')
        self.assertEqual(record['op'], 'roll_num')
        self.driver.pending_buf.records = [record]
        buffer = self.driver.pending_buf
        self.rand._feed['d6'] = [('FREE_PEOPLES', 6)]
        self.rand._feed_pos['d6'] = {0}
        self.rand.step_commit()
        before = pickle.dumps(self.driver.es)
        for _ in range(27):
            self.driver.tick()
        self.assertIs(self.driver.pending_buf, buffer)
        self.assertEqual(buffer.records, [record])
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertFalse(any(kind == 'pending_buffer_stall' for kind, _ in self.driver.findings))
        self.assertIsNone(getattr(self.driver, '_buf_stall', None))
        self.assertNotIn('match none', self.driver.stall_text())
        self.assertIn('1 hit', self.driver.stall_text())
        # A result not yet consumed still deserves the ordinary diagnostic.
        self.rand._feed['d6'] = [('FREE_PEOPLES', 6)]
        self.rand._feed_pos['d6'] = set()
        self.assertIsNone(self.driver._casualty_roll_context_only())

    def test_recovered_declared_payment_waits_for_only_its_required_next_roll(self):
        from test_onslaught_payments import position, choose
        from live import client_board
        import game_logic
        state = position(regulars=2, elites=0)
        game_logic.initialize_decks(state)
        state = ai.apply_action(state, choose(state, count=1))
        self.driver.es = state
        for model in self.driver.hand_models.values():
            model.es = state
        self.driver.human_side, self.driver.ai_side = gs.Side.SHADOW, gs.Side.FREE_PEOPLES
        self.rand.human_side = 'SHADOW'
        elite = next(pid for pid, zone in self.dump.items()
                     if board.sanitize(zone) == 'fords of isen'
                     and client_board.unit_type(pid) == (gs.Nation.ROHAN, 'elite'))
        replacement = next(pid for pid, zone in self.dump.items()
                           if board.sanitize(zone) in ('fp reinf', 'fp reinforcements')
                           and client_board.unit_type(pid) == (gs.Nation.ROHAN, 'regular'))
        self.dump[elite] = 'FP Reinforcements'
        self.dump[replacement] = 'Fords of Isen'
        self.dump[self.shadow_regulars[0]] = 'SA Reinforcements'
        self.write_dump()
        before = pickle.dumps(state)
        items, report = self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.assertIsNotNone(self.driver.pending_buf, report)
        request = self.driver._proven_pending_random_call()
        self.assertIsNotNone(request, report)
        self.assertEqual(request.kw, dict(need=1, ctx='post-casualty Onslaught', side=gs.Side.SHADOW, role='defender'))
        self.assertTrue(any(item[0] == 'rng_ask' and item[1] is request for item in items))
        self.assertTrue(self.driver._last_reset_preserved_combat)


class LiveNoncombatRecovery(unittest.TestCase):
    """Complete native-format board recovery must preserve action ownership."""

    write_dump = LiveCombatRecovery.write_dump

    def setUp(self):
        LiveCombatRecovery.setUp(self)
        self.driver.es.pending_combat = None
        # Establish the native fixture's non-army surfaces via the real adopter.
        # Subsequent tests resync this identical physical board, using the real
        # census to establish that there is no board disagreement to repair.
        _items, report = self.driver.reset_and_resync(hand_turn=False)
        self.assertTrue(self.driver._last_resync_adoption['complete'], report)

    def test_unchanged_real_board_retains_both_possible_actor_states(self):
        for actor in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
            with self.subTest(last_actor=actor):
                self.driver.last_actor = actor
                physical = self.dump_path.read_bytes()
                turn = self.driver.es.turn
                dice = [(die.used, die.current) for side in gs.Side
                        for die in self.driver.es.dice[side]]
                _items, report = self.driver.reset_and_resync()
                self.assertTrue(self.driver._last_resync_adoption['complete'], report)
                self.assertIs(self.driver.last_actor, actor)
                self.assertEqual(self.driver.es.turn, turn)
                self.assertEqual([(die.used, die.current) for side in gs.Side
                                  for die in self.driver.es.dice[side]], dice)
                self.assertEqual(self.dump_path.read_bytes(), physical)
                self.assertEqual(self.driver.n_commits_total, 0)
                self.assertIn('turn order retained', report)

    def test_explicit_handoff_still_requires_complete_adoption(self):
        self.driver.last_actor = gs.Side.SHADOW
        self.write_dump(complete=False)
        self.driver.reset_and_resync(hand_turn=True)
        self.assertIs(self.driver.last_actor, gs.Side.SHADOW)
        self.write_dump()
        self.driver.reset_and_resync(hand_turn=True)
        self.assertIs(self.driver.last_actor, gs.Side.FREE_PEOPLES)

    def test_missing_then_complete_board_cannot_cancel_unanswered_hunt(self):
        hunt = dict(damage=2, phase='hunt_sacrifice', casualty_taken=False)
        self.driver.es.pending_hunt = hunt
        self.driver.last_actor = gs.Side.SHADOW
        before = pickle.dumps(self.driver.es)
        rng = pickle.dumps(vars(self.rand))
        self.write_dump(complete=False)
        self.driver.reset_and_resync()
        self.assertIs(self.driver.es.pending_hunt, hunt)
        self.write_dump()
        items, report = self.driver.reset_and_resync()
        self.assertFalse(self.driver._last_resync_adoption['complete'])
        self.assertIs(self.driver.es.pending_hunt, hunt)
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.assertIs(self.driver.last_actor, gs.Side.SHADOW)
        self.assertTrue(any('pending_hunt' in str(item) and 'cannot prove' in str(item)
                            for item in items), (items, report))

    def test_arrived_hunt_choice_routes_before_board_adoption(self):
        self.driver.es.pending_hunt = dict(
            damage=2, phase='hunt_sacrifice', casualty_taken=False)
        self.dump['252'] = 'Track 2'
        with self.feed.open('a', encoding='utf-8') as stream:
            stream.write('$Human moves Corruption from Fellowship Track 0 to Fellowship Track 2.\n')
        self.write_dump()
        items, report = self.driver.reset_and_resync()
        self.assertIsNone(self.driver.es.pending_hunt, (items, report))
        self.assertEqual(self.driver.es.fellowship.corruption, 2)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertTrue(self.driver._last_resync_adoption['complete'], (items, report))

    def test_failed_adoption_restores_hand_placeholder_counters(self):
        from live import adopt
        before = pickle.dumps(self.driver.es)
        hands = {side: dict(vars(model)) for side, model in self.driver.hand_models.items()}
        # The real hand adopter mints unknown native human cards before this
        # later family fails; its private counter must roll back with the hand.
        with patch.object(adopt, 'adopt_decks', side_effect=ValueError('deck census unavailable')):
            self.driver.reset_and_resync()
        self.assertFalse(self.driver._last_resync_adoption['complete'])
        self.assertEqual(pickle.dumps(self.driver.es), before)
        for side, model in self.driver.hand_models.items():
            self.assertEqual(vars(model), hands[side])

    def companion_pid(self, name):
        import log_replay
        return next(str(pid) for pid in log_replay.COMPANIONS
                    if name in log_replay.names_for_pid(str(pid)))

    def test_lost_guide_selection_is_proved_by_the_single_guide_box_figure(self):
        state = self.driver.es
        state.pending_guide_change = {'candidates': ['Gandalf the Grey', 'Strider']}
        self.dump[self.companion_pid('Gandalf the Grey')] = 'Fellowship Box'
        self.dump[self.companion_pid('Strider')] = 'Guide Box'
        self.write_dump()
        _items, report = self.driver.reset_and_resync()
        self.assertIsNone(self.driver.es.pending_guide_change, report)
        self.assertEqual(self.driver.es.fellowship.guide.name, 'Strider')
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertIs(self.driver.last_actor, gs.Side.SHADOW)
        self.assertFalse(any(item[0] == 'line' for item in _items), _items)

    def test_lost_hunt_corruption_drag_charges_existing_damage_once(self):
        self.driver.es.pending_hunt = dict(damage=2, phase='hunt_sacrifice')
        self.dump['252'] = 'Track 2'
        self.write_dump()
        _items, report = self.driver.reset_and_resync()
        self.assertIsNone(self.driver.es.pending_hunt, report)
        self.assertEqual(self.driver.es.fellowship.corruption, 2)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertFalse(any(key == 'corruption_sweep_emitted' for key, _ in self.driver.findings), _items)
        self.assertNotIn('252', getattr(self.driver, '_track_owed', {}))
        self.driver.reset_and_resync()
        self.assertEqual(self.driver.n_commits_total, 1)

    def test_board_corruption_cannot_increase_the_pending_hunt_damage(self):
        hunt = dict(damage=2, phase='hunt_sacrifice')
        self.driver.es.pending_hunt = hunt
        self.dump['252'] = 'Track 3'
        self.write_dump()
        self.driver.reset_and_resync()
        self.assertIs(self.driver.es.pending_hunt, hunt)
        self.assertEqual(self.driver.es.fellowship.corruption, 0)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_visible_guide_casualty_recovers_its_remaining_hunt_damage(self):
        self.driver.es.pending_hunt = dict(damage=4, phase='hunt_sacrifice')
        self.dump[self.companion_pid('Gandalf the Grey')] = 'FP Dead'
        self.dump[self.companion_pid('Strider')] = 'Guide Box'
        self.dump['252'] = 'Track 1'
        self.write_dump()
        _items, report = self.driver.reset_and_resync()
        self.assertIsNone(self.driver.es.pending_hunt, report)
        self.assertNotIn('Gandalf the Grey', [c.name for c in self.driver.es.fellowship.companions])
        self.assertEqual(self.driver.es.fellowship.guide.name, 'Strider')
        self.assertEqual(self.driver.es.fellowship.corruption, 1)
        self.assertEqual(self.driver.n_commits_total, 1)
        self.assertFalse(any(item[0] == 'line' for item in _items), _items)

    def test_nonguide_casualty_cannot_invent_a_random_companion_draw(self):
        hunt = dict(damage=2, phase='hunt_sacrifice')
        self.driver.es.pending_hunt = hunt
        self.dump[self.companion_pid('Gimli')] = 'FP Dead'
        self.write_dump()
        rng = pickle.dumps(vars(self.rand))
        self.driver.reset_and_resync()
        self.assertIs(self.driver.es.pending_hunt, hunt)
        self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.assertIn('Gimli', [c.name for c in self.driver.es.fellowship.companions])
        self.assertEqual(self.driver.n_commits_total, 0)

    def prepare_reveal(self, *, face):
        self.driver.es.fellowship.hidden = False
        self.driver.es.fellowship.region = self.driver.es.board.region('rivendell')
        self.driver.es.fellowship.progress = 1
        self.driver.es._reveal_place_due = True
        self.dump['247'] = 'Trollshaws'
        self.dump['248'] = 'Track 0'
        if face:
            self.marker_states = {'247': 1}
        self.write_dump()

    def test_reveal_placement_needs_native_face_and_legal_physical_destination(self):
        import game_logic
        self.prepare_reveal(face=True)
        _items, report = self.driver.reset_and_resync()
        self.assertFalse(self.driver.es._reveal_place_due, report)
        self.assertEqual(board.sanitize(str(game_logic.fellowship_region(self.driver.es))), 'trollshaws')
        self.assertEqual(self.driver.es.fellowship.progress, 0)
        self.assertFalse(self.driver.es.fellowship.hidden)
        self.assertEqual(self.driver.n_commits_total, 1)

    def test_reveal_without_native_face_keeps_its_original_placement_choice(self):
        self.prepare_reveal(face=False)
        before = pickle.dumps(self.driver.es)
        self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(self.driver.n_commits_total, 0)

    def test_native_feed_arrival_prevents_board_only_hunt_double_payment(self):
        from live import choice_recovery
        self.driver.es.pending_hunt = dict(damage=2, phase='hunt_sacrifice')
        self.dump['252'] = 'Track 2'
        self.write_dump()
        before = pickle.dumps(self.driver.es)
        real_plan = choice_recovery.plan
        def arrival(*args, **kwargs):
            result = real_plan(*args, **kwargs)
            with self.feed.open('a', encoding='utf-8') as stream:
                stream.write('$Human moves Corruption from Fellowship Track 0 to Fellowship Track 2.\n')
            return result
        with patch.object(choice_recovery, 'plan', side_effect=arrival):
            self.driver.reset_and_resync()
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.write_dump()
        self.driver.reset_and_resync()
        self.assertIsNone(self.driver.es.pending_hunt)
        self.assertEqual(self.driver.es.fellowship.corruption, 2)
        self.assertEqual(self.driver.n_commits_total, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
