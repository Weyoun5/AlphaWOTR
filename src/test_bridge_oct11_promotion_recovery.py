"""Exact native character faces repair missed promotions without a new roll."""
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
import game_logic as gl
import game_state as gs
import board
from live import adopt, client_board
import test_bridge_oct11_combat_recovery as fixtures


class PromotionRecovery(unittest.TestCase):
    write_dump = fixtures.LiveNoncombatRecovery.write_dump

    def setUp(self):
        fixtures.LiveNoncombatRecovery.setUp(self)
        self.marker_states = {'240': 0, '241': 0, '247': 0}
        self.write_dump()

    def prepare_gandalf(self, *, eliminated=False, face=1):
        state = self.driver.es
        grey = next(c for c in state.fellowship.companions
                    if c.name == 'Gandalf the Grey')
        state.fellowship.companions.remove(grey)
        state.fellowship.guide = next(c for c in state.fellowship.companions
                                     if c.name == 'Strider')
        if not eliminated:
            gl.get_or_create_army(state, 'grey havens').leaders.append(grey)
        self.dump['240'] = 'Grey Havens'
        self.dump['241'] = 'Guide Box'
        self.marker_states['240'] = face
        # Early native bonus bookkeeping does not grant another die this turn.
        self.dump['210'] = 'FP Dead'
        self.dump['358'] = 'FP Dead'
        state.dice[gs.Side.FREE_PEOPLES][0].used = True
        state.dice[gs.Side.FREE_PEOPLES][0].current = gs.ActionDieFace.WILL_OF_WEST
        self.write_dump()
        return grey

    def recover(self):
        items, report = self.driver.reset_and_resync()
        self.assertTrue(self.driver._last_resync_adoption['complete'], (items, report))
        return items, report

    def white(self):
        return next(c for c in self.driver.es.armies['grey havens'].leaders
                    if c.name == 'Gandalf the White')

    def test_native_face_exposes_wrong_identity_at_same_location(self):
        self.prepare_gandalf()
        rows = client_board.diff_engine_vs_dump(
            self.driver.es, self.dump, marker_states=self.marker_states)
        self.assertTrue(any('identity' in row and 'Gandalf the White' in row
                            for row in rows), rows)
        self.recover()
        white = self.white()
        self.assertFalse(white.captain_of_the_west)
        self.assertEqual(white.bonus_movement, 1)
        self.assertTrue(white.solo_hobbit_movement)
        self.assertIsNone(white.guide_ability)

    def test_eliminated_grey_restores_white_and_next_turn_capacity_once(self):
        self.prepare_gandalf(eliminated=True)
        state = self.driver.es
        dice = [(d.used, d.current) for d in state.dice[gs.Side.FREE_PEOPLES]]
        actor, turn = self.driver.last_actor, state.turn
        rng = pickle.dumps(vars(self.rand))
        self.recover()
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertEqual(self.white().leadership, 1)
        self.assertEqual([(d.used, d.current) for d in state.dice[gs.Side.FREE_PEOPLES]], dice)
        self.assertEqual(len(state.dice[gs.Side.FREE_PEOPLES]), 4)
        self.assertEqual((self.driver.last_actor, state.turn), (actor, turn))
        self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.assertEqual(self.driver.n_commits_total, 0)
        self.recover()
        self.assertEqual(len(state.dice[gs.Side.FREE_PEOPLES]), 4)
        gl.calculate_bonus_dice(state)
        gl.calculate_bonus_dice(state)
        self.assertEqual(len(state.dice[gs.Side.FREE_PEOPLES]), 5)

    def test_white_shining_face_only_proves_white_identity(self):
        self.prepare_gandalf(eliminated=True, face=2)
        self.recover()
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertEqual(self.white().name, 'Gandalf the White')
        self.assertIsNone(self.driver.es.pending_combat)

    def test_aragorn_face_repairs_traits_without_replaying_politics(self):
        state = self.driver.es
        strider = next(c for c in state.fellowship.companions if c.name == 'Strider')
        state.fellowship.companions.remove(strider)
        gl.get_or_create_army(state, 'pelargir').leaders.append(strider)
        self.dump['241'] = 'Pelargir'
        self.marker_states['241'] = 1
        self.write_dump()
        politics = pickle.dumps(state.political)
        self.recover()
        aragorn = next(c for c in state.armies['pelargir'].leaders if c.name == 'Aragorn')
        self.assertEqual(aragorn.leadership, 2)
        self.assertTrue(aragorn.captain_of_the_west)
        self.assertIsNone(aragorn.guide_ability)
        self.assertTrue(state.aragorn_crowned.crowned)
        self.assertEqual(pickle.dumps(state.political), politics)
        self.assertEqual(len(state.dice[gs.Side.FREE_PEOPLES]), 4)

    def test_later_failure_restores_original_identity_and_promotion_flags(self):
        grey = self.prepare_gandalf()
        state = self.driver.es
        before = pickle.dumps(state)
        with patch.object(adopt, 'adopt_decks', side_effect=ValueError('later adoption failure')):
            self.driver.reset_and_resync()
        self.assertFalse(self.driver._last_resync_adoption['complete'])
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertIs(self.driver.es.armies['grey havens'].leaders[0], grey)
        self.assertFalse(self.driver.es.gandalf_the_white.in_play)

    def test_known_promotion_cannot_be_unremembered_by_grey_face(self):
        self.prepare_gandalf()
        self.recover()
        self.marker_states['240'] = 0
        self.write_dump()
        before = pickle.dumps(self.driver.es)
        items, report = self.driver.reset_and_resync()
        self.assertFalse(self.driver._last_resync_adoption['complete'])
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertTrue(self.driver.es.gandalf_the_white.in_play)
        self.assertIn('history', str((items, report)))

    def test_dead_white_records_entry_without_resurrecting_or_granting_a_die(self):
        self.prepare_gandalf(eliminated=True)
        self.dump['240'] = 'FP Dead'
        self.write_dump()
        self.recover()
        state = self.driver.es
        self.assertTrue(state.gandalf_the_white.in_play)
        self.assertFalse(any(c.name.startswith('Gandalf') for army in state.armies.values()
                             for c in army.leaders))
        state.saruman.was_ever_in_play = True
        self.assertFalse(gl.can_transform_gandalf_white(state)[0])
        gl.calculate_bonus_dice(state)
        self.assertEqual(len(state.dice[gs.Side.FREE_PEOPLES]), 4)
        self.assertEqual(state.fp_bonus_dice, 0)

    def test_unknown_or_missing_face_cannot_reconstruct_an_eliminated_character(self):
        self.prepare_gandalf(eliminated=True, face=7)
        for markers in ({'240': 7, '241': 0}, {'241': 0}):
            with self.subTest(markers=markers):
                self.marker_states = markers
                self.write_dump()
                before = pickle.dumps(self.driver.es)
                self.driver.reset_and_resync()
                self.assertFalse(self.driver._last_resync_adoption['complete'])
                self.assertEqual(pickle.dumps(self.driver.es), before)
                self.assertFalse(self.driver.es.gandalf_the_white.in_play)

    def test_promoted_face_in_reserve_does_not_invent_an_entry_history(self):
        self.prepare_gandalf(eliminated=True)
        self.dump['240'] = 'FP Reinforcements'
        self.write_dump()
        before = pickle.dumps(self.driver.es)
        items, report = self.driver.reset_and_resync()
        self.assertFalse(self.driver._last_resync_adoption['complete'])
        self.assertEqual(pickle.dumps(self.driver.es), before)
        self.assertIn('history', str((items, report)))

    def test_promoted_character_cannot_be_adopted_into_fellowship(self):
        self.prepare_gandalf(eliminated=True)
        self.dump['240'] = 'Fellowship Box'
        self.write_dump()
        before = pickle.dumps(self.driver.es)
        self.driver.reset_and_resync()
        self.assertFalse(self.driver._last_resync_adoption['complete'])
        self.assertEqual(pickle.dumps(self.driver.es), before)

    def test_ordinary_reconciliation_does_not_replay_the_missing_promotion(self):
        grey = self.prepare_gandalf()
        report, _before, after = adopt.adopt_client_board(
            self.driver.es, client_board.read_board(str(self.dump_path)),
            dump_path=str(self.dump_path), manual=False)
        self.assertGreater(after, 0, report)
        self.assertIs(self.driver.es.armies['grey havens'].leaders[0], grey)
        self.assertFalse(self.driver.es.gandalf_the_white.in_play)

    def test_public_face_repairs_missing_promotion_flag_for_correct_existing_character(self):
        self.prepare_gandalf()
        self.recover()
        state = self.driver.es
        white = self.white()
        state.gandalf_the_white.in_play = False
        self.recover()
        self.assertIs(self.white(), white)
        self.assertTrue(state.gandalf_the_white.in_play)

    def test_direct_adoption_rollback_restores_flag_without_driver_outer_transaction(self):
        grey = self.prepare_gandalf()
        state = self.driver.es
        before = pickle.dumps(state)
        with patch.object(adopt, 'adopt_decks', side_effect=ValueError('later adopter')):
            report, _, _ = adopt.adopt_client_board(
                state, client_board.read_board(str(self.dump_path)),
                dump_path=str(self.dump_path), manual=True)
        self.assertTrue(any('rolled back' in row for row in report), report)
        self.assertEqual(pickle.dumps(state), before)
        self.assertIs(state.armies['grey havens'].leaders[0], grey)

    def stale_staff_after_missed_separation(self):
        import cards
        from persistent_effects import create_persistent_card
        state = self.driver.es
        card = cards.get_card("FREE_PEOPLES_Character_Wizard's_Staff")
        self.assertEqual(card.card_name, "Wizard's Staff")
        for deck in (state.fp_char_deck, state.fp_strat_deck,
                     state.hands[gs.Side.FREE_PEOPLES].hand):
            if card.card_id in deck:
                deck.remove(card.card_id)
        state.persistent_cards.add_card(create_persistent_card(
            card.card_id, card.card_name, gs.Side.FREE_PEOPLES))
        # This fixture's previously public Staff binds an actual native card
        # piece. Its title is never inferred from the PID or a hidden deck.
        pid = next(pid for pid, zone in self.dump.items()
                   if board.sanitize(zone) == 'fp char deck')
        self.driver.view.card_pid_map[card.card_id] = pid
        self.dump[pid] = 'FP Dead'  # No longer occupies a native Board slot.
        self.dump['240'] = 'Grey Havens'
        self.dump['241'] = 'Guide Box'
        self.marker_states['240'] = 1
        self.dump['210'] = self.dump['358'] = 'FP Dead'
        state.dice[gs.Side.FREE_PEOPLES][0].used = True
        self.write_dump()
        self.assertEqual(client_board.read_board(str(self.dump_path))[pid], 'FP Dead')
        self.assertEqual(state.fellowship.guide.name, 'Gandalf the Grey')
        return card.card_id

    def test_missed_separation_and_promotion_remove_stale_staff_without_casualty_effects(self):
        cid = self.stale_staff_after_missed_separation()
        state = self.driver.es
        rng = pickle.dumps(vars(self.rand))
        dice = [(d.used, d.current) for d in state.dice[gs.Side.FREE_PEOPLES]]
        items, report = self.recover()
        self.assertFalse(state.persistent_cards.has_card("Wizard's Staff"))
        self.assertEqual(state.discard_face_up['fp_character'].count(cid), 1)
        self.assertEqual(state.fellowship.guide.name, 'Strider')
        self.assertEqual(self.white().name, 'Gandalf the White')
        self.assertEqual(pickle.dumps(vars(self.rand)), rng)
        self.assertEqual([(d.used, d.current) for d in state.dice[gs.Side.FREE_PEOPLES]], dice)
        self.assertFalse(getattr(state, '_wwsat_owed', None))
        self.assertIsNone(getattr(state, 'pending_card_option', None))
        self.assertIn("Wizard's Staff", str((items, report)))
        self.recover()
        self.assertEqual(state.discard_face_up['fp_character'].count(cid), 1)

    def test_stale_staff_cleanup_and_discard_rollback_with_failed_later_adoption(self):
        self.stale_staff_after_missed_separation()
        state = self.driver.es
        before = pickle.dumps(state)
        guide = state.fellowship.guide
        observed = []
        def fail_after_cleanup(*args, **kwargs):
            observed.append(state.persistent_cards.has_card("Wizard's Staff"))
            raise ValueError('after required-companion cleanup')
        with patch.object(adopt, 'adopt_decks', side_effect=fail_after_cleanup):
            report, _, _ = adopt.adopt_client_board(
                state, client_board.read_board(str(self.dump_path)),
                dump_path=str(self.dump_path), manual=True)
        self.assertEqual(observed, [False])
        self.assertTrue(any('rolled back' in row for row in report), report)
        self.assertEqual(pickle.dumps(state), before)
        self.assertIs(state.fellowship.guide, guide)


if __name__ == '__main__':
    unittest.main()
