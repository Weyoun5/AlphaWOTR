"""Native completion Undo, including real unanswered/answered Dead Men requests."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import game_logic as gl
import game_state as gs
import test_bridge_oct12_card_boundary_adversarial as rage
import test_bridge_oct12_dead_men_prefix_live as dead
from live.live_server import LiveServer


class CompletionUndo(unittest.TestCase):
    def rage(self):
        fixture = rage.CardBoundaryAdversarial()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture, fixture.case

    def undo(self, fixture, line, moves=None):
        return fixture.literal('$Wey undoes last action : ' + line.lstrip('$'),
                               (), moves, driver_only=True)

    def test_card_close_undo_extends_original_group_without_repaying(self):
        fixture, case = self.rage()
        case.native(103, 107)
        used = [d.used for d in case.driver.es.dice[gs.Side.SHADOW]]
        self.undo(fixture, '$Wey kills Rage of the Dunlendings/Relentless Assault in Current SA Card.',
            {'185': 'Current SA Card'})
        self.assertEqual(case.regulars('gap of rohan'), 2)
        self.assertIsNotNone(ai.describe_pending(case.driver.es))
        self.assertEqual([d.used for d in case.driver.es.dice[gs.Side.SHADOW]], used)
        case.native(104, 107, driver_only=True)
        self.assertEqual(case.regulars('gap of rohan'), 4)
        self.assertEqual(case.regulars('south dunland'), 0)
        self.assertFalse(case.driver.alarms)
        case.assert_native_census()

    def test_die_close_undo_extends_and_recloses(self):
        fixture, case = self.rage()
        case.native(103, 108, driver_only=True)
        self.undo(fixture, '$Wey kills Shadow Action Die (M) in Current Action.', {'102': 'Current Action'})
        self.assertIsNotNone(ai.describe_pending(case.driver.es))
        self.assertEqual(case.regulars('gap of rohan'), 2)
        case.native(104, 108, driver_only=True)
        self.assertEqual(case.regulars('gap of rohan'), 4)
        self.assertFalse(case.driver.alarms)
        case.assert_native_census()

    def test_grouped_card_and_die_close_undo_restores_one_native_window(self):
        fixture, case = self.rage()
        case.native(103)
        close = '$Wey kills group (Shadow Action Die (M), Rage of the Dunlendings/Relentless Assault) in Current Action.'
        fixture.literal(close, ('102', '185'), {'102': 'SA Casualties', '185': 'SA Casualties'},
                        driver_only=True)
        self.undo(fixture, close, {'102': 'Current Action', '185': 'Current SA Card'})
        self.assertEqual(case.regulars('gap of rohan'), 2)
        self.assertIsNotNone(case.driver.seg.window)
        case.native(104, driver_only=True)
        fixture.literal(close, ('102', '185'), {'102': 'SA Casualties', '185': 'SA Casualties'},
                        driver_only=True)
        self.assertEqual(case.regulars('gap of rohan'), 4)
        self.assertFalse(case.driver.alarms)
        case.assert_native_census()

    def test_cleanup_die_undo_keeps_choice_then_card_undo_reopens(self):
        fixture, case = self.rage()
        case.native(103, 107, 108, driver_only=True)
        self.undo(fixture, '$Wey kills Shadow Action Die (M) in Current Action.', {'102': 'Current Action'})
        self.assertEqual(case.regulars('gap of rohan'), 3)
        self.assertIsNone(ai.describe_pending(case.driver.es))
        self.undo(fixture, '$Wey kills Rage of the Dunlendings/Relentless Assault in Current SA Card.',
                  {'185': 'Current SA Card'})
        self.assertEqual(case.regulars('gap of rohan'), 2)
        case.native(104, 107, 108, driver_only=True)
        self.assertEqual(case.regulars('gap of rohan'), 4)
        self.assertFalse(case.driver.alarms)
        case.assert_native_census()

    def test_done_then_native_move_undo_reopens_the_logical_choice(self):
        fixture, case = self.rage()
        case.native(103)
        fixture.literal('<Wey> done', driver_only=True)
        self.undo(fixture, '$Wey moves Isengard Regular from North Dunland to Gap of Rohan.',
                  {'001': 'North Dunland'})
        self.assertEqual(case.regulars('gap of rohan'), 2)
        self.assertIsNotNone(ai.describe_pending(case.driver.es))
        case.native(104, driver_only=True)
        fixture.literal('<Wey> done', driver_only=True)
        self.assertEqual(case.regulars('gap of rohan'), 3)
        self.assertEqual(case.regulars('north dunland'), 1)
        self.assertEqual(case.regulars('south dunland'), 0)
        self.assertFalse(case.driver.alarms)
        case.assert_native_census()

    def test_cleanup_undo_invalidates_thinking_ai_step_without_rewinding_choice(self):
        fixture, case = self.rage()
        case.native(103, 107, 108, driver_only=True)
        post = case.driver.es
        case.server._open_step = ('ai_decide', None)
        case.server.rand.begin_attempt()
        before_epoch = case.driver.undo_epoch
        fixture.literal('$Wey undoes last action : Wey kills Shadow Action Die (M) in Current Action.',
                        (), {'102': 'Current Action'})
        self.assertIs(case.driver.es, post)
        self.assertIsNone(case.server._open_step)
        self.assertFalse(case.server.rand.step_open)
        self.assertGreater(case.driver.undo_epoch, before_epoch)
        self.assertIsNotNone(case.driver.seg.window)
        self.assertFalse(case.driver.alarms)

    def test_completion_undo_does_not_rebase_a_committed_ai_reply(self):
        fixture, case = self.rage()
        case.native(103, 107, 108, driver_only=True)
        action = next(a for a in ai.get_legal_actions(case.driver.es, gs.Side.FREE_PEOPLES)
                      if a.action_type == 'army_move')
        post = ai.apply_action(case.driver.es, action)
        case.driver._snapshot([], 'ai:action', post=post, action=action)
        case.driver.es = post
        for model in case.driver.hand_models.values():
            model.es = post
        before_commits = case.driver.n_commits_total
        self.undo(fixture, '$Wey kills Shadow Action Die (M) in Current Action.', {'102': 'Current Action'})
        self.assertIs(case.driver.es, post)
        self.assertEqual(case.driver.n_commits_total, before_commits)
        self.assertTrue(any('CARD CHOICE UNDO HELD' in alarm for alarm in case.driver.alarms))
        self.assertFalse(any(finding[0] == 'undo_rebased' for finding in case.driver.findings))

    def test_empty_decline_can_be_undone_then_choose_a_move(self):
        fixture, case = self.rage()
        case.native(107)
        self.undo(fixture, '$Wey kills Rage of the Dunlendings/Relentless Assault in Current SA Card.',
                  {'185': 'Current SA Card'})
        self.assertIsNotNone(ai.describe_pending(case.driver.es))
        case.native(104, 107, driver_only=True)
        self.assertEqual(case.regulars('gap of rohan'), 3)
        self.assertFalse(case.driver.alarms)
        case.assert_native_census()

    def dead_server(self):
        fixture = dead.DeadMenPrefixes()
        rig = fixture.position(recruit=False)
        self.addCleanup(fixture.doCleanups)
        enemy = gl.get_or_create_army(rig.drv.es, 'erech')
        enemy.units = [gs.Unit(gs.Nation.SAURON, kind)
                       for kind in (gs.UnitKind.REGULAR, gs.UnitKind.ELITE)]
        rig.drv.es.political.boxes[gs.Nation.SAURON] = 3
        server = LiveServer(rig.td, dead.SH, load_nets=False, seed=7)
        server.driver.es = rig.drv.es
        for model in server.driver.hand_models.values():
            model.es = server.driver.es
        server.driver.phase = 'actions'
        server.driver.last_actor = dead.FP
        server.driver.enforce_human = True
        server.driver.view.piece_region.update({'241': 'Westemnet', '243': 'Westemnet',
                                                '011': 'Erech', '057': 'Erech'})
        feed = Path(rig.td)/'ai_feed.txt'
        def physical(line):
            with feed.open('a', encoding='utf8') as stream:
                stream.write(line+'\n')
            return server.handle({'lines': ()})
        physical('$Wey moves Strider from Westemnet to Erech.')
        physical('$Wey moves Gimli from Westemnet to Erech.')
        return server, physical

    def test_unanswered_card_close_undo_cancels_request_then_reclose_asks_once(self):
        server, physical = self.dead_server()
        close = '$Wey kills Dead Men of Dunharrow in Current FP Card.'
        output = physical(close)
        self.assertEqual(sum('roll 1 die #' in line for line in output), 1)
        old_nonce = next(iter(server._live_asks))
        self.assertEqual(server.driver.n_commits_total, 0)
        physical('<Wey> done')  # Repeated confirmation must retain the physical closure witness.
        self.assertEqual(list(server._live_asks), [old_nonce])
        output = physical('$Wey undoes last action : ' + close.lstrip('$'))
        self.assertFalse(any('roll 1 die' in line for line in output))
        self.assertEqual(server._live_asks, {})
        self.assertIsNone(server.rand.pending())
        self.assertFalse(server.driver.pending_buf._staged_card_closed)
        for _ in range(2):
            self.assertFalse(any('roll 1 die' in line for line in server.handle({'lines': ()})))
        output = physical(close)
        self.assertEqual(sum('roll 1 die #' in line for line in output), 1)
        self.assertEqual(len(server._live_asks), 1)
        self.assertGreater(next(iter(server._live_asks)), old_nonce)
        self.assertEqual(server.driver.n_commits_total, 0)
        self.assertFalse(server.driver.alarms)

    def test_unanswered_done_then_move_undo_requires_new_completion(self):
        server, physical = self.dead_server()
        physical('<Wey> done')
        output = physical('$Wey undoes last action : Wey moves Gimli from Westemnet to Erech.')
        self.assertEqual(server._live_asks, {})
        self.assertFalse(any('roll 1 die' in line for line in output))
        self.assertFalse(server.driver.pending_buf._staged_card_closed)
        physical('$Wey moves Gimli from Westemnet to Erech.')
        self.assertEqual(server._live_asks, {})
        output = physical('<Wey> done')
        self.assertEqual(sum('roll 1 die #' in line for line in output), 1)
        self.assertFalse(server.driver.alarms)

    def test_answered_event_roll_is_not_erased_by_native_completion_undo(self):
        server, physical = self.dead_server()
        # Driver receives the real outcome, but this test deliberately leaves
        # the learned Shadow casualty choice to the normal server's next tick.
        close = '$Wey kills Dead Men of Dunharrow in Current FP Card.'
        physical(close)
        feed = Path(server.driver.feed.path)
        with feed.open('a', encoding='utf8') as stream:
            stream.write('<game> Wey rolls 1\n')
        server.driver.tick()
        post = server.driver.es
        self.assertTrue(post.pending_card_option['card_casualty'])
        self.assertIs(ai.describe_pending(post)['side'], dead.SH)
        committed = server.driver.n_commits_total
        with feed.open('a', encoding='utf8') as stream:
            stream.write('$Wey undoes last action : ' + close.lstrip('$')+'\n')
        output = server.driver.tick()
        self.assertIs(server.driver.es, post)
        self.assertEqual(server.driver.n_commits_total, committed)
        self.assertIsNone(server.rand.pending())
        self.assertFalse(any(item[0] == 'rng_ask' for item in output))
        self.assertTrue(server.driver.es.pending_card_option['card_casualty'])
        self.assertTrue(server.driver.alarms)  # Native reversal needs correction, never a reroll.

    def test_cancelled_close_then_late_roll_never_commits_an_open_choice(self):
        server, physical = self.dead_server()
        close = '$Wey kills Dead Men of Dunharrow in Current FP Card.'
        physical(close)
        physical('$Wey undoes last action : ' + close.lstrip('$'))
        output = physical('<game> Wey rolls 1')
        self.assertEqual(server.driver.n_commits_total, 0)
        self.assertFalse(server.driver.pending_buf._staged_card_closed)
        self.assertFalse(any('roll 1 die' in line for line in output))
        self.assertTrue(server.rand._feed['d6'])
        # The already produced outcome is retained; closing again consumes it
        # rather than requesting a replacement outcome.
        feed = Path(server.driver.feed.path)
        with feed.open('a', encoding='utf8') as stream:
            stream.write(close+'\n')
        output = server.driver.tick()
        self.assertEqual(server.driver.n_commits_total, 1)
        self.assertTrue(server.driver.es.pending_card_option['card_casualty'])
        self.assertFalse(any(item[0] == 'rng_ask' for item in output))


if __name__ == '__main__':
    unittest.main()
