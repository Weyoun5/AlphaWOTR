"""Dead Men human arrival, real roll, and authoritative Shadow aftermath."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'wotr_bridge'), str(ROOT)]
import ai
import board
import game_logic as gl
import game_state as gs
from live.live_server import LiveServer
import test_bridge_oct12_dead_men_prefix_live as fixtures
FP, SH = fixtures.FP, fixtures.SH


class DeadMenForcedLive(unittest.TestCase):
    def position(self, units=3, leaders=False, multiple_retreats=False):
        base = fixtures.DeadMenPrefixes()
        self.addCleanup(base.doCleanups)
        rig = base.position(recruit=False)
        state = rig.drv.es
        enemy = gl.get_or_create_army(state, 'erech')
        enemy.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR) for _ in range(units)]
        state.political.boxes[gs.Nation.SAURON] = 3
        if leaders:
            enemy.leaders = [gs.Nazgul(gs.Nation.SAURON),
                             gl.Minion(gs.Nation.SAURON, 'Witch King')]
            state.witch_king.in_play = state.witch_king.was_ever_in_play = True
        if multiple_retreats:
            state.enemy_controlled[SH].add('lamedon')
        server = LiveServer(rig.td, SH, load_nets=False, seed=7)
        server.driver.es = state
        for model in server.driver.hand_models.values():
            model.es = state
        server.driver.phase = 'actions'
        server.driver.last_actor = FP
        server.driver.enforce_human = True
        server.driver.view.piece_region.update({'241': 'Westemnet', '243': 'Westemnet',
            **{str(11+i).zfill(3): 'Erech' for i in range(units)}})
        if leaders:
            server.driver.view.piece_region.update({'037': 'Erech', '253': 'Erech'})
        feed = Path(rig.td)/'ai_feed.txt'

        def physical(*lines, driver_only=False):
            with feed.open('a', encoding='utf8') as stream:
                stream.write('\n'.join(lines)+'\n')
            return server.driver.tick() if driver_only else server.handle({'lines': ()})

        return server, physical

    def arrive(self, server, physical):
        output = physical('$Wey moves Strider from Westemnet to Erech.')
        self.assertFalse(any('roll 1 die' in line for line in output))
        output = physical('$Wey moves Gimli from Westemnet to Erech.')
        self.assertFalse(any('roll 1 die' in line for line in output))
        self.assertEqual(server.driver.n_commits_total, 0)
        output = physical('<Wey> done')
        self.assertEqual(sum('roll 1 die' in line for line in output), 1)
        self.assertEqual(server.driver.n_commits_total, 0)

    def test_unique_casualty_and_retreat_emit_once_after_the_real_roll(self):
        server, physical = self.position()
        self.arrive(server, physical)
        response = physical('<game> Wey rolls 1')
        self.assertEqual(server.driver.n_commits_total, 1)
        self.assertEqual(len(server.driver.es.armies['anfalas'].units), 2)
        self.assertEqual([o['recruit_count'] for o in server.driver.es.pending_card_option['options']], [0, 1, 2, 3])
        operations = [line for line in response if 'Scribe.141' in line or 'Controls.296' in line]
        self.assertEqual(len(operations), 3, response)
        self.assertFalse(any('241 ' in line or '243 ' in line for line in operations))
        self.assertFalse(any('roll ' in line for line in response))
        for _ in range(3):
            repeated = server.handle({'lines': ()})
            self.assertFalse(any('Scribe.141' in line or 'Controls.296' in line or 'roll ' in line
                                 for line in repeated), repeated)
            self.assertEqual(server.driver.n_commits_total, 1)
        self.assertFalse(server.driver.alarms)

    def test_wiped_army_emits_nazgul_and_minion_losses(self):
        server, physical = self.position(units=1, leaders=True)
        self.arrive(server, physical)
        response = physical('<game> Wey rolls 1')
        self.assertEqual(server.driver.n_commits_total, 1)
        self.assertFalse(server.driver.es.witch_king.in_play)
        self.assertEqual({leader.name for leader in server.driver.es.armies['erech'].leaders}, {'Strider', 'Gimli'})
        operations = [line for line in response if 'Scribe.141' in line or 'Controls.296' in line]
        self.assertEqual(len(operations), 3, response)
        for pid in ('011', '037', '253'):
            self.assertTrue(any('>'+pid+' ' in line for line in operations), (pid, operations))
        self.assertFalse(any('241 ' in line or '243 ' in line for line in operations))
        self.assertFalse(server.driver.alarms)

    def test_missing_native_nazgul_source_refuses_figure_emission(self):
        server, physical = self.position(units=1, leaders=True)
        server.driver.view.piece_region['037'] = 'Moria'
        self.arrive(server, physical)
        response = physical('<game> Wey rolls 1')
        self.assertTrue(any('FORCED AI ROWS HOLD' in alarm for alarm in server.driver.alarms),
                        server.driver.alarms)
        self.assertFalse(any('Scribe.141' in line or 'Controls.296' in line for line in response), response)

    def test_generic_arrivals_cancelling_named_departures_do_not_become_losses(self):
        server, _ = self.position(units=1)
        pre = ai.clone_state(server.driver.es)
        post = ai.clone_state(pre)
        post.armies['erech'].leaders = post.armies['westemnet'].leaders
        post.armies['westemnet'].leaders = [gs.Nazgul(gs.Nation.SAURON) for _ in range(2)]
        from live import emission
        rows = emission.removal_rows(pre, post, 'shadow')
        self.assertEqual(rows['naz'], {'minus': [], 'plus': [('westemnet', 2)]})
        self.assertEqual(rows['named'], {'leaves': {}, 'joins': {}})
        self.assertFalse(rows['unexplained'])
        reverse = emission.removal_rows(post, pre, 'shadow')
        self.assertEqual(reverse['naz'], {'minus': [('westemnet', 2)], 'plus': []})
        self.assertEqual(reverse['named'], {'leaves': {}, 'joins': {}})
        self.assertFalse(reverse['unexplained'])

    def test_unique_casualty_preserves_multiple_shadow_retreat_choices(self):
        server, physical = self.position(multiple_retreats=True)
        self.arrive(server, physical)
        physical('<game> Wey rolls 1', driver_only=True)
        pending = ai.describe_pending(server.driver.es)
        self.assertEqual(server.driver.n_commits_total, 1)
        self.assertIs(pending['side'], SH)
        self.assertEqual({a.params['choices']['destination'] for a in pending['actions']}, {'anfalas', 'lamedon'})
        self.assertEqual(len(server.driver.es.armies['erech'].units), 2)
        self.assertNotIn('anfalas', server.driver.es.armies)
        self.assertNotIn('lamedon', server.driver.es.armies)

    def test_wrong_voiced_shadow_move_does_not_receive_forced_credit(self):
        server, physical = self.position()
        self.arrive(server, physical)
        physical('$Wey moves Sauron Regular from Erech to Lamedon.', driver_only=True)
        physical('<game> Wey rolls 1', driver_only=True)
        self.assertEqual(server.driver.n_commits_total, 0)
        self.assertEqual(len(server.driver.es.armies['erech'].units), 3)
        self.assertIsNotNone(server.driver.pending_buf)

    def test_incomplete_human_group_does_not_hide_wrong_destination(self):
        server, physical = self.position()
        physical('$Wey moves Strider from Westemnet to Erech.',
                 '$Wey moves Gimli from Westemnet to Lamedon.', '<Wey> done', driver_only=True)
        physical('<game> Wey rolls 1', driver_only=True)
        self.assertEqual(server.driver.n_commits_total, 0)
        self.assertEqual(len(server.driver.es.armies['erech'].units), 3)
        self.assertIsNotNone(server.driver.pending_buf)


if __name__ == '__main__':
    unittest.main()
