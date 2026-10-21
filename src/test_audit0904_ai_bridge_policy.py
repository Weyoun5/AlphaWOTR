"""AI-only policy enforcement at live entry and returned-choice boundaries.

Human legal menus retain Regular/Elite and optional draw/decline. AI menus
prefer Elite where legal and take an available guide draw. A caller's
truncated menu, a network result, or a stale journal cannot bypass that.
"""
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'wotr_bridge'), str(ROOT)]
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
import ai
import test_audit0904_stewards_choices as house
import test_audit0904_stewards_bridge as flow
import test_audit0904_recruit_choices as guards
from live.live_server import LiveServer
from live.driver import DriverHalt


class LiveFixedPolicies(unittest.TestCase):
    def server(self, state):
        temp = tempfile.TemporaryDirectory(prefix='wotr-ai-policy-')
        self.addCleanup(temp.cleanup)
        (Path(temp.name) / 'ai_feed.txt').write_text('', encoding='utf-8')
        srv = LiveServer(temp.name, house.FP, load_nets=False, seed=17)
        srv.driver.es = state
        srv.driver.hands.es = state
        srv.driver.phase = 'actions'
        return srv

    def house_pending(self):
        state = house.before_play()
        return ai.apply_action(state, ai.Action(0, 'play_event',
            {'card_id': house.CID, 'side': 'fp'}))

    def test_ai_projects_elite_without_removing_human_regular(self):
        srv = self.server(self.house_pending())
        human = ai.get_legal_actions(srv.driver.es, house.FP)
        self.assertEqual({a.params['choices']['unit_type'] for a in human},
                         {'regular', 'elite'})
        dp = srv.driver.ai_action_dp()
        self.assertEqual({a.params['choices']['unit_type'] for a in dp['actions']}, {'elite'})

    def test_ai_projects_take_without_removing_human_decline(self):
        srv = self.server(flow.prospective_gandalf_state())
        human = ai.get_legal_actions(srv.driver.es, house.FP)
        self.assertEqual({a.params['choices']['gandalf_draw'] for a in human}, {True, False})
        dp = srv.driver.ai_action_dp()
        self.assertEqual({a.params['choices']['gandalf_draw'] for a in dp['actions']}, {True})

    def test_truncated_regular_menu_cannot_hide_an_available_elite(self):
        srv = self.server(self.house_pending())
        dp = ai.describe_pending(srv.driver.es)
        regular, = [a for a in dp['actions'] if a.params['choices']['unit_type'] == 'regular']
        with self.assertRaises(DriverHalt):
            srv._ai_decide(dict(dp, actions=[regular]))
        self.assertEqual(srv.driver.n_commits_total, 0)
        self.assertEqual(house.figures(srv.driver.es), (3, 1, 1))

    def test_truncated_decline_menu_cannot_hide_an_available_draw(self):
        srv = self.server(flow.prospective_gandalf_state())
        dp = ai.describe_pending(srv.driver.es)
        decline, = [a for a in dp['actions'] if a.params['choices']['gandalf_draw'] is False]
        with self.assertRaises(DriverHalt):
            srv._ai_decide(dict(dp, actions=[decline]))
        self.assertEqual(srv.driver.n_commits_total, 0)
        self.assertEqual(srv.driver.es.pending_card_option['card_id'], flow.GANDALF_ID)

    def test_network_cannot_return_a_policy_forbidden_regular(self):
        srv = self.server(self.house_pending())
        dp = ai.describe_pending(srv.driver.es)
        regular, = [a for a in dp['actions'] if a.params['choices']['unit_type'] == 'regular']
        srv.fp_net = srv.sh_net = 'INJECTED_POLICY_PROBE'
        with patch.object(ai, 'mcts_action_with_policy', return_value=(regular, None)):
            with self.assertRaises(DriverHalt):
                srv._ai_decide(dp)
        self.assertEqual(srv.driver.n_commits_total, 0)

    def test_stale_journal_cannot_return_a_policy_forbidden_decline(self):
        srv = self.server(flow.prospective_gandalf_state())
        dp = ai.describe_pending(srv.driver.es)
        decline, = [a for a in dp['actions'] if a.params['choices']['gandalf_draw'] is False]
        with patch.object(srv.rand, 'journal_pick', return_value=[decline]):
            with self.assertRaises(DriverHalt):
                srv._ai_decide(dp)
        self.assertEqual(srv.driver.n_commits_total, 0)
        self.assertEqual(srv.driver.es.pending_card_option['card_id'], flow.GANDALF_ID)

    def test_ai_uses_regular_from_real_pool_when_all_spare_elites_are_dead(self):
        import emitter
        import log_replay
        state = guards.before_play()
        guards.exhaust_gondor_pool(state, 'elite')
        srv = self.server(state)
        view = srv.driver.view
        positions = json.loads((ROOT / 'wotr_bridge/initial_positions.json').read_text())
        view.piece_region = {pid: log_replay.decode_location(zone) for pid, zone in positions.items()}
        # Four spare Gondor Elites are dead; the fifth is the initial MT
        # Elite. The Regular277 and Leader270 remain in FP Reinforcements.
        for pid in ('273', '274', '275', '276'):
            view.piece_region[pid] = 'FP Casualties'
        view.card_pid_map[guards.CID] = '130'
        view.piece_region['130'] = 'Free Peoples Hand'
        view.piece_region['210'] = 'FP Dice'
        play = ai.Action(0, 'play_event', {'card_id': guards.CID, 'side': 'fp'})
        output = srv._ai_decide({'side': house.FP, 'kind': 'action', 'actions': [play]})
        moves = [emitter.classify_line(emitter.strip_tail(line)) for line in output]
        moves = [fields for op, fields in moves if op == 'move' and fields.get('frm') == 'Game.2038']
        self.assertEqual({tuple(fields['pieces']) for fields in moves}, {('277',), ('270',)})
        self.assertTrue(all(fields['to'] == 'Game.1944' for fields in moves))
        self.assertEqual(guards.gondor_figures(srv.driver.es), (4, 1, 2))
        self.assertEqual(srv.driver.es.discard_face_up['fp_strategy'].count(guards.CID), 1)
        self.assertEqual([die.used for die in srv.driver.es.dice[house.FP]], [True, False])
        self.assertEqual(srv.driver.alarms, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
