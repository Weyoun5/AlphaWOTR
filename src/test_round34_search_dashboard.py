"""Headless readers for the search runner's exact dashboard publication schema."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import td_dashboard as dashboard


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class SearchDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='round34_search_dashboard_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.feed = self.run / 'loop/all_games.jsonl'
        self.settings = {'search_mode': 'afterstate', 'generations': 4, 'learner_device': 'cpu'}
        write(self.run / 'config.json', {'schema': 'search_run_v1', 'config': self.settings})
        write(self.root / 'data/active_training_pipeline.json', {
            'pipeline_status_path': str(self.run / 'pipeline_status.json'), 'selfplay_feed': str(self.feed)})
        write(self.run / 'latest.json', {'path': 'checkpoints/synthetic'})
        write(self.run / 'checkpoints/synthetic/bundle.json', {'metadata': {'next_batch': 1}})
        for side in ('fp', 'sh'):
            write(self.run / ('checkpoints/synthetic/' + side + '/complete.json'), {'sha256': '1' * 64})
        write(self.feed, {'batch': 'batch_000', 'winner': 'fp', 'episode': 'synthetic-0'})
        write(self.feed.parent / 'dashboard_feed_contract.json', {
            'version': 'committed_served_games_v1', 'publication': 'committed_only'})
        self.enterContext(patch.object(dashboard, 'ROOT', str(self.root)))
        self.enterContext(patch.dict(os.environ, {'WOTR_LOOP_DIR': str(self.feed.parent)}))
        self.enterContext(patch.object(dashboard.tk, 'Tk', side_effect=AssertionError('No GUI may start')))
        self.enterContext(patch.object(dashboard.subprocess, 'Popen', side_effect=AssertionError('No subprocess may start')))
        self.enterContext(patch.multiple(dashboard, LOOP_DIR=dashboard.LOOP_DIR, LOOP_CONFIG_PATH=dashboard.LOOP_CONFIG_PATH,
                                       SCRIPT_LOG=dashboard.SCRIPT_LOG, CARD_INDEX_PATH=dashboard.CARD_INDEX_PATH,
                                       LEGACY_CONTROLS_ENABLED=dashboard.LEGACY_CONTROLS_ENABLED))
        dashboard.bind_loop_paths(dashboard._active_pipeline()['selfplay_feed'])

    def publish_status(self, status, stamp=None):
        write(self.run / 'pipeline_status.json', {'schema': 'actor_critic_status_v1',
            'algorithm': 'gumbel_search_target_v1', 'status': status,
            'stage': 'complete' if status == 'completed' else 'selfplay',
            'detail': 'Synthetic search boundary', 'pid': os.getpid(),
            'updated_at': time.time() if stamp is None else stamp,
            'supervisor_status_path': str(self.run / 'pipeline_status.json')})

    def test_current_search_config_routes_to_its_owned_status_models_and_settings(self):
        self.publish_status('running')
        self.assertEqual(dashboard._actor_critic_config(), self.settings)
        status = dashboard._collect_status({})
        self.assertTrue(status['alive'])
        self.assertEqual(status['pipeline']['status'], 'running')
        self.assertNotIn('DOWN', status['state'])
        self.assertEqual(len(status['batches']), 1)
        self.assertEqual({row[0] for row in status['nets']}, {'FP', 'SH'})
        self.assertTrue(all(row[3] == 'Search resume' for row in status['nets']))
        self.assertIn('1 paired Search batches committed', status['loop_state'])
        self.assertIn('selfplay_actor_critic.search_runner --resume', status['config'][0])
        self.assertIn('search_mode = "afterstate"', status['settings'])
        self.assertFalse(any('arena_enabled' in line for line in status['settings']))

    def test_saved_stopped_failed_completed_and_stale_statuses_remain_nonlive(self):
        for saved in ('stopped', 'failed', 'completed'):
            with self.subTest(status=saved):
                self.publish_status(saved)
                status = dashboard._collect_status({})
                self.assertFalse(status['alive'])
                self.assertEqual(status['pipeline']['status'], saved)
        self.publish_status('running', time.time() - 120)
        status = dashboard._collect_status({})
        self.assertFalse(status['alive'])
        self.assertIn('stale', status['pipeline']['status'])

    def test_legacy_schema_retains_ppo_labels_and_unknown_schema_is_not_guessed(self):
        self.publish_status('completed')
        write(self.run / 'config.json', {'schema': 'actor_critic_run_v1', 'config': {'generations': 1}})
        status = dashboard._collect_status({})
        self.assertTrue(all(row[3] == 'PPO resume' for row in status['nets']))
        self.assertIn('selfplay_actor_critic.runner --resume', status['config'][0])
        write(self.run / 'config.json', {'schema': 'unrecognized_future_run_v1', 'config': self.settings})
        self.assertIsNone(dashboard._actor_critic_config())
        self.assertIsNone(dashboard._actor_critic_run())


if __name__ == '__main__':
    unittest.main()
