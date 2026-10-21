"""Literal result ledgers are the oracle, independent of reward functions."""
import copy
import json
import unittest

from profile_metrics import summarize_profile_results


def game(winner, route, **extra):
    return dict({'profiles': {'fp': 'ring', 'sh': 'military'},
                 'reward_contract': 'predecision_profile_v1',
                 'checkpoint_ids': {'fp': 'fp-checkpoint-A', 'sh': 'sh-checkpoint-A'},
                 'winner': winner, 'victory_type': route}, **extra)


class ProfileMetricsTests(unittest.TestCase):
    def only(self, records, side='fp'):
        report = summarize_profile_results(records, side=side)
        self.assertEqual(len(report['cohorts']), 1)
        return report['cohorts'][0]

    def test_literal_ledger_uses_decided_attempted_and_win_denominators(self):
        rows = [game('fp', 'ring_destroyed'), game('fp', 'ring_destroyed'),
                game('fp', 'military'), game('sh', 'military'),
                game('sh', 'ring_corruption'), game('timeout', 'timeout'),
                game('fp', 'ring_destroyed', status='rejected'),
                game(None, None, status='error'), game('fp', 'unknown_route')]
        group = self.only(rows)
        self.assertEqual(group['counts'], {
            'attempted': 9, 'decided': 5, 'wins': 3, 'losses': 2,
            'timeouts': 1, 'rejected': 1, 'errors': 2, 'malformed': 1,
            'unreviewed': 6, 'on_target_wins': 2, 'off_target_wins': 1})
        self.assertEqual(group['rates'], {
            'win_rate_decided': {'numerator': 3, 'denominator': 5, 'value': .6},
            'on_target_win_rate_decided': {'numerator': 2, 'denominator': 5, 'value': .4},
            'on_target_win_rate_attempted': {'numerator': 2, 'denominator': 9, 'value': 2/9},
            'off_target_win_rate_wins': {'numerator': 1, 'denominator': 3, 'value': 1/3}})

    def test_every_specialized_profile_has_explicit_target_route(self):
        cases = [
            ('fp', 'ring', 'ring_destroyed', 'military'),
            ('fp', 'military', 'military', 'ring_destroyed'),
            ('sh', 'corruption', 'ring_corruption', 'military'),
            ('sh', 'military', 'military', 'ring_corruption')]
        for side, profile, on_route, off_route in cases:
            with self.subTest(side=side, profile=profile):
                rows = [game(side, on_route), game(side, off_route)]
                for row in rows:
                    row['profiles'][side] = profile
                counts = self.only(rows, side)['counts']
                self.assertEqual((counts['wins'], counts['on_target_wins'],
                                  counts['off_target_wins']), (2, 1, 1))

    def test_mixed_profile_has_no_target_or_off_target_category(self):
        for side, routes in [('fp', ['ring_destroyed', 'military']),
                             ('sh', ['ring_corruption', 'military'])]:
            rows = [game(side, route) for route in routes]
            for row in rows:
                row['profiles'][side] = 'mixed'
            group = self.only(rows, side)
            self.assertEqual(group['counts']['wins'], 2)
            self.assertIsNone(group['counts']['on_target_wins'])
            self.assertIsNone(group['counts']['off_target_wins'])
            self.assertEqual(group['rates']['win_rate_decided']['value'], 1.0)
            for name in ('on_target_win_rate_decided', 'on_target_win_rate_attempted',
                         'off_target_win_rate_wins'):
                self.assertIsNone(group['rates'][name])

    def test_unknown_impossible_and_missing_routes_are_errors_not_losses(self):
        for winner, route in [('fp', 'ring_corruption'), ('sh', 'ring_destroyed'),
                              ('fp', None), ('timeout', 'military'),
                              ('draw', 'timeout'), ('FP', 'military')]:
            with self.subTest(winner=winner, route=route):
                counts = self.only([game(winner, route)])['counts']
                self.assertEqual((counts['decided'], counts['losses'], counts['errors']), (0, 0, 1))

    def test_profile_aliases_are_not_inferred(self):
        for player, profile in [('fp', 'corruption'), ('sh', 'ring'),
                                ('fp', 'Ring'), ('sh', 'mil'), ('fp', None)]:
            row = game('fp', 'military')
            row['profiles'][player] = profile
            report = summarize_profile_results([row], side='fp')
            self.assertEqual(report['cohorts'][0]['counts']['errors'], 1)
            self.assertIn('profile_label', [d['code'] for d in report['diagnostics']])

    def test_integrity_rejection_never_counts_as_legitimate_loss_or_win(self):
        rejected = {'accepted': False, 'valid': False,
                    'errors': [{'code': 'source_action_exec_failed'}]}
        rows = [game('sh', 'military', admission=rejected),
                game('fp', 'ring_destroyed', admission=rejected)]
        counts = self.only(rows)['counts']
        self.assertEqual((counts['decided'], counts['wins'], counts['losses'],
                          counts['rejected']), (0, 0, 0, 2))

    def test_admission_exclusion_for_intact_timeout_is_still_timeout(self):
        row = game('timeout', 'timeout', admission={
            'accepted': False, 'valid': True, 'errors': [{'code': 'truncated'}]})
        counts = self.only([row])['counts']
        self.assertEqual((counts['timeouts'], counts['rejected'], counts['unreviewed']), (1, 0, 0))

    def test_inconsistent_admission_is_error(self):
        for admission in [
                {'accepted': True, 'valid': False, 'errors': []},
                {'accepted': True, 'errors': [{'code': 'source_action_exec_failed'}]},
                {'accepted': False, 'valid': True, 'errors': [{'code': 'truncated'}]},
                {'accepted': 'false'}, {'accepted': False, 'errors': 'bad'}]:
            with self.subTest(admission=admission):
                counts = self.only([game('fp', 'ring_destroyed', admission=admission)])['counts']
                self.assertEqual((counts['errors'], counts['decided']), (1, 0))

    def test_explicit_error_does_not_inherit_claimed_win(self):
        counts = self.only([game('fp', 'ring_destroyed', error='worker crashed')])['counts']
        self.assertEqual((counts['wins'], counts['errors'], counts['malformed']), (0, 1, 0))

    def test_empty_denominators_are_unknown_not_zero_percent(self):
        group = self.only([game(None, None, status='error')])
        self.assertEqual(group['rates']['win_rate_decided'],
                         {'numerator': 0, 'denominator': 0, 'value': None})
        self.assertEqual(group['rates']['off_target_win_rate_wins'],
                         {'numerator': 0, 'denominator': 0, 'value': None})
        self.assertEqual(group['rates']['on_target_win_rate_attempted'],
                         {'numerator': 0, 'denominator': 1, 'value': 0.0})
        self.assertEqual(summarize_profile_results([], side='sh')['cohorts'], [])

    def test_cohorts_do_not_merge_changed_profiles_opponents_or_contracts(self):
        rows = [game('fp', 'ring_destroyed') for _ in range(6)]
        rows[1]['profiles']['fp'] = 'military'
        rows[2]['profiles']['sh'] = 'corruption'
        rows[3]['checkpoint_ids']['sh'] = 'sh-checkpoint-B'
        rows[4]['reward_contract'] = 'predecision_winloss_v1'
        rows[5].pop('checkpoint_ids')
        report = summarize_profile_results(rows, side='fp')
        self.assertEqual(len(report['cohorts']), 6)
        self.assertNotIn('rates', report)
        self.assertEqual(sum(g['counts']['attempted'] for g in report['cohorts']), 6)
        unknown = next(g for g in report['cohorts'] if g['cohort']['opponent_id'] is None)
        self.assertFalse(unknown['opponent_identity_known'])
        self.assertFalse(unknown['cohort_complete'])

    def test_opponent_checkpoint_selected_from_other_side_and_explicit_override(self):
        row = game('sh', 'military')
        group = self.only([row], 'sh')
        self.assertEqual(group['cohort']['opponent_id'], 'fp-checkpoint-A')
        row['opponent_id'] = 'fixed-arena-opponent'
        self.assertEqual(self.only([row], 'sh')['cohort']['opponent_id'], 'fixed-arena-opponent')

    def test_missing_contract_stays_unknown(self):
        row = game('fp', 'ring_destroyed')
        row.pop('reward_contract')
        group = self.only([row])
        self.assertIsNone(group['cohort']['reward_contract'])
        self.assertFalse(group['cohort_complete'])

    def test_malformed_record_is_counted_and_output_is_json_serializable(self):
        report = summarize_profile_results([None, game('fp', 'ring_destroyed')], side='fp')
        self.assertEqual(report['attempted'], 2)
        self.assertEqual(sum(g['counts']['errors'] for g in report['cohorts']), 1)
        json.dumps(report, allow_nan=False)

    def test_does_not_mutate_inputs_and_accepts_one_pass_iterators(self):
        rows = [game('fp', 'ring_destroyed'), game('sh', 'military')]
        original = copy.deepcopy(rows)
        report = summarize_profile_results(iter(rows), side='fp')
        self.assertEqual(rows, original)
        self.assertEqual(report['attempted'], 2)

    def test_programmer_argument_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            summarize_profile_results([], side='FP')
        with self.assertRaises(ValueError):
            summarize_profile_results(game('fp', 'military'), side='fp')


if __name__ == '__main__':
    unittest.main()
