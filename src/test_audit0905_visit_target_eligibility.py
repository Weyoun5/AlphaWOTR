"""Literal support/evidence controls for visit-target learning admission.

Targets are specified independently; no policy model, action encoder, game
episode or optimizer supplies the expected answers.
"""
import os
import unittest
from unittest.mock import patch
import numpy as np

os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import ai
import delta_trainer as dt


class VisitTargetEligibility(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, WOTR_REWARD_CONTRACT='predecision_profile_v1')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = patch.object(ai, 'SELECT_CFG', ai.SearchSelectConfig(pi_mode='visit_tf'))
        self.config.start()
        self.addCleanup(self.config.stop)

    def row(self, n=(2, 1, 1), **extra):
        k = len(n)
        row = {'p': np.full(k, 1 / k), 'q': np.zeros(k),
               'n': np.array(n, dtype=float), 'sm': np.ones(k, dtype=bool), **extra}
        # Current producer schema; the assertions below remain literal target
        # oracles, independently checking both selector formula and admission.
        from selector_contract import make_evidence
        played, target, diag = ai.select_root_action(row, 0.)
        row['played'] = played
        row.update(make_evidence(ai.SELECT_CFG, 0., target, eligible=diag['elig'],
                                 reward_contract='predecision_profile_v1'))
        return row

    def assert_target(self, row, expected, indices=None):
        pi, keep, _, why = dt._masked_target(row)
        self.assertIsNone(why)
        np.testing.assert_allclose(pi, expected, rtol=0, atol=5e-8)
        np.testing.assert_array_equal(keep, list(range(len(expected))) if indices is None else indices)

    def test_multiple_earned_arms_teach_despite_unique_maximum_below_eight(self):
        self.assert_target(self.row(), [.5, .25, .25])
        self.assert_target(self.row((3, 1, 0)), [.75, .25, 0])

    def test_tied_maxima_keep_identical_target(self):
        self.assert_target(self.row((2, 2, 0)), [.5, .5, 0])

    def test_unique_upper_bound_proof_teaches_against_unresolved_alternatives(self):
        self.assert_target(self.row((1, 0, 0), pv=[1, 0, 0], pvv=[1., 0., 0.]), [1, 0, 0])

    def test_exact_proof_can_be_evidence_without_playout_visits(self):
        self.assert_target(self.row((0, 0), pv=[1, 0], pvv=[1., 0.]), [1, 0])

    def test_off_target_win_and_unvalued_proof_do_not_bypass_visit_gate(self):
        for extra in ({'pv': [1, 0], 'pvv': [.2, 0.]}, {'pv': [1, 0]}):
            with self.subTest(extra=extra):
                self.assertEqual(dt._masked_target(self.row((1, 0), **extra))[-1], 'elig')

    def test_all_proven_losses_keep_existing_one_hot_compatibility(self):
        self.assert_target(self.row((1, 1, 1), pv=[-1, -1, -1], pvv=[-1., -1., -1.]), [1, 0, 0])

    def test_single_visited_arm_and_true_single_legal_choice_remain_excluded(self):
        for row in (self.row((4, 0, 0)), self.row((4,)),
                    self.row((1,), pv=[1], pvv=[1.])):
            with self.subTest(row=row):
                self.assertEqual(dt._masked_target(row)[-1], 'elig')

    def test_zero_visits_cannot_manufacture_uniform_learning_evidence(self):
        self.assertEqual(dt._masked_target(self.row((0, 0, 0)))[-1], 'elig')

    def test_forced_exploration_visits_not_distilled_as_two_supported_arms(self):
        self.assertEqual(dt._masked_target(self.row((4, 3, 0), fn=[0, 3, 0]))[-1], 'elig')
        self.assertEqual(dt._masked_target(self.row((2, 2, 0), fn=[2, 2, 0]))[-1], 'elig')

    def test_selector_earned_visit_convention_preserved_for_played_arm(self):
        # Most-visited play keeps raw n; the other arm contributes earned n.
        self.assert_target(self.row((4, 3, 0), fn=[3, 1, 0]), [2 / 3, 1 / 3, 0])

    def test_stale_supported_arm_cannot_satisfy_minimum(self):
        self.assertEqual(dt._masked_target(self.row((2, 1, 0), sm=[True, False, True]))[-1], 'elig')

    def test_stale_proof_does_not_transfer_its_claim_to_another_action(self):
        self.assertEqual(dt._masked_target(self.row((1, 1, 0), sm=[False, True, True],
                                                   pv=[1, 0, 0], pvv=[1., 0., 0.]))[-1], 'elig')

    def test_stale_masking_keeps_renormalized_supported_target(self):
        self.assert_target(self.row((2, 1, 1), sm=[True, True, False]), [2 / 3, 1 / 3], [0, 1])

    def test_proof_requires_an_actual_matched_alternative_or_regenerated_extra(self):
        row = self.row((1, 0), pv=[1, 0], pvv=[1., 0.], sm=[True, False])
        self.assertEqual(dt._masked_target(row)[-1], 'elig')
        row['extra_ae'] = np.zeros((1, ai.ACTION_ENCODING_DIM))
        self.assert_target(row, [1.], [0])

    def test_fixed_policy_row_is_refused_even_if_fake_visits_are_attached(self):
        with self.assertRaisesRegex(ValueError, 'forced|fixed'):
            dt._masked_target(self.row(policy_forced=True, forced_policy_reason='bonus_draw_always_take'))

    def test_malformed_evidence_is_refused_before_selector_or_loss(self):
        cases = [dict(n=[2, -1, 1]), dict(n=[2, np.nan, 1]), dict(n=[2, np.inf, 1]),
                 dict(n=[2, .5, 1]), dict(n=[[2, 1, 1]]), dict(fn=[3, 0, 0]),
                 dict(fn=[0, -1, 0]), dict(fn=[0, 0]), dict(p=[1, np.nan, 0]),
                 dict(q=[0, np.inf, 0]), dict(p=[0, 0, 0]), dict(p=[1, 0]),
                 dict(pv=[2, 0, 0]), dict(pv=[1]), dict(pvv=[1., float('nan'), 0.]),
                 dict(tf=[0, 2, 0]), dict(sm=[True, False]), dict(sm=[1, np.nan, 0])]
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    row = self.row()
                    row.update(malformed)
                    dt._masked_target(row)

    def test_sigma_mode_keeps_old_eligibility_and_exact_target(self):
        with patch.object(ai, 'SELECT_CFG', ai.SearchSelectConfig(pi_mode='sigma')):
            self.assertEqual(dt._masked_target(self.row((2, 1, 1)))[-1], 'elig')
            self.assertEqual(dt._masked_target(self.row((1, 0, 0), pv=[1, 0, 0], pvv=[1., 0., 0.]))[-1], 'elig')
            self.assert_target(self.row((2, 2, 0)), [.5, .5, 0])


if __name__ == '__main__':
    unittest.main()
