"""Independent controls for retained root targets and selector provenance."""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('WOTR_NATIVE', '0')
import numpy as np
import ai
import delta_trainer as dt

PROFILE = 'predecision_profile_v1'


def identity():
    cfg = asdict(ai.SearchSelectConfig())
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    return dict(selector_contract='root_search_target_v1', selector_config=cfg,
                selector_config_sha256=digest)


def literal_row(*, rounded=False, temperature=0., played=1):
    row = dict(action_encoding_contract='casualty_operations_v2', p=np.array([.6, .4]), q=np.array([.1, .1300000001 if rounded else .14],
                                              dtype=np.float32),
               n=np.array([20, 8]), tf=np.array([0., 1.], dtype=np.float16),
               played=played, sm=np.ones(2, dtype=bool),
               ae=np.zeros((2, ai.ACTION_ENCODING_DIM)), ri=np.zeros(2, dtype=int))
    row.update(identity(), pi_sel=np.array([5 / 14, 9 / 14], dtype=np.float32),
               selector_eligible=np.ones(2, dtype=bool),
               selector_temperature=temperature, selector_reward_contract=PROFILE)
    return row


@contextmanager
def corrected(**cfg_changes):
    with patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE), patch.object(
            ai, 'SELECT_CFG', ai.SearchSelectConfig(**cfg_changes)):
        yield


class SelectorContractControls(unittest.TestCase):
    def test_rounding_cannot_recompute_authoritative_target(self):
        with corrected():
            row = literal_row(rounded=True)
            # This exact rounded Q misses the .03 override; the live target
            # was computed before that rounding and is an independent oracle.
            self.assertEqual(ai.select_root_action(row, 0.)[0], 0)
            pi, eligible = dt._row_target(row)
            np.testing.assert_array_equal(pi, row['pi_sel'])
            np.testing.assert_array_equal(eligible, [True, True])

    def test_resolved_config_change_is_rejected_before_retargeting(self):
        with corrected(min_visits=20):
            with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                dt._masked_target(literal_row())

    def test_corrected_row_missing_evidence_is_not_legacy(self):
        row = literal_row()
        for key in list(row):
            if key.startswith('selector_') or key == 'pi_sel':
                del row[key]
        with corrected(), self.assertRaisesRegex(ValueError, '[Ss]elector'):
            dt._row_target(row)

    def test_corrupt_or_partial_authoritative_targets_are_refused(self):
        for changes in ({'pi_sel': [float('nan'), 1.]}, {'pi_sel': [-.1, 1.1]},
                        {'pi_sel': [.2, .2]}, {'pi_sel': [1.]},
                        {'selector_eligible': [1, 1]},
                        {'selector_contract': 'future'},
                        {'selector_config_sha256': 'wrong'},
                        {'selector_temperature': -1.},
                        {'selector_reward_contract': 'legacy_profile_v1'}):
            with self.subTest(changes=changes), corrected():
                row = literal_row()
                row.update(changes)
                with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                    dt._row_target(row)

    def test_temperature_sample_may_differ_from_deterministic_pick(self):
        row = literal_row(temperature=1., played=1)
        row.update(p=np.array([.5, .5]), q=np.zeros(2), n=np.array([6, 4]),
                   fn=np.array([4, 0]), tf=np.zeros(2), pi_sel=np.array([.6, .4]),
                   selector_eligible=np.array([True, False]))
        with corrected():
            self.assertEqual(ai.select_root_action(row, 0.)[0], 0)
            pi, _, _, why = dt._masked_target(row)
            self.assertIsNone(why)
            np.testing.assert_allclose(pi, [.6, .4])

    def test_explicit_legacy_row_keeps_existing_reconstruction(self):
        row = literal_row()
        for key in list(row):
            if key.startswith('selector_') or key == 'pi_sel':
                del row[key]
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT='legacy_profile_v1'):
            expected, elig = ai.select_root_action(row, 0.)[1:]
            actual, mask = dt._row_target(row)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(mask, elig['elig'])

    def test_current_captured_trace_requires_new_contract_only_for_training(self):
        import game_trace
        import trace_admission
        path = ROOT / 'audit/2026-09-04/performance_pass/lookup_baseline_0_20260905/episode_sh_ring_corruption.trace.gz'
        tr = game_trace.load_trace(str(path))
        # Historical files remain readable; current admission does not certify
        # observations encoded before the casualty-budget lanes existed.
        self.assertTrue(tr['steps'])
        historical = trace_admission.validate_trace(tr)
        if tr['meta'].get('observation_contract') != ai.OBSERVATION_CONTRACT:
            self.assertIn('observation_contract', [e['code'] for e in historical['errors']])
        with corrected():
            report = trace_admission.validate_trace(tr, decision_contract=PROFILE)
            self.assertFalse(report['accepted'])
            self.assertTrue(any('selector' in e['code'] for e in report['errors']))

    def test_live_payload_retains_original_target_and_eligibility(self):
        import selector_contract as sc
        row = literal_row(rounded=True)
        info = dict(action_encs=row['ae'], region_indices=row['ri'],
                    priors_clean=row['p'], q_values=row['q'], visit_counts=row['n'],
                    term_frac=row['tf'], played=1, regret=0.,
                    **{key: row[key] for key in sc.EVIDENCE_FIELDS})
        with corrected(), patch.object(ai, '_STORE_SEARCH_STATS', True):
            ss, regret = ai._search_stats_payload(info)
            np.testing.assert_array_equal(ss['pi_sel'], row['pi_sel'])
            np.testing.assert_array_equal(ss['selector_eligible'], row['selector_eligible'])
            self.assertEqual(ss['selector_config'], row['selector_config'])
            self.assertEqual(regret, 0.)
            info['pi_sel'][0] = 0.
            self.assertNotEqual(ss['pi_sel'][0], 0.)
            del info['selector_config']
            with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                ai._search_stats_payload(info)

    def test_buffer_round_trip_preserves_targets_and_refuses_changed_resume(self):
        import run_td_v33loop as loop
        from trace_decisions import current_learning_context
        row = literal_row()
        with corrected(), tempfile.TemporaryDirectory() as folder:
            data = dict(current_learning_context(), **identity(),
                        reward_contract=PROFILE, value=[1.], delta=[row])
            buffer = loop.ReplayBuffer(folder, [('fp', 'ring')])
            buffer.add(0, {('fp', 'ring'): data})
            result = buffer.load(('fp', 'ring'))
            np.testing.assert_array_equal(result['delta'][0]['pi_sel'], row['pi_sel'])
            self.assertEqual(result['selector_config'], row['selector_config'])
            self.assertEqual(result['delta'][0]['selector_temperature'], 0.)
            with patch.object(ai, 'SELECT_CFG', ai.SearchSelectConfig(pi_mode='sigma')):
                changed = loop.ReplayBuffer(folder, [('fp', 'ring')])
                with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                    changed.load(('fp', 'ring'))

    def test_invalid_row_cannot_write_even_an_earlier_valid_cohort(self):
        import run_td_v33loop as loop
        from trace_decisions import current_learning_context
        with corrected(), tempfile.TemporaryDirectory() as folder:
            keys = [('fp', 'ring'), ('sh', 'military')]
            good = dict(current_learning_context(), **identity(), reward_contract=PROFILE,
                        value=[1.], delta=[literal_row()])
            bad = copy.deepcopy(good)
            del bad['delta'][0]['pi_sel']
            buffer = loop.ReplayBuffer(folder, keys)
            with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                buffer.add(0, {keys[0]: good, keys[1]: bad})
            self.assertEqual(list(Path(folder).rglob('b*.pkl.gz')), [])
            with patch.object(loop, 'NET_KEYS', keys), patch.object(loop, 'load_net') as load:
                with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                    loop.train_nets_v2({keys[0]: good, keys[1]: bad}, {})
                load.assert_not_called()

    def test_direct_operator_refuses_before_touching_model(self):
        from unittest.mock import Mock
        net = Mock()
        row = literal_row()
        del row['pi_sel']
        with corrected(), self.assertRaisesRegex(ValueError, '[Ss]elector'):
            dt.train_policy_delta(net, [row])
        net.parameters.assert_not_called()

    def test_experiment_resume_refuses_changed_settings_before_seed_copy(self):
        import run_td_v33loop as loop
        from trace_decisions import current_learning_context
        with corrected(), tempfile.TemporaryDirectory() as folder:
            saved = dict(current_learning_context(), **identity(), reward_contract=PROFILE)
            path = Path(folder) / 'learning_contract.json'
            path.write_text(json.dumps(saved), encoding='utf-8')
            original = path.read_bytes()
            with patch.object(loop, 'LOOP', folder), patch.object(
                    ai, 'SELECT_CFG', ai.SearchSelectConfig(min_visits=20)), patch.object(
                    loop.shutil, 'copy2') as copyfile:
                with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                    loop.setup()
                copyfile.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_synthetic_current_row_target_survives_actual_legal_menu_collection(self):
        import game_trace
        import selector_contract as sc
        path = ROOT / 'audit/2026-09-04/performance_pass/lookup_baseline_0_20260905/episode_sh_ring_corruption.trace.gz'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        tr = game_trace.load_trace(str(path))
        row = tr['steps'][12]
        self.assertEqual(row['action_type'], 'army_move')
        ss = row['search_stats']
        self.assertEqual(np.flatnonzero(ss['n']).tolist(), [0, 24, 31])
        # Build a synthetic current fixture from a captured board. Re-encode
        # every arm and the decision state now; old vectors are correspondence
        # checks only. Expected selector target below is independently pinned.
        from game_state import Side
        from trace_decisions import current_learning_context
        state = row['decision_context']['state']
        legal = ai.get_legal_actions(state, Side.SHADOW)
        current = {(np.asarray(ai.encode_action(a, state, Side.SHADOW), dtype=np.float16).tobytes(),
                    ai._get_action_region_index(a)): a for a in legal}
        arms = []
        for encoded, region in zip(ss['ae'], ss['ri']):
            key = (np.asarray(encoded, dtype=np.float16).tobytes(), int(region))
            self.assertTrue(key in current, 'captured arm lacks a current legal counterpart')
            arms.append(current[key])
        ss['ae'] = np.asarray([ai.encode_action(a, state, Side.SHADOW) for a in arms], dtype=np.float32)
        ss['ri'] = np.asarray([ai._get_action_region_index(a) for a in arms])
        row['decision_context']['encoding'] = ai.encode_state(state, Side.SHADOW)
        tr['meta'].update(current_learning_context())
        target = np.zeros(40, dtype=np.float32)
        target[[0, 24, 31]] = [.4, .2, .4]
        eligible = np.zeros(40, dtype=bool)
        eligible[[0, 31]] = True
        with corrected():
            ss.update(sc.make_evidence(ai.SELECT_CFG, 0., target, eligible=eligible,
                                       reward_contract=PROFILE))
            tr['meta'].update(identity())
            tr['steps'] = [dict(kind='initial', side=None,
                                state=row['decision_context']['state']), row]
            rows, counts = dt.collect_delta_samples(tr, 'sh')
            self.assertEqual(counts['rows'], 1)
            self.assertEqual(counts['skew_dropped'], 0)
            self.assertEqual(counts['error'], 0)
            np.testing.assert_array_equal(rows[0]['pi_sel'], target)
            np.testing.assert_array_equal(dt._row_target(rows[0])[0], target)
            with patch.object(ai, 'SELECT_CFG', ai.SearchSelectConfig(pi_mode='sigma')):
                with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                    dt.collect_delta_samples(tr, 'sh')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_real_small_search_stamps_actual_settings_and_sampling_temperature(self):
        import game_trace
        import game_state as gs
        import selector_contract as sc
        path = ROOT / 'audit/2026-09-04/performance_pass/lookup_baseline_0_20260905/episode_sh_ring_corruption.trace.gz'
        state = game_trace.load_trace(str(path))['steps'][12]['decision_context']['state']
        with corrected(), patch.object(ai, '_TREE_REUSE', False):
            action, info = ai.mcts_search(state, gs.Side.SHADOW, None, None,
                num_simulations=2, temperature=.75, add_noise=False, determinize=False)
            self.assertIsNotNone(action)
            self.assertEqual(info['selector_temperature'], .75)
            self.assertEqual(info['selector_reward_contract'], PROFILE)
            self.assertEqual(info['selector_config'], asdict(ai.SELECT_CFG))
            with patch.object(ai, '_STORE_SEARCH_STATS', True):
                payload, _ = ai._search_stats_payload(info)
            pi, eligible = sc.validate_evidence(payload, expected=sc.current_identity(),
                                                 reward_contract=PROFILE)
            np.testing.assert_array_equal(pi, info['pi_sel'])
            np.testing.assert_array_equal(eligible, info['selector_eligible'])
            self.assertEqual(payload['played'], info['played'])

    def test_checkpoint_writers_and_roster_resume_bind_selector_identity(self):
        from types import SimpleNamespace
        import torch
        import run_td_v33loop as loop
        import value_trainer_v2 as value
        net = SimpleNamespace(value_head=[None], state_dict=lambda: {})
        with corrected(), tempfile.TemporaryDirectory() as folder:
            for writer, name in ((loop.save_net, 'live_fp_ring.pt'),
                                 (value.save_v2_checkpoint, 'value.pt')):
                path = str(Path(folder) / name)
                writer(net, 'fp', path, profile='ring')
                saved = torch.load(path, map_location='cpu', weights_only=False)
                for field, expected in identity().items():
                    self.assertEqual(saved[field], expected)
            with patch.object(loop, 'LOOP', folder), patch.object(loop, 'NET_KEYS', [('fp', 'ring')]), \
                    patch.object(loop, 'FROZEN_CKPT_FP', None), patch.object(loop, 'FROZEN_CKPT_SH', None):
                loop.preflight_generation()
                with patch.object(ai, 'SELECT_CFG', ai.SearchSelectConfig(min_visits=20)):
                    with self.assertRaisesRegex(ValueError, '[Ss]elector'):
                        loop.preflight_generation()


if __name__ == '__main__':
    unittest.main()
