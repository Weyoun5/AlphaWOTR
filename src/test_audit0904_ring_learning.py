"""Elven Ring observations and search support with literal rule oracles.

No model fitting or data writes. A normal ring changes one die; the same
player then uses the changed face for an ordinary action (rulebook21).
"""
import unittest
from unittest.mock import patch
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import ai
import board
import game_state as gs
import game_trace
import delta_trainer
import run_td_v33loop as loop
import trace_admission
from reward_contract import WINLOSS, LEGACY

if os.environ.get('WOTR_AUDIT_RING_AI_BASELINE'):
    spec = importlib.util.spec_from_file_location('ring_audit_baseline_ai',
        os.environ['WOTR_AUDIT_RING_AI_BASELINE'])
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    ai = baseline

CONTEXT = {'observation_contract': 'v44_fellowship_tail_v1',
           'action_encoding_contract': 'casualty_operations_v2',
           'ai_policy_contract': 'elite_and_bonus_draw_v1'}


FP, SH = gs.Side.FREE_PEOPLES, gs.Side.SHADOW
FACE = gs.ActionDieFace


def position():
    state = gs.setup_initial_game(board.load_board())
    state.fp_elven_rings, state.sh_elven_rings = 2, 1
    for side in (FP, SH):
        die = state.dice[side][0]
        die.current, die.used = FACE.MUSTER, False
        state.dice[side] = [die]
    return state


def ring(face, side=FP):
    return ai.Action(-1, 'elven_ring', {'target_die_idx': 0,
        'old_face': FACE.MUSTER.value, 'new_face': face.value,
        'side': 'fp' if side == FP else 'sh'})


class RingObservationTests(unittest.TestCase):
    def test_used_this_turn_is_visible_to_both_observers(self):
        for tag, actor in (('fp', FP), ('sh', SH)):
            available = position()
            used = ai.clone_state(available)
            setattr(used, tag + '_ring_used_this_turn', True)
            self.assertTrue(any(a.params.get('new_face') == FACE.CHARACTER.value
                                for a in ai._get_elven_ring_actions(available, actor)))
            self.assertEqual(ai._get_elven_ring_actions(used, actor), [])
            for observer in (FP, SH):
                with self.subTest(actor=tag, observer=observer):
                    self.assertFalse(np.array_equal(ai.encode_state(available, observer),
                                                    ai.encode_state(used, observer)))

    def test_ring_availability_changes_reuse_and_chance_signatures(self):
        for tag in ('fp', 'sh'):
            available = position()
            used = ai.clone_state(available)
            setattr(used, tag + '_ring_used_this_turn', True)
            for signature in (ai._state_sig, ai._chance_sig):
                with self.subTest(actor=tag, signature=signature.__name__):
                    self.assertNotEqual(signature(available), signature(used))

    def test_actual_character_and_event_conversions_have_distinct_value_inputs(self):
        for actor in (FP, SH):
            with self.subTest(actor=actor):
                before = position()
                character = ai.apply_action(before, ring(FACE.CHARACTER, actor))
                event = ai.apply_action(before, ring(FACE.EVENT, actor))
                # Literal die results are the independent physical rule oracle.
                self.assertEqual(character.dice[actor][0].current, FACE.CHARACTER)
                self.assertEqual(event.dice[actor][0].current, FACE.EVENT)
                self.assertFalse(np.array_equal(ai.encode_state(character, actor),
                                                ai.encode_state(event, actor)))

    def test_conversion_action_encoding_preserves_both_old_and_new_faces(self):
        state = position()
        character, event = ring(FACE.CHARACTER), ring(FACE.EVENT)
        old_army = ring(FACE.CHARACTER)
        old_army.params['old_face'] = FACE.ARMY.value
        encs = [ai.encode_action(action, state, FP)
                for action in (character, event, old_army)]
        self.assertEqual(len({row.tobytes() for row in encs}), 3)
        self.assertAlmostEqual(float(encs[0][23]), 1 / 7)
        self.assertAlmostEqual(float(encs[0][24]), 3 / 7)


class RingSearchSupportTests(unittest.TestCase):
    def test_candidate_guarantee_keeps_three_low_prior_ring_arms_when_enabled(self):
        # Same category as rings: category balancing alone cannot protect them.
        actions = [ai.Action(-1, 'companion_move', {'fixture': i}) for i in range(50)]
        actions += [ring(FACE.CHARACTER) for _ in range(4)]
        priors = np.array([.02] * 50 + [.0001] * 4)
        ordinary = ai._prune_candidates(actions, priors, 40, free_guarantee=False)
        protected = ai._prune_candidates(actions, priors, 40, free_guarantee=True)
        self.assertEqual(sum(i >= 50 for i in ordinary), 0)
        self.assertEqual(sum(i >= 50 for i in protected), 3)

    def test_ring_exploration_floor_is_not_automatically_a_policy_target(self):
        # Ring has20 raw visits,12 forced exploration visits. Other arm21.
        played, pi, _ = ai.select_root_action(dict(p=[.0001, .9999],
            q=[.5, .5], n=[20, 21], fn=[12, 0]), 0., ai.SearchSelectConfig())
        self.assertEqual(played, 1)
        np.testing.assert_allclose(pi, [8 / 29, 21 / 29])

    def test_search_can_follow_ring_then_same_player_army_move(self):
        state = position()
        seen = []
        def prior(st, actor, net, actions):
            wanted = [i for i, a in enumerate(actions)
                      if actor == FP and a.action_type == 'elven_ring'
                      and a.params.get('new_face') == FACE.CHARACTER.value]
            if not wanted:
                wanted = [i for i, a in enumerate(actions)
                          if actor == FP and a.action_type == 'army_move'
                          and str(a.params.get('from', '')).lower() == 'minas tirith'
                          and str(a.params.get('to', '')).lower() == 'lossarnach'
                          and a.params.get('n_regular') == 1
                          and a.params.get('n_elite') == 0
                          and a.params.get('take_leaders', True)]
            weights = np.zeros(len(actions))
            weights[wanted[0] if wanted else 0] = 1.
            return weights, 0.
        original = ai.apply_action
        def apply(st, action):
            after = original(st, action)
            seen.append((action, after))
            return after
        with patch.object(ai, '_get_priors_and_value', side_effect=prior), \
                patch.object(ai, '_mcts_eval_state', return_value=0.), \
                patch.object(ai, 'apply_action', side_effect=apply), \
                patch.object(ai, '_TREE_REUSE', False), \
                patch.object(ai, '_LEAF_BATCH', 1), \
                patch.object(ai, '_FREE_FORCE', 0):
            selected, stats = ai.mcts_search(state, FP, None, None,
                num_simulations=2, temperature=0., add_noise=False, determinize=False)
        self.assertEqual(selected.action_type, 'elven_ring')
        self.assertEqual([a.action_type for a, _ in seen], ['elven_ring', 'army_move'])
        self.assertEqual(ai._next_acting_side(seen[0][1], FP), FP)
        self.assertEqual(ai._next_acting_side(seen[1][1], FP), SH)
        army, = [army for key, army in seen[1][1].armies.items()
                 if str(key).lower() == 'lossarnach']
        self.assertEqual(len(army.units), 1)
        self.assertEqual(len(army.leaders), 1)
        self.assertEqual(int(stats['visit_counts'].sum()), 2)
        self.assertFalse(state.dice[FP][0].used)


class RingObservationContractTests(unittest.TestCase):
    def trace(self):
        initial = position()
        final = ai.clone_state(initial)
        final.fellowship.corruption = 12
        return dict(meta=dict(winner='sh', victory_type='ring_corruption',
            source_integrity_version=1, source_errors=[], **CONTEXT),
            steps=[dict(kind='initial', turn=1, state=initial, side=None),
                   dict(kind='action', turn=1, state=final, side='FP',
                        action_type='pass', params={})])

    def test_new_recorder_persists_current_observation_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'current.trace.gz')
            tracer = game_trace.TraceRecorder(path)
            tracer.record_initial(position())
            tracer.save(winner='timeout', victory_type='timeout')
            loaded = game_trace.load_trace(path)
        self.assertEqual(loaded['meta'].get('observation_contract'), CONTEXT['observation_contract'])

    def test_fresh_admission_requires_explicit_both_context_contracts(self):
        trace = self.trace()
        self.assertTrue(trace_admission.validate_trace(trace, audit_rules=False,
                                                      **CONTEXT)['accepted'])
        for field in CONTEXT:
            with self.subTest(field=field):
                old = copy.deepcopy(trace)
                del old['meta'][field]
                # The historical diagnostic is not retroactively relabeled.
                self.assertTrue(trace_admission.validate_trace(old, audit_rules=False)['accepted'])
                self.assertFalse(trace_admission.validate_trace(old, audit_rules=False,
                                                               **CONTEXT)['accepted'])

    def test_unknown_observation_version_is_refused_by_admission_and_collectors(self):
        trace = self.trace()
        trace['meta'].update(observation_contract='future-v999', reward_contract=WINLOSS)
        self.assertFalse(trace_admission.validate_trace(trace, audit_rules=False)['accepted'])
        with self.assertRaisesRegex(ValueError, 'observation contract'):
            loop._replay_reward_stream(trace, {'fp': 'ring', 'sh': 'military'})
        with self.assertRaisesRegex(ValueError, 'observation contract'):
            delta_trainer.collect_delta_samples(trace, 'fp')
        with self.assertRaisesRegex(ValueError, 'observation contract'):
            loop._collect_search_samples(trace, 'fp')

    def test_previous_ring_only_trace_is_not_relabelled_as_rearguard_observation(self):
        trace = self.trace()
        trace['meta'].update(observation_contract='v41_ring_state_v1', reward_contract=WINLOSS)
        self.assertFalse(trace_admission.validate_trace(trace, audit_rules=False,
                                                       **CONTEXT)['accepted'])
        for collect in (lambda: loop._replay_reward_stream(trace, {'fp': 'ring', 'sh': 'military'}),
                        lambda: delta_trainer.collect_delta_samples(trace, 'fp'),
                        lambda: loop._collect_search_samples(trace, 'fp')):
            with self.assertRaisesRegex(ValueError, 'observation contract'):
                collect()
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS), \
                tempfile.TemporaryDirectory() as folder:
            buffer = loop.ReplayBuffer(folder, [('fp', 'ring')])
            payload = dict(reward_contract=WINLOSS, value=[1.], delta=[], **CONTEXT)
            payload['observation_contract'] = 'v41_ring_state_v1'
            with self.assertRaisesRegex(ValueError, 'observation_contract contract mismatch'):
                buffer.add(0, {('fp', 'ring'): payload})
            self.assertEqual(buffer.batches(('fp', 'ring')), [])
        self.assertEqual(trace['meta']['observation_contract'], 'v41_ring_state_v1')

    def test_fresh_buffer_rejects_old_observation_or_policy_payload_before_write(self):
        from selector_contract import training_identity
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS), \
                tempfile.TemporaryDirectory() as folder:
            buffer = loop.ReplayBuffer(folder, [('fp', 'ring')])
            payload = dict(reward_contract=WINLOSS, value=[1.], delta=[], **CONTEXT)
            payload.update(training_identity())
            for field in CONTEXT:
                with self.subTest(field=field):
                    old = dict(payload)
                    del old[field]
                    with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                        buffer.add(0, {('fp', 'ring'): old})
                    self.assertEqual(buffer.batches(('fp', 'ring')), [])
            buffer.add(1, {('fp', 'ring'): payload})
            loaded = buffer.load(('fp', 'ring'))
            self.assertEqual(loaded['value'], [1.])

    def test_loop_setup_refuses_old_contract_before_copying_seeds(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS), \
                tempfile.TemporaryDirectory() as folder, patch.object(loop, 'LOOP', folder):
            path = Path(folder) / 'learning_contract.json'
            original = json.dumps({'reward_contract': WINLOSS})
            path.write_text(original)
            with patch.object(loop.shutil, 'copy2', side_effect=AssertionError('no copy')), \
                    self.assertRaisesRegex(ValueError, 'contract mismatch'):
                loop.setup()
            self.assertEqual(path.read_text(), original)

    def test_existing_unversioned_loop_is_not_relabelled_even_in_legacy_reward_mode(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY), \
                tempfile.TemporaryDirectory() as folder, patch.object(loop, 'LOOP', folder):
            seed = Path(folder) / 'live_fp_ring.pt'
            seed.write_bytes(b'old checkpoint sentinel')
            with patch.object(loop.shutil, 'copy2', side_effect=AssertionError('no copy')), \
                    self.assertRaisesRegex(ValueError, 'Unversioned existing loop'):
                loop.setup()
            self.assertEqual(seed.read_bytes(), b'old checkpoint sentinel')
            self.assertFalse((Path(folder) / 'learning_contract.json').exists())

    def test_loading_old_buffer_metadata_refuses_without_rewriting_it(self):
        import gzip
        import pickle
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS), \
                tempfile.TemporaryDirectory() as folder:
            buffer = loop.ReplayBuffer(folder, [('fp', 'ring')])
            member = Path(folder) / 'fp_ring' / 'b000.pkl.gz'
            member.parent.mkdir(exist_ok=True)
            with gzip.open(member, 'wb') as handle:
                pickle.dump(dict(reward_contract=WINLOSS, value=[1.], delta=[]), handle)
            before = member.read_bytes()
            with self.assertRaisesRegex(ValueError, 'observation_contract contract mismatch'):
                buffer.load(('fp', 'ring'))
            self.assertEqual(member.read_bytes(), before)

    def test_both_checkpoint_writers_preserve_observation_and_policy_contract(self):
        from types import SimpleNamespace
        import torch
        import value_trainer_v2
        # A metadata serialization fixture; it is not a usable trained model.
        net = SimpleNamespace(value_head=[None], state_dict=lambda: {})
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=WINLOSS), \
                tempfile.TemporaryDirectory() as folder:
            for name, writer in (('loop', loop.save_net),
                                 ('value', value_trainer_v2.save_v2_checkpoint)):
                with self.subTest(writer=name):
                    path = str(Path(folder) / (name + '.pt'))
                    writer(net, 'fp', path)
                    metadata = torch.load(path, map_location='cpu', weights_only=False)
                    for field, value in CONTEXT.items():
                        self.assertEqual(metadata.get(field), value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
