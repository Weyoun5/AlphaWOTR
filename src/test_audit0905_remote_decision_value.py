"""Literal local/served values at the pre-decision recording boundary."""
import ast
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import ai
import game_trace
from infer_server import RemoteNet
from reward_contract import LEGACY, PROFILE, WINLOSS


class LiteralClient:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def eval(self, key, x):
        self.calls.append((key, x.copy()))
        if isinstance(self.values, Exception):
            raise self.values
        return self.values


class LiteralNet(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().numpy().copy())
        return torch.tensor([[self.value]], device=x.device)


class RemoteDecisionValueTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        baseline = os.environ.get('WOTR_AUDIT_REMOTE_VALUE_BASELINE')
        if baseline:
            source = Path(baseline).read_text(encoding='utf-8')
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == '_decision_context')
            ns = dict(vars(ai))
            exec(compile(ast.Module(body=[node], type_ignores=[]), baseline, 'exec'), ns)
            self.stack.enter_context(patch.object(ai, '_decision_context', ns['_decision_context']))
            # The baseline function must see the same patched collaborators.
            self.baseline_globals = ns
        else:
            self.baseline_globals = None
        self.state = SimpleNamespace(_eye_phase=False)
        self.vector = np.asarray([.125, .5, -.75], dtype=np.float32)
        self.seen_phases = []

        def encode(state, side):
            self.seen_phases.append(state._eye_phase)
            return self.vector

        replacements = dict(DEVICE=torch.device('cpu'), INFER_CLIENT=None,
                            encode_state=encode, _reward_stats=lambda s: {'literal': 7},
                            _profile_shaping=lambda s: .25)
        for name, value in replacements.items():
            self.stack.enter_context(patch.object(ai, name, value))
            if self.baseline_globals is not None:
                self.baseline_globals[name] = value
        self.stack.enter_context(patch.object(game_trace, 'snapshot_state',
                                             side_effect=lambda s: {'eye': s._eye_phase}))
        self.stack.enter_context(patch.dict(os.environ, WOTR_REWARD_CONTRACT=PROFILE))

    def context(self, net, side=ai.Side.FREE_PEOPLES, **kw):
        return ai._decision_context(self.state, side, tracer=object(), net=net, **kw)

    def test_remote_uses_attached_client_and_literal_raw_value_once(self):
        client = LiteralClient([.5])
        global_client = LiteralClient([-.8])
        with patch.object(ai, 'INFER_CLIENT', global_client):
            if self.baseline_globals is not None:
                self.baseline_globals['INFER_CLIENT'] = global_client
            row = self.context(RemoteNet('fp-generation7', client))
        self.assertEqual(row['raw_value'], .75)
        self.assertEqual(row['value'], .75)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], 'fp-generation7')
        np.testing.assert_array_equal(client.calls[0][1], [[.125, .5, -.75]])
        self.assertEqual(client.calls[0][1].dtype, np.float32)
        self.assertEqual(global_client.calls, [])

    def test_global_client_remains_supported_for_unattached_handle(self):
        client = LiteralClient([.5])
        with patch.object(ai, 'INFER_CLIENT', client):
            row = self.context(RemoteNet('sh'), side=ai.Side.SHADOW)
        self.assertEqual(row['raw_value'], .25)
        self.assertEqual(client.calls[0][0], 'sh')

    def test_local_prediction_uses_eval_mode_and_same_literal_value(self):
        net = LiteralNet(.5)
        net.train()
        row = self.context(net)
        self.assertEqual(row['raw_value'], .75)
        self.assertEqual(row['value'], .75)
        self.assertFalse(net.training)
        np.testing.assert_array_equal(net.inputs[0], [[.125, .5, -.75]])

    def test_clip_occurs_after_potential_and_preserves_raw_for_both_contracts(self):
        for contract in (PROFILE, WINLOSS):
            for remote in (False, True):
                with self.subTest(contract=contract, remote=remote), patch.dict(
                        os.environ, WOTR_REWARD_CONTRACT=contract):
                    net = RemoteNet('fp', LiteralClient([1.5])) if remote else LiteralNet(1.5)
                    row = self.context(net)
                    self.assertEqual(row['raw_value'], 1.75)
                    self.assertEqual(row['value'], 1.)

    def test_legacy_raw_value_stays_unclipped(self):
        with patch.dict(os.environ, WOTR_REWARD_CONTRACT=LEGACY):
            row = self.context(RemoteNet('fp', LiteralClient([1.5])))
        self.assertEqual(row['value'], 1.75)

    def test_phase_encoding_snapshot_and_flag_restore(self):
        for old in (False, True):
            with self.subTest(old=old):
                self.state._eye_phase = old
                row = self.context(RemoteNet('sh', LiteralClient([.5])), phase='_eye_phase')
                self.assertEqual(row['state'], {'eye': True})
                self.assertTrue(self.seen_phases[-1])
                self.assertIs(self.state._eye_phase, old)
                self.vector[0] = 99
                self.assertNotEqual(row['encoding'][0], 99)
                self.vector[0] = .125

    def test_bad_reply_count_refuses_to_record_and_restores_phase(self):
        for values in ([], [.1, .2]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, 'Value replies'):
                    self.context(RemoteNet('fp', LiteralClient(values)), phase='_eye_phase')
                self.assertFalse(self.state._eye_phase)

    def test_nonfinite_reply_refuses_to_record_and_restores_phase(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'Non-finite|Nonfinite'):
                    self.context(RemoteNet('fp', LiteralClient([value])), phase='_eye_phase')
                self.assertFalse(self.state._eye_phase)

    def test_client_failure_propagates_without_heuristic_fallback(self):
        client = LiteralClient(RuntimeError('literal disconnected server'))
        with self.assertRaisesRegex(RuntimeError, 'literal disconnected server'):
            self.context(RemoteNet('fp', client), phase='_eye_phase')
        self.assertFalse(self.state._eye_phase)

    def test_untraced_context_does_not_request_prediction(self):
        client = LiteralClient(RuntimeError('must not call'))
        row = ai._decision_context(self.state, ai.Side.FREE_PEOPLES,
                                   tracer=None, net=RemoteNet('fp', client))
        self.assertNotIn('raw_value', row)
        self.assertEqual(client.calls, [])


if __name__ == '__main__':
    unittest.main()
