"""Weight-only conversion is explicit and preserves historical data identity."""
import unittest
import torch
import ai
from warmstart_casualty_schema import convert_checkpoint, OLD_OBSERVATION, COMBAT_OBSERVATION, CARD_OBSERVATION


class WarmStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(905)
        cls.old = ai.WOTRValueNet(3742, side='fp').eval()
        cls.checkpoint = dict(model_state=cls.old.state_dict(), side='fp',
            input_dim=3742, action_dim=ai.ACTION_ENCODING_DIM,
            observation_contract=OLD_OBSERVATION, optimizer={'historical': True},
            steps=100, validation_score=.9)

    def test_weights_preserve_zero_tail_forward_and_discard_training_claims(self):
        converted = convert_checkpoint(self.checkpoint, 'a'*64)
        net = ai.WOTRValueNet(ai.get_feature_dim(), side='fp').eval()
        net.load_state_dict(converted['model_state'])
        # Isolated tensor arithmetic oracle; actual board validity is irrelevant
        # to checking that the added projection columns have zero influence.
        old_x = torch.zeros(2, 3742)
        new_x = torch.cat([old_x, torch.ones(2, ai.get_feature_dim()-3742)], dim=1)
        with torch.no_grad():
            torch.testing.assert_close(self.old(old_x), net(new_x), rtol=1e-6, atol=1e-6)
        self.assertEqual(converted['training_status'], 'warm_start_not_retrained')
        for key in ('optimizer', 'steps', 'validation_score'):
            self.assertNotIn(key, converted)
        self.assertEqual(self.checkpoint['observation_contract'], OLD_OBSERVATION)
        self.assertEqual(self.checkpoint['input_dim'], 3742)

    def test_unidentified_or_wrong_generation_refused(self):
        for observation in (None, 'v40', ai.OBSERVATION_CONTRACT):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                convert_checkpoint(dict(self.checkpoint, observation_contract=observation), 'a'*64)

    def test_unexpected_shape_refused(self):
        state = dict(self.checkpoint['model_state'])
        key = 'tok_global.weight' if self.old.arch_kind == 'transformer' else 'backbone.0.weight'
        state[key] = state[key][:, :-1]
        with self.assertRaisesRegex(ValueError, 'shape'):
            convert_checkpoint(dict(self.checkpoint, model_state=state), 'a'*64)

    def test_explicit_v42_warm_start_keeps_weights_and_prior_lineage(self):
        old_net = ai.WOTRValueNet(3745, side='fp')
        source = dict(self.checkpoint, model_state=old_net.state_dict(),
            input_dim=3745, observation_contract=COMBAT_OBSERVATION,
            action_encoding_contract='casualty_operations_v1',
            warm_start={'source_sha256': 'original_seed'},
            warm_start_updates={'value': 1, 'policy': 4},
            training_status='updated_after_warm_start')
        converted = convert_checkpoint(source, 'b'*64)
        projection = 'tok_global.weight' if old_net.arch_kind == 'transformer' else 'backbone.0.weight'
        for key, tensor in old_net.state_dict().items():
            target = converted['model_state'][key]
            if key == projection:
                torch.testing.assert_close(target[:, :-1], tensor, rtol=0, atol=0)
                self.assertTrue(torch.all(target[:, -1] == 0))
            else:
                torch.testing.assert_close(target, tensor, rtol=0, atol=0)
        self.assertEqual(converted['warm_start']['new_input_columns'], 1)
        self.assertEqual(converted['warm_start']['parent_warm_start'], source['warm_start'])
        self.assertEqual(converted['warm_start']['source_warm_start_updates'], {'value': 1, 'policy': 4})
        self.assertEqual(converted['training_status'], 'warm_start_not_retrained')

    def test_v42_with_wrong_or_missing_action_contract_refused(self):
        for action in (None, 'casualty_operations_v2'):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'original action contract'):
                convert_checkpoint(dict(self.checkpoint, observation_contract=COMBAT_OBSERVATION,
                    action_encoding_contract=action), 'b'*64)

    def test_v43_requires_its_contract_and_preserves_the_zero_tail_forward(self):
        old = ai.WOTRValueNet(3745, side='fp').eval()
        source = dict(self.checkpoint, model_state=old.state_dict(), input_dim=3745,
                      observation_contract=CARD_OBSERVATION, action_encoding_contract='casualty_operations_v2')
        converted = convert_checkpoint(source, 'c'*64)
        net = ai.WOTRValueNet(ai.get_feature_dim(), side='fp').eval()
        net.load_state_dict(converted['model_state'])
        x = torch.zeros(2, 3745)
        with torch.no_grad():
            torch.testing.assert_close(old(x), net(torch.cat([x, torch.ones(2,1)], dim=1)), atol=1e-6, rtol=1e-6)
        self.assertEqual(converted['warm_start']['source_observation_contract'], CARD_OBSERVATION)
        self.assertEqual(source['input_dim'], 3745)
        self.assertTrue(converted['warm_start']['fellowship_tail_policy_requires_training'])
        for action in (None, 'casualty_operations_v1'):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'original action contract'):
                convert_checkpoint(dict(source, action_encoding_contract=action), 'c'*64)


if __name__ == '__main__':
    unittest.main(verbosity=2)
