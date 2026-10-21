"""Fixed AI conventions remove targets without rewriting human history."""
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import ai
import ai_policies
import extract_corpus
import rendered_extract as rx
import test_audit0904_stewards_choices as house
import test_audit0904_stewards_extraction as logs


def muster_menu():
    state = house.before_play(face=house.gs.ActionDieFace.MUSTER)
    state.political.active[house.GONDOR] = True
    state.political.boxes[house.GONDOR] = 3
    menu = ai.get_legal_actions(state, house.FP)
    return state, menu


def emit(state, menu, index, phase='muster'):
    with patch.object(rx, '_RX_RINGS_CUR', None), patch.object(rx, '_RX_LINE_CUR', None), \
            patch.object(rx, '_RX_EYEALLOC', None), patch.object(rx, '_RX_EYEROLL', None):
        return rx._emit_sample(state, house.FP, menu, index, phase, 1)


class FixedPolicyCorpus(unittest.TestCase):
    def test_elite_label_uses_projected_menu_and_remapped_index(self):
        state, menu = muster_menu()
        selected = next(i for i, a in enumerate(menu) if a.action_type == 'muster_elite'
                        and a.params['region'] == 'minas tirith' and a.die_idx == 0)
        regular = next(a for a in menu if a.action_type == 'muster_recruit'
                       and a.params.get('type') == 'regular'
                       and a.params['region'] == 'minas tirith' and a.die_idx == 0)
        sample = emit(state, menu, selected)
        self.assertNotIn('ai_policy_exclusion', sample)
        self.assertEqual(sample['ai_policy_version'], ai_policies.POLICY_VERSION)
        self.assertTrue(np.array_equal(sample['encs'][sample['human']],
                                       ai.encode_action(menu[selected], state, house.FP)))
        self.assertFalse(any(np.array_equal(row, ai.encode_action(regular, state, house.FP))
                             for row in sample['encs']))
        # The source menu and observation remain the literal human game.
        self.assertIn(regular, menu)
        self.assertEqual(house.figures(state), (3, 1, 1))

    def test_human_regular_when_elite_available_is_dropped_never_relabeled(self):
        state, menu = muster_menu()
        selected = next(i for i, a in enumerate(menu) if a.action_type == 'muster_recruit'
                        and a.params.get('type') == 'regular'
                        and a.params['region'] == 'minas tirith' and a.die_idx == 0)
        sample = emit(state, menu, selected)
        self.assertEqual(sample['ai_policy_exclusion'], 'expert_action_outside_fixed_policy')
        stats = defaultdict(Counter)
        self.assertEqual(rx._filter_ai_policy_samples([sample], stats), [])
        with tempfile.TemporaryDirectory(prefix='fixed-policy-pack-') as folder:
            destination = Path(folder) / 'should-not-exist.npz'
            with self.assertRaisesRegex(ValueError, 'filtered before packing'):
                extract_corpus.pack_npz([sample], str(destination))
            self.assertFalse(destination.exists())

    def test_elven_ring_remains_a_learned_choice(self):
        state, menu = muster_menu()
        selected = next(i for i, a in enumerate(menu) if a.action_type == 'elven_ring')
        sample = emit(state, menu, selected, 'elven_ring')
        self.assertNotIn('ai_policy_exclusion', sample)
        self.assertGreater(len(sample['encs']), 1)
        stats = defaultdict(Counter)
        self.assertEqual(len(rx._filter_ai_policy_samples([sample], stats)), 1)

    def test_pack_retains_policy_version_and_actual_decision_phase(self):
        state, menu = muster_menu()
        selected = next(i for i, a in enumerate(menu) if a.action_type == 'elven_ring')
        sample = emit(state, menu, selected, 'elven_ring')
        with tempfile.TemporaryDirectory(prefix='policy-version-') as folder:
            path = Path(folder) / 'sample.npz'
            extract_corpus.pack_npz([sample], str(path))
            with np.load(path) as data:
                self.assertEqual(data['ai_policy_version'].tolist(), [ai_policies.POLICY_VERSION])
                self.assertEqual(data['decision_phase'].tolist(), ['elven_ring'])
                self.assertEqual(data['observation_contract'].tolist(), ['v44_fellowship_tail_v1'])

    def test_human_regular_and_bonus_keep_both_discard_observations_without_type_targets(self):
        batches, ledger = logs.bonus_fixture()
        initial = house.before_play(face=house.gs.ActionDieFace.EVENT, fillers=house.FILLERS)
        stats, observed = defaultdict(Counter), []
        original = rx._emit_sample
        def capture(state, side, menu, index, phase, turn):
            observed.append(ai.clone_state(state))
            return original(state, side, menu, index, phase, turn)
        with patch.object(rx, '_LEDGER', ledger), \
                patch.object(rx, '_mk_state_with_die', return_value=(initial, house.FP)), \
                patch.object(rx, '_emit_sample', side_effect=capture):
            interiors = rx._emit_card_interiors(batches[0], stats)
            discards = rx._emit_discards(batches, stats)
        self.assertEqual(interiors, [])
        self.assertEqual(len(discards), 2, dict(stats))
        self.assertEqual([house.figures(state) for state in observed], [(4, 1, 1)] * 2)
        self.assertIsNotNone(observed[0]._gandalf_draw_pending)
        self.assertIsNone(observed[1]._gandalf_draw_pending)
        self.assertNotIn(house.BONUS, observed[0].hands[house.FP].hand)
        self.assertIn(house.BONUS, observed[1].hands[house.FP].hand)


if __name__ == '__main__':
    unittest.main(verbosity=2)
