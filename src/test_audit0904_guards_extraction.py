"""Guards labels must follow completed, literal player placement evidence.

Fixtures below are authored from the printed card, not from apply_action or
the bridge emitter. Real log controls use the existing GAME06/GAME09 records.
The prospective two-row matcher control separates extraction from menu bugs.
"""
from collections import Counter, defaultdict
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import ai
import rendered_extract as rx
import test_audit0904_recruit_choices as guards

if os.environ.get('WOTR_AUDIT_GUARDS_RX_BASELINE'):
    spec = importlib.util.spec_from_file_location(
        'guards_rx_baseline', os.environ['WOTR_AUDIT_GUARDS_RX_BASELINE'])
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)

ROOT = Path(__file__).resolve().parent.parent
TITLE = 'Guards of the Citadel/Charge'


def move(line, obj, source, target):
    return dict(op='move', line=line, obj=obj, **{'from': source, 'to': target})


def recorded_batch(kind='regular', split=False, complete=True,
                   pool='FP Reinforcements'):
    records = [move(10, 'Free Action Die (M)', 'FP Dice', 'Current Action'),
               move(11, TITLE, 'Free Peoples Hand', 'Current FP Card')]
    unit = 'Gondor ' + kind.title()
    if split:
        records += [move(12, 'Gondor Leader', pool, 'Minas Tirith'),
                    move(13, unit, pool, 'Minas Tirith')]
    else:
        records.append(move(12, 'group (Gondor Leader, ' + unit + ')',
                            pool, 'Minas Tirith'))
    if complete:
        records.append(dict(op='kill', line=16, zone='Current Action',
                            pieces=['Free Action Die (M)', TITLE]))
    pre = {'armies': {'minas tirith': (
        [('GONDOR', 'REGULAR')] * 3 + [('GONDOR', 'ELITE')],
        [('LEADER', 'GONDOR')])}, 'pol': {},
        'fel': {'region': 'rivendell', 'comps': [], 'corruption': 0}}
    post = copy.deepcopy(pre)
    post['armies']['minas tirith'][0].append(('GONDOR', kind.upper()))
    post['armies']['minas tirith'][1].append(('LEADER', 'GONDOR'))
    return dict(kind='bookkeeping', die_side='FREE_PEOPLES', die_face='M',
                start_line=10, end_line=17, turn=1, records=records, rng=[],
                truth_before=pre, truth=post)


def match(batch):
    options = [(i, ai.Action(-1, 'card_option', {
        'card_id': guards.CID, 'choices': {'unit_type': kind}}))
        for i, kind in enumerate(('regular', 'elite'))]
    pco = {'card_name': guards.TITLE, 'side': 'fp', 'decider': 'fp'}
    context = {'delta': rx._snapshot_delta(batch['truth_before'], batch['truth']),
               'play_title': TITLE, 'batch': batch}
    return rx._match_interior(pco, options, context, batch,
                              defaultdict(Counter), True)


class GuardsEvidence(unittest.TestCase):
    def test_grouped_and_leader_first_split_choose_exact_kind(self):
        for kind, expected in (('regular', 0), ('elite', 1)):
            for split in (False, True):
                with self.subTest(kind=kind, split=split):
                    self.assertEqual(match(recorded_batch(kind, split)),
                                     (expected, None))

    def test_partial_card_never_supplies_a_choice_label(self):
        self.assertEqual(match(recorded_batch(complete=False)),
                         (None, 'cardint_guards_incomplete'))

    def test_wrong_pool_cannot_masquerade_as_fp_reinforcement(self):
        for pool in ('FP Casualties', 'SA Reinforcements',
                     'reinforcements(1)/casualties(1)', 'Osgiliath'):
            with self.subTest(pool=pool):
                self.assertEqual(match(recorded_batch(pool=pool)),
                                 (None, 'cardint_guards_not_fp_reinforcement'))

    def test_returned_or_corrected_recruit_is_not_guessed(self):
        batch = recorded_batch()
        batch['records'].insert(-1, move(14, 'Gondor Regular',
                                        'Minas Tirith', 'FP Reinforcements'))
        batch['truth']['armies']['minas tirith'][0].pop()
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_return_ambiguous'))

    def test_overstack_return_requires_separate_attribution(self):
        batch = recorded_batch('elite')
        batch['records'].insert(-1, move(14, 'Gondor Regular',
                                        'Minas Tirith', 'FP Reinforcements'))
        batch['truth']['armies']['minas tirith'][0].pop(0)
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_return_ambiguous'))

    def test_both_types_placed_cannot_label_a_single_recruit(self):
        batch = recorded_batch()
        batch['records'].insert(-1, move(14, 'Gondor Elite',
                                        'FP Reinforcements', 'Minas Tirith'))
        batch['truth']['armies']['minas tirith'][0].append(('GONDOR', 'ELITE'))
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_recruit_count'))

    def test_raw_gesture_and_board_delta_must_agree(self):
        batch = recorded_batch()
        batch['truth']['armies']['minas tirith'][0][-1] = ('GONDOR', 'ELITE')
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_delta_mismatch'))

    def test_card_disposal_before_last_placement_is_incomplete(self):
        batch = recorded_batch()
        batch['records'][-1]['line'] = 11
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_incomplete'))

    def test_unit_without_named_card_play_is_not_attributed(self):
        batch = recorded_batch()
        batch['records'].pop(1)
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_no_play'))

    def test_besieged_box_placement_keeps_the_selected_kind(self):
        batch = recorded_batch('elite')
        batch['records'][2]['to'] = 'Minas Tirith Stronghold'
        for name in ('truth_before', 'truth'):
            snapshot = batch[name]['armies']
            snapshot['minas tirith stronghold'] = snapshot.pop('minas tirith')
        self.assertEqual(match(batch), (1, None))

    def test_leader_without_unit_is_not_a_type_decision(self):
        batch = recorded_batch()
        batch['records'][2]['obj'] = 'Gondor Leader'
        batch['truth']['armies']['minas tirith'][0].pop()
        self.assertEqual(match(batch),
                         (None, 'cardint_guards_recruit_count'))

    def test_complete_human_transcripts_do_not_train_fixed_unit_type(self):
        for kind in ('regular', 'elite'):
            for split in (False, True):
                with self.subTest(kind=kind, split=split):
                    stats = defaultdict(Counter)
                    with patch.object(rx, '_mk_state_with_die',
                                      return_value=(guards.before_play(), guards.FP)):
                        samples = rx._emit_card_interiors(recorded_batch(kind, split), stats)
                    self.assertEqual(samples, [], dict(stats))
                    self.assertEqual(stats['kinds_emit']['cardopt_unit_type'], 0)
                    self.assertEqual(stats['detected']['fixed_policy:recruit_elite_if_available'], 1)

    def test_real_expert_elite_records_are_fixed_policy_transitions(self):
        import rendered_replay as rr
        paths = [
            'GAME06_2025-05-18_170053_FP_PeterM_SP_jan.rendered.txt',
            'GAME09_2025-01-10_054831_FP_SilmarillWill_SP_Upward.rendered.txt']
        for name in paths:
            with self.subTest(game=name):
                game = rr.parse_game(str(ROOT / 'data' / 'rendered_logs' / name))
                batch = next(b for b in game['batches']
                             if rx._card_play_title(b) == TITLE)
                def observed_card(state, _batch):
                    state.hands[guards.FP].hand = [guards.CID]
                stats = defaultdict(Counter)
                with patch.object(rx, '_install_ledger_hands', side_effect=observed_card):
                    samples = rx._emit_card_interiors(batch, stats)
                self.assertEqual(samples, [], dict(stats))
                self.assertEqual(stats['detected']['fixed_policy:recruit_elite_if_available'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
