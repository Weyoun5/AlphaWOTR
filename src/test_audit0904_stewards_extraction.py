"""House recruitment labels follow closed player evidence, never Elite defaults."""
from collections import Counter, defaultdict
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ai
import rendered_extract as rx
import rendered_replay as rr
import test_audit0904_stewards_choices as house

ROOT = Path(__file__).resolve().parents[1]
TITLE = 'House of the Stewards/Brave Stand'


def parse_fixture(lines):
    with tempfile.TemporaryDirectory(prefix='stewards-extraction-') as scratch:
        path = Path(scratch) / 'fixture.rendered.txt'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return rr.parse_game(str(path))


def bonus_fixture(choice='take', full=True, early=False, cleanup_first=False):
    lines = [
        '%Player rolls P C C C .',
        '%Opponent rolls M M M M M M M .',
        '%Player moves Boromir from Fellowship of the Ring to Minas Tirith.',
        '%Player moves Free Action Die (P) from FP Dice to Current Action.',
        '%Player moves ' + TITLE + ' from Free Peoples Hand to Current FP Card.',
        '%Player moves Gondor Regular from FP Reinforcements to Minas Tirith.',
        '%Player draws Free Strategy Card.',
        '%Player draws Free Strategy Card.',
    ]
    cleanup = '%Player kills group (Free Action Die (P), ' + TITLE + ') in Current Action.'
    discard = '%Player kills Guards of the Citadel/Charge in Free Peoples Hand.'
    if full and not early:
        lines.append(discard)
    if cleanup_first:
        lines.append(cleanup)
    if choice == 'take':
        lines.append('%Player draws Free Character Card.')
    elif choice == 'wrong_deck':
        lines.append('%Player draws Free Strategy Card.')
    elif choice == 'decline':
        lines.append("%<Player> I won't draw another card")
    if full and early:
        lines.append(discard)
    if full and choice == 'take':
        lines.append('%Player kills Gwaihir the Windlord/Daring Defiance in Free Peoples Hand.')
    if not cleanup_first:
        lines.append(cleanup)
    lines.append('%Opponent moves Shadow Action Die (M) from SA Dice to Current Action.')
    ledger = rx._HandLedger()
    ledger._fp = [house.CID] + (list(house.FILLERS) if full else [])
    ledger._record(0)
    strategy = iter((house.S1, house.S2, house.S3))
    for line, text in enumerate(lines, 1):
        if 'from Free Peoples Hand to Current FP Card' in text:
            ledger.remove_named('fp', house.CID, line, face_up=True)
        elif 'draws Free Strategy Card' in text:
            ledger.add('fp', next(strategy), line)
        elif 'draws Free Character Card' in text:
            ledger.add('fp', house.BONUS, line)
        elif 'in Free Peoples Hand' in text:
            cid = house.S1 if 'Guards of' in text else house.BONUS
            ledger.remove_named('fp', cid, line)
            ledger.discard_events.append(('fp', line, cid))
    parsed = parse_fixture(lines)
    rx._annotate_remaining_pools(parsed['batches'], {})
    rx._annotate_ring_ledger(parsed['batches'])
    return parsed['batches'], ledger

if os.environ.get('WOTR_AUDIT_STEWARDS_RX_BASELINE'):
    spec = importlib.util.spec_from_file_location(
        'stewards_rx_baseline', os.environ['WOTR_AUDIT_STEWARDS_RX_BASELINE'])
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)


def unit_batch(kind='regular', region='Minas Tirith', complete=True,
               pool='FP Reinforcements'):
    """Printed single unit, no Leader, then two Strategy draws, on a C die."""
    lines = [
        '%Player moves Free Action Die (C) from FP Dice to Current Action.',
        '%Player moves ' + TITLE + ' from Free Peoples Hand to Current FP Card.',
        '%Player moves Gondor ' + kind.title() + ' from ' + pool + ' to ' + region + '.',
        '%Player draws Free Strategy Card.',
        '%Player draws Free Strategy Card.']
    if complete:
        lines.append('%Player kills group (Free Action Die (C), ' + TITLE + ') in Current Action.')
    records = [dict(rr.parse_line(text), line=line) for line, text in enumerate(lines, 10)]
    pre = {'armies': {'minas tirith': (
        [('GONDOR', 'REGULAR')] * 3 + [('GONDOR', 'ELITE')],
        [('LEADER', 'GONDOR'), ('NAMED', 'Boromir')])},
        'pol': {}, 'fel': {'region': 'rivendell', 'comps': [], 'corruption': 0}}
    post = copy.deepcopy(pre)
    post['armies']['minas tirith'][0].append(('GONDOR', kind.upper()))
    if region != 'Minas Tirith':
        for snap in (pre, post):
            snap['armies'][rr.sanitize(region)] = snap['armies'].pop('minas tirith')
    return dict(kind='muster', die_side='FREE_PEOPLES', die_face='C',
                start_line=10, end_line=17, turn=1, records=records, rng=[],
                truth_before=pre, truth=post)


def match(batch):
    region = next(iter(batch['truth_before']['armies']))
    options = [(i, ai.Action(-1, 'card_option', {
        'card_id': house.CID,
        'choices': {'boromir_region': region, 'unit_type': kind}}))
        for i, kind in enumerate(('regular', 'elite'))]
    pending = {'card_id': house.CID, 'card_name': house.TITLE, 'side': 'fp'}
    ctx = {'delta': rx._snapshot_delta(batch['truth_before'], batch['truth']),
           'play_title': TITLE, 'batch': batch}
    return rx._match_interior(pending, options, ctx, batch, defaultdict(Counter), True)


class HouseRecruitLabels(unittest.TestCase):
    def test_observed_regular_and_elite_match_separate_rows(self):
        for kind, selected in (('regular', 0), ('elite', 1)):
            with self.subTest(kind=kind):
                self.assertEqual(match(unit_batch(kind)), (selected, None))

    def test_incomplete_effect_is_not_the_historical_default(self):
        self.assertEqual(match(unit_batch(complete=False)),
                         (None, 'cardint_stewards_incomplete'))

    def test_wrong_pool_cannot_supply_the_unit_label(self):
        for pool in ('FP Casualties', 'SA Reinforcements', 'Reinforcements'):
            with self.subTest(pool=pool):
                self.assertEqual(match(unit_batch(pool=pool)),
                                 (None, 'cardint_stewards_not_fp_reinforcement'))

    def test_returned_unit_remains_explicitly_ambiguous(self):
        batch = unit_batch()
        rec = rr.parse_line('%Player moves Gondor Regular from Minas Tirith to FP Reinforcements.')
        rec['line'] = 14
        batch['records'].insert(-1, rec)
        batch['truth']['armies']['minas tirith'][0].pop()
        self.assertEqual(match(batch), (None, 'cardint_stewards_return_ambiguous'))

    def test_wrong_board_delta_cannot_label_a_literal_regular(self):
        batch = unit_batch()
        batch['truth']['armies']['minas tirith'][0][-1] = ('GONDOR', 'ELITE')
        self.assertEqual(match(batch), (None, 'cardint_stewards_delta_mismatch'))

    def test_human_unit_alternatives_do_not_become_fixed_policy_training_targets(self):
        for kind in ('regular', 'elite'):
            with self.subTest(kind=kind):
                stats = defaultdict(Counter)
                with patch.object(rx, '_mk_state_with_die',
                                  return_value=(house.before_play(), house.FP)):
                    samples = rx._emit_card_interiors(unit_batch(kind), stats)
                self.assertEqual(stats['kinds_emit']['cardopt_unit_type'], 0, dict(stats))
                self.assertEqual(samples, [])
                self.assertEqual(stats['detected']['fixed_policy:recruit_elite_if_available'], 1)

    def test_game10_osgiliath_play_is_not_promoted_to_a_legal_event_label(self):
        path = ROOT / 'data/rendered_logs/GAME10_2023-08-13_201533_FP_HombreLobo_SP_GJB.rendered.txt'
        corrected = ROOT / 'audit/2026-09-04/decision_pass/undo_fixture_refresh_r1/stewards_game10_native_r1/010/client'
        with self.assertRaises(rr.ClientUndoHistoryError):
            rr.parse_game(str(path))
        parsed = rr.parse_game(str(corrected / 'rendered.txt'), raw_path=str(corrected / 'input.log'))
        batch = next(b for b in parsed['batches'] if rx._card_play_title(b) == TITLE)
        def observed_card(state, _batch):
            state.hands[house.FP].hand = [house.CID]
        stats = defaultdict(Counter)
        with patch.object(rx, '_install_ledger_hands', side_effect=observed_card):
            samples = rx._emit_card_interiors(batch, stats)
        self.assertEqual(samples, [])
        self.assertEqual(stats['kinds_emit']['cardopt_unit_type'], 0)


class HouseDiscardObservations(unittest.TestCase):
    def test_each_discard_observes_recruitment_but_only_cards_already_drawn(self):
        # Boromir's first line establishes a reachable midgame fixture. It
        # is setup, not a claim that this was a legal one-step separation.
        lines = [
            '%Player moves Boromir from Fellowship of the Ring to Minas Tirith.',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves ' + TITLE + ' from Free Peoples Hand to Current FP Card.',
            '%Player moves Gondor Regular from FP Reinforcements to Minas Tirith.',
            '%Player draws Free Strategy Card.',
            '%Player draws Free Strategy Card.',
            '%Player kills Guards of the Citadel/Charge in Free Peoples Hand.',
            '%Player draws Free Character Card.',
            '%Player kills Gwaihir the Windlord/Daring Defiance in Free Peoples Hand.',
            '%Player kills group (Free Action Die (P), ' + TITLE + ') in Current Action.',
        ]
        with tempfile.TemporaryDirectory(prefix='stewards-discard-') as scratch:
            path = Path(scratch) / 'fixture.rendered.txt'
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            parsed = rr.parse_game(str(path))
        ledger = rx._HandLedger()
        ledger._fp = [house.CID] + list(house.FILLERS)
        ledger._record(0)
        for line, card, add in ((3, house.CID, False), (5, house.S1, True),
                                (6, house.S2, True), (7, house.S1, False),
                                (8, house.BONUS, True), (9, house.BONUS, False)):
            if add:
                ledger._fp.append(card)
            else:
                ledger._fp.remove(card)
            ledger._record(line)
        ledger.discard_events = [('fp', 7, house.S1), ('fp', 9, house.BONUS)]
        observed = []
        emit = rx._emit_sample
        def capture(state, side, menu, index, kind, turn):
            observed.append((house.figures(state), list(state.hands[side].hand),
                             menu[index].params['card_id']))
            return emit(state, side, menu, index, kind, turn)
        stats = defaultdict(Counter)
        with patch.object(rx, '_LEDGER', ledger), patch.object(rx, '_emit_sample', side_effect=capture):
            samples = rx._emit_discards(parsed['batches'], stats)
        self.assertEqual(len(samples), 2, dict(stats))
        self.assertEqual([row[0] for row in observed], [(4, 1, 1), (4, 1, 1)])
        self.assertCountEqual(observed[0][1], list(house.FILLERS) + [house.S1, house.S2])
        self.assertNotIn(house.BONUS, observed[0][1])
        self.assertCountEqual(observed[1][1], list(house.FILLERS) + [house.S2, house.BONUS])
        self.assertEqual([row[2] for row in observed], [house.S1, house.BONUS])

    def test_record_snapshots_keep_spent_dice_and_exclude_later_board_and_ring_changes(self):
        # These are parser timing controls, not a claim that the later ring
        # and figure gestures constitute another legal action on this die.
        parsed = parse_fixture([
            '%Player rolls P C C C .',
            '%Opponent rolls M M M M M M M .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves ' + TITLE + ' from Free Peoples Hand to Current FP Card.',
            '%Player moves Gondor Regular from FP Reinforcements to Minas Tirith.',
            '%Player draws Free Strategy Card.',
            '%Player draws Free Strategy Card.',
            '%Player kills Guards of the Citadel/Charge in Free Peoples Hand.',
            '%Player moves Gondor Regular from Minas Tirith to Lossarnach.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player draws Free Character Card.',
            '%Player kills Gwaihir the Windlord/Daring Defiance in Free Peoples Hand.',
            '%Player kills group (Free Action Die (P), ' + TITLE + ') in Current Action.',
            '%Opponent moves Shadow Action Die (M) from SA Dice to Current Action.',
        ])
        batches = parsed['batches']
        rx._annotate_remaining_pools(batches, {})
        rx._annotate_ring_ledger(batches)
        records = {r['line']: r for b in batches for r in b['records']}
        first, bonus, second = (records[i] for i in (8, 11, 12))
        for record in (first, bonus, second):
            self.assertIn('decision_truth', record)
            self.assertEqual(record['_rx_pools']['FREE_PEOPLES'],
                             {'unused': ['C', 'C', 'C'], 'used': ['P']})
            self.assertEqual(record['_rx_pools']['SHADOW'],
                             {'unused': ['M'] * 7, 'used': []})
        for record, regulars, fp_rings in ((first, 4, 3), (bonus, 3, 2), (second, 3, 2)):
            state = rx._recon_state(record['decision_truth'], 1)
            self.assertEqual(house.figures(state), (regulars, 1, 1))
            self.assertEqual(record['decision_truth']['_rx_rings']['fp'], fp_rings)
        self.assertEqual(batches[0]['_rx_pools']['FREE_PEOPLES'],
                         {'unused': ['C', 'C', 'C', 'P'], 'used': []})

    def test_missing_versioned_record_snapshot_is_dropped(self):
        batch = unit_batch()
        batch['record_truth_version'] = 1
        ledger = rx._HandLedger()
        ledger._fp = list(house.FILLERS) + [house.S1, house.S2]
        ledger._record(0)
        ledger.discard_events = [('fp', 14, house.S1)]
        stats = defaultdict(Counter)
        with patch.object(rx, '_LEDGER', ledger):
            self.assertEqual(rx._emit_discards([batch], stats), [])
        self.assertEqual(stats['drop_reasons']['discard_no_record_snapshot'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
