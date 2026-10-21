"""Literal ring observations; RB21 and AL31-32 are the independent oracle.

The menu must describe remaining dice, actual ring ownership, and one use
per side per turn. A counter-backed Will conversion is a real ring use.
No learned model or engine-produced expected log is used.
"""
from collections import Counter, defaultdict
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ai
import rendered_extract as rx
import rendered_replay as rr

if os.environ.get('WOTR_AUDIT_RING_RX_BASELINE'):
    spec = importlib.util.spec_from_file_location(
        'ring_rx_baseline', os.environ['WOTR_AUDIT_RING_RX_BASELINE'])
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)


def parse(lines):
    with tempfile.TemporaryDirectory(prefix='ring-extraction-') as folder:
        path = Path(folder) / 'literal.rendered.txt'
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        batches = rr.parse_game(str(path))['batches']
    pools = rx._roll_pools(batches)
    rx._annotate_remaining_pools(batches, pools)
    rx._annotate_ring_ledger(batches)
    return batches, pools


def spent_die_fixture(second_use=False):
    lines = [
        '%Player rolls H P C M .',
        '%Opponent rolls M M M M M M M .',
        '%Player moves Free Action Die (H) from FP Dice to Current Action.',
    ]
    if second_use:
        lines.extend([
            '%Player moves Nenya from Elven Ring 2 to Dark Ring 2.',
            '%Player changes Free Action Die (H) to C.',
        ])
    lines.extend([
        '%Player moves group (Gondor Regular, Gondor Leader) from Minas Tirith to Lossarnach.',
        '%Player kills Free Action Die (' + ('C' if second_use else 'H') + ') in Current Action.',
        '%Opponent moves Shadow Action Die (M) from SA Dice to Current Action.',
        '%Opponent moves Isengard from Political Track 1 to Political Track 0.',
        '%Opponent kills Shadow Action Die (M) in Current Action.',
        '%Player moves Free Action Die (P) from FP Dice to Current Action.',
        '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
        '%Player changes Free Action Die (P) to M.',
        '%Player moves Gondor from Political Track 2 to Political Track 1.',
        '%Player kills Free Action Die (M) in Current Action.',
    ])
    batches, pools = parse(lines)
    return batches[-1], pools


def extract(batch, pools):
    stats, observed = defaultdict(Counter), []
    emit = rx._emit_sample
    def capture(state, side, actions, index, phase, turn):
        if rx._RING_PROBE is None:
            observed.append((ai.clone_state(state), list(actions), index))
        return emit(state, side, actions, index, phase, turn)
    with patch.object(rx, '_emit_sample', side_effect=capture), \
            patch.object(rx, '_LEDGER', None), patch.object(rx, '_TABLE', None), \
            patch.object(rx, '_RX_RINGS_CUR', None), patch.object(rx, '_RX_LINE_CUR', None):
        samples = rx._emit_elven_rings(batch, pools, stats)
    return samples, observed, stats


class ElvenRingEvidence(unittest.TestCase):
    def test_spent_army_die_is_not_a_ring_candidate(self):
        batch, pools = spent_die_fixture()
        samples, observed, stats = extract(batch, pools)
        self.assertEqual(len(samples), 1, dict(stats))
        _, actions, selected = observed[0]
        self.assertEqual(actions[selected].params['old_face'], 'Event')
        self.assertEqual(actions[selected].params['new_face'], 'Muster')
        old_faces = {a.params['old_face'] for a in actions if a.action_type == 'elven_ring'}
        self.assertEqual(old_faces, {'Event', 'Character', 'Muster'})

    def test_actual_second_ring_use_in_one_turn_is_rejected(self):
        batch, pools = spent_die_fixture(second_use=True)
        self.assertTrue(batch['_rx_rings']['pre']['fp_used'])
        samples, _, stats = extract(batch, pools)
        self.assertEqual(samples, [], dict(stats))
        self.assertEqual(stats['detected']['ring_ledger_clamp'], 0)

    def test_shadow_without_an_acquired_ring_is_not_granted_one(self):
        batches, pools = parse([
            '%Opponent rolls P M M M M M M .',
            '%Opponent moves Shadow Action Die (P) from SA Dice to Current Action.',
            '%Opponent moves Narya from Dark Ring 1 to SA Casualties.',
            '%Opponent changes Shadow Action Die (P) to M.',
            '%Opponent kills Shadow Action Die (M) in Current Action.',
        ])
        self.assertEqual(batches[0]['_rx_rings']['pre']['sh'], 0)
        samples, _, stats = extract(batches[0], pools)
        self.assertEqual(samples, [], dict(stats))
        self.assertEqual(stats['detected']['ring_ledger_clamp'], 0)

    def test_counter_backed_will_conversion_is_a_ring_decision(self):
        batches, pools = parse([
            '%Player rolls W W C M .',
            '%Player moves Free Action Die (W) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player changes Free Action Die (W) to C.',
            '%Player kills Free Action Die (C) in Current Action.',
        ])
        samples, observed, stats = extract(batches[0], pools)
        self.assertEqual(len(samples), 1, dict(stats))
        _, menu, index = observed[0]
        self.assertEqual(menu[index].params['old_face'], 'Will of the West')
        self.assertEqual(menu[index].params['new_face'], 'Character')

    def test_unbacked_will_wildcard_is_not_a_ring_use(self):
        batches, pools = parse([
            '%Player rolls W W C M .',
            '%Player moves Free Action Die (W) from FP Dice to Current Action.',
            '%Player changes Free Action Die (W) to C.',
            '%Player kills Free Action Die (C) in Current Action.',
        ])
        samples, _, _ = extract(batches[0], pools)
        self.assertEqual(samples, [])

    def test_second_ring_counter_in_same_batch_cannot_reset_the_limit(self):
        batches, pools = parse([
            '%Player rolls P C C M .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player changes Free Action Die (P) to M.',
            '%Player moves Nenya from Elven Ring 2 to Dark Ring 2.',
            '%Player changes Free Action Die (M) to C.',
            '%Player kills Free Action Die (C) in Current Action.',
        ])
        samples, _, stats = extract(batches[0], pools)
        self.assertEqual(len(samples), 1, dict(stats))
        self.assertEqual(stats['drop_reasons']['ring_used_this_turn'], 1)

    def test_manual_return_of_the_ring_and_face_is_not_a_final_ring_choice(self):
        batches, pools = parse([
            '%Player rolls P C C M .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player changes Free Action Die (P) to M.',
            '%Player moves Narya from Dark Ring 1 to Elven Ring 1.',
            '%Player changes Free Action Die (M) to P.',
            '%Player kills Free Action Die (P) in Current Action.',
        ])
        samples, _, stats = extract(batches[0], pools)
        self.assertEqual(len(samples), 0, dict(stats))

    def test_moving_between_owned_ring_slots_does_not_pay_a_ring(self):
        batches, pools = parse([
            '%Player rolls P C C M .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Elven Ring 2.',
            '%Player changes Free Action Die (P) to M.',
            '%Player kills Free Action Die (M) in Current Action.',
        ])
        samples, _, _ = extract(batches[0], pools)
        self.assertEqual(samples, [])

    def test_explicit_illegal_target_cannot_fall_back_to_an_implied_different_face(self):
        batches, pools = parse([
            '%Player rolls P C C M .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player changes Free Action Die (P) to W.',
            '%Player moves Gondor from Political Track 2 to Political Track 1.',
            '%Player kills Free Action Die (W) in Current Action.',
        ])
        samples, _, stats = extract(batches[0], pools)
        self.assertEqual(len(samples), 0, dict(stats))

    def test_menu_and_encoded_observation_agree_on_both_sides_remaining_dice(self):
        batch, pools = spent_die_fixture()
        samples, observed, _ = extract(batch, pools)
        state, _, _ = observed[0]
        fp, sh = ai.Side.FREE_PEOPLES, ai.Side.SHADOW
        self.assertEqual(Counter(d.current.value for d in state.dice[fp] if not d.used),
                         Counter({'Event': 1, 'Character': 1, 'Muster': 1}))
        self.assertEqual([d.current.value for d in state.dice[fp] if d.used], ['Muster/Army'])
        self.assertEqual(sum(not d.used for d in state.dice[sh]), 6)
        self.assertEqual(sum(d.used for d in state.dice[sh]), 1)
        self.assertTrue((samples[0]['x'] == ai.encode_state(state, fp)).all())

    def test_implied_fp_army_conversion_has_only_physical_fp_faces(self):
        batches, pools = parse([
            '%Player rolls P P P P .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player moves group (Gondor Regular, Gondor Leader) from Minas Tirith to Lossarnach.',
            '%Player kills Free Action Die (P) in Current Action.',
        ])
        analysis = rx._ring_face_analysis(batches[0])
        self.assertEqual(set(analysis['faces']), {'C', 'H'})
        samples, observed, stats = extract(batches[0], pools)
        self.assertEqual(len(samples), 1, dict(stats))
        state, menu, selected = observed[0]
        chosen = {menu[selected].params['new_face']}
        chosen.update(menu[i].params['new_face'] for i in samples[0]['alt'])
        self.assertEqual(chosen, {'Character', 'Muster/Army'})

    def test_action_after_ring_observes_the_changed_die(self):
        batches, _ = parse([
            '%Player rolls P C C C .',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player changes Free Action Die (P) to H.',
            '%Player moves group (Gondor Regular, Gondor Leader) from Minas Tirith to Lossarnach.',
            '%Player kills Free Action Die (H) in Current Action.',
        ])
        batch = batches[0]
        observed = []
        original = ai.encode_state
        def encode(state, side):
            observed.append(Counter(d.current.value for d in state.dice[side] if not d.used))
            return original(state, side)
        stats = defaultdict(Counter)
        with patch.object(rx, '_LEDGER', None), patch.object(rx, '_TABLE', None), \
                patch.object(ai, 'encode_state', side_effect=encode):
            state, side = rx._mk_state_with_die(batch, install_hands=True)
            menu = ai.get_legal_actions(state, side)
            sample = rx._emit_bookkeeping(batch, state, side, menu, stats)
        self.assertIsNotNone(sample, dict(stats))
        self.assertEqual(observed, [Counter({'Character': 3, 'Muster/Army': 1})])
        self.assertEqual(Counter(batch['_rx_pools']['FREE_PEOPLES']['unused']),
                         Counter({'P': 1, 'C': 3}))

    def test_card_after_ring_uses_the_changed_face_and_queues_event_bonus(self):
        import test_audit0904_stewards_choices as house
        title = 'House of the Stewards/Brave Stand'
        batches, _ = parse([
            '%Player rolls H C C C .',
            '%Player moves Boromir from Fellowship of the Ring to Minas Tirith.',
            '%Player moves Free Action Die (H) from FP Dice to Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player changes Free Action Die (H) to P.',
            '%Player moves ' + title + ' from Free Peoples Hand to Current FP Card.',
            '%Player moves Gondor Regular from FP Reinforcements to Minas Tirith.',
            '%Player draws Free Strategy Card.',
            '%Player draws Free Strategy Card.',
            '%Player kills group (Free Action Die (P), ' + title + ') in Current Action.',
        ])
        batch = batches[0]
        ledger = rx._HandLedger()
        ledger._fp = [house.CID]
        ledger._record(0)
        applied_faces = []
        original = ai.apply_action
        def apply(state, action, *args, **kwargs):
            if action.action_type == 'play_event':
                applied_faces.append(state.dice[house.FP][action.die_idx].current.value)
            return original(state, action, *args, **kwargs)
        stats = defaultdict(Counter)
        with patch.object(rx, '_LEDGER', ledger), patch.object(rx, '_TABLE', None), \
                patch.object(ai, 'apply_action', side_effect=apply):
            samples = rx._emit_card_interiors(batch, stats)
        self.assertEqual(applied_faces, ['Event'])
        self.assertEqual(samples, [])
        self.assertEqual(stats['detected']['fixed_policy_bonus_draw'], 1)
        self.assertEqual(batch['_rx_stewards_gandalf']['die_face'], 'EVENT')

    def test_counter_before_staging_preserves_the_explicit_face_label(self):
        batches, pools = parse([
            '%Player rolls P C C M .',
            '%Opponent rolls M M M M M M M .',
            '%Opponent moves Shadow Action Die (M) from SA Dice to Current Action.',
            '%Opponent moves Isengard from Political Track 1 to Political Track 0.',
            '%Opponent kills Shadow Action Die (M) in Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player changes Free Action Die (P) to H.',
            '%Player moves group (Gondor Regular, Gondor Leader) from Minas Tirith to Lossarnach.',
            '%Player kills Free Action Die (H) in Current Action.',
        ])
        self.assertEqual(rx._annotate_ring_carry(batches), 1)
        samples, observed, stats = extract(batches[-1], pools)
        self.assertEqual(len(samples), 1, dict(stats))
        _, menu, selected = observed[0]
        self.assertEqual(menu[selected].params['old_face'], 'Event')
        self.assertEqual(menu[selected].params['new_face'], 'Muster/Army')
        self.assertEqual(samples[0].get('alt', []), [])

    def test_card_between_carried_counter_and_change_blocks_ring_attribution(self):
        batches, pools = parse([
            '%Player rolls P C C M .',
            '%Opponent rolls M M M M M M M .',
            '%Opponent moves Shadow Action Die (M) from SA Dice to Current Action.',
            '%Opponent kills Shadow Action Die (M) in Current Action.',
            '%Player moves Narya from Elven Ring 1 to Dark Ring 1.',
            '%Player moves Mirror of Galadriel from Free Peoples Hand to Current FP Card.',
            '%Player moves Free Action Die (P) from FP Dice to Current Action.',
            '%Player changes Free Action Die (P) to C.',
            '%Player moves group (Gondor Regular, Gondor Leader) from Minas Tirith to Lossarnach.',
            '%Player kills Free Action Die (C) in Current Action.',
        ])
        rx._annotate_ring_carry(batches)
        samples, _, stats = extract(batches[-1], pools)
        # This prefix does not establish the Ring's target/face. A later
        # movement cannot silently replace the intervening card provenance.
        self.assertEqual(samples, [], dict(stats))


if __name__ == '__main__':
    unittest.main(verbosity=2)
