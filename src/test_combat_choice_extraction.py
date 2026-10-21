"""Chronological parser fixtures using real observed combat continuations.

Records are minimal synthetic examples of the existing rendered schema, not
claimed real games. No expected decision comes from engine outcome or kernel.
"""
import os
os.environ['WOTR_ACT_GEN'] = 'v52'
from test_combat_continuation_literal import (
    live_battle, companion, minion, begin, Dice, cb, gs, card_id)
import copy
import pickle
import unittest
from unittest.mock import patch
import ai
from combat_choice_evidence import match_round_choice
from combat_round_observation import SOURCE_STATE_DIM, TARGET_STATE_DIM


def publish(st, a, d, result, fp_card, sh_card=None):
    pc = {'attacker_region': a, 'defender_region': d,
          'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
          'deciding_side': gs.Side.FREE_PEOPLES, 'rounds_fought': 0, 'is_siege': False}
    st.pending_combat = pc
    ai._accept_combat_round_result(st, pc, result, card_id(fp_card) if fp_card else None,
                                  card_id(sh_card) if sh_card else None)
    return st


def items(*entries):
    return [({'line': line, 'ints': values, 'player': 'FP human' if side == 'A' else 'SH human'}, side, phase)
            for line, side, phase, values in entries]


def match(st, records, rounds, after=0):
    return match_round_choice(st, {'records': records},
        ai.get_legal_actions(st, st.pending_combat['deciding_side']), rounds, 0, 100, after)


def move(line, obj, source, target):
    return {'op': 'move', 'line': line, 'obj': obj, 'from': source, 'to': target}


def kill(line, piece, region):
    return {'op': 'kill', 'line': line, 'pieces': [piece], 'zone': region}


class ChronologicalEvidence(unittest.TestCase):
    def mighty(self):
        st, a, d = live_battle(fp_regulars=4,
            fp_leaders=[companion('Aragorn', 3, 2), companion('Meriadoc'), gs.Leader(gs.Nation.ROHAN)])
        # Runtime now pauses for the attacking payer before defender dice.
        # Recorded evidence below deliberately retains its historical order.
        tape = Dice([6, 1, 1, 1, 1])
        with patch.object(cb, 'roll_d6_batch', tape):
            r = begin(st, a, d, 'Mighty Attack')
        self.assertEqual([call[1] for call in tape.calls], [str(gs.Side.FREE_PEOPLES)])
        self.assertFalse(tape.batches)
        publish(st, a, d, r, 'Mighty Attack')
        return st, items((10, 'A', 'main', [6, 1, 1, 1, 1]),
                         (20, 'D', 'main', [6, 6, 6]), (40, 'A', 're', [6, 6]))

    def test_mighty_payer_park_between_combat_and_reroll(self):
        st, rounds = self.mighty()
        before = pickle.dumps(st)
        chosen, line, reason = match(st, [move(30, 'Aragorn', 'Fords of Isen', 'Current FP Card')], rounds)
        menu = ai.get_legal_actions(st, gs.Side.FREE_PEOPLES)
        self.assertEqual(menu[chosen].params['target'], 'Aragorn')
        self.assertEqual((line, reason), (30, 'park_gesture'))
        self.assertEqual(pickle.dumps(st), before)
        observation = ai.encode_state(st, gs.Side.FREE_PEOPLES)
        self.assertTrue(ai._COMBAT_ROUND_CONTEXT_ON)
        self.assertEqual(len(observation), TARGET_STATE_DIM)
        self.assertTrue(any(observation[SOURCE_STATE_DIM:]))
        tape = Dice([6, 6], [6, 6, 6])
        with patch.object(cb, 'roll_d6_batch', tape):
            done = ai.apply_action(st, menu[chosen])
        self.assertEqual([call[1] for call in tape.calls], [str(gs.Side.FREE_PEOPLES), str(gs.Side.SHADOW)])
        self.assertFalse(tape.batches)
        self.assertEqual(done._last_battle_result.deferred_hits_on_defender, 4)

    def test_mighty_missing_late_early_wrong_place_or_reverted_gesture_abstains(self):
        st, rounds = self.mighty()
        for records in ([], [move(9, 'Aragorn', 'Fords of Isen', 'Current FP Card')],
                        [move(41, 'Aragorn', 'Fords of Isen', 'Current FP Card')],
                        [move(30, 'Aragorn', 'Dale', 'Current FP Card')],
                        [move(30, 'Aragorn', 'Fords of Isen', 'Current FP Card'),
                         move(31, 'Aragorn', 'Current FP Card', 'Fords of Isen')]):
            with self.subTest(records=records):
                self.assertIsNone(match(st, records, rounds)[0])

    def test_roll_count_alone_does_not_identify_mighty_payer(self):
        st, rounds = self.mighty()
        # Human may roll fewer than available Leader dice. Two recorded dice
        # cannot prove the player paid Aragorn instead of a1-leadership Hobbit.
        self.assertEqual(match(st, [], rounds)[2], 'missing_park_gesture')

    def test_native_reconstructed_region_objects_use_public_battle_keys(self):
        st, rounds = self.mighty()
        c = st.pending_combat['_round_resume']['cursor'].combat
        # Native reconstruction stores actual Region objects in Army.region;
        # passing them into board.sanitize previously raised unhashable Region.
        import board
        c.attacker_army.region = st.board.region('Fords of Isen')
        c.defender_army.region = st.board.region('Orthanc')
        result = match(st, [move(30, 'Aragorn', 'Fords of Isen', 'Current FP Card')], rounds)
        self.assertIsNotNone(result[0], result[2])

    def test_regular_ra_count_then_nation_payment_follows_same_record(self):
        st, a, d = live_battle(sh_regulars=2)
        st.armies['orthanc'].units[1].nation = gs.Nation.SAURON
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Pre-choice roll')):
            r = begin(st, a, d, sh_card='Relentless Assault')
        publish(st, a, d, r, None, 'Relentless Assault')
        rounds = items((20, 'A', 'main', [1, 1, 1]), (30, 'D', 'main', [1]))
        records = [move(10, 'Sauron Regular', 'Orthanc', 'Current SA Card'),
                   kill(50, 'Sauron Regular', 'Current SA Card')]
        index, _, _ = match(st, records, rounds)
        menu = ai.get_legal_actions(st, gs.Side.SHADOW)
        self.assertEqual(menu[index].params['count'], 1)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Roll before payment')):
            st = ai.apply_action(st, menu[index])
        self.assertEqual(st.pending_combat['effect_choice']['effect'], 'relentless_assault_payment')
        index, _, reason = match(st, records, rounds)
        self.assertEqual(reason, 'physical_payment')
        menu = ai.get_legal_actions(st, gs.Side.SHADOW)
        self.assertEqual(menu[index].params['nation'], 'SAURON')
        with patch.object(cb, 'roll_d6_batch', Dice([1, 1, 1], [1])):
            st = ai.apply_action(st, menu[index])
        self.assertEqual([u.nation for u in st.armies['orthanc'].units], [gs.Nation.ISENGARD])

    def test_elite_park_is_not_assumed_to_mean_one_hit_or_decline(self):
        st, a, d = live_battle(sh_regulars=0, sh_elites=1)
        r = begin(st, a, d, sh_card='Relentless Assault')
        publish(st, a, d, r, None, 'Relentless Assault')
        rounds = items((20, 'A', 'main', [1, 1, 1]), (30, 'D', 'main', [1]))
        self.assertEqual(match(st, [move(10, 'Isengard Elite', 'Orthanc', 'Current SA Card')], rounds)[2],
                         'payment_parked_loss_unconfirmed')
        self.assertEqual(match(st, [], rounds)[2], 'missing_payment_gesture')

    def test_fateful_ordinary_nazgul_token_is_detected_only_after_observed_rolls(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=1,
            fp_leaders=[gs.Leader(gs.Nation.ROHAN)],
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON), minion('Witch King')])
        with patch.object(cb, 'roll_d6_batch', Dice([1], [6], [6])):
            r = begin(st, a, d, 'Fateful Strike')
        publish(st, a, d, r, 'Fateful Strike')
        rounds = items((10, 'A', 'main', [1]), (20, 'D', 'main', [6]), (30, 'A', 're', [6]))
        index, line, _ = match(st, [kill(35, 'Nazgul', 'Orthanc')], rounds)
        self.assertEqual(ai.get_legal_actions(st, gs.Side.FREE_PEOPLES)[index].params['target'], 'nazgul')
        self.assertEqual(line, 35)
        self.assertIsNone(match(st, [kill(25, 'Nazgul', 'Orthanc')], rounds)[0])
        st.pending_combat['_round_resume']['cursor'].observed_rolls.discard('reroll')
        self.assertEqual(match(st, [kill(35, 'Nazgul', 'Orthanc')], rounds)[2], 'missing_completed_roll_observation')

    def test_black_breath_then_heroic_are_two_distinct_chronological_labels(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=2,
            fp_leaders=[companion('Meriadoc'), gs.Leader(gs.Nation.ROHAN)],
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON)])
        with patch.object(cb, 'roll_d6_batch', Dice([6], [1, 1], [6])):
            r = begin(st, a, d, 'Heroic Death', 'Black Breath')
        publish(st, a, d, r, 'Heroic Death', 'Black Breath')
        rounds = items((10, 'A', 'main', [6]), (20, 'D', 'main', [1, 1]), (30, 'D', 're', [6]))
        records = [kill(40, 'Rohan Leader', 'Fords of Isen'), kill(50, 'Merry', 'Fords of Isen')]
        self.assertEqual(match(st, records[:1], rounds)[2], 'black_breath_heroic_attribution_ambiguous')
        index, line, _ = match(st, records, rounds)
        self.assertEqual(line, 40)
        action = ai.get_legal_actions(st, gs.Side.SHADOW)[index]
        self.assertEqual(action.params['target'], 'leader')
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Repeated recorded roll')):
            st = ai.apply_action(st, action)
            index, second_line, _ = match(st, records, rounds, after=line)
            self.assertEqual(second_line, 50)
            action = ai.get_legal_actions(st, gs.Side.FREE_PEOPLES)[index]
            self.assertEqual((action.params['count'], action.params['target']), (1, 'Meriadoc'))
            done = ai.apply_action(st, action)
        self.assertEqual(done._last_battle_result.deferred_hits_on_attacker, 0)

    def test_high_level_heroic_death_does_not_prove_full_level_cancellation(self):
        st, a, d = live_battle(fp_regulars=1, sh_regulars=3,
            fp_leaders=[companion('Gandalf the White', 3, 1)])
        # White Gandalf has no Captain-of-the-West die: one army unit rolls one.
        with patch.object(cb, 'roll_d6_batch', Dice([6], [6, 6, 6])):
            r = begin(st, a, d, 'Heroic Death')
        publish(st, a, d, r, 'Heroic Death')
        rounds = items((10, 'A', 'main', [6]), (20, 'D', 'main', [6, 6, 6]))
        records = [kill(30, 'Gandalf the White', 'Fords of Isen')]
        self.assertEqual(match(st, records, rounds)[2], 'sacrifice_magnitude_ambiguous')
        self.assertIsNone(match(st, [], rounds)[0])

    def test_cleanup_death_wrong_battle_or_reused_line_never_labels_card(self):
        st, rounds = self.mighty()
        # Use a fresh post-roll Blade state for the target controls.
        st, a, d = live_battle(fp_regulars=1, sh_regulars=1,
            fp_leaders=[companion()], sh_leaders=[minion('Witch King'), minion('Mouth of Sauron')])
        with patch.object(cb, 'roll_d6_batch', Dice([1], [6], [6])):
            r = begin(st, a, d, 'Blade of Westernesse')
        publish(st, a, d, r, 'Blade of Westernesse')
        rounds = items((10, 'A', 'main', [1]), (20, 'D', 'main', [6]), (30, 'A', 're', [6]))
        self.assertIsNotNone(match(st, [kill(40, 'Mouth of Sauron', 'Orthanc')], rounds)[0])
        for records in ([kill(40, 'Mouth of Sauron', 'Dale')],
                        [kill(35, 'Isengard Regular', 'Orthanc'), kill(40, 'Mouth of Sauron', 'Orthanc')], []):
            self.assertIsNone(match(st, records, rounds)[0])
        self.assertIsNone(match(st, [kill(40, 'Mouth of Sauron', 'Orthanc')], rounds, after=40)[0])

    def test_ra_elite_downgrade_and_removal_have_different_literal_hit_costs(self):
        for replacement, expected in [(True, 1), (False, 2)]:
            st, a, d = live_battle(sh_regulars=0, sh_elites=1)
            r = begin(st, a, d, sh_card='Relentless Assault')
            publish(st, a, d, r, None, 'Relentless Assault')
            rounds = items((20, 'A', 'main', [1, 1, 1]), (30, 'D', 'main', [1]))
            records = [move(10, 'Isengard Elite', 'Orthanc', 'Casualties')]
            if replacement:
                records.append(move(11, 'Isengard Regular', 'Reinforcements', 'Orthanc'))
            index, _, _ = match(st, records, rounds)
            menu = ai.get_legal_actions(st, gs.Side.SHADOW)
            self.assertEqual(menu[index].params['count'], expected)
            st = ai.apply_action(st, menu[index])
            index, line, _ = match(st, records, rounds)
            menu = ai.get_legal_actions(st, gs.Side.SHADOW)
            self.assertEqual(menu[index].params['operation'], 'downgrade' if replacement else 'remove')
            self.assertEqual(line, 11 if replacement else 10)

    def test_precombat_exact_dice_then_elite_replacement_before_main_rolls(self):
        from collections import Counter
        import rendered_extract as rx
        st, a, d = live_battle(fp_regulars=0, fp_elites=1, sh_regulars=1, sh_elites=1)
        rounds = items((5, 'A', 'pre', [6]), (20, 'A', 'main', [1]), (30, 'D', 'main', [1, 1]))
        tape = rx._RecordedCombatDice([rounds], gs.Side.FREE_PEOPLES,
            {'detected': Counter()}, allow_special=True)
        tape.special_phase, tape.special_actor = 'pre', gs.Side.FREE_PEOPLES
        with patch.object(cb, 'roll_d6_batch', tape.draw):
            r = begin(st, a, d, 'Charge')
        tape.special_phase = tape.special_actor = None
        publish(st, a, d, r, 'Charge')
        self.assertEqual(st.pending_combat['effect_choice']['effect'], 'pre_combat_casualty')
        records = [kill(10, 'Isengard Elite', 'Orthanc'),
                   move(11, 'Isengard Regular', 'Reinforcements', 'Orthanc')]
        index, line, _ = match(st, records, rounds)
        self.assertEqual(line, 11)
        action = ai.get_legal_actions(st, gs.Side.SHADOW)[index]
        self.assertEqual((action.params['kind'], action.params['operation']), ('elite', 'downgrade'))
        with patch.object(cb, 'roll_d6_batch', tape.draw):
            done = ai.apply_action(st, action)
        tape.finish()
        self.assertEqual([u.kind for u in done.armies['orthanc'].units], [gs.UnitKind.REGULAR] * 2)
        self.assertEqual(done.pending_combat['rounds_fought'], 1)

    def test_dread_explicit_ff_chat_is_bound_to_preroll_window(self):
        st, a, d = live_battle(fp_regulars=3, sh_regulars=2,
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON), gs.Nazgul(gs.Nation.SAURON)])
        r = begin(st, a, d, sh_card='Dread and Despair')
        publish(st, a, d, r, None, 'Dread and Despair')
        rounds = items((20, 'A', 'main', [1]), (30, 'D', 'main', [1, 1]))
        menu = ai.get_legal_actions(st, gs.Side.SHADOW)
        result = match_round_choice(st, {'records': []}, menu, rounds, 0, 100, chats=[(10, 'ff 2', 'SH human')])
        self.assertEqual(menu[result[0]].params['count'], 2)
        for chats in ([], [(21, 'ff 2', 'SH human')], [(9, 'ff 1', 'SH human'), (10, 'ff 2', 'SH human')],
                      [(10, 'maybe ff 2?', 'SH human')], [(10, 'ff 2', 'FP human')]):
            self.assertIsNone(match_round_choice(st, {'records': []}, menu, rounds, 0, 100, chats=chats)[0])

    def test_onslaught_count_payment_then_recorded_additional_roll(self):
        from test_onslaught_payments import position
        from combat_choice_evidence import match_onslaught_payment
        from collections import Counter
        import rendered_extract as rx
        st = position(regulars=1, elites=1)
        rounds = items((10, 'A', 'main', [1, 1, 1]), (20, 'D', 'main', [1, 1]),
                       (40, 'D', 'post', [4, 1]))
        records = [move(25, 'Isengard Regular', 'Orthanc', 'Current SA Card'),
                   move(26, 'Isengard Elite', 'Orthanc', 'Casualties'),
                   move(27, 'Isengard Regular', 'Reinforcements', 'Orthanc'),
                   kill(50, 'Isengard Regular', 'Current SA Card')]
        for expected in ({'count': 2}, {'kind': 'regular', 'operation': 'remove'},
                         {'kind': 'elite', 'operation': 'downgrade'}):
            menu = ai.get_legal_actions(st, gs.Side.SHADOW)
            index, _, reason = match_onslaught_payment(st, {'records': records}, menu, 10, 100, rounds)
            self.assertIsNotNone(index, reason)
            self.assertTrue(all(menu[index].params[k] == v for k, v in expected.items()))
            final = expected.get('kind') == 'elite'
            with patch.object(cb, 'roll_d6_batch', side_effect=Dice([4, 1]) if final else AssertionError('Early Onslaught roll')):
                st = ai.apply_action(st, menu[index])
        self.assertEqual(len(st.armies['fords of isen'].units), 2)
        self.assertEqual([u.kind for u in st.armies['orthanc'].units], [gs.UnitKind.REGULAR])

    def test_special_rng_adapter_refuses_wrong_actor_count_and_unused_roll(self):
        from collections import Counter
        import rendered_extract as rx
        rounds = items((5, 'A', 'pre', [6]), (20, 'A', 'main', [1]), (30, 'D', 'main', [1]))
        for actor, count in [(gs.Side.SHADOW, 1), (gs.Side.FREE_PEOPLES, 2)]:
            tape = rx._RecordedCombatDice([rounds], gs.Side.FREE_PEOPLES, {'detected': Counter()}, allow_special=True)
            tape.special_phase, tape.special_actor = 'pre', actor
            with self.assertRaises(rx._RecordedCombatDiceError): tape.draw(count)
        tape = rx._RecordedCombatDice([rounds], gs.Side.FREE_PEOPLES, {'detected': Counter()}, allow_special=True)
        with self.assertRaises(rx._RecordedCombatDiceError): tape.finish()

    def test_central_v52_admission_requires_bound_phase_evidence(self):
        import rendered_extract as rx
        from ai_policies import MIGHTY_PAYER
        from combat_choice_evidence import admission_refusal
        st, rounds = self.mighty()
        menu = ai.get_legal_actions(st, gs.Side.FREE_PEOPLES)
        evidence = {'schema': 'observed_round_v1', 'effect': 'mighty_attack',
            'round': 0, 'line': 30, 'reason': 'park_gesture',
            'observed_rolls': sorted(st.pending_combat['_round_resume']['cursor'].observed_rolls)}
        missing = rx._emit_sample(st, gs.Side.FREE_PEOPLES, menu, 0, 'combat_effect_choice', 1)
        self.assertEqual(missing['ai_policy_exclusion'], MIGHTY_PAYER)
        self.assertEqual(admission_refusal(st, None, menu[0]), 'combat_observed_evidence_missing')
        valid = rx._emit_observed_combat_sample(st, gs.Side.FREE_PEOPLES, menu, 0, 1, evidence)
        self.assertEqual(valid['ai_policy_exclusion'], MIGHTY_PAYER)
        self.assertIsNone(admission_refusal(st, evidence, menu[0]))
        self.assertEqual(valid['combat_choice_evidence'], evidence)
        self.assertFalse(hasattr(st, '_rx_combat_choice_evidence'))
        for delta in ({'effect': 'heroic_death'}, {'line': 0}, {'round': 1}, {'observed_rolls': []}):
            invalid = rx._emit_observed_combat_sample(st, gs.Side.FREE_PEOPLES, menu, 0, 1, evidence | delta)
            self.assertEqual(invalid['ai_policy_exclusion'], MIGHTY_PAYER)
            self.assertIn('combat_', admission_refusal(st, evidence | delta, menu[0]))
        clone = ai.clone_state(st)
        clone.pending_combat.pop('_round_resume')
        invalid = rx._emit_observed_combat_sample(clone, gs.Side.FREE_PEOPLES, menu, 0, 1, evidence)
        self.assertEqual(invalid['ai_policy_exclusion'], 'combat_observed_cursor_missing')

    def test_words_of_power_missing_target_or_later_death_is_not_a_cancellation_label(self):
        import rendered_extract as rx
        st, a, d = live_battle(fp_leaders=[companion('Aragorn', 3, 2), companion('Meriadoc')],
            sh_leaders=[gs.Nazgul(gs.Nation.SAURON)])
        st.pending_combat = {'phase': 'combat_effect_choice', 'attacker_region': a, 'defender_region': d,
            'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
            'deciding_side': gs.Side.SHADOW, 'rounds_fought': 0, 'is_siege': False,
            'effect_choice': {'stage': 'pre', 'effect': 'words_of_power', 'owner': 'def',
                'card_id': card_id('Words of Power'), 'targets': ['Aragorn', 'Meriadoc']}}
        menu = ai.get_legal_actions(st, gs.Side.SHADOW)
        self.assertEqual(len(menu), 2)
        from collections import defaultdict, Counter
        # Words of Power cancels a figure's abilities; a later death cannot
        # identify which earlier legal target the player chose. Both public
        # targets remain available, but neither is a proven corpus label.
        for evidence in (None, {'schema': 'observed_round_v1', 'effect': 'words_of_power',
                                'round': 0, 'line': 30, 'reason': 'figure_gesture',
                                'observed_rolls': [],
                                'record': kill(30, 'Aragorn', 'Fords of Isen')}):
            for index in range(len(menu)):
                with self.subTest(evidence=evidence, target=menu[index].params['target']):
                    st._rx_combat_choice_evidence = evidence
                    sample = rx._emit_sample(st, gs.Side.SHADOW, menu, index, 'combat_effect_choice', 1)
                    self.assertEqual(sample.get('ai_policy_exclusion'), 'combat_words_of_power_target_unproven')
                    self.assertEqual(rx._filter_ai_policy_samples([sample], defaultdict(Counter)), [])
        self.assertEqual({a.params['target'] for a in ai.get_legal_actions(st, gs.Side.SHADOW)},
                         {'Aragorn', 'Meriadoc'})

    def test_legacy_onslaught_recovery_without_round_observation_is_excluded(self):
        import rendered_extract as rx
        from test_onslaught_payments import position
        st = position()
        menu = ai.get_legal_actions(st, gs.Side.SHADOW)
        evidence = {'schema': 'observed_round_v1', 'effect': 'onslaught', 'round': 0,
            'line': 30, 'reason': 'physical_self_hits', 'observed_rolls': ['combat', 'reroll', 'ordinary_casualties']}
        valid = rx._emit_observed_combat_sample(st, gs.Side.SHADOW, menu, 1, 1, evidence)
        self.assertNotIn('ai_policy_exclusion', valid)
        clone = ai.clone_state(st)
        self.assertEqual(clone.pending_combat['_observed_round_completed'], st.pending_combat['_observed_round_completed'])
        self.assertIsNot(clone.pending_combat['_observed_round_completed'], st.pending_combat['_observed_round_completed'])
        count2 = next(a for a in menu if a.params['count'] == 2)
        clone = ai.apply_action(clone, count2)
        payment_menu = ai.get_legal_actions(clone, gs.Side.SHADOW)
        payment_evidence = evidence | {'effect': 'onslaught_payment', 'reason': 'physical_payment'}
        paid = rx._emit_observed_combat_sample(clone, gs.Side.SHADOW, payment_menu, 0, 1, payment_evidence)
        self.assertNotIn('ai_policy_exclusion', paid)
        clone.pending_combat.pop('_observed_round_completed')
        invalid = rx._emit_observed_combat_sample(clone, gs.Side.SHADOW, payment_menu, 0, 1, payment_evidence)
        self.assertEqual(invalid['ai_policy_exclusion'], 'combat_onslaught_observation_unbound')

    def test_next_round_clears_previous_completed_observation_marker(self):
        st, a, d = live_battle(fp_regulars=3,
            fp_leaders=[companion('Aragorn', 3, 2), companion('Meriadoc')])
        pc = {'phase': 'defender_combat_card', 'attacker_region': a, 'defender_region': d,
            'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
            'deciding_side': gs.Side.SHADOW, 'rounds_fought': 1, 'is_siege': False,
            'shine_asked': True, 'atk_combat_card': card_id('Mighty Attack'), 'def_combat_card': None,
            '_observed_round_completed': {'schema': 'observed_round_v1', 'round': 0}}
        st.pending_combat = pc
        with patch.object(cb, 'roll_d6_batch', Dice([1, 1, 1, 1], [6, 6, 6])):
            ai._run_combat_round(st, pc)
        self.assertEqual(pc['effect_choice']['effect'], 'mighty_attack')
        self.assertNotIn('_observed_round_completed', pc)
        self.assertEqual(pc['rounds_fought'], 1)

    def test_v51_archival_gate_is_unchanged_in_fresh_process(self):
        import subprocess, sys
        from pathlib import Path
        code = """import sys
sys.path.insert(0, sys.argv[1])
from test_combat_continuation_literal import live_battle, gs, card_id
import ai, rendered_extract as rx
assert not ai._COMBAT_ROUND_CONTEXT_ON
st,a,d=live_battle()
st.pending_combat={'phase':'combat_effect_choice','attacker_region':a,'defender_region':d,
'attacker_side':gs.Side.FREE_PEOPLES,'defender_side':gs.Side.SHADOW,'deciding_side':gs.Side.SHADOW,
'effect_choice':{'stage':'pre','owner':'def','effect':'relentless_assault','card_id':card_id('Relentless Assault'),'counts':[0,1,2]}}
la=ai.get_legal_actions(st,gs.Side.SHADOW)
s=rx._emit_sample(st,gs.Side.SHADOW,la,0,'combat_effect_choice',1)
assert 'combat_choice_refusal' not in s
assert 'ai_policy_exclusion' not in s
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parent)],
            env=dict(os.environ, WOTR_ACT_GEN='v51'), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)




class CorpusWordsPriority(unittest.TestCase):
    def position(self, fp_card=None, names=None):
        names=names or ['Gandalf the White','Aragorn','Gimli','Meriadoc']
        figures=[companion(n,3 if n not in ('Meriadoc','Peregrin') else 1,2 if n=='Aragorn' else 1) for n in names]
        for f in figures:f.captain_of_the_west=f.name in ('Aragorn','Gimli','Legolas')
        st,a,d=live_battle(fp_leaders=figures,sh_leaders=[gs.Nazgul(gs.Nation.SAURON)])
        st.pending_combat={'phase':'combat_effect_choice','attacker_region':a,'defender_region':d,
            'attacker_side':gs.Side.FREE_PEOPLES,'defender_side':gs.Side.SHADOW,'deciding_side':gs.Side.SHADOW,
            'rounds_fought':0,'is_siege':False,'atk_combat_card':card_id(fp_card) if fp_card else None,
            'def_combat_card':card_id('Words of Power'),
            'effect_choice':{'stage':'pre','effect':'words_of_power','owner':'def',
                'card_id':card_id('Words of Power'),'targets':names}}
        return st

    def matched(self, st, chats=()):
        from combat_choice_evidence import match_words_of_power
        menu=ai.get_legal_actions(st,gs.Side.SHADOW)
        result=match_words_of_power(st,{'records':[]},menu,845,847,
                    items((847,'A','main',[1]),(849,'D','main',[1])),chats)
        return menu,result

    def test_literal637_cancel_gandalf_is_observed_and_beats_anduril_fallback(self):
        from combat_choice_evidence import admission_refusal
        st=self.position('And\u00faril')
        menu,(index,evidence,reason)=self.matched(st,[(846,'cancel gandalf','SH human')])
        self.assertIsNotNone(index,reason)
        self.assertEqual(menu[index].params['target'],'Gandalf the White')
        self.assertEqual(evidence['provenance'],'observed')
        self.assertEqual(reason,'observed_cancel_chat')
        self.assertIsNone(admission_refusal(st,evidence,menu[index]))

    def test_user_priority_all_tiers_and_stable_tie_are_corpus_only(self):
        from combat_choice_evidence import admission_refusal
        for fp,names,expected in [(None,None,'Gandalf the White'),('And\u00faril',None,'Aragorn'),
             (None,['Legolas','Gimli','Meriadoc'],'Gimli'),(None,['Peregrin','Meriadoc'],'Meriadoc')]:
            with self.subTest(expected=expected):
                st=self.position(fp,names);original=pickle.dumps(st)
                menu,(index,evidence,reason)=self.matched(st)
                self.assertEqual(menu[index].params['target'],expected)
                self.assertEqual(evidence['provenance'],'user_authorized_inferred')
                self.assertIsNone(admission_refusal(st,evidence,menu[index]))
                self.assertEqual(pickle.dumps(st),original)
                self.assertEqual(len(ai.get_ai_policy(st,gs.Side.SHADOW).actions),len(menu))

    def test_uncovered_target_abstains_and_conflicting_observed_targets_do_not_guess(self):
        st=self.position(names=['Uncovered Companion'])
        self.assertEqual(self.matched(st)[1][2],'words_priority_has_no_covered_target')
        st=self.position()
        self.assertEqual(self.matched(st,[(846,'cancel gandalf','SH human'),
                     (846,'cancel aragorn','SH human')])[1][2],'words_conflicting_cancel_targets')

    def test_wrong_owner_late_or_tentative_chat_is_not_labeled_observed(self):
        for row in [(846,'cancel aragorn','FP human'),(848,'cancel aragorn','SH human'),
                    (846,'maybe cancel aragorn','SH human')]:
            st=self.position();menu,(index,evidence,_)=self.matched(st,[row])
            self.assertEqual(evidence['provenance'],'user_authorized_inferred')
            self.assertEqual(menu[index].params['target'],'Gandalf the White')

    def test_central_gate_binds_selected_action_round_cards_menu_and_priority(self):
        from combat_choice_evidence import admission_refusal
        st=self.position();menu,(index,evidence,_)=self.matched(st)
        other=next(a for a in menu if a.params['target']=='Aragorn')
        self.assertIsNotNone(admission_refusal(st,evidence,other))
        for key,value in [('target','Aragorn'),('round',1),('provenance','observed')]:
            bad=copy.deepcopy(evidence);bad[key]=value
            self.assertIsNotNone(admission_refusal(st,bad,menu[index]))
        changed=ai.clone_state(st);changed.pending_combat['atk_combat_card']=card_id('And\u00faril')
        self.assertIsNotNone(admission_refusal(changed,evidence,menu[index]))
        changed=ai.clone_state(st);changed.pending_combat['rounds_fought']=1
        self.assertIsNotNone(admission_refusal(changed,evidence,menu[index]))
        changed=ai.clone_state(st);changed.pending_combat['effect_choice']['targets']=['Aragorn','Gimli']
        self.assertIsNotNone(admission_refusal(changed,evidence,menu[index]))

    def test_actual_emitter_preserves_inferred_provenance_and_rejects_other_target(self):
        import rendered_extract as rx
        st=self.position();menu,(index,evidence,_)=self.matched(st)
        with patch.multiple(rx,_RX_RINGS_CUR=None,_RX_LINE_CUR=None,_RX_EYEALLOC=None,_RX_EYEROLL=None):
            sample=rx._emit_observed_combat_sample(st,gs.Side.SHADOW,menu,index,st.turn,evidence)
            self.assertNotIn('ai_policy_exclusion',sample)
            self.assertEqual(sample['combat_choice_evidence']['provenance'],'user_authorized_inferred')
            other=next(i for i,a in enumerate(menu) if a.params['target']=='Aragorn')
            bad=rx._emit_observed_combat_sample(st,gs.Side.SHADOW,menu,other,st.turn,evidence)
            self.assertEqual(bad['ai_policy_exclusion'],'combat_words_of_power_target_unproven')

    def test_reconstructed_captains_use_engine_identity_not_missing_metadata_flag(self):
        import rendered_extract as rx
        import rendered_replay as rr
        for captain in ('Legolas','Gimli','Boromir','Gandalf the Grey','Strider'):
            with self.subTest(captain=captain):
                snap=rr.Mirror().snapshot()
                snap['armies']={'fords of isen':([('ROHAN','REGULAR')],
                                      [('NAMED',captain),('NAMED','Meriadoc')]),
                                'orthanc':([('ISENGARD','REGULAR')],[('NAZGUL','SAURON')])}
                snap['fel']['comps']=[n for n in snap['fel']['comps'] if n not in (captain,'Meriadoc')]
                st=rx._recon_state(snap,1)
                st.pending_combat=self.position(names=[captain,'Meriadoc']).pending_combat
                figure=next(l for l in st.armies['fords of isen'].leaders if l.name==captain)
                self.assertIsInstance(figure,gs.Companion)
                self.assertFalse(figure.captain_of_the_west)  # Actual reconstruction defaults this flag.
                self.assertEqual(cb.calc_combat_strength(st.armies['fords of isen']),2)
                menu,(index,evidence,reason)=self.matched(st)
                self.assertIsNotNone(index,reason)
                self.assertEqual(menu[index].params['target'],captain)
                self.assertTrue(evidence['binding']['traits'][captain]['captain'])
                self.assertFalse(figure.captain_of_the_west)  # Reader does not rewrite the source figure.

if __name__ == '__main__':
    unittest.main(verbosity=2)
