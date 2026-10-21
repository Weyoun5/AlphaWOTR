"""Literal normalized snippets from admitted games855/681, not engine oracles.

855 batch49 lines496..503: ordinary SH3/FP1 deaths, then Onslaught selfhits4.
The reduced fixture preserves that recorded roster and actual pending hit
counts. Full captured-state replay and immutable input hashes live in
performance_pass/combat_choice_driver_recovery_r1; tests need no audit files.
"""
import os
os.environ['WOTR_ACT_GEN'] = 'v52'
import copy
import pickle
import unittest
from unittest.mock import patch
from test_combat_continuation_literal import live_battle, gs, cb, minion
import ai
import rendered_extract as rx
from combat_choice_evidence import match_onslaught_payment


CARD = 'SHADOW_Strategy_Orcs_Multiplying_Again'


def real_records():
    return {'records': [
        {'op': 'kill', 'line': 496, 'pieces': ['[3]Sauron Regular'], 'zone': 'Erebor'},
        {'op': 'kill', 'line': 497, 'pieces': ['North Regular'], 'zone': 'Erebor Stronghold'},
        {'op': 'kill', 'line': 499, 'pieces': ['S&E Elite'], 'zone': 'Erebor'},
        {'op': 'kill', 'line': 500, 'pieces': ['S&E Regular', 'Sauron Regular'], 'zone': 'Erebor'},
        {'op': 'kill', 'line': 503, 'pieces': ['North Regular'], 'zone': 'Erebor Stronghold'}]}


def captured_position():
    state, _, _ = live_battle()
    units = []
    for nation, kind, count in (
            ('THE_NORTH', 'REGULAR', 3), ('DWARVES', 'REGULAR', 1),
            ('ISENGARD', 'ELITE', 1), ('SAURON', 'REGULAR', 4),
            ('SOUTHRON_EASTERLINGS', 'ELITE', 1), ('SOUTHRON_EASTERLINGS', 'REGULAR', 4)):
        units.extend(gs.Unit(gs.Nation[nation], gs.UnitKind[kind]) for _ in range(count))
    state.armies = {'erebor': gs.Army('erebor', units,
        [gs.Leader(gs.Nation.THE_NORTH), gs.Leader(gs.Nation.DWARVES), minion('Witch King')]
        + [gs.Nazgul(gs.Nation.SAURON) for _ in range(5)])}
    state.witch_king.in_play = True
    state.pending_combat = {
        'attacker_region': 'erebor', 'defender_region': 'erebor',
        'attacker_side': gs.Side.SHADOW, 'defender_side': gs.Side.FREE_PEOPLES,
        'deciding_side': gs.Side.SHADOW, 'phase': 'casualty_allocation',
        'alloc_stage': 'attacker', 'atk_hits': 3, 'def_hits': 1,
        'is_siege': True, 'stronghold_region': True, 'rounds_fought': 1,
        'attacker_rearguard': ('erebor', 'SHADOW', ('forced_rearguard_v1', (), ())),
        'atk_casualty_policy': 'preserve_elites', 'def_casualty_policy': 'preserve_elites',
        'shine_asked': True,
        '_p9': {'atk_card': CARD, 'def_card': None, 'atk_cancelled': False, 'def_cancelled': False},
        '_observed_round_completed': {'schema': 'observed_round_v1', 'round': 0,
                                      'attacker_card': CARD, 'defender_card': None}}
    state._casualty_operations = []
    return state


def prefix(st, records=None):
    return rx._drive_onslaught_ordinary_prefix(st, records or real_records(),
                                              {'erebor', 'erebor stronghold'}, 495, 501)


def recorded_case70():
    """Actual855 batch70 local roster/hits and normalized direct-loss prefix."""
    st = captured_position()
    roster = [('GONDOR','REGULAR',3),('ELVES','ELITE',1),
              ('SOUTHRON_EASTERLINGS','ELITE',2),('SOUTHRON_EASTERLINGS','REGULAR',4),
              ('SAURON','ELITE',1)]
    units=[gs.Unit(gs.Nation[n],gs.UnitKind[k]) for n,k,count in roster for _ in range(count)]
    st.armies={'dol amroth':gs.Army('dol amroth',units,
        [minion('Witch King')]+[gs.Nazgul(gs.Nation.SAURON) for _ in range(3)])}
    pc=st.pending_combat
    pc.update(attacker_region='dol amroth',defender_region='dol amroth',atk_hits=2,def_hits=3,
        attacker_rearguard=('dol amroth','SHADOW',('forced_rearguard_v1',(),())))
    pc['_p9']['atk_card']='SHADOW_Strategy_Horde_from_the_East'
    pc['_observed_round_completed']['attacker_card']='SHADOW_Strategy_Horde_from_the_East'
    records={'records':[
        {'op':'kill','line':726,'pieces':['Elven Elite','Gondor Regular'],'zone':'Dol Amroth Stronghold'},
        {'op':'kill','line':727,'pieces':['Sauron Elite'],'zone':'Dol Amroth'},
        {'op':'kill','line':729,'pieces':['S&E Elite'],'zone':'Dol Amroth'},
        {'op':'move','line':730,'obj':'S&E Regular','from':'reinforcements','to':'Dol Amroth'}],
        'rng':[{'line':724,'kind':'roll_num','ints':[6]}]}
    return st,records


def commuting(st,records):
    return rx._drive_commuting_ordinary_cleanup(st,records,
        {'dol amroth','dol amroth stronghold'},725,737)


class RecordedCombatRecall(unittest.TestCase):
    def test_actual_ordinary_prefix_reaches_onslaught_without_using_self_or_bonus_losses(self):
        st = captured_position()
        before = pickle.dumps(st)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Premature post roll')):
            result, rows, boundary = prefix(st)
        self.assertEqual(boundary, 497)
        self.assertEqual(len(rows), 4)
        self.assertEqual([menu[i].params['nation'].name for _, menu, i in rows],
                         ['SAURON', 'SAURON', 'SAURON', 'THE_NORTH'])
        self.assertEqual(result.pending_combat['effect_choice']['effect'], 'onslaught')
        self.assertEqual(result.pending_combat['effect_choice']['max_count'], 4)
        self.assertEqual(len(result.armies['erebor'].units), 10)
        self.assertEqual(pickle.dumps(st), before)

    def test_actual_count_and_first_payment_recovered_later_mixed_order_not_invented(self):
        st, _, boundary = prefix(captured_position())
        rounds = [({'line': 501, 'ints': [1, 2, 3, 6]}, 'A', 'post')]
        def selected(s):
            menu = ai.get_legal_actions(s, gs.Side.SHADOW)
            i, line, reason = match_onslaught_payment(s, real_records(), menu,
                490, 512, rounds, self_loss_after_line=boundary)
            return menu, i, line, reason
        menu, i, _, _ = selected(st)
        self.assertEqual(menu[i].params['count'], 4)
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Premature post roll')):
            st = ai.apply_action(st, menu[i])
            menu, i, line, _ = selected(st)
            self.assertEqual(line, 499)
            self.assertEqual({k:v for k,v in menu[i].params.items() if k!='card_id'},
                             {'nation': 'SOUTHRON_EASTERLINGS', 'kind': 'elite', 'operation': 'remove'})
            st = ai.apply_action(st, menu[i])
        _, i, _, reason = selected(st)
        self.assertIsNone(i)
        self.assertIn('group_order_ambiguous', reason)

    def test_missing_ordinary_gesture_cannot_be_replaced_with_later_sacrifice(self):
        st = captured_position(); before = pickle.dumps(st)
        records = real_records(); records['records'] = records['records'][1:]
        with self.assertRaises(rx._CasualtyGestureError):
            prefix(st, records)
        self.assertEqual(pickle.dumps(st), before)

    def test_wrong_ordinary_count_and_reversed_true_choices_refuse(self):
        for change in ('count', 'order'):
            with self.subTest(change=change):
                records=real_records()
                if change=='count': records['records'][0]['pieces']=['[2]Sauron Regular']
                else:
                    records['records'][0]['line']=497
                    records['records'][1]['line']=496
                with self.assertRaises(rx._CasualtyGestureError): prefix(captured_position(), records)

    def test_onslaught_without_proven_boundary_does_not_guess_plain_kill_partition(self):
        st, _, _ = prefix(captured_position())
        menu = ai.get_legal_actions(st, gs.Side.SHADOW)
        i, _, reason = match_onslaught_payment(st, real_records(), menu,490,512,
            [({'line':501,'ints':[1,2,3,6]},'A','post')])
        self.assertIsNone(i)
        self.assertIn('boundary_unproven', reason)

    def test_actual_681_five_dice_are_not_silently_truncated_to_four(self):
        from collections import Counter
        tape = rx._RecordedCombatDice([[
            ({'line':615,'ints':[6,1,5,2,6]}, 'A','main'),
            ({'line':616,'ints':[5]},'D','main')]], gs.Side.SHADOW,
            {'detected':Counter()}, allow_special=True)
        with self.assertRaisesRegex(rx._RecordedCombatDiceError, 'main_count'):
            tape.draw(4, 'combat roll', gs.Side.SHADOW)
        self.assertEqual(tape.seen, set())

    def test_actual_case70_reversed_cleanup_recovers_endpoint_without_ordinary_labels(self):
        st,records=recorded_case70();before=pickle.dumps(st)
        with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Unexpected dice')):
            out,proof=commuting(st,records)
        self.assertFalse(proof['emits_ordinary_rows'])
        self.assertEqual(proof['suppressed_ordinary_rows'],2)
        self.assertEqual(proof['last_line'],727)
        self.assertEqual(out.pending_combat['phase'],'combat_round_result')
        self.assertEqual([(u.nation.name,u.kind.name) for u in out.armies['dol amroth'].units],
            [('GONDOR','REGULAR')]*2+[('SOUTHRON_EASTERLINGS','ELITE')]*2+
            [('SOUTHRON_EASTERLINGS','REGULAR')]*4)
        self.assertEqual(out.fp_casualties[gs.Nation.ELVES]['elite'],1)
        self.assertEqual(out.fp_casualties[gs.Nation.GONDOR]['regular'],1)
        self.assertEqual(pickle.dumps(st),before)

    def test_commuting_cleanup_refuses_random_event_replacement_and_wrong_hit_count(self):
        for change in ('rng','replacement','hits'):
            with self.subTest(change=change):
                st,records=recorded_case70();before=pickle.dumps(st)
                if change=='rng':records['rng'].append({'line':726,'kind':'draw'})
                elif change=='replacement':records['records'].insert(0,
                    {'op':'move','line':725,'obj':'Elven Regular','from':'reinforcements','to':'Dol Amroth Stronghold'})
                else:st.pending_combat['atk_hits']=1;before=pickle.dumps(st)
                with self.assertRaises(rx._CasualtyGestureError):commuting(st,records)
                self.assertEqual(pickle.dumps(st),before)

    def test_commuting_cleanup_accepts_direct_casualty_drags_with_same_literal_endpoint(self):
        for destination in ('FP Casualties', 'reinforcements'):
            with self.subTest(destination=destination):
                st, records = recorded_case70()
                first = records['records'][0]
                records['records'][0] = {'op': 'move', 'line': first['line'],
                    'obj': 'group (' + ', '.join(first['pieces']) + ')', 'from': first['zone'],
                    'to': destination}
                before = pickle.dumps(st)
                with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Unexpected dice')):
                    out, proof = commuting(st, records)
                self.assertEqual([(u.nation.name, u.kind.name) for u in out.armies['dol amroth'].units],
                    [('GONDOR', 'REGULAR')] * 2 + [('SOUTHRON_EASTERLINGS', 'ELITE')] * 2
                    + [('SOUTHRON_EASTERLINGS', 'REGULAR')] * 4)
                self.assertEqual(out.fp_casualties[gs.Nation.ELVES]['elite'], 1)
                self.assertEqual(out.fp_casualties[gs.Nation.GONDOR]['regular'], 1)
                self.assertEqual(proof['last_line'], 727)
                self.assertFalse(proof['emits_ordinary_rows'])
                self.assertEqual(pickle.dumps(st), before)

    def test_commuting_cleanup_refuses_retreats_and_mixed_leader_drags(self):
        for destination, obj in (('Dol Amroth', 'group (Elven Elite, Gondor Regular)'),
                                 ('FP Casualties', 'group (Elven Elite, Gondor Leader)')):
            with self.subTest(destination=destination, obj=obj):
                st, records = recorded_case70()
                records['records'][0] = {'op': 'move', 'line': 726, 'obj': obj,
                    'from': 'Dol Amroth Stronghold', 'to': destination}
                with self.assertRaises(rx._CasualtyGestureError):
                    commuting(st, records)

    def test_commuting_cleanup_does_not_generalize_to_wipe_or_unproved_prior_phase(self):
        for change in ('wipe','unobserved','already_applied'):
            with self.subTest(change=change):
                st,records=recorded_case70()
                if change=='wipe':
                    units=st.armies['dol amroth'].units
                    st.armies['dol amroth'].units=[units[0]]+units[3:]
                elif change=='unobserved':st.pending_combat.pop('_observed_round_completed')
                else:st._casualty_operations=[{'stage':'attacker'}]
                with self.assertRaises(rx._CasualtyGestureError):commuting(st,records)

    def test_actual_later_ra_window_starts_after_current_commit_not_previous_casualties(self):
        lower=rx._observed_round_choice_lower({'start_line':710},
            [{'last':724},{'last':739}], [{'sh_ln':711},{'sh_ln':735,'fp_ln':None}],1)
        self.assertEqual(lower,735)
        # Actual previous ordinary losses727 and press payment729/730 must
        # not be counted with the next round's RA Elite park736.
        self.assertEqual([n for n in (727,729,730,736) if lower<n<737],[736])


if __name__ == '__main__': unittest.main()
