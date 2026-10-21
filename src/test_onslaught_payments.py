"""Literal Onslaught payments through real AI actions, chance and clone paths."""
import pickle
import unittest
from unittest.mock import patch
from test_combat_continuation_literal import live_battle, card_id, Dice, cb, gs
import ai


def position(regulars=1, elites=1, fp_regulars=3):
    st, a, d = live_battle(fp_regulars=fp_regulars, sh_regulars=regulars, sh_elites=elites)
    pc = {'attacker_region': a, 'defender_region': d,
          'attacker_side': gs.Side.FREE_PEOPLES, 'defender_side': gs.Side.SHADOW,
          'phase': 'defender_combat_card', 'deciding_side': gs.Side.SHADOW,
          'rounds_fought': 0, 'is_siege': False, 'shine_asked': True,
          'atk_combat_card': None, 'def_combat_card': card_id('Onslaught')}
    st.pending_combat = pc
    with patch.object(cb, 'roll_d6_batch', Dice([1] * fp_regulars, [1] * (regulars + elites))):
        ai._run_combat_round(st, pc)
    if st.pending_combat['effect_choice']['effect'] != 'onslaught':
        raise AssertionError('Real combat failed to reach Onslaught')
    return st


def choose(st, **fields):
    side = st.pending_combat['deciding_side']
    return next(a for a in ai.get_legal_actions(st, side)
                if all(a.params.get(k) == v for k, v in fields.items()))


class OnslaughtPayments(unittest.TestCase):
    def test_count_then_each_unit_choice_precedes_roll(self):
        st = position()
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Roll before payment')):
            st = ai.apply_action(st, choose(st, count=2))
            self.assertEqual(st.pending_combat['effect_choice']['stage'], 'onslaught_payment')
            menu = ai.get_legal_actions(st, gs.Side.SHADOW)
            self.assertEqual({(a.params['kind'], a.params['operation']) for a in menu},
                             {('regular', 'remove'), ('elite', 'remove'), ('elite', 'downgrade')})
            st = ai.apply_action(st, choose(st, kind='regular', operation='remove'))
        before = pickle.dumps(st)
        for clone in (ai.clone_state(st), pickle.loads(pickle.dumps(st))):
            with patch.object(cb, 'roll_d6_batch', Dice([4, 1])):
                done = ai.apply_action(clone, choose(clone, kind='elite', operation='downgrade'))
            self.assertEqual([u.kind for u in done.armies['orthanc'].units], [gs.UnitKind.REGULAR])
            self.assertEqual(len(done.armies['fords of isen'].units), 2)
            self.assertNotIn('_onslaught_payment', done.pending_combat)
            self.assertEqual(done.pending_combat['rounds_fought'], 1)
            self.assertEqual(pickle.dumps(st), before)

    def test_remove_elite_consumes_two_hits_and_preserves_regular(self):
        st = position()
        st = ai.apply_action(st, choose(st, count=2))
        with patch.object(cb, 'roll_d6_batch', Dice([1, 1])):
            st = ai.apply_action(st, choose(st, kind='elite', operation='remove'))
        self.assertEqual([u.kind for u in st.armies['orthanc'].units], [gs.UnitKind.REGULAR])
        self.assertEqual(len(st.armies['fords of isen'].units), 3)

    def test_full_selfwipe_still_rolls_and_mutual_wipe_finishes(self):
        st = position(regulars=1, elites=0, fp_regulars=1)
        st = ai.apply_action(st, choose(st, count=1))
        with patch.object(cb, 'roll_d6_batch', Dice([6])):
            st = ai.apply_action(st, choose(st, kind='regular'))
        self.assertIsNone(st.pending_combat)
        self.assertNotIn('orthanc', st.armies)
        self.assertNotIn('fords of isen', st.armies)

    def test_decline_has_no_payment_no_roll(self):
        st = position()
        before = [(key, [u.kind for u in a.units]) for key, a in st.armies.items()]
        with patch.object(cb, 'roll_d6_batch', side_effect=AssertionError('Roll on decline')):
            st = ai.apply_action(st, choose(st, count=0))
        self.assertEqual([(key, [u.kind for u in a.units]) for key, a in st.armies.items()], before)
        self.assertNotIn('_onslaught_payment', st.pending_combat)

    def test_invalid_count_and_payment_fail_before_mutation(self):
        import onslaught_choices as oc
        st = position()
        before = pickle.dumps(st)
        with self.assertRaises(ValueError):
            oc.begin(st, st.pending_combat, {'card_id': card_id('Onslaught'), 'count': 99})
        self.assertEqual(pickle.dumps(st), before)
        st = ai.apply_action(st, choose(st, count=1))
        before = pickle.dumps(st)
        with self.assertRaises(ValueError):
            oc.pay(st, st.pending_combat, {'card_id': card_id('Onslaught'),
                   'kind': 'elite', 'nation': 'ISENGARD', 'operation': 'remove'})
        self.assertEqual(pickle.dumps(st), before)

    def test_final_payment_exact_chance_uses_full_paid_count(self):
        st = position(regulars=2, elites=0)
        st = ai.apply_action(st, choose(st, count=2))
        st = ai.apply_action(st, choose(st, kind='regular'))
        action = choose(st, kind='regular')
        parent = ai.MCTSNode(st, gs.Side.SHADOW)
        chance = ai.MCTSNode(None, gs.Side.SHADOW, parent=parent, action=action)
        with patch.object(ai, 'OVERLAY_ASSERT', False):
            self.assertTrue(ai._kernel_expand_onslaught_post(chance))
        # Two independent4+ dice: 0,1,2 hits have exact weights1/4,1/2,1/4.
        self.assertEqual(sorted(b.prior for b in chance.outcomes), [.25, .25, .5])
        self.assertEqual(chance.kernel_cover, 1.0)


if __name__ == '__main__':
    unittest.main()
