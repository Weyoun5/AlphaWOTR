"""Independent Onslaught supply, timing and exact-branch regressions."""
import pickle
import unittest
from unittest.mock import patch
import ai
import combat as cb
import game_state as gs
import game_logic as gl
from test_onslaught_payments import position, choose
from test_combat_continuation_literal import Dice


class OnslaughtEdges(unittest.TestCase):
    def test_actual_count_menu_is_hit_capacity_and_stale_larger_count_refuses(self):
        import onslaught_choices as oc
        from test_combat_continuation_literal import card_id
        st=position(regulars=1,elites=1)
        self.assertEqual([a.params['count'] for a in ai.get_legal_actions(st,gs.Side.SHADOW)],
                         [0,1,2,3])
        st.pending_combat['effect_choice']['max_count']=4
        before=pickle.dumps(st)
        with self.assertRaisesRegex(ValueError,'participating army'):
            oc.begin(st,st.pending_combat,{'card_id':card_id('Onslaught'),'count':4})
        self.assertEqual(pickle.dumps(st),before)

    def no_isengard_regular_supply(self,st):
        # All twelve printed Isengard Regulars stand outside this battle.
        for region in ('north dunland','south dunland'):
            st.armies[region]=gs.Army(region,[gs.Unit(gs.Nation.ISENGARD,gs.UnitKind.REGULAR) for _ in range(6)],[])
        self.assertEqual(ai.figures_available(st,gs.Nation.ISENGARD,'regular'),0)

    def test_unreplaceable_elite_selfwipe_rolls_only_actual_one_paid_hit(self):
        st=position(regulars=0,elites=1)
        self.no_isengard_regular_supply(st)
        st=ai.apply_action(st,choose(st,count=2))
        action=choose(st,kind='elite',operation='downgrade')
        parent=ai.MCTSNode(st,gs.Side.SHADOW)
        node=ai.MCTSNode(None,gs.Side.SHADOW,parent=parent,action=action)
        with patch.object(ai,'OVERLAY_ASSERT',False):
            self.assertTrue(ai._kernel_expand_onslaught_post(node))
        self.assertEqual(sorted(b.prior for b in node.outcomes),[.5,.5])
        with patch.object(cb,'roll_d6_batch',Dice([4])):
            done=ai.apply_action(st,action)
        self.assertNotIn('orthanc',done.armies)
        self.assertEqual(len(done.armies['fords of isen'].units),2)
        self.assertEqual(ai.figures_available(done,gs.Nation.ISENGARD,'regular'),0)

    def test_shadow_regular_returned_by_first_payment_can_fund_second_downgrade(self):
        st=position(regulars=1,elites=1)
        for region,n in (('north dunland',6),('south dunland',5)):
            st.armies[region]=gs.Army(region,[gs.Unit(gs.Nation.ISENGARD,gs.UnitKind.REGULAR) for _ in range(n)],[])
        self.assertEqual(ai.figures_available(st,gs.Nation.ISENGARD,'regular'),0)
        st=ai.apply_action(st,choose(st,count=2))
        with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Roll before final payment')):
            st=ai.apply_action(st,choose(st,kind='regular'))
        self.assertEqual(ai.figures_available(st,gs.Nation.ISENGARD,'regular'),1)
        with patch.object(cb,'roll_d6_batch',Dice([1,1])):
            st=ai.apply_action(st,choose(st,kind='elite',operation='downgrade'))
        self.assertEqual([u.kind for u in st.armies['orthanc'].units],[gs.UnitKind.REGULAR])
        self.assertEqual(ai.figures_available(st,gs.Nation.ISENGARD,'regular'),0)

    def test_selfwipe_eliminates_witchking_before_enemy_allocation_choice(self):
        st=position(regulars=1,elites=0)
        # Literal current post-round position: Witch-king accompanies the
        # last Shadow Regular; FP has both Regular and Elite targets.
        st.armies['orthanc'].leaders=[gl.Minion(gs.Nation.SAURON,'Witch King')]
        st.witch_king.in_play=True
        st.armies['fords of isen'].units[-1].kind=gs.UnitKind.ELITE
        st=ai.apply_action(st,choose(st,count=1))
        with patch.object(cb,'roll_d6_batch',Dice([4])):
            st=ai.apply_action(st,choose(st,kind='regular'))
        self.assertEqual(st.pending_combat['phase'],'casualty_allocation')
        self.assertFalse(st.witch_king.in_play)
        self.assertNotIn('orthanc',st.armies)
        # Completing the enemy loss cannot restore or re-eliminate the Minion.
        casualty=next(a for a in ai.get_legal_actions(st,gs.Side.FREE_PEOPLES)
                      if a.params.get('operation')=='remove_regular')
        with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Repeated Onslaught attack')):
            done=ai.apply_action(st,casualty)
        self.assertFalse(done.witch_king.in_play)
        self.assertEqual(len(done.armies['fords of isen'].units),2)

    def test_count_selection_and_nonfinal_payment_are_deterministic(self):
        st=position(regulars=2,elites=0)
        original=pickle.dumps(st)
        with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Premature attack')):
            chosen=ai.apply_action(st,choose(st,count=2))
            next_state=ai.apply_action(chosen,choose(chosen,kind='regular'))
        self.assertEqual(pickle.dumps(st),original)
        self.assertEqual(next_state.pending_combat['_onslaught_payment']['paid'],1)
        self.assertEqual(next_state.pending_combat['_onslaught_payment']['remaining'],1)


if __name__=='__main__':unittest.main()
