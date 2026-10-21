"""Literal corpus gestures; no label is generated from a model or rollout.

Lines cited here are from games637/681/855 normalized combat batches. Battle
fixtures deliberately isolate the recorded payments; they are not full replays.
"""
import unittest
from unittest.mock import patch
from test_combat_choice_extraction import live_battle, begin, publish, items, move, kill, cb, gs
from test_onslaught_payments import position, choose
import ai
from combat_choice_evidence import match_round_choice, match_onslaught_payment


def menu(st):
    return ai.get_legal_actions(st, st.pending_combat['deciding_side'])


def ra(regulars=2, elites=0):
    st,a,d=live_battle(sh_regulars=regulars, sh_elites=elites)
    with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Premature roll')):
        result=begin(st,a,d,sh_card='Relentless Assault')
    publish(st,a,d,result,None,'Relentless Assault')
    return st


class CombatEvidenceRecall(unittest.TestCase):
    def test_681_two_regular_payment_count_then_forced_individual_choices(self):
        st=ra(3)
        for u in st.armies['orthanc'].units:u.nation=gs.Nation.SAURON
        # Actual614: [2]Sauron Regular killed before main615. Location alone
        # is adapted from Minas Tirith to the isolated fixture's Orthanc.
        records=[kill(614,'[2]Sauron Regular','Orthanc'),kill(617,'Sauron Regular','Orthanc')]
        rounds=items((615,'D','main',[6,1,5,2,6]),(616,'A','main',[5]))
        idx,line,_=match_round_choice(st,{'records':records},menu(st),rounds,611,625)
        self.assertEqual(menu(st)[idx].params['count'],2)
        st=ai.apply_action(st,menu(st)[idx])
        idx,line,_=match_round_choice(st,{'records':records},menu(st),rounds,611,625)
        self.assertEqual(menu(st)[idx].params['nation'],'SAURON')
        self.assertEqual(line,614)
        st=ai.apply_action(st,menu(st)[idx])
        idx,_,reason=match_round_choice(st,{'records':records},menu(st),rounds,611,625)
        self.assertIsNotNone(idx,reason)

    def test_637_postroll_elite_death_does_not_prove_ra_count_or_decline(self):
        st=ra(1,1)
        records=[kill(973,'Isengard Elite','Orthanc')]
        rounds=items((968,'D','main',[1,6,4,3,5]),(970,'A','main',[6,1,4]))
        self.assertEqual(match_round_choice(st,{'records':records},menu(st),rounds,966,978)[2],
                         'missing_payment_gesture')
        self.assertIsNone(match_round_choice(st,{'records':[]},menu(st),rounds,966,978)[0])

    def test_855_parked_elite_confirmed_in_current_action_cleanup(self):
        st=ra(1,1)
        for u in st.armies['orthanc'].units:u.nation=gs.Nation.SOUTHRON_EASTERLINGS
        records=[move(736,'S&E Elite','Orthanc','Current SA Card'),
                 kill(743,'S&E Elite','Current Action')]
        rounds=items((737,'D','main',[2,1,5,4,1]),(738,'A','main',[1,4]))
        idx,line,reason=match_round_choice(st,{'records':records},menu(st),rounds,735,744)
        self.assertIsNotNone(idx,reason)
        self.assertEqual(menu(st)[idx].params['count'],2)
        self.assertEqual(line,736)

    def test_no_roll_ra_requires_literal_complete_army_removal(self):
        st=ra(2)
        records=[kill(10,'[2]Isengard Regular','Orthanc')]
        idx,_,reason=match_round_choice(st,{'records':records},menu(st),[],0,20)
        self.assertIsNotNone(idx,reason)
        self.assertEqual(menu(st)[idx].params['count'],2)
        st=ai.apply_action(st,menu(st)[idx])
        for _ in range(2):
            idx,_,reason=match_round_choice(st,{'records':records},menu(st),[],0,20)
            self.assertIsNotNone(idx,reason)
            with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Roll after selfwipe')):
                st=ai.apply_action(st,menu(st)[idx])
        partial=ra(3)
        self.assertEqual(match_round_choice(partial,{'records':records},menu(partial),[],0,20)[2],
                         'no_roll_payment_does_not_prove_wipe')

    def test_delayed_replacement_is_not_silently_labeled_two_hits(self):
        st=ra(1,1)
        records=[kill(10,'Isengard Elite','Orthanc'),move(25,'Isengard Regular','reinforcements','Orthanc')]
        rounds=items((20,'A','main',[1]),(21,'D','main',[1]))
        self.assertEqual(match_round_choice(st,{'records':records},menu(st),rounds,0,30)[2],
                         'replacement_after_payment_window')

    def test_855_unparked_onslaught_count_and_first_payment_recovered_without_order_guess(self):
        st=position(regulars=3,elites=1)
        army=st.armies['orthanc']
        army.units[0].nation=gs.Nation.SOUTHRON_EASTERLINGS
        army.units[1].nation=gs.Nation.SAURON
        army.units[-1].nation=gs.Nation.SOUTHRON_EASTERLINGS
        records=[kill(496,'[3]Sauron Regular','Orthanc'),
                 kill(497,'North Regular','Fords of Isen'),
                 kill(499,'S&E Elite','Orthanc'),
                 {'op':'kill','line':500,'pieces':['S&E Regular','Sauron Regular'],'zone':'Orthanc'}]
        rounds=items((490,'D','main',[4,6,4,4,4]),(493,'A','main',[4,3,4,5]),
                     (501,'D','post',[1,2,3,6]))
        args=(st,{'records':records},menu(st),489,508,rounds)
        self.assertEqual(match_onslaught_payment(*args)[2],'onslaught_self_loss_boundary_unproven')
        idx,line,reason=match_onslaught_payment(*args,self_loss_after_line=497)
        self.assertIsNotNone(idx,reason)
        self.assertEqual(menu(st)[idx].params['count'],4)
        st=ai.apply_action(st,menu(st)[idx])
        idx,line,reason=match_onslaught_payment(st,{'records':records},menu(st),489,508,rounds,
                                             self_loss_after_line=497)
        self.assertIsNotNone(idx,reason)
        self.assertEqual((menu(st)[idx].params['nation'],menu(st)[idx].params['operation']),('SOUTHRON_EASTERLINGS','remove'))
        st=ai.apply_action(st,menu(st)[idx])
        self.assertEqual(match_onslaught_payment(st,{'records':records},menu(st),489,508,rounds,
                         self_loss_after_line=497)[2],'payment_group_order_ambiguous')

    def test_terminal_regular_group_has_set_target_and_both_orders_same_final_state(self):
        from combat_choice_evidence import match_unordered_onslaught_regular_group
        from enum import Enum
        import dataclasses
        def signature(value):
            seen={}
            retained=[]
            def freeze(v):
                if isinstance(v,Enum): return (type(v).__name__,v.name)
                if v is None or isinstance(v,(str,int,float,bool)): return v
                if id(v) in seen:return ("ref",seen[id(v)])
                seen[id(v)]=len(seen)
                retained.append(v)
                if isinstance(v,dict):return tuple((freeze(k),freeze(x)) for k,x in sorted(v.items(),key=lambda t:str(t[0])))
                if isinstance(v,(list,tuple)):return tuple(freeze(x) for x in v)
                if isinstance(v,(set,frozenset)):return tuple(freeze(x) for x in sorted(v,key=str))
                if hasattr(v,"dtype") and hasattr(v,"tolist"):return (str(v.dtype),freeze(v.tolist()))
                if dataclasses.is_dataclass(v):return (type(v).__name__,freeze({f.name:getattr(v,f.name) for f in dataclasses.fields(v)}))
                if hasattr(v,"__dict__"):return (type(v).__name__,freeze(vars(v)))
                if hasattr(v,"__slots__"):return (type(v).__name__,freeze({k:getattr(v,k) for k in v.__slots__ if hasattr(v,k)}))
                raise TypeError(type(v).__name__)
            return freeze(value)
        st=position(regulars=3,elites=0)
        st.armies['orthanc'].units[0].nation=gs.Nation.SOUTHRON_EASTERLINGS
        st.armies['orthanc'].units[1].nation=gs.Nation.SAURON
        st=ai.apply_action(st,choose(st,count=2))
        records=[{'op':'kill','line':500,'pieces':['S&E Regular','Sauron Regular'],'zone':'Orthanc'}]
        rounds=items((490,'D','main',[1]),(493,'A','main',[1]),(501,'D','post',[1,2]))
        found=match_unordered_onslaught_regular_group(st,{'records':records},menu(st),489,508,rounds,
                                                     self_loss_after_line=497)
        self.assertEqual({menu(st)[i].params['nation'] for i in found['legal_indices']},
                         {'SAURON','SOUTHRON_EASTERLINGS'})
        self.assertEqual(found['physical_regular_losses'],{'SAURON':1,'SOUTHRON_EASTERLINGS':1})
        self.assertFalse(found['emit_intermediate_choices'])
        finals=[]
        for order in (found['canonical_options'],list(reversed(found['canonical_options']))):
            branch=ai.clone_state(st)
            with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Attack inside atomic group')):
                branch=ai.apply_action(branch,choose(branch,**order[0]))
            with patch.object(cb,'roll_d6_batch',return_value=[1,2]) as rolls:
                branch=ai.apply_action(branch,choose(branch,**order[1]))
            self.assertEqual(rolls.call_count,1)
            self.assertEqual([u.nation for u in branch.armies['orthanc'].units],[gs.Nation.ISENGARD])
            # The literal oracle is the exact recorded loss set; this full
            # graph comparison additionally checks side effects and supply.
            self.assertEqual(branch._casualty_operations[-1]['nation'].name,order[-1]['nation'])
            # Last-operation audit witnesses truthfully record each replay order;
            # they are not future game state and must not be falsely equal.
            finals.append({k:signature(v) for k,v in vars(branch).items() if k != '_casualty_operations'})
        self.assertEqual(finals[0],finals[1], [k for k in finals[0] if finals[0][k] != finals[1].get(k)])
        corrupt={'records':records+[move(501,'Isengard Regular','Orthanc','casualties')]}
        self.assertIsNone(match_unordered_onslaught_regular_group(st,corrupt,menu(st),489,508,items((502,'D','post',[1,2])),
                                                                  self_loss_after_line=497))

    def test_atomic_regular_group_refuses_intervening_rng_from_either_parser_view(self):
        from combat_choice_evidence import match_unordered_onslaught_regular_group
        st=position(regulars=3,elites=0)
        st.armies['orthanc'].units[0].nation=gs.Nation.SOUTHRON_EASTERLINGS
        st.armies['orthanc'].units[1].nation=gs.Nation.SAURON
        st=ai.apply_action(st,choose(st,count=2))
        records=[{'op':'kill','line':500,'pieces':['S&E Regular','Sauron Regular'],'zone':'Orthanc'}]
        rounds=items((490,'D','main',[1]),(493,'A','main',[1]),(503,'D','post',[1,2]))
        def check(batch, recorded):
            return match_unordered_onslaught_regular_group(st,batch,menu(st),489,508,recorded,
                                                           self_loss_after_line=497)
        self.assertIsNotNone(check({'records':records},rounds))
        with self.subTest(source='segmented_round_items'):
            self.assertIsNone(check({'records':records},rounds+items((501,'A','re',[6]))))
        with self.subTest(source='unsegmented_batch_rng'):
            self.assertIsNone(check({'records':records,'rng':[{'line':501,'kind':'roll_num','ints':[6]}]},rounds))
        with self.subTest(source='other_random_event'):
            self.assertIsNone(check({'records':records,'rng':[{'line':502,'kind':'draw_tile'}]},rounds))
        # The actual attack is the permitted boundary; earlier round rolls do
        # not invalidate the subsequent atomic payment group.
        self.assertIsNotNone(check({'records':records,'rng':[{'line':490,'kind':'roll_num','ints':[1]},
                                                            {'line':503,'kind':'roll_num','ints':[1,2]}]},rounds))

if __name__=='__main__':unittest.main()

