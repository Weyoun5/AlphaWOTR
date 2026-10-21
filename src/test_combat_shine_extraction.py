"""White Rider labels require an actual timely, correctly located flip."""
import os
os.environ['WOTR_ACT_GEN']='v52'
import copy
import pickle
import unittest
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
from test_combat_continuation_literal import live_battle, companion, card_id, gs, cb
import ai
import rendered_extract as rx
import rendered_census as rc


def position(eligible=True):
    st,a,d=live_battle(fp_attacks=False,fp_regulars=2,sh_regulars=2,
        fp_leaders=[companion('Gandalf the White',3,2)] if eligible else [],
        sh_leaders=[gs.Nazgul(gs.Nation.SAURON)])
    st.gandalf_the_white.in_play=eligible
    st.pending_combat={'attacker_region':a,'defender_region':d,
        'attacker_side':gs.Side.SHADOW,'defender_side':gs.Side.FREE_PEOPLES,
        'deciding_side':gs.Side.FREE_PEOPLES,'phase':'fp_shine',
        'rounds_fought':0,'is_siege':False}
    return st


def flip(line=15,region='Fords of Isen',player='FP human'):
    return {'op':'flip','obj':'Gandalf the White','region':region,'line':line,'player':player}


def round_items():
    return [[({'kind':'roll_num','line':30,'ints':[1,1],'player':'SH human'},'A','main'),
             ({'kind':'roll_num','line':31,'ints':[1,1],'player':'FP human'},'D','main')]]


def matched(st,records,cards=None):
    legal=ai.get_legal_actions(st,gs.Side.FREE_PEOPLES)
    return rx._match_shine_gesture(st,{'start_line':10,'records':records},legal,
        cards or [{'sh':None,'fp':None,'sh_ln':20,'fp_ln':21}],round_items())


class ShineEvidence(unittest.TestCase):
    def test_unobserved_default_and_decline_are_excluded_by_real_emitter(self):
        st=position();menu=ai.get_legal_actions(st,gs.Side.FREE_PEOPLES)
        for i in range(len(menu)):
            sample=rx._emit_sample(st,gs.Side.FREE_PEOPLES,menu,i,'fp_shine',1)
            self.assertTrue(sample.get('ai_policy_exclusion'))
            stats={'drop_reasons':Counter()}
            self.assertEqual(rx._filter_ai_policy_samples([sample],stats),[])

    def test_authentic_precommit_flip_is_admitted_and_effect_is_chosen_live(self):
        st=position();before=pickle.dumps(st)
        i,evidence,reason=matched(st,[flip()])
        self.assertEqual(reason,'observed_gandalf_flip')
        self.assertEqual(evidence['line'],15)
        self.assertEqual(pickle.dumps(st),before)
        menu=ai.get_legal_actions(st,gs.Side.FREE_PEOPLES)
        sample=rx._emit_shine_sample(st,gs.Side.FREE_PEOPLES,menu,i,1,evidence)
        self.assertNotIn('ai_policy_exclusion',sample)
        self.assertEqual(sample['shine_choice_evidence'],evidence)
        with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Roll before card selection')):
            result=ai.apply_action(st,menu[i])
        self.assertTrue(result.pending_combat['gtw_shine'])
        self.assertEqual(result.pending_combat['phase'],'attacker_combat_card')

    def test_missing_late_wrong_owner_wrong_region_and_duplicate_flip_abstain(self):
        for records in ([],[flip(22)],[flip(32)],[flip(player='SH human')],
                        [flip(region='Dale')],[flip(region='')],[flip(),flip(19)],
                        [flip(),flip(25)]):
            with self.subTest(records=records):
                st=position();before=pickle.dumps(st)
                self.assertIsNone(matched(st,records)[0])
                self.assertEqual(pickle.dumps(st),before)

    def test_missing_player_requires_actual_owned_figure_and_later_round_cannot_reuse_flip(self):
        record=flip();record.pop('player')
        i,evidence,_=matched(position(),[record])
        self.assertIsNotNone(i)
        self.assertEqual(evidence['owner_binding'],'fp_owned_figure')
        self.assertIsNone(matched(position(False),[record])[0])
        st=position();st.pending_combat['rounds_fought']=1
        self.assertIsNone(matched(st,[record])[0])


class LiveDrive(unittest.TestCase):
    def drive(self,records,word=False,chats=()):
        st=position(not word)
        if word:
            st.armies['fords of isen'].leaders=[companion('Aragorn',3,2),companion('Legolas',2,1)]
            st.hands[gs.Side.SHADOW].hand=[card_id('Words of Power')]
        st.pending_combat=None
        batch={'kind':'combat','die_side':'SHADOW','turn':1,'idx':1,'start_line':10,
               'truth_before':{},'truth':{},'records':records,
               'rng':[chunk for chunk,_,_ in round_items()[0]]}
        cards=[{'fp':None,'sh':card_id('Words of Power') if word else None,
                'fp_ln':None,'sh_ln':20 if word else None}]
        stats={'detected':Counter(),'drop_reasons':Counter(),'kinds_emit':Counter()}
        emissions=[];executed=[]
        original_apply=ai.apply_action
        original_emit=rx._emit_sample
        def apply(state,action):
            executed.append(action.action_type)
            return original_apply(state,action)
        def emit(state,side,legal,i,phase,turn):
            row={'phase':phase,'action':legal[i].action_type,
                 'params':dict(legal[i].params),
                 'shine_evidence':getattr(state,'_rx_shine_evidence',None),
                 'combat_evidence':getattr(state,'_rx_combat_choice_evidence',None)}
            if phase=='combat_effect_choice':
                actual=original_emit(state,side,legal,i,phase,turn)
                row['admission_refusal']=actual.get('ai_policy_exclusion')
                row['stored_evidence']=actual.get('combat_choice_evidence')
            emissions.append(row);return row
        with ExitStack() as stack:
            for owner,name,value in (
                (rx,'_recon_state',lambda *_:ai.clone_state(st)),
                (rx,'_install_ledger_hands',lambda *_args,**_kw:None),
                (rx,'_combat_round_windows',lambda *_args,**_kw:[{'first':30,'last':31,'next_first':10**9}]),
                (rx,'_round_cards',lambda *_:(cards,None,set())),
                (rx,'_combat_commit_board_matches',lambda *_:True),
                (rx,'_combat_final_ok',lambda *_:True),
                (rx,'_COMBAT_CHAT_INDEX',list(chats)),
                (rx,'_emit_sample',emit),
                (rc,'_plan_combat',lambda *_args,**_kw:('orthanc','fords of isen',False)),
                (rc,'_combat_chunk_rounds',lambda *_args,**_kw:round_items()),
                (ai,'apply_action',apply)):
                stack.enter_context(patch.object(owner,name,value))
            output=rx._emit_combat_family(batch,stats)
        return output,emissions,executed,stats

    def test_unobserved_shine_stops_real_driver_without_default_or_later_rows(self):
        output,emissions,actions,stats=self.drive([])
        self.assertEqual(output,[])
        self.assertEqual(emissions,[])
        self.assertEqual(actions,['army_attack'])
        self.assertTrue(any(k.startswith('cf_shine_choice_unproven:') for k in stats['drop_reasons']))
        self.assertNotIn('shine_defaulted',stats['detected'])

    def test_actual_flip_drives_shine_once_and_does_not_use_fallback(self):
        output,emissions,actions,stats=self.drive([flip()])
        shines=[r for r in emissions if r['phase']=='fp_shine']
        self.assertEqual(len(shines),1)
        self.assertEqual(shines[0]['shine_evidence']['line'],15)
        self.assertEqual(actions.count('shine'),1)
        self.assertNotIn('shine_none',actions)

    def test_words_user_fallback_is_inferred_and_not_selected_from_later_death(self):
        output,emissions,actions,stats=self.drive([
            {'op':'kill','line':40,'zone':'Fords of Isen','pieces':['Legolas']}],word=True)
        rows=[r for r in emissions if r['phase']=='combat_effect_choice']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['params']['target'],'Aragorn')
        self.assertEqual(rows[0]['combat_evidence']['schema'],'corpus_words_of_power_v1')
        self.assertEqual(rows[0]['combat_evidence']['provenance'],'user_authorized_inferred')
        self.assertIsNone(rows[0]['admission_refusal'])
        self.assertEqual(rows[0]['stored_evidence'],rows[0]['combat_evidence'])
        self.assertEqual(actions.count('combat_effect_choice'),1)
        self.assertEqual(stats['detected']['cc_words_of_power_user_authorized_inferred'],1)

    def test_words_authentic_current_owner_cancel_overrides_corpus_priority(self):
        _,emissions,actions,stats=self.drive([],word=True,
            chats=[(25,'cancel Legolas','SH human')])
        rows=[r for r in emissions if r['phase']=='combat_effect_choice']
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['params']['target'],'Legolas')
        self.assertEqual(rows[0]['stored_evidence']['provenance'],'observed')
        self.assertEqual(rows[0]['stored_evidence']['line'],25)
        self.assertIsNone(rows[0]['admission_refusal'])
        self.assertEqual(actions.count('combat_effect_choice'),1)
        self.assertEqual(stats['detected']['cc_words_of_power_observed'],1)

    def test_conflicting_words_cancellations_stop_before_target_and_aftermath(self):
        output,emissions,actions,stats=self.drive([],word=True,chats=[
            (25,'cancel Legolas','SH human'),(26,'cancel Aragorn','SH human')])
        self.assertNotIn('combat_effect_choice',actions)
        self.assertTrue(all(r['phase']=='combat_card' for r in emissions))
        self.assertEqual(stats['drop_reasons'][
            'cf_words_of_power_target_unproven:words_conflicting_cancel_targets'],1)


if __name__=='__main__':unittest.main()
