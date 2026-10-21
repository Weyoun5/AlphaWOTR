"""Literal public-roll controls and actual cursor menus for appended encoding."""
import os
os.environ.setdefault('WOTR_ACT_GEN','v52')
os.environ.setdefault('WOTR_NATIVE','0')
import copy
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import combat as cb
import combat_dist as cd
import combat_round_resume as cr
import combat_round_observation as obs
import ai
import board
from game_state import Nation, Side, setup_initial_game

class CombatObservation(unittest.TestCase):
    def setUp(self):
        self.state=setup_initial_game(board.load_board())

    def pause(self,context,batches):
        cursor=cr.RoundCursor(cd.build_round(context),{'defer_casualties':True})
        rolls=iter(batches)
        with patch.object(cb,'roll_d6_batch',side_effect=lambda n,ctx='',side=None:next(rolls)):
            pending=cr.advance(cursor)
        c=cursor.combat
        self.state.pending_combat={'_round_resume':{'cursor':cursor},'phase':'combat_effect_choice',
            'attacker_side':c.attacker_side,'defender_side':c.defender_side,
            'attacker_region':c.attacker_region,'defender_region':c.defender_region,
            'deciding_side':c.attacker_side if pending['owner']=='atk' else c.defender_side,
            'effect_choice':dict(pending,stage='round',card_id='')}
        return cursor

    def menu_encodings(self):
        actions=ai._get_pending_combat_actions(self.state)
        encoded=[ai.encode_action(a,self.state,self.state.pending_combat['deciding_side']) for a in actions]
        self.assertTrue(all(len(x)==133 for x in encoded))
        self.assertEqual(len({x.tobytes() for x in encoded}),len(encoded))
        observation=ai.encode_state(self.state,self.state.pending_combat['deciding_side'])
        self.assertEqual(len(observation),obs.TARGET_STATE_DIM)
        np.testing.assert_array_equal(observation[-obs.STATE_TAIL_DIM:],obs.encode_state_tail(self.state))
        return actions,encoded

    def test_actual_mighty_hobbit_alias_menu_and_future_roll_mask(self):
        leaders=(cd.LeaderSpec('companion','Aragorn',3,2),cd.LeaderSpec('companion','Meriadoc Brandybuck',1,1))
        context=cd.ExchangeContext(cd.ArmySpec(3,0,Nation.GONDOR,leaders=leaders),cd.ArmySpec(3,0,Nation.SAURON),attacker_card='Mighty Attack',defer_casualties=True)
        cursor=self.pause(context,[[1]*4,[1]*3])
        actions,_=self.menu_encodings()
        self.assertEqual([x.params['target'] for x in actions],['Aragorn','Meriadoc Brandybuck'])
        before=obs.encode_state_tail(self.state)
        self.assertEqual(before[26:32].tolist(),[1,1,0,1,1,0])
        cursor.combat.attacker_rerolls=[6,6,6]
        cursor.combat.defender_rerolls=[2,3]
        cursor.combat._resume_atk_reroll_hits=3
        np.testing.assert_array_equal(before,obs.encode_state_tail(self.state))
        self.assertEqual(before[88+5],1)

    def test_actual_fateful_generic_nazgul_and_named_minions_are_distinct(self):
        leaders=(cd.LeaderSpec('nazgul'),cd.LeaderSpec('minion','Witch King',nation=Nation.SAURON),cd.LeaderSpec('minion','Mouth of Sauron',nation=Nation.SAURON))
        context=cd.ExchangeContext(cd.ArmySpec(3,0,Nation.GONDOR,leaders=(cd.LeaderSpec('leader'),)*3),cd.ArmySpec(3,0,Nation.SAURON,leaders=leaders),attacker_card='Fateful Strike',defer_casualties=True)
        self.pause(context,[[1]*3,[6]*3,[6,6,1]])
        actions,_=self.menu_encodings()
        self.assertEqual({a.params['target'] for a in actions},{'nazgul','Witch King','Mouth of Sauron'})
        self.assertEqual(obs.encode_state_tail(self.state)[26:32].tolist(),[1]*6)

    def test_actual_black_breath_decline_and_leader_menu(self):
        context=cd.ExchangeContext(cd.ArmySpec(1,0,Nation.GONDOR,leaders=(cd.LeaderSpec('companion','Peregrin Took',1,1),cd.LeaderSpec('leader'))),cd.ArmySpec(2,0,Nation.SAURON,leaders=(cd.LeaderSpec('nazgul'),)),attacker_card='Heroic Death',defender_card='Black Breath',defer_casualties=True)
        self.pause(context,[[6],[1,1],[6]])
        actions,enc=self.menu_encodings()
        self.assertIn({'card_id':'','count':0},[a.params for a in actions])
        self.assertEqual(sum(x[106] for x in enc),1)

    def test_all_nation_payment_operation_encodings_are_distinct(self):
        self.state.pending_combat={'_round_resume':{'cursor':SimpleNamespace(schema='combat_round_cursor_v1')}}
        values=[]
        # Drive the public Nation enum, not the encoder's own lookup table.
        for nation in Nation:
            for kind,operation in [('regular','remove'),('elite','remove'),('elite','downgrade')]:
                action=ai.Action(-1,'combat_effect_choice',{'nation':nation,'kind':kind,'operation':operation})
                values.append(obs.encode_action_tail(action,self.state).tobytes())
        self.assertEqual(len(set(values)),24)

    def test_north_preattack_payment_menu_uses_real_nation_identity(self):
        context = cd.ExchangeContext(
            cd.ArmySpec(1, 1, Nation.THE_NORTH),
            cd.ArmySpec(3, 0, Nation.SAURON),
            defender_card="Durin's Bane", defer_casualties=True)
        cursor = self.pause(context, [[4, 1, 1]])
        self.assertEqual(cursor.pending['owner'], 'atk')
        actions, encodings = self.menu_encodings()
        self.assertEqual({a.params['nation'] for a in actions}, {'THE_NORTH'})
        for encoding in encodings:
            self.assertEqual(encoding[-obs.ACTION_TAIL_DIM:][18:22].tolist(), [1, 0, 1, 0])

    def test_actual_preattack_casualty_masks_unexecuted_other_roll(self):
        context=cd.ExchangeContext(cd.ArmySpec(0,1,Nation.GONDOR),cd.ArmySpec(1,1,Nation.SAURON),attacker_card='Charge',defer_casualties=True)
        cursor=self.pause(context,[[6]])
        self.assertEqual(cursor.pending['effect'],'pre_combat_casualty')
        self.menu_encodings()
        tail=obs.encode_state_tail(self.state)
        self.assertEqual(tail[120:125].tolist(),[1,1,np.float32(.1),0,1])
        self.assertEqual(tail[26:32].tolist(),[1,0,0,0,0,0])
        # Defender's preattack phase has not executed. Neither side's future
        # combat roll may leak into this payment observation.
        cursor.combat.attacker_combat_rolls=[6]*5
        cursor.combat.defender_combat_rolls=[6]*5
        cursor.combat.defender_pre_combat_rolls=[6]*5
        np.testing.assert_array_equal(tail,obs.encode_state_tail(self.state))

    def test_onslaught_count_and_payment_have_context_without_cursor(self):
        self.state.pending_combat={'phase':'combat_effect_choice','effect_choice':{
            'stage':'post','effect':'onslaught','owner':'atk','max_count':2,'card_id':''}}
        count=obs.encode_state_tail(self.state)
        self.assertEqual(count[125],1)
        self.assertFalse(count[26:68].any())
        self.state.pending_combat.update(effect_choice={'stage':'onslaught_payment','effect':'onslaught_payment','owner':'atk'},
            _onslaught_payment={'owner':'atk','card_id':'','count':2,'remaining':1,'paid':1})
        payment=obs.encode_state_tail(self.state)
        self.assertEqual(payment[126],1)
        self.assertFalse(payment[26:68].any())
        a=ai.Action(-1,'combat_effect_choice',{'nation':'SAURON','kind':'elite','operation':'downgrade'})
        b=ai.Action(-1,'combat_effect_choice',{'nation':'SAURON','kind':'regular','operation':'remove'})
        self.assertNotEqual(obs.encode_action_tail(a,self.state).tobytes(),obs.encode_action_tail(b,self.state).tobytes())

    def test_paid_names_survive_pickle_and_deepcopy(self):
        leaders=(cd.LeaderSpec('companion','Aragorn',3,2),cd.LeaderSpec('companion','Meriadoc Brandybuck',1,1))
        context=cd.ExchangeContext(cd.ArmySpec(3,0,Nation.GONDOR,leaders=leaders),cd.ArmySpec(3,0,Nation.SAURON),attacker_card='Mighty Attack',defer_casualties=True)
        cursor=self.pause(context,[[1]*4,[1]*3])
        cursor.combat._leadership_forfeited={'Meriadoc Brandybuck'}
        before=obs.encode_state_tail(self.state)
        self.assertEqual(before[108+5],1)
        np.testing.assert_array_equal(before,obs.encode_state_tail(copy.deepcopy(self.state)))
        np.testing.assert_array_equal(before,obs.encode_state_tail(pickle.loads(pickle.dumps(self.state))))

    def test_outside_cursor_zero_and_unknown_schema_refused(self):
        self.assertFalse(obs.encode_state_tail(self.state).any())
        self.state.pending_combat={'_round_resume':{'cursor':SimpleNamespace(schema='unknown')}}
        with self.assertRaises(ValueError):obs.encode_state_tail(self.state)

if __name__=='__main__':unittest.main()
