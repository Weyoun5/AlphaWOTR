"""Literal initiative-2 casualty choices, before the next attack or Combat roll."""
import copy
import pickle
import unittest
from unittest.mock import patch

import combat as cb
import combat_dist as cd
import combat_round_resume as cr
from game_state import Nation, Unit, UnitKind


class PrecombatPayments(unittest.TestCase):
    def cursor(self, fp_regular=1, fp_elite=1, sh_regular=3,
               fp_card='Charge', sh_card="Durin's Bane"):
        context=cd.ExchangeContext(
            cd.ArmySpec(fp_regular,fp_elite,Nation.GONDOR),
            cd.ArmySpec(sh_regular,0,Nation.SAURON),
            attacker_card=fp_card,defender_card=sh_card,defer_casualties=True)
        return cr.RoundCursor(cd.build_round(context),{'defer_casualties':True})

    def test_defender_attack_then_owner_choice_changes_next_charge_dice(self):
        cursor=self.cursor();calls=[]
        with patch.object(cb,'roll_d6_batch',side_effect=lambda n,ctx='',side=None:
                          calls.append(n) or [4,1,1]):
            pending=cr.advance(cursor)
        self.assertEqual(calls,[3])
        self.assertEqual(pending['effect'],'pre_combat_casualty')
        self.assertEqual(pending['owner'],'atk')
        self.assertEqual(pending['options'],[
            {'nation':'GONDOR','kind':'regular','operation':'remove'},
            {'nation':'GONDOR','kind':'elite','operation':'downgrade'}])
        self.assertEqual(cursor.observed_roll_sides,{('pre_combat','defender')})
        self.assertEqual((cursor.pre_total,cursor.pre_remaining,cursor.pre_paid),(1,1,0))
        frozen=pickle.loads(pickle.dumps(cursor))
        # Removing the Regular preserves the Elite: Charge still rolls one die.
        regular=copy.deepcopy(frozen);batches=[]
        with patch.object(cb,'roll_d6_batch',side_effect=lambda n,ctx='',side=None:
                          batches.append(n) or [1]*n):
            self.assertIsNone(cr.advance(regular,pending['options'][0]))
        self.assertEqual(batches,[1,1,3])
        # Downgrading the Elite leaves two Regulars: zero Charge dice.
        elite=copy.deepcopy(frozen);batches=[]
        with patch.object(cb,'roll_d6_batch',side_effect=lambda n,ctx='',side=None:
                          batches.append(n) or [1]*n):
            self.assertIsNone(cr.advance(elite,pending['options'][1]))
        self.assertEqual(batches,[2,3])
        self.assertEqual([u.kind for u in elite.combat.attacker_army.units],
                         [UnitKind.REGULAR,UnitKind.REGULAR])
        self.assertEqual(cursor.combat.attacker_pre_combat_rolls,[])

    def test_two_hits_allow_elite_removal_and_wipe_prevents_all_later_rolls(self):
        cursor=self.cursor(fp_regular=0,fp_elite=1)
        with patch.object(cb,'roll_d6_batch',return_value=[4,4,1]):
            pending=cr.advance(cursor)
        self.assertEqual(pending['options'],[
            {'nation':'GONDOR','kind':'elite','operation':'downgrade'},
            {'nation':'GONDOR','kind':'elite','operation':'remove'}])
        with patch.object(cb,'roll_d6_batch',side_effect=AssertionError('Rolled after wipe')):
            self.assertIsNone(cr.advance(cursor,pending['options'][1]))
        self.assertTrue(cursor.combat.attacker_destroyed)
        self.assertTrue(cursor.combat.battle_over)
        self.assertEqual(cursor.pre_paid,2)

    def test_payment_nation_is_selected_and_invalid_operation_preserves_cursor(self):
        cursor=self.cursor(fp_elite=0,fp_card='Sudden Strike')
        cursor.combat.attacker_army.units.append(Unit(Nation.ELVES,UnitKind.REGULAR))
        with patch.object(cb,'roll_d6_batch',return_value=[4,1,1]):
            pending=cr.advance(cursor)
        before=pickle.dumps(cursor)
        with self.assertRaises(ValueError):
            cr.advance(cursor,{'nation':'ELVES','kind':'elite','operation':'remove'})
        self.assertEqual(pickle.dumps(cursor),before)
        with patch.object(cb,'roll_d6_batch',side_effect=lambda n,ctx='',side=None:[1]*n):
            cr.advance(cursor,{'nation':'ELVES','kind':'regular','operation':'remove'})
        self.assertEqual([u.nation for u in cursor.combat.attacker_army.units],[Nation.GONDOR])
        self.assertEqual([u.nation for u in cursor.combat.attacker_casualties],[Nation.ELVES])

    def test_two_attacks_pause_separately_without_repeating_first_roll(self):
        cursor=self.cursor();batches=[]
        def roll(n,ctx='',side=None):
            batches.append(n)
            return [4,1,1] if len(batches)==1 else [6]*n
        with patch.object(cb,'roll_d6_batch',side_effect=roll):
            first=cr.advance(cursor)
            second=cr.advance(cursor,first['options'][0])
        self.assertEqual(batches,[3,1])
        self.assertEqual(second['owner'],'def')
        self.assertEqual(cursor.pre_attack_index,1)
        self.assertEqual(cursor.combat.defender_pre_combat_rolls,[4,1,1])
        self.assertEqual(cursor.combat.attacker_pre_combat_rolls,[6])
        with patch.object(cb,'roll_d6_batch',side_effect=lambda n,ctx='',side=None:[1]*n):
            self.assertIsNone(cr.advance(cursor,second['options'][0]))
        self.assertEqual(len(cursor.combat.defender_army.units),2)


if __name__=='__main__':unittest.main()
