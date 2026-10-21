"""Independent physical-piece expectations for Onslaught's owner payments."""
import unittest
from unittest.mock import patch
import ai
import game_state as gs
import combat as cb
from test_combat_continuation_literal import live_battle, card_id, Dice
from test_onslaught_payments import position, choose


class OnslaughtIndependent(unittest.TestCase):
    def test_selected_nation_casualty_preserves_other_nation(self):
        st=position(regulars=2,elites=0)
        st.armies['orthanc'].units[1].nation=gs.Nation.SAURON
        st=ai.apply_action(st,choose(st,count=1))
        self.assertEqual({a.params['nation'] for a in ai.get_legal_actions(st,gs.Side.SHADOW)},
                         {'ISENGARD','SAURON'})
        with patch.object(cb,'roll_d6_batch',Dice([1])):
            st=ai.apply_action(st,choose(st,nation='SAURON'))
        self.assertEqual([u.nation for u in st.armies['orthanc'].units],[gs.Nation.ISENGARD])

    def test_attacker_peaceful_rearguard_cannot_pay_or_roll(self):
        st,a,d=live_battle(fp_attacks=False,sh_regulars=1,shadow_nation=gs.Nation.SAURON)
        st.armies[a].units.append(gs.Unit(gs.Nation.ISENGARD,gs.UnitKind.REGULAR))
        st.political.boxes[gs.Nation.ISENGARD]=2
        pc={'attacker_region':a,'defender_region':d,'attacker_side':gs.Side.SHADOW,
            'defender_side':gs.Side.FREE_PEOPLES,'is_siege':False,'shine_asked':True,
            'phase':'defender_combat_card','deciding_side':gs.Side.FREE_PEOPLES,
            'rounds_fought':0,'atk_combat_card':card_id('Onslaught'),'def_combat_card':None}
        st.pending_combat=pc
        from army_participants import freeze_attack
        freeze_attack(st,pc)
        with patch.object(cb,'roll_d6_batch',Dice([1],[1,1,1])):
            ai._run_combat_round(st,pc)
        st=ai.apply_action(st,choose(st,count=1))
        self.assertEqual([a.params['nation'] for a in ai.get_legal_actions(st,gs.Side.SHADOW)],['SAURON'])
        with patch.object(cb,'roll_d6_batch',Dice([1])):
            st=ai.apply_action(st,choose(st,nation='SAURON'))
        self.assertEqual([u.nation for u in st.armies['orthanc'].units],[gs.Nation.ISENGARD])
        self.assertEqual(len(st.armies['fords of isen'].units),3)

    def test_colocated_siege_selfwipe_keeps_enemy_garrison(self):
        st,a,d=live_battle(fp_attacks=False,sh_regulars=1,shadow_nation=gs.Nation.SAURON)
        fp=st.armies[d];sh=st.armies.pop(a)
        st.armies.pop(d)
        d='helms deep';fp.region=d;st.armies[d]=fp
        fp.units.extend(sh.units)
        st.besieged_strongholds.add(d)
        pc={'attacker_region':d,'defender_region':d,'attacker_side':gs.Side.SHADOW,
            'defender_side':gs.Side.FREE_PEOPLES,'is_siege':True,'shine_asked':True,
            'phase':'defender_combat_card','deciding_side':gs.Side.FREE_PEOPLES,
            'rounds_fought':0,'atk_combat_card':card_id('Onslaught'),'def_combat_card':None}
        st.pending_combat=pc
        from army_participants import freeze_attack
        freeze_attack(st,pc)
        with patch.object(cb,'roll_d6_batch',Dice([1],[1,1,1])):
            ai._run_combat_round(st,pc)
        st=ai.apply_action(st,choose(st,count=1))
        with patch.object(cb,'roll_d6_batch',Dice([1])):
            st=ai.apply_action(st,choose(st,nation='SAURON'))
        self.assertEqual([u.nation for u in st.armies[d].units],[gs.Nation.ROHAN]*3)
        self.assertIsNone(st.pending_combat)


if __name__=='__main__':unittest.main()
