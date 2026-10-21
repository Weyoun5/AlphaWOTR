"""Literal companion-card group prefixes through production LiveDriver."""
from pathlib import Path
import shutil,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge'),str(ROOT)]
import ai,board,event_effects as ee,game_logic as gl,game_state as gs
from test_live_driver import fresh_state
from test_bridge_oct12_dead_men_prefix_live import Rig
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
CARDS=('Gwaihir the Windlord','We Prove Swifter','I Will Go Alone','There and Back Again')

class CompanionCardPrefixes(unittest.TestCase):
    def position(self,name,board_move=False):
        state=fresh_state();state.armies={};state._declare_phase=False;state.dice_rolled=True
        for pool in state.dice.values():
            for die in pool:die.used=True
        state.dice[FP][0].used=False;state.dice[FP][0].current=gs.ActionDieFace.CHARACTER
        if board_move:
            army=gl.get_or_create_army(state,'rivendell')
            for cname in ('Gimli','Legolas'):
                comp=next(c for c in state.fellowship.companions if c.name==cname)
                state.fellowship.companions.remove(comp);army.leaders.append(comp)
        card=ee._card_by_name(name);state.hands[FP].hand=[card.card_id]
        play=next(a for a in ai.get_legal_actions(state,FP) if a.action_type=='play_event' and a.params['card_id']==card.card_id)
        state=ai.apply_action(state,play)
        options=state.pending_card_option['options']
        target='bree'
        def names(o):
            arm=o.get('companion_action','')
            return set(o.get('companions',[])) or ({arm.split('_',1)[1]} if arm.startswith(('sep_','move_')) and arm not in ('move_group',) else set())
        relevant=[o for o in options if o.get('destination')==target and (not board_move or str(o.get('companion_action','')).startswith('move'))]
        self.assertTrue(any(names(o)=={'Gimli'} for o in relevant),(name,relevant))
        self.assertTrue(any(names(o)=={'Gimli','Legolas'} for o in relevant),(name,relevant))
        rig=Rig(state,SH,phase='actions');self.addCleanup(shutil.rmtree,rig.td)
        rig.drv.enforce_human=True;rig.drv.last_actor=FP
        return rig,('Rivendell' if board_move else 'Fellowship Box')

    def landed(self,rig):
        return {l.name for k,a in rig.drv.es.armies.items() if board.sanitize(k)=='bree' for l in a.leaders}

    def exercise_split(self,name,board_move=False):
        rig,source=self.position(name,board_move)
        rig.feed('$Wey moves Gimli from %s to Bree.'%source)
        first=(rig.drv.n_commits_total,bool(rig.drv.es.pending_card_option),self.landed(rig))
        rig.feed('$Wey moves Legolas from %s to Bree.'%source)
        rig.feed('<Wey> done')
        self.assertEqual(self.landed(rig),{'Gimli','Legolas'},(name,'first',first,'commits',rig.drv.n_commits_total,'alarms',rig.drv.alarms))
        self.assertEqual(rig.drv.n_commits_total,1)
        self.assertFalse(rig.drv.alarms)

    def test_completed_single_separation_positive_controls(self):
        for name in CARDS:
            with self.subTest(card=name):
                rig,source=self.position(name)
                rig.feed('$Wey moves Gimli from %s to Bree.'%source,'<Wey> done')
                self.assertEqual(self.landed(rig),{'Gimli'})
                self.assertEqual(rig.drv.n_commits_total,1)
                self.assertFalse(rig.drv.alarms)

    def test_full_native_board_group_positive_controls(self):
        for name in CARDS[:2]:
            with self.subTest(card=name):
                rig,source=self.position(name,True)
                rig.feed('$Wey moves group (Gimli, Legolas) from %s to Bree.'%source,'<Wey> done')
                self.assertEqual(self.landed(rig),{'Gimli','Legolas'})
                self.assertEqual(rig.drv.n_commits_total,1)
                self.assertFalse(rig.drv.alarms)

    def test_separation_split_has_intended_final_group(self):
        for name in CARDS:
            with self.subTest(card=name):self.exercise_split(name)

    def test_board_move_split_has_intended_final_group(self):
        for name in CARDS[:2]:
            with self.subTest(card=name):self.exercise_split(name,True)

    def test_full_native_group_positive_controls(self):
        for name in CARDS:
            with self.subTest(card=name):
                rig,source=self.position(name)
                rig.feed('$Wey moves group (Gimli, Legolas) from %s to Bree.'%source,'<Wey> done')
                self.assertEqual(self.landed(rig),{'Gimli','Legolas'})
                self.assertEqual(rig.drv.n_commits_total,1)
                self.assertFalse(rig.drv.alarms)

if __name__=='__main__':unittest.main()
