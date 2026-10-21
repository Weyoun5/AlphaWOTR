"""Literal native attack politics: activate/advance at declaration, once."""
from pathlib import Path
import sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
import ai,game_state as gs,game_logic as gl
from live.live_server import LiveServer
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
ROHAN=gs.Nation.ROHAN

class AttackPolitics(unittest.TestCase):
    def position(self,ai_side=SH,box=0,active=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        Path(temp.name,'ai_feed.txt').write_text('')
        server=LiveServer(temp.name,ai_side,load_nets=False,seed=61)
        state=server.driver.es
        state._declare_phase=False;state.dice_rolled=True
        state.political.boxes[gs.Nation.ISENGARD]=3
        state.political.boxes[ROHAN]=box;state.political.active[ROHAN]=active
        for side,dice in state.dice.items():
            for d in dice:d.used=True
        state.dice[SH][0].used=False;state.dice[SH][0].current=gs.ActionDieFace.ARMY
        server.driver.phase='actions';server.driver.last_actor=FP
        server.driver.view.piece_region.update({'099':'SA Dice','259':'Political '+str(3-box)})
        action=next(a for a in ai.get_legal_actions(state,SH) if a.action_type=='army_attack' and str(a.params['from']).lower()=='orthanc' and str(a.params['to']).lower()=='fords of isen')
        return server,state,action

    def lines(self,items):return [i[1] for i in items if i[0]=='line']
    def expected(self,nick='WOTR_AI_Shadow',box=0,flip=True):
        lines=[f'${nick}<~Controls.323~>259'] if flip else []
        control=(91,96,101)[box]
        lines.append(f'${nick}<~HighlightLabel.{control}~>259<~HighlightLabel.{control+1}~><~Game.{2000+box}~><~HighlightLabel.{control+2}~><~Game.{2001+box}~>')
        return lines

    def test_ai_declaration_emits_activation_and_step_before_card_choice(self):
        server,pre,action=self.position();server.fp_net=object();server.sh_net=object()
        dp={'kind':'action','side':SH,'actions':[action]};server._open_step=('ai_decide',dp)
        with patch.object(ai,'mcts_action_with_policy',return_value=(action,None)):
            out=server._ai_decide(dp)
        wanted=self.expected();self.assertEqual([l for l in out if '259' in l],wanted)
        self.assertFalse(any('combat_roll' in l for l in out))
        self.assertEqual(ai.describe_pending(server.driver.es)['kind'],'combat:attacker_combat_card')
        self.assertEqual(server.driver.es.political.boxes[ROHAN],1)
        self.assertTrue(server.driver.es.political.active[ROHAN])
        self.assertEqual(sum('Controls.276~>099' in l for l in out),1)

    def test_human_chat_attack_activates_before_advance(self):
        server,pre,action=self.position(FP);post=ai.apply_action(pre,action)
        items=server.driver.post_commit_sweep(pre,post,label='chat:attack',actions=[action])
        self.assertEqual([l for l in self.lines(items) if '259' in l],self.expected('WOTR_AI_FP'))

    def test_repeated_sweep_cannot_double_step_or_flip(self):
        server,pre,action=self.position();post=ai.apply_action(pre,action)
        first=self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action]))
        second=self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action]))
        self.assertEqual([l for l in first if '259' in l],self.expected())
        self.assertEqual([l for l in second if '259' in l],[])

    def test_already_active_nation_only_advances(self):
        server,pre,action=self.position(active=True);post=ai.apply_action(pre,action)
        out=self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action]))
        self.assertEqual([l for l in out if '259' in l],self.expected(flip=False))

    def test_at_war_does_not_move_or_flip(self):
        server,pre,action=self.position(box=3,active=True);post=ai.apply_action(pre,action)
        self.assertEqual([l for l in self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action])) if '259' in l],[])

    def test_expressed_marker_operations_are_not_emitted_twice(self):
        server,pre,action=self.position();post=ai.apply_action(pre,action)
        expressed=[('flip',{'pieces':['259']}),('political_91',{'piece':'259','frm':'Game.2000','to':'Game.2001'})]
        out=self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action],expressed=expressed))
        self.assertEqual([l for l in out if '259' in l],[])

    def test_physical_active_face_prevents_toggle_back(self):
        server,pre,action=self.position();post=ai.apply_action(pre,action)
        server.driver.view.dump_path='literal.native.board'
        with patch('live.client_protocol.read_marker_states',return_value={'259':1}):
            out=self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action]))
        self.assertEqual([l for l in out if '259' in l],self.expected(flip=False))

    def test_human_already_flipped_defending_nation(self):
        server,pre,action=self.position(FP);post=ai.apply_action(pre,action)
        gesture=SimpleNamespace(records=[{'op':'flip','obj':'Rohan','zone':'Political 3'}])
        gesture.body_records=lambda: gesture.records
        out=self.lines(server.driver.post_commit_sweep(pre,post,
            gesture=gesture,label='chat:attack',actions=[action]))
        self.assertEqual([l for l in out if '259' in l],self.expected('WOTR_AI_FP',flip=False))

    def test_mixed_defenders_activate_and_advance_each_nation(self):
        server,pre,action=self.position()
        gondor=gs.Nation.GONDOR
        source=gl.get_or_create_army(pre,'minas tirith')
        unit=next(u for u in source.units if u.nation is gondor)
        source.units.remove(unit);gl.get_or_create_army(pre,'fords of isen').units.append(unit)
        pre.political.boxes[gondor]=0;pre.political.active[gondor]=False
        post=ai.apply_action(pre,action)
        out=self.lines(server.driver.post_commit_sweep(pre,post,label='ai:action',actions=[action]))
        for pid,nation in [('259',ROHAN),('260',gondor)]:
            self.assertEqual(post.political.boxes[nation],1)
            self.assertTrue(post.political.active[nation])
            self.assertEqual([l for l in out if pid in l],[l.replace('259',pid) for l in self.expected()])

if __name__=='__main__':unittest.main()
