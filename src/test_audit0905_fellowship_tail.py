"""Almanac C17: public Balrog→Hunt→Guide chain precedes entry/Guide/eyes."""
from pathlib import Path
import contextlib,importlib.util,io,json,os,random,sys,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(WOTR_NATIVE='0',WOTR_ACT_GEN='v41',WOTR_REWARD_CONTRACT='predecision_profile_v1',CUDA_VISIBLE_DEVICES='-1')
import ai,board,game_logic as gl,game_state as gs
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
HERE=ROOT/'audit/2026-09-04/decision_pass/guide_phase_review'
spec=importlib.util.spec_from_file_location('balrog_timing_probe',HERE/'balrog_timing_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class FellowshipTail(unittest.TestCase):
 def setUp(self):
  self.profiles=dict(ai.PLAYER_PROFILES);ai.PLAYER_PROFILES.update(fp='ring',sh='military')
  self.addCleanup(lambda:(ai.PLAYER_PROFILES.clear(),ai.PLAYER_PROFILES.update(self.profiles)))
 def pick(self,s,side,kind,**params):
  acts=[a for a in ai.get_ai_actions(s,side) if a.action_type==kind and all(a.params.get(k)==v for k,v in params.items())]
  self.assertEqual(len(acts),1,acts);return acts[0]
 def balrog(self,s,draw):
  acts=[a for a in ai.get_ai_actions(s,SH) if a.action_type=='card_option' and a.params.get('choices',{}).get('balrog_declare_draw')==draw]
  self.assertEqual(len(acts),1,acts);return acts[0]
 def test_stationary_balrog_can_be_kept_before_guide_and_eyes(self):
  s,_=probe.position();self.assertFalse(gl.begin_fellowship_tail(s))
  self.assertTrue(s._fellowship_tail_phase);self.assertFalse(getattr(s,'_eye_allocation_phase',False))
  self.assertEqual(ai._pending_decision_side(s),SH)
  ai._execute_action(s,self.balrog(s,False));self.assertTrue(gl.resume_fellowship_tail(s))
  self.assertFalse(s._fellowship_tail_phase);self.assertIsNotNone(s.pending_guide_change)
  self.assertTrue(s._eye_allocation_phase);self.assertEqual(len(s.hunt_discard),0)
  self.assertTrue(gl.begin_fellowship_tail(s));self.assertIsNone(s.pending_card_option)
 def test_damage_and_compulsory_guide_finish_before_optional_guide(self):
  s,_=probe.position();s.fellowship.companions=[c for c in s.fellowship.companions if c.name in ('Gimli','Legolas','Boromir')]
  s.fellowship.guide=next(c for c in s.fellowship.companions if c.name=='Gimli')
  tile=next(t for t in s.hunt_pool if t.damage==2 and not t.reveal and not t.is_eye)
  s.hunt_pool.remove(tile);s.hunt_pool.append(tile)
  gl.begin_fellowship_tail(s);ai._execute_action(s,self.balrog(s,True))
  self.assertFalse(gl.resume_fellowship_tail(s));self.assertEqual(s.pending_hunt['damage'],2)
  ai._execute_action(s,self.pick(s,FP,'hunt_sacrifice_guide'))
  self.assertIsNotNone(s.pending_guide_change);self.assertFalse(gl.resume_fellowship_tail(s))
  self.assertFalse(getattr(s,'_eye_allocation_phase',False));self.assertNotIn('Gimli',s.pending_guide_change['candidates'])
  ai._execute_action(s,self.pick(s,FP,'choose_guide',companion='Legolas'))
  self.assertTrue(gl.resume_fellowship_tail(s));self.assertTrue(s._eye_allocation_phase)
  self.assertEqual(set(s.pending_guide_change['candidates']),{'Legolas','Boromir'})
  self.assertEqual(s.fellowship.corruption,0);self.assertEqual(len(s.hunt_discard),1)
 def test_gateway_refill_waits_for_balrog_and_damage(self):
  s,_=probe.position(8);s._declare_phase=True
  ai._execute_action(s,self.pick(s,FP,'fellowship_declare',region='morannon'))
  tile=next(t for t in s.hunt_pool if t.damage==2 and not t.reveal and not t.is_eye)
  s.hunt_pool.remove(tile);s.hunt_pool.append(tile)
  with patch.object(gl,'_restore_hunt_pool_for_mordor',wraps=gl._restore_hunt_pool_for_mordor) as refill:
   gl.begin_fellowship_tail(s);self.assertFalse(s.fellowship.mordor_entered);refill.assert_not_called()
   with self.assertRaisesRegex(RuntimeError,'effects must finish'):gl._check_mordor_entry(s)
   ai._execute_action(s,self.balrog(s,True));gl.resume_fellowship_tail(s);refill.assert_not_called()
   self.assertFalse(s.fellowship.mordor_entered)
   ai._execute_action(s,self.pick(s,FP,'hunt_take_corruption'));gl.resume_fellowship_tail(s)
   self.assertTrue(s.fellowship.mordor_entered);refill.assert_called_once_with(s)
  self.assertEqual(s.fellowship.corruption,2);self.assertIn(tile,s.hunt_discard)
 def test_sanctioned_auto_declaration_uses_same_balrog_trigger(self):
  s,_=probe.position(8);self.assertTrue(ai._auto_enter_mordor(s))
  self.assertIn(gl.fellowship_region(s),('morannon','minas morgul'))
  self.assertEqual(s.pending_card_option['card_name'],'Balrog of Moria')
  self.assertFalse(s.fellowship.mordor_entered);self.assertFalse(gl.begin_fellowship_tail(s))
 def test_search_analytic_hunt_continuation_preserves_phase(self):
  s,_=probe.position();gl.begin_fellowship_tail(s)
  parent=ai.MCTSNode(state=s,side=SH,parent=None,action=None)
  node=ai.MCTSNode(state=None,side=SH,parent=parent,action=self.balrog(s,False))
  self.assertTrue(ai._hunt_action_gate(node));kind,result=ai._hunt_expand_try(node)
  self.assertEqual(kind,'det');self.assertFalse(result._fellowship_tail_phase)
  self.assertTrue(result._eye_allocation_phase);self.assertIsNotNone(result.pending_guide_change)
  self.assertFalse(result.dice_rolled);self.assertEqual(result.turn,s.turn)
  self.assertTrue(s._fellowship_tail_phase);self.assertIsNotNone(s.pending_card_option)
 def test_analytic_draw_enumerates_real_tiles_and_never_leaks_to_actions(self):
  s,_=probe.position();gl.begin_fellowship_tail(s)
  parent=ai.MCTSNode(state=s,side=SH,parent=None,action=None)
  node=ai.MCTSNode(state=None,side=SH,parent=parent,action=self.balrog(s,True))
  rng=random.getstate();kind,result=ai._hunt_expand_try(node)
  self.assertEqual(kind,'chance');self.assertEqual(random.getstate(),rng)
  self.assertGreater(len(node.outcomes),1)
  states=[n.state for n in node.outcomes]
  self.assertTrue(all(not st.dice_rolled and st.turn==s.turn for st in states))
  self.assertTrue(any(st.pending_hunt is not None for st in states))
  for st in states:
   self.assertFalse(st.pending_card_option)
   if st.pending_hunt is not None:
    self.assertTrue(st._fellowship_tail_phase);self.assertFalse(getattr(st,'_eye_allocation_phase',False))
   else:
    self.assertFalse(st._fellowship_tail_phase);self.assertTrue(st._eye_allocation_phase)
 def test_tail_has_its_own_appended_lane_clone_and_cache_identity(self):
  s,_=probe.position();ai._balrog_stationary_turn_start(s)
  mid=ai.clone_state(s);phase=ai.clone_state(s);gl.begin_fellowship_tail(phase)
  a,b=ai.encode_state(mid,SH),ai.encode_state(phase,SH)
  self.assertEqual(len(b),3746);np.testing.assert_array_equal(a[:-1],b[:-1]);self.assertEqual((a[-1],b[-1]),(0,1))
  self.assertNotEqual(ai._state_sig(mid),ai._state_sig(phase));self.assertNotEqual(ai._chance_sig(mid),ai._chance_sig(phase))
  clone=ai.clone_state(phase);np.testing.assert_array_equal(ai.encode_state(clone,SH),b)
  self.assertEqual(ai._state_sig(clone),ai._state_sig(phase))
  with self.assertRaisesRegex(RuntimeError,'Earlier Fellowship'):gl.open_fellowship_guide_choice(phase)
  with self.assertRaisesRegex(RuntimeError,'refusing rollover'):ai._safe_next_turn(phase)
 def test_actual_episode_resolves_balrog_before_first_guide_or_eye(self):
  s,_=probe.position();order=[]
  def choose(st,side,*args,**kwargs):
   acts=list(ai.get_ai_actions(st,side));kinds={a.action_type for a in acts}
   if 'fellowship_declare' in kinds:a=self.pick(st,side,'fellowship_declare',region='moria')
   elif 'card_option' in kinds:a=self.balrog(st,False)
   elif 'choose_guide' in kinds:a=self.pick(st,side,'choose_guide',companion='Strider')
   elif 'eye_allocate' in kinds:a=self.pick(st,side,'eye_allocate',count=0)
   else:raise AssertionError(kinds)
   order.append(a.action_type);return a,None
  def stop(st,**kw):
   self.assertIsNone(st.pending_card_option);self.assertIsNone(st.pending_guide_change)
   self.assertFalse(st._fellowship_tail_phase);raise probe.BeforeRoll()
  with patch.object(gs,'setup_initial_game',return_value=s),patch.object(gl,'initialize_decks'),patch.object(ai,'mcts_action_with_policy',side_effect=choose),patch.object(gl,'roll_action_dice',side_effect=stop),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(probe.BeforeRoll):ai._run_single_episode(None,None,{'max_turns':1,'mcts_sims':1})
  self.assertEqual(order,['fellowship_declare','card_option','choose_guide','eye_allocate'])
 def test_missing_tail_selection_is_an_error_before_guide_or_roll(self):
  s,_=probe.position()
  def choose(st,side,*args,**kwargs):
   if getattr(st,'_declare_phase',False):return self.pick(st,FP,'pass'),None
   return None,None
  with patch.object(gs,'setup_initial_game',return_value=s),patch.object(gl,'initialize_decks'),patch.object(ai,'mcts_action_with_policy',side_effect=choose),patch.object(gl,'roll_action_dice',side_effect=AssertionError('early roll')),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaisesRegex(RuntimeError,'invalid_fellowship_tail_selection'):ai._run_single_episode(None,None,{'max_turns':1,'mcts_sims':1})

if __name__=='__main__':unittest.main(verbosity=2)
