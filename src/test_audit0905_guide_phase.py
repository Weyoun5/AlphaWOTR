"""Rulebook37: actual guide opportunity and current-turn search continuation."""
from pathlib import Path
import contextlib,io,json,os,random,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ['WOTR_REWARD_CONTRACT']='predecision_profile_v1'
import ai,board,game_logic as gl,game_state as gs,game_trace
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
HERE=ROOT/'audit/2026-09-04/decision_pass/guide_phase_review'
class GuidePhase(unittest.TestCase):
 def setUp(self):
  self.profiles=dict(ai.PLAYER_PROFILES);ai.PLAYER_PROFILES.update(fp='ring',sh='military')
  self.addCleanup(lambda:(ai.PLAYER_PROFILES.clear(),ai.PLAYER_PROFILES.update(self.profiles)))
  self.state=gs.setup_initial_game(board.load_board());gl.initialize_decks(self.state)
 def select(self,state,side,kind,**params):
  matches=[a for a in ai.get_ai_actions(state,side) if a.action_type==kind and all(a.params.get(k)==v for k,v in params.items())]
  self.assertEqual(len(matches),1);return matches[0]
 def test_initial_and_later_real_episode_phase_offer(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('guide_phase_probe',HERE/'probe.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  for search in (False,True):
   with self.subTest(search=search):
    result=module.probe(search)
    self.assertTrue(result['offered_guide']);self.assertEqual(result['guide_before_roll'],'Strider')
    kinds=[a['type'] for c in result['calls'] for a in c['menu']]
    self.assertLess(kinds.index('choose_guide'),kinds.index('eye_allocate'))
 def test_setup_stays_gandalf_and_keep_is_a_legal_decision(self):
  s=self.state;self.assertEqual(s.fellowship.guide.name,'Gandalf the Grey')
  self.assertTrue(gl.open_fellowship_guide_choice(s));self.assertTrue(s._eye_allocation_phase)
  keep=self.select(s,FP,'choose_guide',companion='Gandalf the Grey')
  ai._execute_action(s,keep);self.assertIsNone(s.pending_guide_change)
  self.assertEqual(s.fellowship.guide.name,'Gandalf the Grey')
  self.assertFalse(gl.open_fellowship_guide_choice(s));self.assertIsNone(s.pending_guide_change)
 def test_unique_best_and_gollum_have_no_learned_guide_choice(self):
  for names,wanted in ((['Strider'],'Strider'),([], 'Gollum')):
   with self.subTest(names=names):
    s=ai.clone_state(self.state);s.fellowship.companions=[c for c in s.fellowship.companions if c.name in names]
    self.assertFalse(gl.open_fellowship_guide_choice(s));self.assertIsNone(s.pending_guide_change)
    self.assertEqual(s.fellowship.guide.name,wanted);self.assertTrue(s._eye_allocation_phase)
 def test_already_mordor_still_offers_highest_level_guide(self):
  s=self.state;s.fellowship.mordor_entered=True;s.fellowship.mordor_step=2
  self.assertTrue(gl.open_fellowship_guide_choice(s));a=self.select(s,FP,'choose_guide',companion='Strider')
  ai._execute_action(s,a);self.assertEqual(s.fellowship.guide.name,'Strider');self.assertEqual(s.fellowship.mordor_step,2)
 def test_illegal_lower_level_guide_refused_without_clearing_pending(self):
  s=self.state;gl.open_fellowship_guide_choice(s);pending=dict(s.pending_guide_change)
  with self.assertRaisesRegex(ValueError,'pending legal candidates'):
   ai._execute_action(s,ai.Action(-1,'choose_guide',{'companion':'Gimli'}))
  self.assertEqual(s.pending_guide_change,pending);self.assertEqual(s.fellowship.guide.name,'Gandalf the Grey')
 def test_compulsory_guide_is_not_reclassified_as_optional_keep(self):
  s=self.state;gl.change_guide(s,allow_pending=True);pending=dict(s.pending_guide_change)
  with self.assertRaisesRegex(RuntimeError,'compulsory'):gl.open_fellowship_guide_choice(s)
  self.assertEqual(s.pending_guide_change,pending);self.assertFalse(getattr(s,'_eye_allocation_phase',False))
 def test_guide_opportunity_refused_after_action_rolls(self):
  s=self.state;gl.roll_action_dice(s,pool=FP)
  with self.assertRaisesRegex(RuntimeError,'precede'):gl.open_fellowship_guide_choice(s)
  self.assertIsNone(s.pending_guide_change)
 def test_real_search_decline_guide_eyes_roll_current_turn_once(self):
  s=self.state;s._declare_phase=True;hands={side:list(s.hands[side].hand) for side in (FP,SH)};turn=s.turn
  decline=self.select(s,FP,'pass');s,rng=ai._apply_action_detect_rng(s,decline)
  self.assertFalse(rng);self.assertFalse(s.dice_rolled);self.assertEqual(ai._pending_decision_side(s),FP)
  guide=self.select(s,FP,'choose_guide',companion='Strider');s,rng=ai._apply_action_detect_rng(s,guide)
  self.assertFalse(rng);self.assertEqual(ai._pending_decision_side(s),SH)
  node=ai.MCTSNode(state=s,side=FP,parent=None,action=guide)
  with patch.object(ai,'_TURN_HORIZON',1):ai._settle_new_node(node,FP,None,None,FP)
  self.assertTrue(node.children);self.assertTrue(all(c.action.action_type=='eye_allocate' for c in node.children))
  eye=self.select(s,SH,'eye_allocate',count=0);s,rng=ai._apply_action_detect_rng(s,eye)
  self.assertTrue(rng);self.assertTrue(s.dice_rolled);self.assertEqual(s.turn,turn)
  self.assertFalse(s._eye_allocation_phase);self.assertIsNone(s.pending_guide_change)
  self.assertEqual({side:s.hands[side].hand for side in (FP,SH)},hands)
  self.assertEqual(s.fellowship.guide.name,'Strider')
 def test_live_eye_executor_keeps_roll_ownership_external(self):
  s=self.state;gl.open_fellowship_guide_choice(s);ai._execute_action(s,self.select(s,FP,'choose_guide',companion='Strider'))
  with patch.object(gl,'roll_action_dice',side_effect=AssertionError('live eye must not roll')):
   ai._execute_action(s,self.select(s,SH,'eye_allocate',count=0))
  self.assertFalse(s.dice_rolled);self.assertFalse(s._eye_allocation_phase)
 def test_phase_and_composition_context_are_distinct_in_existing_lanes(self):
  composition=ai.clone_state(self.state);gl.change_guide(composition,allow_pending=True)
  phase=ai.clone_state(self.state);gl.open_fellowship_guide_choice(phase)
  a,b=ai.encode_state(composition,FP),ai.encode_state(phase,FP)
  self.assertEqual(len(a),3746);self.assertEqual(ai.OBSERVATION_CONTRACT,'v44_fellowship_tail_v1')
  self.assertEqual(np.count_nonzero(a!=b),1)
  self.assertNotEqual(ai._state_sig(composition),ai._state_sig(phase));self.assertNotEqual(ai._chance_sig(composition),ai._chance_sig(phase))
  clone=ai.clone_state(phase);np.testing.assert_array_equal(ai.encode_state(clone,FP),b)
  self.assertEqual(ai._state_sig(clone),ai._state_sig(phase))
  actions=ai.get_ai_actions(phase,FP);self.assertEqual(len(actions),2)
  encs=[ai.encode_action(x,phase,FP) for x in actions];self.assertFalse(np.array_equal(encs[0],encs[1]))
 def test_search_current_rolls_are_sampled_chance_outcomes(self):
  s=self.state;gl.open_fellowship_guide_choice(s);ai._execute_action(s,self.select(s,FP,'choose_guide',companion='Strider'))
  a=self.select(s,SH,'eye_allocate',count=0)
  signatures=[]
  for seed in (9001,9002,9003):
   random.seed(seed);after,rng=ai._apply_action_detect_rng(s,a)
   self.assertTrue(rng);self.assertEqual(after.turn,s.turn);signatures.append(ai._chance_sig(after))
  self.assertGreater(len(set(signatures)),1);self.assertFalse(s.dice_rolled)
 def test_real_mcts_guide_target_survives_trace_collection_and_storage(self):
  import delta_trainer as dt,run_td_v33loop as loop,torch,copy
  torch.set_num_threads(1)
  s=self.state;gl.open_fellowship_guide_choice(s)
  with tempfile.TemporaryDirectory() as tmp:
   trace=game_trace.TraceRecorder(str(Path(tmp)/'guide_partial.trace.gz'));trace.record_initial(s)
   context=ai._decision_context(s,FP,trace)
   random.seed(95301);np.random.seed(95301)
   with patch.object(ai,'_TURN_HORIZON',1):
    action,info=ai.mcts_action_with_policy(s,FP,None,None,num_simulations=32,
                                         temperature=1.,add_noise=False,)
   self.assertEqual(action.action_type,'choose_guide');self.assertEqual(len(info['pi_sel']),2)
   stats,regret=ai._search_stats_payload(info)
   ai._traced_execute(s,action,trace)
   trace.record_action(s,FP,action,search_stats=stats,regret=regret,decision_context=context)
   trace.save(status='partial_guide_policy_probe',reward_contract='predecision_profile_v1',
              profiles={'fp':'ring','sh':'military'})
   loaded=game_trace.load_trace(trace.path)
   rows,counts=dt.collect_delta_samples(loaded,'fp')
   self.assertEqual(len(rows),1);self.assertEqual(counts['stale_arms'],0);self.assertEqual(counts['error'],0)
   self.assertTrue(loaded['steps'][-1]['decision_context']['state']._eye_allocation_phase)
   np.testing.assert_array_equal(rows[0]['pi_sel'],stats['pi_sel'])
   packed=loop._buffer_pack({'fp':{'delta':copy.deepcopy(rows)}})
   restored=loop._buffer_unpack_delta(packed['fp']['delta'])
   np.testing.assert_array_equal(restored[0]['ae'],rows[0]['ae'])
   np.testing.assert_array_equal(restored[0]['pi_sel'],rows[0]['pi_sel'])
 def test_selected_strider_unlocks_hide_on_an_army_result(self):
  for name,expected in [('Strider',True),('Gandalf the Grey',False)]:
   with self.subTest(name=name):
    s=ai.clone_state(self.state);gl.open_fellowship_guide_choice(s)
    ai._execute_action(s,self.select(s,FP,'choose_guide',companion=name))
    s._eye_allocation_phase=False;s.fellowship.hidden=False
    for side in (FP,SH):
     for die in s.dice[side]:die.used=True
    s.dice[FP][0].used=False;gl.adopt_rolled_faces(s,[(FP,0,gs.ActionDieFace.ARMY)])
    offered=any(a.action_type=='fellowship_hide' for a in ai.get_ai_actions(s,FP))
    self.assertEqual(offered,expected)
 def test_guide_phase_selection_error_stops_before_eyes_or_rolls(self):
  class EndProbe(BaseException):pass
  calls=[]
  def pick(st,side,*args,**kwargs):
   actions=ai.get_ai_actions(st,side)
   if st.pending_guide_change is not None:
    calls.append('guide');return None,None
   if getattr(st,'_declare_phase',False):return next(a for a in actions if a.action_type=='pass'),None
   raise AssertionError('phase advanced after invalid guide selection')
  with tempfile.TemporaryDirectory() as tmp, patch.object(ai,'mcts_action_with_policy',side_effect=pick), \
       patch.object(gl,'roll_action_dice',side_effect=AssertionError('roll before guide complete')):
   with self.assertRaisesRegex(RuntimeError,'invalid_guide_selection'):
    ai._run_single_episode(None,None,{'max_turns':1,'trace_path':str(Path(tmp)/'error.trace.gz')})
   saved=game_trace.load_trace(str(Path(tmp)/'error.trace.gz'))
   self.assertIn('invalid_guide_selection',[e['code'] for e in saved['meta']['source_errors']])
  self.assertEqual(calls,['guide'])
 def test_shadow_cannot_enumerate_or_apply_a_guide_choice(self):
  s=self.state;gl.open_fellowship_guide_choice(s);pending=dict(s.pending_guide_change)
  self.assertEqual(ai.get_legal_actions(s,SH),[]);self.assertEqual(ai.get_ai_actions(s,SH),[])
  self.assertTrue(ai.get_ai_actions(s,FP))
  with self.assertRaisesRegex(ValueError,'Only Free Peoples'):
   ai.apply_action(s,ai.Action(-1,'choose_guide',{'companion':'Strider','side':'sh'}))
  self.assertEqual(s.pending_guide_change,pending);self.assertEqual(s.fellowship.guide.name,'Gandalf the Grey')
if __name__=='__main__':unittest.main(verbosity=2)
