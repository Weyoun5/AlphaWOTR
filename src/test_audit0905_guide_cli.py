"""Public CLI submissions preserve Fellowship timing and fail closed."""
from pathlib import Path
import contextlib,io,os,sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(WOTR_NATIVE='0',WOTR_ACT_GEN='v41',WOTR_REWARD_CONTRACT='predecision_profile_v1',CUDA_VISIBLE_DEVICES='-1')
import ai,board,engine_play as ep,game_logic as gl,game_state as gs
from test_audit0905_fellowship_tail import probe
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
class GuideCLI(unittest.TestCase):
 def session(self,state=None,**kw):
  if state is None:return ep.EngineSession(seed=95591,human_side=FP,**kw)
  with patch.object(gs,'setup_initial_game',return_value=state),patch.object(gl,'initialize_decks'):
   return ep.EngineSession(seed=95591,human_side=FP,**kw)
 def pick(self,session,kind,**params):
  dp=session.current();matches=[i for i,a in enumerate(dp.actions) if a.action_type==kind and all(a.params.get(k)==v for k,v in params.items())]
  self.assertEqual(len(matches),1,[(a.action_type,a.params) for a in dp.actions]);session.submit(matches[0],dp.side)
 def test_initial_and_later_turn_offer_guide_before_eye(self):
  for later in (False,True):
   with self.subTest(later=later):
    state=None
    if later:
     state=gs.setup_initial_game(board.load_board());gl.initialize_decks(state)
     for dice in state.dice.values():
      for die in dice:die.used=True
     ai._safe_next_turn(state)
    s=self.session(state);self.assertEqual(s.current().kind,'declare')
    self.pick(s,'pass');self.assertEqual(s.current().kind,'guide')
    self.assertEqual(s.current().side,FP);self.assertFalse(s.state.dice_rolled)
    with self.assertRaises(ep.IllegalPlay):s.submit(0,SH)
    self.pick(s,'choose_guide',companion='Strider');self.assertEqual(s.current().kind,'eye')
    self.assertEqual(s.state.fellowship.guide.name,'Strider');self.assertFalse(s.state.dice_rolled)
    self.pick(s,'eye_allocate',count=0);self.assertTrue(s.state.dice_rolled)
    s.abandon()
 def test_balrog_damage_compulsory_guide_then_optional_guide(self):
  state,_=probe.position();state.fellowship.companions=[c for c in state.fellowship.companions if c.name in ('Gimli','Legolas','Boromir')]
  state.fellowship.guide=next(c for c in state.fellowship.companions if c.name=='Gimli')
  tile=next(t for t in state.hunt_pool if t.damage==2 and not t.reveal and not t.is_eye)
  state.hunt_pool.remove(tile);state.hunt_pool.append(tile)
  s=self.session(state);self.pick(s,'fellowship_declare',region='moria')
  self.assertEqual((s.current().kind,s.current().side),('fellowship_tail',SH))
  self.pick(s,'card_option',choices={'balrog_declare_draw':True})
  self.assertEqual((s.current().kind,s.current().side),('fellowship_tail',FP));self.pick(s,'hunt_sacrifice_guide')
  self.assertEqual(s.current().kind,'fellowship_tail');self.assertFalse(getattr(state,'_eye_allocation_phase',False))
  self.pick(s,'choose_guide',companion='Legolas');self.assertEqual(s.current().kind,'guide')
  self.pick(s,'choose_guide',companion='Legolas');self.assertEqual(s.current().kind,'eye')
  self.assertFalse(state.dice_rolled);self.assertEqual(state.fellowship.corruption,0);s.abandon()
 def test_search_invalid_selection_does_not_choose_first_or_decline(self):
  s=self.session();s.attach_nets(None,None)
  with patch.object(ai,'mcts_action_with_policy',return_value=(None,None)):
   with self.assertRaisesRegex(ep.IllegalPlay,'legal current decision'):s.ai_pick()
  self.assertEqual(s.current().kind,'declare');self.assertTrue(s.state._declare_phase);s.abandon()
 def test_stale_guide_menu_is_rejected_before_execution(self):
  s=self.session();self.pick(s,'pass');dp=s.current()
  idx=next(i for i,a in enumerate(dp.actions) if a.params['companion']=='Strider')
  # The model/controller retained a menu across a real externally committed
  # legal Guide choice. Reusing that old menu must not execute it twice.
  ai._execute_action(s.state,dp.actions[idx]);self.assertIsNone(s.state.pending_guide_change)
  with self.assertRaisesRegex(ep.IllegalPlay,'stale'):s.submit(idx,FP)
  self.assertEqual(s.state.fellowship.guide.name,'Strider');self.assertFalse(s.state.dice_rolled);s.abandon()
 def test_executor_error_is_not_replaced_by_pass(self):
  s=self.session();self.pick(s,'pass');before=s.state.fellowship.guide.name
  with patch.object(ai,'_traced_execute',side_effect=RuntimeError('literal phase failure')):
   with self.assertRaisesRegex(RuntimeError,'literal phase failure'):self.pick(s,'choose_guide',companion='Strider')
  self.assertEqual(s.state.fellowship.guide.name,before);self.assertIsNotNone(s.state.pending_guide_change)
  self.assertFalse(s.state.dice_rolled);self.assertFalse(s.action_log);s.abandon()
 def test_cap_cannot_certify_military_victory_with_unused_dice(self):
  state=gs.setup_initial_game(board.load_board());gl.initialize_decks(state)
  state.enemy_controlled[FP].update({'orthanc','morannon'})
  for region in ('orthanc','morannon'):
   army=gs.Army(state.board.region(region));army.units=[gs.Unit(gs.Nation.THE_NORTH,gs.UnitKind.REGULAR)];army.leaders=[]
   state.armies[region]=army
  def rolls(st,*,pool):
   gl.adopt_rolled_faces(st,[(pool,i,gs.ActionDieFace.ARMY) for i,d in enumerate(st.dice[pool]) if not d.used])
  with patch.object(gl,'roll_action_dice',side_effect=rolls):
   s=self.session(state,max_turn_decisions=1);self.pick(s,'pass');self.pick(s,'choose_guide',companion='Strider');self.pick(s,'eye_allocate',count=0)
   self.assertEqual(gl.check_all_victory_conditions(state)['victory'],'free_peoples')
   with self.assertRaisesRegex(RuntimeError,'unfinished_turn.*unused_action_dice'):
    self.pick(s,'pass')
  self.assertIsNone(s.result_info);self.assertEqual(state.turn,1);s.abandon()

if __name__=='__main__':unittest.main(verbosity=2)
