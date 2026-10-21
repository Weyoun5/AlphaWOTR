"""Literal rulebook p21: Eye conversion is legal, but creates no extra die.

The last die enters the Hunt Box immediately. With no remaining dice, the
other player acts or the turn ends. No actual pending decision is waived.
"""
from pathlib import Path
import importlib.util,os,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(WOTR_ACT_GEN='v52',WOTR_NATIVE='0',WOTR_COMBAT_TABLE='',WOTR_OPPONENT_POLICY='learned_policy',CUDA_VISIBLE_DEVICES='-1')
sys.path.insert(0,str(ROOT/'src'))
source=Path(os.environ.get('WOTR_TEST_AI_SOURCE',str(ROOT/'src/ai.py'))).resolve()
spec=importlib.util.spec_from_file_location('ai',source);ai=importlib.util.module_from_spec(spec);sys.modules['ai']=ai;spec.loader.exec_module(ai)
import board,game_state as gs,game_logic


def die(side,face):
    result=gs.ActionDie([face],side);result.current=face;return result


def position(extra_shadow=False,fp=False):
    state=gs.setup_initial_game(board.load_board());game_logic.initialize_decks(state)
    state.dice={gs.Side.SHADOW:[die(gs.Side.SHADOW,gs.ActionDieFace.EVENT)],gs.Side.FREE_PEOPLES:[]}
    if extra_shadow:state.dice[gs.Side.SHADOW].append(die(gs.Side.SHADOW,gs.ActionDieFace.EVENT))
    if fp:state.dice[gs.Side.FREE_PEOPLES].append(die(gs.Side.FREE_PEOPLES,gs.ActionDieFace.EVENT))
    state.sh_elven_rings=1;state.sh_ring_used_this_turn=False;state.dice_rolled=True
    state._declare_phase=False;state._eye_allocation_phase=False;state.hunt_box=[]
    action=next(a for a in ai.get_legal_actions(state,gs.Side.SHADOW)
        if a.action_type=='elven_ring' and a.params['target_die_idx']==0 and a.params['new_face']==gs.ActionDieFace.EYE.value)
    return state,action


class RingExhaustion(unittest.TestCase):
    def assert_ring_payment(self,state):
        self.assertEqual(state.sh_elven_rings,0)
        self.assertTrue(state.sh_ring_used_this_turn)
        self.assertTrue(state.dice[gs.Side.SHADOW][0].used)
        self.assertEqual(state.dice[gs.Side.SHADOW][0].current,gs.ActionDieFace.EYE)
        self.assertEqual(len(state.hunt_box),1)
        self.assertIs(state.hunt_box[0],state.dice[gs.Side.SHADOW][0])

    def test_apply_last_die_eye_finishes_turn_without_fake_pending(self):
        state,action=position();next_state=ai.apply_action(state,action)
        self.assert_ring_payment(next_state)
        self.assertEqual(state.sh_elven_rings,1,'apply_action must clone input')
        self.assertIsNone(getattr(next_state,'_ring_free_side',None))
        self.assertEqual(ai.get_ai_actions(next_state,gs.Side.SHADOW),[])
        self.assertEqual(ai.get_ai_actions(next_state,gs.Side.FREE_PEOPLES),[])
        self.assertIsNone(ai._next_acting_side(next_state,gs.Side.SHADOW))

    def test_direct_execute_last_die_leaves_other_player_action(self):
        state,action=position(fp=True)
        ai._execute_action(state,action)
        self.assert_ring_payment(state)
        self.assertIsNone(getattr(state,'_ring_free_side',None))
        self.assertEqual(ai._next_acting_side(state,gs.Side.SHADOW),gs.Side.FREE_PEOPLES)
        self.assertTrue(ai.get_ai_actions(state,gs.Side.FREE_PEOPLES))
        self.assertEqual(ai.get_ai_actions(state,gs.Side.SHADOW),[])

    def test_remaining_die_keeps_same_owner_then_spends_exactly_once(self):
        state,action=position(extra_shadow=True,fp=True);state=ai.apply_action(state,action)
        self.assertEqual(state._ring_free_side,gs.Side.SHADOW)
        self.assertEqual(ai._next_acting_side(state,gs.Side.SHADOW),gs.Side.SHADOW)
        funded=next(a for a in ai.get_ai_actions(state,gs.Side.SHADOW) if a.action_type=='draw_card')
        state=ai.apply_action(state,funded)
        self.assertTrue(all(d.used for d in state.dice[gs.Side.SHADOW]))
        self.assertIsNone(state._ring_free_side)
        self.assertEqual(ai._next_acting_side(state,gs.Side.SHADOW),gs.Side.FREE_PEOPLES)

    def test_canonical_router_does_not_revive_stale_exhausted_ring_marker(self):
        state,action=position();state=ai.apply_action(state,action)
        state._ring_free_side=gs.Side.SHADOW
        self.assertIsNone(ai._next_acting_side(state,gs.Side.SHADOW))

    def test_real_required_menu_error_is_preserved(self):
        state,action=position();state=ai.apply_action(state,action)
        state.pending_combat={'phase':'defender_combat_card','deciding_side':gs.Side.FREE_PEOPLES}
        node=ai.MCTSNode(state,gs.Side.SHADOW)
        class Overlay:
            def patch(self,state):pass
        with self.assertRaisesRegex(RuntimeError,'Empty sampled menu during required decision'):
            ai._learned_opponent_child(node,Overlay())

    def test_actual_search_settlement_creates_turn_transition(self):
        from unittest.mock import patch
        state,action=position();state=ai.apply_action(state,action)
        node=ai.MCTSNode(state,gs.Side.SHADOW,action=action)
        with patch.object(ai,'_TURN_HORIZON',1),patch.object(ai,'_mcts_eval_state',return_value=0.0):
            ai._settle_new_node(node,gs.Side.SHADOW,None,None,gs.Side.FREE_PEOPLES)
        self.assertEqual([c.action.action_type for c in node.children],['end_turn'])

    def test_learned_search_can_select_only_automatic_turn_end_after_last_eye(self):
        state,action=position();state=ai.apply_action(state,action)
        node=ai.MCTSNode(state,gs.Side.SHADOW)
        end=ai.MCTSNode(None,gs.Side.SHADOW,parent=node,
                       action=ai.Action(-1,'end_turn',{}),prior=1.0)
        node.children=[end]
        class Overlay:
            def patch(self,state):pass
        self.assertIs(ai._learned_opponent_child(node,Overlay()),end)


if __name__=='__main__':unittest.main(verbosity=2)
