"""Literal legal noncard recovery and stochastic opponent policy regressions.

These fixtures use the public engine menu and real executor. Network outputs
are explicit test inputs; expected physical movement is independently stated.
"""
import copy
import random
import unittest
from unittest.mock import patch
import numpy as np
import ai
import board
import game_state as gs
import game_logic as gl
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
def particle(s,root=FP):
    ov=ai._SimOverlay(s,root);prefix='sh_' if root==FP else 'fp_'
    for typ in ('char','strat'):
        ov.hand[typ]=[c for c in s.hands[ov.opp_side].hand if ai._ov_deck_of(c)==typ]
        ov.deck[typ]=list(reversed(getattr(s,prefix+typ+'_deck')));ov.fdown[typ]=[c for c in s.hands[ov.opp_side].discarded_face_down if ai._ov_deck_of(c)==typ]
    return ov

def character_state(side):
    s=gs.setup_initial_game(board.load_board());gl.initialize_decks(s)
    if side==SH:
        s.armies={'barad dur':next(a for k,a in s.armies.items() if board.sanitize(k)=='barad dur')}
        s.political.boxes[gs.Nation.SAURON]=3;s.political.active[gs.Nation.SAURON]=True
        cards=['SHADOW_Character_Dreadful_Spells','SHADOW_Character_Morgul_Wound']
    else:
        cards=['FREE_PEOPLES_Character_The_Ents_Awake_Treebeard','FREE_PEOPLES_Character_The_Ents_Awake_Huorns']
        # Resolve spelling from the actual catalog, retaining literal card identities.
        cards=[next(c for c in s.fp_char_deck if word in c) for word in ('Treebeard','Huorns')]
    prefix='sh_' if side==SH else 'fp_'
    s.hands[side].hand=cards
    for cid in cards:getattr(s,prefix+'char_deck').remove(cid)
    d=gs.ActionDie([gs.ActionDieFace.CHARACTER],side);d.current=gs.ActionDieFace.CHARACTER
    s.dice={SH:[],FP:[]};s.dice[side]=[d];s.dice_rolled=True
    s.shadow_elven_rings=0;s.free_peoples_elven_rings=0
    return s

class LearnedOpponentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.rng=random.getstate()
        self.globals=patch.multiple(ai,_OPPONENT_POLICY_MODE='learned_policy',_OPPONENT_POLICY_EPSILON=.02,_OPPONENT_POLICY_CACHE={},_SEARCH_NETS=(object(),object()),_SEARCH_ROOT_SIDE=FP,_SIM_OVERLAY=None)
        self.globals.start()
        random.seed(10)
    def tearDown(self):
        self.globals.stop()
        random.setstate(self.rng)
    def select(self,s,root=FP):
        ai._SIM_OVERLAY=particle(s,root);ai._SEARCH_ROOT_SIDE=root
        node=ai.MCTSNode(s,SH if root==FP else FP)
        return node

    def test_character_unplayable_cards_recover_literal_leader_move_from_card_only_node(self):
        s=character_state(SH);node=self.select(s)
        node.children=[ai.MCTSNode(None,SH,parent=node,action=ai.Action(0,'play_event',{'card_id':'SHADOW_Character_Wormtongue','side':'sh'}),prior=1.)]
        menu=ai.get_ai_actions(s,SH);self.assertFalse(any(a.action_type=='play_event' for a in menu))
        wanted=ai.Action(0,'army_move',{'from':'barad dur','to':'gorgoroth','n_regular':0,'n_elite':1,'take_leaders':True,'side':'sh'})
        self.assertIn(wanted,menu)
        before=copy.deepcopy(s)
        with patch.object(ai,'_get_priors_and_value',side_effect=lambda st,side,net,actions,**kw:([float(a==wanted) for a in actions],0.)):
            random.seed(1);chosen=ai._select_child(node,9999.)
        self.assertEqual(chosen.action,wanted);self.assertEqual(node.side,SH)
        after=ai.apply_action(s,chosen.action)
        self.assertTrue(after.dice[SH][0].used)
        target=next(a for k,a in after.armies.items() if board.sanitize(k)=='gorgoroth')
        source=next(a for k,a in after.armies.items() if board.sanitize(k)=='barad dur')
        self.assertEqual(sum(u.kind==gs.UnitKind.ELITE for u in target.units),1)
        self.assertEqual(sum(u.kind==gs.UnitKind.ELITE for u in source.units),sum(u.kind==gs.UnitKind.ELITE for u in before.armies['barad dur'].units)-1)
        self.assertEqual(len(target.leaders),len(before.armies['barad dur'].leaders));self.assertGreater(len(target.leaders),0)
        from ai_policies import action_policy_key
        current_menu=ai.get_ai_policy(s,SH).actions
        self.assertGreater(len(current_menu),40)
        self.assertTrue({action_policy_key(a) for a in current_menu} <= node._opponent_edge_catalog.index.keys())
        self.assertEqual(len(node.children),2)

    def test_fp_unplayable_ents_still_allow_fellowship(self):
        s=character_state(FP);node=self.select(s,SH)
        menu=ai.get_ai_actions(s,FP);self.assertFalse(any(a.action_type=='play_event' for a in menu))
        moves=[a for a in menu if a.action_type=='fellowship_move'];self.assertEqual(len(moves),1)
        node.children=[ai.MCTSNode(None,FP,parent=node,action=ai.Action(0,'play_event',{'card_id':'not_held','side':'fp'}),prior=1.)]
        with patch.object(ai,'_get_priors_and_value',side_effect=lambda st,side,net,actions,**kw:([float(a.action_type=='fellowship_move') for a in actions],0.)):
            random.seed(1);chosen=ai._select_child(node,1.)
        self.assertEqual(chosen.action.action_type,'fellowship_move');self.assertEqual(node.side,FP)
        after=ai.apply_action(s,chosen.action);self.assertTrue(after.dice[FP][0].used)
        self.assertEqual(after.fellowship.progress,s.fellowship.progress+1)

    def test_whole_menu_probability_floor_and_q_independence(self):
        state=character_state(SH);node=self.select(state);ai._SIM_OVERLAY.patch(state)
        menu=ai.get_ai_actions(state,SH)
        wanted=next(i for i,a in enumerate(menu) if a.action_type=='army_move')
        raw=np.zeros(len(menu));raw[wanted]=1.
        expected=.98*raw+.02/len(menu)
        with patch.object(ai,'_get_priors_and_value',return_value=(raw,0.)):
            expected_draws=random.Random(173).choices(range(len(menu)),weights=expected,k=100)
            with patch.object(ai.random,'choices',side_effect=random.Random(173).choices):
                actual=[menu.index(ai._select_child(node,100.).action) for _ in range(100)]
            self.assertEqual(actual,expected_draws)
            for i,c in enumerate(node.children):
                c.visit_count=1000;c.value_sum=(-1.)**i*1e6;c.proven='W' if i%2 else 'L'
            with patch.object(ai.random,'choices',side_effect=random.Random(173).choices):
                changed=[menu.index(ai._select_child(node,1e9).action) for _ in range(100)]
            self.assertEqual(changed,expected_draws)
        probabilities=next(iter(ai._OPPONENT_POLICY_CACHE.values()))
        np.testing.assert_allclose(probabilities,expected)
        self.assertTrue(all(p>0 for p in probabilities))

    def test_only_proven_full_menu_empty_hands_off(self):
        state=character_state(SH);state.armies={}
        # No pieces, no playable cards, and no opposing unused dice.
        node=self.select(state);self.assertEqual(ai.get_ai_actions(state,SH),[])
        self.assertIsNone(ai._select_child(node,1.));self.assertEqual(node.side,SH)
        state._ring_free_side=SH
        with self.assertRaisesRegex(RuntimeError,'required decision'):
            ai._select_child(node,1.)

    def test_policy_mode_does_not_propagate_minimax_proof(self):
        node=ai.MCTSNode(None,SH)
        child=ai.MCTSNode(None,FP,parent=node,action=ai.Action(-1,'combat_card_none',{}),prior=1.)
        child.proven='L';child.proven_value=-1.;node.children=[child]
        ai._propagate_proofs([node],FP);self.assertIsNone(node.proven)
        with patch.object(ai,'_OPPONENT_POLICY_MODE','legacy_puct'):
            ai._propagate_proofs([node],FP)
        self.assertEqual(node.proven,'L')

if __name__=='__main__':unittest.main()
