"""Actual first-corruption fixture and independent public-information checks."""
from pathlib import Path
import copy,os,pickle,random,sys,unittest
from collections import Counter
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'audit/2026-09-04/performance_pass/missing_card_failure_r1'
SOURCE=Path(os.environ.get('WOTR_DIAG_SOURCE',str(ROOT/'src'))).resolve()
os.environ.update(CUDA_VISIBLE_DEVICES='-1',WOTR_ACT_GEN='v52',WOTR_NATIVE='0')
sys.path[:0]=[str(SOURCE),str(ROOT/'src')]
import ai,numpy as np
from game_state import Side

class CommitmentConservation(unittest.TestCase):
    def fixture(self):
        with (HERE/'first_pool_r2/first_pool_before.pickle').open('rb') as stream:return pickle.load(stream)
    def cards(self,state,ov):
        info=state.hands[ov.opp_side]
        values=list(info.hand)+list(info.discarded_face_down)
        for deck in ('char','strat'):values.extend(getattr(state,ov.opp_tag+'_'+deck+'_deck'))
        prefix='SHADOW_' if ov.opp_tag=='sh' else 'FREE_PEOPLES_'
        for cards in state.discard_face_up.values():values.extend(c for c in cards if str(c).startswith(prefix))
        return Counter(values)
    def test_actual_first_bad_patch_conserves_every_physical_card(self):
        data=self.fixture();state,ov=data['state'],data['overlay']
        before=self.cards(state,ov)
        self.assertTrue(all(count==1 for count in before.values()),'Captured fixture precedes corruption')
        random.setstate(data['python_rng']);np.random.set_state(data['numpy_rng'])
        ov.patch(state)
        self.assertEqual(self.cards(state,ov),before)
        self.assertFalse(ov.has(ov.commit_cid),'Already committed card cannot return to sampled hand')
    def test_second_patch_preserves_zones_and_rng(self):
        data=self.fixture();state,ov=data['state'],data['overlay']
        random.setstate(data['python_rng']);ov.patch(state)
        zones=lambda:(copy.deepcopy(ov.hand),copy.deepcopy(ov.deck),copy.deepcopy(ov.fdown),copy.deepcopy(state.discard_face_up))
        expected=zones();rng=random.getstate();ov.patch(state)
        self.assertEqual(zones(),expected);self.assertEqual(random.getstate(),rng)
    def test_public_discard_identity_survives_consumed_round_card_fields(self):
        data=self.fixture();first=data['state'];second=ai.clone_state(first)
        deck,index=data['overlay'].commit_slot
        self.assertNotIn('atk_combat_card',first.pending_combat)
        second.discard_face_up[deck][index]='SHADOW_Strategy_Horde_from_the_East'
        with patch.object(ai,'_SEARCH_ROOT_SIDE',Side.FREE_PEOPLES):
            self.assertNotEqual(ai._chance_sig(first),ai._chance_sig(second))
    def test_unknown_hand_identity_does_not_split_public_chance_outcome(self):
        data=self.fixture();first=data['state'];second=ai.clone_state(first)
        hand=second.hands[Side.SHADOW].hand;index=next(i for i,c in enumerate(hand) if '_Strategy_' in c)
        hand[index],second.sh_strat_deck[0]=second.sh_strat_deck[0],hand[index]
        with patch.object(ai,'_SEARCH_ROOT_SIDE',Side.FREE_PEOPLES):
            self.assertEqual(ai._chance_sig(first),ai._chance_sig(second))
    def test_already_public_sample_refused_before_any_world_or_rng_mutation(self):
        data=self.fixture();state,ov=data['state'],data['overlay']
        cid=ov.commit_cid
        info=state.hands[ov.opp_side]
        for zone in (info.hand,info.discarded_face_down,state.sh_char_deck,state.sh_strat_deck):
            if cid in zone:zone.remove(cid);break
        else:self.fail('Captured sample must be in a hidden zone')
        state.discard_face_up[ov.commit_slot[0]].append(cid)
        before=pickle.dumps((state,ov));rng=random.getstate()
        with self.assertRaisesRegex(RuntimeError,'conflicts with public card history'):
            ov.patch(state)
        self.assertEqual(pickle.dumps((state,ov)),before)
        self.assertEqual(random.getstate(),rng)
    def test_actual_root_blinding_equal_observation_for_two_same_type_commitments(self):
        from test_audit0829_AI2 import _blinded_root
        first=_blinded_root().state;second=ai.clone_state(first)
        original=first.pending_combat['atk_combat_card']
        replacement=next(c for c in second.sh_char_deck if c!=original)
        second.sh_char_deck[second.sh_char_deck.index(replacement)]=original
        second.pending_combat['atk_combat_card']=replacement
        for cards in second.discard_face_up.values():
            if original in cards:cards[cards.index(original)]=replacement;break
        captures=[]
        class CapturedRoot(Exception):pass
        def capture(root,*args,**kwargs):
            self.assertTrue(hasattr(root.state,'_commit_slot'))
            captures.append((ai.encode_state(root.state,Side.FREE_PEOPLES).copy(),
                sorted(ai._atk_commit_mixture(root.state,root.state.pending_combat))))
            raise CapturedRoot()
        with patch.object(ai,'_TREE_REUSE',False),patch.object(ai,'_expand_node',side_effect=capture):
            for state in (first,second):
                with self.assertRaises(CapturedRoot):
                    ai.mcts_search(state,Side.FREE_PEOPLES,None,None,num_simulations=1,add_noise=False)
        np.testing.assert_array_equal(captures[0][0],captures[1][0])
        self.assertEqual(captures[0][1],captures[1][1])

if __name__=='__main__':unittest.main()
