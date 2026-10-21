"""One synchronous expansion shares its exact actor observation with policy memory."""
import contextlib,unittest
from unittest.mock import patch
import numpy as np
import ai
from test_learned_opponent_policy import character_state,FP,SH

class ExpansionObservationReuse(unittest.TestCase):
    def test_opponent_cache_and_inference_receive_same_real_actor_vector(self):
        actor=SH;root=FP;state=character_state(actor)
        actions=list(ai.get_ai_policy(state,actor).actions)
        self.assertTrue(actions)
        expected=ai.encode_state(state,actor);seen=[];memory=[];real_encode=ai.encode_state
        def encode(s,side):
            self.assertIs(s,state);self.assertEqual(side,actor)
            value=real_encode(s,side);seen.append(value);return value
        def infer(s,side,net,legal,*,features_vec=None):
            self.assertEqual(legal,actions);self.assertIs(features_vec,seen[0]);np.testing.assert_array_equal(features_vec,expected)
            return np.full(len(legal),1/len(legal),dtype=np.float32),.25
        def remember(side,vec,legal,priors):
            self.assertEqual(side,actor);self.assertIs(vec,seen[0]);self.assertEqual(legal,actions);memory.append(vec)
        node=ai.MCTSNode(state,actor)
        with contextlib.ExitStack() as stack:
            for name,value in [('_SIM_OVERLAY',None),('_PENDING_EVAL_BOX',[None]),('_OPPONENT_POLICY_MODE','learned_policy'),('encode_state',encode),('_get_priors_and_value',infer),('_remember_opponent_policy',remember),('_mcts_eval_state',lambda *a:.75)]:stack.enter_context(patch.object(ai,name,new=value))
            self.assertEqual(ai._expand_node(node,object(),object(),root,max_children=len(actions)),.75)
        self.assertEqual(len(seen),1);self.assertEqual(len(memory),1)
        self.assertEqual([c.action for c in node.children],actions)

    def test_root_actor_expansion_uses_vector_without_opponent_memory(self):
        actor=SH;state=character_state(actor);seen=[]
        expected=ai.encode_state(state,actor)
        def infer(s,side,net,legal,*,features_vec=None):
            np.testing.assert_array_equal(features_vec,expected);seen.append(features_vec)
            return np.full(len(legal),1/len(legal),dtype=np.float32),.25
        with patch.object(ai,'_SIM_OVERLAY',None),patch.object(ai,'_PENDING_EVAL_BOX',[None]),patch.object(ai,'_get_priors_and_value',new=infer),patch.object(ai,'_remember_opponent_policy',side_effect=AssertionError('root expansion must not populate opponent cache')):
            self.assertEqual(ai._expand_node(ai.MCTSNode(state,actor),object(),object(),actor),.25)
        self.assertEqual(len(seen),1)

if __name__=='__main__':unittest.main()
