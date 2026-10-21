"""Army batching agrees with direct per-action encoding on live engine states."""
import copy,pickle,random,unittest
import numpy as np
import ai,board,game_state as gs
from test_learned_opponent_policy import character_state,SH,FP

class SearchArmyBatchTests(unittest.TestCase):
    def verify(self,state,side,actions):
        before=pickle.dumps(state,protocol=5);rng=pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)
        expected=np.asarray([ai.encode_action(a,state,side) for a in actions],np.float32)
        regions=np.asarray([ai._get_action_region_index(a) for a in actions],np.int64)
        ai._ENC_CACHE.clear();ai._RIDX_CACHE.clear();ai._LATTICE_ENC_CACHE.clear()
        for action in actions:
            action.__dict__.pop('_encs',None);action.__dict__.pop('_ri',None)
        actual=ai._encode_actions_batch(actions,state,side,{})
        np.testing.assert_array_equal(actual[0],expected);np.testing.assert_array_equal(actual[1],regions)
        self.assertEqual(before,pickle.dumps(state,protocol=5))
        self.assertEqual(rng,pickle.dumps((random.getstate(),np.random.get_state()),protocol=5))
        return actual[0]

    def variants(self):
        return [ai.Action(0,'army_move',{'from':'barad dur','to':'gorgoroth','n_regular':nr,'n_elite':ne,'take_leaders':take,'side':'sh'})
                for nr in range(8) for ne in range(3) for take in (False,True)]

    def test_complete_actual_fp_and_sh_menus_remain_exact(self):
        for side in (FP,SH):
            state=character_state(side);actions=ai.get_ai_policy(state,side).actions
            self.assertGreater(len(actions),40);self.verify(state,side,actions)

    def test_mixed_nations_literal_counts_left_and_vacated(self):
        state=character_state(SH);army=state.armies['barad dur']
        army.units=[gs.Unit(gs.Nation.SAURON,gs.UnitKind.REGULAR) for _ in range(4)]+[gs.Unit(gs.Nation.ISENGARD,gs.UnitKind.REGULAR) for _ in range(3)]+[gs.Unit(gs.Nation.SOUTHRON_EASTERLINGS,gs.UnitKind.ELITE) for _ in range(2)]
        actions=self.variants();enc=self.verify(state,SH,actions)
        for nr,ne,left in [(3,2,4),(7,2,0)]:
            i=next(i for i,a in enumerate(actions) if a.params['n_regular']==nr and a.params['n_elite']==ne)
            self.assertEqual(enc[i,23],np.float32((nr+ne)/10+ne/100))
            self.assertEqual(enc[i,ai._ACT_COMP_OFF],np.float32(nr/10))
            self.assertEqual(enc[i,ai._ACT_COMP_OFF+1],np.float32(ne/10))
            self.assertEqual(enc[i,ai._ACT_LEAVE_OFF],np.float32(left/10))
            self.assertEqual(enc[i,ai._ACT_EMPTIES_OFF],float(left==0))

    def test_parameter_and_live_source_mutations_do_not_reuse_stale_template(self):
        state=character_state(SH);actions=self.variants();self.verify(state,SH,actions)
        actions[0].params['n_regular']=7;actions[1].params['nation']='elves';actions[2].params['n_regular']='3'
        self.verify(state,SH,actions)
        state.armies['barad dur'].units.pop();self.verify(state,SH,actions)
        state.armies={};self.verify(state,SH,actions)

    def test_companion_lattice_and_small_menus_keep_generic_semantics(self):
        state=character_state(FP);actions=ai.get_ai_policy(state,FP).actions
        self.assertTrue(any('_lattice' in vars(a) for a in actions));self.verify(state,FP,actions)
        self.verify(state,FP,actions[:2])

if __name__=='__main__':unittest.main()
