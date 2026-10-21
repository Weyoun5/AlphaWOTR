"""Real legal menus and immutable execution-edge regressions, CPU only."""
import copy,pickle,random,types,unittest
from unittest.mock import patch
import numpy as np
import ai
import search_edge_catalog as catalog
from ai_policies import action_policy_key
from test_learned_opponent_policy import character_state,particle,SH,FP

class SearchEdgeCatalogTests(unittest.TestCase):
    def setUp(self):
        self.rng=random.getstate()
        self.scope=patch.multiple(ai,_OPPONENT_POLICY_MODE='learned_policy',_OPPONENT_POLICY_EPSILON=.02,
            _OPPONENT_POLICY_CACHE={},_SEARCH_NETS=(object(),object()),_COMBAT_FORECAST_OBSERVER=None,_SIM_OVERLAY=None)
        self.scope.start();self.addCleanup(self.scope.stop);self.addCleanup(random.setstate,self.rng)

    def test_real_changed_hidden_hands_match_full_allocation_choices_priors_rng(self):
        def run(source,side,reference):
            state=copy.deepcopy(source);node=ai.MCTSNode(state,side)
            ov=particle(state,FP if side==SH else SH);rows=[];random.seed(43019)
            ai._OPPONENT_POLICY_CACHE.clear()
            def score(s,side,net,actions,**kw):return np.arange(1,len(actions)+1,dtype=np.float32),0.
            chooser=catalog.full_choice if reference else catalog.choose
            with patch.object(catalog,'choose',chooser),patch.object(ai,'_get_priors_and_value',side_effect=score):
                for i in range(24):
                    if i%2 and ov.hand['char'] and ov.deck['char']:
                        ov.hand['char'][0],ov.deck['char'][0]=ov.deck['char'][0],ov.hand['char'][0]
                    child=ai._learned_opponent_child(node,ov)
                    legal={action_policy_key(a) for a in ai.get_ai_policy(state,side).actions}
                    self.assertIn(action_policy_key(child.action),legal)
                    if not reference:self.assertTrue(legal<=node._opponent_edge_catalog.index.keys())
                    rows.append((action_policy_key(child.action),child.prior,child.visit_count,child.value_sum))
                    child.visit_count+=1;child.value_sum+=.25
            return rows,random.getstate(),node
        for side in (SH,FP):
            with self.subTest(side=side):
                source=character_state(side);full=run(source,side,True);lazy=run(source,side,False)
                self.assertEqual(full[:2],lazy[:2]);self.assertLess(len(lazy[2].children),len(full[2].children))

    def test_snapshot_survives_nested_public_and_execution_parameter_mutations(self):
        node=ai.MCTSNode(types.SimpleNamespace(),SH);ov=types.SimpleNamespace(patch=lambda state:None)
        action=ai.Action(-1,'companion_move',{'companions':['Legolas'],'routes':{'Legolas':['a','b']}})
        action.provenance={'observed':[5]};original=copy.deepcopy(action)
        policy=types.SimpleNamespace(actions=[action],forced_reason='single')
        with patch.object(ai,'get_ai_policy',return_value=policy):
            child=ai._learned_opponent_child(node,ov)
            action.params['companions'].append('Gimli');action.provenance['observed'].append(6)
            self.assertEqual(child.action.__dict__,original.__dict__)
            child.action.params['routes']['Legolas'].append('bad execution mutation')
            policy.actions=[original];again=ai._learned_opponent_child(node,ov)
        self.assertIs(child,again);self.assertEqual(again.action.__dict__,original.__dict__)
        self.assertIs(type(again.action),ai.Action)
        self.assertEqual(pickle.loads(pickle.dumps(again.action)).__dict__,original.__dict__)

    def test_unsupported_payload_fallback_has_one_callback_and_no_forced_rng_draw(self):
        node=ai.MCTSNode(types.SimpleNamespace(),SH);calls={'overlay':0,'observer':0}
        ov=types.SimpleNamespace(patch=lambda s:calls.__setitem__('overlay',calls['overlay']+1))
        observer=types.SimpleNamespace(card_policy=lambda *a:calls.__setitem__('observer',calls['observer']+1))
        marker=object();actions=[ai.Action(-1,'a',{'x':1}),ai.Action(-1,'b',{'custom':marker})]
        before=random.getstate()
        with patch.object(ai,'_COMBAT_FORECAST_OBSERVER',observer):
            for action in actions:
                with patch.object(ai,'get_ai_policy',return_value=types.SimpleNamespace(actions=[action],forced_reason='single')):
                    child=ai._learned_opponent_child(node,ov)
        self.assertIs(child.action.params['custom'],marker);self.assertEqual(calls,{'overlay':2,'observer':2})
        self.assertEqual(before,random.getstate());self.assertTrue(node._opponent_edge_catalog.disabled)

    def test_other_selector_materializes_every_legal_choice(self):
        state=character_state(SH);node=ai.MCTSNode(state,SH);ov=particle(state,FP)
        with patch.object(ai,'_get_priors_and_value',side_effect=lambda s,side,n,a,**kw:(np.ones(len(a)),0.)):
            ai._learned_opponent_child(node,ov)
        legal={action_policy_key(a) for a in ai.get_ai_policy(state,SH).actions}
        self.assertGreater(len(legal),len(node.children))
        with patch.object(ai,'_SIM_OVERLAY',None):chosen=ai._select_child(node,1.)
        self.assertEqual({action_policy_key(c.action) for c in node.children},legal)
        self.assertTrue(node._opponent_edge_catalog.disabled);self.assertIn(action_policy_key(chosen.action),legal)

    def test_existing_duplicate_last_edge_and_unsupported_reorder(self):
        state=types.SimpleNamespace();node=ai.MCTSNode(state,SH);action=ai.Action(-1,'a',{})
        first=ai.MCTSNode(None,SH,node,action,.1);last=ai.MCTSNode(None,SH,node,action,.9)
        node.children=[first,last]
        with patch.object(ai,'get_ai_policy',return_value=types.SimpleNamespace(actions=[action],forced_reason='single')):
            self.assertIs(ai._learned_opponent_child(node,types.SimpleNamespace(patch=lambda s:None)),last)
            node.children.reverse()
            with self.assertRaisesRegex(RuntimeError,'append-only'):
                ai._learned_opponent_child(node,types.SimpleNamespace(patch=lambda s:None))

if __name__=='__main__':unittest.main()
