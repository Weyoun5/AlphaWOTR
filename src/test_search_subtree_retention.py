"""Selected-subtree lifetime and real pending-search regressions; CPU only.

Only maintained engine fixtures are imported. No saved games, audit files,
checkpoints or private implementation path is required.
"""
import ast,copy,gc,hashlib,inspect,os,pickle,random,textwrap,unittest,weakref
from unittest.mock import patch
import numpy as np
import ai,rng_streams

class Payload:
    def __init__(self,size=0):self.bytes=bytearray(size)

def serial(x):
    import numpy as np
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if hasattr(x,'action_type'):return [x.die_idx,x.action_type,serial(x.params)]
    if hasattr(x,'value'):return x.value
    return x

def _search_controls():
    import numpy as np,rng_streams
    from test_onslaught_payments import position
    from test_learned_opponent_policy import character_state,FP,SH
    from test_combat_continuation_literal import live_battle,card_id
    from ai_policies import action_policy_key,POLICY_VERSION
    outputs={};lifetime={}
    def reset(seed):
        ai._TREE_CACHE.clear();ai._TREE_CACHE.update(child=None,sig=None,hits=0,misses=0)
        ai._ENC_CACHE.clear();ai._RIDX_CACHE.clear();ai._OPPONENT_POLICY_CACHE.clear()
        ai._SEARCH_ROOT_SIDE=None;ai._SIM_OVERLAY=None
        rng_streams.start_episode({'rng_episode_seed':seed,'rng_episode_identity':'subtree-retention'})
    def priors(state,side,net,actions,**kw):
        values=np.array([100. if a.action_type=='combat_effect_choice' and a.params.get('count')==2 else 1. for a in actions],np.float32)
        return values/values.sum(),.125
    def call(state,side,sims=16):
        before=pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)
        result=ai.mcts_search(state,side,object(),object(),num_simulations=sims,temperature=0.,add_noise=True)
        assert before==pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)
        cache=ai._TREE_CACHE.get('child')
        meta={'chosen':serial(result[0]) if result else None,'info':serial(result[1]) if result else None,
            'hits':ai._TREE_CACHE['hits'],'misses':ai._TREE_CACHE['misses'],
            'search_rng':hashlib.sha256(pickle.dumps((rng_streams._search_python,rng_streams._search_numpy),protocol=5)).hexdigest(),
            'cache':None if cache is None else {'side':cache.side.value,'children':len(cache.children),'action':serial(cache.action),'is_chance':cache.is_chance}}
        return result,meta
    # Plain callables: Mock call histories would themselves retain every state.
    with patch.object(ai,'_get_priors_and_value',new=priors),patch.object(ai,'_mcts_eval_state',new=lambda *a,**kw:.125),patch.object(ai,'_TURN_HORIZON',0):
        reset(791);state=position(regulars=2,elites=1)
        first,m1=call(state,SH,24);cache=ai._TREE_CACHE.get('child')
        assert first[0].action_type=='combat_effect_choice' and first[0].params.get('count')==2,m1['chosen']
        assert cache is not None and cache.side==SH and cache.children,m1['cache']
        assert cache.parent is None
        after=ai.apply_action(state,first[0]);second,m2=call(after,SH,16)
        assert m2['hits']==1,(m1,m2)
        outputs['pending_reuse_hit']=[m1,m2]
        reset(791);state=position(regulars=2,elites=1)
        first,m1=call(state,SH,24);after=ai.apply_action(state,first[0]);after.fellowship.corruption+=1
        marker=Payload(8192);probe=weakref.ref(marker);ai._TREE_CACHE['child'].state._retention_probe=marker;del marker
        real_expand=ai._expand_node;observations=[]
        def watched_expand(*args,**kwargs):
            if not observations:
                gc.collect();observations.append(probe() is not None)
            return real_expand(*args,**kwargs)
        with patch.object(ai,'_expand_node',side_effect=watched_expand):second,m2=call(after,SH,16)
        assert observations==[False],observations
        lifetime['miss_cached_state_alive_at_fresh_expansion']=observations[0]
        assert m2['hits']==0 and m2['misses']==2
        outputs['signature_miss']=[m1,m2]
        reset(815);state=character_state(SH)
        die=ai.ActionDie([ai.ActionDieFace.MUSTER],FP);die.current=ai.ActionDieFace.MUSTER
        state.dice[FP]=[die]
        first,m1=call(state,SH,12);after=ai.apply_action(state,first[0])
        assert m1['cache'] is not None and m1['cache']['side']==FP.value,m1
        second,m2=call(after,FP,8);assert m2['hits']==0 and m2['misses']==2
        outputs['opposite_side_miss']=[m1,m2]
        reset(791);state=position(regulars=2,elites=1)
        first,m1=call(state,SH,24)
        from test_audit0904_gandalf_draw import opening_for
        forced_state=opening_for('FREE_PEOPLES_Character_I_Will_Go_Alone','character')
        projection=ai.get_ai_policy(forced_state,FP);assert projection.forced_reason
        with patch.object(ai,'_get_priors_and_value',side_effect=AssertionError('Forced action must not infer')):
            second,m2=call(forced_state,FP,16)
        assert second[1] is None and ai._TREE_CACHE.get('child') is None
        outputs['forced_clears_previous_cache']={'reason':projection.forced_reason,'searches':[m1,m2]}
        reset(913);state=character_state(SH)
        first,m1=call(state,SH,12);after=ai.apply_action(state,first[0])
        second,m2=call(after,SH,4);outputs['ordinary_endturn_or_forced']=[m1,m2]
        reset(913);state=character_state(SH)
        with patch.object(ai,'_TURN_HORIZON',1):
            first,m1=call(state,SH,4)
        cached=ai._TREE_CACHE.get('child');assert cached is not None
        assert any(c.action.action_type=='end_turn' and c.parent is cached for c in cached.children)
        outputs['endturn_descendant_parent']={'result':m1,'child_actions':[serial(c.action) for c in cached.children]}
        reset(613);state,attacker,defender=live_battle(fp_attacks=False)
        ai.game_logic.initialize_decks(state);cid=card_id('Onslaught')
        deck='char' if '_Character_' in cid else 'strat'
        getattr(state,'sh_'+deck+'_deck').remove(cid)
        state.discard_face_up['sh_character' if deck=='char' else 'sh_strategy'].append(cid)
        state.pending_combat={'attacker_region':attacker,'defender_region':defender,
            'attacker_side':SH,'defender_side':FP,'phase':'defender_combat_card',
            'deciding_side':FP,'atk_combat_card':cid,'def_combat_card':None,'rounds_fought':0,
            'is_siege':False,'shine_asked':True}
        assert ai._hidden_commit_card(state,FP) is not None
        menu=ai.get_ai_policy(state,FP)
        first,m1=call(state,FP,4)
        assert ai._TREE_CACHE.get('child') is None
        outputs['blinded_or_forced']={'menu':len(menu.actions),'forced':menu.forced_reason,'result':m1}
    return outputs,lifetime

class SubtreeRetentionTests(unittest.TestCase):
    def setUp(self):
        self.py=random.getstate();self.np=np.random.get_state();self.gc_enabled=gc.isenabled()
        self.streams={n:copy.deepcopy(getattr(rng_streams,n)) for n in ('_episode','_search_python','_search_numpy','_depth','_owner_thread','_search_calls')}
        self.env=patch.dict(os.environ,{'WOTR_SEARCH_DEFER_GC':'1','WOTR_RNG_CONTRACT':rng_streams.ISOLATED})
        self.env.start()
        self.globals=patch.multiple(ai,_TREE_REUSE=True,_TREE_CACHE={'child':None,'sig':None,'hits':0,'misses':0},
            _OPPONENT_POLICY_MODE='learned_policy',_OPPONENT_POLICY_EPSILON=.02,_OPPONENT_POLICY_CACHE={},
            _SEARCH_NETS=(None,None),_SEARCH_ROOT_SIDE=None,_SIM_OVERLAY=None,_COMBAT_FORECAST_OBSERVER=None)
        self.globals.start()

    def tearDown(self):
        ai._TREE_CACHE['child']=None;self.globals.stop();self.env.stop()
        for key,value in self.streams.items():setattr(rng_streams,key,value)
        random.setstate(self.py);np.random.set_state(self.np)
        (gc.enable if self.gc_enabled else gc.disable)()

    def stash_function(self,*,blind=False,reuse=True):
        # Compile the actual nested function with literal enclosing context.
        source=ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(ai.mcts_search))))
        outer=next(n for n in source.body if isinstance(n,ast.FunctionDef))
        nested=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='_stash_tree')
        from ai_policies import POLICY_VERSION
        namespace=dict(vars(ai),side=ai.Side.FREE_PEOPLES,_blind9=blind,_TREE_REUSE=reuse,
            POLICY_VERSION=POLICY_VERSION,_state_sig=lambda state:'literal-state-signature')
        exec(compile(ast.Module(body=[nested],type_ignores=[]),'<actual nested stash>','exec'),namespace)
        return namespace['_stash_tree']

    def test_stash_releases_ancestors_and_siblings_but_keeps_descendant_parents(self):
        root=ai.MCTSNode(Payload(256),ai.Side.FREE_PEOPLES)
        keep=ai.MCTSNode(Payload(256),root.side,root,ai.Action(-1,'literal_keep',{}),.5)
        sibling=ai.MCTSNode(Payload(256),root.side,root,ai.Action(-1,'literal_sibling',{}),.5)
        descendant=ai.MCTSNode(Payload(256),keep.side,keep,ai.Action(-1,'end_turn',{}),1.)
        root.children=[keep,sibling];keep.children=[descendant]
        old=weakref.ref(root.state);other=weakref.ref(sibling.state);kept=weakref.ref(keep.state)
        before=random.getstate();self.stash_function()(keep)
        self.assertEqual(random.getstate(),before);self.assertIsNone(keep.parent)
        self.assertIs(descendant.parent,keep);self.assertIs(ai._TREE_CACHE['child'],keep)
        del root,sibling,keep,descendant;gc.collect()
        self.assertIsNone(old());self.assertIsNone(other());self.assertIsNotNone(kept())

    def test_existing_cache_admission_does_not_change_for_leaf_side_blind_or_chance(self):
        for case in ('eligible','opposite_side','leaf','chance','endturn_chance','blinded','lazy','disabled'):
            with self.subTest(case=case):
                ai._TREE_CACHE['child']=None
                root=ai.MCTSNode(Payload(),ai.Side.FREE_PEOPLES)
                child=ai.MCTSNode(Payload(),ai.Side.SHADOW if case=='opposite_side' else root.side,root,ai.Action(-1,'literal',{}),1.)
                child.children=[] if case=='leaf' else [ai.MCTSNode(Payload(),child.side,child,ai.Action(-1,'next',{}),1.)]
                if case in ('chance','endturn_chance'):child.is_chance=True
                if case=='endturn_chance':child.action=ai.Action(-1,'end_turn',{})
                if case=='lazy':child.state=None
                root.children=[child]
                self.stash_function(blind=case=='blinded',reuse=case!='disabled')(child)
                accepted=case in ('eligible','opposite_side','leaf')
                self.assertEqual(ai._TREE_CACHE['child'] is child,accepted)
                self.assertEqual(child.parent is None,accepted)

    def test_real_successive_searches_preserve_physical_choices_rng_and_reuse(self):
        first,lifetime=_search_controls();second,repeated=_search_controls()
        self.assertEqual(first,second)
        self.assertEqual(lifetime,{'miss_cached_state_alive_at_fresh_expansion':False})
        self.assertEqual(lifetime,repeated)

if __name__=='__main__':unittest.main()
