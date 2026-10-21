"""GC boundaries preserve actual search cleanup, CPU model outputs and RNG."""
import copy,gc,os,pickle,random,unittest,weakref
from unittest.mock import patch
import numpy as np
import torch
import ai,rng_streams
from search_collection import deferred_gc,search_collection_scope

class _Cycle:
    pass

class SearchCollectionTests(unittest.TestCase):
    def setUp(self):
        self.enabled=gc.isenabled();self.py=random.getstate();self.np=np.random.get_state()
        self.torch_rng=torch.get_rng_state().clone()
        names=('_episode','_search_python','_search_numpy','_depth','_owner_thread','_search_calls')
        self.streams={n:copy.deepcopy(getattr(rng_streams,n)) for n in names}
        self.env=patch.dict(os.environ,{'WOTR_SEARCH_DEFER_GC':'1','WOTR_RNG_CONTRACT':rng_streams.LEGACY})
        self.env.start()
        self.globals=patch.multiple(ai,_SIM_OVERLAY=None,_SEARCH_NETS=(None,None),_SEARCH_ROOT_SIDE=None,
            _SEARCH_FAILURE_DIAGNOSTIC=None,_OPPONENT_POLICY_CACHE={},_TREE_CACHE={},_COMBAT_FORECAST_OBSERVER=None)
        self.globals.start();gc.enable()

    def tearDown(self):
        self.globals.stop();self.env.stop()
        for key,value in self.streams.items():setattr(rng_streams,key,value)
        random.setstate(self.py);np.random.set_state(self.np);torch.set_rng_state(self.torch_rng)
        (gc.enable if self.enabled else gc.disable)()

    def scope(self,body):return search_collection_scope(ai._search_failure_scope(body))

    def dirty(self):
        ai._SIM_OVERLAY=object();ai._SEARCH_NETS=(object(),object());ai._SEARCH_ROOT_SIDE=object()
        ai._OPPONENT_POLICY_CACHE['sample']=[.25,.75]

    def clean(self):
        self.assertIsNone(ai._SIM_OVERLAY);self.assertEqual(ai._SEARCH_NETS,(None,None))
        self.assertIsNone(ai._SEARCH_ROOT_SIDE);self.assertEqual(ai._OPPONENT_POLICY_CACHE,{})

    def test_default_and_explicit_off_preserve_caller_gc_without_collection(self):
        token=object()
        for value in (None,'0'):
            with self.subTest(value=value),patch.dict(os.environ,{},clear=False),patch('search_collection.gc.collect') as collect:
                if value is None:os.environ.pop('WOTR_SEARCH_DEFER_GC',None)
                else:os.environ['WOTR_SEARCH_DEFER_GC']=value
                def body():self.assertTrue(gc.isenabled());return token
                self.assertIs(search_collection_scope(body)(),token);collect.assert_not_called()

    def test_success_cleans_search_before_collection_and_keeps_reachable_tree(self):
        token=_Cycle();token.self=token;reachable=weakref.ref(token);dead=[];calls=[];real_collect=gc.collect
        ai._TREE_CACHE['child']=token
        def body():
            self.assertFalse(gc.isenabled());self.dirty()
            garbage=_Cycle();garbage.self=garbage;dead.append(weakref.ref(garbage))
            return token
        def collect():
            self.clean();self.assertIs(ai._TREE_CACHE['child'],token);calls.append(1)
            return real_collect()
        with patch('search_collection.gc.collect',side_effect=collect):self.assertIs(self.scope(body)(),token)
        self.assertEqual(calls,[1]);self.assertIsNone(dead[0]());self.assertIs(reachable(),token);self.assertTrue(gc.isenabled())

    def test_exceptions_keep_original_error_and_diagnostic_then_clear_partial_tree(self):
        for error in (RuntimeError('literal failure'),KeyboardInterrupt('literal interrupt')):
            with self.subTest(error=type(error).__name__):
                events=[];real_collect=gc.collect
                ai._TREE_CACHE.update(child=object(),sig='partial',root_side='fp',policy_version=3)
                def body():self.dirty();raise error
                def capture(exc,args,kwargs,rng):
                    self.assertIs(exc,error);self.assertIsNotNone(ai._SIM_OVERLAY);events.append('capture')
                    return {'literal':'failure-only'}
                def collect():
                    self.clean();self.assertTrue(all(ai._TREE_CACHE[k] is None for k in ('child','sig','root_side','policy_version')))
                    self.assertEqual(ai._SEARCH_FAILURE_DIAGNOSTIC,{'literal':'failure-only'});events.append('collect')
                    return real_collect()
                with patch.object(ai,'_capture_search_failure',side_effect=capture),patch('search_collection.gc.collect',side_effect=collect):
                    with self.assertRaises(type(error)) as caught:self.scope(body)()
                self.assertIs(caught.exception,error);self.assertEqual(events,['capture','collect']);self.assertTrue(gc.isenabled())

    def test_nested_scope_restores_disabled_caller_and_collects_once(self):
        gc.disable();threshold=gc.get_threshold();callbacks=list(gc.callbacks);real_collect=gc.collect
        with patch('search_collection.gc.collect',side_effect=real_collect) as collect:
            with deferred_gc() as outer:
                with deferred_gc() as inner:self.assertFalse(gc.isenabled())
                self.assertEqual(collect.call_count,0);self.assertFalse(gc.isenabled())
            self.assertEqual(collect.call_count,1)
        self.assertTrue(outer['outermost']);self.assertFalse(inner['outermost']);self.assertFalse(gc.isenabled())
        self.assertEqual(gc.get_threshold(),threshold);self.assertEqual(gc.callbacks,callbacks)

    def test_isolated_rng_and_cpu_policy_noise_are_identical_on_and_off(self):
        os.environ['WOTR_RNG_CONTRACT']=rng_streams.ISOLATED
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43);model=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Tanh(),torch.nn.Linear(7,3)).eval()
        parameters=[p.detach().clone() for p in model.parameters()];torch_before=torch.get_rng_state().clone()
        def run(enabled):
            os.environ['WOTR_SEARCH_DEFER_GC']=enabled
            rng_streams.start_episode({'rng_episode_seed':419,'rng_episode_identity':'collection-regression'})
            game_before=pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)
            def body():
                self.dirty()
                x=torch.tensor([[random.random(),random.random(),.25,.75]],dtype=torch.float32)
                with torch.no_grad():logits=model(x)[0].numpy()
                noise=np.random.dirichlet([.3,.3,.3]);probs=np.exp(logits-logits.max());probs/=probs.sum()
                mixed=.75*probs+.25*noise;choice=random.choices(range(3),weights=mixed)[0]
                return logits.copy(),mixed,choice
            outputs=[self.scope(body)() for _ in range(2)]
            self.assertEqual(game_before,pickle.dumps((random.getstate(),np.random.get_state()),protocol=5))
            self.assertEqual(rng_streams.search_call_count(),2)
            return pickle.dumps((outputs,rng_streams._search_python,rng_streams._search_numpy),protocol=5)
        self.assertEqual(run('0'),run('1'))
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
        for old,new in zip(parameters,model.parameters()):self.assertTrue(torch.equal(old,new))

if __name__=='__main__':unittest.main()
