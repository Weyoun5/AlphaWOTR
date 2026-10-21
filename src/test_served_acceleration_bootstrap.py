"""Actual worker entry: native bootstrap, failure boundary and honest stats."""
import contextlib,copy,os,queue,random,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import ai,ai_policies,infer_server,policy_keys_backend,run_sp_served

class ServedAccelerationBootstrapTests(unittest.TestCase):
    def invoke(self,*,failure=None,bootstrap_failure=False,disabled_gc=False):
        sink=queue.Queue();calls=[];order=[]
        handles={key:SimpleNamespace(checkpoint_id=letter*64) for key,letter in [('fp','a'),('sh','b')]}
        real_activate=policy_keys_backend.activate
        def activate(policy):
            order.append('activate')
            if bootstrap_failure:raise RuntimeError('literal backend identity refusal')
            result=real_activate(policy);self.assertEqual(result['backend'],'native');return result
        def attach(*args,**kwargs):
            order.append('attach');self.assertEqual(kwargs,{'require_metadata':True});return handles
        def episode(fp,sh,config):
            order.append('episode');calls.append(copy.deepcopy(config))
            self.assertIs(fp,handles['fp']);self.assertIs(sh,handles['sh'])
            if failure:raise failure
            return {'game_stats':{'winner':'fp','literal_fixture':True}}
        before=(random.getstate(),np.random.get_state())
        with contextlib.ExitStack() as scope,tempfile.TemporaryDirectory(prefix='wotr_bootstrap_') as temp:
            scope.enter_context(patch.dict(os.environ,{'WOTR_POLICY_KEYS':'native','WOTR_SEARCH_DEFER_GC':'0' if disabled_gc else '1',
                'WOTR_INFER_CUDA_GRAPH':'1','WOTR_RNG_CONTRACT':'isolated_game_search_v1',
                'WOTR_SP_FP_PROFILES':'ring','WOTR_SP_SH_PROFILES':'military','WOTR_COMBAT_FORECAST_EVAL':'0'}))
            for name in ('mcts_search','_traced_execute','DEVICE'):
                scope.enter_context(patch.object(ai,name,getattr(ai,name)))
            scope.enter_context(patch.dict(ai.PLAYER_PROFILES,dict(ai.PLAYER_PROFILES)))
            scope.enter_context(patch.object(ai,'_run_single_episode',side_effect=episode))
            scope.enter_context(patch.object(infer_server,'attach_client',side_effect=attach))
            scope.enter_context(patch.object(policy_keys_backend,'activate',side_effect=activate))
            try:run_sp_served._worker(1,object(),2,64,temp,30000,'base',sink,worker_stride=4,completed_episodes=(0,))
            finally:random.setstate(before[0]);np.random.set_state(before[1])
        rows=[]
        while not sink.empty():rows.append(sink.get_nowait())
        return rows,calls,order

    def test_native_bootstrap_precedes_attach_and_stats_record_actual_options(self):
        for disabled in (False,True):
            with self.subTest(disabled_gc=disabled):
                rows,calls,order=self.invoke(disabled_gc=disabled)
                self.assertEqual(order,['activate','attach','episode']);self.assertEqual(len(calls),1)
                self.assertEqual(calls[0]['rng_episode_identity'],{'run_seed':30000,'episode_index':5})
                self.assertEqual(calls[0]['mcts_sims'],64)
                games=[r for r in rows if r[0]=='g'];self.assertEqual(len(games),1)
                self.assertFalse(any(r[0]=='e' for r in rows))
                stats=games[0][2];self.assertEqual((stats['worker_id'],stats['episode']),(1,1))
                self.assertEqual(stats['search_acceleration'],{'policy_keys':'native','deferred_gc':not disabled,'cuda_graph_requested':True})
                self.assertEqual(stats['checkpoint_ids'],{'fp':'a'*64,'sh':'b'*64});self.assertEqual(stats['served_arch'],'base')
                self.assertTrue(stats['literal_fixture']);self.assertEqual(rows[0][0],'p')

    def test_backend_refusal_reaches_queue_before_attach_or_episode(self):
        rows,calls,order=self.invoke(bootstrap_failure=True)
        self.assertEqual(order,['activate']);self.assertEqual(calls,[])
        errors=[r for r in rows if r[0]=='e'];self.assertEqual(len(errors),1)
        self.assertEqual(errors[0][3]['stage'],'worker_init');self.assertIn('backend identity refusal',errors[0][3]['error'])
        self.assertFalse(any(r[0]=='g' for r in rows))

    def test_episode_error_preserves_identity_without_success_stats(self):
        rows,calls,order=self.invoke(failure=RuntimeError('literal searched-node failure'))
        self.assertEqual(len(calls),1);self.assertEqual(order,['activate','attach','episode'])
        errors=[r for r in rows if r[0]=='e'];self.assertEqual(len(errors),1)
        record=errors[0][3];self.assertEqual((record['worker_id'],record['episode'],record['stage']),(1,1,'episode'))
        self.assertEqual(record['rng_episode_identity'],{'run_seed':30000,'episode_index':5})
        self.assertEqual(record['error_type'],'RuntimeError');self.assertNotIn('search_acceleration',record)
        self.assertFalse(any(r[0]=='g' for r in rows))

if __name__=='__main__':unittest.main()
