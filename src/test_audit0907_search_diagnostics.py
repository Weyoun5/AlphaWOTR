"""Failure-only capture preserves replay RNG and never retains an MCTS tree."""
import copy,pickle,random,types,unittest
from unittest.mock import patch
from test_audit0907_ring_search_exhaustion import ai,position,gs
import numpy as np
import rng_streams


class FailureDiagnostic(unittest.TestCase):
    def setUp(self):
        ai._COMBAT_FORECAST_OBSERVER=None
        self.state,self.action=position(extra_shadow=True)
        rng_streams.start_episode({})

    def _exercise_failure(self):
        observed={}
        @ai._search_failure_scope
        def search(state,side,fp,sh,num_simulations=100,**kwargs):
            observed['python']=random.random();observed['numpy']=float(np.random.random())
            ai._SIM_OVERLAY=ai._SimOverlay(state,side)
            parent=ai.MCTSNode(state,side)
            node=ai.MCTSNode(ai.clone_state(state),gs.Side.SHADOW,parent=parent,action=self.action)
            raise RuntimeError('literal diagnostic failure')
        with self.assertRaisesRegex(RuntimeError,'literal diagnostic failure'):
            search(self.state,gs.Side.FREE_PEOPLES,None,None,num_simulations=37,add_noise=True,temperature=.3)
        return ai._SEARCH_FAILURE_DIAGNOSTIC,observed

    def test_failure_packet_replays_entry_rng_and_contains_only_state_snapshots(self):
        packet,observed=self._exercise_failure()
        generator=random.Random();generator.setstate(packet['search_entry_rng']['python'])
        self.assertEqual(generator.random(),observed['python'])
        numpy_generator=np.random.RandomState();numpy_generator.set_state(packet['search_entry_rng']['numpy'])
        self.assertEqual(float(numpy_generator.random()),observed['numpy'])
        self.assertEqual(packet['requested_simulations'],37)
        self.assertEqual(packet['search_options'],{'temperature':.3,'add_noise':True})
        self.assertEqual(packet['action']['action_type'],'elven_ring')
        self.assertIn('hand',packet['overlay'])
        self.assertIsNone(packet['root_state'].board)
        self.assertIsNone(packet['node_state'].board)
        self.assertIsNone(packet['parent_state'].board)
        self.assertEqual(packet['capture_errors'],[])
        encoded=pickle.dumps(packet)
        self.assertNotIn(b'MCTSNode',encoded)
        self.assertLess(len(encoded),200000)
        self.assertIsNone(ai._SIM_OVERLAY)

    def test_next_healthy_search_clears_packet_without_snapshot_or_io(self):
        self._exercise_failure()
        @ai._search_failure_scope
        def healthy(*args,**kwargs):return None
        with patch.object(ai,'_capture_search_failure',side_effect=AssertionError('Healthy capture forbidden')):
            self.assertIsNone(healthy(self.state,gs.Side.FREE_PEOPLES,None,None))
        self.assertIsNone(ai._SEARCH_FAILURE_DIAGNOSTIC)

    def test_failed_episode_metadata_gets_detached_packet_not_action_row(self):
        packet,_=self._exercise_failure();saved=[]
        tracer=types.SimpleNamespace(meta={'source_errors':[]},steps=[],_ann_buffer=[],save=lambda **kw:saved.append(kw))
        ai._save_failed_episode({'tracer':tracer,'state':self.state},{},RuntimeError('episode failure'))
        self.assertEqual(tracer.steps,[])
        self.assertEqual(tracer.meta['search_failure']['requested_simulations'],37)
        self.assertIsNot(tracer.meta['search_failure'],packet)
        self.assertEqual(saved[0]['status'],'error')

    def test_new_episode_clears_previous_failure_before_impl(self):
        self._exercise_failure()
        def run(*args,**kwargs):
            self.assertIsNone(ai._SEARCH_FAILURE_DIAGNOSTIC)
            return {}
        with patch.object(ai,'_run_single_episode_impl',side_effect=run):
            self.assertEqual(ai._run_single_episode(None,None,{}),{})

    def test_bad_diagnostic_copy_does_not_replace_original_episode_error(self):
        class BadCopy:
            def __deepcopy__(self,memo):raise ValueError('bad diagnostic copy')
        ai._SEARCH_FAILURE_DIAGNOSTIC={'bad':BadCopy()};saved=[]
        tracer=types.SimpleNamespace(meta={'source_errors':[]},steps=[],_ann_buffer=[],save=lambda **kw:saved.append(kw))
        ai._save_failed_episode({'tracer':tracer},{},RuntimeError('original episode error'))
        self.assertIn('original episode error',saved[0]['error'])
        self.assertTrue(any(item['code']=='search_failure_capture_copy_failed' for item in tracer.meta['source_errors']))


if __name__=='__main__':unittest.main(verbosity=2)
