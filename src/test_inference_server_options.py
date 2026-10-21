"""Actual spawned CPU inference protocol; literal tiny nets, no games/models."""
import os,sys,json,tempfile,time,unittest,multiprocessing as mp
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import infer_server as service


class _Score(torch.nn.Module):
    def forward(self,x):return x[:,2:3]

class _Zero(torch.nn.Module):
    def forward(self,x):return x[:,:1]*0

class _ToyNet(torch.nn.Module):
    def __init__(self,path,fail=False):
        super().__init__();self.checkpoint_id=str(path);self.input_dim=3
        self.reward_profile='ring';self.reward_contract='literal_reward'
        self.observation_contract='literal_observation';self.action_encoding_contract='literal_action'
        self.policy_delta=_Zero();self.spatial_action_scorer=_Score();self.value_head=torch.nn.Identity()
        self.multiplier=2 if str(path)=='revision_b' else 1;self.fail=fail
    def forward_backbone(self,x):
        if self.fail:raise RuntimeError('literal inference execution failure')
        return x[:,:1]*self.multiplier,torch.zeros(len(x),105,1)
    def _embed_action_encs(self,a):return a[:,:1],torch.zeros(len(a),1)
    def forward(self,x):return self.forward_backbone(x)[0].reshape(-1)


def _serve(conns,paths,arch,fail=False):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    def loader(ai,torch_,path,side):
        log=os.environ.get('WOTR_OPTION_LOAD_LOG')
        if log:
            with open(log,'a',encoding='utf8') as stream:stream.write(str(path)+'\n')
        return _ToyNet(path,fail).eval()
    service._load_net=loader
    service.server_main(conns,paths,arch,device='cpu')


class ServerOptionTests(unittest.TestCase):
    def start(self,frozen=False,graph=False,fail=False):
        temporary=tempfile.TemporaryDirectory(prefix='wotr_infer_options_');self.addCleanup(temporary.cleanup)
        log=Path(temporary.name)/'loads.txt'
        parent,child=mp.get_context('spawn').Pipe()
        env={'WOTR_INFER_FROZEN':str(int(frozen)),'WOTR_INFER_CUDA_GRAPH':str(int(graph)),
             'CUDA_VISIBLE_DEVICES':'-1','WOTR_OPTION_LOAD_LOG':str(log)}
        with patch.dict(os.environ,env):
            process=mp.get_context('spawn').Process(target=_serve,args=([child],{'fp':'revision_a'},'base',fail));process.start()
        child.close();client=service.InferClient(parent,response_timeout=10)
        def cleanup():
            try:
                if process.is_alive():parent.send(('stop',))
            except (OSError,EOFError):pass
            process.join(5)
            if process.is_alive():process.terminate();process.join(5)
            client.close();process.close()
        self.addCleanup(cleanup)
        return parent,process,client,log

    def test_zero_item_pvm_replies_without_desynchronizing_next_request(self):
        _,_,client,_=self.start()
        before=client.describe()
        self.assertEqual(client.pv_many('fp',[]),[])
        self.assertEqual(client.describe(),before)

    def test_default_mutable_reload_preserves_metadata_and_changes_output(self):
        parent,process,client,log=self.start()
        first=client.describe()['fp'];self.assertFalse(first['immutable_weights'])
        self.assertEqual(first['reward_profile'],'ring');self.assertEqual(first['observation_contract'],'literal_observation')
        x=np.array([[.25,0,0]],np.float32)
        np.testing.assert_equal(client.eval('fp',x),[.25])
        parent.send(('reload',{'fp':'revision_b'}))
        self.assertEqual(client.describe()['fp']['checkpoint_id'],'revision_b')
        np.testing.assert_equal(client.eval('fp',x),[.5])

    def test_frozen_reload_fails_before_loader_and_connection_closes(self):
        parent,process,client,log=self.start(frozen=True)
        self.assertTrue(client.describe()['fp']['immutable_weights'])
        parent.send(('reload',{'fp':'revision_b'}))
        process.join(5);self.assertFalse(process.is_alive());self.assertNotEqual(process.exitcode,0)
        self.assertEqual(log.read_text().splitlines(),['revision_a'])
        with self.assertRaises(Exception):client.describe(timeout=1)

    def test_graph_opt_in_on_cpu_uses_exact_eager_empty_menu_contract(self):
        _,_,client,_=self.start(frozen=True,graph=True)
        self.assertTrue(client.describe()['fp']['immutable_weights'])
        v=np.array([.25,0,0],np.float32);a=np.array([[0.],[1.]],np.float32);r=np.array([-1,0],np.int64)
        out=client.pv_many('fp',[(v,a,r),(v,a[:0],r[:0])])
        np.testing.assert_allclose(out[0][0],[1/(1+np.e),np.e/(1+np.e)],rtol=1e-6)
        self.assertEqual(out[0][1],.25);self.assertEqual(out[1][1],.25);self.assertEqual(len(out[1][0]),0)

    def test_actual_forward_error_terminates_server_and_reaches_client(self):
        _,process,client,_=self.start(frozen=True,graph=True,fail=True)
        client.describe()
        with self.assertRaises(Exception):client.eval('fp',np.zeros((1,3),np.float32))
        process.join(5);self.assertFalse(process.is_alive());self.assertNotEqual(process.exitcode,0)


def _failing_runtime_server(conns,paths,arch):_serve(conns,paths,arch,True)

def _requesting_runtime_worker(wid,conn,count,sims,trace_dir,seed0,arch,out_q,worker_stride=1):
    client=service.InferClient(conn,response_timeout=10)
    try:
        client.describe();client.eval('fp',np.zeros((1,3),np.float32))
    except Exception as exc:
        out_q.put(('e',wid,str(exc),{'record_type':'game_attempt','worker_id':wid,'episode':0,'stage':'episode','error':str(exc)}))
    finally:client.close()


class RunnerPropagationTests(unittest.TestCase):
    def test_actual_runner_records_error_and_quiesces_all_children(self):
        import test_audit0905_served_lifecycle as existing
        case=existing.ServedLifecycleTests('test_episode_failure_has_no_invented_unreported_result')
        try:
            with patch.object(existing,'_fixture_server_entry',_failing_runtime_server),patch.object(existing,'_fixture_worker',_requesting_runtime_worker):
                phase,_=case.run_case('inference_error',2,1,RuntimeError)
            self.assertEqual(phase['completed'],0)
            self.assertEqual(phase['cleanup']['alive'],[])
            self.assertNotEqual(phase['cleanup']['server_exitcode'],0)
            self.assertEqual(phase['completed']+phase['error_attempts']+phase['unreported'],2)
        finally:case.doCleanups()

if __name__=='__main__':unittest.main(verbosity=2)
