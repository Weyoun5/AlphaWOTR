"""Strict schema conversion controls; reusable for base/wide/xformer runtimes."""
import os
os.environ.setdefault('WOTR_ACT_GEN','v52')
os.environ.setdefault('WOTR_NATIVE','0')
os.environ.setdefault('WOTR_OBSERVATION_TRAINING','full_network_v1')
import copy
import hashlib
import io
import tempfile
from pathlib import Path
import unittest
import torch
import ai
import optimizer_continuity as oc
import warmstart_combat_schema as migration
from combat_round_observation import *
from reward_contract import current_contract, PROFILE
from trace_decisions import current_learning_context

class Migration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        target=ai.WOTRValueNet(TARGET_STATE_DIM,side='fp')
        target.observation_contract=OBSERVATION_CONTRACT
        target.action_encoding_contract=ACTION_CONTRACT
        target.reward_contract=current_contract()
        target.reward_profile='ring' if current_contract()==PROFILE else None
        ident=oc._identity(target)
        try:
            ai.ACTION_ENCODING_DIM=SOURCE_ACTION_DIM
            cls.source_net=ai.WOTRValueNet(SOURCE_STATE_DIM,side='fp')
        finally:ai.ACTION_ENCODING_DIM=TARGET_ACTION_DIM
        cls.source={'model_state':cls.source_net.state_dict(),'arch_name':ai.NET_ARCH_NAME,
            'input_dim':SOURCE_STATE_DIM,'action_dim':SOURCE_ACTION_DIM,
            'reward_profile':target.reward_profile,
            **{key:copy.deepcopy(ident[key]) for key in ('side','head_act','reward_contract','ai_policy_contract',
                'selector_contract','selector_config','selector_config_sha256')},
            'observation_contract':migration.SOURCE_CONTRACT,
            'ai_policy_contract':migration.SOURCE_POLICY_CONTRACT,
            'action_encoding_contract':migration.SOURCE_ACTION_CONTRACT}

    def test_old_tensor_columns_preserved_and_new_columns_zero(self):
        result=migration.convert_checkpoint(self.source,'1'*64)
        for name,tensor in self.source['model_state'].items():
            expected=migration._expand(tensor,name,self.source_net.card_embedding.embedding_dim)
            self.assertTrue(torch.equal(expected,result['model_state'][name]),name)
        self.assertEqual(result['training_status'],'warm_start_not_retrained')
        self.assertNotIn(oc.FIELD,result)
        self.assertFalse(result['warm_start']['historical_rows_migrated'])

    def test_zero_tail_predictions_and_new_columns_learn(self):
        result=migration.convert_checkpoint(self.source,'1'*64)
        target=ai.WOTRValueNet(TARGET_STATE_DIM,side='fp')
        target.load_state_dict(result['model_state']);target.eval();self.source_net.eval()
        x=torch.zeros((1,SOURCE_STATE_DIM))
        spatial=ai.NUM_REGIONS*ai.NODE_FEAT_DIM
        x[:,spatial:spatial+ai.MAX_HAND_SIZE]=-1
        enc=torch.zeros((2,SOURCE_ACTION_DIM));enc[:,33]=-1
        regions=torch.tensor([-1,-1])
        z=torch.nn.functional.pad(x,(0,STATE_TAIL_DIM));ez=torch.nn.functional.pad(enc,(0,ACTION_TAIL_DIM))
        with torch.no_grad():
            a,h,n=self.source_net.forward_spatial(x);b,hh,nn=target.forward_spatial(z)
            p=self.source_net.action_scores(h,n,enc,regions);q=target.action_scores(hh,nn,ez,regions)
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
        z[:,-STATE_TAIL_DIM:]=.5;ez[:,-ACTION_TAIL_DIM:]=.5
        value,h,n=target.forward_spatial(z)
        (value.sum()+target.action_scores(h,n,ez,regions).sum()).backward()
        name='tok_global.weight' if target.arch_kind=='transformer' else 'backbone.0.weight'
        self.assertTrue(torch.count_nonzero(dict(target.named_parameters())[name].grad[:,-STATE_TAIL_DIM:])>0)

    def test_wrong_schema_or_shape_or_source_digest_refused(self):
        for key,value in [('observation_contract','future'),('action_dim',104),('side','invalid')]:
            source=dict(self.source);source[key]=value
            with self.assertRaises(ValueError):migration.convert_checkpoint(source,'1'*64)
        with self.assertRaises(ValueError):migration.convert_checkpoint(self.source,'short')
        source=dict(self.source);source['model_state']=dict(source['model_state'])
        source['model_state']['policy_delta.0.weight']=source['model_state']['policy_delta.0.weight'][:,:-1]
        with self.assertRaises(ValueError):migration.convert_checkpoint(source,'1'*64)

    def test_corrupt_adam_hash_refused_without_dropping_optimizer(self):
        source=dict(self.source);source[oc.FIELD]={'version':oc.VERSION,'blob':b'broken','sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'Adam envelope'):migration.convert_checkpoint(source,'1'*64)

    def test_file_conversion_explicit_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.pt';target=Path(folder)/'target.pt'
            torch.save(self.source,source)
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):migration.convert_file(source,target,'0'*64)
            self.assertFalse(target.exists())
            migration.convert_file(source,target,sha)
            with self.assertRaises(ValueError):migration.convert_file(source,target,sha)
            with self.assertRaises(ValueError):migration.convert_file(source,source,sha)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),sha)

if __name__=='__main__':unittest.main()
