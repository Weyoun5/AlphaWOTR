"""Independent contracts and forward/optimizer checks on actual saved search rows."""
from pathlib import Path
import copy
import gzip
import os
import pickle
import unittest
from unittest.mock import patch
import sys
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
REPO = SOURCE.parent
HERE = REPO / 'audit/2026-09-04/decision_pass/encoded_menu_review/learning_v45'
# These immutable fixtures contain genuine search targets and original optimizer output.
# They are deliberately not regenerated from the implementation under test.
os.environ.update(WOTR_ACT_GEN='v45',WOTR_NATIVE='0',CUDA_VISIBLE_DEVICES='-1',WOTR_REWARD_CONTRACT='predecision_profile_v1')
import numpy as np
import torch
import ai
import delta_trainer as dt
from observation_training import ObservationTrainingScope, validate_training_scope


class TrainingScopeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with gzip.open(HERE/'policy_rows_r1/rows_cases.pkl.gz','rb') as stream:
            cls.data=pickle.load(stream)
        cls.source=HERE.parent/'converter_v45_final/converted_pair/live_sh_military.pt'

    def model(self):
        return ai.load_value_net(str(self.source),side='sh',profile='military')

    def direct(self,net):
        output=[]
        with torch.no_grad():
            for row in self.data['rows']:
                target,keep,masked,why=dt._masked_target(row)
                self.assertIsNone(why)
                enc=np.concatenate([np.asarray(row['ae'])[keep],row['extra_ae']])
                regions=np.concatenate([np.asarray(row['ri'])[keep],row['extra_ri']])
                h,nodes=net.forward_backbone(torch.from_numpy(row['enc'][None]))
                base=net.action_scores(h,nodes,torch.from_numpy(enc.astype(np.float32)),torch.from_numpy(regions.astype(np.int64)),with_delta=False)
                logits=net.action_scores(h,nodes,torch.from_numpy(enc.astype(np.float32)),torch.from_numpy(regions.astype(np.int64)),with_delta=True)
                output.append((torch.log_softmax(logits,0).numpy(),(logits-base).numpy()))
        return output

    def test_current_metrics_match_direct_inference_and_frozen_prefix_buffers(self):
        net=self.model()
        old={key:value.clone() for key,value in net.state_dict().items()}
        before=self.direct(net)
        metrics=dt.train_policy_delta(net,self.data['rows'],dt.DeltaTrainConfig(bs=4,epochs=4,lr=1e-3,wd=.1))
        after=self.direct(net)
        kl=[float(np.sum(np.exp(a[0])*(a[0]-b[0]))) for a,b in zip(before,after)]
        self.assertAlmostEqual(metrics['kl_p50'],float(np.percentile(kl,50)),delta=1e-5)
        self.assertAlmostEqual(metrics['kl_p95'],float(np.percentile(kl,95)),delta=1e-5)
        self.assertAlmostEqual(metrics['max_abs_delta_logit'],max(float(np.max(np.abs(item[1]))) for item in after),delta=6e-5)
        changed=[]
        for key,value in net.state_dict().items():
            if not torch.equal(value,old[key]):changed.append(key)
        self.assertIn('backbone.0.weight',changed)
        self.assertTrue(all(key.startswith('policy_delta.') or key=='backbone.0.weight' for key in changed))
        torch.testing.assert_close(net.backbone[0].weight[:,:682],old['backbone.0.weight'][:,:682],rtol=0,atol=0)
        self.assertGreater(torch.count_nonzero(net.backbone[0].weight[:,-434:]).item(),0)
        scope=net.training_scope['modes']['appended_columns_v1']
        self.assertEqual(scope['policy_steps'],8)
        self.assertEqual(scope['policy_projection_changes'],8)
        self.assertEqual(scope['raw_start'],3742)
        self.assertFalse(net.backbone[0].weight._backward_hooks)

    def test_explicit_legacy_mode_preserves_original_optimizer_weights(self):
        net=self.model()
        with patch.dict(os.environ,WOTR_OBSERVATION_TRAINING='head_only_legacy'):
            dt.train_policy_delta(net,self.data['rows'],dt.DeltaTrainConfig(bs=4,epochs=4,lr=1e-3))
        original=torch.load(HERE/'policy_head_only_r1/diagnostic_sh.pt',map_location='cpu',weights_only=True)
        for key,value in net.state_dict().items():
            torch.testing.assert_close(value,original['model_state'][key],rtol=0,atol=0)
        record=net.training_scope['modes']['head_only_legacy']
        self.assertEqual(record['policy_steps'],8)
        self.assertEqual(record['policy_projection_changes'],0)

    def test_unknown_or_inconsistent_model_refused_before_optimizer(self):
        for field,value in [('observation_contract',None),('observation_contract','unknown'),
                            ('input_dim',4181),('arch_kind','unknown'),('_global_dim',1)]:
            net=self.model()
            before={key:value.clone() for key,value in net.state_dict().items()}
            setattr(net,field,value)
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                dt.train_policy_delta(net,self.data['rows'],dt.DeltaTrainConfig(epochs=1))
            for key,tensor in net.state_dict().items():
                torch.testing.assert_close(tensor,before[key],rtol=0,atol=0)

    def test_invalid_preparation_removes_gradient_hook(self):
        net=self.model()
        rows=copy.deepcopy(self.data['rows'])
        for row in rows:row['enc']=row['enc'][:10]
        with self.assertRaisesRegex(RuntimeError,'shape'):
            dt.train_policy_delta(net,rows)
        self.assertFalse(net.backbone[0].weight._backward_hooks)

    def test_transformer_projection_trains_without_changing_older_columns(self):
        with patch.object(ai,'NET_ARCH_NAME','xformer'),patch.object(ai,'NET_ARCH_CFG',ai._NET_ARCHS['xformer']):
            torch.manual_seed(95622)
            net=ai.WOTRValueNet(4180,side='sh').eval()
            net.observation_contract='v45_deferred_card_context_v1'
            with torch.no_grad():net.tok_global.weight[:,-438:]=0
            before={key:value.clone() for key,value in net.state_dict().items()}
            dt.train_policy_delta(net,self.data['rows'][1:3],dt.DeltaTrainConfig(bs=2,epochs=1,lr=1e-3))
            cut=net.tok_global.weight.shape[1]-438
            torch.testing.assert_close(net.tok_global.weight[:,:cut],before['tok_global.weight'][:,:cut],rtol=0,atol=0)
            self.assertGreater(torch.count_nonzero(net.tok_global.weight[:,-434:]).item(),0)
            for key,value in net.state_dict().items():
                if key!='tok_global.weight' and not key.startswith('policy_delta.'):
                    torch.testing.assert_close(value,before[key],rtol=0,atol=0)
            self.assertFalse(net.tok_global.weight._backward_hooks)

    def test_scope_metadata_refuses_impossible_learning_claim(self):
        net=self.model()
        with ObservationTrainingScope(net) as scope:
            scope.record_step('value')
        record=net.training_scope
        self.assertEqual(record['modes']['appended_columns_v1']['value_projection_changes'],0)
        bad=copy.deepcopy(record)
        bad['modes']['appended_columns_v1']['value_projection_changes']=2
        with self.assertRaises(ValueError):validate_training_scope(bad)

    def test_model_layout_and_header_checked_on_save_and_load(self):
        import tempfile
        from trace_decisions import checkpoint_learning_lineage
        net = self.model()
        with ObservationTrainingScope(net) as scope:
            scope.record_step('value')
        good = copy.deepcopy(net.training_scope)
        for change in ('header', 'parameter', 'columns'):
            bad = copy.deepcopy(good)
            if change == 'header':
                bad['observation_contract'] = 'v44_fellowship_tail_v1'
                bad['input_dim'] = 3746
                bad['modes']['appended_columns_v1'].update(raw_end=3746, column_end=686)
            elif change == 'parameter':
                bad['modes']['appended_columns_v1']['parameter'] = 'tok_global.weight'
            else:
                bad['modes']['appended_columns_v1']['column_start'] += 1
                bad['modes']['appended_columns_v1']['column_end'] += 1
            net.training_scope = bad
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint_learning_lineage(net)
            checkpoint = torch.load(self.source, map_location='cpu', weights_only=True)
            checkpoint['training_scope'] = bad
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'invalid.pt'
                torch.save(checkpoint, path)
                with self.assertRaises(ValueError):
                    ai.load_value_net(str(path), side='sh', profile='military')
        net.training_scope = good
        saved = checkpoint_learning_lineage(net)
        saved['training_scope']['modes']['appended_columns_v1']['value_steps'] = 99
        self.assertEqual(net.training_scope, good)
        with ObservationTrainingScope(net) as scope:
            scope.record_step('policy')
        self.assertEqual(net.training_scope['modes']['appended_columns_v1']['value_steps'], 1)
        self.assertEqual(net.training_scope['modes']['appended_columns_v1']['policy_steps'], 1)


if __name__=='__main__':unittest.main(verbosity=2)
