from pathlib import Path
import copy
import gzip
import hashlib
import os
import pickle
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
SOURCE = Path(os.environ.get('WOTR_TIMING_SOURCE', str(Path(__file__).resolve().parent))).resolve()
REVIEW = Path(os.environ.get('WOTR_REVIEW_ROOT', str(SOURCE.parent / 'audit/2026-09-04/decision_pass/stronghold_hunt_interpose_review'))).resolve()
os.environ.update(WOTR_ACT_GEN='v49', WOTR_TIMING_GENERATION='v49', WOTR_NATIVE='0', CUDA_VISIBLE_DEVICES='-1', WOTR_REWARD_CONTRACT='predecision_profile_v1')
sys.path.insert(0,str(SOURCE))
import ai
assert Path(ai.__file__).resolve().parent == SOURCE
import numpy as np
import torch
import warmstart_reveal_route as converter
assert Path(converter.__file__).resolve().parent == SOURCE
from warmstart_reveal_route import convert_checkpoint, convert_file

class ConverterTests(unittest.TestCase):
    def test_actual_cli_creates_new_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'converted.pt'
            command = [sys.executable, str(SOURCE / 'warmstart_reveal_route.py'), str(self.path), str(output)]
            first = subprocess.run(command, capture_output=True, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr.decode(errors='replace'))
            self.assertTrue(output.is_file())
            before = output.read_bytes()
            second = subprocess.run(command, capture_output=True, timeout=60)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, output.read_bytes())

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.path = REVIEW/'game_update_r2/live_fp_ring.pt'
        cls.source = torch.load(cls.path,map_location='cpu',weights_only=True)
        import test_audit0905_mithril_return as fixture
        cls.states=[fixture.setup(), fixture.move(fixture.setup())]
        import test_audit0905_reveal_route_context as context_fixture
        cls.states.extend(context_fixture.different_public_routes())

    def check_weights_and_forward(self, source):
        original=copy.deepcopy(source)
        converted=convert_checkpoint(source,'a'*64)
        projection=converted['warm_start']['expanded_parameter']
        for key,tensor in source['model_state'].items():
            torch.testing.assert_close(tensor,original['model_state'][key],rtol=0,atol=0)
            new=converted['model_state'][key]
            if key==projection:
                torch.testing.assert_close(new[:,:-64],tensor,rtol=0,atol=0)
                self.assertEqual(torch.count_nonzero(new[:,-64:]).item(),0)
            else:
                torch.testing.assert_close(new,tensor,rtol=0,atol=0)
        old=ai.WOTRValueNet(4216,side='fp').eval();old.load_state_dict(source['model_state'])
        new=ai.WOTRValueNet(4280,side='fp').eval();new.load_state_dict(converted['model_state'])
        with torch.no_grad():
            for state in self.states:
                x=torch.from_numpy(ai.encode_state(state,ai.Side.FREE_PEOPLES)[None])
                actions=ai.get_legal_actions(state,ai.Side.FREE_PEOPLES)
                enc=torch.from_numpy(np.asarray([ai.encode_action(a,state,ai.Side.FREE_PEOPLES) for a in actions]))
                regions=torch.tensor([ai._get_action_region_index(a) for a in actions])
                h1,n1=old.forward_backbone(x[:,:4216]);h2,n2=new.forward_backbone(x)
                torch.testing.assert_close(old(x[:,:4216]),new(x),rtol=2e-6,atol=2e-6)
                torch.testing.assert_close(old.action_scores(h1,n1,enc,regions,with_delta=True),new.action_scores(h2,n2,enc,regions,with_delta=True),rtol=2e-6,atol=2e-6)
        self.assertNotIn('training_scope',converted)
        self.assertEqual(converted['warm_start']['source_training_scope'],source.get('training_scope'))
        self.assertEqual(converted['warm_start_updates'],{'value':0,'policy':0})
        return converted

    def test_actual_updated_gnn_weights_and_literal_context_forward(self):
        self.check_weights_and_forward(self.source)

    def test_transformer_nonzero_old_weights_preserved(self):
        with patch.object(ai,'NET_ARCH_NAME','xformer'),patch.object(ai,'NET_ARCH_CFG',ai._NET_ARCHS['xformer']):
            old=ai.WOTRValueNet(4216,side='fp')
            with torch.no_grad():old.policy_delta[2].weight.fill_(.03)
            source={**self.source,'model_state':old.state_dict(),'arch_name':'xformer'}
            source.pop('training_scope',None)
            self.check_weights_and_forward(source)

    def test_invalid_shape_contract_scope_and_nonfinite_refused(self):
        for kind in ('dimension','contract','shape','scope','nan'):
            source=copy.deepcopy(self.source)
            if kind=='dimension':source['input_dim']=4280
            elif kind=='contract':source['observation_contract']='unknown'
            elif kind=='shape':source['model_state']['backbone.0.weight']=source['model_state']['backbone.0.weight'][:,:-1]
            elif kind=='scope':source['training_scope']['input_dim']=4280
            else:source['model_state']['backbone.0.weight'][0,0]=float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError):convert_checkpoint(source,'a'*64)

    def test_exclusive_output_source_unchanged_and_actual_loader(self):
        before=self.path.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'converted.pt'
            result=convert_file(self.path,destination)
            loaded=ai.load_value_net(str(destination),side='fp',profile='ring')
            self.assertEqual(loaded.input_dim,4280)
            self.assertIsNone(loaded.training_scope)
            with self.assertRaises(ValueError):convert_file(self.path,destination)
            with self.assertRaises(ValueError):convert_file(self.path,self.path)
            self.assertEqual(result['warm_start']['source_sha256'],hashlib.sha256(before).hexdigest())
        self.assertEqual(self.path.read_bytes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
