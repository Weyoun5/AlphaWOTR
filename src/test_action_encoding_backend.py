"""Portable native backend controls; requires a local explicit native build."""
from pathlib import Path
import importlib.util,json,os,pickle,random,shutil,tempfile,types,unittest
from unittest.mock import patch
import numpy as np
import ai,board,game_state as gs
import search_action_encoding as helper
import action_encoding_backend as backend
HERE=Path(__file__).resolve().parent

class ActionEncodingBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state=gs.setup_initial_game(board.load_board())
        cls.state.dice_rolled=True
        for side,pool in cls.state.dice.items():
            for i,die in enumerate(pool):die.current=gs.ActionDieFace.CHARACTER if i==0 else gs.ActionDieFace.ARMY
        with patch.dict(os.environ,{'WOTR_ACTION_ENCODING':'native'}):backend.activate(helper)

    def compare(self,actions,state=None,side=gs.Side.SHADOW):
        state=state or self.state
        a=helper._PYTHON_ENCODE_ARMY_BATCH(ai,actions,state,side,{},ai._encode_actions_batch_base)
        b=helper.encode_army_batch(ai,actions,state,side,{},ai._encode_actions_batch_base)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        if actions:np.testing.assert_array_equal(b[0],np.asarray([ai.encode_action(a,state,side) for a in actions],dtype=np.float32))

    def test_aa_runtime_dependency_replacement_refused_before_first_use(self):
        manifest=json.loads((HERE/'action_encoding_build.json').read_text(encoding='utf-8'))
        source=json.loads((HERE/'action_encoding_source.json').read_text(encoding='utf-8'))
        with patch.object(ai,'_armies_lookup',lambda *args:None):
            with self.assertRaisesRegex(RuntimeError,'runtime AI dependency changed'):
                backend.verify_ai_binding(ai,manifest,source)

    def copied_backend(self,path):
        for name in ('action_encoding_backend.py','action_encoding_source.json','action_encoding_build.json','search_action_encoding.py','wotr_action_encoding.pyx'):
            shutil.copy2(HERE/name,path/name)
        manifest=json.loads((path/'action_encoding_build.json').read_text(encoding='utf-8'))
        shutil.copy2(HERE/manifest['binary'],path/manifest['binary'])
        spec=importlib.util.spec_from_file_location('isolated_action_backend',path/'action_encoding_backend.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        owner=types.SimpleNamespace(__file__=str(path/'search_action_encoding.py'),encode_army_batch=helper._PYTHON_ENCODE_ARMY_BATCH)
        return module,owner

    def negative(self,change,reason):
        with tempfile.TemporaryDirectory(prefix='action_backend_test_') as folder:
            path=Path(folder);module,owner=self.copied_backend(path);change(path)
            with patch.dict(os.environ,{'WOTR_ACTION_ENCODING':'native'}):
                with self.assertRaisesRegex(RuntimeError,reason):module.activate(owner)
            self.assertIs(owner.encode_army_batch,helper._PYTHON_ENCODE_ARMY_BATCH)

    def test_complete_both_side_opening_menus_direct_oracle(self):
        before=pickle.dumps(self.state);rng=pickle.dumps((random.getstate(),np.random.get_state()))
        for side in gs.Side:self.compare(ai.get_ai_policy(self.state,side).actions,side=side)
        self.assertEqual(pickle.dumps(self.state),before);self.assertEqual(pickle.dumps((random.getstate(),np.random.get_state())),rng)

    def test_all_counts_and_later_mutations(self):
        action=next(a for a in ai.get_ai_policy(self.state,gs.Side.SHADOW).actions if a.action_type=='army_move')
        actions=[ai.Action(action.die_idx,'army_move',dict(action.params,n_regular=r,n_elite=e)) for r in range(11) for e in range(11)]
        self.compare(actions)
        for a in actions:a.params.update(mouth_muster=True,card_followup='actual_source',n_regular=2)
        self.compare(actions)

    def test_unknown_shapes_and_small_menus(self):
        self.compare([ai.Action(-1,'pass',{}) for _ in range(35)])
        self.compare([ai.Action(-1,'pass',{})])
        actions=[ai.Action(0,'army_move',{'from':'missing','to':'morannon','n_regular':1,'n_elite':0,'unknown':i}) for i in range(35)]
        self.compare(actions)

    def test_missing_binary(self):
        self.negative(lambda p:(p/json.loads((p/'action_encoding_build.json').read_text(encoding='utf-8'))['binary']).unlink(),'Native action encoding unavailable')

    def test_changed_binary(self):
        def change(p):
            binary=p/json.loads((p/'action_encoding_build.json').read_text(encoding='utf-8'))['binary'];binary.write_bytes(binary.read_bytes()+b'changed')
        self.negative(change,'binary hash differs')

    def test_changed_python_source(self):
        self.negative(lambda p:(p/'search_action_encoding.py').write_text('# changed',encoding='utf-8'),'Python implementation changed')

    def test_changed_cython_source(self):
        self.negative(lambda p:(p/'wotr_action_encoding.pyx').write_text('# changed',encoding='utf-8'),'Cython implementation changed')

    def test_wrong_abi(self):
        def change(p):
            path=p/'action_encoding_build.json';m=json.loads(path.read_text(encoding='utf-8'));m['abi']['machine']='foreign';path.write_text(json.dumps(m),encoding='utf-8')
        self.negative(change,'ABI/NumPy version differs')

    def test_source_manifest_relabel_refused(self):
        self.negative(lambda p:(p/'action_encoding_source.json').write_text(json.dumps({'schema':'wotr_action_encoding_source_v1'}),encoding='utf-8'),'source binding changed')

    def test_default_python_needs_no_build(self):
        with tempfile.TemporaryDirectory(prefix='action_backend_test_') as folder:
            path=Path(folder);module,owner=self.copied_backend(path)
            (path/'action_encoding_build.json').unlink()
            with patch.dict(os.environ,{'WOTR_ACTION_ENCODING':'python'}):diagnostic=module.activate(owner)
            self.assertEqual(diagnostic,{'backend':'python','activated':False})

    def test_activated_native_cannot_silently_switch_to_python(self):
        with patch.dict(os.environ,{'WOTR_ACTION_ENCODING':'python'}):
            with self.assertRaisesRegex(RuntimeError,'cannot switch'):backend.activate(helper)

    def test_bound_ai_identity_cannot_change(self):
        self.compare([ai.Action(-1,'pass',{})])
        with self.assertRaisesRegex(RuntimeError,'another AI module'):
            helper.encode_army_batch(types.SimpleNamespace(__file__=ai.__file__),[],None,None,{},None)

if __name__=='__main__':unittest.main()
