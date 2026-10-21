"""Explicit optional native action encoding; Python default, no runtime builds."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,os,platform,sys,sysconfig
import numpy as np
HERE=Path(__file__).resolve().parent
_activated=None
_snapshot_prepare=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def abi():
    return dict(implementation=sys.implementation.name,cache_tag=sys.implementation.cache_tag,
                python_version=list(sys.version_info[:3]),system=platform.system(),machine=platform.machine(),
                platform=sysconfig.get_platform(),soabi=sysconfig.get_config_var('SOABI'),
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),numpy_version=np.__version__)
def require(condition,message):
    if not condition:raise RuntimeError('Native action encoding unavailable: '+message)

def verify_ai_binding(ai,manifest,source):
    require(sha(ai.__file__)==manifest['ai_source_sha256'],'AI source changed; rebuild before use')
    require(ai.ACT_GEN=='v52' and ai.ACTION_ENCODING_DIM==133,'requires exact v52 action dimensions')
    for name,expected in source['ai_functions_sha256'].items():
        require(hashlib.sha256(inspect.getsource(getattr(ai,name)).encode()).hexdigest()==expected,
                'runtime AI dependency changed: '+name)

def activate(helper):
    global _activated, _snapshot_prepare
    mode=os.environ.get('WOTR_ACTION_ENCODING','python').strip().lower()
    require(mode in ('python','native'),'WOTR_ACTION_ENCODING must be python or native')
    if mode=='python':
        require(_activated is None,'cannot switch an activated process to Python')
        return dict(backend='python',activated=False)
    try:
        source=json.loads((HERE/'action_encoding_source.json').read_text(encoding='utf-8'))
        manifest=json.loads((HERE/'action_encoding_build.json').read_text(encoding='utf-8'))
        require(source['schema']=='wotr_action_encoding_source_v1','source schema differs')
        require(manifest['schema']=='wotr_action_encoding_build_v1','build schema differs')
        require(manifest['abi']==abi(),'ABI/NumPy version differs; rebuild locally')
        require(manifest['source_manifest_sha256']==sha(HERE/'action_encoding_source.json'),'source binding changed')
        require(source['helper_sha256']==sha(helper.__file__),'Python implementation changed')
        require(source['pyx_sha256']==sha(HERE/'wotr_action_encoding.pyx'),'Cython implementation changed')
        require(manifest['ai_functions_sha256']==source['ai_functions_sha256'],'AI function binding changed')
        filename='wotr_action_encoding'+abi()['extension_suffix']
        require(manifest['binary']==filename,'binary filename differs')
        binary=HERE/filename
        require(manifest['binary_sha256']==sha(binary),'binary hash differs')
        if _activated is not None:
            owner,checked,diagnostics=_activated
            require(owner is helper and helper.encode_army_batch is checked,'activated binding was replaced')
            return dict(diagnostics)
        require('wotr_action_encoding' not in sys.modules,'extension loaded outside its backend')
        spec=importlib.util.spec_from_file_location('wotr_action_encoding',binary)
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        bound=[]
        def checked(ai,actions,state,side,comp_memo,original):
            if not bound:
                verify_ai_binding(ai,manifest,source)
                native.bind(ai)
                for name in ('_action_cache_key','_encode_action_cached','_region_index_cached'):
                    setattr(ai,name,getattr(native,name))
                ai._current_action_keys=native.current_action_keys
                bound.append(ai)
            else:require(bound[0] is ai,'native backend is bound to another AI module')
            return native.encode_army_batch(ai,actions,state,side,comp_memo,original)
        diagnostics=dict(backend='native',activated=True,binary=str(binary),binary_sha256=manifest['binary_sha256'],
                         ai_source_sha256=manifest['ai_source_sha256'],source_manifest_sha256=manifest['source_manifest_sha256'],abi=abi())
        sys.modules['wotr_action_encoding']=native
        _snapshot_prepare=native.prepare_snapshot
        helper.encode_army_batch=checked
        _activated=(helper,checked,diagnostics)
        return dict(diagnostics)
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        raise RuntimeError('Native action encoding unavailable: '+str(exc)) from exc


def prepare_snapshot(action):
    """Use only the already verified native module; Python fallback otherwise."""
    if _snapshot_prepare is None:
        return None
    return _snapshot_prepare(action)
