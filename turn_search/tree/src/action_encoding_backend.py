"""Explicit optional native action encoding; Python default, no runtime builds."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,os,platform,sys,sysconfig
import numpy as np
HERE=Path(__file__).resolve().parent
_activated=None
_snapshot_prepare=None
_bind=None

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def abi():
    return dict(implementation=sys.implementation.name,cache_tag=sys.implementation.cache_tag,
                python_version=list(sys.version_info[:3]),system=platform.system(),machine=platform.machine(),
                platform=sysconfig.get_platform(),soabi=sysconfig.get_config_var('SOABI'),
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),numpy_version=np.__version__)
def require(condition,message):
    if not condition:raise RuntimeError('Native action encoding unavailable: '+message)

def _v53_binding(ai):
    """R3-V53-FOLLOWONS: an exact encoding-generation-v53 engine (USER D5
    2026-10-22): ACT_GEN v53, the complete v52 157 row as its prefix and
    ACTION_ENCODING_DIM == 157 + encoding_v53.ACTION_TAIL_DIM."""
    if getattr(ai,'ACT_GEN',None)!='v53' or not getattr(ai,'_ACT_V53_ON',False):return False
    import encoding_v53
    return ai._V53_ACTION_TAIL_START==157 and ai.ACTION_ENCODING_DIM==157+encoding_v53.ACTION_TAIL_DIM

def verify_ai_binding(ai,manifest,source):
    require(sha(ai.__file__)==manifest['ai_source_sha256'],'AI source changed; rebuild before use')
    # R2-TR-NATIVE-EXTENSIONS: the same v52 widths ai._encode_actions_batch_ordinary
    # dispatches here (133 = retired action_schema v6, 157 = v7 cargo lanes); the
    # native fill reads its column offsets from ai at run time.
    # R3-V53-FOLLOWONS: or generation v53 = the v52 157 row ++ the encoding_v53
    # tail (199); under v53 only the generation-agnostic per-row helpers bind
    # (_v53_binding, activate.bind), never the v52 key copy or the v52 army batch.
    require((ai.ACT_GEN=='v52' and ai.ACTION_ENCODING_DIM in (133,157)) or _v53_binding(ai),
            'requires exact v52 action dimensions (or the exact v53 width)')
    for name,expected in source['ai_functions_sha256'].items():
        require(hashlib.sha256(inspect.getsource(getattr(ai,name)).encode()).hexdigest()==expected,
                'runtime AI dependency changed: '+name)

def bind_active(ai):
    """R3-V53-FOLLOWONS: the v53 batch path (search_action_encoding_v53)
    never calls the compiled army batch; it binds the verified per-row
    helpers through here (same verification as the v52 hook).  No-op while
    the Python backend is selected."""
    if _bind is not None:
        _bind(ai)

def activate(helper):
    global _activated, _snapshot_prepare, _bind
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
        def bind(ai):
            if not bound:
                verify_ai_binding(ai,manifest,source)
                native.bind(ai)
                # R3-V53-FOLLOWONS: the compiled _action_cache_key is the v52 key. Under v53 ai keeps
                # its own key (it carries the encoding_v53 roster component); the compiled
                # _encode_action_cached reads it through the bound module globals.
                names=(('_encode_action_cached','_region_index_cached') if _v53_binding(ai)
                       else ('_action_cache_key','_encode_action_cached','_region_index_cached'))
                for name in names:
                    setattr(ai,name,getattr(native,name))
                ai._current_action_keys=native.current_action_keys
                bound.append(ai)
            else:require(bound[0] is ai,'native backend is bound to another AI module')
        def checked(ai,actions,state,side,comp_memo,original):
            bind(ai)
            # the compiled army batch fills v52 columns only; v53 menus take
            # search_action_encoding_v53 (ai._encode_actions_batch_ordinary)
            require(not _v53_binding(ai),'the compiled army batch serves v52 only; v53 uses search_action_encoding_v53')
            return native.encode_army_batch(ai,actions,state,side,comp_memo,original)
        diagnostics=dict(backend='native',activated=True,binary=str(binary),binary_sha256=manifest['binary_sha256'],
                         ai_source_sha256=manifest['ai_source_sha256'],source_manifest_sha256=manifest['source_manifest_sha256'],abi=abi())
        sys.modules['wotr_action_encoding']=native
        _snapshot_prepare=native.prepare_snapshot
        helper.encode_army_batch=checked
        _activated=(helper,checked,diagnostics)
        _bind=bind
        return dict(diagnostics)
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        raise RuntimeError('Native action encoding unavailable: '+str(exc)) from exc


def prepare_snapshot(action):
    """Use only the already verified native module; Python fallback otherwise."""
    if _snapshot_prepare is None:
        return None
    return _snapshot_prepare(action)
