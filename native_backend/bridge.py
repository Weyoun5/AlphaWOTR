"""Thin boundary for the first native state slice; no implicit Python fallback.

Packing still reads Python GameState once. Native encoding/cloning thereafter
use C++ owned storage and make no Python callbacks. This is NOT yet a complete
GameState, legal-action generator, transition engine, or search implementation.
"""
from pathlib import Path
import ctypes as C
import hashlib
import json
import os
import sys
import threading
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
_LIB=None
_LIB_LOCK=threading.Lock()

class Unit(C.Structure):_fields_=[('region',C.c_int32),('nation',C.c_int32),('kind',C.c_int32)]
class Leader(C.Structure):_fields_=[('region',C.c_int32),('side',C.c_int32),('named',C.c_int32),('nazgul',C.c_int32)]
class Region(C.Structure):_fields_=[('occupied',C.c_int32),('besieged',C.c_int32),('controlled_fp',C.c_int32),('controlled_sh',C.c_int32),('fellowship',C.c_int32)]
class Nation(C.Structure):_fields_=[('side',C.c_int32),('box',C.c_int32),('active',C.c_int32)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_reference():
    reference_path=Path(os.environ.get('WOTR_NATIVE_SEARCH_REFERENCE') or (HERE/'reference_sep16_own_current_v3_r2.json'))
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    from rule_data import reference_rule_data
    reference_rule_data(reference)
    source=Path(reference['runtime'])/'src'
    for name,expected in reference['source_sha256'].items():
        if sha(source/name)!=expected:raise RuntimeError('Reference source drift: '+name)
    if 'ai' in sys.modules:
        if Path(sys.modules['ai'].__file__).resolve()!=source/'ai.py':
            raise RuntimeError('Different engine already imported; use a fresh process')
        return sys.modules['ai']
    for name in list(os.environ):
        if name.startswith('WOTR_'):del os.environ[name]
    os.environ.update(reference['environment'])
    sys.path.insert(0,str(source));sys.dont_write_bytecode=True
    import ai
    return ai

def library():
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:return _LIB
        manifest=json.loads((HERE/'build/manifest.json').read_text(encoding='utf-8'))
        if manifest.get('passed') is not True:raise RuntimeError('No successful native build')
        for name,expected in manifest['sources'].items():
            if sha(HERE/name)!=expected:raise RuntimeError('Native source changed; rebuild required: '+name)
        path=(HERE/manifest['binary_relative']).resolve() if manifest.get('binary_relative') else Path(manifest['binary']).resolve()
        if not path.is_relative_to((HERE/'build').resolve()):raise RuntimeError('Binary escaped isolated build folder')
        if sha(path)!=manifest['binary_sha256']:raise RuntimeError('Native binary drift')
        lib=C.CDLL(str(path))
        lib.wotr_abi_version.argtypes=[];lib.wotr_abi_version.restype=C.c_uint32
        lib.wotr_last_error.argtypes=[];lib.wotr_last_error.restype=C.c_char_p
        if lib.wotr_abi_version()!=1:raise RuntimeError('Native ABI mismatch')
        lib.wotr_position_create.argtypes=[C.POINTER(Region),C.c_size_t,C.POINTER(Nation),C.c_size_t,C.POINTER(Unit),C.c_size_t,C.POINTER(Leader),C.c_size_t]
        lib.wotr_position_create.restype=C.c_void_p
        lib.wotr_position_clone.argtypes=[C.c_void_p];lib.wotr_position_clone.restype=C.c_void_p
        lib.wotr_position_destroy.argtypes=[C.c_void_p];lib.wotr_position_destroy.restype=None
        lib.wotr_observation_size.argtypes=[C.c_void_p];lib.wotr_observation_size.restype=C.c_size_t
        lib.wotr_encode_regions_politics.argtypes=[C.c_void_p,C.c_int32,C.POINTER(C.c_float),C.c_size_t]
        lib.wotr_encode_regions_politics.restype=C.c_int
        lib.wotr_set_political_box.argtypes=[C.c_void_p,C.c_size_t,C.c_int32];lib.wotr_set_political_box.restype=C.c_int
        _LIB=lib
        return lib

def native_error(lib):return RuntimeError((lib.wotr_last_error() or b'Native operation failed').decode('utf-8','replace'))

class NativePosition:
    """Owns an opaque native regional/political slice; explicitly not a full game."""
    full_game=False
    def __init__(self,lib,pointer):
        if not pointer:raise native_error(lib)
        self._lib=lib;self._pointer=pointer;self._lock=threading.RLock()
        self.size=int(lib.wotr_observation_size(pointer))
    def _require_open(self):
        if not self._pointer:raise RuntimeError('Native position is closed')
    def encode_into(self,side,output):
        if type(side) is not int or side not in (0,1):raise ValueError('Side must be integer 0=FP or 1=SH')
        if not isinstance(output,np.ndarray) or output.dtype!=np.dtype(np.float32) or output.shape!=(self.size,) or not output.flags.c_contiguous or not output.flags.writeable or not output.flags.aligned:
            raise ValueError('Output must be writable aligned contiguous float32 with exact size')
        with self._lock:
            self._require_open()
            if self._lib.wotr_encode_regions_politics(self._pointer,side,output.ctypes.data_as(C.POINTER(C.c_float)),output.size):raise native_error(self._lib)
        return output
    def encode(self,side):return self.encode_into(side,np.empty(self.size,dtype=np.float32))
    def clone(self):
        with self._lock:
            self._require_open();return NativePosition(self._lib,self._lib.wotr_position_clone(self._pointer))
    def set_political_box(self,nation,box):
        if type(nation) is not int or not 0<=nation<(1<<(8*C.sizeof(C.c_size_t))) or type(box) is not int or not 0<=box<=3:raise ValueError('Invalid political mutation primitive')
        with self._lock:
            self._require_open()
            if self._lib.wotr_set_political_box(self._pointer,nation,box):raise native_error(self._lib)
    def close(self):
        with self._lock:
            if self._pointer:
                self._lib.wotr_position_destroy(self._pointer);self._pointer=None
    def __enter__(self):
        self._require_open();return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        if getattr(self,'_pointer',None):self.close()

def pack_position(ai,state):
    """Copy only observation-relevant regional/political inputs into owned C++.

    Other state, companion abilities, card decisions and RNG are deliberately
    not represented yet. Callers must not use this slice for game transitions.
    """
    nations=list(ai.ALL_NATIONS);nation_index={n:i for i,n in enumerate(nations)}
    def side_id(nation):
        if nation not in nation_index:raise ValueError('Unsupported nation in native slice')
        return 0 if ai._side_of_nation(nation)==ai.Side.FREE_PEOPLES else 1
    armies={ai.sanitize(key):army for key,army in state.armies.items()}
    try:fellowship=ai.game_logic.fellowship_region(state)
    except Exception:fellowship=None  # Exact original prefix semantics.
    fp=state.enemy_controlled.get(ai.Side.FREE_PEOPLES,set())
    sh=state.enemy_controlled.get(ai.Side.SHADOW,set())
    siege=getattr(state,'besieged_strongholds',set())
    regions=[];units=[];leaders=[];politics=[]
    for i,key in enumerate(ai.ALL_REGION_KEYS):
        army=armies.get(key)
        regions.append(Region(bool(army),key in siege,key in fp,key in sh,key==fellowship))
        if not army:continue
        for unit in army.units:
            if unit.nation not in nation_index:raise ValueError('Unsupported unit nation')
            if unit.kind is ai.UnitKind.REGULAR:kind=0
            elif unit.kind is ai.UnitKind.ELITE:kind=1
            else:raise ValueError('Unsupported unit kind')
            units.append(Unit(i,nation_index[unit.nation],kind))
        for leader in army.leaders:
            name=getattr(leader,'name','')
            if name in ('Witch King','Saruman','Mouth of Sauron'):side,named,nazgul=1,1,0
            elif isinstance(leader,ai.Companion):side,named,nazgul=0,1,0
            elif isinstance(leader,ai.Nazgul):side,named,nazgul=1,0,1
            elif type(leader).__name__=='Minion':side,named,nazgul=1,1,0
            elif hasattr(leader,'nation'):side,named,nazgul=side_id(leader.nation),0,0
            else:continue  # Original observation ignores unclassified leaders.
            leaders.append(Leader(i,side,named,nazgul))
    for nation in nations:
        box=state.political.boxes.get(nation,0)
        if type(box) is not int or not 0<=box<=3:raise ValueError('Native political boxes currently require integers 0..3')
        politics.append(Nation(side_id(nation),box,bool(state.political.active.get(nation,False))))
    ra=(Region*len(regions))(*regions);na=(Nation*len(politics))(*politics)
    ua=(Unit*len(units))(*units);la=(Leader*len(leaders))(*leaders)
    lib=library()
    return NativePosition(lib,lib.wotr_position_create(ra,len(ra),na,len(na),ua,len(ua),la,len(la)))
