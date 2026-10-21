"""C unpickler for internally generated native data; never a user pickle loader.

The native writer emits no executable/class opcodes. The reader additionally
refuses class imports and accepts only registered enum persistent tokens.
"""
import ctypes as C
import io,pickle
import snapshot,snapshot_schema

class DataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError('Executable/class pickle opcode in native metadata')
    def persistent_load(self,token):
        if type(token) is not tuple or len(token)!=3 or token[0]!='enum':raise ValueError('Invalid native enum token')
        _,name,member=token
        if name not in snapshot_schema.ENUM_TYPES:raise ValueError('Unregistered native enum token')
        # Preserve canonical source singleton identity and class validation.
        return snapshot._classes()[name][member]

def decode(data):
    stream=io.BytesIO(data);value=DataUnpickler(stream).load()
    if stream.read(1):raise ValueError('Trailing native metadata bytes')
    return value

def payload(ai,state,root=None):
    if root is not None and (type(root) is not int or root not in (0,1)):raise ValueError('Expected None/FP/Shadow root')
    lib=state._lib;fn=lib.wotr_snapshot_chance_signature_wire;fn.argtypes=[C.c_void_p,C.c_int32];fn.restype=C.c_void_p
    with state._lock:
        state._require_open()
        with snapshot.NativeSnapshot(lib,fn(state._pointer,-1 if root is None else root)) as result:data=result.unpack_metadata(ai)
    return decode(data)

def query(ai,state,root=None):
    value=payload(ai,state,root)
    return (None if value[0] is None else hash(value[0]),*value[1:])

def component(ai,state):
    lib=state._lib;fn=lib.wotr_snapshot_metadata_wire;fn.argtypes=[C.c_void_p];fn.restype=C.c_void_p
    with state._lock:
        state._require_open()
        with snapshot.NativeSnapshot(lib,fn(state._pointer)) as result:data=result.unpack_metadata(ai)
    return data,decode(data)
