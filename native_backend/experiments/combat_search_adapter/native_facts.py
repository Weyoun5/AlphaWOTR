"""Small combat-search projections; these never reconstruct a GameState."""
import ctypes as C
import snapshot

def query(ai,state):
    lib=state._lib;fn=lib.wotr_snapshot_combat_search_facts
    fn.argtypes=[C.c_void_p];fn.restype=C.c_void_p
    with state._lock:
        state._require_open()
        with snapshot.NativeSnapshot(lib,fn(state._pointer)) as payload:
            return payload.unpack_metadata(ai)

def units(ai,state,region,side,kind=None,pc=None):
    lib=state._lib
    if snapshot.library() is not lib:raise ValueError('Combat query inputs must use one native library')
    fn=lib.wotr_snapshot_combat_search_units
    fn.argtypes=[C.c_void_p,C.c_void_p];fn.restype=C.c_void_p
    # The source sanitizes str(region) when partitioning and selecting eligibility.
    options=dict(region=str(region),side=side,kind=kind,pc=pc)
    with snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai,options)) as args,state._lock:
        state._require_open()
        with snapshot.NativeSnapshot(lib,fn(state._pointer,args._pointer)) as payload:
            return payload.unpack_metadata(ai)
