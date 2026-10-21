"""Isolated resident query wrapper; no Python gameplay/search callbacks."""
import ctypes as C
import snapshot
def query(ai,state,root=None):
    if root is not None and (type(root) is not int or root not in (0,1)):raise ValueError('Expected None/0/1 root side')
    lib=state._lib;fn=lib.wotr_snapshot_search_metadata;fn.argtypes=[C.c_void_p,C.c_int32];fn.restype=C.c_void_p
    with state._lock:
        state._require_open()
        result=snapshot.NativeSnapshot(lib,fn(state._pointer,-1 if root is None else root))
    try:return result.unpack_metadata(ai)
    finally:result.close()
def source_signatures(payload):
    """Salted Python hashes remain at the compatibility boundary, not in C++."""
    regular,ordered,chance,_metadata=payload
    return (None if regular is None else hash(regular),
            (None if ordered is None else hash(ordered),*chance[1:]))

def blinded_commit_key(metadata, *, active, card, hidden, round_key):
    """Exact ai._blinded_commit_key using explicit overlay inputs and native facts."""
    if not active:return None
    if card is None:
        if hidden:raise RuntimeError('Hidden combat-card commitment has no sampled identity')
        return None
    isdict,committed,nonce,rounds=metadata['commit_context']
    return str(card) if isdict and committed and nonce and rounds==round_key else None
