"""Experimental explicit Hunt-plan ABI with immutable source probability tables."""
import ctypes as C
from contextlib import ExitStack
import threading
import snapshot,random_stream,observation_tables,combat_context
from chance_tables import values
_tables={};_guard=threading.RLock()

def chance_table(ai,lib):
    with _guard:
        if lib not in _tables:_tables[lib]=snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai,values()))
        return _tables[lib]

def apply(parent,ai,rng,side,menu,index,plan,resume=True):
    import os
    combat_context.require_native_bid_context()
    if os.environ.get('MORDOR_STOP_KEEPS_STEP'):raise ValueError('Native actions require standard Mordor stop behavior')
    if type(parent) is not snapshot.NativeSnapshot:raise ValueError('Owned NativeSnapshot required')
    if type(rng) is not random_stream.NativeRandom:raise ValueError('Owned NativeRandom required')
    if rng._lib is not parent._lib:raise ValueError('State and RNG must use one library')
    if type(side) is not int or side not in (0,1):raise ValueError('Side must be FP or SH')
    if type(index) is not int or not 0<=index<2**63:raise ValueError('Invalid action row')
    if type(resume) is not bool:raise ValueError('Preaction flag must be Boolean')
    if type(menu) is not snapshot.NativeSnapshot or menu._lib is not parent._lib:raise ValueError('State and menu must use one library')
    plan=tuple(plan)
    if len(plan)>4096 or any(type(v) is not int or not -2**63<=v<2**63 for v in plan):raise ValueError('Invalid Hunt option-index plan')
    data=(C.c_int64*len(plan))(*plan);lib=parent._lib;chance=chance_table(ai,lib);tables=observation_tables.native_tables(ai)
    if snapshot.library() is not lib:raise ValueError('State and cached rule tables must use one library')
    fn=lib.wotr_snapshot_apply_descriptor_hunt_plan
    fn.argtypes=[C.c_void_p]*5+[C.POINTER(C.c_int64),C.c_size_t,C.c_int32,C.c_int64,C.c_int32]+[C.POINTER(C.c_void_p)]*4+[C.POINTER(C.c_double)];fn.restype=C.c_int
    child_ptr,random_ptr,events_ptr,request_ptr=C.c_void_p(),C.c_void_p(),C.c_void_p(),C.c_void_p();probability=C.c_double()
    with ExitStack() as stack:
        for owner in sorted({id(x):x for x in (parent,menu,chance)}.values(),key=id):stack.enter_context(owner._lock)
        stack.enter_context(rng._lock)
        parent._require_open();menu._require_open();chance._require_open()
        if not rng._pointer:raise RuntimeError('Native random stream is closed')
        status=fn(parent._pointer,tables._pointer,menu._pointer,rng._pointer,chance._pointer,data,len(plan),side,index,resume,C.byref(child_ptr),C.byref(random_ptr),C.byref(events_ptr),C.byref(request_ptr),C.byref(probability))
    if status<0:raise RuntimeError(snapshot._error(lib))
    if status==1:
        with snapshot.NativeSnapshot(lib,request_ptr.value) as request:payload=request.unpack_metadata(ai)
        return dict(fork=True,probability=probability.value,request=payload)
    if status!=0:raise RuntimeError('Unknown native Hunt status')
    child=snapshot.NativeSnapshot(lib,child_ptr.value);advanced=random_stream.NativeRandom._adopt(rng._lib,random_ptr.value)
    try:
        with snapshot.NativeSnapshot(lib,events_ptr.value) as events:trace=events.unpack_metadata(ai)
        return dict(fork=False,probability=probability.value,child=child,random=advanced,events=trace)
    except BaseException:child.close();advanced.close();raise
