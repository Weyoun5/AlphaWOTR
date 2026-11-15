"""Explicit CPython 3.11 Random-state boundary; no implicit global RNG swaps.

Each handle owns one stream. Create separate handles from rng_streams' game
and search Python states. NumPy state and seeding contracts stay external.
Unsupported generic Python inputs are rejected before drawing randomness.
"""
import ctypes as C
import itertools
import math
import struct
import threading
import bridge

LIMIT=1048576
_configured=False
_configure_lock=threading.Lock()


def library():
    global _configured
    lib=bridge.library()
    with _configure_lock:
        if _configured:return lib
        P=C.c_void_p;U=C.c_uint32;Q=C.c_uint64;I=C.c_int32;L=C.c_int64;N=C.c_size_t
        signatures={
            'last_error':([],C.c_char_p),
            'create':([C.POINTER(U),N,U,I,Q],P),'clone':([P],P),'destroy':([P],None),
            'export':([P,C.POINTER(U),N,C.POINTER(U),C.POINTER(I),C.POINTER(Q)],C.c_int),
            'setstate':([P,C.POINTER(U),N,U,I,Q],C.c_int),
            'double':([P,C.POINTER(C.c_double)],C.c_int),
            'getrandbits':([P,U,C.POINTER(U),N],C.c_int),
            'randbelow':([P,Q,C.POINTER(Q)],C.c_int),
            'randint':([P,L,L,C.POINTER(L)],C.c_int),
            'd6':([P,C.POINTER(U),N],C.c_int),
            'shuffle':([P,C.POINTER(Q),N],C.c_int),
            'sample':([P,Q,N,C.POINTER(Q),N],C.c_int),
            'choices':([P,Q,C.POINTER(C.c_double),N,N,C.POINTER(Q),N],C.c_int),
        }
        for name,(arguments,result) in signatures.items():
            function=getattr(lib,'wotr_random_'+name);function.argtypes=arguments;function.restype=result
        _configured=True
    return lib


def _integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high:raise ValueError(name+' outside supported integer range')
    return value


def _state(state):
    if type(state) is not tuple or len(state)!=3 or type(state[0]) is not int or state[0]!=3:
        raise ValueError('Only CPython Random version-3 state tuples are supported')
    inner=state[1]
    if type(inner) is not tuple or len(inner)!=625:raise ValueError('Random state requires 624 words and index')
    for word in inner[:624]:_integer(word,'state word',0,2**32-1)
    index=_integer(inner[-1],'state index',0,624)
    gauss=state[2]
    if gauss is not None and type(gauss) is not float:raise ValueError('Gaussian cache must be None or float')
    bits=0 if gauss is None else struct.unpack('=Q',struct.pack('=d',gauss))[0]
    return (C.c_uint32*624)(*inner[:624]),624,index,int(gauss is not None),bits


def _sequence(value):
    if type(value) is not list and type(value) is not tuple and type(value) is not range:
        raise ValueError('Only literal list, tuple and range populations are supported')
    return len(value)


class NativeRandom:
    def __init__(self,state):
        args=_state(state);self._lib=library();self._lock=threading.RLock()
        self._pointer=self._lib.wotr_random_create(*args)
        if not self._pointer:raise self._error()

    @classmethod
    def _adopt(cls,lib,pointer):
        """Take exclusive ownership of an already-created native RNG handle."""
        result=object.__new__(cls);result._lib=lib;result._lock=threading.RLock();result._pointer=pointer
        if not pointer:raise result._error()
        return result

    def _error(self):return RuntimeError((self._lib.wotr_random_last_error() or b'Native random failure').decode('utf8','replace'))
    def _call(self,name,*args):
        with self._lock:
            if not self._pointer:raise RuntimeError('Native random stream is closed')
            if getattr(self._lib,'wotr_random_'+name)(self._pointer,*args):raise self._error()
    def close(self):
        with self._lock:
            if self._pointer:self._lib.wotr_random_destroy(self._pointer);self._pointer=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
    def __del__(self):
        if getattr(self,'_pointer',None):self.close()

    def getstate(self):
        words=(C.c_uint32*624)();index=C.c_uint32();has=C.c_int32();bits=C.c_uint64()
        self._call('export',words,624,C.byref(index),C.byref(has),C.byref(bits))
        gauss=None if not has.value else struct.unpack('=d',struct.pack('=Q',bits.value))[0]
        return 3,tuple(words)+(index.value,),gauss
    def setstate(self,state):self._call('setstate',*_state(state))
    def clone(self):
        with self._lock:
            if not self._pointer:raise RuntimeError('Native random stream is closed')
            return type(self)._adopt(self._lib,self._lib.wotr_random_clone(self._pointer))
    def random(self):
        result=C.c_double();self._call('double',C.byref(result));return result.value
    def getrandbits(self,bits):
        _integer(bits,'bit count',0,LIMIT);count=(bits+31)//32;words=(C.c_uint32*count)()
        self._call('getrandbits',bits,words,count)
        return int.from_bytes(struct.pack('<'+str(count)+'I',*words),'little')
    def randbelow(self,bound):
        _integer(bound,'randbelow bound',1,2**64-1);result=C.c_uint64()
        self._call('randbelow',bound,C.byref(result));return result.value
    def randint(self,low,high):
        _integer(low,'lower bound',-2**63,2**63-1);_integer(high,'upper bound',-2**63,2**63-1)
        result=C.c_int64();self._call('randint',low,high,C.byref(result));return result.value
    def roll_d6(self,count):
        _integer(count,'dice count',0,LIMIT);result=(C.c_uint32*count)();self._call('d6',result,count);return list(result)
    def choice(self,population):
        count=_sequence(population)
        if not count:raise IndexError('Cannot choose from an empty sequence')
        return population[self.randbelow(count)]
    def shuffle(self,population):
        if type(population) is not list:raise ValueError('Native shuffle requires a literal list')
        count=_integer(len(population),'shuffle count',0,LIMIT);old=population.copy();indices=(C.c_uint64*count)()
        self._call('shuffle',indices,count);population[:]=[old[i] for i in indices]
    def sample(self,population,k,*,counts=None):
        if counts is not None:raise ValueError('Sample counts are not yet supported')
        count=_sequence(population);_integer(k,'sample size',0,min(count,LIMIT));indices=(C.c_uint64*k)()
        self._call('sample',count,k,indices,k);return [population[i] for i in indices]
    def choices(self,population,weights=None,*,cum_weights=None,k=1):
        count=_sequence(population);_integer(k,'choices count',0,LIMIT)
        if weights is not None and cum_weights is not None:raise ValueError('Specify only one weight form')
        raw=weights if weights is not None else cum_weights
        cumulative=None
        if raw is not None:
            _sequence(raw)
            if len(raw)!=count:raise ValueError('Weight count must match population')
            if any(type(v) not in (int,float) or not math.isfinite(v) or (type(v) is int and abs(v)>2**53) for v in raw):
                raise ValueError('Weights require finite literal doubles or exactly representable integers')
            values=list(itertools.accumulate(raw)) if weights is not None else list(raw)
            if any(type(v) is int and abs(v)>2**53 for v in values):raise ValueError('Integer cumulative weights exceed exact double range')
            cumulative=(C.c_double*count)(*values)
        if count==0:
            if cumulative is None and k==0:
                with self._lock:
                    if not self._pointer:raise RuntimeError('Native random stream is closed')
                    return []
            raise ValueError('Empty choices population')
        indices=(C.c_uint64*k)();self._call('choices',count,cumulative,count if cumulative is not None else 0,k,indices,k)
        return [population[i] for i in indices]
