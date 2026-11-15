"""Portable reply_graph_key_v1, separate from action semantic equivalence.

Full reachable values, ordered containers and compound aliases; sets use typed
canonical member order. Native transport metadata and primitive interning are
not game state. This helper is not installed in the frozen r4 engine.
"""
import enum
import math
import struct
import weakref

VERSION=b'RGK1'
def _u64(n):
    if type(n) is not int or not 0<=n<2**64:raise ValueError('Identity count must be uint64')
    return struct.pack('<Q',n)
def _blob(value):return _u64(len(value))+value
def _text(value):return value.encode('utf-8','surrogatepass')
def _name(cls):return cls.__module__+'.'+cls.__qualname__
def _scalar(x):
    t=type(x)
    if x is None:return b'N'
    if t is bool:return b'B'+bytes([int(x)])
    if t is int:return b'I'+_blob(str(x).encode('ascii'))
    if t is float:return b'F'+struct.pack('<d',x)
    if t is str:return b'U'+_blob(_text(x))
    if t is bytes:return b'Y'+_blob(x)
    if isinstance(x,enum.Enum):return b'E'+_blob(_text(_name(t)))+_blob(_text(x.name))
    return None
def _sorted_members(values):
    keyed=sorted((_atom(value),i,value) for i,value in enumerate(values))
    for i in range(1,len(keyed)):
        if keyed[i-1][0]==keyed[i][0]:raise ValueError('Ambiguous typed set member order')
    return [value for _,_,value in keyed]
def _atom(x):
    scalar=_scalar(x)
    if scalar is not None:return scalar
    if type(x) is tuple:return b'T'+_u64(len(x))+b''.join(_atom(v) for v in x)
    if type(x) is frozenset:return b'Z'+_u64(len(x))+b''.join(_atom(v) for v in _sorted_members(x))
    raise ValueError('Unsupported typed set member: '+_name(type(x)))

def world_bytes(state):
    output=bytearray(VERSION+b'G');memo={}
    def visit(x):
        scalar=_scalar(x)
        if scalar is not None:output.extend(scalar);return
        identity=id(x)
        if identity in memo:output.extend(b'R'+_u64(memo[identity]));return
        index=len(memo);memo[identity]=index;t=type(x)
        if t in (list,tuple,dict,set,frozenset):
            output.extend({list:b'L',tuple:b'T',dict:b'D',set:b'S',frozenset:b'Z'}[t]+_u64(index)+_u64(len(x)))
            if t is dict:
                for k,v in x.items():visit(k);visit(v)
            else:
                for v in _sorted_members(x) if t in (set,frozenset) else x:visit(v)
            return
        fields=[]
        if hasattr(x,'__dict__'):fields.append(('__dict__',vars(x)))
        for cls in reversed(t.__mro__):
            slots=vars(cls).get('__slots__',());slots=(slots,) if type(slots) is str else slots
            for slot in slots:
                if slot not in ('__dict__','__weakref__') and hasattr(x,slot):fields.append((slot,getattr(x,slot)))
        if not fields:raise ValueError('Unsupported identity object: '+_name(t))
        output.extend(b'O'+_u64(index)+_blob(_text(_name(t)))+_u64(len(fields)))
        for name,value in fields:output.extend(_blob(_text(name)));visit(value)
    visit(state);return bytes(output)

def worlds_bytes(states_and_weights):
    worlds=tuple(states_and_weights);out=bytearray(VERSION+b'W'+_u64(len(worlds)))
    for state,weight in worlds:
        out.extend(struct.pack('<d',float(weight)));out.extend(_blob(world_bytes(state)))
    return bytes(out)

def policy_key_bytes(key):
    """Mirror existing sc::key_bytes on action_policy_key's frozen value."""
    t=type(key)
    if key is None:return b'N'
    if t in (bool,int):
        value=int(key)
        if -(2**63)<=value<2**63:return b'I'+struct.pack('<q',value)
        raw=str(value).encode('ascii');return b'G'+_blob(raw)
    if t is float:
        if math.isfinite(key) and key.is_integer() and -(2**63)<=key<2**63:return b'I'+struct.pack('<q',int(key))
        return b'F'+struct.pack('<d',key)
    if t in (str,bytes):return (b'S' if t is str else b'B')+_blob(_text(key) if t is str else key)
    if isinstance(key,enum.Enum):return b'E'+policy_key_bytes(_name(t))+policy_key_bytes(key.name)
    if t is tuple:return b'T'+_u64(len(key))+b''.join(policy_key_bytes(x) for x in key)
    if t is frozenset:
        parts=sorted(policy_key_bytes(x) for x in key);return b'Z'+_u64(len(parts))+b''.join(parts)
    raise ValueError('Unsupported frozen action key type: '+_name(t))

class PolicyKeyEncoder:
    """Bounded per-search reuse of the exact immutable RGK1 key fragments.

    Compound cache keys contain *encoded* children: Python equality alone is
    insufficient (IntEnum/int and large integral float/int can compare equal
    while carrying different wire identities). No mutable action or world is
    retained. Eviction affects only serialization work, never reply selection.
    """
    def __init__(self, max_entries=32768, max_bytes=8*1024*1024):
        self.max_entries=max_entries;self.max_bytes=max_bytes
        self.cache={};self.cache_bytes=0

    def encode(self,key):
        t=type(key)
        if t is tuple:
            parts=tuple(self.encode(value) for value in key)
            cache_key=(tuple,parts)
        elif t is frozenset:
            parts=tuple(sorted(self.encode(value) for value in key))
            cache_key=(frozenset,parts)
        elif key is None or t in (bool,int,float,str,bytes):
            cache_key=(t,key)
        elif isinstance(key,enum.Enum):
            # Read the class/member names on each visit, including custom enums.
            name=key.name
            if type(name) is not str:return policy_key_bytes(key)
            cache_key=(enum.Enum,_name(t),name)
        else:
            # Preserve the reference serializer's unsupported-type error.
            return policy_key_bytes(key)
        found=self.cache.get(cache_key)
        if found is not None:return found
        if t is tuple:encoded=b'T'+_u64(len(parts))+b''.join(parts)
        elif t is frozenset:encoded=b'Z'+_u64(len(parts))+b''.join(parts)
        else:encoded=policy_key_bytes(key)
        # Bound retained fragments separately from the public reply-cache charge.
        size=len(encoded)+128
        if size<=self.max_bytes and self.max_entries>0:
            if len(self.cache)>=self.max_entries or self.cache_bytes+size>self.max_bytes:
                self.cache.clear();self.cache_bytes=0
            self.cache[cache_key]=encoded;self.cache_bytes+=size
        return encoded


def actions_bytes(semantic_keys, *, encoder=None):
    encode=policy_key_bytes if encoder is None else encoder.encode
    keys=tuple(semantic_keys);return VERSION+b'A'+_u64(len(keys))+b''.join(_blob(encode(k)) for k in keys)

def logical_charge(world_byte_count,action_byte_count,row_count):
    for value in (world_byte_count,action_byte_count,row_count):_u64(value)
    result=world_byte_count+action_byte_count+512*row_count+4096;_u64(result);return result

class NodeTokens:
    """Per-context first-use identities; weak keys never keep tree nodes alive."""
    def __init__(self):self.next_token=1;self.nodes=weakref.WeakKeyDictionary()
    def token(self,node):
        if node is None:return 0
        found=self.nodes.get(node)
        if found is None:
            found=self.next_token;_u64(found);self.next_token+=1;self.nodes[node]=found
        return found
