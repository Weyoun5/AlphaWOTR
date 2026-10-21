"""Cached action fragments preserve the original portable wire identity."""
import enum
import math
import random
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from portable_reply_identity import PolicyKeyEncoder, actions_bytes, world_bytes


def reference_key(key):
    """Pre-cache encoder kept independent of the implementation under test."""
    def blob(raw):return struct.pack('<Q',len(raw))+raw
    def text(value):return value.encode('utf-8','surrogatepass')
    t=type(key)
    if key is None:return b'N'
    if t in (bool,int):
        value=int(key)
        if -(2**63)<=value<2**63:return b'I'+struct.pack('<q',value)
        return b'G'+blob(str(value).encode('ascii'))
    if t is float:
        if math.isfinite(key) and key.is_integer() and -(2**63)<=key<2**63:
            return b'I'+struct.pack('<q',int(key))
        return b'F'+struct.pack('<d',key)
    if t in (str,bytes):return (b'S' if t is str else b'B')+blob(text(key) if t is str else key)
    if isinstance(key,enum.Enum):
        return b'E'+reference_key(t.__module__+'.'+t.__qualname__)+reference_key(key.name)
    if t is tuple:return b'T'+struct.pack('<Q',len(key))+b''.join(reference_key(x) for x in key)
    if t is frozenset:
        parts=sorted(reference_key(x) for x in key)
        return b'Z'+struct.pack('<Q',len(parts))+b''.join(parts)
    raise ValueError('Unsupported frozen action key type: '+t.__module__+'.'+t.__qualname__)


def reference_actions(keys):
    keys=tuple(keys)
    parts=[reference_key(key) for key in keys]
    return b'RGK1A'+struct.pack('<Q',len(keys))+b''.join(struct.pack('<Q',len(p))+p for p in parts)


class Number(enum.IntEnum):
    ONE=1
    ALIAS=1


class OtherNumber(enum.IntEnum):
    ONE=1


class Text(str,enum.Enum):
    ONE='one'


class DynamicName(enum.Enum):
    ONE=1

    @property
    def name(self):return ('custom',1)


class PolicyKeyEncoderTests(unittest.TestCase):
    def values(self):
        nans=[struct.unpack('<d',struct.pack('<Q',bits))[0]
              for bits in (0x7ff8000000000001,0x7ff8000000000002,0xfff8000000000001)]
        return [None,False,True,0,1,-1,-2**63,2**63-1,2**63,2**100,
                0.,-0.,1.,-1.,0.5,-0.5,float(2**63),float(-2**63),
                float('inf'),float('-inf'),*nans,'','one','\ud800\x00\u2603',
                b'',b'one',b'\xff\x00',Number.ONE,OtherNumber.ONE,Text.ONE,
                DynamicName.ONE,(),frozenset()]

    def test_all_scalars_and_equal_but_wire_distinct_nested_keys(self):
        encoder=PolicyKeyEncoder()
        for value in self.values()*3:
            for key in (value,(value,),('outer',(value,)),frozenset((('nested',value),))):
                with self.subTest(key=key):self.assertEqual(encoder.encode(key),reference_key(key))
        for left,right in ((Number.ONE,1),(Number.ONE,OtherNumber.ONE),
                           (Text.ONE,'one'),(2**63,float(2**63))):
            self.assertEqual(left,right)
            self.assertNotEqual(encoder.encode((left,)),encoder.encode((right,)))

    def test_seeded_nested_keys_and_ordered_duplicate_menus(self):
        rng=random.Random(1910)
        atoms=self.values()
        def make(depth):
            if not depth or rng.randrange(3)==0:return rng.choice(atoms)
            parts=[make(depth-1) for _ in range(rng.randrange(5))]
            return tuple(parts) if rng.randrange(2) else frozenset(parts)
        keys=[make(3) for _ in range(500)]
        encoder=PolicyKeyEncoder()
        for menu in (keys,keys[::-1],keys[:20]*4,[],[keys[1]],keys):
            self.assertEqual(actions_bytes(iter(menu),encoder=encoder),reference_actions(menu))
            self.assertEqual(actions_bytes(menu),reference_actions(menu))

    def test_enum_name_changes_are_read_afresh(self):
        encoder=PolicyKeyEncoder()
        key=('action',(Number.ONE,))
        before=encoder.encode(key)
        original=Number.ONE._name_
        try:
            Number.ONE._name_='CHANGED'
            self.assertEqual(encoder.encode(key),reference_key(key))
            self.assertNotEqual(before,encoder.encode(key))
        finally:Number.ONE._name_=original
        self.assertEqual(before,encoder.encode(key))

    def test_eviction_disabled_cache_and_oversized_fragments_are_exact(self):
        for limit,budget in ((2,1024),(100,300),(0,0),(3,0)):
            encoder=PolicyKeyEncoder(limit,budget)
            for i in range(50):
                key=('action',i,('small','x'*2000))
                self.assertEqual(encoder.encode(key),reference_key(key))
                self.assertLessEqual(len(encoder.cache),limit)
                self.assertLessEqual(encoder.cache_bytes,budget)

    def test_unsupported_mutable_values_still_raise(self):
        encoder=PolicyKeyEncoder()
        for key in ([],{},set(),object(),('action',[])):
            for encode in (encoder.encode,reference_key):
                with self.assertRaises(ValueError):encode(key)

    def test_world_aliases_container_order_and_values_remain_distinct(self):
        shared=[1]
        alias=[shared,shared]
        independent=[[1],[1]]
        self.assertNotEqual(world_bytes(alias),world_bytes(independent))
        self.assertNotEqual(world_bytes({'a':1,'b':2}),world_bytes({'b':2,'a':1}))
        self.assertNotEqual(world_bytes([1,True]),world_bytes([True,1]))

    def test_seeded_reply_choices_cache_and_rng_match_reference(self):
        import own_search
        keys=[('action',i,(('value',i),)) for i in range(7)]
        state=SimpleNamespace(payload=['public'])
        world=SimpleNamespace(state=state,weight=1.)
        def run(cached):
            seen=[]
            context=own_search.SearchContext(SimpleNamespace(),'root',None,
                seed=1910,reply_panels=3,belief_worlds=2)
            def panel(ai,worlds,key,**kwargs):
                seen.append((key,kwargs['seed']))
                return [key[1]/10.],[1.]
            encoder=actions_bytes if cached else lambda values,**kw:reference_actions(values)
            with patch.object(own_search,'sample_worlds',return_value=[world]), \
                 patch.object(own_search,'menu',return_value=(dict.fromkeys(keys),None)), \
                 patch.object(own_search,'evaluate_panel',side_effect=panel), \
                 patch.object(own_search,'actions_bytes',side_effect=encoder):
                chosen=[context.choose(state,'opponent') for _ in range(12)]
            return chosen,seen,dict(context.stats),context.stream.getstate(),tuple(context.cache)
        self.assertEqual(run(False),run(True))


if __name__=='__main__':unittest.main(verbosity=2)
