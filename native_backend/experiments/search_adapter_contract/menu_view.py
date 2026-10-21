"""Thin owned-menu query boundary. Does not unpack a GameState."""
import ctypes as C
import snapshot

class MenuQuery:
    def __init__(self,ai):
        self.ai=ai
        self.tables=snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai,dict(
            category=dict(ai._ATYPE_TO_CAT),category_index=dict(ai._ACTION_TYPE_TO_IDX),
            free_types=tuple(sorted(ai._FREE_ACTION_TYPES)))))
        self.lib=self.tables._lib
        self.view=self.lib.wotr_snapshot_search_menu_view
        self.view.argtypes=[C.c_void_p,C.c_void_p];self.view.restype=C.c_void_p
        self.row=self.lib.wotr_snapshot_search_menu_row
        self.row.argtypes=[C.c_void_p,C.c_int64];self.row.restype=C.c_void_p
        self.rows=self.lib.wotr_snapshot_search_menu_rows
        self.rows.argtypes=[C.c_void_p,C.c_void_p];self.rows.restype=C.c_void_p
        self.groups=self.lib.wotr_snapshot_search_menu_alias_groups
        self.groups.argtypes=[C.c_void_p];self.groups.restype=C.c_void_p

    def metadata(self,menu):
        if type(menu) is not snapshot.NativeSnapshot or menu._lib is not self.lib:
            raise ValueError('Menu must share the resident library')
        with menu._lock,self.tables._lock:
            menu._require_open();self.tables._require_open()
            with snapshot.NativeSnapshot(self.lib,self.view(menu._pointer,self.tables._pointer)) as result:
                return result.unpack_metadata(self.ai)

    def descriptor(self,menu,index):
        if type(index) is not int or not -(1<<63)<=index<(1<<63):raise ValueError('Menu row must be int64')
        if type(menu) is not snapshot.NativeSnapshot or menu._lib is not self.lib:
            raise ValueError('Menu must share the resident library')
        with menu._lock:
            menu._require_open()
            with snapshot.NativeSnapshot(self.lib,self.row(menu._pointer,index)) as result:
                return result.unpack_metadata(self.ai)

    def close(self):self.tables.close()

    def alias_groups(self,menu):
        if type(menu) is not snapshot.NativeSnapshot or menu._lib is not self.lib:
            raise ValueError('Menu must share the resident library')
        with menu._lock:
            menu._require_open()
            with snapshot.NativeSnapshot(self.lib,self.groups(menu._pointer)) as result:
                return result.unpack_metadata(self.ai)

    def descriptors(self,menu,indices):
        if type(menu) is not snapshot.NativeSnapshot or menu._lib is not self.lib:
            raise ValueError('Menu must share the resident library')
        if type(indices) is not list or any(type(i) is not int or not -(1<<63)<=i<(1<<63) for i in indices):
            raise ValueError('Menu rows must be list of int64')
        with snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(self.ai,indices)) as selection,menu._lock:
            menu._require_open()
            with snapshot.NativeSnapshot(self.lib,self.rows(menu._pointer,selection._pointer)) as result:
                return result.unpack_metadata(self.ai)

# Proven metadata-only transport; original descriptor and decoder paths remain.
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('menu_wire_codec',ROOT/'experiments/chance_signature_wire/native_query.py')
codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)

class WireMenuQuery(MenuQuery):
    def _decode(self,value):
        fn=self.lib.wotr_snapshot_metadata_wire
        fn.argtypes=[C.c_void_p];fn.restype=C.c_void_p
        try:
            with snapshot.NativeSnapshot(self.lib,fn(value._pointer)) as wire:
                raw=wire.unpack_metadata(self.ai)
        except RuntimeError as error:
            # The projection has already succeeded. Preserve the original
            # decoder for accepted values outside the data-only wire domain.
            if str(error) not in (
                'Chance wire cannot encode object metadata',
                'Chance wire cannot encode external list bindings',
                'Chance wire immutable cycle exceeds depth bound'):
                raise
            return value.unpack_metadata(self.ai)
        return codec.decode(raw)
    def metadata(self,menu):
        if type(menu) is not snapshot.NativeSnapshot or menu._lib is not self.lib:
            raise ValueError('Menu must share the resident library')
        with menu._lock,self.tables._lock:
            menu._require_open();self.tables._require_open()
            with snapshot.NativeSnapshot(self.lib,self.view(menu._pointer,self.tables._pointer)) as result:
                return self._decode(result)
    def alias_groups(self,menu):
        if type(menu) is not snapshot.NativeSnapshot or menu._lib is not self.lib:
            raise ValueError('Menu must share the resident library')
        with menu._lock:
            menu._require_open()
            with snapshot.NativeSnapshot(self.lib,self.groups(menu._pointer)) as result:
                return self._decode(result)

MenuQuery=WireMenuQuery
