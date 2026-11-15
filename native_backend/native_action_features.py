"""Batched float32/int64 outputs from retained state/menu handles; no callbacks."""
import ctypes as C
import numpy as np
import snapshot
import observation_tables

def action_tables(ai):
    import action_schema
    if action_schema.ACTION_ENCODING_CONTRACT != "observed_combat_choices_v8":
        raise ValueError("Native action features require observed_combat_choices_v8")
    assert ai.ACT_GEN=='v52' and ai.ACTION_ENCODING_DIM==157
    return _v52_row_tables(ai,action_schema.ACTION_ENCODING_CONTRACT)

def _v52_row_tables(ai,contract):
    """The v52 row tables af_encode reads (unchanged body, unchanged key order)."""
    import combat_round_observation as round_observation
    import game_state
    return dict(schema='wotr_action_features_v52_157',action_encoding_contract=contract,width=157,regions=ai.NUM_REGIONS,
        region_index=dict(ai.REGION_INDEX),category=dict(ai._ATYPE_TO_CAT),
        category_index=dict(ai._ACTION_TYPE_TO_IDX),subtypes=dict(ai._SUBTYPE_TO_IDX),
        characters=dict(ai._ACTION_CHAR_IDX),faces=dict(ai._FACE_ORDER),
        offsets={key:int(value) for key,value in vars(ai).items() if key.startswith('_ACT_') and key.endswith('_OFF')},
        targets=dict(round_observation._TARGET_INDEX),payment_nations=list(round_observation.PAYMENT_NATIONS),
        companion_names=list(round_observation.COMPANIONS),casualty_operations=list(ai._CASUALTY_COST),
        nation_values={nation.name:nation.value for nation in game_state.Nation})

V53_SCHEMA='wotr_action_features_v53_199'
V53_EXPORTS=('wotr_snapshot_encode_actions_v53','wotr_snapshot_prepare_action_tables_v53')

def action_tables_v53(ai):
    """R2-D5-V53-NATIVE (USER D5 2026-10-22): encoding generation v53 = the
    complete v52 row (the untouched native af_encode, whose tables keep the
    v52 schema / v8 prefix contract / width 157) ++ the 42-wide encoding_v53
    action tail.  The appended v53 keys pin the tail layout; the native mirror
    (cpp/encoding_v53.inc) checks every one of them and refuses a mismatch."""
    import action_schema
    import combat_round_observation as round_observation
    import encoding_v53 as V
    if getattr(ai,'ACT_GEN',None)!='v53' or action_schema.ACTION_ENCODING_CONTRACT!=V.ACTION_CONTRACT:
        raise ValueError('Native v53 action features require WOTR_ACT_GEN=v53 (%r)'%(getattr(ai,'ACT_GEN',None),))
    if round_observation.ACTION_CONTRACT!='observed_combat_choices_v8' or ai._V53_ACTION_TAIL_START!=157 or ai.ACTION_ENCODING_DIM!=157+V.ACTION_TAIL_DIM:
        raise ValueError('Native v53 action features require the v52 v8 row of width 157 ++ the v53 tail')
    tables=_v52_row_tables(ai,round_observation.ACTION_CONTRACT)
    tables.update(v53_schema=V53_SCHEMA,v53_action_contract=V.ACTION_CONTRACT,v53_width=ai.ACTION_ENCODING_DIM,
        v53_prefix_width=ai._V53_ACTION_TAIL_START,
        v53_tail=dict(ABILITY=V.ABILITY_OFF,FIGURE=V.FIGURE_OFF,LEADER=V.LEADER_OFF,REGULAR=V.REGULAR_OFF,
                      ELITE=V.ELITE_OFF,VIA=V.VIA_OFF,VIA_DIM=V.VIA_DIM,TAIL_DIM=V.ACTION_TAIL_DIM),
        v53_nations=list(V.NATIONS),v53_characters=list(V.CHARACTERS),v53_leader_norm=V.LEADER_NORM,
        v53_unit_norm=V.UNIT_NORM,v53_tdaan_card=V.TDAAN_CARD)
    return tables

class ActionFeatures:
    def __init__(self,ai):
        self.tables=observation_tables.native_tables(ai)
        self.constants=snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai,action_tables(ai)))
        self.lib=self.constants._lib
        prepare=getattr(self.lib,'wotr_snapshot_prepare_action_tables_v52',None)
        if prepare is not None:
            prepare.argtypes=[C.c_void_p];prepare.restype=C.c_int
            if prepare(self.constants._pointer):raise snapshot._error(self.lib)
        self.constants=observation_tables._ImmutableTables(self.constants)
        self.function=self.lib.wotr_snapshot_encode_actions_v52
        self.function.argtypes=[C.c_void_p]*4+[C.c_int32,C.POINTER(C.c_float),C.c_size_t,C.POINTER(C.c_int64),C.c_size_t]
        self.function.restype=C.c_int

    def encode(self,state,menu,side,rows):
        for obj in (state,menu):
            if type(obj) is not snapshot.NativeSnapshot or obj._lib is not self.lib:raise ValueError('Action inputs must share the same resident library')
        features=np.empty((rows,157),dtype=np.float32)
        regions=np.empty(rows,dtype=np.int64)
        first,second=sorted((state,menu),key=id)
        with first._lock,second._lock:
            state._require_open();menu._require_open()
            if self.function(state._pointer,self.tables._pointer,menu._pointer,self.constants._pointer,side,features.ctypes.data_as(C.POINTER(C.c_float)),features.size,regions.ctypes.data_as(C.POINTER(C.c_int64)),rows):raise snapshot._error(self.lib)
        return features,regions

class ActionFeaturesV53(ActionFeatures):
    """R2-D5-V53-NATIVE: (rows, 199) float32 = the native v52 row ++ the
    encoding_v53 tail, and the v52 primary region per row, from retained
    state/menu handles; no callbacks.  Refuses loudly on a v52 engine and on
    a resident DLL built before the v53 exports existed (never a 157 row)."""
    def __init__(self,ai):
        self.tables=observation_tables.native_tables(ai)
        data=action_tables_v53(ai)
        self.width=int(data['v53_width'])
        owner=snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai,data))
        self.lib=owner._lib
        absent=[name for name in V53_EXPORTS if not hasattr(self.lib,name)]
        if absent:
            owner.close()
            raise RuntimeError('Resident native library lacks the v53 exports %s; rebuild the DLL (R2-D5-V53-NATIVE)'%(absent,))
        prepare=self.lib.wotr_snapshot_prepare_action_tables_v53
        prepare.argtypes=[C.c_void_p];prepare.restype=C.c_int
        if prepare(owner._pointer):
            error=snapshot._error(self.lib);owner.close();raise error
        self.constants=observation_tables._ImmutableTables(owner)
        self.function=self.lib.wotr_snapshot_encode_actions_v53
        self.function.argtypes=[C.c_void_p]*4+[C.c_int32,C.POINTER(C.c_float),C.c_size_t,C.POINTER(C.c_int64),C.c_size_t]
        self.function.restype=C.c_int

    def encode(self,state,menu,side,rows):
        for obj in (state,menu):
            if type(obj) is not snapshot.NativeSnapshot or obj._lib is not self.lib:raise ValueError('Action inputs must share the same resident library')
        features=np.empty((rows,self.width),dtype=np.float32)
        regions=np.empty(rows,dtype=np.int64)
        first,second=sorted((state,menu),key=id)
        with first._lock,second._lock:
            state._require_open();menu._require_open()
            if self.function(state._pointer,self.tables._pointer,menu._pointer,self.constants._pointer,side,features.ctypes.data_as(C.POINTER(C.c_float)),features.size,regions.ctypes.data_as(C.POINTER(C.c_int64)),rows):raise snapshot._error(self.lib)
        return features,regions
