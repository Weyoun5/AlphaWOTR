"""Batched float32/int64 outputs from retained state/menu handles; no callbacks."""
import ctypes as C
import numpy as np
import snapshot
import observation_tables

def action_tables(ai):
    import combat_round_observation as round_observation
    import game_state
    import action_schema
    if action_schema.ACTION_ENCODING_CONTRACT != "observed_combat_choices_v8":
        raise ValueError("Native action features require observed_combat_choices_v8")
    assert ai.ACT_GEN=='v52' and ai.ACTION_ENCODING_DIM==157
    return dict(schema='wotr_action_features_v52_157',action_encoding_contract=action_schema.ACTION_ENCODING_CONTRACT,width=157,regions=ai.NUM_REGIONS,
        region_index=dict(ai.REGION_INDEX),category=dict(ai._ATYPE_TO_CAT),
        category_index=dict(ai._ACTION_TYPE_TO_IDX),subtypes=dict(ai._SUBTYPE_TO_IDX),
        characters=dict(ai._ACTION_CHAR_IDX),faces=dict(ai._FACE_ORDER),
        offsets={key:int(value) for key,value in vars(ai).items() if key.startswith('_ACT_') and key.endswith('_OFF')},
        targets=dict(round_observation._TARGET_INDEX),payment_nations=list(round_observation.PAYMENT_NATIONS),
        companion_names=list(round_observation.COMPANIONS),casualty_operations=list(ai._CASUALTY_COST),
        nation_values={nation.name:nation.value for nation in game_state.Nation})

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
