"""Additive own-value constructor; reuse all existing request/reply bindings."""
import ctypes as C
import threading
import controller_binding as base


class WotrOwnValueConfig(C.Structure):
    _fields_=[('struct_size',C.c_uint32),('version',C.c_uint32),
              ('reply_panels',C.c_uint64),('belief_worlds',C.c_uint64),
              ('seed_words',C.c_uint32),('reserved',C.c_uint32),('seed',C.c_uint32*8)]


def own_configuration(seed,reply_panels=2,belief_worlds=2):
    if type(seed) is not int or not 0<=seed<2**256:
        raise ValueError('Own-value seed must be an unsigned integer of at most 256 bits')
    for value in (reply_panels,belief_worlds):
        if type(value) is not int or not 1<=value<2**64:
            raise ValueError('Own-value panel/world counts must be positive uint64 integers')
    cfg=WotrOwnValueConfig();cfg.struct_size=C.sizeof(cfg);cfg.version=1
    cfg.reply_panels=reply_panels;cfg.belief_worlds=belief_worlds
    cfg.seed_words=(seed.bit_length()+31)//32
    for index in range(cfg.seed_words):cfg.seed[index]=(seed>>(32*index))&0xffffffff
    return cfg


def configure(lib):
    base.configure(lib)
    try:
        version=lib.wotr_search_controller_own_value_version
        create=lib.wotr_search_controller_create_own_value
    except AttributeError as exc:
        raise base.ControllerUnavailable('Native library has no own-value controller capability') from exc
    version.argtypes=[];version.restype=C.c_uint32
    create.argtypes=[C.c_void_p]*6+[C.POINTER(base.WotrSearchConfig),C.POINTER(WotrOwnValueConfig)]
    create.restype=C.c_void_p
    if version()!=1:raise base.ControllerUnavailable('Unsupported native own-value controller version')


class OwnController(base.Controller):
    """Fresh native search with private reply RNG; no partner-model parameter."""
    def __init__(self,lib,root_state,observation_tables,action_tables,menu_tables,hunt_tables,rng,config,
                 *,seed,reply_panels=2,belief_worlds=2):
        import snapshot,random_stream,observation_tables as table_module
        configure(lib)
        handles=(root_state,observation_tables,action_tables,menu_tables,hunt_tables)
        # The shared table factories return immutable lifetime-owning wrappers.
        # Validate their actual owners; they are not mutable NativeSnapshot handles.
        states=tuple(s._ImmutableTables__owner if type(s) is table_module._ImmutableTables else s
                     for s in handles)
        if any(type(s) is not snapshot.NativeSnapshot or s._lib is not lib for s in states):
            raise ValueError('Own-value state/tables must belong to the controller library')
        if type(rng) is not random_stream.NativeRandom or rng._lib is not lib:
            raise ValueError('Own-value search RNG must belong to the controller library')
        if not isinstance(config,base.WotrSearchConfig):raise ValueError('Invalid search configuration type')
        config=base.WotrSearchConfig.from_buffer_copy(config)
        config.struct_size=C.sizeof(config)
        if config.opponent_policy_mode!=2 or config.leaf_batch!=1 or config.determinize!=1:
            raise ValueError('Own-value v0 requires mode 2, sequential leaves and hidden-hand sampling')
        own=own_configuration(seed,reply_panels,belief_worlds)
        self._lib=lib;self._config=config;self._own_config=own
        self.capabilities={**base.capabilities(lib),'own_value':True,'tree_reuse':False}
        if config.add_noise and not self.capabilities.get('root_priors_exchange'):
            raise base.ControllerUnavailable('Native own-value noise requires root-priors exchange')
        self._root_priors_pending=False;self.reused=False
        self._keep=(*handles,*states,rng);self._lock=threading.RLock();self._pending=None;self.requests_issued=0
        with root_state._lock,rng._lock:
            for state in states:state._require_open()
            if not rng._pointer:raise RuntimeError('Native random stream is closed')
            pointer=lib.wotr_search_controller_create_own_value(
                *(s._pointer for s in states),rng._pointer,C.byref(config),C.byref(own))
        if not pointer:raise base._error(lib)
        self._pointer=pointer
