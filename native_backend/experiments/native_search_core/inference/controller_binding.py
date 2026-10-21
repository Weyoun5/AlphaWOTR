"""ctypes binding of SHARED_ABI.md section 2 (native search controller exports).

No search arithmetic lives here. The binding exposes the controller's request
buffers as read-only NumPy views, copies them into owned arrays for the numeric
bridge, keeps reply arrays alive across `submit`, and decodes the owned
Snapshot/NativeRandom handles the controller returns. A library without the
controller exports fails loudly at `configure` with the missing symbol names.
"""
from dataclasses import dataclass
from pathlib import Path
import ctypes as C
import threading
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # native_backend
BOX = ROOT / 'experiments/native_search_core/controller/compile_check'
ABI_VERSION = 1          # SHARED_ABI section 9: stays 1 while only NEW exports are added
# Milestone-2 exports (SHARED_ABI 9.3-9.5, FINAL 2026-09-09) are optional symbol groups; a
# capability is present only when its whole group exists.  Detected by symbol, never assumed.
STATUS_REQUEST, STATUS_COMPLETE, STATUS_ROOT_PRIORS = 0, 1, 2
FEATURES = 4647
ACTION_WIDTH = 157
OP_PV, OP_PVM, OP_EVAL, OP_COMBAT_DIST = 1, 2, 3, 4
OPERATION_NAMES = {OP_PV: 'PV', OP_PVM: 'PVM', OP_EVAL: 'EVAL', OP_COMBAT_DIST: 'COMBAT_DIST'}
PURPOSE_NAMES = {1: 'root_expand', 2: 'leaf_expand', 3: 'opponent_policy', 4: 'flush_policy',
                 5: 'flush_root_value', 6: 'eval_state', 7: 'turn_greedy', 8: 'combat_dist'}
PURPOSE_COMBAT_DIST = 8


class ControllerUnavailable(RuntimeError):
    """The candidate library does not (yet) export the controller ABI."""


class WotrSearchConfig(C.Structure):
    _fields_ = [
        ('struct_size', C.c_uint32), ('root_side', C.c_int32), ('num_simulations', C.c_int64),
        ('sims_menu_scale', C.c_int32), ('sims_per_child', C.c_int64), ('sims_floor', C.c_int64),
        ('c_puct', C.c_double), ('leaf_batch', C.c_int64), ('turn_horizon', C.c_int64),
        ('max_children', C.c_int64), ('chance_k_max', C.c_int64), ('kernel_k_max', C.c_int64),
        ('chance_draws', C.c_int64), ('free_cands', C.c_int32), ('free_cands_n', C.c_int64),
        ('free_force', C.c_int64), ('opponent_policy_mode', C.c_int32),
        ('opponent_policy_epsilon', C.c_double), ('conf_leaf', C.c_int32), ('conf_track', C.c_int32),
        ('conf_ref', C.c_int32), ('determinize', C.c_int32), ('add_noise', C.c_int32),
        ('overlay_assert', C.c_int32), ('reward_contract', C.c_int32), ('fp_profile', C.c_int32),
        ('sh_profile', C.c_int32), ('phi_rings_rho', C.c_double), ('trace', C.c_int32)]


class WotrSearchRequest(C.Structure):
    _fields_ = [
        ('struct_size', C.c_uint32), ('request_id', C.c_uint64), ('operation', C.c_int32),
        ('model_side', C.c_int32), ('purpose', C.c_int32), ('batch', C.c_uint64), ('rows', C.c_uint64),
        ('leaf_ids', C.POINTER(C.c_uint64)), ('observations', C.POINTER(C.c_float)),
        ('actions', C.POINTER(C.c_float)), ('regions', C.POINTER(C.c_int64)),
        ('offsets', C.POINTER(C.c_uint64))]


class WotrSearchReply(C.Structure):
    _fields_ = [
        ('struct_size', C.c_uint32), ('request_id', C.c_uint64), ('priors', C.POINTER(C.c_float)),
        ('values', C.POINTER(C.c_double)), ('conf', C.POINTER(C.c_double)),
        ('mixed', C.POINTER(C.c_double))]


class WotrSearchRoot(C.Structure):
    _fields_ = [
        ('struct_size', C.c_uint32), ('children', C.c_uint64), ('visits', C.POINTER(C.c_uint64)),
        ('term_visits', C.POINTER(C.c_uint64)), ('forced_visits', C.POINTER(C.c_uint64)),
        ('value_sum', C.POINTER(C.c_double)), ('value_sq', C.POINTER(C.c_double)),
        ('conf_sum', C.POINTER(C.c_double)), ('phi_sum', C.POINTER(C.c_double)),
        ('priors_clean', C.POINTER(C.c_float)), ('proven', C.POINTER(C.c_int8)),
        ('proven_value', C.POINTER(C.c_double)), ('side', C.POINTER(C.c_int32)),
        ('root_side', C.c_int32), ('simulations_run', C.c_uint64), ('budget', C.c_uint64)]


class WotrSearchSessionInfo(C.Structure):
    """SHARED_ABI 9.6: structural facts of a detached subtree (32 bytes, natural alignment)."""
    _fields_ = [
        ('struct_size', C.c_uint32), ('side', C.c_int32), ('is_chance', C.c_int32), ('has_state', C.c_int32),
        ('children', C.c_uint64), ('visit_count', C.c_uint64)]


STRUCT_SIZES = {name: C.sizeof(cls) for name, cls in (
    ('WotrSearchConfig', WotrSearchConfig), ('WotrSearchRequest', WotrSearchRequest),
    ('WotrSearchReply', WotrSearchReply), ('WotrSearchRoot', WotrSearchRoot),
    ('WotrSearchSessionInfo', WotrSearchSessionInfo))}
assert STRUCT_SIZES['WotrSearchSessionInfo'] == 32, STRUCT_SIZES

P = C.c_void_p
SYMBOLS = {
    'abi_version': ([], C.c_uint32),
    'create': ([P, P, P, P, P, P, C.POINTER(WotrSearchConfig)], P),
    'destroy': ([P], None),
    'advance': ([P, C.POINTER(C.c_int32)], C.c_int),
    'request': ([P, C.POINTER(WotrSearchRequest)], C.c_int),
    'submit': ([P, C.POINTER(WotrSearchReply)], C.c_int),
    'root': ([P, C.POINTER(WotrSearchRoot)], C.c_int),
    'root_menu': ([P], P),
    'rng': ([P], P),
    'counters': ([P], P),
    'export': ([P, C.c_int32], P),
}
# Optional exports grouped by capability (SHARED_ABI 9.3-9.6), keyed by the FULL export name.
# A capability is present only when EVERY symbol of its group exists; partial groups are
# reported and refused.
OPTIONAL_SYMBOLS = {
    # 9.4: status 2 hand-out of the K stored child priors + is_free flags; hand-back of K priors.
    'root_priors_exchange': {
        'wotr_search_controller_root_priors': ([P, C.POINTER(C.c_uint64), C.POINTER(C.POINTER(C.c_double)),
                                                C.POINTER(C.POINTER(C.c_uint8))], C.c_int),
        'wotr_search_controller_set_root_priors': ([P, C.POINTER(C.c_double), C.c_uint64], C.c_int),
    },
    # 9.5: op-4 COMBAT_DIST request payload (owned metadata Snapshot) and reply submit.
    'combat_dist': {
        'wotr_search_controller_request_payload': ([P], P),
        'wotr_search_controller_submit_payload': ([P, C.c_uint64, P], C.c_int),
    },
    # 9.3: observer event export (owned metadata Snapshot), readable after finish or failure.
    'observer_events': {
        'wotr_search_controller_observer_events': ([P], P),
    },
    # 9.6: tree-reuse session (detached subtree) + create_reusing.
    'tree_reuse': {
        'wotr_search_session_detach': ([P, C.c_uint64], P),
        'wotr_search_session_destroy': ([P], None),
        'wotr_search_session_info': ([P, C.POINTER(WotrSearchSessionInfo)], C.c_int),
        'wotr_search_session_state': ([P], P),
        'wotr_search_controller_create_reusing': ([P, P, P, P, P, P, C.POINTER(WotrSearchConfig), P, C.c_int32,
                                                   C.POINTER(C.c_int32)], P),
    },
}
_CONFIGURED = set()
_CAPABILITIES = {}
_CONFIGURE_LOCK = threading.Lock()


def _error(lib):
    import snapshot
    return snapshot._error(lib)


def configure(lib):
    """Attach the controller signatures, failing loudly on missing exports."""
    with _CONFIGURE_LOCK:
        if id(lib) in _CONFIGURED:
            return lib
        missing = [n for n in SYMBOLS if not hasattr(lib, 'wotr_search_controller_' + n)]
        if missing:
            raise ControllerUnavailable(
                'Native library %s lacks the search controller exports %s (SHARED_ABI.md section 2); '
                'build the controller candidate (work package A) before using the orchestrator'
                % (getattr(lib, '_name', '?'), ', '.join('wotr_search_controller_' + n for n in missing)))
        for name, (arguments, result) in SYMBOLS.items():
            function = getattr(lib, 'wotr_search_controller_' + name)
            function.argtypes = arguments
            function.restype = result
        capabilities = {}
        for capability, group in OPTIONAL_SYMBOLS.items():
            present = [n for n in group if hasattr(lib, n)]
            if present and len(present) != len(group):
                raise ControllerUnavailable(
                    'Native library %s exports a PARTIAL %s group (%s of %s); the capability is refused'
                    % (getattr(lib, '_name', '?'), capability, present, sorted(group)))
            capabilities[capability] = len(present) == len(group)
            if capabilities[capability]:
                for name, (arguments, result) in group.items():
                    function = getattr(lib, name)
                    function.argtypes = arguments
                    function.restype = result
        version = lib.wotr_search_controller_abi_version()
        if version != ABI_VERSION:
            raise ControllerUnavailable('Search controller ABI version %d; this binding implements %d'
                                        % (version, ABI_VERSION))
        _CAPABILITIES[id(lib)] = capabilities
        _CONFIGURED.add(id(lib))
        return lib


def capabilities(lib):
    """Optional-capability table of a configured controller library ({name: bool})."""
    configure(lib)
    return dict(_CAPABILITIES[id(lib)])


def has_capability(lib, name):
    """False (never an exception) when the library lacks the controller or the capability."""
    try:
        return bool(capabilities(lib).get(name, False))
    except ControllerUnavailable:
        return False


def candidate_library(box=BOX):
    """Load (or verify) the candidate DLL as THE resident native library.

    bridge.library() caches one library per process; the candidate must be the
    first native library loaded (the `bridge.HERE = BOX; bridge.library()`
    pattern of run_gate.py). Snapshot and random-stream signatures are attached
    so table handles, menus and RNG clones share the controller's library.
    """
    import bridge, snapshot, random_stream
    box = Path(box).resolve()
    if bridge._LIB is None:
        manifest = box / 'build/manifest.json'
        if not manifest.exists():
            raise ControllerUnavailable(
                'Candidate controller build is missing: %s (controller/prepare.py + compile_check/build.py '
                'have not produced a passing build yet)' % manifest)
        original = bridge.HERE
        bridge.HERE = box
        try:
            bridge.library()
        finally:
            bridge.HERE = original
    resident = Path(bridge._LIB._name).resolve()
    if not resident.is_relative_to(box / 'build'):
        raise ControllerUnavailable(
            'Resident native library %s is not the candidate box %s; load the candidate before any '
            'other native call in this process' % (resident, box))
    lib = snapshot.library()
    random_stream.library()
    return configure(lib)


def _view(pointer, ctype, count, dtype):
    """Read-only NumPy view over controller memory (no copy)."""
    if count == 0:
        return np.empty(0, dtype=dtype)
    if not pointer:
        raise RuntimeError('Controller returned a NULL buffer for %d %s entries' % (count, dtype))
    view = np.ctypeslib.as_array(C.cast(pointer, C.POINTER(ctype)), shape=(count,))
    if view.dtype != np.dtype(dtype):
        view = view.view(dtype)
    view.flags.writeable = False
    return view


def _owned(view, dtype, shape):
    result = np.array(view, dtype=dtype, order='C', copy=True).reshape(shape)
    return result


@dataclass(frozen=True)
class RequestView:
    """Owned copies of one pending request (the numeric bridge consumes these).

    `raw` holds the read-only views over the controller's buffers; they stay
    valid until the matching submit succeeds or the controller closes.
    """
    request_id: int
    operation: int
    model_side: int
    purpose: int
    batch: int
    rows: int
    leaf_ids: np.ndarray       # uint64 [B]
    observations: np.ndarray   # float32 [B, 4647]
    actions: np.ndarray        # float32 [rows, 157]
    regions: np.ndarray        # int64 [rows]
    offsets: np.ndarray        # uint64 [B+1]
    raw: dict

    def summary(self):
        return dict(request_id=self.request_id, operation=OPERATION_NAMES.get(self.operation, self.operation),
                    model_side=self.model_side, purpose=PURPOSE_NAMES.get(self.purpose, self.purpose),
                    batch=self.batch, rows=self.rows, leaf_ids=[int(x) for x in self.leaf_ids])


@dataclass(frozen=True)
class RootPriorsView:
    """9.4 root-priors hand-out (owned copies): the K stored child priors and is_free flags."""
    children: int
    priors: np.ndarray        # float64 [K] child priors exactly as stored by the controller
    is_free: np.ndarray       # uint8 [K] 1 = action_type in ai._FREE_ACTION_TYPES


@dataclass(frozen=True)
class RootView:
    children: int
    visits: np.ndarray         # uint64
    term_visits: np.ndarray    # uint64
    forced_visits: np.ndarray  # uint64
    value_sum: np.ndarray      # float64
    value_sq: np.ndarray       # float64
    conf_sum: np.ndarray       # float64
    phi_sum: np.ndarray        # float64
    priors_clean: np.ndarray   # float32
    proven: np.ndarray         # int8
    proven_value: np.ndarray   # float64
    side: np.ndarray           # int32
    root_side: int
    simulations_run: int
    budget: int


@dataclass(frozen=True)
class SessionInfo:
    side: int
    is_chance: bool
    has_state: bool
    children: int
    visit_count: int


class Session:
    """SHARED_ABI 9.6: an owned detached subtree (native).  Destroyed on close()/GC unless a
    controller consumed it (`release()` after a reused create)."""

    def __init__(self, lib, pointer):
        if not pointer:
            raise RuntimeError('Session pointer is NULL')
        self._lib = lib
        self._pointer = pointer
        self._lock = threading.RLock()

    @property
    def open(self):
        return bool(self._pointer)

    def _require_open(self):
        if not self._pointer:
            raise RuntimeError('Search session is closed or consumed')

    def info(self):
        out = WotrSearchSessionInfo()
        out.struct_size = C.sizeof(WotrSearchSessionInfo)
        with self._lock:
            self._require_open()
            if self._lib.wotr_search_session_info(self._pointer, C.byref(out)):
                raise _error(self._lib)
        if int(out.side) not in (0, 1) or int(out.is_chance) not in (0, 1) or int(out.has_state) not in (0, 1):
            raise RuntimeError('Session info fields out of domain')
        return SessionInfo(side=int(out.side), is_chance=bool(out.is_chance), has_state=bool(out.has_state),
                           children=int(out.children), visit_count=int(out.visit_count))

    def state(self):
        """Owned NativeSnapshot clone of the retained child.state (caller closes it)."""
        import snapshot
        with self._lock:
            self._require_open()
            pointer = self._lib.wotr_search_session_state(self._pointer)
            if not pointer:
                raise _error(self._lib)
            return snapshot.NativeSnapshot(self._lib, pointer)

    def release(self):
        """The controller consumed the session (reused): forget the pointer without destroying."""
        with self._lock:
            self._pointer = None

    def close(self):
        with self._lock:
            if self._pointer:
                self._lib.wotr_search_session_destroy(self._pointer)
                self._pointer = None

    def __del__(self):
        if getattr(self, '_pointer', None):
            self.close()


class Controller:
    """One native search controller instance (ABI section 1 lifetime rules).

    session/adopt (9.6): when `session` is a Session the controller is created with
    `wotr_search_controller_create_reusing` (entry flush of the cached root's edge catalog
    whether or not it adopts); `reused` reports the outcome and a reused session is released
    (owned by the controller), an unreused one stays owned by the caller.
    """

    def __init__(self, lib, root_state, observation_tables, action_tables, menu_tables, hunt_tables, rng, config,
                 session=None, adopt=False):
        import snapshot, random_stream
        configure(lib)
        if type(root_state) is not snapshot.NativeSnapshot or root_state._lib is not lib:
            raise ValueError('Root state must be an open NativeSnapshot of the controller library')
        if type(rng) is not random_stream.NativeRandom or rng._lib is not lib:
            raise ValueError('RNG must be a NativeRandom of the controller library')
        if not isinstance(config, WotrSearchConfig):
            raise ValueError('config must be a WotrSearchConfig')
        config.struct_size = C.sizeof(WotrSearchConfig)
        self._lib = lib
        self._config = config
        self.capabilities = capabilities(lib)
        if config.add_noise and not self.capabilities.get('root_priors_exchange'):
            raise ControllerUnavailable(
                'add_noise requires the SHARED_ABI 9.4 exports (root_priors/set_root_priors); '
                'this library does not provide them')
        if session is not None:
            if not self.capabilities.get('tree_reuse'):
                raise ControllerUnavailable('tree reuse requires the SHARED_ABI 9.6 exports; this library lacks them')
            if type(session) is not Session or session._lib is not lib:
                raise ValueError('session must be an open Session of the controller library')
        self._root_priors_pending = False
        self.reused = False
        # Tables must outlive the controller (ABI section 1): retain them here.
        self._keep = (root_state, observation_tables, action_tables, menu_tables, hunt_tables, rng)
        self._lock = threading.RLock()
        self._pending = None
        self.requests_issued = 0
        with root_state._lock, rng._lock:
            root_state._require_open()
            if not rng._pointer:
                raise RuntimeError('Native random stream is closed')
            if session is None:
                pointer = lib.wotr_search_controller_create(
                    root_state._pointer, observation_tables._pointer, action_tables._pointer,
                    menu_tables._pointer, hunt_tables._pointer, rng._pointer, C.byref(config))
            else:
                reused = C.c_int32(0)
                with session._lock:
                    session._require_open()
                    pointer = lib.wotr_search_controller_create_reusing(
                        root_state._pointer, observation_tables._pointer, action_tables._pointer,
                        menu_tables._pointer, hunt_tables._pointer, rng._pointer, C.byref(config),
                        session._pointer, C.c_int32(1 if adopt else 0), C.byref(reused))
                    if pointer and reused.value not in (0, 1):
                        lib.wotr_search_controller_destroy(pointer)
                        raise RuntimeError('create_reusing reported reused=%d' % reused.value)
                    if pointer and reused.value == 1:
                        if not adopt:
                            lib.wotr_search_controller_destroy(pointer)
                            raise RuntimeError('Controller adopted a session although adopt was 0')
                        session.release()
                    self.reused = bool(pointer) and reused.value == 1
        if not pointer:
            raise _error(lib)
        self._pointer = pointer

    def detach_child(self, index):
        """9.6 _stash_tree: detach root child `index` after status 1; returns (Session, None) or
        (None, error text) when the child has no state or is a chance node (source: no stash)."""
        if not self.capabilities.get('tree_reuse'):
            raise ControllerUnavailable('detach_child requires the SHARED_ABI 9.6 exports')
        with self._lock:
            self._require_open()
            pointer = self._lib.wotr_search_session_detach(self._pointer, C.c_uint64(int(index)))
            if not pointer:
                return None, str(_error(self._lib))
            return Session(self._lib, pointer), None

    def _require_open(self):
        if not self._pointer:
            raise RuntimeError('Search controller is closed')

    def advance(self):
        """Run native work; returns 0 (request pending), 1 (search complete) or, only when
        the library has the root-priors exchange, 2 (root priors pending, proposal)."""
        status = C.c_int32(-1)
        with self._lock:
            self._require_open()
            if self._lib.wotr_search_controller_advance(self._pointer, C.byref(status)):
                raise _error(self._lib)
        if status.value == STATUS_ROOT_PRIORS:
            if not self.capabilities.get('root_priors_exchange'):
                raise RuntimeError('Controller reported status 2 (root priors pending) without the exchange exports')
            if not self._config.add_noise:
                raise RuntimeError('Controller paused for root priors although add_noise is 0')
            return status.value
        if status.value not in (STATUS_REQUEST, STATUS_COMPLETE):
            raise RuntimeError('Controller advance returned unknown status %d' % status.value)
        return status.value

    def root_priors(self):
        """9.4 hand-out (status 2 only): owned copies of the K stored child priors (double) and
        the is_free flags (uint8), in child order."""
        if not self.capabilities.get('root_priors_exchange'):
            raise ControllerUnavailable('root_priors requires the SHARED_ABI 9.4 exports')
        count = C.c_uint64(0)
        priors_ptr = C.POINTER(C.c_double)()
        free_ptr = C.POINTER(C.c_uint8)()
        with self._lock:
            self._require_open()
            if self._lib.wotr_search_controller_root_priors(self._pointer, C.byref(count), C.byref(priors_ptr),
                                                            C.byref(free_ptr)):
                raise _error(self._lib)
            k = int(count.value)
            if k < 1:
                raise RuntimeError('Controller handed out an empty root-priors vector')
            priors = _owned(_view(priors_ptr, C.c_double, k, np.float64), np.float64, (k,))
            free = _owned(_view(free_ptr, C.c_uint8, k, np.uint8), np.uint8, (k,))
            self._root_priors_pending = True
        if not np.isfinite(priors).all() or np.any(priors < 0.0):
            raise RuntimeError('Controller handed out non-finite or negative root priors')
        if np.any((free != 0) & (free != 1)):
            raise RuntimeError('Controller is_free flags must be 0/1')
        return RootPriorsView(children=k, priors=priors, is_free=free)

    def set_root_priors(self, priors):
        """9.4 hand-back: the K noised priors (double, child order); the controller stores them as
        the children's priors and the next advance runs the simulations."""
        if not self.capabilities.get('root_priors_exchange'):
            raise ControllerUnavailable('set_root_priors requires the SHARED_ABI 9.4 exports')
        values = np.ascontiguousarray(priors, dtype=np.float64)
        with self._lock:
            self._require_open()
            if not self._root_priors_pending:
                raise RuntimeError('No root-priors hand-out is pending')
            if values.ndim != 1 or values.shape[0] < 1:
                raise RuntimeError('Noised root priors must be a non-empty vector')
            if not np.isfinite(values).all() or np.any(values < 0.0):
                raise RuntimeError('Noised root priors must be finite and non-negative')
            if self._lib.wotr_search_controller_set_root_priors(
                    self._pointer, values.ctypes.data_as(C.POINTER(C.c_double)), C.c_uint64(values.shape[0])):
                raise _error(self._lib)
            self._root_priors_pending = False

    # -- 9.5 combat distribution request (op 4) --
    def request_payload(self, ai):
        """Decode the pending op-4 payload: {'context', 'node', 'action_type', 'root_branch_row', 'parent_battle'}."""
        import snapshot
        if not self.capabilities.get('combat_dist'):
            raise ControllerUnavailable('The controller issued a COMBAT_DIST request but the library lacks the '
                                        'SHARED_ABI 9.5 exports (request_payload/submit_payload)')
        with self._lock:
            self._require_open()
            pending = self._pending
            if pending is None or pending.operation != OP_COMBAT_DIST:
                raise RuntimeError('No COMBAT_DIST request is pending')
            pointer = self._lib.wotr_search_controller_request_payload(self._pointer)
            if not pointer:
                raise _error(self._lib)
        with snapshot.NativeSnapshot(self._lib, pointer) as payload:
            value = payload.unpack_metadata(ai)
        if not isinstance(value, dict) or not {'context', 'node', 'action_type', 'root_branch_row', 'parent_battle'} <= set(value):
            raise RuntimeError('COMBAT_DIST payload must decode to the 9.5 dict; got %r' % (type(value),))
        if int(value['node']) != int(pending.leaf_ids[0]):
            raise RuntimeError('COMBAT_DIST payload node %r differs from the request leaf id %r'
                               % (value['node'], int(pending.leaf_ids[0])))
        return value

    def submit_payload(self, reply_snapshot):
        """Deliver the 9.5 reply Snapshot for the pending op-4 request; the Snapshot stays open
        until the call returns (the caller closes it afterwards)."""
        import snapshot
        if not self.capabilities.get('combat_dist'):
            raise ControllerUnavailable('submit_payload requires the SHARED_ABI 9.5 exports')
        if type(reply_snapshot) is not snapshot.NativeSnapshot or reply_snapshot._lib is not self._lib:
            raise ValueError('Reply must be an open NativeSnapshot of the controller library')
        with self._lock, reply_snapshot._lock:
            self._require_open()
            reply_snapshot._require_open()
            pending = self._pending
            if pending is None or pending.operation != OP_COMBAT_DIST:
                raise RuntimeError('No COMBAT_DIST request is pending')
            if self._lib.wotr_search_controller_submit_payload(self._pointer, C.c_uint64(pending.request_id),
                                                               reply_snapshot._pointer):
                raise _error(self._lib)
            self._pending = None

    # -- 9.3 observer event export --
    def observer_events(self, ai):
        """Owned metadata Snapshot -> list of ('hidden_commit', ...) / ('card_policy', ...) tuples in order."""
        import snapshot
        if not self.capabilities.get('observer_events'):
            raise ControllerUnavailable('observer_events requires the SHARED_ABI 9.3 export')
        with self._lock:
            self._require_open()
            pointer = self._lib.wotr_search_controller_observer_events(self._pointer)
            if not pointer:
                raise _error(self._lib)
        with snapshot.NativeSnapshot(self._lib, pointer) as payload:
            value = payload.unpack_metadata(ai)
        if not isinstance(value, (list, tuple)):
            raise RuntimeError('Observer events must decode to a list; got %r' % (type(value),))
        return list(value)

    def request(self):
        out = WotrSearchRequest()
        out.struct_size = C.sizeof(WotrSearchRequest)
        with self._lock:
            self._require_open()
            if self._lib.wotr_search_controller_request(self._pointer, C.byref(out)):
                raise _error(self._lib)
            batch, rows = int(out.batch), int(out.rows)
            operation = int(out.operation)
            if operation not in OPERATION_NAMES:
                raise RuntimeError('Controller issued unknown operation %d' % operation)
            if batch < 1:
                raise RuntimeError('Controller issued an empty batch')
            if operation == OP_EVAL and rows != 0:
                raise RuntimeError('EVAL request carries %d action rows' % rows)
            if operation == OP_PV and batch != 1:
                raise RuntimeError('PV request batch %d; expected 1' % batch)
            if operation == OP_COMBAT_DIST:
                # 9.5: batch 1, rows 0, leaf_ids [node id], model_side = node side, purpose 8;
                # numeric buffers empty/NULL and never read.
                if batch != 1 or rows != 0 or int(out.purpose) != PURPOSE_COMBAT_DIST:
                    raise RuntimeError('COMBAT_DIST request must have batch 1, rows 0, purpose 8 (got %d, %d, %d)'
                                       % (batch, rows, int(out.purpose)))
                leaf = _view(out.leaf_ids, C.c_uint64, 1, np.uint64)
                view = RequestView(
                    request_id=int(out.request_id), operation=operation, model_side=int(out.model_side),
                    purpose=int(out.purpose), batch=1, rows=0, leaf_ids=_owned(leaf, np.uint64, (1,)),
                    observations=np.empty((0, FEATURES), dtype=np.float32),
                    actions=np.empty((0, ACTION_WIDTH), dtype=np.float32), regions=np.empty(0, dtype=np.int64),
                    offsets=np.zeros(2, dtype=np.uint64), raw=dict(leaf_ids=leaf))
                self._pending = view
                self.requests_issued += 1
                return view
            raw = dict(
                leaf_ids=_view(out.leaf_ids, C.c_uint64, batch, np.uint64),
                observations=_view(out.observations, C.c_float, batch * FEATURES, np.float32),
                actions=_view(out.actions, C.c_float, rows * ACTION_WIDTH, np.float32),
                regions=_view(out.regions, C.c_int64, rows, np.int64),
                offsets=_view(out.offsets, C.c_uint64, batch + 1, np.uint64))
            view = RequestView(
                request_id=int(out.request_id), operation=operation, model_side=int(out.model_side),
                purpose=int(out.purpose), batch=batch, rows=rows,
                leaf_ids=_owned(raw['leaf_ids'], np.uint64, (batch,)),
                observations=_owned(raw['observations'], np.float32, (batch, FEATURES)),
                actions=_owned(raw['actions'], np.float32, (rows, ACTION_WIDTH)),
                regions=_owned(raw['regions'], np.int64, (rows,)),
                offsets=_owned(raw['offsets'], np.uint64, (batch + 1,)), raw=raw)
            if int(view.offsets[0]) != 0 or int(view.offsets[-1]) != rows:
                raise RuntimeError('Controller request offsets do not span the action rows')
            self._pending = view
            self.requests_issued += 1
            return view

    def submit(self, reply):
        """Deliver a numeric_bridge.Reply; its arrays stay alive until return."""
        with self._lock:
            self._require_open()
            pending = self._pending
            if pending is None:
                raise RuntimeError('No request is pending')
            if pending.operation == OP_COMBAT_DIST:
                raise RuntimeError('A COMBAT_DIST request is answered with submit_payload, not a numeric reply')
            if int(reply.request_id) != pending.request_id:
                raise RuntimeError('Reply %d does not answer pending request %d' % (reply.request_id, pending.request_id))
            values = np.ascontiguousarray(reply.values, dtype=np.float64)
            conf = np.ascontiguousarray(reply.conf, dtype=np.float64)
            if values.shape != (pending.batch,) or conf.shape != (pending.batch,):
                raise RuntimeError('Reply values/conf shape differs from the request batch')
            struct = WotrSearchReply()
            struct.struct_size = C.sizeof(WotrSearchReply)
            struct.request_id = pending.request_id
            struct.values = values.ctypes.data_as(C.POINTER(C.c_double))
            struct.conf = conf.ctypes.data_as(C.POINTER(C.c_double))
            keep = [values, conf]
            if pending.operation == OP_EVAL:
                if reply.priors is not None or reply.mixed is not None:
                    raise RuntimeError('EVAL reply must not carry priors/mixed')
                struct.priors = None
                struct.mixed = None
            else:
                priors = np.ascontiguousarray(reply.priors, dtype=np.float32)
                mixed = np.ascontiguousarray(reply.mixed, dtype=np.float64)
                if priors.shape != (pending.rows,) or mixed.shape != (pending.rows,):
                    raise RuntimeError('Reply priors/mixed shape differs from the request rows')
                struct.priors = priors.ctypes.data_as(C.POINTER(C.c_float))
                struct.mixed = mixed.ctypes.data_as(C.POINTER(C.c_double))
                keep += [priors, mixed]
            if self._lib.wotr_search_controller_submit(self._pointer, C.byref(struct)):
                raise _error(self._lib)
            del keep
            self._pending = None

    def root(self):
        out = WotrSearchRoot()
        out.struct_size = C.sizeof(WotrSearchRoot)
        with self._lock:
            self._require_open()
            if self._lib.wotr_search_controller_root(self._pointer, C.byref(out)):
                raise _error(self._lib)
            k = int(out.children)
            arrays = dict(
                visits=_owned(_view(out.visits, C.c_uint64, k, np.uint64), np.uint64, (k,)),
                term_visits=_owned(_view(out.term_visits, C.c_uint64, k, np.uint64), np.uint64, (k,)),
                forced_visits=_owned(_view(out.forced_visits, C.c_uint64, k, np.uint64), np.uint64, (k,)),
                value_sum=_owned(_view(out.value_sum, C.c_double, k, np.float64), np.float64, (k,)),
                value_sq=_owned(_view(out.value_sq, C.c_double, k, np.float64), np.float64, (k,)),
                conf_sum=_owned(_view(out.conf_sum, C.c_double, k, np.float64), np.float64, (k,)),
                phi_sum=_owned(_view(out.phi_sum, C.c_double, k, np.float64), np.float64, (k,)),
                priors_clean=_owned(_view(out.priors_clean, C.c_float, k, np.float32), np.float32, (k,)),
                proven=_owned(_view(out.proven, C.c_int8, k, np.int8), np.int8, (k,)),
                proven_value=_owned(_view(out.proven_value, C.c_double, k, np.float64), np.float64, (k,)),
                side=_owned(_view(out.side, C.c_int32, k, np.int32), np.int32, (k,)))
            return RootView(children=k, root_side=int(out.root_side), simulations_run=int(out.simulations_run),
                            budget=int(out.budget), **arrays)

    def root_menu(self):
        import snapshot
        with self._lock:
            self._require_open()
            return snapshot.NativeSnapshot(self._lib, self._lib.wotr_search_controller_root_menu(self._pointer))

    def rng_state(self):
        """Python `random` state tuple of the search stream after the search."""
        import random_stream
        with self._lock:
            self._require_open()
            pointer = self._lib.wotr_search_controller_rng(self._pointer)
        with random_stream.NativeRandom._adopt(self._lib, pointer) as clone:
            return clone.getstate()

    def counters(self, ai):
        import snapshot
        with self._lock:
            self._require_open()
            pointer = self._lib.wotr_search_controller_counters(self._pointer)
        with snapshot.NativeSnapshot(self._lib, pointer) as payload:
            return payload.unpack_metadata(ai)

    def export(self, ai, with_states=False):
        import snapshot
        with self._lock:
            self._require_open()
            pointer = self._lib.wotr_search_controller_export(self._pointer, int(bool(with_states)))
        with snapshot.NativeSnapshot(self._lib, pointer) as payload:
            return payload.unpack_metadata(ai)

    def close(self):
        with self._lock:
            if self._pointer:
                self._lib.wotr_search_controller_destroy(self._pointer)
                self._pointer = None
                self._pending = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if getattr(self, '_pointer', None):
            self.close()
