"""Python side of the native raw-game generator: loads THIS folder's DLL (box/) and drives one game per seed.

The native engine owns the game (rules, menus, encoders, the game RNG); Python owns only what the Python generator's
own code does there: building the opening position (environment.PythonEnvironment.__init__, unchanged), the policy
forward (PolicyActor.predict) and the sampling (Prediction.sample with environment's per-side numpy samplers).

    ctx = native_env.load(ai)                       # once per process
    game = native_env.NativeRawGame(ctx, ai, seed)  # PythonEnvironment's opening, packed into the native engine
    while game.next():                              # next learned decision (forced frames applied natively)
        obs, watch, cand, regions, kinds = game.encode()
        ...                                         # predict + sample exactly as valuedata.play
        game.apply(selected)
    game.end  -> environment.EpisodeEnd

Loading never falls back: a missing / drifted DLL, a drifted shared boundary file, or a different engine refuses.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("WOTR_TREE") or HERE.parents[2])     # the code tree: tscommon names it (a packed kit's own)
ORIGINAL = ROOT / "claude_bugscan_2026-10-21" / "native_w2"
BACKEND = ROOT / "native_backend"
BOX = HERE / "box"
# The library that is loaded: on Windows the box's own MSVC build; on every other platform the runtime box that
# box/build_portable.py made there for this platform (box/platform/<tag>).  WOTR_TS_BOX names another (a check).
RUNTIME_BOX = Path(os.environ.get("WOTR_TS_BOX") or (
    BOX if os.name == "nt" else BOX / "platform" / ("%s-%s" % (sys.platform, platform.machine().lower())))).resolve()
PIN = ORIGINAL / "NATIVE_RAW_PIN.json"      # the certified folder's engine pin (this copy adds one export)
PHASES = {0: "opening", 1: "declaration", 3: "preaction", 5: "actions"}
ROUTES = {1: "military", 2: "ring_destroyed", 3: "ring_corruption"}
# action types encode_probabilities gives a non-zero row (probabilities.py): any other frame's rows are all zero
CHANCE_KINDS = {"fellowship_move", "mordor_move", "army_attack", "siege_assault", "combat_card_play", "combat_card_none",
                "combat_effect_choice"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Context:
    pass


class NativeRefusal(RuntimeError):
    """The native engine refused a step (an unported rules path; it never mis-simulates silently). The caller replays
    the whole game on the Python generator."""


_CTX = None


def load(ai, *, verify_pin=True):
    """Load the raw-game DLL as THE resident native library of this process, with the shared Python boundary
    (bridge / snapshot / random_stream / observation_tables / native_action_features, read-only)."""
    global _CTX
    if _CTX is not None:
        if _CTX.ai is not ai:
            raise RuntimeError("native raw generator: engine module changed within one process")
        return _CTX
    if getattr(ai, "ACT_GEN", None) != "v53" or ai.get_feature_dim() != 4745 or ai.ACTION_ENCODING_DIM != 199:
        raise RuntimeError("native raw generator is v53 only (4745 / 199)")
    pin = json.loads(PIN.read_text(encoding="utf-8")) if (verify_pin and PIN.exists()) else None
    if verify_pin and pin is None:
        raise RuntimeError("native raw generator: no NATIVE_RAW_PIN.json (run tools/write_pin.py after parity)")
    if pin is not None:
        verify(ai, pin)
    for p in (BACKEND,):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import bridge
    if bridge._LIB is None:
        original = bridge.HERE
        if not (RUNTIME_BOX / "build" / "manifest.json").is_file():
            raise RuntimeError("the fast engine is not built for this platform: run  python %s  (%s is missing)" % (
                BOX / "build_portable.py", RUNTIME_BOX / "build" / "manifest.json"))
        bridge.HERE = RUNTIME_BOX
        try:
            bridge.library()
        finally:
            bridge.HERE = original
    resident = Path(bridge._LIB._name).resolve()
    if not resident.is_relative_to((RUNTIME_BOX / "build").resolve()):
        raise RuntimeError(f"another native library is resident ({resident}); load the raw generator first")
    bridge.load_reference = lambda: ai
    import snapshot
    import random_stream
    import observation_tables
    import native_action_features
    lib = snapshot.library()
    random_stream.library()
    P, I64, I32, F, N = C.c_void_p, C.c_int64, C.c_int32, C.POINTER(C.c_float), C.c_size_t
    sig = {
        "wotr_raw_abi_version": ([], C.c_uint32),
        "wotr_raw_last_error": ([], C.c_char_p),
        "wotr_raw_create": ([P, P, P, P, I64, I64], P),
        "wotr_raw_create_at": ([P, P, P, P, I64, I64, I64, I64], P),
        "wotr_raw_destroy": ([P], None),
        "wotr_raw_next": ([P, C.POINTER(I64)], C.c_int),
        "wotr_raw_encode": ([P, F, F, N, F, N, C.POINTER(I64), C.POINTER(I32), N], C.c_int),
        "wotr_raw_encode_view": ([P, I32, F, N], C.c_int),
        "wotr_raw_encode_menu": ([P, F, N, C.POINTER(I64), N], C.c_int),
        "wotr_raw_memo": ([C.c_int], C.c_int),
        "wotr_raw_vocab_size": ([P], I64),
        "wotr_raw_vocab": ([P, I64], C.c_char_p),
        "wotr_raw_apply": ([P, I64], C.c_int),
        "wotr_raw_end": ([P, C.POINTER(I64), C.c_char_p, N], C.c_int),
        "wotr_raw_state": ([P], P),
        "wotr_raw_random": ([P], P),
        "wotr_raw_menu": ([P], P),
        "wotr_raw_stage": ([P, C.POINTER(I64)], C.c_int),
        "wotr_raw_set_options": ([P, I64], C.c_int),
        "wotr_raw_fixed_reason": ([P], C.c_char_p),
        "wotr_raw_row": ([P, I64], P),
        "wotr_raw_chance_width": ([], I64),
        "wotr_raw_chance": ([P, F, N, N], C.c_int),
        # v4: the chance logger's alternatives on the native engine (box/cpp/chance_log.inc)
        "wotr_raw_chance_exact": ([P, I64, I64, P, C.POINTER(I64)], C.c_int),
        "wotr_raw_chance_why": ([P], C.c_char_p),
        "wotr_raw_chance_clear": ([P], C.c_int),
        "wotr_raw_chance_sample": ([P, I64, P], C.c_int),
        "wotr_raw_chance_roll": ([P, P], C.c_int),
        "wotr_raw_chance_leaf": ([P, I64, F, F, N, C.POINTER(C.c_double)], C.c_int),
        "wotr_raw_views": ([P, F, F, N, C.POINTER(C.c_double)], C.c_int),
        # 3a: hypothetical worlds of the afterstate search (box/cpp/search_worlds.inc)
        "wotr_raw_world_expand": ([P, I64, I64, I64, C.c_uint64, P, C.POINTER(I64)], C.c_int),
        "wotr_raw_world_why": ([P], C.c_char_p),
        "wotr_raw_world_leaf": ([P, I64, C.POINTER(C.c_double)], P),
        "wotr_raw_world_release": ([P], C.c_int),
        "wotr_raw_world_base": ([P, I64, C.c_uint64, P], P),
        "wotr_raw_world_sample": ([P, P, I64, P], P),
        "wotr_raw_world_roll": ([P, P], P),
        "wotr_raw_world_marker": ([P, C.POINTER(I64)], C.c_int),
        "wotr_raw_world_victory": ([P, C.POINTER(I64)], C.c_int),
        "wotr_raw_world_info": ([P, C.POINTER(I64)], C.c_int),
        "wotr_raw_world_copy": ([P], P),
        "wotr_raw_world_clone": ([P, P, I64], P),
        "wotr_raw_world_determinize": ([P, I64, C.POINTER(C.c_uint64)], C.c_int),
        "wotr_raw_world_apply_from": ([P, P, I64], C.c_int),
        # 2b: chance.kernel_outcomes natively (box/cpp/kernel_expand.inc over combat_kernel.inc)
        "wotr_raw_world_kernel": ([P, P, I64, C.POINTER(I64)], C.c_int),
        "wotr_raw_determinized": ([P, I64, C.c_uint64], P),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    if lib.wotr_raw_abi_version() != 4:
        raise RuntimeError("native raw ABI mismatch")
    ctx = Context()
    ctx.ai, ctx.lib, ctx.snapshot, ctx.random_stream = ai, lib, snapshot, random_stream
    ctx.tables = observation_tables.native_tables(ai)
    ctx.features = native_action_features.ActionFeaturesV53(ai)
    ctx.width = int(ai.get_feature_dim())
    ctx.action_width = int(ai.ACTION_ENCODING_DIM)
    # the stream every chance walk starts from: chance.enumerate_chance's clone_environment(env, 0) -> GameRandom(0)
    ctx.walk_random = random_stream.NativeRandom(game_python_state(0))
    ctx.dll = str(resident)
    ctx.dll_sha256 = sha256_file(resident)
    # NATIVE_RAW_DEV_ANY_DLL=1: development only (PGO training on the instrumented DLL); never set by the league
    if pin is not None and ctx.dll_sha256 != pin["dll_sha256"] and os.environ.get("NATIVE_RAW_DEV_ANY_DLL") != "1":
        raise RuntimeError("native raw generator: resident DLL differs from the pinned one")
    _CTX = ctx
    return ctx


def game_python_state(seed):
    """environment.GameRandom(seed).python (the numpy half is never drawn by the engine)."""
    import random
    from selfplay_actor_critic.environment import derived_seed
    return random.Random(derived_seed(seed, "game_python")).getstate()


def boundary_files():
    """The shared native_backend Python files this generator executes (read-only; pinned)."""
    return {name: BACKEND / name for name in ("bridge.py", "snapshot.py", "snapshot_schema.py", "random_stream.py",
                                              "observation_tables.py", "native_action_features.py", "rule_data.py")}


def verify(ai, pin):
    """Refuse unless the Python engine, the generator code it mirrors and the shared boundary are the pinned ones."""
    src = Path(ai.__file__).resolve().parent
    bad = [n for n, h in pin["engine_sha256"].items() if sha256_file(src / n) != h]
    # a NEW engine module is drift too (it may be imported by the rules the native engine mirrors)
    bad += [p.name for p in src.glob("*.py") if not p.name.startswith("test_") and p.name not in pin["engine_sha256"]]
    bad += [n for n, h in pin["data_sha256"].items() if sha256_file(src.parent / "data" / n) != h]
    bad += [n for n, h in pin["selfplay_sha256"].items() if sha256_file(ROOT / "selfplay_actor_critic" / n) != h]
    bad += [n for n, p in boundary_files().items() if sha256_file(p) != pin["boundary_sha256"].get(n)]
    # (TURNSEARCH copy: this folder's generator files are not the pinned ones - one export was added)
    # round 17 (C37 / T3): the Python code the native tools mirror or execute; a pin without the map is refused
    if "mirrored_sha256" not in pin:
        bad.append("mirrored_sha256 (pin predates round 17; rewrite it with tools/write_pin.py)")
    bad += [n for n, h in pin.get("mirrored_sha256", {}).items() if sha256_file(ROOT / n) != h]
    if bad:
        raise RuntimeError("native raw generator pin drift: " + ", ".join(sorted(bad)[:12]))


class NativeRawGame:
    """One raw game on the native engine; the PythonEnvironment interface valuedata.play uses, reduced to what it
    reads: next learned frame, its encodings, apply, end."""

    def __init__(self, ctx, ai, seed, *, max_turns=100, max_transitions=10000, episode_id=None, expose_forced=False):
        from selfplay_actor_critic.environment import PythonEnvironment
        # The opening exactly as the Python generator builds it (setup, decks, four draws under the game RNG).
        episode_id = episode_id or f"raw-{seed}"
        self.max_turns, self.max_transitions = max_turns, max_transitions
        opening = PythonEnvironment(ai, episode_id, seed, max_turns=max_turns, max_transitions=max_transitions)
        self.ctx, self.seed, self.episode_id = ctx, seed, episode_id
        self.samplers = opening.samplers
        self.first_turn = opening.first_turn
        lib = ctx.lib
        snap = ctx.snapshot.pack_snapshot(ai, opening.state)
        rng = ctx.random_stream.NativeRandom(opening.rng.python)
        try:
            self.handle = lib.wotr_raw_create(snap._pointer, ctx.tables._pointer, ctx.features.constants._pointer,
                                              rng._pointer, max_turns, max_transitions)
        finally:
            snap.close()
            rng.close()
        if not self.handle:
            raise NativeRefusal(self._error())
        if expose_forced and lib.wotr_raw_set_options(self.handle, 1):
            raise NativeRefusal(self._error())
        self.info = (C.c_int64 * 5)()
        self.forced = False
        self.final_turn = None
        self.capacity = 0
        self._grow(1024)
        self.obs = np.empty(ctx.width, np.float32)
        self.watch = np.empty(ctx.width, np.float32)
        self.vocab = []
        self.end = None
        self.side = self.phase = self.turn = self.rows = None

    def _error(self):
        return (self.ctx.lib.wotr_raw_last_error() or b"native raw game failure").decode("utf-8", "replace")

    def _grow(self, rows):
        if rows <= self.capacity:
            return
        self.capacity = max(rows, 2 * self.capacity)
        self.cand = np.empty((self.capacity, self.ctx.action_width), np.float32)
        self.regions = np.empty(self.capacity, np.int64)
        self.kinds = np.empty(self.capacity, np.int32)

    def next(self):
        rc = self.ctx.lib.wotr_raw_next(self.handle, self.info)
        if rc < 0:
            raise NativeRefusal(self._error())
        if rc == 0:
            self._finish()
            return False
        side, phase, turn, rows, forced = (int(v) for v in self.info)
        self.side, self.phase, self.turn, self.rows, self.forced = ("fp", "sh")[side], PHASES[phase], turn, rows, bool(forced)
        return True

    def encode(self):
        """(observation float32 [W], watch float32 [W], candidates float32 [N, A], regions int64 [N], kinds int64 [N]):
        fresh arrays (the caller keeps them)."""
        n = self.rows
        self._grow(n)
        F = C.POINTER(C.c_float)
        rc = self.ctx.lib.wotr_raw_encode(self.handle, self.obs.ctypes.data_as(F), self.watch.ctypes.data_as(F), self.ctx.width,
                                          self.cand.ctypes.data_as(F), self.ctx.action_width,
                                          self.regions.ctypes.data_as(C.POINTER(C.c_int64)),
                                          self.kinds.ctypes.data_as(C.POINTER(C.c_int32)), self.capacity)
        if rc:
            raise NativeRefusal(self._error())
        size = int(self.ctx.lib.wotr_raw_vocab_size(self.handle))
        while len(self.vocab) < size:
            self.vocab.append(self.ctx.lib.wotr_raw_vocab(self.handle, len(self.vocab)).decode("utf-8"))
        return (self.obs.copy(), self.watch.copy(), self.cand[:n].copy(), self.regions[:n].copy(),
                self.kinds[:n].astype(np.int64))

    def encode_view(self, side):
        """TURNSEARCH: the observation of ONE side ("fp" / "sh") of the open decision, float32 [W] - encode()'s first
        array when it is the mover's side, its second when it is the other's.  A fresh array."""
        out = np.empty(self.ctx.width, np.float32)
        if self.ctx.lib.wotr_raw_encode_view(self.handle, ("fp", "sh").index(side), out.ctypes.data_as(C.POINTER(C.c_float)),
                                             self.ctx.width):
            raise NativeRefusal(self._error())
        return out

    def encode_menu(self):
        """TURNSEARCH: (candidates float32 [N, A], regions int64 [N]) of the open decision - encode()'s third and fourth
        arrays.  Fresh arrays."""
        n = self.rows
        cand, regions = np.empty((n, self.ctx.action_width), np.float32), np.empty(n, np.int64)
        if self.ctx.lib.wotr_raw_encode_menu(self.handle, cand.ctypes.data_as(C.POINTER(C.c_float)), self.ctx.action_width,
                                             regions.ctypes.data_as(C.POINTER(C.c_int64)), n):
            raise NativeRefusal(self._error())
        return cand, regions

    def apply(self, index):
        if self.ctx.lib.wotr_raw_apply(self.handle, int(index)):
            raise NativeRefusal(self._error())

    def _finish(self):
        from selfplay_actor_critic.contracts import EpisodeEnd
        out = (C.c_int64 * 7)()
        reason = C.create_string_buffer(256)
        if self.ctx.lib.wotr_raw_end(self.handle, out, reason, 256) < 0:
            raise RuntimeError(self._error())
        status, winner, kind, committed, forced, learned, turn = (int(v) for v in out)
        self.forced_frames, self.learned, self.final_turn = forced, learned, turn
        if status == 1:
            self.end = EpisodeEnd(self.episode_id, "terminated", transition_count=committed, reason=None,
                                  winner=("fp", "sh")[winner], victory_route=ROUTES[kind])
        elif status == 2:
            self.end = EpisodeEnd(self.episode_id, "truncated", transition_count=committed,
                                  reason=reason.value.decode("utf-8"), winner=None, victory_route=None)
        else:
            raise RuntimeError("native raw game stopped without an accounted episode end")

    STAGES = {0: "opening", 1: "declaration", 2: "post_declaration", 3: "preaction", 4: "post_allocation", 5: "actions"}

    def stage_info(self):
        """(PythonEnvironment.stage, active seat index or -1, committed transitions) of the native game now."""
        out = (C.c_int64 * 3)()
        if self.ctx.lib.wotr_raw_stage(self.handle, out):
            raise NativeRefusal(self._error())
        return self.STAGES[int(out[0])], int(out[1]), int(out[2])

    def fixed_reason(self):
        """environment.PreparedDecision.fixed_reason of the open frame (None when there is none)."""
        text = self.ctx.lib.wotr_raw_fixed_reason(self.handle)
        if text is None:
            raise NativeRefusal(self._error())
        return text.decode("utf-8") or None

    def row(self, ai, index):
        """(die_idx, action_type, params) of one row of the open frame, decoded to the Python values ai uses."""
        snap = self.ctx.snapshot.NativeSnapshot(self.ctx.lib, self.ctx.lib.wotr_raw_row(self.handle, int(index)))
        try:
            die, kind, params = snap.unpack(ai)
        finally:
            snap.close()
        return die, kind, params

    def actions(self, ai):
        """The open frame's menu as ai.Action objects (the frame.actions the Python environment hands its readers)."""
        return [ai.Action(die_idx=d, action_type=k, params=p) for d, k, p in self.menu(ai)]

    def probabilities(self, ai):
        """probabilities.encode_probabilities(ai, state, side, actions) of the open frame, computed natively
        (box/cpp/chance_features.inc, ABI 3): the C++ port of the feature code on the native state, byte-identical to
        the Python environment's rows (tools/frame_check.py).  NATIVE_CHANCE_PYTHON=1 (diagnostics) uses the Python
        feature code on the decoded state instead (probabilities_python)."""
        if os.environ.get("NATIVE_CHANCE_PYTHON") == "1":
            return self.probabilities_python(ai)
        from selfplay_actor_critic.probability_schema import WIDTH
        out = np.zeros((self.rows, WIDTH), np.float32)
        if self.ctx.lib.wotr_raw_chance_width() != WIDTH:
            raise NativeRefusal("native chance features: schema width differs from probability_schema.WIDTH")
        if self.rows and self.ctx.lib.wotr_raw_chance(self.handle, out.ctypes.data_as(C.POINTER(C.c_float)), WIDTH, self.rows):
            raise NativeRefusal(self._error())
        return out

    def probabilities_python(self, ai):
        """probabilities.encode_probabilities(ai, state, side, actions) of the open frame, computed by the Python feature
        code itself on the native state decoded to a GameState (byte-identical to the Python environment's rows).
        Frames without a chance-bearing row get the all-zero rows encode_probabilities returns for them, without
        decoding anything."""
        from selfplay_actor_critic.probability_schema import WIDTH
        from selfplay_actor_critic.probabilities import encode_probabilities
        kinds = {self.vocab[int(k)] for k in self.kinds[:self.rows]}
        if not (kinds & CHANCE_KINDS):
            return np.zeros((self.rows, WIDTH), np.float32)
        side = ai.gs.Side.FREE_PEOPLES if self.side == "fp" else ai.gs.Side.SHADOW
        return encode_probabilities(ai, self.state(ai), side, self.actions(ai))

    # ---- v4: chance-log alternatives (box/cpp/chance_log.inc)
    def chance_exact(self, index, max_walks):
        """chance.enumerate_chance for row `index` of the open frame: (status 0 exact / 1 Unsupported / 2 replay on
        Python, leaves kept, walks, reason)."""
        out = (C.c_int64 * 3)()
        if self.ctx.lib.wotr_raw_chance_exact(self.handle, int(index), int(max_walks), self.ctx.walk_random._pointer, out):
            raise NativeRefusal(self._error())
        why = self.ctx.lib.wotr_raw_chance_why(self.handle)
        return int(out[0]), int(out[1]), int(out[2]), (why or b"").decode("utf-8", "replace")

    def chance_clear(self):
        if self.ctx.lib.wotr_raw_chance_clear(self.handle):
            raise NativeRefusal(self._error())

    def chance_sample(self, index, python_state):
        """Append one sampled clone (the row applied with the stream `python_state`); False when the engine refused."""
        rng = self.ctx.random_stream.NativeRandom(python_state)
        try:
            return self.ctx.lib.wotr_raw_chance_sample(self.handle, int(index), rng._pointer) == 0
        finally:
            rng.close()

    def chance_roll(self, python_state):
        """Append one rolled clone of the current position (PythonEnvironment._roll); False when the engine refused."""
        rng = self.ctx.random_stream.NativeRandom(python_state)
        try:
            return self.ctx.lib.wotr_raw_chance_roll(self.handle, rng._pointer) == 0
        finally:
            rng.close()

    def chance_leaf(self, i):
        """(probability, sh observation float32, fp observation float32, terminal) of kept leaf i."""
        sh, fp, out = np.empty(self.ctx.width, np.float32), np.empty(self.ctx.width, np.float32), (C.c_double * 2)()
        F = C.POINTER(C.c_float)
        if self.ctx.lib.wotr_raw_chance_leaf(self.handle, int(i), sh.ctypes.data_as(F), fp.ctypes.data_as(F), self.ctx.width, out):
            raise NativeRefusal(self._error())
        return float(out[0]), sh, fp, float(out[1])

    def views(self):
        """(sh observation, fp observation, terminal) of the live position."""
        sh, fp, out = np.empty(self.ctx.width, np.float32), np.empty(self.ctx.width, np.float32), (C.c_double * 1)()
        F = C.POINTER(C.c_float)
        if self.ctx.lib.wotr_raw_views(self.handle, sh.ctypes.data_as(F), fp.ctypes.data_as(F), self.ctx.width, out):
            raise NativeRefusal(self._error())
        return sh, fp, float(out[0])

    # ---- diagnostics (lockstep harness)
    def state(self, ai):
        snap = self.ctx.snapshot.NativeSnapshot(self.ctx.lib, self.ctx.lib.wotr_raw_state(self.handle))
        try:
            return snap.unpack(ai)
        finally:
            snap.close()

    def rng_state(self):
        r = self.ctx.random_stream.NativeRandom._adopt(self.ctx.random_stream.library(), self.ctx.lib.wotr_raw_random(self.handle))
        try:
            return r.getstate()
        finally:
            r.close()

    def menu(self, ai):
        snap = self.ctx.snapshot.NativeSnapshot(self.ctx.lib, self.ctx.lib.wotr_raw_menu(self.handle))
        try:
            return snap.unpack(ai)
        finally:
            snap.close()

    def close(self):
        if getattr(self, "handle", None):
            self.ctx.lib.wotr_raw_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    # ---- 3a: hypothetical worlds (box/cpp/search_worlds.inc)
    @classmethod
    def at(cls, ctx, ai, state, stream, stage, active, *, max_turns=100, max_transitions=10000):
        """TURNSEARCH: a game started at `state` (a Python GameState at a decision).  stage = the name or index of the
        turn stage, active = the seat index whose action it is (-1 none), stream = a CPython Random state tuple."""
        g = object.__new__(cls)
        g.max_turns, g.max_transitions = max_turns, max_transitions
        g.ctx, g.seed, g.episode_id = ctx, 0, "at"
        g.samplers, g.first_turn = None, None
        index = stage if isinstance(stage, int) else {v: k for k, v in cls.STAGES.items()}[stage]
        snap = ctx.snapshot.pack_snapshot(ai, state)
        rng = ctx.random_stream.NativeRandom(stream)
        try:
            g.handle = ctx.lib.wotr_raw_create_at(snap._pointer, ctx.tables._pointer, ctx.features.constants._pointer,
                                                  rng._pointer, max_turns, max_transitions, int(index), int(active))
        finally:
            snap.close()
            rng.close()
        if not g.handle:
            raise NativeRefusal(g._error())
        g.info = (C.c_int64 * 5)()
        g.forced, g.final_turn, g.capacity = False, None, 0
        g._grow(1024)
        g.obs = np.empty(ctx.width, np.float32)
        g.watch = np.empty(ctx.width, np.float32)
        g.vocab = []
        g.end = None
        g.side = g.phase = g.turn = g.rows = None
        return g

    @classmethod
    def adopt(cls, parent, handle):
        """A world handle returned by the engine (owned from now on), with the parent's game identity."""
        if not handle:
            raise NativeRefusal(parent._error())
        w = object.__new__(cls)
        w.ctx, w.seed, w.episode_id = parent.ctx, parent.seed, parent.episode_id
        w.samplers, w.first_turn, w.max_turns = parent.samplers, parent.first_turn, parent.max_turns
        w.handle = handle
        w.info = (C.c_int64 * 5)()
        w.forced, w.final_turn, w.capacity = False, None, 0
        w._grow(256)
        w.obs = np.empty(parent.ctx.width, np.float32)
        w.watch = np.empty(parent.ctx.width, np.float32)
        w.vocab = list(parent.vocab)
        w.end = None
        w.side = w.phase = w.turn = w.rows = None
        w.max_transitions = w.world_info()[5]
        return w

    def world_info(self):
        out = (C.c_int64 * 10)()
        if self.ctx.lib.wotr_raw_world_info(self.handle, out):
            raise NativeRefusal(self._error())
        return tuple(int(v) for v in out)

    def world_expand(self, index, max_walks, determinize_side, determinize_seed):
        """chance.enumerate_chance for row `index` of the open frame from a (determinized) clone: (status, [(p, world)],
        walks, reason)."""
        out = (C.c_int64 * 3)()
        lib = self.ctx.lib
        if lib.wotr_raw_world_expand(self.handle, int(index), int(max_walks), int(determinize_side),
                                     int(determinize_seed), self.ctx.walk_random._pointer, out):
            raise NativeRefusal(self._error())
        why = (lib.wotr_raw_world_why(self.handle) or b"").decode("utf-8", "replace")
        leaves = []
        try:
            if out[0] == 0:
                p = C.c_double()
                for i in range(int(out[1])):
                    leaves.append((float(0.0), NativeRawGame.adopt(self, lib.wotr_raw_world_leaf(self.handle, i, C.byref(p)))))
                    leaves[-1] = (float(p.value), leaves[-1][1])
        finally:
            lib.wotr_raw_world_release(self.handle)
        return int(out[0]), leaves, int(out[2]), why

    def world_kernel(self, base, index):
        """chance.kernel_outcomes(world = base or self, frame = self's open frame, choice = index): (status 0 leaves /
        1 None / 2 outside the port, [(p, world)], reason)."""
        out = (C.c_int64 * 2)()
        lib = self.ctx.lib
        if lib.wotr_raw_world_kernel(self.handle, base.handle if base is not None else None, int(index), out):
            raise NativeRefusal(self._error())
        why = (lib.wotr_raw_world_why(self.handle) or b"").decode("utf-8", "replace")
        leaves = []
        try:
            if out[0] == 0:
                p = C.c_double()
                for i in range(int(out[1])):
                    world = NativeRawGame.adopt(self, lib.wotr_raw_world_leaf(self.handle, i, C.byref(p)))
                    leaves.append((float(p.value), world))
        finally:
            lib.wotr_raw_world_release(self.handle)
        return int(out[0]), leaves, why

    def world_base(self, determinize_side, determinize_seed):
        """chance.expand's base: clone_environment(self, 0) + determinize."""
        return NativeRawGame.adopt(self, self.ctx.lib.wotr_raw_world_base(self.handle, int(determinize_side), int(determinize_seed),
                                                                          self.ctx.walk_random._pointer))

    def world_sample(self, base, index, python_state):
        rng = self.ctx.random_stream.NativeRandom(python_state)
        try:
            return NativeRawGame.adopt(self, self.ctx.lib.wotr_raw_world_sample(self.handle, base.handle if base is not None else None,
                                                                                int(index), rng._pointer))
        finally:
            rng.close()

    def world_copy(self):
        """An exact copy (state, own stream, bookkeeping) of this game / world."""
        return NativeRawGame.adopt(self, self.ctx.lib.wotr_raw_world_copy(self.handle))

    def world_clone(self, python_state, extra_transitions):
        """search.clone_environment(self, seed, extra_transitions) with the stream GameRandom(seed).python."""
        rng = self.ctx.random_stream.NativeRandom(python_state)
        try:
            return NativeRawGame.adopt(self, self.ctx.lib.wotr_raw_world_clone(self.handle, rng._pointer, int(extra_transitions)))
        finally:
            rng.close()

    def world_determinize(self, side, generator):
        """search.determinize_state(self's state, ..., generator, ai) in place, drawing from (and advancing) the numpy
        PCG64 Generator `generator` itself."""
        state = generator.bit_generator.state
        if state.get("bit_generator") != "PCG64":
            raise NativeRefusal("native determinize needs a PCG64 generator")
        M = (1 << 64) - 1
        s, inc = int(state["state"]["state"]), int(state["state"]["inc"])
        buf = (C.c_uint64 * 6)(s >> 64, s & M, inc >> 64, inc & M, int(state["has_uint32"]), int(state["uinteger"]))
        if self.ctx.lib.wotr_raw_world_determinize(self.handle, int(side), buf):
            raise NativeRefusal(self._error())
        state["state"]["state"] = (int(buf[0]) << 64) | int(buf[1])
        state["state"]["inc"] = (int(buf[2]) << 64) | int(buf[3])
        state["has_uint32"], state["uinteger"] = int(buf[4]), int(buf[5])
        generator.bit_generator.state = state

    def world_apply_from(self, source, index):
        """PythonEnvironment.apply(source's open frame, index) on this world."""
        if self.ctx.lib.wotr_raw_world_apply_from(self.handle, source.handle, int(index)):
            raise NativeRefusal(self._error())

    def world_roll(self, python_state):
        rng = self.ctx.random_stream.NativeRandom(python_state)
        try:
            return NativeRawGame.adopt(self, self.ctx.lib.wotr_raw_world_roll(self.handle, rng._pointer))
        finally:
            rng.close()

    def world_marker(self):
        out = (C.c_int64 * 2)()
        if self.ctx.lib.wotr_raw_world_marker(self.handle, out):
            raise NativeRefusal(self._error())
        return bool(out[0]), int(out[1])

    def world_victory(self):
        """PythonEnvironment._victory(): the winner tag when the game ends here (the world is ended), else None."""
        out = (C.c_int64 * 2)()
        if self.ctx.lib.wotr_raw_world_victory(self.handle, out):
            raise NativeRefusal(self._error())
        return ("fp", "sh")[int(out[1])] if out[0] else None

    def determinized_state(self, ai, side, seed):
        snap = self.ctx.snapshot.NativeSnapshot(self.ctx.lib, self.ctx.lib.wotr_raw_determinized(self.handle, int(side), int(seed)))
        try:
            return snap.unpack(ai)
        finally:
            snap.close()
