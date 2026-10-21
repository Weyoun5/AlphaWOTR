"""Python orchestrator for the native search controller (SHARED_ABI.md section 4).

`native_mcts_search(ai, state, side, fp_net, sh_net, ...)` mirrors ai.mcts_search
17331-17884 outside the tree: the same four decorators (RNG stream swap, observer
begin/finish, lattice/companion scopes, failure scope), the same entry checks,
a controller advance/bridge/submit loop instead of _simulate_from, and the same
visit_info / select_root_action / make_evidence tail. Every milestone-1
unsupported branch (ABI section 0) is refused by name; nothing falls back.

Accepted roots: a Python GameState (packed once with snapshot.pack_snapshot), a
NativeSnapshot, or an object exposing `.native` (driver State / adapter
NativeState). The controller clones the root itself; the input is never mutated.

Milestone 2 (SHARED_ABI section 9, FINAL 9.3-9.5, 2026-09-09):
* 9.4 root noise: advance status 2 -> `root_priors` hand-out -> `priors_clean` float32
  capture -> `np.random.dirichlet([0.3]*K)` on the search NumPy stream ->
  `_apply_root_noise` arithmetic (`apply_root_noise_numeric`) -> `set_root_priors`.
  add_noise is accepted iff the library has the 9.4 exports.
* 9.5 combat distribution: op-4 requests are serviced by `service_combat_request`
  (frozen `ai._kernel_dist`, `plan_conf_case`, WCtK side, reply dict encoded with
  snapshot.encode_graph) with `observer.exact_round` on a node facade at service time.
* 9.3 observer replay: after the controller completes, `observer_events` are replayed
  into `ai._COMBAT_FORECAST_OBSERVER` (`replay_observer_events`) before the decorator's
  `finish_search`; blinded-root / observer-window refusals lift with that capability.
* Temperature > 0 uses the source's own `select_root_action` inside `build_visit_info`.
* 9.6 tree reuse: `_TREE_CACHE['child']` holds a `TreeSession` owning the detached native
  subtree; at entry the cached session is passed to `create_reusing` (entry flush of its
  edge catalog) with the adopt decision of `reuse_decision` (17399-17431, sig hash + own
  hand from the `search_metadata` projection); hits/misses from `reused`; `stash_tree`
  detaches the chosen child after selection under the source's condition (17843-17859).

Per-search budget guard (package Y2, loop_resilience, 2026-09-11; incident: supervisor_r3_native
batch 4 episode (3,5) - ONE search never finished, 42 GB private bytes, heartbeat alive):
`SearchBudgetGuard.before_advance` runs BETWEEN controller advances (never inside one) and raises
`SearchBudgetExceeded` (a RuntimeError carrying the root state, the search kwargs, the controller
counters and a bounded export tail) when the wall time since controller creation, the process'
private-bytes growth since then, or the advance count exceeds WOTR_SEARCH_MAX_SECONDS (300) /
WOTR_SEARCH_MAX_PRIVATE_GB (4.0 growth) / WOTR_SEARCH_MAX_ADVANCES (20 x num_simulations); 0
disables a check.  When it does not fire the search is untouched: no RNG draw, no controller call,
no change to any decision.  The runner records the raised game as status=error (fixture) and the
batch continues.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import importlib
import importlib.util
import inspect
import math
import os
import random
import sys
import threading
import time
import numpy as np

OWN_PACKAGE = Path(__file__).resolve().parent.parent
CORE = OWN_PACKAGE.parent
HERE = CORE / "inference"
ROOT = HERE.parents[2]
for _path in (OWN_PACKAGE, CORE / "own_value_numeric_sep13"):
    sys.path.insert(0, str(_path))
for _p in (str(HERE), str(ROOT), str(ROOT / 'experiments/search_adapter_contract'), str(ROOT / 'experiments/forced_hunt')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import snapshot, random_stream, observation_tables                       # noqa: E402
from native_action_features import ActionFeatures                        # noqa: E402
from menu_view import MenuQuery                                          # noqa: E402
import controller_binding as binding                                     # noqa: E402
from numeric_bridge import Request                                      # noqa: E402
from root_numeric import RootNumericInferenceBridge
from own_binding import OwnController

EXPECTED_DECORATORS = ('@search_collection_scope', '@_companion_search_cache.search_scope',
                       '@_search_lattice.search_scope', '@_search_failure_scope')
PROFILE_CODES = {'fp': {None: 0, 'ring': 1, 'military': 2, 'mixed': 3},
                 'sh': {None: 0, 'corruption': 1, 'military': 2, 'mixed': 3}}
REWARD_CODES = {'predecision_profile_v1': 2}
OBSERVER_PHASES = ('attacker_combat_card', 'defender_combat_card', 'combat_effect_choice')


class Unsupported(NotImplementedError):
    """A milestone-1 unsupported branch (SHARED_ABI.md section 0); names the branch."""


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Resources:
    """Python-owned table handles; they must outlive every controller (ABI section 1)."""

    def __init__(self, ai):
        self.ai = ai
        # Controller exports are checked when a Controller is created (binding.configure);
        # table construction and root materialization work on any resident library.
        self.lib = snapshot.library()
        random_stream.library()
        self.tables = observation_tables.native_tables(ai)
        self.features = ActionFeatures(ai)
        self.menu = MenuQuery(ai)
        forcing = _load('native_search_core_inference_forcing', 'experiments/forced_hunt/native_forcing.py')
        self.chance = forcing.chance_table(ai, self.lib)
        self.combat_facts = _load('native_search_core_inference_combat_facts',
                                  'experiments/combat_search_adapter/native_facts.py')
        # 9.6: _state_sig tuple + own hand of a native state, without unpacking (hash-equal to
        # ai._state_sig; own_hand == tuple(sorted(map(str, hands[side].hand)))).
        self.search_metadata = _load('native_search_core_inference_search_metadata',
                                     'experiments/search_metadata/native_queries.py')
        for handle in (self.menu.tables, self.chance):
            if handle._lib is not self.lib:
                raise RuntimeError('Table handle does not belong to the resident native library')
        if self.features.lib is not self.lib:
            raise RuntimeError('Action feature tables do not belong to the resident native library')


_RESOURCES = {}
_RESOURCE_LOCK = threading.Lock()


def resources(ai):
    with _RESOURCE_LOCK:
        result = _RESOURCES.get(id(ai))
        if result is None:
            result = _RESOURCES[id(ai)] = Resources(ai)
        return result


class NativeRoot:
    """Minimal root wrapper for a NativeSnapshot input: `.native`, `pending_combat`, `turn`.

    The observer's begin_search reads pending_combat/turn; the refusal checks need
    pending_combat. Facts come from the reviewed combat projection, never a GameState.
    """
    __slots__ = ('ai', 'native', '_facts', '_res')

    def __init__(self, ai, native, res):
        self.ai, self.native, self._facts, self._res = ai, native, None, res

    def facts(self):
        if self._facts is None:
            self._facts = self._res.combat_facts.query(self.ai, self.native)
        return self._facts

    @property
    def pending_combat(self):
        values = self.facts()
        if 'pending_combat' not in values:
            raise AttributeError('pending_combat')
        return values['pending_combat']

    def __getattr__(self, name):
        if name in ('turn', 'turn_number', '_last_round_call'):
            values = self.facts()
            if name in values:
                return values[name]
            raise AttributeError(name)
        raise AttributeError(name)


@dataclass
class SearchSession:
    """Debug/oracle record of one orchestrated search (never in timed runs)."""
    trace: bool = False
    with_states: bool = False
    result: object = None
    requests: list = field(default_factory=list)
    root: object = None
    root_descriptors: list = None
    counters: dict = None
    export: dict = None
    rng_state: tuple = None
    budget: int = None
    simulations_run: int = None
    struct_sizes: dict = field(default_factory=lambda: dict(binding.STRUCT_SIZES))
    capabilities: dict = None
    root_noise: dict = None        # {'children', 'clean', 'noise', 'noised', 'exempt'} when the step ran
    combat_requests: list = field(default_factory=list)   # 9.5 service summaries in request order
    observer_events: list = None   # 9.3 decoded event list (as replayed)
    deferred_exact_rounds: int = 0  # exact_round calls replayed after completion (root menu unavailable earlier)
    reuse: dict = None             # 9.6: entry decision, reused flag, hits/misses, stash record
    budget_guard: dict = None      # Y2: resolved limits + advances / wall / private-bytes growth of the search


def verify_decorator_chain(ai):
    """The live source must still carry exactly the four decorators, in this order."""
    lines, _ = inspect.getsourcelines(ai.mcts_search)
    found = tuple(line.strip() for line in lines[:4])
    if found != EXPECTED_DECORATORS:
        raise RuntimeError('mcts_search decorator chain changed: %r' % (found,))
    return (ai.search_collection_scope, ai._companion_search_cache.search_scope,
            ai._search_lattice.search_scope, ai._search_failure_scope)


def refuse_unsupported(ai, add_noise, hot_regions, bonus_sims, determinize, noise_capable=False, reuse_capable=False):
    rc = importlib.import_module('reward_contract')
    if add_noise and not noise_capable:
        raise Unsupported('add_noise (Dirichlet root noise) is refused: the controller does not expose the '
                          'SHARED_ABI 9.4 exports (root_priors/set_root_priors); the Python step is ported and waits for them')
    # Production path: the episode loop passes a NON-EMPTY hot_regions set at every main decision
    # after the first (ai.py 19085/19478/19422-19425) while bonus_sims comes from config (19008,
    # default 0). The source reads them only under `if hot_regions and bonus_sims > 0` (17773),
    # so hot_regions alone (None, empty or not) is a no-op and is accepted and ignored here.
    if bonus_sims and int(bonus_sims) > 0:
        raise Unsupported('bonus_sims=%r focused search (hot-zone bonus simulations, ai.py 17773-17790) is refused; '
                          'hot_regions without bonus_sims is a source no-op and is accepted' % (bonus_sims,))
    if ai._OPPONENT_POLICY_MODE != 'own_value_reply_v0':
        raise Unsupported('Isolated own-value controller requires own_value_reply_v0')
    if ai._TREE_REUSE or ai._TREE_CACHE.get('child') is not None or ai._LEAF_BATCH != 1:
        raise Unsupported('Own-value v0 requires fresh trees and sequential leaves')
    if not reuse_capable:
        if ai._TREE_REUSE:
            raise Unsupported('tree reuse (_TREE_REUSE true) is refused until the controller exports the SHARED_ABI 9.6 '
                              'session symbols')
        if ai._TREE_CACHE.get('child') is not None:
            raise Unsupported('tree reuse (a cached child is present) is refused until the controller exports the '
                              'SHARED_ABI 9.6 session symbols')
    if ai._CONF_REF != 'abs':
        raise Unsupported('milestone 1 refuses _CONF_REF=%r (only "abs")' % (ai._CONF_REF,))
    if rc.current_contract() not in REWARD_CODES:
        raise Unsupported('milestone 1 refuses reward contract %r' % (rc.current_contract(),))
    if float(ai._PHI_RINGS_RHO) != 0.0:
        raise Unsupported('milestone 1 refuses _PHI_RINGS_RHO=%r (must be 0.0)' % (ai._PHI_RINGS_RHO,))
    for key in ('fp', 'sh'):
        if ai.PLAYER_PROFILES.get(key) not in PROFILE_CODES[key]:
            raise Unsupported('unknown %s profile %r' % (key, ai.PLAYER_PROFILES.get(key)))
    if int(ai._LEAF_BATCH) < 1:
        raise Unsupported('_LEAF_BATCH must be >= 1')
    if not determinize:
        # Source entry check (17348-17349), kept verbatim for identical errors.
        raise ValueError('Learned opponent policy requires joint hidden-hand sampling')


def refuse_root(ai, state, side, observer_capable=False):
    """Blinded roots and active CombatForecastObserver roots are refused until the library
    exports the 9.3 observer events (their replay is what makes the observer exact)."""
    if isinstance(state, NativeRoot) and 'pending_combat' not in state.facts():
        raise Unsupported('native root facts carry no pending_combat; blinding/observer state cannot be verified')
    try:
        pc = getattr(state, 'pending_combat', None) or {}
    except Exception as error:
        raise Unsupported('cannot read the root pending_combat to verify blinding: %s' % error)
    if observer_capable:
        return
    if ai._hidden_commit_card(state, side) is not None:
        raise Unsupported('blinded roots (_hidden_commit_card is not None) are refused until the controller '
                          'exports the SHARED_ABI 9.3 observer events')
    if pc.get('phase') in OBSERVER_PHASES:
        raise Unsupported('active CombatForecastObserver roots (phase %r) are refused until the controller '
                          'exports the SHARED_ABI 9.3 observer events' % (pc.get('phase'),))


def build_config(ai, root_side_index, num_simulations, c_puct, determinize, trace, add_noise=False):
    cfg = binding.WotrSearchConfig()
    cfg.struct_size = binding.C.sizeof(binding.WotrSearchConfig)
    cfg.root_side = int(root_side_index)
    cfg.num_simulations = int(num_simulations)
    cfg.sims_menu_scale = int(bool(ai._SIMS_MENU_SCALE))
    cfg.sims_per_child = int(ai._SIMS_PER_CHILD)
    cfg.sims_floor = int(ai._SIMS_FLOOR)
    cfg.c_puct = float(c_puct)
    cfg.leaf_batch = int(ai._LEAF_BATCH)
    cfg.turn_horizon = int(ai._TURN_HORIZON)
    cfg.max_children = int(ai.MCTS_MAX_CHILDREN or 0)
    cfg.chance_k_max = int(ai._CHANCE_K_MAX)
    cfg.kernel_k_max = int(ai._KERNEL_K_MAX)
    cfg.chance_draws = int(ai._CHANCE_DRAWS)
    cfg.free_cands = int(bool(ai._FREE_CANDS))
    cfg.free_cands_n = int(ai._FREE_CANDS_N)
    cfg.free_force = int(ai._FREE_FORCE)
    cfg.opponent_policy_mode = 2
    cfg.opponent_policy_epsilon = float(ai._OPPONENT_POLICY_EPSILON)
    cfg.conf_leaf = int(bool(ai._CONF_LEAF))
    cfg.conf_track = int(bool(ai._CONF_TRACK))
    cfg.conf_ref = 0
    cfg.determinize = int(bool(determinize))
    # add_noise reaches the controller only when refuse_unsupported accepted it (capability
    # present); the controller must then pause with status 2 after root expansion.
    cfg.add_noise = int(bool(add_noise))
    cfg.overlay_assert = int(bool(ai.OVERLAY_ASSERT))
    cfg.reward_contract = REWARD_CODES[importlib.import_module('reward_contract').current_contract()]
    cfg.fp_profile = PROFILE_CODES['fp'][ai.PLAYER_PROFILES.get('fp')]
    cfg.sh_profile = PROFILE_CODES['sh'][ai.PLAYER_PROFILES.get('sh')]
    cfg.phi_rings_rho = float(ai._PHI_RINGS_RHO)
    cfg.trace = int(bool(trace))
    return cfg


def config_dict(cfg):
    return {name: getattr(cfg, name) for name, _ in binding.WotrSearchConfig._fields_}


# ---------------------------------------------------------------------------------------------
# Root Dirichlet noise (source 17489-17495 + _apply_root_noise 13773-13811), node-free port.
# ---------------------------------------------------------------------------------------------
NOISE_ALPHA = 0.3      # np.random.dirichlet([0.3] * len(root.children))
NOISE_FRAC = 0.25      # _apply_root_noise(children, noise, frac=0.25)


def dirichlet_root_noise(k):
    """The source's single NumPy draw (17494): `np.random.dirichlet([0.3] * k)`, float64 [k].

    Must run inside the decorated search (rng_streams.search_scope swapped the search
    NumPy stream in) after root expansion and before the first simulation; the
    controller consumes no NumPy randomness, so the draw sits at the source's point.
    """
    k = int(k)
    if k < 1:
        raise ValueError('root noise needs at least one child')
    return np.random.dirichlet([NOISE_ALPHA] * k)


def apply_root_noise_numeric(ai, priors, noise, free_mask=None, frac=NOISE_FRAC, exempt_free=None):
    """ai._apply_root_noise over K numeric child priors (no node objects).

    priors: the controller's K child priors as Python floats / float64 (the doubles the
    controller stores: `priors.tolist()` values or prune renormalization quotients);
    noise: float64 [K] from dirichlet_root_noise; free_mask: bool [K] (child action_type
    in ai._FREE_ACTION_TYPES), required only when the exemption is active.
    Returns (noised priors as Python floats, exempt mask) with the SAME arithmetic:
    * exemption inactive (default WOTR_FREE_NOISE_EXEMPT=0, or no/only free arms):
      `p = (1.0 - frac) * p + frac * noise[i]` per child, NO renormalization;
    * exemption active: noise zeroed on exempt arms, `noise / float(noise.sum())` when
      positive, non-exempt `p = (1.0 - frac) * p + frac * float(noise[i])`, then
      `tot = float(sum(all priors))` (sequential) and `p / tot` for every arm when > 0.
    """
    if exempt_free is None:
        exempt_free = ai._FREE_NOISE_EXEMPT
    values = [float(p) for p in priors]
    k = len(values)
    ex = np.zeros(k, dtype=bool)
    if exempt_free:
        if free_mask is None:
            raise Unsupported('WOTR_FREE_NOISE_EXEMPT needs the per-child free-action mask from the controller')
        mask = np.asarray(free_mask, dtype=bool)
        if mask.shape != (k,):
            raise ValueError('free mask length differs from the prior vector')
        ex[:] = mask
        if ex.all():
            ex[:] = False      # degenerate all-free menu: noise as usual
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape != (k,):
        raise ValueError('noise length differs from the prior vector')
    if not ex.any():
        # exemption inactive: EXACT legacy arithmetic (byte-preserved default path, no renormalization)
        out = [float((1.0 - frac) * values[i] + frac * noise[i]) for i in range(k)]
        return out, ex
    noise = noise.copy()
    noise[ex] = 0.0
    s = float(noise.sum())
    if s > 0.0:
        noise = noise / s
    out = list(values)
    for i in range(k):
        if not ex[i]:
            out[i] = (1.0 - frac) * values[i] + frac * float(noise[i])
    tot = float(sum(out))
    if tot > 0.0:
        out = [p / tot for p in out]
    return [float(p) for p in out], ex


def root_noise_step(ai, clean_priors, free_mask=None):
    """Draw + apply exactly as the source does between root expansion and the budget
    (17493-17495): returns (noised Python floats, noise float64 [K], exempt mask)."""
    noise = dirichlet_root_noise(len(clean_priors))
    noised, ex = apply_root_noise_numeric(ai, clean_priors, noise, free_mask)
    return noised, noise, ex


# ---------------------------------------------------------------------------------------------
# Combat distribution request (SHARED_ABI 9.5, Python side) + observer facades (9.3/9.5).
# ---------------------------------------------------------------------------------------------
EXCHANGE_PAYLOAD_FIELDS = ('attacker_card', 'defender_card', 'is_siege', 'city', 'shine', 'choices',
                           'defer_casualties', 'attacker_region', 'defender_region', 'saruman_in_play')
BATTLE_KEYS = ('attacker_region', 'defender_region', 'attacker_side', 'defender_side')


def exchange_context_payload(ctx):
    """The 12-entry metadata layout of experiments/kernel_context (what wotr_snapshot_kernel_context
    already produces): two army rows then the ten scalar/tuple fields in dataclass order."""
    def army(a):
        return (a.regulars, a.elites, a.nation, a.elite_nation,
                tuple((x.kind, x.name, x.level, x.leadership, x.nation) for x in a.leaders),
                a.elites2, a.elite2_nation)
    return (army(ctx.attacker), army(ctx.defender)) + tuple(getattr(ctx, name) for name in EXCHANGE_PAYLOAD_FIELDS)


def exchange_context_from_payload(ai, values):
    """Rebuild the frozen combat_dist dataclasses from a decoded payload (kernel_context/native_query.py)."""
    from combat_dist import ArmySpec, LeaderSpec, ExchangeContext
    values = tuple(values)
    if len(values) != 2 + len(EXCHANGE_PAYLOAD_FIELDS):
        raise ValueError('ExchangeContext payload must have %d entries' % (2 + len(EXCHANGE_PAYLOAD_FIELDS)))

    def army(row):
        row = tuple(row)
        if len(row) != 7:
            raise ValueError('ArmySpec row must have 7 entries')
        return ArmySpec(regulars=row[0], elites=row[1], nation=row[2], elite_nation=row[3],
                        leaders=tuple(LeaderSpec(*tuple(x)) for x in row[4]), elites2=row[5], elite2_nation=row[6])
    scalars = dict(zip(EXCHANGE_PAYLOAD_FIELDS, values[2:]))
    scalars['choices'] = tuple(tuple(item) for item in scalars['choices'])
    return ExchangeContext(army(values[0]), army(values[1]), **scalars)


def wctk_side(ai, ctx):
    """We Come to Kill side exactly as _kernel_expand_combat 15684-15695: card name match, then
    the engine's own stage0 cancellation check."""
    import combat_dist as cd
    side = None
    for name, who in ((ctx.attacker_card, 'attacker'), (ctx.defender_card, 'defender')):
        if name and 'we come to kill' in name.lower():
            side = who
    if side is not None:
        p0, _c0 = cd.stage0(ctx)
        sp0 = p0.atk if side == 'attacker' else p0.dfn
        if 'we come to kill' not in sp0.name:
            side = None
    return side


def combat_distribution_reply(ai, ctx):
    """9.5 reply dict from the frozen `ai._kernel_dist(ctx, want_witness=True)` (memo + combat_lookup +
    RNG shield, unchanged), `combat_dist.plan_conf_case(ctx)` and the WCtK side. Returns (reply, dist).

    reply = {'outcomes': [19-tuple obs, ...] in dist.outcomes insertion order,
             'masses': [float64 ...], 'witness': [plan as list of int | (int, int) ...],
             'plan_reprs': [repr(plan tuple) ...], 'conf': bool, 'wctk_side': None|'attacker'|'defender',
             'ctx_repr': repr(ctx)}.  A missing witness plan (never produced by exact_distribution /
    combat_lookup, which cover every outcome) is encoded as None so the controller can take its
    `no_witness` fallback. Consumes no search RNG (asserted in tests)."""
    import combat_dist as cd
    if not ctx.defer_casualties:
        raise ValueError('witness plans are defer-mode only (in-tree rounds)')
    dist = ai._kernel_dist(ctx, want_witness=True)
    witness = dist.witness if dist.witness is not None else {}
    outcomes, masses, plans, reprs = [], [], [], []
    for obs, mass in dist.outcomes.items():
        plan = witness.get(obs)
        outcomes.append(tuple(obs))
        masses.append(float(mass))
        plans.append(None if plan is None else [tuple(x) if isinstance(x, tuple) else int(x) for x in plan])
        reprs.append(repr(plan))
    reply = {'outcomes': outcomes, 'masses': masses, 'witness': plans, 'plan_reprs': reprs,
             'conf': bool(cd.plan_conf_case(ctx)), 'wctk_side': wctk_side(ai, ctx), 'ctx_repr': repr(ctx)}
    return reply, dist


def reply_snapshot(ai, reply):
    """The 9.5 reply as an owned NativeSnapshot (`NativeSnapshot.from_graph(encode_graph(ai, dict))`)."""
    return snapshot.NativeSnapshot.from_graph(snapshot.encode_graph(ai, reply))


def battle_state_facade(ai, battle, phase=None):
    """A state facade whose `pending_combat`/`turn` make observer.battle() return the source's
    value for a raw (turn, attacker_region, defender_region, attacker_side, defender_side,
    rounds_fought) tuple; None -> no battle (observer.battle returns None)."""
    if battle is None:
        return SimpleNamespace(pending_combat={} if phase is None else {'phase': phase}, turn=None)
    battle = tuple(battle)
    if len(battle) != 6:
        raise RuntimeError('battle tuple must have 6 entries; got %r' % (battle,))
    pc = {'attacker_region': battle[1], 'defender_region': battle[2], 'attacker_side': battle[3],
          'defender_side': battle[4], 'rounds_fought': battle[5]}
    if phase is not None:
        pc['phase'] = phase
    return SimpleNamespace(pending_combat=pc, turn=battle[0])


def node_facade(ai, root_actions, root_branch_row, side=None, state=None, parent_state=None):
    """Node chain root <- branch(action = root child at root_branch_row) <- parent <- node so that
    CombatForecastObserver.root_branch(node) == action_key(root child) and node.parent.state is
    the parent facade; root_branch_row None -> the node IS the root (root_branch -> None)."""
    if root_branch_row is None:
        return SimpleNamespace(parent=None, action=None, side=side, state=state)
    row = int(root_branch_row)
    if not 0 <= row < len(root_actions):
        raise RuntimeError('root_branch_row %d outside the root menu (%d rows)' % (row, len(root_actions)))
    root = SimpleNamespace(parent=None, action=None, side=None, state=None)
    branch = SimpleNamespace(parent=root, action=root_actions[row], side=None, state=None)
    parent = SimpleNamespace(parent=branch, action=None, side=None, state=parent_state)
    return SimpleNamespace(parent=parent, action=None, side=side, state=state)


def side_member(ai, value):
    if isinstance(value, ai.Side):
        return value
    return ai.Side.FREE_PEOPLES if int(value) == 0 else ai.Side.SHADOW


def observer_exact_round(ai, observer, root_actions, payload, ctx, dist):
    """observer.exact_round(node, ctx, dist) exactly as _kernel_expand_combat 15635-15636, on a facade
    built from the 9.5 payload ('root_branch_row', 'parent_battle')."""
    node = node_facade(ai, root_actions, payload.get('root_branch_row'),
                       parent_state=battle_state_facade(ai, payload.get('parent_battle')))
    observer.exact_round(node, ctx, dist)


def card_action_rows(rows):
    """Facade actions for the observer's card_action(): rows = [(action_type, card_id|None), ...]."""
    actions = []
    for row in rows:
        action_type, card_id = tuple(row)
        params = {} if card_id is None else {'card_id': card_id}
        actions.append(SimpleNamespace(action_type=action_type, params=params, die_idx=None))
    return actions


def replay_observer_events(ai, observer, events, root_actions, root_state, root_side):
    """9.3: replay the controller's ordered event list into the observer (it only accumulates)."""
    replayed = {'hidden_commit': 0, 'card_policy': 0}
    for event in events:
        event = tuple(event)
        kind = event[0]
        if kind == 'hidden_commit':
            if len(event) != 2:
                raise RuntimeError('hidden_commit event must be (kind, mixture); got %d fields' % len(event))
            mixture = [(item[0], item[1]) for item in (tuple(x) for x in event[1])]
            observer.hidden_commit(root_state, root_side, mixture)
            replayed[kind] += 1
        elif kind == 'card_policy':
            if len(event) != 8:
                raise RuntimeError('card_policy event must have 8 fields; got %d' % len(event))
            _kind, _node_id, side, root_branch_row, phase, battle, rows, probabilities = event
            state = battle_state_facade(ai, battle, phase=phase)
            node = node_facade(ai, root_actions, root_branch_row, side=side_member(ai, side), state=state)
            observer.card_policy(node, card_action_rows(rows), [float(p) for p in probabilities])
            replayed[kind] += 1
        else:
            raise RuntimeError('Unknown observer event kind %r' % (kind,))
    return replayed


class RootActions:
    """Root Actions materialized once from the controller's root menu; available lazily (the menu
    may be refused before completion) and definitely after the search completes."""

    def __init__(self, ai, res, ctl, native, side_index):
        self.ai, self.res, self.ctl, self.native, self.side_index = ai, res, ctl, native, side_index
        self.actions = None
        self.descriptors = None
        self.action_encs = None
        self.failure = None

    def try_materialize(self, count):
        """K is known only from the root view (after status 1): before completion the root menu
        is not readable (the controller's root_menu requires completion and the descriptor query
        needs K), so service-time callers find `actions is None` and defer (see
        service_combat_request); the deferral is exact for the observer's keyed `exact` dict."""
        if self.actions is not None:
            return True
        try:
            with self.ctl.root_menu() as menu:
                self.actions, self.descriptors = materialize_root_actions(self.ai, self.res.menu, menu, count)
                self.action_encs, _ = self.res.features.encode(self.native, menu, self.side_index, count)
            return True
        except Exception as error:
            self.failure = '%s: %s' % (type(error).__name__, error)
            return False

    def materialize(self, count):
        if not self.try_materialize(count):
            raise RuntimeError('Root menu unavailable after completion: %s' % self.failure)
        if len(self.actions) != count:
            raise RuntimeError('Root menu has %d rows; expected %d' % (len(self.actions), count))
        return self.actions


def service_combat_request(ai, ctl, view, roots, session=None, deferred=None):
    """Answer one op-4 COMBAT_DIST request (SHARED_ABI 9.5) from the frozen kernel functions."""
    payload = ctl.request_payload(ai)
    ctx = exchange_context_from_payload(ai, payload['context'])
    reply, dist = combat_distribution_reply(ai, ctx)
    observer = ai._COMBAT_FORECAST_OBSERVER
    deferred_here = False
    if observer is not None:
        if roots.actions is not None:
            observer_exact_round(ai, observer, roots.actions, payload, ctx, dist)
        elif deferred is not None:
            # Root menu not readable before completion: record and replay exact_round after the
            # search (the observer's `exact` dict is keyed; order across kinds is irrelevant).
            deferred.append((dict(payload), ctx, dist))
            deferred_here = True
        else:
            raise RuntimeError('Cannot service exact_round: root menu unavailable and no deferral list')
    with reply_snapshot(ai, reply) as snap:
        ctl.submit_payload(snap)
    if session is not None:
        session.combat_requests.append(dict(
            request_id=view.request_id, node=payload['node'], action_type=payload['action_type'],
            root_branch_row=payload['root_branch_row'], outcomes=len(reply['outcomes']), conf=reply['conf'],
            wctk_side=reply['wctk_side'], cards=[ctx.attacker_card, ctx.defender_card],
            observer_deferred=deferred_here))
    return payload, ctx, dist


def materialize_root_actions(ai, menu_query, menu, count):
    """Root Actions ONCE from the controller's root menu (descriptor rows in order)."""
    descriptors = menu_query.descriptors(menu, list(range(count)))
    if len(descriptors) != count:
        raise RuntimeError('Root menu returned %d descriptors; expected %d' % (len(descriptors), count))
    actions = []
    for descriptor in descriptors:
        if len(descriptor) != 3:
            raise RuntimeError('Root menu descriptor is not (die_idx, action_type, params): %r' % (descriptor,))
        actions.append(ai.Action(*descriptor))
    return tuple(actions), descriptors


def build_visit_info(ai, root, root_actions, action_encs, region_indices, temperature):
    """visit_info exactly as ai.mcts_search 17705-17760 + final selection 17790-17884.

    `root` is a controller_binding.RootView (or any object with the same arrays).
    Returns (index, visit_info). Python-float arithmetic is kept where the source
    reads node attributes (value_sum / visit_count etc.) so signed zeros and
    rounding match the source bit for bit.
    """
    k = int(root.children)
    visits = [int(x) for x in root.visits]
    value_sum = [float(x) for x in root.value_sum]
    value_sq = [float(x) for x in root.value_sq]
    term_visits = [int(x) for x in root.term_visits]
    conf_sum = [float(x) for x in root.conf_sum]
    phi_sum = [float(x) for x in root.phi_sum]
    forced_visits = [int(x) for x in root.forced_visits]
    proven_marks = [int(x) for x in root.proven]
    proven_value = [float(x) for x in root.proven_value]
    sides = [int(x) for x in root.side]
    root_side = int(root.root_side)
    if not (len(root_actions) == k == len(visits)):
        raise RuntimeError('Root view/children/actions length mismatch')
    if any(mark not in (-1, 0, 1) for mark in proven_marks):
        raise RuntimeError('Root proven marks must be -1/0/1')

    def q_value(i):
        return value_sum[i] / visits[i] if visits[i] > 0 else 0.0

    child_visits_raw = np.array(visits, dtype=np.float32)
    total_v = child_visits_raw.sum()
    child_visits = (child_visits_raw / total_v if total_v > 0
                    else child_visits_raw.copy())
    q_values = np.array(
        [(q_value(i) if sides[i] == root_side else -q_value(i)) for i in range(k)], dtype=np.float32)
    q_var = np.array(
        [max(0.0, value_sq[i] / visits[i] - (value_sum[i] / visits[i]) ** 2)
         if visits[i] > 0 else 0.0 for i in range(k)], dtype=np.float32)
    term_frac = np.array(
        [term_visits[i] / visits[i] if visits[i] > 0 else 0.0 for i in range(k)], dtype=np.float32)
    priors_clean = np.array(root.priors_clean, dtype=np.float32)
    conf_on = bool(ai._CONF_LEAF or ai._CONF_TRACK)
    visit_info = {
        'root_actions': tuple(root_actions),
        'action_encs': np.array(action_encs, dtype=np.float32),
        'region_indices': np.array(region_indices, dtype=np.int32),
        'visits': child_visits,
        'visit_counts': child_visits_raw.astype(np.int32),
        'q_values': q_values,
        'q_var': q_var,
        'term_frac': term_frac,
        'priors_clean': priors_clean,
        'proven': np.array(proven_marks, dtype=np.int8),
        'proven_value': np.array([float(v) for v in proven_value], dtype=np.float32),
        'forced': np.array(forced_visits, dtype=np.int32),
        'conf': np.array(
            [(conf_sum[i] / visits[i]) if (conf_on and visits[i] > 0) else 1.0 for i in range(k)],
            dtype=np.float32),
        'phi': np.array(
            [(phi_sum[i] / visits[i]) if (conf_on and visits[i] > 0) else 0.0 for i in range(k)],
            dtype=np.float32),
    }
    selector_contract = importlib.import_module('selector_contract')
    reward_contract = importlib.import_module('reward_contract')
    selector_cfg = dataclasses.replace(ai.SELECT_CFG)
    idx, pi_sel, sel_diag = ai.select_root_action(
        {'p': priors_clean, 'q': q_values, 'n': child_visits_raw,
         'tf': term_frac, 'pv': visit_info['proven'],
         'pvv': visit_info['proven_value'],
         'fn': visit_info['forced'], 'cf': visit_info['conf'],
         'ph': visit_info['phi']},
        temperature, cfg=selector_cfg)
    visit_info.update(selector_contract.make_evidence(
        selector_cfg, temperature, pi_sel, eligible=sel_diag['elig'],
        reward_contract=reward_contract.current_contract()))
    visit_info['q_hat'] = sel_diag['q_hat'].astype(np.float32)
    visit_info['sigma'] = sel_diag['sigma']
    visit_info['played'] = idx
    visit_info['regret'] = sel_diag['regret']
    return idx, visit_info


# ---------------------------------------------------------------------------------------------
# Tree reuse (SHARED_ABI 9.6): _TREE_CACHE wrapper, reuse decision (17399-17431), stash (17843-17859).
# ---------------------------------------------------------------------------------------------
TREE_CACHE_STASH_KEYS = ('sig', 'root_side', 'policy_version', 'opponent_policy_mode', 'opponent_policy_epsilon')


class TreeSession:
    """The object stored in `ai._TREE_CACHE['child']`: owns the detached native subtree.

    `_TREE_CACHE.get('child') is not None` keeps its source meaning. The native session is
    destroyed when the wrapper is closed (replaced by a new stash, rejected at reuse) or
    collected (the source's resets `_TREE_CACHE['child'] = None` at 17317-17318 / 18964-18966
    drop the last reference); a reused session is released to the controller instead.
    """
    __slots__ = ('session', 'side_index', 'side', 'action', 'visit_count', 'children', 'is_chance', 'has_state',
                 '__weakref__')

    def __init__(self, session, side_index, side=None, action=None, info=None):
        self.session = session
        self.side_index = int(side_index)
        # cache_evidence-compatible attributes (experiments/full_mcts_reuse/continuation.py reads
        # child.side, child.action.action_type, child.visit_count): the detached child's side as an
        # ai.Side member, its root-menu Action, and the SessionInfo scalars at detach time.
        self.side = side
        self.action = action
        self.visit_count = None if info is None else int(info.visit_count)
        self.children = None if info is None else int(info.children)
        self.is_chance = None if info is None else bool(info.is_chance)
        self.has_state = None if info is None else bool(info.has_state)

    @property
    def open(self):
        return self.session is not None and self.session.open

    def info(self):
        """Live SessionInfo of the native session (raises when closed/consumed)."""
        if not self.open:
            raise RuntimeError('TreeSession is closed or consumed')
        return self.session.info()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def drop_cached_child(ai):
    """`_TREE_CACHE['child'] = None` with the native session destroyed (forced roots 17376)."""
    previous = ai._TREE_CACHE.get('child')
    ai._TREE_CACHE['child'] = None
    if isinstance(previous, TreeSession):
        previous.close()


def continuation_root(ai, side):
    """The second root of a reuse pair (experiments/full_mcts_reuse/continuation.py next_root on
    the native route): the same assertions, evaluated on the cached TreeSession's live info, then an
    OWNED NativeSnapshot of the retained child.state (wotr_search_session_state). The caller owns
    and closes it; it is accepted by native_mcts_search as any NativeSnapshot root. The session is
    neither consumed nor mutated."""
    child = ai._TREE_CACHE.get('child')
    assert child is not None, 'Original selector did not retain a reusable continuation'
    assert isinstance(child, TreeSession) and child.open, 'Cached child is not an open native TreeSession'
    info = child.info()
    assert info.has_state and not info.is_chance, 'Selected continuation is not a materialized decision'
    assert info.children > 0, 'Selected continuation was not searched deeply enough for reuse'
    assert info.side == int(side == ai.Side.SHADOW), 'Selected continuation belongs to the other side'
    assert ai._TREE_CACHE.get('root_side') == side, 'Stored root currency differs'
    return child.session.state()


def state_projection(ai, res, native, side_index):
    """(sig, own_hand) of a native state: sig = Python hash of the _state_sig tuple (== ai._state_sig),
    own_hand = tuple(sorted(map(str, hands[side].hand))) - both from one metadata projection."""
    payload = res.search_metadata.query(ai, native, int(side_index))
    sig = res.search_metadata.source_signatures(payload)[0]
    own_hand = payload[3]['own_hand']
    if own_hand is None:
        raise RuntimeError('search_metadata projection returned no own hand for side %d' % side_index)
    return sig, tuple(own_hand)


def reuse_decision(ai, cache, side, blind, info, projection_now, projection_stash):
    """Adopt decision EXACTLY as ai.mcts_search 17399-17431 (short-circuit order, lazy sig).

    cache: the _TREE_CACHE dict (after the single-shot read of 'child'); info: SessionInfo of the
    cached subtree (None when no cached child); projection_now(): (sig_now, own_hand_now) of the
    new root; projection_stash(): (sig_stash_unused, own_hand_stash) of the retained child.state.
    Returns {'adopt', 'failed' (first failing condition or None), 'sig_now', ...}.
    """
    from ai_policies import POLICY_VERSION
    side_index = int(side == ai.Side.SHADOW)
    result = {'adopt': False, 'failed': None, 'sig_now': None}
    structural = (('not_blind', lambda: not blind),
                  ('cand_present', lambda: info is not None),
                  ('cand_state', lambda: info.has_state),
                  ('cand_not_chance', lambda: not info.is_chance),
                  ('cand_children', lambda: info.children > 0),
                  ('cand_side', lambda: info.side == side_index),
                  ('root_side', lambda: cache.get('root_side') == side))
    for name, condition in structural:
        if not condition():
            result['failed'] = name
            return result
    sig_now, own_hand_now = projection_now()
    result['sig_now'] = sig_now
    mode = ai._OPPONENT_POLICY_MODE
    scalar = (('policy_version', lambda: cache.get('policy_version') == POLICY_VERSION),
              ('opponent_policy_mode', lambda: cache.get('opponent_policy_mode', 'legacy_puct') == mode),
              ('opponent_policy_epsilon', lambda: mode == 'legacy_puct'
               or cache.get('opponent_policy_epsilon') == ai._OPPONENT_POLICY_EPSILON),
              ('sig_not_none', lambda: sig_now is not None),
              ('sig_equal', lambda: sig_now == cache.get('sig')),
              ('own_hand_equal', lambda: tuple(sorted(map(str, projection_stash()[1])))
               == tuple(sorted(map(str, own_hand_now)))))
    for name, condition in scalar:
        if not condition():
            result['failed'] = name
            return result
    result['adopt'] = True
    return result


def _stash_projection(ai, res, wrapper, side_index):
    """(sig, own_hand) of the retained child.state (owned clone from wotr_search_session_state)."""
    with wrapper.session.state() as stash_state:
        return state_projection(ai, res, stash_state, side_index)


def stash_fields(ai, side, sig):
    """The _TREE_CACHE writes of _stash_tree 17854-17859 (besides 'child')."""
    from ai_policies import POLICY_VERSION
    return {'sig': sig, 'root_side': side, 'policy_version': POLICY_VERSION,
            'opponent_policy_mode': ai._OPPONENT_POLICY_MODE, 'opponent_policy_epsilon': ai._OPPONENT_POLICY_EPSILON}


def stash_tree(ai, res, ctl, idx, side, blind, root_actions=None):
    """_stash_tree(root.children[idx]) over the finished controller: detach under the source
    condition (`_TREE_REUSE and not _blind9 and child.state is not None and not child.is_chance`;
    the controller refuses the last two with NULL -> no stash), then write _TREE_CACHE."""
    record = {'attempted': False, 'stashed': False, 'reason': None}
    if not (ai._TREE_REUSE and not blind):
        record['reason'] = 'tree_reuse_off' if not ai._TREE_REUSE else 'blinded_root'
        return record
    record['attempted'] = True
    session, error = ctl.detach_child(idx)
    if session is None:
        record['reason'] = 'not_detachable: ' + error
        return record
    side_index = int(side == ai.Side.SHADOW)
    info = session.info()
    with session.state() as stash_state:
        sig, own_hand = state_projection(ai, res, stash_state, side_index)
    previous = ai._TREE_CACHE.get('child')
    ai._TREE_CACHE['child'] = TreeSession(session, side_index, side=side_member(ai, info.side),
                                          action=(None if root_actions is None else root_actions[int(idx)]), info=info)
    ai._TREE_CACHE.update(stash_fields(ai, side, sig))
    if isinstance(previous, TreeSession):
        previous.close()
    record.update(stashed=True, sig=sig, own_hand=list(own_hand), child_index=int(idx),
                  child_info=dict(side=info.side, children=info.children, visit_count=info.visit_count))
    return record


def _add_counter(target, key, value):
    # Coordinator fix (2026-09-09): the source creates most counters lazily
    # (`STATS.get(k, 0) + 1`, `setdefault`), so a zero/empty native delta for a key the
    # live dict does not hold must not create it; nonzero deltas and existing keys add.
    if isinstance(value, list):
        if value or key in target:
            target.setdefault(key, []).extend(value)
    elif isinstance(value, dict):
        if value or key in target:
            sub = target.setdefault(key, {})
            for inner_key, inner_value in value.items():
                _add_counter(sub, inner_key, inner_value)
    else:
        if value or key in target:
            target[key] = target.get(key, 0) + value


# Source key-creation order per counter dict.  The source creates most keys lazily
# (`.get(k, 0) + 1`, `setdefault`) while the controller's export is a sorted std::map, so
# NEW keys must be inserted into the live dicts in the source's order (dict key order is part
# of the whole-game counters comparison).  Predefined keys are in the module definition order
# (they already exist in the live dicts); lazily created keys follow the code order of their
# first-creation sites.  Caveat: keys whose first creation depends on search history
# (overlay commit_blind / learned_policy_draws / no_action_samples; kernel high_drop /
# fallback_detail / by_class / dropped_at; edge / lattice Counters) can only be ordered by
# code order here - an exact history order would need the controller to export creation order.
COUNTER_KEY_ORDER = {
    # ai.py 15383 predefined; _flush_pending 14004, 14005, then 14017/14033
    'search': ('apply_fail', 'leaf_batches', 'leaf_batched', 'leaf_batch_fallback'),
    # ai.py 14658-14663 predefined; lazy: 14859 (per blinded sample), 17059 (learned draw), 17038/17109
    'overlay': ('samples', 'sample_error', 'patches', 'patch_error', 'cond_swaps', 'require_fail', 'mask_empty',
                'mask_error', 'lazy_event', 'lazy_discard', 'lazy_combat', 'lazy_no_template', 'lazy_rejected',
                'det_draw_chance', 'assert_checks', 'commit_blind', 'learned_policy_draws', 'no_action_samples'),
    # ai.py 15361-15365 predefined; lazy: 15372 high_drop, 15409 fallback_detail, 15852 by_class, 15855 dropped_at
    'kernel': ('combat_chance_nodes', 'kernel_nodes', 'fallback', 'dropped_mass', 'support', 'opp_card_nodes',
               'opp_card_fallback', 'hunt_class_applies', 'hunt_nodes', 'hunt_det', 'hunt_walks',
               'high_drop', 'fallback_detail', 'by_class', 'dropped_at'),
    # search_edge_catalog.py Counter: 58 materialized, 69 catalog_fallbacks, 101 snapshots_created, 109 selections
    'edge': ('materialized', 'catalog_fallbacks', 'snapshots_created', 'selections'),
    # search_lattice.py Counter, code order of the increment sites
    'lattice': ('full_iteration', 'flat_projection_menus', 'flat_projection_rows', 'lattice_encoded_rows',
                'dynamic_encoded_rows', 'block_created', 'block_snapshot_rows', 'materialized', 'encoding_hit_rows',
                'encoding_miss_rows', 'original_matrix_direct_rows', 'block_reuse_rows', 'borrowed_pruning_rows',
                'numeric_list_pruning_rows'),
}
# Nested dicts with a fixed source order (ai.py 15855-15862: the 16/24/32 loop).
NESTED_KEY_ORDER = {('kernel', 'dropped_at'): (16, 24, 32)}


def ordered_items(mapping, order):
    """Items of `mapping` sorted by the source order table; unknown keys keep the export order after it."""
    position = {key: index for index, key in enumerate(order)}
    keys = list(mapping)
    return [(key, mapping[key]) for key in sorted(keys, key=lambda k: (position.get(k, len(order)), keys.index(k)))]


def merge_counters(ai, counters):
    """Add the controller's deltas into the live OVERLAY/KERNEL/SEARCH (+edge/lattice) stats,
    inserting new keys in the source's creation order (COUNTER_KEY_ORDER)."""
    if not isinstance(counters, dict):
        raise RuntimeError('Controller counters must decode to a dict; got %r' % type(counters))
    targets = {'overlay': ai.OVERLAY_STATS, 'kernel': ai.KERNEL_STATS, 'search': ai.SEARCH_STATS}
    try:
        targets['edge'] = importlib.import_module('search_edge_catalog').STATS
        targets['lattice'] = importlib.import_module('search_lattice').STATS
    except Exception:
        pass
    for name, target in targets.items():
        for key, value in ordered_items(counters.get(name) or {}, COUNTER_KEY_ORDER.get(name, ())):
            nested = NESTED_KEY_ORDER.get((name, key))
            if nested is not None and isinstance(value, dict):
                value = dict(ordered_items(value, nested))
            _add_counter(target, key, value)


def _resolve_root(ai, state, res):
    """Return (object the decorators/observer see, NativeSnapshot, owned_snapshot_or_None)."""
    if type(state) is snapshot.NativeSnapshot:
        return NativeRoot(ai, state, res), state, None
    native = getattr(state, 'native', None)
    if type(native) is snapshot.NativeSnapshot:
        return state, native, None
    if snapshot._identity(type(state)) == 'game_state.GameState':
        packed = snapshot.pack_snapshot(ai, state)
        return state, packed, packed
    raise TypeError('native_mcts_search accepts a GameState, a NativeSnapshot or an object with .native')


# ---------------------------------------------------------------------------------------------
# Per-search budget guard (package Y2, loop_resilience, 2026-09-11).  Checked between advances.
# ---------------------------------------------------------------------------------------------
SEARCH_MAX_SECONDS_ENV = 'WOTR_SEARCH_MAX_SECONDS'          # wall seconds per search (float), 0 = off
SEARCH_MAX_PRIVATE_GB_ENV = 'WOTR_SEARCH_MAX_PRIVATE_GB'    # private-bytes GROWTH per search in GB, 0 = off
SEARCH_MAX_ADVANCES_ENV = 'WOTR_SEARCH_MAX_ADVANCES'        # advance calls per SIMULATION (factor), 0 = off
SEARCH_BUDGET_EXPORT_ENV = 'WOTR_SEARCH_BUDGET_EXPORT'      # auto (default) | always | never
SEARCH_BUDGET_DEFAULTS = {'max_seconds': 300.0, 'max_private_gb': 4.0, 'max_advances_per_sim': 20.0}
SEARCH_BUDGET_MEMORY_SAMPLE_SECONDS = 1.0   # psutil cadence; wall/advance checks run at every advance
SEARCH_BUDGET_EXPORT_TAIL = 64              # nodes / events kept from the controller export on a breach
SEARCH_BUDGET_EXPORT_MAX_GROWTH_GB = 1.0    # 'auto': export only when the tree is still small
GB = float(1024 ** 3)


def _env_nonnegative(name, default, kind):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError('%s must be a nonnegative number (0 disables the check); got %r' % (name, raw))
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError('%s must be a nonnegative number (0 disables the check); got %r' % (name, raw))
    return value


def search_budget_limits(num_simulations):
    """Resolved per-search limits from the environment (0 = that check is disabled):
    {'max_seconds', 'max_private_gb', 'max_advances_per_sim', 'max_advances', 'export'}.
    max_advances = ceil(max_advances_per_sim x num_simulations)."""
    factor = _env_nonnegative(SEARCH_MAX_ADVANCES_ENV, SEARCH_BUDGET_DEFAULTS['max_advances_per_sim'], float)
    export = (os.environ.get(SEARCH_BUDGET_EXPORT_ENV) or 'auto').strip().lower()
    if export not in ('auto', 'always', 'never'):
        raise ValueError('%s must be auto, always or never; got %r' % (SEARCH_BUDGET_EXPORT_ENV, export))
    return {'max_seconds': _env_nonnegative(SEARCH_MAX_SECONDS_ENV, SEARCH_BUDGET_DEFAULTS['max_seconds'], float),
            'max_private_gb': _env_nonnegative(SEARCH_MAX_PRIVATE_GB_ENV, SEARCH_BUDGET_DEFAULTS['max_private_gb'], float),
            'max_advances_per_sim': factor,
            'max_advances': int(math.ceil(factor * max(int(num_simulations), 1))) if factor > 0 else 0,
            'export': export}


def process_private_bytes():
    """Private bytes of this process (psutil; Windows `private` = commit charge, POSIX falls back
    to RSS).  Raises when psutil is unavailable - the memory check must never silently vanish."""
    import psutil
    info = psutil.Process().memory_info()
    return int(getattr(info, 'private', None) or info.rss)


def export_tail(export, tail=SEARCH_BUDGET_EXPORT_TAIL):
    """Bounded view of a controller export (SHARED_ABI section 5): counts, the root node, the
    last `tail` nodes and events.  Never holds the whole tree."""
    if not isinstance(export, dict):
        return {'error': 'export is not a dict: %s' % type(export).__name__}
    nodes = export.get('nodes') or []
    events = export.get('events') or []
    out = {'nodes_total': len(nodes), 'events_total': len(events), 'tail': int(tail),
           'root': nodes[0] if nodes else None,
           'nodes_tail': list(nodes[-int(tail):]) if tail else [],
           'events_tail': list(events[-int(tail):]) if tail else [],
           'other_keys': sorted(k for k in export if k not in ('nodes', 'events'))}
    return out


class SearchBudgetExceeded(RuntimeError):
    """'native search budget exceeded: <which>=<value> (limit <limit>) sims=<n> advances=<k>'.

    Attributes: which, value, limit, sims, advances, elapsed_seconds, private_growth_bytes,
    root_state (the search's root object), search_kwargs (dict), counters (controller counters
    or None), export_tail (bounded export or None), capture_errors (list of str) and
    search_budget (a JSON-able summary without the state object) - everything the runner's
    failure fixture / attempt row needs.  Not a native refusal (run_sp_served.is_native_refusal).
    """

    def __init__(self, message, *, which, value, limit, sims, advances, elapsed_seconds, private_growth_bytes,
                 root_state=None, search_kwargs=None, counters=None, export_tail=None, capture_errors=()):
        super().__init__(message)
        self.which, self.value, self.limit = which, value, limit
        self.sims, self.advances = int(sims), int(advances)
        self.elapsed_seconds = float(elapsed_seconds)
        self.private_growth_bytes = None if private_growth_bytes is None else int(private_growth_bytes)
        self.root_state = root_state
        self.search_kwargs = dict(search_kwargs or {})
        self.counters = counters
        self.export_tail = export_tail
        self.capture_errors = list(capture_errors)
        self.search_budget = {
            'source': 'orchestrator', 'which': which, 'value': value, 'limit': limit, 'sims': self.sims,
            'advances': self.advances, 'elapsed_seconds': round(self.elapsed_seconds, 3),
            'private_growth_gb': (None if self.private_growth_bytes is None
                                  else round(self.private_growth_bytes / GB, 4)),
            'counters': counters, 'export_tail': export_tail, 'capture_errors': self.capture_errors,
            'search_kwargs': {k: (v if isinstance(v, (str, int, float, bool)) or v is None else repr(v))
                              for k, v in self.search_kwargs.items()}}


class SearchBudgetGuard:
    """Wall / private-bytes-growth / advance-count budget of ONE native search.

    `before_advance(ctl)` is called immediately before every `ctl.advance()`: it counts the
    advance, compares the wall clock and the advance count against the limits at every call and
    samples psutil private bytes at most every SEARCH_BUDGET_MEMORY_SAMPLE_SECONDS.  On a breach
    it collects the controller counters (always attempted) and a bounded export tail ('auto':
    only when the growth is still below SEARCH_BUDGET_EXPORT_MAX_GROWTH_GB and the breach is not
    the memory one - an export copies the tree) and raises SearchBudgetExceeded.  With every
    limit 0 it only counts.  No randomness, no controller call on the non-breach path.
    """

    def __init__(self, ai, limits, num_simulations, root_state=None, search_kwargs=None, *,
                 clock=time.monotonic, memory=process_private_bytes):
        self.ai = ai
        self.limits = dict(limits)
        self.sims = int(num_simulations)
        self.root_state = root_state
        self.search_kwargs = dict(search_kwargs or {})
        self.clock = clock
        self.memory = memory if self.limits.get('max_private_gb', 0) > 0 else None
        self.started = clock()
        self.advances = 0
        self.private_start = self.memory() if self.memory is not None else None
        self.private_last = self.private_start
        self._last_sample = self.started
        self.enabled = bool(self.limits.get('max_seconds') or self.limits.get('max_private_gb')
                            or self.limits.get('max_advances'))

    @classmethod
    def from_env(cls, ai, num_simulations, root_state=None, search_kwargs=None):
        return cls(ai, search_budget_limits(num_simulations), num_simulations, root_state, search_kwargs)

    def growth(self):
        return None if self.private_last is None else self.private_last - self.private_start

    def summary(self):
        growth = self.growth()
        return {'limits': dict(self.limits), 'sims': self.sims, 'advances': self.advances,
                'elapsed_seconds': round(self.clock() - self.started, 3),
                'private_growth_gb': None if growth is None else round(growth / GB, 4)}

    def before_advance(self, ctl):
        self.advances += 1
        if not self.enabled:
            return
        now = self.clock()
        elapsed = now - self.started
        max_advances = self.limits.get('max_advances') or 0
        if max_advances and self.advances > max_advances:
            self.breach(ctl, 'advances', self.advances, max_advances, elapsed)
        max_seconds = self.limits.get('max_seconds') or 0
        if max_seconds and elapsed > max_seconds:
            self.breach(ctl, 'wall_seconds', round(elapsed, 1), max_seconds, elapsed)
        if self.memory is not None and now - self._last_sample >= SEARCH_BUDGET_MEMORY_SAMPLE_SECONDS:
            self._last_sample = now
            self.private_last = self.memory()
            growth = self.private_last - self.private_start
            limit_gb = float(self.limits['max_private_gb'])
            if growth > limit_gb * GB:
                self.breach(ctl, 'private_gb', round(growth / GB, 3), limit_gb, elapsed)

    def evidence(self, ctl, which):
        """(counters, export_tail, errors): best effort, never raises."""
        errors = []
        counters = export = None
        try:
            counters = ctl.counters(self.ai)
        except Exception as error:
            errors.append('counters: %s: %s' % (type(error).__name__, error))
        mode = self.limits.get('export', 'auto')
        growth = self.growth()
        if mode == 'never':
            export = {'skipped': 'export disabled (%s=never)' % SEARCH_BUDGET_EXPORT_ENV}
        elif mode == 'auto' and (which == 'private_gb'
                                 or (growth is not None and growth > SEARCH_BUDGET_EXPORT_MAX_GROWTH_GB * GB)):
            export = {'skipped': 'export would copy a runaway tree (breach %s, private growth %s GB > %s GB auto cap)'
                      % (which, None if growth is None else round(growth / GB, 3), SEARCH_BUDGET_EXPORT_MAX_GROWTH_GB)}
        else:
            try:
                export = export_tail(ctl.export(self.ai, False))
            except Exception as error:
                errors.append('export: %s: %s' % (type(error).__name__, error))
        return counters, export, errors

    def breach(self, ctl, which, value, limit, elapsed):
        message = ('native search budget exceeded: %s=%s (limit %s) sims=%d advances=%d'
                   % (which, value, limit, self.sims, self.advances))
        counters, export, errors = self.evidence(ctl, which)
        raise SearchBudgetExceeded(message, which=which, value=value, limit=limit, sims=self.sims,
                                   advances=self.advances, elapsed_seconds=elapsed,
                                   private_growth_bytes=self.growth(), root_state=self.root_state,
                                   search_kwargs=self.search_kwargs, counters=counters, export_tail=export,
                                   capture_errors=errors)


_SEARCHES = {}
_SEARCH_LOCK = threading.Lock()


def _build(ai):
    decorators = verify_decorator_chain(ai)

    def native_search(state, side, fp_net, sh_net, num_simulations=100, c_puct=None, temperature=1.0,
                      add_noise=True, hot_regions=None, bonus_sims=0, determinize=True,
                      _native=None, _owned=None, _session=None):
        if c_puct is None:
            c_puct = ai._C_PUCT
        res = resources(ai)
        # Capability checks: each milestone-2 step is wired only when the controller exposes its
        # SHARED_ABI 9.x exports; otherwise the corresponding flag is refused by name (path unchanged).
        noise_capable = binding.has_capability(res.lib, 'root_priors_exchange')
        reuse_capable = binding.has_capability(res.lib, 'tree_reuse')
        refuse_unsupported(ai, add_noise, hot_regions, bonus_sims, determinize, noise_capable=noise_capable,
                           reuse_capable=reuse_capable)
        side_index = int(side == ai.Side.SHADOW)
        selected = None
        try:
            # 1. policy projection; forced -> (actions[0], None) exactly as the source (17376: the
            #    cached child is dropped; the source's entry flush of its edge catalog (17361-17366)
            #    cannot run without a controller - reported in INFERENCE_NOTES 7.6/8.x).
            if _owned is not None:
                policy = ai.get_ai_policy(state, side)
                if policy.forced_reason:
                    drop_cached_child(ai)
                    return policy.actions[0], None
            else:
                with _native.menu_handle(ai, side_index, 'full') as raw:
                    selected, forced, reasons = _native.project_policy(ai, side_index, raw)
                if forced:
                    actions, _ = materialize_root_actions(ai, res.menu, selected, 1)
                    drop_cached_child(ai)
                    return actions[0], None
            if not ai.HAS_TORCH:
                return None
            ai._SEARCH_NETS = (fp_net, sh_net)
            ai._SEARCH_ROOT_SIDE = side
            observer_capable = binding.has_capability(res.lib, 'observer_events')
            refuse_root(ai, state, side, observer_capable=observer_capable)
            blind = ai._hidden_commit_card(state, side) is not None
            # 9.6 entry (17361-17366, 17399-17431): any cached session goes to create_reusing (the
            # controller performs the entry flush whether or not it adopts); the single-shot read and
            # the adopt decision happen only under _TREE_REUSE, exactly as the source.
            cached = ai._TREE_CACHE.get('child')
            if cached is not None and not (isinstance(cached, TreeSession) and cached.open):
                raise Unsupported('_TREE_CACHE holds a child that is not an open native TreeSession '
                                  '(mixing the Python and native search routes is unsupported)')
            reuse_record = {'capable': reuse_capable, 'cached': cached is not None, 'tree_reuse': bool(ai._TREE_REUSE),
                            'decision': None, 'adopt': False, 'reused': False}
            cand = None
            adopt = False
            if ai._TREE_REUSE:
                cand = cached
                ai._TREE_CACHE['child'] = None      # single-shot
                if cand is not None:
                    decision = reuse_decision(
                        ai, ai._TREE_CACHE, side, blind, cand.session.info(),
                        lambda: state_projection(ai, res, _native, side_index),
                        lambda: _stash_projection(ai, res, cand, side_index))
                    reuse_record['decision'] = decision
                    adopt = bool(decision['adopt'])
                    reuse_record['adopt'] = adopt
            # 2. controller over the search stream state (the decorator swapped it in).
            config = build_config(ai, side_index, num_simulations, c_puct, determinize,
                                  trace=_session is not None and _session.trace, add_noise=bool(add_noise))
            context = ai._OWN_VALUE_CONTEXT
            if context is None or context.side != side:
                raise ValueError('Missing matching own-value root context')
            expected = (context.net, None) if side_index == 0 else (None, context.net)
            if fp_net is not expected[0] or sh_net is not expected[1]:
                raise ValueError('Own-value orchestration accepts only the root model')
            bridge = RootNumericInferenceBridge(ai, context.net, side)
            noised_once = False
            priors_clean_py = None
            deferred_exact = []
            # Y2 budget guard: clock + private-bytes baseline taken here, just before the controller
            # is created; checked before every advance below.  Non-breach path: no side effects.
            guard = SearchBudgetGuard.from_env(
                ai, num_simulations, root_state=state,
                search_kwargs={'side': side, 'num_simulations': num_simulations, 'c_puct': c_puct,
                               'temperature': temperature, 'add_noise': add_noise, 'hot_regions': hot_regions,
                               'bonus_sims': bonus_sims, 'determinize': determinize})
            with random_stream.NativeRandom(random.getstate()) as rng, \
                    OwnController(res.lib, _native, res.tables, res.features.constants, res.menu.tables,
                                  res.chance, rng, config, seed=context.native_seed,
                                  reply_panels=context.panels, belief_worlds=context.world_count) as ctl:
                reused = bool(ctl.reused)
                reuse_record['reused'] = reused
                if reused != adopt:
                    raise RuntimeError('Controller reuse outcome %r differs from the Python adopt decision %r '
                                       '(structural conditions were checked on both sides)' % (reused, adopt))
                if ai._TREE_REUSE:
                    # 17427-17429: hits on reuse, misses otherwise (inside the _TREE_REUSE block only).
                    ai._TREE_CACHE['hits' if reused else 'misses'] += 1
                    if cand is not None and not reused:
                        cand.close()           # the source drops the rejected subtree (del _cand9)
                roots = RootActions(ai, res, ctl, _native, side_index)
                if _session is not None:
                    _session.capabilities = dict(ctl.capabilities)
                while True:
                    guard.before_advance(ctl)
                    status = ctl.advance()
                    if status == binding.STATUS_COMPLETE:
                        break
                    if status == binding.STATUS_ROOT_PRIORS:
                        # 9.4 / source 17440, 17489-17495: priors_clean = float32 capture BEFORE noise,
                        # one Dirichlet draw from the search NumPy stream, _apply_root_noise in place.
                        if noised_once:
                            raise RuntimeError('Controller paused for root priors twice')
                        if reused:
                            raise RuntimeError('Controller paused for root priors on a reused root (the source skips noise there)')
                        hand = ctl.root_priors()
                        priors_clean_py = np.array(hand.priors, dtype=np.float32)
                        noised, noise, exempt = root_noise_step(ai, hand.priors.tolist(), hand.is_free.astype(bool))
                        ctl.set_root_priors(np.asarray(noised, dtype=np.float64))
                        noised_once = True
                        if _session is not None:
                            _session.root_noise = dict(children=hand.children, clean=hand.priors.tolist(),
                                                       is_free=hand.is_free.tolist(), noise=noise.tolist(),
                                                       noised=list(noised), exempt=exempt.tolist())
                        continue
                    view = ctl.request()
                    if _session is not None:
                        _session.requests.append(view.summary())
                    if reused and view.purpose == 1:
                        raise RuntimeError('A reused root must not issue a root expansion request (9.6)')
                    if view.operation == binding.OP_COMBAT_DIST:
                        # 9.5: metadata request answered from the frozen combat kernel functions.
                        service_combat_request(ai, ctl, view, roots, _session, deferred_exact)
                        continue
                    ctl.submit(bridge.run(Request.from_view(view)))
                if add_noise and not noised_once and not reused:
                    raise RuntimeError('add_noise was accepted but the controller never paused for root priors')
                # 3. root view + root menu -> Actions once, encodings over the input state.
                root = ctl.root()
                if root.children < 1:
                    raise RuntimeError('Controller finished with no root children')
                root_actions = roots.materialize(root.children)
                descriptors, action_encs = roots.descriptors, roots.action_encs
                if priors_clean_py is not None and priors_clean_py.tobytes() != np.asarray(root.priors_clean, dtype=np.float32).tobytes():
                    raise RuntimeError('priors_clean captured at the noise pause differs from the controller root view')
                # 9.5 deferred exact_round calls (root menu was not readable at service time) and the
                # 9.3 observer replay, both BEFORE the decorator's finish_search.
                observer = ai._COMBAT_FORECAST_OBSERVER
                if deferred_exact:
                    if observer is None:
                        raise RuntimeError('deferred exact_round calls without an observer')
                    for payload, ctx, dist in deferred_exact:
                        observer_exact_round(ai, observer, root_actions, payload, ctx, dist)
                    if _session is not None:
                        _session.deferred_exact_rounds = len(deferred_exact)
                if observer_capable:
                    events = ctl.observer_events(ai)
                    if _session is not None:
                        _session.observer_events = events
                    if observer is not None:
                        replay_observer_events(ai, observer, events, root_actions, state, side)
                region_indices = np.array([ai._get_action_region_index(a) for a in root_actions], dtype=np.int32)
                # temperature > 0: select_root_action samples with np.random.choice on the search
                # NumPy stream (the decorator swapped it in); same function, same inputs as the source.
                idx, visit_info = build_visit_info(ai, root, root_actions, action_encs, region_indices, temperature)
                # 4. RNG/counters hand-back, _stash_tree (9.6), then the debug export. The source tracer
                #    reads its tree after mcts_search returned, i.e. after _stash_tree cut the chosen
                #    child's parent link (17849); the native detach cuts exactly that link and the
                #    controller's export is taken AFTER the detach so the cut is exported natively
                #    (one owner). When the controller refuses views after a detach (older builds) the
                #    pre-detach export is used and recorded as such (the oracle then sees parent 0).
                rng_state = ctl.rng_state()
                counters = ctl.counters(ai)
                export_before = None
                if _session is not None and _session.trace:
                    export_before = ctl.export(ai, _session.with_states)
                if reuse_capable:
                    reuse_record['stash'] = stash_tree(ai, res, ctl, idx, side, blind, root_actions)
                if _session is not None:
                    _session.root = root
                    _session.root_descriptors = descriptors
                    _session.counters = counters
                    _session.rng_state = rng_state
                    _session.budget = root.budget
                    _session.simulations_run = root.simulations_run
                    reuse_record['hits'] = ai._TREE_CACHE.get('hits')
                    reuse_record['misses'] = ai._TREE_CACHE.get('misses')
                    _session.budget_guard = guard.summary()
                    if _session.trace:
                        _session.export = export_before
                        if reuse_capable and reuse_record['stash'].get('stashed'):
                            try:
                                _session.export = ctl.export(ai, _session.with_states)
                                reuse_record['stash']['export_after_detach'] = True
                            except Exception as error:
                                reuse_record['stash']['export_after_detach'] = False
                                reuse_record['stash']['export_error'] = '%s: %s' % (type(error).__name__, error)
                    _session.reuse = reuse_record
            random.setstate(rng_state)
            merge_counters(ai, counters)
            ai._SIM_OVERLAY = None
            visit_info['own_value_search'] = {'contract':'own_value_reply_v0',
                'reply_identity':'RGK1','reply_cache_charge':'wire_plus_512_per_row_plus_4096',
                'reply_panels_per_visit':context.panels,'belief_worlds_per_panel':context.world_count,
                'leaf_batch':1,'tree_reuse':False,'stats':counters.get('own_value_search',{})}
            result = root_actions[idx], visit_info
            if _session is not None:
                _session.result = result
            return result
        finally:
            if selected is not None:
                selected.close()

    decorated = native_search
    for decorator in reversed(decorators):
        decorated = decorator(decorated)
    return decorated


def _decorated(ai):
    with _SEARCH_LOCK:
        result = _SEARCHES.get(id(ai))
        if result is None:
            result = _SEARCHES[id(ai)] = _build(ai)
        return result


def native_mcts_search(ai, state, side, fp_net, sh_net, num_simulations=100, c_puct=None, temperature=1.0,
                       add_noise=True, hot_regions=None, bonus_sims=0, determinize=True, session=None):
    """Drop-in for ai.mcts_search(state, side, fp_net, sh_net, ...) over the native controller."""
    res = resources(ai)
    root_object, native, owned = _resolve_root(ai, state, res)
    try:
        return _decorated(ai)(root_object, side, fp_net, sh_net, num_simulations, c_puct, temperature,
                              add_noise, hot_regions, bonus_sims, determinize,
                              _native=native, _owned=owned, _session=session)
    finally:
        if owned is not None:
            owned.close()


def native_search_session(ai, state, side, fp_net, sh_net, *, trace=True, with_states=False, **kwargs):
    """Run native_mcts_search with a debug session; returns the SearchSession (result in .result)."""
    session = SearchSession(trace=trace, with_states=with_states)
    session.result = native_mcts_search(ai, state, side, fp_net, sh_net, session=session, **kwargs)
    return session


def own_search_session(ai,state,side,own_net,*,profile,simulations,seed,reply_panels=2,belief_worlds=2,trace=True):
    """Use reference root context/RNG scopes; controller owns all reply traversal."""
    import own_search
    with own_search.active(ai,side,own_net,profile=profile,seed=seed,
                          reply_panels=reply_panels,belief_worlds=belief_worlds) as context:
        context.native_seed=seed
        with own_search.reference_rng(ai,seed):
            fp,sh=ai._SEARCH_NETS
            return native_search_session(ai,state,side,fp,sh,num_simulations=simulations,
                                         temperature=0,add_noise=False,determinize=True,trace=trace)
