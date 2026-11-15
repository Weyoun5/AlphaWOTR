# hunt_dist.py - EXACT hunt-resolution distribution kernel.
"""Exact joint distribution over one fellowship-move hunt event's resolved
outcome, engine functions as single source of truth per branch; stochastic
parts (hunt roll, rerolls, tile draw, Shelob d6, random companion) enumerated
analytically.  Certified by cert_hunt_dist.py (fork-oracle + Monte Carlo)."""

from __future__ import annotations

import math
import os
import random
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import board                               # noqa: E402
import game_state as gs                    # noqa: E402
import game_logic as gl                    # noqa: E402
from game_state import (Side, HuntTile, PersistentCardState,
                        SPECIAL_HUNT_TILES)            # noqa: E402
from persistent_effects import create_persistent_card  # noqa: E402


# --------------------------------------------------------------------------
# Tile specs (keys are ascii; special names matched accent-insensitively)
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


_STD_TILES = {
    '0r': dict(damage=0, reveal=True),
    '1':  dict(damage=1),
    '1r': dict(damage=1, reveal=True),
    '2':  dict(damage=2),
    '2r': dict(damage=2, reveal=True),
    '3':  dict(damage=3),
    'eye': dict(is_eye=True),
}

_SPECIAL_KEYS = {
    'oonw': 'on on they went',
    'shelob': "shelob's lair",
    'ringmine': 'the ring is mine!',
    'giveuss': 'give it to uss!',
    'smeagol': 'smeagol helps nice master',
    'ecloak': 'elven cloaks',
    'erope': 'elven rope',
    'phial': 'phial of galadriel',
}

_SPECIAL_TEMPLATES: Dict[str, HuntTile] = {}
for _k, _n in _SPECIAL_KEYS.items():
    for _name, _t in SPECIAL_HUNT_TILES.items():
        if _norm(_name) == _n:
            _SPECIAL_TEMPLATES[_k] = _t
            break
assert len(_SPECIAL_TEMPLATES) == len(_SPECIAL_KEYS)

STANDARD_POOL = ('3', '3', '3', '2', '2', '1', '1',
                 'eye', 'eye', 'eye', 'eye',
                 '2r', '1r', '1r', '0r', '0r')


def mk_tile(key: str) -> HuntTile:
    if key in _STD_TILES:
        return HuntTile(**_STD_TILES[key])
    t = _SPECIAL_TEMPLATES[key]
    return HuntTile(damage=t.damage, is_eye=t.is_eye, reveal=t.reveal,
                    is_special=t.is_special, special_name=t.special_name,
                    stops_movement=t.stops_movement,
                    is_dice_roll=t.is_dice_roll,
                    is_fellowship_special=t.is_fellowship_special,
                    image_file=t.image_file)


def tile_key(tile: HuntTile) -> str:
    if getattr(tile, 'special_name', None):
        nm = _norm(tile.special_name)
        for k, n in _SPECIAL_KEYS.items():
            if n == nm:
                return k
        return 'special?' + nm.replace(' ', '_')
    if tile.is_eye:
        return 'eye'
    return '%d%s' % (tile.damage, 'r' if tile.reveal else '')


# --------------------------------------------------------------------------
# Context spec
# --------------------------------------------------------------------------

_NAME_ABILITY = {
    'Gandalf the Grey': 'palantir_draw',
    'Strider': 'hide_any_die',
    'Meriadoc': 'sacrifice_reduce_hunt',
    'Peregrin': 'sacrifice_reduce_hunt',
    'Gollum': 'no_reveal_on_damage',
}


@dataclass(frozen=True)
class CompanionSpec:
    name: str
    level: int
    leadership: int = 1
    guide_ability: Optional[str] = None    # None = engine default by name


def CS(name: str, level: int) -> CompanionSpec:
    return CompanionSpec(name, level)


DEFAULT_ROSTER = (CS('Gandalf the Grey', 3), CS('Legolas', 2),
                  CS('Gimli', 2), CS('Meriadoc', 1))


@dataclass(frozen=True)
class HuntContext:
    """One fellowship-move hunt event; learned pendings are inputs."""
    eyes: int = 1                      # Shadow dice in the hunt box
    fp_dice: int = 0                   # FP dice in the hunt box (C dice)
    pool: Tuple[str, ...] = STANDARD_POOL
    discard: Tuple[str, ...] = ()      # used pile (recycle-rule cases)
    companions: Tuple[CompanionSpec, ...] = DEFAULT_ROSTER
    guide: str = ''                    # '' = highest level, first listed
    corruption: int = 0
    hidden: bool = True
    progress: int = 0
    region: str = 'dagorlad'
    in_mordor: bool = False
    mordor_step: int = 0
    nazgul_at_fs: bool = False         # hunt re-roll source fixtures
    army_at_fs: bool = False
    crebain_bonus: int = 0             # resolved Crebain discard (+1 all dice)
    mithril: bool = False
    mithril_redraw: Tuple[str, ...] = ()   # tile keys the FP redraws on
    wizard_staff: bool = False
    wizard_prevent: bool = False
    axe_horn: bool = False
    use_reduction: bool = False
    damage_choice: str = 'corruption'  # corruption|guide|random|separate
    chain_choice: str = 'corruption'   # chained sacrifice phases
    gollum_reveal: bool = False


# --------------------------------------------------------------------------
# Fixture construction (shared with the certification harness)
# --------------------------------------------------------------------------

_SCRATCH = None


def scratch_state():
    global _SCRATCH
    if _SCRATCH is None:
        b = board.load_board()
        _SCRATCH = gs.setup_initial_game(b)
    return _SCRATCH


def build_hunt_state(ctx: HuntContext):
    """Reset the scratch state to the ctx fixture (fresh mutable objects).
    The pool shuffle IS the untouched engine's draw randomness; driven
    walks pick by index, so their pool order must stay replay-stable."""
    st = scratch_state()
    st.hunt_pool = [mk_tile(k) for k in ctx.pool]
    if _DRIVER is None:
        random.shuffle(st.hunt_pool)
    st.hunt_discard = [mk_tile(k) for k in ctx.discard]
    st.hunt_removed = []
    st.hunt_box = []
    for _ in range(ctx.eyes):
        d = gs.ActionDie(gs.SH_FACES, Side.SHADOW)
        d.current = gs.ActionDieFace.EYE
        st.hunt_box.append(d)
    for _ in range(ctx.fp_dice):
        d = gs.ActionDie(gs.FP_FACES, Side.FREE_PEOPLES)
        d.current = gs.ActionDieFace.CHARACTER
        st.hunt_box.append(d)

    comps = []
    for s in ctx.companions:
        ab = s.guide_ability if s.guide_ability is not None \
            else _NAME_ABILITY.get(s.name)
        comps.append(gs.Companion(name=s.name, level=s.level,
                                  leadership=s.leadership,
                                  guide_ability=ab))
    if comps:
        if ctx.guide:
            guide = next(c for c in comps if c.name == ctx.guide)
        else:
            guide = max(comps, key=lambda c: c.level)
    else:
        guide = gs.Companion('Gollum', 0, 0,
                             guide_ability='no_reveal_on_damage')
    st.fellowship = gs.Fellowship(
        companions=comps, guide=guide,
        region=st.board.region(ctx.region))
    st.fellowship.progress = ctx.progress
    st.fellowship.corruption = ctx.corruption
    st.fellowship.hidden = ctx.hidden
    st.fellowship.mordor_entered = ctx.in_mordor
    st.fellowship.mordor_step = ctx.mordor_step

    st.armies.clear()
    if ctx.nazgul_at_fs or ctx.army_at_fs:
        a = gl.get_or_create_army(st, ctx.region)
        if ctx.army_at_fs:
            a.units = [gs.Unit(gs.Nation.SAURON, gs.UnitKind.REGULAR)]
        if ctx.nazgul_at_fs:
            a.leaders = [gs.Nazgul(nation=gs.Nation.SAURON)]
    st.enemy_controlled = {Side.FREE_PEOPLES: set(), Side.SHADOW: set()}
    st.besieged_strongholds = set()

    pcs = PersistentCardState()
    if ctx.mithril:
        pcs.add_card(create_persistent_card(
            'FP_Character_hd_mithril', 'Mithril Coat and Sting',
            Side.FREE_PEOPLES))
    if ctx.wizard_staff:
        pcs.add_card(create_persistent_card(
            'FP_Character_hd_staff', "Wizard's Staff", Side.FREE_PEOPLES))
    if ctx.axe_horn:
        pcs.add_card(create_persistent_card(
            'FP_Character_hd_axebow', 'Axe and Bow', Side.FREE_PEOPLES))
    st.persistent_cards = pcs

    st._pending_hunt_damage = 0
    st.pending_hunt = None
    st.pending_card_option = None
    st.pending_guide_change = None
    st._mithril_stash = None
    st._wizardstaff_stash = None
    st._reveal_place_due = False
    st._fellowship_route_event = None
    st._pending_reveal_strongholds = ()
    st._hobbit_place_queue = []
    st._sh_tile_drawn_regions = None
    st._crebain_hunt_bonus = ctx.crebain_bonus
    st.fellowship_acted_in_mordor = False
    return st


def snapshot(st) -> dict:
    return {
        'comps': tuple(c.name for c in st.fellowship.companions),
    }


# --------------------------------------------------------------------------
# Observable extraction (shared with certification)
# --------------------------------------------------------------------------

OBS_FIELDS = ('corruption', 'hidden', 'fx', 'guide', 'tiles_used',
              'pool_left', 'progress', 'region', 'mordor_step', 'victory',
              'residual')


def observe(st, pre: dict) -> tuple:
    """Canonical outcome tuple; requires every pending fully resolved.
    A NEGATIVE accumulator is a terminal engine state (outside-Mordor
    FP-special tiles: no heal branch, the residual offsets the next
    hunt's damage) - carried as the 'residual' field, never rejected."""
    if st.pending_hunt is not None or st.pending_card_option is not None \
            or st.pending_guide_change is not None \
            or getattr(st, '_pending_hunt_damage', 0) > 0 \
            or getattr(st, '_hobbit_place_queue', None) \
            or getattr(st, '_reveal_place_due', False) \
            or getattr(st, '_fellowship_route_event', None) \
            or getattr(st, '_pending_reveal_strongholds', None):
        raise RuntimeError('observe() on unresolved hunt state')
    remaining = {c.name for c in st.fellowship.companions}
    placed = set()
    for a in st.armies.values():
        for l in a.leaders:
            if isinstance(l, gs.Companion):
                placed.add(l.name)
    fx = []
    for nm in pre['comps']:
        if nm not in remaining:
            fx.append(('P:' if nm in placed else 'E:') + nm)
    corr = st.fellowship.corruption
    victory = ''
    if corr >= 12:
        victory = 'sh'
    elif st.fellowship.mordor_entered and st.fellowship.mordor_step >= 5:
        victory = 'fp'
    return (corr,
            bool(st.fellowship.hidden),
            tuple(sorted(fx)),
            getattr(st.fellowship.guide, 'name', ''),
            tuple(sorted(tile_key(t) for t in st.hunt_discard)),
            tuple(sorted(tile_key(t) for t in st.hunt_pool)),
            st.fellowship.progress,
            gl.fellowship_region(st),
            st.fellowship.mordor_step,
            victory,
            int(getattr(st, '_pending_hunt_damage', 0) or 0))


# --------------------------------------------------------------------------
# RNG funnels + fork driver
# --------------------------------------------------------------------------

class HuntFork(BaseException):
    """BaseException: the ai.py-mirrored `except Exception` blocks in the
    walk must never swallow a fork."""


class HuntDriver:
    """Replays a plan of option indices; forks at the first unplanned
    stochastic request (combat-cert fork technique)."""
    __slots__ = ('plan', 'ptr', 'prob', 'pending', 'source')

    def __init__(self, plan, source):
        self.plan = plan
        self.ptr = 0
        self.prob = 1.0
        self.pending = None
        self.source = source

    def choose(self, options):
        # Action-finally upkeep can request another draw while HuntFork is
        # unwinding. Preserve the FIRST unresolved draw and its branch menu;
        # this partial walk must be discarded before any later draw runs.
        if self.pending is not None:
            raise HuntFork()
        if self.ptr < len(self.plan):
            i = self.plan[self.ptr]
            self.ptr += 1
            val, pr = options[i]
            self.prob *= pr
            return val
        self.pending = options
        raise HuntFork()

    def roll(self, n, ctxstr):
        return self.choose(self.source.roll_options(n, ctxstr))

    def pool_pick(self, pool):
        return self.choose(self.source.pool_options(pool))

    def comp_pick(self, cands):
        return self.choose(self.source.comp_options(cands))


_DRIVER: Optional[HuntDriver] = None
_ORIG_ROLL = gl.roll_d6_batch
_ORIG_POP = gl.pop_hunt_tile
_ORIG_PICK = gl.pick_hunt_tile
_ORIG_RANDOM = gl.random
_ORIG_SHUFFLE = random.shuffle


def _noop_shuffle(seq, *a, **k):
    # driven walks: every later draw is an enumerated uniform pick, so
    # pool order is irrelevant; a live shuffle would break plan replay
    return None


def _w_roll(n, ctx=''):
    if _DRIVER is None:
        return _ORIG_ROLL(n, ctx)
    return _DRIVER.roll(n, ctx)


def _w_pop(state):
    if _DRIVER is None:
        return _ORIG_POP(state)
    i = _DRIVER.pool_pick(state.hunt_pool)
    return state.hunt_pool.pop(i)


def _w_pick(state):
    if _DRIVER is None:
        return _ORIG_PICK(state)
    i = _DRIVER.pool_pick(state.hunt_pool)
    return state.hunt_pool[i]


class _RandomProxy:
    """Intercepts game_logic's random.choice on Companion lists only."""

    def __getattr__(self, name):
        return getattr(_ORIG_RANDOM, name)

    def choice(self, seq):
        if _DRIVER is not None and seq \
                and isinstance(seq[0], gs.Companion):
            return seq[_DRIVER.comp_pick(seq)]
        return _ORIG_RANDOM.choice(seq)


def _install(driver: HuntDriver):
    global _DRIVER
    _DRIVER = driver
    gl.roll_d6_batch = _w_roll
    gl.pop_hunt_tile = _w_pop
    gl.pick_hunt_tile = _w_pick
    gl.random = _RandomProxy()
    random.shuffle = _noop_shuffle


def _uninstall():
    global _DRIVER
    _DRIVER = None
    gl.roll_d6_batch = _ORIG_ROLL
    gl.pop_hunt_tile = _ORIG_POP
    gl.pick_hunt_tile = _ORIG_PICK
    gl.random = _ORIG_RANDOM
    random.shuffle = _ORIG_SHUFFLE


# --------------------------------------------------------------------------
# Analytic option source (the kernel's probability model)
# --------------------------------------------------------------------------

def _hit_faces(mod: int) -> int:
    """Faces (of 6) that hit at threshold 6: natural 1 never, else
    face+mod >= 6 (game_logic._hunt_roll's rule)."""
    return sum(1 for r in range(2, 7) if r + mod >= 6)


def _binom_terms(n: int, p: float):
    out = []
    for k in range(n + 1):
        pr = math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        if pr > 0.0:
            out.append((k, pr))
    return out


class AnalyticSource:
    """Sufficient-statistic branch model: hit counts w/ binomial mass,
    uniform collapsed tile kinds, uniform companions, uniform d6."""

    def __init__(self, ctx: HuntContext):
        self.p_hit = _hit_faces(ctx.fp_dice + ctx.crebain_bonus) / 6.0

    def roll_options(self, n, ctxstr):
        if ctxstr == 'hunt roll':
            return [([6] * h + [1] * (n - h), pr)
                    for h, pr in _binom_terms(n, self.p_hit)]
        if ctxstr == 'hunt reroll':
            return [([6], self.p_hit), ([1], 1.0 - self.p_hit)]
        if ctxstr == 'shelob d6 damage':
            return [([v], 1.0 / 6.0) for v in range(1, 7)]
        raise RuntimeError('unexpected roll ctx: %r' % ctxstr)

    def pool_options(self, pool):
        by_kind: Dict[str, int] = {}
        first: Dict[str, int] = {}
        for i, t in enumerate(pool):
            k = tile_key(t)
            by_kind[k] = by_kind.get(k, 0) + 1
            first.setdefault(k, i)
        n = float(len(pool))
        return [(first[k], c / n) for k, c in sorted(by_kind.items())]

    def comp_options(self, cands):
        n = float(len(cands))
        return [(i, 1.0 / n) for i in range(len(cands))]


# --------------------------------------------------------------------------
# The composed-engine walk (decision bodies = game_logic, 1.230 factoring;
# only the ai.py fellowship_move/card_option TAIL sequencing is mirrored)
# --------------------------------------------------------------------------

_CHOICE_ACTION = {'corruption': 'hunt_take_corruption',
                  'guide': 'hunt_sacrifice_guide',
                  'random': 'hunt_sacrifice_random',
                  'separate': 'hunt_guide_separate'}


def _do_gollum(st, ctx: HuntContext):
    kind = ('hunt_gollum_reveal' if ctx.gollum_reveal
            else 'hunt_gollum_no_reveal')
    menu = [a for a, _ in gl.hunt_decision_menu(st)]
    if kind not in menu:
        raise ValueError('choice %r unavailable (menu %r)' % (kind, menu))
    gl.resolve_hunt_decision(st, kind)


def _do_sacrifice(st, ctx: HuntContext, first: bool):
    choice = ctx.damage_choice if first else ctx.chain_choice
    kind = _CHOICE_ACTION[choice]
    menu = [a for a, _ in gl.hunt_decision_menu(st)]
    if kind not in menu:
        raise ValueError('choice %r unavailable (menu %r)'
                         % (choice, menu))
    gl.resolve_hunt_decision(st, kind)


def _resolve_pco(st, ctx: HuntContext):
    """Resolve one pending_card_option (learned inputs from ctx)."""
    from event_effects import CARD_EFFECT_HANDLERS
    pco = st.pending_card_option
    opts = pco.get('options') or []
    if any('wizardstaff_prevent' in (o or {}) for o in opts):
        st.pending_card_option = None
        CARD_EFFECT_HANDLERS["wizard's staff"](
            st, None, {'wizardstaff_prevent': bool(ctx.wizard_prevent)})
    elif any('mithril_redraw' in (o or {}) for o in opts):
        tile = st._mithril_stash['tile']
        st.pending_card_option = None
        CARD_EFFECT_HANDLERS['mithril coat and sting'](
            st, None,
            {'mithril_redraw': tile_key(tile) in ctx.mithril_redraw})
    elif pco.get('hunt_reduction'):
        # mirror ai card_option hunt_reduction branch
        st.pending_card_option = None
        if ctx.use_reduction:
            cid = pco.get('card_id')
            if cid is not None:
                st.persistent_cards.remove_card(cid)
            st._pending_hunt_damage = max(
                0, getattr(st, '_pending_hunt_damage', 0) - 1)
        gl._settle_hunt_damage(st)
        return True
    else:
        raise RuntimeError('unexpected pending_card_option: %r'
                           % (pco.get('card_name'),))
    # 1.230: the card_option resume runs the SAME full settle as the
    # fellowship_move tail (mirror of the unified ai tail)
    if getattr(st, '_pending_hunt_damage', 0) > 0 \
            and st.pending_hunt is None:
        if not gl._offer_hunt_reduction(st):
            gl._settle_hunt_damage(st)
    return True


def drive_hunt_event(st, ctx: HuntContext):
    """Run the composed self-play hunt resolution to quiescence.
    Decision bodies are game_logic.resolve_hunt_decision (1.230); the
    ai.py executor-tail sequencing is mirrored (ai stays import-free)."""
    gl.move_fellowship(st, skip_hunt=False, skip_auto_sacrifice=True)
    # ai fellowship_move tail settles pending damage immediately; here
    # the settle is DEFERRED while an interpose (pending_hunt /
    # pending_card_option) is open — the loop below settles it at
    # quiescence, so the composed effect is identical
    if getattr(st, '_pending_hunt_damage', 0) > 0 \
            and st.pending_hunt is None and st.pending_card_option is None:
        if not gl._offer_hunt_reduction(st):
            gl._settle_hunt_damage(st)
    first_sacrifice = True
    for _ in range(200):
        gl.settle_hunt_reveal(st)
        if st.pending_card_option is not None:
            _resolve_pco(st, ctx)
            continue
        ph = st.pending_hunt
        if ph is not None:
            phase = ph['phase']
            if phase == 'hunt_gollum':
                _do_gollum(st, ctx)
            elif phase == 'hunt_sacrifice':
                _do_sacrifice(st, ctx, first_sacrifice)
                first_sacrifice = False
            elif phase == 'hobbit_place':
                # canonical region input: place at the current region
                # (the ai region pick is a deferred learned decision)
                gl.resolve_hunt_decision(
                    st, 'hobbit_place',
                    place_region=gl.fellowship_region(st))
            else:
                raise RuntimeError('unknown pending_hunt phase %r'
                                   % phase)
            continue
        if st.pending_guide_change is not None:
            # deferred learned input: canonical tie-break, first LIVING
            # name — the candidate list was recorded when the tie
            # opened, and a chained hunt casualty may have killed that
            # companion since (change_guide on a dead name raises)
            cands = st.pending_guide_change.get('candidates') or []
            st.pending_guide_change = None
            alive = {c.name for c in st.fellowship.companions}
            cands = sorted(n for n in cands if n in alive)
            if cands:
                gl.change_guide(st, chosen=cands[0])
            continue
        if getattr(st, '_reveal_place_due', False) \
                and not getattr(st, '_pending_hunt_damage', 0):
            # mirror ai fellowship_move 1.132 tail
            st._reveal_place_due = False
            gl._auto_reveal_place(st)
            if getattr(st, '_pending_hunt_damage', 0) > 0:
                if not gl._offer_hunt_reduction(st):
                    gl._settle_hunt_damage(st)
            continue
        if getattr(st, '_pending_hunt_damage', 0) > 0:
            if not gl._offer_hunt_reduction(st):
                gl._settle_hunt_damage(st)
                if st.pending_hunt is None and st.pending_card_option \
                        is None and getattr(st, '_pending_hunt_damage',
                                            0) > 0:
                    raise RuntimeError('hunt damage settle stalled')
            continue
        if getattr(st, '_hobbit_place_queue', None):
            gl.pop_hobbit_place_queue(st)
            continue
        if (getattr(st, '_fellowship_route_event', None)
                or getattr(st, '_pending_reveal_strongholds', None)):
            from fellowship_reveal_route import drain
            drain(st)
            continue
        gl.return_mithril_tile_if_settled(st)
        return
    raise RuntimeError('drive_hunt_event: no quiescence in 200 steps')


# --------------------------------------------------------------------------
# Exact enumeration
# --------------------------------------------------------------------------

class HuntDist:
    """outcome tuple (OBS_FIELDS) -> probability, with marginal helpers."""

    def __init__(self, outcomes: Dict[tuple, float], ctx: HuntContext,
                 walks: int):
        self.outcomes = outcomes
        self.ctx = ctx
        self.walks = walks

    def total(self) -> float:
        return sum(self.outcomes.values())

    def corruption_delta_pmf(self) -> Dict[int, float]:
        base = self.ctx.corruption
        out: Dict[int, float] = {}
        for o, p in self.outcomes.items():
            d = o[0] - base
            out[d] = out.get(d, 0.0) + p
        return out

    def mean_corruption_delta(self) -> float:
        return sum(k * p for k, p in self.corruption_delta_pmf().items())

    def p_reveal(self) -> float:
        return sum(p for o, p in self.outcomes.items() if not o[1])


def enumerate_walks(ctx: HuntContext, source, max_nodes=2_000_000):
    """Exhaustive fork-driver enumeration of drive_hunt_event under the
    given stochastic-option source.  Returns ({obs: prob}, walk count)."""
    dist: Dict[tuple, float] = {}
    stack: List[list] = [[]]
    nodes = 0
    while stack:
        plan = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError('hunt enumeration node budget exceeded')
        drv = HuntDriver(plan, source)
        _install(drv)
        try:
            st = build_hunt_state(ctx)
            pre = snapshot(st)
            try:
                drive_hunt_event(st, ctx)
            except HuntFork:
                for i in range(len(drv.pending)):
                    stack.append(plan + [i])
                continue
            obs = observe(st, pre)
        finally:
            _uninstall()
        dist[obs] = dist.get(obs, 0.0) + drv.prob
    return dist, nodes


_MEMO: Dict[HuntContext, HuntDist] = {}


def exact_hunt_distribution(ctx: HuntContext) -> HuntDist:
    """Exact outcome distribution for one hunt event; memoized on ctx."""
    hit = _MEMO.get(ctx)
    if hit is not None:
        return hit
    outcomes, walks = enumerate_walks(ctx, AnalyticSource(ctx))
    d = HuntDist(outcomes, ctx, walks)
    total = d.total()
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError('hunt distribution mass %.12f != 1' % total)
    _MEMO[ctx] = d
    return d


# --------------------------------------------------------------------------
# Lookup tables
# --------------------------------------------------------------------------

def build_roll_table() -> Dict[Tuple[int, int, int], Tuple[float, ...]]:
    """(eyes 0-5, modifier 0-4, reroll sources 0-3) -> pmf of hunt
    successes 0-5 (closed form: binomial roll + capped binomial reroll)."""
    table: Dict[Tuple[int, int, int], Tuple[float, ...]] = {}
    for eyes in range(6):
        n = min(5, eyes)
        for mod in range(5):
            p = _hit_faces(mod) / 6.0
            for src in range(4):
                pmf = [0.0] * 6
                for h, ph in _binom_terms(n, p):
                    k = min(src, n - h)
                    for h2, ph2 in _binom_terms(k, p):
                        pmf[h + h2] += ph * ph2
                table[(eyes, mod, src)] = tuple(pmf)
    return table


