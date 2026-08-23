# combat_dist.py - EXACT one-round combat exchange distribution kernel.
"""
Exchange-distribution calculator for War of the Ring combat (ISMCTS kernel).

Computes the EXACT joint distribution over one combat round's observable
outcome -- (attacker hits, defender hits, pre-combat hits, bonus hits,
cancelled hits, surviving army composition, destroyed flags, leader
eliminations, deferred-casualty stash) -- for a given exchange context
(armies, leadership, field/siege/city thresholds, ordered combat-card
pair, White Rider shine, learned effect-choice magnitudes).

ARCHITECTURE (engine = single source of truth):
  * The DETERMINISTIC card phases (cancel/initiative, Words of Power,
    dice modifiers, dice-count modifiers, Dread & Despair, Relentless
    Assault, Confusion arming, Foul Stench re-roll cancel, They are
    Terrible) are executed by the REAL engine appliers from combat.py on
    a real CombatState built by the real setup_combat().  The kernel
    only snapshots the resulting round parameters.
  * The STOCHASTIC phases (pre-combat additional attacks, the combat
    roll, Mighty Attack / Anduril conversions, the leader re-roll,
    Fateful Strike / Blade of Westernesse / bonus-hit windows /
    Shield-Wall / Black Breath / Heroic Death, casualties, and the
    post-casualty Onslaught / We Come to Kill / Great Host effects) are
    enumerated analytically with binomial/multinomial algebra, phase
    logic mirrored line-for-line from combat.resolve_combat_round
    (RULING 1.228 initiative semantics: defender-first ties,
    resolution-time magnitudes, once-only leadership forfeit,
    battle-end interrupts).
  * cert_combat_dist.py certifies the mirror two ways: an exhaustive
    ENGINE-DRIVEN enumeration oracle (exact equality) and >=100k Monte
    Carlo rounds through the untouched engine path (TV distance vs
    multinomial noise).  It is a STANDING GATE: any combat.py edit
    that shifts a distribution fails the oracle leg immediately.

Search integration surface (ai.py kernel chance nodes):
  * exact_distribution(ctx, want_witness=True) additionally records ONE
    witness batch-statistic plan per outcome (engine batch order), and
    run_with_plan() drives the REAL engine apply through those
    sufficient statistics (witness faces) -- branch states are always
    engine-built, never hand-assembled.
  * card_play_posterior() is the closed-form opponent combat-card model
    (exchangeable-uniform hand posterior x greedy best-response).

One round only.  Multi-round battles, sieges pressing, retreats
(Scouts-class), and choice-bearing magnitudes remain tree decision
nodes; the kernel accepts the chosen magnitudes through `choices`.

cp1252-safe: no unicode in prints.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import combat as cb
import board
import game_state as gs
import game_logic
from cards import get_cards
from game_state import (Side, Nation, UnitKind, Companion, Nazgul)


# --------------------------------------------------------------------------
# Card lookup by printed combat name (accent-insensitive)
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


_CARD_CACHE: Dict[str, object] = {}
_CARD_ID_CACHE: Dict[str, str] = {}


def combat_card(combat_name: str):
    """CombatCard by printed combat_name; None passthrough."""
    if not combat_name:
        return None
    key = _norm(combat_name)
    if key not in _CARD_CACHE:
        for c in get_cards().values():
            if _norm(getattr(c, 'combat_name', '') or '') == key:
                _CARD_CACHE[key] = c.to_combat_card()
                break
        else:
            raise ValueError('combat card not found: %s' % combat_name)
    return _CARD_CACHE[key]


def combat_card_id(combat_name: str) -> Optional[str]:
    """A card id whose combat side prints combat_name (any copy)."""
    if not combat_name:
        return None
    key = _norm(combat_name)
    if key not in _CARD_ID_CACHE:
        for cid, c in get_cards().items():
            if _norm(getattr(c, 'combat_name', '') or '') == key:
                _CARD_ID_CACHE[key] = cid
                break
        else:
            raise ValueError('combat card not found: %s' % combat_name)
    return _CARD_ID_CACHE[key]


# --------------------------------------------------------------------------
# Context spec
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LeaderSpec:
    """kind: 'leader' | 'nazgul' | 'companion' | 'minion'."""
    kind: str
    name: str = ''
    level: int = 1
    leadership: int = 1
    nation: Optional[Nation] = None


@dataclass(frozen=True)
class ArmySpec:
    regulars: int = 0
    elites: int = 0
    nation: Nation = Nation.GONDOR
    elite_nation: Optional[Nation] = None      # default = nation
    leaders: Tuple[LeaderSpec, ...] = ()
    # second elite block (mixed-nation armies where the elite NATION is
    # math-bearing: Isengard elites + Saruman leadership rule)
    elites2: int = 0
    elite2_nation: Optional[Nation] = None


@dataclass(frozen=True)
class ExchangeContext:
    attacker: ArmySpec
    defender: ArmySpec
    attacker_card: Optional[str] = None        # printed combat_name
    defender_card: Optional[str] = None
    is_siege: bool = False
    city: bool = False                         # defender in city/fort, round 1
    shine: Optional[bool] = None               # GtW White Rider: None=legacy
    choices: Tuple[Tuple[str, object], ...] = ()   # engine choices dict items
    defer_casualties: bool = False             # ai.py in-tree round mode
    attacker_region: Optional[str] = None
    defender_region: Optional[str] = None
    # Saruman anywhere in play: Isengard Elites count as Leaders
    saruman_in_play: bool = False

    def choices_dict(self) -> dict:
        return dict(self.choices)


# --------------------------------------------------------------------------
# Fixture construction (shared with the certification harness)
# --------------------------------------------------------------------------

_SCRATCH = None
_REGION_PICKS: Dict[Tuple[str, str], str] = {}


def scratch_state():
    global _SCRATCH
    if _SCRATCH is None:
        b = board.load_board()
        st = gs.setup_initial_game(b)
        game_logic.initialize_decks(st)
        _SCRATCH = st
    return _SCRATCH


def _pick_region(side: Side, kind: str) -> str:
    """A region of the given settlement kind natively of `side`.
    kind: 'field' (no city/fort/stronghold), 'stronghold', 'city'."""
    key = (side.name, kind)
    if key in _REGION_PICKS:
        return _REGION_PICKS[key]
    scratch_state()          # board loaded
    fixed = {
        ('FREE_PEOPLES', 'field'): 'dagorlad',
        ('SHADOW', 'field'): 'gorgoroth',
        ('FREE_PEOPLES', 'stronghold'): 'minas tirith',
        ('SHADOW', 'stronghold'): 'morannon',
    }
    if key in fixed and fixed[key] in board.regions:
        _REGION_PICKS[key] = fixed[key]
        return fixed[key]
    for name, meta in sorted(board.regions.items()):
        is_sh = meta.get('is_stronghold', False)
        is_ci = meta.get('is_city', False)
        is_fo = meta.get('is_fort', False)
        fp = game_logic.is_fp_region(name)
        if side == Side.FREE_PEOPLES and not fp:
            continue
        if side == Side.SHADOW and fp:
            continue
        if kind == 'field' and not (is_sh or is_ci or is_fo):
            _REGION_PICKS[key] = name
            return name
        if kind == 'stronghold' and is_sh:
            _REGION_PICKS[key] = name
            return name
        if kind == 'city' and is_ci:
            _REGION_PICKS[key] = name
            return name
    raise RuntimeError('no region for %r' % (key,))


def _mk_leader(spec: LeaderSpec, default_nation: Nation):
    nat = spec.nation or default_nation
    if spec.kind == 'nazgul':
        return gs.Nazgul(nation=Nation.SAURON)
    if spec.kind == 'companion':
        return gs.Companion(name=spec.name, level=spec.level,
                            leadership=spec.leadership)
    if spec.kind == 'minion':
        return game_logic.Minion(nation=nat, name=spec.name)
    return gs.Leader(nation=nat)


def _mk_units(spec: ArmySpec) -> List[gs.Unit]:
    en = spec.elite_nation or spec.nation
    units = ([gs.Unit(spec.nation, UnitKind.REGULAR)
              for _ in range(spec.regulars)]
             + [gs.Unit(en, UnitKind.ELITE) for _ in range(spec.elites)])
    if spec.elites2:
        en2 = spec.elite2_nation or spec.nation
        units += [gs.Unit(en2, UnitKind.ELITE)
                  for _ in range(spec.elites2)]
    return units


def _army_side(spec: ArmySpec) -> Side:
    return cb.side_of_nation(spec.nation)


def build_round(ctx: ExchangeContext, state=None):
    """Build the engine CombatState for this exchange (fresh armies).

    Mirrors the test_combat_initiative fixture pattern: armies placed via
    game_logic.get_or_create_army, cards assigned AFTER setup_combat so
    the FP-legality nulling cannot drop them, dice reductions installed
    the way _resolve_combat_body does.
    """
    st = state if state is not None else scratch_state()
    st.armies.clear()
    st.fp_casualties = {}
    st.pending_combat = ({'gtw_shine': ctx.shine}
                         if ctx.shine is not None else None)
    atk_side = _army_side(ctx.attacker)
    def_side = _army_side(ctx.defender)
    def_kind = ('stronghold' if ctx.is_siege
                else ('city' if ctx.city else 'field'))
    atk_region = ctx.attacker_region or _pick_region(atk_side, 'field')
    def_region = ctx.defender_region or _pick_region(def_side, def_kind)

    a = game_logic.get_or_create_army(st, atk_region)
    a.units = _mk_units(ctx.attacker)
    a.leaders = [_mk_leader(ls, ctx.attacker.nation)
                 for ls in ctx.attacker.leaders]
    d = game_logic.get_or_create_army(st, def_region)
    d.units = _mk_units(ctx.defender)
    d.leaders = [_mk_leader(ls, ctx.defender.nation)
                 for ls in ctx.defender.leaders]

    if ctx.saruman_in_play:
        # Saruman-in-play rule (setup_combat scans ALL armies): plant
        # him OUTSIDE the battle unless a battle leader already is him
        _in_battle = any(
            type(l).__name__ == 'Minion'
            and 'saruman' in str(getattr(l, 'name', '')).lower()
            for l in (a.leaders + d.leaders))
        if not _in_battle:
            _sar_reg = next(
                (r for r in ('orthanc', 'moria', 'dol guldur')
                 if r in board.regions
                 and r not in (atk_region, def_region)), None)
            if _sar_reg is None:
                _sar_reg = next(
                    n for n in sorted(board.regions)
                    if not game_logic.is_fp_region(n)
                    and n not in (atk_region, def_region))
            s_army = game_logic.get_or_create_army(st, _sar_reg)
            s_army.leaders = [game_logic.Minion(nation=Nation.ISENGARD,
                                                name='Saruman')]

    c = cb.setup_combat(st, atk_region, def_region, None, None,
                        ctx.is_siege)
    c.attacker_card = combat_card(ctx.attacker_card)
    c.defender_card = combat_card(ctx.defender_card)
    ch = ctx.choices_dict()
    c.attacker_dice_reduction = int(ch.get('attacker_dice_reduction', 0)
                                    or 0)
    c.defender_dice_reduction = int(ch.get('defender_dice_reduction', 0)
                                    or 0)
    if ch.get('disable_city_defense'):
        c.city_defense = False
    return c


def round_choices(ctx: ExchangeContext) -> dict:
    ch = ctx.choices_dict()
    if ctx.defer_casualties:
        ch['defer_casualties'] = True
    return ch


# --------------------------------------------------------------------------
# Observable extraction (shared with MC / oracle certification)
# --------------------------------------------------------------------------

def _fx_desc(ld) -> str:
    if isinstance(ld, Companion):
        return 'C:' + ld.name
    if isinstance(ld, Nazgul):
        return 'N'
    if type(ld).__name__ == 'Minion':
        return 'M:' + str(getattr(ld, 'name', ''))
    return 'L'


def _re_counts(army) -> Tuple[int, int]:
    r = sum(1 for u in army.units if u.kind == UnitKind.REGULAR)
    e = sum(1 for u in army.units if u.kind == UnitKind.ELITE)
    return r, e


def observe(combat, atk_leaders0, def_leaders0) -> tuple:
    """Canonical observable tuple read from an engine CombatState after
    resolve_combat_round.  atk/def_leaders0: leader object lists captured
    right after build_round (pre-round rosters)."""
    ar, ae = _re_counts(combat.attacker_army)
    dr, de = _re_counts(combat.defender_army)
    cur_a = combat.attacker_army.leaders
    cur_d = combat.defender_army.leaders
    fxa = tuple(sorted(_fx_desc(l) for l in atk_leaders0
                       if not any(l is x for x in cur_a)))
    fxd = tuple(sorted(_fx_desc(l) for l in def_leaders0
                       if not any(l is x for x in cur_d)))
    return (combat.attacker_hits, combat.defender_hits,
            combat.attacker_pre_combat_hits,
            combat.defender_pre_combat_hits,
            combat.attacker_bonus_hits, combat.defender_bonus_hits,
            combat.attacker_hits_cancelled,
            combat.defender_hits_cancelled,
            ar, ae, dr, de,
            bool(combat.attacker_destroyed),
            bool(combat.defender_destroyed),
            fxa, fxd,
            bool(getattr(combat, 'casualties_deferred', False)),
            int(getattr(combat, 'deferred_hits_on_defender', 0) or 0),
            int(getattr(combat, 'deferred_hits_on_attacker', 0) or 0))


OBS_FIELDS = ('atk_hits', 'def_hits', 'atk_pre', 'def_pre', 'atk_bonus',
              'def_bonus', 'atk_cancelled', 'def_cancelled',
              'atk_r', 'atk_e', 'def_r', 'def_e',
              'atk_destroyed', 'def_destroyed', 'fx_atk', 'fx_def',
              'deferred', 'deferred_on_def', 'deferred_on_atk')


# --------------------------------------------------------------------------
# Probability helpers
# --------------------------------------------------------------------------

def _hit_faces(threshold: int, mod: int) -> int:
    """Faces (of 6) that HIT: natural 6 always, natural 1 never,
    else face+mod >= threshold.  (combat.combat_roll's rule.)"""
    return sum(1 for r in range(2, 7)
               if r == 6 or r + mod >= threshold)


def _binom_terms(n: int, p: float):
    """[(k, P(k))] for Binomial(n, p), zero-prob terms dropped."""
    out = []
    for k in range(n + 1):
        pr = math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))
        if pr > 0.0:
            out.append((k, pr))
    return out


def _trinom_terms(n: int, p_hit: float, p_one: float):
    """[( (hits, ones), P )] for the (hit / natural-1 / other-miss)
    trinomial over n dice, zero-prob terms dropped."""
    p_oth = 1.0 - p_hit - p_one
    out = []
    for h in range(n + 1):
        for o in range(n - h + 1):
            r = n - h - o
            pr = (math.factorial(n)
                  / (math.factorial(h) * math.factorial(o)
                     * math.factorial(r))
                  * (p_hit ** h) * (p_one ** o) * (p_oth ** r))
            if pr > 0.0:
                out.append(((h, o), pr))
    return out


def _apply_hits(r: int, e: int, h: int, policy: str) -> Tuple[int, int]:
    """Mirror combat.apply_casualties on (regulars, elites) counts.
    Downgrades always available (standalone path: _ACTIVE_STATE is None
    -> _consume_downgrade_source always allows)."""
    if policy == 'preserve_units':
        while h > 0 and (r + e) > 0:
            if e > 0:
                e -= 1
                r += 1          # downgrade absorbs the hit
            else:
                r -= 1
            h -= 1
        return r, e
    if policy == 'spend_elites':
        while h > 0 and (r + e) > 0:
            if e > 0:
                e -= 1
                if h >= 2:
                    h -= 2
                else:
                    r += 1
                    h -= 1
            else:
                r -= 1
                h -= 1
        return r, e
    # preserve_elites (default)
    k = min(r, h)
    r -= k
    h -= k
    while h > 0 and e > 0:
        e -= 1
        if h >= 2:
            h -= 2
        else:
            r += 1              # downgrade
            h -= 1
    return r, e


# --------------------------------------------------------------------------
# Stage 0: engine-applier deterministic setup + parameter snapshot
# --------------------------------------------------------------------------

class _SideParams:
    __slots__ = ('t', 'm', 'rr_m', 'cap', 'red', 'L3', 'L5',
                 'rr_cancelled', 'confused', 'track_ones',
                 'eff', 'name', 'has_card',
                 'ma_kind', 'ma_pay', 'ma_max', 'ma_blocked',
                 'captains', 'policy', 'pre_kind')

    def __init__(self):
        self.t = 5
        self.m = 0
        self.rr_m = 0
        self.cap = None
        self.red = 0
        self.L3 = 0
        self.L5 = 0
        self.rr_cancelled = False
        self.confused = False
        self.track_ones = False
        self.eff = ''            # lowercased combat_effect ('' if none)
        self.name = ''           # lowercased combat_name
        self.has_card = False    # card present and NOT cancelled
        self.ma_kind = None      # 'anduril' | 'ma' | None
        self.ma_pay = 0
        self.ma_max = 0
        self.ma_blocked = False
        self.captains = 0
        self.policy = 'preserve_elites'
        self.pre_kind = None     # 'ss' | 'charge' | 'db' | None


class _RoundParams:
    __slots__ = ('atk', 'dfn', 'def_first', 'naz',
                 'ra', 'ea', 'rd', 'ed',
                 'atk_leaders', 'def_leaders', 'companions',
                 'wop', 'wop_is_attacker',
                 'is_siege', 'city_r1', 'battle_over',
                 'destA', 'destD', 'choices', 'defer')


_CAPTAINS = cb.CAPTAINS_OF_THE_WEST


def _mk_side_params(c, is_attacker: bool, ch: dict,
                    conf_present: bool) -> _SideParams:
    s = _SideParams()
    card = c.attacker_card if is_attacker else c.defender_card
    cancelled = (c.attacker_card_cancelled if is_attacker
                 else c.defender_card_cancelled)
    s.has_card = bool(card) and not cancelled
    s.eff = card.combat_effect.lower() if s.has_card else ''
    s.name = card.combat_name.lower() if s.has_card else ''
    if is_attacker:
        s.t = 6 if (c.is_siege or (c.city_defense and c.siege_round == 1)) \
            else 5
        s.m = c.attacker_dice_mod
        s.rr_m = c.attacker_reroll_mod
        s.cap = c.attacker_dice_cap
        s.red = getattr(c, 'attacker_dice_reduction', 0)
        s.rr_cancelled = c.attacker_reroll_cancelled
        s.confused = c.attacker_confusion
        s.policy = ch.get('atk_casualty_policy', 'preserve_elites')
    else:
        s.t = 5
        s.m = c.defender_dice_mod
        s.rr_m = c.defender_reroll_mod
        s.cap = c.defender_dice_cap
        s.red = getattr(c, 'defender_dice_reduction', 0)
        s.rr_cancelled = c.defender_reroll_cancelled
        s.confused = c.defender_confusion
        s.policy = ch.get('def_casualty_policy', 'preserve_elites')
    s.track_ones = s.confused or conf_present or (1 + s.m >= s.t)
    # pre-combat additional attack (engine if/elif order)
    if s.has_card and ("before the combat roll" in s.eff
                       and "additional attack" in s.eff):
        if "leadership" in s.eff:
            s.pre_kind = 'ss'
        elif "elite units" in s.eff:
            s.pre_kind = 'charge'
        elif "three combat dice" in s.eff:
            s.pre_kind = 'db'
    # Mighty Attack / Anduril
    if s.has_card and ("forfeit" in s.eff and "leadership" in s.eff
                       and "change" in s.eff and "missed" in s.eff):
        s.ma_blocked = (getattr(c, '_foul_stench_cancel_atk', False)
                        if is_attacker
                        else getattr(c, '_foul_stench_cancel_def', False))
        shined = getattr(c, '_shined_companion', None)
        wop = getattr(c, '_wop_cancelled', None)
        avail = [x for x in c.companions_in_battle
                 if x is not shined and x is not wop]
        if "aragorn" in s.eff or "strider" in s.eff:
            heroes = [x for x in avail
                      if "aragorn" in x.name.lower()
                      or "strider" in x.name.lower()]
            if heroes:
                comp = heroes[0]
                crowned = (getattr(comp, 'crowned', False)
                           or getattr(comp, 'leadership', 1) >= 2)
                s.ma_kind = 'anduril'
                s.ma_max = 2 if crowned else 1
                s.ma_pay = getattr(comp, 'leadership', 1)
            # no Strider/Aragorn -> card refuses (returns 0)
        elif avail:
            target = min(avail, key=lambda x: getattr(x, 'leadership', 1))
            s.ma_kind = 'ma'
            s.ma_max = 1
            s.ma_pay = getattr(target, 'leadership', 1)
    return s


def stage0(ctx: ExchangeContext):
    """Run the engine's deterministic phases and snapshot round params."""
    c = build_round(ctx)
    ch = round_choices(ctx)
    cb.apply_cancel_effects(c)
    cb.apply_words_of_power(c)
    cb.apply_dice_modifiers(c)
    cb.apply_dice_count_modifiers(c)
    cb.apply_dread_and_despair(c, ch)
    cb.apply_relentless_assault(c, ch)

    p = _RoundParams()
    p.choices = ch
    p.defer = bool(ch.get('defer_casualties'))
    p.is_siege = c.is_siege
    p.city_r1 = c.city_defense and c.siege_round == 1
    p.battle_over = c.battle_over
    p.destA = c.attacker_destroyed
    p.destD = c.defender_destroyed
    p.ra, p.ea = _re_counts(c.attacker_army)
    p.rd, p.ed = _re_counts(c.defender_army)
    p.atk_leaders = list(c.attacker_army.leaders)
    p.def_leaders = list(c.defender_army.leaders)
    p.companions = list(c.companions_in_battle)
    p.wop = getattr(c, '_wop_cancelled', None)
    p.wop_is_attacker = getattr(c, '_wop_cancelled_is_attacker', False)
    p.def_first = cb._defender_first_initiative(c)

    # phase-3 leadership (pre TaT / pre MA forfeit)
    L3a = c.attacker_leadership
    L3d = c.defender_leadership

    if not c.battle_over:
        # hoisted deterministic phase 4/5 appliers (read no rolls)
        cb.apply_confusion_effect(c)
        cb.apply_reroll_cancellation(c)
        cb.apply_they_are_terrible(c)

    conf_present = c.attacker_confusion or c.defender_confusion
    p.atk = _mk_side_params(c, True, ch, conf_present)
    p.dfn = _mk_side_params(c, False, ch, conf_present)
    p.atk.L3 = L3a
    p.dfn.L3 = L3d
    p.atk.L5 = c.attacker_leadership
    p.dfn.L5 = c.defender_leadership
    p.naz = c.nazgul_in_battle

    # captains of the west (WoP-cancelled captain dropped on his side).
    # No own-side filter here: 1.233 battle rosters are own-side pure
    # (an enemy Companion can never appear in `leaders`), so the
    # engine's calc_combat_strength side skip has nothing to skip.
    def _capt(leaders, side_is_attacker):
        cancelled = (p.wop if p.wop is not None
                     and p.wop_is_attacker == side_is_attacker else None)
        n = 0
        for l in leaders:
            if isinstance(l, Companion) and l.name in _CAPTAINS \
                    and l is not cancelled:
                n += 1
        return n
    p.atk.captains = _capt(p.atk_leaders, True)
    p.dfn.captains = _capt(p.def_leaders, False)
    return p, c


# --------------------------------------------------------------------------
# The exact enumeration
# --------------------------------------------------------------------------

class ExchangeDist:
    """outcome tuple (see OBS_FIELDS) -> probability, plus hit-total
    marginals for decision-value helpers.  `witness` (optional): one
    engine-batch-order statistic plan per outcome for forced applies."""

    def __init__(self):
        self.outcomes: Dict[tuple, float] = {}
        self.hits_joint: Dict[Tuple[int, int], float] = {}
        self.witness: Optional[Dict[tuple, tuple]] = None

    def _add(self, obs: tuple, on_def: int, on_atk: int, pr: float):
        self.outcomes[obs] = self.outcomes.get(obs, 0.0) + pr
        k = (on_def, on_atk)
        self.hits_joint[k] = self.hits_joint.get(k, 0.0) + pr

    def total(self) -> float:
        return sum(self.outcomes.values())

    def mean_hits(self) -> Tuple[float, float]:
        md = sum(k[0] * p for k, p in self.hits_joint.items())
        ma = sum(k[1] * p for k, p in self.hits_joint.items())
        return md, ma

    def var_hits(self) -> Tuple[float, float]:
        md, ma = self.mean_hits()
        vd = sum((k[0] - md) ** 2 * p for k, p in self.hits_joint.items())
        va = sum((k[1] - ma) ** 2 * p for k, p in self.hits_joint.items())
        return vd, va


def exact_distribution(ctx: ExchangeContext,
                       want_witness: bool = False) -> ExchangeDist:
    if want_witness and not ctx.defer_casualties:
        raise ValueError('witness plans are defer-mode only (in-tree '
                         'rounds); full-round mode branches inside '
                         'phase 9')
    p, _c0 = stage0(ctx)
    dist = ExchangeDist()
    if want_witness:
        dist.witness = {}
    ch = p.choices
    _cur_plan = [()]

    def emit(prob, ah, dh, pa, pd, ba, bd, ca, cd,
             ra, ea, rd, ed, destA, destD, fxa, fxd,
             deferred=False, dhd=0, dha=0,
             on_def=None, on_atk=None):
        obs = (ah, dh, pa, pd, ba, bd, ca, cd, ra, ea, rd, ed,
               bool(destA), bool(destD),
               tuple(sorted(fxa)), tuple(sorted(fxd)),
               bool(deferred), int(dhd), int(dha))
        if on_def is None:
            on_def = pa + max(0, ah + ba - cd)
        if on_atk is None:
            on_atk = pd + max(0, dh + bd - ca)
        dist._add(obs, on_def, on_atk, prob)
        if dist.witness is not None and obs not in dist.witness:
            dist.witness[obs] = _cur_plan[0]

    # ---- stage 0 terminal (Relentless Assault self-wipe) ----
    if p.battle_over:
        emit(1.0, 0, 0, 0, 0, 0, 0, 0, 0,
             p.ra, p.ea, p.rd, p.ed, p.destA, p.destD, (), ())
        return dist

    # ---- Stage A: pre-combat additional attacks (interleaved) ----
    # A-state: (ra, ea, rd, ed, preA, preD, over, destA, destD) -> prob
    a_init = (p.ra, p.ea, p.rd, p.ed, 0, 0, False, False, False)
    a_states = {a_init: 1.0}
    a_plans: Dict[tuple, tuple] = {a_init: ()}
    order = [False, True] if p.def_first else [True, False]
    for is_att in order:
        sp = p.atk if is_att else p.dfn
        if sp.pre_kind is None:
            continue
        nxt: Dict[tuple, float] = {}
        nxt_p: Dict[tuple, tuple] = {}
        for st_, pr in a_states.items():
            ra, ea, rd, ed, preA, preD, over, dA, dD = st_
            base_plan = a_plans[st_]
            if over:
                nxt[st_] = nxt.get(st_, 0.0) + pr
                nxt_p.setdefault(st_, base_plan)
                continue
            if sp.pre_kind == 'ss':
                dice = min(5, sp.L3)
                t = 6 if (is_att and (p.is_siege or p.city_r1)) else 5
            elif sp.pre_kind == 'charge':
                my_e = ea if is_att else ed
                dice = min(5, my_e)
                t = 6 if (is_att and (p.is_siege or p.city_r1)) else 5
            else:                      # Durin's Bane
                dice = 3
                t = 4
            n = max(1, dice)
            ph = _hit_faces(t, 0) / 6.0
            for h, ppr in _binom_terms(n, ph):
                ra2, ea2, rd2, ed2 = ra, ea, rd, ed
                over2, dA2, dD2 = over, dA, dD
                preA2, preD2 = preA, preD
                if is_att:
                    preA2 += h
                    if h > 0:
                        rd2, ed2 = _apply_hits(rd, ed, h, p.dfn.policy)
                        if rd2 + ed2 == 0:
                            dD2, over2 = True, True
                else:
                    preD2 += h
                    if h > 0:
                        ra2, ea2 = _apply_hits(ra, ea, h, p.atk.policy)
                        if ra2 + ea2 == 0:
                            dA2, over2 = True, True
                k = (ra2, ea2, rd2, ed2, preA2, preD2, over2, dA2, dD2)
                nxt[k] = nxt.get(k, 0.0) + pr * ppr
                nxt_p.setdefault(k, base_plan + (h,))
        a_states = nxt
        a_plans = nxt_p

    # ---- per-A-state stochastic core ----
    reroll_memo: Dict[Tuple[int, int], list] = {}

    def _reroll_terms(k, h6):
        if k <= 0:
            return [(0, 1.0)]
        key = (k, h6)
        if key not in reroll_memo:
            reroll_memo[key] = _binom_terms(k, h6 / 6.0)
        return reroll_memo[key]

    for st_, pr0 in a_states.items():
        ra0, ea0, rd0, ed0, preA, preD, over, dA0, dD0 = st_
        planA = a_plans[st_]
        if over:
            _cur_plan[0] = planA
            emit(pr0, 0, 0, preA, preD, 0, 0, 0, 0,
                 ra0, ea0, rd0, ed0, dA0, dD0, (), (),
                 on_def=preA, on_atk=preD)
            continue

        # ---- Stage B: combat rolls ----
        def _n_dice(sp, units, capt):
            s = min(5, units + capt)
            eff = s - sp.red
            if sp.cap is not None:
                eff = min(eff, sp.cap)
            return max(1, eff)

        nA = _n_dice(p.atk, ra0 + ea0, p.atk.captains)
        nD = _n_dice(p.dfn, rd0 + ed0, p.dfn.captains)
        h6A = _hit_faces(p.atk.t, p.atk.m)
        h6D = _hit_faces(p.dfn.t, p.dfn.m)
        h6Ar = _hit_faces(p.atk.t, p.atk.rr_m)
        h6Dr = _hit_faces(p.dfn.t, p.dfn.rr_m)

        def _side_branches(sp, n, h6):
            """[( (hits, ones), P )]; ones=0 when untracked."""
            if sp.track_ones:
                return _trinom_terms(n, h6 / 6.0, 1.0 / 6.0)
            return [((h, 0), q) for h, q in _binom_terms(n, h6 / 6.0)]

        brA = _side_branches(p.atk, nA, h6A)
        brD = _side_branches(p.dfn, nD, h6D)

        def _post_roll(sp, n, hits, ones, t, m):
            """(conv, k, L_after) for one side given its combat stats."""
            L = sp.L5
            conv = 0
            if sp.ma_kind and not sp.ma_blocked:
                ma_miss = n - hits - (ones if (1 + m >= t) else 0)
                if ma_miss > 0:
                    conv = min(sp.ma_max, ma_miss) \
                        if sp.ma_kind == 'anduril' else 1
                    L = max(0, L - sp.ma_pay)
            oth = n - hits - ones
            if sp.confused:
                pool0 = oth
            elif 1 + m >= t:
                pool0 = oth
            else:
                pool0 = oth + ones
            pool = max(0, pool0 - conv)
            if sp.rr_cancelled or L <= 0:
                k = 0
            else:
                k = min(5, L, pool)
            return conv, k

        for (hA, oA), prA in brA:
            convA, kA = _post_roll(p.atk, nA, hA, oA, p.atk.t, p.atk.m)
            statA = (hA, oA) if p.atk.track_ones else hA
            for (hD, oD), prD in brD:
                convD, kD = _post_roll(p.dfn, nD, hD, oD,
                                       p.dfn.t, p.dfn.m)
                statD = (hD, oD) if p.dfn.track_ones else hD
                # confusion: a confused side's natural 1s score hits
                # for the OPPONENT (engine phase 4)
                base_atk = hA + convA + (oD if p.dfn.confused else 0)
                base_def = hD + convD + (oA if p.atk.confused else 0)
                for rrA, prra in _reroll_terms(kA, h6Ar):
                    for rrD, prrd in _reroll_terms(kD, h6Dr):
                        prob = pr0 * prA * prD * prra * prrd
                        _cur_plan[0] = (planA + (statA, statD)
                                        + ((rrA,) if kA > 0 else ())
                                        + ((rrD,) if kD > 0 else ()))
                        _finish(p, ch, dist, emit, prob,
                                ra0, ea0, rd0, ed0, preA, preD,
                                base_atk + rrA, base_def + rrD,
                                rrA, rrD)
    return dist


# --------------------------------------------------------------------------
# Stages C-G (deterministic given realized rolls) - mirrors combat.py
# phases 5d, 6, 7, 8, 9 line-for-line.
# --------------------------------------------------------------------------

def _finish(p, ch, dist, emit, prob, ra, ea, rd, ed, preA, preD,
            atkH, defH, rrA, rrD):
    rostA = list(p.atk_leaders)
    rostD = list(p.def_leaders)
    comps = list(p.companions)
    naz = p.naz
    fxa: List[str] = []
    fxd: List[str] = []
    bonusA = bonusD = 0
    cancA = cancD = 0          # cancX protects side X

    # ---- Phase 5d: Fateful Strike then Blade of Westernesse ----
    def _fs(sp, rh, enemy_rost, enemy_fx):
        nonlocal naz
        if not sp.has_card:
            return
        if "re-roll" not in sp.eff or "nazg" not in sp.eff:
            return
        if rh >= 1:
            if rh >= 2:
                for l in list(enemy_rost):
                    if type(l).__name__ == 'Minion':
                        enemy_rost.remove(l)
                        enemy_fx.append(_fx_desc(l))
                        return
            for l in list(enemy_rost):
                if isinstance(l, Nazgul):
                    enemy_rost.remove(l)
                    enemy_fx.append(_fx_desc(l))
                    naz = max(0, naz - 1)
                    return

    def _bow(sp, rh, enemy_rost, enemy_fx):
        if not sp.has_card:
            return
        if "eliminate" not in sp.eff or "minion" not in sp.eff \
                or "westernesse" not in sp.name:
            return
        if rh >= 1:
            for l in list(enemy_rost):
                if type(l).__name__ == 'Minion':
                    enemy_rost.remove(l)
                    enemy_fx.append(_fx_desc(l))
                    return

    _fs(p.atk, rrA, rostD, fxd)
    _fs(p.dfn, rrD, rostA, fxa)
    _bow(p.atk, rrA, rostD, fxd)
    _bow(p.dfn, rrD, rostA, fxa)

    # ---- Phase 6: bonus hits window (defender side always first) ----
    def _bonus(sp, is_att):
        nonlocal bonusA, bonusD
        if not sp.has_card:
            return
        at = atkH + preA + bonusA
        dt = defH + preD + bonusD
        my, en = (at, dt) if is_att else (dt, at)
        add = 0
        if ("score at least one hit" in sp.eff
                or "scores at least one hit" in sp.eff) \
                and "score one additional hit" in sp.eff:
            if my >= 1:
                add += 1
        if ("score at least one hit" in sp.eff
                or "scores at least one hit" in sp.eff) \
                and "score two additional hits" in sp.eff:
            if my >= 1:
                add += 2
        if "more total hits" in sp.eff \
                and "score one additional hit" in sp.eff:
            if my > en:
                add += 1
        if is_att:
            bonusA += add
        else:
            bonusD += add

    _bonus(p.dfn, False)
    _bonus(p.atk, True)

    # ---- Phase 7: Shield-Wall, then Black Breath / Heroic Death ----
    if p.atk.has_card and "two or more hits" in p.atk.eff \
            and "cancel one hit" in p.atk.eff:
        if defH >= 2:
            cancA += 1
    if p.dfn.has_card and "two or more hits" in p.dfn.eff \
            and "cancel one hit" in p.dfn.eff:
        if atkH >= 2:
            cancD += 1

    def _bb(sp, rh, enemy_rost, enemy_fx):
        if not sp.has_card:
            return
        if naz < 1:
            return
        if "re-roll" not in sp.eff or "eliminate" not in sp.eff \
                or "leader" not in sp.eff:
            return
        if "black breath" not in sp.name:
            return
        if rh < 1:
            return
        viable = [x for x in comps if x.level <= rh]
        if viable:
            target = max(viable, key=lambda x: x.level)
            comps.remove(target)
            for l in list(enemy_rost):
                if isinstance(l, Companion) and l.name == target.name:
                    enemy_rost.remove(l)
                    break
            enemy_fx.append('C:' + target.name)
            return
        for l in list(enemy_rost):
            if not isinstance(l, Companion) and not isinstance(l, Nazgul):
                enemy_rost.remove(l)
                enemy_fx.append(_fx_desc(l))
                return

    def _hd_sac(is_atk, rost, fx, fc, total_enemy):
        nonlocal cancA, cancD
        if fc <= 0 or total_enemy <= 0:
            return
        if fc == 1:
            for l in list(rost):
                if not isinstance(l, Companion) \
                        and not isinstance(l, Nazgul):
                    rost.remove(l)
                    fx.append(_fx_desc(l))
                    if is_atk:
                        cancA += 1
                    else:
                        cancD += 1
                    return
        viable = sorted((x for x in comps if x.level >= max(1, fc)),
                        key=lambda x: x.level)
        if not viable:
            viable = sorted(comps, key=lambda x: -x.level)
        if not viable:
            return
        target = viable[0]
        comps.remove(target)
        for l in list(rost):
            if isinstance(l, Companion) and l.name == target.name:
                rost.remove(l)
                break
        fx.append('C:' + target.name)
        amt = min(max(fc, 1), target.level, total_enemy)
        if is_atk:
            cancA += amt
        else:
            cancD += amt

    def _hd(sp, is_atk, rost, fx):
        nonlocal cancA, cancD
        if not sp.has_card or "heroic death" not in sp.name:
            return
        if is_atk:
            total_enemy = defH + preD + bonusD - cancA
        else:
            total_enemy = atkH + preA + bonusA - cancD
        total_enemy = max(0, total_enemy)
        army_size = (ra + ea) if is_atk else (rd + ed)
        fc = ch.get('atk_effect_count' if is_atk else 'def_effect_count')
        if fc is not None:
            _hd_sac(is_atk, rost, fx, int(fc), total_enemy)
            return
        if total_enemy <= army_size // 2:
            return
        best = None
        best_value = 0
        for x in comps:
            cancel = min(x.level, total_enemy)
            if cancel > best_value:
                best = x
                best_value = cancel
        if best is not None and best_value >= 2:
            comps.remove(best)
            for l in list(rost):
                if isinstance(l, Companion) and l.name == best.name:
                    rost.remove(l)
                    break
            fx.append('C:' + best.name)
            if is_atk:
                cancA += best_value
            else:
                cancD += best_value
            return
        for l in list(rost):
            if not isinstance(l, Companion) and not isinstance(l, Nazgul):
                rost.remove(l)
                fx.append(_fx_desc(l))
                if is_atk:
                    cancA += 1
                else:
                    cancD += 1
                return

    def _bb_hd_side(is_att):
        if is_att:
            _bb(p.atk, rrA, rostD, fxd)
            _hd(p.atk, True, rostA, fxa)
        else:
            _bb(p.dfn, rrD, rostA, fxa)
            _hd(p.dfn, False, rostD, fxd)

    if p.def_first:
        _bb_hd_side(False)
        _bb_hd_side(True)
    else:
        _bb_hd_side(True)
        _bb_hd_side(False)

    # ---- final hits ----
    finalA = max(0, atkH + bonusA - cancD)      # hits ON the defender
    finalD = max(0, defH + bonusD - cancA)      # hits ON the attacker

    if p.defer:
        # ai.py in-tree mode: stash and stop before casualties/phase 9
        emit(prob, atkH, defH, preA, preD, bonusA, bonusD,
             cancA, cancD, ra, ea, rd, ed, False, False, fxa, fxd,
             deferred=True, dhd=finalA, dha=finalD,
             on_def=preA + finalA, on_atk=preD + finalD)
        return

    # ---- Phase 8: casualties ----
    rd2, ed2 = _apply_hits(rd, ed, finalA, p.dfn.policy)
    ra2, ea2 = _apply_hits(ra, ea, finalD, p.atk.policy)
    destA = (ra2 + ea2 == 0)
    destD = (rd2 + ed2 == 0)
    over = destA or destD

    on_def = preA + finalA
    on_atk = preD + finalD

    # ---- Phase 9: post-casualty effects ----
    if over:
        emit(prob, atkH, defH, preA, preD, bonusA, bonusD,
             cancA, cancD, ra2, ea2, rd2, ed2, destA, destD, fxa, fxd,
             on_def=on_def, on_atk=on_atk)
        return

    # branch-carrying phase 9 (at most ONE side holds a Shadow
    # post-casualty card, but mirror the engine's two-side walk)
    def _g_side(state, is_att):
        """state: (ra,ea,rd,ed,bonusA,bonusD,destA,destD,over,
                   extra_on_def, extra_on_atk, prob_factor)
        returns list of successor states."""
        (sra, sea, srd, sed, sbA, sbD, sdA, sdD, sover,
         xd, xa, pf) = state
        sp = p.atk if is_att else p.dfn
        if sover:
            return [state]
        if not sp.has_card:
            return [state]
        eff = sp.eff
        my_r, my_e = (sra, sea) if is_att else (srd, sed)
        en_r, en_e = (srd, sed) if is_att else (sra, sea)

        is_onslaught = (("up to four additional hits" in eff
                         and "4+" in eff.replace("result of ", ""))
                        or ("four additional hits" in eff
                            and "score one hit" in eff))
        if is_onslaught:
            cap = 2 * my_e + my_r
            k = min(4, cap)
            if k > 0 and ch.get('defer_onslaught'):
                return [state]          # deferred to the decision node
            if en_r + en_e == 0 or k <= 0:
                return [state]
            m_r, m_e = _apply_hits(my_r, my_e, k, 'preserve_elites')
            out = []
            for succ, spr in _binom_terms(k, 0.5):
                e_r, e_e = en_r, en_e
                for _ in range(succ):
                    e_r, e_e = _apply_hits(e_r, e_e, 1,
                                           'preserve_elites')
                ndA = m_r + m_e == 0 if is_att else e_r + e_e == 0
                ndD = e_r + e_e == 0 if is_att else m_r + m_e == 0
                nover = ndA or ndD
                if is_att:
                    st2 = (m_r, m_e, e_r, e_e, sbA, sbD,
                           sdA or ndA, sdD or ndD, nover,
                           xd + succ, xa + k, pf * spr)
                else:
                    st2 = (e_r, e_e, m_r, m_e, sbA, sbD,
                           sdA or ndA, sdD or ndD, nover,
                           xd + k, xa + succ, pf * spr)
                # NOTE: the sacrificer's own k self-hits count as hits
                # applied to that side's army (self-inflicted).
                out.append(st2)
            return out
        if "elite" in eff and "additional" in eff \
                and "after removing casualties" in eff:
            if en_r + en_e == 0:
                return [state]
            ec = min(5, my_e)
            if ec <= 0:
                return [state]
            out = []
            for h, spr in _binom_terms(ec, _hit_faces(5, 0) / 6.0):
                e_r, e_e = (en_r, en_e) if h == 0 else \
                    _apply_hits(en_r, en_e, h, 'preserve_elites')
                nb_A = sbA + (h if is_att else 0)
                nb_D = sbD + (0 if is_att else h)
                dead = e_r + e_e == 0
                if is_att:
                    st2 = (sra, sea, e_r, e_e, nb_A, nb_D,
                           sdA, sdD or dead, sover or dead,
                           xd + h, xa, pf * spr)
                else:
                    st2 = (e_r, e_e, srd, sed, nb_A, nb_D,
                           sdA or dead, sdD, sover or dead,
                           xd, xa + h, pf * spr)
                out.append(st2)
            return out
        if "twice as many" in eff and "automatic hit" in eff:
            if en_r + en_e == 0:
                return [state]
            if (my_r + my_e) >= 2 * (en_r + en_e):
                e_r, e_e = _apply_hits(en_r, en_e, 1, 'preserve_elites')
                dead = e_r + e_e == 0
                if is_att:
                    return [(sra, sea, e_r, e_e, sbA + 1, sbD,
                             sdA, sdD or dead, sover or dead,
                             xd + 1, xa, pf)]
                return [(e_r, e_e, srd, sed, sbA, sbD + 1,
                         sdA or dead, sdD, sover or dead,
                         xd, xa + 1, pf)]
        return [state]

    states9 = [(ra2, ea2, rd2, ed2, bonusA, bonusD,
                destA, destD, over, 0, 0, 1.0)]
    order9 = [False, True] if p.def_first else [True, False]
    for is_att in order9:
        nxt = []
        for st9 in states9:
            nxt.extend(_g_side(st9, is_att))
        states9 = nxt

    for (sra, sea, srd, sed, sbA, sbD, sdA, sdD, sover,
         xd, xa, pf) in states9:
        emit(prob * pf, atkH, defH, preA, preD, sbA, sbD,
             cancA, cancD, sra, sea, srd, sed, sdA, sdD, fxa, fxd,
             on_def=on_def + xd, on_atk=on_atk + xa)


# --------------------------------------------------------------------------
# Witness-plan forced applies (branch STATE construction for ai.py):
# drive the REAL engine apply through one outcome's sufficient batch
# statistics, feeding witness faces -- never hand-build states.
# --------------------------------------------------------------------------

class PlanMismatch(Exception):
    """The engine consumed RNG batches the witness plan cannot drive
    (extra/raw batches, statistic not in the batch's support)."""


def _witness_faces(n, hits, ones, t, m):
    """Concrete faces realizing (hits, ones) under (t, m): hits as 6s,
    ones as 1s, other misses as the largest non-1 missing face."""
    faces = [6] * hits + [1] * ones
    rest = n - hits - ones
    if rest > 0:
        wm = next((f for f in (5, 4, 3, 2) if f + m < t), 1)
        faces += [wm] * rest
    return faces


def batch_options(n, t, m, track):
    """[(stat, P)] for one engine dice batch: stat = hits, or
    (hits, ones) when natural 1s are statistic-bearing."""
    h6 = _hit_faces(t, m)
    if track:
        return _trinom_terms(n, h6 / 6.0, 1.0 / 6.0)
    return _binom_terms(n, h6 / 6.0)


class _PlanDriver:
    __slots__ = ('plan', 'i')

    def __init__(self, plan):
        self.plan = tuple(plan)
        self.i = 0

    def pick(self, options):
        if self.i >= len(self.plan):
            raise PlanMismatch('plan exhausted (extra batch)')
        want = self.plan[self.i]
        self.i += 1
        for val, _pr in options:
            if val == want:
                return val
            # plain-h plan against a ones-tracked batch: witness o=0
            if isinstance(val, tuple) and not isinstance(want, tuple) \
                    and val[0] == want and val[1] == 0:
                return val
        raise PlanMismatch('stat %r not in batch support' % (want,))

    def done(self):
        return self.i == len(self.plan)


def run_with_plan(plan, conf_case, thunk):
    """Run `thunk` (a real engine apply) with combat dice batches driven
    by `plan` (engine-order sufficient statistics).  Restores patches
    and the `random` module state unconditionally; raises PlanMismatch
    if the plan and the engine's batch consumption do not line up
    exactly, or if any non-batch RNG was consumed."""
    import random as _rnd
    drv = _PlanDriver(plan)
    spec_box = {'spec': None}
    o_roll = cb.roll_d6_batch
    o_cr = cb.combat_roll
    o_lr = cb.leader_reroll

    def p_roll(n, ctx=''):
        spec = spec_box['spec']
        if spec is None:
            if ctx == 'sacrifice hits':
                # Onslaught's direct 4+ batch (post-casualty resume)
                spec = (4, 0, False)
            else:
                raise PlanMismatch('raw roll_d6_batch(%r)' % ctx)
        t, m, track = spec
        try:
            val = drv.pick(batch_options(n, t, m, track))
        except PlanMismatch as e:
            raise PlanMismatch('%s [batch ctx=%r n=%d t=%d m=%d]'
                               % (e, ctx, n, t, m))
        if isinstance(val, tuple):
            h, o = val
        else:
            h, o = val, 0
        return _witness_faces(n, h, o, t, m)

    def p_combat_roll(dice_count, threshold, modifier=0, cap=None):
        prev = spec_box['spec']
        spec_box['spec'] = (threshold, modifier,
                            conf_case or (1 + modifier >= threshold))
        try:
            return o_cr(dice_count, threshold, modifier, cap)
        finally:
            spec_box['spec'] = prev

    def p_leader_reroll(rolls, threshold, leadership, modifier=0,
                        confused=False, miss_modifier=None,
                        exclude_misses=0):
        prev = spec_box['spec']
        spec_box['spec'] = (threshold, modifier, False)
        try:
            return o_lr(rolls, threshold, leadership, modifier,
                        confused, miss_modifier, exclude_misses)
        finally:
            spec_box['spec'] = prev

    rng0 = _rnd.getstate()
    cb.roll_d6_batch = p_roll
    cb.combat_roll = p_combat_roll
    cb.leader_reroll = p_leader_reroll
    try:
        out = thunk()
    finally:
        cb.roll_d6_batch = o_roll
        cb.combat_roll = o_cr
        cb.leader_reroll = o_lr
        rng_dirty = _rnd.getstate() != rng0
        if rng_dirty:
            _rnd.setstate(rng0)
    if rng_dirty:
        raise PlanMismatch('non-batch RNG consumed during forced apply')
    if not drv.done():
        raise PlanMismatch('plan not fully consumed')
    return out


def plan_conf_case(ctx: ExchangeContext) -> bool:
    """Ones-tracking arm flag for run_with_plan: Confusion in the pair."""
    for nm in (ctx.attacker_card, ctx.defender_card):
        if nm and 'confusion' in nm.lower():
            return True
    return False


# --------------------------------------------------------------------------
# Opponent combat-card play posterior (design Section 2 / 3B closed form)
# --------------------------------------------------------------------------

def hyperg_no_hold(U: int, n: int, k: int) -> float:
    """P(a uniform n-subset of a U-card pool avoids all k marked cards)."""
    if k <= 0:
        return 1.0
    if U <= 0 or n <= 0:
        return 1.0
    if n > U or U - k < n:
        return 0.0
    return math.comb(U - k, n) / math.comb(U, n)


def card_play_posterior(pool_sizes: Dict[str, int],
                        hand_counts: Dict[str, int],
                        cands: Dict[str, Tuple[str, int, object]],
                        none_value: object
                        ) -> Tuple[Dict[str, float], float]:
    """Closed-form P(opponent plays each candidate combat card).

    Exchangeable-uniform posterior (per-deck independent n_d-subsets of
    the unseen pools U_d) x greedy best-response: the opponent plays the
    best-valued candidate they hold, declining when nothing held beats
    none_value.  cands: key -> (deck_tag, copies_in_pool, value); value
    ordering is total (ties broken by key -- copies of equal value split
    a hand's mass to exactly one of them, keeping the total exact).
    Returns ({key: P(plays)}, P(none))."""
    order = sorted(cands.items(),
                   key=lambda kv: (kv[1][2], str(kv[0])), reverse=True)
    out: Dict[str, float] = {}
    total = 0.0
    better = {d: 0 for d in pool_sizes}
    for key, (deck, copies, val) in order:
        if not (val > none_value):
            out[key] = 0.0
            better[deck] = better.get(deck, 0) + int(copies)
            continue
        p_nob = 1.0
        p_nob_nox = 1.0
        for d in pool_sizes:
            U = int(pool_sizes[d])
            n = int(hand_counts.get(d, 0))
            k = better.get(d, 0)
            kx = k + (int(copies) if d == deck else 0)
            p_nob *= hyperg_no_hold(U, n, k)
            p_nob_nox *= hyperg_no_hold(U, n, kx)
        p = max(0.0, p_nob - p_nob_nox)
        out[key] = p
        total += p
        better[deck] = better.get(deck, 0) + int(copies)
    return out, max(0.0, 1.0 - total)


# --------------------------------------------------------------------------
# Card coverage registry
# --------------------------------------------------------------------------

# v1-COVERED: the kernel models the full printed effect (choice-bearing
# magnitudes flow in via `choices`, as the engine's learned pendings do).
COVERED_CARDS = [
    'Daring Defiance', 'Swarm of Bats', 'Words of Power',
    'Charge', "Durin's Bane", 'Sudden Strike',
    'Advantageous Position', 'Brave Stand', 'Cruel as Death', 'Daylight',
    'Deadly Strife', 'Desperate Battle', 'Devilry of Orthanc',
    'Dread and Despair', "Ent's Rage", 'Foul Stench', 'Huorn-dark',
    'It is a Gift', 'Mumakil', 'One for the Dark Lord',
    'Relentless Assault', 'Servant of the Secret Fire', 'Valour',
    'Anduril', 'Confusion', 'Mighty Attack', 'They are Terrible',
    'Nameless Wood', 'No Quarter',
    'Black Breath', 'Blade of Westernesse', 'Fateful Strike',
    'Heroic Death', 'Shield-Wall',
    'Great Host', 'Onslaught', 'We Come to Kill',
]

# DEFERRED: effects that are decisions / movement / multi-round scope --
# the tree keeps them as decision nodes exactly as today.
DEFERRED_CARDS = {
    'Scouts': 'retreat-before-combat is a movement DECISION (adjacent '
              'free region / withdraw into siege) resolved before any '
              'dice; stays a tree decision node',
}

# Choice-bearing magnitudes the kernel takes as INPUT (decision nodes
# above the chance node pick them; engine keys atk/def_effect_count):
CHOICE_INPUT_CARDS = ['Dread and Despair', 'Relentless Assault',
                      'Heroic Death', 'Onslaught']
