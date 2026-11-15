# cert_combat_dist.py - certification of the combat_dist kernel.
"""
STANDING GATE for the exact combat-exchange kernel (run after ANY
combat.py / combat_dist.py change; --quick for the gate list).

Per-case legs:

  LEG 1 (exact): an ENGINE-DRIVEN exhaustive enumeration oracle.  The
  real resolve_combat_round runs once per RNG-branch path; the only
  patched function is the roll_d6_batch randomness funnel, replaced by
  a fork-driver that enumerates the sufficient statistics of each dice
  batch ((hits) or (hits, natural-1s)) with their exact multinomial
  probabilities and feeds the engine WITNESS faces realizing them.
  Kernel vs oracle must agree to float precision (TV ~ 0).

  LEG 2 (statistical): Monte Carlo rounds through the REAL, UNPATCHED
  engine path.  Empirical-vs-kernel TV compared against a parametric-
  bootstrap noise bound (mean + 3*sigma under H0).

Composed legs (Phase-9 deferred-path fix):

  RESUME: the ai.py deferred round COMPOSED end-to-end on the engine -
  resolve_combat_round(defer_casualties) -> policy allocation (Phase-8
  semantics) -> combat.resume_deferred_post_casualty - enumerated with
  the same fork driver and MC-sampled, must match the CERTIFIED
  full-round kernel distribution marginalized onto army outcomes.

  WITNESS: every defer-mode outcome's witness plan, forced through the
  real resolve_combat_round via combat_dist.run_with_plan, must
  reproduce exactly that outcome (certifies the ai.py kernel-node
  branch construction).

  POSTERIOR: combat_dist.card_play_posterior vs brute-force hand
  enumeration on hand-computable pools (opponent combat-card model).

Run:  python cert_combat_dist.py [--quick] [--mc N] [--only base_,defer_]
Writes data/cert_combat_dist_results.json + prints the table
(cp1252-safe).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import random
import sys
import time

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import combat as cb                                  # noqa: E402
from game_state import Nation                        # noqa: E402
import combat_dist as cd                             # noqa: E402
from combat_dist import (ExchangeContext, ArmySpec, LeaderSpec,
                         build_round, round_choices, observe,
                         exact_distribution)         # noqa: E402

OUT_JSON = os.path.join(_ROOT, 'data', 'cert_combat_dist_results.json')


# ==========================================================================
# LEG 1: engine-driven exhaustive enumeration oracle
# ==========================================================================

class _Fork(Exception):
    pass


class _Driver:
    __slots__ = ('plan', 'ptr', 'prob', 'pending')

    def __init__(self, plan):
        self.plan = plan
        self.ptr = 0
        self.prob = 1.0
        self.pending = None

    def choose(self, options):
        if self.ptr < len(self.plan):
            i = self.plan[self.ptr]
            self.ptr += 1
            val, pr = options[i]
            self.prob *= pr
            return val
        self.pending = options
        raise _Fork()


class _Enum:
    driver = None
    spec = None          # (threshold, modifier, track_ones)
    conf_case = False    # Confusion present in this case


_ORIG_ROLL = cb.roll_d6_batch
_ORIG_COMBAT_ROLL = cb.combat_roll
_ORIG_LEADER_REROLL = cb.leader_reroll


# The oracle shims mirror the engine signatures INCLUDING the G1
# `side=` keyword (whose dice these are) - audit CB-1.
def _enum_roll(n, ctx='', side=None):
    spec = _Enum.spec
    if spec is None:
        if ctx == 'sacrifice hits':      # Onslaught's direct 4+ batch
            t, m, track = 4, 0, False
        else:
            raise RuntimeError('unexpected raw roll_d6_batch: %r' % ctx)
    else:
        t, m, track = spec
    options = cd.batch_options(n, t, m, track)
    if track:
        h, o = _Enum.driver.choose(options)
    else:
        h, o = _Enum.driver.choose(options), 0
    return cd._witness_faces(n, h, o, t, m)


def _enum_combat_roll(dice_count, threshold, modifier=0, cap=None,
                      side=None):
    prev = _Enum.spec
    _Enum.spec = (threshold, modifier,
                  _Enum.conf_case or (1 + modifier >= threshold))
    try:
        return _ORIG_COMBAT_ROLL(dice_count, threshold, modifier, cap,
                                 side=side)
    finally:
        _Enum.spec = prev


def _enum_leader_reroll(rolls, threshold, leadership, modifier=0,
                        confused=False, miss_modifier=None,
                        exclude_misses=0, side=None):
    prev = _Enum.spec
    _Enum.spec = (threshold, modifier, False)
    try:
        return _ORIG_LEADER_REROLL(rolls, threshold, leadership,
                                   modifier, confused, miss_modifier,
                                   exclude_misses, side=side)
    finally:
        _Enum.spec = prev


def _run_engine_round(ctx):
    """Build the fixture and run one REAL engine round; returns the
    observable tuple."""
    c = build_round(ctx)
    la0 = list(c.attacker_army.leaders)
    ld0 = list(c.defender_army.leaders)
    cb.resolve_combat_round(c, round_choices(ctx))
    return observe(c, la0, ld0)


def oracle_distribution(run_fn, conf=False, max_nodes=2_000_000):
    """Exact engine distribution of run_fn's observable by exhaustive
    RNG-branch enumeration through the fork driver."""
    dist = {}
    stack = [[]]
    nodes = 0
    cb.roll_d6_batch = _enum_roll
    cb.combat_roll = _enum_combat_roll
    cb.leader_reroll = _enum_leader_reroll
    _Enum.conf_case = conf
    try:
        while stack:
            plan = stack.pop()
            nodes += 1
            if nodes > max_nodes:
                raise RuntimeError('oracle node budget exceeded')
            drv = _Driver(plan)
            _Enum.driver = drv
            try:
                obs = run_fn()
            except _Fork:
                for i in range(len(drv.pending)):
                    stack.append(plan + [i])
                continue
            dist[obs] = dist.get(obs, 0.0) + drv.prob
    finally:
        cb.roll_d6_batch = _ORIG_ROLL
        cb.combat_roll = _ORIG_COMBAT_ROLL
        cb.leader_reroll = _ORIG_LEADER_REROLL
        _Enum.driver = None
        _Enum.spec = None
    return dist, nodes


# ==========================================================================
# LEG 2: Monte Carlo through the untouched engine path
# ==========================================================================

def mc_distribution(run_fn, n_trials, seed):
    random.seed(seed)
    counts = {}
    for _ in range(n_trials):
        obs = run_fn()
        counts[obs] = counts.get(obs, 0) + 1
    return counts


# ==========================================================================
# Comparison metrics
# ==========================================================================

def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def project_state(dist_map):
    """State-faithful outcome projection obs[8:19] (rosters, destroyed
    flags, fx, deferred stash) — the space the in-tree kernel nodes
    branch on after outcome-class coarsening.  Certification compares
    kernel and oracle PROJECTED IDENTICALLY (never coarse-vs-fine)."""
    out = {}
    for o, p in dist_map.items():
        k = o[8:19]
        out[k] = out.get(k, 0.0) + p
    return out


def mc_tv(kernel, counts, n):
    keys = set(kernel) | set(counts)
    return 0.5 * sum(abs(kernel.get(k, 0.0) - counts.get(k, 0) / n)
                     for k in keys)


def bootstrap_tv_bound(kernel, n, m=300, seed=1234):
    """mean + 3*sigma of TV(empirical_n, kernel) under H0 via numpy
    multinomial bootstrap."""
    import numpy as np
    rng = np.random.default_rng(seed)
    probs = np.array(list(kernel.values()), dtype=float)
    probs = probs / probs.sum()
    draws = rng.multinomial(n, probs, size=m)
    emp = draws / float(n)
    tvs = 0.5 * np.abs(emp - probs[None, :]).sum(axis=1)
    return float(tvs.mean()), float(tvs.mean() + 3.0 * tvs.std())


# ==========================================================================
# Case matrix
# ==========================================================================

GON = Nation.GONDOR
SAU = Nation.SAURON
SE = Nation.SOUTHRON_EASTERLINGS


def L(n=1):
    return tuple(LeaderSpec('leader') for _ in range(n))


def NZ(n=1):
    return tuple(LeaderSpec('nazgul') for _ in range(n))


def C(name, level, ldr):
    return LeaderSpec('companion', name, level, ldr)


def M(name):
    return LeaderSpec('minion', name, nation=SAU)


def _sh(r, e=0, leaders=()):
    return ArmySpec(regulars=r, elites=e, nation=SAU, leaders=leaders)


def _se(r, e=0, leaders=()):
    return ArmySpec(regulars=r, elites=e, nation=SAU, elite_nation=SE,
                    leaders=leaders)


def _fp(r, e=0, leaders=()):
    return ArmySpec(regulars=r, elites=e, nation=GON, leaders=leaders)


def build_cases():
    cs = []

    def add(name, **kw):
        cs.append((name, ExchangeContext(**kw)))

    # ---- baselines: dice 1-5 x leadership 0-3, field + siege ----
    add('base_f_1v1_L0',
        attacker=_sh(1), defender=_fp(1))
    add('base_f_3v2_L1',
        attacker=_sh(3, 0, NZ(1)), defender=_fp(2, 0, L(1)))
    add('base_f_5v5_L3',
        attacker=_sh(3, 2, NZ(3)), defender=_fp(3, 2, L(3)))
    add('base_f_2v4_L2v0',
        attacker=_sh(2, 0, NZ(2)), defender=_fp(4))
    add('base_f_4v3_L0v2',
        attacker=_sh(4), defender=_fp(3, 0, L(2)))
    add('base_s_5v3_L2v1',
        attacker=_sh(3, 2, NZ(2)), defender=_fp(2, 1, L(1)),
        is_siege=True)
    add('base_s_2v2_L0',
        attacker=_sh(2), defender=_fp(2), is_siege=True)
    add('base_s_5v5_L3',
        attacker=_sh(3, 2, NZ(3)), defender=_fp(3, 2, L(3)),
        is_siege=True)
    add('base_c_4v3_L1',
        attacker=_sh(4, 0, NZ(1)), defender=_fp(3, 0, L(1)), city=True)
    add('base_f_fpatk_3v3_L1',            # FP attacking, none/none
        attacker=_fp(3, 0, L(1)), defender=_sh(3, 0, NZ(1)))
    add('base_s_fpatk_4v3_L1',            # FP assaulting a SH stronghold
        attacker=_fp(3, 1, L(1)), defender=_sh(2, 1, NZ(1)),
        is_siege=True)

    # ---- casualty policies (learned allocation inputs) ----
    add('policy_units_f_4v4',
        attacker=_sh(2, 2, NZ(1)), defender=_fp(2, 2, L(1)),
        choices=(('def_casualty_policy', 'preserve_units'),
                 ('atk_casualty_policy', 'spend_elites')))

    # ---- single-card cases ----
    add('shieldwall_def_f',
        attacker=_sh(4, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        defender_card='Shield-Wall')
    add('shieldwall_def_s',
        attacker=_sh(4, 1, NZ(2)), defender=_fp(2, 1, L(1)),
        defender_card='Shield-Wall', is_siege=True)
    add('heroicdeath_def_choice2',
        attacker=_sh(4, 0, NZ(2)), defender=_fp(4, 0, (LeaderSpec('leader'),
                                                       C('Boromir', 2, 1))),
        defender_card='Heroic Death',
        choices=(('def_effect_count', 2),))
    add('heroicdeath_def_heuristic',
        attacker=_sh(5, 0, NZ(2)),
        defender=_fp(2, 0, (LeaderSpec('leader'), C('Boromir', 2, 1))),
        defender_card='Heroic Death')
    add('blackbreath_atk_f',
        attacker=_sh(4, 0, NZ(2)),
        defender=_fp(3, 0, (C('Meriadoc', 1, 1), C('Legolas', 2, 1))),
        attacker_card='Black Breath')
    add('durinsbane_atk_f',
        attacker=_sh(4, 0, NZ(1)), defender=_fp(2, 1, L(1)),
        attacker_card="Durin's Bane")
    add('durinsbane_wipe_f',
        attacker=_sh(4), defender=_fp(1),
        attacker_card="Durin's Bane")
    add('suddenstrike_def_f',
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Aragorn', 3, 2), C('Legolas', 2, 1))),
        defender_card='Sudden Strike')
    add('suddenstrike_atk_f',
        attacker=ArmySpec(regulars=3, nation=GON,
                          leaders=(C('Aragorn', 3, 2), L(1)[0])),
        defender=_sh(3, 0, NZ(1)),
        attacker_card='Sudden Strike')
    add('charge_atk_vs_db_def',           # interaction J pairing
        attacker=ArmySpec(regulars=0, elites=3, nation=GON),
        defender=_sh(5),
        attacker_card='Charge', defender_card="Durin's Bane")
    add('charge_def_f',
        attacker=_sh(4, 0, NZ(1)),
        defender=ArmySpec(regulars=2, elites=2, nation=GON, leaders=L(1)),
        defender_card='Charge')
    add('mumakil_atk_vs_noquarter_def',   # interaction F5a
        attacker=_se(2, 2, NZ(1)), defender=_fp(3, 0, L(1)),
        attacker_card='Mumakil', defender_card='No Quarter')
    add('noquarter_atk_vs_mumakil_def',   # interaction F5b
        attacker=ArmySpec(regulars=3, elites=1, nation=GON, leaders=L(1)),
        defender=_se(2, 2, NZ(1)),
        attacker_card='No Quarter', defender_card='Mumakil')
    add('namelesswood_def_f',
        attacker=_sh(4, 0, NZ(1)), defender=_fp(3, 1, L(1)),
        defender_card='Nameless Wood')
    add('desperatebattle_atk_f',
        attacker=_sh(3, 1, NZ(2)), defender=_fp(3, 0, L(1)),
        attacker_card='Desperate Battle')
    add('deadlystrife_atk_f',
        attacker=_sh(3, 1, NZ(2)), defender=_fp(3, 0, L(2)),
        attacker_card='Deadly Strife')
    add('wctk_atk_f',
        attacker=_sh(3, 2, NZ(1)), defender=_fp(4, 0, L(1)),
        attacker_card='We Come to Kill')
    add('wctk_atk_vs_daylight_def',       # interaction E1
        attacker=_sh(0, 5, NZ(1)), defender=_fp(3, 0, L(1)),
        attacker_card='We Come to Kill', defender_card='Daylight')

    # ---- White Rider / forfeit interactions ----
    fpdd = _fp(3, 0, (C('Legolas', 2, 1), C('Aragorn', 3, 2)))
    add('wop_atk_vs_dd_def',              # interaction A normal
        attacker=_sh(4, 0, NZ(1)), defender=fpdd,
        attacker_card='Words of Power', defender_card='Daring Defiance')
    add('wop_atk_vs_dd_def_WR',           # interaction A White Rider
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Gandalf the White', 3, 1),
                            C('Legolas', 2, 1), C('Aragorn', 3, 2))),
        attacker_card='Words of Power', defender_card='Daring Defiance',
        shine=True)
    add('sob_atk_vs_dd_def',              # interaction B(i)
        attacker=_sh(5, 0, NZ(1)),
        defender=_fp(3, 0, (C('Legolas', 2, 1), C('Gimli', 2, 1))),
        attacker_card='Swarm of Bats', defender_card='Daring Defiance')
    add('dd_atk_vs_sob_def',              # interaction B(ii)
        attacker=ArmySpec(regulars=3, nation=GON,
                          leaders=(C('Legolas', 2, 1), C('Gimli', 2, 1))),
        defender=_sh(5, 0, NZ(1)),
        attacker_card='Daring Defiance', defender_card='Swarm of Bats')
    add('cruelasdeath_atk_f',
        attacker=_sh(5, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        attacker_card='Cruel as Death')
    add('cruelasdeath_atk_WR',            # interaction I
        attacker=_sh(5, 0, NZ(2)),
        defender=_fp(3, 0, (C('Gandalf the White', 3, 1),)),
        attacker_card='Cruel as Death', shine=True)
    add('terrible_atk_f',
        attacker=_sh(5, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        attacker_card='They are Terrible')
    add('terrible_atk_WR',
        attacker=_sh(5, 0, NZ(2)),
        defender=_fp(3, 0, (C('Gandalf the White', 3, 1),)),
        attacker_card='They are Terrible', shine=True)
    add('dread_atk_choice2',
        attacker=_sh(5, 0, NZ(3)), defender=_fp(4, 0, L(1)),
        attacker_card='Dread and Despair',
        choices=(('atk_effect_count', 2),))
    add('dread_atk_WR',
        attacker=_sh(5, 0, NZ(3)),
        defender=_fp(4, 0, (C('Gandalf the White', 3, 1),)),
        attacker_card='Dread and Despair', shine=True,
        choices=(('atk_effect_count', 2),))
    add('sob_atk_nocard_def',             # SoB re-roll bonus clause
        attacker=_sh(4, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        attacker_card='Swarm of Bats')

    # ---- re-roll window cards ----
    add('foulstench_atk_vs_mightyattack_def',   # interaction K1
        attacker=_sh(4, 0, NZ(3)),
        defender=_fp(3, 0, (C('Legolas', 2, 1),)),
        attacker_card='Foul Stench', defender_card='Mighty Attack')
    add('mightyattack_def_f',
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Legolas', 2, 1), C('Gimli', 2, 1))),
        defender_card='Mighty Attack')
    add('anduril_def_f',
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Aragorn', 3, 2),)),
        defender_card='Anduril')
    add('confusion_def_f',
        attacker=_sh(4, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        defender_card='Confusion')
    add('fs_atk_vs_bb_def',               # interaction N pairing
        attacker=ArmySpec(regulars=5, nation=GON,
                          leaders=(C('Meriadoc', 1, 1),)),
        defender=_sh(3, 0, NZ(2)),
        attacker_card='Fateful Strike', defender_card='Black Breath')
    add('bow_atk_f',
        attacker=ArmySpec(regulars=4, nation=GON,
                          leaders=(C('Meriadoc', 1, 1),
                                   C('Legolas', 2, 1))),
        defender=_sh(3, 0, (M('Mouth of Sauron'),) + NZ(1)),
        attacker_card='Blade of Westernesse')
    add('fs_atk_minion_f',
        attacker=ArmySpec(regulars=4, nation=GON,
                          leaders=(C('Meriadoc', 1, 1),
                                   C('Legolas', 2, 1))),
        defender=_sh(3, 0, (M('Mouth of Sauron'),) + NZ(1)),
        attacker_card='Fateful Strike')
    add('wop_atk_vs_ss_def',              # interaction M pairing
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Aragorn', 3, 2), C('Legolas', 2, 1))),
        attacker_card='Words of Power', defender_card='Sudden Strike')

    # ---- dice-count / modifier cards ----
    add('daylight_def_f',
        attacker=_sh(4, 1, NZ(2)), defender=_fp(3, 0, L(1)),
        defender_card='Daylight')
    add('huorndark_def_vs_db_atk',        # interaction E2
        attacker=_sh(5), defender=_fp(4, 0, L(1)),
        attacker_card="Durin's Bane", defender_card='Huorn-dark')
    add('advpos_def_f',
        attacker=_sh(4, 0, NZ(1)), defender=_fp(3, 0, L(1)),
        defender_card='Advantageous Position')
    add('bravestand_def_f',
        attacker=_sh(5, 0, NZ(1)),
        defender=_fp(3, 0, (C('Legolas', 2, 1), C('Gimli', 2, 1))),
        defender_card='Brave Stand')
    add('valour_def_f',
        attacker=_sh(4, 0, NZ(1)), defender=_fp(2, 2, L(1)),
        defender_card='Valour')
    add('entsrage_def_vs_deadlystrife_atk_oneedge',   # 1+m>=t edge
        attacker=_sh(3, 1, NZ(1)), defender=_fp(3, 1, L(1)),
        attacker_card='Deadly Strife', defender_card="Ent's Rage")

    # ---- self-damage / post-casualty Shadow cards ----
    add('relentless_atk_choice2',
        attacker=_sh(4, 1, NZ(1)), defender=_fp(3, 0, L(1)),
        attacker_card='Relentless Assault',
        choices=(('atk_effect_count', 2),))
    add('relentless_atk_selfwipe',
        attacker=_sh(1), defender=_fp(3, 0, L(1)),
        attacker_card='Relentless Assault',
        choices=(('atk_effect_count', 2),))
    add('greathost_atk_f',
        attacker=_sh(6), defender=_fp(2, 0, L(1)),
        attacker_card='Great Host')
    add('onslaught_atk_auto',
        attacker=_sh(4, 2, NZ(1)), defender=_fp(4, 0, L(1)),
        attacker_card='Onslaught')

    # ---- extra edge-path coverage ----
    add('suddenstrike_atk_siege',         # FP assault: pre-combat at 6+
        attacker=ArmySpec(regulars=4, nation=GON,
                          leaders=(C('Aragorn', 3, 2), C('Legolas', 2, 1))),
        defender=_sh(2, 1, NZ(1)),
        attacker_card='Sudden Strike', is_siege=True)
    add('charge_atk_siege',
        attacker=ArmySpec(regulars=2, elites=3, nation=GON,
                          leaders=L(1)),
        defender=_sh(2, 1, NZ(1)),
        attacker_card='Charge', is_siege=True)
    add('blackbreath_atk_fallback',       # no viable comp -> plain leader
        attacker=_sh(4, 0, NZ(2)),
        defender=_fp(3, 0, (C('Legolas', 2, 1), LeaderSpec('leader'))),
        attacker_card='Black Breath')
    add('heroicdeath_def_choice1',        # fc=1: plain-leader sacrifice
        attacker=_sh(4, 0, NZ(2)),
        defender=_fp(4, 0, (LeaderSpec('leader'), C('Boromir', 2, 1))),
        defender_card='Heroic Death',
        choices=(('def_effect_count', 1),))
    add('wop_atk_targets_gtw_WR',         # WoP on shining GtW: restore
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Gandalf the White', 3, 1),
                            C('Meriadoc', 1, 1))),
        attacker_card='Words of Power', shine=True)
    add('anduril_def_strider',            # uncrowned: 1 convert max
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Strider', 3, 1),)),
        defender_card='Anduril')
    add('mumakil_atk_vs_suddenstrike_def',   # F2: pre hits count
        attacker=_se(2, 2, NZ(1)),
        defender=_fp(3, 0, (C('Aragorn', 3, 2),)),
        attacker_card='Mumakil', defender_card='Sudden Strike')
    add('city_disabled_4v3',              # ai round>0: city 6+ dropped
        attacker=_sh(4, 0, NZ(1)), defender=_fp(3, 0, L(1)), city=True,
        choices=(('disable_city_defense', True),))
    add('greathost_def_f',                # SH defender owns Great Host
        attacker=ArmySpec(regulars=2, nation=GON, leaders=L(1)),
        defender=_sh(6),
        defender_card='Great Host')
    add('wctk_def_f',                     # SH defender owns WCtK
        attacker=ArmySpec(regulars=4, nation=GON, leaders=L(1)),
        defender=_sh(2, 2, NZ(1)),
        defender_card='We Come to Kill')

    # ---- audit 2026-08-29 (CB) pairings ----
    add('durinsbane_atk_vs_heroicdeath_def',   # CB-2: HD scope excludes pre-combat hits
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(4, 0, (LeaderSpec('leader'), C('Boromir', 2, 1))),
        attacker_card="Durin's Bane", defender_card='Heroic Death',
        choices=(('def_effect_count', 2),))
    add('wop_atk_vs_ss_def_L0',            # CB-3: Sudden Strike at Leadership 0 rolls nothing
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Legolas', 2, 1),)),
        attacker_card='Words of Power', defender_card='Sudden Strike')
    add('charge_atk_2E_vs_db_def',         # CB-3: Charge after DB kills both Elites
        attacker=ArmySpec(regulars=0, elites=2, nation=GON),
        defender=_sh(5),
        attacker_card='Charge', defender_card="Durin's Bane")
    add('foulstench_atk_WR',               # CB-4: shine negates the compared Nazgul Leadership
        attacker=_sh(4, 0, NZ(3)),
        defender=_fp(3, 0, (C('Gandalf the White', 3, 1),
                            LeaderSpec('leader'))),
        attacker_card='Foul Stench', shine=True)
    add('foulstench_atk_stranded_naz',     # CB-6: FS gate is card-exact (1.233 stranded Nazgul)
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (LeaderSpec('leader'),) + NZ(1)),
        attacker_card='Foul Stench')
    add('blackbreath_atk_target',          # CB-7: the Shadow names the figure
        attacker=_sh(4, 0, NZ(2)),
        defender=_fp(3, 0, (C('Meriadoc', 1, 1), C('Legolas', 2, 1),
                            LeaderSpec('leader'))),
        attacker_card='Black Breath',
        choices=(('atk_effect_target', 'Meriadoc'),))
    add('heroicdeath_def_target',          # CB-7: the FP names the Companion
        attacker=_sh(4, 0, NZ(2)),
        defender=_fp(4, 0, (C('Boromir', 2, 1), C('Gimli', 2, 1),
                            LeaderSpec('leader'))),
        defender_card='Heroic Death',
        choices=(('def_effect_count', 2), ('def_effect_target', 'Gimli')))
    add('wop_atk_target_vs_ss_def',        # CB-7: WoP names Legolas (SS keeps Aragorn's 2)
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Aragorn', 3, 2), C('Legolas', 2, 1))),
        attacker_card='Words of Power', defender_card='Sudden Strike',
        choices=(('atk_effect_target', 'Legolas'),))
    add('wctk_atk_vs_units_policy',        # 1.254: enemy hits under the enemy's policy
        attacker=_sh(3, 2, NZ(1)), defender=_fp(2, 2, L(1)),
        attacker_card='We Come to Kill',
        choices=(('def_casualty_policy', 'preserve_units'),))

    # ---- Saruman / Isengard context bits (ai.py ArmySpec mapping) ----
    add('saruman_isengard_elites_f',
        attacker=ArmySpec(regulars=2, nation=SAU, elite_nation=SAU,
                          elites=0, elites2=3,
                          elite2_nation=Nation.ISENGARD),
        defender=_fp(3, 0, L(1)),
        saruman_in_play=True)

    # ---- ai.py in-tree round mode (defer_casualties) ----
    add('defer_base_f_3v3_L1',
        attacker=_sh(3, 0, NZ(1)), defender=_fp(3, 0, L(1)),
        defer_casualties=True)
    add('defer_wctk_atk',                 # stash stops before Phase 9
        attacker=_sh(3, 2, NZ(1)), defender=_fp(4, 0, L(1)),
        attacker_card='We Come to Kill', defer_casualties=True)
    add('defer_shieldwall_def',
        attacker=_sh(4, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        defender_card='Shield-Wall', defer_casualties=True)
    # defer-mode witness coverage (pre-combat, trinomial, MA/rerolls,
    # siege/city thresholds, Great Host present)
    add('defer_charge_atk_f',
        attacker=ArmySpec(regulars=1, elites=3, nation=GON,
                          leaders=L(1)),
        defender=_sh(4, 0, NZ(1)),
        attacker_card='Charge', defer_casualties=True)
    add('defer_confusion_def_f',
        attacker=_sh(4, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        defender_card='Confusion', defer_casualties=True)
    add('defer_mightyattack_def_f',
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Legolas', 2, 1), C('Gimli', 2, 1))),
        defender_card='Mighty Attack', defer_casualties=True)
    add('defer_siege_5v3',
        attacker=_sh(3, 2, NZ(2)), defender=_fp(2, 1, L(1)),
        is_siege=True, defer_casualties=True)
    add('defer_city_4v3',
        attacker=_sh(4, 0, NZ(1)), defender=_fp(3, 0, L(1)), city=True,
        defer_casualties=True)
    add('defer_greathost_atk_f',
        attacker=_sh(6), defender=_fp(2, 0, L(1)),
        attacker_card='Great Host', defer_casualties=True)
    add('defer_suddenstrike_def_f',
        attacker=_sh(4, 0, NZ(1)),
        defender=_fp(3, 0, (C('Aragorn', 3, 2), C('Legolas', 2, 1))),
        defender_card='Sudden Strike', defer_casualties=True)
    add('defer_charge_atk_2E_vs_db_def',  # CB-3 witness: the skipped 0-dice batch
        attacker=ArmySpec(regulars=0, elites=2, nation=GON),
        defender=_sh(5), attacker_card='Charge',
        defender_card="Durin's Bane", defer_casualties=True)
    return cs


# ==========================================================================
# Composed deferred-path RESUME leg (Phase-9 engine fix)
# ==========================================================================

def build_resume_cases():
    """(name, base ctx WITHOUT defer flag, defer_onslaught)"""
    cs = []

    def add(name, defer_onslaught=False, **kw):
        cs.append((name, ExchangeContext(**kw), defer_onslaught))

    add('resume_base_f_3v3',
        attacker=_sh(3, 0, NZ(1)), defender=_fp(3, 0, L(1)))
    add('resume_wctk_atk',
        attacker=_sh(3, 2, NZ(1)), defender=_fp(4, 0, L(1)),
        attacker_card='We Come to Kill')
    add('resume_wctk_def',
        attacker=ArmySpec(regulars=4, nation=GON, leaders=L(1)),
        defender=_sh(2, 2, NZ(1)),
        defender_card='We Come to Kill')
    add('resume_greathost_atk',
        attacker=_sh(6), defender=_fp(2, 0, L(1)),
        attacker_card='Great Host')
    add('resume_greathost_def',
        attacker=ArmySpec(regulars=2, nation=GON, leaders=L(1)),
        defender=_sh(6),
        defender_card='Great Host')
    add('resume_onslaught_auto',
        attacker=_sh(4, 2, NZ(1)), defender=_fp(4, 0, L(1)),
        attacker_card='Onslaught')
    add('resume_onslaught_defer', defer_onslaught=True,
        attacker=_sh(4, 2, NZ(1)), defender=_fp(4, 0, L(1)),
        attacker_card='Onslaught')
    add('resume_shieldwall_def',
        attacker=_sh(4, 0, NZ(2)), defender=_fp(3, 0, L(1)),
        defender_card='Shield-Wall')
    add('resume_policy_units',
        attacker=_sh(2, 2, NZ(1)), defender=_fp(2, 2, L(1)),
        attacker_card='We Come to Kill',
        choices=(('def_casualty_policy', 'preserve_units'),))
    return cs


def composed_defer_run(ctx, defer_onslaught):
    """One composed run of the SHIPPED deferred-round path: deferred
    round -> Phase-8 policy allocation -> resume_deferred_post_casualty.
    Returns ((atk_r, atk_e, def_r, def_e, atk_dead, def_dead), role)."""
    dctx = dataclasses.replace(ctx, defer_casualties=True)
    c = build_round(dctx)
    ch = round_choices(dctx)
    cb.resolve_combat_round(c, ch)
    role = None
    if getattr(c, 'casualties_deferred', False):
        chd = ctx.choices_dict()
        c.defender_casualties.extend(cb.apply_casualties(
            c.defender_army, int(c.deferred_hits_on_defender or 0),
            chd.get('def_casualty_policy', 'preserve_elites')))
        c.attacker_casualties.extend(cb.apply_casualties(
            c.attacker_army, int(c.deferred_hits_on_attacker or 0),
            chd.get('atk_casualty_policy', 'preserve_elites')))
        atk_dead = not c.attacker_army.units
        def_dead = not c.defender_army.units
        if not (atk_dead and def_dead):
            st = cd.scratch_state()
            # the owners' allocation policies reach Phase 9 (1.254:
            # post-casualty enemy hits are the enemy owner's allocation)
            _ch9 = {k: v for k, v in chd.items()
                    if k in ('atk_casualty_policy', 'def_casualty_policy')}
            if defer_onslaught:
                _ch9['defer_onslaught'] = True
            _a, _d, role = cb.resume_deferred_post_casualty(
                st, c.attacker_region, c.defender_region,
                c.attacker_side, c.defender_side,
                cd.combat_card_id(ctx.attacker_card),
                cd.combat_card_id(ctx.defender_card),
                c.attacker_card_cancelled, c.defender_card_cancelled,
                is_siege=ctx.is_siege, choices=_ch9)
    ar, ae = cd._re_counts(c.attacker_army)
    dr, de = cd._re_counts(c.defender_army)
    return ((ar, ae, dr, de, ar + ae == 0, dr + de == 0), role)


def _kernel_full_marginal(ctx, defer_onslaught):
    """CERTIFIED full-round kernel marginal onto ((armies, dead), role):
    the composed deferred path must reproduce it exactly."""
    ons_owner = None
    for nm, who in ((ctx.attacker_card, 'attacker'),
                    (ctx.defender_card, 'defender')):
        if nm and 'onslaught' in nm.lower():
            ons_owner = who
    fch = ctx.choices
    if defer_onslaught:
        fch = fch + (('defer_onslaught', True),)
    fctx = dataclasses.replace(ctx, defer_casualties=False, choices=fch)
    kd = exact_distribution(fctx)
    marg = {}
    for obs, pr in kd.outcomes.items():
        m = (obs[8], obs[9], obs[10], obs[11],
             obs[8] + obs[9] == 0, obs[10] + obs[11] == 0)
        role = None
        if defer_onslaught and ons_owner is not None:
            own_alive = (m[0] + m[1] > 0 if ons_owner == 'attacker'
                         else m[2] + m[3] > 0)
            if own_alive:
                role = ons_owner
        key = (m, role)
        marg[key] = marg.get(key, 0.0) + pr
    return marg


# ==========================================================================
# WITNESS leg: forced applies must reproduce each defer outcome
# ==========================================================================

def witness_check(ctx):
    kd = exact_distribution(ctx, want_witness=True)
    conf = cd.plan_conf_case(ctx)
    bad = 0
    for obs, plan in sorted(kd.witness.items(), key=str):
        def _thunk():
            c = build_round(ctx)
            la0 = list(c.attacker_army.leaders)
            ld0 = list(c.defender_army.leaders)
            cb.resolve_combat_round(c, round_choices(ctx))
            return observe(c, la0, ld0)
        try:
            got = cd.run_with_plan(plan, conf, _thunk)
        except cd.PlanMismatch as e:
            print('    witness %r: PlanMismatch %s' % (obs[:8], e))
            bad += 1
            continue
        if got != obs:
            print('    witness %r != forced %r' % (obs[:8], got[:8]))
            bad += 1
    return bad, len(kd.witness)


# ==========================================================================
# POSTERIOR leg: closed form vs brute-force hand enumeration
# ==========================================================================

def _posterior_bruteforce(pools, hand_counts, cands, none_value):
    """pools: deck -> list of card ids (with duplicates)."""
    decks = sorted(pools)
    subsets = [list(itertools.combinations(range(len(pools[d])),
                                           hand_counts.get(d, 0)))
               for d in decks]
    total = 0
    play = {k: 0 for k in cands}
    none = 0
    order = sorted(cands.items(),
                   key=lambda kv: (kv[1][2], str(kv[0])), reverse=True)
    for combo in itertools.product(*subsets):
        total += 1
        held = set()
        for d, idxs in zip(decks, combo):
            for i in idxs:
                held.add(pools[d][i])
        pick = None
        for key, (_d, _c, val) in order:
            if key in held and val > none_value:
                pick = key
                break
        if pick is None:
            none += 1
        else:
            play[pick] += 1
    return ({k: v / total for k, v in play.items()}, none / total)


def posterior_check():
    """Hand-computable cases: closed form == brute force (<=1e-12)."""
    cases = []
    # 1: one deck, distinct values
    cases.append(({'char': ['A', 'B', 'X', 'Y']}, {'char': 2},
                  {'A': ('char', 1, 3.0), 'B': ('char', 1, 2.0)}, 0.0))
    # 2: duplicate copies of one candidate
    cases.append(({'char': ['A', 'A', 'B', 'X', 'Y']}, {'char': 2},
                  {'A': ('char', 2, 3.0), 'B': ('char', 1, 2.0)}, 0.0))
    # 3: cross-deck with a value tie (broken by key)
    cases.append(({'char': ['A', 'X', 'Y'], 'strat': ['B', 'Z']},
                  {'char': 2, 'strat': 1},
                  {'A': ('char', 1, 2.0), 'B': ('strat', 1, 2.0)}, 0.0))
    # 4: candidate not better than declining
    cases.append(({'char': ['A', 'B', 'X']}, {'char': 2},
                  {'A': ('char', 1, 3.0), 'B': ('char', 1, -1.0)}, 0.0))
    # 5: certain holding (table pseudo-deck U=1, n=1)
    cases.append(({'char': ['A', 'X', 'Y'], 'table': ['T']},
                  {'char': 1, 'table': 1},
                  {'A': ('char', 1, 5.0), 'T': ('table', 1, 3.0)}, 0.0))
    n_bad = 0
    for i, (pools, hc, cands, v_none) in enumerate(cases):
        sizes = {d: len(v) for d, v in pools.items()}
        got_p, got_none = cd.card_play_posterior(sizes, hc, cands,
                                                 v_none)
        exp_p, exp_none = _posterior_bruteforce(pools, hc, cands,
                                                v_none)
        ok = abs(got_none - exp_none) < 1e-12 and all(
            abs(got_p.get(k, 0.0) - exp_p[k]) < 1e-12 for k in exp_p)
        print('  posterior case %d: %s  (closed %s none=%.6f | brute '
              '%s none=%.6f)'
              % (i + 1, 'PASS' if ok else 'FAIL',
                 {k: round(v, 6) for k, v in sorted(got_p.items())},
                 got_none,
                 {k: round(v, 6) for k, v in sorted(exp_p.items())},
                 exp_none))
        if not ok:
            n_bad += 1
    return n_bad, len(cases)


# ==========================================================================
# 1.233 stranded-leader ruling leg
# ==========================================================================

def stranded_check():
    """1.233: a stranded enemy leader grants NO Leadership and NO
    captain die; own-side leaders unchanged; leader-only entries keep
    legacy unfiltered behavior."""
    import game_state as gsx
    import game_logic as glx
    st = cd.scratch_state()
    st.armies.clear()
    a = glx.get_or_create_army(st, 'gorgoroth')
    a.units = [gsx.Unit(SAU, gsx.UnitKind.REGULAR) for _ in range(3)]
    checks = []
    a.leaders = [gsx.Leader(nation=GON)]
    checks.append(('stranded plain FP leader: L=0, dice=3',
                   cb.calc_leadership(a) == 0
                   and cb.calc_combat_strength(a) == 3))
    a.leaders = [gsx.Companion(name='Boromir', level=2, leadership=1)]
    checks.append(('stranded FP captain: L=0, no captain die',
                   cb.calc_leadership(a) == 0
                   and cb.calc_combat_strength(a) == 3))
    a.leaders = [gsx.Nazgul(nation=SAU)]
    checks.append(('own nazgul still counts: L=1',
                   cb.calc_leadership(a) == 1))
    b = glx.get_or_create_army(st, 'dagorlad')
    b.units = []
    b.leaders = [gsx.Leader(nation=GON)]
    checks.append(('leader-only entry: legacy unfiltered L=1',
                   cb.calc_leadership(b) == 1))
    st.armies.clear()
    bad = 0
    for label, ok in checks:
        print('  %-46s %s' % (label, 'PASS' if ok else 'FAIL'))
        if not ok:
            bad += 1
    return bad, len(checks)


# ==========================================================================
# A4 conservation regression leg (batch-0 mint): resume paths must
# SOURCE downgrade replacements (box first / pool / refuse)
# ==========================================================================

def conservation_check():
    import game_state as gsx
    import game_logic as glx
    EL = gsx.Nation.ELVES
    st = cd.scratch_state()
    checks = []

    def _reset(box_regs, parked):
        st.armies.clear()
        st.fp_casualties = {}
        st.pending_combat = None
        if parked > 0:
            far = glx.get_or_create_army(st, 'grey havens')
            far.units = [gsx.Unit(EL, gsx.UnitKind.REGULAR)
                         for _ in range(parked)]
        if box_regs:
            st.fp_casualties[EL] = {'regular': box_regs, 'elite': 0,
                                    'leader': 0}
        atk = glx.get_or_create_army(st, 'gorgoroth')
        atk.units = [gsx.Unit(gsx.Nation.SAURON, gsx.UnitKind.REGULAR)
                     for _ in range(6)]
        dfn = glx.get_or_create_army(st, 'dagorlad')
        dfn.units = [gsx.Unit(EL, gsx.UnitKind.ELITE)
                     for _ in range(2)]

    def _totals():
        dep = sum(1 for a in st.armies.values() for u in a.units
                  if u.nation == EL
                  and u.kind == gsx.UnitKind.REGULAR)
        box = (st.fp_casualties.get(EL) or {}).get('regular', 0)
        return dep, box

    gh = cd.combat_card_id('Great Host')

    # 1: Great Host resume, pool+box EMPTY -> downgrade REFUSED
    _reset(box_regs=0, parked=5)
    d0, b0 = _totals()
    cb.resume_deferred_post_casualty(
        st, 'gorgoroth', 'dagorlad', gsx.Side.SHADOW,
        gsx.Side.FREE_PEOPLES, gh, None, False, False,
        is_siege=False, choices={'defer_onslaught': True})
    d1, b1 = _totals()
    checks.append(('GH resume, no source: elite dies UNREPLACED',
                   (d1, b1) == (d0, b0)
                   and (st.fp_casualties.get(EL) or {})
                   .get('elite', 0) == 1))

    # 2: Great Host resume, BOX has 1 regular -> box-pull downgrade
    _reset(box_regs=1, parked=4)
    d0, b0 = _totals()          # dep 4 (parked at grey havens), box 1
    cb.resume_deferred_post_casualty(
        st, 'gorgoroth', 'dagorlad', gsx.Side.SHADOW,
        gsx.Side.FREE_PEOPLES, gh, None, False, False,
        is_siege=False, choices={'defer_onslaught': True})
    d1, b1 = _totals()
    checks.append(('GH resume, box-pull: dep+1/box-1, conserved',
                   d1 == d0 + 1 and b1 == b0 - 1))

    # 3: Onslaught resume, pool+box EMPTY -> downgrade REFUSED
    _reset(box_regs=0, parked=5)
    d0, b0 = _totals()
    _orig_roll = cb.roll_d6_batch
    cb.roll_d6_batch = lambda n, ctx='', side=None: [6] * n     # G1 side=
    try:
        cb.apply_onslaught_choice(
            st, 'gorgoroth', 'dagorlad', gsx.Side.SHADOW,
            gsx.Side.FREE_PEOPLES, 1)
    finally:
        cb.roll_d6_batch = _orig_roll
    d1, b1 = _totals()
    checks.append(('Onslaught resume, no source: no mint',
                   (d1, b1) == (d0, b0)))

    # 4: Great Host resume, POOL free -> pool downgrade legal, <= max
    _reset(box_regs=0, parked=3)
    cb.resume_deferred_post_casualty(
        st, 'gorgoroth', 'dagorlad', gsx.Side.SHADOW,
        gsx.Side.FREE_PEOPLES, gh, None, False, False,
        is_siege=False, choices={'defer_onslaught': True})
    d1, b1 = _totals()
    checks.append(('GH resume, pool source: downgrade legal, <= max',
                   d1 == 4 and b1 == 0 and d1 + b1 <= 5))
    st.armies.clear()
    st.fp_casualties = {}
    bad = 0
    for label, ok in checks:
        print('  %-46s %s' % (label, 'PASS' if ok else 'FAIL'))
        if not ok:
            bad += 1
    return bad, len(checks)


# ==========================================================================
# Main
# ==========================================================================

def run(mc_n=100_000, only=None, quick=False, out=OUT_JSON):
    cases = build_cases()
    resume_cases = build_resume_cases()
    prefixes = ([p.strip() for p in only.split(',') if p.strip()]
                if only else None)

    def _want(name):
        return prefixes is None or any(name.startswith(p)
                                       for p in prefixes)
    cases = [c for c in cases if _want(c[0])]
    resume_cases = [c for c in resume_cases if _want(c[0])]
    if prefixes is not None and not (cases or resume_cases):
        raise SystemExit('--only %r matched no cases' % only)
    if quick and mc_n == 100_000:
        mc_n = 5000     # --quick default; explicit --mc wins
    results = []
    print('cases: %d (+%d resume)   mc per case: %d'
          % (len(cases), len(resume_cases), mc_n))
    print('%-38s %8s %10s %10s %10s %7s %6s %s' %
          ('case', 'support', 'TV_exact', 'TV_mc', 'bound3s',
           'kern_ms', 'oracle', 'verdict'))
    for i, (name, ctx) in enumerate(cases):
        t0 = time.perf_counter()
        kd = exact_distribution(ctx)
        t1 = time.perf_counter()
        kernel = kd.outcomes
        ksum = sum(kernel.values())
        od, nodes = oracle_distribution(
            lambda: _run_engine_round(ctx), conf=cd.plan_conf_case(ctx))
        t2 = time.perf_counter()
        tv_exact = tv_distance(kernel, od)
        # coarsened-space leg: kernel and oracle projected identically
        tv_proj = tv_distance(project_state(kernel), project_state(od))
        counts = mc_distribution(lambda: _run_engine_round(ctx),
                                 mc_n, seed=1000 + i)
        t3 = time.perf_counter()
        tvmc = mc_tv(kernel, counts, mc_n)
        tv_mean, tv_bound = bootstrap_tv_bound(kernel, mc_n)
        ok = tv_exact < 1e-9 and tv_proj < 1e-9 and tvmc <= tv_bound \
            and abs(ksum - 1.0) < 1e-9
        verdict = 'PASS' if ok else 'FAIL'
        results.append({
            'case': name, 'leg': 'round', 'support': len(kernel),
            'support_proj': len(project_state(kernel)),
            'kernel_mass': ksum,
            'tv_exact': tv_exact, 'tv_proj': tv_proj, 'tv_mc': tvmc,
            'tv_bound_3sigma': tv_bound, 'tv_mean_h0': tv_mean,
            'mc_n': mc_n, 'oracle_nodes': nodes,
            'kernel_ms': (t1 - t0) * 1e3,
            'oracle_ms': (t2 - t1) * 1e3,
            'mc_ms': (t3 - t2) * 1e3,
            'mean_hits': kd.mean_hits(), 'var_hits': kd.var_hits(),
            'verdict': verdict,
        })
        print('%-38s %8d %10.2e %10.4f %10.4f %7.1f %6d %s' %
              (name, len(kernel), tv_exact, tvmc, tv_bound,
               (t1 - t0) * 1e3, nodes, verdict))
        if verdict == 'FAIL' and tv_exact >= 1e-9:
            _dump_diff(name, kernel, od)

    # ---- WITNESS leg over every defer-mode case ----
    print('')
    print('WITNESS leg (forced applies reproduce each defer outcome):')
    for name, ctx in cases:
        if not ctx.defer_casualties:
            continue
        bad, tot = witness_check(ctx)
        verdict = 'PASS' if bad == 0 else 'FAIL'
        results.append({'case': name, 'leg': 'witness',
                        'outcomes': tot, 'bad': bad,
                        'verdict': verdict})
        print('  %-36s %3d outcomes  %s' % (name, tot, verdict))

    # ---- RESUME leg (composed deferred path vs certified full mode) --
    if resume_cases:
        print('')
        print('RESUME leg (deferred round + allocation + Phase-9 '
              'resume == certified full-round kernel):')
        print('%-38s %8s %10s %10s %10s %6s %s' %
              ('case', 'support', 'TV_exact', 'TV_mc', 'bound3s',
               'oracle', 'verdict'))
    for i, (name, ctx, d_ons) in enumerate(resume_cases):
        expected = _kernel_full_marginal(ctx, d_ons)
        od, nodes = oracle_distribution(
            lambda: composed_defer_run(ctx, d_ons),
            conf=cd.plan_conf_case(ctx))
        # oracle observables are ((marg), role) already
        tv_exact = tv_distance(expected, od)
        counts = mc_distribution(
            lambda: composed_defer_run(ctx, d_ons), mc_n,
            seed=7000 + i)
        tvmc = mc_tv(expected, counts, mc_n)
        tv_mean, tv_bound = bootstrap_tv_bound(expected, mc_n)
        ok = tv_exact < 1e-9 and tvmc <= tv_bound
        verdict = 'PASS' if ok else 'FAIL'
        results.append({
            'case': name, 'leg': 'resume',
            'support': len(expected), 'tv_exact': tv_exact,
            'tv_mc': tvmc, 'tv_bound_3sigma': tv_bound,
            'tv_mean_h0': tv_mean,
            'mc_n': mc_n, 'oracle_nodes': nodes, 'verdict': verdict})
        print('%-38s %8d %10.2e %10.4f %10.4f %6d %s' %
              (name, len(expected), tv_exact, tvmc, tv_bound, nodes,
               verdict))
        if verdict == 'FAIL' and tv_exact >= 1e-9:
            _dump_diff(name, expected, od)

    # ---- POSTERIOR leg ----
    print('')
    print('POSTERIOR leg (card_play_posterior vs brute force):')
    p_bad, p_tot = posterior_check()
    results.append({'case': 'card_play_posterior', 'leg': 'posterior',
                    'cases': p_tot, 'bad': p_bad,
                    'verdict': 'PASS' if p_bad == 0 else 'FAIL'})

    # ---- 1.233 STRANDED-LEADER ruling leg ----
    print('')
    print('STRANDED leg (1.233: enemy leaders grant nothing):')
    s_bad, s_tot = stranded_check()
    results.append({'case': 'stranded_leader_1233', 'leg': 'stranded',
                    'cases': s_tot, 'bad': s_bad,
                    'verdict': 'PASS' if s_bad == 0 else 'FAIL'})

    # ---- A4 conservation leg (resume downgrade sourcing) ----
    print('')
    print('CONSERVATION leg (resume paths source replacements):')
    c_bad, c_tot = conservation_check()
    results.append({'case': 'resume_downgrade_sourcing',
                    'leg': 'conservation', 'cases': c_tot,
                    'bad': c_bad,
                    'verdict': 'PASS' if c_bad == 0 else 'FAIL'})

    n_fail = sum(1 for r in results if r['verdict'] == 'FAIL')
    print('')
    print('=== %d/%d legs PASS ===' % (len(results) - n_fail,
                                       len(results)))
    try:
        with open(out, 'w') as f:
            json.dump(results, f, indent=1)
        print('results: %s' % out)
    except OSError as e:
        print('results write FAILED: %r' % e)
        results.append({'case': '_results_write', 'verdict': 'FAIL',
                        'error': repr(e)})
    return results


def _dump_diff(name, kernel, oracle, top=12):
    keys = set(kernel) | set(oracle)
    diffs = sorted(keys,
                   key=lambda k: -abs(kernel.get(k, 0.0)
                                      - oracle.get(k, 0.0)))
    print('  -- %s: top outcome diffs (kernel vs oracle):' % name)
    for k in diffs[:top]:
        dk = kernel.get(k, 0.0)
        do = oracle.get(k, 0.0)
        if abs(dk - do) < 1e-12:
            break
        print('     %r  k=%.6f o=%.6f' % (k, dk, do))


if __name__ == '__main__':
    args = sys.argv[1:]
    kw = {}
    used = set()
    if '--quick' in args:
        kw['quick'] = True
        used.add(args.index('--quick'))
    if '--mc' in args:
        j = args.index('--mc')
        kw['mc_n'] = int(args[j + 1])
        used |= {j, j + 1}
    if '--only' in args:
        j = args.index('--only')
        kw['only'] = args[j + 1]
        used |= {j, j + 1}
    left = [a for j, a in enumerate(args) if j not in used]
    if left:
        sys.exit('unknown args: %r' % left)
    res = run(**kw)
    sys.exit(1 if any(r['verdict'] == 'FAIL' for r in res) else 0)
