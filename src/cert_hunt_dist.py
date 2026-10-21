# cert_hunt_dist.py - certification of the hunt_dist kernel.
"""Two-leg certification per case: (1) engine-driven exhaustive oracle at
raw d6-face granularity (kernel vs oracle must match to float precision),
(2) Monte Carlo through the untouched engine path (TV vs bootstrap 3-sigma
bound).  Run: python cert_hunt_dist.py [--quick] [--mc N] [--only PREFIX]
     [--after NAME] [--select K:N] [--out FILE]."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from itertools import combinations_with_replacement

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hunt_dist import (HuntContext, CS,                   # noqa: E402
                       build_hunt_state, snapshot, drive_hunt_event,
                       observe, enumerate_walks, exact_hunt_distribution,
                       build_roll_table)

_OUT_DIR = (os.environ.get('CERT_HUNT_OUT_DIR')
            or os.path.join(os.path.dirname(_HERE), 'data',
                            'cert_hunt'))


# ==========================================================================
# LEG 1: raw-granularity engine-driven oracle
# ==========================================================================

class RawSource:
    """Face multisets with pure multinomial mass (no hit-rule knowledge);
    per-index uniform pool and companion picks."""

    def roll_options(self, n, ctxstr):
        opts = []
        for combo in combinations_with_replacement(range(1, 7), n):
            counts = {}
            for f in combo:
                counts[f] = counts.get(f, 0) + 1
            coef = math.factorial(n)
            for c in counts.values():
                coef //= math.factorial(c)
            opts.append((list(combo), coef / (6.0 ** n)))
        return opts

    def pool_options(self, pool):
        n = float(len(pool))
        return [(i, 1.0 / n) for i in range(len(pool))]

    def comp_options(self, cands):
        n = float(len(cands))
        return [(i, 1.0 / n) for i in range(len(cands))]


def oracle_distribution(ctx, max_nodes=2_000_000):
    return enumerate_walks(ctx, RawSource(), max_nodes=max_nodes)


# ==========================================================================
# LEG 2: Monte Carlo through the untouched engine path
# ==========================================================================

def mc_distribution(ctx, n_trials, seed):
    random.seed(seed)
    counts = {}
    for _ in range(n_trials):
        st = build_hunt_state(ctx)
        pre = snapshot(st)
        drive_hunt_event(st, ctx)
        obs = observe(st, pre)
        counts[obs] = counts.get(obs, 0) + 1
    return counts


# ==========================================================================
# Comparison metrics
# ==========================================================================

def tv_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def mc_tv(kernel, counts, n):
    keys = set(kernel) | set(counts)
    return 0.5 * sum(abs(kernel.get(k, 0.0) - counts.get(k, 0) / n)
                     for k in keys)


def bootstrap_tv_bound(kernel, n, m=300, seed=1234):
    import numpy as np
    rng = np.random.default_rng(seed)
    probs = np.array(list(kernel.values()), dtype=float)
    probs = probs / probs.sum()
    draws = rng.multinomial(n, probs, size=m)
    emp = draws / float(n)
    tvs = 0.5 * np.abs(emp - probs[None, :]).sum(axis=1)
    return float(tvs.mean()), float(tvs.mean() + 3.0 * tvs.std())


# ==========================================================================
# Exhaustive case matrix: every reachable branch-class gets a case; the
# unreachable/collapsed classes carry engine-verified reasons (probes below)
# ==========================================================================

GRID_POOL = ('eye', 'eye', '3', '1r')

STD_KINDS = ('0r', '1', '1r', '2', '2r', '3', 'eye')
SPECIAL_KINDS = ('oonw', 'shelob', 'ringmine', 'giveuss',
                 'smeagol', 'ecloak', 'erope', 'phial')


def _src_kwargs(n):
    if n == 0:
        return {}
    if n == 1:
        return {'nazgul_at_fs': True}
    if n == 2:
        return {'nazgul_at_fs': True, 'army_at_fs': True}
    return {'nazgul_at_fs': True, 'army_at_fs': True, 'region': 'moria'}


def build_matrix():
    """(cases, unreachable, deferred): one cert case per reachable
    branch class; unreachable = (class, reason) engine-verified by the
    probes; deferred = learned-decision inputs outside kernel scope."""
    cs = []

    def add(name, **kw):
        cs.append((name, HuntContext(**kw)))

    # ---- DIM R: roll cells (eyes x C x sources x crebain) ----
    add('r_e0', eyes=0, pool=GRID_POOL, corruption=3)
    for e in range(1, 6):
        for c in range(4):
            for s in range(4):
                for b in range(2):
                    add('r_e%dc%ds%db%d' % (e, c, s, b), eyes=e,
                        fp_dice=c, crebain_bonus=b, pool=GRID_POOL,
                        corruption=3, **_src_kwargs(s))
    # source-composition symmetry + the C=4 modifier cap edge
    add('r_sym_army1', eyes=2, fp_dice=1, pool=GRID_POOL, corruption=3,
        army_at_fs=True)
    add('r_sym_moria1', eyes=2, fp_dice=1, pool=GRID_POOL, corruption=3,
        region='moria')
    add('r_sym_pair_army_moria', eyes=2, fp_dice=1, pool=GRID_POOL,
        corruption=3, army_at_fs=True, region='moria')
    add('r_e2_c4', eyes=2, fp_dice=4, pool=GRID_POOL, corruption=3)

    # ---- DIM T: tile kind x location (single-tile pools isolate) ----
    for k in STD_KINDS + SPECIAL_KINDS:
        add('t_out_%s' % k, eyes=2, fp_dice=1, pool=(k,), corruption=3)
        add('t_mor_%s' % k, eyes=2, fp_dice=1, pool=(k,), corruption=3,
            in_mordor=True, mordor_step=1)

    # ---- DIM I: interpose cards ----
    add('i_mithril_keep_out', eyes=2, fp_dice=1, pool=('3', '1r'),
        corruption=3, mithril=True)
    add('i_mithril_redraw_out', eyes=2, fp_dice=1, pool=('3', '1', '1r'),
        corruption=3, mithril=True, mithril_redraw=('3',))
    add('i_mithril_keep_mor', eyes=2, fp_dice=0, pool=('2', '1r'),
        corruption=3, in_mordor=True, mordor_step=1, mithril=True)
    add('i_mithril_redraw_mor', eyes=2, fp_dice=0, pool=('shelob', '1'),
        corruption=3, in_mordor=True, mordor_step=2, mithril=True,
        mithril_redraw=('shelob',))
    add('i_mithril_rebuild', eyes=2, fp_dice=1, pool=('3',),
        discard=('1',), corruption=3, mithril=True,
        mithril_redraw=('3',))
    add('i_mithril_gollum_resume', eyes=2, fp_dice=1, pool=('2',),
        companions=(), corruption=3, mithril=True, gollum_reveal=True)
    add('i_staff_decline_out', eyes=2, fp_dice=1, pool=('2',),
        corruption=3, wizard_staff=True)
    add('i_staff_prevent_out', eyes=2, fp_dice=1, pool=('2',),
        corruption=3, wizard_staff=True, wizard_prevent=True)
    # 1.232: Staff x Mordor-track combinations are REACHABLE
    add('i_staff_mor_prevent', eyes=2, fp_dice=0, pool=('2',),
        corruption=3, in_mordor=True, mordor_step=1, wizard_staff=True,
        wizard_prevent=True)
    add('i_staff_mor_decline', eyes=2, fp_dice=0, pool=('2', '1r'),
        corruption=3, in_mordor=True, mordor_step=1, wizard_staff=True)
    add('i_staff_mor_stop_prevent', eyes=2, fp_dice=0, pool=('oonw',),
        corruption=3, in_mordor=True, mordor_step=2, wizard_staff=True,
        wizard_prevent=True)
    add('i_staff_mor_stop_decline', eyes=2, fp_dice=0, pool=('oonw',),
        corruption=3, in_mordor=True, mordor_step=2, wizard_staff=True)
    add('i_staff_mor_shelob_decline', eyes=2, fp_dice=0,
        pool=('shelob',), corruption=3, in_mordor=True, mordor_step=1,
        wizard_staff=True)
    add('i_staff_mor_ringmine_prevent', eyes=2, fp_dice=1,
        pool=('ringmine',), corruption=3, in_mordor=True, mordor_step=1,
        wizard_staff=True, wizard_prevent=True)
    add('i_staff_mor_ringmine_decline', eyes=2, fp_dice=1,
        pool=('ringmine',), corruption=3, in_mordor=True, mordor_step=1,
        wizard_staff=True)
    add('i_staff_mor_giveuss_decline', eyes=2, fp_dice=0,
        pool=('giveuss',), corruption=3, in_mordor=True, mordor_step=1,
        wizard_staff=True)
    add('i_staff_mithril_mor_decline', eyes=2, fp_dice=0,
        pool=('shelob', '1'), corruption=3, in_mordor=True,
        mordor_step=2, wizard_staff=True, mithril=True,
        mithril_redraw=('shelob',))
    add('i_staff_mithril_decline', eyes=2, fp_dice=1, pool=('3', '1'),
        corruption=3, wizard_staff=True, mithril=True,
        mithril_redraw=('3',))
    add('i_staff_mithril_prevent', eyes=2, fp_dice=1, pool=('3', '1'),
        corruption=3, wizard_staff=True, wizard_prevent=True,
        mithril=True, mithril_redraw=('3',))
    add('i_axe_use_out', eyes=2, fp_dice=1, pool=('2',), corruption=3,
        axe_horn=True, use_reduction=True)
    add('i_axe_decline_out', eyes=2, fp_dice=1, pool=('2',),
        corruption=3, axe_horn=True)
    add('i_axe_use_mordor', eyes=2, fp_dice=0, pool=('2',), corruption=3,
        in_mordor=True, mordor_step=1, axe_horn=True,
        use_reduction=True)

    # ---- DIM G: guide/roster x damage-choice classes ----
    small = (CS('Legolas', 2), CS('Meriadoc', 1))
    add('g_corr_default', eyes=3, fp_dice=1, pool=('3',), corruption=3)
    add('g_guide_full_absorb', eyes=3, fp_dice=1, pool=('2',),
        companions=small, guide='Legolas', damage_choice='guide',
        corruption=3)
    add('g_guide_partial_chain_corr', eyes=3, fp_dice=1, pool=('3',),
        companions=small, guide='Legolas', damage_choice='guide',
        chain_choice='corruption', corruption=3)
    add('g_guide_partial_chain_sep', eyes=3, fp_dice=1, pool=('3',),
        companions=small, guide='Legolas', damage_choice='guide',
        chain_choice='separate', corruption=3)
    add('g_random_mixed', eyes=3, fp_dice=1, pool=('3',),
        companions=(CS('Boromir', 2), CS('Legolas', 2),
                    CS('Meriadoc', 1)),
        guide='Boromir', damage_choice='random', corruption=3)
    add('g_random_hobbits_only', eyes=2, fp_dice=1, pool=('2',),
        companions=(CS('Meriadoc', 1), CS('Peregrin', 1)),
        guide='Meriadoc', damage_choice='random', corruption=3)
    add('g_random_single_guide', eyes=2, fp_dice=1, pool=('2',),
        companions=(CS('Legolas', 2),), damage_choice='random',
        corruption=3)
    add('g_random_hobbit_mordor', eyes=2, fp_dice=0, pool=('2',),
        companions=(CS('Meriadoc', 1), CS('Peregrin', 1)),
        guide='Meriadoc', damage_choice='random', corruption=3,
        in_mordor=True, mordor_step=1)
    add('g_separate_chain_sep', eyes=2, fp_dice=1, pool=('2',),
        companions=(CS('Meriadoc', 1), CS('Peregrin', 1)),
        guide='Meriadoc', damage_choice='separate',
        chain_choice='separate', corruption=3)
    add('g_separate_chain_corr', eyes=2, fp_dice=1, pool=('3',),
        companions=(CS('Meriadoc', 1), CS('Peregrin', 1)),
        guide='Meriadoc', damage_choice='separate',
        chain_choice='corruption', corruption=3)
    add('g_gollum_reveal', eyes=2, fp_dice=1, pool=('2', '1r', 'eye'),
        companions=(), corruption=3, gollum_reveal=True)
    add('g_gollum_no_reveal', eyes=2, fp_dice=1, pool=('2', '1r', 'eye'),
        companions=(), corruption=3)
    add('g_gollum_fp_settlement', eyes=2, fp_dice=1, pool=('2',),
        companions=(), corruption=3, region='minas tirith',
        gollum_reveal=True)
    add('g_gollum_revealed_already', eyes=2, fp_dice=1, pool=('2',),
        companions=(), corruption=3, hidden=False, gollum_reveal=True)
    add('g_gollum_mordor', eyes=2, fp_dice=0, pool=('1r', 'eye'),
        companions=(), corruption=3, in_mordor=True, mordor_step=1)
    add('g_chain_to_gollum_rev', eyes=3, fp_dice=1, pool=('3',),
        companions=(CS('Legolas', 2),), damage_choice='guide',
        gollum_reveal=True, corruption=3)
    add('g_chain_to_gollum_norev', eyes=3, fp_dice=1, pool=('3',),
        companions=(CS('Legolas', 2),), damage_choice='guide',
        corruption=3)

    # ---- DIM E: clamp / victory / recycle / reveal-placement edges ----
    add('e_heal_smeagol_c0', eyes=2, fp_dice=0, pool=('smeagol',),
        corruption=0, in_mordor=True, mordor_step=1)
    add('e_heal_phial_c1', eyes=2, fp_dice=0, pool=('phial',),
        corruption=1, in_mordor=True, mordor_step=1)
    add('e_clamp10_mor', eyes=2, fp_dice=0, pool=('3',), corruption=10,
        in_mordor=True, mordor_step=1)
    add('e_clamp11_out', eyes=2, fp_dice=1, pool=('3',), corruption=11)
    add('e_clamp_exact12', eyes=2, fp_dice=1, pool=('3',), corruption=9)
    add('e_step5_win', eyes=1, fp_dice=0, pool=('1',), corruption=0,
        in_mordor=True, mordor_step=4)
    add('e_step5_stop_denied', eyes=1, fp_dice=0, pool=('oonw',),
        corruption=0, in_mordor=True, mordor_step=4)
    add('e_step5_mithril_gate', eyes=1, fp_dice=0, pool=('1',),
        corruption=0, in_mordor=True, mordor_step=4, mithril=True)
    add('e_step5_staff_prevent_win', eyes=1, fp_dice=0, pool=('3',),
        corruption=0, in_mordor=True, mordor_step=4, wizard_staff=True,
        wizard_prevent=True)
    add('e_step5_staff_stop_decline', eyes=1, fp_dice=0, pool=('oonw',),
        corruption=0, in_mordor=True, mordor_step=4, wizard_staff=True)
    add('e_recycle_out', eyes=2, fp_dice=1, pool=(), discard=('2', 'eye'),
        corruption=3)
    add('e_recycle_mor', eyes=2, fp_dice=0, pool=(), discard=('2', '1'),
        corruption=3, in_mordor=True, mordor_step=1)
    add('e_recycle_stronghold', eyes=2, fp_dice=1, pool=('1r',),
        discard=('2',), corruption=3, region='moria', progress=1)
    add('e_recycle_special_excluded', eyes=2, fp_dice=1, pool=(),
        discard=('giveuss', '1'), corruption=3)
    add('e_fizzle_out', eyes=2, fp_dice=1, pool=(), discard=(),
        corruption=3)
    add('e_fizzle_mor', eyes=2, fp_dice=0, pool=(), discard=(),
        corruption=3, in_mordor=True, mordor_step=2)
    add('e_sh_eye_noop', eyes=2, fp_dice=1, pool=('1r', 'eye'),
        corruption=3, region='moria', progress=1)
    add('e_reveal_prog0', eyes=2, fp_dice=1, pool=('1r',), corruption=3)
    add('e_reveal_prog2_gateway', eyes=2, fp_dice=1, pool=('1r',),
        corruption=3, progress=2)
    add('e_reveal_in_moria', eyes=2, fp_dice=1, pool=('1r',),
        corruption=3, region='moria', progress=1)
    add('e_gollum_reveal_moria', eyes=2, fp_dice=1, pool=('2', '1'),
        companions=(), corruption=3, region='moria', progress=1,
        gollum_reveal=True)
    add('e_revealed_already_1r', eyes=2, fp_dice=1, pool=('1r',),
        corruption=3, hidden=False, progress=1)

    unreachable = [
        ('r_e0 x C/sources/crebain (31 combos)',
         'eyes=0 -> num_dice=0, _hunt_roll never runs; modifier and '
         'reroll dimensions are dead (probe: exact dist equality)'),
        ('eyes>5 roll cells',
         'num_dice=min(5,hunt_level) caps the roll; outside-Mordor '
         'classes collapse onto eyes=5 (probe: exact dist equality); '
         'Mordor eye-box damage is its own certified dim (t_mor_eye)'),
        ('C>=5 modifier cells',
         'threshold 6, natural 1 never hits: modifier>=4 saturates '
         'p_hit at 5/6 (probe: exact dist equality C=4 vs C=5)'),
        ('crebain_bonus>1', 'one Flocks of Crebain card exists; the '
         'engine consumes a scalar +1 once per hunt'),
        ('guide=Gollum with companions present',
         'change_guide invariant: guide = max-level companion whenever '
         'companions exist (probe: fixture construction fails)'),
        ('separate choice without hobbit guide',
         'menu-gated: hunt_decision_menu offers hunt_guide_separate '
         'only for sacrifice_reduce_hunt guides (probe: ValueError)'),
        ('guide/random choice after casualty_taken',
         'menu-gated: one casualty per hunt (probe: ValueError on '
         'chained guide choice)'),
        ('stop-tile movement effect outside Mordor',
         'stops_movement is read only on the track; outside path '
         'ignores it (certified t_out stop-tile cases: no effect)'),
        ('Shelob d6 outside Mordor',
         'is_dice_roll read only in _apply_mordor_hunt_tile; outside '
         'the tile deals its printed damage 0 (certified t_out_shelob)'),
        ('sources compositions beyond canonical',
         'the three re-roll sources are commutative +1 counts '
         '(certified: each singleton + one pairing + all-three)'),
        ('reveal-place "leaving"/"entering"/"passing through" draws',
         'same _draw_shadow_stronghold_tile body as "revealed in" '
         'after the 1.231 guard; reason-string differs only'),
        ('special tiles in the outside pool (production)',
         'stage_special_hunt_tile routes card tiles to the used pile '
         'pre-Mordor; they enter the pool on Mordor entry - the '
         't_out special cases certify the latent engine behavior '
         'anyway'),
    ]

    deferred = [
        'Balrog of Moria extra-draw trigger (persistent-card input)',
        'hobbit_place region pick (learned; kernel fixes current region)',
        'guide-tie choose_guide pick (learned; kernel canonical pick)',
        'Flocks of Crebain discard decision (input as crebain_bonus)',
        "card-driven hunts (Isildur's Bane / Foul Thing / Orc Patrol / "
        'Nazgul Strike) - separate entry points, same tile machinery',
        'Worn with Sorrow and Toil on sacrificed companions',
        'per-information-set adaptive policies (ctx fixes one vector)',
    ]
    return cs, unreachable, deferred


def probe_unreachable():
    """Engine-verified unreachability/collapse checks; returns failures."""
    fails = []

    def same(ca, cb, tag):
        da = exact_hunt_distribution(ca).outcomes
        db = exact_hunt_distribution(cb).outcomes
        if tv_distance(da, db) > 1e-12:
            fails.append(tag)

    base0 = HuntContext(eyes=0, pool=GRID_POOL, corruption=3)
    same(base0, HuntContext(eyes=0, fp_dice=2, crebain_bonus=1,
                            pool=GRID_POOL, corruption=3,
                            nazgul_at_fs=True), 'e0-collapse')
    same(HuntContext(eyes=5, fp_dice=1, pool=GRID_POOL, corruption=3),
         HuntContext(eyes=7, fp_dice=1, pool=GRID_POOL, corruption=3),
         'eyes-cap-collapse')
    same(HuntContext(eyes=2, fp_dice=4, pool=GRID_POOL, corruption=3),
         HuntContext(eyes=2, fp_dice=5, pool=GRID_POOL, corruption=3),
         'modifier-saturation')
    same(HuntContext(eyes=2, fp_dice=1, crebain_bonus=1,
                     pool=GRID_POOL, corruption=3),
         HuntContext(eyes=2, fp_dice=2, pool=GRID_POOL, corruption=3),
         'crebain-additivity')
    try:
        build_hunt_state(HuntContext(
            eyes=1, pool=('2',), corruption=3, guide='Gollum',
            companions=(CS('Legolas', 2),)))
        fails.append('gollum-guide-with-companions CONSTRUCTIBLE')
    except StopIteration:
        pass
    try:
        exact_hunt_distribution(HuntContext(
            eyes=5, fp_dice=3, pool=('2',), corruption=3,
            damage_choice='separate'))
        fails.append('separate-without-hobbit-guide LEGAL')
    except ValueError:
        pass
    try:
        # the chained menu OPENS only when the next guide is a hobbit
        # (plain-companion chains self-resolve in continue_hunt_chain);
        # the chained guide/random exclusion is what must hold there
        exact_hunt_distribution(HuntContext(
            eyes=5, fp_dice=3, pool=('3',), corruption=3,
            companions=(CS('Legolas', 2), CS('Meriadoc', 1)),
            guide='Legolas', damage_choice='guide',
            chain_choice='guide'))
        fails.append('chained-guide-choice LEGAL')
    except ValueError:
        pass
    return fails


def build_cases():
    """Public hook: reachable cert cases only (interactive / ad-hoc
    retest tooling); run() consumes build_matrix() directly."""
    cs, _, _ = build_matrix()
    return cs


# ==========================================================================
# Roll-table consistency (T1 vs the engine-certified kernel)
# ==========================================================================

def roll_table_check():
    """Max |T1 - kernel| over the FULL eye 0-5 x mod 0-4 x src 0-3
    grid on an all-eye pool (eye tile damage = successes, so the
    corruption delta reads the pmf); src levels mirror _src_kwargs."""
    t1 = build_roll_table()
    worst = 0.0
    for eyes in range(6):
        for mod in range(5):
            for src in range(4):
                kw9 = dict(_src_kwargs(src))
                kw9.setdefault('region', 'dagorlad')
                ctx = HuntContext(eyes=eyes, fp_dice=mod,
                                  pool=('eye',) * 4, corruption=0,
                                  **kw9)
                kd = exact_hunt_distribution(ctx)
                pmf = t1[(eyes, mod, src)]
                delta = kd.corruption_delta_pmf()
                for s in range(6):
                    got = delta.get(s, 0.0)
                    worst = max(worst, abs(got - pmf[s]))
    return worst


# ==========================================================================
# Main
# ==========================================================================

def _oracle_ctx(ctx):
    """Raw-oracle cost gate: over-budget cells run the exact-oracle leg
    on the same roll cell with a single non-reveal tile (identical
    roll/reroll engine path, no chain multiplier); the full-pool cell
    keeps its 100k-MC leg.  Measured growth: ~18x nodes per eye level
    on 3-source cells (e3=58.6k), so e5 x 3-source exceeds the node budget
    (e4 x 3-source fits at est ~1.31M)."""
    from dataclasses import replace
    n = min(5, ctx.eyes)
    src = ((1 if ctx.nazgul_at_fs else 0) + (1 if ctx.army_at_fs else 0)
           + (1 if ctx.region == 'moria' else 0))
    est = (math.comb(n + 5, 5) * 6 ** min(src, n)
           * (len(set(ctx.pool)) + 1) * 12)
    if est <= 1_500_000:
        return ctx, 'full'
    return (replace(ctx, pool=('3',), discard=(), mithril_redraw=()),
            'reduced-pool')


def _run_case(name, ctx, i, mc_n):
    t0 = time.perf_counter()
    kd = exact_hunt_distribution(ctx)
    t1 = time.perf_counter()
    kernel = kd.outcomes
    ksum = sum(kernel.values())
    octx, omode = _oracle_ctx(ctx)
    od, onodes = oracle_distribution(octx)
    t2 = time.perf_counter()
    tv_exact = tv_distance(exact_hunt_distribution(octx).outcomes, od)
    counts = mc_distribution(ctx, mc_n, seed=1000 + i)
    t3 = time.perf_counter()
    tvmc = mc_tv(kernel, counts, mc_n)
    tv_mean, tv_bound = bootstrap_tv_bound(kernel, mc_n)
    ok = tv_exact < 1e-9 and tvmc <= tv_bound \
        and abs(ksum - 1.0) < 1e-9
    verdict = 'PASS' if ok else 'FAIL'
    rec = {
        'case': name, 'support': len(kernel), 'kernel_mass': ksum,
        'tv_exact': tv_exact, 'tv_mc': tvmc,
        'tv_bound_3sigma': tv_bound, 'tv_mean_h0': tv_mean,
        'mc_n': mc_n, 'kernel_walks': kd.walks,
        'oracle_walks': onodes, 'oracle_mode': omode,
        'kernel_ms': (t1 - t0) * 1e3,
        'oracle_ms': (t2 - t1) * 1e3,
        'mc_ms': (t3 - t2) * 1e3,
        'mean_corruption_delta': kd.mean_corruption_delta(),
        'p_reveal': kd.p_reveal(),
        'verdict': verdict,
    }
    print('%-26s %8d %10.2e %10.4f %10.4f %8.1f %8d %s' %
          (name, len(kernel), tv_exact, tvmc, tv_bound,
           (t1 - t0) * 1e3, kd.walks, verdict))
    if verdict == 'FAIL' and tv_exact >= 1e-9:
        _dump_diff(name, exact_hunt_distribution(octx).outcomes, od)
    return rec


def run(mc_n=100_000, only=None, quick=False,
        out='cert_hunt_results.json', after=None, select=None):
    cases, unreachable, deferred = build_matrix()
    names = [n for n, _ in cases]
    assert len(names) == len(set(names)), 'duplicate case names'
    print('reachable branch classes certified: %d' % len(cases))
    print('unreachable/collapsed classes (engine-verified): %d'
          % len(unreachable))
    print('deferred learned-decision inputs: %d' % len(deferred))
    indexed = list(enumerate(cases))
    if only:
        indexed = [(i, c) for i, c in indexed if c[0].startswith(only)]
    if after:
        cut = names.index(after) + 1
        indexed = [(i, c) for i, c in indexed if i >= cut]
    if select:
        k, n = (int(x) for x in select.split(':'))
        indexed = indexed[k::n]
    if quick and mc_n == 100_000:
        mc_n = 5000     # --quick default; explicit --mc wins
    results = []
    print('cases: %d   mc per case: %d' % (len(indexed), mc_n))
    print('%-26s %8s %10s %10s %10s %8s %8s %s' %
          ('case', 'support', 'TV_exact', 'TV_mc', 'bound3s',
           'kern_ms', 'walks', 'verdict'))
    for i, (name, ctx) in indexed:
        # fault isolation: one case never kills the sweep
        try:
            results.append(_run_case(name, ctx, i, mc_n))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print('%-26s EXCEPTION -> FAIL' % name)
            print(tb)
            results.append({'case': name, 'verdict': 'FAIL',
                            'error': tb.splitlines()[-1],
                            'traceback': tb})
    n_fail = sum(1 for r in results if r['verdict'] == 'FAIL')
    print('')
    print('=== %d/%d cases PASS ===' % (len(results) - n_fail,
                                        len(results)))
    if not only and not after and not select:
        w = roll_table_check()
        print('roll-table T1 vs kernel max |diff| = %.3e' % w)
        results.append({'case': '_roll_table_T1',
                        'verdict': 'PASS' if w < 1e-9 else 'FAIL',
                        'max_diff': w})
        pf = probe_unreachable()
        print('unreachability probes: %s'
              % ('ALL VERIFIED' if not pf else 'FAIL %r' % pf))
        results.append({'case': '_probe_unreachable',
                        'verdict': 'FAIL' if pf else 'PASS',
                        'fails': pf})
        print('')
        print('--- unreachable/collapsed classes ---')
        for cl, reason in unreachable:
            print('  %s: %s' % (cl, reason))
        print('--- deferred inputs ---')
        for d in deferred:
            print('  %s' % d)
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(os.path.join(_OUT_DIR, out), 'w') as f:
        json.dump({'results': results,
                   'unreachable': unreachable,
                   'deferred': deferred,
                   'filters': {'only': only, 'after': after,
                               'select': select, 'quick': quick}},
                  f, indent=1)
    return results


def _dump_diff(name, kernel, oracle, top=12):
    keys = set(kernel) | set(oracle)
    diffs = sorted(keys, key=lambda k: -abs(kernel.get(k, 0.0)
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
    _used = set()
    if '--quick' in args:
        kw['quick'] = True
        _used.add(args.index('--quick'))
    for _flag, _key in (('--mc', 'mc_n'), ('--only', 'only'),
                        ('--after', 'after'), ('--select', 'select'),
                        ('--out', 'out')):
        if _flag in args:
            _j = args.index(_flag)
            if _j + 1 >= len(args):
                sys.exit('%s needs a value' % _flag)
            kw[_key] = (int(args[_j + 1]) if _key == 'mc_n'
                        else args[_j + 1])
            _used |= {_j, _j + 1}
    _left = [a for _j, a in enumerate(args) if _j not in _used]
    if _left:
        sys.exit('unknown args: %r' % _left)
    _res = run(**kw)
    # hard gate: nonzero exit on any FAIL (gate wiring reads the code)
    sys.exit(1 if any(r.get('verdict') == 'FAIL' for r in _res) else 0)
