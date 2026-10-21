#!/usr/bin/env python3
"""Audit 2026-08-29 regression pins - CB family (combat.py, combat_dist.py,
cert_combat_dist.py).

One plain test_* function per finding, states built the test_rules way
(board.load_board / setup_initial_game / mutate / call the engine) or the
combat_dist fixture builder (build_round).  main() runs them all and
prints TOTAL=<n> FAIL=<m>; exit 1 on any failure.

Findings pinned (ids from CB_report.md; rulings from rules_amendments.md):
  CB-1   run_with_plan / cert oracle shims accept side= (G1 keyword)
  CB-2   kernel Heroic Death scope excludes pre-combat hits (engine parity)
  CB-3   a 0-dice additional attack (Charge / Sudden Strike) rolls NOTHING
  CB-4   Foul Stench sees the shine-negated Nazgul Leadership (0)
  1.255  field-battle wipe captures nothing; entry captures
  CB-5   internal multi-round loop re-derives Leadership / Nazgul pool
  CB-6   Fateful Strike gate is card-identity exact
  CB-8   _capture_advance is loud (no blanket except)
  CB-7 / 1.254  post-casualty enemy hits + target picks route through the
         owner's decision hooks (policy threading, defer stash, targets)

cp1252 console: ASCII only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import board
import combat as cb
import combat_dist as cd
import game_logic
import game_state as gs
from board import sanitize
from combat_dist import ArmySpec, ExchangeContext, LeaderSpec
from game_state import Nation, Side, UnitKind

GON, SAU = Nation.GONDOR, Nation.SAURON


# ---------------------------------------------------------------------------
# helpers (mirrors scratchpad/cb_probe.py)
# ---------------------------------------------------------------------------

def L(n=1):
    return tuple(LeaderSpec('leader') for _ in range(n))


def NZ(n=1):
    return tuple(LeaderSpec('nazgul') for _ in range(n))


def C(name, level, ldr):
    return LeaderSpec('companion', name, level, ldr)


def sh(r, e=0, leaders=()):
    return ArmySpec(regulars=r, elites=e, nation=SAU, leaders=tuple(leaders))


def fp(r, e=0, leaders=()):
    return ArmySpec(regulars=r, elites=e, nation=GON, leaders=tuple(leaders))


class Rec:
    """scripted roller: pops scripted batches, else fills with `fill`."""

    def __init__(self, scripts=(), fill=1):
        self.scripts = list(scripts)
        self.fill = fill
        self.calls = []

    def __call__(self, n, ctx='', side=None):
        self.calls.append((n, ctx))
        if self.scripts:
            s = self.scripts.pop(0)
            return list(s)[:n] + [self.fill] * max(0, n - len(s))
        return [self.fill] * n


def with_roller(rec, fn):
    o = cb.roll_d6_batch
    cb.roll_d6_batch = rec
    try:
        return fn()
    finally:
        cb.roll_d6_batch = o


def fresh():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    st.armies.clear()
    st.fp_casualties = {}
    st.pending_combat = None
    st.enemy_controlled = {Side.SHADOW: set(), Side.FREE_PEOPLES: set()}
    return st


def place(st, region, units, leaders=None):
    a = game_logic.get_or_create_army(st, region)
    a.units = list(units)
    a.leaders = list(leaders or [])
    return a


def units(nation, n, kind=UnitKind.REGULAR):
    return [gs.Unit(nation, kind) for _ in range(n)]


def resolve_with(st, rec, *args, **kw):
    prev = cb.set_active_state(st)
    try:
        return with_roller(rec, lambda: cb.resolve_combat(st, *args, **kw))
    finally:
        cb.set_active_state(prev)


def _has_boromir(obs):
    return 'C:Boromir' in obs[15]


# ---------------------------------------------------------------------------
# CB-1: the forced-apply shims must accept the G1 side= keyword
# ---------------------------------------------------------------------------

def test_cb1_run_with_plan_passes_side():
    faces, hits = cd.run_with_plan(
        (1,), False,
        lambda: cb.combat_roll(2, 5, 0, None, side=Side.SHADOW))
    assert len(faces) == 2 and hits == 1, (faces, hits)
    # every witness plan of a defer-mode exchange drives the REAL round
    # apply (the ai.py kernel path) and reproduces its outcome
    ctxw = ExchangeContext(attacker=sh(3, 0, NZ(1)), defender=fp(2, 0, L(1)),
                           defer_casualties=True)
    dw = cd.exact_distribution(ctxw, want_witness=True)
    assert dw.witness, 'no witness plans'
    n = 0
    for obs, plan in dw.witness.items():
        cw = cd.build_round(ctxw)
        la0, ld0 = list(cw.attacker_army.leaders), list(cw.defender_army.leaders)
        cd.run_with_plan(plan, cd.plan_conf_case(ctxw),
                         lambda: cb.resolve_combat_round(cw, cd.round_choices(ctxw)))
        got = cd.observe(cw, la0, ld0)
        assert got == obs, (plan, got, obs)
        assert getattr(cw, 'casualties_deferred', False), 'round not resolved'
        n += 1
    assert n == len(dw.witness)


def test_cb1_cert_oracle_accepts_side():
    import cert_combat_dist as cert
    ctx = ExchangeContext(attacker=sh(2, 0, NZ(1)), defender=fp(2, 0, L(1)))
    kd = cd.exact_distribution(ctx)
    od, nodes = cert.oracle_distribution(
        lambda: cert._run_engine_round(ctx), conf=cd.plan_conf_case(ctx))
    assert nodes > 0
    tv = cert.tv_distance(kd.outcomes, od)
    assert tv < 1e-9, 'oracle TV %r' % tv


# ---------------------------------------------------------------------------
# CB-2: kernel Heroic Death must not count pre-combat hits (engine parity)
# ---------------------------------------------------------------------------

def _db_vs_hd_ctx():
    return ExchangeContext(attacker=sh(4, 0, NZ(1)),
                           defender=fp(4, 0, L(1) + (C('Boromir', 2, 1),)),
                           attacker_card="Durin's Bane",
                           defender_card='Heroic Death',
                           choices=(('def_effect_count', 2),))


def test_cb2_kernel_heroic_death_ignores_pre_combat_hits():
    ctx = _db_vs_hd_ctx()
    dist = cd.exact_distribution(ctx)
    bad = sum(p for obs, p in dist.outcomes.items()
              if obs[0] == 0 and _has_boromir(obs))
    assert bad == 0.0, \
        'kernel sacrifices Boromir with 0 cancellable hits: mass %.6f' % bad
    # engine reference: DB [6,6,6], all other dice miss -> no sacrifice
    c = cd.build_round(ctx)
    rec = Rec(scripts=[[6, 6, 6]], fill=1)
    with_roller(rec, lambda: cb.resolve_combat_round(c, cd.round_choices(ctx)))
    assert c.attacker_pre_combat_hits == 3 and c.attacker_hits == 0
    assert c.defender_hits_cancelled == 0
    assert any(getattr(l, 'name', '') == 'Boromir'
               for l in c.defender_army.leaders)
    import cert_combat_dist as cert
    od, _n = cert.oracle_distribution(
        lambda: cert._run_engine_round(ctx), conf=cd.plan_conf_case(ctx))
    tv = cert.tv_distance(dist.outcomes, od)
    assert tv < 1e-9, 'DB-vs-HD oracle TV %r' % tv


# ---------------------------------------------------------------------------
# CB-3: an additional attack with ZERO dice rolls nothing (engine + kernel)
# ---------------------------------------------------------------------------

def test_cb3_zero_dice_additional_attack_engine():
    # Charge after Durin's Bane (defender, resolves first) killed both Elites
    ctx = ExchangeContext(attacker=ArmySpec(regulars=0, elites=2, nation=GON),
                          defender=sh(5), attacker_card='Charge',
                          defender_card="Durin's Bane")
    c = cd.build_round(ctx)
    rec = Rec(scripts=[[6, 6, 6]], fill=6)
    with_roller(rec, lambda: cb._resolve_pre_combat(c))
    assert cb.count_elite_units(c.attacker_army) == 0
    assert rec.calls == [(3, 'combat roll')], \
        'Charge with 0 Elites consumed a batch: %r' % (rec.calls,)
    assert c.attacker_pre_combat_rolls == [] and c.attacker_pre_combat_hits == 0
    # Sudden Strike after Words of Power zeroed the only Companion's Leadership
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(1)),
                          defender=fp(3, 0, (C('Legolas', 2, 1),)),
                          attacker_card='Words of Power',
                          defender_card='Sudden Strike')
    c = cd.build_round(ctx)
    cb.apply_words_of_power(c)
    assert c.defender_leadership == 0
    rec = Rec(fill=6)
    with_roller(rec, lambda: cb._apply_pre_combat_for_side(c, is_attacker=False))
    assert rec.calls == [], \
        'Sudden Strike with Leadership 0 consumed a batch: %r' % (rec.calls,)
    assert c.defender_pre_combat_hits == 0, 'phantom pre-combat hit'


def test_cb3_zero_dice_additional_attack_kernel():
    # Sudden Strike at Leadership 0: every kernel outcome has def_pre == 0
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(1)),
                          defender=fp(3, 0, (C('Legolas', 2, 1),)),
                          attacker_card='Words of Power',
                          defender_card='Sudden Strike')
    dist = cd.exact_distribution(ctx)
    bad = sum(p for obs, p in dist.outcomes.items() if obs[3] > 0)
    assert bad == 0.0, 'kernel phantom Sudden Strike die mass %.6f' % bad
    # Charge with the Elites gone (Durin's Bane scored 3): atk_pre == 0
    ctx = ExchangeContext(attacker=ArmySpec(regulars=0, elites=2, nation=GON),
                          defender=sh(5), attacker_card='Charge',
                          defender_card="Durin's Bane")
    dist = cd.exact_distribution(ctx)
    bad = sum(p for obs, p in dist.outcomes.items()
              if obs[3] == 3 and obs[2] > 0)
    assert bad == 0.0, 'kernel phantom Charge die mass %.6f' % bad
    # the batch plan stays aligned with the engine (no plan entry for the
    # skipped roll): every defer-mode witness plan reproduces its outcome
    ctxw = ExchangeContext(attacker=ArmySpec(regulars=0, elites=2, nation=GON),
                           defender=sh(5), attacker_card='Charge',
                           defender_card="Durin's Bane", defer_casualties=True)
    dw = cd.exact_distribution(ctxw, want_witness=True)
    seen_skip = False
    for obs, plan in dw.witness.items():
        cw = cd.build_round(ctxw)
        la0, ld0 = list(cw.attacker_army.leaders), list(cw.defender_army.leaders)
        cd.run_with_plan(plan, cd.plan_conf_case(ctxw),
                         lambda: cb.resolve_combat_round(cw, cd.round_choices(ctxw)))
        assert cd.observe(cw, la0, ld0) == obs, (plan, obs)
        if obs[3] == 3:
            seen_skip = True
    assert seen_skip, 'no DB=3 branch in the witness set'


# ---------------------------------------------------------------------------
# CB-4: Foul Stench vs a shining Gandalf the White
# ---------------------------------------------------------------------------

def test_cb4_foul_stench_respects_shine():
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(3)),
                          defender=fp(3, 0, (C('Gandalf the White', 3, 1),) + L(1)),
                          attacker_card='Foul Stench', shine=True)
    c = cd.build_round(ctx)
    assert c.attacker_leadership == 0 and c.defender_leadership == 1
    cb.apply_reroll_cancellation(c)
    assert c.defender_reroll_cancelled is False, \
        'Foul Stench cancelled the FP re-roll against negated Nazgul Leadership'
    assert getattr(c, '_foul_stench_cancel_def', False) is False
    # control: shine declined -> 3 >= 2 -> cancelled
    ctx2 = ExchangeContext(attacker=sh(4, 0, NZ(3)),
                           defender=fp(3, 0, (C('Gandalf the White', 3, 1),) + L(1)),
                           attacker_card='Foul Stench', shine=False)
    c2 = cd.build_round(ctx2)
    cb.apply_reroll_cancellation(c2)
    assert c2.defender_reroll_cancelled is True
    # play-legality: Foul Stench requires Nazgul Leadership >= 1, which the
    # shine negates (cards.can_play_combat_card, USER 2026-08-03 case I)
    from cards import can_play_combat_card, get_card
    ok, _why = can_play_combat_card(get_card(cd.combat_card_id('Foul Stench')),
                                    c, is_attacker=True)
    assert ok is False, 'Foul Stench playable under a shine'


# ---------------------------------------------------------------------------
# 1.255: capture only on ENTRY (USER 2026-08-29)
# ---------------------------------------------------------------------------

def test_1255_field_wipe_does_not_capture():
    st = fresh()
    place(st, 'west harondor', units(SAU, 5))
    place(st, 'pelargir', units(GON, 1))
    b0 = st.political.boxes[GON]
    a0 = st.political.active.get(GON)
    res = resolve_with(st, Rec(scripts=[[6, 6, 6, 6, 6]], fill=1),
                       'west harondor', 'pelargir')
    assert res.defender_destroyed and not res.attacker_destroyed
    assert cb._army_entry(st, 'pelargir')[1] is None
    assert 'pelargir' not in st.enemy_controlled[Side.SHADOW], \
        '1.255: captured at the wipe without entering'
    assert st.political.boxes[GON] == b0 and st.political.active.get(GON) == a0, \
        '1.255: capture-advance fired at the wipe (%r -> %r)' % (
            (b0, a0), (st.political.boxes[GON], st.political.active.get(GON)))
    assert cb._wipe_captures_in_place('west harondor', 'pelargir') is False
    # the victor ENTERS (the victor_advance executor's move + capture)
    game_logic.move_army(st, 'west harondor', 'pelargir', move_units=5,
                         move_elite=0, move_leaders=0, side=Side.SHADOW)
    cb._check_settlement_capture(st, 'pelargir', Side.SHADOW)
    assert 'pelargir' in st.enemy_controlled[Side.SHADOW]
    assert st.political.boxes[GON] == b0 + 1 and st.political.active.get(GON)


def test_1255_onslaught_field_wipe_does_not_capture():
    # enemy (defender) wiped by the Onslaught dice from an ADJACENT region
    st = fresh()
    place(st, 'west harondor', units(SAU, 4))
    place(st, 'pelargir', units(GON, 1))
    prev = cb.set_active_state(st)
    try:
        own_dead, enemy_dead, _r, hits = with_roller(
            Rec(fill=6), lambda: cb.apply_onslaught_choice(
                st, 'west harondor', 'pelargir', Side.SHADOW,
                Side.FREE_PEOPLES, count=1, enemy_is_defender=True))
    finally:
        cb.set_active_state(prev)
    assert enemy_dead and not own_dead and hits == 1
    assert 'pelargir' not in st.enemy_controlled[Side.SHADOW], \
        '1.255: Onslaught wipe captured without entry'
    # self-wiped DEFENDER in the field: the adjacent attacker did not enter
    st2 = fresh()
    place(st2, 'west harondor', units(SAU, 4))
    place(st2, 'pelargir', units(GON, 2))
    prev = cb.set_active_state(st2)
    try:
        own_dead2, enemy_dead2, _r2, _h2 = with_roller(
            Rec(fill=1), lambda: cb.apply_onslaught_choice(
                st2, 'pelargir', 'west harondor', Side.FREE_PEOPLES,
                Side.SHADOW, count=2, enemy_is_defender=False))
    finally:
        cb.set_active_state(prev)
    assert own_dead2 and not enemy_dead2
    assert 'pelargir' not in st2.enemy_controlled[Side.SHADOW], \
        '1.255: self-wiped field defender yielded the city without entry'


def test_1255_colocated_siege_wipe_captures():
    # the besieger already stands IN the stronghold region: the wipe is an
    # entry (rulebook: garrison eliminated, besieging army remains)
    st = fresh()
    mt = place(st, 'minas tirith', units(GON, 1) + units(SAU, 5))
    st.besieged_strongholds = {sanitize('minas tirith')}
    res = resolve_with(st, Rec(fill=6), 'minas tirith', 'minas tirith',
                       is_siege=True)
    assert res.defender_destroyed and not res.attacker_destroyed
    assert cb._wipe_captures_in_place('minas tirith', 'minas tirith') is True
    assert 'minas tirith' in st.enemy_controlled[Side.SHADOW], \
        'co-located siege wipe must capture (the attacker is inside)'
    assert cb._side_units(mt, Side.SHADOW) and not cb._side_units(mt, Side.FREE_PEOPLES)


# ---------------------------------------------------------------------------
# CB-5: internal multi-round loop re-derives Leadership each round
# ---------------------------------------------------------------------------

def test_cb5_multi_round_resets_leadership():
    st = fresh()
    place(st, 'dagorlad', units(GON, 5),
          [gs.Companion(name='Aragorn', level=3, leadership=2),
           gs.Companion(name='Legolas', level=2, leadership=1)])
    place(st, 'gorgoroth', units(SAU, 5), [gs.Nazgul(nation=SAU)])
    rec = Rec(fill=1)
    res = resolve_with(st, rec, 'dagorlad', 'gorgoroth',
                       attacker_card_id=cd.combat_card_id('Mighty Attack'),
                       max_rounds=2)
    assert res.rounds_fought == 2
    rr = [n for n, x in rec.calls if x == 'leader reroll']
    # round 1: FP L 3-1 (Mighty Attack forfeit) = 2, SH 1; round 2: FP 3, SH 1
    assert rr == [2, 1, 3, 1], 'leader re-roll batches %r' % (rr,)


def test_cb5_recompute_helper_matches_setup():
    st = fresh()
    wk = game_logic.Minion(nation=SAU, name='Witch King')
    place(st, 'gorgoroth', units(SAU, 4), [gs.Nazgul(nation=SAU),
                                            gs.Nazgul(nation=SAU), wk])
    place(st, 'dagorlad', units(GON, 3),
          [gs.Companion(name='Gandalf the White', level=3, leadership=1),
           gs.Leader(nation=GON)])
    for shine, want in ((True, (0, 1)), (False, (4, 2)), (None, (0, 2))):
        st.pending_combat = ({'gtw_shine': shine} if shine is not None
                             else None)
        c = cb.setup_combat(st, 'gorgoroth', 'dagorlad')
        assert (c.attacker_leadership, c.defender_leadership) == want, \
            (shine, c.attacker_leadership, c.defender_leadership)
        assert c.nazgul_in_battle == 4
        # per-round forfeits are wiped by the round reset; the shine stays
        c.attacker_leadership = 0
        c.defender_leadership = 0
        c.nazgul_in_battle = 1
        cb._recompute_round_leadership(c)
        assert (c.attacker_leadership, c.defender_leadership) == want, \
            ('recompute', shine, c.attacker_leadership, c.defender_leadership)
        assert c.nazgul_in_battle == 4
    # Saruman in play: Isengard Elites count as Leaders (recomputed from
    # the live roster, so a siege-press downgrade drops the bonus)
    st2 = fresh()
    place(st2, 'orthanc', [], [game_logic.Minion(nation=Nation.ISENGARD,
                                                 name='Saruman')])
    a = place(st2, 'gorgoroth', units(SAU, 2) + units(Nation.ISENGARD, 3,
                                                       UnitKind.ELITE))
    place(st2, 'dagorlad', units(GON, 3), [gs.Leader(nation=GON)])
    c2 = cb.setup_combat(st2, 'gorgoroth', 'dagorlad')
    assert c2.attacker_leadership == 3, c2.attacker_leadership
    a.units[-1] = gs.Unit(Nation.ISENGARD, UnitKind.REGULAR)
    cb._recompute_round_leadership(c2)
    assert c2.attacker_leadership == 2, c2.attacker_leadership


# ---------------------------------------------------------------------------
# CB-6: Fateful Strike gate is exact (Foul Stench / They are Terrible never
# eliminate a stranded Nazgul from the FP entry)
# ---------------------------------------------------------------------------

def _stranded_state(card_name):
    atk = gs.Army(region='gorgoroth', units=units(SAU, 4),
                  leaders=[gs.Nazgul(nation=SAU)])
    dfn = gs.Army(region='dagorlad', units=units(GON, 3),
                  leaders=[gs.Leader(nation=GON), gs.Nazgul(nation=SAU)])
    c = gs.CombatState(attacker_region='gorgoroth', defender_region='dagorlad',
                       attacker_army=atk, defender_army=dfn,
                       attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
                       attacker_card=cd.combat_card(card_name))
    c.nazgul_in_battle = 1
    return c


def test_cb6_fateful_strike_gate_exact():
    for nm in ('Foul Stench', 'They are Terrible'):
        c = _stranded_state(nm)
        cb.apply_fateful_strike(c, 2, is_attacker=True)
        assert any(isinstance(l, gs.Nazgul) for l in c.defender_army.leaders), \
            '%s eliminated the stranded Nazgul (Fateful Strike gate)' % nm
        assert c.nazgul_in_battle == 1
    c = _stranded_state('Foul Stench')
    c.attacker_card = cd.combat_card('Fateful Strike')
    cb.apply_fateful_strike(c, 1, is_attacker=True)
    assert not any(isinstance(l, gs.Nazgul) for l in c.defender_army.leaders)
    # kernel mirror: a Shadow Foul Stench never books an 'N' kill on the FP side
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(1)),
                          defender=fp(3, 0, L(1) + NZ(1)),
                          attacker_card='Foul Stench')
    dist = cd.exact_distribution(ctx)
    bad = sum(p for obs, p in dist.outcomes.items() if 'N' in obs[15])
    assert bad == 0.0, 'kernel Foul Stench killed a stranded Nazgul: %.6f' % bad


# ---------------------------------------------------------------------------
# CB-8: _capture_advance propagates unexpected failures
# ---------------------------------------------------------------------------

def test_cb8_capture_advance_is_loud():
    st = fresh()
    st.political.boxes[GON] = 1
    st.political.active[GON] = False
    orig = game_logic.activate_nation

    def boom(*a, **k):
        raise ValueError('boom')
    game_logic.activate_nation = boom
    try:
        raised = False
        try:
            cb._check_settlement_capture(st, 'pelargir', Side.SHADOW)
        except ValueError:
            raised = True
        assert raised, '_capture_advance swallowed a ValueError'
    finally:
        game_logic.activate_nation = orig
    # the intended Wormtongue block (RuntimeError) is still tolerated: the
    # box still advances (1 -> 2, not the passive->At-War clamp)
    st2 = fresh()
    st2.political.boxes[GON] = 1
    st2.political.active[GON] = False

    def blocked(*a, **k):
        raise RuntimeError('Wormtongue')
    game_logic.activate_nation = blocked
    try:
        cb._check_settlement_capture(st2, 'pelargir', Side.SHADOW)
    finally:
        game_logic.activate_nation = orig
    assert 'pelargir' in st2.enemy_controlled[Side.SHADOW]
    assert st2.political.boxes[GON] == 2


# ---------------------------------------------------------------------------
# CB-7 / 1.254: post-casualty enemy hits and target picks are the owner's
# decisions (policy threading, deferral hooks, target hooks)
# ---------------------------------------------------------------------------

def _wctk_state():
    atk = gs.Army(region='gorgoroth', units=units(SAU, 2, UnitKind.ELITE),
                  leaders=[])
    dfn = gs.Army(region='dagorlad',
                  units=units(GON, 1) + units(GON, 1, UnitKind.ELITE), leaders=[])
    c = gs.CombatState(attacker_region='gorgoroth', defender_region='dagorlad',
                       attacker_army=atk, defender_army=dfn,
                       attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
                       attacker_card=cd.combat_card('We Come to Kill'))
    return c


def test_cb7_wctk_hits_honor_owner_policy_and_defer():
    # 2 Elites -> WCtK rolls 2; script one hit
    c = _wctk_state()
    with_roller(Rec(scripts=[[6, 1]]), lambda: cb._apply_post_casualty_for_side(
        c, is_attacker=True, choices={'def_casualty_policy': 'preserve_units'}))
    kinds = sorted(u.kind.name for u in c.defender_army.units)
    assert kinds == ['REGULAR', 'REGULAR'], \
        'WCtK ignored the owner policy (preserve_units): %r' % (kinds,)
    assert c.attacker_bonus_hits == 1
    # deferral: the hit is stashed for the learned allocation, not applied
    c = _wctk_state()
    with_roller(Rec(scripts=[[6, 1]]), lambda: cb._apply_post_casualty_for_side(
        c, is_attacker=True, choices={'defer_post_hits': True}))
    assert len(c.defender_army.units) == 2, 'deferred WCtK hit was applied'
    assert getattr(c, 'deferred_post_hits_on_defender', 0) == 1
    assert getattr(c, 'deferred_post_hits_on_attacker', 0) == 0
    assert c.attacker_bonus_hits == 1
    assert not c.defender_destroyed and not c.battle_over
    # Great Host: the automatic hit defers the same way
    c = _wctk_state()
    c.attacker_card = cd.combat_card('Great Host')
    c.attacker_army.units = units(SAU, 4)
    cb._apply_post_casualty_for_side(c, is_attacker=True,
                                     choices={'defer_post_hits': True})
    assert len(c.defender_army.units) == 2
    assert getattr(c, 'deferred_post_hits_on_defender', 0) == 1


def test_cb7_resume_deferred_reports_post_hits():
    st = fresh()
    place(st, 'gorgoroth', units(SAU, 2, UnitKind.ELITE))
    place(st, 'dagorlad', units(GON, 1) + units(GON, 1, UnitKind.ELITE))
    out = {}
    a_d, d_d, role = with_roller(Rec(scripts=[[6, 1]]), lambda: cb.resume_deferred_post_casualty(
        st, 'gorgoroth', 'dagorlad', Side.SHADOW, Side.FREE_PEOPLES,
        cd.combat_card_id('We Come to Kill'), None,
        choices={'defer_onslaught': True, 'defer_post_hits': True}, out=out))
    assert (a_d, d_d, role) == (False, False, None)
    assert out.get('post_hits_on_defender') == 1 and out.get('post_hits_on_attacker') == 0
    assert len(cb._army_entry(st, 'dagorlad')[1].units) == 2, 'hit applied despite deferral'
    # without the hook the historical path applies (default policy)
    st2 = fresh()
    place(st2, 'gorgoroth', units(SAU, 2, UnitKind.ELITE))
    place(st2, 'dagorlad', units(GON, 1) + units(GON, 1, UnitKind.ELITE))
    with_roller(Rec(scripts=[[6, 1]]), lambda: cb.resume_deferred_post_casualty(
        st2, 'gorgoroth', 'dagorlad', Side.SHADOW, Side.FREE_PEOPLES,
        cd.combat_card_id('We Come to Kill'), None,
        choices={'defer_onslaught': True}))
    assert len(cb._army_entry(st2, 'dagorlad')[1].units) == 1


def test_cb7_onslaught_defers_enemy_hits():
    st = fresh()
    place(st, 'gorgoroth', units(SAU, 4))
    place(st, 'dagorlad', units(GON, 1) + units(GON, 1, UnitKind.ELITE))
    prev = cb.set_active_state(st)
    try:
        own_dead, enemy_dead, rolls, hits = with_roller(
            Rec(scripts=[[6, 6]]), lambda: cb.apply_onslaught_choice(
                st, 'gorgoroth', 'dagorlad', Side.SHADOW, Side.FREE_PEOPLES,
                count=2, enemy_is_defender=True, defer_enemy_hits=True))
    finally:
        cb.set_active_state(prev)
    assert hits == 2 and len(rolls) == 2
    assert not own_dead and not enemy_dead
    assert len(cb._army_entry(st, 'gorgoroth')[1].units) == 2, 'self hits not applied'
    assert len(cb._army_entry(st, 'dagorlad')[1].units) == 2, 'enemy hits applied despite deferral'


def test_cb7_target_hooks():
    # Black Breath: the Shadow picks WHICH figure dies (Meriadoc over the
    # heuristic's max-level Legolas), or a plain Leader
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(2)),
                          defender=fp(3, 0, (C('Meriadoc', 1, 1), C('Legolas', 2, 1)) + L(1)),
                          attacker_card='Black Breath')
    c = cd.build_round(ctx)
    assert sorted(cb.combat_effect_targets(c, True, reroll_hits=2)) == \
        ['Legolas', 'Meriadoc', 'leader']
    assert sorted(cb.combat_effect_targets(c, True, reroll_hits=1)) == \
        ['Meriadoc', 'leader']
    cb.apply_black_breath(c, 2, is_attacker=True,
                          choices={'atk_effect_target': 'Meriadoc'})
    names = [getattr(l, 'name', type(l).__name__) for l in c.defender_army.leaders]
    assert 'Meriadoc' not in names and 'Legolas' in names, names
    c = cd.build_round(ctx)
    cb.apply_black_breath(c, 2, is_attacker=True,
                          choices={'atk_effect_target': 'leader'})
    assert not any(type(l).__name__ == 'Leader' for l in c.defender_army.leaders)
    assert len(c.companions_in_battle) == 2
    # Words of Power: the Shadow chooses the Companion (Legolas over the
    # heuristic's max-Leadership Aragorn)
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(1)),
                          defender=fp(3, 0, (C('Aragorn', 3, 2), C('Legolas', 2, 1))),
                          attacker_card='Words of Power')
    c = cd.build_round(ctx)
    assert sorted(cb.combat_effect_targets(c, True)) == ['Aragorn', 'Legolas']
    cb.apply_words_of_power(c, choices={'atk_effect_target': 'Legolas'})
    assert getattr(c._wop_cancelled, 'name', None) == 'Legolas'
    assert c.defender_leadership == 2
    # Blade of Westernesse: the FP picks the Minion
    ctx = ExchangeContext(
        attacker=ArmySpec(regulars=4, nation=GON,
                          leaders=(C('Meriadoc', 1, 1), C('Legolas', 2, 1))),
        defender=sh(3, 0, (LeaderSpec('minion', 'Mouth of Sauron'),
                           LeaderSpec('minion', 'Witch King')) + NZ(1)),
        attacker_card='Blade of Westernesse')
    c = cd.build_round(ctx)
    assert sorted(cb.combat_effect_targets(c, True, reroll_hits=1)) == \
        ['Mouth of Sauron', 'Witch King']
    cb.apply_blade_of_westernesse(c, 1, is_attacker=True,
                                  choices={'atk_effect_target': 'Witch King'})
    left = [getattr(l, 'name', '') for l in c.defender_army.leaders]
    assert 'Witch King' not in left and 'Mouth of Sauron' in left, left
    # Heroic Death: the FP picks the Companion of equal level (Gimli over
    # the list-order Boromir)
    ctx = ExchangeContext(attacker=sh(4, 0, NZ(2)),
                          defender=fp(4, 0, (C('Boromir', 2, 1), C('Gimli', 2, 1)) + L(1)),
                          defender_card='Heroic Death',
                          choices=(('def_effect_count', 2),
                                   ('def_effect_target', 'Gimli')))
    c = cd.build_round(ctx)
    assert sorted(cb.combat_effect_targets(c, False)) == ['Boromir', 'Gimli', 'leader']
    c.attacker_hits = 3
    cb._heroic_death_for_side(c, False, cd.round_choices(ctx))
    names = [getattr(l, 'name', type(l).__name__) for l in c.defender_army.leaders]
    assert 'Gimli' not in names and 'Boromir' in names, names
    assert c.defender_hits_cancelled == 2
    # the kernel mirrors the picks (oracle parity keeps the cert exact)
    import cert_combat_dist as cert
    for kctx in (
            ExchangeContext(attacker=sh(4, 0, NZ(2)),
                            defender=fp(3, 0, (C('Meriadoc', 1, 1), C('Legolas', 2, 1)) + L(1)),
                            attacker_card='Black Breath',
                            choices=(('atk_effect_target', 'Meriadoc'),)),
            ExchangeContext(attacker=sh(4, 0, NZ(2)),
                            defender=fp(4, 0, (C('Boromir', 2, 1), C('Gimli', 2, 1)) + L(1)),
                            defender_card='Heroic Death',
                            choices=(('def_effect_count', 2),
                                     ('def_effect_target', 'Gimli')))):
        kd = cd.exact_distribution(kctx)
        od, _n = cert.oracle_distribution(
            lambda: cert._run_engine_round(kctx), conf=cd.plan_conf_case(kctx))
        tv = cert.tv_distance(kd.outcomes, od)
        assert tv < 1e-9, 'target-hook oracle TV %r for %r' % (tv, kctx)


# ---------------------------------------------------------------------------

TESTS = [
    test_cb1_run_with_plan_passes_side,
    test_cb1_cert_oracle_accepts_side,
    test_cb2_kernel_heroic_death_ignores_pre_combat_hits,
    test_cb3_zero_dice_additional_attack_engine,
    test_cb3_zero_dice_additional_attack_kernel,
    test_cb4_foul_stench_respects_shine,
    test_1255_field_wipe_does_not_capture,
    test_1255_onslaught_field_wipe_does_not_capture,
    test_1255_colocated_siege_wipe_captures,
    test_cb5_multi_round_resets_leadership,
    test_cb5_recompute_helper_matches_setup,
    test_cb6_fateful_strike_gate_exact,
    test_cb8_capture_advance_is_loud,
    test_cb7_wctk_hits_honor_owner_policy_and_defer,
    test_cb7_resume_deferred_reports_post_hits,
    test_cb7_onslaught_defers_enemy_hits,
    test_cb7_target_hooks,
]


def main():
    import traceback
    fail = 0
    for t in TESTS:
        try:
            t()
            print('PASS %s' % t.__name__)
        except Exception:
            fail += 1
            print('FAIL %s' % t.__name__)
            tb = traceback.format_exc().splitlines()
            for ln in tb[-4:]:
                print('    ' + ln)
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fail))
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
