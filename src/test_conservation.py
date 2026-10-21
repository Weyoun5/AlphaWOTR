# -*- coding: utf-8 -*-
"""Piece-conservation invariant suite (campaign 2026-07-20).

Locks the casualty / downgrade / card-handler cluster that yielded
bugs F-8, 1.188, 1.189, 1.190: for EVERY nation, at all times
    deployed(regular/elite/leader) + tracked FP casualties  <=  max.
A violation means a figure was MINTED (created beyond the physical
set). This is the unit-level counterpart to the oracle_fuzz runtime
tripwire — faster, deterministic, and it exercises EVERY event-card
handler (the fuzz only reaches the cards its low-sim games happen to
play).

Run: python src/test_conservation.py   (exit 0 = all conserved)
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('PYTHONHASHSEED', '0')

import random

import ai
import board as board_mod
import cards
import combat
import event_effects as ee
import game_logic
import game_state as gs
from board import sanitize
from game_state import Nation, Side, Unit, UnitKind


def conservation_violations(state):
    """List of (nation, type, deployed, cas, max) where dep+cas > max."""
    dep = ai._count_deployed_all(state)
    fcas = getattr(state, 'fp_casualties', None) or {}
    out = []
    for nat, mx in ai.NATION_MAX_FIGURES.items():
        d = dep.get(nat, {})
        c = (fcas.get(nat, {})
             if ai._side_of_nation(nat) == Side.FREE_PEOPLES else {})
        for typ, m in mx.items():
            tot = d.get(typ, 0) + c.get(typ, 0)
            if tot > m:
                out.append((nat.value, typ, d.get(typ, 0),
                            c.get(typ, 0), m))
    return out


def rich_state():
    """A board with every nation deployed, nazgul + FP armies present,
    some FP casualties, elven rings — a plausible mid-game so card
    handlers actually fire."""
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for _s in (Side.FREE_PEOPLES, Side.SHADOW):
        for _d in ('character', 'strategy'):
            try:
                game_logic.draw_event_card(st, _s, _d)
            except Exception:
                pass
    game_logic.roll_action_dice(st, pool=Side.FREE_PEOPLES)
    game_logic.roll_action_dice(st, pool=Side.SHADOW)
    # a Shadow nazgul army adjacent to / on an FP army (Dreadful Spells
    # / Foul Thing targets) — Dagorlad besiegers vs Gondor at Osgiliath
    a = game_logic.get_or_create_army(st, 'osgiliath')
    a.units = [Unit(Nation.GONDOR, UnitKind.REGULAR),
               Unit(Nation.GONDOR, UnitKind.ELITE)]
    sh = game_logic.get_or_create_army(st, 'osgiliath')
    return st


def check(label, state, fails):
    v = conservation_violations(state)
    if v:
        fails.append((label, v))
    return not v


def test_unit_ops(fails):
    """Recruit-to-max then downgrade/kill cycles must never exceed max."""
    st = rich_state()
    combat.set_active_state(st)
    # Elves: deploy 4 regulars + 2 elites at Lorien, 1 reg in reinf
    for k, a in st.armies.items():
        a.units = [u for u in a.units if u.nation != Nation.ELVES]
    L = game_logic.get_or_create_army(st, 'lorien')
    L.units = ([Unit(Nation.ELVES, UnitKind.REGULAR) for _ in range(4)]
               + [Unit(Nation.ELVES, UnitKind.ELITE) for _ in range(2)])
    check('unit_ops:setup', st, fails)
    # apply many hits under each policy — must not mint
    for pol in ('preserve_elites', 'preserve_units', 'spend_elites'):
        s2 = ai.clone_state(st)
        combat.set_active_state(s2)
        tgt = next(a for k, a in s2.armies.items()
                   if sanitize(k) == 'lorien')
        combat.apply_casualties(tgt, 5, policy=pol)
        check('unit_ops:apply_casualties:' + pol, s2, fails)
    combat.set_active_state(None)


def test_pool_exhausted_multihit(fails):
    """Feanor/Hynch teacher-replay regression (task 164, 1.191).

    A multi-hit 'preserve_elites' allocation on a nation whose REGULAR
    reinforcement pool is already EXHAUSTED must not mint. The policy
    removes regulars FIRST (vacating deployment slots that become
    permanent casualties), then reduces an elite with the last hit — that
    downgrade has to be REFUSED (elite eliminated), never draw a regular
    from the empty pool. Pre-fix _downgrade_allowed read a LIVE deployed
    count deflated by the same-call kills, saw avail>0, and minted a 6th
    Elf regular (deployed 2 + casualties 4 = 6 > max 5)."""
    st = rich_state()
    for k, a in st.armies.items():
        a.units = [u for u in a.units if u.nation != Nation.ELVES]
    # 3 Elf regulars + 3 elites in lorien, 2 MORE regulars parked
    # deployed in rivendell => regular pool = max(5) - deployed(5) = 0,
    # casualty box EMPTY (2026-07-20d: box regulars are now a LEGAL
    # downgrade source, so the negative needs pool 0 AND box 0).
    # 4 hits: the first 3 remove the 3 regulars (the pre-fix trap —
    # those freed deployment slots must NOT read as reinforcement
    # capacity; the kills are tracked only at write-back, so the box
    # is still empty during the call), the 4th reaches an elite whose
    # reduction MUST be refused (no source anywhere).
    L = game_logic.get_or_create_army(st, 'lorien')
    L.units = ([Unit(Nation.ELVES, UnitKind.REGULAR) for _ in range(3)]
               + [Unit(Nation.ELVES, UnitKind.ELITE) for _ in range(3)])
    P = game_logic.get_or_create_army(st, 'rivendell')
    P.units = [u for u in P.units if u.nation != Nation.ELVES] \
        + [Unit(Nation.ELVES, UnitKind.REGULAR) for _ in range(2)]
    st.fp_casualties = {}
    check('pool_exhausted:setup', st, fails)
    # apply 4 hits then TRACK, exactly as resolve_combat's write-back does
    combat.set_active_state(st)
    tgt = next(a for k, a in st.armies.items() if sanitize(k) == 'lorien')
    cas = combat.apply_casualties(tgt, 4, policy='preserve_elites')
    combat._track_fp_casualties(st, cas, Side.FREE_PEOPLES)
    combat.set_active_state(None)
    # the exhausted-pool downgrade must have been refused (elite killed)
    if any(getattr(u, '_downgraded', False) for u in cas):
        fails.append(('pool_exhausted:minted_downgrade',
                      [('Elves', 'regular', 0, 0, 5)]))
    check('pool_exhausted:after_4hits_preserve_elites', st, fails)


def test_split_siege_tracking_order(fails):
    """Giampy/Hynch teacher-replay regression (task 164, 1.191).

    In a co-located SIEGE the two fighting armies are TEMP splits, merged
    back into the shared entry only at the END of resolve_combat. Casualty
    tracking must therefore run AFTER that merge-back: otherwise a real FP
    regular death is booked while state.armies still shows the STALE
    pre-battle deployed count, so deployed+casualties transiently crosses
    max at the _track site (the conservation tripwire wraps _track and read
    that ordering artifact as a mint — Giampy lorien siege: a real Elf
    regular dies dep 3->2 / cas 2->3, but _track ran while dep still read
    3 => transient 6/5). Conservation must hold at the MOMENT each casualty
    is booked, not only in the final state (which was always clean)."""
    st = rich_state()
    for k, a in st.armies.items():
        a.units = [u for u in a.units if u.nation != Nation.ELVES]
    # co-located Lorien: FP Elf garrison (3 regulars + 2 elites — at max
    # given the 2 pre-existing regular casualties) sharing ONE army entry
    # with Shadow besiegers (4 Sauron regulars => 4 guaranteed hits under
    # the all-6 dice stub, enough to kill Elf regulars).
    L = game_logic.get_or_create_army(st, 'lorien')
    L.units = ([Unit(Nation.ELVES, UnitKind.REGULAR) for _ in range(3)]
               + [Unit(Nation.ELVES, UnitKind.ELITE) for _ in range(2)]
               + [Unit(Nation.SAURON, UnitKind.REGULAR) for _ in range(4)])
    st.besieged_strongholds = {'lorien'}
    st.fp_casualties = {Nation.ELVES: {'regular': 2, 'elite': 0, 'leader': 0}}
    check('split_siege:setup', st, fails)

    seen = []
    orig_track = combat._track_fp_casualties
    orig_roll = combat.roll_d6_batch

    def spy_track(state, casualties, side):
        r = orig_track(state, casualties, side)
        v = conservation_violations(state)  # check AT the tracking moment
        if v:
            seen.append(v)
        return r

    combat.roll_d6_batch = lambda n, label=None, **kw: [6] * n
    combat.set_active_state(st)
    combat._track_fp_casualties = spy_track
    try:
        combat.resolve_combat(st, 'lorien', 'lorien', is_siege=True,
                              choices={'attacker_side': Side.SHADOW})
    finally:
        combat._track_fp_casualties = orig_track
        combat.roll_d6_batch = orig_roll
        combat.set_active_state(None)
    if seen:
        fails.append(('split_siege:transient_mint_at_track', seen[0]))
    check('split_siege:final_state', st, fails)


def test_teacher_clamp(fails):
    """Log-alignment placement conservation regression (W9 / 1.194).

    Replay forcing paths append units to the live board.  Before 1.194
    several did so with no reinforcement-pool check, injecting
    rules-illegal OVER-MAX board STATES that were then encoded into every
    subsequent training sample's feature vector (W9: Isengard 13/12,
    Sauron 37/36, Gondor 15+2cas>15).  The fix routes all such placements
    through game_logic._clamp_place, which returns how many pieces may
    LEGALLY be added and DROPS the excess.  This test locks the helper."""
    import game_logic as clt

    # --- _clamp_place unit behaviour ---
    st = rich_state()
    for k, a in list(st.armies.items()):
        a.units = [u for u in a.units if u.nation != Nation.GONDOR]
    G = game_logic.get_or_create_army(st, 'minas tirith')
    G.units = [Unit(Nation.GONDOR, UnitKind.REGULAR) for _ in range(15)]
    st.fp_casualties = {}
    # deployed at max (15/15) -> nothing may be added
    if clt._clamp_place(st, Nation.GONDOR, 'regular', 3) != 0:
        fails.append(('clamp_place:at_max',
                      [('Gondor', 'regular', 15, 0, 15)]))
    # 10 deployed + 3 permanent casualties => remaining = 15-10-3 = 2
    G.units = [Unit(Nation.GONDOR, UnitKind.REGULAR) for _ in range(10)]
    st.fp_casualties = {Nation.GONDOR: {'regular': 3, 'elite': 0,
                                        'leader': 0}}
    if clt._clamp_place(st, Nation.GONDOR, 'regular', 5) != 2:
        fails.append(('clamp_place:fp_casualty_accounting',
                      [('Gondor', 'regular', 10, 3, 15)]))
    # Shadow: casualties return to pool, NOT subtracted; UnitKind arg form
    st2 = rich_state()
    for k, a in list(st2.armies.items()):
        a.units = [u for u in a.units if u.nation != Nation.SAURON]
    S = game_logic.get_or_create_army(st2, 'minas morgul')
    S.units = [Unit(Nation.SAURON, UnitKind.REGULAR) for _ in range(35)]
    if clt._clamp_place(st2, Nation.SAURON, UnitKind.REGULAR, 4) != 1:
        fails.append(('clamp_place:shadow_enum_arg',
                      [('Sauron', 'regular', 35, 0, 36)]))

    # (the _board_snapshot/_apply_resync resync-clamp test died with
    #  the retired teacher 2026-07-27 — _clamp_place unit checks above
    #  still lock the 1.194 conservation semantics)


def test_all_cards(fails):
    """Play EVERY registered card handler from the rich state; assert
    conservation holds after each (catches card-handler mints like
    1.190 Dreadful Spells)."""
    played = 0
    errored = 0
    for cid in sorted(ai._get_card_idx_map()):
        card = cards.get_card(cid)
        if card is None:
            continue
        st = rich_state()
        # give both sides the card + resources so handlers proceed
        side = getattr(card, 'side', Side.SHADOW)
        try:
            st.hands[side].hand.append(cid)
        except Exception:
            pass
        st.fp_elven_rings = 2
        st.sh_elven_rings = 1
        random.seed(1234)
        try:
            ee.play_event_card(st, side, cid, {})
            played += 1
        except Exception:
            errored += 1  # a handler that rejects our generic state is fine
        check('card:' + str(cid), st, fails)
    return played, errored


def main():
    fails = []
    test_unit_ops(fails)
    test_pool_exhausted_multihit(fails)
    test_split_siege_tracking_order(fails)
    test_teacher_clamp(fails)
    played, errored = test_all_cards(fails)
    print('=' * 62)
    print('CONSERVATION SUITE')
    print('  cards exercised: %d played, %d rejected-our-setup' % (
        played, errored))
    print('  violations: %d' % len(fails))
    for label, v in fails[:20]:
        print('  FAIL %s' % label)
        for row in v:
            print('       %s %s: deployed %d + cas %d > max %d' % row)
    print('=' * 62)
    print('RESULT:', 'PASS' if not fails else 'FAIL (%d)' % len(fails))
    # review P2 (2026-07-27): the unconditional `return 0` made
    # exit-code callers see PASS on FAIL
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
