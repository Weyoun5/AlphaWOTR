#!/usr/bin/env python3
# -*- coding: ascii -*-
"""FLC regression pins (2026-08-30) - fleet Family C (par5 engine triage).

FLC-C10 (FIX, RED->GREEN): _effect_faramirs_rangers choiceless
        multi-target silent default - 1.248-class (decision-ownership
        doctrine): the card text says CHOOSE a Shadow Army; with 2+
        valid targets a choiceless play must open pending_card_option
        owned by the card player (same fix as The Spirit of Mordor,
        SQG item 1).  Single target stays forced; teacher replay keeps
        the documented default.

TRIAGE GUARDS (verdicts, expected GREEN on the fixed tree):
FLC-C1  passive->At-War clamp (rulebook 3009-3011): Gondor box 2 +
        passive is offered on NO lane (plain muster, companion
        ability, token) and the executor raises; flipping active
        makes the row appear (sensitivity).  The par5_g1 event was a
        live-state active-flag corruption, not an engine-surface hole
        (probe-confirmed; see FLC_HANDOFF.md).
FLC-C2  a battle whose pending entry lost one whole side terminates
        before any round (menu guard publishes only the decline; the
        executor twin finishes the battle); AND the par5_g1 "phantom
        die" truth: a 1-hit downgrade of the garrison's last elite
        sources its replacement Regular from the CASUALTY BOX first
        (rulebook 2760-2767 verbatim), so the garrison legally lives
        on as 1 Regular and rolls 1 die.
FLC-C4  casualty menu k-rows are never supply-gated (K6 / USER
        2026-08-18) but the executor mints NO phantom Regular: empty
        box+pool => elites eliminated outright (0 arrivals); box
        supply => the arrivals come FROM the box (decremented), the
        pool untouched.  Menu and engine executor agree on the empty
        pool.
FLC-C5  the Gandalf-the-Grey same-deck redraw fires ONLY while the
        living Gandalf the Grey guides (dead Gandalf + Strider guide
        = no grant); the par5_g1 "phantom redraw" was Thranduil's
        Archers' own printed "Then, draw one Strategy Event card".
FLC-C6  The Ents Awake cards are battles for the Political Track
        (rulebook ~3080-3087): one play advances Isengard exactly one
        step (2->3 = At War; Isengard starts active).
FLC-C7  play_event surface gating equals the printed play condition
        (1.230(a): only check_event_requirement gates): Lure of the
        Ring appears exactly when the Fellowship is revealed; Rage of
        the Dunlendings exactly when Isengard is At War - from the
        very same (boot-shaped) hand.
FLC-C3  attacked-nation politics (rulebook 3047-3056 + the printed
        Dale example): one attack on a North army in Dale + the
        capture of Dale = exactly TWO steps once (activate + 2
        advances); an attack without capture = ONE step.

Run:  python test_audit0829_FLC.py     (exit 0 = all pass)
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai                                    # noqa: E402
import board                                 # noqa: E402
import combat as cb                          # noqa: E402
import event_effects as ee                   # noqa: E402
import game_logic                            # noqa: E402
import game_state as gs                      # noqa: E402
from board import sanitize                   # noqa: E402
from cards import get_cards                  # noqa: E402

Side = gs.Side
Nation = gs.Nation
Kind = gs.UnitKind
Face = gs.ActionDieFace

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def entry(st, region_key):
    san = sanitize(str(region_key))
    for k, a in st.armies.items():
        if sanitize(k) == san:
            return a
    return None


def clear(st, region_key):
    a = game_logic.get_or_create_army(st, region_key)
    a.units.clear()
    a.leaders.clear()
    return a


def force_d6(value):
    o_ee = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [value] * n

    def restore():
        ee._d6_batch = o_ee
    return restore


def cid_of(name):
    cid = next((c for c, card in get_cards().items()
                if getattr(card, 'card_name', '') == name), None)
    assert cid is not None, '%s card id resolves' % name
    return cid


def arm_hand(st, side, cid):
    for a in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck',
              'sh_strat_deck'):
        d = getattr(st, a, None)
        if d and cid in d:
            d.remove(cid)
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        h = st.hands[s].hand
        while cid in h:
            h.remove(cid)
    for v in (getattr(st, 'discard_face_up', None) or {}).values():
        while cid in v:
            v.remove(cid)
    st.hands[side].hand.append(cid)


def set_dice(st, side, faces):
    dice = st.dice[side]
    for i, d in enumerate(dice):
        d.used = False
        d.current = faces[i % len(faces)]


def count_units(st, nation, kind):
    n = 0
    for a in st.armies.values():
        n += sum(1 for u in a.units
                 if u.nation == nation and u.kind == kind)
    return n


# ---------------------------------------------------------------------------
# FLC-C10: Faramir's Rangers choiceless multi-target -> owner pending
# ---------------------------------------------------------------------------

@test
def test_faramirs_rangers_multi_target_opens_owner_pending():
    """C10 (1.248-class): 2+ valid Shadow-army targets + no chosen
    target must OPEN pending_card_option for the card player - never
    silently strike the largest army."""
    CID = cid_of("Faramir's Rangers")
    st = setup_game()
    clear(st, 'osgiliath')                 # no recruit leg in this pin
    a1 = clear(st, 'south ithilien')
    for _ in range(3):
        a1.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a2 = clear(st, 'north ithilien')
    a2.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    arm_hand(st, Side.FREE_PEOPLES, CID)
    restore = force_d6(2)                  # if a roll leaks: 0 hits
    try:
        res = ee.play_event_card(st, Side.FREE_PEOPLES, CID, {})
    finally:
        restore()
    assert res.get('success'), 'play refused: %r' % (res,)
    pco = getattr(st, 'pending_card_option', None)
    assert pco is not None, \
        'FLC-C10 RED: choiceless multi-target play opened NO pending ' \
        '(silent default substituted the card player\'s learned ' \
        'decision): %r' % (res.get('effects'),)
    assert pco.get('side') == 'fp' and pco.get('decider') == 'fp', \
        'the target pick belongs to the CARD PLAYER (fp): %r' % (pco,)
    got = sorted(sanitize(str(o.get('target_region')))
                 for o in pco.get('options') or [])
    assert got == ['north ithilien', 'south ithilien'], \
        'one option per valid region: %r' % (got,)
    assert not any(str(e).startswith('Rolled')
                   for e in res.get('effects') or []), \
        'no roll may fire before the target decision: %r' \
        % (res.get('effects'),)
    # card conservation across the deferral: hand -> face-up discard
    assert CID not in st.hands[Side.FREE_PEOPLES].hand, \
        'the played card left the hand'
    assert CID in st.discard_face_up['fp_strategy'], \
        'the played card reached the face-up pile'
    # the FP menu is exactly the option rows; the Shadow is never asked
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    assert acts and all(a.action_type == 'card_option' for a in acts), \
        'the FP menu is exactly the option rows: %r' \
        % ([a.action_type for a in acts],)
    try:
        ai.get_legal_actions(st, Side.SHADOW)
        raise AssertionError('the Shadow must not be offered the '
                             'FP card player\'s target pick')
    except RuntimeError:
        pass
    # resolving the pick through the REAL direct card_option lane
    # strikes the CHOSEN region (north ithilien), not the old
    # largest-army default (south ithilien)
    pick = next(a for a in acts
                if sanitize(str((a.params.get('choices') or {})
                                .get('target_region')))
                == 'north ithilien')
    restore = force_d6(6)                  # 5+ = hit; wipes 1 regular
    try:
        st2 = ai.apply_action(st, pick)
    finally:
        restore()
    a_ni = entry(st2, 'north ithilien')
    assert a_ni is None or not a_ni.units, \
        'the CHOSEN target took the hits: %r' \
        % (a_ni.units if a_ni else None,)
    a_si = entry(st2, 'south ithilien')
    assert a_si is not None and len(a_si.units) == 3, \
        'the unchosen (old-default) target is untouched: %r' % (a_si,)


@test
def test_faramirs_rangers_single_target_stays_forced():
    """C10: exactly ONE valid target = a forced pick, resolved
    immediately with no pending (no decision exists)."""
    CID = cid_of("Faramir's Rangers")
    st = setup_game()
    clear(st, 'osgiliath')
    clear(st, 'north ithilien')
    a1 = clear(st, 'south ithilien')
    a1.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    arm_hand(st, Side.FREE_PEOPLES, CID)
    restore = force_d6(2)                  # 0 hits: no casualty leg
    try:
        res = ee.play_event_card(st, Side.FREE_PEOPLES, CID, {})
    finally:
        restore()
    assert res.get('success'), 'play refused: %r' % (res,)
    assert getattr(st, 'pending_card_option', None) is None, \
        'a forced single-target pick must NOT open a pending: %r' \
        % (st.pending_card_option,)
    rolled = [e for e in res.get('effects') or []
              if str(e).startswith('Rolled')]
    assert rolled and 'south ithilien' in sanitize(rolled[0]), \
        'the forced pick resolves against the one valid target: %r' \
        % (res.get('effects'),)


@test
def test_faramirs_rangers_teacher_replay_keeps_the_default():
    """C10 guard: teacher replay (FINAL_HITS_HOOK, log = truth) never
    opens the pending - the documented largest-army default stands."""
    CID = cid_of("Faramir's Rangers")
    st = setup_game()
    clear(st, 'osgiliath')
    a1 = clear(st, 'south ithilien')
    for _ in range(3):
        a1.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a2 = clear(st, 'north ithilien')
    a2.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    arm_hand(st, Side.FREE_PEOPLES, CID)
    prev_hook = cb.FINAL_HITS_HOOK
    cb.FINAL_HITS_HOOK = lambda *a, **k: None
    restore = force_d6(2)                  # 0 hits: roll only
    try:
        res = ee.play_event_card(st, Side.FREE_PEOPLES, CID, {})
    finally:
        restore()
        cb.FINAL_HITS_HOOK = prev_hook
    assert res.get('success'), 'play refused: %r' % (res,)
    assert getattr(st, 'pending_card_option', None) is None, \
        'teacher replay must not stall on a pending: %r' \
        % (st.pending_card_option,)
    rolled = [e for e in res.get('effects') or []
              if str(e).startswith('Rolled')]
    assert rolled and 'south ithilien' in sanitize(rolled[0]), \
        'teacher default = the largest valid army: %r' \
        % (res.get('effects'),)


# ---------------------------------------------------------------------------
# FLC-C1: passive->At-War clamp holds on EVERY lane (triage guard)
# ---------------------------------------------------------------------------

@test
def test_c1_passive_at_war_clamp_all_lanes():
    """Rulebook 3009-3011: "As long as a Nation remains passive, it
    can never be moved to the last step of the track ('At War')".
    par5_g1 shape: Gondor box 2, passive, FP muster dice live."""
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 2
    st.political.active[Nation.GONDOR] = False
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    st.current_player = Side.FREE_PEOPLES
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    bad = [a for a in acts
           if a.action_type in ('advance_political', 'use_token')
           and a.params.get('nation') == Nation.GONDOR]
    assert not bad, \
        'FLC-C1: passive Gondor at box 2 must NOT be offered an ' \
        'advance into At War: %r' % (bad,)
    for params in ({'nation': Nation.GONDOR},
                   {'nation': Nation.GONDOR,
                    'companion_ability': 'Boromir'}):
        try:
            ai.apply_action(st, ai.Action(die_idx=0,
                                          action_type='advance_political',
                                          params=params))
            raise AssertionError('executor applied a passive->At-War '
                                 'advance: %r' % (params,))
        except RuntimeError as e:
            assert 'Cannot advance' in str(e), \
                'unexpected refusal: %r' % (e,)
    # sensitivity: the pin bites the FLAG - active Gondor gets the row
    st.political.active[Nation.GONDOR] = True
    acts2 = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    ok = [a for a in acts2
          if a.action_type == 'advance_political'
          and a.params.get('nation') == Nation.GONDOR]
    assert ok, 'active Gondor at box 2 must be offered the advance'


# ---------------------------------------------------------------------------
# FLC-C2: wiped-side battle terminates; box-first downgrade lives on
# ---------------------------------------------------------------------------

def _lorien_siege_pc(st):
    return {
        'attacker_region': 'lorien', 'defender_region': 'lorien',
        'attacker_side': Side.SHADOW, 'defender_side': Side.FREE_PEOPLES,
        'is_siege': True, 'stronghold_region': 'lorien',
        'phase': 'defender_combat_card',
        'deciding_side': Side.FREE_PEOPLES, 'rounds_fought': 2,
    }


@test
def test_c2_wiped_garrison_battle_terminates_before_any_roll():
    """A pending battle whose entry lost one whole side must publish
    only the decline row and FINISH on the next action - no round, no
    phantom defender die (the CHAOS-1 guard pair)."""
    st = setup_game()
    a = clear(st, 'lorien')                # co-located siege entry:
    for _ in range(5):                     # besiegers only, garrison 0
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    st.besieged_strongholds = {'lorien'}
    st.pending_combat = _lorien_siege_pc(st)
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    assert [a9.action_type for a9 in acts] == ['combat_card_none'], \
        'the wiped-side menu is exactly the decline row: %r' \
        % ([a9.action_type for a9 in acts],)
    st2 = ai.apply_action(st, acts[0])
    assert st2.pending_combat is None, \
        'FLC-C2: the battle with a destroyed side must terminate ' \
        'before any roll: %r' % (st2.pending_combat,)
    assert 'lorien' in st2.enemy_controlled[Side.SHADOW], \
        'the co-located wipe captures the stronghold (1.255): %r' \
        % (st2.enemy_controlled[Side.SHADOW],)


@test
def test_c2_last_elite_downgrade_sources_the_casualty_box():
    """The par5_g1 "empty-garrison round" truth (rulebook 2760-2767:
    "the Regular unit can be taken from the previous casualties"): a
    1-hit allocation against a garrison of one Elite with a Regular in
    the FP casualty box DOWNGRADES - the garrison legally lives on as
    1 Regular (and rolls 1 die next round); the box is consumed."""
    st = setup_game()
    a = clear(st, 'lorien')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ELVES, Kind.ELITE))
    st.besieged_strongholds = {'lorien'}
    st.fp_casualties.setdefault(
        Nation.ELVES, {'regular': 0, 'elite': 0, 'leader': 0})['regular'] = 1
    ai._apply_allocation(st, 'lorien', Side.FREE_PEOPLES, 1, 0)
    garrison = ai._alloc_side_units(st, 'lorien', Side.FREE_PEOPLES)
    assert [ (u.nation, u.kind) for u in garrison ] \
        == [(Nation.ELVES, Kind.REGULAR)], \
        'the downgraded garrison lives on as 1 Regular: %r' \
        % ([(u.nation, u.kind) for u in garrison],)
    assert st.fp_casualties[Nation.ELVES]['regular'] == 0, \
        'the replacement Regular came FROM the casualty box'
    pc = _lorien_siege_pc(st)
    assert ai._pending_battle_wiped(st, pc) is None, \
        'a live 1-Regular garrison is NOT a wiped side'
    src = (getattr(st, '_dg_src_log', None) or [(None, None)])[-1]
    assert src[-1] == 'cas', \
        'the downgrade source stamp must say cas (box first): %r' \
        % (getattr(st, '_dg_src_log', None),)


# ---------------------------------------------------------------------------
# FLC-C4: casualty k-rows honest against supply (menu == executor)
# ---------------------------------------------------------------------------

def _lorien_alloc_pc(hits):
    return {
        'attacker_region': 'dimrill dale', 'defender_region': 'lorien',
        'attacker_side': Side.SHADOW, 'defender_side': Side.FREE_PEOPLES,
        'is_siege': True, 'stronghold_region': 'lorien',
        'phase': 'casualty_allocation', 'alloc_stage': 'defender',
        'deciding_side': Side.FREE_PEOPLES,
        'def_hits': hits, 'atk_hits': 0, 'rounds_fought': 1,
    }


def _drain_fp_regular_pool(st, nation):
    """Deploy the nation's whole free Regular pool onto the board so
    figures_available == 0 (physical-limits shape)."""
    n = ai.figures_available(st, nation, 'regular')
    if n > 0:
        holder = game_logic.get_or_create_army(st, 'rivendell')
        for _ in range(n):
            holder.units.append(gs.Unit(nation, Kind.REGULAR))
    assert ai.figures_available(st, nation, 'regular') == 0, \
        'pool drained'


@test
def test_c4_empty_pool_and_box_mints_no_phantom_regulars():
    """K6 + USER 2026-08-18: the k rows stay on the menu regardless of
    supply, and the executor eliminates the elites OUTRIGHT with no
    replacement - no phantom Regular anywhere (physical limits)."""
    st = setup_game()
    a = clear(st, 'lorien')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ELVES, Kind.ELITE))
    a.units.append(gs.Unit(Nation.ELVES, Kind.ELITE))
    st.besieged_strongholds = {'lorien'}
    _drain_fp_regular_pool(st, Nation.ELVES)
    st.fp_casualties.setdefault(
        Nation.ELVES, {'regular': 0, 'elite': 0, 'leader': 0})['regular'] = 0
    st.pending_combat = _lorien_alloc_pc(2)
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    ks = sorted(int(a9.params.get('count', -1)) for a9 in acts
                if a9.action_type == 'casualty_allocate')
    assert ks == [0, 1, 2], \
        'k rows are never supply-gated (K6): %r' % (ks,)
    before = count_units(st, Nation.ELVES, Kind.REGULAR)
    pick = next(a9 for a9 in acts
                if a9.action_type == 'casualty_allocate'
                and int(a9.params.get('count', -1)) == 2)
    st2 = ai.apply_action(st, pick)
    after = count_units(st2, Nation.ELVES, Kind.REGULAR)
    assert after == before, \
        'FLC-C4: an empty box+pool downgrade must mint NO Regular ' \
        '(elite eliminated outright): %d -> %d' % (before, after)
    assert not ai._alloc_side_units(st2, 'lorien', Side.FREE_PEOPLES), \
        'both elites died outright (no replacement existed)'


@test
def test_c4_box_supply_arrivals_come_from_the_box():
    """With 2 Regulars in the FP casualty box, k=2 revives exactly
    those two (box -> garrison); the reinforcement pool stays
    untouched at 0 - the +2 arrivals are physically real."""
    st = setup_game()
    a = clear(st, 'lorien')
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ELVES, Kind.ELITE))
    a.units.append(gs.Unit(Nation.ELVES, Kind.ELITE))
    st.besieged_strongholds = {'lorien'}
    _drain_fp_regular_pool(st, Nation.ELVES)
    st.fp_casualties.setdefault(
        Nation.ELVES, {'regular': 0, 'elite': 0, 'leader': 0})['regular'] = 2
    st.pending_combat = _lorien_alloc_pc(2)
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    pick = next(a9 for a9 in acts
                if a9.action_type == 'casualty_allocate'
                and int(a9.params.get('count', -1)) == 2)
    st2 = ai.apply_action(st, pick)
    garrison = ai._alloc_side_units(st2, 'lorien', Side.FREE_PEOPLES)
    assert [(u.nation, u.kind) for u in garrison] \
        == [(Nation.ELVES, Kind.REGULAR)] * 2, \
        'k=2 downgraded both elites into box-sourced Regulars: %r' \
        % ([(u.nation, u.kind) for u in garrison],)
    assert st2.fp_casualties[Nation.ELVES]['regular'] == 0, \
        'both replacement Regulars came FROM the box'
    assert ai.figures_available(st2, Nation.ELVES, 'regular') == 0, \
        'the reinforcement pool was never touched'


# ---------------------------------------------------------------------------
# FLC-C5: Gandalf-the-Grey redraw gate reads the LIVING guide
# ---------------------------------------------------------------------------

def _kill_gandalf_make_strider_guide(st):
    fs = st.fellowship
    g = next((c for c in fs.companions
              if c.name == 'Gandalf the Grey'), None)
    assert g is not None, 'setup Fellowship holds Gandalf the Grey'
    fs.companions.remove(g)
    strider = next(c for c in fs.companions if c.name == 'Strider')
    fs.guide = strider


@test
def test_c5_dead_gandalf_no_redraw_live_gandalf_redraws():
    """par5_g1 01:01:01 triage: the same-deck redraw belongs to the
    LIVING guide Gandalf the Grey only.  Bilbo's Song (no printed
    draw) on an Event die: dead Gandalf + Strider guide = decks
    untouched; Gandalf guiding = one same-deck (character) draw."""
    CID = cid_of("Bilbo's Song")
    # leg A: Gandalf dead, Strider guides -> NO redraw
    st = setup_game()
    _kill_gandalf_make_strider_guide(st)
    arm_hand(st, Side.FREE_PEOPLES, CID)
    set_dice(st, Side.FREE_PEOPLES, [Face.EVENT])
    st.current_player = Side.FREE_PEOPLES
    n_char = len(st.fp_char_deck)
    n_strat = len(st.fp_strat_deck)
    n_hand = len(st.hands[Side.FREE_PEOPLES].hand)
    st2 = ai.apply_action(st, ai.Action(
        die_idx=0, action_type='play_event',
        params={'card_id': CID, 'side': 'fp', 'choices': {}}))
    assert len(st2.fp_char_deck) == n_char \
        and len(st2.fp_strat_deck) == n_strat, \
        'FLC-C5: dead Gandalf granted a redraw (decks %d/%d -> %d/%d)' \
        % (n_char, n_strat, len(st2.fp_char_deck),
           len(st2.fp_strat_deck))
    assert len(st2.hands[Side.FREE_PEOPLES].hand) == n_hand - 1, \
        'the play only left the hand (no draw landed)'
    # leg B (sensitivity): Gandalf the Grey guiding -> same-deck draw
    st = setup_game()
    fs = st.fellowship
    fs.guide = next(c for c in fs.companions
                    if c.name == 'Gandalf the Grey')
    arm_hand(st, Side.FREE_PEOPLES, CID)
    set_dice(st, Side.FREE_PEOPLES, [Face.EVENT])
    st.current_player = Side.FREE_PEOPLES
    n_char = len(st.fp_char_deck)
    st2 = ai.apply_action(st, ai.Action(
        die_idx=0, action_type='play_event',
        params={'card_id': CID, 'side': 'fp', 'choices': {}}))
    assert len(st2.fp_char_deck) == n_char, \
        'the human optional bonus waits for its take/decline decision'
    assert st2.pending_card_option.get('gandalf_draw'), \
        'the living guide earns the bonus opportunity'
    policy = ai.get_ai_policy(st2, Side.FREE_PEOPLES)
    assert len(policy.actions) == 1 and policy.forced_reason == 'bonus_draw_always_take'
    st2 = ai.apply_action(st2, policy.actions[0])
    assert len(st2.fp_char_deck) == n_char - 1, \
        'the fixed AI bonus acceptance draws exactly one matching card'


@test
def test_c5_thranduils_archers_draw_is_the_cards_own():
    """The demanded strategy draw in par5_g1 was the card's printed
    "Then, draw one Strategy Event card" - it fires with Gandalf DEAD
    (and is not the guide ability)."""
    st = setup_game()
    _kill_gandalf_make_strider_guide(st)
    card = None
    for c, cd in get_cards().items():
        if getattr(cd, 'card_name', '') == "Thranduil's Archers":
            card = cd
    assert card is not None
    n_strat = len(st.fp_strat_deck)
    res = ee.CARD_EFFECT_HANDLERS["thranduil's archers"](st, card, {})
    assert any('Strategy card' in str(e)
               for e in res.get('effects') or []), \
        'the printed draw fired: %r' % (res,)
    assert len(st.fp_strat_deck) == n_strat - 1, \
        'exactly one strategy card drawn by the card text'


# ---------------------------------------------------------------------------
# FLC-C6: Ents Awake advances Isengard (a battle for the track)
# ---------------------------------------------------------------------------

@test
def test_c6_ents_awake_advances_isengard_one_step():
    """Rulebook ~3080-3087: the Ents Awake cards are battles in which
    the enemy cannot fight back -> the attacked nation (Isengard,
    active from setup) advances one step: 2 -> 3 (At War)."""
    st = setup_game()
    a = clear(st, 'orthanc')
    a.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    assert st.political.boxes[Nation.ISENGARD] == 2, 'setup box'
    card = None
    for c, cd in get_cards().items():
        if getattr(cd, 'card_name', '') == 'The Ents Awake: Entmoot':
            card = cd
    assert card is not None
    restore = force_d6(2)                  # 4+ hits: none -> politics only
    try:
        ee.CARD_EFFECT_HANDLERS['the ents awake: entmoot'](st, card, {})
    finally:
        restore()
    assert st.political.boxes[Nation.ISENGARD] == 3, \
        'FLC-C6: one Ents Awake play = exactly one Isengard step ' \
        '(At War): box=%r' % (st.political.boxes[Nation.ISENGARD],)


# ---------------------------------------------------------------------------
# FLC-C7: printed play conditions gate the surface (1.230(a))
# ---------------------------------------------------------------------------

@test
def test_c7_condition_gating_matches_printed_text():
    """Boot-shaped Shadow hand with Lure of the Ring ("Play if the
    Fellowship is revealed.") and Rage of the Dunlendings ("Play if
    Isengard is At War."): rows absent while the printed conditions
    are unmet, present the moment each condition holds - from the SAME
    hand (no boot-hand registration asymmetry)."""
    LURE = cid_of('Lure of the Ring')
    RAGE = cid_of('Rage of the Dunlendings')
    st = setup_game()
    arm_hand(st, Side.SHADOW, LURE)
    arm_hand(st, Side.SHADOW, RAGE)
    set_dice(st, Side.SHADOW, [Face.EVENT])
    st.current_player = Side.SHADOW
    st.fellowship.hidden = True
    st.political.boxes[Nation.ISENGARD] = 2

    def plays(st9):
        return {a.params.get('card_id')
                for a in ai.get_legal_actions(st9, Side.SHADOW)
                if a.action_type == 'play_event'}

    p0 = plays(st)
    assert LURE not in p0 and RAGE not in p0, \
        'unmet printed conditions gate the surface (1.230(a)): %r' \
        % (sorted(p0),)
    st.fellowship.hidden = False
    p1 = plays(st)
    assert LURE in p1, \
        'Lure of the Ring appears the moment the Fellowship reveals'
    assert RAGE not in p1, 'Rage still gated (Isengard not At War)'
    st.political.boxes[Nation.ISENGARD] = 3
    p2 = plays(st)
    assert RAGE in p2, \
        'Rage of the Dunlendings appears when Isengard is At War'


# ---------------------------------------------------------------------------
# FLC-C3: attack + capture = two steps once (the printed Dale example)
# ---------------------------------------------------------------------------

@test
def test_c3_attack_plus_capture_two_steps_once():
    """Rulebook 3047-3056 + the page-34 Dale example verbatim: "the
    North nation is activated and its Political counter is moved two
    steps ... since a North Army was attacked and a North City
    captured".  One attack = ONE step; the capture = ONE more."""
    st = setup_game()
    a = clear(st, 'dale')
    a.units.append(gs.Unit(Nation.THE_NORTH, Kind.REGULAR))
    assert st.political.boxes[Nation.THE_NORTH] == 0, 'setup box'
    assert not st.political.is_active(Nation.THE_NORTH), 'passive'
    game_logic.advance_defenders_political(st, 'dale',
                                           attacker_side=Side.SHADOW)
    assert st.political.boxes[Nation.THE_NORTH] == 1, \
        'the attack advance is ONE step at declaration: %r' \
        % (st.political.boxes[Nation.THE_NORTH],)
    assert st.political.is_active(Nation.THE_NORTH), \
        'the attacked nation activates'
    # the wipe + capture leg (attacker takes the city)
    a.units.clear()
    cb._check_settlement_capture(st, 'dale', Side.SHADOW)
    assert st.political.boxes[Nation.THE_NORTH] == 2, \
        'FLC-C3: attack + capture = exactly TWO steps (rulebook Dale ' \
        'example): %r' % (st.political.boxes[Nation.THE_NORTH],)


def main():
    fails = 0
    for fn in TESTS:
        try:
            fn()
            print('[PASS] %s' % fn.__name__)
        except Exception:
            fails += 1
            print('[FAIL] %s' % fn.__name__)
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
