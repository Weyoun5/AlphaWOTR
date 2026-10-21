#!/usr/bin/env python3
"""Regression pins for the 2026-08-29 audit findings GL-1..GL-11
(src/game_logic.py).  Each test went RED against the unfixed code and
GREEN after the fix (see GL_FIX_REPORT.md).

Authorities: rulebook (data/_rulebook.txt line refs in the comments),
rules_amendments.md 1.83 / 1.178a / 1.215, standing rulings 2026-08-16
(hunt-damage ordering: the SAME damage event continues with the NEW
guide's abilities live; a random draw is the casualty-taken row).
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import board                                   # noqa: E402
import game_state as gs                        # noqa: E402
import game_logic as gl                        # noqa: E402
from game_state import (Side, ActionDieFace,   # noqa: E402
                        Nation, UnitKind)


# ---------------------------------------------------------------- helpers
def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    gl.initialize_decks(st)
    return st


def comp(st, name):
    return next(c for c in st.fellowship.companions if c.name == name)


def keep(st, names, guide=None):
    """Trim the Fellowship to *names*; guide = *guide* or the first."""
    st.fellowship.companions[:] = [c for c in st.fellowship.companions
                                   if c.name in names]
    if names:
        st.fellowship.guide = comp(st, guide or names[0])
    else:
        gl.change_guide(st)          # Gollum takes over


def set_dice(st, side, faces):
    st.dice[side] = []
    for face in faces:
        d = gs.ActionDie(faces=[face], side=side)
        d.current = face
        st.dice[side].append(d)


def pawn(st, region, progress, hidden=True):
    st.fellowship.region = st.board.region(region)
    st.fellowship.progress = progress
    st.fellowship.hidden = hidden


def menu_kinds(st):
    return [k for k, _ in gl.hunt_decision_menu(st)]


def leaders_at(st, region):
    out = []
    for k, a in st.armies.items():
        if board.sanitize(k) == board.sanitize(region):
            out.extend(getattr(l, 'name', type(l).__name__)
                       for l in a.leaders)
    return out


def on_map(st, name):
    return any(getattr(l, 'name', '') == name
               for a in st.armies.values() for l in a.leaders)


def queue_names(st):
    return [getattr(h, 'name', str(h))
            for h in (getattr(st, '_hobbit_place_queue', None) or [])]


def wwsat_armed(st):
    """WWSAT on the table + exactly ONE FP hand Character card and no FP
    table Character card -> a casualty resolves the discard inline; the
    FP hand shrinking is the observable (same rig as the streak test)."""
    from persistent_effects import create_persistent_card
    st.persistent_cards.add_card(create_persistent_card(
        'SHADOW_Character_Worn_with_Sorrow_and_Toil',
        'Worn with Sorrow and Toil', Side.SHADOW))
    fpc = next(c for c in st.fp_char_deck if 'Character' in c)
    st.hands[Side.FREE_PEOPLES].hand = [fpc]
    return st


def fp_hand_n(st):
    return len(st.hands[Side.FREE_PEOPLES].hand)


def raises(fn, exc=RuntimeError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- GL-1
def test_gl1_gollum_reveal_offered_in_fp_settlement_after_a_move():
    """Gollum card (rulebook 660-664): no settlement condition.  With
    progress 1 the pawn can leave Rivendell on the reveal placement."""
    st = setup_game()
    keep(st, [])
    pawn(st, 'rivendell', 1)
    st._pending_hunt_damage = 2
    gl._settle_hunt_damage(st)
    assert (st.pending_hunt or {}).get('phase') == 'hunt_gollum', (
        "GL-1: Gollum reveal-to-reduce not offered at Rivendell with "
        "progress 1: pending=%r corruption=%d"
        % (st.pending_hunt, st.fellowship.corruption))
    assert st.fellowship.corruption == 0


def test_gl1_gollum_reveal_withheld_in_fp_settlement_at_progress0():
    """Rulebook 3352-3355 (progress 0: the Fellowship stays) + 3401-3405
    (a reveal placement may not END in an FP city/stronghold)."""
    st = setup_game()
    keep(st, [])
    pawn(st, 'rivendell', 0)
    st._pending_hunt_damage = 2
    gl._settle_hunt_damage(st)
    assert st.pending_hunt is None, st.pending_hunt
    assert st.fellowship.corruption == 2, st.fellowship.corruption


def test_gl1_chain_path_shares_the_settle_predicate():
    """continue_hunt_chain must answer exactly as _settle_hunt_damage."""
    st = setup_game()
    keep(st, [])
    pawn(st, 'rivendell', 0)
    st.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    gl.continue_hunt_chain(st, 2, casualty_taken=True)
    assert st.pending_hunt is None, (
        "GL-1: chain path offered the reveal at progress 0 in Rivendell: %r"
        % (st.pending_hunt,))
    assert st.fellowship.corruption == 2, st.fellowship.corruption

    st = setup_game()
    keep(st, [])
    pawn(st, 'rivendell', 1)
    st.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    gl.continue_hunt_chain(st, 2, casualty_taken=True)
    assert (st.pending_hunt or {}).get('phase') == 'hunt_gollum', \
        st.pending_hunt
    assert st.pending_hunt.get('casualty_taken') is True
    assert st.fellowship.corruption == 0


# ---------------------------------------------------------------- GL-4
def test_gl4_random_draw_of_the_guide_offers_the_new_guides_ability():
    """Rulebook p.42 (new Guide appointed during the Hunt -> ability
    usable immediately), amendment 1.83, standing ruling 2026-08-16."""
    st = setup_game()
    keep(st, ['Legolas', 'Meriadoc'], guide='Legolas')
    pawn(st, 'hollin', 1)
    st.pending_hunt = {'damage': 3, 'phase': 'hunt_sacrifice'}
    gl.resolve_hunt_decision(st, 'hunt_sacrifice_random',
                             forced_companion='Legolas')
    names = [c.name for c in st.fellowship.companions]
    assert names == ['Meriadoc'], names
    assert st.fellowship.guide.name == 'Meriadoc'
    ph = st.pending_hunt or {}
    assert ph.get('phase') == 'hunt_sacrifice' and ph.get('damage') == 1 \
        and ph.get('casualty_taken') is True, (
            "GL-4: remainder not chained to the new guide: pending=%r "
            "corruption=%d" % (st.pending_hunt, st.fellowship.corruption))
    assert st.fellowship.corruption == 0, st.fellowship.corruption
    kinds = menu_kinds(st)
    assert 'hunt_guide_separate' in kinds, kinds
    assert 'hunt_sacrifice_guide' not in kinds, kinds
    assert 'hunt_sacrifice_random' not in kinds, kinds
    # the new guide separates: 0 corruption, placement is a learned pending
    gl.resolve_hunt_decision(st, 'hunt_guide_separate')
    assert st.fellowship.corruption == 0
    assert (st.pending_hunt or {}).get('phase') == 'hobbit_place', \
        st.pending_hunt
    assert st.fellowship.guide.name == 'Gollum'


def test_gl4_random_hobbit_victim_queues_placement_and_chains_remainder():
    st = setup_game()
    keep(st, ['Meriadoc', 'Peregrin'], guide='Meriadoc')
    pawn(st, 'hollin', 1)
    st.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    gl.resolve_hunt_decision(st, 'hunt_sacrifice_random',
                             forced_companion='Meriadoc')
    assert [c.name for c in st.fellowship.companions] == ['Peregrin']
    assert st.fellowship.guide.name == 'Peregrin'
    assert queue_names(st) == ['Meriadoc'], (
        "GL-4: hobbit victim not queued for the learned placement: "
        "queue=%r pending=%r" % (queue_names(st), st.pending_hunt))
    ph = st.pending_hunt or {}
    assert ph.get('phase') == 'hunt_sacrifice' and ph.get('damage') == 1 \
        and ph.get('casualty_taken') is True, ph
    assert st.fellowship.corruption == 0, st.fellowship.corruption
    kinds = menu_kinds(st)
    assert sorted(kinds) == ['hunt_guide_separate',
                             'hunt_take_corruption'], kinds
    gl.resolve_hunt_decision(st, 'hunt_take_corruption')
    assert st.fellowship.corruption == 1
    assert (st.pending_hunt or {}).get('phase') == 'hobbit_place' and \
        st.pending_hunt.get('companion') == 'Meriadoc', st.pending_hunt


def test_gl4_random_hobbit_victim_then_gollum_may_reveal_for_the_rest():
    st = setup_game()
    keep(st, ['Meriadoc'])
    pawn(st, 'hollin', 1)
    st.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    gl.resolve_hunt_decision(st, 'hunt_sacrifice_random',
                             forced_companion='Meriadoc')
    assert st.fellowship.guide.name == 'Gollum'
    assert (st.pending_hunt or {}).get('phase') == 'hunt_gollum', (
        "GL-4: Gollum (new guide) reveal not offered on the remainder: "
        "pending=%r corruption=%d" % (st.pending_hunt,
                                      st.fellowship.corruption))
    assert st.pending_hunt.get('damage') == 1
    assert st.fellowship.corruption == 0
    gl.resolve_hunt_decision(st, 'hunt_gollum_reveal')
    assert st.fellowship.hidden is False
    assert st.fellowship.corruption == 0
    assert (st.pending_hunt or {}).get('phase') == 'hobbit_place', \
        st.pending_hunt


def test_gl4_mordor_random_draw_of_a_hobbit_guide_fires_wwsat():
    """Standing ruling 2026-08-16: a random draw is the casualty-TAKEN
    row (WWSAT fires); only guide ABSORPTION is exempt."""
    st = wwsat_armed(setup_game())
    keep(st, ['Meriadoc', 'Peregrin'], guide='Meriadoc')
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 1
    st.fellowship.hidden = True
    st.pending_hunt = {'damage': 1, 'phase': 'hunt_sacrifice'}
    gl.resolve_hunt_decision(st, 'hunt_sacrifice_random',
                             forced_companion='Meriadoc')
    assert [c.name for c in st.fellowship.companions] == ['Peregrin']
    assert not on_map(st, 'Meriadoc'), "dies on the Mordor track"
    assert fp_hand_n(st) == 0, (
        "GL-4: WWSAT did not fire on a Mordor random draw of the hobbit "
        "guide (FP hand %d)" % fp_hand_n(st))
    assert st.fellowship.guide.name == 'Peregrin'
    assert st.pending_hunt is None, st.pending_hunt
    assert st.fellowship.corruption == 0


# ---------------------------------------------------------------- GL-5
def test_gl5_hobbit_guide_menu_offers_one_separation_row():
    """Amendment 1.83: separated hobbits queue for hobbit_place; the
    separation is never a casualty.  hunt_sacrifice_guide (immediate
    placement + casualty_taken) is suppressed for a hobbit guide;
    hunt_guide_separate is the row."""
    st = setup_game()
    keep(st, ['Meriadoc'])
    pawn(st, 'rivendell', 2)
    st.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    kinds = menu_kinds(st)
    assert 'hunt_guide_separate' in kinds, kinds
    assert 'hunt_sacrifice_guide' not in kinds, (
        "GL-5: duplicate row offered with a hobbit guide: %r" % (kinds,))
    assert 'hunt_take_corruption' in kinds and \
        'hunt_sacrifice_random' in kinds, kinds
    gl.resolve_hunt_decision(st, 'hunt_guide_separate')
    assert 'Meriadoc' not in leaders_at(st, 'rivendell'), \
        "placement is a learned decision, not an immediate drop"
    assert queue_names(st) == ['Meriadoc']
    ph = st.pending_hunt or {}
    assert ph.get('phase') == 'hunt_gollum' and ph.get('damage') == 1 \
        and ph.get('casualty_taken') is False, ph
    # stability: a non-hobbit guide keeps the sacrifice row
    st2 = setup_game()
    keep(st2, ['Legolas', 'Meriadoc'], guide='Legolas')
    pawn(st2, 'hollin', 1)
    st2.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice'}
    kinds2 = menu_kinds(st2)
    assert 'hunt_sacrifice_guide' in kinds2 and \
        'hunt_guide_separate' not in kinds2, kinds2


# ---------------------------------------------------------------- GL-3
def test_gl3_witch_king_alone_is_a_nazgul_for_the_hunt_reroll():
    """Rulebook 2131-2135 (the Witch-king is a Nazgul) + 3666-3676."""
    st = setup_game()
    fr = gl.fellowship_region(st)
    army = gl.get_or_create_army(st, fr)
    army.leaders.append(gl.Minion(nation=Nation.SAURON, name='Witch King'))
    assert gl._nazgul_present(st), "GL-3: WK alone not a Nazgul"
    assert 'Nazgul present' in gl.get_hunt_params(st)['reroll_sources']
    # _hunt_roll: one die misses (1), the WK re-roll hits (6)
    seq = [[1], [6]]
    orig = gl.roll_d6_batch
    gl.roll_d6_batch = lambda n, ctx='': seq.pop(0)
    try:
        got = gl._hunt_roll(st, 1, 6, 0)
    finally:
        gl.roll_d6_batch = orig
    assert got == 1, "GL-3: WK re-roll not applied in _hunt_roll: %r" % got


# ---------------------------------------------------------------- GL-2
def test_gl2_ring_waiver_never_funds_a_will_only_action():
    """Rulebook 1712-1714 / amendments 1.178a, 1.215: a ring never
    produces Will of the West."""
    st = setup_game()
    strider = comp(st, 'Strider')
    st.fellowship.companions.remove(strider)
    gl.get_or_create_army(st, 'minas tirith').leaders.append(strider)
    set_dice(st, Side.FREE_PEOPLES, [ActionDieFace.CHARACTER] * 2)
    st.fp_ring_waiver = True
    assert raises(lambda: gl.crown_aragorn(st, 1)), \
        "GL-2: a CHARACTER die + ring waiver crowned Aragorn"
    assert not st.dice[Side.FREE_PEOPLES][1].used
    assert st.fp_ring_waiver is True, "waiver must not be spent"
    assert not st.aragorn_crowned.crowned
    # a real Will die still pays
    st.dice[Side.FREE_PEOPLES][0].current = ActionDieFace.WILL_OF_WEST
    gl.crown_aragorn(st, 0)
    assert st.aragorn_crowned.crowned


# ---------------------------------------------------------------- GL-6
def test_gl6_reveal_at_progress0_in_a_shadow_stronghold_draws_the_tile():
    """Rulebook 3429-3436: 'remains stationary in a Shadow Stronghold'
    -> a Hunt tile is drawn."""
    import ai
    st = setup_game()
    pawn(st, 'moria', 0)
    st.fellowship.guide = comp(st, 'Gandalf the Grey')
    st.hunt_pool = [gs.HuntTile(0, reveal=False) for _ in range(4)]
    gl._apply_hunt_tile(st, gs.HuntTile(1, reveal=True), 1, mordor=False)
    assert st.fellowship.hidden is False
    assert getattr(st, '_reveal_place_due', False) is True, (
        "GL-6: reveal placement not armed at progress 0 "
        "(_reveal_place_due=%r)" % getattr(st, '_reveal_place_due', None))
    # the tile's own damage settles first (1.132), then the deferred tail
    gl._settle_hunt_damage(st)
    if st.pending_hunt is not None:
        gl.resolve_hunt_decision(st, 'hunt_take_corruption')
    pool0 = len(st.hunt_pool)
    ai.open_reveal_placement(st)
    assert len(st.hunt_pool) == pool0 - 1, (
        "GL-6: stationary-in-Moria tile not drawn (pool %d -> %d)"
        % (pool0, len(st.hunt_pool)))
    assert st.fellowship.progress == 0
    assert gl.fellowship_region(st) == 'moria'


# ---------------------------------------------------------------- GL-7
def test_gl7_refused_special_actions_keep_the_die():
    st = setup_game()
    set_dice(st, Side.SHADOW, [ActionDieFace.MUSTER] * 2)
    assert raises(lambda: gl.muster_minion(st, 'saruman', 'orthanc', 0))
    assert not st.dice[Side.SHADOW][0].used, \
        "GL-7: refused Saruman muster burned the die"
    set_dice(st, Side.FREE_PEOPLES, [ActionDieFace.WILL_OF_WEST] * 2)
    assert raises(lambda: gl.crown_aragorn(st, 0))      # Strider not out
    assert not st.dice[Side.FREE_PEOPLES][0].used, \
        "GL-7: refused crowning burned the die"
    assert raises(lambda: gl.transform_gandalf_white(st, 'fangorn', 1))
    assert not st.dice[Side.FREE_PEOPLES][1].used, \
        "GL-7: refused Gandalf the White burned the die"


# ---------------------------------------------------------------- GL-8
def test_gl8_besieged_cap_counts_the_garrison_side_only():
    """Rulebook 2870-2871: five units INSIDE the besieged stronghold."""
    st = setup_game()
    mt = gl.get_or_create_army(st, 'minas tirith')
    mt.units[:] = [u for u in mt.units if u.nation is Nation.GONDOR][:2]
    for _ in range(4):
        mt.units.append(gs.Unit(Nation.SAURON, UnitKind.REGULAR))
    st.besieged_strongholds = {'minas tirith'}
    gl._muster_one(st, Side.FREE_PEOPLES, 'minas tirith', 'regular',
                   nation=Nation.GONDOR)
    gondor = sum(1 for u in mt.units if u.nation is Nation.GONDOR)
    assert gondor == 3, "GL-8: garrison muster refused/miscounted: %d" % gondor
    while sum(1 for u in mt.units if u.nation is Nation.GONDOR) < 5:
        mt.units.append(gs.Unit(Nation.GONDOR, UnitKind.REGULAR))
    assert raises(lambda: gl._muster_one(
        st, Side.FREE_PEOPLES, 'minas tirith', 'regular',
        nation=Nation.GONDOR)), "garrison of 5 must be refused"


# ---------------------------------------------------------------- GL-11
def test_gl11_declare_requires_a_hidden_fellowship():
    """Rulebook 3361-3363."""
    st = setup_game()
    pawn(st, 'rivendell', 1, hidden=False)
    st.fellowship.corruption = 3
    target = board.sanitize(next(iter(st.board.adjacent('rivendell'))))
    assert raises(lambda: gl.declare_fellowship(st, region=target)), \
        "GL-11: a revealed Fellowship declared (free hide)"
    assert st.fellowship.hidden is False
    assert st.fellowship.progress == 1
    assert st.fellowship.corruption == 3


# ---------------------------------------------------------------- GL-10
def test_gl10_capture_advance_never_takes_a_passive_nation_to_war():
    """Rulebook 3009-3011."""
    from persistent_effects import create_persistent_card
    st = setup_game()
    st.persistent_cards.add_card(create_persistent_card(
        'SHADOW_Character_Wormtongue', 'Wormtongue', Side.SHADOW))
    st.political.active[Nation.ROHAN] = False
    st.political.boxes[Nation.ROHAN] = 2
    gl.activate_nation_for_attack(st, 'westemnet', captured=True)
    assert not st.political.active.get(Nation.ROHAN, False)
    assert st.political.boxes[Nation.ROHAN] == 2, (
        "GL-10: passive Rohan advanced to At War on capture: box %d"
        % st.political.boxes[Nation.ROHAN])
    # an ACTIVE nation still advances on capture
    st.political.active[Nation.ROHAN] = True
    gl.activate_nation_for_attack(st, 'westemnet', captured=True)
    assert st.political.boxes[Nation.ROHAN] == 3


# ---------------------------------------------------------------- GL-9
def _boom(*a, **k):
    raise ValueError('boom')


def test_gl9_turn_start_draw_swallows_only_deck_empty():
    st = setup_game()
    st.fp_char_deck = []                       # documented: tolerated
    gl.begin_turn_reset(st, 2, deal_cards=True)
    orig = gl.draw_event_card
    gl.draw_event_card = _boom
    try:
        assert raises(lambda: gl.begin_turn_reset(st, 3, deal_cards=True),
                      ValueError), "GL-9: turn-start draw error swallowed"
    finally:
        gl.draw_event_card = orig


def test_gl9_next_turn_settlement_control_errors_propagate():
    st = setup_game()
    orig = gl.enforce_settlement_control
    gl.enforce_settlement_control = _boom
    try:
        assert raises(lambda: gl.next_turn(st), ValueError), \
            "GL-9: enforce_settlement_control error swallowed"
    finally:
        gl.enforce_settlement_control = orig


def test_gl9_overstack_detection_errors_propagate():
    import event_effects as ee
    st = setup_game()
    a = gl.get_or_create_army(st, 'osgiliath')
    a.units[:] = [gs.Unit(Nation.GONDOR, UnitKind.REGULAR)
                  for _ in range(11)]
    orig = ee.resolve_overstacks
    ee.resolve_overstacks = _boom
    try:
        assert raises(lambda: gl.cleanup_sieges(st), ValueError), \
            "GL-9: overstack resolution error swallowed"
    finally:
        ee.resolve_overstacks = orig


def test_gl9_reroll_source_errors_propagate():
    st = setup_game()
    orig = gl._nazgul_present
    gl._nazgul_present = _boom
    try:
        assert raises(lambda: gl.get_hunt_params(st), ValueError), \
            "GL-9: get_hunt_params swallowed a re-roll source error"
        assert raises(lambda: gl._hunt_roll(st, 1, 6, 0), ValueError), \
            "GL-9: _hunt_roll swallowed a re-roll source error"
    finally:
        gl._nazgul_present = orig


def test_gl9_advance_defenders_political_narrowed():
    from persistent_effects import create_persistent_card
    st = setup_game()
    st.persistent_cards.add_card(create_persistent_card(
        'SHADOW_Character_Wormtongue', 'Wormtongue', Side.SHADOW))
    st.political.active[Nation.ROHAN] = False
    st.political.boxes[Nation.ROHAN] = 2
    gl.get_or_create_army(st, 'westemnet').units.append(
        gs.Unit(Nation.ROHAN, UnitKind.REGULAR))
    # expected: the passive-at-2 refusal stays silent (box unchanged)
    gl.advance_defenders_political(st, 'westemnet',
                                   attacker_side=Side.SHADOW)
    assert st.political.boxes[Nation.ROHAN] == 2
    assert not st.political.active.get(Nation.ROHAN, False)
    orig = gl.advance_political
    gl.advance_political = _boom
    try:
        assert raises(lambda: gl.advance_defenders_political(
            st, 'westemnet', attacker_side=Side.SHADOW), ValueError), \
            "GL-9: advance_defenders_political swallowed a foreign error"
    finally:
        gl.advance_political = orig


def test_gl9_companion_persistent_discard_errors_propagate():
    st = setup_game()
    st.persistent_cards.check_companion_requirements = _boom
    assert raises(lambda: gl.check_companion_persistent_discards(st),
                  ValueError), \
        "GL-9: check_companion_persistent_discards swallowed an error"


# ---------------------------------------------------------------- EX-5 (AI handoff)
def test_ex5_turn_boundary_resets_routing_marker_and_waiver_faces():
    """EX-5 / EX-13 (AI handoff item 2): begin_turn_reset is the live
    driver's turn boundary and must be the member-for-member twin of
    ai._safe_next_turn - the battle-attacker routing marker and the
    recorded Elven-Ring waiver faces never cross a turn; and
    ring_waiver_spend clears the recorded face like the ai.py spend
    sites do."""
    st = setup_game()
    st._battle_resolved_attacker = Side.SHADOW
    st.fp_ring_waiver = True
    st.fp_ring_waiver_face = ActionDieFace.ARMY
    st.sh_ring_waiver_face = ActionDieFace.MUSTER
    gl.begin_turn_reset(st, 2, deal_cards=False)
    assert getattr(st, '_battle_resolved_attacker', None) is None, (
        "EX-5: _battle_resolved_attacker crossed the turn boundary: %r"
        % (st._battle_resolved_attacker,))
    assert getattr(st, 'fp_ring_waiver_face', None) is None, \
        "EX-13: fp_ring_waiver_face crossed the turn boundary"
    assert getattr(st, 'sh_ring_waiver_face', None) is None, \
        "EX-13: sh_ring_waiver_face crossed the turn boundary"
    assert st.fp_ring_waiver is False
    # next_turn runs the same reset
    st = setup_game()
    st._battle_resolved_attacker = Side.FREE_PEOPLES
    st.sh_ring_waiver_face = ActionDieFace.EYE
    gl.next_turn(st)
    assert getattr(st, '_battle_resolved_attacker', None) is None, \
        "EX-5: next_turn kept the routing marker"
    assert getattr(st, 'sh_ring_waiver_face', None) is None, \
        "EX-13: next_turn kept sh_ring_waiver_face"
    # a spent waiver forgets its face
    st = setup_game()
    st.fp_ring_waiver = True
    st.fp_ring_waiver_face = ActionDieFace.ARMY
    gl.ring_waiver_spend(st, Side.FREE_PEOPLES)
    assert st.fp_ring_waiver is False
    assert getattr(st, 'fp_ring_waiver_face', None) is None, \
        "ring_waiver_spend left the recorded face behind"


# ---------------------------------------------------------------- runner
TESTS = [
    test_gl1_gollum_reveal_offered_in_fp_settlement_after_a_move,
    test_gl1_gollum_reveal_withheld_in_fp_settlement_at_progress0,
    test_gl1_chain_path_shares_the_settle_predicate,
    test_gl4_random_draw_of_the_guide_offers_the_new_guides_ability,
    test_gl4_random_hobbit_victim_queues_placement_and_chains_remainder,
    test_gl4_random_hobbit_victim_then_gollum_may_reveal_for_the_rest,
    test_gl4_mordor_random_draw_of_a_hobbit_guide_fires_wwsat,
    test_gl5_hobbit_guide_menu_offers_one_separation_row,
    test_gl3_witch_king_alone_is_a_nazgul_for_the_hunt_reroll,
    test_gl2_ring_waiver_never_funds_a_will_only_action,
    test_gl6_reveal_at_progress0_in_a_shadow_stronghold_draws_the_tile,
    test_gl7_refused_special_actions_keep_the_die,
    test_gl8_besieged_cap_counts_the_garrison_side_only,
    test_gl11_declare_requires_a_hidden_fellowship,
    test_gl10_capture_advance_never_takes_a_passive_nation_to_war,
    test_gl9_turn_start_draw_swallows_only_deck_empty,
    test_gl9_next_turn_settlement_control_errors_propagate,
    test_gl9_overstack_detection_errors_propagate,
    test_gl9_reroll_source_errors_propagate,
    test_gl9_advance_defenders_political_narrowed,
    test_gl9_companion_persistent_discard_errors_propagate,
    test_ex5_turn_boundary_resets_routing_marker_and_waiver_faces,
]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print('PASS  ' + t.__name__)
        except Exception as e:           # noqa: BLE001 - report every kind
            fails += 1
            print('FAIL  %s: %s: %s' % (t.__name__, type(e).__name__, e))
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
