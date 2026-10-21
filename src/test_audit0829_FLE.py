#!/usr/bin/env python3
# -*- coding: ascii -*-
"""FLE regression pins (2026-08-30) - par5b fleet, engine items 11-15.

FIX PINS (RED -> GREEN):
FLE-11r ring-waived event-category play: the Elven-Ring waiver is SPENT
        by the play (par5b_g1#2: one ring bought two die-face changes)
        and, per the LA-1/LA-2 doctrine (the die is treated AS the
        recorded waiver face), a waiver recorded for EVENT earns the
        Gandalf-the-Grey same-deck redraw; one recorded for MUSTER does
        not.
FLE-13b Pits of Mordor: a committed stronghold that receives fewer than
        the printed 2 Sauron Regulars (pool exhausted - physical
        component limits, USER 2026-08-18) gets a LOUD effect line
        naming the short leg (g3#13: 3 strongholds committed, 2
        silently served).
FLE-13c rage_split: a split answer the board cannot fully pay refuses
        loudly (RuntimeError out of the live/self-play path) - never a
        silent partial move (g3#14c: {1,1} silently became 1 unit).
FLE-13d casualty_allocate rows carry 'downgrades'/'kills' decomposition
        riders so two k-rows can never render identically (g3#9,
        g2#23); 'count' stays the decision key.

VERDICT GUARDS (expected GREEN before and after - they pin the triage):
FLE-11q the four redraw quadrants: the same-deck redraw fires ONLY for
        (living Gandalf the Grey guiding) x (EVENT/WILL funding face);
        the par5b "redraw on M/MUSTER_ARMY die / for a dead Gandalf"
        findings (g1#1, g5#8, g2 T7) were the played cards' OWN printed
        "Then, draw one Strategy Event card." (card_list.csv rows
        40/44/45/49) - no engine bug on either axis.
FLE-12  the g3#3 EVENT-die control: the 3 never-offered cards are
        legally gated by their PRINTED play conditions (1.230(a)):
        Corsairs of Umbar "Play if the Southrons & Easterlings are At
        War", The Fighting Uruk-hai "Play if Saruman is in play, and if
        an Army containing an Isengard unit is besieging a Stronghold",
        Candles of Corpses "Play if the Fellowship is not in a region
        containing a Free Peoples Settlement" - each flips ON when its
        condition is met (sensitivity legs).
FLE-12b same-face die fungibility at the EXECUTOR: the menu folds
        same-face dice (one row per face, ai.py normal-generation
        dedup) but _consume_die_for_face accepts the row with die_idx
        overridden to ANY unused die of that face - the bridge's half
        of the g1#3/g3#4 parity gap (see FLE_HANDOFF.md).
FLE-13a Stormcrow is the printed Shadow Muster card (card_list.csv row
        79): retreats the not-At-War nation holding the Ring-bearers/a
        Companion one step and opens the FP-owned elimination pending -
        the g3#5 "Elves-punishing effect" IS the card.
FLE-13k the 2-hit / 5r+1e casualty k-space offers exactly k=0 and k=1,
        and k=1 IS the "mixed" allocation (1 regular kill + 1 elite
        downgrade; rulebook 2745-2767) the g3#9 finding thought was
        missing.
FLE-14  Boromir separation from Moria at movement 2: cardolan / fords
        of bruinen / south downs / trollshaws are genuinely TWO steps
        away (moria-hollin-X, moria-north dunland-cardolan) - the
        g1#11 "union of ranges" menu was board-true; no Rivendell-only
        region leaks in.
FLE-15  Westemnet IS a settlement (a Rohan town, board is_town=True;
        USER rule 2026-07-08: ALL settlements are capturable, towns
        included) - the g2 control marker was legal.

Run:  python test_audit0829_FLE.py     (exit 0 = all pass)
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai                                    # noqa: E402
import board                                 # noqa: E402
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


def cid_of(name):
    def _n(x):
        import unicodedata
        t = unicodedata.normalize('NFKD', str(x))
        t = ''.join(ch for ch in t if not unicodedata.combining(ch))
        return ''.join(ch for ch in t.lower()
                       if ch.isalpha() and ord(ch) < 128)
    for c, card in get_cards().items():
        tn, wn = _n(getattr(card, 'card_name', '')), _n(name)
        if tn == wn or (len(wn) > 6 and wn in tn):
            return c
    raise AssertionError('%s card id resolves' % name)


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


def guide_gandalf(st):
    fs = st.fellowship
    fs.guide = next(c for c in fs.companions
                    if c.name == 'Gandalf the Grey')


def kill_gandalf(st):
    fs = st.fellowship
    g = next(c for c in fs.companions if c.name == 'Gandalf the Grey')
    fs.companions.remove(g)
    fs.guide = next(c for c in fs.companions if c.name == 'Strider')


def fp_play(st, cid, die_idx):
    return ai.apply_action(st, ai.Action(
        die_idx=die_idx, action_type='play_event',
        params={'card_id': cid, 'side': 'fp', 'choices': {}}))


# ---------------------------------------------------------------------------
# FLE-11q: the four redraw quadrants (verdict guard)
# ---------------------------------------------------------------------------

@test
def test_11q_redraw_guide_and_face_quadrants():
    """Redraw = (living Gandalf the Grey guiding) AND (EVENT/WILL
    funding face).  Swords in Eriador / Celeborn's Galadhrim print
    their OWN 'Then, draw one Strategy Event card.' - that single draw
    is the baseline on every quadrant."""
    cid = cid_of('Swords in Eriador')
    # Q1: alive guide + EVENT face -> printed draw + redraw = -2
    st = setup_game()
    guide_gandalf(st)
    arm_hand(st, Side.FREE_PEOPLES, cid)
    set_dice(st, Side.FREE_PEOPLES, [Face.EVENT])
    st.current_player = Side.FREE_PEOPLES
    n0 = len(st.fp_strat_deck)
    st2 = fp_play(st, cid, 0)
    assert len(st2.fp_strat_deck) == n0 - 2, \
        'Q1 alive+EVENT: printed draw + guide redraw (%d -> %d)' \
        % (n0, len(st2.fp_strat_deck))
    # Q2: alive guide + MUSTER face -> printed draw only = -1 (g1#1)
    st = setup_game()
    guide_gandalf(st)
    arm_hand(st, Side.FREE_PEOPLES, cid)
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    st.current_player = Side.FREE_PEOPLES
    n0 = len(st.fp_strat_deck)
    st2 = fp_play(st, cid, 0)
    assert len(st2.fp_strat_deck) == n0 - 1, \
        'Q2 alive+MUSTER: the ONLY draw is the card\'s printed one ' \
        '(g1#1 verdict; %d -> %d)' % (n0, len(st2.fp_strat_deck))
    # Q3: dead Gandalf + EVENT face -> printed draw only = -1
    st = setup_game()
    kill_gandalf(st)
    arm_hand(st, Side.FREE_PEOPLES, cid)
    set_dice(st, Side.FREE_PEOPLES, [Face.EVENT])
    st.current_player = Side.FREE_PEOPLES
    n0 = len(st.fp_strat_deck)
    st2 = fp_play(st, cid, 0)
    assert len(st2.fp_strat_deck) == n0 - 1, \
        'Q3 dead+EVENT: no guide, no redraw (%d -> %d)' \
        % (n0, len(st2.fp_strat_deck))
    # Q4: dead Gandalf + MUSTER_ARMY face (g2 T7 / g5#8 shape) -> -1
    cid2 = cid_of("Celeborn's Galadhrim")
    st = setup_game()
    kill_gandalf(st)
    arm_hand(st, Side.FREE_PEOPLES, cid2)
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER_ARMY])
    st.current_player = Side.FREE_PEOPLES
    n0 = len(st.fp_strat_deck)
    st2 = fp_play(st, cid2, 0)
    assert len(st2.fp_strat_deck) == n0 - 1, \
        'Q4 dead+MUSTER_ARMY: the g2-T7/g5#8 draw is the card\'s own ' \
        'printed text (%d -> %d)' % (n0, len(st2.fp_strat_deck))


# ---------------------------------------------------------------------------
# FLE-11r: ring-waived play spends the waiver + effective-face redraw
# ---------------------------------------------------------------------------

@test
def test_11r_ring_waived_play_spends_waiver_and_uses_waiver_face():
    """par5b_g1#2: a play only fundable through the Elven-Ring waiver
    must SPEND the waiver.  LA-1/LA-2: the play is treated AS the
    recorded waiver face - EVENT earns the Gandalf redraw, MUSTER does
    not."""
    cid = cid_of("Celeborn's Galadhrim")       # Muster-type card
    # leg A: waiver recorded for EVENT, CHARACTER-rolled die
    st = setup_game()
    guide_gandalf(st)
    arm_hand(st, Side.FREE_PEOPLES, cid)
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    st.current_player = Side.FREE_PEOPLES
    st.fp_ring_waiver = True
    st.fp_ring_waiver_face = Face.EVENT
    n0 = len(st.fp_strat_deck)
    st2 = fp_play(st, cid, 0)
    assert not game_logic.ring_waiver_active(st2, Side.FREE_PEOPLES), \
        'FLE-11r RED: the waiver survived the play it alone funded ' \
        '(g1#2: one ring = two die-face changes)'
    assert len(st2.fp_strat_deck) == n0 - 2, \
        'a waiver recorded for EVENT is an Event-category spend: ' \
        'printed draw + guide redraw (%d -> %d)' \
        % (n0, len(st2.fp_strat_deck))
    # leg B: waiver recorded for MUSTER, CHARACTER-rolled die
    st = setup_game()
    guide_gandalf(st)
    arm_hand(st, Side.FREE_PEOPLES, cid)
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    st.current_player = Side.FREE_PEOPLES
    st.fp_ring_waiver = True
    st.fp_ring_waiver_face = Face.MUSTER
    n0 = len(st.fp_strat_deck)
    st2 = fp_play(st, cid, 0)
    assert not game_logic.ring_waiver_active(st2, Side.FREE_PEOPLES), \
        'the MUSTER-recorded waiver is spent too'
    assert len(st2.fp_strat_deck) == n0 - 1, \
        'a waiver recorded for MUSTER is NOT an Event-category spend: ' \
        'printed draw only (%d -> %d)' % (n0, len(st2.fp_strat_deck))


# ---------------------------------------------------------------------------
# FLE-12: the g3#3 EVENT-die control is printed-condition gating
# ---------------------------------------------------------------------------

@test
def test_12_event_die_gating_matches_printed_conditions():
    """g3#3/#6: pids 170/203/207 absent on the EVENT die because their
    printed conditions were unmet (1.230(a)); each flips ON when its
    condition is met."""
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.active[Nation.SAURON] = True
    st.political.boxes[Nation.ISENGARD] = 3
    st.political.active[Nation.ISENGARD] = True
    if st.saruman is None:
        from game_state import Minion
        st.saruman = Minion('Saruman')
    st.saruman.in_play = True
    c_cor = cid_of('Corsairs of Umbar')
    c_can = cid_of('Candles of Corpses')
    c_uru = cid_of('The Fighting Uruk-hai')
    c_sto = cid_of('Stormcrow')
    for c in (c_cor, c_can, c_uru, c_sto):
        arm_hand(st, Side.SHADOW, c)
    set_dice(st, Side.SHADOW, [Face.EVENT])
    st.current_player = Side.SHADOW
    acts = ai.get_legal_actions(st, Side.SHADOW)
    offered = set(a.params['card_id'] for a in acts
                  if a.action_type == 'play_event')
    assert c_sto in offered, 'condition-free-here Stormcrow offered'
    assert c_cor not in offered, \
        'Corsairs gated: S&E not At War (printed condition)'
    assert c_can not in offered, \
        'Candles gated: Fellowship at Rivendell = FP settlement region'
    assert c_uru not in offered, \
        'Fighting Uruk-hai gated: no Isengard army besieging'
    # sensitivity legs: each condition flips its card ON
    st.political.boxes[Nation.SOUTHRON_EASTERLINGS] = 3
    st.political.active[Nation.SOUTHRON_EASTERLINGS] = True
    assert ee.check_event_requirement(st, c_cor)[0], \
        'Corsairs ON once S&E At War'
    a = game_logic.get_or_create_army(st, 'helm s deep')
    a.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ROHAN, Kind.REGULAR))
    st.besieged_strongholds = set(
        getattr(st, 'besieged_strongholds', set()))
    st.besieged_strongholds.add('helm s deep')
    assert ee.check_event_requirement(st, c_uru)[0], \
        'Uruk-hai ON once an Isengard army besieges a stronghold'
    st.fellowship.region = 'hollin'
    assert ee.check_event_requirement(st, c_can)[0], \
        'Candles ON once the Fellowship leaves FP-settlement regions'


# ---------------------------------------------------------------------------
# FLE-12b: same-face die fungibility at the executor
# ---------------------------------------------------------------------------

@test
def test_12b_executor_accepts_same_face_die_override():
    """g1#3/g3#4 engine half: the menu folds same-face dice (one row
    per face), and the EXECUTOR accepts that row with die_idx pointed
    at ANY unused die of the face - the specific-die gesture is
    expressible (bridge normalization = FLE_HANDOFF.md)."""
    st = setup_game()
    st.fellowship.progress = 1                # Nazgul Search req
    cid = cid_of('Nazgul Search')
    arm_hand(st, Side.SHADOW, cid)
    set_dice(st, Side.SHADOW, [Face.CHARACTER, Face.CHARACTER, Face.EYE,
                               Face.EYE, Face.EYE, Face.EYE, Face.EYE])
    st.current_player = Side.SHADOW
    acts = ai.get_legal_actions(st, Side.SHADOW)
    dies = sorted(set(a.die_idx for a in acts
                      if a.action_type == 'play_event'))
    assert dies == [0], \
        'menu folds same-face dice to the first (die0): %r' % (dies,)
    st2 = ai.apply_action(st, ai.Action(
        die_idx=1, action_type='play_event',
        params={'card_id': cid, 'side': 'sh', 'choices': {}}))
    assert st2.dice[Side.SHADOW][1].used \
        and not st2.dice[Side.SHADOW][0].used, \
        'the executor spends exactly the NAMED same-face die (die1)'
    assert cid not in st2.hands[Side.SHADOW].hand, 'the play completed'


# ---------------------------------------------------------------------------
# FLE-13a: Stormcrow = the printed card (verdict guard)
# ---------------------------------------------------------------------------

@test
def test_13a_stormcrow_is_the_printed_elves_effect():
    """card_list.csv row 79: retreat the not-At-War nation holding the
    Ring-bearers one step; the FP player owns the elimination pick.
    g3#5's 'Elves-punishing effect' is exactly this."""
    st = setup_game()
    cid = cid_of('Stormcrow')
    arm_hand(st, Side.SHADOW, cid)
    st.political.boxes[Nation.ELVES] = 1      # advanced once, not at war
    e0 = st.political.boxes[Nation.ELVES]
    res = ee.play_event_card(st, Side.SHADOW, cid,
                             {'nation': 'elves', '_from_pending': True})
    assert res.get('success'), 'play succeeded: %r' % (res,)
    assert st.political.boxes[Nation.ELVES] == e0 - 1, \
        'Elves moved one step AWAY from war (%d -> %d)' \
        % (e0, st.political.boxes[Nation.ELVES])
    pco = getattr(st, 'pending_card_option', None)
    assert pco is not None and pco.get('decider') == 'fp', \
        'the elimination is the FP player\'s learned decision: %r' \
        % (pco and {k: pco[k] for k in ('decider', 'side')},)
    assert any(sanitize(str(o.get('eliminate', {}).get('region')))
               == 'grey havens' for o in pco['options']), \
        'a Grey Havens Elven piece is an offered target (g3#5 feed)'


# ---------------------------------------------------------------------------
# FLE-13b: Pits of Mordor short leg is LOUD
# ---------------------------------------------------------------------------

@test
def test_13b_pits_pool_short_leg_is_loud():
    """g3#13: 3 committed strongholds, pool holds only 4 Regulars ->
    2+2+0 is LEGAL (component limits) but the third leg must say so."""
    card = get_cards()[cid_of('Pits of Mordor')]
    # control: full pool serves all three
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.active[Nation.SAURON] = True
    picks = ee._free_sauron_strongholds(st)[:3]
    assert len(picks) == 3, 'three free Sauron strongholds at setup'
    res = ee._effect_pits_of_mordor(
        st, card, {'strongholds': picks, '_from_pending': True})
    rec = [e for e in res['effects'] if e.startswith('Recruited 2')]
    assert len(rec) == 3, 'full pool: 2 Regulars in EACH pick: %r' \
        % (res['effects'],)
    # starved pool: exactly 4 Sauron Regulars left off-board
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.active[Nation.SAURON] = True
    avail = ai.figures_available(st, Nation.SAURON, 'regular')
    a = game_logic.get_or_create_army(st, 'gorgoroth')
    for _ in range(avail - 4):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    res = ee._effect_pits_of_mordor(
        st, card, {'strongholds': picks, '_from_pending': True})
    rec = [e for e in res['effects'] if e.startswith('Recruited 2')]
    assert len(rec) == 2, 'starved pool: only two picks served: %r' \
        % (res['effects'],)
    short = [e for e in res['effects']
             if 'only 0 of 2' in e and picks[2] in e]
    assert short, \
        'FLE-13b RED: the short leg (%s, 0 of 2) has NO explaining ' \
        'effect line: %r' % (picks[2], res['effects'])


# ---------------------------------------------------------------------------
# FLE-13c: rage_split partial answers refuse loudly
# ---------------------------------------------------------------------------

@test
def test_13c_rage_partial_split_refuses_loudly():
    """g3#14c: a {n_north:1, n_south:1} answer with South Dunland empty
    must raise (action rejected, pending kept) - never silently move
    one unit and commit a downgraded pick."""
    st = setup_game()
    nd = game_logic.get_or_create_army(st, 'north dunland')
    nd.units.clear()
    nd.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    for k in [k for k in st.armies
              if sanitize(k) == 'south dunland']:
        st.armies.pop(k)
    raised = False
    try:
        ee._rage_apply_split(st, {'n_north': 1, 'n_south': 1},
                             'tharbad')
    except RuntimeError:
        raised = True
    assert raised, \
        'FLE-13c RED: the unpayable {1,1} split applied silently ' \
        '(the g3#14c downgrade class)'
    # control: a payable {1,0} split applies cleanly
    st = setup_game()
    nd = game_logic.get_or_create_army(st, 'north dunland')
    nd.units.clear()
    nd.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    moved = ee._rage_apply_split(st, {'n_north': 1, 'n_south': 0},
                                 'tharbad')
    assert moved == 1, 'payable split moves exactly what was asked'


# ---------------------------------------------------------------------------
# FLE-13d + FLE-13k: casualty k-space complete and self-describing
# ---------------------------------------------------------------------------

@test
def test_13d_casualty_rows_decompose_and_kspace_complete():
    """g3#9: 2 hits vs 5r+1e -> exactly k=0 and k=1; k=1 IS the mixed
    allocation (1 regular kill + 1 elite downgrade, rulebook
    2745-2767).  Rows carry 'downgrades'/'kills' riders so no two rows
    can render identically (g2#23)."""
    st = setup_game()
    a = game_logic.get_or_create_army(st, 'westemnet')
    a.units.clear()
    for _ in range(5):
        a.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.ISENGARD, Kind.ELITE))
    st.pending_combat = {
        'phase': 'casualty_allocation', 'alloc_stage': 'defender',
        'defender_region': 'westemnet', 'defender_side': Side.SHADOW,
        'attacker_region': 'westemnet',
        'attacker_side': Side.FREE_PEOPLES,
        'def_hits': 2, 'atk_hits': 0}
    rows = [x for x in ai.get_legal_actions(st, Side.SHADOW)
            if x.action_type == 'casualty_allocate']
    ks = sorted(x.params['count'] for x in rows)
    assert ks == [0, 1], \
        'FLE-13k: k-space for 2 hits vs 5r+1e is exactly {0,1} ' \
        '(k=1 = the mixed allocation): %r' % (ks,)
    for x in rows:
        assert x.params.get('downgrades') == x.params['count'] \
            and x.params.get('kills') == 2 - x.params['count'], \
            'FLE-13d RED: row k=%r lacks the downgrades/kills ' \
            'decomposition riders: %r' % (x.params.get('count'),
                                          x.params)
    sigs = set((x.params.get('kills'), x.params.get('downgrades'))
               for x in rows)
    assert len(sigs) == len(rows), 'row descriptions are distinct'


# ---------------------------------------------------------------------------
# FLE-14: Boromir separation range from Moria is board-true
# ---------------------------------------------------------------------------

@test
def test_14_boromir_separation_from_moria_is_board_true():
    """g1#11 verdict: the 'Eriador' destinations are genuinely 2 steps
    from Moria (moria-hollin-{fords of bruinen, south downs,
    trollshaws}, moria-north dunland-cardolan); nothing leaks in from
    the old Rivendell position."""
    bd = board.load_board()
    hol = set(sanitize(x) for x in bd.adjacent('hollin'))
    ndl = set(sanitize(x) for x in bd.adjacent('north dunland'))
    assert {'fords of bruinen', 'south downs', 'trollshaws'} <= hol, \
        'board: hollin borders the three (client-extracted adjacency)'
    assert 'cardolan' in ndl, 'board: north dunland borders cardolan'
    st = setup_game()
    st.fellowship.region = 'moria'
    st.fellowship.progress = 0
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    st.current_player = Side.FREE_PEOPLES
    acts = ai._get_companion_separate_actions(st, 0)
    dests = set(x.params['to_region'] for x in acts
                if x.params.get('companion') == 'Boromir')
    assert {'cardolan', 'fords of bruinen', 'south downs',
            'trollshaws'} <= dests, \
        'the contested destinations ARE in range 2 of Moria: %r' \
        % (sorted(dests),)
    leak = dests & {'rivendell', 'high pass', 'weather hills',
                    'ettenmoors', 'bree', 'buckland', 'goblins gate',
                    'the shire'}
    assert not leak, \
        'no Rivendell-neighborhood region leaks into the Moria menu ' \
        '(a real union-of-ranges would show these): %r' % (leak,)


# ---------------------------------------------------------------------------
# FLE-15: Westemnet is a capturable settlement (town)
# ---------------------------------------------------------------------------

@test
def test_15_westemnet_is_a_capturable_town():
    """g2 triage: the control marker on Westemnet was legal - the
    board (client-extracted) marks it a Rohan TOWN, and USER rule
    2026-07-08 makes ALL settlements capturable, towns included
    (combat._check_settlement_capture's explicit is_town clause)."""
    board.load_board()
    m = board.regions['westemnet']
    assert m.get('is_town') and not m.get('is_city') \
        and not m.get('is_stronghold'), \
        'Westemnet is a town: %r' % (m.get('is_town'),)
    st = setup_game()
    from combat import _check_settlement_capture
    a = game_logic.get_or_create_army(st, 'westemnet')
    a.units.clear()
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    _check_settlement_capture(st, 'westemnet', Side.SHADOW)
    assert 'westemnet' in st.enemy_controlled.get(Side.SHADOW, set()), \
        'a Shadow capture of the Rohan town Westemnet is tracked ' \
        '(control marker legal)'


def main():
    fails = 0
    for fn in TESTS:
        try:
            fn()
            print('PASS %s' % fn.__name__)
        except Exception:
            fails += 1
            print('FAIL %s' % fn.__name__)
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
