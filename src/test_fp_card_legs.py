#!/usr/bin/env python3
"""FREE PEOPLES EVENT-CARD LEG BATTERY (BL-90, 2026-08-25).

A live FP tester (session fpgame9) reported that several Free Peoples
event cards were charged a die and left the hand while delivering no
board effect, whereas the SHADOW twins of the same effect classes fired
in full.  This battery is the engine-side half of that claim: it drives
`event_effects.play_event_card` (the exact function
`ai._execute_action`'s 'play_event' branch calls) and asserts that each
printed LEG of the named cards actually lands in the GameState, side by
side with the Shadow twin of the same effect class.

Effect classes covered (FP leg | SHADOW twin leg):
  recruit    Celeborn's Galadhrim, Guards of the Citadel | Horde from
             the East, Rage of the Dunlendings
  heal       Athelas, There is Another Way, Bilbo's Song, Mirror of
             Galadriel | Candles of Corpses (adds), Morgul Wound (adds)
  table      Mithril Coat and Sting + the other 6 FP table cards |
             the 7 Shadow table cards
  draw       Celeborn's Galadhrim's Strategy draw; Athelas grants NONE

Run:  python test_fp_card_legs.py       (exit 0 = all pass)
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import board                                 # noqa: E402
import event_effects as ee                   # noqa: E402
import game_logic                            # noqa: E402
import game_state as gs                      # noqa: E402
from cards import get_card, get_cards        # noqa: E402

Side = gs.Side
Nation = gs.Nation
UnitKind = gs.UnitKind
PCT = gs.PersistentCardType

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Harness (same shape as test_card_conservation.py)
# ---------------------------------------------------------------------------

def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def cid_for(name):
    """Card id for a printed card name (accent/apostrophe tolerant)."""
    key = board.sanitize(name)
    for c, card in get_cards().items():
        if board.sanitize(card.card_name) == key:
            return c
    raise AssertionError('no such card: %r' % name)


def put_in_hand(st, cid):
    """Move a card out of whatever pile holds it and into its owner's
    hand, so play_event_card's 'Card not in hand' guard passes."""
    side = get_card(cid).side
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
    return side


def arm(st, name):
    """Seat the named card in its owner's hand and return (cid, side).

    ALWAYS call this BEFORE snapshotting deck/hand counts: arming moves
    the card out of its deck, which is a bookkeeping change, not a card
    effect.  Measuring across the arm step is what makes a card-draw
    assertion lie by one.
    """
    cid = cid_for(name)
    return cid, put_in_hand(st, cid)


def play_armed(st, cid, side, choices=None):
    return ee.play_event_card(st, side, cid, choices)


def play(st, name, choices=None):
    """Arm + play in one step (for tests that do not count decks)."""
    cid, side = arm(st, name)
    return ee.play_event_card(st, side, cid, choices)


def army(st, region):
    return st.armies.get(ee._find_army_key(st, region))


def n_units(st, region, nation, kind=None):
    a = army(st, region)
    if a is None:
        return 0
    return sum(1 for u in a.units
               if u.nation == nation and (kind is None or u.kind == kind))


def n_leaders(st, region, nation=None):
    a = army(st, region)
    if a is None:
        return 0
    return sum(1 for l in a.leaders
               if type(l).__name__ == 'Leader'
               and (nation is None or getattr(l, 'nation', None) == nation))


def exhaust_pool(st, nation, *types):
    """Spend a Free Peoples nation's reinforcement capacity without
    touching the board: FP casualties are permanent, so charging the
    cap to casualties leaves figures_available() at 0 (rulebook p.30).
    """
    from ai import NATION_MAX_FIGURES
    cas = st.fp_casualties.setdefault(nation, {})
    for t in types:
        cas[t] = NATION_MAX_FIGURES[nation][t]


def on_table(st, cid):
    pc = st.persistent_cards
    return next((c for c in list(pc.fp_cards) + list(pc.sh_cards)
                 if c.card_id == cid), None)


def in_face_up(st, cid):
    return any(cid in v
               for v in (getattr(st, 'discard_face_up', None) or {}).values())


class ForceD6(object):
    """Pin the card-effect d6 funnel so a roll-and-apply card's leg is
    deterministic (event_effects._d6_batch is the ONE funnel every card
    handler rolls through)."""

    def __init__(self, faces):
        self.faces = list(faces)

    def __enter__(self):
        self._prev = ee._d6_batch
        faces = self.faces
        ee._d6_batch = lambda n, ctx='card effect': list(faces[:n])
        return self

    def __exit__(self, *a):
        ee._d6_batch = self._prev
        return False


def set_guide(st, name, level=1):
    """Make `name` the Fellowship guide (creating the Companion when the
    setup Company does not hold one by that name, as for Gollum)."""
    for c in st.fellowship.companions:
        if board.sanitize(c.name) == board.sanitize(name):
            st.fellowship.guide = c
            return c
    c = gs.Companion(name=name, level=level)
    st.fellowship.companions.append(c)
    st.fellowship.guide = c
    return c


def add_die(st, side, face):
    d = gs.ActionDie([face], side)
    d.current = face
    d.used = False
    st.dice.setdefault(side, []).append(d)
    return d


# ---------------------------------------------------------------------------
# 1. RECRUIT LEGS -- "Celeborn's Galadhrim" / "Guards of the Citadel"
#    vs their Shadow twins
# ---------------------------------------------------------------------------

@test
def test_celeborns_galadhrim_recruits_and_draws():
    """'Recruit one Elven unit (Regular or Elite) in Lorien. Then, draw
    one Strategy Event card.'  BOTH legs, not just the draw."""
    st = setup_game()
    cid, side = arm(st, "Celeborn's Galadhrim")
    before_units = n_units(st, 'Lorien', Nation.ELVES)
    before_hand = len(st.hands[Side.FREE_PEOPLES].hand)
    before_deck = len(st.fp_strat_deck)
    r = play_armed(st, cid, side)
    assert r['success'], r
    assert n_units(st, 'Lorien', Nation.ELVES) == before_units + 1, \
        'the RECRUIT leg places one Elven unit in Lorien (%d -> %d)' \
        % (before_units, n_units(st, 'Lorien', Nation.ELVES))
    # the card itself left the hand; the draw must replace it
    assert len(st.hands[Side.FREE_PEOPLES].hand) == before_hand, \
        'the DRAW leg replaces the played card in hand'
    assert len(st.fp_strat_deck) == before_deck - 1, \
        'the DRAW leg takes one card off the FP Strategy deck'


@test
def test_celeborns_galadhrim_recruit_is_not_gated_on_the_draw():
    """An empty FP Strategy deck must not cost the recruit leg."""
    st = setup_game()
    st.fp_strat_deck[:] = []
    before = n_units(st, 'Lorien', Nation.ELVES)
    r = play(st, "Celeborn's Galadhrim")
    assert r['success'], r
    assert n_units(st, 'Lorien', Nation.ELVES) == before + 1, \
        'the recruit leg is independent of the draw leg'


@test
def test_celeborns_galadhrim_draw_is_not_gated_on_the_recruit():
    """An exhausted Elven pool must not cost the draw leg."""
    st = setup_game()
    exhaust_pool(st, Nation.ELVES, 'regular', 'elite')
    cid, side = arm(st, "Celeborn's Galadhrim")
    before_deck = len(st.fp_strat_deck)
    before_units = n_units(st, 'Lorien', Nation.ELVES)
    r = play_armed(st, cid, side)
    assert r['success'], r
    assert n_units(st, 'Lorien', Nation.ELVES) == before_units, \
        'no Elven figure exists to place'
    assert len(st.fp_strat_deck) == before_deck - 1, \
        'the draw leg is independent of the recruit leg'


@test
def test_guards_of_the_citadel_places_unit_and_leader():
    """'Recruit one Gondor unit (Regular or Elite) AND one Gondor Leader
    in Minas Tirith.'"""
    st = setup_game()
    bu = n_units(st, 'Minas Tirith', Nation.GONDOR)
    bl = n_leaders(st, 'Minas Tirith', Nation.GONDOR)
    r = play(st, 'Guards of the Citadel')
    assert r['success'], r
    assert n_units(st, 'Minas Tirith', Nation.GONDOR) == bu + 1, \
        'the UNIT leg places one Gondor unit in Minas Tirith'
    assert n_leaders(st, 'Minas Tirith', Nation.GONDOR) == bl + 1, \
        'the LEADER leg places one Gondor Leader in Minas Tirith'


@test
def test_guards_of_the_citadel_leader_not_gated_on_the_unit():
    """CARD-FLEET LOW-30: an exhausted Gondor UNIT pool must not eat the
    entitled Leader (the two pieces are placed independently)."""
    st = setup_game()
    exhaust_pool(st, Nation.GONDOR, 'regular', 'elite')
    bu = n_units(st, 'Minas Tirith', Nation.GONDOR)
    bl = n_leaders(st, 'Minas Tirith', Nation.GONDOR)
    r = play(st, 'Guards of the Citadel')
    assert r['success'], r
    assert n_units(st, 'Minas Tirith', Nation.GONDOR) == bu, \
        'no Gondor unit figure exists to place'
    assert n_leaders(st, 'Minas Tirith', Nation.GONDOR) == bl + 1, \
        'the Leader is still placed when the unit pool is empty'


def besiege(st, region, nation=Nation.SAURON, n=2):
    """Put `region` under siege by a Shadow army (USER 2026-07-08: you
    may still recruit into a stronghold you are besieged in)."""
    a = army(st, region)
    assert a is not None, 'no army at %r to besiege' % region
    for _ in range(n):
        a.units.append(gs.Unit(nation, UnitKind.REGULAR))
    st.besieged_strongholds.add(board.sanitize(region))
    return a


@test
def test_celeborns_galadhrim_recruits_into_a_besieged_lorien():
    """Recruiting into a stronghold you are besieged in stays legal; the
    garrison cap is 5 and Lorien's setup garrison is 3."""
    st = setup_game()
    besiege(st, 'Lorien')
    before = n_units(st, 'Lorien', Nation.ELVES)
    r = play(st, "Celeborn's Galadhrim")
    assert r['success'], r
    assert n_units(st, 'Lorien', Nation.ELVES) == before + 1, \
        'the recruit leg survives a siege (%d -> %d)' \
        % (before, n_units(st, 'Lorien', Nation.ELVES))


@test
def test_guards_of_the_citadel_recruits_into_a_besieged_minas_tirith():
    st = setup_game()
    besiege(st, 'Minas Tirith')
    bu = n_units(st, 'Minas Tirith', Nation.GONDOR)
    bl = n_leaders(st, 'Minas Tirith', Nation.GONDOR)
    r = play(st, 'Guards of the Citadel')
    assert r['success'], r
    assert n_units(st, 'Minas Tirith', Nation.GONDOR) == bu + 1, \
        'the unit leg survives a siege (garrison cap 5, 4 deployed)'
    assert n_leaders(st, 'Minas Tirith', Nation.GONDOR) == bl + 1, \
        'the leader leg survives a siege'


@test
def test_fully_blocked_fp_muster_completes_as_zero_effect():
    """USER RULING 1.230 (2026-08-28, verbatim: "zero effect plays are
    allowed, always. even in self play") SUPERSEDES CARD-FLEET
    LOW-6/LOW-8 here: a muster card with NOTHING left to place is a
    completed zero-effect play -- success True, card discarded face-up,
    NOT refunded (the die stays spent in ai.apply_action).  (Celeborn's
    still has its draw leg, so use Guards of the Citadel with both
    Gondor pools and the Gondor leader pool spent.)"""
    st = setup_game()
    exhaust_pool(st, Nation.GONDOR, 'regular', 'elite', 'leader')
    cid, side = arm(st, 'Guards of the Citadel')
    r = play_armed(st, cid, side)
    assert not r.get('refund'), \
        'a fully-blocked muster is a zero-effect play, not a refund: %r' \
        % (r,)
    assert r['success'] and r.get('zero_effect'), \
        'the play completes with no effect, got %r' % (r,)
    assert cid not in st.hands[Side.FREE_PEOPLES].hand, \
        'a zero-effect play leaves the hand'
    assert in_face_up(st, cid), 'a zero-effect play is discarded face-up'


@test
def test_athelas_with_no_five_plus_is_a_legal_zero_heal():
    """Zero heals is a legal outcome of the printed text, NOT a dropped
    leg: the play still succeeds and is not refunded."""
    st = setup_game()
    st.fellowship.corruption = 4
    set_guide(st, 'Gandalf the Grey')
    with ForceD6([4, 2, 1]):
        r = play(st, 'Athelas')
    assert r['success'] and not r.get('refund'), r
    assert st.fellowship.corruption == 4, \
        'no result of 5+ heals nothing'


@test
def test_tiaw_hide_option_when_revealed():
    """USER RULING 2026-08-25 (1.229-b, verbatim: 'When gollum is
    guide you get to move or hide depending on whether fellowship is
    revealed. you always get the heal though'): a REVEALED fellowship
    with Gollum as guide gets the HIDE option, and applying it hides
    the fellowship AND heals."""
    from card_options import get_card_options
    st = setup_game()
    gollum = gs.Companion('Gollum', 0, leadership=0)
    st.fellowship.guide = gollum
    st.fellowship.hidden = False
    st.fellowship.corruption = 2
    cid, side = arm(st, 'There is Another Way')
    opts = get_card_options(st, side, 'There is Another Way')
    acts = sorted(str(o.get('action')) for o in opts)
    assert 'hide' in acts, 'revealed+Gollum must offer hide: %r' % acts
    assert 'move' not in acts, \
        'revealed fellowship must not offer move: %r' % acts
    res = ee.play_event_card(st, side, cid,
                             choices={'action': 'hide'})
    assert res.get('success', True), res
    assert st.fellowship.hidden is True, 'hide leg did not hide'
    assert st.fellowship.corruption == 1, \
        'heal always applies (got %d)' % st.fellowship.corruption


@test
def test_tiaw_move_option_when_hidden_and_heal_only_no_gollum():
    """1.229-b's other two legs: HIDDEN + Gollum offers move (not
    hide); no Gollum -> automatic heal-only completion, and the heal still
    lands."""
    from card_options import get_card_options
    st = setup_game()
    st.fellowship.guide = gs.Companion('Gollum', 0, leadership=0)
    st.fellowship.hidden = True
    opts = get_card_options(st, Side.FREE_PEOPLES,
                            'There is Another Way')
    acts = sorted(str(o.get('action')) for o in opts)
    assert 'move' in acts and 'hide' not in acts, acts
    st2 = setup_game()          # default guide (not Gollum)
    st2.fellowship.hidden = False
    st2.fellowship.corruption = 3
    assert get_card_options(st2, Side.FREE_PEOPLES,
                            'There is Another Way') == [{'action': 'heal_only'}]
    cid2, side2 = arm(st2, 'There is Another Way')
    res2 = ee.play_event_card(st2, side2, cid2, choices=None)
    assert res2.get('success', True), res2
    assert st2.fellowship.corruption == 2, \
        'heal-only leg (got %d)' % st2.fellowship.corruption
    assert st2.fellowship.hidden is False, \
        'no Gollum must not hide'


@test
def test_shadow_recruit_twin_horde_from_the_east():
    """The Shadow twin of the recruit class fires (control)."""
    st = setup_game()
    ee.set_at_war(st, Nation.SOUTHRON_EASTERLINGS)
    before = sum(len(a.units) for a in st.armies.values())
    r = play(st, 'Horde from the East')
    assert r['success'], r
    assert sum(len(a.units) for a in st.armies.values()) > before, \
        'the Shadow recruit twin places units'


@test
def test_shadow_recruit_twin_rage_of_the_dunlendings():
    st = setup_game()
    ee.set_at_war(st, Nation.ISENGARD)
    before = sum(len(a.units) for a in st.armies.values())
    r = play(st, 'Rage of the Dunlendings')
    assert r['success'], r
    assert sum(len(a.units) for a in st.armies.values()) > before, \
        'the Shadow recruit twin places units'


# ---------------------------------------------------------------------------
# 2. CORRUPTION HEAL LEGS -- Athelas / There is Another Way / Bilbo's
#    Song / Mirror of Galadriel, against the Shadow ADD twins
# ---------------------------------------------------------------------------

@test
def test_athelas_heals_one_per_five_plus():
    """'Roll three dice. Heal one Corruption point for each result of
    5+.'  Forced roll 6/4/4 = exactly one heal."""
    st = setup_game()
    st.fellowship.corruption = 3
    set_guide(st, 'Gandalf the Grey')
    with ForceD6([6, 4, 4]):
        r = play(st, 'Athelas')
    assert r['success'], r
    assert st.fellowship.corruption == 2, \
        'one 5+ result heals exactly one Corruption (3 -> 2), got %d' \
        % st.fellowship.corruption


@test
def test_athelas_strider_guide_heals_on_three_plus():
    """'If Strider is the Guide, heal ... for each result of 3+.'"""
    st = setup_game()
    st.fellowship.corruption = 4
    set_guide(st, 'Strider')
    with ForceD6([3, 4, 2]):
        r = play(st, 'Athelas')
    assert r['success'], r
    assert st.fellowship.corruption == 2, \
        'Strider guide heals on 3+ (two of 3/4/2), got %d' \
        % st.fellowship.corruption


@test
def test_athelas_grants_no_card_draw():
    """Athelas has NO draw leg -- the engine must not draw or owe one
    (fpgame9 T2: the play demanded a card draw the card never grants)."""
    st = setup_game()
    st.fellowship.corruption = 2
    cid, side = arm(st, 'Athelas')
    decks = {a: len(getattr(st, a)) for a in
             ('fp_char_deck', 'fp_strat_deck')}
    hand = len(st.hands[Side.FREE_PEOPLES].hand)
    with ForceD6([6, 6, 6]):
        r = play_armed(st, cid, side)
    assert r['success'], r
    for a, n in decks.items():
        assert len(getattr(st, a)) == n, \
            'Athelas draws nothing from %s (%d -> %d)' \
            % (a, n, len(getattr(st, a)))
    assert len(st.hands[Side.FREE_PEOPLES].hand) == hand - 1, \
        'the only hand change is Athelas itself leaving'


@test
def test_there_is_another_way_heals_unconditionally():
    """'Heal one Corruption point.  Then, if Gollum is the Guide ...' --
    the heal is unconditional; a non-Gollum guide still heals."""
    st = setup_game()
    st.fellowship.corruption = 1
    set_guide(st, 'Gandalf the Grey')
    r = play(st, 'There is Another Way')
    assert r['success'], r
    assert st.fellowship.corruption == 0, \
        'the heal leg applies with any guide (1 -> 0), got %d' \
        % st.fellowship.corruption


@test
def test_bilbos_song_heals_one_or_two():
    st = setup_game()
    st.fellowship.corruption = 5
    set_guide(st, 'Gandalf the Grey')
    r = play(st, "Bilbo's Song")
    assert r['success'], r
    assert st.fellowship.corruption == 4, \
        "Bilbo's Song heals 1 with a non-Gollum guide, got %d" \
        % st.fellowship.corruption

    st2 = setup_game()
    st2.fellowship.corruption = 5
    set_guide(st2, 'Gollum', level=0)
    r2 = play(st2, "Bilbo's Song")
    assert r2['success'], r2
    assert st2.fellowship.corruption == 3, \
        "Bilbo's Song heals 2 with Gollum as Guide, got %d" \
        % st2.fellowship.corruption


@test
def test_mirror_of_galadriel_heals_in_lorien():
    """'Change any one unused Character Action Die into a Will of the
    West.  Then, if the Fellowship is Lorien, heal one Corruption.'"""
    st = setup_game()
    st.fellowship.corruption = 3
    st.fellowship.region = st.board.region('lorien')
    add_die(st, Side.FREE_PEOPLES, gs.ActionDieFace.CHARACTER)
    r = play(st, 'Mirror of Galadriel')
    assert r['success'], r
    assert any(d.current == gs.ActionDieFace.WILL_OF_WEST
               for d in st.dice[Side.FREE_PEOPLES]), \
        'the die-change leg fired'
    assert st.fellowship.corruption == 2, \
        'the Lorien heal leg fired (3 -> 2), got %d' \
        % st.fellowship.corruption


@test
def test_shadow_heal_twin_candles_of_corpses_adds():
    """The Shadow roll-and-apply twin of Athelas fires (control)."""
    st = setup_game()
    st.fellowship.corruption = 1
    st.fellowship.region = st.board.region('hollin')   # not a FP settlement
    set_guide(st, 'Gandalf the Grey')
    with ForceD6([6, 4, 1]):
        r = play(st, 'Candles of Corpses')
    assert r['success'], r
    assert st.fellowship.corruption == 3, \
        'two 4+ results add two Corruption (1 -> 3), got %d' \
        % st.fellowship.corruption


@test
def test_shadow_heal_twin_morgul_wound_adds():
    st = setup_game()
    st.fellowship.corruption = 1
    st.fellowship.hidden = False
    r = play(st, 'Morgul Wound')
    assert r['success'], r
    assert st.fellowship.corruption == 3, \
        'Morgul Wound adds 2 at Corruption <= 3 (1 -> 3), got %d' \
        % st.fellowship.corruption


# ---------------------------------------------------------------------------
# 2b. TRAILING "Then, ..." LEGS -- a leg printed after the card's main
#     imperative must survive EVERY path that performs that imperative.
# ---------------------------------------------------------------------------

def separate_to_board(st, name, region):
    """Take a Companion out of the Fellowship and stand him in `region`
    (the board form the 'if Gimli or Legolas ARE in ...' trigger reads).
    """
    comp = next(c for c in st.fellowship.companions if c.name == name)
    st.fellowship.companions.remove(comp)
    if st.fellowship.guide is comp:
        st.fellowship.guide = st.fellowship.companions[0]
    a = army(st, region)
    assert a is not None, 'no army at %r' % region
    a.leaders.append(comp)
    return comp


@test
def test_there_and_back_again_trigger_survives_a_mordor_separation():
    """'Separate one Companion or one group of Companions ... Then, if
    Gimli or Legolas ARE in Dale, Erebor or the Woodland Realm, activate
    the Dwarven and North Nations and advance the Dwarven, Elven, and
    North Nations one step each.'

    The trigger is a BOARD test taken after the separation -- it does
    not depend on where the separated group went.  A Mordor-Track
    separation removes the companions from play (almanac QA 2026-07-11
    cluster B) but it IS a separation, so the trailing leg must still
    fire.  The sibling I Will Go Alone already applies its trailing
    heal on exactly that path.
    """
    st = setup_game()
    separate_to_board(st, 'Gimli', 'Erebor')
    st.fellowship.mordor_entered = True
    before = {n: st.political.boxes[n]
              for n in (Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH)}
    r = play(st, 'There and Back Again',
             {'companions': ['Meriadoc'],
              'destination': '__remove_from_play__'})
    assert r['success'], r
    assert not any(c.name == 'Meriadoc'
                   for c in st.fellowship.companions), \
        'the Mordor-Track separation happened'
    assert st.political.active[Nation.DWARVES], \
        'the trigger activates the Dwarven Nation'
    assert st.political.active[Nation.THE_NORTH], \
        'the trigger activates the North Nation'
    for n in (Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH):
        assert st.political.boxes[n] == before[n] + 1, \
            '%s advances one step (%d -> %d)' \
            % (n.value, before[n], st.political.boxes[n])


@test
def test_there_and_back_again_trigger_still_fires_on_the_board_path():
    """Control: the same trailing leg on the ordinary separation path."""
    st = setup_game()
    separate_to_board(st, 'Legolas', 'Woodland Realm')
    before = {n: st.political.boxes[n]
              for n in (Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH)}
    r = play(st, 'There and Back Again',
             {'companions': ['Meriadoc'], 'destination': 'Rivendell'})
    assert r['success'], r
    for n in (Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH):
        assert st.political.boxes[n] == before[n] + 1, \
            '%s advances one step on the board path' % n.value


@test
def test_there_and_back_again_no_trigger_without_gimli_or_legolas():
    """Sensitivity control: with neither Gimli nor Legolas standing in a
    trigger region the political leg must NOT fire (guards the two
    tests above against a trigger that fires unconditionally)."""
    st = setup_game()
    st.fellowship.mordor_entered = True
    before = {n: st.political.boxes[n]
              for n in (Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH)}
    r = play(st, 'There and Back Again',
             {'companions': ['Meriadoc'],
              'destination': '__remove_from_play__'})
    assert r['success'], r
    for n in (Nation.DWARVES, Nation.ELVES, Nation.THE_NORTH):
        assert st.political.boxes[n] == before[n], \
            ('%s must NOT advance with no Gimli/Legolas in a trigger '
             'region' % n.value)


@test
def test_i_will_go_alone_heal_survives_a_mordor_separation():
    """The sibling contract this card family shares (almanac QA
    2026-07-11 cluster B): a Mordor-Track separation still heals."""
    st = setup_game()
    st.fellowship.mordor_entered = True
    st.fellowship.corruption = 5
    r = play(st, 'I Will Go Alone',
             {'companions': ['Meriadoc'],
              'destination': '__remove_from_play__'})
    assert r['success'], r
    assert st.fellowship.corruption == 4, \
        'the trailing heal survives the Mordor-Track path (5 -> 4), ' \
        'got %d' % st.fellowship.corruption


# ---------------------------------------------------------------------------
# 3. TABLE-CARD SEATING -- every FP table card takes a typed Board slot,
#    exactly as the Shadow ones do
# ---------------------------------------------------------------------------

FP_TABLE_CARDS = [
    ('Mithril Coat and Sting', PCT.MITHRIL_COAT),
    ('Axe and Bow', PCT.AXE_AND_BOW),
    ('Horn of Gondor', PCT.HORN_OF_GONDOR),
    ("Wizard's Staff", PCT.WIZARD_STAFF),
    ('The Last Battle', PCT.LAST_BATTLE),
    ('A Power too Great', PCT.POWER_TOO_GREAT),
    ('The Power of Tom Bombadil', PCT.TOM_BOMBADIL),
]

SH_TABLE_CARDS = [
    ('Worn with Sorrow and Toil', PCT.WORN_WITH_SORROW),
    ('Flocks of Crebain', PCT.FLOCKS_OF_CREBAIN),
    ('Balrog of Moria', PCT.BALROG_OF_MORIA),
    ('The Palantir of Orthanc', PCT.PALANTIR_OF_ORTHANC),
    ('Wormtongue', PCT.WORMTONGUE),
    ("Denethor's Folly", PCT.DENETHORS_FOLLY),
    ('Threats and Promises', PCT.THREATS_AND_PROMISES),
]


def _seat_via_handler(name):
    """Drive the registered handler directly (the play requirements of
    several table cards need board setups irrelevant to seating)."""
    st = setup_game()
    cid = cid_for(name)
    card = get_card(cid)
    h = ee.CARD_EFFECT_HANDLERS.get(card.card_name.lower())
    assert h is not None, 'no handler registered for %r' % name
    h(st, card, {})
    return st, cid


@test
def test_every_fp_table_card_takes_a_typed_slot():
    fails = []
    for name, ptype in FP_TABLE_CARDS:
        st, cid = _seat_via_handler(name)
        seat = on_table(st, cid)
        if seat is None:
            fails.append('%s: never seated' % name)
        elif seat.card_type is not ptype:
            fails.append('%s: seated untyped/wrong (%r)'
                         % (name, seat.card_type))
        elif seat.side is not Side.FREE_PEOPLES:
            fails.append('%s: seated on the wrong side' % name)
    assert not fails, 'FP table cards not seated: %s' % fails


@test
def test_every_shadow_table_card_takes_a_typed_slot():
    fails = []
    for name, ptype in SH_TABLE_CARDS:
        st, cid = _seat_via_handler(name)
        seat = on_table(st, cid)
        if seat is None:
            fails.append('%s: never seated' % name)
        elif seat.card_type is not ptype:
            fails.append('%s: seated untyped/wrong (%r)'
                         % (name, seat.card_type))
        elif seat.side is not Side.SHADOW:
            fails.append('%s: seated on the wrong side' % name)
    assert not fails, 'Shadow table cards not seated: %s' % fails


@test
def test_mithril_coat_seats_through_play_event_card():
    """fpgame9 T7: the FP table card must reach the table through the
    full play path -- seated, out of hand, and NOT in the discard."""
    st = setup_game()
    cid = cid_for('Mithril Coat and Sting')
    r = play(st, 'Mithril Coat and Sting')
    assert r['success'], r
    assert on_table(st, cid) is not None, \
        'Mithril Coat and Sting is seated on the table'
    assert cid not in st.hands[Side.FREE_PEOPLES].hand, \
        'the played card left the hand'
    assert not in_face_up(st, cid), \
        'a table card is NOT filed to the face-up discard when played'


@test
def test_worn_with_sorrow_seats_through_play_event_card():
    """The Shadow twin of the same seating path (control)."""
    st = setup_game()
    cid = cid_for('Worn with Sorrow and Toil')
    r = play(st, 'Worn with Sorrow and Toil')
    assert r['success'], r
    assert on_table(st, cid) is not None, \
        'Worn with Sorrow and Toil is seated on the table'
    assert not in_face_up(st, cid), \
        'a table card is NOT filed to the face-up discard when played'


@test
def test_every_table_card_is_recognised_by_the_discard_predicate():
    """play_event_card suppresses the face-up discard with the literal
    test `"play on the table" in card.event_effect.lower()`.  cards.py
    re-parses the CSV's event_req/event_effect split, so a table card
    whose prefix ends up outside `event_effect` would be seated AND
    filed to the discard (a duplicated physical card).  Pin the
    predicate on all 14 table cards, both sides."""
    fails = []
    for name, _ in FP_TABLE_CARDS + SH_TABLE_CARDS:
        card = get_card(cid_for(name))
        if 'play on the table' not in (card.event_effect or '').lower():
            fails.append('%s: event_effect=%r'
                         % (name, (card.event_effect or '')[:60]))
    assert not fails, \
        'table cards invisible to the discard-suppression predicate: %s' \
        % fails


@test
def test_table_card_handlers_seat_with_choices_present():
    """The interposed-decision handlers (Mithril redraw / Wizard's Staff
    prevent / Crebain discard) branch on a choices KEY.  A play carrying
    unrelated choices -- every live play does, e.g. {'_from_pending':
    True} -- must still take the table seat."""
    fails = []
    for name in ('Mithril Coat and Sting', "Wizard's Staff",
                 'Flocks of Crebain'):
        st = setup_game()
        cid = cid_for(name)
        card = get_card(cid)
        h = ee.CARD_EFFECT_HANDLERS[card.card_name.lower()]
        h(st, card, {'_from_pending': True, 'region': 'Hollin'})
        if on_table(st, cid) is None:
            fails.append(name)
    assert not fails, \
        'table cards dropped their seat when choices were present: %s' \
        % fails


# ---------------------------------------------------------------------------
# 4. The FP recruit class as a whole (twin sweep): every FP muster card
#    whose printed text places a piece actually places one.
# ---------------------------------------------------------------------------

FP_MUSTER_EXPECT = [
    # (card name, region, nation, expects a leader too)
    ("Celeborn's Galadhrim", 'Lorien', Nation.ELVES, False),
    ('Guards of the Citadel', 'Minas Tirith', Nation.GONDOR, True),
    ('Grimbeorn the Old Son of Beorn', 'Carrock', Nation.THE_NORTH, True),
    ('Imrahil of Dol Amroth', 'Dol Amroth', Nation.GONDOR, True),
    ("King Brand's Men", 'Dale', Nation.THE_NORTH, False),
    ('Kindred of Glorfindel', 'Rivendell', Nation.ELVES, False),
    ("Dain Ironfoot's Guard", 'Erebor', Nation.DWARVES, True),
    ("Thranduil's Archers", 'Woodland Realm', Nation.ELVES, False),
]


@test
def test_every_fp_fixed_region_muster_card_places_its_pieces():
    fails = []
    for name, region, nation, wants_leader in FP_MUSTER_EXPECT:
        st = setup_game()
        bu = n_units(st, region, nation)
        bl = n_leaders(st, region, nation)
        r = play(st, name)
        if not r.get('success'):
            fails.append('%s: play refused (%s)' % (name, r.get('message')))
            continue
        if n_units(st, region, nation) <= bu:
            fails.append('%s: no unit placed in %s' % (name, region))
        if wants_leader and n_leaders(st, region, nation) <= bl:
            fails.append('%s: no Leader placed in %s' % (name, region))
    assert not fails, 'FP muster legs missing: %s' % fails


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FP EVENT-CARD LEG BATTERY (BL-90)")
    print("=" * 70)
    npass = nfail = 0
    for fn in TESTS:
        try:
            fn()
            print("  [PASS] %s" % fn.__name__)
            npass += 1
        except Exception:
            print("  [FAIL] %s" % fn.__name__)
            for line in traceback.format_exc().splitlines()[-6:]:
                print("         " + line)
            nfail += 1
    print("-" * 70)
    print("TOTAL %d  |  PASS %d  FAIL %d" % (npass + nfail, npass, nfail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
