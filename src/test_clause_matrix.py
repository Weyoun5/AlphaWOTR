#!/usr/bin/env python3
"""Rulebook-clause TRACEABILITY suite (agent P2, rules-assurance program).

Each testable clause of the War of the Ring (2nd ed) rules gets a POSITIVE
test (a legal thing is offered / succeeds) and a NEGATIVE test (an illegal
thing is NOT offered / is rejected).  The point of this file is the
NEGATIVE half: our historical suites verified the engine DOES legal things
but almost never that it REFUSES illegal ones.

Authority order for "what the rules are": (1) data/CURRENT_RULES.md,
(2) data/rules_amendments.md, (3) the official 2nd-ed rulebook.  Where a
house ruling contradicts the rulebook, the house ruling WINS.

The runner never crashes on a failing clause -- failures are DATA, recorded
and printed in an ASCII summary; the process exits 1 when any clause FAILs
or ERRORs (a runner that prints a failure and exits 0 is a defect).  A FAIL is
either (a) an engine rule error -> see data/rules_flags_p2.md, or (b) an
intentional flag-driver test that encodes the CORRECT rule so its failure
documents the engine's divergence.

Run:  python test_clause_matrix.py
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai
import game_state as gs
import game_logic
import combat as cb
import board
from board import sanitize
from cards import get_card, get_cards

Side = gs.Side
Nation = gs.Nation
Face = gs.ActionDieFace
Kind = gs.UnitKind


# ---------------------------------------------------------------------------
# Test registry + runner
# ---------------------------------------------------------------------------

REGISTRY = []            # list of (cid, chapter, desc, fn, page)


class Skip(Exception):
    pass


def skip(msg):
    raise Skip(msg)


# ---------------------------------------------------------------------------
# Rulebook page anchors  (War of the Ring, 2nd edition rulebook)
# ---------------------------------------------------------------------------
# One anchor per clause id.  Values:
#   int        - a single rulebook page (the page the clause verifies)
#   (lo, hi)   - an inclusive page range
#   'house'    - a house ruling from data/CURRENT_RULES.md / rules_amendments.md
#                that has NO rulebook page (the house ruling WINS over the book)
#   None       - a genuine rulebook rule whose EXACT page could not be pinned
#                with confidence (honest gap; never a wrong guess)
#
# Pages were taken from the Authority column of data/rules_clause_matrix.md,
# which records "rb p.N" where the clause maps to a rulebook page and "CR.n"
# where it maps to a CURRENT_RULES house ruling.  Clauses whose Authority is
# only CR/amendment (no "rb p.N") are 'house'.  A handful of clauses cite the
# rulebook with no page ("rb", "rb card text", "rb/almanac"); where the page
# is not confidently knowable they are None with the reason noted below.
CLAUSE_PAGES = {
    # Ch.1 - game turn sequence & action dice (rb p.15-17, p.19-26)
    'CL-001': 15, 'CL-002': 15, 'CL-003': 15,
    'CL-004': 25, 'CL-005': 20, 'CL-006': 24,
    'CL-007': 26, 'CL-008': 26,
    'CL-009': 'house',                 # CR.1: WoW-as-any-family + no burn-a-die 'skip'
    'CL-010': 17, 'CL-011': 15, 'CL-012': 22, 'CL-013': 17,
    'CL-014': 'house', 'CL-015': 'house', 'CL-016': 'house',   # CR.1 eye-allocation rulings
    'CL-017': 'house', 'CL-018': 'house',                       # CR.2 GtW/crowned-Aragorn bonus die
    'CL-019': 19, 'CL-020': 17, 'CL-021': 17,
    # Ch.2 - Fellowship & companions (rb p.36-38; much is CR.2 house)
    'CL-030': 38, 'CL-031': 38,
    'CL-032': 38,                      # rb (guide/hide rules); anchored with CL-031
    'CL-033': 20, 'CL-034': 'house', 'CL-035': 36, 'CL-036': 36,
    'CL-037': 'house', 'CL-038': 'house', 'CL-039': 20,
    'CL-040': 'house', 'CL-041': 'house', 'CL-042': 'house',
    'CL-043': 'house', 'CL-044': 'house',
    # Ch.3 - Hunt (rb p.39-40)
    'CL-050': 39, 'CL-051': 40, 'CL-052': 40, 'CL-053': 39,
    'CL-054': 'house',                 # CR.1 Mordor-gate ruling
    # Ch.4 - Armies & battle (rb p.30-33)
    'CL-060': 33, 'CL-061': 33, 'CL-062': 33,
    'CL-063': 31, 'CL-064': 31, 'CL-065': 30, 'CL-066': 30,
    'CL-067': 32, 'CL-068': 31,
    'CL-069': 'house', 'CL-070': 'house',        # CR.4 siege/garrison rulings
    'CL-071': 30,
    'CL-072': None,                    # rb (Saruman elite-leadership); page not pinned
    'CL-073': None,                    # rb (Saruman leaderless-as-led); page not pinned
    'CL-074': None,                    # rb/almanac (repeat-attack clarification); page not pinned
    'CL-075': 20, 'CL-076': 19, 'CL-077': 'house',   # CL-077 CR.4 besieger-move ruling
    # Ch.5 - Politics & muster (rb p.24, p.26)
    'CL-080': 'house',                 # CR.6 / session-21 activation ruling
    'CL-081': 24, 'CL-082': 24, 'CL-083': 36,
    'CL-084': 26, 'CL-085': 26, 'CL-086': 26, 'CL-087': 20,
    'CL-088': 'house', 'CL-089': 'house', 'CL-090': 'house',   # CR.3 minion-entry rulings
    'CL-091': None,                    # rb (Saruman->Orthanc muster); page not pinned
    'CL-092': 'house', 'CL-093': 'house', 'CL-094': 'house',
    'CL-095': 24,
    # Ch.6 - Event & table cards (mostly CR.5 house; CL-105 rb p.18)
    'CL-100': 'house', 'CL-101': 'house', 'CL-102': 'house',
    'CL-103': 'house', 'CL-104': 'house', 'CL-105': 18,
    'CL-106': 'house',                 # USER ruling 2026-07-20 (card text, no rulebook page)
    'CL-107': 'house',                 # card text; deferred (SKIP)
    # Ch.7 - Characters (rb; several abilities not page-pinnable)
    'CL-110': None,                    # rb (Saruman immobile); page not pinned
    'CL-111': 19,
    'CL-112': None,                    # rb (Witch-king stat block); page not pinned
    'CL-113': None,                    # rb (Mouth stat block); page not pinned
    'CL-114': 'house',                 # CR.3 lone-Mouth movement ruling
    # Ch.8 - Victory conditions & timing (rb p.8, p.17)
    'CL-120': 8, 'CL-121': 8, 'CL-122': 8, 'CL-123': 8,
    'CL-124': 17, 'CL-125': 8,
    # Ch.9 - Fellowship reveal placement (rb p.38; CL-131 CR.1 house)
    'CL-130': 38, 'CL-131': 'house',
    # ---- Rules-assurance blind-spot sweep (2026-07-27) --------------------
    # Politics ch.: "ENTERING WAR" / non-belligerent restrictions.  Marker
    # p.34 -> printed p.35 (header "Chapter VII: The Politics ... 35").
    'CL-300': 35, 'CL-301': 35, 'CL-302': 35, 'CL-303': 35, 'CL-304': 35,
    # Armies ch.: move-through Settlement capture + 10-unit stacking.
    # Marker p.26-27 -> printed p.27-28.
    'CL-310': 28, 'CL-311': (27, 28),
    # Fellowship ch.: declare/reveal base mechanic.  Marker p.37 -> printed
    # p.38 (header "... The Fellowship of the Ring 38").
    'CL-320': 38, 'CL-321': 38, 'CL-322': 38, 'CL-323': 38,
    # Winning the Game ch.  Marker p.43 -> printed p.44 (header "Chapter X:
    # Winning the Game 44").
    'CL-330': 44, 'CL-331': 44, 'CL-332': 44, 'CL-333': 44,
    # Setting Up the Game ch.  STEPs on printed p.14; Army Setup diagram on
    # printed p.16-17; standard Hunt-tile values p.39.
    'CL-340': (16, 17), 'CL-341': 14, 'CL-342': 14, 'CL-343': 39,
}


def clause(cid, chapter, desc, page=None):
    """Register a clause test.

    `page` anchors the clause to WOTR 2nd-ed rulebook page(s):
      * an int             - a single page
      * an inclusive tuple - page=(28, 29)
      * 'house'            - a CURRENT_RULES/amendment house ruling (no page)
      * None               - rulebook rule whose exact page is unknown
    Backward compatible: page=None is allowed.  When no page is given inline
    the anchor is looked up from CLAUSE_PAGES (all anchors live in one
    auditable table); an explicit page= on the decorator wins.
    """
    if page is None:
        page = CLAUSE_PAGES.get(cid)
    def deco(fn):
        REGISTRY.append((cid, chapter, desc, fn, page))
        return fn
    return deco


def _page_covers(page, n):
    """True if a clause's page anchor covers rulebook page `n`.
    'house' and None never cover a page (they are not rulebook-anchored)."""
    if page is None or page == 'house':
        return False
    if isinstance(page, tuple):
        return page[0] <= n <= page[1]
    return page == n


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def setup_game():
    b = board.load_board()
    state = gs.setup_initial_game(b)
    game_logic.initialize_decks(state)
    return state


def set_dice(state, side, faces):
    state.dice[side] = []
    for face in faces:
        d = gs.ActionDie(faces=[face], side=side)
        d.current = face
        state.dice[side].append(d)


def army_at(state, region_key):
    return game_logic.get_or_create_army(state, region_key)


def place(state, region_key, nation, regulars=0, elites=0, leaders=0, nazgul=0,
          clear=False):
    """Build/extend an army ONLY via game_logic.get_or_create_army."""
    a = army_at(state, region_key)
    if clear:
        a.units.clear()
        a.leaders.clear()
    for _ in range(regulars):
        a.units.append(gs.Unit(nation, Kind.REGULAR))
    for _ in range(elites):
        a.units.append(gs.Unit(nation, Kind.ELITE))
    for _ in range(leaders):
        a.leaders.append(gs.Leader(nation))
    for _ in range(nazgul):
        a.leaders.append(gs.Nazgul(nation))
    return a


def types_of(actions):
    return set(a.action_type for a in actions)


def _find_sh_army_card():
    for cid, c in get_cards().items():
        if cid.startswith('SHADOW') and getattr(c, 'card_type', None) == 'Army':
            return cid
    return None


def _find_sh_character_card():
    for cid, c in get_cards().items():
        if cid.startswith('SHADOW') and '_Character_' in cid:
            return cid
    return None


def _bfs_dist(state, start, goal):
    from collections import deque
    start = sanitize(start)
    goal = sanitize(goal)
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        r, d = q.popleft()
        for adj in state.board.adjacent(r):
            a = sanitize(adj)
            if a == goal:
                return d + 1
            if a not in seen:
                seen.add(a)
                q.append((a, d + 1))
    return 999


def _strip_nation(state, nation):
    """Remove every unit of `nation` from every army (leaders untouched).
    Used to force a known reinforcement-pool availability for that nation."""
    for _k, a in state.armies.items():
        a.units[:] = [u for u in a.units if u.nation != nation]


def _set_reg_pool_free(state, nation, free):
    """Leave EXACTLY `free` Regulars available in `nation`'s
    reinforcements, with an EMPTY casualty box. 2026-07-20d: the old
    approach parked the difference as casualty-box regulars — the box
    is now a LEGAL downgrade source (USER ruling), so the negative
    pool-exhausted tests need the difference parked as DEPLOYED units
    (remote region) instead."""
    _strip_nation(state, nation)
    state.fp_casualties = {}
    mx = ai.NATION_MAX_FIGURES[nation]['regular']
    if mx - free > 0:
        park = game_logic.get_or_create_army(state, 'forlindon')
        for _ in range(mx - free):
            park.units.append(gs.Unit(nation, Kind.REGULAR))


# ===========================================================================
# CHAPTER 1 - Game turn sequence + action dice
# ===========================================================================

@clause("CL-001", "1-dice", "FP starts with 4 action dice; SH with 7")
def cl_001():
    st = setup_game()
    assert len(st.dice[Side.FREE_PEOPLES]) == 4, len(st.dice[Side.FREE_PEOPLES])
    assert len(st.dice[Side.SHADOW]) == 7, len(st.dice[Side.SHADOW])


@clause("CL-002", "1-dice", "FP die faces include Will of the West, exclude Eye (neg)")
def cl_002():
    assert Face.WILL_OF_WEST in gs.FP_FACES
    assert Face.EYE not in gs.FP_FACES          # negative
    assert Face.ARMY not in gs.FP_FACES          # FP has no plain Army face


@clause("CL-003", "1-dice", "SH die faces include Eye, exclude Will of the West (neg)")
def cl_003():
    assert Face.EYE in gs.SH_FACES
    assert Face.WILL_OF_WEST not in gs.SH_FACES  # negative


@clause("CL-284", "1-dice", "Elven Ring targets: FP excludes Will of the West; Shadow includes Eye, excludes Will, cannot change FROM Eye (rulebook p.21)", page=21)
def cl_284():
    # FP ring may change an unused die to any FP face EXCEPT Will of the West
    # (rulebook _rulebook.txt:1712-1714; USER ruling 2026-07-23, amendment 1.215)
    st = setup_game()
    st.fp_elven_rings = 1
    st.fp_ring_used_this_turn = False
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER, Face.ARMY])
    fp_ring = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
               if a.action_type == 'elven_ring']
    assert fp_ring, "FP with a ring + unused dice must offer elven_ring changes"
    fp_new = {str(a.params['new_face']) for a in fp_ring}
    assert Face.WILL_OF_WEST.value not in fp_new, ("FP ring offered Will of the West", fp_new)
    assert Face.EYE.value not in fp_new, ("FP ring offered Eye", fp_new)
    assert Face.MUSTER.value in fp_new and Face.EVENT.value in fp_new, \
        ("FP ring must offer the normal faces", fp_new)

    # Shadow ring MAY change a die TO an Eye, NEVER to Will of the West
    # (rulebook 1716-1719; Will is not a Shadow face at all)
    st2 = setup_game()
    st2.sh_elven_rings = 1
    st2.sh_ring_used_this_turn = False
    set_dice(st2, Side.SHADOW, [Face.CHARACTER])
    sh_ring = [a for a in ai.get_legal_actions(st2, Side.SHADOW)
               if a.action_type == 'elven_ring']
    assert sh_ring, "Shadow with a ring + unused die must offer elven_ring changes"
    sh_new = {str(a.params['new_face']) for a in sh_ring}
    assert Face.EYE.value in sh_new, ("Shadow ring must offer Eye", sh_new)
    assert Face.WILL_OF_WEST.value not in sh_new, ("Shadow ring offered Will of the West", sh_new)

    # Shadow ring cannot change a die ALREADY showing an Eye (rulebook 1721-1723)
    st3 = setup_game()
    st3.sh_elven_rings = 1
    st3.sh_ring_used_this_turn = False
    set_dice(st3, Side.SHADOW, [Face.EYE])
    sh_eye_ring = [a for a in ai.get_legal_actions(st3, Side.SHADOW)
                   if a.action_type == 'elven_ring']
    assert not sh_eye_ring, ("Shadow ring changed a die already showing Eye", sh_eye_ring)


@clause("CL-004", "1-dice", "Army die: 2 moves (first sets pending) OR 1 attack (no pending)")
def cl_004():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.ARMY])
    moves = [a for a in ai.get_legal_actions(st, Side.SHADOW)
             if a.action_type == 'army_move']
    assert moves, "Army die must offer moves"
    ai._execute_action(st, moves[0])
    assert st.pending_army_move_die is not None, "first army move sets pending"
    # After the first move only army_move / pass remain
    t2 = types_of(ai.get_legal_actions(st, Side.SHADOW))
    assert t2 <= {'army_move', 'pass'}, t2


@clause("CL-005", "1-dice", "Character die requires a leader for army move (neg: leaderless not offered)")
def cl_005():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.CHARACTER])
    moves = [a for a in ai.get_legal_actions(st, Side.SHADOW)
             if a.action_type == 'army_move']
    for a in moves:
        army = next((ar for k, ar in st.armies.items()
                     if sanitize(k) == sanitize(a.params['from'])), None)
        assert army and army.leaders, f"char-die move from leaderless {a.params['from']}"


@clause("CL-006", "1-dice", "Character die does NOT advance politics; Muster die does")
def cl_006():
    st = setup_game()
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    assert not [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                if a.action_type == 'advance_political'], "char die must not advance politics"
    st2 = setup_game()
    set_dice(st2, Side.FREE_PEOPLES, [Face.MUSTER])
    assert [a for a in ai.get_legal_actions(st2, Side.FREE_PEOPLES)
            if a.action_type == 'advance_political'], "muster die must advance politics"


@clause("CL-007", "1-dice", "Muster die: two recruits, second in a DIFFERENT settlement")
def cl_007():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    recs = [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'muster_recruit']
    assert recs, "muster die offers recruits"
    first = recs[0]
    first_region = first.params['region']
    ai._execute_action(st, first)
    assert st.pending_muster_die is not None
    recs2 = [a for a in ai.get_legal_actions(st, Side.SHADOW)
             if a.action_type == 'muster_recruit']
    same = [a for a in recs2 if sanitize(a.params['region']) == sanitize(first_region)]
    assert not same, "second recruit must not reuse the first settlement (neg)"


@clause("CL-008", "1-dice", "Muster die elite placement consumes the whole die (no pending)")
def cl_008():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    elites = [a for a in ai.get_legal_actions(st, Side.SHADOW)
              if a.action_type == 'muster_elite']
    assert elites, "muster die offers elite placement"
    ai._execute_action(st, elites[0])
    assert st.pending_muster_die is None, "elite placement must NOT open a second recruit"
    assert all(d.used for d in st.dice[Side.SHADOW])


@clause("CL-009", "1-dice", "Will of the West acts as any family; 'skip' never generated (neg)")
def cl_009():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    set_dice(st, Side.FREE_PEOPLES, [Face.WILL_OF_WEST])
    t = types_of(ai.get_legal_actions(st, Side.FREE_PEOPLES))
    for need in ('army_move', 'fellowship_move', 'advance_political'):
        assert need in t, f"WoW should allow {need}: {t}"
    assert 'skip' not in t, "'skip' (burn a die) must never be generated"


@clause("CL-010", "1-dice", "Eye face maps to NO spendable action family (neg)")
def cl_010():
    assert ai._die_faces_for_action(Face.EYE) == set()
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.EYE])
    t = types_of(ai.get_legal_actions(st, Side.SHADOW))
    assert not (t & {'army_move', 'army_attack', 'muster_recruit',
                     'advance_political'}), f"Eye die must spend nothing: {t}"


@clause("CL-011", "1-dice", "Muster/Army combined face allows BOTH muster and army families")
def cl_011():
    fams = ai._die_faces_for_action(Face.MUSTER_ARMY)
    assert 'muster' in fams and 'army' in fams, fams
    assert 'army' not in ai._die_faces_for_action(Face.MUSTER)   # plain Muster != army (neg)


@clause("CL-012", "1-dice", "Event face plays any card type; Character die only Character (per icon)")
def cl_012():
    assert ai._die_faces_for_action(Face.EVENT) == {'event'}
    assert ai._die_faces_for_action(Face.CHARACTER) == {'character'}


# --- Eye allocation (Hunt Box) ---

@clause("CL-013", "1-dice", "Eye allocation cap = number of Companions in the Fellowship")
def cl_013():
    st = setup_game()
    # Trim to 2 companions -> cap must be 2 (Ring-bearers excluded)
    st.fellowship.companions = st.fellowship.companions[:2]
    st._eye_allocation_phase = True
    acts = ai.get_legal_actions(st, Side.SHADOW)
    counts = sorted(a.params['count'] for a in acts if a.action_type == 'eye_allocate')
    assert counts, "eye phase must offer allocations"
    assert max(counts) == 2, f"cap must equal #companions=2, got {counts}"
    assert 3 not in counts, "must never offer more eyes than companions (neg)"


@clause("CL-014", "1-dice", "At least 1 eye is permitted even with zero Companions")
def cl_014():
    st = setup_game()
    st.fellowship.companions = []
    st._eye_allocation_phase = True
    counts = sorted(a.params['count'] for a in ai.get_legal_actions(st, Side.SHADOW)
                    if a.action_type == 'eye_allocate')
    assert 1 in counts, f"min 1 eye always permitted: {counts}"


@clause("CL-015", "1-dice", "Eye cap has NO Mordor exception (cap still = #companions on the track)")
def cl_015():
    st = setup_game()
    st.fellowship.companions = st.fellowship.companions[:2]
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 2
    st._eye_allocation_phase = True
    counts = [a.params['count'] for a in ai.get_legal_actions(st, Side.SHADOW)
              if a.action_type == 'eye_allocate']
    assert max(counts) == 2, f"Mordor must NOT lift the eye cap: {counts}"


@clause("CL-016", "1-dice", "One eye is REQUIRED if FP retrieved a hunt-box die last turn")
def cl_016():
    st = setup_game()
    st.fellowship.companions = st.fellowship.companions[:3]
    st._fp_used_hunt_box_last_turn = True
    st._eye_allocation_phase = True
    counts = sorted(a.params['count'] for a in ai.get_legal_actions(st, Side.SHADOW)
                    if a.action_type == 'eye_allocate')
    assert 0 not in counts, f"0 eyes must be forbidden after FP retrieval: {counts}"
    assert min(counts) == 1, counts


# --- Bonus dice from special characters ---

@clause("CL-017", "1-dice", "Gandalf the White grants +1 FP action die")
def cl_017():
    st = setup_game()
    st.gandalf_the_white.in_play = True
    # A14 (2026-07-26): the die requires the character ALIVE on the map
    # (marker p.18 "lose the additional die if ... eliminated"), so the
    # figure must exist — flag-only states cannot occur in real play.
    import game_state as _gs17
    game_logic.get_or_create_army(st, 'fangorn').leaders.append(
        _gs17.Companion(name='Gandalf the White', level=3, leadership=1))
    game_logic.calculate_bonus_dice(st)
    assert len(st.dice[Side.FREE_PEOPLES]) == 5, len(st.dice[Side.FREE_PEOPLES])


@clause("CL-018", "1-dice", "Crowned Aragorn grants +1 FP action die")
def cl_018():
    st = setup_game()
    st.aragorn_crowned.crowned = True
    # A14 (2026-07-26): see CL-017 — the crowned figure must be on the map.
    import game_state as _gs18
    game_logic.get_or_create_army(st, 'minas tirith').leaders.append(
        _gs18.Companion(name='Aragorn', level=3, leadership=2))
    game_logic.calculate_bonus_dice(st)
    assert len(st.dice[Side.FREE_PEOPLES]) == 5, len(st.dice[Side.FREE_PEOPLES])


@clause("CL-019", "1-dice", "Each Shadow minion in play grants +1 SH die (max 10)")
def cl_019():
    st = setup_game()
    st.witch_king.in_play = True
    st.saruman.in_play = True
    st.mouth_of_sauron.in_play = True
    game_logic.calculate_bonus_dice(st)
    assert len(st.dice[Side.SHADOW]) == 10, len(st.dice[Side.SHADOW])


@clause("CL-020", "1-dice", "next_turn resets per-turn one-shot flags and eye count")
def cl_020():
    st = setup_game()
    st.fp_ring_used_this_turn = True
    st.sh_mouth_muster_used_this_turn = True
    st.fp_token_used_this_turn = True
    st.eyes_allocated = 3
    game_logic.next_turn(st)
    assert not st.fp_ring_used_this_turn
    assert not st.sh_mouth_muster_used_this_turn
    assert not st.fp_token_used_this_turn
    assert st.eyes_allocated == 0


@clause("CL-021", "1-dice", "A rolled Eye die is auto-consumed into the Hunt Box")
def cl_021():
    st = setup_game()
    st.dice[Side.SHADOW] = [gs.ActionDie(faces=[Face.EYE], side=Side.SHADOW)]
    game_logic.roll_action_dice(st, pool=Side.SHADOW)
    d = st.dice[Side.SHADOW][0]
    assert d.used and d in st.hunt_box, "an Eye die must move to the Hunt Box on roll"


# ===========================================================================
# CHAPTER 2 - Fellowship & companions
# ===========================================================================

@clause("CL-030", "2-fellow", "Fellowship moves only while hidden (neg: revealed offers no move)")
def cl_030():
    st = setup_game()
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    assert [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'fellowship_move'], "hidden fellowship may move"
    st.fellowship.hidden = False
    assert not [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                if a.action_type == 'fellowship_move'], "revealed fellowship must NOT move (neg)"


@clause("CL-031", "2-fellow", "Hide offered only when revealed (neg: hidden offers no hide)")
def cl_031():
    st = setup_game()
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    assert not [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                if a.action_type == 'fellowship_hide'], "hidden fellowship has nothing to hide (neg)"
    st.fellowship.hidden = False
    assert [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'fellowship_hide'], "revealed fellowship may hide"


@clause("CL-032", "2-fellow", "Strider (guide) hides with ANY die face")
def cl_032():
    st = setup_game()
    strider = next(c for c in st.fellowship.companions if c.name == 'Strider')
    st.fellowship.guide = strider
    st.fellowship.hidden = False
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])   # non-character die
    hides = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a.action_type == 'fellowship_hide']
    assert hides, "Strider guide must hide with a Muster die"


@clause("CL-033", "2-fellow", "Non-Strider guide cannot hide with a non-Character die (neg)")
def cl_033():
    st = setup_game()
    gandalf = next(c for c in st.fellowship.companions if c.name == 'Gandalf the Grey')
    st.fellowship.guide = gandalf
    st.fellowship.hidden = False
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    hides = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a.action_type == 'fellowship_hide']
    assert not hides, "only a Character/WoW die hides without Strider (neg)"


@clause("CL-034", "2-fellow", "Only a HIDDEN fellowship may declare (neg: revealed cannot)")
def cl_034():
    st = setup_game()
    st.fellowship.progress = 3
    assert ai._get_declare_actions(st, allow_in_place=True), "hidden may declare"
    st.fellowship.hidden = False
    assert not ai._get_declare_actions(st, allow_in_place=True), "revealed must NOT declare (neg)"


@clause("CL-035", "2-fellow", "Declaration reaches up to 'progress' regions, any direction")
def cl_035():
    st = setup_game()
    st.fellowship.progress = 1
    cur = sanitize(game_logic.fellowship_region(st))
    dests = {sanitize(a.params['region']) for a in ai._get_declare_actions(st, allow_in_place=True)}
    neigh = {sanitize(x) for x in st.board.adjacent(cur)}
    assert cur in dests, "declaring in place is legal"
    assert neigh <= dests, "all 1-step regions must be reachable at progress=1"


@clause("CL-036", "2-fellow", "Declaration blocked once on the Mordor track (neg)")
def cl_036():
    st = setup_game()
    st.fellowship.progress = 3
    st.fellowship.mordor_entered = True
    assert not ai._get_declare_actions(st, allow_in_place=True), "no declare inside Mordor (neg)"


@clause("CL-037", "2-fellow", "Companion separation IS an offered learned action")
def cl_037():
    st = setup_game()
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    assert [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'companion_separate'], "separation must be offered"


@clause("CL-038", "2-fellow", "Companion separation blocked on the Mordor track (neg)")
def cl_038():
    st = setup_game()
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 2
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    assert not [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
                if a.action_type == 'companion_separate'], "no separation in Mordor (neg)"
    raised = False
    try:
        game_logic.separate_companions(st, ['Gandalf the Grey'], 'Minas Tirith', die_idx=0)
    except RuntimeError as e:
        raised = 'Mordor' in str(e)
    assert raised, "separate_companions must reject a Mordor separation"


@clause("CL-039", "2-fellow", "Shadow never has Fellowship actions (neg)")
def cl_039():
    st = setup_game()
    set_dice(st, Side.SHADOW, [Face.CHARACTER])
    t = types_of(ai.get_legal_actions(st, Side.SHADOW))
    assert not (t & {'fellowship_move', 'fellowship_hide',
                     'fellowship_declare', 'companion_separate'}), t


@clause("CL-040", "2-fellow", "Crown Aragorn: Strider in unconquered MT/Pelargir/Dol Amroth")
def cl_040():
    st = setup_game()
    strider = gs.Companion(name='Strider', level=3, leadership=1, captain_of_the_west=True)
    place(st, 'minas tirith', Nation.GONDOR, regulars=1)
    army_at(st, 'minas tirith').leaders.append(strider)
    # remove Strider from the fellowship so the on-board copy is found
    st.fellowship.companions = [c for c in st.fellowship.companions if c.name != 'Strider']
    can, _, regions = game_logic.can_crown_aragorn(st)
    assert can and 'minas tirith' in [sanitize(r) for r in regions], (can, regions)


@clause("CL-041", "2-fellow", "Crown Aragorn refused where Strider is NOT in a crown city (neg)")
def cl_041():
    st = setup_game()
    strider = gs.Companion(name='Strider', level=3, leadership=1)
    place(st, 'bree', Nation.THE_NORTH, regulars=1)
    army_at(st, 'bree').leaders.append(strider)
    st.fellowship.companions = [c for c in st.fellowship.companions if c.name != 'Strider']
    can, _, _ = game_logic.can_crown_aragorn(st)
    assert not can, "crowning at Bree must be refused (neg)"


@clause("CL-042", "2-fellow", "Crowned Aragorn is level 3, leadership 2, Captain of the West")
def cl_042():
    st = setup_game()
    strider = gs.Companion(name='Strider', level=3, leadership=1, captain_of_the_west=True)
    place(st, 'dol amroth', Nation.GONDOR, regulars=1)
    army_at(st, 'dol amroth').leaders.append(strider)
    st.fellowship.companions = [c for c in st.fellowship.companions if c.name != 'Strider']
    set_dice(st, Side.FREE_PEOPLES, [Face.WILL_OF_WEST])
    game_logic.crown_aragorn(st, die_idx=0)
    ar = next(l for l in army_at(st, 'dol amroth').leaders
              if isinstance(l, gs.Companion) and l.name == 'Aragorn')
    assert ar.level == 3 and ar.leadership == 2 and ar.captain_of_the_west


@clause("CL-043", "2-fellow", "Gandalf the White carries NO Captain-of-the-West bonus")
def cl_043():
    st = setup_game()
    assert st.gandalf_the_white.captain_of_the_west is False


@clause("CL-044", "2-fellow", "A hunt-drawn Hobbit is PLACED off-board, except in Mordor where he dies")
def cl_044():
    st = setup_game()
    assert game_logic.hobbit_hunt_placement_applies(st, 'Meriadoc') is True
    st.fellowship.mordor_entered = True
    assert game_logic.hobbit_hunt_placement_applies(st, 'Meriadoc') is False, \
        "on the Mordor track a drawn Hobbit is eliminated, not placed (neg)"


# ===========================================================================
# CHAPTER 3 - Hunt pool, tiles, resolution
# ===========================================================================

@clause("CL-050", "3-hunt", "Initial hunt pool is the 16 standard tiles")
def cl_050():
    st = setup_game()
    pool = st.hunt_pool
    assert len(pool) == 16, len(pool)
    eyes = sum(1 for t in pool if t.is_eye)
    reveals = sum(1 for t in pool if t.reveal and not t.is_eye)
    assert eyes == 4, f"4 eye tiles, got {eyes}"
    assert reveals == 5, f"5 numbered reveal tiles, got {reveals}"


@clause("CL-051", "3-hunt", "Hunt threshold is 6; FP dice add fp_modifier, not hunt level")
def cl_051():
    st = setup_game()
    p = game_logic.get_hunt_params(st)
    assert p['threshold'] == 6 and p['fp_modifier'] == 0
    for _ in range(2):
        d = gs.ActionDie(faces=[Face.CHARACTER], side=Side.FREE_PEOPLES)
        d.current = Face.CHARACTER
        d.used = True
        st.hunt_box.append(d)
    p = game_logic.get_hunt_params(st)
    assert p['threshold'] == 6, "FP dice never change the threshold"
    assert p['fp_modifier'] == 2, p['fp_modifier']
    assert p['hunt_level'] == 0, "FP dice must NOT raise the hunt level (neg)"


@clause("CL-052", "3-hunt", "Hunt roll: natural 1 always misses")
def cl_052():
    random.seed(7)
    st = setup_game()
    hits = sum(game_logic._hunt_roll(st, 1, 1) for _ in range(600))
    # threshold 1 -> everything hits except natural 1s (~5/6)
    assert 420 < hits < 580, f"nat-1 miss expected ~500/600, got {hits}"


@clause("CL-053", "3-hunt", "Eye tiles ALWAYS reveal the Fellowship")
def cl_053():
    assert gs.HuntTile(is_eye=True).reveal is True
    assert gs.HuntTile(damage=3).reveal is False   # plain damage tile does not (neg)


@clause("CL-054", "3-hunt", "Mordor is entered only from Morannon or Minas Morgul (neg: elsewhere)")
def cl_054():
    st = setup_game()
    st.fellowship.region = st.board.region('rivendell')
    assert game_logic._check_mordor_entry(st) is False, "cannot enter Mordor from Rivendell (neg)"
    st2 = setup_game()
    st2.fellowship.region = st2.board.region('morannon')
    assert game_logic._check_mordor_entry(st2) is True, "Morannon is a Mordor gate"
    st3 = setup_game()
    st3.fellowship.region = st3.board.region('minas morgul')
    assert game_logic._check_mordor_entry(st3) is True, "Minas Morgul is a Mordor gate"


# ===========================================================================
# CHAPTER 4 - Armies & battle
# ===========================================================================

@clause("CL-060", "4-battle", "Only an At-War nation may attack (neg: non-belligerent cannot)")
def cl_060():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    place(st, 'osgiliath', Nation.SAURON, regulars=3, clear=True)
    # Sauron not at war -> no attack on Minas Tirith
    st.political.boxes[Nation.SAURON] = 2
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'osgiliath']
    assert not atk, "non-belligerent Sauron must NOT attack (neg)"
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'osgiliath']
    assert atk, "At-War Sauron must be able to attack"


@clause("CL-061", "4-battle", "Non-belligerent army may not cross into another nation (neg)")
def cl_061():
    st = setup_game()
    # Sauron active(box2) not at war; every move dest must be Sauron/neutral
    set_dice(st, Side.SHADOW, [Face.ARMY])
    for a in ai.get_legal_actions(st, Side.SHADOW):
        if a.action_type != 'army_move':
            continue
        army = st.armies.get(a.params['from'])
        if not army or not army.units:
            continue
        nat = army.units[0].nation
        if st.political.is_at_war(nat):
            continue
        dest = ai._region_nation(sanitize(a.params['to']))
        assert dest is None or dest == nat, \
            f"non-belligerent {nat.value} crossed into {dest} (neg)"


@clause("CL-062", "4-battle", "Mixed army attacks only if ALL units' nations are At War (neg one passive)")
def cl_062():
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3
    st.political.boxes[Nation.SAURON] = 2       # passive
    st.political.boxes[Nation.ROHAN] = 3
    place(st, 'gap of rohan', Nation.ISENGARD, regulars=1, clear=True)
    army_at(st, 'gap of rohan').units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    place(st, 'fords of isen', Nation.ROHAN, regulars=2, clear=True)
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'gap of rohan']
    assert not atk, "mixed army with a passive nation must NOT attack (neg)"
    st.political.boxes[Nation.SAURON] = 3
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'gap of rohan']
    assert atk, "all-At-War mixed army may attack"


@clause("CL-063", "4-battle", "City / fortification gives attacker a first-round 6+ penalty")
def cl_063():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.ROHAN] = 3
    place(st, 'westemnet', Nation.SAURON, regulars=5)
    c = cb.setup_combat(st, 'Westemnet', 'Edoras')
    assert c.city_defense, "Edoras (city) must set city_defense"
    place(st, 'north ithilien', Nation.SAURON, regulars=5)
    c2 = cb.setup_combat(st, 'North Ithilien', 'Osgiliath')
    assert c2.city_defense, "Osgiliath (fortification) must set city_defense"


@clause("CL-064", "4-battle", "Stronghold FIELD battle has no city penalty (neg)")
def cl_064():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    place(st, 'osgiliath', Nation.SAURON, regulars=5)
    field = cb.setup_combat(st, 'Osgiliath', 'Minas Tirith')
    assert not field.city_defense, "stronghold field battle must NOT set city_defense (neg)"
    assert field.stronghold_defense
    siege = cb.setup_combat(st, 'Osgiliath', 'Minas Tirith', is_siege=True)
    assert siege.is_siege and not siege.city_defense


@clause("CL-065", "4-battle", "Combat roll: natural 1 always misses, natural 6 always hits")
def cl_065():
    random.seed(11)
    one_hits = one_tot = six_hits = six_tot = 0
    for _ in range(8000):
        rolls, hits = cb.combat_roll(1, 5, modifier=4)
        if rolls[0] == 1:
            one_tot += 1
            one_hits += (hits > 0)
    for _ in range(8000):
        rolls, hits = cb.combat_roll(1, 5, modifier=-10)
        if rolls[0] == 6:
            six_tot += 1
            six_hits += (hits > 0)
    assert one_tot > 50 and one_hits == 0, f"nat-1 must miss ({one_hits}/{one_tot})"
    assert six_tot > 50 and six_hits == six_tot, f"nat-6 must hit ({six_hits}/{six_tot})"


@clause("CL-066", "4-battle", "Leader re-rolls are capped at 5")
def cl_066():
    random.seed(3)
    rolls = [2, 2, 2, 2, 2, 2, 2]           # 7 misses vs threshold 5
    rerolls, _ = cb.leader_reroll(rolls, 5, leadership=99)
    assert len(rerolls) == 5, f"re-rolls capped at 5, got {len(rerolls)}"


@clause("CL-067", "4-battle", "Siege press downgrades one elite; fails without elites (neg)")
def cl_067():
    st = setup_game()
    a = place(st, 'osgiliath', Nation.SAURON, regulars=3, elites=2, clear=True)
    assert cb.count_elite_units(a) == 2
    assert cb.siege_press(a) and cb.count_elite_units(a) == 1
    assert cb.siege_press(a) and cb.count_elite_units(a) == 0
    assert not cb.siege_press(a), "press must fail with no elites (neg)"


@clause("CL-068", "4-battle", "Max 5 units in a besieged stronghold; setup trims the excess")
def cl_068():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    a = army_at(st, 'minas tirith')
    a.units.clear()
    a.leaders.clear()
    for _ in range(7):
        a.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    for _ in range(3):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    st.besieged_strongholds.add('minas tirith')
    c = cb.setup_combat(st, 'minas tirith', 'minas tirith')
    garrison = c.defender_army if c.defender_side == Side.FREE_PEOPLES else c.attacker_army
    assert len(garrison.units) == 5, f"garrison must be trimmed to 5, got {len(garrison.units)}"


@clause("CL-069", "4-battle", "Besieged garrison may sortie (from==to) but not attack adjacent (neg)")
def cl_069():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    a = army_at(st, 'minas tirith')
    a.units.clear()
    a.leaders.clear()
    a.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.GONDOR, Kind.ELITE))
    a.leaders.append(gs.Leader(Nation.GONDOR))
    # besieging Shadow army co-located
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    st.besieged_strongholds.add('minas tirith')
    # give an adjacent Shadow army too (to see the garrison cannot hit it)
    place(st, 'osgiliath', Nation.SAURON, regulars=2)
    set_dice(st, Side.FREE_PEOPLES, [Face.ARMY])
    acts = [a2 for a2 in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a2.action_type == 'army_attack'
            and sanitize(a2.params['from']) == 'minas tirith']
    sortie = [a2 for a2 in acts if sanitize(a2.params['to']) == 'minas tirith']
    adjacent = [a2 for a2 in acts if sanitize(a2.params['to']) != 'minas tirith']
    assert sortie, "garrison must be able to sortie (from==to)"
    assert not adjacent, "garrison must NOT attack adjacent regions through the siege (neg)"


@clause("CL-070", "4-battle", "Besieged garrison cannot move out; its side cannot move in (neg)")
def cl_070():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    a = army_at(st, 'minas tirith')
    a.units.clear()
    a.leaders.clear()
    a.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))   # besieger co-located
    st.besieged_strongholds.add('minas tirith')
    place(st, 'osgiliath', Nation.GONDOR, regulars=2)      # relief force adjacent
    set_dice(st, Side.FREE_PEOPLES, [Face.ARMY])
    moves = [a2 for a2 in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a2.action_type == 'army_move']
    out = [m for m in moves if sanitize(m.params['from']) == 'minas tirith']
    into = [m for m in moves if sanitize(m.params['to']) == 'minas tirith']
    assert not out, "garrison may not move OUT of the besieged stronghold (neg)"
    assert not into, "garrison's side may not move INTO the besieged region (neg)"


@clause("CL-071", "4-battle", "FP casualties are permanent; Shadow casualties return to pool")
def cl_071():
    st = setup_game()
    st.fp_casualties = {Nation.GONDOR: {'regular': 5, 'elite': 0, 'leader': 0}}
    dep = ai._count_deployed(st, Nation.GONDOR)
    gmax = ai.NATION_MAX_FIGURES[Nation.GONDOR]['regular']
    # available accounts for the 5 permanent losses
    assert gmax - dep['regular'] - 5 >= 0
    # Shadow recruit ignores casualties (units return)
    sdep = ai._count_deployed(st, Nation.SAURON)
    smax = ai.NATION_MAX_FIGURES[Nation.SAURON]['regular']
    assert ai._can_recruit(st, Nation.SAURON, 'regular') == (sdep['regular'] < smax)


@clause("CL-072", "4-battle", "Saruman: each Isengard elite adds +1 combat leadership (live setup_combat)")
def cl_072():
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3
    st.political.boxes[Nation.ROHAN] = 3
    place(st, 'gap of rohan', Nation.ISENGARD, elites=2, clear=True)     # leaderless
    place(st, 'fords of isen', Nation.ROHAN, regulars=2, clear=True)
    base = cb.setup_combat(st, 'gap of rohan', 'fords of isen').attacker_leadership
    army_at(st, 'orthanc').leaders.append(game_logic.Minion(Nation.ISENGARD, name='Saruman'))
    withsar = cb.setup_combat(st, 'gap of rohan', 'fords of isen').attacker_leadership
    assert base == 0, f"leaderless Isengard has 0 leadership w/o Saruman (neg), got {base}"
    assert withsar - base == 2, f"2 Isengard elites -> +2 leadership, got +{withsar - base}"


@clause("CL-073", "4-battle", "Saruman: leaderless Isengard-elite army counts as led for Character die")
def cl_073():
    st = setup_game()
    a = place(st, 'north dunland', Nation.ISENGARD, elites=1, clear=True)   # no leaders
    assert not game_logic.army_has_leader_figure(st, a), "leaderless w/o Saruman => no leader (neg)"
    army_at(st, 'orthanc').leaders.append(game_logic.Minion(Nation.ISENGARD, name='Saruman'))
    assert game_logic.army_has_leader_figure(st, a), "Saruman grants Isengard-elite leadership"


@clause("CL-074", "4-battle", "A region MAY be attacked repeatedly with separate dice")
def cl_074():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    place(st, 'osgiliath', Nation.SAURON, regulars=3, clear=True)
    place(st, 'minas morgul', Nation.SAURON, regulars=3)
    # (tracker deleted 2026-07-27; the assertion below still guards
    # against anyone re-adding a once-per-turn attack gate)
    set_dice(st, Side.SHADOW, [Face.ARMY, Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['to']) == 'minas tirith']
    assert atk, "already-attacked region must still be attackable"


@clause("CL-075", "4-battle", "A lone FP Leader (no units) is removed; a lone Nazgul survives")
def cl_075():
    st = setup_game()
    army_at(st, 'bree').leaders.append(gs.Leader(Nation.THE_NORTH))
    army_at(st, 'bree').units.clear()
    game_logic.remove_orphaned_leaders(st)
    left = [l for l in army_at(st, 'bree').leaders if type(l) is gs.Leader]
    assert not left, "orphaned FP leader must be removed"
    naz = army_at(st, 'carrock')
    naz.units.clear()
    naz.leaders.append(gs.Nazgul(Nation.SAURON))
    game_logic.remove_orphaned_leaders(st)
    assert any(isinstance(l, gs.Nazgul) for l in army_at(st, 'carrock').leaders), \
        "a lone Nazgul survives cleanup"


@clause("CL-076", "4-battle", "Own-unit stacking cap is 10 in an open region (neg: no recruit at cap)")
def cl_076():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    a = army_at(st, 'nurn')
    a.units.clear()
    for _ in range(game_logic.MAX_ARMY_UNITS):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    acts = ai.get_legal_actions(st, Side.SHADOW)
    # The cap is on ARMY UNITS (regular/elite); Leaders/Nazgul do not count.
    unit_recs = [x for x in acts
                 if sanitize(x.params.get('region', '')) == 'nurn'
                 and ((x.action_type == 'muster_recruit'
                       and x.params.get('type') == 'regular')
                      or x.action_type == 'muster_elite')]
    assert not unit_recs, "no unit recruit into a region already at the 10-unit cap (neg)"


@clause("CL-077", "4-battle", "The besieger (not the garrison) may move out of a besieged region")
def cl_077():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    a = army_at(st, 'minas tirith')
    a.units.clear()
    a.leaders.clear()
    a.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))       # garrison
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))       # besieger co-located
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    st.besieged_strongholds.add('minas tirith')
    set_dice(st, Side.SHADOW, [Face.ARMY])
    out = [x for x in ai.get_legal_actions(st, Side.SHADOW)
           if x.action_type == 'army_move' and sanitize(x.params['from']) == 'minas tirith']
    assert out, "the besieging Shadow force may thin/lift the siege by moving out"


# ===========================================================================
# CHAPTER 5 - Politics & muster
# ===========================================================================

@clause("CL-080", "5-politics", "Advancing the political track NEVER activates a nation")
def cl_080():
    st = setup_game()
    assert not st.political.is_active(Nation.ROHAN)
    game_logic.advance_political(st, Nation.ROHAN, method='muster_die')
    assert not st.political.is_active(Nation.ROHAN), "advance must NOT activate (neg)"
    assert st.political.boxes[Nation.ROHAN] == 1


@clause("CL-081", "5-politics", "A passive nation cannot advance from box 2 to At War (neg)")
def cl_081():
    st = setup_game()
    st.political.boxes[Nation.ROHAN] = 2
    st.political.active[Nation.ROHAN] = False
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    offers = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
              if a.action_type == 'advance_political' and a.params['nation'] == Nation.ROHAN]
    assert not offers, "passive nation may not advance to At War (neg)"
    raised = False
    try:
        game_logic.advance_political(st, Nation.ROHAN)
    except RuntimeError:
        raised = True
    assert raised, "engine must reject the box2->3 advance for a passive nation"


@clause("CL-082", "5-politics", "Enemy army entering a nation's region activates that nation")
def cl_082():
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3
    assert not st.political.is_active(Nation.ROHAN)
    place(st, 'gap of rohan', Nation.ISENGARD, regulars=3)
    game_logic.move_army(st, 'Gap of Rohan', 'Fords of Isen')
    assert st.political.is_active(Nation.ROHAN), "Rohan activates on Shadow entry"


@clause("CL-083", "5-politics", "Fellowship declared in a nation's stronghold activates it")
def cl_083():
    st = setup_game()
    assert not st.political.is_active(Nation.DWARVES)
    st.fellowship.progress = 0
    st.fellowship.region = st.board.region('Erebor')
    game_logic.declare_fellowship(st, region='Erebor')
    assert st.political.is_active(Nation.DWARVES), "declaring in Erebor activates Dwarves"


@clause("CL-084", "5-politics", "Recruits only at own-nation settlements (neg: other nation)")
def cl_084():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.ISENGARD] = 2      # not at war
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    recs = [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'muster_recruit']
    for a in recs:
        assert ai._region_nation(a.params['region']) == Nation.SAURON, \
            f"recruit at non-Sauron settlement {a.params['region']} (neg)"


@clause("CL-085", "5-politics", "Recruiting requires the nation to be At War (neg: not at war)")
def cl_085():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 2        # active-ish but not at war
    st.political.active[Nation.GONDOR] = True
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    recs = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'muster_recruit' and a.params['nation'] == Nation.GONDOR]
    assert not recs, "Gondor not At War must not recruit (neg)"


@clause("CL-086", "5-politics", "Reinforcement pool limit blocks recruiting at max figures (neg)")
def cl_086():
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3
    assert ai._can_recruit(st, Nation.ISENGARD, 'regular')
    orth = army_at(st, 'orthanc')
    while ai._count_deployed(st, Nation.ISENGARD)['regular'] < \
            ai.NATION_MAX_FIGURES[Nation.ISENGARD]['regular']:
        orth.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    assert not ai._can_recruit(st, Nation.ISENGARD, 'regular'), "at max => cannot recruit (neg)"


@clause("CL-087", "5-politics", "FP leader muster needs a friendly unit in the region (neg: lone leader)")
def cl_087():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    army_at(st, 'pelargir').units.clear()        # empty Gondor settlement
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    recs = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'muster_recruit'
            and sanitize(a.params['region']) == 'pelargir']
    regs = [a for a in recs if a.params['type'] == 'regular']
    leads = [a for a in recs if a.params['type'] == 'leader']
    assert regs, "a regular may be recruited into an empty own settlement"
    assert not leads, "no lone FP leader muster into an empty region (neg)"


@clause("CL-088", "5-politics", "Witch-king entry needs Sauron At War AND >=1 FP nation At War")
def cl_088():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    can, _, regions = game_logic.can_muster_witch_king(st)
    assert can and regions, "WK musters with Sauron + FP at war"


@clause("CL-089", "5-politics", "Witch-king refused when Sauron not At War, or no FP at war (neg)")
def cl_089():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 2       # Sauron NOT at war
    st.political.boxes[Nation.GONDOR] = 3
    can, _, _ = game_logic.can_muster_witch_king(st)
    assert not can, "WK must be refused with Sauron passive (neg)"
    st2 = setup_game()
    st2.political.boxes[Nation.SAURON] = 3       # no FP nation at war
    can2, _, _ = game_logic.can_muster_witch_king(st2)
    assert not can2, "WK must be refused with no FP nation at war (neg)"


@clause("CL-090", "5-politics", "Mouth of Sauron entry: all 5 FP nations At War, OR Fellowship in Mordor")
def cl_090():
    st = setup_game()
    for n in (Nation.GONDOR, Nation.ROHAN, Nation.ELVES, Nation.DWARVES, Nation.THE_NORTH):
        st.political.boxes[n] = 3
    can, _, _ = game_logic.can_muster_mouth_of_sauron(st)
    assert can, "Mouth musters with all 5 FP nations at war"
    st2 = setup_game()
    st2.fellowship.mordor_entered = True
    can2, _, _ = game_logic.can_muster_mouth_of_sauron(st2)
    assert can2, "Mouth musters with the Fellowship in Mordor"
    st3 = setup_game()
    st3.political.boxes[Nation.GONDOR] = 3      # only one FP nation
    can3, _, _ = game_logic.can_muster_mouth_of_sauron(st3)
    assert not can3, "Mouth refused with only one FP nation at war (neg)"


@clause("CL-091", "5-politics", "Saruman entry: Isengard At War, into unconquered Orthanc only")
def cl_091():
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 2
    can, _, _ = game_logic.can_muster_saruman(st)
    assert not can, "Saruman refused when Isengard not At War (neg)"
    st.political.boxes[Nation.ISENGARD] = 3
    can, _, regions = game_logic.can_muster_saruman(st)
    assert can and [sanitize(r) for r in regions] == ['orthanc'], (can, regions)


@clause("CL-092", "5-politics", "A killed minion cannot be re-mustered (neg)")
def cl_092():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    st.witch_king.was_ever_in_play = True       # was eliminated
    st.witch_king.in_play = False
    can, _, _ = game_logic.can_muster_witch_king(st)
    assert not can, "eliminated WK may never return (neg)"


@clause("CL-093", "5-politics", "Mouth muster-as-army: a plain Muster die yields army actions once/turn")
def cl_093():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.mouth_of_sauron.in_play = False
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    assert not [a for a in ai.get_legal_actions(st, Side.SHADOW)
                if a.action_type == 'army_move'], "plain Muster die => no army move w/o Mouth (neg)"
    st.mouth_of_sauron.in_play = True
    st.sh_mouth_muster_used_this_turn = False
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    mv = [a for a in ai.get_legal_actions(st, Side.SHADOW)
          if a.action_type == 'army_move' and a.params.get('mouth_muster')]
    assert mv, "Mouth in play => Muster die can move armies"


@clause("CL-094", "5-politics", "Mustering a minion consumes the entire Muster die (no second recruit)")
def cl_094():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.boxes[Nation.GONDOR] = 3
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    minion = [a for a in ai.get_legal_actions(st, Side.SHADOW)
              if a.action_type == 'muster_minion']
    assert minion, "WK minion muster should be offered"
    ai._execute_action(st, minion[0])
    assert st.pending_muster_die is None, "minion muster must consume the whole die (neg pending)"


@clause("CL-095", "5-politics", "A companion ending movement in its nation's stronghold activates it")
def cl_095():
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    st.political.boxes[Nation.DWARVES] = 1
    st.political.active[Nation.DWARVES] = False
    place(st, 'iron hills', Nation.DWARVES, regulars=2)
    army_at(st, 'iron hills').leaders.append(gs.Companion(name='Gimli', level=2))
    game_logic.move_army(st, 'Iron Hills', 'Erebor')
    assert st.political.is_active(Nation.DWARVES), "Gimli reaching Erebor activates Dwarves"


# ===========================================================================
# CHAPTER 6 - Event cards & table cards
# ===========================================================================

@clause("CL-100", "6-cards", "Shadow table-kill offered ONLY with 1 Army card + 1 Character card")
def cl_100():
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_Strategy_A_Power_too_Great', card_name='A Power too Great',
        side=Side.FREE_PEOPLES, card_type=gs.PersistentCardType.POWER_TOO_GREAT))
    army_c = _find_sh_army_card()
    char_c = _find_sh_character_card()
    assert army_c and char_c
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    # missing the character card -> not offered
    st.hands[Side.SHADOW].hand = [army_c]
    assert not [a for a in ai.get_legal_actions(st, Side.SHADOW)
                if a.action_type == 'discard_table_card'], "no kill without a Character card (neg)"
    # both cards -> offered on any die
    st.hands[Side.SHADOW].hand = [army_c, char_c]
    assert [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'discard_table_card'], "kill offered with Army + Character card"


@clause("CL-101", "6-cards", "Shadow table-kill: the Army + Character payment pair is the Shadow's learned pending; the chosen pair is discarded")
def cl_101():
    # EX-8 (audit 2026-08-29, USER "fix the open issues"; learned-decision
    # doctrine / 1.254): WHICH Army-type + Character pair pays is a
    # pending_card_option owned by the Shadow (every legal pair), resolved
    # by card_option - the first-in-hand auto-pay this clause used to pin
    # survives only on the teacher path (log = truth).
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_Strategy_A_Power_too_Great', card_name='A Power too Great',
        side=Side.FREE_PEOPLES, card_type=gs.PersistentCardType.POWER_TOO_GREAT))
    army_c = _find_sh_army_card()
    char_c = _find_sh_character_card()
    st.hands[Side.SHADOW].hand = [army_c, char_c]
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    act = ai.Action(die_idx=0, action_type='discard_table_card',
                    params={'card_id': 'FREE_Strategy_A_Power_too_Great',
                            'card_name': 'A Power too Great'})
    ai._execute_action(st, act)
    pco = getattr(st, 'pending_card_option', None)
    assert pco is not None and pco.get('table_kill_payment') \
        and pco.get('side') == 'sh', "the payment pair opens the Shadow's pending"
    assert pco.get('options') == [{'army_card': army_c, 'character_card': char_c}], pco.get('options')
    assert army_c in st.hands[Side.SHADOW].hand and char_c in st.hands[Side.SHADOW].hand, \
        "hand intact until the pair is chosen"
    assert not st.persistent_cards.has_card('A Power too Great'), "table card removed"
    assert st.dice[Side.SHADOW][0].used, "die spent by the kill"
    pick = ai.table_kill_payment_action(st, army_c, char_c)
    assert [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'card_option'
            and a.params.get('choices') == pick.params.get('choices')], "pick is on the menu"
    ai._execute_action(st, pick)
    assert getattr(st, 'pending_card_option', None) is None, "pending resolved"
    assert army_c not in st.hands[Side.SHADOW].hand, "Army payment card must be discarded"
    assert char_c not in st.hands[Side.SHADOW].hand, "Character payment card must be discarded"
    # 1.261: payment cards are discarded FACE-DOWN (payer's private pile)
    fdown = st.hands[Side.SHADOW].discarded_face_down
    assert army_c in fdown and char_c in fdown, \
        "1.261: both payment cards reach the Shadow face-down pile (conservation)"
    assert not any(army_c in lst or char_c in lst
                   for lst in st.discard_face_up.values()), \
        "1.261: payment cards must not reach a face-up pile"


@clause("CL-102", "6-cards", "FP kills the Palantir with a Will die alone, or any die + Elven Ring")
def cl_102():
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='SHADOW_Strategy_The_Palantir', card_name='The Palantir of Orthanc',
        side=Side.SHADOW, card_type=gs.PersistentCardType.PALANTIR_OF_ORTHANC))
    # Will die alone
    set_dice(st, Side.FREE_PEOPLES, [Face.WILL_OF_WEST])
    kills = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a.action_type == 'discard_table_card']
    assert any(a.params.get('payment') == 'will' for a in kills), "Will-alone kill offered"
    # any die + a ring
    st.fp_elven_rings = 3
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    kills = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a.action_type == 'discard_table_card']
    assert any(a.params.get('payment') == 'ring' for a in kills), "ring-paid kill offered"


@clause("CL-103", "6-cards", "FP Palantir kill refused with a non-Will die and no ring (neg)")
def cl_103():
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='SHADOW_Strategy_The_Palantir', card_name='The Palantir of Orthanc',
        side=Side.SHADOW, card_type=gs.PersistentCardType.PALANTIR_OF_ORTHANC))
    st.fp_elven_rings = 0
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    kills = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a.action_type == 'discard_table_card']
    assert not kills, "no kill with a plain die and no ring (neg)"


@clause("CL-104", "6-cards", "Palantir ring-kill spends one FP Elven Ring (transfers to Shadow)")
def cl_104():
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='SHADOW_Strategy_The_Palantir', card_name='The Palantir of Orthanc',
        side=Side.SHADOW, card_type=gs.PersistentCardType.PALANTIR_OF_ORTHANC))
    st.fp_elven_rings = 3
    st.sh_elven_rings = 0
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    act = ai.Action(die_idx=0, action_type='discard_table_card',
                    params={'card_id': 'SHADOW_Strategy_The_Palantir',
                            'card_name': 'The Palantir of Orthanc', 'payment': 'ring'})
    ai._execute_action(st, act)
    assert st.fp_elven_rings == 2, st.fp_elven_rings
    assert st.sh_elven_rings == 1, st.sh_elven_rings
    assert not st.persistent_cards.has_card('The Palantir of Orthanc')


@clause("CL-105", "6-cards", "Elven Ring is a free LEGALITY EXPANSION (die keeps its face, no die consumed); FP->SH on use")
def cl_105():
    # USER RULING 2026-08-19 (A7), replacing the face-change reading
    # this clause used to assert: "the ring is a legality expansion,
    # NOT a face change... the die keeps its face; the ring widens what
    # that die may fund for that action - any non-Will-of-the-West die
    # for Free Peoples, any die for Shadow.  Anything later reading the
    # die's face still sees the original face."
    # Measured basis: the client's ring command emits a marker walk and
    # no die face at all (540 ring uses over 300 logs -> 1 nearby face
    # re-render vs 2.6 by chance), so a face change is not
    # reconstructible from the rendered log by any parser.
    st = setup_game()
    set_dice(st, Side.FREE_PEOPLES, [Face.MUSTER])
    before = st.fp_elven_rings
    game_logic.use_elven_ring(st, Side.FREE_PEOPLES, 0, Face.CHARACTER)
    assert st.dice[Side.FREE_PEOPLES][0].current == Face.MUSTER, "the die KEEPS its rolled face"
    assert not st.dice[Side.FREE_PEOPLES][0].used, "the ring consumes NO die"
    assert st.fp_elven_rings == before - 1 and st.sh_elven_rings == 1, "ring transfers to Shadow"
    assert game_logic.ring_waiver_active(st, Side.FREE_PEOPLES), "the ring must buy SOMETHING"
    # and what it buys: the Muster die now funds a Character action
    used = ai._consume_die_for_face(
        st, Side.FREE_PEOPLES, 0, (Face.CHARACTER, Face.WILL_OF_WEST))
    assert used == 0, ("the STAGED die pays; consuming a different die "
                       "is the client-vs-engine desync (USER C2/C4)")
    assert not game_logic.ring_waiver_active(st, Side.FREE_PEOPLES), \
        "one ring buys ONE action"


@clause("CL-106", "6-cards", "Axe and Bow stays while EITHER hero remains; discards when both gone (USER ruling 2026-07-20)")
def cl_106():
    # USER RULING 2026-07-20 (P2-01 resolved): the card only requires
    # ONE of Gimli/Legolas in the Fellowship — the engine's keep-while-
    # either-remains behavior is CORRECT; the OR-discard reading was
    # rejected.
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_Strategy_Axe_and_Bow', card_name='Axe and Bow',
        side=Side.FREE_PEOPLES, card_type=gs.PersistentCardType.AXE_AND_BOW,
        requires_companion='Gimli, Legolas'))
    # Gimli leaves, Legolas stays -> card STAYS (pos)
    st.fellowship.companions = [c for c in st.fellowship.companions if c.name != 'Gimli']
    discarded = game_logic.check_companion_persistent_discards(st)
    assert 'Axe and Bow' not in discarded, \
        "card must STAY while Legolas remains (USER ruling: requires only 1)"
    assert st.persistent_cards.has_card('Axe and Bow')
    # Legolas leaves too -> card discards (neg boundary)
    st.fellowship.companions = [c for c in st.fellowship.companions if c.name != 'Legolas']
    discarded = game_logic.check_companion_persistent_discards(st)
    assert 'Axe and Bow' in discarded, \
        "card discards once BOTH heroes have left"


@clause("CL-107", "6-cards", "Denethor's Folly removal IS offered: any die with Aragorn/Gandalf in Minas Tirith", page='card')
def cl_107():
    # USER RULING 1.177 (P2-03): removable via Will-of-West, or ANY die
    # when Aragorn/Gandalf (Grey or White) is with the besieged MT army.
    # (Was a stale skip from before the ruling landed - converted
    # 2026-07-21.)
    from persistent_effects import get_denethors_folly_discard_options
    st = setup_game()
    # Folly on the table + MT besieged (both sides co-located so
    # cleanup keeps the siege)
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='SHADOW_Strategy_Denethor\'s_Folly',
        card_name="Denethor's Folly",
        card_type=gs.PersistentCardType.DENETHORS_FOLLY,
        side=Side.SHADOW))
    st.besieged_strongholds.add(sanitize('minas tirith'))
    mt = game_logic.get_or_create_army(st, 'minas tirith')
    mt.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    mt.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    # no hero yet -> any-die removal NOT allowed (neg)
    assert not get_denethors_folly_discard_options(st).get('any_die'), \
        "no Aragorn/Gandalf in MT => no any-die removal (neg)"
    # Gandalf the Grey with the garrison -> any-die removal allowed
    mt.leaders.append(gs.Companion(name='Gandalf the Grey', level=3,
                                   leadership=1))
    assert get_denethors_folly_discard_options(st).get('any_die'), \
        "Gandalf in MT => any-die removal allowed"
    # and the MENU actually offers discard_table_card for the Folly
    # (the generator is per-die-face: give FP one CHARACTER die)
    st.dice[Side.FREE_PEOPLES] = [gs.ActionDie(faces=[Face.CHARACTER],
                                               side=Side.FREE_PEOPLES)]
    st.dice[Side.FREE_PEOPLES][0].current = Face.CHARACTER
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    offers = [a for a in acts if a.action_type == 'discard_table_card'
              and 'folly' in str(a.params.get('card_name', '')).lower()]
    assert offers, "get_legal_actions must offer the Folly removal"


# ===========================================================================
# CHAPTER 7 - Characters
# ===========================================================================

@clause("CL-110", "7-chars", "Saruman never leaves Orthanc (an army moving out does not carry him)")
def cl_110():
    st = setup_game()
    assert st.saruman.immobile is True
    st.saruman.in_play = True
    st.political.boxes[Nation.ISENGARD] = 3
    orth = army_at(st, 'orthanc')
    orth.units.clear()
    orth.leaders.clear()
    orth.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    orth.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    orth.leaders.append(game_logic.Minion(Nation.ISENGARD, name='Saruman'))
    set_dice(st, Side.SHADOW, [Face.ARMY])
    moves = [a for a in ai.get_legal_actions(st, Side.SHADOW)
             if a.action_type == 'army_move' and sanitize(a.params['from']) == 'orthanc']
    assert moves, "the Orthanc army should be able to move out"
    ai._execute_action(st, moves[0])
    still = any(type(l).__name__ == 'Minion' and 'saruman' in getattr(l, 'name', '').lower()
               for l in army_at(st, 'orthanc').leaders)
    assert still, "Saruman must remain in Orthanc when an army moves out (neg: never rides along)"


@clause("CL-111", "7-chars", "Gandalf the White transform needs a Shadow minion in/ever-in play")
def cl_111():
    st = setup_game()
    # remove Gandalf the Grey from the fellowship (a prerequisite)
    st.fellowship.companions = [c for c in st.fellowship.companions
                                if c.name != 'Gandalf the Grey']
    can_no, _, _ = game_logic.can_transform_gandalf_white(st)
    assert not can_no, "no Gandalf-White while no minion has appeared (neg)"
    st.witch_king.was_ever_in_play = True
    can_yes, _, _ = game_logic.can_transform_gandalf_white(st)
    assert can_yes, "a minion having appeared enables Gandalf the White"


@clause("CL-112", "7-chars", "Witch-king data: leadership 2, infinite movement, first-round card draw")
def cl_112():
    st = setup_game()
    wk = st.witch_king
    assert wk.leadership == 2 and wk.infinite_movement and wk.combat_card_draw


@clause("CL-113", "7-chars", "Mouth of Sauron data: level 3, leadership 2; the muster-as-army ability is the live mouth_muster mechanism")
def cl_113():
    # The old muster_to_army / moves_with_nazgul stub flags were
    # DELETED 2026-08-24 (vestigial, no production reader).  The REAL
    # abilities: amendment 1.2 (once-per-turn Muster die as Army die,
    # sh_mouth_muster_used_this_turn) and 1.177b (mouth_move).
    st = setup_game()
    m = st.mouth_of_sauron
    assert m.level == 3 and m.leadership == 2
    assert hasattr(st, 'sh_mouth_muster_used_this_turn'), \
        "the live muster-as-army flag must exist"
    assert not hasattr(m, 'muster_to_army'), \
        "the vestigial stub must STAY deleted"


@clause("CL-114", "7-chars", "In-play lone Mouth can move independently on a Character die (flag-driver)")
def cl_114():
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    # strip every real Nazgul so only the Mouth (a minion) remains to move
    for k, a in st.armies.items():
        a.leaders = [l for l in a.leaders if not isinstance(l, gs.Nazgul)]
    st.mouth_of_sauron.in_play = True
    m = army_at(st, 'gorgoroth')
    m.units.clear()                 # lone Mouth: no army to carry it out
    m.leaders.clear()
    m.leaders.append(game_logic.Minion(Nation.SAURON, name='Mouth'))
    start = sanitize('gorgoroth')

    def mouth_region(state):
        for k, a in state.armies.items():
            for l in a.leaders:
                if type(l).__name__ == 'Minion' and 'mouth' in getattr(l, 'name', '').lower():
                    return sanitize(k)
        return None

    set_dice(st, Side.SHADOW, [Face.CHARACTER])
    movable = False
    for act in ai.get_legal_actions(st, Side.SHADOW):
        s2 = ai.clone_state(st)
        try:
            ai._execute_action(s2, act)
        except Exception:
            continue
        if mouth_region(s2) not in (start, None):
            movable = True
            break
    assert movable, "the in-play Mouth must be movable on a Character die (CURRENT_RULES 3)"


# ===========================================================================
# CHAPTER 8 - Victory conditions & timing
# ===========================================================================

@clause("CL-120", "8-victory", "Shadow military victory at 10 VP; below 10 no win (neg)")
def cl_120():
    st = setup_game()
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth', 'helm s deep',
                                        'lorien', 'erebor'}   # 5 strongholds = 10 VP
    r = game_logic.check_military_victory(st)
    assert r['victory'] == 'shadow', r
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth'}   # 4 VP
    assert game_logic.check_military_victory(st)['victory'] is None, "4 VP is not a win (neg)"


@clause("CL-121", "8-victory", "FP military victory at 4 VP; below 4 no win (neg)")
def cl_121():
    st = setup_game()
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon', 'minas morgul'}   # 2 strongholds = 4 VP
    r = game_logic.check_military_victory(st)
    assert r['victory'] == 'free_peoples', r
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon'}   # 2 VP
    assert game_logic.check_military_victory(st)['victory'] is None, "2 VP is not a win (neg)"


@clause("CL-122", "8-victory", "Ring corruption of 12 is a Shadow win; 11 is not (neg)")
def cl_122():
    st = setup_game()
    st.fellowship.corruption = 12
    r = game_logic.check_ring_victory(st)
    assert r['victory'] == 'shadow', r
    st.fellowship.corruption = 11
    assert game_logic.check_ring_victory(st)['victory'] is None, "11 corruption is not a win (neg)"


@clause("CL-123", "8-victory", "Mordor step 5 with corruption<12 is a FP win; pending hunt blocks it (neg)")
def cl_123():
    st = setup_game()
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 5
    st.fellowship.corruption = 4
    st.pending_hunt = None
    assert game_logic.check_ring_victory(st)['victory'] == 'free_peoples'
    st.pending_hunt = {'damage': 2}
    assert game_logic.check_ring_victory(st)['victory'] is None, \
        "unresolved hunt must block the step-5 win (neg)"


@clause("CL-124", "8-victory", "Military victory fires only at end-of-turn; ring victory fires immediately")
def cl_124():
    st = setup_game()
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth', 'helm s deep',
                                        'lorien', 'erebor'}
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory_type'] == 'military'
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "military VP must NOT end the game mid-turn (neg)"
    assert game_logic.should_game_end(vc, end_of_turn=True), "military ends at end-of-turn"
    st2 = setup_game()
    st2.fellowship.corruption = 12
    vc2 = game_logic.check_all_victory_conditions(st2)
    assert game_logic.should_game_end(vc2, end_of_turn=False), "ring victory ends immediately"


@clause("CL-125", "8-victory", "VP: enemy-held city=1, stronghold=2; own settlements score nothing (neg)")
def cl_125():
    st = setup_game()
    st.enemy_controlled[Side.SHADOW] = {'pelargir'}          # FP city
    assert game_logic.calculate_victory_points(st)['shadow'] == 1
    st.enemy_controlled[Side.SHADOW] = {'minas tirith'}      # FP stronghold
    assert game_logic.calculate_victory_points(st)['shadow'] == 2
    st.enemy_controlled = {Side.FREE_PEOPLES: set(), Side.SHADOW: set()}
    vp = game_logic.calculate_victory_points(st)
    assert vp['shadow'] == 0 and vp['free_peoples'] == 0, "uncaptured settlements score nothing (neg)"


# ===========================================================================
# CHAPTER 9 - Fellowship reveal placement (edge)
# ===========================================================================

@clause("CL-130", "9-reveal", "Reveal moves the Ring-bearers figure EXACTLY 'progress' regions")
def cl_130():
    st = setup_game()
    st.fellowship.region = st.board.region('rivendell')
    st.fellowship.progress = 2
    st.fellowship.hidden = False
    start = sanitize('rivendell')
    game_logic._auto_reveal_place(st)
    end = sanitize(game_logic.fellowship_region(st))
    if end != start:
        assert _bfs_dist(st, start, end) == 2, \
            f"reveal placed at distance {_bfs_dist(st, start, end)}, expected exactly 2"


@clause("CL-131", "9-reveal", "Reveal never ends in an unconquered FP city/stronghold (P2-04 fallback)", page=42)
def cl_131():
    # USER RULING 1.177 (P2-04): place at exactly Progress when legal;
    # when every region at that distance is an unconquered FP
    # settlement, fall back to SHORTER distances — never inside the
    # settlement (the old fallback revealed the Fellowship INSIDE Grey
    # Havens from Forlindon). (Stale skip converted 2026-07-21.)
    def _is_unconq_fp_settlement(st, r):
        meta = board.regions.get(sanitize(r), {})
        return ((meta.get('is_city') or meta.get('is_stronghold'))
                and game_logic.is_fp_region(sanitize(r))
                and sanitize(r) not in
                st.enemy_controlled.get(Side.SHADOW, set()))

    # the original P2-04 scenario: Forlindon, progress 1 (Grey Havens
    # is the FP-stronghold neighbor)
    st = setup_game()
    st.fellowship.region = st.board.region('forlindon')
    st.fellowship.progress = 1
    st.fellowship.hidden = False
    game_logic._auto_reveal_place(st)
    end = game_logic.fellowship_region(st)
    assert not _is_unconq_fp_settlement(st, end), \
        f"reveal placed INSIDE an unconquered FP settlement: {end}"
    assert st.fellowship.progress == 0, "progress resets on reveal"
    # and at progress 2 as well (wider frontier, same contract)
    st2 = setup_game()
    st2.fellowship.region = st2.board.region('forlindon')
    st2.fellowship.progress = 2
    st2.fellowship.hidden = False
    game_logic._auto_reveal_place(st2)
    end2 = game_logic.fellowship_region(st2)
    assert not _is_unconq_fp_settlement(st2, end2), \
        f"reveal placed INSIDE an unconquered FP settlement: {end2}"


# ===========================================================================
# CHAPTER 0 - Game turn sequence  (rulebook p.13 sequence of play; p.21 cards)
# ===========================================================================
# Page anchors below are set INLINE on the decorator (they win over the
# CLAUSE_PAGES table).  Battle-chapter pages follow the TASK's chapter map
# (p.28-29 battle, p.30-31 casualties); the p.41-44 band matches the repo's
# own data/rulebook_page_index.md label "advanced/special Hunt rules
# (p.41-44)"; p.16/21/23/34/37 are interpolations between established anchors
# and are commented as such.  Where a rule is a CURRENT_RULES/amendment house
# ruling with no rulebook page it is page='house'.

@clause("CL-170", "0-turn", "next_turn advances the counter, clears the Hunt Box, un-uses dice", page=13)
def cl_170():
    st = setup_game()
    # dirty the state as a mid-turn snapshot
    st.dice[Side.SHADOW][0].used = True
    st.dice[Side.SHADOW][0].current = Face.ARMY
    d = gs.ActionDie(faces=[Face.EYE], side=Side.SHADOW)
    d.used = True
    st.hunt_box.append(d)
    st.eyes_allocated = 2
    t0 = st.turn
    game_logic.next_turn(st)
    assert st.turn == t0 + 1, "the turn counter advances"
    assert not st.hunt_box, "the Hunt Box is emptied at turn start (neg: not carried over)"
    assert st.eyes_allocated == 0, "the eye count resets"
    assert all(not dd.used and dd.current is None
               for pool in st.dice.values() for dd in pool), \
        "every die is un-used (recovered) for the new Action Roll"


@clause("CL-165", "6-cards", "Each turn both sides draw 1 Character + 1 Strategy card; hand limit 6 is enforced by a LEARNED discard decision (USER 2026-08-15), never an engine auto-trim", page=21)
def cl_165():
    # rb p.21 (Event cards: drawing/playing), interpolated between the
    # established leaders anchor p.20 and the event-face anchor p.22, and
    # consistent with data/rulebook_page_index.md's p.21 label.
    st = setup_game()
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        st.hands[s].hand = []
    game_logic.next_turn(st)
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        assert len(st.hands[s].hand) == 2, \
            f"{s.name} draws exactly 2 event cards/turn, got {len(st.hands[s].hand)}"
    # hand limit 6 (USER RULING 2026-08-15): seed 6 cards, draw 2 -> the
    # engine must NOT pick the discard.  It opens the learned pending;
    # resolving it enforces the limit, one card per decision.
    st2 = setup_game()
    filler = list(get_cards().keys())[:6]
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        st2.hands[s].hand = list(filler)
    game_logic.next_turn(st2)
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        assert len(st2.hands[s].hand) == 8, \
            f"the rollover DRAWS and trims nothing itself, got {len(st2.hands[s].hand)}"
    pd = getattr(st2, 'pending_discard', None)
    assert pd is not None and pd['side'] == Side.FREE_PEOPLES, \
        f"an over-limit hand opens the learned discard pending, got {pd!r}"
    # neg: the menu is the hand itself, and one execution discards ONE card
    guard = 0
    while getattr(st2, 'pending_discard', None) is not None and guard < 20:
        guard += 1
        s2 = st2.pending_discard['side']
        acts = ai.get_legal_actions(st2, s2)
        assert acts and all(a.action_type == 'discard_card' for a in acts), \
            "the pending offers exactly the hand-limit discard menu"
        n_before = len(st2.hands[s2].hand)
        ai._execute_action(st2, acts[0])
        assert len(st2.hands[s2].hand) == n_before - 1, \
            "each decision discards exactly one card"
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        assert len(st2.hands[s].hand) == 6, \
            f"resolving the pendings enforces the 6-card limit, got {len(st2.hands[s].hand)}"
        assert len(st2.hands[s].discarded_face_down) == 2, \
            "both discarded cards reached the face-down pile (never annihilated)"


# ===========================================================================
# CHAPTER 1b - Action-die results table  (rulebook p.16)
# ===========================================================================

@clause("CL-160", "1-dice", "Action-die results table: each face maps to its allowed action family", page=16)
def cl_160():
    # rb p.16 (Action Dice Results), between the established p.15 (dice
    # counts/faces) and p.17 (Eye / Hunt Box) anchors.
    fam = ai._die_faces_for_action
    assert fam(Face.ARMY) == {'army'}, fam(Face.ARMY)
    assert fam(Face.MUSTER) == {'muster'}, fam(Face.MUSTER)
    assert fam(Face.MUSTER_ARMY) == {'army', 'muster'}, fam(Face.MUSTER_ARMY)
    assert fam(Face.CHARACTER) == {'character'}, fam(Face.CHARACTER)
    assert fam(Face.EVENT) == {'event'}, fam(Face.EVENT)
    # negatives: a plain Muster face is NOT an Army face; an Army face is
    # NOT a Muster face; the Eye face yields NOTHING spendable.
    assert 'army' not in fam(Face.MUSTER), "plain Muster is not an Army face (neg)"
    assert 'muster' not in fam(Face.ARMY), "plain Army is not a Muster face (neg)"
    assert fam(Face.EYE) == set(), "Eye maps to no action family (neg)"


@clause("CL-161", "1-dice", "Will of the West is the wild face (army/muster/character/event)", page=16)
def cl_161():
    fams = ai._die_faces_for_action(Face.WILL_OF_WEST)
    for need in ('army', 'muster', 'character', 'event'):
        assert need in fams, f"WoW must cover {need}: {fams}"
    # Only the FP die carries Will; the Shadow pool has none (neg).
    assert Face.WILL_OF_WEST in gs.FP_FACES
    assert Face.WILL_OF_WEST not in gs.SH_FACES, "Will is never a Shadow face (neg)"


# ===========================================================================
# CHAPTER 4c - Battle resolution sequence  (rulebook p.28-29)
# ===========================================================================

@clause("CL-140", "4-battle", "Combat dice = number of Army units, capped at 5 (neg: 8 units still 5)", page=(28, 29))
def cl_140():
    st = setup_game()
    a = place(st, 'nurn', Nation.SAURON, regulars=3, clear=True)
    assert cb.calc_combat_strength(a) == 3, "3 units roll 3 combat dice"
    for _ in range(5):
        a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))    # now 8
    assert cb.calc_combat_strength(a) == 5, "8 units still roll only 5 dice (neg: no >5)"


@clause("CL-141", "4-battle", "A Captain of the West present adds one combat die", page=(28, 29))
def cl_141():
    st = setup_game()
    a = place(st, 'minas tirith', Nation.GONDOR, regulars=2, clear=True)
    base = cb.calc_combat_strength(a)
    assert base == 2
    a.leaders.append(gs.Companion(name='Aragorn', level=3, leadership=2,
                                  captain_of_the_west=True))
    assert cb.calc_combat_strength(a) == 3, "Captain of the West adds +1 combat die"
    # a plain (non-captain) leader adds NO combat die (neg)
    b = place(st, 'pelargir', Nation.GONDOR, regulars=2, clear=True)
    b.leaders.append(gs.Leader(Nation.GONDOR))
    assert cb.calc_combat_strength(b) == 2, "a plain leader is not a combat die (neg)"


@clause("CL-142", "4-battle", "Battle leadership sums per figure (plain/Nazgul 1, Witch-king/Mouth 2)", page=(28, 29))
def cl_142():
    st = setup_game()
    a = gs.Army(region='x')
    a.leaders.append(gs.Leader(Nation.SAURON))                              # 1
    a.leaders.append(gs.Nazgul(Nation.SAURON))                             # 1
    a.leaders.append(game_logic.Minion(Nation.SAURON, name='Witch-king of Angmar'))  # 2
    assert cb.calc_leadership(a) == 4, f"1+1+2 leadership, got {cb.calc_leadership(a)}"
    # leadership drives re-rolls: 5 misses with leadership 3 -> only 3 re-rolls
    random.seed(5)
    rr, _ = cb.leader_reroll([2, 2, 2, 2, 2], threshold=5, leadership=3)
    assert len(rr) == 3, f"re-rolls limited to leadership=3, got {len(rr)}"


# ===========================================================================
# CHAPTER 4d - Casualties & downgrades  (rulebook p.30-31)
# ===========================================================================

@clause("CL-144", "4-casualty", "A downgrade needs a Regular in reinforcements; else the Elite is eliminated (neg)", page=30)
def cl_144():
    # rulebook p.30 (1.188/1.191): reduce an Elite by taking a Regular from
    # REINFORCEMENTS; FP casualties are PERMANENT and cannot source it.
    st = setup_game()
    prev = cb.set_active_state(st)
    try:
        assert cb._downgrade_allowed(Nation.GONDOR, 0, {Nation.GONDOR: 1}) is True
        assert cb._downgrade_allowed(Nation.GONDOR, 0, {Nation.GONDOR: 0}) is False, \
            "no Regular in reinforcements => no downgrade (neg)"
        assert cb._downgrade_allowed(Nation.GONDOR, 1, {Nation.GONDOR: 1}) is False, \
            "a Regular already consumed this scope => no second downgrade (neg)"
        # integration, POSITIVE: one Regular free -> elite downgrades in place
        _set_reg_pool_free(st, Nation.DWARVES, 1)
        a = place(st, 'erebor', Nation.DWARVES, elites=1, clear=True)
        cb.apply_casualties(a, 1, policy='preserve_units')
        assert cb.count_elite_units(a) == 0 and len(a.units) == 1 \
            and a.units[0].kind == Kind.REGULAR, \
            "elite -> Regular downgrade when a Regular is available"
        # integration, NEGATIVE: pool exhausted -> the elite is eliminated
        st2 = setup_game()
        cb.set_active_state(st2)
        _set_reg_pool_free(st2, Nation.DWARVES, 0)
        a2 = place(st2, 'erebor', Nation.DWARVES, elites=1, clear=True)
        cb.apply_casualties(a2, 1, policy='preserve_units')
        assert len(a2.units) == 0, "no reinforcement Regular => elite eliminated (neg)"
    finally:
        cb.set_active_state(prev)


@clause("CL-145", "4-casualty", "Two hits fully remove an Elite; a single hit only downgrades it", page=30)
def cl_145():
    st = setup_game()
    prev = cb.set_active_state(st)
    try:
        _set_reg_pool_free(st, Nation.DWARVES, 5)   # plenty of Regulars free
        # TWO hits: the elite is removed outright, no Regular minted (neg)
        a = place(st, 'erebor', Nation.DWARVES, elites=1, clear=True)
        cas = cb.apply_casualties(a, 2, policy='preserve_elites')
        assert len(a.units) == 0 and len(cas) == 1, \
            "2 hits eliminate the elite, no downgrade (neg: no Regular left)"
        # ONE hit: the elite absorbs it by downgrading to a Regular
        b = place(st, 'iron hills', Nation.DWARVES, elites=1, clear=True)
        cb.apply_casualties(b, 1, policy='preserve_elites')
        assert len(b.units) == 1 and b.units[0].kind == Kind.REGULAR, \
            "a single hit downgrades the elite to a Regular"
    finally:
        cb.set_active_state(prev)


@clause("CL-146", "4-casualty", "A downgraded FP Elite is a PERMANENT elite casualty (does not return to the pool)", page=30)
def cl_146():
    # USER RULING 2026-07-20 (supersedes 1.188 pool-swap doctrine, via
    # trace w5_g2 T4 Woodland Realm): the downgraded FP elite is
    # permanently removed — booked as an ELITE casualty; only the
    # replacement Regular comes from reinforcements.
    st = setup_game()
    dg = gs.Unit(Nation.GONDOR, Kind.ELITE)
    dg._downgraded = True
    killed = gs.Unit(Nation.GONDOR, Kind.REGULAR)
    st.fp_casualties = {}
    cb._track_fp_casualties(st, [dg, killed], Side.FREE_PEOPLES)
    box = st.fp_casualties.get(Nation.GONDOR, {})
    assert box.get('elite', 0) == 1, "a downgraded FP elite IS a permanent elite casualty"
    assert box.get('regular', 0) == 1, "a fully-removed Regular IS a permanent casualty"
    # Shadow losses never track (they return to the pool)
    st.fp_casualties = {}
    cb._track_fp_casualties(st, [gs.Unit(Nation.SAURON, Kind.REGULAR)], Side.SHADOW)
    assert not st.fp_casualties, "Shadow casualties return to the pool, never tracked (neg)"


@clause("CL-147", "4-casualty", "Withdrawing >5 units into a besieged stronghold trims to 5 (excess to reinforcements)", page='house')
def cl_147():
    # 5-unit cap = rulebook p.31 (CL-068 covers setup trim); WHICH-box the
    # excess returns to is the USER 1.181 house ruling: REINFORCEMENTS, not
    # casualties (a computed pool, so an untracked removal IS a return).
    st = setup_game()
    a = army_at(st, 'minas tirith')
    a.units.clear()
    a.leaders.clear()
    for _ in range(7):
        a.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    a.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))   # besieger co-located
    st.besieged_strongholds.add('minas tirith')
    st.fp_casualties = {}
    c = cb.setup_combat(st, 'minas tirith', 'minas tirith')
    garrison = c.defender_army if c.defender_side == Side.FREE_PEOPLES else c.attacker_army
    assert len(garrison.units) == 5, f"garrison trimmed to 5, got {len(garrison.units)}"
    assert not st.fp_casualties.get(Nation.GONDOR), \
        "the 2 excess units go to REINFORCEMENTS, not the casualties box (neg)"
    # B2 (2026-08-24): the MERGE-BACK half of the same rule.  A full
    # battle used to book the 2 cap-removed units as FP casualties in
    # _publish_fp_casualties, contradicting the 1.181 ruling this
    # clause pins.  Post-battle, the casualty box must hold ONLY the
    # battle's genuine FP casualties (all-Regular garrison: booking is
    # 1:1 with the result's casualty lists), never the cap excess.
    st2 = setup_game()
    a2 = army_at(st2, 'minas tirith')
    a2.units.clear()
    a2.leaders.clear()
    for _ in range(7):
        a2.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    a2.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    st2.besieged_strongholds.add('minas tirith')
    st2.fp_casualties = {}
    import random as _rnd147
    _rnd147.seed(147)
    res = cb.resolve_combat(st2, 'minas tirith', 'minas tirith',
                            is_siege=True, max_rounds=1)
    _genuine = sum(1 for u in (list(res.attacker_casualties)
                               + list(res.defender_casualties))
                   if getattr(u, 'nation', None) == Nation.GONDOR)
    _boxed = st2.fp_casualties.get(Nation.GONDOR, {})
    _boxed_n = sum(_boxed.values()) if isinstance(_boxed, dict) else 0
    assert _boxed_n == _genuine, \
        ("cap excess leaked into the casualty box: boxed=%d genuine=%d"
         % (_boxed_n, _genuine))


# ===========================================================================
# CHAPTER 5b - Politics track & activation  (rulebook p.34-35)
# ===========================================================================

@clause("CL-150", "5-politics", "An attacked nation activates AND advances one step (not past At War)", page=35)
def cl_150():
    # rulebook p.35 (advance_defenders_political docstring).
    st = setup_game()
    assert not st.political.is_active(Nation.ROHAN) and st.political.boxes[Nation.ROHAN] == 0
    place(st, 'fords of isen', Nation.ROHAN, regulars=2, clear=True)
    game_logic.advance_defenders_political(st, 'fords of isen', attacker_side=Side.SHADOW)
    assert st.political.is_active(Nation.ROHAN), "the attacked nation activates"
    assert st.political.boxes[Nation.ROHAN] == 1, "and advances exactly one step"
    # already At War (box 3) does not advance past box 3 (neg)
    st.political.boxes[Nation.ROHAN] = 3
    game_logic.advance_defenders_political(st, 'fords of isen', attacker_side=Side.SHADOW)
    assert st.political.boxes[Nation.ROHAN] == 3, "cannot advance past At War (neg)"


@clause("CL-151", "5-politics", "Capturing a nation's settlement activates it and advances one step", page=35)
def cl_151():
    st = setup_game()
    b0 = st.political.boxes[Nation.GONDOR]
    game_logic.activate_nation_for_attack(st, 'minas tirith', captured=True)
    assert st.political.is_active(Nation.GONDOR), "capture activates the owning nation"
    assert st.political.boxes[Nation.GONDOR] == b0 + 1, "capture advances one political step"


@clause("CL-152", "5-politics", "Political track: At War = box 3; advancing stops at 3; below 3 is not At War (neg)", page=(34, 35))
def cl_152():
    st = setup_game()
    st.political.active[Nation.ROHAN] = True
    st.political.boxes[Nation.ROHAN] = 2
    assert not st.political.is_at_war(Nation.ROHAN), "box 2 is not At War (neg)"
    game_logic.advance_political(st, Nation.ROHAN, method='muster_die')
    assert st.political.boxes[Nation.ROHAN] == 3 and st.political.is_at_war(Nation.ROHAN), \
        "box 3 is At War"
    game_logic.advance_political(st, Nation.ROHAN, method='muster_die')
    assert st.political.boxes[Nation.ROHAN] == 3, "the track never advances past box 3 (neg)"


# ===========================================================================
# CHAPTER 2b - Fellowship movement & the Guide  (rulebook p.37)
# ===========================================================================

@clause("CL-175", "2-fellow", "A Fellowship move advances the Progress counter by exactly 1 (outside Mordor)", page=37)
def cl_175():
    # rb p.37 (Fellowship movement / the Guide), between the declaration
    # anchor p.36 and the hide/reveal anchor p.38.
    st = setup_game()
    assert not st.fellowship.mordor_entered
    p0 = st.fellowship.progress
    game_logic.move_fellowship(st, skip_hunt=True)
    assert st.fellowship.progress == p0 + 1, "one move = +1 Progress"
    game_logic.move_fellowship(st, skip_hunt=True)
    assert st.fellowship.progress == p0 + 2, "a second move = +1 more Progress"


@clause("CL-176", "2-fellow", "The Guide is a Companion in the Fellowship (default the highest level)", page=37)
def cl_176():
    st = setup_game()
    guide = st.fellowship.guide
    assert guide is not None and guide in st.fellowship.companions, \
        "the Guide is one of the Fellowship's Companions"
    top = max(c.level for c in st.fellowship.companions)
    assert guide.level == top, f"default Guide is a highest-level Companion (lvl {top})"


# ===========================================================================
# CHAPTER 3b - Advanced Hunt & the Mordor track  (rulebook p.41-44)
# ===========================================================================
# The exact page within the advanced-hunt band is not individually pinned;
# data/rulebook_page_index.md records "advanced/special Hunt rules (p.41-44)",
# so these rulebook clauses are anchored to the (41,44) band (honest range,
# not a false-precise single page).

@clause("CL-180", "3-hunt", "On the Mordor track an Eye tile's damage = dice in the Hunt Box; off it = successes", page=(41, 44))
def cl_180():
    st = setup_game()
    for _ in range(3):
        d = gs.ActionDie(faces=[Face.EYE], side=Side.SHADOW)
        d.current = Face.EYE
        d.used = True
        st.hunt_box.append(d)
    st.fellowship.mordor_entered = True
    st.fellowship.progress = 0
    st._pending_hunt_damage = 0
    game_logic._apply_hunt_tile(st, gs.HuntTile(is_eye=True), successes=1, mordor=True)
    assert st._pending_hunt_damage == 3, \
        f"Mordor Eye damage = 3 (hunt-box dice), not 1 success; got {st._pending_hunt_damage}"
    st2 = setup_game()
    st2._pending_hunt_damage = 0
    game_logic._apply_hunt_tile(st2, gs.HuntTile(is_eye=True), successes=2, mordor=False)
    assert st2._pending_hunt_damage == 2, "off the track an Eye tile deals `successes` (neg: not box count)"


@clause("CL-181", "3-hunt", "Tiles reveal the Fellowship on the Mordor track too (reveal tile flips hidden)", page=(41, 44))
def cl_181():
    st = setup_game()
    st.fellowship.mordor_entered = True
    st.fellowship.progress = 0
    st.fellowship.hidden = True
    st._pending_hunt_damage = 0
    game_logic._apply_hunt_tile(st, gs.HuntTile(damage=1, reveal=True), successes=1, mordor=True)
    assert st.fellowship.hidden is False, "a reveal tile reveals the Fellowship on the track"
    # a non-reveal damage tile does NOT reveal (neg)
    st.fellowship.hidden = True
    game_logic._apply_hunt_tile(st, gs.HuntTile(damage=2, reveal=False), successes=1, mordor=True)
    assert st.fellowship.hidden is True, "a plain damage tile does not reveal (neg)"


@clause("CL-182", "3-hunt", "Gollum guide blocks reveals on ordinary damage tiles, but Eyes reveal through him", page='house')
def cl_182():
    # USER 2026-07-11 house ruling (CURRENT_RULES s1): Gollum's guide
    # ability blocks ordinary damage-tile reveals; Eyes/specials reveal.
    st = setup_game()
    gollum = gs.Companion(name='Gollum', level=0, leadership=0,
                          guide_ability='no_reveal_on_damage')
    st.fellowship.guide = gollum
    st.fellowship.hidden = True
    st.fellowship.progress = 0
    st._pending_hunt_damage = 0
    game_logic._apply_hunt_tile(st, gs.HuntTile(damage=1, reveal=True), successes=1, mordor=False)
    assert st.fellowship.hidden is True, "Gollum blocks an ordinary damage-tile reveal (neg reveal)"
    st.fellowship.hidden = True
    game_logic._apply_hunt_tile(st, gs.HuntTile(is_eye=True), successes=1, mordor=False)
    assert st.fellowship.hidden is False, "an Eye tile reveals through Gollum"


@clause("CL-183", "3-hunt", "An empty Hunt Pool refills from the used pile (special tiles stay out pre-Mordor)", page='house')
def cl_183():
    # USER 1.136 house ruling: an empty pool returns ALL used non-special
    # tiles (special/Mordor tiles stay out until Mordor entry).
    st = setup_game()
    st.hunt_pool.clear()
    st.hunt_discard[:] = [gs.HuntTile(damage=1), gs.HuntTile(damage=2, reveal=True),
                          gs.HuntTile(is_special=True, special_name='Shelob')]
    n = game_logic._recycle_hunt_pool(st)
    assert n == 2, f"2 ordinary used tiles recycle, got {n}"
    assert len(st.hunt_pool) == 2, "the pool is refilled from the used pile"
    assert any(getattr(t, 'is_special', False) for t in st.hunt_discard), \
        "a special tile stays OUT until Mordor entry (neg: not recycled)"


# ===========================================================================
# CHAPTER 7b - Character special abilities  (rulebook p.23)
# ===========================================================================
# p.23 matches the Characters-chapter placement inferred in
# data/rulebook_page_index.md (leaders overview is the established p.19-20).

@clause("CL-190", "7-chars", "Gandalf the White nullifies ALL Nazgul leadership (incl. the Witch-king) in his battle", page=23)
def cl_190():
    st = setup_game()
    a = gs.Army(region='x')
    a.leaders.append(gs.Nazgul(Nation.SAURON))                             # 1
    a.leaders.append(game_logic.Minion(Nation.SAURON, name='Witch-king of Angmar'))  # 2
    a.leaders.append(gs.Leader(Nation.SAURON))                             # 1 (non-nazgul)
    assert cb.calc_leadership(a) == 4, "normal leadership 1+2+1 = 4"
    assert cb.calc_leadership(a, cancel_nazgul=True) == 1, \
        "Gandalf-White cancels the Nazgul (1) and Witch-king (2); only the plain leader (1) remains"


@clause("CL-191", "7-chars", "Gandalf the White carries leadership 1 and no Captain-of-the-West bonus", page=23)
def cl_191():
    st = setup_game()
    g = st.gandalf_the_white
    assert g.leadership == 1, f"Gandalf the White leadership 1, got {g.leadership}"
    assert g.captain_of_the_west is False, "Gandalf the White is not a Captain of the West (neg)"


# ===========================================================================
# CHAPTER 6b - Elven Rings, negative face rules  (rulebook p.18)
# ===========================================================================

@clause("CL-195", "6-cards", "An Elven Ring may NOT change a die to Will of the West (neg)", page=18)
def cl_195():
    # rb p.18 (Elven Rings): any result EXCEPT Will of the West; F-4 (1.178a).
    st = setup_game()
    st.dice[Side.FREE_PEOPLES] = [gs.ActionDie(faces=[Face.MUSTER], side=Side.FREE_PEOPLES)]
    st.dice[Side.FREE_PEOPLES][0].current = Face.MUSTER
    raised = False
    try:
        game_logic.use_elven_ring(st, Side.FREE_PEOPLES, 0, Face.WILL_OF_WEST)
    except RuntimeError:
        raised = True
    assert raised, "the ring must refuse a change to Will of the West (neg)"
    assert st.dice[Side.FREE_PEOPLES][0].current == Face.MUSTER, "the die face is unchanged"


@clause("CL-196", "6-cards", "The Shadow Elven Ring cannot change an Eye die, but MAY change a die to an Eye", page=18)
def cl_196():
    st = setup_game()
    st.sh_elven_rings = 1
    st.dice[Side.SHADOW] = [gs.ActionDie(faces=[Face.EYE], side=Side.SHADOW)]
    st.dice[Side.SHADOW][0].current = Face.EYE
    raised = False
    try:
        game_logic.use_elven_ring(st, Side.SHADOW, 0, Face.ARMY)
    except RuntimeError:
        raised = True
    assert raised and st.dice[Side.SHADOW][0].current == Face.EYE, \
        "an already-Eye die cannot be changed (neg)"
    # but the Shadow ring MAY convert a spendable die INTO an Eye
    st2 = setup_game()
    st2.sh_elven_rings = 1
    st2.dice[Side.SHADOW] = [gs.ActionDie(faces=[Face.ARMY], side=Side.SHADOW)]
    st2.dice[Side.SHADOW][0].current = Face.ARMY
    game_logic.use_elven_ring(st2, Side.SHADOW, 0, Face.EYE)
    assert st2.dice[Side.SHADOW][0].current == Face.EYE, "Shadow may change a die TO an Eye"


@clause("CL-197", "5-hunt", "An Eye tile drawn by a card-forced draw has NO effect (no reveal)", page=35)
def cl_197():
    # USER RULING 2026-07-20 (w5_g2 T7): Foul Thing / Isildur's Bane /
    # Orc Patrol / stronghold extra tile — a drawn Eye does nothing.
    import event_effects as ee
    from game_state import HuntTile
    st = setup_game()
    st.fellowship.hidden = True
    eye = HuntTile(damage=0, reveal=True, is_eye=True)
    st.hunt_pool = [eye]
    ee._effect_foul_thing(st, None, {})
    assert st.fellowship.hidden is True, \
        "card-drawn Eye must NOT reveal the fellowship"
    # positive companion: a non-eye reveal tile via the card DOES reveal
    st2 = setup_game()
    st2.fellowship.hidden = True
    st2.hunt_pool = [HuntTile(damage=0, reveal=True, is_eye=False)]
    ee._effect_foul_thing(st2, None, {})
    assert st2.fellowship.hidden is False, \
        "a non-eye reveal tile drawn by the card still reveals (pos)"


@clause("CL-198", "9-reveal", "Revealed AT a Mordor gateway: pawn stays; never region-steps into the interior", page=42)
def cl_198():
    # USER RULING 2026-07-20b (w5_g2 T9 Gorgoroth): reveal placement
    # from a gateway must not put the pawn in gorgoroth/barad dur/nurn;
    # it stays at the gate (progress resets) and enters via the track.
    import combat as cb9
    assert cb9.FINAL_HITS_HOOK is None, "self-play mode expected"
    for gate, prog in (('minas morgul', 1), ('morannon', 2)):
        st = setup_game()
        st.fellowship.region = st.board.region(gate)
        st.fellowship.progress = prog
        st.fellowship.hidden = False
        game_logic._auto_reveal_place(st)
        end = sanitize(game_logic.fellowship_region(st))
        assert end == sanitize(gate), \
            f"revealed at {gate}: pawn must stay (got {end})"
        assert st.fellowship.progress == 0, "progress resets on reveal"


@clause("CL-199", "9-reveal", "The fellowship can never DECLARE into Mordor's interior (neg)", page=42)
def cl_199():
    st = setup_game()
    st.fellowship.region = st.board.region('minas morgul')
    st.fellowship.progress = 1
    st.fellowship.hidden = True
    raised = False
    try:
        game_logic.declare_fellowship(st, region='gorgoroth')
    except RuntimeError:
        raised = True
    assert raised, "declare into gorgoroth must be rejected (1.155)"
    # positive companion: declaring AT the gateway itself stays legal
    st2 = setup_game()
    st2.fellowship.region = st2.board.region('north ithilien')
    st2.fellowship.progress = 1
    st2.fellowship.hidden = True
    game_logic.declare_fellowship(st2, region='minas morgul')
    assert sanitize(game_logic.fellowship_region(st2)) == sanitize('minas morgul'), \
        "declaring AT a gateway is legal (pos)"


@clause("CL-200", "4-casualty", "A downgrade pulls the replacement Regular from the CASUALTY BOX first (the one return path)", page=30)
def cl_200():
    # USER RULING 2026-07-20d: "when you downgrade elites, you can pull
    # regulars back from casualties. that is the only time units in
    # casualties can come back (aside from Gandalf the White)."
    st = setup_game()
    st.fp_casualties = {Nation.GONDOR: {'regular': 1, 'elite': 0,
                                        'leader': 0}}
    army = gs.Army(region='minas tirith')
    army.units.append(gs.Unit(Nation.GONDOR, Kind.ELITE))
    prev = cb.set_active_state(st)
    try:
        cas = cb.apply_casualties(army, 1, 'preserve_units')
    finally:
        cb.set_active_state(prev)
    cb._track_fp_casualties(st, cas, Side.FREE_PEOPLES)
    box = st.fp_casualties[Nation.GONDOR]
    assert len(army.units) == 1 and army.units[0].kind == Kind.REGULAR, \
        "elite downgraded to a Regular"
    assert box['regular'] == 0, \
        "the replacement Regular came BACK from the casualty box"
    assert box['elite'] == 1, \
        "the downgraded elite is a permanent elite casualty"


@clause("CL-201", "4-casualty", "Anduril converts misses ONLY via Strider (1) or crowned Aragorn (2); no hero = no effect", page='card')
def cl_201():
    # CARD-FLEET LOW-1 (2026-07-20): the battle layer must self-enforce
    # the hero requirement — the old fall-through converted a miss off
    # ANY companion (Gimli paid Anduril's cost).
    from cards import get_card
    athelas = get_card('FREE_PEOPLES_Character_Athelas').to_combat_card()
    axebow = get_card('FREE_PEOPLES_Character_Axe_and_Bow').to_combat_card()

    def probe(card, comps, ldr):
        c = gs.CombatState(
            attacker_region='dagorlad', defender_region='osgiliath',
            attacker_army=gs.Army(region='dagorlad'),
            defender_army=gs.Army(region='osgiliath'),
            attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
            defender_card=card)
        c.companions_in_battle = comps
        c.defender_leadership = ldr
        auto = cb.apply_mighty_attack(c, [1, 2, 3], 5, 0,
                                      is_attacker=False)
        return auto, c.defender_leadership

    strider = gs.Companion(name='Strider', level=3, leadership=1)
    aragorn = gs.Companion(name='Aragorn', level=3, leadership=2)
    gimli = gs.Companion(name='Gimli', level=2, leadership=1)
    a, l = probe(athelas, [strider], 1)
    assert (a, l) == (1, 0), "Strider converts exactly 1 (P1): %s" % ((a, l),)
    a, l = probe(athelas, [aragorn], 2)
    assert (a, l) == (2, 0), "crowned Aragorn converts up to 2 (P2): %s" % ((a, l),)
    a, l = probe(athelas, [gimli], 1)
    assert (a, l) == (0, 1), "no hero -> Anduril NO effect, no forfeit (N1): %s" % ((a, l),)
    a, l = probe(axebow, [gimli], 1)
    assert (a, l) == (1, 0), "generic Mighty Attack unchanged (R1): %s" % ((a, l),)


@clause("CL-202", "6-cards", "Mirror of Galadriel heals at uncaptured or BESIEGED Lorien; only CAPTURED blocks", page='card')
def cl_202():
    # Almanac (Mirror_of_Galadriel.md:21): "possible to heal at Lorien
    # if it is under siege, but not if Lorien is captured." Locks the
    # LOW-10 adjudication incl. the siege-vs-captured precision.
    import event_effects as ee
    from cards import get_card
    cid = next(c for c in
               ('FREE_PEOPLES_Character_Mirror_of_Galadriel',)
               if get_card(c))

    def probe(captured, besieged):
        st = setup_game()
        st.fellowship.region = st.board.region('lorien')
        st.fellowship.corruption = 5
        if captured:
            st.enemy_controlled[Side.SHADOW].add(sanitize('lorien'))
        if besieged:
            st.besieged_strongholds.add(sanitize('lorien'))
        st.hands[Side.FREE_PEOPLES].hand.append(cid)
        ee.play_event_card(st, Side.FREE_PEOPLES, cid)
        return st.fellowship.corruption

    assert probe(False, False) == 4, "uncaptured Lorien heals 1"
    assert probe(True, False) == 5, "CAPTURED Lorien does not heal (neg)"
    assert probe(False, True) == 4, "BESIEGED (not captured) Lorien still heals"


@clause("CL-203", "6-cards", "Gwaihir: a no-op play refunds the card (self-play); teacher still consumes", page='card')
def cl_203():
    # CARD-FLEET LOW-6 (2026-07-21)
    import ai as ai_m
    import event_effects as ee
    cid = 'FREE_PEOPLES_Character_Gwaihir_the_Windlord'

    def play(choices, hook):
        st = setup_game()
        st.hands[Side.FREE_PEOPLES].hand.append(cid)
        # (hook fixture collapsed 2026-07-27 — teacher retired)
        res = ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                                     choices=choices)
        return st, res

    # negative (self-play): out-of-range dest -> refund, card kept
    st, res = play({'companion_action': 'sep_Legolas',
                    'destination': 'dol amroth'}, None)
    assert res.get('refund') is True and res.get('success') is False, \
        "no-op play must refund (self-play)"
    assert cid in st.hands[Side.FREE_PEOPLES].hand, "card stays in hand"
    assert any(c.name == 'Legolas' for c in st.fellowship.companions), \
        "Legolas not separated"
    # (teacher-half removed 2026-07-27 — teacher retired)
    # positive (self-play): in-range separation works and consumes
    st, res = play({'companion_action': 'sep_Legolas',
                    'destination': 'hollin'}, None)
    assert res.get('success') and not res.get('refund')
    assert not any(c.name == 'Legolas'
                   for c in st.fellowship.companions), "Legolas separated"


@clause("CL-204", "6-cards", "I Will Go Alone: mandatory separation - no heal-only leak (self-play)", page='card')
def cl_204():
    # CARD-FLEET LOW-8 (2026-07-21)
    import ai as ai_m
    import event_effects as ee
    cid = 'FREE_PEOPLES_Character_I_Will_Go_Alone'

    def play(choices, hook):
        st = setup_game()
        st.fellowship.corruption = 3
        st.hands[Side.FREE_PEOPLES].hand.append(cid)
        # (hook fixture collapsed 2026-07-27 — teacher retired)
        res = ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                                     choices=choices)
        return st, res

    # negative empty choices (self-play): refund, NO heal
    st, res = play({}, None)
    assert res.get('refund') is True and st.fellowship.corruption == 3, \
        "empty choices: no heal, refunded"
    assert cid in st.hands[Side.FREE_PEOPLES].hand
    # negative out-of-range (self-play): refund, NO heal
    st, res = play({'companions': ['Legolas'],
                    'destination': 'dol amroth'}, None)
    assert res.get('refund') is True and st.fellowship.corruption == 3, \
        "out-of-range: no heal, refunded"
    # (teacher-half removed 2026-07-27 — teacher retired)
    # positive (self-play): valid separation + heal + consume
    st, res = play({'companions': ['Legolas'],
                    'destination': 'hollin'}, None)
    assert res.get('success') and st.fellowship.corruption == 2
    assert not any(c.name == 'Legolas'
                   for c in st.fellowship.companions)


@clause("CL-205", "5-hunt", "Axe&Bow/Horn hunt-damage discard is a LEARNED decision (use reduces 1; decline keeps card)", page='card')
def cl_205():
    # USER RULING 2026-07-21: "keep axe and bow and horn of gondor to
    # be learned decisions" — rides pending_card_option (no encoding
    # change; get_feature_dim() stability asserted).
    import combat as cb9
    assert cb9.FINAL_HITS_HOOK is None, "self-play mode expected"
    dim0 = ai.get_feature_dim()

    def setup():
        st = setup_game()
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='FREE_PEOPLES_Character_Axe_and_Bow',
            card_name='Axe and Bow',
            card_type=gs.PersistentCardType.AXE_AND_BOW,
            side=Side.FREE_PEOPLES))
        st.fellowship.companions = [c for c in st.fellowship.companions
                                    if c.name in ('Gollum',)]
        st.fellowship.corruption = 0
        st._pending_hunt_damage = 2
        return st

    # offer opens with the right shape
    st = setup()
    assert game_logic._offer_hunt_reduction(st) is True
    pco = st.pending_card_option
    assert pco and pco.get('hunt_reduction') and pco['decider'] == 'fp'
    assert len(pco['options']) == 2
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    assert len(acts) == 2 and all(a.action_type == 'card_option'
                                  for a in acts), \
        "interpose must expose exactly the use/decline pair"
    # USE: card discarded, damage reduced to 1, settled as corruption
    use = next(a for a in acts
               if a.params['choices'].get('use_hunt_reduction'))
    ai._execute_action(st, use)
    assert st.persistent_cards.get_by_type(
        gs.PersistentCardType.AXE_AND_BOW) is None, "card discarded"
    assert st.fellowship.corruption == 1, \
        "2 damage - 1 reduction = 1 corruption (no companions)"
    assert st._pending_hunt_damage == 0
    # DECLINE: card stays, full damage confronted
    st2 = setup()
    game_logic._offer_hunt_reduction(st2)
    acts2 = ai.get_legal_actions(st2, Side.FREE_PEOPLES)
    dec = next(a for a in acts2
               if not a.params['choices'].get('use_hunt_reduction'))
    ai._execute_action(st2, dec)
    assert st2.persistent_cards.get_by_type(
        gs.PersistentCardType.AXE_AND_BOW) is not None, "card kept"
    assert st2.fellowship.corruption == 2, "full 2 damage taken"
    # no card on table -> no interpose (negative)
    st3 = setup_game()
    st3._pending_hunt_damage = 1
    assert game_logic._offer_hunt_reduction(st3) is False
    # encoding freeze intact
    assert ai.get_feature_dim() == dim0, "encoding dim must not move"


@clause("CL-206", "5-hunt", "Eye-tile hunt damage (= successes) feeds the Axe&Bow/Horn reduction decision", page='card')
def cl_206():
    # USER RULING 2026-07-21b: "you can use axe and bow / horn of
    # gondor on eye tiles." In a NORMAL hunt an Eye tile deals damage
    # equal to the hunt successes; that damage must open the learned
    # reduction interpose exactly like numbered tiles. (Card-forced
    # Eye draws remain no-effect — no hunt roll, no successes.)
    from game_state import HuntTile
    st = setup_game()
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_PEOPLES_Character_Horn_of_Gondor',
        card_name='Horn of Gondor',
        card_type=gs.PersistentCardType.HORN_OF_GONDOR,
        side=Side.FREE_PEOPLES))
    eye = HuntTile(damage=0, is_eye=True)
    game_logic._apply_hunt_tile(st, eye, 2, mordor=False)
    assert getattr(st, '_pending_hunt_damage', 0) == 2, \
        "Eye damage = hunt successes (2)"
    assert game_logic._offer_hunt_reduction(st) is True, \
        "Eye-tile damage opens the reduction decision"
    assert st.pending_card_option.get('hunt_reduction')


def _l3_play(cid, choices, hook, prep=None):
    import ai as ai_m
    import event_effects as ee
    st = setup_game()
    if prep:
        prep(st)
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    # (hook fixture collapsed 2026-07-27 — teacher retired)
    res = ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                             choices=choices)
    return st, res


def _l3_exhaust(st, nation, dump, leaders=False):
    import ai as ai_m
    mx = ai_m.NATION_MAX_FIGURES[nation]
    d = ai_m._count_deployed(st, nation)
    place(st, dump, nation,
          regulars=max(0, mx['regular'] - d.get('regular', 0)),
          elites=max(0, mx['elite'] - d.get('elite', 0)),
          leaders=(max(0, mx.get('leader', 4) - d.get('leader', 0))
                   if leaders else 0))


@clause("CL-207", "6-cards", "Recruit cards: honest per-piece reporting; Guards decouples leader; fully-blocked refunds (self-play)", page='card')
def cl_207():
    # CARD-FLEET LOW-24/27/29/30 + family sweep (2026-07-21)
    cid = 'FREE_PEOPLES_Strategy_Guards_of_the_Citadel'
    # unit pool exhausted, leader available -> leader STILL places
    st, res = _l3_play(cid, {}, None,
                       prep=lambda s: _l3_exhaust(s, Nation.GONDOR,
                                                  'pelargir'))
    mt = army_at(st, 'minas tirith')
    n_led = sum(1 for l in mt.leaders
                if getattr(l, 'nation', None) == Nation.GONDOR)
    assert res.get('success') and not res.get('refund'), \
        "partial success consumes"
    assert n_led >= 2, "entitled leader placed despite exhausted units"
    assert not any('and Leader' in e for e in res['effects']), \
        "per-piece strings only"
    # both blocked -> refund (self-play); teacher consumes
    def _both(s):
        _l3_exhaust(s, Nation.GONDOR, 'pelargir', leaders=True)
    st, res = _l3_play(cid, {}, None, prep=_both)
    # USER RULING 1.230 (2026-08-28): a fully-blocked recruit is a
    # completed ZERO-EFFECT play (card discarded, die spent), never a
    # refund - supersedes the 2026-07-21 self-play refund here
    assert res.get('success') and res.get('zero_effect') \
        and not res.get('refund') \
        and cid not in st.hands[Side.FREE_PEOPLES].hand, \
        "fully blocked completes as a zero-effect play (1.230)"
    # (teacher-half removed 2026-07-27 — teacher retired)


@clause("CL-208", "6-cards", "No-op plays refund (WPS/TABA/Cirdan/Faramir); revealed TIAW heals but never moves", page='card')
def cl_208():
    # CARD-FLEET LOW-18/19/20/25/28 (2026-07-21)
    from game_state import Companion as _Comp
    # WPS out-of-range -> refund; teacher consumes
    c20 = 'FREE_PEOPLES_Character_We_Prove_Swifter'
    st, res = _l3_play(c20, {'companion_action': 'sep_Legolas',
                             'destination': 'dol amroth'}, None)
    assert res.get('refund') and cid_in(st, c20)
    # (teacher-half removed 2026-07-27 — teacher retired)
    # TABA empty pick -> refund
    c18 = 'FREE_PEOPLES_Character_There_and_Back_Again'
    st, res = _l3_play(c18, {}, None)
    # USER RULING 1.230: an ABSENT choice while legal choices exist is
    # an invalid action (refund); with NO legal choice at all the play
    # completes as zero-effect.  Measured against the spec's own answer.
    from card_options import get_card_options as _gco208
    _s18 = setup_game()
    _s18.hands[Side.FREE_PEOPLES].hand.append(c18)
    if _gco208(_s18, Side.FREE_PEOPLES, 'There and Back Again'):
        assert res.get('refund') and cid_in(st, c18), \
            "absent choice with legal choices refunds"
    else:
        assert res.get('success') and res.get('zero_effect') \
            and not cid_in(st, c18), "no legal choice completes (1.230)"
    # Cirdan: no region -> refund; invalid target -> refund;
    # valid coastal FP-army region -> recruits (req: Elves At War)
    c25 = "FREE_PEOPLES_Strategy_Cirdan's_Ships"
    def _elves_war(s):
        s.political.boxes[Nation.ELVES] = 3
        s.political.active[Nation.ELVES] = True
    st, res = _l3_play(c25, {}, None, prep=_elves_war)
    assert res.get('refund') and cid_in(st, c25)
    st, res = _l3_play(c25, {'region': 'gorgoroth'}, None,
                       prep=_elves_war)
    assert res.get('refund'), "non-coastal target refunds"
    def _coast(s):
        _elves_war(s)
        place(s, 'pelargir', Nation.GONDOR, regulars=2, clear=True)
    st, res = _l3_play(c25, {'region': 'pelargir'}, None, prep=_coast)
    assert res.get('success') and not res.get('refund')
    # Faramir has neither a Shadow target nor the AL18 recruitment-only
    # condition. RB22 refuses the play; it is not a legal zero-effect action.
    c28 = "FREE_PEOPLES_Strategy_Faramir's_Rangers"
    def _strip(s):
        for k in list(s.armies):
            if sanitize(k) == 'osgiliath':
                s.armies.pop(k)
    st, res = _l3_play(c28, {'target_region': 'osgiliath'}, None,
                       prep=_strip)
    assert not res.get('success') and cid_in(st, c28), \
        "missing Faramir position refuses play and preserves the card"
    # TIAW revealed move — USER RE-RULING 2026-08-24 (amendment 1.229,
    # supersedes 21f): the heal ALWAYS applies, so a revealed play is
    # a REAL play (heal kept, illegal move skipped) — never a refund.
    # Full doctrine locked in CL-214; this leg just guards the
    # no-move-while-revealed half.
    c19 = 'FREE_PEOPLES_Character_There_is_Another_Way'
    def _gollum_rev(s):
        g = _Comp(name='Gollum', level=0)
        s.fellowship.companions.append(g)
        s.fellowship.guide = g
        s.fellowship.hidden = False
        s.fellowship.corruption = 2
    st, res = _l3_play(c19, {'action': 'move'}, None, prep=_gollum_rev)
    assert res.get('success') and not res.get('refund') \
        and st.fellowship.corruption == 1, "heal kept (1.229)"
    assert st.fellowship.progress == 0, "revealed fellowship never moves"


def cid_in(st, cid):
    return cid in st.hands[Side.FREE_PEOPLES].hand


@clause("CL-209", "6-cards", "Ents free-chain excludes the resolving card; another Character card still chains", page='card')
def cl_209():
    # CARD-FLEET LOW-13/15/17 (2026-07-21)
    import event_effects as ee
    from cards import get_card
    cid = 'FREE_PEOPLES_Character_The_Ents_Awake:_Treebeard'
    other = 'FREE_PEOPLES_Character_Athelas'
    st = setup_game()
    # GtW SEPARATED in Fangorn (the chain gate scans army leaders) +
    # the engine's dual-representation tracker (requirement reads it)
    fan = game_logic.get_or_create_army(st, 'fangorn')
    fan.leaders.append(gs.Companion(name='Gandalf the White', level=3,
                                    leadership=1))
    if getattr(st, 'gandalf_the_white', None) is not None:
        st.gandalf_the_white.in_play = True
    st.hands[Side.FREE_PEOPLES].hand[:] = [cid, other]
    ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                       choices={'_from_pending': True})
    pco = st.pending_card_option
    # USER RULING 1.254 (2026-08-29): when the roll hits the Orthanc
    # stack with a real allocation choice, the Shadow OWNER's casualty
    # allocation opens FIRST; the free chain resumes once it resolves
    # (event_effects._complete_card_casualty continuation).
    if pco is not None and pco.get('card_casualty'):
        st = ai.apply_action(st, ai.Action(
            die_idx=-1, action_type='card_option',
            params={'idx': 0, 'card_id': pco['card_id'],
                    'card_name': pco['card_name'], 'side': pco['side'],
                    'choices': pco['options'][0]}))
        pco = st.pending_card_option
    assert pco is not None, "chain pending opens with another card in hand"
    plays = [o.get('ents_chain_play') for o in pco['options']
             if 'ents_chain_play' in o]
    assert other in plays, "the OTHER Character card is offered"
    assert cid not in plays, "the resolving card must NOT offer itself"


@clause("CL-210", "6-cards", "House of the Stewards: Boromir must be SEPARATED in Gondor; parked Fellowship never counts", page='card')
def cl_210():
    # USER RULING 2026-07-21e
    from event_effects import check_event_requirement
    HOS = 'FREE_PEOPLES_Character_House_of_the_Stewards'
    st = setup_game()
    st.fellowship.region = st.board.region('minas tirith')
    if not any(c.name == 'Boromir' for c in st.fellowship.companions):
        st.fellowship.companions.append(
            gs.Companion(name='Boromir', level=2, leadership=1))
    assert not check_event_requirement(st, HOS)[0], \
        "parked Fellowship with Boromir inside must NOT satisfy (neg)"
    st2 = setup_game()
    st2.fellowship.companions = [c for c in st2.fellowship.companions
                                 if c.name != 'Boromir']
    army_at(st2, 'pelargir').leaders.append(
        gs.Companion(name='Boromir', level=2, leadership=1))
    assert check_event_requirement(st2, HOS)[0], \
        "separated Boromir in a Gondor region satisfies (pos)"


@clause("CL-211", "6-cards", "Ents Awake: companion must be SEPARATED in Fangorn; parked Fellowship never counts", page='card')
def cl_211():
    # USER RULING 2026-07-21e (also resolves the QA CASE9 premise)
    from event_effects import check_event_requirement
    ENTS = 'FREE_PEOPLES_Character_The_Ents_Awake:_Entmoot'
    st = setup_game()
    st.gandalf_the_white.in_play = True
    st.fellowship.region = st.board.region('fangorn')
    assert not check_event_requirement(st, ENTS)[0], \
        "parked Fellowship in Fangorn must NOT satisfy (neg)"
    st2 = setup_game()
    st2.gandalf_the_white.in_play = True
    army_at(st2, 'fangorn').leaders.append(
        gs.Companion(name='Gandalf the White', level=3, leadership=1))
    assert check_event_requirement(st2, ENTS)[0], \
        "GtW separated in Fangorn satisfies (pos)"


@clause("CL-212", "6-cards", "Guards over besieged-5: 3/2/1 MT + elite+leader -> overstack -> owner returns a regular -> 2/3/2", page='card')
def cl_212():
    # USER RULING 2026-07-21g, example 1 verbatim
    import event_effects as ee
    cid = 'FREE_PEOPLES_Strategy_Guards_of_the_Citadel'
    st = setup_game()
    for a in st.armies.values():
        a.units[:] = [u for u in a.units if u.nation != Nation.GONDOR]
        a.leaders[:] = [l for l in a.leaders
                        if getattr(l, 'nation', None) != Nation.GONDOR]
    place(st, 'minas tirith', Nation.GONDOR, regulars=3, elites=2,
          leaders=1, clear=True)
    place(st, 'minas tirith', Nation.SAURON, regulars=4)
    st.besieged_strongholds.add(sanitize('minas tirith'))
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    res = ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                             choices={'unit_type': 'elite',
                                      '_from_pending': True})
    assert res.get('success') and not res.get('refund'), \
        "must recruit over the besieged-5 cap, not refund: %s" % res
    pco = getattr(st, 'pending_card_option', None)
    assert pco and pco['card_name'] == '__overstack__', \
        "owner offered the overstack trim"
    pick = next(o for o in pco['options']
                if o['overstack_remove']['kind'] == 'regular')
    st.pending_card_option = None
    ee._effect_overstack(st, None, pick)
    a = army_at(st, 'minas tirith')
    r = sum(1 for u in a.units
            if u.nation == Nation.GONDOR and u.kind == Kind.REGULAR)
    e = sum(1 for u in a.units
            if u.nation == Nation.GONDOR and u.kind == Kind.ELITE)
    assert (r, e) == (2, 3), "3/2/1 +Guards elite -> 2/3/2: %s" % ((r, e),)


@clause("CL-213", "6-cards", "Cirdan over besieged-5 @ Dol Amroth: 5R + 2E -> owner returns 2 regulars -> 3/2; besieger ordering must not drop the target", page='card')
def cl_213():
    # USER RULING 2026-07-21g, example 2 verbatim (+ the units[0] bug)
    import event_effects as ee
    import card_options as co
    cid = "FREE_PEOPLES_Strategy_Cirdan's_Ships"
    st = setup_game()
    for a in st.armies.values():
        a.units[:] = [u for u in a.units if u.nation != Nation.ELVES]
    place(st, 'dol amroth', Nation.SAURON, regulars=4, clear=True)
    place(st, 'dol amroth', Nation.GONDOR, regulars=5)
    st.besieged_strongholds.add(sanitize('dol amroth'))
    st.political.boxes[Nation.ELVES] = 3
    st.political.active[Nation.ELVES] = True
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    opts = co.get_card_options(st, Side.FREE_PEOPLES, "Cirdan's Ships")
    assert {'region': 'dol amroth'} in opts, \
        "besieged Dol Amroth must stay a Cirdan target: %s" % opts
    res = ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                             choices={'region': 'Dol Amroth',
                                      'unit_combo': '2elite',
                                      '_from_pending': True})
    assert res.get('success') and not res.get('refund'), \
        "must recruit under siege, not refund: %s" % res
    for _ in range(3):
        pco = getattr(st, 'pending_card_option', None)
        if not (pco and pco['card_name'] == '__overstack__'):
            break
        pick = next(o for o in pco['options']
                    if o['overstack_remove']['kind'] == 'regular')
        st.pending_card_option = None
        ee._effect_overstack(st, None, pick)
    a = army_at(st, 'dol amroth')
    gr = sum(1 for u in a.units
             if u.nation == Nation.GONDOR and u.kind == Kind.REGULAR)
    el = sum(1 for u in a.units
             if u.nation == Nation.ELVES and u.kind == Kind.ELITE)
    fp = sum(1 for u in a.units if u.nation in ee.FP_NATIONS)
    assert (gr, el, fp) == (3, 2, 5), \
        "5R +Cirdan 2E -> 3/2 (5 FP total): %s" % ((gr, el, fp),)


@clause("CL-214", "6-cards", "There is Another Way: heal 1 ALWAYS; Gollum guide: move if hidden / HIDE if revealed (1.229-b), NO die to the Hunt Box", page='card')
def cl_214():
    # USER RE-RULING 2026-08-24 (supersedes 2026-07-21f exclusivity):
    # "applies the heal regardless, and the move conditional on
    # gollum. but not exclusive either or; if gollum is guide, you
    # can heal 1 + move in the same card. and it lets you move
    # without a die in the hunt box"
    import event_effects as ee
    import card_options as co
    cid = 'FREE_PEOPLES_Character_There_is_Another_Way'

    def play(gollum, hidden, choices, corr=3):
        st = setup_game()
        st.fellowship.corruption = corr
        st.fellowship.hidden = hidden
        if gollum:
            g = gs.Companion(name='Gollum', level=0, leadership=0,
                             guide_ability='no_reveal_on_damage')
            st.fellowship.companions.append(g)
            st.fellowship.guide = g
        st.hands[Side.FREE_PEOPLES].hand.append(cid)
        res = ee.play_event_card(st, Side.FREE_PEOPLES, cid,
                                 choices=choices)
        return st, res

    st, res = play(False, True, {})
    assert res.get('success') and st.fellowship.corruption == 2, \
        "no Gollum -> heal 1"
    # Gollum + hidden + move: heal AND move, and the Hunt Box stays
    # EMPTY (no die enters it for a card move)
    st, res = play(True, True, {'action': 'move'})
    assert res.get('success'), res
    assert st.fellowship.corruption == 2, \
        "Gollum move: the heal STILL applies (not either/or)"
    assert st.fellowship.progress == 1, \
        "Gollum move: the Fellowship moved (progress %d)" \
        % st.fellowship.progress
    assert not st.hunt_box, \
        "card move must put NO die in the Hunt Box: %r" % (
            st.hunt_box,)
    assert all(not d.used for d in st.dice[Side.FREE_PEOPLES]), \
        "the card move consumes no extra FP die"
    # heal_only decline: heal, no move
    st, res = play(True, True, {'action': 'heal_only'})
    assert res.get('success') and st.fellowship.corruption == 2 \
        and st.fellowship.progress == 0, "heal_only: heal, no move"
    # revealed: heal kept, move skipped (never a refund)
    st, res = play(True, False, {'action': 'move'})
    assert res.get('success') and not res.get('refund') \
        and st.fellowship.corruption == 2 \
        and st.fellowship.progress == 0, \
        "revealed -> heal kept, illegal move skipped"
    # 1.229-b (USER 2026-08-25, corrects 1.229's misrecorded
    # hide-removed clause): REVEALED + Gollum -> the HIDE option
    # ("move or hide depending on whether fellowship is revealed.
    # you always get the heal though")
    st, res = play(True, False, {'action': 'hide'})
    assert res.get('success') and st.fellowship.corruption == 2 \
        and st.fellowship.hidden is True, \
        "revealed + Gollum hide: heals AND hides (1.229-b)"
    # menus: Gollum+hidden = move vs heal_only; Gollum+revealed =
    # hide vs heal_only; no Gollum = no decision (direct heal)
    st2 = setup_game()
    g = gs.Companion(name='Gollum', level=0, leadership=0)
    st2.fellowship.companions.append(g)
    st2.fellowship.guide = g
    st2.fellowship.hidden = True
    o1 = co.get_card_options(st2, Side.FREE_PEOPLES,
                             'There is Another Way')
    assert o1 == [{'action': 'move'}, {'action': 'heal_only'}], o1
    st2.fellowship.hidden = False
    o2 = co.get_card_options(st2, Side.FREE_PEOPLES,
                             'There is Another Way')
    assert o2 == [{'action': 'hide'}, {'action': 'heal_only'}], \
        "revealed + Gollum -> hide vs heal_only (1.229-b): %r" % o2
    st3 = setup_game()
    st3.fellowship.hidden = False
    assert co.get_card_options(st3, Side.FREE_PEOPLES,
                               'There is Another Way') == [], \
        "no Gollum -> no decision (direct heal)"


@clause("CL-216", "4-casualty", "Words of Power / Black Breath resolvers self-gate on a Nazgul in battle; combat-roll bonuses do NOT reach the re-roll (1.208h supersedes 1.79); WoP also cancels the Captain COMBAT die + Gandalf's shine + the Mighty Attack/Anduril payer, and Foul Stench cancels Mighty Attack/Anduril (1.208i/j)", page='card')
def cl_216():
    # CARD-FLEET combat-scope cluster (2026-07-21): flags 5/34 fixed;
    # 1.208h REOPENED flags 1/2/3 as real bugs -- combat-roll bonuses are
    # FULLY SEPARATE from re-roll bonuses (SUPERSEDES 1.79).
    from cards import get_card
    wop = get_card('SHADOW_Character_The_Lidless_Eye').to_combat_card()
    comp = gs.Companion(name='Legolas', level=2, leadership=1)

    def wop_probe(with_nazgul):
        c = gs.CombatState(
            attacker_region='dagorlad', defender_region='osgiliath',
            attacker_army=gs.Army(region='dagorlad'),
            defender_army=gs.Army(region='osgiliath'),
            attacker_side=Side.SHADOW,
            defender_side=Side.FREE_PEOPLES,
            attacker_card=wop)
        c.companions_in_battle = [comp]
        c.defender_leadership = 1
        c.nazgul_in_battle = 1 if with_nazgul else 0
        cb.apply_words_of_power(c)
        return c.defender_leadership

    assert wop_probe(True) == 0, "with a Nazgul: leadership cancelled"
    assert wop_probe(False) == 1, \
        "NO Nazgul: resolver must not cancel (flags 5/34)"
    # 1.208h lock (supersedes 1.79): a COMBAT bonus (miss_modifier) sizes
    # the miss pool but must NOT boost the re-roll HIT. [3] at 5+ with
    # combat +1 misses (3+1=4 < 5) so it re-rolls; the re-rolled 4 uses
    # modifier=0 (reroll_mod) and stays a miss -> (1 die, 0 hits).
    _seq = iter([4])
    _orig_ri = random.randint
    random.randint = lambda a, b: next(_seq)
    try:
        rr, hits = cb.leader_reroll([3], 5, leadership=1,
                                    modifier=0, miss_modifier=1)
    finally:
        random.randint = _orig_ri
    assert (len(rr), hits) == (1, 0), \
        "combat +1 must NOT boost the re-roll hit (1.208h supersedes 1.79)"

    # (2) 1.208i: Words of Power cancels the chosen Companion's
    # Captain-of-the-West COMBAT die (pre-cap) AND his Leadership. 4 Gondor
    # regulars + Boromir = 4 units + 1 Captain = 5 dice; after WoP the
    # Captain die is dropped (5->4) and defender Leadership falls to 0.
    def_army = gs.Army(region='osgiliath')
    for _ in range(4):
        def_army.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    boromir = gs.Companion(name='Boromir', level=2, leadership=1)
    def_army.leaders.append(boromir)
    assert cb.calc_combat_strength(def_army) == 5, \
        "pre-WoP: 4R + Boromir's Captain die = 5"
    c2 = gs.CombatState(
        attacker_region='dagorlad', defender_region='osgiliath',
        attacker_army=gs.Army(region='dagorlad'),
        defender_army=def_army,
        attacker_side=Side.SHADOW,
        defender_side=Side.FREE_PEOPLES,
        attacker_card=wop)
    c2.companions_in_battle = [boromir]
    c2.defender_leadership = 1
    c2.nazgul_in_battle = 1
    cb.apply_words_of_power(c2)
    assert c2._wop_cancelled is boromir, "WoP marks the cancelled Captain"
    assert cb.calc_combat_strength(
        def_army, cancelled_captain=c2._wop_cancelled) == 4, \
        "1.208i: Captain COMBAT die dropped pre-cap (5->4)"
    assert c2.defender_leadership == 0, "WoP also cancels his Leadership"

    # (3) 1.208i(c): Words of Power on a SHINING Gandalf the White cancels
    # the shine -- the Nazgul Leadership the shine removed is RESTORED to
    # the Shadow side -- and cancels his own Leadership. Fixtures: shine had
    # removed 1 Nazgul Leadership; Gandalf's Leadership (3, synthetic) is
    # still on the FP side (legacy-default shine keeps it, so _shined_
    # companion is left unset and the (a)-subtract lands it on 0).
    gtw = gs.Companion(name='Gandalf the White', level=3, leadership=3)
    c3 = gs.CombatState(
        attacker_region='dagorlad', defender_region='osgiliath',
        attacker_army=gs.Army(region='dagorlad'),
        defender_army=gs.Army(region='osgiliath'),
        attacker_side=Side.SHADOW,
        defender_side=Side.FREE_PEOPLES,
        attacker_card=wop)
    c3.companions_in_battle = [gtw]
    c3._shine_companion = gtw
    c3._shine_cancelled_leadership = 1
    c3.attacker_leadership = 2
    c3.defender_leadership = 3
    c3.nazgul_in_battle = 1
    cb.apply_words_of_power(c3)
    assert c3.attacker_leadership == 3, \
        "1.208i(c): shine cancelled -> Nazgul Leadership restored to Shadow"
    assert c3.defender_leadership == 0, \
        "1.208i: Gandalf's own Leadership cancelled"

    # (4) 1.208i(c): a Companion cancelled by Words of Power this round may
    # not pay Mighty Attack / Anduril. Simulate WoP already applied by
    # setting _wop_cancelled to the sole companion.
    athelas = get_card('FREE_PEOPLES_Character_Athelas').to_combat_card()
    axebow = get_card('FREE_PEOPLES_Character_Axe_and_Bow').to_combat_card()

    def wop_payer(card, comp):
        c = gs.CombatState(
            attacker_region='dagorlad', defender_region='osgiliath',
            attacker_army=gs.Army(region='dagorlad'),
            defender_army=gs.Army(region='osgiliath'),
            attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
            defender_card=card)
        c.companions_in_battle = [comp]
        c.defender_leadership = getattr(comp, 'leadership', 1)
        c._wop_cancelled = comp
        return cb.apply_mighty_attack(c, [1, 2, 3], 5, 0, is_attacker=False)

    aragorn = gs.Companion(name='Aragorn', level=3, leadership=2)
    gimli = gs.Companion(name='Gimli', level=2, leadership=1)
    assert wop_payer(athelas, aragorn) == 0, \
        "1.208i: WoP-cancelled Aragorn cannot pay Anduril"
    assert wop_payer(axebow, gimli) == 0, \
        "1.208i: WoP-cancelled Gimli cannot pay generic Mighty Attack"

    # (5) 1.208j: Foul Stench also cancels that side's Mighty Attack /
    # Anduril. The real carrier's combat_effect must match the resolver's
    # trigger text.
    stench = get_card(
        'SHADOW_Character_The_Black_Captain_Commands').to_combat_card()
    assert "leader re-roll is cancelled" in stench.combat_effect.lower(), \
        "Foul Stench carrier text sanity"
    aragorn5 = gs.Companion(name='Aragorn', level=3, leadership=2)
    c5 = gs.CombatState(
        attacker_region='dagorlad', defender_region='osgiliath',
        attacker_army=gs.Army(region='dagorlad'),
        defender_army=gs.Army(region='osgiliath'),
        attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
        attacker_card=stench,
        defender_card=athelas)   # FP holds Anduril, would convert a miss
    c5.companions_in_battle = [aragorn5]
    c5.defender_leadership = 1
    c5.nazgul_in_battle = 2
    cb.apply_reroll_cancellation(c5)
    assert c5.defender_reroll_cancelled, "Foul Stench cancels the FP re-roll"
    assert getattr(c5, '_foul_stench_cancel_def', False), \
        "1.208j: Foul Stench raises the MA/Anduril block flag"
    assert cb.apply_mighty_attack(
        c5, [1, 2, 3], 5, 0, is_attacker=False) == 0, \
        "1.208j: Foul Stench blocks Anduril too"


@clause("CL-217", "6-cards", "Corruption clamps at 12; Wisdom of Elrond refuses a non-FP nation (self-play refund, teacher consumes); played combat/table cards route to the face-up discard pile", page='card')
def cl_217():
    # CARD-FLEET guards+clamps+discards cluster (2026-07-21)
    import ai as ai_m
    import event_effects as ee
    from persistent_effects import discard_table_card

    # (a) corruption ceiling: 11 + Candles all-hits -> clamp at 12 exactly
    st = setup_game()
    st.fellowship.corruption = 11
    _orig_d6 = ee._d6_batch
    ee._d6_batch = lambda n, *a, **k: [6] * n
    try:
        ee._effect_candles_of_corpses(st, None, {})
    finally:
        ee._d6_batch = _orig_d6
    assert st.fellowship.corruption == 12, \
        "corruption clamps at 12: %s" % st.fellowship.corruption

    # (b) Wisdom of Elrond: a non-FP nation is refused (self-play refund),
    #     an FP nation advances + consumes, and the teacher never refunds
    cid_woe = 'FREE_PEOPLES_Strategy_Wisdom_of_Elrond'

    def play_woe(nation, hook):
        s = setup_game()
        s.hands[Side.FREE_PEOPLES].hand.append(cid_woe)
        # (hook fixture collapsed 2026-07-27 — teacher retired)
        r = ee.play_event_card(s, Side.FREE_PEOPLES, cid_woe,
                               choices={'nation': nation})
        return s, r

    base = setup_game()
    base_rohan = base.political.boxes.get(Nation.ROHAN)
    base_sauron = base.political.boxes.get(Nation.SAURON)
    # self-play, Sauron (not an FP nation) -> refund, card kept, no advance
    s, r = play_woe('sauron', None)
    assert r.get('refund') is True and r.get('success') is False, \
        "non-FP nation must refund (self-play)"
    assert cid_woe in s.hands[Side.FREE_PEOPLES].hand, "card kept"
    assert s.political.boxes.get(Nation.SAURON) == base_sauron, \
        "SAURON political unchanged on refund"
    # self-play, Rohan (control) -> advances one step and consumes
    s, r = play_woe('rohan', None)
    assert not r.get('refund') and cid_woe not in \
        s.hands[Side.FREE_PEOPLES].hand, "FP nation consumes the card"
    assert s.political.boxes.get(Nation.ROHAN) == base_rohan + 1, \
        "Rohan advanced one step"
    # (teacher-half removed 2026-07-27 — teacher retired)

    # (c) a played combat card leaves the hand for its FACE-UP discard
    #     pile; the draw decks are untouched
    cid_cc = 'FREE_PEOPLES_Character_Elven_Cloaks'
    st = setup_game()
    st.hands[Side.FREE_PEOPLES].hand.append(cid_cc)
    decks_before = (st.decks.fp_character, st.decks.fp_strategy,
                    st.decks.sh_character, st.decks.sh_strategy)
    st.hands[Side.FREE_PEOPLES].hand.remove(cid_cc)
    ai_m._discard_played_combat_card(st, cid_cc)
    decks_after = (st.decks.fp_character, st.decks.fp_strategy,
                   st.decks.sh_character, st.decks.sh_strategy)
    assert cid_cc in st.discard_face_up['fp_character'], \
        "played FP Character combat card goes to fp_character discard"
    assert cid_cc not in st.hands[Side.FREE_PEOPLES].hand, "absent from hand"
    assert decks_before == decks_after, "draw decks unchanged by a discard"
    # a table (persistent) card removed from play routes to the same pile
    st2 = setup_game()
    wt = gs.PersistentCard(card_id='SHADOW_Character_Wormtongue',
                           card_name='Wormtongue', side=Side.SHADOW)
    discard_table_card(st2, wt)
    assert 'SHADOW_Character_Wormtongue' in \
        st2.discard_face_up['sh_character'], "table card -> sh_character discard"


@clause("CL-218", "6-cards", "Honest-reporting lock: Olog-hai empty pool / The Shadow Lengthens no-move refund on self-play (teacher consumes); Riders of Theoden non-Rohan target refunds (self-play), teacher consumes, Edoras is legal", page='card')
def cl_218():
    # honest-reporting wave (2026-07-21): per-piece recruit strings +
    # _selfplay_refund on no-op/all-blocked exits + self-play target guards
    import ai as ai_m
    import event_effects as ee

    def play(side, cid, choices, hook, prep=None):
        st = setup_game()
        if prep:
            prep(st)
        st.hands[side].hand.append(cid)
        # (hook fixture collapsed 2026-07-27 — teacher retired)
        res = ee.play_event_card(st, side, cid, choices=choices)
        return st, res

    # --- Olog-hai: Sauron pool empty -> honest refund (self-play) ---
    olog = 'SHADOW_Strategy_Olog-hai'
    assert get_card(olog) is not None, "Olog-hai card id resolves"

    def _exhaust_sauron(s):
        # exactly NATION_MAX_FIGURES (36R + 6E) at Gorgoroth: exhausts the
        # Sauron reinforcement pool AND doubles as the required Shadow army
        place(s, 'gorgoroth', Nation.SAURON, regulars=36, elites=6)
        s.political.boxes[Nation.SAURON] = 3     # Sauron At War (req gate)
        s.political.active[Nation.SAURON] = True

    st, res = play(Side.SHADOW, olog, {'region': 'gorgoroth'}, None,
                   prep=_exhaust_sauron)
    # USER RULING 1.230 (2026-08-28): a VALID choice whose effect cannot
    # apply (empty pool) is a completed zero-effect play - card
    # discarded, not refunded (supersedes the 2026-07-21 refund)
    assert res.get('success') and res.get('zero_effect') \
        and not res.get('refund'), \
        "Olog-hai empty pool completes as a zero-effect play (1.230)"
    assert olog not in st.hands[Side.SHADOW].hand, "card is discarded"
    # (teacher-half removed 2026-07-27 — teacher retired)

    # --- Riders of Theoden: non-Rohan target -> refund (self-play) ---
    riders = 'FREE_PEOPLES_Strategy_Riders_of_Théoden'   # accented id
    assert get_card(riders) is not None, "Riders card id resolves"
    st, res = play(Side.FREE_PEOPLES, riders,
                   {'region': 'dol amroth'}, None)
    assert res.get('refund') is True and \
        riders in st.hands[Side.FREE_PEOPLES].hand, \
        "non-Rohan target refunds (self-play)"
    # (teacher-half removed 2026-07-27 — teacher retired)
    # positive: Edoras recruits and consumes (self-play)
    st, res = play(Side.FREE_PEOPLES, riders, {'region': 'Edoras'}, None)
    assert res.get('success') and not res.get('refund'), \
        "Edoras is a legal target"
    assert res['effects'], "Edoras placement reports at least one piece"

    # --- The Shadow Lengthens: no-move -> refund (self-play) ---
    tsl = 'SHADOW_Strategy_The_Shadow_Lengthens'
    assert get_card(tsl) is not None, "The Shadow Lengthens card id resolves"
    st, res = play(Side.SHADOW, tsl, {}, None)
    # USER RULING 1.230: absent choice with legal choices -> refund;
    # no legal move at all -> completes as zero-effect
    from card_options import get_card_options as _gco218
    _s_tsl = setup_game()
    _s_tsl.hands[Side.SHADOW].hand.append(tsl)
    if _gco218(_s_tsl, Side.SHADOW, 'The Shadow Lengthens'):
        assert res.get('refund') is True and \
            tsl in st.hands[Side.SHADOW].hand, \
            "absent choice with legal moves refunds"
    else:
        assert res.get('success') and res.get('zero_effect') \
            and tsl not in st.hands[Side.SHADOW].hand, \
            "no legal move completes (1.230)"
    # (teacher-half removed 2026-07-27 — teacher retired)


@clause("CL-219", "6-cards", "Adjudication: a separated Guide re-picks by HIGHEST Level (Level tie -> pending_guide_change), not companions[0]; Horde from the East is unplayable when all four legal regions are FP-controlled", page='card')
def cl_219():
    # adjudication batch (2026-07-21): #3 guide re-pick + #7 Horde gate
    import event_effects as ee

    # (T1) The Breaking of the Fellowship: the drawn damage tile separates
    # the (lower-Level, first-in-list) Guide; the survivor Guide must be
    # the HIGHEST-Level remaining companion - NOT arbitrary companions[0].
    def run_bof(comps):
        st = setup_game()
        st.fellowship.companions = list(comps)
        st.fellowship.guide = st.fellowship.companions[0]  # lower-Level first
        st.pending_guide_change = None
        st.fellowship.mordor_entered = False
        # a single damage-1 tile -> deterministic draw, 1 companion separated
        st.hunt_pool = [gs.HuntTile(damage=1)]
        ee._effect_breaking_of_fellowship(st, None, {})
        # USER 2026-07-27: WHICH companion separates is the FP player's
        # LEARNED decision — the handler opens a chained pending on the
        # self-play path instead of auto-picking lowest-Level.  Resolve
        # it here by choosing Peregrin (the old auto-pick's outcome) so
        # the guide re-pick assertions below are unchanged.
        pco = getattr(st, 'pending_card_option', None)
        assert pco is not None and pco.get('bof_separation'), \
            "BoF opens the learned separation pending"
        assert len(pco.get('options') or []) == len(
            [c for c in comps if c.name not in ('Frodo', 'Sam',
                                                'Gollum')]), \
            "one option per movable companion"
        st.pending_card_option = None
        ee._bof_apply_separation(st, pco, 'Peregrin')
        return st

    st = run_bof([gs.Companion(name='Peregrin', level=1),
                  gs.Companion(name='Aragorn', level=3),
                  gs.Companion(name='Legolas', level=2)])
    assert 'Peregrin' not in [c.name for c in st.fellowship.companions], \
        "the first-in-list Guide was separated"
    assert st.fellowship.guide is not None and \
        st.fellowship.guide.name == 'Aragorn', \
        "Guide re-picks the HIGHEST-Level survivor, not companions[0]: %s" % (
            st.fellowship.guide and st.fellowship.guide.name)
    assert getattr(st, 'pending_guide_change', None) is None, \
        "no Level tie -> no pending_guide_change"

    st = run_bof([gs.Companion(name='Peregrin', level=1),
                  gs.Companion(name='Aragorn', level=3),
                  gs.Companion(name='Boromir', level=3)])
    assert getattr(st, 'pending_guide_change', None) is not None, \
        "equal-Level survivors open pending_guide_change"
    _cands = set(st.pending_guide_change.get('candidates', []))
    assert {'Aragorn', 'Boromir'} <= _cands, \
        "both tied companions are candidates: %s" % _cands

    # (T2) Horde from the East: unplayable when all four legal recruit
    # regions are FP-controlled; freeing one restores playability.
    cid = 'SHADOW_Strategy_Horde_from_the_East'
    assert get_card(cid) is not None, "Horde card id resolves"
    st = setup_game()
    st.political.boxes[Nation.SOUTHRON_EASTERLINGS] = 3   # S&E At War (req)
    st.political.active[Nation.SOUTHRON_EASTERLINGS] = True
    _fp = st.enemy_controlled.setdefault(Side.FREE_PEOPLES, set())
    _fp |= {'east rhun', 'south rhun', 'khand', 'far harad'}
    ok, _why = ee.check_event_requirement(st, cid)
    assert ok is False, "all four regions FP-controlled -> unplayable"
    _fp.discard('east rhun')                              # free one region
    ok, _why = ee.check_event_requirement(st, cid)
    assert ok is True, "freeing one region restores playability: %s" % _why


@clause("CL-220", "6-cards", "Battle-path: Flocks of Crebain interposes on the Nazgul Strike! hunt option (self-play, pre-roll); The Shadow is Moving teacher over-stack returns whole-stack regulars-first (== resolve_overstacks(learned=False)); learned path opens __overstack__; conservation holds", page='card')
def cl_220():
    import ai as ai_m
    import event_effects as ee
    from game_state import PersistentCardType as PCT

    def _crebain_state():
        st = setup_game()
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='SHADOW_Character_Flocks_of_Crebain',
            card_name='Flocks of Crebain', side=Side.SHADOW,
            card_type=PCT.FLOCKS_OF_CREBAIN))
        st.fellowship.mordor_entered = False
        d = gs.ActionDie(faces=[Face.EYE], side=Side.SHADOW)
        d.current = Face.EYE
        st.hunt_box = [d]
        return st

    def _ns_pending():
        return {
            'card_id': 'SHADOW_Character_Nazgûl_Strike!',
            'card_name': 'Nazgûl Strike!',        # sanitize -> 'nazgul strike'
            'side': 'sh', 'decider': 'sh', 'direct': True,
            'options': [{'action': 'hunt'}, {'action': 'none'}]}

    def _dispatch_hunt(st):
        # (hook save/restore collapsed 2026-07-27 — teacher retired)
        ai_m._execute_action(st, ai_m.Action(
            die_idx=-1, action_type='card_option',
            params={'choices': {'action': 'hunt'}}))

    # (a1) self-play + Crebain on table + Nazgul Strike hunt option -> the
    #      discard-Crebain decision interposes PRE-ROLL (no tile drawn yet)
    st = _crebain_state()
    st.pending_card_option = _ns_pending()
    _pool_n = len(st.hunt_pool)
    _dispatch_hunt(st)
    assert st.pending_card_option is not None and \
        st.pending_card_option.get('card_name') == 'Flocks of Crebain', \
        "Crebain discard decision interposes before the hunt roll"
    assert 'resume_action' in st.pending_card_option, \
        "the interpose carries a resume_action to re-dispatch the hunt"
    assert len(st.hunt_pool) == _pool_n, "no tile drawn yet (pre-roll)"

    # (a2) discard=True sets _crebain_hunt_bonus and removes the table card
    st = _crebain_state()
    ee._effect_flocks_of_crebain(st, None, {'crebain_discard': True})
    assert getattr(st, '_crebain_hunt_bonus', 0) == 1, \
        "discarding Crebain sets the +1 Hunt-dice bonus"
    assert st.persistent_cards.get_by_type(PCT.FLOCKS_OF_CREBAIN) is None, \
        "the Crebain table card is removed on discard"

    # (a3) negative: NO Crebain on table -> no interpose (immediate roll)
    st = setup_game()
    st.persistent_cards = gs.PersistentCardState()
    st.fellowship.mordor_entered = False
    d = gs.ActionDie(faces=[Face.EYE], side=Side.SHADOW)
    d.current = Face.EYE
    st.hunt_box = [d]
    st.pending_card_option = _ns_pending()
    _dispatch_hunt(st)
    assert not (st.pending_card_option and
                st.pending_card_option.get('card_name') == 'Flocks of Crebain'), \
        "no Crebain -> no interpose (the hunt resolves immediately)"

    # (b) The Shadow is Moving over-stack: the teacher path merges the
    #     WHOLE moving group then returns the excess whole-stack
    #     regulars-first (== resolve_overstacks(learned=False)); the
    #     learned path defers to the __overstack__ pending; conservation.
    def comp(a):
        return (sum(1 for u in a.units if u.kind == Kind.REGULAR),
                sum(1 for u in a.units if u.kind == Kind.ELITE))

    def _tsim_scenario():
        st = setup_game()
        S = 'gorgoroth'
        fsp = sanitize(game_logic.fellowship_region(st))
        _bs = getattr(st, 'besieged_strongholds', set())
        adj = [sanitize(x) for x in st.board.adjacent(S)]
        D = next(d for d in adj if d != fsp and d not in _bs)
        # D holds 8 S&E regulars; S holds 4 S&E elites (all Shadow)
        place(st, D, Nation.SOUTHRON_EASTERLINGS, regulars=8, clear=True)
        place(st, S, Nation.SOUTHRON_EASTERLINGS, elites=4, clear=True)
        return st, S, D

    # (teacher-half removed 2026-07-27 — teacher retired)
    # excess returns regulars-first -> 6 regulars + 4 elites survive
    st, S, D = _tsim_scenario()
    total_before = sum(comp(army_at(st, S))) + sum(comp(army_at(st, D)))
    ee._effect_shadow_is_moving(st, None,
                                {'from_region': S, 'to_region': D})
    d_after = comp(army_at(st, D))
    total_after = sum(comp(army_at(st, S))) + sum(comp(army_at(st, D)))
    assert d_after == (6, 4), \
        "teacher over-stack returns regulars first (6R+4E): %s" % (d_after,)
    assert total_before - total_after == 2, \
        "conservation: exactly the 2-unit excess left the board"

    # the same merged stack under resolve_overstacks(learned=False) matches
    st2 = setup_game()
    place(st2, D, Nation.SOUTHRON_EASTERLINGS, regulars=8, elites=4,
          clear=True)
    st2._overstack_regions = {D}
    ee.resolve_overstacks(st2, learned=False)
    assert comp(army_at(st2, D)) == d_after, \
        "teacher path == resolve_overstacks(learned=False)"

    # learned path (_from_pending, move-chain suppressed): the handler
    # records the over-stack and lands the WHOLE group without trimming;
    # resolve_overstacks(learned=True) then opens the __overstack__ pending
    st3, S3, D3 = _tsim_scenario()
    ee._effect_shadow_is_moving(st3, None,
                                {'from_region': S3, 'to_region': D3,
                                 '_from_pending': True, '_tsim_n': 3})
    assert D3 in getattr(st3, '_overstack_regions', set()), \
        "learned path records the over-stack region"
    assert comp(army_at(st3, D3)) == (8, 4), \
        "learned path lands the WHOLE group (no immediate trim)"
    st3.pending_card_option = None
    ee.resolve_overstacks(st3, learned=True)
    assert st3.pending_card_option is not None and \
        st3.pending_card_option.get('card_name') == '__overstack__', \
        "learned path opens the __overstack__ owner decision"


@clause("CL-221", "6-cards", "Foul Thing: hobbit guide separates to reduce damage pre-pick (interpose offered, chains to the new guide)", page='card')
def cl_221():
    # USER 1.208l: BEFORE the forced random casualty, the FP may use the
    # guide's reducing ability (hobbit separates -1); a separation
    # re-checks the NEW guide (chain).  Self-play only.
    import event_effects as ee
    import combat as cb_m
    _sv_fh = cb_m.FINAL_HITS_HOOK
    cb_m.FINAL_HITS_HOOK = None
    try:
        st = setup_game()
        st.pending_card_option = None
        merry = gs.Companion(name='Meriadoc', level=1, leadership=1,
                             guide_ability='sacrifice_reduce_hunt')
        pippin = gs.Companion(name='Peregrin', level=2, leadership=1,
                              guide_ability='sacrifice_reduce_hunt')
        boromir = gs.Companion(name='Boromir', level=1, leadership=1)
        st.fellowship.companions = [merry, pippin, boromir]
        st.fellowship.guide = merry
        st.fellowship.hidden = True
        st.fellowship.mordor_entered = False
        # the pre-pick guide stage opens for the reducing hobbit guide
        ee._foul_confront(st, {'victim': None}, 2)
        pco = st.pending_card_option
        assert pco is not None and pco.get('foul_guide'), \
            "foul_guide interpose offered for a reducing hobbit guide"
        assert {'use_foul_guide': 1} in pco['options'] and \
            {'use_foul_guide': 0} in pco['options'], \
            "the 2-option use/decline pair is offered"
        # use=1: Merry separates (queued for placement), damage -> 1, and
        # a FRESH interpose opens for the new guide (Peregrin, a hobbit)
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='card_option',
            params={'choices': {'use_foul_guide': 1}}))
        q = getattr(st, '_hobbit_place_queue', None) or []
        assert any(getattr(h, 'name', '') == 'Meriadoc' for h in q), \
            "the separated Merry is queued for placement"
        assert merry not in st.fellowship.companions, \
            "Merry left the Fellowship"
        pco2 = st.pending_card_option
        assert pco2 is not None and pco2.get('foul_guide'), \
            "the chain re-offers the guide stage for the new guide"
        assert int((pco2.get('foul_resume') or {}).get('damage', -1)) == 1, \
            "the fresh interpose carries the reduced damage (1)"
    finally:
        cb_m.FINAL_HITS_HOOK = _sv_fh


@clause("CL-222", "6-cards", "Foul Thing: Gollum reveal one-shot (damage->0, no casualty); decline goes straight to the forced random casualty (no re-offer)", page='card')
def cl_222():
    import event_effects as ee
    import combat as cb_m
    _sv_fh = cb_m.FINAL_HITS_HOOK
    cb_m.FINAL_HITS_HOOK = None
    try:
        # use=1: Gollum reveals -> hidden False, damage 0, no casualty
        st = setup_game()
        st.pending_card_option = None
        gollum = gs.Companion(name='Gollum', level=0, leadership=0,
                              guide_ability='no_reveal_on_damage')
        boromir = gs.Companion(name='Boromir', level=2, leadership=1)
        st.fellowship.companions = [gollum, boromir]
        st.fellowship.guide = gollum
        st.fellowship.hidden = True
        st.fellowship.mordor_entered = False
        ee._foul_confront(st, {'victim': None}, 1)
        assert st.pending_card_option is not None and \
            st.pending_card_option.get('foul_guide'), \
            "Gollum's reveal reduction is offered while hidden"
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='card_option',
            params={'choices': {'use_foul_guide': 1}}))
        assert st.fellowship.hidden is False, "Gollum revealed the Fellowship"
        assert any(c.name == 'Boromir' for c in st.fellowship.companions), \
            "damage fully neutralised -> no casualty"
        assert st.pending_card_option is None, "one-shot: no re-offer"
        # decline: goes straight to the forced random casualty, and MUST
        # NOT re-open the same interpose (guide unchanged) forever
        st2 = setup_game()
        st2.pending_card_option = None
        gollum2 = gs.Companion(name='Gollum', level=0, leadership=0,
                               guide_ability='no_reveal_on_damage')
        boromir2 = gs.Companion(name='Boromir', level=2, leadership=1)
        st2.fellowship.companions = [gollum2, boromir2]
        st2.fellowship.guide = gollum2
        st2.fellowship.hidden = True
        st2.fellowship.mordor_entered = False
        ee._foul_confront(st2, {'victim': None}, 1)
        assert st2.pending_card_option is not None
        ai._execute_action(st2, ai.Action(
            die_idx=-1, action_type='card_option',
            params={'choices': {'use_foul_guide': 0}}))
        assert st2.pending_card_option is None, \
            "decline does not re-open the guide interpose"
        assert not any(c.name == 'Boromir'
                       for c in st2.fellowship.companions), \
            "decline -> the forced random casualty removes Boromir"
    finally:
        cb_m.FINAL_HITS_HOOK = _sv_fh


@clause("CL-223", "6-cards", "Palantir of Orthanc: bonus draw is a LEARNED 2-option deck choice (not an auto same-deck draw); no card drawn before the pick", page='card')
def cl_223():
    from game_state import PersistentCardType as PCT
    try:
        st = setup_game()
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='SHADOW_Strategy_The_Palantir_of_Orthanc',
            card_name='The Palantir of Orthanc', side=Side.SHADOW,
            card_type=PCT.PALANTIR_OF_ORTHANC))
        st.pending_card_option = None
        st.pending_combat = None
        _h0 = len(st.hands[Side.SHADOW].hand)
        ai._open_palantir_deck_choice(st)
        pco = st.pending_card_option
        assert pco is not None and pco.get('palantir_draw'), \
            "the learned Palantir deck choice opens"
        assert {'palantir_deck': 'character'} in pco['options'] and \
            {'palantir_deck': 'strategy'} in pco['options'], \
            "both decks (character / strategy) are offered"
        assert len(st.hands[Side.SHADOW].hand) == _h0, \
            "NO card is auto-drawn before the choice is made"
        acts = ai.get_legal_actions(st, Side.SHADOW)
        _copts = [a for a in acts if a.action_type == 'card_option']
        assert len(_copts) == 2, \
            "get_legal_actions surfaces exactly the 2 deck options: %d" \
            % len(_copts)
        # resolving a deck choice draws exactly one card from that deck
        st._palantir_draw_pending = True
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='card_option',
            params={'choices': {'palantir_deck': 'strategy'}}))
        assert len(st.hands[Side.SHADOW].hand) == _h0 + 1, \
            "the chosen-deck draw adds exactly one card"
        assert st.pending_card_option is None
        assert getattr(st, '_palantir_draw_pending', True) is False, \
            "the deferred-draw flag is cleared after the draw"
    finally:
        pass  # (hook restore collapsed 2026-07-27)


@clause("CL-224", "6-cards", "Palantir bonus draw is DEFERRED past an unresolved pending_combat and fires only once the battle fully resolves", page='card')
def cl_224():
    from game_state import PersistentCardType as PCT
    try:
        st = setup_game()
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='SHADOW_Strategy_The_Palantir_of_Orthanc',
            card_name='The Palantir of Orthanc', side=Side.SHADOW,
            card_type=PCT.PALANTIR_OF_ORTHANC))
        # a P-die event action opened a combat -> the draw is deferred
        st.pending_combat = {'phase': 'x', 'attacker_region': 'a',
                             'defender_region': 'b',
                             'attacker_side': Side.SHADOW,
                             'defender_side': Side.FREE_PEOPLES}
        st._palantir_draw_pending = True
        st.pending_card_option = None
        ai._maybe_open_palantir_draw(st)
        assert st.pending_card_option is None, \
            "no deck choice opens while pending_combat is unresolved"
        assert getattr(st, '_palantir_draw_pending', False) is True, \
            "the deferred-draw flag persists across the unresolved battle"
        # battle ends -> the terminal guard opens the deck choice
        st.pending_combat = None
        ai._maybe_open_palantir_draw(st)
        assert st.pending_card_option is not None and \
            st.pending_card_option.get('palantir_draw'), \
            "the deck choice opens once the battle fully resolves"
        assert getattr(st, '_palantir_draw_pending', True) is False
    finally:
        pass  # (hook restore collapsed 2026-07-27)


@clause("CL-225", "6-cards", "Shadows Gather: learned split grid; a partial move relocates the chosen composition and the remainder stays; conservation holds", page='card')
def cl_225():
    import event_effects as ee
    from card_options import get_card_options
    st = setup_game()
    S = 'gorgoroth'
    _bs = getattr(st, 'besieged_strongholds', set())
    fsp = sanitize(game_logic.fellowship_region(st))
    adj = [sanitize(x) for x in st.board.adjacent(S)]
    D = next(d for d in adj if d != fsp and d not in _bs)
    # source: 3 regulars + 1 elite; destination: an existing Shadow army
    place(st, S, Nation.SOUTHRON_EASTERLINGS, regulars=3, elites=1,
          clear=True)
    place(st, D, Nation.SOUTHRON_EASTERLINGS, regulars=1, clear=True)

    def comp(a):
        return (sum(1 for u in a.units if u.kind == Kind.REGULAR),
                sum(1 for u in a.units if u.kind == Kind.ELITE))

    total_before = sum(comp(army_at(st, S))) + sum(comp(army_at(st, D)))
    opts = get_card_options(st, Side.SHADOW, 'Shadows Gather')
    assert {'from_region': S, 'to_region': D} in opts, \
        "the legacy whole-army option is still offered (teacher parity)"
    assert {'from_region': S, 'to_region': D,
            'n_regular': 2, 'n_elite': 0} in opts, \
        "the learned split grid offers the (2R, 0E) composition"
    ee._effect_shadows_gather(st, None, {'from_region': S, 'to_region': D,
                                         'n_regular': 2, 'n_elite': 0})
    assert comp(army_at(st, S)) == (1, 1), \
        "the unchosen 1R+1E remainder stays at the source: %s" \
        % (comp(army_at(st, S)),)
    assert comp(army_at(st, D)) == (3, 0), \
        "the chosen 2R joined the destination (1R -> 3R): %s" \
        % (comp(army_at(st, D)),)
    total_after = sum(comp(army_at(st, S))) + sum(comp(army_at(st, D)))
    assert total_before == total_after, \
        "conservation: no unit created or destroyed"


@clause("CL-226", "6-cards", "Return to Valinor gate: a BESIEGED Elven stronghold no longer qualifies (1.208o capture-only); a Shadow-captured one does", page='card')
def cl_226():
    import event_effects as ee
    CID = 'SHADOW_Strategy_Return_to_Valinor'
    st = setup_game()
    if getattr(st, 'besieged_strongholds', None) is None:
        st.besieged_strongholds = set()
    st.besieged_strongholds.add('lorien')
    ok, why = ee.check_event_requirement(st, CID)
    assert ok is False, \
        "a besieged-only Elven SH must NOT satisfy the RtV gate: %r" % (why,)
    st.enemy_controlled[Side.SHADOW].add('rivendell')
    ok2, why2 = ee.check_event_requirement(st, CID)
    assert ok2 is True, \
        "a Shadow-captured Elven SH satisfies the RtV gate: %r" % (why2,)


@clause("CL-227", "6-cards", "Parser deck-index tables: every VASSAL piece title in log_replay's four deck dicts resolves to a real card id via ai._lookup_card_id (t138 Grimbeorn / 'A Balrog is Come!' drift class)", page='card')
def cl_227():
    # USER 2026-07-21: 'A Balrog is Come!' surfaced from the parser's
    # hand-typed deck tables (log_replay.py) -- the VASSAL module's piece
    # title for the card officially named 'Balrog of Moria'.  The engine
    # translates through ai._lookup_card_id's alias/normalization shim;
    # this clause locks the CLOSED LOOP: any future module-title drift
    # (new alias needed) fails here instead of silently producing
    # undecodable plays (t138: 16/16 Grimbeorn plays undecodable).
    import ai as _ai
    import log_replay as _lr
    from cards import get_cards as _gc
    cards = _gc()
    lookup = {c.card_name: cid for cid, c in cards.items()}
    bad = []
    for tbl_name in ('FP_CHARACTER_CARDS', 'FP_STRATEGY_CARDS',
                     'SA_CHARACTER_CARDS', 'SA_STRATEGY_CARDS'):
        tbl = getattr(_lr, tbl_name)
        assert len(tbl) == 24, "%s has %d entries, expected 24" % (
            tbl_name, len(tbl))
        for idx, title in tbl.items():
            cid = _ai._lookup_card_id(lookup, title)
            if cid is None or cid not in cards:
                bad.append("%s[%d]=%r" % (tbl_name, idx, title))
    assert not bad, "unresolvable VASSAL piece titles: %s" % "; ".join(bad)
    # The known aliases keep resolving to the intended cards.
    assert _ai._lookup_card_id(lookup, 'A Balrog is Come!') == \
        _ai._lookup_card_id(lookup, 'Balrog of Moria')
    assert _ai._lookup_card_id(lookup, 'Grimbeorn the Old') == \
        _ai._lookup_card_id(lookup, 'Grimbeorn the Old, Son of Beorn')


@clause("CL-228", "4-battle", "1.209: in-Fellowship companions are INVISIBLE to combat -- a co-located Fellowship gives NO Captain combat die, NO leadership, does NOT satisfy requires_companion, and never appears in companions_in_battle (no GtW shine candidacy); It is a Gift / One for the Dark Lord stay playable via the Fellowship-in-region gate", page='card')
def cl_228():
    # USER 2026-07-21: "in fellowship companions NEVER EVER EVER contribute
    # to combat.  the only time the fellowship position matters FOR COMBAT is
    # for the specific combat cards that mention it: it is a gift and the
    # shadow counterpart [One for the Dark Lord]."  Supersedes the Session-18
    # setup_combat fellowship inclusion.
    from cards import get_card, can_play_combat_card
    st = setup_game()
    # Defender: an FP army with ZERO board companions.
    place(st, 'lorien', Nation.GONDOR, regulars=3, clear=True)
    # Attacker: a Shadow army in an adjacent region.
    place(st, 'moria', Nation.SAURON, regulars=5, clear=True)
    def_army = army_at(st, 'lorien')
    # The (hidden) Fellowship sits in the DEFENDER's region and carries a
    # Captain (Boromir, leadership 1) -- if the old code path were still
    # live he would add a combat die AND +1 leadership AND be a card-gate
    # companion.  1.209 says he contributes NOTHING.
    boromir = gs.Companion(name='Boromir', level=2, leadership=1)
    st.fellowship.region = 'lorien'
    st.fellowship.hidden = True
    st.fellowship.companions = [boromir]
    assert sanitize(game_logic.fellowship_region(st)) == 'lorien', \
        "Fellowship is co-located with the defending army"

    # Cross-check that Boromir WOULD count if he were on the board -- makes
    # the exclusion assertions load-bearing, not vacuous.
    assert cb.calc_combat_strength(def_army, [boromir]) == 4, \
        "sanity: a board Boromir would raise strength 3 -> 4"
    assert cb.calc_leadership(def_army, [boromir]) == 1, \
        "sanity: a board Boromir would raise leadership 0 -> 1"

    combat = cb.setup_combat(st, 'moria', 'lorien',
                             attacker_side=Side.SHADOW)

    # (a) strength counts UNITS ONLY -- no Captain-of-the-West die.
    assert combat.defender_combat_strength == len(def_army.units) == 3, \
        "defender strength must count units only (no in-Fellowship Captain die): %d" \
        % (combat.defender_combat_strength,)
    # (b) leadership is ARMY-ONLY (this army has no leaders -> 0).
    assert combat.defender_leadership == cb.calc_leadership(def_army) == 0, \
        "defender leadership must be army-only (no in-Fellowship leadership): %d" \
        % (combat.defender_leadership,)
    # (c) companions_in_battle holds board companions only -> empty here.
    assert combat.companions_in_battle == [], \
        "in-Fellowship companions must NOT appear in companions_in_battle: %r" \
        % (combat.companions_in_battle,)
    # (d) no Gandalf-the-White shine candidate (GtW can never be in the
    #     Fellowship, 2026-07-21e; the scan no longer looks there).
    assert not any(getattr(c, 'can_cancel_nazgul_leadership', False)
                   or getattr(c, 'name', '') == 'Gandalf the White'
                   for c in combat.companions_in_battle), \
        "no shine candidate may be sourced from the Fellowship"
    assert getattr(combat, '_shine_companion', 'MISSING') is None, \
        "no companion shines from inside the Fellowship"

    # A requires_companion combat card is NOT playable: the only companion
    # (Boromir) is hidden in the Fellowship, so the board-only gate fails.
    horn = get_card('FREE_PEOPLES_Character_Horn_of_Gondor')  # 'Mighty Attack'
    assert getattr(horn, 'requires_companion', False), \
        "fixture sanity: Horn of Gondor requires a Companion"
    ok_h, why_h = can_play_combat_card(horn, combat, is_attacker=False,
                                       state=st)
    assert ok_h is False and 'Companion' in (why_h or ''), \
        "requires_companion must NOT be satisfied by an in-Fellowship companion: %r" \
        % (why_h,)

    # It is a Gift (FP) stays playable via the Fellowship-in-region gate.
    gift = get_card('FREE_PEOPLES_Character_Elven_Cloaks')  # 'It is a Gift'
    assert getattr(gift, 'requires_fellowship_region', False)
    ok_g, why_g = can_play_combat_card(gift, combat, is_attacker=False,
                                       state=st)
    assert ok_g is True, \
        "It is a Gift must remain playable when the Fellowship is in the battle region: %r" \
        % (why_g,)

    # One for the Dark Lord (SH, played by the attacker) likewise stays
    # playable via the same Fellowship-in-region gate.
    ofdl = get_card("SHADOW_Character_Shelob's_Lair")  # 'One for the Dark Lord'
    assert getattr(ofdl, 'requires_fellowship_region', False)
    ok_o, why_o = can_play_combat_card(ofdl, combat, is_attacker=True,
                                       state=st)
    assert ok_o is True, \
        "One for the Dark Lord must remain playable via the Fellowship gate: %r" \
        % (why_o,)


def _resolve_combat_cards(st):
    """Resolve the per-round combat-card decline phases (attacker then
    defender) so the pending combat round actually rolls.  The card phases
    always fire (forced 'combat_card_none' when nothing is playable)."""
    for _ in range(4):
        pc = getattr(st, 'pending_combat', None)
        if pc is None or pc.get('phase') not in (
                'attacker_combat_card', 'defender_combat_card'):
            return
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='combat_card_none', params={}))


@clause("CL-229", "6-cards", "1.210(a) Help Unlooked For: MULTI-round relief battle - the Shadow besieger's dice reduction (one per FP unit inside the besieged stronghold, capped 5 then floored 1) PERSISTS every round; the FP attacker is offered combat_cease", page='card')
def cl_229():
    import event_effects as ee
    st = setup_game()
    st.besieged_strongholds = {'minas tirith'}
    # co-located siege entry: 4 Gondor garrison + 10 Sauron besiegers
    place(st, 'minas tirith', Nation.GONDOR, regulars=4, clear=True)
    place(st, 'minas tirith', Nation.SAURON, regulars=10)
    # relief army: 4 Gondor regulars in adjacent Osgiliath, Gondor At War
    place(st, 'osgiliath', Nation.GONDOR, regulars=4, clear=True)
    st.political.boxes[Nation.GONDOR] = 3
    for _s in (Side.SHADOW, Side.FREE_PEOPLES):
        st.hands[_s].hand = []           # no combat cards -> clean rounds
    assert cb.garrison_side_for(st, 'minas tirith') == Side.FREE_PEOPLES, \
        "fixture: MT garrison is Free Peoples (Shadow besieges)"

    _dice = []
    _orig = cb.combat_roll

    def _spy(dc, thr, mod=0, cap=None, **kw):
        # **kw: combat_roll gained side= for G1 (2026-08-18) - the spy
        # forwards it so the funnel still knows whose dice these are.
        r = _orig(dc, thr, mod, cap, **kw)
        _dice.append(len(r[0]))
        return r
    cb.combat_roll = _spy
    try:
        ee._effect_help_unlooked_for(
            st, None, {'stronghold': 'minas tirith',
                       'attack_from': 'osgiliath'})
        _resolve_combat_cards(st)                # roll round 1
        _r1 = list(_dice)
        pc = st.pending_combat
        assert pc is not None, "the relief battle opened a pending_combat"
        assert pc.get('def_dice_reduction') == 4, \
            "Shadow dice reduced by the 4 FP garrison units: %r" \
            % pc.get('def_dice_reduction')
        assert pc.get('attacker_side') == Side.FREE_PEOPLES, \
            "the FP relief army is the ATTACKER"
        assert pc.get('phase') == 'attacker_continue', \
            "field battle -> attacker_continue after round 1 (multi-round): %r" \
            % pc.get('phase')
        # round 1: the Shadow DEFENDER rolled exactly 1 die (10 -> cap 5 -> -4 -> 1)
        assert len(_r1) == 2 and _r1[1] == 1, \
            "Shadow defender rolls exactly 1 die in round 1: %r" % _r1
        # 1.210(a): the FP attacker IS offered combat_cease (may stop pressing)
        _t = types_of(ai._get_pending_combat_actions(st))
        assert 'combat_cease' in _t and 'combat_continue' in _t, \
            "FP attacker offered BOTH continue and cease: %r" % _t
        # drive round 2 -> the reduction PERSISTS (never popped)
        _dice.clear()
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='combat_continue', params={}))
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='combat_no_retreat', params={}))
        _resolve_combat_cards(st)                # roll round 2
        _r2 = list(_dice)
        assert st.pending_combat is not None and \
            st.pending_combat.get('def_dice_reduction') == 4, \
            "the dice reduction persists into round 2"
        assert len(_r2) == 2 and _r2[1] == 1, \
            "Shadow defender rolls exactly 1 die AGAIN in round 2: %r" % _r2
    finally:
        cb.combat_roll = _orig


@clause("CL-230", "6-cards", "1.210(b) Corsairs of Umbar: field target (Pelargir) - attacker CANNOT cease (no combat_cease), defender may retreat; stronghold target (Dol Amroth) - retreat_into_siege ENDS the battle; a Shadow-garrison coastal stronghold under FP siege is an admitted RELIEF landing (siege battle, no-cease rule off, siege_no_press legal)", page='card')
def cl_230():
    import event_effects as ee
    import card_options as co
    try:
        # (T1) Pelargir (a City, not a stronghold): field battle; the Shadow
        # attacker CANNOT voluntarily cease; the FP defender may retreat.
        st = setup_game()
        place(st, 'umbar', Nation.SOUTHRON_EASTERLINGS, regulars=8, clear=True)
        place(st, 'pelargir', Nation.GONDOR, regulars=6, clear=True)
        for _s in (Side.SHADOW, Side.FREE_PEOPLES):
            st.hands[_s].hand = []
        ee._effect_corsairs_of_umbar(st, None, {'target_region': 'pelargir'})
        _resolve_combat_cards(st)                # roll the field round
        pc = st.pending_combat
        assert pc is not None and pc.get('phase') == 'attacker_continue', \
            "Pelargir field battle -> attacker_continue: %r" \
            % (pc and pc.get('phase'))
        assert pc.get('attacker_no_retreat') is True, \
            "the landed Shadow army cannot retreat (no-cease)"
        _t = types_of(ai._get_pending_combat_actions(st))
        assert 'combat_continue' in _t and 'combat_cease' not in _t, \
            "attacker offered continue but NOT cease: %r" % _t
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='combat_continue', params={}))
        _t2 = types_of(ai._get_pending_combat_actions(st))
        assert 'combat_retreat' in _t2, \
            "the FP defender is offered a field retreat: %r" % _t2

        # (T2) Dol Amroth (a stronghold, not besieged): the FP defender
        # withdraws into the siege and THAT ENDS THE BATTLE.
        st = setup_game()
        place(st, 'umbar', Nation.SOUTHRON_EASTERLINGS, regulars=6, clear=True)
        place(st, 'dol amroth', Nation.GONDOR, regulars=3, clear=True)
        for _s in (Side.SHADOW, Side.FREE_PEOPLES):
            st.hands[_s].hand = []
        ee._effect_corsairs_of_umbar(st, None, {'target_region': 'dol amroth'})
        pc = st.pending_combat
        assert pc is not None and pc.get('phase') == 'defender_siege_choice', \
            "Dol Amroth (FP-controlled stronghold) -> defender_siege_choice: %r" \
            % (pc and pc.get('phase'))
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='retreat_into_siege', params={}))
        assert st.pending_combat is None, \
            "retreat_into_siege ENDS the battle (no auto siege continuation)"

        # (T3) Dol Amroth captured by Shadow and besieged by the FP: a legal
        # RELIEF landing (validator widening) - a siege battle where the
        # no-cease rule does NOT apply (Shadow may siege_no_press).
        st = setup_game()
        # Shadow CAPTURED Dol Amroth (Shadow garrison), FP now besieges it
        st.enemy_controlled.setdefault(Side.SHADOW, set()).add('dol amroth')
        st.besieged_strongholds = {'dol amroth'}
        place(st, 'umbar', Nation.SOUTHRON_EASTERLINGS, regulars=6, clear=True)
        place(st, 'dol amroth', Nation.SOUTHRON_EASTERLINGS, regulars=3,
              clear=True)                                   # Shadow garrison
        place(st, 'dol amroth', Nation.GONDOR, regulars=4)  # FP besiegers
        for _s in (Side.SHADOW, Side.FREE_PEOPLES):
            st.hands[_s].hand = []
        assert cb.garrison_side_for(st, 'dol amroth') == Side.SHADOW, \
            "fixture: Dol Amroth garrison is Shadow (FP besieges)"
        _opts = co.get_card_options(st, Side.SHADOW, 'Corsairs of Umbar')
        assert any(o.get('target_region') == 'dol amroth' for o in _opts), \
            "the besieged Shadow stronghold is admitted as a landing: %r" % _opts
        ee._effect_corsairs_of_umbar(st, None, {'target_region': 'dol amroth'})
        pc = st.pending_combat
        assert pc is not None and pc.get('is_siege') is True, \
            "the pre-besieged landing forms a siege battle"
        assert not pc.get('attacker_no_retreat'), \
            "the no-cease rule does NOT apply to a pre-besieged landing"
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='siege_assault', params={}))
        _resolve_combat_cards(st)                # roll the siege round
        _t3 = types_of(ai._get_pending_combat_actions(st))
        assert 'siege_no_press' in _t3, \
            "Shadow may stop pressing (siege_no_press legal): %r" % _t3
    finally:
        pass  # (hook restore collapsed 2026-07-27)


@clause("CL-231", "6-cards", "1.210(d) The Fighting Uruk-hai: the play opens a siege pc with mandatory_rounds=3 + no_def_card_r1_unless_companion; the press decision (combat_round_result) is WITHHELD until the 3 mandatory rounds are fought, then offers BOTH siege_press and siege_no_press (never auto-press)", page='card')
def cl_231():
    import event_effects as ee
    st = setup_game()
    st.besieged_strongholds = {'minas tirith'}
    place(st, 'minas tirith', Nation.GONDOR, regulars=4, clear=True)  # garrison
    place(st, 'minas tirith', Nation.ISENGARD, regulars=5, elites=1)  # besiegers

    # (a) the handler builds the correct siege pc WITHOUT auto-pressing -
    #     capture it via a no-op _run_combat_round (no dice rolled here).
    _cap = {}
    _orig = ai._run_combat_round

    def _fake(state, pc, is_siege=None):
        _cap['pc'] = dict(pc)
    ai._run_combat_round = _fake
    try:
        ee._effect_fighting_uruk_hai(
            st, None, {'attack_from': 'minas tirith',
                       'attack_target': 'minas tirith'})
    finally:
        ai._run_combat_round = _orig
    pc = _cap.get('pc')
    assert pc is not None, "the play opened an interactive siege pc"
    assert pc.get('mandatory_rounds') == 3, \
        "3 mandatory siege rounds: %r" % pc.get('mandatory_rounds')
    assert pc.get('no_def_card_r1_unless_companion') is True, \
        "FP may not play a combat card in round 1 (no companion)"
    assert pc.get('is_siege') is True and \
        pc.get('attacker_side') == Side.SHADOW

    # (b) the mandatory-rounds gate: the press decision (combat_round_result)
    #     is WITHHELD while rounds_fought < mandatory_rounds, and only opens
    #     after round 3.  Drive _set_post_round_phase with a no-op round so
    #     the recursion does not roll dice.
    ai._run_combat_round = _fake
    try:
        _pc = {'is_siege': True, 'mandatory_rounds': 3, 'rounds_fought': 1,
               'phase': 'sentinel', 'attacker_side': Side.SHADOW,
               'defender_side': Side.FREE_PEOPLES,
               'deciding_side': Side.SHADOW}
        ai._set_post_round_phase(st, _pc, True)
        assert _pc['phase'] != 'combat_round_result', \
            "no press decision during the mandatory rounds (round 1)"
        _pc['rounds_fought'] = 3
        ai._set_post_round_phase(st, _pc, True)
        assert _pc['phase'] == 'combat_round_result', \
            "after round 3 the normal press decision opens"
    finally:
        ai._run_combat_round = _orig

    # (c) combat_round_result offers BOTH press options when the attacker has
    #     an elite (never auto-pressed); only no_press when it has none.
    st2 = setup_game()
    place(st2, 'minas tirith', Nation.ISENGARD, regulars=3, elites=1,
          clear=True)
    # FIXTURE REPAIR (BRIDGE-6 2026-08-17, CHAOS-1 wipe guard): the
    # press menu is queried on a pending battle, and a battle whose
    # DEFENDER has zero units is now auto-finished (a wiped side =
    # battle over) - the old fixture had no garrison at all.  The
    # clause's subject (press offered iff the attacker has an elite)
    # needs a live battle: give the siege its garrison.
    place(st2, 'minas tirith', Nation.GONDOR, regulars=1)
    st2.pending_combat = {
        'phase': 'combat_round_result', 'is_siege': True,
        'attacker_region': 'minas tirith', 'defender_region': 'minas tirith',
        'attacker_side': Side.SHADOW, 'defender_side': Side.FREE_PEOPLES,
        'deciding_side': Side.SHADOW}
    _tp = types_of(ai._get_pending_combat_actions(st2))
    assert 'siege_press' in _tp and 'siege_no_press' in _tp, \
        "with an attacker elite BOTH press and no_press are offered: %r" % _tp
    # strip the elite -> only no_press
    a2 = army_at(st2, 'minas tirith')
    a2.units[:] = [u for u in a2.units if u.kind != Kind.ELITE]
    _tp2 = types_of(ai._get_pending_combat_actions(st2))
    assert 'siege_press' not in _tp2 and 'siege_no_press' in _tp2, \
        "with no elite only siege_no_press is offered: %r" % _tp2


@clause("CL-232", "6-cards", "1.210(c) Rage of the Dunlendings: the 'move up to 4' clause is ONE learned split grid over (n_north, n_south) - it ALWAYS includes (0,0) (do not move) and (2,2); executing (2,2) moves 2 Isengard from EACH Dunland to the destination (conservation)", page='card')
def cl_232():
    import event_effects as ee
    st = setup_game()
    place(st, 'north dunland', Nation.ISENGARD, regulars=3, clear=True)
    place(st, 'south dunland', Nation.ISENGARD, regulars=3, clear=True)
    place(st, 'orthanc', Nation.ISENGARD, regulars=0, clear=True)  # empty dest

    def _iseng(reg):
        a = st.armies.get(ee._find_army_key(st, reg))
        return sum(1 for u in (a.units if a else [])
                   if u.nation == Nation.ISENGARD)

    _before = _iseng('north dunland') + _iseng('south dunland') + \
        _iseng('orthanc')
    grid = ee._rage_split_options(st, 'orthanc')
    _splits = [tuple(sorted((o.get('rage_split') or {}).items())) for o in grid]
    assert any((o.get('rage_split') or {}) == {'n_north': 0, 'n_south': 0,
                                              'n_north_elite': 0, 'n_south_elite': 0}
               for o in grid), \
        "the grid ALWAYS includes (0,0) - the option NOT to move: %r" % _splits
    assert any((o.get('rage_split') or {}) == {'n_north': 2, 'n_south': 2,
                                              'n_north_elite': 0, 'n_south_elite': 0}
               for o in grid), \
        "the grid includes the (2,2) split: %r" % _splits

    # execute (2,2) via the handler STAGE-2 path (rage_split, no recruit)
    ee._effect_rage_of_dunlendings(
        st, None, {'rage_split': {'n_north': 2, 'n_south': 2},
                   '_rage_dest': 'orthanc'})
    assert _iseng('north dunland') == 1 and _iseng('south dunland') == 1, \
        "2 Isengard moved from EACH Dunland: N=%d S=%d" % (
            _iseng('north dunland'), _iseng('south dunland'))
    assert _iseng('orthanc') == 4, \
        "4 Isengard arrived at the destination: %d" % _iseng('orthanc')
    _after = _iseng('north dunland') + _iseng('south dunland') + \
        _iseng('orthanc')
    assert _before == _after == 6, \
        "conservation: no Isengard unit minted or lost (%d -> %d)" % (
            _before, _after)


@clause("CL-233", "6-cards", "1.210(e) The Spirit of Mordor playability gate: the card (empty event_req) is UNPLAYABLE with no legal target (no Shadow army of 2+ Shadow Nations) and PLAYABLE once such an army exists", page='card')
def cl_233():
    import event_effects as ee
    _cid = next((cid for cid, c in get_cards().items()
                 if getattr(c, 'card_name', '') == 'The Spirit of Mordor'),
                None)
    assert _cid is not None, "The Spirit of Mordor card id resolves"
    st = setup_game()
    assert not ee._get_multi_nation_shadow_armies(st), \
        "fixture: no multi-nation Shadow army in the initial setup"
    ok, why = ee.check_event_requirement(st, _cid)
    assert ok is False and 'Nation' in (why or ''), \
        "unplayable with no 2-nation Shadow army: %r" % (why,)
    # a single mixed Shadow army (Sauron + Isengard) restores playability
    place(st, 'moria', Nation.SAURON, regulars=2, clear=True)
    place(st, 'moria', Nation.ISENGARD, regulars=1)
    ok2, why2 = ee.check_event_requirement(st, _cid)
    assert ok2 is True, \
        "playable once a 2-nation Shadow army exists: %r" % (why2,)


# ---------------------------------------------------------------------------
# MED CLUSTER A: musters / region lists / gates (CL-234..240, 1.211a-e/l)
# ---------------------------------------------------------------------------

def _exhaust_elites(st, nation, dump='forlindon'):
    """Park EXACTLY every Elite of `nation` as deployed units in a remote
    region (empty casualty box), leaving 0 elites in the reinforcement
    pool but the Regular pool untouched - so _default_unit_type falls back
    to 'regular' (a la _set_reg_pool_free, but for elites)."""
    st.fp_casualties = {}
    mx = ai.NATION_MAX_FIGURES[nation]['elite']
    d = ai._count_deployed(st, nation).get('elite', 0)
    if mx - d > 0:
        place(st, dump, nation, elites=(mx - d))


# Swept muster cards (1.211a): unit type is elite-first HARDCODED, no
# learned/choice surface remains anywhere.
_SWEPT_MUSTER = {
    'guardsofthecitadel', 'celebornsgaladhrim', 'ridersoftheoden',
    'grimbeorntheoldsonofbeorn', 'imrahilofdolamroth',
    'kindredofglorfindel', 'dainironfootsguard', 'eomersonofeomund',
    'thranduilsarchers', 'houseofthestewards', 'theredarrow',
    'halforcsandgoblinmen', 'ologhai', 'faramirsrangers',
    'cirdansships', 'shadowsonthemistymountains', 'anewpowerisrising',
    'swordsineriador',
}


@clause("CL-234", "6-cards", "Guards exposes its printed Regular-or-Elite choice; legacy direct defaults and other recruitment-card restrictions are retained pending separate review", page='card')
def cl_234():
    import event_effects as ee
    from card_options import get_card_options
    # Guards of the Citadel: elites available -> ELITE placed
    guards, _g = _card_by_norm('guardsofthecitadel')
    _, res = _l3_play(guards, {}, None)
    assert any('elite' in e.lower() and 'minas tirith' in e.lower()
               for e in res['effects']), \
        "elites available -> Guards recruits an Elite: %r" % (res['effects'],)
    # elite pool exhausted (regulars still available) -> REGULAR placed
    _, res = _l3_play(guards, {}, None,
                      prep=lambda s: _exhaust_elites(s, Nation.GONDOR))
    assert any('regular' in e.lower() and 'minas tirith' in e.lower()
               for e in res['effects']), \
        "elite pool dry -> Guards recruits a Regular: %r" % (res['effects'],)
    # Faramir: legal recruitment-only opens the human kind choice. The
    # user's fixed AI convention selects Elite from that printed menu.
    faramir, _f = _card_by_norm('faramirsrangers')
    far_state, res = _l3_play(faramir, {}, None,
                      prep=lambda s: place(s, 'osgiliath', Nation.GONDOR,
                                           regulars=1, clear=True))
    assert far_state.pending_card_option.get('faramir_recruit'), \
        "Faramir exposes the human Regular/Elite decision"
    far_policy = ai.get_ai_policy(far_state, Side.FREE_PEOPLES)
    assert len(far_policy.actions) == 1 \
        and far_policy.actions[0].params['choices']['unit_type'] == 'elite'
    far_state = ai.apply_action(far_state, far_policy.actions[0])
    assert sum(u.nation == Nation.GONDOR and u.kind == Kind.ELITE
               for u in army_at(far_state, 'osgiliath').units) == 1
    # Guards' printed alternative is now exposed (RB22, AL21-22). Both
    # figures exist in the initial supply. This assertion is independent
    # of the legacy direct handler's unspecified-choice default above.
    st = setup_game()
    guards_card = get_card(guards)
    assert {o.get('unit_type') for o in get_card_options(
        st, Side.FREE_PEOPLES, guards_card.card_name)} == {'regular', 'elite'}
    assert get_card_options(st, Side.SHADOW, guards_card.card_name) == []
    # Other cards retain their historical restriction in this bounded
    # change; this is a compatibility assertion, not a printed-rule oracle.
    for cid, c in get_cards().items():
        norm = ai._normalize_card_name(getattr(c, 'card_name', ''))
        if norm not in _SWEPT_MUSTER or norm == 'guardsofthecitadel':
            continue
        gco = (get_card_options(st, Side.FREE_PEOPLES, c.card_name)
               + get_card_options(st, Side.SHADOW, c.card_name))
        assert not any(k in ('unit_type', 'unit_combo')
                       for o in gco for k in o), \
            "get_card_options unit-type surface leaked: %s" % (c.card_name,)


@clause("CL-235", "6-cards", "1.211b Eomer legal settlements: Folde and Westemnet are offered, 'Westfold' never appears anywhere, a besieged Helm's Deep is excluded from the option surface AND a self-play besieged-HD play is REFUNDED (no silent Edoras redirect)", page='card')
def cl_235():
    import event_effects as ee
    from card_options import get_card_options
    eomer, card = _card_by_norm('eomersonofeomund')
    st = setup_game()
    regs = {o['region'] for o in get_card_options(st, Side.FREE_PEOPLES,
                                                  card.card_name)}
    assert 'folde' in regs and 'westemnet' in regs, \
        "Folde + Westemnet are legal Eomer settlements: %r" % (regs,)
    assert not any('westfold' in r for r in regs), "'Westfold' does not exist"
    assert not any('westfold' in sanitize(s) for s in
                   ee._get_free_rohan_settlements(st, exclude_besieged=True)), \
        "GUI settlement list has no Westfold"
    # besieged Helm's Deep is excluded from the option surface
    st2 = setup_game()
    st2.besieged_strongholds = {sanitize("Helm's Deep")}
    regs2 = {o['region'] for o in get_card_options(st2, Side.FREE_PEOPLES,
                                                   card.card_name)}
    assert sanitize("Helm's Deep") not in regs2, \
        "besieged Helm's Deep is not offered: %r" % (regs2,)
    # self-play besieged-HD play REFUNDS (no silent Edoras redirect)
    def _besiege_hd(s):
        s.besieged_strongholds = {sanitize("Helm's Deep")}
    st3, res = _l3_play(eomer, {'region': "Helm's Deep"}, None,
                        prep=_besiege_hd)
    assert res.get('refund') is True and \
        eomer in st3.hands[Side.FREE_PEOPLES].hand, \
        "self-play besieged-HD Eomer play refunds: %r" % (res,)
    assert not any('redirected to Edoras' in e
                   for e in res.get('effects', [])), \
        "no silent besieged-HD -> Edoras redirect"


@clause("CL-236", "6-cards", "1.211c Riders of Theoden PER-REGION rule (corrected): a board Companion in Westemnet offers exactly {edoras, westemnet}; a Companion inside a besieged Helm's Deep offers Helm's Deep; with NO board Companion only Edoras is offered; a Fellowship-parked Companion never unlocks its region (doctrine l)", page='card')
def cl_236():
    from card_options import get_card_options
    _, card = _card_by_norm('ridersoftheoden')
    name = card.card_name
    # companion in Westemnet -> options are EXACTLY {edoras, westemnet}
    st = setup_game()
    army_at(st, 'westemnet').leaders.append(
        gs.Companion(name='Eowyn', level=1, leadership=1))
    regs = {o['region'] for o in get_card_options(st, Side.FREE_PEOPLES, name)}
    assert regs == {'edoras', 'westemnet'}, \
        "companion in Westemnet -> {edoras, westemnet} only: %r" % (regs,)
    # companion inside a BESIEGED Helm's Deep -> HD offered (even under siege)
    st2 = setup_game()
    st2.besieged_strongholds = {sanitize("Helm's Deep")}
    army_at(st2, "Helm's Deep").leaders.append(
        gs.Companion(name='Eowyn', level=1, leadership=1))
    regs2 = {o['region'] for o in get_card_options(st2, Side.FREE_PEOPLES, name)}
    assert sanitize("Helm's Deep") in regs2, \
        "companion inside besieged Helm's Deep -> HD offered: %r" % (regs2,)
    # no board companion -> Edoras only
    st3 = setup_game()
    assert get_card_options(st3, Side.FREE_PEOPLES, name) == \
        [{'region': 'edoras'}], "no board companion -> Edoras only"
    # a Fellowship-parked companion does NOT unlock its region
    st4 = setup_game()
    st4.fellowship.region = st4.board.region('westemnet')
    if not any(c.name == 'Eowyn' for c in st4.fellowship.companions):
        st4.fellowship.companions.append(
            gs.Companion(name='Eowyn', level=1, leadership=1))
    regs4 = {o['region'] for o in get_card_options(st4, Side.FREE_PEOPLES, name)}
    assert regs4 == {'edoras'}, \
        "fellowship-parked companion does not unlock Westemnet: %r" % (regs4,)


@clause("CL-237", "6-cards", "1.211d House of the Stewards: Boromir ALONE (no army) in a Gondor region musters an Elite (army created); Osgiliath is not Gondor (req False); Boromir in the Fellowship is unplayable; a Shadow-CAPTURED Gondor region gets NO muster and control is NOT flipped; with empty pools the card still plays and draws 2", page='card')
def cl_237():
    import event_effects as ee
    hos, _c = _card_by_norm('houseofthestewards')

    def _sep_boromir(s, region):
        s.fellowship.companions = [c for c in s.fellowship.companions
                                   if c.name != 'Boromir']
        army_at(s, region).leaders.append(
            gs.Companion(name='Boromir', level=2, leadership=1))

    # (1) Boromir ALONE (no units) in Pelargir -> Elite + army created
    def _p1(s):
        _sep_boromir(s, 'pelargir')
        army_at(s, 'pelargir').units.clear()
    st, res = _l3_play(hos, {'boromir_region': 'pelargir'}, None, prep=_p1)
    assert any('elite' in e.lower() and 'pelargir' in e.lower()
               for e in res['effects']), \
        "Boromir alone musters an Elite: %r" % (res['effects'],)
    assert any(u.nation == Nation.GONDOR and u.kind == Kind.ELITE
               for u in army_at(st, 'pelargir').units), \
        "Gondor elite placed into the created army"

    # (2) Osgiliath is NOT a Gondor region -> requirement False
    st2 = setup_game()
    _sep_boromir(st2, 'osgiliath')
    assert not ee.check_event_requirement(st2, hos)[0], \
        "Osgiliath is not Gondor -> unplayable"

    # (3) Boromir INSIDE the Fellowship -> unplayable (board-only, doctrine l)
    st3 = setup_game()
    st3.fellowship.region = st3.board.region('minas tirith')
    if not any(c.name == 'Boromir' for c in st3.fellowship.companions):
        st3.fellowship.companions.append(
            gs.Companion(name='Boromir', level=2, leadership=1))
    assert not ee.check_event_requirement(st3, hos)[0], \
        "Boromir in the Fellowship never satisfies"

    # (4) Shadow-CAPTURED Gondor region -> NO muster, control NOT flipped
    def _p4(s):
        _sep_boromir(s, 'pelargir')
        army_at(s, 'pelargir').units.clear()
        s.enemy_controlled.setdefault(Side.SHADOW, set()).add(
            sanitize('pelargir'))
    st4, res4 = _l3_play(hos, {'boromir_region': 'pelargir'}, None, prep=_p4)
    assert not any(u.nation == Nation.GONDOR
                   for u in army_at(st4, 'pelargir').units), \
        "no Gondor muster into a captured settlement (1.211d leg ii)"
    assert sanitize('pelargir') in \
        st4.enemy_controlled.get(Side.SHADOW, set()), \
        "control NOT flipped (still Shadow-captured)"
    assert any('Drew 2' in e for e in res4['effects']), "still draws 2"

    # (5) empty Gondor pool -> still playable, draws 2
    def _p5(s):
        _sep_boromir(s, 'pelargir')
        _l3_exhaust(s, Nation.GONDOR, 'forlindon')
    st5, res5 = _l3_play(hos, {'boromir_region': 'pelargir'}, None, prep=_p5)
    assert not res5.get('refund') and \
        any('Drew 2' in e for e in res5['effects']), \
        "empty pool: still played, draws 2: %r" % (res5,)


@clause("CL-238", "6-cards", "1.211d Grey Company draw-only playability: Strider with an ELITES-ONLY Free Peoples Army satisfies the requirement, the play draws 2 Strategy cards and is NOT refunded, and no Regular is upgraded (none present)", page='card')
def cl_238():
    import event_effects as ee
    grey, _c = _card_by_norm('thegreycompany')

    def _prep(s):
        a = place(s, 'pelargir', Nation.GONDOR, elites=2, clear=True)
        a.leaders.append(gs.Companion(name='Strider', level=3, leadership=1))

    st0 = setup_game(); _prep(st0)
    assert ee.check_event_requirement(st0, grey)[0], \
        "Strider with an FP Army satisfies the requirement (elites-only)"
    st, res = _l3_play(grey, {'strider_region': 'pelargir',
                              'nation': 'gondor'}, None, prep=_prep)
    assert not res.get('refund'), "elites-only Grey Company is NOT refunded"
    assert any('Drew 2' in e for e in res['effects']), \
        "draws 2 Strategy cards: %r" % (res['effects'],)
    a = army_at(st, 'pelargir')
    assert sum(1 for u in a.units if u.kind == Kind.REGULAR) == 0 and \
        sum(1 for u in a.units if u.kind == Kind.ELITE) == 2, \
        "no upgrade (no Regular to convert): %r" % (a.units,)


@clause("CL-239", "6-cards", "1.211e Paths of the Woses source: a lone ELF Regular in Fords of Isen (a Rohan region) with Rohan At War offers the move {fords of isen -> minas tirith} (units need NOT be Rohan); with Rohan NOT At War the card is unplayable", page='card')
def cl_239():
    import event_effects as ee
    paths, _c = _card_by_norm('pathsofthewoses')
    # lone Elf Regular in Fords of Isen, Rohan NOT At War -> unplayable
    st = setup_game()
    place(st, 'fords of isen', Nation.ELVES, regulars=1, clear=True)
    ok, why = ee.check_event_requirement(st, paths)
    assert ok is False and 'At War' in (why or ''), \
        "Rohan not At War -> unplayable: %r" % (why,)
    # Rohan At War -> playable; the Elf source is legal (non-Rohan unit)
    st.political.boxes[Nation.ROHAN] = 3
    st.political.active[Nation.ROHAN] = True
    assert ee.check_event_requirement(st, paths)[0], \
        "Rohan At War + FP army in a Rohan region -> playable"
    assert {'from_region': 'fords of isen',
            'to_region': 'minas tirith'} in ee._paths_of_woses_options(st), \
        "Elf in Fords of Isen is a legal source (units need not be Rohan)"


@clause("CL-240", "6-cards", "1.211l Through a Day and a Night gate: an FP Army containing a BOARD Companion is playable; with companions only in the Fellowship it is unplayable; a Fellowship-only Companion never satisfies the gate", page='card')
def cl_240():
    import event_effects as ee
    tdaan, _c = _card_by_norm('throughadayandanight')
    # default setup: companions only in the Fellowship -> unplayable
    st = setup_game()
    ok, why = ee.check_event_requirement(st, tdaan)
    assert ok is False and 'Companion' in (why or ''), \
        "Fellowship-only companions -> unplayable: %r" % (why,)
    # an FP Army containing a board Companion -> playable
    st2 = setup_game()
    a = next(x for k, x in st2.armies.items()
             if any(u.nation in ee.FP_NATIONS for u in x.units))
    a.leaders.append(gs.Companion(name='Legolas', level=2, leadership=1))
    assert ee.check_event_requirement(st2, tdaan)[0], \
        "FP Army with a board Companion -> playable"
    # no Companion anywhere -> unplayable
    st3 = setup_game()
    st3.fellowship.companions = []
    assert not ee.check_event_requirement(st3, tdaan)[0], \
        "no Companion anywhere -> unplayable"


# ---------------------------------------------------------------------------
# MED CLUSTER B: combat math (Sudden Strike / Shield-Wall / Onslaught /
#                Dreadful Spells / Daring Defiance)  (CL-241..246)
# ---------------------------------------------------------------------------

@clause("CL-241", "4-battle", "Sudden Strike / Charge additional attack uses the ROUND's base hit number (field 5+; a stronghold assault 6+ every round, a city/fort 6+ only in round 1), not a hardcoded 5+; the DEFENDER's own additional attack stays 5+ even in a siege", page='card')
def cl_241():
    from cards import get_card
    ss = get_card('FREE_PEOPLES_Character_Challenge_of_the_King').to_combat_card()

    def probe(is_attacker, is_siege=False, city=False, rnd=1):
        c = gs.CombatState(
            attacker_region='dagorlad', defender_region='minas tirith',
            attacker_army=gs.Army(region='dagorlad'),
            defender_army=gs.Army(region='minas tirith'),
            attacker_side=Side.FREE_PEOPLES, defender_side=Side.SHADOW,
            attacker_card=(ss if is_attacker else None),
            defender_card=(None if is_attacker else ss))
        c.is_siege = is_siege
        c.city_defense = city
        c.siege_round = rnd
        if is_attacker:
            c.attacker_leadership = 3
        else:
            c.defender_leadership = 3
        _orig = cb.roll_d6_batch
        cb.roll_d6_batch = lambda n, ctx='', **kw: [5] * n          # forced 5s
        try:
            cb._apply_pre_combat_for_side(c, is_attacker=is_attacker)
        finally:
            cb.roll_d6_batch = _orig
        return (c.attacker_pre_combat_hits if is_attacker
                else c.defender_pre_combat_hits)

    assert probe(True) == 3, "field attacker: base 5+ -> [5,5,5] = 3 hits"
    assert probe(True, is_siege=True) == 0, "siege attacker: base 6+ -> 0 hits"
    assert probe(True, city=True, rnd=1) == 0, "city round 1: base 6+ -> 0 hits"
    assert probe(True, city=True, rnd=2) == 3, "city round 2: back to 5+ -> 3"
    assert probe(False, is_siege=True) == 3, \
        "DEFENDER additional attack stays 5+ even in a siege -> 3 hits"


@clause("CL-242", "4-battle", "Shield-Wall's 2+ trigger reads ONLY the opponent's Combat roll + Leader re-roll (defender_hits / attacker_hits); pre-combat additional attacks and post-reroll bonus hits are OUT of the printed scope", page='card')
def cl_242():
    from cards import get_card
    # A1 (2026-08-24): Book of Mazarbul's combat side is Advantageous
    # Position (client + physical card; the CSV said Shield-Wall for
    # that one card only) — the Shield-Wall specimen is King Brand's
    # Men, a client-confirmed Shield-Wall carrier.
    sw = get_card("FREE_PEOPLES_Strategy_King_Brand's_Men")
    assert sw is not None and sw.combat_name == 'Shield-Wall', \
        'specimen must carry Shield-Wall: %r' % (sw and sw.combat_name,)
    sw = sw.to_combat_card()

    def probe(hits, pre, bonus):
        c = gs.CombatState(
            attacker_region='dagorlad', defender_region='osgiliath',
            attacker_army=gs.Army(region='dagorlad'),
            defender_army=gs.Army(region='osgiliath'),
            attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
            defender_card=sw)
        c.attacker_hits = hits
        c.attacker_pre_combat_hits = pre
        c.attacker_bonus_hits = bonus
        cb.apply_casualty_prevention(c)
        return c.defender_hits_cancelled

    assert probe(2, 0, 0) == 1, "2 combat hits -> cancel exactly 1"
    assert probe(1, 0, 0) == 0, "1 combat hit -> no cancel"
    assert probe(1, 1, 0) == 0, "a pre-combat hit does NOT add to the trigger"
    assert probe(0, 2, 0) == 0, "two PRE-combat hits never trigger Shield-Wall"
    assert probe(1, 0, 1) == 0, "a post-reroll bonus hit is OUT of scope"
    assert probe(3, 0, 0) == 1, "3 combat hits -> still cancels exactly 1"


@clause("CL-243", "4-battle", "Onslaught HIT-capacity cap (2*elites + regulars) permits a FULL self-wipe; the deferred path eliminates the wiped side's leaders and pops the entry once both sides are gone; a self-wiped ATTACKER captures nothing, a self-wiped DEFENDER yields the settlement to the surviving attacker", page='card')
def cl_243():
    # Leg A: an ATTACKER self-wipe in the field captures nothing.
    st = setup_game()
    own = army_at(st, 'morannon')
    own.units = [gs.Unit(Nation.SAURON, Kind.REGULAR),
                 gs.Unit(Nation.SAURON, Kind.REGULAR)]
    own.leaders = [gs.Nazgul(Nation.SAURON)]
    enemy = army_at(st, 'dagorlad')
    enemy.units = [gs.Unit(Nation.GONDOR, Kind.REGULAR)]
    enemy.leaders = []
    _prev = cb.set_active_state(st)
    _orig = cb.roll_d6_batch
    cb.roll_d6_batch = lambda n, ctx='', **kw: [1] * n              # forced all-miss
    try:
        # 2-unit army -> HIT-capacity cap == 2 (no keep-one floor)
        assert cb.onslaught_max_sacrifice(st, 'morannon', Side.SHADOW) == 2, \
            "2 regulars -> capacity cap 2"
        own_dead, enemy_dead, _rolls, _hits = cb.apply_onslaught_choice(
            st, 'morannon', 'dagorlad', Side.SHADOW, Side.FREE_PEOPLES,
            count=2, enemy_is_defender=True)
    finally:
        cb.roll_d6_batch = _orig
        cb.set_active_state(_prev)
    assert own_dead and not enemy_dead, \
        "attacker fully self-wiped; enemy survives the all-miss roll"
    assert cb._army_entry(st, 'morannon')[1] is None, \
        "wiped attacker entry (units + leaders gone) is popped"
    assert sanitize('morannon') not in st.enemy_controlled.get(
        Side.FREE_PEOPLES, set()), "a self-wiped ATTACKER captures nothing"

    # Leg B: a besieged DEFENDER self-wipe yields Minas Tirith to the Shadow.
    st2 = setup_game()
    mt = army_at(st2, 'minas tirith')
    mt.units = [gs.Unit(Nation.GONDOR, Kind.REGULAR),
                gs.Unit(Nation.GONDOR, Kind.REGULAR),
                gs.Unit(Nation.SAURON, Kind.REGULAR),
                gs.Unit(Nation.SAURON, Kind.REGULAR)]
    mt.leaders = [gs.Leader(Nation.GONDOR)]
    st2.besieged_strongholds = {sanitize('minas tirith')}
    st2.fp_casualties = {}
    _prev2 = cb.set_active_state(st2)
    _orig2 = cb.roll_d6_batch
    cb.roll_d6_batch = lambda n, ctx='', **kw: [1] * n
    try:
        own_dead2, enemy_dead2, _r2, _h2 = cb.apply_onslaught_choice(
            st2, 'minas tirith', 'minas tirith',
            Side.FREE_PEOPLES, Side.SHADOW,
            count=2, enemy_is_defender=False)
    finally:
        cb.roll_d6_batch = _orig2
        cb.set_active_state(_prev2)
    ent = cb._army_entry(st2, 'minas tirith')[1]
    assert own_dead2 and not enemy_dead2, "FP garrison fully self-wiped"
    assert ent is not None, "shared entry survives (Shadow besieger remains)"
    assert cb._side_units(ent, Side.FREE_PEOPLES) == [], "no FP units remain"
    assert cb._side_units(ent, Side.SHADOW), "Shadow besieger survived"
    assert not any(type(l).__name__ == 'Leader' for l in ent.leaders), \
        "the FP plain garrison Leader is eliminated with the army"
    assert st2.fp_casualties.get(Nation.GONDOR, {}).get('leader', 0) >= 1, \
        "the garrison Leader death is BOOKED (permanent, rulebook p.30)"
    assert sanitize('minas tirith') in st2.enemy_controlled.get(
        Side.SHADOW, set()), \
        "self-wiped DEFENDER yields Minas Tirith to the surviving Shadow"


@clause("CL-244", "4-battle", "Onslaught direct path rolls ONE die per HIT inflicted (max_sacrifice), not per figure removed -- an Elite absorbs 2 hits, so 4 hits remove only 2 elites yet still roll 4 dice", page='card')
def cl_244():
    from cards import get_card
    onsl = get_card('SHADOW_Strategy_The_Fighting_Uruk-hai').to_combat_card()
    atk = gs.Army(region='morannon')
    for _ in range(5):
        atk.units.append(gs.Unit(Nation.SAURON, Kind.ELITE))
    dfn = gs.Army(region='osgiliath')
    dfn.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    c = gs.CombatState(
        attacker_region='morannon', defender_region='osgiliath',
        attacker_army=atk, defender_army=dfn,
        attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
        attacker_card=onsl)
    # 5 elites -> hit capacity 10 -> max_sacrifice = min(4, 10) = 4
    captured = []
    _orig = cb.roll_d6_batch
    cb.roll_d6_batch = lambda n, ctx='', **kw: (captured.append(n) or [1] * n)
    try:
        cb._apply_post_casualty_for_side(c, is_attacker=True, choices={})
    finally:
        cb.roll_d6_batch = _orig
    assert captured == [4], \
        "one die per HIT (max_sacrifice=4), not per figure removed: %r" % (
            captured,)
    assert len(atk.units) == 3, \
        "4 hits removed exactly two elites (each absorbs 2): %d left" % (
            len(atk.units),)


@clause("CL-245", "6-cards", "Dreadful Spells wiping a SHARED besieged FP garrison (Shadow besieger keeps the entry non-empty, so remove_orphaned_leaders can't reach it) books the FP plain Leader as a PERMANENT casualty and strips it, keeps the Companion with the capturing Shadow army, and captures the stronghold", page='card')
def cl_245():
    import event_effects as ee
    from cards import get_card
    CID = 'SHADOW_Character_Dreadful_Spells'
    st = setup_game()
    mt = army_at(st, 'minas tirith')
    mt.units = [gs.Unit(Nation.GONDOR, Kind.REGULAR),
                gs.Unit(Nation.SAURON, Kind.REGULAR)]
    mt.leaders = [gs.Leader(Nation.GONDOR),
                  gs.Companion(name='Boromir', level=2, leadership=1),
                  gs.Nazgul(Nation.SAURON)]
    st.besieged_strongholds = {sanitize('minas tirith')}
    st.fp_casualties = {}
    choices = {'target_region': sanitize('minas tirith'),
               'target_fp_army': sanitize('minas tirith')}
    _orig = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [5] * n       # forced 5s (5+ hits)
    try:
        ee._effect_dreadful_spells(st, get_card(CID), choices)
    finally:
        ee._d6_batch = _orig
    ent = next((a for k, a in st.armies.items()
                if sanitize(k) == sanitize('minas tirith')), None)
    assert ent is not None, "the shared besieged entry survives"
    assert [u for u in ent.units if u.nation not in ee.SHADOW_NATIONS] == [], \
        "the FP garrison unit is gone"
    assert not any(type(l).__name__ == 'Leader' for l in ent.leaders), \
        "no FP plain Leader remains (eliminated with the garrison)"
    assert any(isinstance(l, gs.Companion) for l in ent.leaders), \
        "the Companion is retained with the capturing Shadow army"
    assert st.fp_casualties.get(Nation.GONDOR, {}).get('leader', 0) >= 1, \
        "the garrison Leader death is BOOKED (fp_casualties, rulebook p.30)"
    assert sanitize('minas tirith') in st.enemy_controlled.get(
        Side.SHADOW, set()), "Minas Tirith is captured by the Shadow"


@clause("CL-246", "4-battle", "Daring Defiance forfeits each participating Companion's LEADERSHIP RATING (crowned Aragorn = 2), not a flat 1-per-head, and cancels the Shadow combat card", page='card')
def cl_246():
    from cards import get_card
    dd = get_card('FREE_PEOPLES_Character_I_Will_Go_Alone').to_combat_card()
    shc = get_card('SHADOW_Strategy_The_Fighting_Uruk-hai').to_combat_card()

    def probe(comps, ldr):
        c = gs.CombatState(
            attacker_region='dagorlad', defender_region='osgiliath',
            attacker_army=gs.Army(region='dagorlad'),
            defender_army=gs.Army(region='osgiliath'),
            attacker_side=Side.FREE_PEOPLES, defender_side=Side.SHADOW,
            attacker_card=dd, defender_card=shc)
        c.companions_in_battle = comps
        c.attacker_leadership = ldr
        cb.apply_cancel_effects(c)
        return c.attacker_leadership, c.defender_card_cancelled

    aragorn = gs.Companion(name='Aragorn', level=3, leadership=2)
    legolas = gs.Companion(name='Legolas', level=2, leadership=1)
    gimli = gs.Companion(name='Gimli', level=2, leadership=1)
    assert probe([aragorn], 2) == (0, True), \
        "crowned Aragorn forfeits his rating (2), card cancelled"
    assert probe([legolas], 1) == (0, True), "a level-1 Companion forfeits 1"
    assert probe([legolas, gimli], 2) == (0, True), \
        "two level-1 Companions forfeit 2 total"


# ---------------------------------------------------------------------------
# MED CLUSTER C: decisions / gates / guards (CL-247..256)
# ---------------------------------------------------------------------------

def _card_by_norm(norm):
    """(card_id, card) for the card whose normalized name == norm."""
    for cid, c in get_cards().items():
        if ai._normalize_card_name(getattr(c, 'card_name', '')) == norm:
            return cid, c
    return None, None


@clause("CL-247", "6-cards", "1.211f Worn with Sorrow SOURCE: with BOTH an FP hand Character card and an FP table Character card, a companion casualty defers a LEARNED Shadow pending [table..., hand] (NO decline, 1.210g); the table pick removes the table card (hand untouched), the hand pick removes a hand card (table untouched); a SINGLE source resolves directly with NO pending", page='card')
def cl_247():
    import event_effects as ee  # noqa
    hand_card = next((cid for cid in get_cards()
                      if cid.startswith('FREE') and '_Character_' in cid),
                     None)
    assert hand_card is not None, "an FP Character card id resolves"

    def _mk(with_hand=True, with_table=True):
        st = setup_game()
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='SHADOW_Character_Worn', card_name='Worn with Sorrow and Toil',
            side=Side.SHADOW,
            card_type=gs.PersistentCardType.WORN_WITH_SORROW))
        if with_table:
            st.persistent_cards.add_card(gs.PersistentCard(
                card_id='FREE_Character_Horn_of_Gondor',
                card_name='Horn of Gondor', side=Side.FREE_PEOPLES))
        st.hands[Side.FREE_PEOPLES].hand = [hand_card] if with_hand else []
        return st

    # both sources -> DEFER, then a learned pending [table, hand]
    st = _mk()
    game_logic._on_companion_casualty(st, 'Boromir')
    assert getattr(st, '_wwsat_owed', None) == [True], \
        "both sources: casualty is deferred to the learned queue"
    assert getattr(st, 'pending_card_option', None) is None, \
        "no pending until drained"
    ai._offer_wwsat_discard(st)
    pco = st.pending_card_option
    assert pco is not None and pco['card_name'] == '__wwsat_discard__' \
        and pco['side'] == 'sh' and pco['decider'] == 'sh'
    opts = pco['options']
    assert opts == [{'wwsat_source': 'table',
                     'card_id': 'FREE_Character_Horn_of_Gondor'},
                    {'wwsat_source': 'hand'}], \
        "options are exactly [table..., hand] (no decline): %r" % (opts,)

    # TABLE pick removes the table card, leaves the hand card
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='card_option',
        params={'idx': 0, 'card_id': 'Worn with Sorrow and Toil',
                'card_name': '__wwsat_discard__', 'side': 'sh',
                'choices': opts[0]}))
    assert not any(c.card_id == 'FREE_Character_Horn_of_Gondor'
                   for c in st.persistent_cards.fp_cards), "table card killed"
    assert hand_card in st.hands[Side.FREE_PEOPLES].hand, "hand untouched"

    # HAND pick removes the hand card, leaves the table card
    st = _mk()
    game_logic._on_companion_casualty(st, 'Boromir')
    ai._offer_wwsat_discard(st)
    hopt = st.pending_card_option['options'][-1]
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='card_option',
        params={'idx': 1, 'card_id': 'Worn with Sorrow and Toil',
                'card_name': '__wwsat_discard__', 'side': 'sh',
                'choices': hopt}))
    assert hand_card not in st.hands[Side.FREE_PEOPLES].hand, "hand card gone"
    assert any(c.card_id == 'FREE_Character_Horn_of_Gondor'
               for c in st.persistent_cards.fp_cards), "table untouched"

    # SINGLE source (hand only) -> direct discard, no pending, no queue
    st = _mk(with_hand=True, with_table=False)
    game_logic._on_companion_casualty(st, 'Boromir')
    assert not getattr(st, '_wwsat_owed', None), "single source not deferred"
    assert getattr(st, 'pending_card_option', None) is None, \
        "single source opens no pending"
    assert hand_card not in st.hands[Side.FREE_PEOPLES].hand, "hand discarded"


@clause("CL-248", "6-cards", "1.211j Nazgul Strike!: with 4 Nazgul ELSEWHERE and NONE co-located with the Ring-bearers, BOTH arms are refused - the Hunt does not roll and the FP table Character card is not discarded", page='card')
def cl_248():
    import event_effects as ee
    _, card = _card_by_norm('nazgulstrike')
    st = setup_game()
    fsp = sanitize(game_logic.fellowship_region(st))
    other = 'barad dur' if fsp != 'barad dur' else 'minas morgul'
    place(st, other, Nation.SAURON, nazgul=4, clear=True)
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_Character_Horn_of_Gondor', card_name='Horn of Gondor',
        side=Side.FREE_PEOPLES))
    n_pool = len(st.hunt_pool)
    r1 = ee._effect_nazgul_strike(st, card, {'action': 'hunt'})
    assert 'No Nazgul' in r1.get('message', ''), \
        "Hunt arm refused with no co-located Nazgul: %r" % (r1,)
    assert len(st.hunt_pool) == n_pool, "no Hunt tile drawn"
    r2 = ee._effect_nazgul_strike(
        st, card, {'action': 'discard_table_card',
                   'target_card': 'FREE_Character_Horn_of_Gondor'})
    assert 'No Nazgul' in r2.get('message', ''), "discard arm refused"
    assert st.persistent_cards.has_card('Horn of Gondor'), \
        "table Character card not discarded"


@clause("CL-249", "6-cards", "1.211j Nazgul Strike! discard target validation: co-located Nazgul present -> targeting a Shadow card or an FP STRATEGY table card is REFUSED (wrong deck/side); targeting an FP CHARACTER-deck table card removes it", page='card')
def cl_249():
    import event_effects as ee
    _, card = _card_by_norm('nazgulstrike')
    st = setup_game()
    fsp = sanitize(game_logic.fellowship_region(st))
    place(st, fsp, Nation.SAURON, nazgul=1)          # co-located Nazgul
    fp_char = next(cid for cid in get_cards()
                   if cid.startswith('FREE') and '_Character_' in cid)
    fp_strat = next(cid for cid in get_cards()
                    if cid.startswith('FREE') and '_Strategy_' in cid)
    sh_card = next(cid for cid in get_cards() if cid.startswith('SHADOW'))
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id=fp_char, card_name=get_card(fp_char).card_name,
        side=Side.FREE_PEOPLES))
    # a Shadow card target -> refused (not FP)
    r_sh = ee._effect_nazgul_strike(
        st, card, {'action': 'discard_table_card', 'target_card': sh_card})
    assert 'FP Character' in r_sh.get('message', ''), \
        "Shadow-card target refused: %r" % (r_sh,)
    # an FP Strategy target -> refused (wrong deck)
    r_st = ee._effect_nazgul_strike(
        st, card, {'action': 'discard_table_card', 'target_card': fp_strat})
    assert 'FP Character' in r_st.get('message', ''), \
        "FP Strategy target refused: %r" % (r_st,)
    # the FP Character table card -> removed
    r_ok = ee._effect_nazgul_strike(
        st, card, {'action': 'discard_table_card', 'target_card': fp_char})
    assert not any(c.card_id == fp_char
                   for c in st.persistent_cards.fp_cards), \
        "FP Character table card removed: %r" % (r_ok,)


@clause("CL-250", "6-cards", "1.211g / row 33 Horn of Gondor: 3 hunt damage with Horn on the table -> the discard-to-reduce is OFFERED BEFORE the sacrifice pending; using Horn discards it and reduces 3->2; sacrificing Meriadoc(L1) absorbs 1 -> exactly 1 corruption (reduction STACKS with the sacrifice)", page='card')
def cl_250():
    st = setup_game()
    st.fellowship.companions = [gs.Companion(name='Meriadoc', level=1,
                                             leadership=0)]
    st.fellowship.guide = st.fellowship.companions[0]
    st.fellowship.corruption = 0
    st.fellowship.hidden = True
    st.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_Character_Horn_of_Gondor', card_name='Horn of Gondor',
        side=Side.FREE_PEOPLES,
        card_type=gs.PersistentCardType.HORN_OF_GONDOR))
    st._pending_hunt_damage = 3
    opened = game_logic._offer_hunt_reduction(st)
    assert opened is True, "Horn discard-to-reduce is offered"
    assert st.pending_card_option is not None \
        and st.pending_card_option.get('hunt_reduction'), "reduction pending"
    assert st.pending_hunt is None, "the Horn offer PRECEDES the sacrifice"
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='card_option',
        params={'idx': 0, 'card_id': 'FREE_Character_Horn_of_Gondor',
                'card_name': 'Horn of Gondor', 'side': 'fp',
                'choices': {'use_hunt_reduction': 1,
                            'card_id': 'FREE_Character_Horn_of_Gondor'}}))
    assert not st.persistent_cards.has_card('Horn of Gondor'), "Horn discarded"
    assert st.pending_hunt is not None \
        and st.pending_hunt.get('phase') == 'hunt_sacrifice', \
        "a sacrifice pending confronts the reduced damage"
    assert int(st.pending_hunt.get('damage')) == 2, "3 - 1 (Horn) = 2"
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='hunt_sacrifice_random', params={}))
    assert st.fellowship.corruption == 1, \
        "2 damage - 1 absorbed (Meriadoc L1) = 1 corruption: %r" \
        % (st.fellowship.corruption,)


@clause("CL-251", "6-cards", "row 42 Stormcrow gate + row 26/1.211i The Last Battle: Stormcrow is UNPLAYABLE with no not-At-War FP nation holding the Ring-bearers/a separated Companion, PLAYABLE once a Companion sits in a not-At-War Gondor region; playing it retreats Gondor and eliminates one Gondor piece; and The Last Battle discards on fellowship DECLARATION", page='card')
def cl_251():
    import event_effects as ee
    scid, scard = _card_by_norm('stormcrow')
    assert scid is not None, "Stormcrow card id resolves"

    # (a) no valid nation -> gate False (fellowship's Elves forced At War)
    st = setup_game()
    st.political.boxes[Nation.ELVES] = 3          # At War -> not a target
    assert not ee._stormcrow_valid_nations(st), "no valid Stormcrow nation"
    ok, why = ee.check_event_requirement(st, scid)
    assert ok is False and 'not-At-War' in (why or ''), \
        "unplayable with no target: %r" % (why,)

    # (b) a separated Companion in a not-At-War Gondor region -> playable
    st.political.boxes[Nation.GONDOR] = 2         # active-ish, not At War
    army_at(st, 'minas tirith').leaders.append(
        gs.Companion(name='Boromir', level=2, leadership=1))
    place(st, 'minas tirith', Nation.GONDOR, regulars=1)
    assert Nation.GONDOR in ee._stormcrow_valid_nations(st)
    ok2, _ = ee.check_event_requirement(st, scid)
    assert ok2 is True, "playable once a Companion holds a Gondor region"

    # (c) playing it: retreat Gondor 1 + eliminate one Gondor piece
    box0 = st.political.boxes[Nation.GONDOR]
    n_gondor0 = sum(1 for u in army_at(st, 'minas tirith').units
                    if u.nation == Nation.GONDOR)
    ee._effect_stormcrow(st, scard, {'nation': 'gondor'})
    assert st.political.boxes[Nation.GONDOR] == box0 - 1, "Gondor retreated 1"
    n_gondor1 = sum(1 for u in army_at(st, 'minas tirith').units
                    if u.nation == Nation.GONDOR)
    assert n_gondor1 == n_gondor0 - 1, "one Gondor piece eliminated"

    # (d) 1.211i: The Last Battle discards on DECLARATION (not just reveal)
    st2 = setup_game()
    st2.persistent_cards.add_card(gs.PersistentCard(
        card_id='FREE_Strategy_The_Last_Battle', card_name='The Last Battle',
        side=Side.FREE_PEOPLES,
        card_type=gs.PersistentCardType.LAST_BATTLE))
    st2.fellowship.hidden = True
    st2.fellowship.progress = 0
    game_logic.declare_fellowship(
        st2, region=game_logic.fellowship_region(st2))
    assert not st2.persistent_cards.has_card_type(
        gs.PersistentCardType.LAST_BATTLE), \
        "The Last Battle discarded on declaration"


@clause("CL-252", "6-cards", "row 31 Grond: a co-located Witch-king siege (besieged FP stronghold with the Witch-king among the besiegers) runs the 3-round siege; with NO valid co-located siege the play is REFUNDED (card kept, die unspent) - never the old adjacent-scan, and never the old bare-Sauron-unit proxy (game 16 phantom siege)", page='card')
def cl_252():
    import event_effects as ee
    _, card = _card_by_norm('grondhammeroftheunderworld')
    # co-located WK+Sauron siege of Minas Tirith.  FIX (game 16): Grond
    # requires the Witch-king among the besiegers, not merely a Sauron
    # unit - the fixture now places him (was a stale bare-Sauron setup).
    st = setup_game()
    place(st, 'minas tirith', Nation.GONDOR, regulars=2, clear=True)  # garrison
    place(st, 'minas tirith', Nation.SAURON, regulars=3)              # besieger
    army_at(st, 'minas tirith').leaders.append(
        game_logic.Minion(Nation.SAURON, name='Witch-king of Angmar'))
    st.besieged_strongholds = {'minas tirith'}
    assert cb.garrison_side_for(st, 'minas tirith') == Side.FREE_PEOPLES
    r = ee._effect_grond(st, card, {})
    assert any('Grond siege' in e for e in r.get('effects', [])), \
        "co-located siege runs: %r" % (r,)
    assert not r.get('refund'), "a valid siege is not refunded"

    # no valid siege -> refund (card kept, die unspent)
    st2 = setup_game()
    r2 = ee._effect_grond(st2, card, {})
    assert any('No co-located Witch-king siege' in e
               for e in r2.get('effects', [])), "no-siege message: %r" % (r2,)
    assert r2.get('refund') is True, "no valid siege -> self-play refund"


@clause("CL-253", "6-cards", "1.211h/row 43 besieged-garrison source guards: a besieged SHADOW garrison cannot march OUT via a join-move (The Shadow Lengthens / Shadows Gather) or The Shadow is Moving; a Shadow BESIEGER at a besieged FP stronghold is NOT a garrison source (its relocation stays legal)", page='card')
def cl_253():
    import event_effects as ee
    st = setup_game()
    place(st, 'morannon', Nation.SAURON, regulars=3, clear=True)  # garrison
    st.besieged_strongholds = {'morannon'}
    assert cb.garrison_side_for(st, 'morannon') == Side.SHADOW
    assert ee._besieged_garrison_source(st, 'morannon', Side.SHADOW) is True
    # join-move OUT (TSL single / Shadows Gather path) refused
    res = {"effects": []}
    moved = ee._sg_join_move(st, 'morannon', 'udun', 3, res, set())
    assert moved is False and any('besieged garrison' in e
                                  for e in res['effects']), \
        "join-move out of a besieged garrison refused: %r" % (res,)
    assert len(army_at(st, 'morannon').units) == 3, "garrison stays put"
    # The Shadow is Moving out refused (find a clean adjacent destination)
    to0 = None
    for adj in st.board.adjacent('morannon'):
        asan = sanitize(adj)
        if asan in st.besieged_strongholds:
            continue
        a_to = st.armies.get(ee._find_army_key(st, asan))
        if a_to and any(u.nation not in ee.SHADOW_NATIONS
                        for u in a_to.units):
            continue
        to0 = asan
        break
    assert to0 is not None, "a clean adjacent destination exists"
    r2 = ee._effect_shadow_is_moving(
        st, None, {'from_region': 'morannon', 'to_region': to0})
    assert any('besieged garrison' in e for e in r2['effects']), \
        "TSiM out of a besieged garrison refused: %r" % (r2,)
    assert len(army_at(st, 'morannon').units) == 3, "garrison still put"
    # control: a Shadow BESIEGER at a besieged FP stronghold is NOT a source
    st2 = setup_game()
    place(st2, 'minas tirith', Nation.GONDOR, regulars=2)   # FP garrison
    place(st2, 'minas tirith', Nation.SAURON, regulars=4)   # Shadow besieger
    st2.besieged_strongholds = {'minas tirith'}
    assert cb.garrison_side_for(st2, 'minas tirith') == Side.FREE_PEOPLES
    assert ee._besieged_garrison_source(
        st2, 'minas tirith', Side.SHADOW) is False, \
        "besieger relocation stays legal (control)"


@clause("CL-254", "6-cards", "rows 10/17/1.149b Gwaihir arrival activation: separating Boromir to Minas Tirith uses the standard place_separated_companion helper, so the arrival ACTIVATES Gondor", page='card')
def cl_254():
    import event_effects as ee
    _, card = _card_by_norm('gwaihirthewindlord')
    st = setup_game()
    if not any(c.name == 'Boromir' for c in st.fellowship.companions):
        st.fellowship.companions.append(
            gs.Companion(name='Boromir', level=2, leadership=1))
    st.political.active[Nation.GONDOR] = False
    st.fellowship.progress = 20            # ensure Minas Tirith in range
    ee._effect_gwaihir(st, card, {'companion_action': 'sep_Boromir',
                                  'destination': 'minas tirith'})
    assert any(getattr(l, 'name', '') == 'Boromir'
               for l in army_at(st, 'minas tirith').leaders), \
        "Boromir separated onto the board at Minas Tirith"
    assert st.political.is_active(Nation.GONDOR), \
        "Boromir's arrival activates Gondor (place_separated_companion)"


@clause("CL-255", "6-cards", "rows 11/18/CL-255 We Prove Swifter own-level range: Meriadoc(L1) alone from progress 0 has range 3 (own level, NOT a Fellowship max), so a distance-4 region is out of range; a GROUP {Meriadoc, Legolas(L2)} uses the group max (range 4) and reaches it", page='card')
def cl_255():
    import event_effects as ee
    _, card = _card_by_norm('weproveswifter')
    st = setup_game()
    st.fellowship.companions = [
        gs.Companion(name='Meriadoc', level=1, leadership=0),
        gs.Companion(name='Legolas', level=2, leadership=1),
        gs.Companion(name='Gandalf the Grey', level=3, leadership=1)]
    st.fellowship.guide = st.fellowship.companions[2]
    st.fellowship.progress = 0
    st.fellowship.region = st.board.region('rivendell')
    # a region at distance EXACTLY 4 from Rivendell (illegal for range 3,
    # legal for range 4) with no enemy blocking on the fresh board
    target = None
    for rk in board.regions:
        if _bfs_dist(st, 'rivendell', rk) == 4:
            target = rk
            break
    assert target is not None, "a distance-4 region exists"
    # Meriadoc(L1) alone: range 3 -> distance-4 target ILLEGAL, no separation
    r1 = ee._effect_we_prove_swifter(
        st, card, {'companion_action': 'sep_Meriadoc', 'destination': target})
    assert r1.get('refund') or 'out of range' in r1.get('message', ''), \
        "Meriadoc L1 cannot reach the dist-4 region (own level): %r" % (r1,)
    assert any(c.name == 'Meriadoc' for c in st.fellowship.companions), \
        "Meriadoc stays in the Fellowship (independence from Gandalf L3)"
    assert not any(getattr(l, 'name', '') == 'Meriadoc'
                   for a in st.armies.values() for l in a.leaders), \
        "Meriadoc NOT placed on the board"
    # GROUP {Meriadoc, Legolas}: group max L2 -> range 4 -> reaches target
    r2 = ee._effect_we_prove_swifter(
        st, card, {'companion_action': 'group',
                   'companions': ['Meriadoc', 'Legolas'],
                   'destination': target})
    assert not r2.get('refund'), "group reaches the dist-4 region: %r" % (r2,)
    on_board = {getattr(l, 'name', '')
                for a in st.armies.values() for l in a.leaders}
    assert {'Meriadoc', 'Legolas'} <= on_board, \
        "group separated both companions to the target"


@clause("CL-344", "6-cards", "Balrog of Moria MOVEMENT trigger is a Shadow CHOICE and only applies to a declared/revealed move: a HIDDEN move through Moria neither draws nor discards the card; a move that leaves the Fellowship revealed opens the learned balrog_declare_draw pending instead of force-spending the one-shot card", page='card')
def cl_344():
    """USER 2026-08-20 'fix balrog'.  Card text (card_list.csv:68):
    'YOU MAY discard "Balrog of Moria" to draw an additional Hunt tile
    if the Fellowship moves into, out of, or through Moria WHILE BEING
    DECLARED OR REVEALED.'

    REPRODUCTION: game_logic.move_fellowship honoured neither half - it
    drew the tile and force-discarded the one-shot card with no Shadow
    decision, and keyed on the Fellowship's LAST KNOWN region, so a
    HIDDEN move while the marker sat in Moria fired it too.  The Shadow
    lost a card it never chose to spend."""
    import combat as _cmb
    import game_logic as _gl

    def _mk(hidden):
        st = setup_game()
        st.fellowship.region = 'moria'
        st.fellowship.hidden = hidden
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='SHADOW_Character_Balrog',
            card_name='Balrog of Moria', side=Side.SHADOW,
            card_type=gs.PersistentCardType.BALROG_OF_MORIA))
        for d in st.dice.get(Side.SHADOW, [])[:2]:
            d.current = gs.ActionDieFace.EYE
            d.used = True
            st.hunt_box.append(d)
        return st

    def _asked(st):
        p = getattr(st, 'pending_card_option', None)
        return bool(p) and any('balrog_declare_draw' in o
                               for o in (p.get('options') or []))

    assert _cmb.FINAL_HITS_HOOK is None, \
        "this leg measures the LEARNED path; the teacher hook must be off"

    # HIDDEN move: the card requires declared/revealed - nothing fires
    st = _mk(True)
    # CONTROL (2026-08-28): the hidden move still HUNTS (two Eyes in
    # the box); a hitting roll that draws a reveal tile reveals the
    # Fellowship and the Balrog then asks LEGITIMATELY.  This leg
    # measures the HIDDEN case, so the hunt roll is pinned to a miss
    # (the clause was RNG-coupled: alone it failed on seeds 0 and 4
    # and passed on 1/2/3/5; ruling 1.230's play changes shifted the
    # runner's shared stream onto a hitting roll).
    _orig_roll344 = _gl.roll_d6_batch
    _gl.roll_d6_batch = lambda n, ctx='', side=None: [1] * n
    try:
        _gl.move_fellowship(st)
    finally:
        _gl.roll_d6_batch = _orig_roll344
    assert st.fellowship.hidden, "control: the pinned miss keeps it hidden"
    assert st.persistent_cards.has_card('Balrog of Moria'), \
        "hidden move must NOT spend the Balrog (card: 'while being " \
        "declared or revealed')"
    assert not _asked(st), "hidden move must not even ask"

    # REVEALED move: the Shadow is ASKED, the card is NOT auto-spent
    st2 = _mk(False)
    _gl.move_fellowship(st2)
    assert st2.persistent_cards.has_card('Balrog of Moria'), \
        "'you may' - the movement trigger must not force-discard"
    assert _asked(st2), \
        "a revealed Moria move must open the learned " \
        "balrog_declare_draw pending"

    # ...and answering YES still draws and discards (the shared handler)
    import event_effects as ee
    _, _bcard = _card_by_norm('balrogofmoria')
    n0 = len(st2.hunt_pool)
    ee._effect_balrog_of_moria(st2, _bcard, {'balrog_declare_draw': True})
    assert not st2.persistent_cards.has_card('Balrog of Moria'), \
        "accepting spends the one-shot card"
    assert len(st2.hunt_pool) == n0 - 1, "accepting draws one extra tile"


@clause("CL-256", "6-cards", "row 29 Balrog of Moria declaration draw: hidden in an uncaptured Minas Tirith, drawing a non-Eye REVEAL tile keeps the Fellowship HIDDEN (reveal ignored in an FP City/Stronghold) and discards the Balrog; an Eye tile is discarded without effect", page='card')
def cl_256():
    import event_effects as ee
    _, card = _card_by_norm('balrogofmoria')

    def _mk(tile):
        st = setup_game()
        st.persistent_cards.add_card(gs.PersistentCard(
            card_id='SHADOW_Character_Balrog', card_name='Balrog of Moria',
            side=Side.SHADOW,
            card_type=gs.PersistentCardType.BALROG_OF_MORIA))
        st.fellowship.hidden = True
        st.fellowship.region = st.board.region('minas tirith')
        st.hunt_pool = [tile]
        return st

    # non-Eye REVEAL tile -> stays hidden (ignored in FP City/Stronghold)
    st = _mk(gs.HuntTile(damage=1, reveal=True))
    ee._effect_balrog_of_moria(st, card, {'balrog_declare_draw': True})
    assert st.fellowship.hidden is True, \
        "reveal ignored (declared in an uncaptured FP City/Stronghold)"
    assert not st.persistent_cards.has_card('Balrog of Moria'), \
        "Balrog discarded after the draw"

    # Eye tile -> discarded without effect
    st2 = _mk(gs.HuntTile(is_eye=True))
    n_before = len(st2.hunt_pool)
    ee._effect_balrog_of_moria(st2, card, {'balrog_declare_draw': True})
    assert st2.fellowship.hidden is True, "Eye draw has no effect on hiding"
    assert len(st2.hunt_pool) == n_before - 1, "Eye tile removed from pool"
    assert not st2.persistent_cards.has_card('Balrog of Moria'), \
        "Balrog discarded after an Eye draw"


# ---------------------------------------------------------------------------
# CL-257..259 - RC4 casualty-forcing v5 (row 132): parser emits the removal
# COMPOSITION, apply_casualties_spec forces the logged mix, and the
# consume-once round cursor applies an owned mark exactly once.
# ---------------------------------------------------------------------------

@clause("CL-257", "4-casualty", "RC4 v5 apply_casualties_spec forces the LOGGED mix: {kill_regular:1, downgrade:1} on a 6R+1E army removes one Regular and downgrades the Elite -> 6R,0E (downgrade product), where policy-based preserve_elites on the same 2 hits gives 4R,1E", page=30)
def cl_257():
    st = setup_game()
    prev = cb.set_active_state(st)
    try:
        # Regulars free in reinforcements -> the downgrade is allowed
        _set_reg_pool_free(st, Nation.GONDOR, 10)
        a = place(st, 'erebor', Nation.GONDOR, regulars=6, elites=1,
                  clear=True)
        cas = cb.apply_casualties_spec(
            a, {'kill_regular': 1, 'kill_elite': 0, 'downgrade': 1},
            side=Side.FREE_PEOPLES)
        nreg = sum(1 for u in a.units if u.kind == Kind.REGULAR)
        nel = cb.count_elite_units(a)
        assert nreg == 6 and nel == 0, \
            "forced mix {R:1,D:1}: 1 Regular gone + Elite downgraded -> " \
            "6R,0E, got %dR,%dE" % (nreg, nel)
        assert len(cas) == 2, \
            "two units booked (the killed Regular + the downgraded Elite): " \
            "%r" % (cas,)
        # CONTRAST: same board, policy-based 2 hits under preserve_elites
        st2 = setup_game()
        cb.set_active_state(st2)
        _set_reg_pool_free(st2, Nation.GONDOR, 10)
        b = place(st2, 'erebor', Nation.GONDOR, regulars=6, elites=1,
                  clear=True)
        cb.apply_casualties(b, 2, policy='preserve_elites')
        nreg2 = sum(1 for u in b.units if u.kind == Kind.REGULAR)
        nel2 = cb.count_elite_units(b)
        assert nreg2 == 4 and nel2 == 1, \
            "preserve_elites 2 hits removes 2 Regulars -> 4R,1E, got " \
            "%dR,%dE (proves the spec is NOT policy)" % (nreg2, nel2)
    finally:
        cb.set_active_state(prev)


@clause("CL-258", "4-casualty", "RC4 v5 parser Patch A: a NON-stronghold field elite->regular swap (Edoras) inside a battle window emits the downgrade COMPOSITION spec {R:0,E:0,D:1} with a real cause (not the blind sentinel); a stronghold swap that failed press-detection (Minas Tirith) KEEPS the weight-ambiguity sentinel cause='downgrade', spec=None", page=30)
def cl_258():
    import log_replay as lr
    st = setup_game()
    rep = lr.LogReplayer(st, board.load_board(), verbose=False)

    def _mk(region):
        # synthetic Controls.308 elite->regular swap inside an open battle
        # window with press-detection NOT satisfied (no live _combat_region)
        rep.casualty_marks = []
        rep._battle_synth_region = region
        rep._card_window_side = None
        rep._synth_attack_idx = None
        rep._cc_pre_roll = {}
        rep._last_swap_remove_elite = True
        rep._last_swap_remove_side = Side.FREE_PEOPLES
        rep._combat_region = None
        rep._combat_casualties_this_round = False
        rep._pending_304 = None
        rep._handle_piece_swap_place([], '308', region)
        return list(rep.casualty_marks)

    # NON-stronghold field swap -> composition mark, cause != sentinel
    field = _mk('edoras')
    assert len(field) == 1, \
        "one casualty mark for the field swap, got %r" % (field,)
    assert field[0]['spec'] == {'kill_regular': 0, 'kill_elite': 0,
                                'downgrade': 1}, \
        "field elite->regular swap carries the downgrade COMPOSITION: %r" \
        % (field[0],)
    assert field[0]['cause'] != 'downgrade', \
        "field composition mark is NOT the blind sentinel (cause=%r)" \
        % (field[0]['cause'],)

    # Stronghold swap that failed press-detection -> spec=None sentinel
    sh = _mk('minas tirith')
    assert len(sh) == 1, \
        "one casualty mark for the stronghold swap, got %r" % (sh,)
    assert sh[0]['spec'] is None, \
        "stronghold weight-ambiguity sentinel keeps spec=None: %r" % (sh[0],)
    assert sh[0]['cause'] == 'downgrade', \
        "stronghold sentinel keeps cause='downgrade': %r" % (sh[0],)


@clause("CL-259", "4-casualty", "RC4 v5 consume-once: one owned kill-mark is applied EXACTLY once across N engine rounds -- the round with the mark removes its composition and every engine_hits==0 markless (phantom) round forces a ZERO spec {0,0,0} that removes nothing, so a lone mark never re-applies N times", page=30)
def cl_259():
    st = setup_game()
    prev = cb.set_active_state(st)
    try:
        _set_reg_pool_free(st, Nation.GONDOR, 12)
        a = place(st, 'osgiliath', Nation.GONDOR, regulars=5, clear=True)
        # round with the single owned kill-mark (one Regular)
        cb.apply_casualties_spec(
            a, {'kill_regular': 1, 'kill_elite': 0, 'downgrade': 0},
            side=Side.FREE_PEOPLES)
        assert sum(1 for u in a.units if u.kind == Kind.REGULAR) == 4, \
            "the mark removes exactly one Regular"
        # phantom / die-less rounds: a ZERO spec removes nothing
        for _ in range(4):
            cas = cb.apply_casualties_spec(
                a, {'kill_regular': 0, 'kill_elite': 0, 'downgrade': 0},
                side=Side.FREE_PEOPLES)
            assert cas == [], "a zero spec removes nothing: %r" % (cas,)
        assert sum(1 for u in a.units if u.kind == Kind.REGULAR) == 4, \
            "no N x re-application: still 4 Regulars after the phantom rounds"
        # CONTRAST: re-applying the mark every round (the v4 cache bug)
        # would empty the army -- the consume-once cursor forces zero instead
        b = place(st, 'pelargir', Nation.GONDOR, regulars=5, clear=True)
        for _ in range(5):
            cb.apply_casualties_spec(
                b, {'kill_regular': 1, 'kill_elite': 0, 'downgrade': 0},
                side=Side.FREE_PEOPLES)
        assert sum(1 for u in b.units if u.kind == Kind.REGULAR) == 0, \
            "re-applying the mark 5x (the bug) empties the army -- the " \
            "consume-once cursor prevents exactly this"
    finally:
        cb.set_active_state(prev)


# ---------------------------------------------------------------------------
# Witch-king 'Sorcerer' first-round combat-card draw (rows 1.150 / 1.152).
# The WK minion flag makes Shadow draw an event card of the SAME deck as the
# combat card it just played, but ONLY on the FIRST round of a battle, only
# when a WK is co-located in the battle, and only in self-play (the teacher
# replays the human's logged draw via forced_draw -- no engine auto-draw).
# ---------------------------------------------------------------------------

def _find_sh_combat_card(deck):
    """A real Shadow card id for `deck` ('strategy'|'character').  The
    Sorcerer deck-match keys purely on '_Character_' in the id, so any
    Shadow card of the matching deck exercises the same code path."""
    tag = '_Character_' if deck == 'character' else '_Strategy_'
    for cid in get_cards():
        if cid.startswith('SHADOW') and tag in cid:
            return cid
    return None


def _wk_sorcerer_scenario(wk_present, rounds_fought, deck='strategy'):
    """Drive a Shadow attacker's combat-card commit (Morannon -> Dagorlad)
    and report whether the WK Sorcerer auto-draw fired plus the deck state.

    A KNOWN Shadow combat card of `deck` is placed in the SH hand and pulled
    OUT of its deck first, so the drawn card is unambiguously the former TOP
    of that deck (never the played card).  Monkeypatched emit_annotation is
    always restored in a finally block."""
    import game_trace
    st = setup_game()
    atk = place(st, 'morannon', Nation.SAURON, regulars=4, clear=True)
    if wk_present:
        atk.leaders.append(game_logic.Minion(Nation.SAURON, name='WitchKing'))
        # RULING 1.228b (2026-08-03): the Sorcerer draw now gates on the
        # character sheet (a card-killed WK's queued draw dies with him).
        # A board WK always has in_play=True in a real game (every muster
        # path sets it); mirror that invariant here.
        st.witch_king.in_play = True
    place(st, 'dagorlad', Nation.GONDOR, regulars=3, clear=True)

    cid = _find_sh_combat_card(deck)
    assert cid is not None, "a Shadow %s combat card id resolves" % deck
    own_deck = st.sh_char_deck if deck == 'character' else st.sh_strat_deck
    if cid in own_deck:
        own_deck.remove(cid)
    st.hands[Side.SHADOW].hand = [cid]
    top = own_deck[-1] if own_deck else None
    strat_before, char_before = len(st.sh_strat_deck), len(st.sh_char_deck)

    st.pending_combat = {
        'attacker_region': 'morannon', 'defender_region': 'dagorlad',
        'attacker_side': Side.SHADOW, 'defender_side': Side.FREE_PEOPLES,
        'is_siege': False, 'phase': 'attacker_combat_card',
        'deciding_side': Side.SHADOW, 'rounds_fought': rounds_fought,
    }

    anns = []
    _orig = game_trace.emit_annotation
    game_trace.emit_annotation = \
        (lambda state, kind, data=None, side=None: anns.append(kind))
    try:
        guard = 0
        while st.pending_combat and guard < 6:
            ph = st.pending_combat.get('phase')
            if ph == 'attacker_combat_card':
                act = ai.Action(die_idx=-1, action_type='combat_card_play',
                                params={'card_id': cid, 'side': 'sh'})
            elif ph == 'defender_combat_card':
                act = ai.Action(die_idx=-1, action_type='combat_card_none',
                                params={'side': 'fp'})
            else:
                break
            st = ai.apply_action(st, act)
            guard += 1
    finally:
        game_trace.emit_annotation = _orig

    return {
        'cid': cid, 'top': top,
        'fired': 'wk_sorcerer_draw' in anns,
        'hand': list(st.hands[Side.SHADOW].hand),
        'strat_before': strat_before, 'strat_after': len(st.sh_strat_deck),
        'char_before': char_before, 'char_after': len(st.sh_char_deck),
    }


@clause("CL-260", "6-cards", "1.150/1.152 Witch-king Sorcerer FIRST-round auto-draw: a co-located WK in the Shadow ATTACKER with rounds_fought=0 makes a Shadow STRATEGY combat-card play fire wk_sorcerer_draw and draw the former TOP of sh_strat_deck (deck -1, char deck untouched); a CHARACTER combat-card play instead draws from sh_char_deck (1.150 deck-MATCH lock)", page='card')
def cl_260():
    import game_trace  # noqa
    # STRATEGY combat card -> draws from the STRATEGY deck
    rs = _wk_sorcerer_scenario(wk_present=True, rounds_fought=0,
                               deck='strategy')
    assert rs['fired'] is True, "wk_sorcerer_draw fires on the first round"
    assert rs['top'] is not None and rs['top'] != rs['cid'] \
        and rs['top'] in rs['hand'], \
        "SH hand gains the former top of sh_strat_deck: %r" % (rs,)
    assert rs['strat_after'] == rs['strat_before'] - 1, \
        "strategy deck popped exactly one: %r" % (rs,)
    assert rs['char_after'] == rs['char_before'], \
        "character deck untouched by a strategy-card draw: %r" % (rs,)

    # CHARACTER combat card -> draws from the CHARACTER deck (deck-match lock)
    rc = _wk_sorcerer_scenario(wk_present=True, rounds_fought=0,
                               deck='character')
    assert rc['fired'] is True, "wk_sorcerer_draw fires for a character play"
    assert rc['top'] is not None and rc['top'] != rc['cid'] \
        and rc['top'] in rc['hand'], \
        "SH hand gains the former top of sh_char_deck: %r" % (rc,)
    assert rc['char_after'] == rc['char_before'] - 1, \
        "character deck popped exactly one (1.150 deck-match): %r" % (rc,)
    assert rc['strat_after'] == rc['strat_before'], \
        "strategy deck untouched by a character-card draw: %r" % (rc,)


@clause("CL-261", "6-cards", "1.152 Sorcerer FIRST-round lock: the same co-located-WK Shadow combat-card play with rounds_fought=1 does NOT fire wk_sorcerer_draw and leaves BOTH Shadow decks unchanged (later rounds never redraw)", page='card')
def cl_261():
    import game_trace  # noqa
    r = _wk_sorcerer_scenario(wk_present=True, rounds_fought=1,
                              deck='strategy')
    assert r['fired'] is False, "no Sorcerer draw after the first round"
    assert r['strat_after'] == r['strat_before'], \
        "strategy deck unchanged on a later round: %r" % (r,)
    assert r['char_after'] == r['char_before'], \
        "character deck unchanged on a later round: %r" % (r,)


@clause("CL-262", "6-cards", "Sorcerer WK-presence lock: a first-round Shadow combat-card play with NO Witch-king in either army does NOT fire wk_sorcerer_draw and leaves both Shadow decks unchanged", page='card')
def cl_262():
    import game_trace  # noqa
    r = _wk_sorcerer_scenario(wk_present=False, rounds_fought=0,
                              deck='strategy')
    assert r['fired'] is False, "no draw without a co-located Witch-king"
    assert r['strat_after'] == r['strat_before'], \
        "strategy deck unchanged with no WK: %r" % (r,)
    assert r['char_after'] == r['char_before'], \
        "character deck unchanged with no WK: %r" % (r,)


# CL-264..267 - v5 casualty-window robustness pack (P1 item 1, closed-loop
# teacher).  FIX A round cursor, FIX C span clamp, FIX D combat-card region
# rebind + '?' tag.  (The pack's FIX B -- a 307 downgrade-pair derivation --
# was withdrawn: its target symptom, a persistent +1 Elite at game-13 MT T4,
# does not exist in the current build, and the naive derivation regressed
# game 13, so no lock test is added for it.)
# ---------------------------------------------------------------------------

@clause("CL-273", "8-victory",
        "Both sides at their VP thresholds at turn end -> SHADOW wins the "
        "tiebreaker (1.212a, game-12 shape)", page='house')
def cl_273():
    st = setup_game()
    # Shadow 10 (5 strongholds), FP 4 (2 shadow strongholds) simultaneously
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth',
                                        'helm s deep', 'lorien', 'erebor'}
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon', 'minas morgul'}
    vp = game_logic.calculate_victory_points(st)
    assert vp['shadow'] >= 10 and vp['free_peoples'] >= 4, vp
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'shadow' and vc['victory_type'] == 'military', vc
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "both-at-threshold resolves at the end-of-turn Victory Check"
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "military never ends the game mid-turn (neg)"


@clause("CL-274", "8-victory",
        "FP at 4 VP with Shadow below 10 at turn end -> Free Peoples "
        "military win", page=17)
def cl_274():
    st = setup_game()
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon', 'minas morgul'}  # 4
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth'}      # 4
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'free_peoples' \
        and vc['victory_type'] == 'military', vc
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "FP military resolves at the end-of-turn Victory Check"
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "FP military does not end the game mid-turn (neg)"


@clause("CL-275", "8-victory",
        "A VP threshold reached MID-turn does not end the game (military "
        "waits for the Victory Check)", page=17)
def cl_275():
    st = setup_game()
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth',
                                        'helm s deep', 'lorien', 'erebor'}
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory_type'] == 'military', vc
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "a threshold reached mid-turn must NOT end the game (neg)"
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "the same threshold ends the game at the Victory Check"


@clause("CL-276", "8-victory",
        "Corruption 12 ends the game IMMEDIATELY, even mid-turn (ring "
        "ending)", page=17)
def cl_276():
    st = setup_game()
    st.fellowship.corruption = 12
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'shadow' \
        and vc['victory_type'] == 'ring_corruption', vc
    assert game_logic.should_game_end(vc, end_of_turn=False), \
        "a ring ending is immediate, not gated to the end of turn"


# CL-278..283 - 2026-07-22 rendered-log sweep (six evidence-backed fixes)

@clause("CL-279", "0-turn",
        "Item 2 guide-is-a-role: designating an already-present Companion as "
        "Guide (Fellowship Box -> Guide Box) keeps him IN the Fellowship (a "
        "role change, never a removal) and lists him EXACTLY once (no "
        "double-count); he becomes the guide", page=20)
def cl_279():
    from log_replay import LogReplayer
    st = setup_game()
    rep = LogReplayer(st, st.board, verbose=False)
    assert 'Legolas' in [c.name for c in st.fellowship.companions], \
        "precondition: Legolas starts in the Fellowship"
    # piece 242 = Legolas
    rep._handle_fellowship_companions(['242'], 'Fellowship Box', 'Guide Box')
    names = [c.name for c in st.fellowship.companions]
    assert names.count('Legolas') == 1, \
        "Legolas stays a Companion exactly once (guide is a role, no dup)"
    assert st.fellowship.guide is not None \
        and st.fellowship.guide.name == 'Legolas', \
        "Legolas holds the guide role"


@clause("CL-280", "6-cards",
        "Item 3 Faramir's Rangers auto-target + type fallback: with no logged "
        "target and the default Osgiliath empty of Shadow, the strike falls "
        "to the Osgiliath-area region that DOES hold a Shadow army (South "
        "Ithilien) and removes an AVAILABLE type (a Regular, elite-less "
        "stack) - never a silent no-op that keeps the unit", page='card')
def cl_280():
    import event_effects as ee
    st = setup_game()
    place(st, 'south ithilien', Nation.SAURON, regulars=9, clear=True)
    for k in [k for k in list(st.armies) if sanitize(k) == 'osgiliath']:
        del st.armies[k]
    a = army_at(st, 'south ithilien')
    n0 = sum(1 for u in a.units if u.nation == Nation.SAURON)
    saved = ee._d6_batch
    try:
        ee._d6_batch = lambda n: [6, 1, 1]  # exactly one hit (5+)
        res = ee._effect_faramirs_rangers(st, None, {})
    finally:
        ee._d6_batch = saved
    n1 = sum(1 for u in a.units if u.nation == Nation.SAURON)
    assert n1 == n0 - 1, \
        "the strike removes one Sauron Regular from South Ithilien (%d->%d)" \
        % (n0, n1)
    assert any('south ithilien' in str(e).lower()
               for e in res.get('effects', [])), \
        "the strike is reported against the auto-resolved target"


@clause("CL-281", "6-cards",
        "Item 4 game-8 bug A: Half-Orcs & Goblin-Men recruits its Isengard "
        "unit at the LOGGED region carried in choices (derived from the "
        "play's muster fx) - Dimrill Dale - instead of the no-region default "
        "(the first Shadow army = Orthanc), so no phantom is planted at the "
        "Isengard capital", page='card')
def cl_281():
    import event_effects as ee
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3
    place(st, 'orthanc', Nation.ISENGARD, regulars=1, clear=True)
    place(st, 'dimrill dale', Nation.SAURON, regulars=1, clear=True)
    orc_n0 = len(army_at(st, 'orthanc').units)
    ee._effect_half_orcs(st, None,
                         {'region': 'Dimrill Dale', 'unit_type': 'elite'})
    dd = army_at(st, 'dimrill dale')
    orc = army_at(st, 'orthanc')
    assert any(u.nation == Nation.ISENGARD and u.kind == Kind.ELITE
               for u in dd.units), \
        "the Isengard unit lands at the logged Dimrill Dale"
    assert len(orc.units) == orc_n0, \
        "no phantom minted at the Orthanc no-region default"


@clause("CL-282", "1-dice",
        "Item 5 / 1.213 subset-nation entry: a mixed Dale stack (Elven elite "
        "At War + North regular NOT At War) is OFFERED the Dale->Erebor "
        "crossing - legality keys on the MOVING SUBSET (the Elven detachment "
        "may enter Dwarf territory), not the whole army; a not-At-War "
        "North-only stack is still REFUSED the same foreign crossing", page=33)
def cl_282():
    st = setup_game()
    st.political.boxes[Nation.ELVES] = 3      # Elves At War
    st.political.boxes[Nation.THE_NORTH] = 1  # North NOT At War
    place(st, 'dale', Nation.ELVES, elites=1, clear=True)
    place(st, 'dale', Nation.THE_NORTH, regulars=1)
    set_dice(st, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.ARMY])
    acts = ai._get_army_move_actions(st, gs.Side.FREE_PEOPLES, 0)
    de = [a for a in acts
          if a.action_type == 'army_move'
          and sanitize(a.params.get('from', '')) == 'dale'
          and sanitize(a.params.get('to', '')) == 'erebor']
    assert de, "the Elven detachment's Dale->Erebor crossing is offered"

    st2 = setup_game()
    st2.political.boxes[Nation.THE_NORTH] = 1  # North NOT At War
    place(st2, 'dale', Nation.THE_NORTH, regulars=2, clear=True)
    set_dice(st2, gs.Side.FREE_PEOPLES, [gs.ActionDieFace.ARMY])
    acts2 = ai._get_army_move_actions(st2, gs.Side.FREE_PEOPLES, 0)
    de2 = [a for a in acts2
           if a.action_type == 'army_move'
           and sanitize(a.params.get('from', '')) == 'dale'
           and sanitize(a.params.get('to', '')) == 'erebor']
    assert not de2, \
        "a not-At-War North-only stack is refused the foreign crossing"


@clause("CL-283", "6-cards",
        "Item 6 Shadows Gather log-truth defer: on the extraction path "
        "(FINAL_HITS_HOOK installed) with NO logged from/to, the handler "
        "makes NO join move (it must not auto-pick a consolidation and "
        "drain a stack, e.g. Orthanc->North Dunland); in self-play it "
        "still auto-picks a legal join", page='card')
def cl_283():
    import event_effects as ee
    import combat as cb
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3
    place(st, 'north dunland', Nation.ISENGARD, regulars=4, clear=True)
    place(st, 'south dunland', Nation.ISENGARD, regulars=2, clear=True)

    def _census():
        return {sanitize(k): len(a.units) for k, a in st.armies.items()
                if any(u.nation in ee.SHADOW_NATIONS for u in a.units)}
    before = _census()
    saved = cb.FINAL_HITS_HOOK
    try:
        cb.FINAL_HITS_HOOK = (lambda *a, **k: ([], []))
        ee._effect_shadows_gather(st, None, {})
        assert _census() == before, \
            "extraction defer: no auto-join move is made (no drain)"
        cb.FINAL_HITS_HOOK = None
        ee._effect_shadows_gather(st, None, {})
        assert _census() != before, \
            "self-play still auto-picks a legal join"
    finally:
        cb.FINAL_HITS_HOOK = saved


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_clauses(only=None):
    """Execute registered clauses (all, or only those whose cid is in `only`).
    Never raises: each failure becomes a result row.  Returns a list of
    (cid, chapter, status, desc, msg, page)."""
    random.seed(0)
    results = []
    for cid, chapter, desc, fn, page in REGISTRY:
        if only is not None and cid not in only:
            continue
        try:
            fn()
            results.append((cid, chapter, 'PASS', desc, '', page))
        except Skip as e:
            results.append((cid, chapter, 'SKIP', desc, str(e), page))
        except AssertionError as e:
            results.append((cid, chapter, 'FAIL', desc, str(e) or 'assertion failed', page))
        except Exception as e:
            results.append((cid, chapter, 'ERROR', desc, f"{type(e).__name__}: {e}", page))
    return results


def _tally(results):
    counts = {'PASS': 0, 'FAIL': 0, 'SKIP': 0, 'ERROR': 0}
    for row in results:
        counts[row[2]] += 1
    return counts


def _print_full(results):
    counts = _tally(results)
    print("=" * 78)
    print("WOTR RULE-CLAUSE TRACEABILITY MATRIX  (agent P2)")
    print("=" * 78)
    cur_ch = None
    for cid, chapter, status, desc, msg, page in results:
        if chapter != cur_ch:
            print("\n--- chapter %s ---" % chapter)
            cur_ch = chapter
        line = "  [%-5s] %-8s %s" % (status, cid, desc)
        print(line[:100])
        if status in ('FAIL', 'ERROR') and msg:
            print("           -> %s" % msg[:88])
    print("\n" + "=" * 78)
    print("TOTAL %d  |  PASS %d  FAIL %d  SKIP %d  ERROR %d"
          % (len(results), counts['PASS'], counts['FAIL'],
             counts['SKIP'], counts['ERROR']))
    if counts['FAIL'] or counts['ERROR']:
        print("\nFAIL / ERROR clauses (data for flag review):")
        for cid, chapter, status, desc, msg, page in results:
            if status in ('FAIL', 'ERROR'):
                print("  %-8s %-6s %s" % (cid, status, desc))
    print("=" * 78)


def _page_map():
    """Return (pages, house, unknown): pages = {int page -> sorted [cids]};
    house / unknown = sorted cid lists for 'house' and None anchors."""
    from collections import defaultdict
    pages = defaultdict(list)
    house, unknown = [], []
    for cid, chapter, desc, fn, page in REGISTRY:
        if page is None:
            unknown.append(cid)
        elif page == 'house':
            house.append(cid)
        elif isinstance(page, tuple):
            for p in range(page[0], page[1] + 1):
                pages[p].append(cid)
        else:
            pages[page].append(cid)
    return ({p: sorted(set(c)) for p, c in pages.items()},
            sorted(house), sorted(unknown))


def _print_page_map():
    pages, house, unknown = _page_map()
    print("=" * 78)
    print("WOTR CLAUSE -> RULEBOOK PAGE MAP")
    print("=" * 78)
    for p in sorted(pages):
        print("  p.%-3d  %s" % (p, ", ".join(pages[p])))
    print("\n  house rulings (no rulebook page; CURRENT_RULES / amendments):")
    print("    %s" % (", ".join(house) or "(none)"))
    print("\n  rulebook rule, exact page not confidently determined:")
    print("    %s" % (", ".join(unknown) or "(none)"))
    print("=" * 78)
    print("pages anchored: %d   |   house: %d   |   page-unknown: %d"
          % (len(pages), len(house), len(unknown)))
    print("=" * 78)
    return 0


def _run_page(n):
    only = [cid for (cid, ch, desc, fn, page) in REGISTRY if _page_covers(page, n)]
    print("=" * 78)
    print("WOTR RULE-CLAUSE TRACEABILITY  --  rulebook page %d" % n)
    print("=" * 78)
    if not only:
        print("\n  no clause is anchored to rulebook page %d  ->  UNVERIFIED" % n)
        print("\n" + "=" * 78)
        print("page %d:  0 clauses anchored  |  PASS 0  FAIL 0  SKIP 0  ERROR 0" % n)
        print("=" * 78)
        return 0
    results = _run_clauses(only=set(only))
    counts = _tally(results)
    for cid, chapter, status, desc, msg, page in results:
        line = "  [%-5s] %-8s %s" % (status, cid, desc)
        print(line[:100])
        if status in ('FAIL', 'ERROR') and msg:
            print("           -> %s" % msg[:88])
    print("\n" + "=" * 78)
    print("page %d:  %d clause(s) anchored  |  PASS %d  FAIL %d  SKIP %d  ERROR %d"
          % (n, len(results), counts['PASS'], counts['FAIL'],
             counts['SKIP'], counts['ERROR']))
    print("=" * 78)
    return 1 if (counts['FAIL'] or counts['ERROR']) else 0


@clause("CL-285", "6-cards", "1.216 item 2 (emission spec): The Shadow Lengthens CHAINED/SWAP pair (which bypasses _sg_join_move) captures the empty unconquered FP settlement it walks THROUGH - a swap A<->B routing through an FP stronghold captures that settlement (pass-through capture), matching the single-move path", page='card')
def cl_285():
    import event_effects as ee
    st = setup_game()
    A, MID, B = 'ered luin', 'grey havens', 'forlindon'
    B2 = board.Board()

    def adj(r):
        return [sanitize(x) for x in B2.adjacent(r)]

    # MID must be an empty, unconquered FP settlement between A and B
    assert MID not in st.enemy_controlled[Side.SHADOW], \
        "precondition: MID is unconquered"
    for r in (A, MID, B):
        place(st, r, Nation.SAURON, clear=True)
    # two Shadow armies (the join/swap endpoints)
    place(st, A, Nation.SAURON, regulars=2, clear=True)
    place(st, B, Nation.SAURON, regulars=2, clear=True)
    # block every non-MID neighbour of A and B so the ONLY free 2-step
    # path each way runs through the FP settlement MID
    for src in (A, B):
        for nb in adj(src):
            if nb == MID:
                continue
            a = army_at(st, nb)
            if not a.units:
                a.units.append(gs.Unit(Nation.GONDOR, Kind.REGULAR))
    assert ee._free_path_mids(st, A, B, 2) == [MID], \
        "precondition: A->B walk-through is exactly [MID]"
    assert ee._free_path_mids(st, B, A, 2) == [MID], \
        "precondition: B->A walk-through is exactly [MID]"
    # the CHAINED/SWAP pair (from2 == to1) exercises the branch that
    # bypasses _sg_join_move
    ee._effect_shadow_lengthens(st, None, {
        'from_region': A, 'to_region': B,
        'from_region_2': B, 'to_region_2': A})
    assert MID in st.enemy_controlled[Side.SHADOW], \
        "the walked-through FP settlement is captured (pass-through, 1.216)"
    # both endpoints still hold their (swapped) Shadow armies
    assert len(army_at(st, A).units) == 2 and len(army_at(st, B).units) == 2, \
        "the swap conserves both armies"


@clause("CL-286", "6-cards", "1.216 item 1 (emission spec): Return of the Witch-king has NO decision - it teleports the WK to Angmar and recruits 2 Sauron Regulars + 1 Elite THERE (never a muster-region choice); card_options offers no spec for it", page='card')
def cl_286():
    import event_effects as ee
    from card_options import get_card_options
    st = setup_game()
    ang = sanitize('Angmar')
    # find & relocate the Witch-king somewhere other than Angmar
    import re as _re
    wk = None
    for rk, a in list(st.armies.items()):
        for l in list(a.leaders):
            if 'witchking' in _re.sub(r'[^a-z]', '',
                                      getattr(l, 'name', '').lower()):
                wk = l
                a.leaders.remove(l)
                break
        if wk is not None:
            break
    if wk is None:
        wk = gs.Nazgul(Nation.SAURON)
        wk.name = 'The Witch-king'
        wk.is_witch_king = True
    src = 'morannon'
    army_at(st, src).leaders.append(wk)
    # no optioned decision surface for this card
    opts = get_card_options(st, Side.SHADOW, 'Return of the Witch-king')
    assert not opts, "Return of the Witch-king offers NO card option (no region choice)"

    def _sauron(region, kind):
        a = army_at(st, region)
        return sum(1 for u in a.units
                   if u.nation == Nation.SAURON and u.kind == kind)

    r0 = _sauron(ang, Kind.REGULAR)
    e0 = _sauron(ang, Kind.ELITE)
    ee._effect_return_of_witch_king(st, None, {})
    # WK teleported to Angmar (not left at morannon, not a chosen region)
    assert any('witchking' in _re.sub(r'[^a-z]', '',
                                      getattr(l, 'name', '').lower())
               for l in army_at(st, ang).leaders), \
        "the Witch-king is moved to Angmar"
    assert _sauron(ang, Kind.REGULAR) - r0 == 2, \
        "exactly 2 Sauron Regulars recruited in Angmar"
    assert _sauron(ang, Kind.ELITE) - e0 == 1, \
        "exactly 1 Sauron Elite recruited in Angmar"


@clause("CL-287", "6-cards", "Wizard's Staff (1.217): PRE-draw nullify prevents Isildur's Bane AND Foul Thing tile draws when chosen; Isildur's Bane damage is NOT reducible by Horn/Axe (no reduction offered); Foul Thing IS reducible (Axe/Horn interpose) and preventable by the Staff", page='card')
def cl_287():
    # USER RULING 2026-07-23 (amendment 1.217): "wizard's staff can be
    # used against isildur's bane." The staff (card_list.csv:9) PREVENTS
    # the tile DRAW itself (pre-draw nullify) - distinct from post-draw
    # damage reduction. FP still cannot REDUCE Isildur's Bane damage once
    # a tile is drawn (no Horn/Axe there); Foul Thing IS reducible by
    # Horn/Axe AND preventable by the Staff. Learned path only.
    from game_state import PersistentCardType as PCT
    import event_effects as ee
    import combat as cb_m
    _sv_fh = cb_m.FINAL_HITS_HOOK
    cb_m.FINAL_HITS_HOOK = None

    def _staff():
        return gs.PersistentCard(
            card_id='FP_staff', card_name="Wizard's Staff",
            side=Side.FREE_PEOPLES, card_type=PCT.WIZARD_STAFF)

    def _horn():
        return gs.PersistentCard(
            card_id='FP_horn', card_name='Horn of Gondor',
            side=Side.FREE_PEOPLES, card_type=PCT.HORN_OF_GONDOR)

    def _resolve(st, val):
        ai._execute_action(st, ai.Action(
            die_idx=-1, action_type='card_option',
            params={'choices': {'wizardstaff_prevent': val}}))

    try:
        # (A) Staff PREVENTS Isildur's Bane's draw when chosen: the staff
        # is discarded, NO tile is drawn, NO corruption is taken.
        st = setup_game()
        st.pending_card_option = None
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(_staff())
        st.fellowship.corruption = 0
        st.hunt_pool = [gs.HuntTile(damage=3)]
        ee._effect_isildurs_bane(st, None, {})
        pco = st.pending_card_option
        assert pco is not None and pco.get('card_name') == "Wizard's Staff", \
            "Isildur's Bane offers the Wizard's Staff prevent decision"
        assert {'wizardstaff_prevent': True} in pco['options'] and \
            {'wizardstaff_prevent': False} in pco['options'], \
            "the 2-option prevent/allow pair is offered"
        _resolve(st, True)
        assert st.persistent_cards.get_by_type(PCT.WIZARD_STAFF) is None, \
            "prevent discards the Wizard's Staff"
        assert st.fellowship.corruption == 0, \
            "prevent nullifies the draw -> no corruption"
        assert len(st.hunt_pool) == 1, "prevent -> no tile drawn"
        assert st.pending_card_option is None

        # (B) Isildur's Bane damage is NOT reducible by Horn/Axe: declining
        # the staff draws the tile and applies FULL damage as corruption
        # with NO Axe/Horn reduction offered (Horn stays on the table).
        st = setup_game()
        st.pending_card_option = None
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(_staff())
        st.persistent_cards.add_card(_horn())
        st.fellowship.corruption = 0
        st.hunt_pool = [gs.HuntTile(damage=3)]
        ee._effect_isildurs_bane(st, None, {})
        _resolve(st, False)
        assert st.persistent_cards.get_by_type(PCT.WIZARD_STAFF) is not None, \
            "declining the staff keeps it on the table"
        assert st.pending_card_option is None, \
            "Isildur's Bane opens NO Axe/Horn reduction (not reducible)"
        assert st.fellowship.corruption == 3, \
            "FULL tile damage as corruption, unreduced by Horn"
        assert st.persistent_cards.get_by_type(PCT.HORN_OF_GONDOR) is not None, \
            "Horn is NOT consumed (does not apply to Isildur's Bane)"

        # (C) Staff PREVENTS Foul Thing's draw when chosen: staff
        # discarded, no tile, no casualty.
        st = setup_game()
        st.pending_card_option = None
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(_staff())
        bor = gs.Companion(name='Boromir', level=2, leadership=1)
        st.fellowship.companions = [bor]
        st.fellowship.guide = bor
        st.fellowship.hidden = True
        st.fellowship.mordor_entered = False
        st.hunt_pool = [gs.HuntTile(damage=3)]
        ee._effect_foul_thing(st, None, {})
        pco = st.pending_card_option
        assert pco is not None and pco.get('card_name') == "Wizard's Staff", \
            "Foul Thing ALSO offers the Wizard's Staff prevent decision"
        _resolve(st, True)
        assert st.persistent_cards.get_by_type(PCT.WIZARD_STAFF) is None, \
            "prevent discards the Wizard's Staff"
        assert any(c.name == 'Boromir' for c in st.fellowship.companions), \
            "prevent nullifies the draw -> no random casualty"
        assert len(st.hunt_pool) == 1, "prevent -> no tile drawn"

        # (D) Foul Thing IS reducible by Horn/Axe: declining the staff
        # draws the tile and opens the Axe/Horn -1 reduction interpose
        # BEFORE the forced random casualty.
        st = setup_game()
        st.pending_card_option = None
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(_staff())
        st.persistent_cards.add_card(_horn())
        bor = gs.Companion(name='Boromir', level=2, leadership=1)
        st.fellowship.companions = [bor]
        st.fellowship.guide = bor
        st.fellowship.hidden = True
        st.fellowship.mordor_entered = False
        st.hunt_pool = [gs.HuntTile(damage=3)]
        ee._effect_foul_thing(st, None, {})
        _resolve(st, False)
        pco2 = st.pending_card_option
        assert pco2 is not None and pco2.get('hunt_reduction'), \
            "Foul Thing opens the Axe/Horn reduction after the draw (reducible)"
        assert any(c.name == 'Boromir' for c in st.fellowship.companions), \
            "the random casualty is not applied until the reduction resolves"

        # (E) TEACHER path (combat.FINAL_HITS_HOOK set) is UNAFFECTED: the
        # staff interpose is learned-path-only, so the tile draws normally.
        cb_m.FINAL_HITS_HOOK = (lambda *a, **k: None)
        st = setup_game()
        st.pending_card_option = None
        st.persistent_cards = gs.PersistentCardState()
        st.persistent_cards.add_card(_staff())
        st.fellowship.corruption = 0
        st.hunt_pool = [gs.HuntTile(damage=2)]
        ee._effect_isildurs_bane(st, None, {})
        assert st.pending_card_option is None, \
            "teacher path opens NO staff interpose (replays logged use)"
        assert st.fellowship.corruption == 2, \
            "teacher path draws the tile normally"
    finally:
        cb_m.FINAL_HITS_HOOK = _sv_fh


# ===========================================================================
# RULES-ASSURANCE BLIND-SPOT SWEEP (2026-07-27)
# Added coverage for audited gaps: non-belligerent restrictions (printed
# p.35), move-through Settlement capture + stacking (p.27-28), declare/reveal
# base mechanic (p.38), the Winning-the-Game chapter (p.44), and initial
# setup composition (p.14-17).  Rulebook line refs are into data/_rulebook.txt.
# ===========================================================================

# --- Non-belligerent restrictions (rulebook printed p.35, "ENTERING WAR") ---

@clause("CL-300", "5-politics",
        "Non-belligerent army may move outside its own borders (neutral) and within its own Nation, but NEVER across another Nation's border (neg)")
def cl_300():
    # rulebook printed p.35 (_rulebook.txt L3093-3099): a non-belligerent
    # Nation's Army units "can move outside their National borders but can
    # never move across the borders of other Nations (including friendly ones)."
    st = setup_game()
    # Isengard starts at box 2 = Active but NOT At War (is_at_war needs 3)
    assert not st.political.is_at_war(Nation.ISENGARD)
    place(st, 'gap of rohan', Nation.ISENGARD, regulars=2, clear=True)
    # empty the adjacent Rohan settlement so geography (not occupancy) is the
    # only thing under test
    for _k, _a in list(st.armies.items()):
        if sanitize(_k) == 'fords of isen':
            _a.units.clear(); _a.leaders.clear()
    set_dice(st, Side.SHADOW, [Face.ARMY])
    dests = {sanitize(a.params['to'])
             for a in ai.get_legal_actions(st, Side.SHADOW)
             if a.action_type == 'army_move'
             and sanitize(a.params['from']) == 'gap of rohan'}
    # fords of isen belongs to Rohan (another Nation) -> blocked
    assert 'fords of isen' not in dests, \
        ("non-belligerent Isengard crossed into Rohan (neg)", sorted(dests))
    # enedwaith is neutral (no Nation owner) -> outside-own-borders move OK;
    # orthanc / south dunland are Isengard (own Nation) -> OK
    assert 'enedwaith' in dests, ("must move into a neutral region", sorted(dests))
    assert {'orthanc', 'south dunland'} & dests, \
        ("must move within its own Nation", sorted(dests))


@clause("CL-301", "5-politics",
        "Retreat/advance border exception: a non-belligerent army MAY cross another Nation's border when NOT making a voluntary move")
def cl_301():
    # rulebook printed p.35 (_rulebook.txt L3110-3113): "Non-belligerent
    # figures are allowed to cross another Nation's border when retreating
    # from a battle."  The engine models the restriction as voluntary-move-
    # only: the army-move menu filters passive units at the border while
    # _get_retreat_actions applies no political gate.
    st = setup_game()
    assert not st.political.is_at_war(Nation.ISENGARD)
    place(st, 'gap of rohan', Nation.ISENGARD, regulars=2, clear=True)
    for _k, _a in list(st.armies.items()):
        if sanitize(_k) == 'fords of isen':
            _a.units.clear(); _a.leaders.clear()
    set_dice(st, Side.SHADOW, [Face.ARMY])
    vol = {sanitize(a.params['to'])
           for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_move'
           and sanitize(a.params['from']) == 'gap of rohan'}
    assert 'fords of isen' not in vol, "voluntary crossing must be refused"
    # hook proxy retired 2026-07-27: test the REAL retreat mechanism —
    # _get_retreat_actions carries no political border gate (1.222(b):
    # retreats cross freely)
    pc = {'defender_region': 'gap of rohan', 'defender_side': Side.SHADOW,
          'attacker_region': 'orthanc', 'attacker_side': Side.FREE_PEOPLES}
    ret = {sanitize(a.params['to_region'])
           for a in ai._get_retreat_actions(st, pc, defending=True)}
    assert 'fords of isen' in ret,         "retreat may cross another Nation's border (rulebook exception)"


@clause("CL-302", "5-politics",
        "Non-belligerent Nation cannot attack (neg); once At War it can (rulebook p.35)")
def cl_302():
    # rulebook printed p.35 (_rulebook.txt L3100-3103): non-belligerent units
    # "cannot attack enemy Armies (but can defend if attacked)."
    st = setup_game()
    st.political.boxes[Nation.GONDOR] = 3
    place(st, 'osgiliath', Nation.SAURON, regulars=3, clear=True)
    st.political.boxes[Nation.SAURON] = 2       # Sauron non-belligerent
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'osgiliath']
    assert not atk, "non-belligerent Sauron must NOT attack (neg)"
    st.political.boxes[Nation.SAURON] = 3        # now At War
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'osgiliath']
    assert atk, "At-War Sauron may attack Minas Tirith"


@clause("CL-303", "5-politics",
        "Non-belligerent Nation cannot be recruited via a Muster die (neg); once At War it can (rulebook p.35)")
def cl_303():
    # rulebook printed p.35 (_rulebook.txt L3104-3107): non-belligerent units
    # "can never be recruited by using Muster Action die results."
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 2        # non-belligerent
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    recs = [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'muster_recruit'
            and ai._region_nation(sanitize(a.params['region'])) == Nation.SAURON]
    assert not recs, "non-belligerent Sauron must not be muster-recruited (neg)"
    st.political.boxes[Nation.SAURON] = 3        # At War
    set_dice(st, Side.SHADOW, [Face.MUSTER])
    recs2 = [a for a in ai.get_legal_actions(st, Side.SHADOW)
             if a.action_type == 'muster_recruit'
             and ai._region_nation(sanitize(a.params['region'])) == Nation.SAURON]
    assert recs2, "At-War Sauron may be muster-recruited"


@clause("CL-304", "5-politics",
        "Characters are always At War: a passive-Nation Nazgul does NOT block an all-At-War army's attack, while a passive-Nation Army unit does (neg)")
def cl_304():
    # rulebook printed p.35 (_rulebook.txt L3127-3133, CHARACTERS AT WAR):
    # "Companions, Minions, and Nazgul can freely move and be involved in
    # battles regardless of the political position of the Nation they belong
    # to ... The Nazgul can participate in an attack even if Sauron is not yet
    # At War."  Attack eligibility keys on unit Nations only; Nazgul/leaders
    # never block.
    st = setup_game()
    st.political.boxes[Nation.ISENGARD] = 3      # attacker units At War
    st.political.boxes[Nation.SAURON] = 2        # the Nazgul's Nation is passive
    st.political.boxes[Nation.ROHAN] = 3
    place(st, 'gap of rohan', Nation.ISENGARD, regulars=2, clear=True)
    army_at(st, 'gap of rohan').leaders.append(gs.Nazgul(Nation.SAURON))
    place(st, 'fords of isen', Nation.ROHAN, regulars=2, clear=True)
    set_dice(st, Side.SHADOW, [Face.ARMY])
    atk = [a for a in ai.get_legal_actions(st, Side.SHADOW)
           if a.action_type == 'army_attack'
           and sanitize(a.params['from']) == 'gap of rohan']
    assert atk, "a passive-Sauron Nazgul (a Character) must not block the attack"
    # contrast: a passive-Sauron ARMY UNIT in the stack DOES block (units,
    # unlike Characters, are bound by their Nation's political status)
    st2 = setup_game()
    st2.political.boxes[Nation.ISENGARD] = 3
    st2.political.boxes[Nation.SAURON] = 2
    st2.political.boxes[Nation.ROHAN] = 3
    place(st2, 'gap of rohan', Nation.ISENGARD, regulars=2, clear=True)
    army_at(st2, 'gap of rohan').units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    place(st2, 'fords of isen', Nation.ROHAN, regulars=2, clear=True)
    set_dice(st2, Side.SHADOW, [Face.ARMY])
    atk2 = [a for a in ai.get_legal_actions(st2, Side.SHADOW)
            if a.action_type == 'army_attack'
            and sanitize(a.params['from']) == 'gap of rohan']
    assert not atk2, "a passive-Nation Army unit DOES block the attack (neg)"


# --- Move-through Settlement capture + stacking (rulebook printed p.27-28) --

@clause("CL-310", "4-battle",
        "Army moving into an empty enemy Settlement captures it on entry; a defended Settlement is an attack, not a silent capture (neg)")
def cl_310():
    # rulebook printed p.28 (_rulebook.txt L2508-2509): "If an Army moves
    # through an enemy-controlled Settlement, it captures that Settlement."
    # POSITIVE: walk a Shadow army into empty Pelargir (a Gondor City).
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    for _k, _a in list(st.armies.items()):
        if sanitize(_k) == 'pelargir':
            _a.units.clear(); _a.leaders.clear()
    place(st, 'osgiliath', Nation.SAURON, regulars=2, clear=True)
    assert 'pelargir' not in st.enemy_controlled[Side.SHADOW]
    set_dice(st, Side.SHADOW, [Face.ARMY])
    mv = [a for a in ai.get_legal_actions(st, Side.SHADOW)
          if a.action_type == 'army_move'
          and sanitize(a.params['from']) == 'osgiliath'
          and sanitize(a.params['to']) == 'pelargir']
    assert mv, "a move into the empty enemy City must be offered"
    full = next((a for a in mv if a.params.get('n_regular') in (None, 2)), mv[0])
    ai._execute_action(st, full)
    assert 'pelargir' in st.enemy_controlled[Side.SHADOW], \
        "walking into an empty enemy Settlement captures it (rulebook p.28)"
    # NEGATIVE: a Settlement still holding an enemy defender is an ATTACK
    # target, never a move (so it is never silently captured by a move).
    st2 = setup_game()
    st2.political.boxes[Nation.SAURON] = 3       # Pelargir keeps its Gondor defender
    place(st2, 'osgiliath', Nation.SAURON, regulars=2, clear=True)
    set_dice(st2, Side.SHADOW, [Face.ARMY])
    la = ai.get_legal_actions(st2, Side.SHADOW)
    mv2 = [a for a in la if a.action_type == 'army_move'
           and sanitize(a.params['from']) == 'osgiliath'
           and sanitize(a.params['to']) == 'pelargir']
    atk2 = [a for a in la if a.action_type == 'army_attack'
            and sanitize(a.params['from']) == 'osgiliath'
            and sanitize(a.params['to']) == 'pelargir']
    assert not mv2, "moving into a defended Settlement must NOT be offered (neg)"
    assert atk2, "a defended Settlement is an attack target"
    assert 'pelargir' not in st2.enemy_controlled[Side.SHADOW], \
        "no silent capture of a defended Settlement (neg)"


@clause("CL-311", "4-battle",
        "10-unit stacking limit: a move exceeding 10 own units is rejected; exactly 10 is allowed; the Army-die second leg is independently capped")
def cl_311():
    # rulebook printed p.27 (_rulebook.txt L2476-2478): "After moving an Army
    # into a region, you can not exceed the stacking limit of 10 units."  p.28
    # (L2510-2514): stacking is checked once each movement has completed (the
    # engine checks each move's destination; the standard 2-move die ends
    # each move in a region, so per-move == "after all movements").
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    place(st, 'moria', Nation.SAURON, regulars=10, clear=True)
    place(st, 'dimrill dale', Nation.SAURON, regulars=1, clear=True)
    raised = False
    try:
        game_logic.move_army(st, 'moria', 'dimrill dale', side=Side.SHADOW)
    except RuntimeError as e:
        raised = 'stacking' in str(e).lower()
    assert raised, "moving 10 into a region already holding 1 must be rejected (11 > 10)"
    # exactly 10 is allowed
    st2 = setup_game()
    st2.political.boxes[Nation.SAURON] = 3
    place(st2, 'moria', Nation.SAURON, regulars=9, clear=True)
    place(st2, 'dimrill dale', Nation.SAURON, regulars=1, clear=True)
    game_logic.move_army(st2, 'moria', 'dimrill dale', side=Side.SHADOW)
    dd = next(a for _k, a in st2.armies.items() if sanitize(_k) == 'dimrill dale')
    assert len(dd.units) == 10, ("exactly 10 must be allowed", len(dd.units))
    # Army-die SECOND leg: after leg 1 fills Moria to 9, the pending second
    # move from a different army into Moria is capped so the destination
    # never exceeds 10 (checked at ITS destination, not deferred past the die).
    st3 = setup_game()
    st3.political.boxes[Nation.SAURON] = 3
    place(st3, 'moria', Nation.SAURON, regulars=6, clear=True)
    place(st3, 'dimrill dale', Nation.SAURON, regulars=3, clear=True)
    place(st3, 'hollin', Nation.SAURON, regulars=3, clear=True)
    set_dice(st3, Side.SHADOW, [Face.ARMY])
    leg1 = [a for a in ai.get_legal_actions(st3, Side.SHADOW)
            if a.action_type == 'army_move'
            and sanitize(a.params['from']) == 'dimrill dale'
            and sanitize(a.params['to']) == 'moria']
    full1 = next((a for a in leg1 if a.params.get('n_regular') in (None, 3)), leg1[0])
    ai._execute_action(st3, full1)          # Moria now 9, pending set
    assert st3.pending_army_move_die is not None, "first Army-die move sets pending"
    moria3 = next(a for _k, a in st3.armies.items() if sanitize(_k) == 'moria')
    assert len(moria3.units) == 9, ("leg 1 brought Moria to 9", len(moria3.units))
    leg2 = [a for a in ai.get_legal_actions(st3, Side.SHADOW)
            if a.action_type == 'army_move'
            and sanitize(a.params['from']) == 'hollin'
            and sanitize(a.params['to']) == 'moria']
    assert leg2, "a second-leg move into Moria must still be offered"
    for a in leg2:
        nr = a.params.get('n_regular')
        nr = 3 if nr is None else nr
        assert 9 + nr <= 10, \
            ("the second leg must not push Moria over 10", nr)


# --- Declare / reveal base mechanic (rulebook printed p.38) ------------------

@clause("CL-320", "2-fellow",
        "Declaration requires a HIDDEN Fellowship (neg revealed); it places the Ring-bearers up to 'progress' regions away and resets progress to 0 (stays Hidden)")
def cl_320():
    # rulebook printed p.38 (_rulebook.txt L3362-3380): declaration is only
    # for a Hidden Fellowship; the pawn may move up to Progress regions and
    # the counter then resets to 0, remaining Hidden side up.
    st = setup_game()
    st.fellowship.progress = 2
    st.fellowship.hidden = True
    assert ai._get_declare_actions(st, allow_in_place=True), "hidden may declare"
    st.fellowship.hidden = False
    assert not ai._get_declare_actions(st, allow_in_place=True), \
        "revealed Fellowship must NOT declare (neg)"
    # place up to Progress and reset
    st2 = setup_game()
    st2.fellowship.region = st2.board.region('rivendell')
    st2.fellowship.progress = 2
    st2.fellowship.hidden = True
    from collections import deque
    cur = sanitize('rivendell'); tgt = None
    seen = {cur}; q = deque([(cur, 0)])
    while q:
        r, d = q.popleft()
        if d == 2:
            tgt = r; break
        for adj in st2.board.adjacent(r):
            a = sanitize(adj)
            if a not in seen:
                seen.add(a); q.append((a, d + 1))
    assert tgt is not None, "expected a region exactly 2 away from Rivendell"
    game_logic.declare_fellowship(st2, region=tgt)
    assert sanitize(game_logic.fellowship_region(st2)) == tgt, \
        "declaration places the pawn at the chosen (<=progress) region"
    assert _bfs_dist(st2, 'rivendell', tgt) == 2, "placed exactly 2 (up to progress)"
    assert st2.fellowship.progress == 0, "progress resets to 0 on declaration"
    assert st2.fellowship.hidden is True, "the Fellowship stays Hidden after declaring"


@clause("CL-321", "2-fellow",
        "Declaring in an unconquered FP City/Stronghold heals 1 Corruption; a Shadow-captured settlement heals nothing (neg)")
def cl_321():
    # rulebook printed p.38 (_rulebook.txt L3393-3395): "If the Ring-bearers
    # suffered Corruption, one Corruption point could now be healed as the
    # Fellowship is in a Free Peoples Stronghold" (City or Stronghold, not
    # captured by the Shadow).
    st = setup_game()
    st.fellowship.region = st.board.region('rivendell')
    st.fellowship.progress = 0
    st.fellowship.hidden = True
    st.fellowship.corruption = 5
    game_logic.declare_fellowship(st, region='Rivendell')
    assert st.fellowship.corruption == 4, \
        ("declaring in an unconquered FP Stronghold heals 1", st.fellowship.corruption)
    # Shadow-captured FP settlement: no heal
    st2 = setup_game()
    st2.fellowship.region = st2.board.region('rivendell')
    st2.fellowship.progress = 0
    st2.fellowship.hidden = True
    st2.fellowship.corruption = 5
    st2.enemy_controlled[Side.SHADOW].add(sanitize('rivendell'))
    game_logic.declare_fellowship(st2, region='Rivendell')
    assert st2.fellowship.corruption == 5, \
        "no healing in a Shadow-captured FP settlement (neg)"


@clause("CL-322", "9-reveal",
        "Hunt-reveal placement may NEVER end in an unconquered FP City/Stronghold (rulebook p.38); progress resets")
def cl_322():
    # rulebook printed p.38 (_rulebook.txt L3404-3407): a hunt-reveal
    # "movement can never end in a region containing a Free Peoples
    # Stronghold or City controlled by the Free Peoples."
    def _is_unconq_fp_settlement(st, r):
        meta = board.regions.get(sanitize(r), {})
        return ((meta.get('is_city') or meta.get('is_stronghold'))
                and game_logic.is_fp_region(sanitize(r))
                and sanitize(r) not in st.enemy_controlled.get(Side.SHADOW, set()))
    for prog in (1, 2, 3):
        st = setup_game()
        st.fellowship.region = st.board.region('forlindon')
        st.fellowship.progress = prog
        st.fellowship.hidden = False
        game_logic._auto_reveal_place(st)
        end = game_logic.fellowship_region(st)
        assert not _is_unconq_fp_settlement(st, end), \
            ("reveal ended in an unconquered FP settlement", prog, end)
        assert st.fellowship.progress == 0, "progress resets on reveal"


@clause("CL-323", "9-reveal",
        "ENGINE declaration draws NO Shadow-stronghold Hunt tile (2026-07-27 ruling reverting 'fix 101'); a hunt-reveal in a Shadow stronghold DOES draw one")
def cl_323():
    # rulebook printed p.38 (_rulebook.txt L3431-3446): the Shadow-stronghold
    # tile draw happens "only if the Fellowship is REVEALED by the Shadow
    # player" - it is reveal-only, never on a declaration.  USER RULING
    # 2026-07-27 (game_logic.declare_fellowship L1021-1046): the engine path
    # (no teacher FINAL_HITS_HOOK) draws NO tile on a declaration through/into
    # a Shadow stronghold.  If this ever FAILS it means the reversal regressed.
    import combat as _cb
    assert _cb.FINAL_HITS_HOOK is None, "engine path requires no teacher hook"
    moria = sanitize('moria')
    assert (board.regions.get(moria, {}).get('is_stronghold')
            and game_logic.is_shadow_region(moria)), "Moria is a Shadow stronghold"
    st = setup_game()
    st.fellowship.region = st.board.region('dimrill dale')   # a Moria neighbour
    st.fellowship.progress = 1
    st.fellowship.hidden = True
    before = len(st.hunt_pool)
    game_logic.declare_fellowship(st, region='moria')
    assert sanitize(game_logic.fellowship_region(st)) == moria, "declared into Moria"
    assert len(st.hunt_pool) == before, \
        ("a declaration must draw NO Shadow-stronghold tile (2026-07-27)",
         before, len(st.hunt_pool))
    # contrast: a hunt-reveal STARTING in a Shadow stronghold DOES draw a tile
    st2 = setup_game()
    st2.fellowship.region = st2.board.region('moria')
    st2.fellowship.progress = 0
    st2.fellowship.hidden = False
    before2 = len(st2.hunt_pool)
    game_logic._auto_reveal_place(st2)
    assert len(st2.hunt_pool) == before2 - 1, \
        ("a hunt-reveal in a Shadow stronghold draws one tile", before2, len(st2.hunt_pool))


# --- Winning the Game chapter (rulebook printed p.44) -----------------------

@clause("CL-330", "8-victory",
        "Victory precedence: a lower-numbered condition wins ties (Ring corruption > FP military; Shadow military > FP military; Ring destruction > Shadow military)")
def cl_330():
    # rulebook printed p.44 (_rulebook.txt L3967-3969): "lower-numbered
    # Victory conditions take precedence over higher-numbered Victory
    # conditions, if two or more are achieved on the same turn."  Order:
    # (1) corruption 12, (2) Ring destroyed, (3) Shadow military, (4) FP military.
    # (1) beats (4)
    st = setup_game()
    st.fellowship.corruption = 12
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon', 'minas morgul'}   # 4 VP
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'shadow' and vc['victory_type'] == 'ring_corruption', vc
    # (3) beats (4)
    st2 = setup_game()
    st2.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth',
                                         'helm s deep', 'lorien', 'erebor'}   # 10 VP
    st2.enemy_controlled[Side.FREE_PEOPLES] = {'morannon', 'minas morgul'}    # 4 VP
    vc2 = game_logic.check_all_victory_conditions(st2)
    assert vc2['victory'] == 'shadow' and vc2['victory_type'] == 'military', vc2
    # (2) beats (3)
    st3 = setup_game()
    st3.fellowship.mordor_entered = True
    st3.fellowship.mordor_step = 5
    st3.fellowship.corruption = 4
    st3.pending_hunt = None
    st3.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth',
                                         'helm s deep', 'lorien', 'erebor'}   # 10 VP
    vc3 = game_logic.check_all_victory_conditions(st3)
    assert vc3['victory'] == 'free_peoples' and vc3['victory_type'] == 'ring_destroyed', vc3


@clause("CL-331", "8-victory",
        "Shadow Military Victory needs 10 VP and fires ONLY at end-of-turn (neg mid-turn)")
def cl_331():
    # rulebook printed p.44 (_rulebook.txt L4003-4007 + timing L3961-3964):
    # the Shadow wins if it controls FP Settlements worth 10+ VP, checked only
    # in phase 6 (the Victory Check), i.e. at end-of-turn.
    st = setup_game()
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth',
                                        'helm s deep', 'lorien', 'erebor'}   # 5 SHs = 10 VP
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'shadow' and vc['victory_type'] == 'military', vc
    assert vc['shadow_vp'] == 10, vc
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "military VP must NOT end the game mid-turn (neg)"
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "military victory fires at end-of-turn"
    st.enemy_controlled[Side.SHADOW] = {'minas tirith', 'dol amroth'}        # 4 VP
    assert game_logic.check_all_victory_conditions(st)['victory'] is None, \
        "under 10 VP is not a Shadow military win (neg)"


@clause("CL-332", "8-victory",
        "FP Military Victory needs 4 VP and fires ONLY at end-of-turn (neg mid-turn)")
def cl_332():
    # rulebook printed p.44 (_rulebook.txt L4008-4012 + timing L3961-3964).
    st = setup_game()
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon', 'minas morgul'}    # 2 SHs = 4 VP
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'free_peoples' and vc['victory_type'] == 'military', vc
    assert vc['free_peoples_vp'] == 4, vc
    assert not game_logic.should_game_end(vc, end_of_turn=False), \
        "FP military VP must NOT end the game mid-turn (neg)"
    assert game_logic.should_game_end(vc, end_of_turn=True), \
        "FP military victory fires at end-of-turn"
    st.enemy_controlled[Side.FREE_PEOPLES] = {'morannon'}                    # 2 VP
    assert game_logic.check_all_victory_conditions(st)['victory'] is None, \
        "under 4 VP is not an FP military win (neg)"


@clause("CL-333", "8-victory",
        "Ring victories fire IMMEDIATELY (mid-turn): corruption 12 (Shadow) and Ring destruction at Mordor step 5 (FP)")
def cl_333():
    # rulebook printed p.44 (_rulebook.txt L3974-3990): the two Ring-based
    # conditions "end the game immediately without waiting for the Victory
    # Check phase."
    st = setup_game()
    st.fellowship.corruption = 12
    vc = game_logic.check_all_victory_conditions(st)
    assert vc['victory'] == 'shadow' and vc['victory_type'] == 'ring_corruption', vc
    assert game_logic.should_game_end(vc, end_of_turn=False), \
        "corruption-12 Shadow win must fire mid-turn"
    st2 = setup_game()
    st2.fellowship.mordor_entered = True
    st2.fellowship.mordor_step = 5
    st2.fellowship.corruption = 4
    st2.pending_hunt = None
    vc2 = game_logic.check_all_victory_conditions(st2)
    assert vc2['victory'] == 'free_peoples' and vc2['victory_type'] == 'ring_destroyed', vc2
    assert game_logic.should_game_end(vc2, end_of_turn=False), \
        "Ring destruction (Mordor step 5, <12) must fire mid-turn"


# --- Setting Up the Game (rulebook printed p.14-17) -------------------------

@clause("CL-340", "0-setup",
        "Initial Army setup composition matches the rulebook diagram (p.16-17); NOTE Minas Morgul is 5 Regular + 1 Nazgul (NOT 3 Nazgul)")
def cl_340():
    # rulebook printed p.16-17 ARMY SETUP (_rulebook.txt L1236-1331).
    st = setup_game()

    def comp(rkey):
        a = next((x for k, x in st.armies.items()
                  if sanitize(k) == sanitize(rkey)), None)
        assert a is not None, ("no army at", rkey)
        r = sum(1 for u in a.units if u.kind == Kind.REGULAR)
        e = sum(1 for u in a.units if u.kind == Kind.ELITE)
        L = sum(1 for l in a.leaders if type(l).__name__ == 'Leader')
        N = sum(1 for l in a.leaders if type(l).__name__ == 'Nazgul')
        return (r, e, L, N)

    expected = {
        'rivendell':    (0, 2, 1, 0),   # Elves: 2 Elite, 1 Leader
        'erebor':       (1, 2, 1, 0),   # Dwarves: 1 Reg, 2 Elite, 1 Leader
        'minas tirith': (3, 1, 1, 0),   # Gondor: 3 Reg, 1 Elite, 1 Leader
        'orthanc':      (4, 1, 0, 0),   # Isengard: 4 Reg, 1 Elite
        'barad dur':    (4, 1, 0, 1),   # Sauron: 4 Reg, 1 Elite, 1 Nazgul
        'dol guldur':   (5, 1, 0, 1),   # Sauron: 5 Reg, 1 Elite, 1 Nazgul
        'minas morgul': (5, 0, 0, 1),   # Sauron: 5 Reg, 1 Nazgul (NOT 3)
        'morannon':     (5, 0, 0, 1),   # Sauron: 5 Reg, 1 Nazgul
    }
    for rk, exp in expected.items():
        assert comp(rk) == exp, (rk, "got", comp(rk), "expected", exp)


@clause("CL-341", "0-setup",
        "Setup STEP 11: starting Political boxes (Sauron/Isengard 2, S&E/Gondor 1, others 0) and Active flags (Elves + all Shadow nations active)")
def cl_341():
    # rulebook printed p.14 STEP 11 (_rulebook.txt L1194-1207): Rohan/North/
    # Elves/Dwarves top box (0); Gondor + S&E second box (1); Sauron +
    # Isengard third box, immediately above At War (2).  Elves and all Shadow
    # Army nations start Active; the other FP nations start Passive.
    st = setup_game()
    b = st.political.boxes
    assert b[Nation.SAURON] == 2 and b[Nation.ISENGARD] == 2, dict(b)
    assert b[Nation.SOUTHRON_EASTERLINGS] == 1 and b[Nation.GONDOR] == 1, dict(b)
    for n in (Nation.ROHAN, Nation.THE_NORTH, Nation.ELVES, Nation.DWARVES):
        assert b[n] == 0, (n, b[n])
    assert not any(game_logic.is_nation_at_war(st, n) for n in b), \
        "no nation starts At War"
    for n in (Nation.ELVES, Nation.SAURON, Nation.ISENGARD,
              Nation.SOUTHRON_EASTERLINGS):
        assert st.political.active[n], ("must start Active", n)
    for n in (Nation.GONDOR, Nation.ROHAN, Nation.THE_NORTH, Nation.DWARVES):
        assert not st.political.active[n], ("must start Passive", n)


@clause("CL-342", "0-setup",
        "Setup: 3 FP Elven Rings (0 Shadow); 4 FP + 7 Shadow Action dice; Ring-bearers at Rivendell with 7 Companions and Gandalf the Grey as Guide")
def cl_342():
    # rulebook printed p.14 STEPs 2,4,6,10 (_rulebook.txt L1154-1193).
    st = setup_game()
    assert st.fp_elven_rings == 3 and st.sh_elven_rings == 0, \
        (st.fp_elven_rings, st.sh_elven_rings)
    assert len(st.dice[Side.FREE_PEOPLES]) == 4, len(st.dice[Side.FREE_PEOPLES])
    assert len(st.dice[Side.SHADOW]) == 7, len(st.dice[Side.SHADOW])
    assert sanitize(game_logic.fellowship_region(st)) == 'rivendell', \
        game_logic.fellowship_region(st)
    assert len(st.fellowship.companions) == 7, len(st.fellowship.companions)
    assert st.fellowship.guide.name == 'Gandalf the Grey', st.fellowship.guide.name
    names = {c.name for c in st.fellowship.companions}
    for want in ('Gandalf the Grey', 'Strider', 'Legolas', 'Gimli',
                 'Boromir', 'Meriadoc', 'Peregrin'):
        assert want in names, ("missing starting Companion", want, sorted(names))


@clause("CL-343", "3-hunt",
        "Initial Hunt Pool is 16 standard tiles with the exact mix: 3x'3', 2x'2', 2x'1', 4 Eye, 1 reveal-'2', 2 reveal-'1', 2 reveal-'0'")
def cl_343():
    # rulebook printed p.14 STEP 9 (standard beige tiles) + hunt-tile values
    # p.39 (_rulebook.txt L3560-3575).  Composition verified against
    # game_state._initial_hunt_pool; complements CL-050.
    st = setup_game()
    pool = st.hunt_pool
    assert len(pool) == 16, len(pool)
    from collections import Counter
    c = Counter()
    for t in pool:
        if t.is_eye:
            c['eye'] += 1
        elif t.reveal:
            c[('reveal', t.damage)] += 1
        else:
            c[('plain', t.damage)] += 1
    assert c['eye'] == 4, ("4 Eye tiles", dict(c))
    assert c[('plain', 3)] == 3, ("3 plain '3' tiles", dict(c))
    assert c[('plain', 2)] == 2, ("2 plain '2' tiles", dict(c))
    assert c[('plain', 1)] == 2, ("2 plain '1' tiles", dict(c))
    assert c[('reveal', 2)] == 1, ("1 reveal '2' tile", dict(c))
    assert c[('reveal', 1)] == 2, ("2 reveal '1' tiles", dict(c))
    assert c[('reveal', 0)] == 2, ("2 reveal '0' tiles", dict(c))
    assert c[('plain', 0)] == 0, ("no plain '0' tile", dict(c))
    assert sum(v for k, v in c.items()
               if isinstance(k, tuple) and k[0] == 'reveal') == 5, dict(c)


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if '--pages' in argv:
        return _print_page_map()
    if '--page' in argv:
        i = argv.index('--page')
        try:
            n = int(argv[i + 1])
        except (IndexError, ValueError):
            print("usage: python test_clause_matrix.py --page N   (N = rulebook page)")
            return 0
        return _run_page(n)
    results = _run_clauses()
    _print_full(results)
    counts = _tally(results)
    return 1 if (counts['FAIL'] or counts['ERROR']) else 0


if __name__ == '__main__':
    sys.exit(main())
