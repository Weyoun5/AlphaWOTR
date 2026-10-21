#!/usr/bin/env python3
"""CARD CONSERVATION REGRESSION (2026-08-15).

The 96 event cards are PHYSICAL objects: every one is always in exactly
one place -- a deck, a hand, the table (persistent_cards), a face-up
discard pile, or a face-down discard pile.  A card removed from the
table or paid out of a hand is DISCARDED, never deleted.

Found by auditing the auto35 live game's engine state: 94/96 cards
accounted, 2 annihilated (FREE_PEOPLES_Character_Horn_of_Gondor via the
hunt-reduction discard, SHADOW_Character_The_Palantir_of_Orthanc via the
table-card kill action).  Nine sites removed a table card without ever
putting it in a pile.

Run:  python test_card_conservation.py       (exit 0 = all pass)
"""

import os
import sys
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai                                    # noqa: E402
import board                                 # noqa: E402
import event_effects as ee                   # noqa: E402
import game_logic                            # noqa: E402
import game_state as gs                      # noqa: E402
from cards import get_card, get_cards        # noqa: E402

Side = gs.Side
PCT = gs.PersistentCardType

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def setup_game(deal=0):
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for _ in range(deal):
        for s in (Side.FREE_PEOPLES, Side.SHADOW):
            for dk in ('character', 'strategy'):
                game_logic.draw_event_card(st, s, dk)
    return st


def card_census(st):
    """Counter of card_id -> number of places it occupies."""
    seen = Counter()

    def add(items):
        for c in items or []:
            if isinstance(c, str):
                seen[c] += 1

    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        add(st.hands[s].hand)
        add(getattr(st.hands[s], 'discarded_face_down', []))
    for v in (getattr(st, 'discard_face_up', None) or {}).values():
        add(v)
    for a in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck',
              'sh_strat_deck'):
        add(getattr(st, a, []))
    pc = getattr(st, 'persistent_cards', None)
    if pc is not None:
        for attr in ('fp_cards', 'sh_cards'):
            for c in getattr(pc, attr, []) or []:
                seen[getattr(c, 'card_id', '?')] += 1
    return seen


def assert_conserved(st, note=''):
    """Every registry card in exactly one place, no duplicates."""
    seen = card_census(st)
    allc = set(get_cards().keys())
    dups = sorted(k for k, v in seen.items() if v > 1)
    missing = sorted(allc - set(seen))
    unknown = sorted(set(seen) - allc)
    assert not dups, '%s DUPLICATED cards: %s' % (note, dups)
    assert not unknown, '%s UNKNOWN card ids: %s' % (note, unknown)
    assert not missing, '%s ANNIHILATED cards: %s' % (note, missing)


def put_on_table(st, cid, ptype=None):
    """Move a card from wherever it is onto the table (as a played
    persistent card), preserving conservation."""
    c = get_card(cid)
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        h = st.hands[s].hand
        while cid in h:
            h.remove(cid)
        fd = getattr(st.hands[s], 'discarded_face_down', None)
        if fd and cid in fd:
            fd.remove(cid)
    for v in (getattr(st, 'discard_face_up', None) or {}).values():
        while cid in v:
            v.remove(cid)
    for a in ('fp_char_deck', 'fp_strat_deck', 'sh_char_deck',
              'sh_strat_deck'):
        d = getattr(st, a, None)
        if d and cid in d:
            d.remove(cid)
    pcard = gs.PersistentCard(
        card_id=cid, card_name=c.card_name, side=c.side,
        card_type=ptype if ptype is not None else _ptype_for(cid))
    st.persistent_cards.add_card(pcard)
    return pcard


def _ptype_for(cid):
    """Best-effort PersistentCardType for a card id (name match)."""
    nm = get_card(cid).card_name
    for t in PCT:
        if str(t.value).strip().lower() == nm.strip().lower():
            return t
    return None


def on_table(st, cid):
    pc = st.persistent_cards
    return any(c.card_id == cid
               for c in list(pc.fp_cards) + list(pc.sh_cards))


def in_face_up(st, cid):
    return any(cid in v
               for v in (getattr(st, 'discard_face_up', None) or {}).values())


# ---------------------------------------------------------------------------
# 0. The invariant holds at setup (guards the census helper itself)
# ---------------------------------------------------------------------------

@test
def test_fresh_game_conserves_96_cards():
    st = setup_game()
    seen = card_census(st)
    assert sum(seen.values()) == 96, \
        'a fresh game holds all 96 cards, got %d' % sum(seen.values())
    assert_conserved(st, 'fresh game:')


# ---------------------------------------------------------------------------
# 1. Table-card removals discard face-up instead of deleting
# ---------------------------------------------------------------------------

@test
def test_hunt_reduction_discards_horn_of_gondor():
    """ai card_option hunt_reduction (self-play): the spent Horn of
    Gondor reaches the face-up pile.  auto35 lost it here."""
    CID = 'FREE_PEOPLES_Character_Horn_of_Gondor'
    st = setup_game()
    put_on_table(st, CID)
    st._pending_hunt_damage = 2
    st.pending_card_option = {'card_id': CID, 'card_name': 'Horn of Gondor',
                              'side': 'fp', 'decider': 'fp',
                              'hunt_reduction': True, 'direct': True}
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='card_option',
        params={'idx': 0, 'card_id': CID, 'card_name': 'Horn of Gondor',
                'side': 'fp',
                'choices': {'use_hunt_reduction': True, 'card_id': CID}}))
    assert not on_table(st, CID), 'the Horn leaves the table'
    assert in_face_up(st, CID), \
        'the spent Horn of Gondor is DISCARDED face-up, not deleted'
    assert_conserved(st, 'hunt reduction:')


@test
def test_table_card_kill_discards_the_palantir():
    """The table-card kill action physically discards its target.
    auto35 lost The Palantir of Orthanc here."""
    CID = 'SHADOW_Character_The_Palantír_of_Orthanc'
    if CID not in get_cards():
        CID = next(c for c in get_cards() if 'Palant' in c and 'Orthanc' in c)
    st = setup_game()
    put_on_table(st, CID)
    before = len(st.persistent_cards.sh_cards)
    from persistent_effects import discard_table_card
    _rm = st.persistent_cards.remove_card(CID)
    discard_table_card(st, _rm)
    assert before and _rm is not None, 'the card was on the table'
    assert in_face_up(st, CID), 'a killed table card is discarded face-up'
    assert_conserved(st, 'table-card kill:')


@test
def test_crebain_discard_reaches_the_pile():
    CID = 'SHADOW_Character_Flocks_of_Crebain'
    st = setup_game()
    put_on_table(st, CID, PCT.FLOCKS_OF_CREBAIN)
    res = ee._effect_flocks_of_crebain(
        st, get_card(CID), {'crebain_discard': True})
    assert not on_table(st, CID), 'Crebain leaves the table when discarded'
    assert in_face_up(st, CID), \
        'the discarded Flocks of Crebain reaches the face-up pile'
    assert getattr(st, '_crebain_hunt_bonus', 0) == 1, \
        'the +1 hunt bonus still applies'
    assert_conserved(st, 'crebain discard:')


@test
def test_gui_helper_remove_persistent_card_discards():
    """game_logic.remove_persistent_card is the GUI's sole table-removal
    entry point (Crebain / Wizard's Staff / Mithril / Axe and Bow /
    Horn) -- it must discard, not delete."""
    for CID in ('FREE_PEOPLES_Character_Axe_and_Bow',
                "FREE_PEOPLES_Character_Wizard's_Staff",
                'FREE_PEOPLES_Character_Mithril_Coat_and_Sting'):
        if CID not in get_cards():
            continue
        st = setup_game()
        pcard = put_on_table(st, CID)
        r = game_logic.remove_persistent_card(st, pcard.card_name)
        assert r.get('success'), '%s was on the table' % CID
        assert in_face_up(st, CID), \
            '%s is discarded face-up by the GUI helper' % CID
        assert_conserved(st, 'gui helper %s:' % CID)


@test
def test_denethors_folly_auto_discard_conserves():
    """Denethor's Folly is auto-discarded when Minas Tirith is no longer
    besieged (both the game_logic upkeep and the ai stable-point sweep)."""
    CID = "SHADOW_Strategy_Denethor's_Folly"
    if CID not in get_cards():
        CID = next((c for c in get_cards() if 'Denethor' in c), None)
        assert CID, "Denethor's Folly card id resolves"
    st = setup_game()
    put_on_table(st, CID, PCT.DENETHORS_FOLLY)
    st.besieged_strongholds = set()
    game_logic.cleanup_sieges(st)
    assert not on_table(st, CID), "Denethor's Folly leaves the table"
    assert in_face_up(st, CID), \
        "the auto-discarded Denethor's Folly reaches the face-up pile"
    assert_conserved(st, "denethor's folly:")


# ---------------------------------------------------------------------------
# 2. Hand accounting: a played/paid card always lands in a pile
# ---------------------------------------------------------------------------

@test
def test_played_event_leaves_hand_and_lands_in_a_pile():
    """The ordinary event-play path removes from hand and discards
    (persistent cards go to the TABLE instead -- also a real place)."""
    st = setup_game(deal=3)
    hand = st.hands[Side.SHADOW].hand
    assert hand, 'the Shadow was dealt a hand'
    cid = hand[0]
    n_before = len(hand)
    ee.play_event_card(st, Side.SHADOW, cid, {})
    seen = card_census(st)
    assert seen.get(cid, 0) == 1, \
        'the played card %s is in EXACTLY one place, got %d' \
        % (cid, seen.get(cid, 0))
    assert len(st.hands[Side.SHADOW].hand) in (n_before - 1, n_before), \
        'hand shrinks by one (or the play was refunded)'
    assert_conserved(st, 'event play:')


@test
def test_next_turn_opens_a_learned_discard_and_never_trims():
    """USER RULING 2026-08-15: the hand-limit discard is a LEARNED
    decision.  next_turn must OPEN the pending and touch no card - it
    used to auto-trim (first with an unseeded random.choice, then
    deterministically), which stole the decision from the nets."""
    import copy
    import random

    random.seed(7)
    base = setup_game(deal=4)            # 8 cards a side, over the limit
    n_fp = len(base.hands[Side.FREE_PEOPLES].hand)
    n_sh = len(base.hands[Side.SHADOW].hand)
    assert n_fp > 6 and n_sh > 6, 'both hands start over the limit'

    fd0 = {s: len(base.hands[s].discarded_face_down)
           for s in (Side.FREE_PEOPLES, Side.SHADOW)}

    def run(seed):
        st = copy.deepcopy(base)
        random.seed(seed)
        game_logic.next_turn(st)          # draws 2/side, then OPENS the pending
        assert_conserved(st, 'after next_turn:')
        return st

    a, b = run(1), run(999)
    for st in (a, b):
        # the rollover only ADDS (its two draws); it discards nothing
        assert len(st.hands[Side.FREE_PEOPLES].hand) >= n_fp, \
            'next_turn discards NOTHING by itself (FP %d -> %d)' \
            % (n_fp, len(st.hands[Side.FREE_PEOPLES].hand))
        assert len(st.hands[Side.SHADOW].hand) >= n_sh, \
            'next_turn discards NOTHING by itself (SH %d -> %d)' \
            % (n_sh, len(st.hands[Side.SHADOW].hand))
        for s in (Side.FREE_PEOPLES, Side.SHADOW):
            assert len(st.hands[s].discarded_face_down) == fd0[s], \
                'the rollover put NOTHING in the discard pile itself (%s)' % s
        pd = getattr(st, 'pending_discard', None)
        assert pd is not None, 'the learned discard pending is OPEN'
        assert pd['side'] == Side.FREE_PEOPLES, \
            'FP is asked first when both sides are over: %r' % (pd,)
        assert st.hands[Side.FREE_PEOPLES].needs_discard is True
    assert (list(a.hands[Side.SHADOW].hand)
            == list(b.hands[Side.SHADOW].hand)), \
        'the turn path is RNG-independent for hands'


@test
def test_discard_pending_offers_every_card_and_resolves_one_at_a_time():
    """The pending is a real menu (every distinct hand card is an
    option) and each execution resolves EXACTLY one card."""
    st = setup_game(deal=4)
    game_logic.next_turn(st)
    d_side = st.pending_discard['side']
    hand0 = list(st.hands[d_side].hand)
    acts = ai.get_legal_actions(st, d_side)
    assert acts and all(a.action_type == 'discard_card' for a in acts), \
        'the pending offers only discard_card actions'
    assert {a.params['card_id'] for a in acts} == set(hand0), \
        'every distinct hand card is offered'

    chosen = hand0[2]                      # a deliberate, non-edge pick
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='discard_card',
        params={'card_id': chosen,
                'side': 'fp' if d_side == Side.FREE_PEOPLES else 'sh'}))
    assert chosen not in st.hands[d_side].hand, 'the CHOSEN card left the hand'
    assert chosen in st.hands[d_side].discarded_face_down, \
        'the chosen card landed in the face-down discard pile'
    assert len(st.hands[d_side].hand) == len(hand0) - 1, \
        'exactly one card was discarded'
    assert_conserved(st, 'after one discard:')

    # still over the limit -> the SAME side is asked again (one at a time)
    if len(st.hands[d_side].hand) > 6:
        assert st.pending_discard is not None \
            and st.pending_discard['side'] == d_side, \
            'a 2-card overdraw is TWO decisions, not one batch'

    # drain to completion and confirm the pending clears
    guard = 0
    while getattr(st, 'pending_discard', None) is not None and guard < 12:
        guard += 1
        s2 = st.pending_discard['side']
        a2 = ai.get_legal_actions(st, s2)[0]
        ai._execute_action(st, a2)
        assert_conserved(st, 'mid-drain:')
    assert st.pending_discard is None, 'the pending clears once both hands fit'
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        assert len(st.hands[s].hand) <= 6, 'hand limit 6 is enforced'


@test
def test_gui_discard_button_conserves_and_advances_the_pending():
    """gui.py is not editable, so its Discard button's shared helper
    must do the whole job: real pile + pending advance."""
    st = setup_game(deal=4)
    game_logic.next_turn(st)
    d_side = st.pending_discard['side']
    hand0 = list(st.hands[d_side].hand)
    res = game_logic.discard_card_from_hand(st, d_side, 1)
    assert res['success'], 'the GUI helper reports success'
    gone = hand0[1]
    assert res['discarded'] == gone
    assert gone not in st.hands[d_side].hand, 'the card left the hand'
    assert gone in st.hands[d_side].discarded_face_down, \
        'the GUI discard reaches the pile (was annihilated before)'
    assert_conserved(st, 'gui discard:')
    # pending advanced, never left stale
    pd = getattr(st, 'pending_discard', None)
    if len(st.hands[d_side].hand) <= 6 \
            and len(st.hands[Side.SHADOW if d_side == Side.FREE_PEOPLES
                             else Side.FREE_PEOPLES].hand) <= 6:
        assert pd is None, 'pending cleared once every hand fits'
    else:
        assert pd is not None, 'pending re-derived, never stale'


@test
def test_no_auto_trim_idiom_remains_in_the_turn_paths():
    """Guard the ruling itself: no engine path may pick the discard."""
    import io
    bad = []
    for fn in ('game_logic.py', 'ai.py', 'engine_play.py',
               'event_effects.py'):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)
        src = io.open(p, encoding='utf-8', errors='replace').read()
        for idiom in ('random.choice(hand)', 'random.choice(_h)',
                      'hand.remove(random.', 'hand.pop()'):
            if idiom in src:
                bad.append('%s: %s' % (fn, idiom))
    assert not bad, 'auto-trim idiom still present: %s' % bad


@test
def test_extraction_discard_contract():
    """rendered_extract._emit_discards resolves the human's LOGGED pick
    by setting pending_discard on a reconstructed board, calling
    get_legal_actions and matching card_id.  Guard that contract."""
    st = setup_game(deal=4)
    d_side = Side.SHADOW
    logged_cid = st.hands[d_side].hand[3]      # "what the human discarded"
    st.pending_discard = {'side': d_side}
    legal = ai.get_legal_actions(st, d_side)
    hidx = next((i for i, a in enumerate(legal)
                 if a.action_type == 'discard_card'
                 and str(a.params.get('card_id')) == str(logged_cid)), None)
    assert hidx is not None, \
        'the logged discard is matchable in the legal menu (BC sample)'
    ai._execute_action(st, legal[hidx])
    assert logged_cid not in st.hands[d_side].hand, \
        'replaying the logged pick discards exactly that card'
    assert logged_cid in st.hands[d_side].discarded_face_down
    assert_conserved(st, 'extraction contract:')


@test
def test_hand_limit_discard_conserves():
    st = setup_game(deal=3)
    hand = st.hands[Side.SHADOW].hand
    cid = hand[0]
    ai._execute_action(st, ai.Action(
        die_idx=-1, action_type='discard_card',
        params={'side': 'sh', 'card_id': cid}))
    assert cid not in st.hands[Side.SHADOW].hand, 'the card left the hand'
    assert cid in st.hands[Side.SHADOW].discarded_face_down, \
        'a hand-limit discard goes to the face-DOWN pile'
    assert_conserved(st, 'hand-limit discard:')


# ---------------------------------------------------------------------------
# 2b. PARSER hand tracking (log_replay): a reveal must never delete a
#     card from a hand that did not hold it.  Both live streak breaks
#     (auto35 pid182 Cruel Weather, auto36 pid183 The Day Without Dawn)
#     were this: the reveal missed every matcher and fell through to a
#     blind `hand.pop(0)`, then the deficit cascaded through the hand.
# ---------------------------------------------------------------------------

def _replayer():
    import log_replay
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    r = log_replay.LogReplayer(st, b, verbose=False)
    r._ai_bridge_log = True
    # give each side a tracked hand of piece-id dicts (what the parser
    # actually carries before the extraction harness converts them)
    st.hands[Side.FREE_PEOPLES].hand = [
        {'piece_id': p, 'card_name': 'Unknown'} for p in ('109', '127', '115')]
    st.hands[Side.SHADOW].hand = [
        {'piece_id': p, 'card_name': 'Unknown'} for p in ('183', '209', '168')]
    return r, st


def _hand_pids(st, side):
    return [c.get('piece_id') for c in st.hands[side].hand
            if isinstance(c, dict)]


@test
def test_reveal_from_a_board_zone_never_touches_a_hand():
    """Both reveal grammars carry a SOURCE zone: Controls.867 (play)
    and Controls.862 (combat-card commitment).  A Game.* source means
    the card was already on the board/table."""
    cases = [
        # auto35 root: the on-table Balrog played as a combat card
        ('<auto><~Controls.866~>reveal 174 ABCDEF01 23'
         '<~Controls.867~><~Game.2034~><~Controls.868~><~Game.2036~>'),
        # auto36 root: an FP commitment revealed from Current FP Card
        ('<auto><~Controls.861~>reveal 156 94983E1C 20'
         '<~Controls.862~><~Game.2037~><~Controls.863~><~Game.2037~>'),
    ]
    for line in cases:
        r, st = _replayer()
        before_sh = _hand_pids(st, Side.SHADOW)
        before_fp = _hand_pids(st, Side.FREE_PEOPLES)
        r._process_line(line)
        assert _hand_pids(st, Side.SHADOW) == before_sh, \
            'a Game.*-sourced reveal must not touch the SH hand: %s -> %s' \
            % (before_sh, _hand_pids(st, Side.SHADOW))
        assert _hand_pids(st, Side.FREE_PEOPLES) == before_fp, \
            'a Game.*-sourced reveal must not touch the FP hand: %s -> %s' \
            % (before_fp, _hand_pids(st, Side.FREE_PEOPLES))


@test
def test_reveal_never_touches_the_hand_its_pid_contradicts():
    """Piece-id RANGE is authoritative (106-157 FP, 158-209 SH).  The
    auto36 break sniffed an FP pid-156 commitment as SHADOW (no hand
    marker in the line) and blind-popped pid 183 out of the SH hand."""
    r, st = _replayer()
    before_sh = _hand_pids(st, Side.SHADOW)
    # FP piece id, but nothing in the line marks it FP
    r._process_line('<auto><~Controls.861~>reveal 156 94983E1C 20'
                    '<~Controls.863~><~Game.2037~>')
    assert _hand_pids(st, Side.SHADOW) == before_sh, \
        'an FP pid must never pop a SHADOW hand card: %s -> %s' \
        % (before_sh, _hand_pids(st, Side.SHADOW))
    assert '183' in _hand_pids(st, Side.SHADOW), \
        'The Day Without Dawn (pid 183) survives the FP commitment'


@test
def test_ordinary_hand_reveal_still_removes_its_card():
    """The guards must not break the 99.9% case: a reveal sourced from
    the hand still removes exactly its own piece."""
    r, st = _replayer()
    r._process_line('<auto><~Controls.866~>reveal 209 8B348E00 16'
                    '<~Controls.867~><~Controls.46~><~Controls.868~><~Game.2036~>')
    pids = _hand_pids(st, Side.SHADOW)
    assert '209' not in pids, 'the revealed card leaves the hand: %s' % pids
    assert '183' in pids and '168' in pids, \
        'no other card is disturbed: %s' % pids


# ---------------------------------------------------------------------------
# 3. Whole-game sweep: conservation survives real self-play
# ---------------------------------------------------------------------------

@test
def test_selfplay_game_conserves_cards():
    """A short scripted self-play game must never annihilate a card."""
    import random
    random.seed(11)
    st = setup_game(deal=3)
    assert_conserved(st, 'turn 0:')

    def _piles(state):
        n = sum(len(v) for v in
                (getattr(state, 'discard_face_up', None) or {}).values())
        for s in (Side.FREE_PEOPLES, Side.SHADOW):
            n += len(getattr(state.hands[s], 'discarded_face_down', []))
        pc = getattr(state, 'persistent_cards', None)
        if pc is not None:
            n += len(pc.fp_cards) + len(pc.sh_cards)
        return n

    piles0 = _piles(st)
    steps = 0
    for _round in range(24):
        for s in (Side.FREE_PEOPLES, Side.SHADOW):
            game_logic.roll_action_dice(st, pool=s)
            for dk in ('character', 'strategy'):
                try:
                    game_logic.draw_event_card(st, s, dk)
                except Exception:
                    pass
        assert_conserved(st, 'after upkeep draws:')
        for _ in range(60):
            side = Side.SHADOW if steps % 2 else Side.FREE_PEOPLES
            try:
                acts = ai.get_legal_actions(st, side)
            except Exception:
                acts = []
            if not acts:
                other = (Side.SHADOW if side == Side.FREE_PEOPLES
                         else Side.FREE_PEOPLES)
                try:
                    acts = ai.get_legal_actions(st, other)
                except Exception:
                    acts = []
                if not acts:
                    break
            a = random.choice(acts)
            try:
                ai._execute_action(st, a)
            except Exception:
                steps += 1
                continue
            steps += 1
            assert_conserved(st, 'after %s:' % a.action_type)
    assert steps >= 15, 'the sweep executed real actions (%d)' % steps
    assert _piles(st) > piles0, \
        'cards actually LEFT hands during the sweep (%d -> %d)' \
        % (piles0, _piles(st))


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("CARD CONSERVATION REGRESSION (2026-08-15)")
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
