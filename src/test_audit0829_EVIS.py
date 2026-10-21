# -*- coding: ascii -*-
"""Regression pins for amendment 1.261 (2026-08-31), owner EVIS
(src/ai.py + src/game_logic.py).

1.261: table-kill payment cards and the WWSAT random FP-character hand
discard are FACE-DOWN (payer's / owner's discarded_face_down, never the
face-up piles), and the encoding's 96-dim card-status block is
PER-SIDE: public discard_face_up union the ENCODING side's OWN
discarded_face_down at 0.5 - the opponent's face-down pile stays dark.

Every fix pin ran RED against the unfixed tree (EVIS_FIX_REPORT.md
quotes the RED lines); test_1261_clone_deep_copies_face_down is a
verification guard (expected GREEN by design, brief item 4).
"""
import os
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import board
import game_state as gs
import game_logic
import ai
import cards
from game_state import (Side, ActionDieFace as Face, PersistentCard,
                        PersistentCardType)

FP = Side.FREE_PEOPLES
SH = Side.SHADOW


# ---------------------------------------------------------------- helpers
def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for attr in ('_declare_phase', '_eye_allocation_phase'):
        setattr(st, attr, False)
    st.pending_discard = None
    return st


def set_dice(st, side, faces):
    st.dice[side] = []
    for f in faces:
        d = gs.ActionDie(faces=[f], side=side)
        d.current = f
        st.dice[side].append(d)


def _faceup_snapshot(st):
    return {k: list(v) for k, v in (st.discard_face_up or {}).items()}


def _in_faceup(st, cid):
    return any(cid in lst for lst in (st.discard_face_up or {}).values())


def _table_kill_fixture():
    """The EX-8 fixture: Shadow hand of exactly 2 Army-type + 2
    Character cards, an FP table card to kill, one Shadow Army die."""
    st = setup_game()
    hand = st.hands[SH].hand
    del hand[:]
    army = [c for c in st.sh_strat_deck
            if getattr(cards.get_card(c), 'card_type', None) == 'Army'][:2]
    chars = [c for c in st.sh_char_deck if '_Character_' in c][:2]
    for c in army:
        st.sh_strat_deck.remove(c)
    for c in chars:
        st.sh_char_deck.remove(c)
    hand[:] = [army[0], chars[0], army[1], chars[1]]
    st.persistent_cards.add_card(PersistentCard(
        card_id='FREE_PEOPLES_Strategy_A_Power_too_Great',
        card_name='A Power too Great', side=FP,
        card_type=PersistentCardType.POWER_TOO_GREAT))
    set_dice(st, SH, [Face.ARMY])
    set_dice(st, FP, [Face.CHARACTER])
    return st, army, chars


def _pay_table_kill(st, army_cid, char_cid):
    """Drive the full path: kill action -> learned payment pending ->
    card_option pick (the pair the caller names)."""
    kills = [a for a in ai.get_legal_actions(st, SH)
             if a.action_type == 'discard_table_card']
    assert kills, 'fixture: table kill not offered'
    ai._execute_action(st, kills[0])
    pick = ai.table_kill_payment_action(st, army_cid, char_cid)
    ai._execute_action(st, pick)


def _status_dim(cid):
    """The card-status encoding dim for `cid`, found differentially:
    append cid to its face-up pile on a fresh state and diff the
    encodings - exactly one dim moves (0 -> 0.5).  Layout-independent."""
    st = setup_game()
    base = ai.encode_state(st, SH)
    c = cards.get_card(cid)
    pfx = 'fp' if c.side == FP else 'sh'
    st.discard_face_up['%s_%s' % (pfx, str(c.deck).lower())].append(cid)
    after = ai.encode_state(st, SH)
    diff = np.nonzero(after - base)[0]
    assert len(diff) == 1, \
        'status-dim probe for %r moved %r dims' % (cid, diff)
    assert after[diff[0]] == 0.5, after[diff[0]]
    return int(diff[0])


# ------------------------------------------------------------- 1.261 (a)
def test_1261_table_kill_payment_files_face_down():
    """1.261(a): both payment cards leave the Shadow hand to the
    Shadow's discarded_face_down - NOT to any face-up pile (the
    2026-08-15 face-up filing was a hidden-information leak).  Card
    conservation: hand -2, both cards tracked face-down, table card
    gone."""
    st, army, chars = _table_kill_fixture()
    hand = st.hands[SH].hand
    _pay_table_kill(st, army[1], chars[1])
    assert st.persistent_cards.get_by_type(
        PersistentCardType.POWER_TOO_GREAT) is None, 'table card not killed'
    assert sorted(hand) == sorted([army[0], chars[0]]), \
        '1.261: hand not -2 after payment: %r' % (hand,)
    fdown = st.hands[SH].discarded_face_down
    assert army[1] in fdown and chars[1] in fdown, \
        '1.261 RED: payment cards not in Shadow discarded_face_down: %r' \
        % (fdown,)
    assert not _in_faceup(st, army[1]) and not _in_faceup(st, chars[1]), \
        '1.261 RED: payment cards leaked to a face-up pile'


def test_1261_pay_table_card_kill_direct_body():
    """1.261(a) on the ONE payment body directly (both callers - the
    card_option executor and the teacher auto-pay - route through
    pay_table_card_kill)."""
    st, army, chars = _table_kill_fixture()
    ai.pay_table_card_kill(st, army[0], chars[0])
    fdown = st.hands[SH].discarded_face_down
    assert army[0] in fdown and chars[0] in fdown, \
        '1.261 RED: pay_table_card_kill filed outside discarded_face_down'
    assert not _in_faceup(st, army[0]) and not _in_faceup(st, chars[0]), \
        '1.261 RED: pay_table_card_kill filed face-up'
    assert sorted(st.hands[SH].hand) == sorted([army[1], chars[1]])


# ------------------------------------------------------------- 1.261 (b)
def test_1261_wwsat_hand_discard_files_face_down():
    """1.261(b): the WWSAT random FP-character hand discard goes to the
    FP's discarded_face_down (the client renders NO identity - blinding
    holds); every face-up pile is untouched; hand -1."""
    st = setup_game()
    cid = 'FREE_PEOPLES_Character_Athelas'
    st.hands[FP].hand = [cid]
    before = _faceup_snapshot(st)
    game_logic._wwsat_resolve(st, 'hand')
    assert st.hands[FP].hand == [], 'WWSAT did not discard the hand card'
    assert cid in st.hands[FP].discarded_face_down, \
        '1.261 RED: WWSAT hand discard not in FP discarded_face_down: %r' \
        % (st.hands[FP].discarded_face_down,)
    assert _faceup_snapshot(st) == before, \
        '1.261 RED: WWSAT hand discard touched a face-up pile'


# ------------------------------------------------------------- 1.261 (c)
def test_1261_encoding_lights_own_face_down_only():
    """1.261(c): the card-status block is PER-SIDE - after a table-kill
    payment encode_state(st, SH) lights both payment cards at 0.5 while
    encode_state(st, FP) keeps them at 0.0 (the opponent only sees two
    cards leave the hand).  Symmetric control: an FP hand-limit discard
    lights for FP, stays dark for SH.  Layout unchanged."""
    st, army, chars = _table_kill_fixture()
    _pay_table_kill(st, army[1], chars[1])
    d_army, d_char = _status_dim(army[1]), _status_dim(chars[1])
    e_sh = ai.encode_state(st, SH)
    e_fp = ai.encode_state(st, FP)
    assert e_sh[d_army] == 0.5 and e_sh[d_char] == 0.5, \
        '1.261 RED: payer does not see its own face-down payment ' \
        '(sh=%r/%r)' % (e_sh[d_army], e_sh[d_char])
    assert e_fp[d_army] == 0.0 and e_fp[d_char] == 0.0, \
        '1.261 RED: opponent reads the face-down payment identities ' \
        '(fp=%r/%r)' % (e_fp[d_army], e_fp[d_char])
    # symmetric control: FP files its own face-down (hand-limit primitive)
    st2 = setup_game()
    cid = 'FREE_PEOPLES_Character_Athelas'
    if cid in st2.fp_char_deck:
        st2.fp_char_deck.remove(cid)
    st2.hands[FP].hand.append(cid)
    assert game_logic.discard_hand_card(st2, FP, cid)
    d_cid = _status_dim(cid)
    e_fp2 = ai.encode_state(st2, FP)
    e_sh2 = ai.encode_state(st2, SH)
    assert e_fp2[d_cid] == 0.5, \
        '1.261 RED: FP does not see its own face-down discard'
    assert e_sh2[d_cid] == 0.0, \
        '1.261 RED: SH reads the FP face-down discard identity'


# ------------------------------------------------- clone_state guard (4)
def test_1261_clone_deep_copies_face_down():
    """Guard (expected GREEN by design, brief item 4): clone_state
    list-copies discarded_face_down per side - mutating the clone's
    list never touches the parent."""
    st = setup_game()
    st.hands[SH].discarded_face_down.append('SH_MARKER_A')
    st.hands[FP].discarded_face_down.append('FP_MARKER_A')
    c2 = ai.clone_state(st)
    assert c2.hands[SH].discarded_face_down == ['SH_MARKER_A']
    assert c2.hands[FP].discarded_face_down == ['FP_MARKER_A']
    c2.hands[SH].discarded_face_down.append('SH_MARKER_B')
    del c2.hands[FP].discarded_face_down[:]
    assert st.hands[SH].discarded_face_down == ['SH_MARKER_A'], \
        'clone mutation reached the parent (SH)'
    assert st.hands[FP].discarded_face_down == ['FP_MARKER_A'], \
        'clone mutation reached the parent (FP)'


# ------------------------------------------------------------------ main
def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    fail = 0
    for t in tests:
        try:
            t()
            print('PASS %s' % t.__name__)
        except Exception:
            fail += 1
            print('FAIL %s' % t.__name__)
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(tests), fail))
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
