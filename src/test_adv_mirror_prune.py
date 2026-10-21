"""RED-TEAM finding EX-1: Mirror of Galadriel is menu-pruned as a
no-op everywhere outside a Lorien heal, because the no-op dry-run
signature (_event_state_signature) hashes dice as (count, unused)
per side and is BLIND TO DIE FACES -- while the card's headline
effect IS a die-face change (Character -> Will of the West).

Mechanism walkthrough (per the bug-explanation directive):
  RULE: the card is playable whenever FP holds it and an unused
  Character die exists to convert; the Lorien heal is a bonus rider.
  ENGINE: check_event_requirement returns (True, 'OK'); the menu then
  dry-runs the effect and compares _event_state_signature before/after.
  The handler converts one unused CHARACTER die to WILL_OF_WEST --
  neither the die count nor the unused count changes, so the
  signature compares EQUAL and _event_card_has_effect returns False.
  NUMBERS: 162 net-free chaos games (red_team chaos_ledger.jsonl,
  seeds 90000-91149), Mirror of Galadriel offered 0 times; on the
  initial deal with Mirror in hand + unused Character die + unused
  Event die, get_legal_actions offers no play_event for it, while
  ai.apply_action applies the same play and converts the die.
  CONSEQUENCE: self-play NEVER trains the card (BC prior mismatch --
  humans do play it), the sided AI can never play it, and the human
  CAN (the bridge N1 card-play path calls apply_action directly),
  an asymmetry against ruling 1.248's shared gate.  This is the
  sibling of the QA 2026-07-10 pcards signature gap, fixed for table
  cards but never for die faces.

EXPECTED: FAILS against the current tree (that is the point).
Leg 2 pins the mechanism (signature must distinguish die faces),
not the code shape.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def _mk_state():
    import board
    import game_state as gs
    import game_logic
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def _arm(st):
    """Mirror into the FP hand from its real deck; one EVENT die (to
    play with) + one CHARACTER die (the convert target), both unused."""
    from game_state import Side, ActionDieFace
    cid = next(c for c in (st.fp_char_deck + st.fp_strat_deck)
               if 'Mirror_of_Galadriel' in c)
    for cont in (st.fp_char_deck, st.fp_strat_deck):
        if cid in cont:
            cont.remove(cid)
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    d = st.dice[Side.FREE_PEOPLES]
    d[0].current = ActionDieFace.EVENT
    d[0].used = False
    d[1].current = ActionDieFace.CHARACTER
    d[1].used = False
    return cid


def test_mirror_of_galadriel_offered_outside_lorien():
    """The menu must offer Mirror of Galadriel when its requirement
    holds and a Character die exists to convert -- the die-face change
    IS an effect."""
    import ai
    from game_state import Side
    st = _mk_state()
    cid = _arm(st)
    from event_effects import check_event_requirement
    ok, why = check_event_requirement(st, cid)
    assert ok, ('precondition of this test broke: requirement oracle '
                'says %r' % (why,))
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    offered = [a for a in acts
               if a.action_type == 'play_event'
               and a.params.get('card_id') == cid]
    assert offered, (
        'MENU PRUNED A LEGAL CARD: Mirror of Galadriel in hand, '
        'requirement OK, unused Character die present, unused Event '
        'die present -- and get_legal_actions offers no play_event '
        'for it (die-face change invisible to '
        '_event_state_signature).')


def test_event_signature_sees_die_faces():
    """Mechanism pin: two states differing ONLY in one die face must
    produce different event-state signatures, or every die-face-only
    card effect is pruned as a no-op."""
    import ai
    from game_state import Side, ActionDieFace
    st = _mk_state()
    d = st.dice[Side.FREE_PEOPLES]
    d[0].current = ActionDieFace.CHARACTER
    d[0].used = False
    sig_a = ai._event_state_signature(st, Side.FREE_PEOPLES)
    d[0].current = ActionDieFace.WILL_OF_WEST
    sig_b = ai._event_state_signature(st, Side.FREE_PEOPLES)
    assert sig_a != sig_b, (
        'signature blind to die faces: CHARACTER vs WILL_OF_WEST '
        'hash identically, so a die-face-changing effect compares '
        'as a no-op')


if __name__ == '__main__':
    fails = 0
    for fn in (test_mirror_of_galadriel_offered_outside_lorien,
               test_event_signature_sees_die_faces):
        try:
            fn()
            print('PASS %s' % fn.__name__)
        except AssertionError as e:
            fails += 1
            print('FAIL %s: %s' % (fn.__name__, e))
    sys.exit(1 if fails else 0)
