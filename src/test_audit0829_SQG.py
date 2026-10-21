#!/usr/bin/env python3
# -*- coding: ascii -*-
"""SQG regression pins (2026-08-30) - SQF handoff items 1 and 2.

SQG-1  (SQF F4, 1.248-class latent gap): The Spirit of Mordor's
       target_region choice belongs to the CARD PLAYER.  Decision-
       ownership doctrine (every player choice is a learned decision /
       pending; ruling 1.248: the human's actions ride the same legal
       surface as the AI's) forbids a silent in-handler default when
       TWO OR MORE valid multi-nation Shadow armies stand: the
       choiceless play must open a pending_card_option owned by the
       playing side, one option per valid region.  A SINGLE valid
       target stays a forced pick (rules-correct, no decision exists).

SQG-2  (SQF F9(b), harness): run_scenario.human_action's by-fields
       menu filter matched only TOP-LEVEL a.params[k], but a
       card_option row's fields live in the NESTED params['choices']
       dict - the seat's spec matched nothing and the act lane refused
       a row that WAS published (seqack_fp1 row #2337).  The shared
       matcher (run_scenario._filter_menu_rows) must match a filter
       key at EITHER level.

Run:  python test_audit0829_SQG.py     (exit 0 = all pass)
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
from cards import get_card, get_cards        # noqa: E402

Side = gs.Side
Nation = gs.Nation
Kind = gs.UnitKind

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
    """Force the card-roll funnel to `value`; returns restore()."""
    o_ee = ee._d6_batch
    ee._d6_batch = lambda n, ctx='card effect': [value] * n

    def restore():
        ee._d6_batch = o_ee
    return restore


def spirit_cid():
    cid = next((c for c, card in get_cards().items()
                if getattr(card, 'card_name', '')
                == 'The Spirit of Mordor'), None)
    assert cid is not None, 'The Spirit of Mordor card id resolves'
    return cid


def arm_fp(st, cid):
    """Seat cid in the FP hand (out of whatever pile holds it)."""
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
    st.hands[Side.FREE_PEOPLES].hand.append(cid)


def two_targets(st):
    """Two multi-nation Shadow armies; dol guldur is the LARGER one
    (the old silent default's pick)."""
    a1 = clear(st, 'moria')
    a1.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a1.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a1.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    a2 = clear(st, 'dol guldur')
    for _ in range(4):
        a2.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a2.units.append(gs.Unit(Nation.SOUTHRON_EASTERLINGS, Kind.REGULAR))


# ---------------------------------------------------------------------------
# SQG-1: multi-target choiceless play opens the CARD PLAYER's pending
# ---------------------------------------------------------------------------

@test
def test_spirit_of_mordor_multi_target_opens_owner_pending():
    """SQF F4 / 1.248-class + decision-ownership doctrine: with 2+
    valid multi-nation Shadow armies, a choiceless play (the human
    client lane threads no choices) must OPEN pending_card_option
    owned by the playing side - never silently pick 'the largest'."""
    CID = spirit_cid()
    st = setup_game()
    two_targets(st)
    arm_fp(st, CID)
    restore = force_d6(2)                    # no hits if a roll leaks
    try:
        res = ee.play_event_card(st, Side.FREE_PEOPLES, CID, {})
    finally:
        restore()
    assert res.get('success'), 'play refused: %r' % (res,)
    pco = getattr(st, 'pending_card_option', None)
    assert pco is not None, \
        'SQG-1 RED: choiceless multi-target play opened NO pending ' \
        '(silent default substituted the card player\'s learned ' \
        'decision): %r' % (res.get('effects'),)
    assert pco.get('side') == 'fp' and pco.get('decider') == 'fp', \
        'the target pick belongs to the CARD PLAYER (fp): %r' % (pco,)
    got = sorted(sanitize(str(o.get('target_region')))
                 for o in pco.get('options') or [])
    assert got == ['dol guldur', 'moria'], \
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
    # ownership on the legal surface: the pending is the FP's to
    # answer; asking the Shadow is a routing bug (owner-guard)
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
    # strikes the CHOSEN region (moria), not the old default (the
    # larger dol guldur)
    pick = next(a for a in acts
                if sanitize(str((a.params.get('choices') or {})
                                .get('target_region'))) == 'moria')
    restore = force_d6(5)                    # 5+ = hit, wipe of 3 regs
    try:
        st2 = ai.apply_action(st, pick)
    finally:
        restore()
    a_moria = entry(st2, 'moria')
    assert a_moria is None or not a_moria.units, \
        'the CHOSEN target (moria) took the hits: %r' \
        % (a_moria.units if a_moria else None,)
    a_dg = entry(st2, 'dol guldur')
    assert a_dg is not None and len(a_dg.units) == 5, \
        'the unchosen (old-default) target is untouched: %r' % (a_dg,)


@test
def test_spirit_of_mordor_single_target_stays_forced():
    """SQF F4 verdict: exactly ONE valid target = a FORCED pick (no
    decision exists) - the play resolves immediately, no pending."""
    CID = spirit_cid()
    st = setup_game()
    a1 = clear(st, 'moria')
    a1.units.append(gs.Unit(Nation.SAURON, Kind.REGULAR))
    a1.units.append(gs.Unit(Nation.ISENGARD, Kind.REGULAR))
    arm_fp(st, CID)
    restore = force_d6(2)                    # 0 hits: no casualty leg
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
    assert rolled and 'moria' in sanitize(rolled[0]), \
        'the forced pick resolves against the one valid target: %r' \
        % (res.get('effects'),)


@test
def test_spirit_of_mordor_teacher_replay_keeps_the_default():
    """Teacher replay (combat.FINAL_HITS_HOOK installed, log = truth)
    never opens pendings - the documented default stands there, same
    guard pattern as _route_card_casualties (1.254 block)."""
    CID = spirit_cid()
    st = setup_game()
    two_targets(st)
    arm_fp(st, CID)
    prev_hook = cb.FINAL_HITS_HOOK
    cb.FINAL_HITS_HOOK = lambda *a, **k: None
    restore = force_d6(2)                    # 0 hits: roll only
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
    assert rolled and 'dol guldur' in sanitize(rolled[0]), \
        'teacher default = the largest valid army: %r' \
        % (res.get('effects'),)


# ---------------------------------------------------------------------------
# SQG-2: by-fields menu filter matches nested params['choices'] too
# ---------------------------------------------------------------------------

class _Row(object):
    def __init__(self, i, action_type, params, die_idx=None):
        self.i = i
        self.action_type = action_type
        self.params = params
        self.die_idx = die_idx


def _rs():
    lab = r'C:\Users\sarve\Downloads\WOTR_AI_training_lab\scenarios'
    if lab not in sys.path:
        sys.path.insert(0, lab)
    import run_scenario
    return run_scenario


@test
def test_by_fields_spec_matches_card_option_choices_fields():
    """SQF F9(b) (seqack_fp1 row #2337): a card_option spec filtering
    by the row's fields must resolve the published row even though
    those fields live under params['choices'], exactly as the session
    tried - the same row pick/act-by-index resolve."""
    rs = _rs()
    fn = getattr(rs, '_filter_menu_rows', None)
    assert fn is not None, \
        'SQG-2 RED: run_scenario has no shared by-fields matcher ' \
        '(_filter_menu_rows) - human_action filters inline on ' \
        'top-level params only'
    rows = [
        _Row(2336, 'card_option',
             {'idx': 0, 'card_id': 'FP_C_X', 'card_name': 'X',
              'side': 'fp',
              'choices': {'companion_action': 'move_Legolas',
                          'destination': 'lorien'}}),
        _Row(2337, 'card_option',
             {'idx': 1, 'card_id': 'FP_C_X', 'card_name': 'X',
              'side': 'fp',
              'choices': {'companion_action': 'move_Legolas',
                          'destination': 'dimrill dale'}}),
        _Row(2338, 'army_move',
             {'from': 'lorien', 'to': 'dimrill dale'}),
    ]
    spec = {'type': 'card_option', 'companion_action': 'move_Legolas',
            'destination': 'dimrill dale'}
    cands = [a for a in rows if a.action_type == 'card_option']
    out = fn(cands, spec)
    assert [a.i for a in out] == [2337], \
        'SQG-2 RED: the by-fields spec resolved %r, wanted [2337] ' \
        '(choices-level fields invisible to the filter)' \
        % ([a.i for a in out],)


@test
def test_by_fields_top_level_and_negative_legs_hold():
    """Sensitivity legs: top-level matching is unchanged; a spec that
    matches at NEITHER level still resolves nothing (the refusal
    stays loud for genuinely unpublished rows)."""
    rs = _rs()
    fn = getattr(rs, '_filter_menu_rows', None)
    assert fn is not None, 'shared matcher missing'
    rows = [
        _Row(1, 'army_move', {'from': 'lorien', 'to': 'dimrill dale'}),
        _Row(2, 'army_move', {'from': 'lorien', 'to': 'caras galadhon'}),
    ]
    out = fn(list(rows), {'type': 'army_move', 'to': 'dimrill dale'})
    assert [a.i for a in out] == [1], \
        'top-level filtering regressed: %r' % ([a.i for a in out],)
    out2 = fn(list(rows), {'type': 'army_move', 'to': 'moria'})
    assert out2 == [], \
        'a nowhere-matching spec must resolve NOTHING: %r' \
        % ([a.i for a in out2],)
    # a choices dict must not shadow a top-level mismatch into a
    # false positive for OTHER keys
    row3 = _Row(3, 'card_option',
                {'idx': 0, 'choices': {'destination': 'moria'}})
    out3 = fn([row3], {'type': 'card_option', 'destination': 'lorien'})
    assert out3 == [], \
        'choices-level mismatch must not match: %r' \
        % ([a.i for a in out3],)


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
