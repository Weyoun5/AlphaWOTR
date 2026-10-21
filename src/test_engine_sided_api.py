"""Phase-1 engine additions for the sided-play rebuild (REBUILD_PLAN.md).

Each test states its MEASURED QUANTITY first (instrument protocol,
USER directive 2026-08-19).  Every zero ships with a proven non-zero;
every instrument asserts the product's own quantity, never a proxy.

Covers:
  grant_draw / settle_owed_draw   (A13: the owed-draw debt writer)
  adopt_rolled_faces              (client-rolled faces, Eye->Hunt-Box)
  pick_random_companion           (named funnel, stream parity)
  _event_card_has_effect          (zero-randomness enumeration)
  describe_pending                (one-cascade decision packaging)
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai
import board as board_mod
import game_logic
import game_state as gs
from game_state import Side, ActionDieFace


class _Deferred(Exception):
    """Stand-in for a driver's client-call escape (NOT RuntimeError)."""


def fresh_state():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def _debt(st, side, deck):
    return (getattr(st, '_owed_draws', None) or {}).get(
        (side.name, deck), 0)


def test_grant_draw_success_no_residue(fails):
    """MEASURES: a successful grant draws exactly one card and leaves
    ZERO debt (specificity: success must not stamp)."""
    st = fresh_state()
    n0 = len(st.hands[Side.SHADOW].hand)
    game_logic.grant_draw(st, Side.SHADOW, 'character')
    if len(st.hands[Side.SHADOW].hand) != n0 + 1:
        fails.append(('grant_success_draws', 'hand did not grow by 1'))
    if _debt(st, Side.SHADOW, 'character') != 0:
        fails.append(('grant_success_no_residue',
                      'debt=%d after clean success'
                      % _debt(st, Side.SHADOW, 'character')))


def test_grant_draw_deferral_stamps_debt(fails):
    """MEASURES: a deferred draw (non-RuntimeError escape out of
    draw_event_card) records exactly ONE unit of (side.name, deck)
    debt and re-raises.  Sensitivity is built in: the asserted
    transition is 0 -> 1, so a missing writer goes RED."""
    st = fresh_state()
    if _debt(st, Side.SHADOW, 'character') != 0:
        fails.append(('deferral_precondition', 'debt not 0 at start'))
        return
    orig = game_logic.draw_event_card

    def _defer(state, side, deck):
        raise _Deferred('client owns this draw')
    game_logic.draw_event_card = _defer
    try:
        raised = False
        try:
            game_logic.grant_draw(st, Side.SHADOW, 'character')
        except _Deferred:
            raised = True
        if not raised:
            fails.append(('deferral_reraises', 'escape was swallowed'))
        if _debt(st, Side.SHADOW, 'character') != 1:
            fails.append(('deferral_stamps', 'debt=%d, expected 1'
                          % _debt(st, Side.SHADOW, 'character')))
        # specificity: the OTHER keys stay untouched
        for k, v in (getattr(st, '_owed_draws', {}) or {}).items():
            if k != (Side.SHADOW.name, 'character') and v:
                fails.append(('deferral_specificity',
                              'unrelated key %r stamped' % (k,)))
    finally:
        game_logic.draw_event_card = orig


def test_grant_draw_deck_empty_no_debt(fails):
    """MEASURES: RuntimeError (deck empty) propagates from grant_draw
    with NO debt recorded (an unfulfillable grant is not owed)."""
    st = fresh_state()
    st.sh_char_deck = []
    raised = False
    try:
        game_logic.grant_draw(st, Side.SHADOW, 'character')
    except RuntimeError:
        raised = True
    if not raised:
        fails.append(('empty_deck_raises', 'RuntimeError not raised'))
    if _debt(st, Side.SHADOW, 'character') != 0:
        fails.append(('empty_deck_no_debt', 'debt=%d, expected 0'
                      % _debt(st, Side.SHADOW, 'character')))


def test_grant_retry_settles_residue(fails):
    """MEASURES: deferral (+1) followed by a successful retry of the
    SAME grant nets debt 0 — a transactional driver's retry loop is
    self-cancelling."""
    st = fresh_state()
    orig = game_logic.draw_event_card

    def _defer(state, side, deck):
        raise _Deferred('client owns this draw')
    game_logic.draw_event_card = _defer
    try:
        try:
            game_logic.grant_draw(st, Side.SHADOW, 'strategy')
        except _Deferred:
            pass
    finally:
        game_logic.draw_event_card = orig
    if _debt(st, Side.SHADOW, 'strategy') != 1:
        fails.append(('retry_precondition', 'deferral did not stamp'))
        return
    game_logic.grant_draw(st, Side.SHADOW, 'strategy')   # the retry
    if _debt(st, Side.SHADOW, 'strategy') != 0:
        fails.append(('retry_settles', 'debt=%d after retry, expected 0'
                      % _debt(st, Side.SHADOW, 'strategy')))


def test_settle_owed_draw(fails):
    """MEASURES: settle_owed_draw consumes exactly one unit and reports
    truthfully.  Non-zero control first (a zero without a proven
    non-zero is unreadable): stamp 2, settle twice True, third False."""
    st = fresh_state()
    st._owed_draws = {(Side.FREE_PEOPLES.name, 'character'): 2}
    r1 = game_logic.settle_owed_draw(st, Side.FREE_PEOPLES, 'character')
    r2 = game_logic.settle_owed_draw(st, Side.FREE_PEOPLES, 'character')
    r3 = game_logic.settle_owed_draw(st, Side.FREE_PEOPLES, 'character')
    if (r1, r2, r3) != (True, True, False):
        fails.append(('settle_sequence', '(%s,%s,%s)' % (r1, r2, r3)))
    if _debt(st, Side.FREE_PEOPLES, 'character') != 0:
        fails.append(('settle_exhausts', 'debt nonzero after settles'))


def test_adopt_rolled_faces(fails):
    """MEASURES: adopt_rolled_faces lands externally-rolled faces with
    the SAME Eye rule as roll_action_dice (used + Hunt Box), is
    idempotent under re-scan, sets dice_rolled, and refuses an
    out-of-range index."""
    st = fresh_state()
    if st.dice_rolled:
        fails.append(('adopt_precondition', 'dice already rolled'))
        return
    hb0 = len(st.hunt_box)
    eyes = game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.EYE),
        (Side.SHADOW, 1, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    d_eye = st.dice[Side.SHADOW][0]
    d_chr = st.dice[Side.SHADOW][1]
    d_fp = st.dice[Side.FREE_PEOPLES][0]
    if not (d_eye.used and d_eye.current is ActionDieFace.EYE
            and d_eye in st.hunt_box):
        fails.append(('adopt_eye_rule',
                      'rolled Eye not used+hunt_box (parity with '
                      'roll_action_dice broken)'))
    if len(st.hunt_box) != hb0 + 1 or eyes != [(Side.SHADOW, 0)]:
        fails.append(('adopt_eye_return', 'eyes=%r hb=%d->%d'
                      % (eyes, hb0, len(st.hunt_box))))
    if d_chr.used or d_chr.current is not ActionDieFace.CHARACTER:
        fails.append(('adopt_plain_face', 'SH die 1 wrong'))
    if d_fp.current is not ActionDieFace.CHARACTER:
        fails.append(('adopt_fp_face', 'FP die 0 wrong'))
    if not st.dice_rolled:
        fails.append(('adopt_marks_rolled', 'dice_rolled not set'))
    # idempotent re-scan: a second pass must change NOTHING
    eyes2 = game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.CHARACTER),   # already set
        (Side.SHADOW, 1, ActionDieFace.EYE),         # already set
    ])
    if eyes2 or d_eye.current is not ActionDieFace.EYE \
            or d_chr.current is not ActionDieFace.CHARACTER \
            or len(st.hunt_box) != hb0 + 1:
        fails.append(('adopt_idempotent', 're-scan mutated state'))
    raised = False
    try:
        game_logic.adopt_rolled_faces(
            st, [(Side.SHADOW, 99, ActionDieFace.EYE)])
    except RuntimeError:
        raised = True
    if not raised:
        fails.append(('adopt_range_check', 'index 99 accepted'))


def test_pick_funnel_stream_parity(fails):
    """MEASURES: the funnel's default body consumes the IDENTICAL RNG
    stream as the inline random.choice it replaced (seeded twins) —
    the byte-for-byte contract that keeps gate-1 parity green."""
    cands = ['a', 'b', 'c', 'd', 'e']
    random.seed(4242)
    want = [random.choice(cands) for _ in range(20)]
    random.seed(4242)
    got = [game_logic.pick_random_companion(cands, 'parity probe')
           for _ in range(20)]
    if want != got:
        fails.append(('funnel_stream_parity', '%r != %r'
                      % (got[:5], want[:5])))
    random.seed(977)
    want = [random.choice(cands) for _ in range(20)]
    random.seed(977)
    got = [game_logic.pick_random_card(cands, 'parity probe')
           for _ in range(20)]
    if want != got:
        fails.append(('card_funnel_stream_parity', '%r != %r'
                      % (got[:5], want[:5])))


def test_pick_funnel_site_patchable(fails):
    """MEASURES: patching game_logic.pick_random_companion redirects a
    REAL engine pick site (sacrifice_random_companion) — the product's
    own victim, not a proxy.  Sensitivity: the patch names the LAST
    candidate; the unpatched default with this seed names a different
    one (proven in-test), so agreement-by-luck is excluded."""
    st = fresh_state()
    comps = [c for c in st.fellowship.companions if c.name != 'Gollum']
    if len(comps) < 3:
        fails.append(('patch_precondition', 'not enough companions'))
        return
    target = comps[-1].name
    # non-vacuity control: with seed 7 the DEFAULT pick differs from
    # the patched target (otherwise the test cannot distinguish them)
    random.seed(7)
    if random.choice(comps).name == target:
        target = comps[0].name  # pick the other end instead
        random.seed(7)
        if random.choice(comps).name == target:
            fails.append(('patch_control', 'cannot build a control'))
            return
    orig = game_logic.pick_random_companion
    game_logic.pick_random_companion = \
        lambda cands, ctx='': next(c for c in cands
                                   if c.name == target)
    try:
        random.seed(7)
        res = game_logic.sacrifice_random_companion(st)
    finally:
        game_logic.pick_random_companion = orig
    if not res.get('success'):
        fails.append(('patch_site_ran', 'sacrifice failed: %r' % res))
        return
    if res.get('victim') != target:
        fails.append(('patch_site_redirects',
                      'victim=%r, patched target=%r'
                      % (res.get('victim'), target)))


def test_zero_effect_play_is_legal_and_completes(fails):
    """USER RULING 1.230 (2026-08-28, verbatim: "zero effect plays are
    allowed, always. even in self play"; context: Bilbo's Song refused
    at corruption 0 as 'not in the engine menu').  MEASURES on the
    product path: with corruption 0 the FP menu OFFERS Bilbo's Song on
    a Character die (no no-op prune), and apply_action COMPLETES the
    play as a zero-effect play - card out of the hand into the FP
    character discard, the die spent, corruption still 0, no refund.
    SENSITIVITY: at corruption 1 the same play heals to 0 (the effect
    path is intact)."""
    from game_state import ActionDieFace
    cid = "FREE_PEOPLES_Character_Bilbo's_Song"
    for corr in (0, 1):
        st = fresh_state()
        for lst in (st.fp_char_deck, st.fp_strat_deck):
            if cid in lst:
                lst.remove(cid)
        st.hands[Side.FREE_PEOPLES].hand[:] = [cid]
        st.fellowship.corruption = corr
        game_logic.adopt_rolled_faces(st, [
            (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
            (Side.SHADOW, 0, ActionDieFace.ARMY)])
        menu = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        plays = [a for a in menu if a.action_type == 'play_event'
                 and a.params.get('card_id') == cid]
        if not plays:
            fails.append(('zero_effect_not_offered', corr))
            continue
        post = ai.apply_action(st, plays[0])
        hand = post.hands[Side.FREE_PEOPLES].hand
        disc = (getattr(post, 'discard_face_up', None) or {}).get(
            'fp_character', [])
        if cid in hand or cid not in disc:
            fails.append(('zero_effect_not_completed',
                          (corr, list(hand), list(disc))))
        if not post.dice[Side.FREE_PEOPLES][0].used:
            fails.append(('zero_effect_die_refunded', corr))
        want = 0
        if post.fellowship.corruption != want:
            fails.append(('zero_effect_corruption', (corr,
                                                    post.fellowship.corruption)))


def _optioned_fp_strategy_card(st):
    """(cid, card_name) of an FP strategy card whose option spec yields
    >= 2 choices on `st` (Riders of Theoden / Eomer: a Rohan region
    choice), or (None, None)."""
    import cards
    from card_options import get_card_options
    for cid in list(st.fp_strat_deck):
        c = cards.get_card(cid)
        if c is None or not c.card_name:
            continue
        try:
            if len(get_card_options(st, Side.FREE_PEOPLES,
                                    c.card_name)) >= 2:
                return cid, c.card_name
        except Exception:
            continue
    return None, None


def test_bl150_deferred_play_draw_waits_for_the_option(fails):
    """BL-150 E1 (USER manual game 2026-08-28): Gandalf the Grey's
    same-deck redraw fired while an OPTIONED card still sat in the hand
    (the play had deferred into pending_card_option), pushing a 6-card
    hand to 7 and opening a hand-limit discard pending that went stale
    once the option resolved - the live matcher then closed it by
    discarding hand[0], Challenge of the King.  CONTRACT: for a
    deferred play NO draw is granted and NO discard pending opens while
    the option is open; once the option resolves the card has left the
    hand and the redraw is granted (hand back to 6, no pending).
    E2 rides along: a stale pending re-derives instead of persisting."""
    from game_state import ActionDieFace
    st = fresh_state()
    cid, name = _optioned_fp_strategy_card(st)
    if cid is None:
        fails.append(('bl150_control', 'NOT-MEASURED: no optioned FP '
                      'strategy card yields >=2 options on the fresh state'))
        return
    if getattr(st.fellowship.guide, 'name', '') != 'Gandalf the Grey':
        fails.append(('bl150_control', 'NOT-MEASURED: guide'))
        return
    st.fp_strat_deck.remove(cid)
    fillers = [c for c in st.fp_char_deck[:5]]
    for c in fillers:
        st.fp_char_deck.remove(c)
    st.hands[Side.FREE_PEOPLES].hand[:] = [cid] + fillers        # 6
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.WILL_OF_WEST),
        (Side.SHADOW, 0, ActionDieFace.ARMY)])
    plays = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
             if a.action_type == 'play_event'
             and a.params.get('card_id') == cid and a.die_idx == 0]
    if not plays:
        fails.append(('bl150_control', 'NOT-MEASURED: %s not offered on '
                      'the W die' % name))
        return
    mid = ai.apply_action(st, plays[0])
    pco = getattr(mid, 'pending_card_option', None)
    if not pco:
        fails.append(('bl150_control', 'NOT-MEASURED: play did not defer'))
        return
    hand_mid = mid.hands[Side.FREE_PEOPLES].hand
    if len(hand_mid) != 6 or cid not in hand_mid:
        fails.append(('bl150_draw_granted_while_deferred',
                      (len(hand_mid), cid in hand_mid)))
    if getattr(mid, 'pending_discard', None) is not None:
        fails.append(('bl150_phantom_discard_pending', mid.pending_discard))
    opts = [a for a in ai.get_legal_actions(mid, Side.FREE_PEOPLES)
            if a.action_type == 'card_option']
    if not opts:
        fails.append(('bl150_control', 'NOT-MEASURED: no card_option rows'))
        return
    post = ai.apply_action(mid, opts[0])
    hand_post = post.hands[Side.FREE_PEOPLES].hand
    owed = getattr(post, '_owed_draws', None) or {}
    drawn = len(hand_post) - 5
    if cid in hand_post:
        fails.append(('bl150_card_not_played', True))
    if drawn != 1 and not owed:
        fails.append(('bl150_redraw_not_granted_after_resolution',
                      (len(hand_post), dict(owed))))
    if getattr(post, 'pending_discard', None) is not None:
        fails.append(('bl150_discard_pending_after_resolution',
                      post.pending_discard))
    # E2: a stale pending re-derives instead of persisting
    st2 = fresh_state()
    extra = list(st2.fp_char_deck[:7])
    for c in extra:
        st2.fp_char_deck.remove(c)
    st2.hands[Side.FREE_PEOPLES].hand[:] = extra                  # 7
    if not game_logic.open_hand_limit_discard(st2) \
            or st2.pending_discard is None:
        fails.append(('bl150_e2_control', 'pending did not open at 7'))
        return
    st2.hands[Side.FREE_PEOPLES].hand.pop()                      # 6 again
    if game_logic.open_hand_limit_discard(st2) \
            or st2.pending_discard is not None:
        fails.append(('bl150_e2_stale_pending_kept', st2.pending_discard))


def test_enumeration_zero_randomness(fails):
    """MEASURES: the play_event dry-run (_event_card_has_effect)
    consumes ZERO global randomness and never reaches the companion
    funnel.  Non-zero control: at least one card in hand is actually
    dry-run (n stated), else this is reported as NOT-MEASURED."""
    st = fresh_state()
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for d in ('character', 'strategy'):
            try:
                game_logic.draw_event_card(st, s, d)
            except Exception:
                pass
    hand_fp = list(st.hands[Side.FREE_PEOPLES].hand)
    hand_sh = list(st.hands[Side.SHADOW].hand)
    n = len(hand_fp) + len(hand_sh)
    if n == 0:
        fails.append(('enum_control', 'NOT-MEASURED: no cards drawn'))
        return
    funnel_hits = []
    orig = game_logic.pick_random_companion
    game_logic.pick_random_companion = \
        lambda cands, ctx='': funnel_hits.append(ctx) or cands[0]
    try:
        before = random.getstate()
        for cid in hand_fp:
            ai._event_card_has_effect(st, Side.FREE_PEOPLES, cid)
        for cid in hand_sh:
            ai._event_card_has_effect(st, Side.SHADOW, cid)
        after = random.getstate()
    finally:
        game_logic.pick_random_companion = orig
    if before != after:
        fails.append(('enum_zero_randomness',
                      'global RNG advanced across %d dry-runs' % n))
    if funnel_hits:
        fails.append(('enum_funnel_leak',
                      'dry-run reached the live companion funnel: %r'
                      % funnel_hits))
    print('  [enum] n=%d cards dry-run (non-zero control)' % n)


def test_pick_die_ring_waiver(fails):
    """MEASURES the WAVE-F engine gap fix: game_logic._pick_die honors
    a live ring waiver exactly as ai._consume_die_for_face does — a
    wrong-faced die is rescued AND the waiver is spent; a right-faced
    die under a live waiver does NOT spend it (spend-only-on-rescue);
    without a waiver the wrong face still refuses (specificity)."""
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.CHARACTER),
        (Side.SHADOW, 1, ActionDieFace.MUSTER),
    ])
    # no waiver: wrong face refuses
    raised = False
    try:
        game_logic._pick_die(st, Side.SHADOW,
                             (ActionDieFace.MUSTER,), 0)
    except RuntimeError:
        raised = True
    if not raised:
        fails.append(('waiver_specificity', 'wrong face accepted '
                                            'without a waiver'))
        return
    # live waiver: the wrong-faced die is rescued and the waiver spent
    setattr(st, game_logic._RING_WAIVER_ATTR[Side.SHADOW], True)
    d = game_logic._pick_die(st, Side.SHADOW,
                             (ActionDieFace.MUSTER,), 0)
    if not d.used:
        fails.append(('waiver_rescue_consumes', 'die not used'))
    if game_logic.ring_waiver_active(st, Side.SHADOW):
        fails.append(('waiver_spent', 'waiver survived the rescue'))
    # live waiver + RIGHT face: no spend (ring_waiver_spend contract)
    setattr(st, game_logic._RING_WAIVER_ATTR[Side.SHADOW], True)
    game_logic._pick_die(st, Side.SHADOW, (ActionDieFace.MUSTER,), 1)
    if not game_logic.ring_waiver_active(st, Side.SHADOW):
        fails.append(('waiver_kept', 'waiver spent though the die '
                                     'needed no rescue'))


def test_describe_pending(fails):
    """MEASURES: describe_pending packages (kind, side, actions) from
    the ONE cascade, agreeing with _pending_decision_side, and returns
    None when nothing is pending.  Also guards the refactor:
    _pending_decision_side answers are unchanged on crafted pendings."""
    st = fresh_state()
    if ai.describe_pending(st) is not None:
        k = ai._pending_decision_kind_side(st)[0]
        fails.append(('describe_none', 'fresh state pending kind=%r' % k))
    st._eye_allocation_phase = True
    try:
        dp = ai.describe_pending(st)
        if dp is None or dp['kind'] != 'eye' \
                or dp['side'] is not Side.SHADOW:
            fails.append(('describe_eye', repr(dp and dp['kind'])))
        elif not dp['actions']:
            fails.append(('describe_eye_menu', 'empty eye menu'))
        elif dp['side'] is not ai._pending_decision_side(st):
            fails.append(('describe_eye_side_agree', 'cascade drift'))
    finally:
        st._eye_allocation_phase = False
    st._declare_phase = True
    try:
        dp = ai.describe_pending(st)
        if dp is None or dp['kind'] != 'declare' \
                or dp['side'] is not Side.FREE_PEOPLES:
            fails.append(('describe_declare', repr(dp and dp['kind'])))
        elif dp['side'] is not ai._pending_decision_side(st):
            fails.append(('describe_declare_side_agree', 'drift'))
    finally:
        st._declare_phase = False
    # discard pending: the dict shape get_legal_actions consumes
    st2 = fresh_state()
    for _ in range(8):
        try:
            game_logic.draw_event_card(st2, Side.SHADOW, 'character')
        except Exception:
            break
    if len(st2.hands[Side.SHADOW].hand) > 6:
        opener = getattr(game_logic, 'open_hand_limit_discard', None)
        if opener is not None:
            opener(st2)   # scans both sides itself
        if getattr(st2, 'pending_discard', None) is not None:
            dp = ai.describe_pending(st2)
            if dp is None or dp['kind'] != 'discard':
                fails.append(('describe_discard',
                              repr(dp and dp['kind'])))
            elif dp['side'] is not ai._pending_decision_side(st2):
                fails.append(('describe_discard_side_agree', 'drift'))
        else:
            print('  [describe] discard leg NOT-MEASURED '
                  '(no opener path)')
    else:
        print('  [describe] discard leg NOT-MEASURED (deck ran dry)')


def test_menu_coexists_with_opponent_discard(fails):
    """MEASURES (B6, tester soak crash, fixed 2026-08-24): the
    turn-start coexistence of one side's hand-limit discard with the
    OTHER side's declare phase (order D2-D4) must not crash the menu.
    get_legal_actions(FP) returns FP's own declare menu (never
    discard_card actions, never a raise); get_legal_actions(SH)
    returns the discard menu; describe_pending routes the discard to
    its owner first.  SENSITIVITY: the owner's menu really is the
    discard (all discard_card, owner's tag)."""
    st = fresh_state()
    for _ in range(7):
        game_logic.draw_event_card(st, Side.SHADOW, 'character')
    game_logic.open_hand_limit_discard(st)
    if getattr(st, 'pending_discard', None) is None:
        fails.append(('b6_precondition', 'no discard pending'))
        return
    game_logic.open_declare_phase(st)
    try:
        fp = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    except Exception as e:
        fails.append(('b6_fp_menu_raises', repr(e)[:160]))
        return
    if not fp:
        fails.append(('b6_fp_menu_empty', ''))
    if any(a.action_type == 'discard_card' for a in fp):
        fails.append(('b6_fp_handed_discard',
                      [a.action_type for a in fp][:5]))
    sh = ai.get_legal_actions(st, Side.SHADOW)
    if not sh or any(a.action_type != 'discard_card' for a in sh) \
            or any(a.params.get('side') != 'sh' for a in sh):
        fails.append(('b6_owner_menu',
                      [(a.action_type, a.params.get('side'))
                       for a in sh][:5]))
    dp = ai.describe_pending(st)
    if dp is None or dp['kind'] != 'discard' \
            or dp['side'] is not Side.SHADOW:
        fails.append(('b6_routing', repr(dp and dp['kind'])))


def test_day_without_dawn_repoints_stored_die_index(fails):
    """MEASURES (B1, 2026-08-24): The Day Without Dawn removes unused
    FP Will-of-the-West dice from the pool (correct: results are lost;
    the pool rebuilds next turn), which SHIFTS every later FP die's
    index - and pending_army_move_die / pending_muster_die store an
    integer index across applies.  The handler must re-point them.
    SPECIFICITY: a Shadow-owned stored index is untouched."""
    from game_state import ActionDieFace
    import event_effects
    st = fresh_state()
    fp = st.dice[Side.FREE_PEOPLES]
    if len(fp) < 4:
        fails.append(('dwd_precondition', len(fp)))
        return
    fp[0].current = ActionDieFace.WILL_OF_WEST
    fp[0].used = False
    fp[1].current = ActionDieFace.CHARACTER
    fp[2].current = ActionDieFace.ARMY
    fp[2].used = True                      # the spent first-leg die
    fp[3].current = ActionDieFace.MUSTER
    st.pending_army_move_die = 2
    st.pending_army_move_side = Side.FREE_PEOPLES
    st.pending_muster_die = 1
    st.pending_muster_side = Side.SHADOW   # NOT ours - must not move
    h = event_effects.CARD_EFFECT_HANDLERS.get('the day without dawn')
    if h is None:
        fails.append(('dwd_handler_missing', ''))
        return
    res = h(st, None, {})
    if 'Discarded 1' not in ' '.join(res.get('effects', [])):
        fails.append(('dwd_effect', res))
    if len(fp) != 3:
        fails.append(('dwd_pool', len(fp)))
    if st.pending_army_move_die != 1:
        fails.append(('dwd_army_idx_repointed',
                      st.pending_army_move_die))
    elif fp[st.pending_army_move_die].current \
            is not ActionDieFace.ARMY:
        fails.append(('dwd_army_idx_wrong_die',
                      fp[st.pending_army_move_die].current))
    if st.pending_muster_die != 1:
        fails.append(('dwd_shadow_idx_untouched',
                      st.pending_muster_die))


def main():
    tests = [
        test_grant_draw_success_no_residue,
        test_grant_draw_deferral_stamps_debt,
        test_grant_draw_deck_empty_no_debt,
        test_grant_retry_settles_residue,
        test_settle_owed_draw,
        test_adopt_rolled_faces,
        test_pick_funnel_stream_parity,
        test_pick_funnel_site_patchable,
        test_enumeration_zero_randomness,
        test_zero_effect_play_is_legal_and_completes,
        test_bl150_deferred_play_draw_waits_for_the_option,
        test_pick_die_ring_waiver,
        test_describe_pending,
        test_menu_coexists_with_opponent_discard,
        test_day_without_dawn_repoints_stored_die_index,
    ]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
