"""Audit 2026-08-29 - DR fix-wave pins (live driver / adopt seams).

Every pin builds the OFFLINE LiveDriver rig from test_live_driver
(Rig / fresh_state: real client grammar in, engine state judged) and
went RED against the unfixed code before its fix landed:

  DR-1  eye DRAG before the AI-FP's declare step keeps the phase at
        'declare' (no decline committed for the AI); the AI's step
        still finds its menu and the Mordor entry check runs
  DR-2  a live human pending is never force-committed by absence
        (no 'pending_silent_noop'); the new window is HELD and an
        explicit wait state is published
  DR-3  the chat-attack lane spends the die the client STAGED
        (BL-108 rebind), not the face-deduped menu row's idx
  DR-4  the physical reveal of a chat-committed combat card never
        re-inserts the played cid into the engine hand
  DR-5  a census exception in reconcile_board_quiet is loud and keeps
        the reconcile due (never reads as 'no divergence')
  DR-6  a depleted human deck (rulebook 1763) does not stick the
        turn start at 'draws'
  DR-7  adopting a companion as DEAD books the engine's leaves-play
        cleanup (companion-linked table cards re-derived)
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ai                                        # noqa: E402
import game_logic                                # noqa: E402
from game_state import Side, ActionDieFace       # noqa: E402
from test_live_driver import Rig, fresh_state    # noqa: E402


def _shadow_attack_state():
    """Turn-1 board with two ARMY dice rolled for the Shadow and every
    Shadow nation at war: the engine offers army_attack rows."""
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.ARMY),
        (Side.SHADOW, 2, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    for nat in list(st.political.boxes):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st.political.boxes[nat] = 3
            st.political.active[nat] = True
    return st


def _first_attack(st):
    menu = ai.get_legal_actions(st, Side.SHADOW)
    atks = [a for a in menu if a.action_type == 'army_attack']
    assert atks, 'precondition: no army_attack on the menu'
    return atks[0]


def _ai_apply(rig, drv, action, label):
    """The server's own commit shape for an AI decision (clone-applied
    inside advance(), snapshotted, committed)."""
    rig.rand.begin_attempt()
    with rig.rand.advance():
        post = ai.apply_action(drv.es, action)
    drv._snapshot([], label)
    drv.es = post
    drv.hands.es = post
    rig.rand.step_commit()


# ---------------------------------------------------------------- DR-1
def test_dr1_eye_drag_keeps_ai_declare_open():
    st = fresh_state()
    st.fellowship.region = st.board.region('minas morgul')
    st.fellowship.hidden = False          # revealed ON the gate
    st.fellowship.progress = 0
    rig = Rig(st, Side.FREE_PEOPLES, phase='draws', seed=3)   # AI = FP
    drv = rig.drv
    drv._ai_turn_start_draws()            # the hand-off after the draws
    assert drv.phase == 'declare', drv.phase
    dp = ai.describe_pending(drv.es)
    assert dp is not None and dp['kind'] == 'declare' \
        and dp['side'] is Side.FREE_PEOPLES, dp and (dp['kind'], dp['side'])
    n_commits = len(drv.commits)
    hb0 = len(drv.es.hunt_box)
    rig.feed('$Wey moves Shadow Action Die (A) from SA Dice to The Hunt '
             'For The Ring.')
    assert len(drv.es.hunt_box) == hb0 + 1, 'the eye must land engine-side'
    labels = [c[2] for c in drv.commits[n_commits:]]
    assert 'declare-decline' not in labels, \
        'DR-1 RED: eye drag committed the AI declaration decline %r' % labels
    assert drv.phase == 'declare', \
        'DR-1 RED: phase moved to %r before the AI declare step' % drv.phase
    assert any(k == 'eye_before_ai_declare' for k, _ in drv.findings), \
        [k for k, _ in drv.findings]
    # live_server._ai_declare: guard 'phase != declare' must not abandon
    dp = drv.ai_action_dp()
    assert dp is not None and dp['kind'] == 'declare' and dp['actions'], \
        'DR-1 RED: the AI declare step lost its menu (%r)' % (dp and dp['kind'],)
    _ai_apply(rig, drv, dp['actions'][0], 'ai:declare')
    items = drv.after_ai_declare()
    assert drv.phase == 'eyes', drv.phase
    assert drv.es.fellowship.mordor_entered is True, \
        'revealed on Minas Morgul enters Mordor at the close of the ' \
        'declaration opportunity (FINAL ruling 2026-08-16)'
    assert any(i[0] == 'mordor_entry' for i in items), items
    assert ai.describe_pending(drv.es) is None
    # SPECIFICITY: with the AI's declaration already settled this turn
    # the same drag moves the phase on (the _on_eye lane's contract)
    st2 = fresh_state()
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='draws', seed=3)
    drv2 = rig2.drv
    drv2._ai_turn_start_draws()
    drv2._declare_done_turn = drv2.es.turn
    rig2.feed('$Wey moves Shadow Action Die (A) from SA Dice to The Hunt '
              'For The Ring.')
    assert drv2.phase == 'eyes', drv2.phase
    assert not getattr(drv2.es, '_declare_phase', False)


# ---------------------------------------------------------------- DR-2
def test_dr2_live_human_pending_never_silently_nooped():
    st = _shadow_attack_state()
    atk = _first_attack(st)
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions', seed=5)  # human=SH
    drv = rig.drv
    rig.feed('$Wey moves Shadow Action Die (A) from SA Dice to Current '
             'Action.')
    rig.feed('<Wey> %s from %s' % (atk.params['to'], atk.params['from']))
    pc = drv.es.pending_combat or {}
    assert pc.get('phase') == 'attacker_combat_card', pc.get('phase')
    rig.feed('$Wey kills Shadow Action Die (A) in Current Action.')
    n_find = len(drv.findings)
    out = rig.feed(
        '$Wey moves Shadow Action Die (C) from SA Dice to Current Action.',
        '$Wey moves Nazgul from Minas Morgul to Osgiliath.',
        '$Wey kills Shadow Action Die (C) in Current Action.')
    new = drv.findings[n_find:]
    assert not any(k == 'pending_silent_noop' for k, _ in new), \
        'DR-2 RED: human decision force-committed by absence: %r' % (new,)
    dp = ai.describe_pending(drv.es)
    assert dp is not None and dp['side'] is Side.SHADOW \
        and dp['kind'] == 'combat:attacker_combat_card', \
        'DR-2 RED: pending %r' % (dp and (dp['kind'], dp['side']),)
    assert (drv.es.pending_combat or {}).get('phase') \
        == 'attacker_combat_card'
    assert drv.window_stats['hold'] == 1, dict(drv.window_stats)
    assert not any(i[0] == 'ai_decide' for i in out), out
    # the explicit wait state: the human is told which decision is
    # owed (chat reminder) and the panel alarm names the held decision
    chats = [i for i in out if i[0] == 'chat']
    assert chats and 'combat card' in str(chats[0][2]).lower(), out
    assert any('HUMAN DECISION HELD' in a for a in drv.alarms), drv.alarms


# ---------------------------------------------------------------- DR-3
def test_dr3_chat_attack_spends_the_staged_die():
    st = _shadow_attack_state()
    atk = _first_attack(st)
    assert atk.die_idx == 0, atk.die_idx      # face-dedup: ONE row, idx 0
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions', seed=5)  # human=SH
    drv = rig.drv
    for pid, i in (('100', 0), ('101', 1), ('102', 2)):
        drv.view.bind_die(pid, Side.SHADOW, i)
    rig.feed('$Wey moves Shadow Action Die (A) from SA Dice to Current '
             'Action.')
    w = drv.seg.window
    w.records[0]['meta'] = {'pids': ['101']}          # the SECOND A die
    assert drv._staged_die_idx(w) == (Side.SHADOW, 1)
    rig.feed('<Wey> %s from %s' % (atk.params['to'], atk.params['from']))
    assert (drv.es.pending_combat or {}).get('phase') \
        == 'attacker_combat_card'
    dice = drv.es.dice[Side.SHADOW]
    spent = [i for i, d in enumerate(dice) if d.used]
    assert spent == [1], \
        'DR-3 RED: client staged pid 101 (idx 1) but the engine spent ' \
        'idx %r' % (spent,)
    assert any(k == 'die_rebound_to_staged' for k, _ in drv.findings), \
        [k for k, _ in drv.findings]
    assert not drv.alarms, drv.alarms
    m0 = drv.window_stats['matched']
    rig.feed('$Wey kills Shadow Action Die (A) in Current Action.')
    assert drv.window_stats['matched'] == m0 + 1, dict(drv.window_stats)


# ---------------------------------------------------------------- DR-4
def test_dr4_physical_reveal_never_resurrects_played_card():
    from wotr_bridge.live import gesture_match as gm
    title = 'The Shadow is Moving/Swarm of Bats'
    cid = gm.title_to_cid(title)
    assert cid is not None, 'precondition: title_to_cid'
    deck = game_logic.card_deck_of(cid)
    assert deck in ('character', 'strategy'), deck
    st = fresh_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SHADOW
    drv = rig.drv
    ph = drv.hands._mint(deck)               # one UNKNOWN same-deck card
    hand = drv.es.hands[Side.SHADOW].hand
    hand[:] = [ph]
    # the chat commit already applied the card: it left the hand for
    # the face-up discard (ai._discard_played_combat_card)
    drv.es.discard_face_up['sh_%s' % deck].append(cid)
    drv._cc_reveal_owed = {str(cid): 'owed'}
    rig.feed('<auto> moves %s from Shadow Hand (chosen) to Current SA '
             'Card.' % title)
    assert drv._cc_reveal_owed.get(str(cid)) == 'claimed', \
        drv._cc_reveal_owed
    assert any(k == 'cc_reveal_physical' for k, _ in drv.findings)
    hand = drv.es.hands[Side.SHADOW].hand
    assert cid not in hand, \
        'DR-4 RED: the played card was re-inserted into the engine ' \
        'hand: %r' % (hand,)
    assert hand == [ph], hand
    assert drv.es.discard_face_up['sh_%s' % deck].count(cid) == 1


# ---------------------------------------------------------------- DR-5
def test_dr5_census_exception_is_loud_and_keeps_reconcile_due():
    from wotr_bridge.live import client_board
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    drv.view.refresh_dump = lambda: True
    drv._political_recon = lambda: []
    drv._guide_box_recon = lambda: []
    drv._reconcile_due = True
    drv._recon_prev_sig = ('units somewhere: 1 -> 0',)
    orig = client_board.diff_engine_vs_dump

    def _boom(*a, **k):
        raise RuntimeError('witness broken')
    client_board.diff_engine_vs_dump = _boom
    try:
        items = drv.reconcile_board_quiet()
    finally:
        client_board.diff_engine_vs_dump = orig
    assert drv._reconcile_due is True, \
        'DR-5 RED: a census exception read as "no divergence" ' \
        '(_reconcile_due cleared)'
    assert drv._recon_prev_sig == ('units somewhere: 1 -> 0',), \
        drv._recon_prev_sig
    assert any('CENSUS' in a.upper() and 'witness broken' in a
               for a in drv.alarms), drv.alarms
    assert any(k == 'reconcile_census_failed' for k, _ in drv.findings), \
        drv.findings[-3:]
    assert any(i[0] == 'note' and 'census' in str(i[1]).lower()
               for i in items), items
    # SPECIFICITY: a clean census still clears the flag
    client_board.diff_engine_vs_dump = lambda *a, **k: []
    try:
        drv.reconcile_board_quiet()
    finally:
        client_board.diff_engine_vs_dump = orig
    assert drv._reconcile_due is False


# ---------------------------------------------------------------- DR-6
def test_dr6_depleted_deck_does_not_stick_the_turn_start():
    # (a) one human deck depleted: the single draw hands off
    st = fresh_state()
    del st.sh_strat_deck[:]
    rig = Rig(st, Side.FREE_PEOPLES, phase='draws', seed=3)  # human = SH
    drv = rig.drv
    items = rig.feed('$Wey draws Shadow Character Card.')
    assert drv.phase == 'declare', \
        'DR-6 RED: phase %r stuck at the turn start with the strategy ' \
        'deck depleted' % drv.phase
    assert any(i[0] == 'ai_draws' for i in items), items
    assert any(k == 'draw_skipped_deck_depleted' for k, _ in drv.findings), \
        [k for k, _ in drv.findings]
    assert any(i[0] == 'note' and 'depleted' in str(i[1]).lower()
               for i in items), items
    # SPECIFICITY: with both decks stocked one draw does not hand off
    st2 = fresh_state()
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='draws', seed=3)
    items2 = rig2.feed('$Wey draws Shadow Character Card.')
    assert rig2.drv.phase == 'draws', rig2.drv.phase
    assert not any(i[0] == 'ai_draws' for i in items2), items2
    # (b) BOTH human decks depleted: no draw line will ever come - the
    # rollover itself hands off (after the AI's own recover item)
    st3 = fresh_state()
    game_logic.adopt_rolled_faces(st3, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    del st3.fp_char_deck[:]
    del st3.fp_strat_deck[:]
    rig3 = Rig(st3, Side.SHADOW, phase='actions')          # human = FP
    drv3 = rig3.drv
    drv3.last_actor = Side.FREE_PEOPLES
    drv3.turn_draws = {(Side.FREE_PEOPLES, 'character'): 1,
                       (Side.FREE_PEOPLES, 'strategy'): 1}
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for d in drv3.es.dice[s]:
            d.used = True
    turn0 = drv3.es.turn
    items3 = rig3.feed('$Wey moves group (Free Action Die (C), Free '
                       'Action Die (P), [2]Free Action Die (A)) from '
                       'Current Action to FP Dice.')
    assert drv3.es.turn == turn0 + 1, (drv3.es.turn, drv3.phase)
    kinds3 = [i[0] for i in items3]
    assert 'ai_draws' in kinds3, \
        'DR-6 RED: both FP decks depleted, rollover left the phase at ' \
        '%r with items %r' % (drv3.phase, kinds3)
    assert kinds3.index('recover_ai') < kinds3.index('ai_draws'), kinds3
    assert drv3.phase == 'declare', drv3.phase


# ---------------------------------------------------------------- DR-7
def test_dr7_dead_companion_adoption_rederives_table_cards():
    import cards
    from wotr_bridge.live import adopt
    from persistent_effects import create_persistent_card
    cid = next((c for c in cards.get_cards()
                if 'Character' in c and 'Wizard' in c and 'Staff' in c),
               None)
    assert cid is not None, 'precondition: Wizard\'s Staff cid'
    st = fresh_state()
    assert any(c.name == 'Gandalf the Grey'
               for c in st.fellowship.companions)
    st.persistent_cards.add_card(
        create_persistent_card(cid, "Wizard's Staff", Side.FREE_PEOPLES))
    assert st.persistent_cards.has_card("Wizard's Staff")
    rep = adopt.adopt_companions(st, {'240': 'FP Dead'})
    assert not any(c.name == 'Gandalf the Grey'
                   for c in st.fellowship.companions), rep
    assert not st.persistent_cards.has_card("Wizard's Staff"), \
        'DR-7 RED: Wizard\'s Staff still on the table with Gandalf ' \
        'adopted into the dead pile: %r' % (rep,)
    assert cid in st.discard_face_up['fp_character'], \
        st.discard_face_up['fp_character']
    assert any('Staff' in r or cid in r for r in rep), rep
    # SPECIFICITY: a companion the card does not depend on leaves the
    # table card alone
    st2 = fresh_state()
    st2.persistent_cards.add_card(
        create_persistent_card(cid, "Wizard's Staff", Side.FREE_PEOPLES))
    rep2 = adopt.adopt_companions(st2, {'244': 'FP Dead'})     # Boromir
    assert not any(c.name == 'Boromir' for c in st2.fellowship.companions)
    assert st2.persistent_cards.has_card("Wizard's Staff"), rep2


# ---------------------------------------------------------------- DR-8
def test_dr8_human_ring_walk_records_no_face():
    """LA-1/LA-2 (ai.py, 2026-08-29): the elven_ring executor records
    the REQUESTED face and the menu widens a permitted die only to
    that face's categories.  The client's ring command is a bare
    marker walk (no face), so the human's row must carry
    params['face_unknown']: no face recorded, the waiver keeps the
    widen-all model (every ordinary category), and a widened action
    spends the waiver once.  SPECIFICITY: the AI's own ring play (the
    server's apply path) keeps the face the search chose."""
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 1, ActionDieFace.MUSTER),
    ])
    st.fellowship.hidden = False   # fellowship_hide: a character action
    rig = Rig(st, Side.SHADOW, phase='actions', seed=5)     # human = FP
    drv = rig.drv
    drv.last_actor = Side.SHADOW                           # FP to act
    rings0 = st.fp_elven_rings
    items = rig.feed('$Wey moves Nenya from Elven Ring 2 to Dark Ring 2.')
    es = drv.es
    assert es.fp_elven_rings == rings0 - 1, (rings0, es.fp_elven_rings)
    assert game_logic.ring_waiver_active(es, Side.FREE_PEOPLES)
    assert not any(i[0] == 'chat' for i in items), items
    assert getattr(es, 'fp_ring_waiver_face', None) is None, \
        'DR-8 RED: the human ring walk recorded an arbitrary menu face ' \
        '%r' % (es.fp_ring_waiver_face,)
    menu = ai.get_legal_actions(es, Side.FREE_PEOPLES)
    muster = sorted({a.action_type for a in menu if a.die_idx == 1})
    assert 'fellowship_hide' in muster and 'army_move' in muster, \
        'DR-8 RED: the human MUSTER die widened to %r only (widen-all ' \
        'offered character + army + muster + event)' % (muster,)
    # the widened CHARACTER action on the MUSTER die: the die pays and
    # the waiver is spent exactly once
    hide = next(a for a in menu
                if a.die_idx == 1 and a.action_type == 'fellowship_hide')
    assert drv._commit_chain([hide], [], 'test:widened-hide'), \
        drv.findings[-2:]
    es = drv.es
    assert es.dice[Side.FREE_PEOPLES][1].used
    assert es.fellowship.hidden is True
    assert not game_logic.ring_waiver_active(es, Side.FREE_PEOPLES)
    assert getattr(es, 'fp_ring_waiver_face', None) is None
    assert es.fp_elven_rings == rings0 - 1
    # SPECIFICITY: the AI-FP's own ring (server apply path) keeps the
    # face it chose
    st2 = fresh_state()
    game_logic.adopt_rolled_faces(st2, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='actions', seed=5)  # AI = FP
    drv2 = rig2.drv
    row = next(a for a in ai.get_legal_actions(drv2.es, Side.FREE_PEOPLES)
               if a.action_type == 'elven_ring'
               and str(a.params.get('new_face', '')).lower() == 'muster')
    _ai_apply(rig2, drv2, row, 'ai:ring')
    assert drv2.es.fp_ring_waiver_face == ActionDieFace.MUSTER, \
        drv2.es.fp_ring_waiver_face


def main():
    tests = [
        test_dr1_eye_drag_keeps_ai_declare_open,
        test_dr2_live_human_pending_never_silently_nooped,
        test_dr3_chat_attack_spends_the_staged_die,
        test_dr4_physical_reveal_never_resurrects_played_card,
        test_dr5_census_exception_is_loud_and_keeps_reconcile_due,
        test_dr6_depleted_deck_does_not_stick_the_turn_start,
        test_dr7_dead_companion_adoption_rederives_table_cards,
        test_dr8_human_ring_walk_records_no_face,
    ]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', default=None)
    args = ap.parse_args()
    if args.k:
        tests = [t for t in tests if args.k in t.__name__]
    fails = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append((t.__name__, '%s: %s' % (type(e).__name__, e)))
        print('  [%s] %s' % ('FAIL' if fails and fails[-1][0] == t.__name__
                             else 'PASS', t.__name__))
    print('=' * 60)
    print('TOTAL=%d FAIL=%d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
