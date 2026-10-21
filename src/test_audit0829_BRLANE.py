"""Audit 2026-08-29 - BRLANE fix-wave pins (the lanes between client lines
and engine actions: gesture windows, chat, undo, restore watch, the step
queue's human-pending ranking, the AI emission acknowledgement).

Every pin builds the OFFLINE rigs (test_live_driver.Rig / fresh_state,
test_live_server.LiveServer): real client grammar in, the engine judged.
Each went RED against the unfixed tree before its fix (BRLANE_FIX_REPORT
carries the RED and GREEN lines); the two regression pins that were
already GREEN (items 14-plain / 15) say so in their docstrings.

  BRLANE-1  an open human gesture window is not quiescent; the AI's own
            commit never arms the auto-reconcile while it is open
  BRLANE-2  a HELD window is retried when its blocking condition clears;
            a hold never ages out silently
  BRLANE-3  ai_declare parks LOUDLY behind the human's pending; the status
            line ranks the human's decision first; N parks -> self-repair ASK
  BRLANE-4  free-text chat that matches no grammar is never silent
  BRLANE-5  die-less human moves / kills are matched or refused, never dropped
  BRLANE-6  the client undo of an eye allocation unwinds the engine
  BRLANE-7  the human's undo of the AI's own line is RE-ASSERTED, not adopted
  BRLANE-8  restore-watch zone aliases (Fellowship Box == Fellowship Of The Ring)
  BRLANE-9  restore-watch expects a two-hop card back in the HAND
  BRLANE-10 'you cannot pass' prints both unused-die counts and the rule
  BRLANE-11 declare-in-place is unique; a pending ambiguity is voiced
  BRLANE-12 a region chatted over a card-play window answers the card's
            target decision after the die kill; the attack refusal names
            the faces that can attack
  BRLANE-13 an emitted AI line is an EXPECTED ECHO: the step queue waits,
            expiry is loud (finding + alarm + self-repair ASK)
  BRLANE-14 a human-deck draw demand is asked in chat
  BRLANE-15 the roll side comes from the sidecar's die family first
  EX-8      the human Shadow's two physical discards pay the table-card
            kill with the pair the human KILLED
  item 14   a card play commits by CARD IDENTITY; extra rows are explained
  item 15   a multi-companion drag maps to the group separation row
  item 16   a battle the human resolved by hand while owing a combat
            decision asks the self-repair question at once
"""
import os
import sys
import tempfile
import types

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ai                                        # noqa: E402
import game_logic                                # noqa: E402
import game_state as gs                          # noqa: E402
from game_state import Side, ActionDieFace as F  # noqa: E402
from board import sanitize                       # noqa: E402
from test_live_driver import (Rig, fresh_state, adopted_state,   # noqa: E402
                              _army_window_lines)

FP, SH = Side.FREE_PEOPLES, Side.SHADOW


def _faces(st, fp, sh):
    rolls = [(FP, i, f) for i, f in enumerate(fp)]
    rolls += [(SH, i, f) for i, f in enumerate(sh)]
    game_logic.adopt_rolled_faces(st, rolls)
    return st


def _chats(items):
    return [str(i[-1]) for i in items if i and i[0] == 'chat']


def _kinds(drv):
    return [k for k, _ in drv.findings]


def _write_meta(path, recs):
    import json
    with open(path, 'a', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')


def _undo_lines(items):
    return [i for i in items if i and i[0] == 'line'
            and 'Controls.819' in str(i[1])]


# ------------------------------------------------------------- BRLANE-1
def test_brlane1_open_window_parks_reconcile_and_is_not_quiescent():
    """A human die staged + pieces dragged (no kill yet) is an OPEN
    gesture: the driver publishes it (gesture_open), the comparator's
    quiescent() is False, and the AI's own commit (live_server sets
    `_reconcile_due = True`) never arms the auto-reconcile while the
    window is open - the arm is parked and promoted at the kill."""
    st = adopted_state()
    a, lines = _army_window_lines(st)
    assert a is not None, 'precondition: a simple army move'
    rig = Rig(st, FP, phase='actions')                 # human = SHADOW
    drv = rig.drv
    rig.feed(*lines[:2])                               # stage + drag
    assert drv.seg.window is not None
    assert hasattr(drv, 'gesture_open') and drv.gesture_open() is True, \
        'BRLANE-1 RED: the driver publishes no open-gesture state'
    assert hasattr(drv, 'quiescent') and drv.quiescent() is False, \
        'BRLANE-1 RED (BRCMP predicate): quiescent() missing or True ' \
        'with a human window open'
    drv._reconcile_due = True                          # the AI's commit
    assert drv._reconcile_due is False, \
        'BRLANE-1 RED: the auto-reconcile armed while the human window ' \
        'is open'
    # the comparator side: two quiet reads over a diverging dump must
    # adopt NOTHING while the gesture is open (BRCMP's gate)
    from wotr_bridge.live import client_board
    calls = []
    drv._adopt_board_now = lambda why: calls.append(why) or []
    drv.view.refresh_dump = lambda: True
    drv._political_recon = lambda: []
    drv._guide_box_recon = lambda: []
    orig = client_board.diff_engine_vs_dump
    client_board.diff_engine_vs_dump = lambda *a, **k: [
        'units x sauron/regular: engine=1 client=0']
    try:
        drv.reconcile_board_quiet()
        drv.reconcile_board_quiet()
    finally:
        client_board.diff_engine_vs_dump = orig
    assert not calls, \
        'BRLANE-1 RED (BRCMP gate): adopted %r over an open window' % calls
    # the kill closes the gesture: the window commits, the parked arm
    # is promoted, the gesture is no longer open
    rig.feed(lines[2])
    assert drv.commits and drv.commits[-1][2] == 'window', \
        [c[2] for c in drv.commits]
    assert drv.window_stats['matched'] == 1, dict(drv.window_stats)
    assert drv.gesture_open() is False
    assert drv._reconcile_due is True


# ------------------------------------------------------------- BRLANE-2
def test_brlane2_held_window_retries_when_condition_clears():
    """A window HELD behind the AI's open multi-step die is re-matched
    the moment the AI half closes (finding hold_retried ->
    hold_retry_matched, the move commits, holds empty).  A hold whose
    condition never clears expires LOUDLY (hold_expired + alarm +
    the self-repair ASK), never silently."""
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [(SH, 0, F.ARMY), (SH, 1, F.CHARACTER),
                                       (FP, 0, F.CHARACTER)])
    a, lines = _army_window_lines(st)
    assert a is not None
    rig = Rig(st, FP, phase='actions')                 # human = SHADOW
    drv = rig.drv
    drv.last_actor = FP                                # AI acted last
    drv.es.pending_army_move_die = 0                   # AI half open
    drv.es.pending_army_move_side = FP
    rig.feed(*lines)
    assert drv.window_stats['hold'] == 1 and len(drv.holds) == 1, \
        (dict(drv.window_stats), len(drv.holds))
    drv.es.pending_army_move_die = None                # the AI closes it
    drv.es.pending_army_move_side = None
    drv.tick()
    assert drv.window_stats['matched'] == 1 and not drv.holds, \
        'BRLANE-2 RED: held window never retried (stats %r, holds %d)' \
        % (dict(drv.window_stats), len(drv.holds))
    assert drv.commits and drv.commits[-1][2] == 'window'
    assert 'hold_retried' in _kinds(drv) and 'hold_retry_matched' in _kinds(drv), \
        _kinds(drv)[-4:]
    # expiry leg: the condition never clears
    st2 = fresh_state()
    game_logic.adopt_rolled_faces(st2, [(SH, 0, F.ARMY), (SH, 1, F.CHARACTER),
                                        (FP, 0, F.CHARACTER)])
    a2, lines2 = _army_window_lines(st2)
    rig2 = Rig(st2, FP, phase='actions')
    drv2 = rig2.drv
    drv2.last_actor = FP
    drv2.es.pending_army_move_die = 0
    drv2.es.pending_army_move_side = FP
    rig2.feed(*lines2)
    assert len(drv2.holds) == 1
    w2 = drv2.holds[0][0]
    w2._hold_tick = drv2._tick_n - 100000              # aged past the bound
    out = drv2.tick()
    assert 'hold_expired' in _kinds(drv2), \
        'BRLANE-2 RED: an aged hold expired silently: %r' % (_kinds(drv2)[-4:],)
    assert any('HELD WINDOW' in x for x in drv2.alarms), drv2.alarms
    assert drv2._repair is not None and drv2._repair.get('phase') == 'offered', \
        drv2._repair
    assert any('self-repair' in c for c in _chats(out)), out


# ------------------------------------------------------------- BRLANE-3
def test_brlane3_ai_declare_parks_loudly_and_status_names_human_pending():
    """AI = FP; the human Shadow's turn-start draws take the hand to 8:
    the engine's open decision is the HUMAN's discard, the AI's queued
    ai_declare step parks.  The status line must name the human's
    discard (not 'AI deciding: ai_declare'), the park must be said in
    chat once, the starve counter must not count parks, and after N
    parks the self-repair ASK goes out."""
    from wotr_bridge.live.live_server import LiveServer
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, FP, load_nets=False, seed=5)  # human = SHADOW
    drv = srv.driver
    hand = drv.es.hands[SH].hand
    while len(hand) < 6:
        hand.append(drv.es.sh_strat_deck.pop())
    with open(feed, 'a', encoding='utf-8') as f:
        f.write('$Wey draws Shadow Character Card.\n'
                '$Wey draws Shadow Strategy Card.\n')
    outs = []
    for _ in range(6):
        out = srv.handle({'lines': ()})
        outs.extend(out)
        for s in out:
            if s == '!CALL draw_fp_char':
                srv.rand.feed('card', ('FREE_PEOPLES', 'character',
                                       drv.es.fp_char_deck[-1]))
            if s == '!CALL draw_fp_strat':
                srv.rand.feed('card', ('FREE_PEOPLES', 'strategy',
                                       drv.es.fp_strat_deck[-1]))
    dp = ai.describe_pending(drv.es)
    assert dp is not None and dp['side'] is SH and dp['kind'] == 'discard', \
        'precondition: %r' % (dp and (dp['kind'], dp['side']),)
    assert srv._open_step is not None and srv._open_step[0] == 'ai_declare', \
        srv._open_step
    wo = srv._waiting_on()
    assert 'SHADOW' in wo and 'discard' in wo.lower() \
        and not wo.startswith('AI deciding'), \
        'BRLANE-3 RED: status line %r hides the human discard' % wo
    said = [s for s in outs if s.startswith('<WOTR_AI')
            and 'discard' in s.lower()]
    assert said, 'BRLANE-3 RED: the park was never said in chat: %r' % (
        [s for s in outs if s.startswith('<')],)
    assert 'ai_declare_parked' in _kinds(drv), _kinds(drv)[-5:]
    assert getattr(srv, '_step_starved', 0) <= 1, \
        'BRLANE-3 RED: parks counted as starvation (%d)' % srv._step_starved
    # the starvation loop ends in the ASK (never an eternal park)
    for _ in range(40):
        outs.extend(srv.handle({'lines': ()}))
        if drv._repair is not None:
            break
    assert drv._repair is not None and drv._repair.get('phase') == 'offered', \
        'BRLANE-3 RED: no self-repair ASK after 40 parks (%r)' % (drv._repair,)
    assert any('self-repair' in s for s in outs), outs[-5:]
    assert ai.describe_pending(drv.es) is not None       # nothing forced


# ------------------------------------------------------------- BRLANE-4
def test_brlane4_unknown_chat_is_never_silent():
    """Human chat that is neither a command nor a region gets a chat
    reply naming what is accepted now (finding chat_unrecognised); with
    a card-play window open, a region / 'done' chat is answered in CHAT
    (the guard text), not only by a panel note."""
    st = _faces(fresh_state(),
                [F.CHARACTER, F.MUSTER, F.MUSTER_ARMY, F.MUSTER_ARMY],
                [F.EYE, F.EYE, F.CHARACTER, F.ARMY, F.MUSTER, F.MUSTER,
                 F.EVENT])
    rig = Rig(st, SH, phase='actions')                 # human = FP
    drv = rig.drv
    drv.last_actor = SH
    for body in ('done', 'finished', 'nazgul done', 'ok', 'reveal',
                 'banana', 'i am stuck'):
        out = rig.feed('<Wey> %s' % body)
        assert _chats(out), \
            'BRLANE-4 RED: %r answered with silence (%r)' % (body, out)
    assert 'chat_unrecognised' in _kinds(drv), _kinds(drv)[-4:]
    # the card-play window guard answers in chat
    st2 = _faces(fresh_state(),
                 [F.CHARACTER, F.MUSTER, F.MUSTER_ARMY, F.MUSTER_ARMY],
                 [F.EYE, F.EYE, F.CHARACTER, F.ARMY, F.MUSTER, F.MUSTER,
                  F.EVENT])
    rig2 = Rig(st2, FP, phase='actions')               # human = SHADOW
    rig2.drv.last_actor = FP
    rig2.feed('$Wey moves Shadow Action Die (P) from SA Dice to Current '
              'Action.',
              '$Wey moves Nazgul Search/Foul Stench from Shadow Hand to '
              'Current SA Card.')
    out2 = rig2.feed("<Wey> helm's deep")
    assert _chats(out2), \
        'BRLANE-4 RED: the card-window guard answered with a note only: ' \
        '%r' % (out2,)
    assert not any('not legal' in c for c in _chats(out2)), out2
    out3 = rig2.feed('<Wey> done')
    assert _chats(out3), 'BRLANE-4 RED: done over a card window: %r' % (out3,)


# ------------------------------------------------------------- BRLANE-5
def test_brlane5_dieless_move_and_kill_are_loud():
    """A human piece move / kill with NO die staged and NO pending open
    is matched against the engine's die-less rows or REFUSED (chat +
    one client-native undo press + finding), never dropped; the engine
    stays untouched.  Inert bookkeeping (a VP marker walk) stays quiet."""
    from wotr_bridge.live import client_board
    st = _faces(fresh_state(),
                [F.CHARACTER, F.MUSTER, F.MUSTER_ARMY, F.MUSTER_ARMY],
                [F.EYE, F.EYE, F.CHARACTER, F.ARMY, F.MUSTER, F.MUSTER,
                 F.EVENT])
    rig = Rig(st, FP, phase='actions')                 # human = SHADOW
    drv = rig.drv
    drv.last_actor = FP
    pre = client_board.proj_board(drv.es)
    out = rig.feed('$Wey moves group (Nazgul, Sauron Elite, [4]Sauron '
                   'Regular) from Barad-Dur to Gorgoroth.')
    assert _chats(out) and _undo_lines(out), \
        'BRLANE-5 RED: die-less army drag dropped silently: %r' % (out,)
    assert 'dieless_gesture_refused' in _kinds(drv), _kinds(drv)[-4:]
    assert client_board.proj_board(drv.es) == pre
    out = rig.feed('$Wey kills Shadow Character Card in Shadow Hand.')
    assert _chats(out) and _undo_lines(out), \
        'BRLANE-5 RED: hand kill with no discard pending dropped: %r' % (out,)
    out = rig.feed('$Wey kills Sauron Regular in Gorgoroth.')
    assert _chats(out) and _undo_lines(out), \
        'BRLANE-5 RED: unit kill with no pending dropped: %r' % (out,)
    assert client_board.proj_board(drv.es) == pre
    # SPECIFICITY: engine-derived bookkeeping is not refused
    out = rig.feed('$Wey moves Victory Points Marker Shadow from Victory '
                   'Points 0 to Victory Points 1.')
    assert not _chats(out) and not _undo_lines(out), out


# ------------------------------------------------------------- BRLANE-6
def test_brlane6_eye_undo_unwinds_engine():
    """The client undo of 'allocates an eye.' (menu form) and of the
    SA Dice -> Hunt box drag (corpus form) pops the allocation ledger
    entry: the eye leaves the engine's Hunt Box, the counter resets,
    no 'undo with empty commit ledger' / ORDER MISMATCH alarm."""
    for line in ('$Wey allocates an eye.',
                 '$Wey moves Shadow Action Die (A) from SA Dice to The '
                 'Hunt For The Ring.'):
        st = fresh_state()
        rig = Rig(st, FP, phase='eyes')                # human = SHADOW
        drv = rig.drv
        hb0 = len(st.hunt_box)
        rig.feed(line)
        assert len(drv.es.hunt_box) == hb0 + 1, 'precondition: eye landed'
        rig.feed('$Wey undoes last action : %s' % line.lstrip('$'))
        assert len(drv.es.hunt_box) == hb0, \
            'BRLANE-6 RED: the undone eye stays in the engine hunt box ' \
            '(%r)' % (line,)
        assert int(getattr(drv.es, 'eyes_allocated', 0) or 0) == 0
        assert not any('empty commit ledger' in x or 'ORDER MISMATCH' in x
                       for x in drv.alarms), drv.alarms
        assert 'eye_undone' in _kinds(drv), _kinds(drv)[-4:]


# ------------------------------------------------------------- BRLANE-7
def test_brlane7_undo_of_ai_line_reasserts():
    """A human undo of the AI's OWN rendered line is re-asserted: the
    raw op line the server emitted is re-injected, the reconcile is
    NOT armed (the adoption path that erased the AI's move), finding
    undo_of_ai_line_reasserted."""
    import emitter
    st = _faces(fresh_state(), [F.CHARACTER], [F.ARMY, F.CHARACTER])
    rig = Rig(st, SH, phase='actions')                 # human = FP
    drv = rig.drv
    drv.last_actor = SH
    raw = emitter.render('move', {
        'nick': 'WOTR_AI_Shadow', 'pieces': ['001'],
        'frm': emitter.zone_code_for('barad dur'),
        'to': emitter.zone_code_for('gorgoroth')}, local=False)
    assert hasattr(drv, 'note_own_emission'), \
        'BRLANE-7 RED: the driver keeps no ledger of its own emitted lines'
    drv.note_own_emission([raw])
    drv._reconcile_due = False
    out = rig.feed('$Wey undoes last action : $WOTR_AI_Shadow moves Sauron '
                   'Regular from Barad-Dur to Gorgoroth.')
    assert any(i[0] == 'line' and i[1] == raw for i in out), \
        'BRLANE-7 RED: the undone AI line was not re-injected: %r' % (out,)
    assert drv._reconcile_due is False, \
        'BRLANE-7 RED: the AI-line undo armed the auto-reconcile'
    assert 'undo_of_ai_line_reasserted' in _kinds(drv), _kinds(drv)[-4:]
    # an AI-line undo the ledger cannot place is LOUD, still no adoption arm
    out2 = rig.feed('$Wey undoes last action : $WOTR_AI_Shadow moves Nazgul '
                    'from Minas Morgul to Osgiliath.')
    assert _chats(out2), out2
    assert drv._reconcile_due is False
    assert any('UNDO OF AI LINE' in x for x in drv.alarms), drv.alarms


# ------------------------------------------------------------- BRLANE-8
def test_brlane8_restore_watch_zone_aliases():
    """The dump names zones by their decode-table spelling (Fellowship
    Box, Guide Box, FP Dead) while the rendered line names them
    (Fellowship Of The Ring, Guide Of The Fellowship, FP Casualties):
    one zone, two spellings - a landed restore verifies.  SPECIFICITY:
    a genuinely different zone still fails."""
    from wotr_bridge.live import undo_model as um
    rw = um.RestoreWatch()
    rw.arm({'241': 'Fellowship Of The Ring', '244': 'Fellowship Of The Ring',
            '240': 'Guide Of The Fellowship', '245': 'FP Casualties'},
           {}, ticks=1)
    bad = rw.verify({'241': 'Fellowship Box', '244': 'Fellowship Box',
                     '240': 'Guide Box', '245': 'FP Dead'})
    assert bad == [] and rw.alarm is None, \
        'BRLANE-8 RED: landed restore reported as drift: %r' % (bad,)
    rw2 = um.RestoreWatch()
    rw2.arm({'241': 'Fellowship Of The Ring'}, {}, ticks=1)
    bad2 = rw2.verify({'241': 'Moria'})
    assert bad2 and rw2.alarm, (bad2, rw2.alarm)


# ------------------------------------------------------------- BRLANE-9
def test_brlane9_restore_watch_two_hop_card():
    """A card that left the hand (redacted pids) and then walked
    Current FP Card -> Board1 is expected back in the HAND after the
    refusal's undo, not at its second hop's origin."""
    from wotr_bridge.live import undo_model as um
    lines = ['$Wey moves Free Action Die (P) from FP Dice to Current Action.',
             '$Wey moves A Power Too Great/No Quarter from Free Peoples Hand '
             'to Current FP Card.',
             '$Wey moves A Power Too Great/No Quarter from Current FP Card '
             'to Board1.',
             '$Wey draws Free Strategy Card.',
             '$Wey kills Free Action Die (P) in Current Action.']
    pids = [['213'], None, ['111'], ['119'], ['213']]
    expect, notat, unver = um.expectations_from_window(lines, pids=pids)
    assert expect.get('111') == 'Free Peoples Hand', \
        'BRLANE-9 RED: card 111 expected at %r' % (expect.get('111'),)
    assert expect.get('213') == 'FP Dice', expect
    assert not any('Power' in str(u[0]) for u in unver), unver


# ------------------------------------------------------------ BRLANE-10
def test_brlane10_pass_refusal_prints_counts():
    """A rule-correct pass refusal names the numbers it was judged on
    (unused dice per side, Hunt Box dice excluded) and the rule."""
    st = _faces(fresh_state(),
                [F.CHARACTER, F.MUSTER, F.MUSTER_ARMY, F.MUSTER_ARMY],
                [F.EYE, F.EYE, F.EYE, F.ARMY, F.MUSTER, F.MUSTER, F.EVENT])
    rig = Rig(st, SH, phase='actions')                 # human = FP
    rig.drv.last_actor = SH
    out = rig.feed('<Wey> pass')
    txt = ' '.join(_chats(out)).lower()
    assert 'you cannot pass' in txt, out
    assert 'you have 4 unused' in txt and 'i have 4' in txt, \
        'BRLANE-10 RED: no counts in %r' % (txt,)
    assert 'hunt box' in txt and 'fewer' in txt, txt
    assert not rig.drv.commits


# ------------------------------------------------------------ BRLANE-11
def test_brlane11_declare_in_place_unique_and_pending_ambiguity_voiced():
    """(a) an FSP track reset with NO pawn walk is a declaration IN
    PLACE - the only candidate whose pawn stays; it commits.  (b) a
    pending-buffer match that comes back ambiguous is VOICED once
    (chat ambiguous_hold + finding), never a silent 'not yet'."""
    st = fresh_state()
    st.fellowship.progress = 2
    st.fellowship.hidden = True
    rig = Rig(st, SH, phase='draws')                   # human = FP
    drv = rig.drv
    rig.feed('$Wey draws Free Character Card.',
             '$Wey draws Free Strategy Card.')
    dp = ai.describe_pending(drv.es)
    assert dp is not None and dp['kind'] == 'declare', dp and dp['kind']
    rig.feed('$Wey moves FSP from Fellowship Track 2 to Fellowship Track 0.')
    assert ai.describe_pending(drv.es) is None, \
        'BRLANE-11 RED: declare-in-place not committed (pending %r, ' \
        'buffer %r)' % (ai.describe_pending(drv.es)['kind'],
                        drv.pending_buf and [r.get('norm')
                                             for r in drv.pending_buf.records])
    assert sanitize(str(game_logic.fellowship_region(drv.es))) == 'rivendell'
    assert drv.es.fellowship.progress == 0
    assert drv.es.fellowship.hidden is True
    assert drv.commits[-1][2] == 'pending:declare', [c[2] for c in drv.commits]
    # (b) the voice on an ambiguous pending match
    from wotr_bridge.live import gesture_match as gm
    from test_live_adopt import hunt_pending
    st2 = fresh_state()
    hunt_pending(st2)
    rig2 = Rig(st2, SH, phase='actions')               # human = FP
    drv2 = rig2.drv
    # a record that buffers into the hunt decision but answers none of
    # its options (the _stalled_rig shape)
    rig2.feed('$Wey moves Gondor Regular from Minas Tirith to Osgiliath.')
    assert drv2.pending_buf is not None and drv2.pending_buf.records, \
        (drv2.pending_buf, ai.describe_pending(drv2.es))
    orig = gm.match_window
    fake = types.SimpleNamespace(action_type='hunt_take_corruption', params={})
    gm.match_window = lambda *a, **k: gm.MatchResult(
        'ambiguous', chain=[[fake], [fake]],
        note='2 distinct actions produce the window delta')
    try:
        drv2._try_pending()
        n1 = len([i for i in drv2._out_queue if i[0] == 'chat'])
        drv2._try_pending()
        n2 = len([i for i in drv2._out_queue if i[0] == 'chat'])
    finally:
        gm.match_window = orig
    assert n1 == 1 and 'undo' in str(drv2._out_queue[-1][-1]).lower(), \
        'BRLANE-11 RED: the pending ambiguity was swallowed (%r)' % (
            drv2._out_queue,)
    assert n2 == 1, 'voiced more than once per buffer epoch'
    assert 'pending_ambiguous_voiced' in _kinds(drv2), _kinds(drv2)[-4:]


def _two_siege_uruk_state():
    from game_state import Nation, UnitKind
    st = fresh_state()
    _faces(st, [F.CHARACTER, F.WILL_OF_WEST, F.MUSTER, F.EVENT],
           [F.EVENT, F.ARMY, F.MUSTER, F.CHARACTER, F.EYE, F.MUSTER_ARMY,
            F.ARMY])
    for reg, nat in (("helm's deep", Nation.ROHAN),
                     ('minas tirith', Nation.GONDOR)):
        a = game_logic.get_or_create_army(st, reg)
        a.units = [gs.Unit(nat, UnitKind.REGULAR),
                   gs.Unit(Nation.ISENGARD, UnitKind.REGULAR),
                   gs.Unit(Nation.ISENGARD, UnitKind.REGULAR)]
        st.besieged_strongholds.add(sanitize(reg))
    st.saruman.in_play = True
    st.political.boxes[Nation.ISENGARD] = 3
    st.political.active[Nation.ISENGARD] = True
    cid = 'SHADOW_Strategy_The_Fighting_Uruk-hai'
    for dk in ('sh_strat_deck', 'sh_char_deck'):
        dl = getattr(st, dk, [])
        if cid in dl:
            dl.remove(cid)
    return st


# ------------------------------------------------------------ BRLANE-12
def test_brlane12_card_target_chat_applies_after_kill():
    """The Fighting Uruk-hai with TWO sieges: the human stages the P die,
    drags the card and chats the target.  The guard answers in chat and
    REMEMBERS the region; after the die kill the play opens the card's
    target decision and the remembered region answers it (the battle
    opens at Helm's Deep; finding card_target_from_chat).  The plain
    attack lane's refusal names the faces that can attack."""
    st = _two_siege_uruk_state()
    rig = Rig(st, FP, phase='actions')                 # human = SHADOW
    drv = rig.drv
    drv.last_actor = FP
    drv.es.hands[SH].hand.append(drv.hands._mint('strategy'))
    rig.feed('$Wey moves Shadow Action Die (P) from SA Dice to Current '
             'Action.',
             '$Wey moves The Fighting Uruk-Hai/Onslaught from Shadow Hand '
             'to Board4.')
    out = rig.feed("<Wey> helm's deep")
    assert _chats(out) and not any('not legal' in c for c in _chats(out)), \
        'BRLANE-12 RED: region chat over the card window: %r' % (out,)
    rig.feed('$Wey kills Shadow Action Die (P) in Current Action.')
    assert drv.window_stats['matched'] == 1, dict(drv.window_stats)
    pc = getattr(drv.es, 'pending_combat', None) or {}
    assert 'helm' in str(pc.get('defender_region', '')), \
        'BRLANE-12 RED: the chatted target never answered the card ' \
        'decision (pending %r)' % (ai.describe_pending(drv.es)
                                   and ai.describe_pending(drv.es)['kind'],)
    assert 'card_target_from_chat' in _kinds(drv), _kinds(drv)[-4:]
    # the plain attack lane: a P die staged, an A die available
    st2 = fresh_state()
    game_logic.adopt_rolled_faces(st2, [(SH, 0, F.ARMY), (SH, 1, F.EVENT),
                                        (FP, 0, F.CHARACTER)])
    for nat in list(st2.political.boxes):
        if ai._side_of_nation(nat) == SH:
            st2.political.boxes[nat] = 3
            st2.political.active[nat] = True
    atk = next(a for a in ai.get_legal_actions(st2, SH)
               if a.action_type == 'army_attack')
    rig2 = Rig(st2, FP, phase='actions', seed=5)
    rig2.drv.last_actor = FP
    rig2.feed('$Wey moves Shadow Action Die (P) from SA Dice to Current '
              'Action.')
    out2 = rig2.feed('<Wey> %s from %s' % (atk.params['to'], atk.params['from']))
    txt = ' '.join(_chats(out2))
    assert 'staged P die' in txt, out2
    assert 'A' in txt.split('staged P die', 1)[1], \
        'BRLANE-12 RED: the refusal does not name the faces that can ' \
        'attack: %r' % (txt,)


# ------------------------------------------------------------ BRLANE-13
def test_brlane13_emission_echo_gates_step_and_expires_loudly():
    """With the sidecar present, every op line the AI emits is an
    EXPECTED ECHO: echo_pending() holds the step queue (no starve count)
    until the client's echo under the AI nick returns the pids; expiry
    is a finding + alarm + the self-repair ASK."""
    import emitter
    from wotr_bridge.live import sidecar as sc_mod
    from wotr_bridge.live.live_server import LiveServer
    td = tempfile.mkdtemp()
    feed = os.path.join(td, 'ai_feed.txt')
    open(feed, 'w').close()
    srv = LiveServer(td, SH, load_nets=False, seed=3)   # human = FP
    drv = srv.driver
    meta_p = feed + '.meta.jsonl'
    _write_meta(meta_p, [{'seq': 0, 'gen': 1, 'attest': True, 'nick': '',
                          'tags': [], 'pids': [], 'ts': 0}])
    drv.sidecar.poll()
    assert drv.sidecar.present
    raw = emitter.render('move', {
        'nick': srv.nick, 'pieces': ['001'],
        'frm': emitter.zone_code_for('barad dur'),
        'to': emitter.zone_code_for('gorgoroth')}, local=False)
    assert hasattr(drv, 'expect_echo') and hasattr(drv, 'echo_pending'), \
        'BRLANE-13 RED: no expected-echo ledger on the driver'
    drv.expect_echo([raw])
    assert drv.echo_pending(), 'BRLANE-13 RED: emitted line not expected'
    srv._open_step = ('ai_decide', None)
    srv._step_starved = 0
    out = srv._run_open_step()
    assert srv._open_step == ('ai_decide', None) and srv._step_starved == 0, \
        'BRLANE-13 RED: the step queue advanced with an echo outstanding ' \
        '(%r, starved %d)' % (srv._open_step, srv._step_starved)
    assert 'render' in srv._waiting_on().lower() \
        or 'echo' in srv._waiting_on().lower(), srv._waiting_on()
    # the client's echo releases it
    with open(feed, 'a', encoding='utf-8') as f:
        f.write('$%s moves Sauron Regular from Barad-Dur to Gorgoroth.\n'
                % srv.nick)
    _write_meta(meta_p, [{'seq': 1, 'gen': 1, 'nick': srv.nick,
                          'tags': ['Scribe.141'], 'pids': ['001'], 'ts': 1}])
    drv.tick()
    assert not drv.echo_pending(), 'echo not released by the client line'
    assert 'emission_echo_released' in _kinds(drv), _kinds(drv)[-4:]
    srv._open_step = None
    # expiry: no echo ever comes
    drv.expect_echo([raw])
    for _ in range(12):
        out = drv.tick()
        if 'emission_unacknowledged' in _kinds(drv):
            break
    assert 'emission_unacknowledged' in _kinds(drv), \
        'BRLANE-13 RED: an unacknowledged emission expired silently'
    assert any('EMISSION UNACKNOWLEDGED' in x for x in drv.alarms), drv.alarms
    assert drv._repair is not None and drv._repair.get('phase') == 'offered', \
        drv._repair
    assert not drv.echo_pending()


# ------------------------------------------------------------ BRLANE-14
def test_brlane14_human_deck_draw_demand_is_chatted():
    """A deferred gesture starving on a draw from the HUMAN's deck is
    asked in chat (draw_ask), once per demand - not a panel note only."""
    st = adopted_state()
    rig = Rig(st, SH, phase='actions')                 # human = FP
    drv = rig.drv
    w = types.SimpleNamespace(records=[], _starve_call=types.SimpleNamespace(
        kind='draws', kw={'items': [('FREE_PEOPLES', 'strategy')]}))
    drv.deferred = (w, FP)
    drv.rand._pending = None
    out = drv.tick()
    asks = [c for c in _chats(out) if c.lower().startswith('draw a')]
    assert asks and 'strategy' in asks[0].lower(), \
        'BRLANE-14 RED: the human-deck draw demand is not chatted: %r' % (out,)
    assert 'human_draw_asked' in _kinds(drv), _kinds(drv)[-4:]
    out2 = drv.tick()
    assert not [c for c in _chats(out2) if c.lower().startswith('draw a')], \
        'asked twice for one demand'


# ------------------------------------------------------------ BRLANE-15
def test_brlane15_roll_side_from_sidecar_pids():
    """Three eyes allocated leave 4 unset Shadow dice = 4 FP dice; a
    rendered roll with no E/W face is ambiguous by faces - the sidecar's
    die channel (pids 210-213 = the FP family) decides, no alarm, the
    FP pool adopts."""
    from wotr_bridge.live import live_parser
    st = fresh_state()
    rig = Rig(st, FP, phase='eyes')                    # human = SHADOW
    drv = rig.drv
    for _ in range(3):
        rig.feed('$Wey allocates an eye.')
    rec = live_parser.parse('$Wey rolls C A M P .')
    rec['meta'] = {'pids': ['210', '211', '212', '213'],
                   'segs': [[], ['210', '211', '212', '213'], []]}
    drv._on_action_roll(rec)
    assert not any('ROLL SIDE AMBIGUOUS' in x for x in drv.alarms), \
        'BRLANE-15 RED: %r' % (drv.alarms,)
    assert all(d.current is not None for d in drv.es.dice[FP]), \
        [getattr(d.current, 'name', None) for d in drv.es.dice[FP]]
    assert sum(1 for d in drv.es.dice[SH]
               if not d.used and d.current is None) == 4
    assert 'roll_side_from_sidecar' in _kinds(drv), _kinds(drv)[-4:]


# ----------------------------------------------------------------- EX-8
def test_ex8_table_kill_payment_matches_the_killed_pair():
    """The engine's EX-8 payment pending (every Army-type + Character
    pair) is answered by the pair the human physically KILLED - read
    from the client's own dump (the decoded hand pids that left the
    hand zone) - never the first pair in hand order."""
    import cards
    st = _faces(fresh_state(), [F.CHARACTER], [F.ARMY, F.CHARACTER])
    rig = Rig(st, FP, phase='actions', seed=5)         # human = SHADOW
    drv = rig.drv
    drv.last_actor = FP
    hand = drv.es.hands[SH].hand
    army = [c for c in st.sh_strat_deck
            if getattr(cards.get_card(c), 'card_type', None) == 'Army'][:2]
    chars = [c for c in st.sh_char_deck if '_Character_' in c][:2]
    assert len(army) == 2 and len(chars) == 2
    hand[:] = []
    for c in army + chars:
        (st.sh_strat_deck if c in st.sh_strat_deck else st.sh_char_deck).remove(c)
        hand.append(c)
    pids = {}
    for i, c in enumerate(hand):
        pid = '%03d' % (150 + i)
        drv.view.card_pid_map[c] = pid
        pids[c] = pid
    drv.view.piece_region = {p: 'SA Hand' for p in pids.values()}
    drv.view.refresh_dump = lambda: True
    ai._open_table_kill_payment(drv.es, {'card_id': 'X', 'card_name': 'X'})
    dp = ai.describe_pending(drv.es)
    assert dp is not None and dp['kind'] == 'card_option' and dp['side'] is SH
    # the human kills the SECOND army card and the FIRST character card
    drv.view.piece_region.pop(pids[army[1]])
    drv.view.piece_region.pop(pids[chars[0]])
    rig.feed('$Wey kills Shadow Strategy Card in Shadow Hand.',
             '$Wey kills Shadow Character Card in Shadow Hand.')
    assert ai.describe_pending(drv.es) is None, 'payment not committed'
    assert sorted(drv.es.hands[SH].hand) == sorted([army[0], chars[1]]), \
        'EX-8 RED: the engine paid with %r, the human killed %r + %r' % (
            [c for c in army + chars if c not in drv.es.hands[SH].hand],
            army[1], chars[0])
    assert 'table_kill_payment_from_dump' in _kinds(drv), _kinds(drv)[-4:]


# --------------------------------------------------------------- item 14
def test_item14_card_play_commits_by_identity_and_explains_extras():
    """The card leaving the hand with a staged Event-capable die IS the
    play: A Power too Great with an EXTRA contradicting drag (the Rohan
    marker; the card advances Elves) still COMMITS by card identity -
    the engine's own Elves advance renders, no undo press - and the row
    the engine did not produce is EXPLAINED in chat."""
    st = fresh_state()
    _faces(st, [F.EVENT, F.CHARACTER, F.MUSTER, F.WILL_OF_WEST],
           [F.ARMY, F.CHARACTER, F.MUSTER, F.EYE, F.EVENT, F.MUSTER_ARMY,
            F.CHARACTER])
    cid = 'FREE_PEOPLES_Strategy_A_Power_too_Great'
    if cid in st.fp_strat_deck:
        st.fp_strat_deck.remove(cid)
    rig = Rig(st, SH, phase='actions', seed=12)        # human = FP
    drv = rig.drv
    drv.last_actor = SH
    drv.es.hands[FP].hand.append(drv.hands._mint('strategy'))
    elves0 = drv.es.political.boxes[gs.Nation.ELVES]
    out = rig.feed(
        '$Wey moves Free Action Die (P) from FP Dice to Current Action.',
        '$Wey moves A Power Too Great/No Quarter from Free Peoples Hand to '
        'Current FP Card.',
        '$Wey moves Rohan from Political Track 3 to Political Track 2.',
        '$Wey kills Free Action Die (P) in Current Action.')
    out += rig.feed('$Wey draws Free Strategy Card.')
    assert drv.window_stats['matched'] == 1 and not _undo_lines(out), \
        'item 14 RED: the play was refused whole (stats %r, presses %d)' % (
            dict(drv.window_stats), len(_undo_lines(out)))
    pc = getattr(drv.es, 'persistent_cards', None)
    assert pc is not None and any(getattr(c, 'card_id', None) == cid
                                  for c in (getattr(pc, 'fp_cards', [])
                                            or [])), 'card not on the table'
    assert drv.es.political.boxes[gs.Nation.ELVES] == elves0 + 1
    assert 'card_play_extra_rows' in _kinds(drv), _kinds(drv)[-5:]
    assert any('rohan' in c.lower() or 'political' in c.lower()
               for c in _chats(out)), _chats(out)


# --------------------------------------------------------------- item 15
def test_item15_group_companion_drag_maps_to_group_row():
    """REGRESSION PIN (already GREEN on the 2026-08-29 tree: the 08-22
    refusal was the <=4 subset cap removed that day): a two-companion
    drag out of the Fellowship box with a C die matches the engine's
    GROUP separation row - both companions land, no refusal."""
    st = _faces(fresh_state(),
                [F.CHARACTER, F.MUSTER, F.MUSTER_ARMY, F.WILL_OF_WEST],
                [F.EYE, F.ARMY, F.MUSTER, F.MUSTER, F.CHARACTER, F.EVENT,
                 F.ARMY])
    rig = Rig(st, SH, phase='actions')                 # human = FP
    drv = rig.drv
    drv.last_actor = SH
    out = rig.feed('$Wey moves Free Action Die (C) from FP Dice to Current '
                   'Action.',
                   '$Wey moves group (Strider, Boromir) from Fellowship Of '
                   'The Ring to Moria.',
                   '$Wey kills Free Action Die (C) in Current Action.')
    assert drv.window_stats['matched'] == 1 and not _undo_lines(out), \
        (dict(drv.window_stats), out)
    names = [c.name for c in drv.es.fellowship.companions]
    assert 'Strider' not in names and 'Boromir' not in names, names
    moria = next(a for k, a in drv.es.armies.items()
                 if sanitize(k) == 'moria')
    assert sorted(getattr(l, 'name', '') for l in moria.leaders) \
        == ['Boromir', 'Strider']


# --------------------------------------------------------------- item 16
def test_item16_battle_resolved_by_hand_asks_immediately():
    """The human FP owes defender_combat_card and instead drags the AI's
    army into the region, walks the VP marker and kills the dice: the
    shape is recognised AT ONCE - chat names the owed combat-card
    decision and the two ways out (nc / undo), the self-repair ASK is
    offered without the 25-tick stall, nothing is adopted, the pending
    stays open."""
    from test_live_driver import _human_defender_cc_rig
    fails = []
    r = _human_defender_cc_rig(fails, 'item16')
    assert r is not None, fails
    rig, drv = r
    pc = drv.es.pending_combat
    atk_reg = str(pc.get('attacker_region')).title()
    def_reg = str(pc.get('defender_region')).title()
    out = rig.feed('$Wey moves group (Nazgul, [2]Sauron Regular) from %s to '
                   '%s.' % (atk_reg, def_reg),
                   '$Wey moves Victory Points Marker Shadow from Victory '
                   'Points 0 to Victory Points 1.',
                   '$Wey kills group (Free Action Die (C), Shadow Action Die '
                   '(A)) in Current Action.')
    out += drv.tick()
    txt = ' '.join(_chats(out)).lower()
    assert 'combat card' in txt and 'nc' in txt and 'undo' in txt, \
        'item 16 RED: no immediate explanation, chat=%r' % (txt,)
    assert 'battle_resolved_by_hand' in _kinds(drv), _kinds(drv)[-5:]
    assert drv._repair is not None and drv._repair.get('phase') == 'offered', \
        'item 16 RED: no self-repair ASK at once (%r)' % (drv._repair,)
    dp = ai.describe_pending(drv.es)
    assert dp is not None and dp['kind'] == 'combat:defender_combat_card' \
        and dp['side'] is FP, dp and (dp['kind'], dp['side'])
    assert not any(k in ('board_adopt', 'resync_adopt') for k in _kinds(drv))


def main():
    tests = [
        test_brlane1_open_window_parks_reconcile_and_is_not_quiescent,
        test_brlane2_held_window_retries_when_condition_clears,
        test_brlane3_ai_declare_parks_loudly_and_status_names_human_pending,
        test_brlane4_unknown_chat_is_never_silent,
        test_brlane5_dieless_move_and_kill_are_loud,
        test_brlane6_eye_undo_unwinds_engine,
        test_brlane7_undo_of_ai_line_reasserts,
        test_brlane8_restore_watch_zone_aliases,
        test_brlane9_restore_watch_two_hop_card,
        test_brlane10_pass_refusal_prints_counts,
        test_brlane11_declare_in_place_unique_and_pending_ambiguity_voiced,
        test_brlane12_card_target_chat_applies_after_kill,
        test_brlane13_emission_echo_gates_step_and_expires_loudly,
        test_brlane14_human_deck_draw_demand_is_chatted,
        test_brlane15_roll_side_from_sidecar_pids,
        test_ex8_table_kill_payment_matches_the_killed_pair,
        test_item14_card_play_commits_by_identity_and_explains_extras,
        test_item15_group_companion_drag_maps_to_group_row,
        test_item16_battle_resolved_by_hand_asks_immediately,
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
