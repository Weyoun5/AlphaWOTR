"""LiveDriver pipeline instruments: feed -> parse -> segment -> match
-> transactional commit -> per-line undo, plus the turn-start
protocol gates.  Real client grammar throughout (specimen shapes);
the engine is the only state authority in every assertion.
"""
import os
import sys
import tempfile

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as board_mod
import game_logic
import game_state as gs
from game_state import Side, ActionDieFace

from wotr_bridge.live import client_board
from wotr_bridge.live.driver import LiveDriver


def fresh_state():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def adopted_state():
    st = fresh_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
    ])
    return st


class Rig(object):
    """A temp feed file + a LiveDriver over it, wired with a REAL
    installed ClientRandomness (production shape — the PHASE3-D ports
    feed/consume the funnels on every window match)."""

    def __init__(self, state, ai_side, phase=None, seed=7):
        from wotr_bridge.live.randomness import ClientRandomness
        self.td = tempfile.mkdtemp()
        self.path = os.path.join(self.td, 'feed.txt')
        open(self.path, 'w').close()
        self.rand = ClientRandomness(seed=seed)
        self.rand.install()     # idempotent; re-arms _ACTIVE per rig
        self.drv = LiveDriver(state, ai_side, self.path,
                              randomness=self.rand)
        if phase:
            self.drv.phase = phase

    def feed(self, *lines):
        with open(self.path, 'a', encoding='utf-8') as f:
            for ln in lines:
                f.write(ln + '\n')
        return self.drv.tick()


def _army_window_lines(st):
    """Client lines for a REAL menu army_move (matcher-test recipe)."""
    import ai
    from test_live_matcher import _first_simple_move, token_for
    menu = ai.get_legal_actions(st, Side.SHADOW)
    a = _first_simple_move(st, menu)
    if a is None:
        return None, None
    p = a.params
    from board import sanitize
    nat = None
    for k in st.armies:
        if sanitize(k) == sanitize(p['from']):
            for u in st.armies[k].units:
                nat = u.nation
    kind_word = 'Elite' if p.get('n_elite') else 'Regular'
    count = p.get('n_elite') or p.get('n_regular')
    tok = token_for(nat, kind_word, count)
    lines = [
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves %s from %s to %s.' % (tok, str(p['from']).title(),
                                          str(p['to']).title()),
        '$Wey kills Shadow Action Die (A) in Current Action.',
    ]
    return a, lines


def test_window_commit(fails):
    """MEASURES: a human gesture window flows feed->match->COMMIT: the
    driver's engine state advances to the matched action's post state
    (asserted on the engine's own army rows), one 'window' commit is
    recorded, no alarms."""
    st = adopted_state()
    pre = client_board.proj_board(st)
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('commit_precondition', 'no simple army_move'))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    rig.feed(*lines)
    post = client_board.proj_board(rig.drv.es)
    d = client_board.board_delta(st, rig.drv.es) \
        if rig.drv.es is not st else {}
    if rig.drv.es is st or pre == post:
        fails.append(('commit_advances', 'engine state unchanged'))
    if not rig.drv.commits or rig.drv.commits[-1][2] != 'window':
        fails.append(('commit_ledger',
                      repr([c[2] for c in rig.drv.commits])))
    if rig.drv.alarms:
        fails.append(('commit_alarms', repr(rig.drv.alarms)))


def test_undo_per_line(fails):
    """MEASURES fact 1 (per-LINE client undo): three undo records in
    LIFO order restore the pre-window engine state EXACTLY (projection
    equality); one or two presses do NOT pop (partial unwind)."""
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('undo_precondition', 'no simple army_move'))
        return
    import ai
    ref = ai.clone_state(st)
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    rig.feed(*lines)
    mid = rig.drv.es
    rig.feed('$Wey undoes last action : %s' % lines[2].lstrip('$'))
    if rig.drv.es is not mid:
        fails.append(('undo_partial_pop', 'popped after 1 press'))
    rig.feed('$Wey undoes last action : %s' % lines[1].lstrip('$'))
    rig.feed('$Wey undoes last action : %s' % lines[0].lstrip('$'))
    got = client_board.proj_board(rig.drv.es)
    want = client_board.proj_board(ref)
    if got != want:
        fails.append(('undo_restores', 'projection differs after '
                                       'full unwind'))
    if rig.drv.commits and rig.drv.commits[-1][2] == 'window':
        fails.append(('undo_ledger', 'window commit not popped'))


def test_undo_mismatch_alarm(fails):
    """MEASURES (sensitivity): an undo record whose inner text is not
    the expected tail ALARMS and does NOT pop (never a guessed
    rollback)."""
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('mismatch_precondition', 'no simple army_move'))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    rig.feed(*lines)
    mid = rig.drv.es
    out = rig.feed('$Wey undoes last action : Wey moves Something '
                   'Else from A to B.')
    if rig.drv.es is not mid:
        fails.append(('mismatch_no_pop', 'engine popped on mismatch'))
    if not any('UNDO ORDER MISMATCH' in x for x in rig.drv.alarms):
        fails.append(('mismatch_alarm', repr(rig.drv.alarms)))


def test_owed_draw_gate(fails):
    """MEASURES: turn-start draws (one per deck) are accepted; a
    second same-deck draw is refused UNLESS the engine's owed-draw
    ledger covers it (A13 end-to-end through the driver)."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')   # human = FP
    n0 = len(st.hands[Side.FREE_PEOPLES].hand)
    rig.feed('$Wey draws Free Character Card.')
    if len(rig.drv.es.hands[Side.FREE_PEOPLES].hand) != n0 + 1:
        fails.append(('draw_turnstart', 'first draw not applied'))
    out = rig.feed('$Wey draws Free Character Card.')
    if len(rig.drv.es.hands[Side.FREE_PEOPLES].hand) != n0 + 1:
        fails.append(('draw_unowed_refused', 'un-owed draw applied'))
    if not any(k == 'unowed_draw_refused' for k, _ in rig.drv.findings):
        fails.append(('draw_refusal_finding', repr(rig.drv.findings)))
    if any('UN-OWED' in x for x in rig.drv.alarms):
        fails.append(('draw_refusal_alarm_d_t2_4', repr(rig.drv.alarms)))
    # engine grants a draw (bid token class) -> the SAME line is owed
    rig.drv.es._owed_draws = {(Side.FREE_PEOPLES.name,
                               'character'): 1}
    rig.feed('$Wey draws Free Character Card.')
    if len(rig.drv.es.hands[Side.FREE_PEOPLES].hand) != n0 + 2:
        fails.append(('draw_owed_accepted', 'owed draw refused'))
    hand = rig.drv.es.hands[Side.FREE_PEOPLES].hand
    if not any(str(c).startswith('HU_') for c in hand):
        fails.append(('draw_placeholder', 'no placeholder minted: %r'
                      % hand[-3:]))


def test_fp_ok_and_roll(fails):
    """MEASURES: 'fp ok' closes the declaration window (D2); the D3
    specimen roll line adopts faces (rolled Eye -> Hunt Box) and, with
    only the SHADOW pool rolled, leaves the phase at 'roll' -- action
    resolution opens only once the human's FP pool rolls too.

    This leg used to assert 'actions' after that single roll, which
    ENCODED the turn-order defect the USER hit on 2026-08-20 (the AI
    took every action of the turn because an unrolled FP pool reads to
    the engine exactly like an exhausted one).  See
    driver.pools_rolled and test_live_server's
    test_action_phase_waits_for_both_rolls."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='declare')  # human = FP
    rig.feed('<Wey> fp ok')
    if rig.drv.phase != 'eyes':
        fails.append(('fp_ok_phase', rig.drv.phase))
    hb0 = len(rig.drv.es.hunt_box)
    rig.feed('$Wey rolls P C A E P P H .')
    dice = rig.drv.es.dice[Side.SHADOW]
    if len(rig.drv.es.hunt_box) != hb0 + 1:
        fails.append(('roll_eye_hunt', 'rolled Eye not in Hunt Box'))
    set_faces = [d.current for d in dice if d.current is not None]
    if len(set_faces) != 7:
        fails.append(('roll_adopts', '%d faces set' % len(set_faces)))
    if rig.drv.phase != 'roll':
        fails.append(('roll_phase', rig.drv.phase))
    # ...and the FP pool's own roll opens resolution
    fp_unset = [d for d in rig.drv.es.dice[Side.FREE_PEOPLES]
                if not d.used and d.current is None]
    rig.feed('$Wey rolls %s .' % ' '.join(['A'] * len(fp_unset)))
    if rig.drv.phase != 'actions':
        fails.append(('roll_phase_both', rig.drv.phase))


def test_eye_allocation(fails):
    """MEASURES (D3, specimen game 2023-01-02 turn 1): a human-Shadow
    'allocates an eye.' line flips ONE unset Shadow die to Eye in the
    Hunt Box (engine allocate_eyes), so the subsequent 6-face roll
    line resolves its roller by unset-dice count and adopts.  Without
    the allocation the roll side is unresolvable and the whole turn's
    menus are empty (the first-census root failure)."""
    st = fresh_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='eyes')   # human = SHADOW
    hb0 = len(st.hunt_box)
    rig.feed('$Wey allocates an eye.')
    if len(rig.drv.es.hunt_box) != hb0 + 1:
        fails.append(('alloc_hunt_box', len(rig.drv.es.hunt_box)))
        return
    if rig.drv.es.eyes_allocated != 1:
        fails.append(('alloc_counter', rig.drv.es.eyes_allocated))
    # the 6-face roll now matches the 6 remaining unset Shadow dice
    rig.feed('$Wey rolls M P P A A H .')
    dice = rig.drv.es.dice[Side.SHADOW]
    set_faces = [d.current for d in dice if d.current is not None]
    if len(set_faces) != 7:      # 6 adopted + 1 allocated Eye
        fails.append(('alloc_roll_adopts', '%d faces set'
                      % len(set_faces)))
    if any('ROLL side unresolved' in x for x in rig.drv.alarms):
        fails.append(('alloc_roll_side', repr(rig.drv.alarms)))
    # SPECIFICITY: the AI's own allocation echo must NOT double-apply
    st2 = fresh_state()
    rig2 = Rig(st2, Side.SHADOW, phase='eyes')       # AI = SHADOW
    rig2.drv.nick = 'WOTR_AI'
    rig2.feed('$WOTR_AI_Shadow allocates an eye.')
    if len(rig2.drv.es.hunt_box) != hb0:
        fails.append(('alloc_echo_guard', len(rig2.drv.es.hunt_box)))


_FS_WINDOW = [
    '$Wey moves Free Action Die (C) from FP Dice to Current Action.',
    '$Wey moves FSP from Fellowship Track 0 to Fellowship Track 1.',
    '$Wey moves Free Action Die (C) from Current Action to The Hunt '
    'For The Ring.',
]


def _hunt_armed_state():
    """Adopted state with ONE eye in the Hunt Box and an empty FP hand
    (no play_event candidates -> the only rng-starving matcher probe
    is the fellowship move itself, so rand.pending() is deterministic)."""
    st = adopted_state()
    game_logic.allocate_eyes(st, 1, indices=[2])
    st.hands[Side.FREE_PEOPLES].hand[:] = []
    return st


def test_window_rng_deferral(fails):
    """MEASURES (PHASE3-D port, mechanism 1): a human fellowship-move
    window whose hunt roll renders AFTER the terminator (an eye is in
    the Hunt Box) is DEFERRED — engine state untouched, NO refusal —
    with the starving d6 ask surfaced; the '<game> ... rolls' phase
    line then retries and TRANSACTIONALLY commits the move.
    SENSITIVITY: while the roll is missing, nothing commits and no
    undo press is emitted (the pre-port driver mis-refused here).
    Honesty: the fed roll is fully consumed at commit."""
    st = _hunt_armed_state()
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    out = rig.feed(*_FS_WINDOW)
    if rig.drv.deferred is None:
        fails.append(('deferral_armed', 'window not deferred: %r'
                      % (out,)))
        return
    if rig.drv.es.fellowship.progress != 0:
        fails.append(('deferral_no_commit',
                      rig.drv.es.fellowship.progress))
    if any(item[0] == 'line' for item in out):
        fails.append(('deferral_no_refusal',
                      [i for i in out if i[0] == 'line']))
    if rig.drv.window_stats['none']:
        fails.append(('deferral_not_none',
                      dict(rig.drv.window_stats)))
    asks = [i for i in out if i[0] == 'rng_ask']
    if not asks or asks[-1][1].kind != 'd6':
        fails.append(('deferral_ask', repr(asks)))
    # WAVE-J (wjg12, THIRD live expiry): the deferral must carry the
    # starving CALL OBJECT itself - the matcher's backtracking
    # restores rand._pending to None, and a deferral without its call
    # can never re-order the ask once the global slot is clobbered
    sc = getattr(rig.drv.deferred[0], '_starve_call', None)
    if sc is None or getattr(sc, 'kind', None) != 'd6':
        fails.append(('deferral_starve_call', repr(sc)))
    # the client's roll arrives as a phase record -> retry -> commit
    rig.feed('<game> Wey rolls 1')
    if rig.drv.deferred is not None:
        fails.append(('deferral_settles', 'still deferred'))
    if rig.drv.es.fellowship.progress != 1:
        fails.append(('deferral_commit',
                      rig.drv.es.fellowship.progress))
    if rig.drv.window_stats['matched'] != 1:
        fails.append(('deferral_stats', dict(rig.drv.window_stats)))
    if not rig.drv.commits or rig.drv.commits[-1][2] != 'window':
        fails.append(('deferral_ledger',
                      [c[2] for c in rig.drv.commits]))
    exc = rig.rand.unconsumed()
    if any(exc.values()):
        fails.append(('deferral_honesty', exc))


def test_fs_window_with_cards_in_hand_defers_on_hunt_not_draw(fails):
    """MEASURES (BL-134, fpgame19 T1 / fpgame18 T2 - the phantom
    draw demand's true root): a human fellowship-move window on a
    WILL die, one eye in the Hunt Box, Gandalf the Grey guiding and a
    playable STRATEGY card in the hand.  The play_event candidate
    would starve on the same-deck redraw (draws) AFTER the
    fellowship_move starved on the hunt d6 - last-starve-wins made
    the deferral wait on 'client card draw ([(FREE_PEOPLES,
    strategy)])' that nobody owed (turn 1, first action, 47s stall;
    feeding it cost an illegal third card + 'deck fp strat deck:
    engine=23 client=22').  CONTRACT: no card left the hand -> no
    play_event candidate is probed; the deferral's starve call is the
    hunt d6; the rendered hunt roll commits the move; the hand is
    untouched; nothing demanded a draw.  Controls: the play_event
    must be ON the menu for that die (else NOT-MEASURED)."""
    import ai
    st = adopted_state()
    st.dice[Side.FREE_PEOPLES][0].current = ActionDieFace.WILL_OF_WEST
    game_logic.allocate_eyes(st, 1, indices=[2])
    g = st.fellowship.guide
    if getattr(g, 'name', '') != 'Gandalf the Grey':
        fails.append(('fswc_control', 'NOT-MEASURED: guide is %r'
                      % getattr(g, 'name', None)))
        return
    cid = 'FREE_PEOPLES_Strategy_Kindred_of_Glorfindel'
    if cid in st.fp_strat_deck:
        st.fp_strat_deck.remove(cid)
    st.hands[Side.FREE_PEOPLES].hand[:] = [cid]
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    with rig.rand.pin():
        menu = ai.get_legal_actions(drv.es, Side.FREE_PEOPLES)
    if not any(a.action_type == 'play_event' and a.die_idx == 0
               and (a.params or {}).get('card_id') == cid
               for a in menu):
        fails.append(('fswc_control', 'NOT-MEASURED: no play_event '
                      'for the Kindred on FP die 0'))
        return
    win = [
        '$Wey moves Free Action Die (W) from FP Dice to Current '
        'Action.',
        '$Wey moves FSP from Fellowship Track 0 to Fellowship Track 1.',
        '$Wey moves Free Action Die (W) from Current Action to The '
        'Hunt For The Ring.',
    ]
    out = rig.feed(*win)
    if drv.deferred is None:
        fails.append(('fswc_deferred', 'window not deferred: %r %r'
                      % (dict(drv.window_stats), drv.findings[-3:])))
        return
    sc = getattr(drv.deferred[0], '_starve_call', None)
    if sc is None or getattr(sc, 'kind', None) != 'd6':
        fails.append(('fswc_starve_kind',
                      'deferral asks for %r (want the hunt d6)'
                      % (getattr(sc, 'kind', sc),)))
    asks = [i for i in out if i[0] == 'rng_ask']
    if any(getattr(a[1], 'kind', None) in ('draws', 'card')
           for a in asks):
        fails.append(('fswc_draw_asked', [getattr(a[1], 'kind', None)
                                          for a in asks]))
    rig.feed('<game> Wey rolls 1')
    if drv.deferred is not None or drv.es.fellowship.progress != 1:
        fails.append(('fswc_commit', (drv.deferred is not None,
                                      drv.es.fellowship.progress,
                                      drv.findings[-3:])))
    if list(drv.es.hands[Side.FREE_PEOPLES].hand) != [cid]:
        fails.append(('fswc_hand', list(drv.es.hands[
            Side.FREE_PEOPLES].hand)))
    if any(k in ('demanded_draw_fed', 'post_terminator_draw_fed')
           for k, _ in drv.findings):
        fails.append(('fswc_draw_lane', list(drv.findings[-4:])))
    exc = rig.rand.unconsumed()
    if any(exc.values()):
        fails.append(('fswc_honesty', exc))


def test_pending_phase_buffer(fails):
    """MEASURES (PHASE3-D port, mechanism 2): a hand-limit discard —
    an engine pending opened by the human's own over-draw — resolves
    from the PHASE record stream: the identity-less 'kills ... in
    Free Peoples Hand.' record buffers, matches the pending's own menu
    (placeholder-preferred) and commits (hand back to 6, pending
    closed).  SENSITIVITY: the pending stays open until the discard
    record arrives.  SPECIFICITY: a dice-bookkeeping move while the
    pending is open is NOT buffered (the guide-swallowed-recover
    class) and the pending survives it."""
    import ai
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    drv.es._owed_draws = {(Side.FREE_PEOPLES.name, 'character'): 10}
    n_guard = 0
    while len(drv.es.hands[Side.FREE_PEOPLES].hand) <= 6:
        rig.feed('$Wey draws Free Character Card.')
        n_guard += 1
        if n_guard > 10:
            fails.append(('buffer_precondition', 'hand never crossed'))
            return
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'discard':
        fails.append(('buffer_pending_open', repr(dp and dp['kind'])))
        return
    n7 = len(drv.es.hands[Side.FREE_PEOPLES].hand)
    # SPECIFICITY: dice bookkeeping must not enter the buffer
    rig.feed('$Wey moves Free Action Die (P) from Used Dice to '
             'FP Dice.')
    if drv.pending_buf is not None and drv.pending_buf.records:
        fails.append(('buffer_dice_excluded',
                      [r.get('norm') for r in
                       drv.pending_buf.records]))
    if ai.describe_pending(drv.es) is None:
        fails.append(('buffer_pending_survives', 'pending vanished'))
        return
    # the human's discard gesture: an identity-less deck-label kill
    rig.feed('$Wey kills Free Character Card in Free Peoples Hand.')
    if ai.describe_pending(drv.es) is not None:
        fails.append(('buffer_resolved',
                      repr(ai.describe_pending(drv.es)['kind'])))
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n7 - 1:
        fails.append(('buffer_hand_len',
                      len(drv.es.hands[Side.FREE_PEOPLES].hand)))
    if drv.pending_stats['matched'] < 1:
        fails.append(('buffer_stats', dict(drv.pending_stats)))
    if not drv.commits or drv.commits[-1][2] != 'pending:discard':
        fails.append(('buffer_ledger', [c[2] for c in drv.commits]))


def test_d6_side_binding(fails):
    """MEASURES (PHASE3-D port, mechanism 3): roller nicks bind to
    sides from ZONE-attributed draw lines (the deck names the owner,
    fact 3), so '<game> X rolls N' d6 values land side-tagged in the
    feed lanes.  SPECIFICITY (the attribution ruling): a nick that
    draws for BOTH sides — the one-client sided game, where the local
    seat's nick stamps the AI's !CALL draws too — is POISONED to
    untagged, never last-wins flipped (a FP-tagged value would starve
    a Shadow hunt-roll consumer forever)."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')       # ai = SHADOW
    drv = rig.drv
    rig.feed('$Alice draws Free Character Card.')       # human draw
    rig.feed('$Bob draws Shadow Character Card.')       # AI echo
    if drv.nick_side.get('Alice') != 'FREE_PEOPLES' \
            or drv.nick_side.get('Bob') != 'SHADOW':
        fails.append(('bind_zone', dict(drv.nick_side)))
        return
    rig.feed('<game> Alice rolls 3 4')
    rig.feed('<game> Bob rolls 2')
    tags = [t for (t, _v) in rig.rand._feed['d6']]
    if tags != ['FREE_PEOPLES', 'FREE_PEOPLES', 'SHADOW']:
        fails.append(('bind_tags', rig.rand._feed['d6']))
    # the conflict: 'Alice' now draws for the SHADOW side too
    rig.feed('$Alice draws Shadow Strategy Card.')
    if 'Alice' not in drv.nick_side \
            or drv.nick_side['Alice'] is not None:
        fails.append(('bind_poison', dict(drv.nick_side)))
    rig.feed('<game> Alice rolls 5')
    if rig.rand._feed['d6'][-1] != (None, 5):
        fails.append(('bind_poison_untagged',
                      rig.rand._feed['d6'][-1]))
    # a poisoned nick must stay poisoned (no re-bind on a later draw)
    rig.feed('$Alice draws Free Strategy Card.')
    if drv.nick_side['Alice'] is not None:
        fails.append(('bind_poison_sticky', dict(drv.nick_side)))


def test_alternation_wakes_ai(fails):
    """MEASURES the PHASE3-B gap fix: after a committed HUMAN die
    action with no pending open, the driver surfaces the AI's plain
    action turn via the ONE engine answerer (ai._next_acting_side) --
    an ('ai_decide', kind='action') item with a non-empty menu.
    Specificity: before any commit (last_actor None) no such item."""
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('alt_precondition', 'no simple army_move'))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    pre = rig.drv.tick()
    if any(i[0] == 'ai_decide' for i in pre):
        fails.append(('alt_specificity',
                      'ai_decide before any commit'))
    out = rig.feed(*lines)
    dps = [i[1] for i in out if i[0] == 'ai_decide']
    if not dps:
        fails.append(('alt_wakes', 'no ai_decide after human commit: '
                      '%r' % [i[0] for i in out]))
        return
    dp = dps[-1]
    if dp['kind'] != 'action' or dp['side'] is not Side.FREE_PEOPLES             or not dp['actions']:
        fails.append(('alt_dp', '%r n=%d' % (dp['kind'],
                                             len(dp['actions']))))


def test_human_plain_pass(fails):
    """MEASURES (WAVE-I, WAVE-H blocker 4): chat 'pass' with no
    pending open, during the human's alternation turn, matches the
    ENGINE MENU's own pass action (1.223: 0 < my dice < theirs) and
    commits — alternation flips to the human.  SPECIFICITY: out of
    turn -> chat refusal, nothing committed; pass absent from the
    menu -> chat refusal (the rule lives IN the menu, never a driver
    counter)."""
    # legal: FP has 1 unused < Shadow's 2 -> the menu carries pass
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    rig.drv.last_actor = Side.SHADOW                # AI just acted
    out = rig.feed('<Wey> pass')
    if not rig.drv.commits or rig.drv.commits[-1][2] != 'chat-pass':
        fails.append(('pass_commit',
                      [c[2] for c in rig.drv.commits]))
    if rig.drv.last_actor is not Side.FREE_PEOPLES:
        fails.append(('pass_alternates', rig.drv.last_actor))
    if any(i[0] == 'chat' and i[1] == 'refusal' for i in out):
        fails.append(('pass_no_refusal', out))
    # out of turn: the engine says the AI acts next
    st2 = adopted_state()
    rig2 = Rig(st2, Side.SHADOW, phase='actions')
    rig2.drv.last_actor = Side.FREE_PEOPLES         # next = SHADOW
    out2 = rig2.feed('<Wey> pass')
    if rig2.drv.commits:
        fails.append(('pass_oot_commit',
                      [c[2] for c in rig2.drv.commits]))
    if not any(i[0] == 'chat' and i[1] == 'refusal' for i in out2):
        fails.append(('pass_oot_refusal', out2))
    # illegal per the menu: FP unused (2) >= Shadow unused (0)
    st3 = fresh_state()
    game_logic.adopt_rolled_faces(st3, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER),
        (Side.FREE_PEOPLES, 1, ActionDieFace.CHARACTER),
    ])
    rig3 = Rig(st3, Side.SHADOW, phase='actions')
    rig3.drv.last_actor = Side.SHADOW
    out3 = rig3.feed('<Wey> pass')
    if rig3.drv.commits:
        fails.append(('pass_illegal_commit',
                      [c[2] for c in rig3.drv.commits]))
    if not any(i[0] == 'chat' and i[1] == 'refusal'
               and 'pass' in str(i[2]) for i in out3):
        fails.append(('pass_illegal_refusal', out3))


def _pop_unit(post, nation_name, kind_name):
    """Remove one (nation, kind) unit from the first army holding one;
    returns the army key (engine-forced-allocation simulation)."""
    for k, a in post.armies.items():
        for i, u in enumerate(a.units):
            if u.nation.name == nation_name \
                    and u.kind.name == kind_name:
                a.units.pop(i)
                return k
    return None


def test_fp_acts_first(fails):
    """MEASURES (WAVE-J, USER live specimen: 'shadow is not supposed
    to go first' - the AI mustered before the human's first action):
    the FREE PEOPLES side takes the FIRST action of every action-
    resolution phase, SIDE-based never seat-based (the certified
    machine's convention, ai.py:15891 `active = Side.FREE_PEOPLES`,
    mirrored by seeding last_actor=SHADOW at action-phase entry).
    (a) AI=SHADOW with last_actor carried as FREE_PEOPLES from the
        previous turn (the wjg6/wjg7 defect shape): after adoption
        the AI WAITS (ai_action_dp None);
    (b) AI=FREE_PEOPLES: after adoption the AI fires first."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='roll')
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES     # carried across rollover
    rig.feed('$Wey rolls A C H M A C H .')
    rig.feed('$Wey rolls C C M W .')
    if drv.phase != 'actions':
        fails.append(('fpfirst_phase', drv.phase))
        return
    if drv.last_actor is not Side.SHADOW:
        fails.append(('fpfirst_seed', drv.last_actor))
    dp = drv.ai_action_dp()
    if dp is not None:
        fails.append(('fpfirst_ai_waits',
                      'AI=SHADOW acted FIRST: %r'
                      % (dp and dp.get('kind'),)))
    # (b) the AI holding FREE PEOPLES acts first
    st2 = fresh_state()
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='roll')
    drv2 = rig2.drv
    drv2.last_actor = Side.SHADOW
    rig2.feed('$Wey rolls A C H M A C H .')
    rig2.feed('$Wey rolls C C M W .')
    dp2 = drv2.ai_action_dp()
    if dp2 is None:
        fails.append(('fpfirst_ai_fp_fires',
                      'AI=FP did not act first'))


def test_cc_reveal_echo(fails):
    """MEASURES: the AI's OWN combat-card reveal echo ('<auto> moves
    <card> from Shadow Hand (chosen) to Current SA Card.') arriving
    while the HUMAN's combat-card decision is open (an engine-built
    battle: the AI attacks, declines its card, the defender's decision
    is the human's) is consumed as a SELF echo (finding cc_reveal_echo)
    and never buffered into the human's open pending.  SPECIFICITY: the
    HUMAN side's own '(chosen)' reveal line is NOT eaten by this
    filter."""
    got = _human_defender_cc_rig(fails, 'cc_echo_precondition')
    if got is None:
        return
    rig, drv = got
    rig.feed('<auto> moves The Shadow is Moving/Swarm of Bats from '
             'Shadow Hand (chosen) to Current SA Card.')
    if drv.pending_buf is not None and drv.pending_buf.records:
        fails.append(('cc_echo_buffered',
                      [r.get('norm') for r in
                       drv.pending_buf.records]))
    if not any(k == 'cc_reveal_echo' for k, _ in drv.findings):
        fails.append(('cc_echo_row', drv.findings[-3:]))
    pc = drv.es.pending_combat or {}
    if pc.get('phase') != 'defender_combat_card':
        fails.append(('cc_echo_pending_disturbed', pc.get('phase')))
    # SPECIFICITY: the HUMAN's (chosen) reveal is not eaten
    n_rows = len([1 for k, _ in drv.findings
                  if k == 'cc_reveal_echo'])
    rig.feed('<auto> moves Test Card/Other from Free Peoples Hand '
             '(chosen) to Current FP Card.')
    n_rows2 = len([1 for k, _ in drv.findings
                   if k == 'cc_reveal_echo'])
    if n_rows2 != n_rows:
        fails.append(('cc_echo_specificity',
                      'the human reveal was eaten'))


def test_two_army_moves_one_die(fails):
    """BL-78 (shgame9 D2, rulebook p.22): ONE Army-family die moves up
    to TWO different armies.  The human's physical window carries both
    drags before the die kill; the chain matcher commits both legs and
    the multi-step half closes (no phantom open pending)."""
    import ai
    from board import sanitize
    from test_live_matcher import _first_simple_move, token_for
    st = adopted_state()
    menu = ai.get_legal_actions(st, Side.SHADOW)
    a1 = _first_simple_move(st, menu)
    if a1 is None:
        fails.append(('two_move_precondition', 'no simple army_move'))
        return
    # a second simple move from a DIFFERENT army, disjoint regions
    used = {sanitize(str(a1.params['from'])),
            sanitize(str(a1.params['to']))}
    a2 = None
    for a in menu:
        if a.action_type != 'army_move' or a.die_idx != a1.die_idx:
            continue
        p = a.params
        if sanitize(str(p['from'])) in used \
                or sanitize(str(p['to'])) in used:
            continue
        key = next((k for k in st.armies
                    if sanitize(k) == sanitize(p['from'])), None)
        if key is None or getattr(st.armies[key], 'leaders', None):
            continue
        if p.get('n_regular', 0) + p.get('n_elite', 0) >= 1:
            a2 = a
            break
    if a2 is None:
        fails.append(('two_move_precondition', 'no second disjoint '
                                               'simple army_move'))
        return

    def _tok(p):
        nat = None
        for k in st.armies:
            if sanitize(k) == sanitize(p['from']):
                for u in st.armies[k].units:
                    nat = u.nation
        kind_word = 'Elite' if p.get('n_elite') else 'Regular'
        return token_for(nat, kind_word,
                         p.get('n_elite') or p.get('n_regular'))
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    rig.feed(
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves %s from %s to %s.' % (
            _tok(a1.params), str(a1.params['from']).title(),
            str(a1.params['to']).title()),
        '$Wey moves %s from %s to %s.' % (
            _tok(a2.params), str(a2.params['from']).title(),
            str(a2.params['to']).title()),
        '$Wey kills Shadow Action Die (A) in Current Action.',
    )
    es = rig.drv.es
    if es is st:
        fails.append(('two_move_commit', 'window did not commit'))
        return
    for tag, p in (('leg1', a1.params), ('leg2', a2.params)):
        dst = next((k for k in es.armies
                    if sanitize(k) == sanitize(str(p['to']))), None)
        n_units = (len(es.armies[dst].units) if dst is not None
                   else 0)
        if n_units < p.get('n_regular', 0) + p.get('n_elite', 0):
            fails.append(('two_move_' + tag,
                          'destination %r has %d units'
                          % (p['to'], n_units)))
    if getattr(es, 'pending_army_move_die', None) is not None:
        fails.append(('two_move_pending_open',
                      es.pending_army_move_die))
    if rig.drv.alarms:
        fails.append(('two_move_alarms', rig.drv.alarms[:3]))


def test_fellowship_death_banks_named_demand(fails):
    """fpgame16 P1: a guide SACRIFICE (hunt_sacrifice_guide via the
    pick lane) kills the companion engine-side with NO army delta -
    the sweep must bank the named kill demand or the dead guide
    stands in the client guide box forever."""
    import ai
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    _g = next((c for c in post.fellowship.companions
               if 'Gandalf' in str(getattr(c, 'name', ''))), None)
    if _g is None:
        fails.append(('fdb_precondition', 'no Gandalf in setup'))
        return
    post.fellowship.companions = [
        c for c in post.fellowship.companions if c is not _g]
    if getattr(getattr(post.fellowship, 'guide', None), 'name',
               '') == getattr(_g, 'name', ''):
        post.fellowship.guide = next(
            iter(post.fellowship.companions), None)
    import log_replay as lr
    gpid = next((p for p in lr.COMPANIONS
                 if getattr(_g, 'name', '') in
                 set(lr.names_for_pid(str(p)) or ())), None)
    if gpid is not None:
        drv.view.piece_region[str(gpid)] = 'Guide Of The Fellowship'
    drv.post_commit_sweep(pre, post, label='chat:chat')
    rows = [e for e in drv.echo_ledger
            if e.get('kind') == 'named'
            and str(e.get('tkey', '')) == str(getattr(_g, 'name',
                                                      ''))]
    if not rows:
        fails.append(('fdb_not_banked',
                      [dict(e) for e in drv.echo_ledger][:4]))


def test_two_recruits_one_muster_die(fails):
    """BL-91/N2 (shgame14, rulebook 5.3): one MUSTER die places TWO
    regulars in up to two different settlements - the physical window
    (stage + recruit1 + recruit2 + kill) must commit both legs
    through the engine's pending_muster continuation, like the
    two-army pin."""
    import ai
    from board import sanitize
    from game_state import Nation
    st = fresh_state()
    st.political.boxes[Nation.ISENGARD] = 3
    from game_state import ActionDieFace as _F
    import game_logic as _gl
    _gl.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, _F.MUSTER),
        (Side.FREE_PEOPLES, 0, _F.CHARACTER),
    ])
    menu = ai.get_legal_actions(st, Side.SHADOW)
    r1 = next((a for a in menu if a.action_type == 'muster_recruit'
               and sanitize(str(a.params.get('region')))
               == 'orthanc'), None)
    r2 = next((a for a in menu if a.action_type == 'muster_recruit'
               and sanitize(str(a.params.get('region')))
               == 'north dunland'), None)
    if r1 is None or r2 is None:
        fails.append(('recruit2_precondition',
                      'menu lacks rows: orthanc=%s north=%s'
                      % (r1 is not None, r2 is not None)))
        return

    def _n(es, reg):
        a = next((ar for k, ar in es.armies.items()
                  if sanitize(k) == reg), None)
        return len(a.units) if a is not None else 0
    n_orth0 = _n(st, 'orthanc')
    n_nd0 = _n(st, 'north dunland')
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    rig.feed(
        '$Wey moves Shadow Action Die (M) from SA Dice to Current '
        'Action.',
        '$Wey moves Isengard Regular from SA Reinforcements to '
        'Orthanc.',
        '$Wey moves Isengard Regular from SA Reinforcements to '
        'North Dunland.',
        '$Wey kills Shadow Action Die (M) in Current Action.',
    )
    es = rig.drv.es
    if es is st:
        fails.append(('recruit2_commit',
                      'window did not commit; alarms=%r'
                      % rig.drv.alarms[:2]))
        return
    if _n(es, 'orthanc') != n_orth0 + 1 \
            or _n(es, 'north dunland') != n_nd0 + 1:
        fails.append(('recruit2_board',
                      'orthanc %d->%d north %d->%d (want +1 each)'
                      % (n_orth0, _n(es, 'orthanc'),
                         n_nd0, _n(es, 'north dunland'))))
    if getattr(es, 'pending_muster_die', None) is not None:
        fails.append(('recruit2_pending_open',
                      es.pending_muster_die))


def test_progress_sweep_destination_aware(fails):
    """BL-101 (fpgame11 I1): the reveal's progress RESET (pre=2 ->
    post=0) must walk marker 248 even when (a) the commit's own
    gesture rendered the 1->2 walk and (b) an old feed line contains
    the SOURCE phrase 'from Fellowship Track 0'.  The old coverage
    checks were destination-blind: both shapes false-covered the
    reset, the client marker stayed at 2, and every later fellowship
    window refused 'differs on Fellowship progress'."""
    import ai
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    drv.feed.lines.append(
        '$Wey moves FSP from Fellowship Track 0 to Fellowship '
        'Track 1.')

    class _G(object):
        die_side = None
        die_face = None
        closed = 'die_killed'
        records = [{'norm': 'moves FSP from Fellowship Track 1 to '
                            'Fellowship Track 2.', 'op': 'move'}]

        def body_records(self):
            return self.records
    pre = ai.clone_state(drv.es)
    pre.fellowship.progress = 2
    post = ai.clone_state(drv.es)
    post.fellowship.progress = 0
    items = drv.post_commit_sweep(pre, post, gesture=_G(),
                                  label='window')
    walk = [i for i in items
            if i[0] == 'line' and 'Game.2014' in str(i[1])
            and '248' in str(i[1])]
    if not walk:
        fails.append(('progress_reset_not_walked',
                      [str(i)[:80] for i in items[:6]]))
    if (getattr(drv, '_track_owed', None) or {}).get('248', (None,
                                                            0))[0] \
            != 0:
        fails.append(('progress_reset_not_owed',
                      getattr(drv, '_track_owed', None)))


def test_card_leg_gate_wiring(fails):
    """fpgame11 (wave-18): the BL-94 tolerance shipped keyed on
    `rng is None`, which is NEVER true in a live driver (rctl is
    always set) - the flag was INERT in the exact games it was built
    for, and no pin existed to notice.  Pin the WIRING: a live-shaped
    match (Rig passes rng + advance_tolerance=True) must reach
    _covers with card_leg_tolerance=True."""
    from wotr_bridge.live import gesture_match as gm
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('wiring_precondition', 'no simple army_move'))
        return
    seen = []
    orig = gm._covers

    def spy(win, eng, **kw):
        # the live wiring passes the FOLLOW_THROUGH table lookup (a
        # frozenset, possibly empty); the spectator passes None
        seen.append(kw.get('follow_through') is not None)
        return orig(win, eng, **kw)
    gm._covers = spy
    try:
        rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
        rig.feed(*lines)
    finally:
        gm._covers = orig
    if not seen:
        fails.append(('wiring_no_covers_calls', 'match never ran'))
    elif True not in seen:
        fails.append(('wiring_gate_inert', seen[:5]))


def test_two_army_moves_vacated_source(fails):
    """BL-99 (shgame11 H1-3): the LIVE refusal shape - leg 2 moves
    INTO leg 1's just-vacated source (South Dunland -> Gap of Rohan,
    then North Dunland -> South Dunland) on ONE Army die.  The
    matcher refused this twice live ('differs on army units, the
    action die') while accepting leg 1 alone and while the AI played
    the same two-drag shape - if the engine's second-leg enumeration
    forbids the vacated source, this pin catches the menu gap."""
    import ai
    from board import sanitize
    from game_state import Nation, UnitKind
    from test_live_engine_fixwave import place, set_dice, Face
    st = fresh_state()
    st.political.boxes[Nation.ISENGARD] = 3
    place(st, 'south dunland', Nation.ISENGARD, regulars=1,
          clear=True)
    place(st, 'north dunland', Nation.ISENGARD, regulars=1,
          clear=True)
    set_dice(st, Side.SHADOW, [Face.ARMY])
    set_dice(st, Side.FREE_PEOPLES, [Face.CHARACTER])
    menu = ai.get_legal_actions(st, Side.SHADOW)
    a1 = next((a for a in menu if a.action_type == 'army_move'
               and sanitize(str(a.params.get('from'))) ==
               'south dunland'
               and sanitize(str(a.params.get('to'))) ==
               'gap of rohan'
               and a.params.get('n_regular') == 1), None)
    a2 = next((a for a in menu if a.action_type == 'army_move'
               and sanitize(str(a.params.get('from'))) ==
               'north dunland'
               and sanitize(str(a.params.get('to'))) ==
               'south dunland'
               and a.params.get('n_regular') == 1), None)
    if a1 is None or a2 is None:
        fails.append(('vac_precondition',
                      'menu lacks the legs: a1=%s a2=%s'
                      % (a1 is not None, a2 is not None)))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    rig.feed(
        '$Wey moves Shadow Action Die (A) from SA Dice to Current '
        'Action.',
        '$Wey moves Isengard Regular from South Dunland to '
        'Gap of Rohan.',
        '$Wey moves Isengard Regular from North Dunland to '
        'South Dunland.',
        '$Wey kills Shadow Action Die (A) in Current Action.',
    )
    es = rig.drv.es
    if es is st:
        fails.append(('vac_commit', 'window did not commit; alarms=%r'
                      % rig.drv.alarms[:2]))
        return

    def _n(reg):
        a = next((ar for k, ar in es.armies.items()
                  if sanitize(k) == reg), None)
        return len(a.units) if a is not None else 0
    if _n('gap of rohan') != 1 or _n('south dunland') != 1 \
            or _n('north dunland') != 0:
        fails.append(('vac_board',
                      'gap=%d south=%d north=%d (want 1/1/0)'
                      % (_n('gap of rohan'), _n('south dunland'),
                         _n('north dunland'))))
    if getattr(es, 'pending_army_move_die', None) is not None:
        fails.append(('vac_pending_open', es.pending_army_move_die))


def test_echo_move_drag_claims_ledger(fails):
    """BL-80 (shgame9 D4): a region->region drag matching an open
    '<kind>_move' echo entry is the demand's PERFORMANCE - consumed
    against the ledger (mirror_kill_echo), never routed as a fresh
    gesture or pending content.  SPECIFICITY: a to_region mismatch
    does NOT claim the entry."""
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    drv.echo_ledger = [{'kind': 'naz_move', 'side': 'shadow',
                        'region': 'minas morgul',
                        'to_region': 'morannon',
                        'tkey': 'nazgul', 'n': 1, 'echo_id': 1}]
    # mismatched destination first: must NOT claim
    rig.feed('$Wey moves Nazgul from Minas Morgul to Gorgoroth.')
    if not drv.echo_ledger:
        fails.append(('echo_move_specificity',
                      'to_region mismatch claimed the entry'))
        return
    rig.feed('$Wey moves Nazgul from Minas Morgul to Morannon.')
    if drv.echo_ledger:
        fails.append(('echo_move_claim', drv.echo_ledger))
    if not any(k == 'mirror_kill_echo' for k, _ in drv.findings):
        fails.append(('echo_move_finding', drv.findings[-3:]))


def test_move_row_fate(fails):
    """BL-77 (shgame9 blocker): client-facts fate for a to-be-banked
    relocation row - 'satisfied' when the client already shows the
    pieces at the destination (the engine adopted the client's OWN
    drag), 'rebase:' when they stand at exactly one other region (a
    combined delta named the pre-approach source), None otherwise."""
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    row = {'kind': 'unit_move', 'side': 'shadow',
           'region': 'gap of rohan', 'to_region': 'fords of isen',
           'tkey': 'isengard:elite', 'n': 1}
    cases = [
        ({'fords of isen': 1}, dict(row), 'satisfied'),
        ({'fords of isen': 1},
         dict(row, to_region='helms deep'), 'rebase:fords of isen'),
        ({'gap of rohan': 1},
         dict(row, to_region='helms deep'), None),
        ({}, dict(row), None),
        (None, dict(row), None),
    ]
    for census, r, want in cases:
        drv._client_tkey_census = (lambda c: (lambda tkey: c))(census)
        got = drv._move_row_fate(r)
        if got != want:
            fails.append(('move_row_fate',
                          'census=%r row=%r -> %r (want %r)'
                          % (census, r, got, want)))


def test_cc_reveal_claim_once(fails):
    """BL-81 (shgame9 D5): the chat-committed combat card's physical
    reveal claims its ledger entry ONCE; the client's duplicate /
    re-prompt reveal line consumes silently (cc_reveal_dup_echo) -
    the old single slot refused the repeat and pressed the card back,
    duplicating the line."""
    from wotr_bridge.live import gesture_match as gm
    title = 'The Shadow is Moving/Swarm of Bats'
    cid = gm.title_to_cid(title)
    if cid is None:
        fails.append(('cc81_precondition', 'title_to_cid failed'))
        return
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    drv._cc_reveal_owed = {str(cid): 'owed'}
    line = ('<auto> moves %s from Shadow Hand (chosen) to Current '
            'SA Card.' % title)
    rig.feed(line)
    if drv._cc_reveal_owed.get(str(cid)) != 'claimed':
        fails.append(('cc81_claim', drv._cc_reveal_owed))
    if not any(k == 'cc_reveal_physical' for k, _ in drv.findings):
        fails.append(('cc81_finding', drv.findings[-3:]))
    n_alarms = len(drv.alarms)
    rig.feed(line)
    if not any(k == 'cc_reveal_dup_echo' for k, _ in drv.findings):
        fails.append(('cc81_dup_echo', drv.findings[-3:]))
    if len(drv.alarms) != n_alarms:
        fails.append(('cc81_dup_alarmed', drv.alarms[n_alarms:]))


def _resolved_round(atk_reg, def_reg, def_elite, accept, n=300):
    """A REAL battle round resolved by the engine: Shadow (Sauron) at
    West Harondor attacks the Gondor garrison of Pelargir, both sides
    decline their combat card, the round rolls under successive seeds
    until `accept(post)` holds.  Returns (pre, post) where pre is the
    state the defender's decline was applied TO (the driver's
    uncommitted state) and post the engine's resolution, or (None,
    None)."""
    import random
    import ai
    from game_state import Nation
    from test_live_engine_fixwave import field_battle_state, pick
    st = field_battle_state(atk_reg=atk_reg, def_nation=Nation.GONDOR,
                            def_reg=def_reg, def_elite=def_elite)
    nc = pick(ai.get_legal_actions(st, Side.SHADOW), 'combat_card_none')
    if nc is None:
        return None, None
    pre = ai.apply_action(st, nc)
    nc2 = pick(ai.get_legal_actions(pre, Side.FREE_PEOPLES),
               'combat_card_none')
    if nc2 is None:
        return None, None
    for seed in range(n):
        random.seed(seed)
        post = ai.apply_action(pre, nc2)
        if accept(post):
            return pre, post
    return None, None


def _mirror_units(view, zone, side_tag, nation, region, n_regular=0,
                  n_elite=0):
    """The client holds the placed army where the engine placed it
    (the rig's view starts from the setup positions)."""
    import emitter
    for p in emitter.pick_units(view, zone, side_tag, n_regular=n_regular,
                                n_elite=n_elite, nation=nation,
                                side_hint=side_tag):
        view.piece_region[p] = region


def test_forced_downgrade_arrival_echo(fails):
    """MEASURES (WAVE-J): a FORCED human-side downgrade (engine-auto
    allocation inside a committed apply: elite out, replacement
    regular in - reached live as 1 hit on an all-elite human army,
    where _alloc_choice_exists is False and the engine auto-applies;
    here a REAL round: Sauron x5 attacks a 2-elite Gondor garrison)
    banks BOTH rows on the echo ledger - the removal (kind 'unit')
    AND the arrival (kind 'unit_in') - and the human's physical
    mirror (kill elite + drag a regular in from FP Reinforcements) is
    consumed as echoes, never gestures.  SPECIFICITY: an arrival move
    from another REGION (not a reserve zone) is NOT consumed.

    REPRODUCTION 2026-08-20 (WAVE-J): _bank_net dropped the arrival
    plus-row entirely, so the client board could never receive the
    replacement regular - a guaranteed resync desync on every forced
    downgrade of a human-side unit."""
    import ai
    from game_state import UnitKind
    from test_live_engine_fixwave import own_units

    def downgraded(post):
        kinds = sorted(u.kind.name for u in
                       own_units(post, 'pelargir', Side.FREE_PEOPLES))
        return kinds == ['ELITE', 'REGULAR']
    pre, post = _resolved_round(5, 0, 2, downgraded)
    if post is None:
        fails.append(('dg_precondition', 'NOT-MEASURED: no seed forced '
                                         'a downgrade'))
        return
    rig = Rig(pre, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    _mirror_units(drv.view, 'fp reinf', 'fp', 'Gondor', 'Pelargir',
                  n_elite=2)
    drv.post_commit_sweep(pre, post, label='dg-test')
    rem = [e for e in drv.echo_ledger
           if e['kind'] == 'unit' and e['side'] == 'fp'
           and e['tkey'].endswith(':elite') and e['n'] == 1]
    arr = [e for e in drv.echo_ledger
           if e.get('kind') == 'unit_in' and e['side'] == 'fp'
           and e['tkey'].endswith(':regular') and e['n'] == 1]
    if not rem:
        fails.append(('dg_removal_banked', drv.echo_ledger))
    if not arr:
        fails.append(('dg_arrival_banked',
                      'the plus row was DROPPED (reproduction): %r'
                      % drv.echo_ledger))
        return
    # the human's physical mirror arrives: kill + reserve drag
    drv.es = post
    drv.hands.es = post
    n0 = len([1 for k, _ in drv.findings if 'echo' in k])
    rig.feed('$Wey kills Gondor Elite in Pelargir.')
    rig.feed('$Wey moves Gondor Regular from FP Reinforcements to '
             'Pelargir.')
    n1 = len([1 for k, _ in drv.findings if 'echo' in k])
    if n1 != n0 + 2:
        fails.append(('dg_mirror_consumed',
                      (n0, n1, drv.findings[-3:])))
    if any(e['n'] > 0 for e in drv.echo_ledger):
        fails.append(('dg_ledger_cleared', drv.echo_ledger))
    # SPECIFICITY: a region->region arrival is NOT echo-consumable
    st2 = adopted_state()
    rig2 = Rig(st2, Side.SHADOW, phase='actions')
    rig2.drv.echo_ledger.append(
        {'kind': 'unit_in', 'side': 'fp', 'region': 'grey havens',
         'tkey': 'elves:regular', 'n': 1})
    n2 = len([1 for k, _ in rig2.drv.findings if 'echo' in k])
    rig2.feed('$Wey moves Elven Regular from Rivendell to '
              'Grey Havens.')
    n3 = len([1 for k, _ in rig2.drv.findings if 'echo' in k])
    if n3 != n2:
        fails.append(('dg_specificity', rig2.drv.findings[-2:]))


def test_forced_rows_and_echo(fails):
    """MEASURES (WAVE-I, WAVE-H blocker 1): an AI-side unit removal a
    committed apply resolved with NO gesture (the ENGINE's own round:
    the Shadow attacker takes one hit on an all-regular army, a forced
    kill with no allocation choice) is (a) EMITTED as the AI's own
    kill line via the bounded row renderer, (b) BANKED on the echo
    ledger, and (c) the human's subsequent mirror kill of the same
    unit is consumed as an ECHO (finding row), never a gesture.
    SPECIFICITY: a kill with no matching ledger entry is NOT consumed
    as an echo.  HUMAN-side banking is NET-removal only: a real FP
    army_move (relocation) banks nothing; a human-side casualty from
    the same round banks its row.  HOLD discipline (ai = FP): an
    FP-side removal the casualty ledger does not cover - a delta the
    engine never books (defensive control on a hand-popped clone) -
    HOLDS + ALARMS instead of guessing."""
    import ai
    from test_live_engine_fixwave import own_units

    def one_sh_hit(post):
        return (len(own_units(post, 'west harondor', Side.SHADOW)) == 2
                and len(own_units(post, 'pelargir',
                                  Side.FREE_PEOPLES)) == 3)
    pre, post = _resolved_round(3, 3, 0, one_sh_hit)
    if post is None:
        fails.append(('forced_precondition', 'NOT-MEASURED: no seed '
                                             'killed one attacker'))
        return
    rig = Rig(pre, Side.SHADOW, phase='actions')     # AI = SHADOW
    drv = rig.drv
    _mirror_units(drv.view, 'sa reinf', 'shadow', 'Sauron',
                  'West Harondor', n_regular=3)
    items = drv.post_commit_sweep(pre, post, label='test')
    lines = [i[1] for i in items if i[0] == 'line']
    if len(lines) != 1:
        fails.append(('forced_kill_emitted',
                      (len(lines), drv.alarms[-1:],
                       drv.findings[-1:])))
        return
    rows = [e for e in drv.echo_ledger
            if e['kind'] == 'unit' and e['side'] == 'shadow'
            and e['tkey'].endswith(':regular')]
    if not rows:
        fails.append(('forced_banked', drv.echo_ledger))
        return
    # commit-time adoption, then the human's mirror kill arrives
    drv.es = post
    drv.hands.es = post
    n_echo0 = len([1 for k, _ in drv.findings
                   if k == 'mirror_kill_echo'])
    rig.feed('$Wey kills Sauron Regular in West Harondor.')
    n_echo1 = len([1 for k, _ in drv.findings
                   if k == 'mirror_kill_echo'])
    if n_echo1 != n_echo0 + 1:
        fails.append(('echo_consumed', drv.findings[-2:]))
    if any(e['kind'] == 'unit' and e['n'] > 0
           and e['tkey'].endswith(':regular')
           and e['side'] == 'shadow' for e in drv.echo_ledger):
        fails.append(('echo_ledger_decrement', drv.echo_ledger))
    # SPECIFICITY: no ledger entry -> not an echo
    rig.feed('$Wey kills Sauron Regular in West Harondor.')
    n_echo2 = len([1 for k, _ in drv.findings
                   if k == 'mirror_kill_echo'])
    if n_echo2 != n_echo1:
        fails.append(('echo_specificity', drv.findings[-2:]))
    # HUMAN-side banking is NET-removal only (wir3 live finding): a
    # real relocation of a human unit banks NOTHING
    st3 = adopted_state()
    mv = next((a for a in ai.get_legal_actions(st3, Side.FREE_PEOPLES)
               if a.action_type == 'army_move'
               and int(a.params.get('n_regular') or 0) >= 1), None)
    if mv is None:
        fails.append(('bank_precondition', 'NOT-MEASURED: no FP move'))
        return
    post3 = ai.apply_action(st3, mv)
    rig3 = Rig(st3, Side.SHADOW, phase='actions')
    rig3.drv.post_commit_sweep(st3, post3, label='t3')
    if any(e['side'] == 'fp' for e in rig3.drv.echo_ledger):
        fails.append(('bank_relocation_none', rig3.drv.echo_ledger))
    # ...a real human-side casualty banks its row

    def one_fp_hit(p):
        return (len(own_units(p, 'pelargir', Side.FREE_PEOPLES)) == 2
                and len(own_units(p, 'west harondor', Side.SHADOW)) == 3)
    pre4, post4 = _resolved_round(3, 3, 0, one_fp_hit)
    if post4 is None:
        fails.append(('bank_precondition', 'NOT-MEASURED: no seed killed '
                                           'one defender'))
        return
    rig4 = Rig(pre4, Side.SHADOW, phase='actions')
    rig4.drv.post_commit_sweep(pre4, post4, label='t3b')
    if not any(e['side'] == 'fp' and e['kind'] == 'unit'
               for e in rig4.drv.echo_ledger):
        fails.append(('bank_removal_banked', rig4.drv.echo_ledger))
    # HOLD discipline (ai = FP): uncovered FP removal never renders
    st5 = adopted_state()
    rig5 = Rig(st5, Side.FREE_PEOPLES, phase='actions')
    pre5 = rig5.drv.es
    post5 = ai.clone_state(pre5)
    if _pop_unit(post5, 'GONDOR', 'REGULAR') is None:
        fails.append(('forced_fp_precondition', 'no gondor regular'))
        return
    items5 = rig5.drv.post_commit_sweep(pre5, post5, label='t4')
    if any(i[0] == 'line' for i in items5):
        fails.append(('fp_hold_no_lines', items5))
    if not any('FORCED AI ROWS HOLD' in a for a in rig5.drv.alarms):
        fails.append(('fp_hold_alarm', rig5.drv.alarms))


def test_restore_watch_wired(fails):
    """MEASURES (WAVE-I, retire-bar requirement): a live REFUSAL arms
    the RestoreWatch on the refused window's own lines, and tick()
    VERIFIES it against the client dump end-to-end —
      * alarm path: a dump that never restores fires the STICKY alarm
        into driver.alarms (proven ABLE to fire through the wiring,
        not just at module level — the C4 class);
      * disarm path (specificity): a dump showing the piece back home
        disarms with NO alarm."""
    from wotr_bridge.live.undo_model import resolve_named
    pid = resolve_named('Gandalf the Grey')
    if pid is None:
        fails.append(('rw_precondition', 'gandalf unresolvable'))
        return
    bad_window = [
        '$Wey moves Free Action Die (C) from FP Dice to Current '
        'Action.',
        '$Wey moves Gandalf the Grey from Rivendell to Barad-Dur.',
        '$Wey kills Free Action Die (C) in Current Action.',
    ]

    def run(dump_zone):
        st = adopted_state()
        rig = Rig(st, Side.SHADOW, phase='actions')
        drv = rig.drv
        out = rig.feed(*bad_window)
        if not drv.restore_watch.armed:
            fails.append(('rw_armed', (dict(drv.window_stats),
                                       [i[0] for i in out])))
            return None
        drv.view.refresh_dump = lambda: True
        drv.view.piece_region = {str(pid): dump_zone}
        for _ in range(8):
            drv._check_restore_watch()
        return drv
    drv = run('Barad-Dur')          # never restored
    if drv is not None and not any('RESTORE WATCH' in a
                                   for a in drv.alarms):
        fails.append(('rw_alarm_fires', drv.alarms))
    drv2 = run('Rivendell')         # restored: disarm, no alarm
    if drv2 is not None:
        if drv2.restore_watch.armed:
            fails.append(('rw_disarms', 'still armed on a good dump'))
        if any('RESTORE WATCH' in a for a in drv2.alarms):
            fails.append(('rw_specificity', drv2.alarms))


def test_refuse_ineligible_arms_watch(fails):
    """MEASURES (WAVE-K R5 / ledger D-03): a refusal whose client-native
    undo is INELIGIBLE (the window is not at the feed tail - fact 2:
    the human's NEXT gesture landed in the same poll batch, after the
    terminator, so no press can reach the window) still ARMS the
    restore watch, and the watch fires the sticky RESTORE WATCH alarm
    on a dump that never restores (hold-and-witness doctrine).  Before
    the fix the early return left the board unwatched."""
    from wotr_bridge.live.undo_model import resolve_named
    pid = resolve_named('Gandalf the Grey')
    if pid is None:
        fails.append(('ri_precondition', 'gandalf unresolvable'))
        return
    bad_window = [
        '$Wey moves Free Action Die (C) from FP Dice to Current '
        'Action.',
        '$Wey moves Gandalf the Grey from Rivendell to Barad-Dur.',
        '$Wey kills Free Action Die (C) in Current Action.',
        '$Wey moves Free Action Die (A) from FP Dice to Current '
        'Action.',
    ]
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    rig.feed(*bad_window)
    if not any('UNDO-INELIGIBLE' in a for a in drv.alarms):
        fails.append(('ri_control', 'NOT-MEASURED: the ineligible path '
                      'was not taken (%r)' % (drv.alarms,)))
        return
    if not drv.restore_watch.armed:
        fails.append(('ri_armed', 'restore watch NOT armed on the '
                      'undo-ineligible hold path'))
        return
    drv.view.refresh_dump = lambda: True
    drv.view.piece_region = {str(pid): 'Barad-Dur'}
    for _ in range(8):
        drv._check_restore_watch()
    if not any('RESTORE WATCH' in a for a in drv.alarms):
        fails.append(('ri_alarm_fires', drv.alarms))


def test_illegal_window_not_rng_deferred(fails):
    """MEASURES (WAVE-K wkg2 LIVE FAILURE, FAILED-ALARM 'REFUSAL UNDO-
    INELIGIBLE'): an ILLEGAL army-move window on a die whose menu also
    offers fellowship_move, with eyes in the hunt box, must be REFUSED
    outright - never rng-DEFERRED.  Live, the matcher probed the
    fellowship_move candidate, starved on the hunt roll, deferred the
    window, ordered a REAL hunt roll + hunt tile for a non-gesture (the
    client's hunt pool lost a tile the engine never drew) and the client
    lines pushed after the window made the refusal's undo INELIGIBLE.
    A fellowship gesture ALWAYS renders fellowship vocabulary (FSP
    track walk / pawn / flip); a window with none is never one.
    Controls: fellowship_move must be IN the menu for that die (else
    NOT-MEASURED); the real C2 fellowship window still matches
    (test_fs_move_match in the matcher suite)."""
    import ai
    st = adopted_state()
    game_logic.allocate_eyes(st, 1, indices=[0])     # a hunt WOULD roll
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    with rig.rand.pin():
        menu = ai.get_legal_actions(drv.es, Side.FREE_PEOPLES)
    if not any(a.action_type == 'fellowship_move' and a.die_idx == 0
               for a in menu):
        fails.append(('iw_control', 'NOT-MEASURED: no fellowship_move '
                      'on FP die 0'))
        return
    bad_window = [
        '$Wey moves Free Action Die (C) from FP Dice to Current '
        'Action.',
        '$Wey moves North Regular from Bree to Forlindon.',
        '$Wey kills Free Action Die (C) in Current Action.',
    ]
    out = rig.feed(*bad_window)
    kinds = [i[0] for i in out]
    if drv.deferred is not None or 'rng_ask' in kinds:
        fails.append(('iw_rng_deferred',
                      'illegal army window was rng-DEFERRED: deferred=%r '
                      'items=%r stats=%r'
                      % (drv.deferred is not None, kinds,
                         dict(drv.window_stats))))
        return
    if drv.window_stats.get('none', 0) != 1:
        fails.append(('iw_refused', dict(drv.window_stats)))
    # the refusal must be ANNOUNCED and must EXPLAIN itself (USER
    # 2026-08-20: "if it says a move isn't legal it should specify what
    # rule it thinks was broken").  Either wording is a refusal: the
    # bare 'that move is not legal - <note>' when the matcher supplied
    # a reason, or the explained 'closest legal move: ... - your
    # gesture differs on ...' built by driver._explain_refusal.
    _ref = [i for i in out
            if i[0] == 'chat'
            and ('not legal' in str(i[-1])
                 or 'closest legal move' in str(i[-1]))]
    if not _ref:
        fails.append(('iw_refusal_chat', kinds))
    elif not any('closest legal move' in str(i[-1])
                 or ' - ' in str(i[-1]) for i in _ref):
        fails.append(('iw_refusal_unexplained', [str(i[-1])
                                                 for i in _ref]))


def test_siege_withdraw_window(fails):
    """MEASURES (WAVE-K wkg3 LIVE FAILURE - the human defender's siege
    withdrawal): a Shadow army attacks a FP STRONGHOLD region; the
    engine opens defender_siege_choice for the FP human; the human's
    PHYSICAL group drag region -> '<region> Stronghold' (corpus
    grammar) must match the engine's retreat_into_siege through the
    pending buffer and COMMIT (besieged_strongholds gains the region,
    pending_stats matched, no alarms/holds).  Controls: the pending
    must be open before the drag (else NOT-MEASURED); a drag of the
    wrong pieces is not this leg's question."""
    import ai
    from board import sanitize
    import game_logic
    from game_state import ActionDieFace
    from game_state import Nation
    st = adopted_state()
    game_logic.adopt_rolled_faces(st, [
        (Side.SHADOW, 2, ActionDieFace.ARMY),
        (Side.SHADOW, 3, ActionDieFace.ARMY),
        (Side.SHADOW, 4, ActionDieFace.MUSTER),
        (Side.SHADOW, 5, ActionDieFace.MUSTER)])
    with_rig = Rig(st, Side.SHADOW, phase='actions')
    drv = with_rig.drv

    def _pick(menu, at, frm, to, die):
        c = [a for a in menu if a.action_type == at
             and sanitize(str(a.params.get('from'))).lower() == frm
             and sanitize(str(a.params.get('to'))).lower() == to
             and a.die_idx == die]
        c.sort(key=lambda a: -(int(a.params.get('n_regular') or 0)
                               + int(a.params.get('n_elite') or 0)))
        return c[0] if c else None
    # Sauron to WAR first (setup box 2; the engine's can_attack_with_army
    # refuses a not-at-war attacker), then Moria -> Dimrill Dale (die 0)
    # and the attack on Lorien (die 2): the wkg1 live specimen's path
    for _die in (4, 5):
        if drv.es.political.is_at_war(Nation.SAURON):
            break
        with with_rig.rand.pin():
            menu = ai.get_legal_actions(drv.es, Side.SHADOW)
        ap = next((a for a in menu if a.action_type == 'advance_political'
                   and 'SAURON' in str(a.params.get('nation')).upper()
                   and a.die_idx == _die), None)
        if ap is None:
            fails.append(('sw_control', 'NOT-MEASURED: no Sauron '
                          'advance on die %d' % _die))
            return
        with with_rig.rand.pin():
            drv.es = ai.apply_action(drv.es, ap)
            drv.hands.es = drv.es
    if not drv.es.political.is_at_war(Nation.SAURON):
        fails.append(('sw_control', 'NOT-MEASURED: Sauron not at war '
                      'after two advances'))
        return
    for at, frm, to, die in (('army_move', 'moria', 'dimrill dale', 0),
                             ('army_attack', 'dimrill dale', 'lorien',
                              2)):
        with with_rig.rand.pin():
            menu = ai.get_legal_actions(drv.es, Side.SHADOW)
        act = _pick(menu, at, frm, to, die)
        if act is None:
            fails.append(('sw_control', 'NOT-MEASURED: no %s %s -> %s '
                          'on die %d' % (at, frm, to, die)))
            return
        with with_rig.rand.pin():
            drv.es = ai.apply_action(drv.es, act)
            drv.hands.es = drv.es
            # an ARMY die opens the second-leg state (move another
            # army or pass; GATED item 2: describe_pending does not
            # expose it - the MENU does): CLOSE it with the menu's pass
            if at == 'army_move'                     and getattr(drv.es, 'pending_army_move_die',
                                None) is not None:
                m0 = ai.get_legal_actions(drv.es, Side.SHADOW)
                pa = next((a for a in m0 if a.action_type == 'pass'),
                          None)
                if pa is not None:
                    drv.es = ai.apply_action(drv.es, pa)
                    drv.hands.es = drv.es
    # the rig applied the Shadow's moves ENGINE-side (no client
    # rendering) and the rig's view has NO dump: give it a synthetic
    # client dump (initial positions) with the besieger's two Sauron
    # regulars already in Dimrill Dale (live, the AI's own emitted move
    # lines track the view) - the post-commit sweep of the forced
    # move-in must find its pieces or it HOLDS (an alarm)
    import json as _json
    import os as _os
    import emitter as _em
    from wotr_bridge.live import client_board as _cb
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    with open(_os.path.join(_root, 'wotr_bridge',
                            'initial_positions.json'),
              encoding='utf-8') as _f:
        _init = dict(_json.load(_f))
    for _i in range(900, 930):          # pad past the torn-dump floor
        _init[str(_i)] = 'Game.2034'
    _mor = _em.zone_code_for('moria')
    _dd = _em.zone_code_for('dimrill dale')
    moved = 0
    for _pid, _code in sorted(_init.items()):
        if moved >= 2:
            break
        if str(_code) == str(_mor):
            _t = _cb.unit_type(_pid)
            if _t and 'sauron' in str(_t[0]).lower()                     and str(_t[1]).lower() == 'regular':
                _init[_pid] = _dd
                moved += 1
    if moved != 2:
        fails.append(('sw_control', 'NOT-MEASURED: could not place the '
                      'besieger in the synthetic dump (%d)' % moved))
        return
    _by = {}
    for _pid, _code in _init.items():
        _by.setdefault(_code, []).append(_pid)
    _dp = _os.path.join(with_rig.td, 'ai_client_board.txt')
    with open(_dp, 'w', encoding='utf-8') as _f2:
        _f2.write('\n'.join(
            '%d\t<~%s~>\t%s\t%s' % (_i, _c, ' '.join(sorted(_ps)),
                                    ' '.join('-' for _ in _ps))
            for _i, (_c, _ps) in enumerate(sorted(_by.items()))) + '\n')
    drv.view.dump_path = _dp
    if not drv.view.refresh_dump():
        fails.append(('sw_control', 'NOT-MEASURED: synthetic dump '
                      'unreadable'))
        return
    with with_rig.rand.pin():
        dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'combat:defender_siege_choice' \
            or dp['side'] is not Side.FREE_PEOPLES:
        fails.append(('sw_control', 'NOT-MEASURED: pending is %r'
                      % (dp and (dp['kind'], dp['side']),)))
        return
    if not any(a.action_type == 'retreat_into_siege'
               for a in dp['actions']):
        fails.append(('sw_control', 'NOT-MEASURED: retreat_into_siege '
                      'not offered'))
        return
    # the corpus drag: ONE group move of the FP garrison into the
    # stronghold zone, piece names from the engine's own army
    names = []
    try:
        from collections import Counter
        from game_logic import get_or_create_army
        a = get_or_create_army(drv.es, 'lorien')
        cnt = Counter()
        for u in (getattr(a, 'units', None) or []):
            cnt[(str(getattr(u.nation, 'value', u.nation)),
                 str(getattr(u.kind, 'value', u.kind)))] += 1
        disp_map = {'Elves': 'Elven', 'Dwarves': 'Dwarven',
                    'The North': 'North'}
        for (nat, kind), n in sorted(cnt.items()):
            names.append(('[%d]' % n if n > 1 else '')
                         + disp_map.get(nat, nat) + ' '
                         + kind.capitalize())
        n_lead = len(getattr(a, 'leaders', None) or [])
        if n_lead:
            names.append(('[%d]' % n_lead if n_lead > 1 else '')
                         + 'Elven Leader')
    except Exception as e:
        fails.append(('sw_control', 'NOT-MEASURED: garrison read '
                      'failed %r' % (e,)))
        return
    if not names:
        fails.append(('sw_control', 'NOT-MEASURED: empty garrison'))
        return
    line = ('$Wey moves group (%s) from Lorien to Lorien Stronghold.'
            % ', '.join(names))
    out = with_rig.feed(line)
    kinds = [i[0] for i in out]
    bes = {sanitize(str(b)) for b in
           (getattr(drv.es, 'besieged_strongholds', set()) or set())}
    # W2-025 (amendment 1.49 sub-bullet 2026-09-03): the withdrawal
    # COMMITS as the attacker's open victor_advance decision (the AI
    # Shadow attacker decides whether to move in; the siege flag is
    # set by ITS advance, never inside the human's retreat)
    _pc25 = getattr(drv.es, 'pending_combat', None) or {}
    _committed25 = ('lorien' in bes) or (
        _pc25.get('phase') == 'victor_advance'
        and _pc25.get('post_siege_retreat')
        and _pc25.get('deciding_side') is Side.SHADOW
        and sanitize(str(_pc25.get('defender_region'))) == 'lorien')
    if not _committed25:
        fails.append(('sw_match', 'siege drag NOT committed: besieged=%r '
                      'pending=%r pending_stats=%r items=%r line=%r'
                      % (sorted(bes), _pc25.get('phase'),
                         dict(drv.pending_stats), kinds, line)))
        return
    if drv.alarms or drv.holds:
        fails.append(('sw_clean', (drv.alarms, len(drv.holds))))


def test_human_chat_attack_declaration(fails):
    """MEASURES (human-attacker lane, 2026-08-22): with the action die
    STAGED, the human declares an attack in chat in the AI's own form
    ('<target> from <source>'); the ENGINE menu's army_attack for that
    die commits (battle opens on the human's combat-card decision, the
    die is spent) and the closing die kill is consumed without a second
    match or a refusal.  Controls: a declaration with NO staged die is
    refused; a region pair the menu does not offer is refused and the
    engine is untouched."""
    import ai
    from wotr_bridge.live.gesture_match import FACE_LETTER
    st = adopted_state()
    for nat, box in list(st.political.boxes.items()):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st.political.boxes[nat] = 3
            st.political.active[nat] = True
    menu = ai.get_legal_actions(st, Side.SHADOW)
    atk = next((a for a in menu if a.action_type == 'army_attack'),
               None)
    if atk is None:
        fails.append(('chat_attack_precondition', 'no army_attack'))
        return
    die = st.dice[Side.SHADOW][atk.die_idx]
    letter = FACE_LETTER[str(die.current.value).upper()
                         .replace(' ', '_').replace('/', '_')]
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions', seed=5)  # human=SH
    # control 1: no staged die -> refused, engine untouched
    items = rig.feed('<Wey> %s from %s' % (atk.params['to'],
                                            atk.params['from']))
    if not any(i[0] == 'chat' and i[1] == 'refusal' for i in items):
        fails.append(('no_die_refusal', items))
    if rig.drv.es.pending_combat is not None:
        fails.append(('no_die_engine_touched', 'battle opened'))
    rig.feed('$Wey moves Shadow Action Die (%s) from SA Dice to '
             'Current Action.' % letter)
    # control 2: an unoffered pair -> refused, engine untouched
    items = rig.feed('<Wey> minas tirith from %s' % atk.params['from'])
    if not any(i[0] == 'chat' and i[1] == 'refusal' for i in items):
        fails.append(('bad_pair_refusal', items))
    if rig.drv.es.pending_combat is not None:
        fails.append(('bad_pair_engine_touched', 'battle opened'))
    # the declaration
    rig.feed('<Wey> %s from %s' % (atk.params['to'], atk.params['from']))
    pc = rig.drv.es.pending_combat or {}
    if pc.get('phase') != 'attacker_combat_card'             or pc.get('deciding_side') is not Side.SHADOW:
        fails.append(('battle_not_opened', (pc.get('phase'),
                                            pc.get('deciding_side'))))
    if not rig.drv.es.dice[Side.SHADOW][atk.die_idx].used:
        fails.append(('die_not_spent', atk.die_idx))
    if rig.drv.alarms:
        fails.append(('chat_attack_alarms', rig.drv.alarms[-1]))
    m0, n0 = rig.drv.window_stats['matched'], rig.drv.window_stats['none']
    rig.feed('$Wey kills Shadow Action Die (%s) in Current Action.'
             % letter)
    if rig.drv.window_stats['none'] != n0 or rig.drv.uncommitted:
        fails.append(('die_kill_refused', rig.drv.uncommitted[-1:]))
    if rig.drv.window_stats['matched'] != m0 + 1:
        fails.append(('die_kill_not_consumed', rig.drv.window_stats))

    # bare target, ONE eligible army -> commits without 'from'
    st2 = adopted_state()
    for nat, box in list(st2.political.boxes.items()):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st2.political.boxes[nat] = 3
            st2.political.active[nat] = True
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='actions', seed=5)
    rig2.feed('$Wey moves Shadow Action Die (%s) from SA Dice to '
              'Current Action.' % letter)
    rig2.feed('<Wey> %s' % atk.params['to'])
    pc2 = rig2.drv.es.pending_combat or {}
    if pc2.get('phase') != 'attacker_combat_card':
        fails.append(('bare_target_single_source', pc2.get('phase')))
    # bare target, TWO eligible armies -> 'from? a, b' ask, engine
    # untouched, then 'from <b>' commits from b
    from game_state import Unit, UnitKind, Nation
    import game_logic as gl
    st3 = adopted_state()
    for nat, box in list(st3.political.boxes.items()):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st3.political.boxes[nat] = 3
            st3.political.active[nat] = True
    other = next(r for r in st3.board.adjacent(atk.params['to'])
                 if r.lower() != str(atk.params['from']).lower())
    army = gl.get_or_create_army(st3, other)
    army.units.append(Unit(Nation.ISENGARD, UnitKind.REGULAR))
    army.units.append(Unit(Nation.ISENGARD, UnitKind.REGULAR))
    rig3 = Rig(st3, Side.FREE_PEOPLES, phase='actions', seed=5)
    rig3.feed('$Wey moves Shadow Action Die (%s) from SA Dice to '
              'Current Action.' % letter)
    items = rig3.feed('<Wey> %s' % atk.params['to'])
    asks = [i for i in items if i[0] == 'chat' and i[1] == 'attack_from_ask']
    if not asks or not asks[0][2].startswith('from? ')             or other.lower() not in asks[0][2]:
        fails.append(('ambiguous_ask', items))
    if rig3.drv.es.pending_combat is not None:
        fails.append(('ambiguous_engine_touched', 'battle opened'))
    rig3.feed('<Wey> from %s' % other)
    pc3 = rig3.drv.es.pending_combat or {}
    if pc3.get('phase') != 'attacker_combat_card'             or str(pc3.get('attacker_region', '')).lower() != other.lower():
        fails.append(('from_answer', (pc3.get('phase'),
                                      pc3.get('attacker_region'))))


def test_defender_retreat_advance(fails):
    """MEASURES: the human defender's PHYSICAL retreat drag resolves
    through the pending buffer and COMMITS (pending_stats.matched);
    the attacker's advance is then the AI's LEARNED victor_advance
    decision (R6 / W1#132) - surfaced as the AI's pending with a
    victor_stay option, never an engine-forced relocation of the AI
    army into the vacated region.  SPECIFICITY: no alarms/holds."""
    import ai
    from board import sanitize
    st = adopted_state()
    for nat, box in list(st.political.boxes.items()):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st.political.boxes[nat] = 3
            st.political.active[nat] = True
    menu = ai.get_legal_actions(st, Side.SHADOW)
    atk = next((a for a in menu if a.action_type == 'army_attack'),
               None)
    if atk is None:
        fails.append(('retreat_precondition', 'no army_attack'))
        return
    rig = Rig(st, Side.SHADOW, phase='actions', seed=5)
    drv = rig.drv

    def ai_apply(action, label):
        rig.rand.begin_attempt()
        with rig.rand.advance():
            post = ai.apply_action(drv.es, action)
        pre = drv.es
        drv._snapshot([], label)
        drv.es = post
        drv.hands.es = post
        drv._out_queue.extend(
            drv.post_commit_sweep(pre, post, label=label))
        rig.rand.step_commit()

    ai_apply(atk, 'ai:attack')
    drv.last_actor = Side.SHADOW
    rets = []
    for _ in range(24):
        dp = ai.describe_pending(drv.es)
        if dp is None:
            if drv.chat_deferred is not None \
                    or drv.deferred is not None:
                rig.feed('<game> Wey rolls 1 1 1 1 1')
                continue
            break
        rets = [a for a in dp['actions']
                if a.action_type == 'combat_retreat']
        if dp['side'] is Side.FREE_PEOPLES and rets:
            break
        if dp['side'] is Side.FREE_PEOPLES:
            types = {a.action_type for a in dp['actions']}
            word = ('nc' if 'combat_card_none' in types
                    else 'field battle')
            rig.feed('<Wey> %s' % word)
            if drv.chat_deferred is not None:
                rig.feed('<game> Wey rolls 1 1 1 1 1')
            continue
        pick = next((a for a in dp['actions']
                     if a.action_type in ('combat_card_none',
                                          'shine_none',
                                          'combat_continue')),
                    dp['actions'][0])
        try:
            ai_apply(pick, 'ai:%s' % dp['kind'])
        except BaseException:
            rig.feed('<game> Wey rolls 1 1 1 1 1')
    if not rets:
        fails.append(('retreat_window', 'never reached the retreat '
                                        'pending'))
        return
    ra = rets[0]
    pc = getattr(drv.es, 'pending_combat', None) or {}
    frm = ra.params.get('from') or pc.get('defender_region')
    dest = ra.params.get('to_region', ra.params.get('to'))
    atk_reg = sanitize(str(pc.get('attacker_region', '')))
    key = next(k for k in drv.es.armies
               if sanitize(k) == sanitize(str(frm)))
    from collections import Counter
    cnt = Counter()
    for u in drv.es.armies[key].units:
        if ai._side_of_nation(u.nation) == Side.FREE_PEOPLES:
            cnt['%s %s' % (str(u.nation.value).title(),
                           'Elite' if u.kind.name == 'ELITE'
                           else 'Regular')] += 1
    parts = [('[%d]%s' % (n, t)) if n > 1 else t
             for t, n in sorted(cnt.items())]
    obj = ('group (%s)' % ', '.join(parts)) \
        if sum(cnt.values()) > 1 else parts[0]
    out = rig.feed('$Wey moves %s from %s to %s.'
                   % (obj, str(frm).title(), str(dest).title()))
    if drv.pending_stats.get('matched', 0) < 1:
        fails.append(('retreat_committed',
                      (dict(drv.pending_stats),
                       drv.findings[-3:])))
        return
    # R6 (W1#132): the attacker's advance after the retreat is the
    # AI's LEARNED victor_advance decision, surfaced to live_server
    # as its pending - never an engine-forced relocation
    adv_dps = [i[1] for i in out if i[0] == 'ai_decide'
               and str(i[1].get('kind', '')).endswith('victor_advance')]
    adv_key = next((k for k in drv.es.armies
                    if sanitize(k) == sanitize(str(frm))), None)
    has_sh = adv_key is not None and any(
        ai._side_of_nation(u.nation) == Side.SHADOW
        for u in drv.es.armies[adv_key].units)
    if adv_dps:
        if adv_dps[-1]['side'] is not Side.SHADOW:
            fails.append(('advance_decider', adv_dps[-1]['side']))
        if has_sh:
            fails.append(('advance_forced', 'Shadow units already in '
                                            '%r before the decision'
                          % frm))
        if not any(a.action_type == 'victor_stay'
                   for a in adv_dps[-1]['actions']):
            fails.append(('advance_stay_option',
                          sorted({a.action_type
                                  for a in adv_dps[-1]['actions']})))
    else:
        fails.append(('advance_learned', 'no victor_advance decision '
                                         'surfaced: %r'
                      % [i[0] for i in out]))
    bad = [a for a in drv.alarms if 'HOLD' in a or 'FAILED' in a]
    if bad:
        fails.append(('retreat_clean', bad))


def test_capture_marker_sweep(fails):
    """MEASURES: a settlement capture the ENGINE resolved (a real
    battle: the AI-Shadow army at West Harondor wipes the Gondor
    garrison of Pelargir, then ENTERS it with victor_advance -
    USER RULING 1.255 (2026-08-29): the wipe alone captures NOTHING,
    combat._check_settlement_capture registers the city on entry) makes
    the post-commit sweep emit the control-marker walk (pool ->
    settlement, donor pid) and the VP-marker drag (city = 1 VP) from
    ENGINE EVENTS; the ledger makes a re-sweep of the same state
    IDEMPOTENT (no duplicate lines); after the Shadow army marches on
    (the marker stays), a real FP army_move back into the emptied city
    (the engine's own recapture) returns the marker to the pool and
    walks the VP marker back."""
    import ai
    import random
    from board import sanitize
    from test_live_engine_fixwave import (field_battle_state,
                                          decline_cards, seed_search,
                                          own_units, pick, place,
                                          set_dice)
    from game_state import Nation
    pre = field_battle_state(atk_reg=4, def_reg=1)
    rig = Rig(pre, Side.SHADOW, phase='actions')     # AI = attacker
    drv = rig.drv

    def make():
        return decline_cards(pre)

    def wiped(p):
        return (own_units(p, 'west harondor', Side.SHADOW)
                and not own_units(p, 'pelargir', Side.FREE_PEOPLES))
    seed, post = seed_search(make, wiped)
    if post is None:
        fails.append(('cap_control', 'NOT-MEASURED: no seed wiped the '
                                     'Pelargir garrison'))
        return
    # 1.255: the victor still stands at West Harondor - no capture yet
    if 'pelargir' in post.enemy_controlled[Side.SHADOW]:
        fails.append(('cap_1255_wipe', 'captured at the wipe without '
                                       'entering (rules_amendments 1.255)'))
        return
    pc = post.pending_combat
    if pc is None or pc.get('phase') != 'victor_advance':
        fails.append(('cap_control', 'NOT-MEASURED: no victor_advance '
                                     'pending after the wipe: %r' % (pc,)))
        return
    advs = [a for a in ai.get_legal_actions(post, Side.SHADOW)
            if a.action_type == 'victor_advance'
            and int(a.params.get('n_regular', 0)) > 0]
    if not advs:
        fails.append(('cap_control', 'NOT-MEASURED: no victor_advance '
                                     'with units on the menu'))
        return
    adv = max(advs, key=lambda a: int(a.params.get('n_regular', 0)))
    entered = ai.apply_action(post, adv)
    n_in = len(own_units(entered, 'pelargir', Side.SHADOW))
    if not n_in:
        fails.append(('cap_control', 'NOT-MEASURED: the advance did not '
                                     'move the army in'))
        return
    if 'pelargir' not in entered.enemy_controlled[Side.SHADOW]:
        fails.append(('cap_1255_entry', 'the engine did not register the '
                                        'capture on ENTRY (1.255)'))
        return
    items = drv.post_commit_sweep(pre, entered, label='cap')
    lines = [i[1] for i in items if i[0] == 'line']
    if len(lines) < 2:
        fails.append(('cap_lines', (lines, drv.findings[-2:])))
        return
    if drv.ctrl_markers['sh'].get('pelargir') is None:
        fails.append(('cap_ledger', dict(drv.ctrl_markers)))
    if drv.vp_marks['sh'] < 1:      # city = 1 VP
        fails.append(('cap_vp', dict(drv.vp_marks)))
    # idempotent on re-sweep of the same state
    items2 = drv.post_commit_sweep(entered, entered, label='cap2')
    if any(i[0] == 'line' for i in items2):
        fails.append(('cap_idempotent', items2))
    # the engine's recapture: the Shadow army marches on (a real
    # army_move out of the city keeps the marker), then a Gondor army
    # walks back into the empty city with an Army die
    set_dice(entered, Side.SHADOW, [ActionDieFace.ARMY])
    out = next((a for a in ai.get_legal_actions(entered, Side.SHADOW)
                if a.action_type == 'army_move'
                and sanitize(str(a.params.get('from'))) == 'pelargir'
                and sanitize(str(a.params.get('to'))) == 'west harondor'
                and int(a.params.get('n_regular', 0)) == n_in), None)
    if out is None:
        fails.append(('cap_control', 'NOT-MEASURED: no full-strength '
                                     'Shadow army_move out of Pelargir'))
        return
    mid = ai.apply_action(entered, out)
    if own_units(mid, 'pelargir', Side.SHADOW):
        fails.append(('cap_control', 'NOT-MEASURED: the move-out left '
                                     'Shadow units in Pelargir'))
        return
    if 'pelargir' not in mid.enemy_controlled[Side.SHADOW]:
        fails.append(('cap_marker_stays', 'leaving the captured city '
                                          'released the marker'))
        return
    place(mid, 'lamedon', Nation.GONDOR, regulars=1, clear=True)
    set_dice(mid, Side.FREE_PEOPLES, [ActionDieFace.ARMY])
    back = next((a for a in ai.get_legal_actions(mid, Side.FREE_PEOPLES)
                 if a.action_type == 'army_move'
                 and sanitize(str(a.params.get('to'))) == 'pelargir'),
                None)
    if back is None:
        fails.append(('cap_control', 'NOT-MEASURED: no FP army_move into '
                                     'Pelargir'))
        return
    post2 = ai.apply_action(mid, back)
    if 'pelargir' in post2.enemy_controlled[Side.SHADOW]:
        fails.append(('cap_control', 'NOT-MEASURED: the engine did not '
                                     'release the city on the FP move'))
        return
    items3 = drv.post_commit_sweep(mid, post2, label='cap3')
    lines3 = [i[1] for i in items3 if i[0] == 'line']
    if not lines3 or 'pelargir' in drv.ctrl_markers['sh']:
        fails.append(('cap_return', (lines3,
                                     dict(drv.ctrl_markers))))
    if drv.vp_marks['sh'] != 0:
        fails.append(('cap_vp_return', dict(drv.vp_marks)))


# ---------------------------------------------------------------------
# FIX WAVE 1 (agent B) pins: ids = audit FINDINGS_FULL.md W1-playpath#n
# ---------------------------------------------------------------------
def _write_meta(path, recs):
    import json
    with open(path, 'a', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')


def test_route_error_isolated_per_line(fails):
    """MEASURES (W1#0): a routing error on ONE feed line (here the
    human-draw halt on an EMPTY engine deck - a real DriverHalt site)
    neither drops the later lines of the same poll batch nor
    misaligns the sidecar seq.  The later draw in the batch is routed
    (turn_draws grows), the error is a sticky alarm (never silent,
    never a raise out of tick, surfaced ONCE per distinct message),
    and the next line's seq still equals its 1-based feed index (no
    'sidecar_gap').  A client REDO press is an alarm, not a halt."""
    from wotr_bridge.live import sidecar as sc_mod
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    meta_p = rig.path + '.meta.jsonl'
    drv.sidecar = sc_mod.SidecarReader(meta_p)
    _write_meta(meta_p, [
        {'seq': 1, 'gen': 1, 'nick': 'Wey', 'tags': ['Controls.491'],
         'pids': 'redacted', 'side': 'FP', 'ts': 1},
        {'seq': 2, 'gen': 1, 'nick': 'Wey', 'tags': ['Controls.491'],
         'pids': 'redacted', 'side': 'FP', 'ts': 2},
        {'seq': 3, 'gen': 1, 'nick': 'Wey', 'tags': ['Controls.491'],
         'pids': 'redacted', 'side': 'FP', 'ts': 3},
        {'seq': 4, 'gen': 1, 'nick': 'Wey', 'tags': [], 'pids': [],
         'ts': 4}])
    drv.es.fp_char_deck[:] = []          # the draw halt site
    try:
        rig.feed('$Wey draws Free Character Card.',
                 '$Wey draws Free Strategy Card.')
    except Exception as e:
        fails.append(('route_batch_raised', repr(e)))
        return
    if drv.turn_draws.get((Side.FREE_PEOPLES, 'strategy')) != 1:
        fails.append(('route_rest_of_batch_dropped',
                      dict(drv.turn_draws)))
    errs = [a for a in drv.alarms if 'ROUTE ERROR' in a]
    if not errs:
        fails.append(('route_error_silent', list(drv.alarms)))
    rig.feed('$Wey draws Free Character Card.')
    if drv._feed_seq != 3 or len(drv.feed.lines) != 3:
        fails.append(('route_seq', (drv._feed_seq,
                                    len(drv.feed.lines))))
    gaps = [d for k, d in drv.findings if k == 'sidecar_gap']
    if gaps:
        fails.append(('route_seq_misaligned', gaps))
    # surfaced ONCE per distinct message, not once per line
    errs2 = [a for a in drv.alarms if 'ROUTE ERROR' in a]
    if len(errs2) != len(errs):
        fails.append(('route_error_respammed', errs2))
    try:
        rig.feed('$Wey redoes last action : Wey draws Free Strategy '
                 'Card.')
    except Exception as e:
        fails.append(('route_redo_raised', repr(e)))
    if not any('REDO' in a for a in drv.alarms):
        fails.append(('route_redo_silent', list(drv.alarms)))


def test_undo_restores_driver_turn_state(fails):
    """MEASURES (W1#2 / K5): the undo ledger carries the driver's
    per-turn state with the engine state.  (a) undoing a turn-start
    draw restores turn_draws, so the human's REDRAW is owed again
    (no UN-OWED refusal, hand back to +1); (b) undoing the recover
    line restores phase 'actions' and the turn's draw counters."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    n0 = len(drv.es.hands[Side.FREE_PEOPLES].hand)
    rig.feed('$Wey draws Free Character Card.')
    if drv.turn_draws.get((Side.FREE_PEOPLES, 'character')) != 1:
        fails.append(('undo_ts_precondition', dict(drv.turn_draws)))
        return
    rig.feed('$Wey undoes last action : Wey draws Free Character '
             'Card.')
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n0:
        fails.append(('undo_ts_engine', 'engine hand not restored'))
    if drv.turn_draws.get((Side.FREE_PEOPLES, 'character'), 0) != 0:
        fails.append(('undo_ts_turn_draws', dict(drv.turn_draws)))
    rig.feed('$Wey draws Free Character Card.')
    if any('UN-OWED' in a for a in drv.alarms):
        fails.append(('undo_ts_redraw_refused', list(drv.alarms)))
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n0 + 1:
        fails.append(('undo_ts_redraw_hand',
                      len(drv.es.hands[Side.FREE_PEOPLES].hand)))
    # (b) the rollover line
    st2 = adopted_state()
    rig2 = Rig(st2, Side.SHADOW, phase='actions')
    drv2 = rig2.drv
    drv2.last_actor = Side.FREE_PEOPLES
    drv2.turn_draws = {(Side.FREE_PEOPLES, 'character'): 1,
                       (Side.FREE_PEOPLES, 'strategy'): 1}
    for d in drv2.es.dice[Side.FREE_PEOPLES]:
        d.used = True
    # BL-13: recover refuses while ANY die is unspent - the rollover
    # precondition is BOTH pools spent
    for d in drv2.es.dice[Side.SHADOW]:
        d.used = True
    turn0 = drv2.es.turn
    line = ('$Wey moves group (Free Action Die (C), Free Action Die '
            '(P), [2]Free Action Die (A)) from Current Action to FP '
            'Dice.')
    rig2.feed(line)
    if drv2.phase != 'draws' or drv2.es.turn != turn0 + 1:
        fails.append(('undo_ts_rollover_precondition',
                      (drv2.phase, drv2.es.turn)))
        return
    rig2.feed('$Wey undoes last action : %s' % line.lstrip('$'))
    if drv2.es.turn != turn0:
        fails.append(('undo_ts_rollover_engine', drv2.es.turn))
    if drv2.phase != 'actions':
        fails.append(('undo_ts_rollover_phase', drv2.phase))
    if drv2.last_actor is not Side.FREE_PEOPLES:
        fails.append(('undo_ts_rollover_actor', drv2.last_actor))
    if drv2.turn_draws.get((Side.FREE_PEOPLES, 'character')) != 1:
        fails.append(('undo_ts_rollover_draws',
                      dict(drv2.turn_draws)))


def test_human_fp_declaration(fails):
    """MEASURES (W1#1 / K2): a HUMAN Free Peoples player can declare
    the Fellowship.  After the turn-start draws the driver arms the
    engine's declare menu for the human FP; the client's declaration
    records (pawn walk + 'Previously on Fellowship Track N' flip)
    match fellowship_declare through the pending buffer and COMMIT
    (engine region/progress updated); 'fp ok' then closes the phase
    with declaration and Guide settled and Hunt allocation pending.
    SPECIFICITY: 'fp ok' with NO declaration settles the decline
    (menu pass) and explicitly keeps the current Guide."""
    import ai
    from board import sanitize
    if not hasattr(game_logic, 'open_declare_phase'):
        fails.append(('declare_engine_surface',
                      'NOT-MEASURED: game_logic.open_declare_phase '
                      'missing (K2 engine half not landed)'))
        return
    st = fresh_state()
    st.fellowship.progress = 2
    st.fellowship.hidden = True
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    rig.feed('$Wey draws Free Character Card.',
             '$Wey draws Free Strategy Card.')
    if drv.phase != 'declare':
        fails.append(('declare_phase', drv.phase))
        return
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'declare':
        fails.append(('declare_armed', repr(dp and dp['kind'])))
        return
    cur = sanitize(str(game_logic.fellowship_region(drv.es)))
    if cur != 'rivendell':
        fails.append(('declare_precondition', cur))
        return
    rig.feed('$Wey moves The Fellowship from Rivendell to Fords of '
             'Bruinen.',
             '$Wey moves The Fellowship from Fords of Bruinen to '
             'Hollin.',
             '$Wey flips The Fellowship in Hollin. Previously on '
             'Fellowship Track 2.',
             '$Wey flips The Fellowship in Hollin.')
    reg = sanitize(str(game_logic.fellowship_region(drv.es)))
    if reg != 'hollin':
        fails.append(('declare_committed',
                      (reg, drv.findings[-3:], drv.alarms[-2:])))
    if drv.es.fellowship.progress != 0:
        fails.append(('declare_progress', drv.es.fellowship.progress))
    if not drv.es.fellowship.hidden:
        fails.append(('declare_hidden', 'revealed by a declaration'))
    out = rig.feed('<Wey> fp ok')
    if drv.phase != 'eyes':
        fails.append(('declare_fp_ok_phase', drv.phase))
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'eye' or dp['side'] is not Side.SHADOW:
        fails.append(('declare_expected_eye_pending', repr(dp and dp['kind'])))
    if drv.es.pending_guide_change is not None:
        fails.append(('declare_guide_pending_left', drv.es.pending_guide_change))
    if getattr(drv.es, '_declare_phase', False):
        fails.append(('declare_flag_left', 'flag still armed'))
    if not any(i[0] == 'ai_eyes' for i in out):
        fails.append(('declare_fp_ok_items', out))
    # SPECIFICITY: no declaration -> 'fp ok' settles the decline
    st2 = fresh_state()
    st2.fellowship.progress = 2
    rig2 = Rig(st2, Side.SHADOW, phase='draws')
    drv2 = rig2.drv
    rig2.feed('$Wey draws Free Character Card.',
              '$Wey draws Free Strategy Card.')
    rig2.feed('<Wey> fp ok')
    dp2 = ai.describe_pending(drv2.es)
    if dp2 is None or dp2['kind'] != 'eye' or dp2['side'] is not Side.SHADOW:
        fails.append(('decline_expected_eye_pending', repr(dp2 and dp2['kind'])))
    if drv2.es.pending_guide_change is not None:
        fails.append(('decline_guide_pending_left', drv2.es.pending_guide_change))
    if sanitize(str(game_logic.fellowship_region(drv2.es))) \
            != 'rivendell' or drv2.es.fellowship.progress != 2:
        fails.append(('decline_moved', 'a decline changed the '
                                       'fellowship'))
    if not any(lbl == 'declare-decline' for _p, _l, lbl in
               [(c[0], c[1], c[2]) for c in drv2.commits]):
        fails.append(('decline_ledger',
                      [c[2] for c in drv2.commits]))


def _human_defender_cc_rig(fails, tag):
    """An AI-Shadow attack applied to the engine, walked to the
    HUMAN FP's defender_combat_card decision.  Returns (rig, drv) or
    None (NOT-MEASURED recorded under `tag`)."""
    import ai
    st = adopted_state()
    for nat, box in list(st.political.boxes.items()):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st.political.boxes[nat] = 3
            st.political.active[nat] = True
    menu = ai.get_legal_actions(st, Side.SHADOW)
    atk = next((a for a in menu if a.action_type == 'army_attack'),
               None)
    if atk is None:
        fails.append((tag, 'NOT-MEASURED: no army_attack'))
        return None
    rig = Rig(st, Side.SHADOW, phase='actions', seed=5)
    drv = rig.drv

    def ai_apply(action, label):
        rig.rand.begin_attempt()
        with rig.rand.advance():
            post = ai.apply_action(drv.es, action)
        drv._snapshot([], label)
        drv.es = post
        drv.hands.es = post
        rig.rand.step_commit()

    ai_apply(atk, 'ai:attack')
    drv.last_actor = Side.SHADOW
    for _ in range(6):
        dp = ai.describe_pending(drv.es)
        if dp is None:
            break
        if dp['side'] is Side.FREE_PEOPLES:
            break
        pick = next((a for a in dp['actions']
                     if a.action_type == 'combat_card_none'), None)
        if pick is None:
            break
        ai_apply(pick, 'ai:%s' % dp['kind'])
    dp = ai.describe_pending(drv.es)
    pc = getattr(drv.es, 'pending_combat', None) or {}
    if dp is None or dp['side'] is not Side.FREE_PEOPLES \
            or pc.get('phase') != 'defender_combat_card':
        fails.append((tag, 'NOT-MEASURED: defender_combat_card not '
                           'reached (%r)' % (pc.get('phase'),)))
        return None
    return rig, drv


def test_live_pending_never_forced(fails):
    """MEASURES (W1#3 / R5): a HUMAN decision the feed never resolves
    is HELD, never force-committed with stubbed dice.  Two consecutive
    human die windows close behind an unanswered defender_combat_card
    pending: the pending is still open afterwards (same kind, human
    side), no 'pending_forced' / shim row exists, the windows were
    held (not matched), and the human got at most ONE chat reminder
    naming the owed decision."""
    import ai
    got = _human_defender_cc_rig(fails, 'force_precondition')
    if got is None:
        return
    rig, drv = got
    win = [
        '$Wey moves Free Action Die (C) from FP Dice to Current '
        'Action.',
        '$Wey moves Gondor Regular from Minas Tirith to Osgiliath.',
        '$Wey kills Free Action Die (C) in Current Action.',
    ]
    out1 = rig.feed(*win)
    out2 = rig.feed(*win)
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['side'] is not Side.FREE_PEOPLES \
            or not str(dp['kind']).startswith('combat'):
        fails.append(('force_pending_lost',
                      repr(dp and (dp['kind'], dp['side']))))
    forced = [(k, d) for k, d in drv.findings
              if k in ('pending_forced', 'pending_silent_noop',
                       'engine_gap_hunt_reroll_declined',
                       'engine_gap_draw_grant_declined')]
    if forced:
        fails.append(('force_committed', forced))
    if drv.window_stats.get('matched', 0):
        fails.append(('force_window_matched', dict(drv.window_stats)))
    chats = [i for i in out1 + out2 if i[0] == 'chat']
    if len(chats) > 1:
        fails.append(('force_reminder_spam', chats))
    if not chats:
        fails.append(('force_reminder_missing',
                      [i[0] for i in out1 + out2]))
    if 'reroll_decline_shim' in dir(drv.rand) \
            and getattr(drv.rand, 'reroll_decline_shim', False):
        fails.append(('force_shim_left_on', True))


_SCOUTS = 'FREE_PEOPLES_Strategy_The_Red_Arrow'


def _arm_scouts(fails, drv, tag):
    """Put ONE strategy placeholder in the human FP hand and prove
    (on a probe clone) that the engine offers Scouts as a playable
    combat card here; NOT-MEASURED otherwise."""
    import ai
    hm = drv.hand_models[Side.FREE_PEOPLES]
    hand = drv.es.hands[Side.FREE_PEOPLES].hand
    ph = hm._mint('strategy')
    hand.append(ph)
    probe = ai.clone_state(drv.es)
    ph_hand = probe.hands[Side.FREE_PEOPLES].hand
    ph_hand[ph_hand.index(ph)] = _SCOUTS
    dp = ai.describe_pending(probe)
    ok = dp is not None and any(
        a.action_type == 'combat_card_play'
        and a.params.get('card_id') == _SCOUTS for a in dp['actions'])
    if not ok:
        fails.append((tag, 'NOT-MEASURED: Scouts not offered as a '
                           'defender combat card here'))
        return False
    return True


def test_dragged_combat_card_commits_via_buffer(fails):
    """MEASURES (W1#30, end to end): the human defender's combat card
    played by DRAG (hand -> Current FP Card) while the engine's
    defender_combat_card pending is open buffers as a COMBAT card,
    the placeholder is swapped for the named cid, the pending's menu
    offers combat_card_play and the buffer COMMITS once the round's
    rolls render (the card leaves the hand; a pending commit is
    ledgered).  The drag lane is the one the USER's 'scouts is
    nonfunctional' report exercised."""
    import ai
    got = _human_defender_cc_rig(fails, 'drag_cc_precondition')
    if got is None:
        return
    rig, drv = got
    if not _arm_scouts(fails, drv, 'drag_cc_precondition'):
        return
    rig.feed('$Wey moves The Red Arrow/Scouts from Free Peoples Hand '
             'to Current FP Card.')
    for _ in range(8):
        if drv.pending_stats.get('matched', 0) >= 1:
            break
        rig.feed('<game> Wey rolls 1 1 1 1 1')
    if drv.pending_stats.get('matched', 0) < 1:
        fails.append(('drag_cc_committed',
                      (dict(drv.pending_stats), drv.findings[-3:],
                       drv.alarms[-2:])))
        return
    if _SCOUTS in drv.es.hands[Side.FREE_PEOPLES].hand:
        fails.append(('drag_cc_left_in_hand', 'Scouts still in hand'))
    if not any(str(c[2]).startswith('pending:combat')
               for c in drv.commits):
        fails.append(('drag_cc_ledger', [c[2] for c in drv.commits]))


def test_hidden_combat_card_commit_and_reveal(fails):
    """MEASURES (K1 step 3): the human's chooser line
    '<nick> chooses a card.[Free Strategy Card]' while the engine's
    open decision is the human's combat card is a HIDDEN commit
    (ai.commit_hidden_combat_card: deck type only, identity with the
    client; human_cc_committed set); the later '<auto> moves TITLE
    from Free Peoples Hand (chosen) to Current FP Card.' reveal binds
    the identity (ai.reveal_combat_card; the card leaves the hand;
    flag cleared).  SPECIFICITY: the AI's own choose echo (the label
    names the AI's deck) commits nothing.  Until the engine half
    lands, the missing surface is a LOUD alarm, never silence."""
    import ai
    got = _human_defender_cc_rig(fails, 'hidden_cc_precondition')
    if got is None:
        return
    rig, drv = got
    if not _arm_scouts(fails, drv, 'hidden_cc_precondition'):
        return
    # SPECIFICITY first: the AI's own choose_card echo
    rig.feed('$Wey chooses a card.[Shadow Strategy Card]')
    if any(k == 'cc_hidden_commit' for k, _ in drv.findings) \
            or drv.human_cc_committed:
        fails.append(('hidden_cc_ai_echo_committed',
                      drv.findings[-2:]))
    out = rig.feed('$Wey chooses a card.[Free Strategy Card]')
    if not hasattr(ai, 'commit_hidden_combat_card'):
        if not any('ENGINE SURFACE MISSING' in a for a in drv.alarms):
            fails.append(('hidden_cc_guard_silent', list(drv.alarms)))
        fails.append(('hidden_cc_engine_surface',
                      'NOT-MEASURED: ai.commit_hidden_combat_card '
                      'missing (K1 step 1 not landed)'))
        return
    if not drv.human_cc_committed:
        fails.append(('hidden_cc_flag', (drv.findings[-3:],
                                         drv.alarms[-2:], out)))
        return
    unrev = set(ai.combat_cards_unrevealed(drv.es) or ())
    if Side.FREE_PEOPLES not in unrev:
        fails.append(('hidden_cc_engine_unrevealed', repr(unrev)))
    if _SCOUTS in drv.es.hands[Side.FREE_PEOPLES].hand:
        fails.append(('hidden_cc_identity_leaked', 'cid in hand '
                      'before the reveal'))
    rig.feed('<auto> moves The Red Arrow/Scouts from Free Peoples '
             'Hand (chosen) to Current FP Card.')
    for _ in range(8):
        if not drv.human_cc_committed and drv.chat_deferred is None:
            break
        rig.feed('<game> Wey rolls 1 1 1 1 1')
    if drv.human_cc_committed:
        fails.append(('reveal_flag', (drv.findings[-3:],
                                      drv.alarms[-2:])))
    if _SCOUTS in drv.es.hands[Side.FREE_PEOPLES].hand:
        fails.append(('reveal_left_in_hand', 'Scouts still in hand'))
    if Side.FREE_PEOPLES in set(ai.combat_cards_unrevealed(drv.es)
                                or ()):
        fails.append(('reveal_engine', 'engine still lists FP as '
                                       'unrevealed'))
    if not any(k == 'cc_reveal' for k, _ in drv.findings):
        fails.append(('reveal_row', drv.findings[-3:]))


def test_nonhand_chosen_reveal_refused(fails):
    """MEASURES (final3 real-client leg 2026-08-22): after the human's
    hidden combat-card commit, a reveal whose source is NOT a hand
    ('<auto> moves TITLE from FP Casualties (chosen) to Current FP
    Card.' - the chooser let the human pick a card from the casualties
    pile) is REFUSED: the hidden commit is withdrawn (the engine's
    decision re-opens as defender_combat_card), ONE client-native undo
    press, a refusal notice, finding 'cc_reveal_refused', NO alarm.
    Live: the line was inert to every lane, the commit stayed
    outstanding and the battle wedged silently.  SPECIFICITY: the
    same title from 'Free Peoples Hand (chosen)' is the normal reveal
    path (exercised by test_hidden_combat_card_commit_and_reveal)."""
    import ai
    got = _human_defender_cc_rig(fails, 'nonhand_cc_precondition')
    if got is None:
        return
    rig, drv = got
    if not _arm_scouts(fails, drv, 'nonhand_cc_precondition'):
        return
    rig.feed('$Wey chooses a card.[Free Strategy Card]')
    if not drv.human_cc_committed:
        fails.append(('nonhand_cc_commit', drv.findings[-2:]))
        return
    n_alarm0 = len(drv.alarms)
    out = rig.feed('<auto> moves Guards of the Citadel/Charge from FP '
                   'Casualties (chosen) to Current FP Card.')
    if drv.human_cc_committed:
        fails.append(('nonhand_cc_still_committed', drv.findings[-3:]))
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'combat:defender_combat_card':
        fails.append(('nonhand_cc_pending', dp and dp['kind']))
    if Side.FREE_PEOPLES in set(ai.combat_cards_unrevealed(drv.es)
                                or ()):
        fails.append(('nonhand_cc_engine_unrevealed', 'still listed'))
    if not any(i[0] == 'line' and 'Controls.819' in str(i[1])
               for i in out):
        fails.append(('nonhand_cc_undo_press', out))
    if not any(i[0] == 'chat' and 'not in your hand' in str(i[2])
               for i in out):
        fails.append(('nonhand_cc_notice', out))
    if not any(k == 'cc_reveal_refused' and 'not in hand' in str(v)
               for k, v in drv.findings):
        fails.append(('nonhand_cc_finding', drv.findings[-3:]))
    if len(drv.alarms) != n_alarm0:
        fails.append(('nonhand_cc_alarm', drv.alarms[n_alarm0:]))


def test_single_die_from_hunt_box_not_recover(fails):
    """MEASURES (W1#4): ONE action die dragged out of the Hunt Box
    back to its pool while the side still has unused dice is the
    human's self-undo shape, not the turn recover - the turn does not
    roll over.  SPECIFICITY: a multi-die drag from the same zone is
    the recover (turn +1)."""
    st = adopted_state()                    # FP: die 0 set, 3 unset
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    turn0 = drv.es.turn
    rig.feed('$Wey moves Free Action Die (C) from The Hunt For The '
             'Ring to FP Dice.')
    if drv.es.turn != turn0 or drv.phase != 'actions':
        fails.append(('hunt_die_rollover', (drv.es.turn, drv.phase)))
    # BL-13: the group recover only rolls the turn once BOTH pools
    # are spent - complete the precondition
    for _s in (Side.FREE_PEOPLES, Side.SHADOW):
        for _d in drv.es.dice[_s]:
            _d.used = True
    rig.feed('$Wey moves group ([2]Free Action Die (C)) from The Hunt '
             'For The Ring to FP Dice.')
    if drv.es.turn != turn0 + 1:
        fails.append(('hunt_group_recover', (drv.es.turn, drv.phase)))


def test_window_before_action_phase_refused(fails):
    """MEASURES (W1#5): a human die window closing before BOTH action
    pools rolled (phase 'declare') commits nothing, alternation is
    untouched (last_actor None), and the human gets the client-native
    refusal (undo press + chat notice).  SPECIFICITY: the same window
    in the action phase commits (test_window_commit)."""
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('early_window_precondition', 'no army_move'))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='declare')   # human = SH
    drv = rig.drv
    out = rig.feed(*lines)
    if drv.window_stats.get('matched', 0) or drv.es is not st:
        fails.append(('early_window_committed',
                      dict(drv.window_stats)))
    if drv.last_actor is not None:
        fails.append(('early_window_actor', drv.last_actor))
    if not any(i[0] == 'chat' for i in out):
        fails.append(('early_window_notice', [i[0] for i in out]))
    if not any(i[0] == 'line' and 'Controls.819' in str(i[1])
               for i in out):
        fails.append(('early_window_undo', out))


def test_lenient_mode_accepts_refused_window(fails):
    """MEASURES (USER 2026-08-22 'Enforce legal moves for HUMAN player
    (beta)' OFF): a human die window the gate would REFUSE (here: a
    window before the action phase opened - test_window_before_action_
    phase_refused pins the enforced twin) is ACCEPTED AS GIVEN - no
    chat refusal, no undo press, no hold; the staged die is spent on
    the engine side, the human counts as having acted, the finding
    'lenient_accept' names it.  The same holds for a window the gate
    would HOLD (a window while a pending is open).  SPECIFICITY: with
    enforcement ON the refusal/hold is what happens."""
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('lenient_precondition', 'no army_move'))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='declare')   # human = SH
    drv = rig.drv
    drv.enforce_human = False
    n_used0 = sum(1 for d in drv.es.dice[Side.SHADOW] if d.used)
    out = rig.feed(*lines)
    # BL-32b (2026-08-25): a lenient degrade must be VISIBLE - one
    # 'prompt' chat says it; a 'refusal'-kind chat is still forbidden
    if any(i[0] == 'chat' and i[1] == 'refusal' for i in out):
        fails.append(('lenient_chat_refusal', [i for i in out
                                               if i[0] == 'chat']))
    if not any(i[0] == 'chat' and 'accepted as given' in str(i[2])
               for i in out):
        fails.append(('lenient_degrade_invisible', out))
    if any(i[0] == 'line' and 'Controls.819' in str(i[1])
           for i in out):
        fails.append(('lenient_undo_press', out))
    if drv.holds:
        fails.append(('lenient_hold', len(drv.holds)))
    if not any(k == 'lenient_accept' for k, _ in drv.findings):
        fails.append(('lenient_finding', drv.findings[-3:]))
    n_used1 = sum(1 for d in drv.es.dice[Side.SHADOW] if d.used)
    if n_used1 != n_used0 + 1:
        fails.append(('lenient_die_spent', (n_used0, n_used1)))
    if drv.last_actor is not Side.SHADOW:
        fails.append(('lenient_actor', drv.last_actor))
    if drv.alarms:
        fails.append(('lenient_alarm', list(drv.alarms)))
    # a window the gate would HOLD (pending open) is accepted too
    st2 = adopted_state()
    a2, lines2 = _army_window_lines(st2)
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='actions')
    drv2 = rig2.drv
    drv2.enforce_human = False
    game_logic.open_declare_phase(drv2.es)      # an FP pending is open
    out2 = rig2.feed(*lines2)
    if drv2.holds or any(i[0] == 'chat' and i[1] == 'refusal'
                         for i in out2):
        fails.append(('lenient_hold_accepted', (len(drv2.holds), out2)))
    if not any(k == 'lenient_accept' for k, _ in drv2.findings):
        fails.append(('lenient_hold_finding', drv2.findings[-3:]))
    # enforcement ON: the very same window is refused (the twin pin)
    st3 = adopted_state()
    a3, lines3 = _army_window_lines(st3)
    rig3 = Rig(st3, Side.FREE_PEOPLES, phase='declare')
    out3 = rig3.feed(*lines3)
    if not any(i[0] == 'chat' for i in out3):
        fails.append(('enforced_twin_not_refused', [i[0] for i in out3]))


def test_lenient_mode_zero_eyes_and_reset(fails):
    """MEASURES (contract updated 2026-08-25, shgame6 S9): a CHAT
    ANSWER is validated regardless of enforce_human - '0e' under the
    engine's eye minimum is refused even in lenient mode (the lenient
    doctrine covers BOARD gestures; shgame6 played turn 2 a hunt die
    short through the old gate). Reset during a battle retains its
    choice, paid cards, buffered physical work and actor; it must not
    turn a recoverable bridge problem into a skipped combat action."""
    from wotr_bridge.live.undo_model import RestoreWatch
    st = fresh_state()
    st._fp_used_hunt_box_last_turn = True
    rig = Rig(st, Side.FREE_PEOPLES, phase='eyes')   # human = SHADOW
    drv = rig.drv
    drv.enforce_human = False
    out = rig.feed('<Wey> 0e')
    if not any(i[0] == 'chat' and 'at least' in str(i[2])
               for i in out):
        fails.append(('lenient_zero_eyes_not_refused', out))
    if not any(f[0] == 'eye_chat_refused' for f in drv.findings):
        fails.append(('lenient_zero_eyes_no_finding',
                      list(drv.findings)[-3:]))
    # Combat retry retains all evidence rather than clearing the battle.
    st2 = fresh_state()
    rig2 = Rig(st2, Side.SHADOW, phase='actions')
    d2 = rig2.drv
    d2.es.pending_combat = {'phase': 'attacker_combat_card',
                           'attacker_region': 'Minas Tirith',
                           'defender_region': 'Osgiliath',
                           'attacker_side': Side.FREE_PEOPLES,
                           'defender_side': Side.SHADOW,
                           'deciding_side': Side.FREE_PEOPLES}
    d2.holds.append(('w', 'res'))
    d2.alarms.append('SOMETHING BROKE')
    d2.human_cc_committed = True
    d2._attack_ask = ('x', 'y')
    d2.restore_watch.arm({'a': 'b'}, None)
    d2.deferred_tail.append({'op': 'move'})
    pending_before = dict(d2.es.pending_combat)
    actor_before = d2.last_actor
    items, report = d2.reset_and_resync()
    if (len(d2.holds) != 1 or d2.alarms != ['SOMETHING BROKE']
            or not d2.human_cc_committed or d2._attack_ask != ('x', 'y')
            or len(d2.deferred_tail) != 1 or not d2.restore_watch.armed):
        fails.append(('combat_reset_lost_evidence', (len(d2.holds), d2.alarms,
                                           d2.human_cc_committed,
                                           d2._attack_ask,
                                           len(d2.deferred_tail),
                                           d2.restore_watch.armed)))
    if d2.es.pending_combat != pending_before:
        fails.append(('combat_reset_changed_pending', d2.es.pending_combat))
    for word in ('combat retained', 'rolls', 'without board adoption'):
        if not any(word in r for r in report):
            fails.append(('reset_report_' + word.replace(' ', '_'),
                          report))
    if d2.last_actor is not actor_before:
        fails.append(('combat_reset_changed_actor', d2.last_actor))
    if any(item[0] == 'line' for item in items):
        fails.append(('combat_reset_emitted_piece_change', items))


def test_unowed_draw_refused_by_client_undo(fails):
    """MEASURES (W1#9): an UN-OWED human draw is refused through the
    client's OWN undo (one undo press for the draw line, the newest
    line) - never a forged corrective drag of the card piece; the
    engine hand never grew; the undo echo is consumed as bookkeeping
    (no ledger pop of the earlier OWED draw).  D-T2-4: a refused +
    restored mistake is a FINDING + notice, never an alarm."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    rig.feed('$Wey draws Free Character Card.')
    n1 = len(drv.es.hands[Side.FREE_PEOPLES].hand)
    n_commits = len(drv.commits)
    out = rig.feed('$Wey draws Free Character Card.')
    if not any(k == 'unowed_draw_refused' for k, _ in drv.findings):
        fails.append(('unowed_precondition', list(drv.findings[-3:])))
        return
    if drv.alarms:
        fails.append(('unowed_restored_alarm', list(drv.alarms)))
    if not any(i[0] == 'chat' for i in out):
        fails.append(('unowed_notice', out))
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n1:
        fails.append(('unowed_hand_grew',
                      len(drv.es.hands[Side.FREE_PEOPLES].hand)))
    undos = [i for i in out if i[0] == 'line'
             and 'Controls.819' in str(i[1])]
    if len(undos) != 1:
        fails.append(('unowed_undo_press', out))
    forged = [i for i in out if i[0] == 'line'
              and 'Controls.819' not in str(i[1])]
    if forged:
        fails.append(('unowed_forged_drag', forged))
    if any(k in ('unowed_draw_returned',) for k, _ in drv.findings):
        fails.append(('unowed_forged_row', drv.findings[-3:]))
    # the client's undo echo of OUR press is bookkeeping, not a pop
    rig.feed('$WOTR_AI undoes last action : Wey draws Free Character '
             'Card.')
    if len(drv.commits) != n_commits:
        fails.append(('unowed_echo_popped', [c[2] for c in drv.commits]))
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n1:
        fails.append(('unowed_echo_hand',
                      len(drv.es.hands[Side.FREE_PEOPLES].hand)))


def test_demanded_draw_routes_to_funnel_not_gesture(fails):
    """MEASURES (BL-132 L1+L2, fpgame18 wave 26): a human draw
    arriving while a deferral starves on the CARD funnel routes by
    OWED-NESS: it credits the funnel and NEVER joins the gesture
    window's fact bundle (the old post-terminator append made a
    legal fellowship_move refuse with 'your gesture differs on card
    draws'), and the consuming lane SETTLES in the same motion - one
    unit of owed-draw debt closes and turn_draws counts (the
    un-settled draw left a phantom (FP, character) demand alive into
    the action phase).  The window is stamped OWNER of the fed draw
    (the L3 re-issue gate reads it).  The engine hand must NOT grow
    here (the fed placeholder rides the deck; hands.human_draw
    would double-count).  The real deferral retry is pinned by
    test_window_rng_deferral - stubbed here so the fabricated window
    never reaches the matcher."""
    import types
    from wotr_bridge.live.driver import PhaseWindow
    st = fresh_state()
    st._owed_draws = {('FREE_PEOPLES', 'character'): 1}
    rig = Rig(st, Side.SHADOW, phase='actions')      # human = FP
    drv = rig.drv
    win = PhaseWindow()
    win._starve_call = types.SimpleNamespace(
        kind='draws', kw={'items': [('FREE_PEOPLES', 'character')]})
    drv.deferred = (win, Side.FREE_PEOPLES)
    drv._retry_blocked = lambda: None
    n_hand = len(drv.es.hands[Side.FREE_PEOPLES].hand)
    rig.feed('$Wey draws Free Character Card.')
    if win.records:
        fails.append(('demand_route_gesture_append',
                      [r.get('norm') for r in win.records]))
    if not any(k == 'demanded_draw_fed' for k, _ in drv.findings):
        fails.append(('demand_route_lane', list(drv.findings[-4:])))
    if drv.turn_draws.get((Side.FREE_PEOPLES, 'character')) != 1:
        fails.append(('demand_settle_count', dict(drv.turn_draws)))
    if (st._owed_draws or {}).get(('FREE_PEOPLES',
                                   'character')) != 0:
        fails.append(('demand_settle_debt', dict(st._owed_draws)))
    if getattr(win, '_fed_draws132', None) != {('fp', 'char'): 1}:
        fails.append(('demand_owner_stamp',
                      getattr(win, '_fed_draws132', None)))
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n_hand:
        fails.append(('demand_hand_double_count',
                      len(drv.es.hands[Side.FREE_PEOPLES].hand)))


def test_refusal_draw_reissue_gate(fails):
    """MEASURES (BL-132 L3, fpgame18 wave 26): _refuse re-issues a
    popped fed draw ONLY when it was externally granted (BL-94(b):
    identity kept in the engine hand across the refusal).  A draw
    stamped WINDOW-OWNED - fed against the refused window's own
    starving demand - dies with the rollback: NO '!CALL draw_*' goes
    out, no echo token arms, finding refusal_draw_not_reissued
    (fpgame18's re-issue drew a fresh physical card into a hand the
    engine had rolled back: 'deck fp char deck: engine=21
    client=20').  SPECIFICITY: the same shape UN-stamped still
    re-issues and arms the one-shot echo token (the BL-94(b)
    regression guard).  Span shape: the draw line sits BETWEEN the
    window's own lines (interleaved render) - a fed draw ABOVE the
    window instead makes the whole refusal undo-INELIGIBLE ('his
    later gesture' gate), i.e. hold-and-witness with no pop and no
    re-issue, which satisfies the same rule by never popping the
    draw at all."""
    import types
    from wotr_bridge.live.driver import PhaseWindow
    for stamped in (True, False):
        st = fresh_state()
        rig = Rig(st, Side.SHADOW, phase='actions')  # human = FP
        drv = rig.drv
        die_ln = ('$Wey moves Free Action Die (C) from FP Dice to '
                  'Current Action.')
        draw_ln = '$Wey draws Free Character Card.'
        kill_ln = ('$Wey kills Free Action Die (C) in Current '
                   'Action.')
        drv.feed.lines.append(die_ln)
        drv.feed.lines.append(draw_ln)
        drv.feed.lines.append(kill_ln)
        w = PhaseWindow()
        w.records.append({'op': 'move', 'norm': die_ln,
                          'raw': die_ln})
        w.records.append({'op': 'kill', 'norm': kill_ln,
                          'raw': kill_ln})
        if stamped:
            w._fed_draws132 = {('fp', 'char'): 1}
        out = drv._refuse(w, types.SimpleNamespace(
            note='pin refusal', starve_call=None))
        calls = [i for i in out if i[0] == 'call'
                 and 'draw_fp_char' in str(i[1])]
        tok = (getattr(drv, '_reissued_draws94', None)
               or {}).get(('fp', 'char'), 0)
        if stamped:
            if calls:
                fails.append(('gate_reissued_owned', out))
            if tok:
                fails.append(('gate_token_owned', tok))
            if not any(k == 'refusal_draw_not_reissued'
                       for k, _ in drv.findings):
                fails.append(('gate_finding',
                              list(drv.findings[-4:])))
        else:
            if not calls:
                fails.append(('gate_bl94_regressed', out))
            if not tok:
                fails.append(('gate_bl94_token', tok))


def _akey(st, name):
    from board import sanitize
    return next((k for k in st.armies
                 if sanitize(str(k)) == sanitize(name)), None)


def test_siege_movein_banks_human_move_demand(fails):
    """MEASURES (BL-136(i), shgame16 S2/S3): a HUMAN attacker's siege
    move-in - the engine relocates the besiegers INSIDE the defender's
    (AI) retreat apply - is DEMANDED: the stamped move-in opens the
    relocation lane for an 'ai:' sweep (the 'chat:' gate banked
    nothing, no demand named the move, and auto-reconcile adopted the
    client's un-moved army back OUT of the stronghold region: two
    ARMY dice burned, no siege ever formed).  CONTRACT: unit_move /
    naz_move rows osgiliath -> minas tirith on the ledger, an
    'advance your army' note, the client's group drag CLAIMS them
    (ledger drains), and the post-condition alarm stays silent.
    SPECIFICITY: the same relocation in an 'ai:' sweep WITHOUT the
    stamp banks nothing (the wir3 rule intact)."""
    import ai
    from board import sanitize
    pre = fresh_state()
    osg = _akey(pre, 'Osgiliath')
    mm = _akey(pre, 'Minas Morgul')
    if osg is None or mm is None:
        fails.append(('smi_control', 'NOT-MEASURED: setup armies '
                      'missing (%r, %r)' % (osg, mm)))
        return
    # stage the Shadow attacker at Osgiliath (adjacent to Minas
    # Tirith) by moving the Minas Morgul stack's objects - move_army
    # enforces adjacency and Minas Morgul is not adjacent
    pre.armies[osg].units[:] = list(pre.armies[mm].units)
    pre.armies[osg].leaders[:] = list(pre.armies[mm].leaders)
    pre.armies[mm].units[:] = []
    pre.armies[mm].leaders[:] = []
    post = ai.clone_state(pre)
    pc = {'attacker_region': 'Osgiliath',
          'defender_region': 'Minas Tirith',
          'attacker_side': Side.SHADOW}
    ai._begin_siege(post, pc)
    if not getattr(post, '_siege_movein', None):
        fails.append(('smi_control', 'NOT-MEASURED: no move-in stamp '
                      '(%r)' % getattr(post, '_siege_movein_failed',
                                       None)))
        return
    rig = Rig(pre, Side.FREE_PEOPLES, phase='actions')  # human = SA
    drv = rig.drv
    items = drv.post_commit_sweep(pre, post, label='ai:retreat_into_siege')
    moves = [e for e in drv.echo_ledger
             if str(e.get('kind', '')).endswith('_move')
             and sanitize(str(e.get('region', ''))) == 'osgiliath'
             and sanitize(str(e.get('to_region', '')))
             == 'minas tirith']
    if not any(e['kind'] == 'unit_move' for e in moves):
        fails.append(('smi_banked', (drv.echo_ledger[-4:],
                                     drv.findings[-4:],
                                     drv.alarms[-2:])))
        return
    if not any(i[0] == 'chat' and 'drag it from osgiliath' in str(i[2])
               for i in items):
        fails.append(('smi_note', [i for i in items if i[0] == 'chat']))
    if any('UNDEMANDED' in a for a in drv.alarms):
        fails.append(('smi_postcondition_false_alarm', drv.alarms))
    # the human performs the demand: one group drag claims every row
    toks = []
    for e in moves:
        n = int(e['n'])
        if e['kind'] == 'naz_move':
            nm = 'Nazgul'
        else:
            nat, kd = str(e['tkey']).split(':', 1)
            nm = '%s %s' % (nat.title(), kd.title())
        toks.append(('[%d]%s' % (n, nm)) if n > 1 else nm)
    drv.es = post
    rig.feed('$Wey moves group (%s) from Osgiliath to Minas Tirith.'
             % ', '.join(toks))
    left = [e for e in drv.echo_ledger
            if str(e.get('kind', '')).endswith('_move')]
    if left:
        fails.append(('smi_claim', (toks, left, drv.findings[-3:])))
    # specificity: no stamp -> no relocation demand in an 'ai:' sweep
    post2 = ai.clone_state(pre)
    game_logic.move_army(post2, 'Osgiliath', 'Minas Tirith',
                         side=Side.SHADOW)
    post2._siege_movein = None
    rig2 = Rig(ai.clone_state(pre), Side.FREE_PEOPLES, phase='actions')
    rig2.drv.post_commit_sweep(pre, post2, label='ai:other')
    if any(str(e.get('kind', '')).endswith('_move')
           for e in rig2.drv.echo_ledger):
        fails.append(('smi_specificity', rig2.drv.echo_ledger))


def test_named_minion_kill_claims_echo(fails):
    """MEASURES (BL-139(a), shgame16 S6): a banked NAMED minion
    removal in a REGION zone is claimable by the human's kill line
    ('$Wey kills Saruman in Orthanc.') - the row sat on echo_open past
    an explicit raw kill while 'leaders orthanc (shadow): engine=0
    client=1' settled and escalated.  R2 covered the fellowship-box
    zones only."""
    st = fresh_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')     # human = SA
    drv = rig.drv
    drv.echo_ledger.append({'kind': 'named', 'side': 'shadow',
                            'region': 'orthanc', 'tkey': 'Saruman',
                            'n': 1, 'echo_id': 11})
    rig.feed('$Wey kills Saruman in Orthanc.')
    if any(e.get('tkey') == 'Saruman' for e in drv.echo_ledger):
        fails.append(('named_kill_unclaimed',
                      (drv.echo_ledger, drv.findings[-4:],
                       drv.alarms[-2:])))
    if not any(k == 'mirror_kill_echo' for k, _ in drv.findings):
        fails.append(('named_kill_finding', drv.findings[-4:]))


def test_decision_born_move_demand_not_rebased(fails):
    """MEASURES (BL-137, shgame16 S4): a relocation demand born of an
    EXPLICIT decision (card_option naming its regions) is REBASE-
    EXEMPT - the BL-77 rebase marched a conquest army out of Fords of
    Isen because the client held pieces there; the decision named
    North Dunland.  CONTRACT: with rebase_exempt the row keeps the
    engine's source + finding echo_move_rebase_exempt; without it the
    legacy rebase still applies (specificity)."""
    import ai
    from board import sanitize
    for exempt in (True, False):
        pre = fresh_state()
        if _akey(pre, 'North Dunland') is None:
            fails.append(('rbx_control', 'NOT-MEASURED: no army at '
                          'North Dunland in setup'))
            return
        post = ai.clone_state(pre)
        nd, sd = _akey(post, 'North Dunland'), _akey(post, 'South Dunland')
        if nd is None or sd is None:
            fails.append(('rbx_control', 'NOT-MEASURED: Dunland armies '
                          'missing (%r, %r)' % (nd, sd)))
            return
        # a pure relocation of the North Dunland stack (object move -
        # the delta pairing is what is under test, not adjacency)
        post.armies[sd].units.extend(post.armies[nd].units)
        post.armies[nd].units[:] = []
        rig = Rig(pre, Side.FREE_PEOPLES, phase='actions')  # human = SA
        drv = rig.drv
        drv._client_tkey_census = lambda tkey: {'fords of isen': 3}
        drv.post_commit_sweep(pre, post, label='chat:pick',
                              rebase_exempt=exempt)
        rows = [e for e in drv.echo_ledger
                if str(e.get('kind', '')).endswith('_move')]
        if not rows:
            fails.append(('rbx_banked', (exempt, drv.findings[-4:])))
            continue
        src = sanitize(str(rows[0].get('region', '')))
        if exempt:
            if src != 'north dunland':
                fails.append(('rbx_rerouted', (src, rows)))
            if not any(k == 'echo_move_rebase_exempt'
                       for k, _ in drv.findings):
                fails.append(('rbx_finding', drv.findings[-4:]))
        else:
            if src != 'fords of isen':
                fails.append(('rbx_legacy_rebase', (src, rows)))


def test_reconcile_stale_dump_and_erase_witness(fails):
    """MEASURES (BL-136(ii), shgame16 - the eraser behind S1/S2/S4):
    (1) DUMP FRESHNESS: an adoption over a dump OLDER than the last
    engine commit is DEFERRED (finding adopt_deferred_stale_dump, no
    board_adopt) while the commit is younger than 60s; an old commit
    lets the adoption proceed.  (2) ERASE WITNESS: an adoption report
    removing engine pieces from a region the engine touched seconds
    ago raises the RECONCILE ERASED alarm ('units minas morgul
    sauron/regular: 6 -> 5' - S1's silent deletion); a gain row and a
    stale region stay silent."""
    import os as _os
    import time as _time
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    now = _time.time()
    dp = _os.path.join(rig.td, 'ai_client_board.txt')
    with open(dp, 'w') as f:
        f.write('')
    _os.utime(dp, (now - 100, now - 100))
    drv.view.dump_path = dp
    drv.view.refresh_dump = lambda: True
    drv._last_commit_ts = now
    drv._adopt_board_now('auto-reconcile')
    if not any(k == 'adopt_deferred_stale_dump' for k, _ in drv.findings):
        fails.append(('stale_deferred', drv.findings[-4:]))
    if any(k == 'board_adopt' for k, _ in drv.findings):
        fails.append(('stale_adopted_anyway', drv.findings[-4:]))
    drv.findings[:] = []
    drv._last_commit_ts = now - 120
    drv._adopt_board_now('auto-reconcile')
    if any(k == 'adopt_deferred_stale_dump' for k, _ in drv.findings):
        fails.append(('stale_deferred_forever', drv.findings[-4:]))
    # (2) the eraser witness
    drv.alarms[:] = []
    drv._recent_delta136 = {'minas morgul': now, 'lorien': now - 200}
    drv._erase_witness136(['units minas morgul sauron/regular: 6 -> 5',
                           'units osgiliath sauron/regular: 0 -> 9',
                           'units lorien elves/regular: 3 -> 2'],
                          now=now + 5)
    hits = [a for a in drv.alarms if 'RECONCILE ERASED' in a]
    if len(hits) != 1 or 'minas morgul' not in hits[0]:
        fails.append(('erase_witness', drv.alarms))


def test_card_play_window_two_ticks(fails):
    """MEASURES (BL-134 interaction pin, tracker BL-66 race): the
    'no hand exit -> no play_event' prefilter is recomputed per match
    call, so a card play whose die stage lands one tick BEFORE its
    card row still matches: tick 1 (die stage alone) neither refuses
    nor defers; tick 2 (card row + recruit + kill) reaches the
    play_event candidate and defers on ITS draw (the Kindred flow of
    test_post_terminator_draw_feeds_deferred_window, split)."""
    st = adopted_state()
    st.dice[Side.FREE_PEOPLES][0].current = ActionDieFace.EVENT
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    drv.turn_draws = {(Side.FREE_PEOPLES, 'character'): 1,
                      (Side.FREE_PEOPLES, 'strategy'): 1}
    hm = drv.hand_models[Side.FREE_PEOPLES]
    drv.es.hands[Side.FREE_PEOPLES].hand.append(hm._mint('strategy'))
    cid = 'FREE_PEOPLES_Strategy_Kindred_of_Glorfindel'
    if cid not in drv.es.fp_strat_deck:
        drv.es.fp_strat_deck.append(cid)
    out1 = rig.feed('$Wey moves Free Action Die (P) from FP Dice to '
                    'Current Action.')
    if any(i[0] == 'line' for i in out1) or drv.deferred is not None:
        fails.append(('two_tick_stage', (out1, drv.deferred is not None,
                                         drv.findings[-3:])))
        return
    rig.feed(
        "$Wey moves Kindred of Glorfindel/Advantageous Position from "
        'Free Peoples Hand to Current FP Card.',
        '$Wey moves Elven Elite from FP Reinforcements to Rivendell.',
        "$Wey kills group (Free Action Die (P), Kindred of "
        'Glorfindel/Advantageous Position) in Current Action.')
    if drv.deferred is None:
        fails.append(('two_tick_deferred', (dict(drv.window_stats),
                                            drv.findings[-3:],
                                            drv.alarms[-2:])))
        return
    sc = getattr(drv.deferred[0], '_starve_call', None)
    if getattr(sc, 'kind', None) not in ('draws', 'card'):
        fails.append(('two_tick_starve_kind', getattr(sc, 'kind', sc)))


def test_track_ops_on_shim(fails):
    """MEASURES (BL-135, shgame16 S1): the seat's multi-leg planner
    mirrors leg 1's moves onto the shim before leg 2 plans - the
    reserve-box piece it picked for leg 1 is no longer in the box, so
    leg 2 cannot pick the SAME pid (one physical piece rendered into
    two settlements; the first recruit silently deleted)."""
    import importlib.util
    lab = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                       'audit', 'human_gestures.py')
    if not os.path.exists(lab):
        fails.append(('shim_control', 'NOT-MEASURED: lab audit module '
                      'absent at %s' % lab))
        return
    try:
        spec = importlib.util.spec_from_file_location('hg_pin135', lab)
        hg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hg)
    except Exception as e:
        fails.append(('shim_control', 'NOT-MEASURED: import %r' % (e,)))
        return
    shim = hg.DumpReplayer({'337': 'SA Reinforcements'}, {})
    hg.track_ops_on_shim(shim, [('move', {'pieces': ['337'],
                                          'to': 'Game.1969'})])
    z = shim.piece_region.get('337')
    if not z or 'reinf' in str(z).lower():
        fails.append(('shim_untracked', z))


def test_ai_companion_death_rendered_from_client_zone(fails):
    """MEASURES (shgame19 physical lane, T1): the AI is the FP and its
    companion dies inside the HUMAN's commit (Foul Thing from the
    Deep -> random companion casualty; the client's random draw parks
    the piece in 'SA Card Play').  The sweep renders the AI's OWN kill
    of that piece from wherever the client holds it (finding
    ai_companion_death_emitted) - before the fix nothing rendered and
    'companion Boromir: engine=dead/out, client zone=sa card play'
    settled into a divergence.  SPECIFICITY: a companion who SEPARATED
    (now an army leader) is not a death; a piece the client already
    shows dead renders nothing."""
    import ai
    import log_replay as lr
    pid = next((p for p in lr.COMPANIONS
                if 'Boromir' in set(lr.names_for_pid(str(p)) or ())),
               None)
    if pid is None:
        fails.append(('acd_control', 'NOT-MEASURED: no Boromir pid'))
        return
    pre = fresh_state()
    if not any(getattr(c, 'name', '') == 'Boromir'
               for c in pre.fellowship.companions):
        fails.append(('acd_control', 'NOT-MEASURED: Boromir not in the '
                      'setup fellowship'))
        return
    post = ai.clone_state(pre)
    post.fellowship.companions = [c for c in post.fellowship.companions
                                  if getattr(c, 'name', '') != 'Boromir']
    rig = Rig(pre, Side.FREE_PEOPLES, phase='actions')  # AI = FP
    drv = rig.drv
    drv.view.piece_region[str(pid)] = 'SA Card Play'
    items = drv.post_commit_sweep(pre, post, label='window')
    lines = [i[1] for i in items if i[0] == 'line' and str(pid) in i[1]]
    if not lines:
        fails.append(('acd_rendered', (items, drv.findings[-3:],
                                       drv.alarms[-2:])))
    if not any(k == 'ai_companion_death_emitted' for k, _ in drv.findings):
        fails.append(('acd_finding', drv.findings[-4:]))
    # specificity: already dead client-side -> nothing
    rig2 = Rig(ai.clone_state(pre), Side.FREE_PEOPLES, phase='actions')
    rig2.drv.view.piece_region[str(pid)] = 'FP Casualties'
    items2 = rig2.drv.post_commit_sweep(pre, post, label='window')
    if any(i[0] == 'line' and str(pid) in i[1] for i in items2):
        fails.append(('acd_double_kill', items2))


def test_blind_hand_discard_identity_from_dump(fails):
    """MEASURES (shgame21, physical lane): the feed sidecar REDACTS
    the human's own hand pids, so a hand-limit discard renders as a
    deck back with no identity ('kills Shadow Character Card in
    Shadow Hand.').  The identity lane resolves it from the CLIENT'S
    OWN DUMP: the decoded hand pid that left the hand zone is the
    killed card -> that cid's discard row matches and commits (engine
    hand loses exactly that card).  Before: 'discard_identity_missing'
    refusal + undo on every attempt, the seat re-acted the row forever
    and the AI's declare step starved behind it.  SPECIFICITY: two
    decoded pids gone -> loud refusal (discard_identity_ambiguous);
    none gone yet -> quiet hold (no chat)."""
    st = fresh_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')    # human = SA
    drv = rig.drv
    hand = drv.es.hands[Side.SHADOW].hand
    real = [c for c in list(drv.es.sh_char_deck)[-7:]]
    del drv.es.sh_char_deck[-7:]
    hand[:] = list(real)
    pid_map = {cid: str(160 + k) for k, cid in enumerate(real)}
    drv.view.card_pid_map = dict(pid_map)
    drv.view.refresh_dump = lambda: True
    for cid, pid in pid_map.items():
        drv.view.piece_region[pid] = 'SA Hand'
    game_logic.open_hand_limit_discard(drv.es)
    import ai
    dp = ai.describe_pending(drv.es)
    if dp is None or str(dp.get('kind')) != 'discard':
        fails.append(('bhd_control', 'NOT-MEASURED: no discard pending '
                      '(%r)' % (dp and dp.get('kind'),)))
        return
    victim = real[2]
    # the client killed the card: its pid left the hand zone
    drv.view.piece_region[pid_map[victim]] = 'SA Dead'
    rig.feed('$Wey kills Shadow Character Card in Shadow Hand.')
    if victim in drv.es.hands[Side.SHADOW].hand:
        fails.append(('bhd_commit', (drv.findings[-4:], drv.alarms[-2:])))
    if len(drv.es.hands[Side.SHADOW].hand) != 6:
        fails.append(('bhd_hand_size', len(drv.es.hands[Side.SHADOW].hand)))
    if any(k == 'discard_identity_missing' for k, _ in drv.findings):
        fails.append(('bhd_refused', drv.findings[-4:]))
    if not any(k == 'discard_identity_from_dump' for k, _ in drv.findings):
        fails.append(('bhd_finding', drv.findings[-4:]))
    # specificity: two decoded pids gone -> ambiguous, loud
    st2 = fresh_state()
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='actions')
    d2 = rig2.drv
    real2 = list(d2.es.sh_char_deck)[-7:]
    del d2.es.sh_char_deck[-7:]
    d2.es.hands[Side.SHADOW].hand[:] = list(real2)
    pm2 = {cid: str(170 + k) for k, cid in enumerate(real2)}
    d2.view.card_pid_map = dict(pm2)
    d2.view.refresh_dump = lambda: True
    for cid, pid in pm2.items():
        d2.view.piece_region[pid] = 'SA Hand'
    game_logic.open_hand_limit_discard(d2.es)
    d2.view.piece_region[pm2[real2[0]]] = 'SA Dead'
    d2.view.piece_region[pm2[real2[1]]] = 'SA Dead'
    rig2.feed('$Wey kills Shadow Character Card in Shadow Hand.')
    if not any(k == 'discard_identity_ambiguous' for k, _ in d2.findings):
        fails.append(('bhd_ambiguous', d2.findings[-4:]))
    if len(d2.es.hands[Side.SHADOW].hand) != 7:
        fails.append(('bhd_ambiguous_committed',
                      len(d2.es.hands[Side.SHADOW].hand)))


def test_human_declaration_planner_form(fails):
    """MEASURES (fpgame24, physical lane): the human's declaration in
    the EMITTER'S OWN corpus form (the seat renders it with the same
    planner the AI uses): pawn hop(s) + the FSP progress-reset DRAG
    ('moves FSP from Fellowship Track 2 to Fellowship Track 0') + the
    flip-unflip marker pair.  Both shapes must commit through the
    pending buffer: (a) declare IN PLACE (reset + 2 flips, no pawn
    move), (b) declare two regions away (hops + reset + 2 flips).
    Live, shape (a) sat in the pending buffer forever ('gesture match
    for 1387 buffered record(s)') while the seat re-acted the row."""
    import ai
    from board import sanitize
    for shape in ('in_place', 'hops', 'heal'):
        st = fresh_state()
        st.fellowship.progress = 2
        st.fellowship.hidden = True
        if shape == 'heal':
            # declaring in Rivendell (an FP stronghold) HEALS 1 - the
            # fpgame24 wedge: the human voices no corruption marker
            st.fellowship.corruption = 2
        rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
        drv = rig.drv
        rig.feed('$Wey draws Free Character Card.',
                 '$Wey draws Free Strategy Card.')
        dp = ai.describe_pending(drv.es)
        if dp is None or dp['kind'] != 'declare':
            fails.append(('dpf_control', 'NOT-MEASURED: declare not '
                          'armed (%r)' % (dp and dp['kind'],)))
            return
        if shape in ('in_place', 'heal'):
            lines = ['$Wey moves FSP from Fellowship Track 2 to '
                     'Fellowship Track 0.',
                     '$Wey flips The Fellowship in Rivendell.',
                     '$Wey flips The Fellowship in Rivendell.']
            want = 'rivendell'
        else:
            lines = ['$Wey moves The Fellowship from Rivendell to Fords '
                     'of Bruinen.',
                     '$Wey moves The Fellowship from Fords of Bruinen to '
                     'Hollin.',
                     '$Wey moves FSP from Fellowship Track 2 to '
                     'Fellowship Track 0.',
                     '$Wey flips The Fellowship in Hollin.',
                     '$Wey flips The Fellowship in Hollin.']
            want = 'hollin'
        rig.feed(*lines)
        reg = sanitize(str(game_logic.fellowship_region(drv.es)))
        buf = drv.pending_buf
        if reg != want or drv.es.fellowship.progress != 0 \
                or (buf is not None and buf.records):
            fails.append(('dpf_%s' % shape,
                          (reg, drv.es.fellowship.progress,
                           len(buf.records) if buf else 0,
                           drv.findings[-4:], drv.alarms[-2:])))
        if not drv.es.fellowship.hidden:
            fails.append(('dpf_%s_revealed' % shape, 'declare revealed'))
        if shape == 'heal':
            if drv.es.fellowship.corruption != 1:
                fails.append(('dpf_heal_corruption',
                              drv.es.fellowship.corruption))
            owed = (getattr(drv, '_track_owed', None) or {}).get('252')
            if owed is None and not any(
                    k == 'corruption_sweep_emitted'
                    for k, _ in drv.findings):
                fails.append(('dpf_heal_marker_undemanded',
                              drv.findings[-5:]))


def test_human_reveal_placement_marker_flip(fails):
    """MEASURES (fpgame26, physical lane): the hunt-forced REVEAL
    PLACEMENT in the human's corpus choreography - progress-reset drag
    + the marker flip (the pawn already at the placement region).  The
    engine revealed the Fellowship inside the hunt (hidden False
    before the pending), so the candidate carries no hidden change;
    the flip is the client marker catching up and must not read as a
    toggle demand.  CONTRACT: the placement commits (region, progress
    0, still revealed, buffer drained).  SPECIFICITY: a plain flip on
    a HIDDEN fellowship still demands the toggle (a hide gesture is
    not a placement)."""
    import ai
    from board import sanitize
    st = fresh_state()
    st.fellowship.progress = 3
    st.fellowship.hidden = False
    st._reveal_place_due = True
    dp = ai.describe_pending(st)
    if dp is None or dp['kind'] != 'reveal_place':
        fails.append(('hrp_control', 'NOT-MEASURED: no reveal_place '
                      'pending (%r)' % (dp and dp['kind'],)))
        return
    rig = Rig(st, Side.SHADOW, phase='actions')      # human = FP
    drv = rig.drv
    rig.feed('$Wey moves The Fellowship from Rivendell to Fords of '
             'Bruinen.',
             '$Wey moves FSP from Fellowship Track 3 to Fellowship '
             'Track 0.',
             '$Wey flips The Fellowship in Fords of Bruinen.')
    reg = sanitize(str(game_logic.fellowship_region(drv.es)))
    buf = drv.pending_buf
    if reg != 'fords of bruinen' or drv.es.fellowship.progress != 0 \
            or (buf is not None and buf.records) \
            or ai.describe_pending(drv.es) is not None:
        fails.append(('hrp_commit', (reg, drv.es.fellowship.progress,
                                     len(buf.records) if buf else 0,
                                     drv.findings[-4:], drv.alarms[-2:])))
    if drv.es.fellowship.hidden:
        fails.append(('hrp_hidden', 'placement re-hid the fellowship'))


def test_ambiguous_hold_is_voiced(fails):
    """MEASURES (BL-146, USER manual test): an AMBIGUOUS window hold
    speaks in chat at the instant - naming how many actions the
    gesture matched - instead of a silent engine-behind hold that
    surfaced twelve census rows later.  The match result is stubbed
    (the ambiguity itself is the matcher's business, pinned there);
    this pins the driver's VOICE on that branch."""
    import types
    from wotr_bridge.live import gesture_match as gm
    st = adopted_state()
    a, lines = _army_window_lines(st)
    if a is None:
        fails.append(('amb_control', 'NOT-MEASURED: no army move'))
        return
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')     # human = SA
    drv = rig.drv
    orig = gm.match_window
    fake_b = types.SimpleNamespace(action_type='army_move', params={})
    gm.match_window = lambda *args, **kw: gm.MatchResult(
        'ambiguous', chain=[[a], [fake_b]],
        note='2 distinct actions produce the window delta')
    try:
        out = rig.feed(*lines)
    finally:
        gm.match_window = orig
    chats = [i for i in out if i[0] == 'chat']
    if not any('2' in str(i[2]) and 'undo' in str(i[2]).lower()
               for i in chats):
        fails.append(('amb_voiced', (chats, drv.findings[-3:])))
    if not any(k == 'ambiguous_hold_voiced' for k, _ in drv.findings):
        fails.append(('amb_finding', drv.findings[-3:]))


def test_fear_fire_foes_group_split_is_one_gesture(fails):
    """MEASURES (USER manual test 2026-08-25, reproduced exactly): the
    human plays Fear! Fire! Foes! on an Event die - Legolas South
    Downs -> The Shire, Strider+Boromir Eastemnet -> Minas Tirith as
    ONE group drag, North activated + At War, Gondor activated, the
    Gandalf same-deck redraw, the group kill.  The engine offers the
    same moves as 'Strider+Boromir in one group' AND 'Strider, then
    Boromir' - IDENTICAL post states - and the matcher held the play
    'ambiguous'; the engine kept the card and the companions while
    the client moved on (12 census rows).  CONTRACT: same-state
    chains collapse to ONE gesture and the play COMMITS: companions
    moved, North at war, Gondor active, the card gone from the hand,
    one strategy card drawn."""
    import ai
    from board import sanitize
    from game_state import Nation
    st = fresh_state()

    def _sep(name, region):
        comp = next(c for c in st.fellowship.companions
                    if getattr(c, 'name', '') == name)
        st.fellowship.companions.remove(comp)
        game_logic.get_or_create_army(st, region).leaders.append(comp)
    _sep('Legolas', 'South Downs')
    _sep('Strider', 'Eastemnet')
    _sep('Boromir', 'Eastemnet')
    cid = 'FREE_PEOPLES_Strategy_Fear!_Fire!_Foes!'
    for lst in (st.fp_strat_deck, st.fp_char_deck):
        if cid in lst:
            lst.remove(cid)
    st.hands[Side.FREE_PEOPLES].hand[:] = [
        cid, 'FREE_PEOPLES_Character_Gwaihir_the_Windlord']
    game_logic.adopt_rolled_faces(
        st, [(Side.FREE_PEOPLES, 0, ActionDieFace.EVENT),
             (Side.SHADOW, 0, ActionDieFace.ARMY)])
    if getattr(st.fellowship.guide, 'name', '') != 'Gandalf the Grey':
        fails.append(('fff_control', 'NOT-MEASURED: guide'))
        return
    rig = Rig(st, Side.SHADOW, phase='actions')      # human = FP
    drv = rig.drv
    drv.turn_draws = {(Side.FREE_PEOPLES, 'character'): 1,
                      (Side.FREE_PEOPLES, 'strategy'): 1}
    n_hand = len(drv.es.hands[Side.FREE_PEOPLES].hand)
    rig.feed(
        '$Wey moves Free Action Die (P) from FP Dice to Current Action.',
        '$Wey moves Fear! Fire! Foes!/Shield-wall from Free Peoples '
        'Hand to Current FP Card.',
        '$Wey moves Legolas from South Downs to The Shire.',
        '$Wey moves group (Strider, Boromir) from Eastemnet to Minas '
        'Tirith.',
        '$Wey flips North in Political Track 3.',
        '$Wey moves North from Political Track 3 to Political Track 0.',
        '$Wey flips Gondor in Political Track 2.',
        '$Wey draws Free Strategy Card.',
        '$Wey kills group (Free Action Die (P), Fear! Fire! '
        'Foes!/Shield-wall) in Current Action.')
    if drv.window_stats.get('matched', 0) != 1:
        fails.append(('fff_commit', (dict(drv.window_stats),
                                     drv.findings[-4:], drv.alarms[-2:])))
        return
    es = drv.es

    def _leaders_at(region):
        k = next((k for k in es.armies
                  if sanitize(str(k)) == sanitize(region)), None)
        return sorted(getattr(l, 'name', '') for l in
                      (es.armies[k].leaders if k else [])
                      if getattr(l, 'name', ''))      # companions only
    if _leaders_at('Minas Tirith') != ['Boromir', 'Strider']:
        fails.append(('fff_group_moved', _leaders_at('Minas Tirith')))
    if _leaders_at('The Shire') != ['Legolas']:
        fails.append(('fff_legolas', _leaders_at('The Shire')))
    if es.political.boxes[Nation.THE_NORTH] != 3 \
            or not es.political.active[Nation.THE_NORTH]:
        fails.append(('fff_north', (es.political.boxes[Nation.THE_NORTH],
                                    es.political.active[Nation.THE_NORTH])))
    hand = es.hands[Side.FREE_PEOPLES].hand
    if cid in hand or len(hand) != n_hand:
        fails.append(('fff_hand', (list(hand), n_hand)))


def test_bl149_companion_death_removal_claims_echo(fails):
    """BL-149 (USER manual game 2026-08-28, the exact line): hunt tile
    1r, the human's guide Gandalf takes the damage; the engine banked
    the removal demand {'kind':'named','region':'fellowship','tkey':
    'Gandalf the Grey'} and said 'your destroyed pieces are still on
    the board - remove them'; the human performed it as 'kills group
    (1r, Gandalf the Grey) in Current Action' and the driver could NOT
    claim it: the guide/fellowship zone test in _consume_mirror_echo
    was dead code (a zone_key TUPLE compared against _GUIDE_ZONES'
    squashed STRINGS, and zk[0] - the class 'other' - read instead of
    the zone name), and the spent tile vetoed the group.  The kill
    fell into the open reveal_place pending and STALLED forever
    ('waiting on: gesture match for 1 buffered record(s)').  The R2
    (fpgame17) 'fix' for this shape shipped with no claim test.

    CONTRACT: every physical form of the demanded removal claims the
    echo - ledger empty, finding mirror_kill_echo, and NOTHING buffered
    into the open human pending.  SENSITIVITY: another companion's
    kill does not claim it; a fungible unit kill from a region-less
    zone is still refused (it needs its region); a tile-only kill is
    not an echo; a named kill in a DIFFERENT real region does not
    claim a region-banked entry."""
    from wotr_bridge.live import live_parser
    from test_live_adopt import hunt_pending
    entry = {'kind': 'named', 'side': 'fp', 'region': 'fellowship',
             'tkey': 'Gandalf the Grey', 'n': 1, 'echo_id': 1}
    forms = [
        '$Wey kills group (1r, Gandalf the Grey) in Current Action.',
        '$Wey kills Gandalf the Grey in Current Action.',
        '$Wey kills Gandalf the Grey in Guide Of The Fellowship.',
        '$Wey kills group (1r, Gandalf the Grey) in Guide Of The '
        'Fellowship.',
        '$Wey moves Gandalf the Grey from Current FP Card to FP '
        'Casualties.',
        '$Wey moves Gandalf the Grey from Guide Of The Fellowship to FP '
        'Casualties.',
    ]
    for line in forms:
        st = fresh_state()
        hunt_pending(st)                 # an OPEN human (FP) pending
        rig = Rig(st, Side.SHADOW, phase='actions')
        drv = rig.drv
        drv.echo_ledger = [dict(entry)]
        rig.feed(line)
        pb = drv.pending_buf
        buffered = len(pb.records) if (pb is not None and pb.records) else 0
        if drv.echo_ledger or buffered \
                or not any(k == 'mirror_kill_echo' for k, _ in drv.findings):
            fails.append(('bl149_unclaimed',
                          (line, drv.echo_ledger, buffered,
                           [k for k, _ in drv.findings[-3:]])))
    st = fresh_state()
    drv = Rig(st, Side.SHADOW, phase='actions').drv
    P = live_parser.parse
    drv.echo_ledger = [dict(entry)]
    if drv._consume_mirror_echo(P('$Wey kills Boromir in Current Action.')) \
            or not drv.echo_ledger:
        fails.append(('bl149_wrong_companion_claimed', True))
    drv.echo_ledger = [{'kind': 'unit', 'side': 'shadow', 'region': 'lorien',
                        'tkey': 'sauron:regular', 'n': 1, 'echo_id': 2}]
    if drv._consume_mirror_echo(P('$Wey kills Sauron Regular in Current '
                                  'Action.')):
        fails.append(('bl149_unit_claimed_without_region', True))
    drv.echo_ledger = [dict(entry)]
    if drv._consume_mirror_echo(P('$Wey kills 1r in Current Action.')) \
            or not drv.echo_ledger:
        fails.append(('bl149_tile_only_claimed', True))
    drv.echo_ledger = [{'kind': 'named', 'side': 'shadow',
                        'region': 'minas morgul', 'tkey': 'Saruman',
                        'n': 1, 'echo_id': 3}]
    if drv._consume_mirror_echo(P('$Wey kills Saruman in Orthanc.')):
        fails.append(('bl149_region_mismatch_claimed', True))


def _stalled_rig():
    """A human-FP rig with an OPEN hunt pending, a staged FP die and a
    card in the current slots (the repair's kill targets), and one
    buffered line that matches no hunt option."""
    from test_live_adopt import hunt_pending
    st = fresh_state()
    hunt_pending(st)
    rig = Rig(st, Side.SHADOW, phase='actions')          # human = FP
    drv = rig.drv
    drv.view.piece_region = {'210': 'Current Action', '145': 'FP Card Play',
                             '099': 'SA Dice'}
    drv.view.refresh_dump = lambda: True
    rig.feed('$Wey moves Gondor Regular from Minas Tirith to Osgiliath.')
    return rig, drv


def _tick_until_offer(drv, n=40):
    chats = []
    for _ in range(n):
        chats += [it for it in drv.tick() if it and it[0] == 'chat']
        if drv._repair is not None:
            break
    return chats


def test_self_repair_offer_is_explicit_logged_and_declinable(fails):
    """USER 2026-08-28: never a generic 'open pending'; on a stall the
    AI logs everything FIRST, then ASKS explicitly and touches nothing
    until 'yes'.  MEASURES: the stall text names the side, the pending
    kind, the buffered line, the option count and the expected
    gesture; the first stall signal writes ai_stall_*.log (with the
    full feed) and voices ONE offer naming what it will do and who acts
    after (the last-die-killed rule: a staged FP die counts as FP's
    kill -> Shadow to act); 'no' changes nothing."""
    import ai
    rig, drv = _stalled_rig()
    if drv.pending_buf is None or not drv.pending_buf.records:
        fails.append(('repair_control', 'NOT-MEASURED: nothing buffered'))
        return
    txt = drv.stall_text()
    for need in ("FREE_PEOPLES's", 'hunt', 'Gondor Regular',
                 'match none of its', 'legal options', 'expected:'):
        if need not in txt:
            fails.append(('stall_text_missing', (need, txt)))
    chats = _tick_until_offer(drv)
    r = drv._repair
    if r is None or r.get('phase') != 'offered':
        fails.append(('repair_not_offered',
                      (drv._buf_stall, [k for k, _ in drv.findings[-5:]])))
        return
    asks = [c for c in chats if 'self-repair' in str(c[2])]
    if len(asks) != 1:
        fails.append(('repair_ask_count', [str(c[2])[:60] for c in chats]))
    ask = str(asks[0][2]).lower() if asks else ''
    for need in ('hunt', 'say yes', 'shadow to act', 'kill the 2 staged'):
        if need not in ask:
            fails.append(('repair_ask_missing', (need, ask)))
    if r.get('to_act') is not Side.SHADOW:
        fails.append(('repair_turn_rule', r.get('to_act')))
    log = r.get('log')
    if not log or not os.path.exists(log) \
            or not os.path.basename(log).startswith('ai_stall_'):
        fails.append(('repair_log_missing', log))
    else:
        body = open(log, encoding='utf-8').read()
        for need in ('FULL FEED', 'Gondor Regular from Minas Tirith',
                     'explicit:', 'buffered records'):
            if need not in body:
                fails.append(('repair_log_content', need))
    # later stall notes must not re-chat the offer
    more = [it for it in drv.tick() for _ in range(1)
            if it and it[0] == 'chat' and 'self-repair' in str(it[2])]
    for _ in range(26):
        more += [it for it in drv.tick()
                 if it and it[0] == 'chat' and 'self-repair' in str(it[2])]
    if more:
        fails.append(('repair_ask_spam', len(more)))
    # 'no' -> nothing changed
    out = rig.feed('<Wey> no')
    if drv._repair is not None:
        fails.append(('repair_not_withdrawn_on_no', True))
    if not any(it[0] == 'chat' and 'nothing changed' in str(it[2])
               for it in out):
        fails.append(('repair_no_unvoiced', out))
    if drv.pending_buf is None or not drv.pending_buf.records:
        fails.append(('repair_no_touched_buffer', True))
    if ai.describe_pending(drv.es) is None:
        fails.append(('repair_no_touched_engine', True))


def test_self_repair_yes_kills_waits_resyncs_and_sets_turn(fails):
    """'yes' runs the repair in the USER's order: (3) one group kill of
    the staged die + current card under the AI's nick, the engine die
    marked spent, the AI never acting meanwhile; the client's echo of
    that kill is consumed, never a gesture; (4) once the client shows
    the slots empty: reset + resync client-as-is with the political
    track kept engine-side, the stuck buffer and the engine pending
    gone; (5) the turn from the last die killed (the staged FP die ->
    Shadow to act); (6) the announcement."""
    import ai
    rig, drv = _stalled_rig()
    _tick_until_offer(drv)
    if drv._repair is None:
        fails.append(('yes_control', 'NOT-MEASURED: no offer'))
        return
    pol_before = dict(drv.es.political.boxes)
    out = rig.feed('<Wey> yes')
    lines = [it[1] for it in out if it and it[0] == 'line']
    if not lines or '210' not in lines[0] or '145' not in lines[0] \
            or 'Controls.644' not in lines[0]:
        fails.append(('yes_kill_line', lines))
    if not drv.es.dice[Side.FREE_PEOPLES][0].used:
        fails.append(('yes_engine_die_not_spent', True))
    if drv._repair is None or drv._repair.get('phase') != 'kill':
        fails.append(('yes_phase', drv._repair))
        return
    if drv._ai_turn():
        fails.append(('yes_ai_acted_mid_repair', True))
    # the client has not shown the kill yet: the repair waits
    drv.tick()
    if drv._repair is None or drv._repair.get('phase') != 'kill':
        fails.append(('yes_did_not_wait_for_client', drv._repair))
        return
    # the client executes: pieces leave the slots, our kill echoes back
    drv.view.piece_region = {'210': 'FP Dice Used', '145': 'FP Discard',
                             '099': 'SA Dice'}
    out2 = rig.feed('$WOTR_AI_Shadow kills group (Free Action Die (C), '
                    'Gwaihir the Windlord) in Current Action.')
    if not any(k == 'stall_repair_kill_echo' for k, _ in drv.findings):
        fails.append(('yes_echo_not_consumed',
                      [k for k, _ in drv.findings[-6:]]))
    if drv._repair is not None:
        out2 += drv.tick()
    if drv._repair is not None:
        fails.append(('yes_not_finished', drv._repair.get('phase')))
        return
    if not any(k == 'stall_repair_done' for k, _ in drv.findings):
        fails.append(('yes_done_finding', [k for k, _ in drv.findings[-6:]]))
    if drv.pending_buf is not None and drv.pending_buf.records:
        fails.append(('yes_buffer_not_cleared', True))
    if ai.describe_pending(drv.es) is not None:
        fails.append(('yes_engine_pending_not_cleared',
                      ai.describe_pending(drv.es).get('kind')))
    if dict(drv.es.political.boxes) != pol_before:
        fails.append(('yes_politics_adopted_from_client', True))
    if drv.last_actor is not Side.FREE_PEOPLES:      # FP's die was last
        fails.append(('yes_turn_rule', drv.last_actor))
    done = [it for it in out2 if it and it[0] == 'chat'
            and 'self-repair done' in str(it[2])]
    if not done or 'shadow to act' not in str(done[0][2]).lower():
        fails.append(('yes_done_unvoiced', [str(i)[:80] for i in out2[-4:]]))


def test_side_of_last_die_kill_rule(fails):
    """The USER's turn rule, unit-pinned on raw feed lines: the last
    action-die kill (either nick, side from the die's own name), an
    undo cancels it, a die moved to the Hunt Box is not a kill, no
    kill -> None (the caller then lets the Free Peoples act first)."""
    from wotr_bridge.live.driver import side_of_last_die_kill as f
    S, F = Side.SHADOW, Side.FREE_PEOPLES
    cases = [
        ([], None),
        (['$Wey kills Shadow Action Die (M) in Current Action.'], S),
        (['$WOTR_AI_Shadow kills Shadow Action Die (A) in Current Action.',
          '$Wey kills group (Free Action Die (P), Fear! Fire! '
          'Foes!/Shield-wall) in Current Action.'], F),
        (['$Wey kills Free Action Die (C) in Current Action.',
          '$Wey undoes last action : Wey kills Free Action Die (C) in '
          'Current Action.'], None),
        (['$Wey kills Shadow Action Die (M) in Current Action.',
          '$Wey kills Free Action Die (C) in Current Action.',
          '$Wey undoes last action : Wey kills Free Action Die (C) in '
          'Current Action.'], S),
        (['$Wey moves Free Action Die (C) from Current Action to The Hunt '
          'For The Ring.'], None),
    ]
    for lines, want in cases:
        got = f(lines)
        if got is not want:
            fails.append(('last_die_kill', (lines[-1:] or ['<empty>'],
                                            got, want)))


def test_bl150_unvoiced_discard_leg_is_loud(fails):
    """BL-150 D2 (USER manual game 2026-08-28): the chain that
    discarded Challenge of the King committed with NO chat and NO
    finding - the human learned of it three refused plays later.
    CONTRACT: a committed chain carrying a discard_card leg the gesture
    never voiced (no hand-zone kill, no card played) raises a finding
    AND an alarm naming the card.  (D1 - the chain never traversing a
    discard pending - is pinned at the matcher level in
    test_live_matcher.test_bl150_chain_stops_before_a_discard_pending.)"""
    import ai
    from wotr_bridge.live.driver import PhaseWindow
    st2 = fresh_state()
    extra = list(st2.fp_char_deck[:7])
    for c in extra:
        st2.fp_char_deck.remove(c)
    st2.hands[Side.FREE_PEOPLES].hand[:] = extra
    game_logic.open_hand_limit_discard(st2)
    rig2 = Rig(st2, Side.SHADOW, phase='actions')
    d2 = rig2.drv
    menu = [a for a in ai.get_legal_actions(d2.es, Side.FREE_PEOPLES)
            if a.action_type == 'discard_card']
    if not menu:
        fails.append(('bl150_d2_control', 'no discard_card row'))
        return
    ok = d2._commit_chain([menu[0]], ['<test>'], 'test', gesture=PhaseWindow())
    if not ok:
        fails.append(('bl150_d2_control', 'commit failed'))
        return
    if not any(k == 'unvoiced_discard_leg' for k, _ in d2.findings) \
            or not any('UNVOICED DISCARD' in str(a) for a in d2.alarms):
        fails.append(('bl150_d2_silent', (d2.findings[-3:], d2.alarms[-2:])))


def test_paid_noop_sees_the_aragorn_crowning(fails):
    """USER manual game 2026-08-28: crowning Aragorn with a Will die
    changes a leader's IDENTITY (Strider -> Aragorn), not a count, and
    the paid-no-op effect signature (counts only) raised 'PAID NO-OP:
    window consumed a die with no board effect' on a legal, effective
    play.  CONTRACT: a commit whose only effect is the crowning is NOT
    a paid no-op; SENSITIVITY: a commit that spends a die and changes
    nothing still is."""
    import ai
    from game_state import ActionDieFace, Companion
    st = fresh_state()
    comp = next(c for c in st.fellowship.companions
                if getattr(c, 'name', '') == 'Strider')
    st.fellowship.companions.remove(comp)
    game_logic.get_or_create_army(st, 'Pelargir').leaders.append(comp)
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.WILL_OF_WEST),
        (Side.SHADOW, 0, ActionDieFace.ARMY)])
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    from copy import deepcopy
    post = deepcopy(st)              # independent objects (no aliasing)
    post.dice[Side.FREE_PEOPLES][0].used = True
    pel = next(a for k, a in post.armies.items()
               if 'pelargir' in str(k).lower())
    for l in pel.leaders:
        if getattr(l, 'name', '') == 'Strider':
            l.name = 'Aragorn'
    drv._paid_noop_check(st, post, 'crown-test')
    if any('PAID NO-OP' in str(a) for a in drv.alarms):
        fails.append(('crowning_called_noop', drv.alarms[-1]))
    post2 = deepcopy(st)
    post2.dice[Side.FREE_PEOPLES][0].used = True
    drv.alarms[:] = []
    drv._paid_noop_check(st, post2, 'noop-control')
    if not any('PAID NO-OP' in str(a) for a in drv.alarms):
        fails.append(('true_noop_missed', True))


def test_bl152_in_window_leader_recruit_is_not_owed(fails):
    """BL-152 (USER manual game 2026-08-28): the human played Eomer on a
    Muster die and dragged the Rohan leader + elite into Helm's Deep
    INSIDE the window; the post-commit sweep still said 'your recruited
    pieces are owed' and left 'mirror_echo_unclaimed: fp unit_in
    rohan:leader' as an error - _gesture_rows could not key a plain
    FP leader (count-only facts), so it was never subtracted.
    CONTRACT: after the window commits, NO arrival echo is banked for
    the leader or the elite and no 'recruited pieces are owed' chat is
    spoken; SENSITIVITY: _gesture_rows keys the leader arrival as
    rows['units']['rohan:leader']['plus'] and is not blocked."""
    import ai
    from game_state import ActionDieFace
    st = fresh_state()
    eomer = next((c for c in st.fp_strat_deck if 'omer' in c), None)
    if eomer is None:
        fails.append(('bl152_control', 'NOT-MEASURED: no Eomer cid'))
        return
    st.fp_strat_deck.remove(eomer)
    st.hands[Side.FREE_PEOPLES].hand[:] = [eomer]
    game_logic.adopt_rolled_faces(st, [
        (Side.FREE_PEOPLES, 0, ActionDieFace.MUSTER),
        (Side.SHADOW, 0, ActionDieFace.ARMY)])
    rig = Rig(st, Side.SHADOW, phase='actions')          # human = FP
    drv = rig.drv
    out = rig.feed(
        '$Wey moves Free Action Die (M) from FP Dice to Current Action.',
        '$Wey moves Eomer, Son of Eomund/Valour from Free Peoples Hand '
        'to Current FP Card.',
        "$Wey moves group (Rohan Leader, Rohan Elite) from FP "
        "Reinforcements to Helm's Deep.",
        '$Wey kills group (Free Action Die (M), Eomer, Son of '
        'Eomund/Valour) in Current Action.')
    if drv.window_stats.get('matched', 0) != 1:
        fails.append(('bl152_control', 'NOT-MEASURED: window did not '
                      'commit (%r %r)' % (dict(drv.window_stats),
                                          drv.findings[-3:])))
        return
    owed = [e for e in drv.echo_ledger
            if str(e.get('kind', '')).endswith('_in')]
    if owed:
        fails.append(('bl152_arrival_banked_as_owed', owed))
    if any(it[0] == 'chat' and 'owed' in str(it[2]) for it in out
           if it and len(it) > 2):
        fails.append(('bl152_owed_chat_spoken', True))
    # sensitivity: the converter keys the leader from the record token
    from wotr_bridge.live.driver import PhaseWindow
    pw = PhaseWindow()
    from wotr_bridge.live import live_parser
    pw.records.append(live_parser.parse(
        "$Wey moves group (Rohan Leader, Rohan Elite) from FP "
        "Reinforcements to Helm's Deep."))
    rows, blocked = drv._gesture_rows(pw, 'fp')
    if blocked or rows['units'].get('rohan:leader', {}).get('plus') \
            != [('helm s deep', 1)]:
        fails.append(('bl152_gesture_rows_leader',
                      (blocked, rows['units'].get('rohan:leader'))))


def test_livelock_breakers(fails):
    """MEASURES (BL-141/BL-142 belts): (1) a pending buffer that keeps
    GROWING without a match (the stall counter keyed on the record
    count, so 1400 unmatched declare records filled in silence) trips
    PENDING BUFFER WEDGED past 40 records: the records are released
    (buffer cleared), the pending stays open, the alarm is DIVERGENCE-
    class.  (2) the SAME refusal text three times inside the window
    raises REFUSAL LOOP once (shgame21: refusal + seat retry with no
    budget starved the AI 4400 ticks)."""
    import ai
    st = fresh_state()
    st.fellowship.progress = 2
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    rig.feed('$Wey draws Free Character Card.',
             '$Wey draws Free Strategy Card.')
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'declare':
        fails.append(('llb_control', 'NOT-MEASURED: declare not armed'))
        return
    # 42 pawn flips that never equal a declaration (odd count each pair
    # of ticks keeps the buffer unmatched and growing)
    for _ in range(21):
        rig.feed('$Wey flips The Fellowship in Rivendell.',
                 '$Wey moves The Fellowship from Rivendell to Hollin.')
    if not any('PENDING BUFFER WEDGED' in a for a in drv.alarms):
        fails.append(('llb_wedge_alarm',
                      (len(drv.pending_buf.records)
                       if drv.pending_buf else 0, drv.alarms[-3:])))
    if drv.pending_buf is not None and len(drv.pending_buf.records) > 40:
        fails.append(('llb_wedge_released', len(drv.pending_buf.records)))
    if ai.describe_pending(drv.es) is None:
        fails.append(('llb_wedge_pending_closed',
                      'the wedge must not answer the decision'))
    # (2) the refusal loop breaker
    drv2 = Rig(fresh_state(), Side.SHADOW, phase='actions').drv
    for _ in range(3):
        drv2._refusal_loop_check('that discard names no card identity')
    if not any('REFUSAL LOOP' in a for a in drv2.alarms):
        fails.append(('llb_refusal_loop', drv2.alarms))
    drv2._refusal_loop_check('that discard names no card identity')
    if sum(1 for a in drv2.alarms if 'REFUSAL LOOP' in a) != 1:
        fails.append(('llb_refusal_loop_once', drv2.alarms))


def test_human_eyes_judged_by_engine_menu(fails):
    """MEASURES (W1#6): the human Shadow's eye allocation is judged by
    the ENGINE's eye menu (eye_allocate counts: at most one per
    Companion in the Fellowship, at least one when the FP retrieved a
    die from the Hunt Box).  With ONE companion left, the second
    'allocates an eye.' is REFUSED (hunt box stays at 1, alarm, chat
    notice, client-native undo press) - the driver never answers
    legality itself.  SPECIFICITY: the first eye is accepted; the
    under-minimum roll is a loud alarm (the human cannot be forced)."""
    st = fresh_state()
    st.fellowship.companions = st.fellowship.companions[:1]
    rig = Rig(st, Side.FREE_PEOPLES, phase='eyes')   # human = SHADOW
    drv = rig.drv
    hb0 = len(drv.es.hunt_box)
    out1 = rig.feed('$Wey allocates an eye.')
    if len(drv.es.hunt_box) != hb0 + 1:
        fails.append(('eye_menu_first_refused',
                      (len(drv.es.hunt_box), drv.alarms[-2:])))
        return
    if any(i[0] == 'chat' for i in out1):
        fails.append(('eye_menu_first_notice', out1))
    out2 = rig.feed('$Wey allocates an eye.')
    if len(drv.es.hunt_box) != hb0 + 1:
        fails.append(('eye_menu_cap', len(drv.es.hunt_box)))
    if not any('EYE' in a for a in drv.alarms):
        fails.append(('eye_menu_alarm', list(drv.alarms)))
    if not any(i[0] == 'chat' for i in out2):
        fails.append(('eye_menu_notice', out2))
    if not any(i[0] == 'line' and 'Controls.819' in str(i[1])
               for i in out2):
        fails.append(('eye_menu_undo', out2))
    # the minimum: FP retrieved a die last turn -> at least one eye
    st2 = fresh_state()
    st2._fp_used_hunt_box_last_turn = True
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='eyes')
    rig2.feed('$Wey rolls M P P A A H C .')
    if not any('EYE' in a and 'MINIMUM' in a.upper()
               for a in rig2.drv.alarms):
        fails.append(('eye_minimum_silent', list(rig2.drv.alarms)))


def test_zero_eyes_chat_refused_under_minimum(fails):
    """MEASURES (D-T1-R3-2 / D-T2-7): the human Shadow's '0e' is judged
    by the ENGINE's eye menu like every eye gesture - when the Free
    Peoples retrieved a Hunt Box die last turn (p.17: at least one eye)
    '0e' is REFUSED (refusal chat, finding, phase stays 'eyes', no
    hunt-box change) instead of silently closing the window at zero.
    SENSITIVITY: the same '0e' WITHOUT the minimum is accepted (no
    refusal) - the refusal is the menu's, not a blanket ban."""
    st = fresh_state()
    st._fp_used_hunt_box_last_turn = True
    rig = Rig(st, Side.FREE_PEOPLES, phase='eyes')   # human = SHADOW
    drv = rig.drv
    hb0 = len(drv.es.hunt_box)
    out = rig.feed('<Wey> 0e')
    chats = [i for i in out if i[0] == 'chat']
    if not chats or not any('at least 1 eye' in str(c[2]) for c in chats):
        fails.append(('zero_eyes_min_refusal', out))
    if drv.phase != 'eyes':
        fails.append(('zero_eyes_min_phase', drv.phase))
    if len(drv.es.hunt_box) != hb0:
        fails.append(('zero_eyes_min_hunt_box', len(drv.es.hunt_box)))
    if not any(f[0] == 'eye_chat_refused' and 'minimum' in str(f[1])
               for f in drv.findings):
        fails.append(('zero_eyes_min_finding', list(drv.findings)[-3:]))
    # sensitivity: no minimum -> '0e' accepted silently
    st2 = fresh_state()
    st2._fp_used_hunt_box_last_turn = False
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='eyes')
    out2 = rig2.feed('<Wey> 0e')
    if any(i[0] == 'chat' and 'at least' in str(i[2]) for i in out2):
        fails.append(('zero_eyes_free_refused', out2))
    if any(f[0] == 'eye_chat_refused' for f in rig2.drv.findings):
        fails.append(('zero_eyes_free_finding', list(rig2.drv.findings)))


def test_ai_deck_draw_starve_ordered(fails):
    """MEASURES (D-T1-R3-3 / D-T2-8): a 'draws' starve whose every item
    is the AI's OWN deck (the Witch-king Sorcerer draw inside a
    human-decided round) is surfaced as an 'rng_ask' (the server turns
    it into '!CALL draw_*'), not as the blocked note that starved the
    live game.  SPECIFICITY: a starve naming the HUMAN's deck still
    gets the note (never auto-ordered) and a mixed one too."""
    from wotr_bridge.live.randomness import NeedClientCall
    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='actions')     # AI = SHADOW
    drv = rig.drv

    class _EmptyBuf(object):
        records = ()

    def route_with(items):
        # the starve-surfacing tail of tick(): a buffered human
        # pending + a starving call in the randomness slot
        drv.rand._pending = NeedClientCall('draws', items=items)
        drv.pending_buf = _EmptyBuf()   # an OPEN (empty) human buffer
        drv._last_blocked_kind = None
        try:
            return drv.tick()
        finally:
            drv.pending_buf = None
            drv.rand._pending = None
    out = route_with([('SHADOW', 'char')])
    asks = [i for i in out if i[0] == 'rng_ask']
    if not asks or asks[-1][1].kind != 'draws':
        fails.append(('ai_deck_draw_ask', out))
    out2 = route_with([('FREE_PEOPLES', 'char')])
    if any(i[0] == 'rng_ask' for i in out2) or not any(
            i[0] == 'note' and 'blocked' in str(i[1]) for i in out2):
        fails.append(('human_deck_draw_not_ordered', out2))
    out3 = route_with([('SHADOW', 'strat'), ('FREE_PEOPLES', 'strat')])
    if any(i[0] == 'rng_ask' for i in out3):
        fails.append(('mixed_deck_draw_not_ordered', out3))


def test_special_hunt_tile_draw_feeds_tile_funnel(fails):
    """MEASURES (D-T2-9): a SPECIAL hunt tile the client draws
    ('$Wey draws Phial of Galadriel.') is fed to the tile funnel and
    answers a starving engine tile pop with THAT tile - for every
    special tile name the client can render (ONE answerer:
    log_replay.HUNT_TILE_PIECES).  Live: the line parsed as an
    unknown piece, the AI's Mordor-track move starved on 'tile'
    forever.  SENSITIVITY: a non-tile draw ('draws Gandalf the
    Grey.') feeds NOTHING to the tile funnel."""
    import game_logic
    from wotr_bridge.live import live_parser
    from wotr_bridge.live import randomness as R
    from log_replay import HUNT_TILE_PIECES
    names = sorted(set(v for v in HUNT_TILE_PIECES.values()
                       if not v[0].isdigit() and v != 'Eye'))
    if len(names) != 8:
        fails.append(('special_tile_table', names))
    for name in names:
        st = fresh_state()
        st.hunt_pool = list(gs.SPECIAL_HUNT_TILES.values())             + list(st.hunt_pool)
        rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
        drv = rig.drv
        cr = drv.rand
        n0 = len(drv.es.hunt_pool)
        cr.begin_attempt()
        got = None
        with cr.advance():
            try:
                game_logic.pop_hunt_tile(drv.es)
            except R.NeedClientCall as e:
                got = e
        if got is None or got.kind != 'tile':
            fails.append(('special_tile_starve', (name, repr(got))))
            continue
        rec = live_parser.parse('$Wey draws %s.' % name)
        drv._feed_draw_rec(rec)
        if any(f[0] == 'rng_tile_feed' for f in drv.findings):
            fails.append(('special_tile_feed_error',
                          (name, list(drv.findings))))
            continue
        cr.begin_attempt()
        try:
            with cr.advance():
                t = game_logic.pop_hunt_tile(drv.es)
        except BaseException as e:     # NeedClientCall / halt
            fails.append(('special_tile_not_fed', (name, repr(e))))
            continue
        want = R.tile_face(name)
        if R.tile_face(t) != want:
            fails.append(('special_tile_identity',
                          (name, R.tile_face(t))))
        if len(drv.es.hunt_pool) != n0 - 1:
            fails.append(('special_tile_pop', (name, n0,
                                               len(drv.es.hunt_pool))))
        cr.step_commit()
    # sensitivity: a named companion draw never feeds the tile funnel
    st = fresh_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')
    drv = rig.drv
    before = len(drv.rand._feed['tiles'])
    drv._feed_draw_rec(live_parser.parse('$Wey draws Gandalf the Grey.'))
    if len(drv.rand._feed['tiles']) != before:
        fails.append(('non_tile_draw_fed_as_tile',
                      drv.rand._feed['tiles']))


def test_used_die_out_of_hunt_box_refused(fails):
    """MEASURES (D-T2-6): after a COMMITTED Fellowship move (the die
    sits in the Hunt Box, used), the human dragging that single die
    back to the pool mid-turn (the early recover click / a mistaken
    drag) is REFUSED: client-native undo press(es) of that ONE line +
    a notice + the 'hunt_box_die_refused' finding; the engine die stays
    used, the move stays applied, NO alarm.  Live: the line was booked
    as a silent 'self-undo' and the boards split.  SENSITIVITY: a
    single die back from 'Current Action' (merely staged, no pid)
    keeps the silent self-undo - no refusal, no undo press."""
    st = _hunt_armed_state()
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    rig.feed(*_FS_WINDOW)
    rig.feed('<game> Wey rolls 1')
    if drv.es.fellowship.progress != 1 or drv.deferred is not None:
        fails.append(('hb_die_precondition',
                      (drv.es.fellowship.progress, drv.deferred)))
        return
    dice = drv.es.dice[Side.FREE_PEOPLES]
    used0 = [bool(d.used) for d in dice]
    if not any(used0):
        fails.append(('hb_die_precondition_used', used0))
        return
    n_alarm0 = len(drv.alarms)
    out = rig.feed('$Wey moves Free Action Die (C) from The Hunt For '
                   'The Ring to FP Dice.')
    if not any(i[0] == 'chat' and 'end of the turn' in str(i[2])
               for i in out):
        fails.append(('hb_die_notice', out))
    if not any(i[0] == 'line' and 'Controls.819' in str(i[1])
               for i in out):
        fails.append(('hb_die_undo_press', [i for i in out
                                            if i[0] != 'ai_decide']))
    if not any(f[0] == 'hunt_box_die_refused' for f in drv.findings):
        fails.append(('hb_die_finding', list(drv.findings)[-3:]))
    if any(f[0] == 'die_self_undo' for f in drv.findings):
        fails.append(('hb_die_booked_self_undo', list(drv.findings)))
    if [bool(d.used) for d in dice] != used0:
        fails.append(('hb_die_engine_changed', [bool(d.used)
                                                for d in dice]))
    if drv.es.fellowship.progress != 1:
        fails.append(('hb_die_move_changed', drv.es.fellowship.progress))
    if len(drv.alarms) != n_alarm0:
        fails.append(('hb_die_alarm', drv.alarms[n_alarm0:]))
    # sensitivity: a staged die back from Current Action = self-undo
    st2 = _hunt_armed_state()
    rig2 = Rig(st2, Side.SHADOW, phase='actions')
    out2 = rig2.feed('$Wey moves Free Action Die (C) from Current '
                     'Action to FP Dice.')
    if any(i[0] == 'chat' for i in out2) or any(
            i[0] == 'line' and 'Controls.819' in str(i[1])
            for i in out2):
        fails.append(('staged_die_refused', out2))
    if not any(f[0] == 'die_self_undo' for f in rig2.drv.findings):
        fails.append(('staged_die_not_self_undo',
                      list(rig2.drv.findings)))


def test_early_eye_keeps_ai_declare_open(fails):
    """MEASURES (W1#7): in an AI-as-FP game a human Shadow eye
    gesture arriving BEFORE the AI's declare step closed keeps the
    Fellowship phase open. The premature eye is refused without
    changing the Hunt Box or choosing for the AI. Declaration then
    Guide must settle before a fresh legal eye gesture is accepted."""
    import ai
    st = fresh_state()
    st.fellowship.progress = 2
    rig = Rig(st, Side.FREE_PEOPLES, phase='draws')   # AI = FP
    drv = rig.drv
    rig.feed('$Wey draws Shadow Character Card.',
             '$Wey draws Shadow Strategy Card.')
    if drv.phase != 'declare':
        fails.append(('early_eye_precondition', drv.phase))
        return
    hb0 = len(drv.es.hunt_box)
    refused = rig.feed('$Wey allocates an eye.')
    if len(drv.es.hunt_box) != hb0:
        fails.append(('early_eye_prematurely_allocated', len(drv.es.hunt_box)))
    if not any(item[0] == 'chat' and item[1] == 'refusal' for item in refused):
        fails.append(('early_eye_refusal_missing', refused))
    if drv.phase != 'declare':
        fails.append(('early_eye_phase', drv.phase))
    if any(c[2] == 'declare-decline' for c in drv.commits):
        fails.append(('early_eye_declined_for_ai',
                      [c[2] for c in drv.commits]))
    drv.ai_action_dp()          # the AI's tick re-arms its menu
    dp = ai.describe_pending(drv.es, Side.FREE_PEOPLES)
    if dp is None or dp['kind'] != 'declare':
        fails.append(('early_eye_ai_menu_lost',
                      repr(dp and dp['kind'])))
    drv.after_ai_declare()
    dp = ai.describe_pending(drv.es, Side.FREE_PEOPLES)
    if drv.phase != 'guide' or dp is None or dp['kind'] != 'guide_change':
        fails.append(('early_eye_expected_guide', (drv.phase, repr(dp))))
        return
    keep = next(a for a in dp['actions'] if a.params.get('companion') == drv.es.fellowship.guide.name)
    drv.es = ai.apply_action(drv.es, keep)
    drv.hands.es = drv.es
    drv._complete_fellowship_guide()
    if drv.phase != 'eyes':
        fails.append(('early_eye_after_guide', drv.phase))
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'eye' or dp['side'] is not Side.SHADOW:
        fails.append(('early_eye_expected_hunt_pending', repr(dp and dp['kind'])))
    if len(drv.es.hunt_box) != hb0:
        fails.append(('early_eye_replayed_without_new_gesture', len(drv.es.hunt_box)))
    rig.feed('$Wey allocates an eye.')
    if len(drv.es.hunt_box) != hb0 + 1:
        fails.append(('legal_eye_not_allocated_after_guide', len(drv.es.hunt_box)))


def test_post_terminator_draw_feeds_deferred_window(fails):
    """MEASURES (W1#8): a human card play whose effect draws a card
    (Kindred of Glorfindel: recruit in Rivendell, then draw one
    Strategy card; Gandalf the Grey then offers an optional same-deck
    draw for an Event-die play) with the draws rendered
    AFTER the die kill: the window defers on the card starve, each
    post-terminator draw line is routed to the draw funnel
    (placeholder swap + feed) and the window COMMITS on the retry -
    never refused as UN-OWED, never undone; the recruit and the
    drawn placeholders are engine-side."""
    import ai
    st = adopted_state()
    st.dice[Side.FREE_PEOPLES][0].current = ActionDieFace.EVENT
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    drv.turn_draws = {(Side.FREE_PEOPLES, 'character'): 1,
                      (Side.FREE_PEOPLES, 'strategy'): 1}
    hm = drv.hand_models[Side.FREE_PEOPLES]
    hand = drv.es.hands[Side.FREE_PEOPLES].hand
    hand.append(hm._mint('strategy'))
    cid = 'FREE_PEOPLES_Strategy_Kindred_of_Glorfindel'
    if cid not in drv.es.fp_strat_deck:
        drv.es.fp_strat_deck.append(cid)
    n_hand = len(hand)
    out = rig.feed(
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        "$Wey moves Kindred of Glorfindel/Advantageous Position from "
        'Free Peoples Hand to Current FP Card.',
        '$Wey moves Elven Elite from FP Reinforcements to Rivendell.',
        "$Wey kills group (Free Action Die (P), Kindred of "
        'Glorfindel/Advantageous Position) in Current Action.')
    if drv.deferred is None:
        fails.append(('post_draw_precondition',
                      'NOT-MEASURED: window not deferred on the draw '
                      '(%r %r)' % (dict(drv.window_stats),
                                   drv.findings[-3:])))
        return
    out2 = rig.feed('$Wey draws Free Strategy Card.')
    pending = getattr(drv.es, 'pending_card_option', None) or {}
    if drv.deferred is not None or pending.get('card_id') != '__gandalf_draw__':
        fails.append(('post_draw_control', 'printed draw must finish '
                      'before the optional Gandalf decision: %r' % pending))
        return
    if len(drv.es.hands[Side.FREE_PEOPLES].hand) != n_hand:
        fails.append(('post_draw_before_bonus_hand',
                      drv.es.hands[Side.FREE_PEOPLES].hand))
    out2 += rig.feed('<Wey> draw')
    if drv.chat_deferred is None:
        fails.append(('post_draw_bonus_not_awaiting', drv.findings[-3:]))
    out2 += rig.feed('$Wey draws Free Strategy Card.')
    if any('UN-OWED' in a for a in drv.alarms):
        fails.append(('post_draw_refused', list(drv.alarms)))
    if any(i[0] == 'line' for i in out2):
        fails.append(('post_draw_undone', out2))
    if drv.window_stats.get('matched', 0) != 1 \
            or drv.deferred is not None:
        fails.append(('post_draw_committed',
                      (dict(drv.window_stats), drv.findings[-3:],
                       drv.alarms[-2:])))
        return
    hand2 = drv.es.hands[Side.FREE_PEOPLES].hand
    # the played card left, two granted draws arrived
    if len(hand2) != n_hand + 1 or cid in hand2:
        fails.append(('post_draw_hand', (n_hand, hand2)))
    exc = rig.rand.unconsumed()
    if any(exc.values()):
        fails.append(('post_draw_honesty', exc))
    from board import sanitize
    key = next((k for k in drv.es.armies
                if sanitize(k) == 'rivendell'), None)
    n_elf = (len([u for u in drv.es.armies[key].units
                  if 'ELVE' in str(u.nation.name).upper()])
             if key else 0)
    if n_elf < 3:                 # setup 2 + the recruit
        fails.append(('post_draw_recruit', n_elf))


def test_two_draws_in_window_match(fails):
    """MEASURES (the USER's own feed, play_client/ai_feed.txt L27-32,
    refused live as 'differs on the card played, the action die'):
    King Brand's Men played with an Event die under Gandalf the Grey
    as guide = recruit two North regulars in Dale + TWO Strategy
    draws (the card's one + the guide's same-deck draw), all rendered
    inside the window.  The window COMMITS: the second draw's
    placeholder swap must not eat the first's (the deck-desync halt
    that silently broke the chain)."""
    st = adopted_state()
    st.dice[Side.FREE_PEOPLES][0].current = ActionDieFace.EVENT
    rig = Rig(st, Side.SHADOW, phase='actions')     # human = FP
    drv = rig.drv
    hm = drv.hand_models[Side.FREE_PEOPLES]
    drv.es.hands[Side.FREE_PEOPLES].hand.append(hm._mint('strategy'))
    cid = "FREE_PEOPLES_Strategy_King_Brand's_Men"
    if cid not in drv.es.fp_strat_deck:
        drv.es.fp_strat_deck.append(cid)
    out = rig.feed(
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        "$Wey moves King Brand's Men/Shield-wall from Free Peoples "
        'Hand to Current FP Card.',
        '$Wey moves group ([2]North Regular) from FP Reinforcements '
        'to Dale.',
        '$Wey draws Free Strategy Card.',
        '$Wey draws Free Strategy Card.',
        "$Wey kills group (Free Action Die (P), King Brand's Men/"
        'Shield-wall) in Current Action.')
    if drv.window_stats.get('matched', 0) != 1:
        fails.append(('two_draws_committed',
                      (dict(drv.window_stats), drv.findings[-3:],
                       [i for i in out if i[0] == 'chat'])))
        return
    hand = drv.es.hands[Side.FREE_PEOPLES].hand
    if len(hand) != 2 or cid in hand:
        fails.append(('two_draws_hand', list(hand)))
    exc = rig.rand.unconsumed()
    if any(exc.values()):
        fails.append(('two_draws_honesty', exc))


def test_lenient_hand_facts(fails):
    """MEASURES (tester 2026-08-24 'prompted to discard at 6 cards' -
    reproduced, then pinned here): lenient mode takes a card-play
    window AS GIVEN INCLUDING the card the client showed leaving the
    human's hand.  Before the fix the engine kept the played card as a
    phantom (_adopt_board_now never adopts hands), so the next turn's
    draws opened a hand-limit pending the player's one discard could
    never close: engine 7, player 6, 'your discard decision is still
    open'.  Legs: (1) the lenient-accepted play removes the REAL cid
    from the modeled hand and conserves it in discarded_face_down;
    (2) the tester cycle - draws to 7, ONE discard closes the pending
    at hand 6; SPECIFICITY (3): with enforcement ON the same window is
    refused and the hand keeps all 6 (the removal is lenient-gated)."""
    import ai

    def _to_six(rig_):
        rig_.drv.es._owed_draws = {
            (Side.FREE_PEOPLES.name, 'character'): 10}
        n = 0
        while len(rig_.drv.es.hands[Side.FREE_PEOPLES].hand) < 6:
            rig_.feed('$Wey draws Free Character Card.')
            n += 1
            if n > 10:
                return False
        return True

    st = fresh_state()
    rig = Rig(st, Side.SHADOW, phase='draws')       # human = FP
    drv = rig.drv
    drv.enforce_human = False
    if not _to_six(rig):
        fails.append(('lh_precondition', 'hand never reached 6'))
        return
    # a real FP CHARACTER cid still in the fictional deck names the
    # title the client renders for the play
    cid = next((c for c in drv.es.fp_char_deck
                if str(c).startswith('FREE_PEOPLES_Character_')), None)
    if cid is None:
        fails.append(('lh_precondition', 'no char cid in deck'))
        return
    title = str(cid)[len('FREE_PEOPLES_Character_'):].replace('_', ' ')
    play_lines = (
        '$Wey moves Free Action Die (E) from FP Dice to Current '
        'Action.',
        '$Wey moves %s from Free Peoples Hand to Current FP Card.'
        % title,
        '$Wey kills Free Action Die (E) in Current Action.')
    hands_fp = drv.es.hands[Side.FREE_PEOPLES]
    n_dfd0 = len(hands_fp.discarded_face_down)
    rig.feed(*play_lines)
    if not any(k == 'lenient_accept' for k, _ in drv.findings):
        fails.append(('lh_lenient_path', drv.findings[-3:]))
    if len(hands_fp.hand) != 5:
        fails.append(('lh_hand_removed', len(hands_fp.hand)))
    if cid not in hands_fp.discarded_face_down \
            or len(hands_fp.discarded_face_down) != n_dfd0 + 1:
        fails.append(('lh_conserved',
                      hands_fp.discarded_face_down[-2:]))
    if any(k == 'lenient_hand_title_unresolved'
           for k, _ in drv.findings):
        fails.append(('lh_title_resolved', drv.findings[-3:]))
    # (2) the tester cycle: turn draws to 7, ONE discard closes it
    drv.turn_draws.clear()
    rig.feed('$Wey draws Free Character Card.')
    rig.feed('$Wey draws Free Strategy Card.')
    dp = ai.describe_pending(drv.es)
    if dp is None or str(dp['kind']) != 'discard':
        fails.append(('lh_pending_opens', repr(dp and dp['kind'])))
        return
    rig.feed('$Wey kills Free Character Card in Free Peoples Hand.')
    # the pending-buffer commit adopts a CLONE: re-read drv.es
    n_hand = len(drv.es.hands[Side.FREE_PEOPLES].hand)
    if ai.describe_pending(drv.es) is not None or n_hand != 6:
        fails.append(('lh_one_discard_closes',
                      (n_hand, ai.describe_pending(drv.es))))
    # (3) SPECIFICITY: enforcement ON refuses; the hand keeps all 6
    st2 = fresh_state()
    rig2 = Rig(st2, Side.SHADOW, phase='draws')
    if not _to_six(rig2):
        fails.append(('lh_enforced_precondition', ''))
        return
    rig2.feed(*play_lines)
    if len(rig2.drv.es.hands[Side.FREE_PEOPLES].hand) != 6:
        fails.append(('lh_enforced_keeps_hand',
                      len(rig2.drv.es.hands[Side.FREE_PEOPLES].hand)))
    if any(k == 'lenient_accept' for k, _ in rig2.drv.findings):
        fails.append(('lh_enforced_no_lenient',
                      rig2.drv.findings[-3:]))


def test_scouts_retreat_drag_in_reveal_gap(fails):
    """MEASURES (tester 2026-08-24 'played Scouts, retreated, it asked
    me for a gesture and resync did not clear it'): a human board
    record arriving in the K1 REVEAL GAP (the AI's combat card is
    committed hidden, describe_pending is None for one server
    round-trip) is HELD in the pending buffer - never silently
    dropped - and resolves the decision the reveal opens (the Scouts
    mandatory retreat).  SPECIFICITY: outside a battle the same drag
    still routes to the normal phase handlers."""
    import game_logic  # noqa: F401  (fixture parity with the repro)
    import random as _rnd
    _rnd.seed(4242)     # deck order (and so the planted cc) pinned
    from cards import get_card
    from wotr_bridge.live import gesture_match as gm
    st = adopted_state()
    import ai
    for nat in list(st.political.boxes):
        if ai._side_of_nation(nat) == Side.SHADOW:
            st.political.boxes[nat] = 3
            st.political.active[nat] = True
    menu = ai.get_legal_actions(st, Side.SHADOW)
    atk = next((a for a in menu if a.action_type == 'army_attack'),
               None)
    if atk is None:
        fails.append(('gap_precondition', 'no army_attack'))
        return
    rig = Rig(st, Side.SHADOW, phase='actions', seed=5)
    drv = rig.drv

    def ai_apply(action, label):
        rig.rand.begin_attempt()
        with rig.rand.advance():
            post = ai.apply_action(drv.es, action)
        drv._snapshot([], label)
        drv.es = post
        drv.hands.es = post
        rig.rand.step_commit()

    ai_apply(atk, 'ai:attack')
    drv.last_actor = Side.SHADOW
    pc = drv.es.pending_combat
    tmp_c, _ok = ai._combat_card_check_ctx(drv.es, pc, True)
    sh_hand = drv.es.hands[Side.SHADOW].hand
    cand = list(getattr(drv.es, 'sh_char_deck', []) or []) \
        + list(getattr(drv.es, 'sh_strat_deck', []) or []) + sh_hand
    pick_cid = next(
        (cid for cid in cand
         if (get_card(cid) is not None
             and get_card(cid).side == Side.SHADOW
             and ai._combat_card_playable(drv.es, get_card(cid),
                                          tmp_c, True))), None)
    if pick_cid is None:
        fails.append(('gap_precondition', 'no playable SH cc'))
        return
    if pick_cid not in sh_hand:
        sh_hand.append(pick_cid)
    card = get_card(pick_cid)
    title = '%s/%s' % (card.card_name, card.combat_name)
    ai.commit_hidden_combat_card(drv.es, Side.SHADOW,
                                 str(card.deck).lower())
    for _ in range(4):
        dp = ai.describe_pending(drv.es)
        if dp is None or dp['side'] is Side.FREE_PEOPLES:
            break
        pick = next((a for a in dp['actions']
                     if a.action_type == 'combat_card_none'), None)
        if pick is None:
            break
        ai_apply(pick, 'ai:%s' % dp['kind'])
    if not _arm_scouts(fails, drv, 'gap'):
        return
    rig.feed('$Wey moves The Red Arrow/Scouts from Free Peoples Hand '
             'to Current FP Card.')
    if ai.describe_pending(drv.es) is not None \
            or drv.es.pending_combat.get('phase') != 'combat_card_reveal':
        fails.append(('gap_not_parked',
                      drv.es.pending_combat.get('phase')))
        return
    drag = ('$Wey moves group ([2]Rohan Regular, Rohan Leader) from '
            'Fords Of Isen to Druwaith Iaur.')
    rig.feed(drag)
    if not any(k == 'battle_gap_buffered' for k, _ in drv.findings):
        fails.append(('gap_not_held', drv.findings[-3:]))
    if drv.pending_buf is None or len(drv.pending_buf.records) != 1:
        fails.append(('gap_buffer', drv.pending_buf
                      and len(drv.pending_buf.records)))
    if any('Druwaith' in u for u in drv.uncommitted):
        fails.append(('gap_uncommitted', list(drv.uncommitted)[-2:]))
    rig.feed('<auto> moves %s from Shadow Hand (chosen) to '
             'Current SA Card.' % title)
    if 'pending:combat:defender_retreat' not in [c[2] for c in
                                                 drv.commits]:
        fails.append(('gap_retreat_committed',
                      [c[2] for c in drv.commits][-4:]))
    from board import sanitize as _san
    key = next((k for k in drv.es.armies
                if _san(k) == _san('druwaith iaur')), None)
    a = drv.es.armies.get(key) if key else None
    if a is None or len(a.units) != 2:
        fails.append(('gap_army_moved', a and len(a.units)))
    if any('HUMAN DECISION HELD' in al or 'ORPHANED' in al
           for al in drv.alarms):
        fails.append(('gap_alarms', list(drv.alarms)))
    # SPECIFICITY: no battle open -> the drag is NOT battle-gap-held
    st2 = adopted_state()
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='actions', seed=6)
    rig2.feed('$Wey moves group ([2]Sauron Regular) from Morannon to '
              'Dagorlad.')
    if any(k == 'battle_gap_buffered' for k, _ in rig2.drv.findings):
        fails.append(('gap_specificity', rig2.drv.findings[-3:]))


def test_card_play_political_rider_not_required(fails):
    """MEASURES (tester 2026-08-24 'A Power too Great never stayed on
    the board'; USER's own 2026-08-22 23:24 session: the play was
    refused 'differs on the political track' + 5 undo presses because
    the human did not drag the Elves marker in the same window): a
    card effect's AUTOMATIC political advance is engine bookkeeping -
    the play window matches WITHOUT the marker drag, the engine files
    the table card + advance, and the post-commit sweep emits the
    marker walk itself (pid 256 Game.2000->2001) so the client board
    ends right.  SPECIFICITY: a VOICED political row that contradicts
    the engine still refuses."""
    import ai
    import game_state as gsm
    import game_logic
    from game_state import ActionDieFace
    st = fresh_state()
    faces = [(Side.FREE_PEOPLES, i, f) for i, f in enumerate(
        (ActionDieFace.EVENT, ActionDieFace.CHARACTER,
         ActionDieFace.MUSTER, ActionDieFace.WILL_OF_WEST))]
    faces += [(Side.SHADOW, i, f) for i, f in enumerate(
        (ActionDieFace.ARMY, ActionDieFace.CHARACTER,
         ActionDieFace.MUSTER, ActionDieFace.EYE,
         ActionDieFace.EVENT, ActionDieFace.MUSTER_ARMY,
         ActionDieFace.CHARACTER))]
    game_logic.adopt_rolled_faces(st, faces)
    cid = 'FREE_PEOPLES_Strategy_A_Power_too_Great'
    if cid in st.fp_strat_deck:
        st.fp_strat_deck.remove(cid)
    rig = Rig(st, Side.SHADOW, phase='actions', seed=11)
    drv = rig.drv
    drv.last_actor = Side.SHADOW
    drv.es.hands[Side.FREE_PEOPLES].hand.append(
        drv.hands._mint('strategy'))
    elves0 = drv.es.political.boxes[gsm.Nation.ELVES]
    out = rig.feed(
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '$Wey moves A Power Too Great/No Quarter from Free Peoples '
        'Hand to Current FP Card.',
        '$Wey kills Free Action Die (P) in Current Action.')
    # The printed table effect commits before Gandalf's optional draw.
    # Choosing draw creates the demand; the recorded physical draw supplies it.
    if not any(i[0] == 'chat' and i[1] == 'prompt'
               and 'Gandalf' in str(i[2]) and 'optional' in str(i[2])
               and 'Strategy' in str(i[2]) and 'decline' in str(i[2])
               for i in out):
        fails.append(('aptg_draw_not_asked', [i for i in out
                                              if i[0] == 'chat']))
    if (getattr(drv.es, 'pending_card_option', None) or {}).get('card_id') \
            != '__gandalf_draw__':
        fails.append(('aptg_optional_pending_missing', drv.es.pending_card_option))
    out += rig.feed('<Wey> draw')
    out += rig.feed('$Wey draws Free Strategy Card.')
    if any(i[0] == 'chat' and i[1] == 'refusal' for i in out):
        fails.append(('aptg_refused', [i for i in out
                                       if i[0] == 'chat']))
    if drv.window_stats['matched'] != 1:
        fails.append(('aptg_matched', dict(drv.window_stats)))
    pc = getattr(drv.es, 'persistent_cards', None)
    if pc is None or not any(
            getattr(c, 'card_id', None) == cid
            for c in (getattr(pc, 'fp_cards', []) or [])):
        fails.append(('aptg_on_table', 'missing'))
    if drv.es.political.boxes[gsm.Nation.ELVES] != elves0 + 1:
        fails.append(('aptg_elves_advanced',
                      drv.es.political.boxes[gsm.Nation.ELVES]))
    if not any(i[0] == 'line' and '256' in str(i[1])
               and 'Game.2001' in str(i[1]) for i in out):
        fails.append(('aptg_marker_swept',
                      [i[1] for i in out if i[0] == 'line']))
    if not any(k == 'political_sweep_emitted'
               for k, _ in drv.findings):
        fails.append(('aptg_sweep_finding', drv.findings[-3:]))
    # SPECIFICITY: a voiced CONTRADICTING political row still refuses
    st2 = fresh_state()
    game_logic.adopt_rolled_faces(st2, faces)
    if cid in st2.fp_strat_deck:
        st2.fp_strat_deck.remove(cid)
    rig2 = Rig(st2, Side.SHADOW, phase='actions', seed=12)
    rig2.drv.last_actor = Side.SHADOW
    rig2.drv.es.hands[Side.FREE_PEOPLES].hand.append(
        rig2.drv.hands._mint('strategy'))
    out2 = rig2.feed(
        '$Wey moves Free Action Die (P) from FP Dice to Current '
        'Action.',
        '$Wey moves A Power Too Great/No Quarter from Free Peoples '
        'Hand to Current FP Card.',
        # ROHAN marker dragged (the card advances ELVES): contradicts
        '$Wey moves Rohan from Political Track 3 to Political '
        'Track 2.',
        '$Wey kills Free Action Die (P) in Current Action.')
    out2 += rig2.feed('<Wey> draw')
    out2 += rig2.feed('$Wey draws Free Strategy Card.')
    # item 14 / M10 (BRLANE 2026-08-29): the card leaving the hand IS
    # the play - it commits by CARD IDENTITY and the contradicting
    # Rohan drag is EXPLAINED in chat (finding card_play_extra_rows),
    # never a whole-play refusal with undo presses (the old pin)
    if rig2.drv.window_stats['matched'] != 1 \
            or any(i[0] == 'line' and 'Controls.819' in str(i[1])
                   for i in out2):
        fails.append(('aptg_identity_commit',
                      (dict(rig2.drv.window_stats),
                       [i for i in out2 if i[0] == 'chat'][:1])))
    if not any(k == 'card_play_extra_rows' for k, _ in rig2.drv.findings) \
            or not any(i[0] == 'chat' and 'political' in str(i[2]).lower()
                       for i in out2):
        fails.append(('aptg_extra_rows_explained',
                      (rig2.drv.findings[-4:],
                       [i for i in out2 if i[0] == 'chat'])))


def test_engine_truth_reconcilers(fails):
    """MEASURES (USER trust ruling 2026-08-24): (1) _political_recon
    walks the CLIENT's political marker to the ENGINE box - forward
    and backward - with an in-flight guard against re-emission; (2)
    _guide_box_recon (AI playing FP) drags the engine's guide into
    the client Guide box and the displaced piece back to the
    Fellowship box, once."""
    import game_state as gsm
    st = fresh_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # AI = FP
    drv = rig.drv
    # (1) political: engine says Rohan box 2; client marker shows
    # box 0 (client zone 'Political 3')
    drv.es.political.boxes[gsm.Nation.ROHAN] = 2
    drv.view.piece_region = {'259': 'Political 3'}
    out = drv._political_recon()
    lines = [i[1] for i in out if i[0] == 'line']
    if len(lines) != 1 or '259' not in lines[0] \
            or 'Game.2002' not in lines[0]:
        fails.append(('pol_recon_forward', lines))
    if drv._political_recon():
        fails.append(('pol_recon_inflight', 'ni: re-emitted'))
    # client catches up -> guard clears; a BACKWARD divergence (client
    # ahead of engine) walks back
    drv.view.piece_region = {'259': 'Political 1'}      # client box 2
    if drv._political_recon():
        fails.append(('pol_recon_settled', 'emitted at parity'))
    drv.es.political.boxes[gsm.Nation.ROHAN] = 1
    out = drv._political_recon()
    lines = [i[1] for i in out if i[0] == 'line']
    if len(lines) != 1 or 'Game.2001' not in lines[0]:
        fails.append(('pol_recon_backward', lines))
    # (2) guide box: engine guide = Gandalf the Grey (fresh state);
    # the client box holds Strider (pid 241? use names_for_pid to
    # find pids) - the recon must move Gandalf in, Strider out
    import log_replay as lr
    gpid = next(p for p in lr.COMPANIONS
                if 'Gandalf the Grey' in set(
                    lr.names_for_pid(str(p)) or ()))
    spid = next(p for p in lr.COMPANIONS
                if 'Strider' in set(lr.names_for_pid(str(p)) or ()))
    drv.view.piece_region = {str(spid): 'Guide Of The Fellowship',
                             str(gpid): 'Fellowship Of The Ring'}
    out = drv._guide_box_recon()
    lines = [i[1] for i in out if i[0] == 'line']
    if len(lines) != 2 or not any(str(gpid) in l and 'Game.1999' in l
                                  for l in lines):
        fails.append(('guide_recon_lines', lines))
    if drv._guide_box_recon():
        fails.append(('guide_recon_inflight', 're-emitted'))
    # box correct -> clears, silent
    drv.view.piece_region = {str(gpid): 'Guide Of The Fellowship'}
    if drv._guide_box_recon():
        fails.append(('guide_recon_settled', 'emitted at parity'))
    # SPECIFICITY: human-FP game (AI = SHADOW) never touches the box
    rig2 = Rig(fresh_state(), Side.SHADOW, phase='actions')
    rig2.drv.view.piece_region = {str(spid): 'Guide Of The Fellowship'}
    if rig2.drv._guide_box_recon():
        fails.append(('guide_recon_human_fp', 'AI touched the '
                                              "human's guide box"))


def test_card_play_region_chat_guard(fails):
    """MEASURES (tester 2026-08-24, The Fighting Uruk-hai): a region
    chat while an un-terminated CARD-PLAY window is open (human's die
    staged + card dragged from hand) is HELD with a note - never
    routed to the attack lane ('no legal attack ... with the staged P
    die').  Killing the die then completes the play and the assault
    chain opens for the human.  SPECIFICITY: a plain die-staged
    window with NO card drag keeps the chat-attack lane."""
    import ai
    import game_logic
    import game_state as gsm
    from game_state import ActionDieFace, Nation, UnitKind
    st = fresh_state()
    faces = [(Side.SHADOW, i, f) for i, f in enumerate(
        (ActionDieFace.EVENT, ActionDieFace.ARMY,
         ActionDieFace.MUSTER, ActionDieFace.CHARACTER,
         ActionDieFace.EYE, ActionDieFace.MUSTER_ARMY,
         ActionDieFace.ARMY))]
    faces += [(Side.FREE_PEOPLES, i, f) for i, f in enumerate(
        (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST,
         ActionDieFace.MUSTER, ActionDieFace.EVENT))]
    game_logic.adopt_rolled_faces(st, faces)
    # Isengard besieging Helm's Deep; Saruman in play; Isengard at war
    a = game_logic.get_or_create_army(st, "helm's deep")
    a.units = [gsm.Unit(Nation.ROHAN, UnitKind.REGULAR),
               gsm.Unit(Nation.ISENGARD, UnitKind.REGULAR),
               gsm.Unit(Nation.ISENGARD, UnitKind.REGULAR)]
    st.besieged_strongholds.add('helm s deep')
    st.saruman.in_play = True
    st.political.boxes[Nation.ISENGARD] = 3
    st.political.active[Nation.ISENGARD] = True
    cid = 'SHADOW_Strategy_The_Fighting_Uruk-hai'
    for dk in ('sh_strat_deck', 'sh_char_deck'):
        dl = getattr(st, dk, [])
        if cid in dl:
            dl.remove(cid)
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES
    drv.es.hands[Side.SHADOW].hand.append(drv.hands._mint('strategy'))
    rig.feed('$Wey moves Shadow Action Die (P) from SA Dice to '
             'Current Action.',
             '$Wey moves The Fighting Uruk-Hai/Onslaught from Shadow '
             'Hand to Current SA Card.')
    out = rig.feed("<Wey> helm's deep")
    # BRLANE-4 (2026-08-29): the guard answers in CHAT too (never the
    # attack refusal) - the old pin required silence in chat
    if any(i[0] == 'chat' and 'not legal' in str(i[2]) for i in out):
        fails.append(('guard_refused', [i for i in out
                                        if i[0] == 'chat']))
    if not any(i[0] == 'chat' and 'card-play window' in str(i[2])
               for i in out):
        fails.append(('guard_chat_silent', out))
    if not any(i[0] == 'note' and 'card-play window' in str(i[1])
               for i in out):
        fails.append(('guard_note', out))
    if not any(k == 'card_play_region_chat_held'
               for k, _ in drv.findings):
        fails.append(('guard_finding', drv.findings[-3:]))
    out = rig.feed('$Wey kills Shadow Action Die (P) in Current '
                   'Action.')
    pc = getattr(drv.es, 'pending_combat', None)
    if not isinstance(pc, dict) \
            or 'helm' not in str(pc.get('defender_region', '')):
        fails.append(('guard_play_committed', pc))
    if drv.window_stats['matched'] != 1:
        fails.append(('guard_matched', dict(drv.window_stats)))
    # SPECIFICITY: die staged, NO card drag -> attack lane untouched
    st2 = fresh_state()
    game_logic.adopt_rolled_faces(st2, faces)
    rig2 = Rig(st2, Side.FREE_PEOPLES, phase='actions')
    rig2.drv.last_actor = Side.FREE_PEOPLES
    rig2.feed('$Wey moves Shadow Action Die (A) from SA Dice to '
              'Current Action.')
    out2 = rig2.feed('<Wey> osgiliath')
    if any(i[0] == 'note' and 'card-play window' in str(i[1])
           for i in out2):
        fails.append(('guard_specificity', out2))


def test_nazgul_chain_chat_vocabulary(fails):
    """MEASURES (USER 2026-08-24: "'done' / 'rolling for hunt' /
    'hunt' / 'discarding card', fuzzy string matched"): with a
    human-side __nazgul_moves__ chain open, chat 'done' commits the
    terminator; the AI ANNOUNCES the chain when it opens; 'help'
    answers with the waiting-on line + the command list; the Strike
    either/or vocabulary maps via match_choice (unit-level)."""
    import ai
    from wotr_bridge.live import chat as cm
    import game_logic
    import game_state as gsm
    from game_state import ActionDieFace, Nation, UnitKind
    st = fresh_state()
    st.fellowship.progress = 1
    faces = [(Side.SHADOW, i, f) for i, f in enumerate(
        (ActionDieFace.EVENT, ActionDieFace.ARMY,
         ActionDieFace.MUSTER, ActionDieFace.CHARACTER,
         ActionDieFace.EYE, ActionDieFace.MUSTER_ARMY,
         ActionDieFace.ARMY))]
    faces += [(Side.FREE_PEOPLES, i, f) for i, f in enumerate(
        (ActionDieFace.CHARACTER, ActionDieFace.WILL_OF_WEST,
         ActionDieFace.MUSTER, ActionDieFace.EVENT))]
    game_logic.adopt_rolled_faces(st, faces)
    cid = 'SHADOW_Character_Nazgûl_Search'
    _found = None
    for dk in ('sh_char_deck', 'sh_strat_deck'):
        for c in list(getattr(st, dk, [])):
            if 'Nazg' in str(c) and 'Search' in str(c):
                _found = c
                getattr(st, dk).remove(c)
        if _found:
            break
    cid = _found or cid
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    drv.last_actor = Side.FREE_PEOPLES
    drv.es.hands[Side.SHADOW].hand.append(cid)
    import event_effects as ee
    res = ee.play_event_card(drv.es, Side.SHADOW, cid)
    if not res.get('success'):
        fails.append(('naz_precondition', res))
        return
    dp = ai.describe_pending(drv.es, Side.SHADOW)
    if dp is None or str(dp['kind']) != 'card_option':
        fails.append(('naz_chain_open', repr(dp and dp['kind'])))
        return
    # the announce (one-shot)
    ann = drv._maybe_announce_pending()
    if not any(i[0] == 'chat' and 'say done' in str(i[2])
               for i in ann):
        fails.append(('naz_announce', ann))
    if drv._maybe_announce_pending():
        fails.append(('naz_announce_once', 'repeated'))
    # 'help' answers with waiting-on + the command list
    out = rig.feed('<Wey> help')
    chats = [i for i in out if i[0] == 'chat']
    if not any('waiting on your' in str(c[2]) for c in chats) \
            or sum(1 for c in chats
                   if str(c[2]).startswith('help ')) != 5:
        # 5 lines since BL-3/BL-4/BL-5 (2026-08-25): recover + the
        # board-eye advert + the event-card route joined the list
        fails.append(('naz_help', chats))
    if not any('Recover Action Dice' in str(c[2]) for c in chats):
        fails.append(('naz_help_recover_line', chats))
    if any(" 'Ne'" in str(c[2]) or '= 0e or Ne' in str(c[2])
           for c in chats):
        fails.append(('naz_help_ne_advert_back', chats))
    # 'done' commits the terminator and closes the chain
    out = rig.feed('<Wey> done')
    dp2 = ai.describe_pending(drv.es, Side.SHADOW)
    if dp2 is not None and str(dp2['kind']) == 'card_option':
        fails.append(('naz_done_commits', repr(dp2['kind'])))
    # Strike either/or vocabulary at the match_choice level
    class _A(object):
        def __init__(self, ch):
            self.action_type = 'card_option'
            self.params = {'choices': ch, 'card_id': 'x'}
            self.die_idx = -1
    menu = [_A({'action': 'hunt'}),
            _A({'action': 'discard_table_card',
                'target_card': 'FREE_PEOPLES_Character_Elven_Rope'}),
            _A({'action': 'discard_table_card',
                'target_card': 'FREE_PEOPLES_Character_Axe_and_Bow'}),
            _A({'action': 'none'})]
    if cm.match_choice('rolling for hunt', menu) is not menu[0] \
            or cm.match_choice('hunt', menu) is not menu[0]:
        fails.append(('naz_hunt_word', 'miss'))
    if cm.match_choice('discarding card elven rope', menu) \
            is not menu[1]:
        fails.append(('naz_discard_named', 'miss'))
    if cm.match_choice('discarding card', menu) is not None:
        fails.append(('naz_discard_ambiguous', 'guessed'))
    if cm.match_choice('none', menu) is not menu[3]:
        fails.append(('naz_none_word', 'miss'))
    # 'done' must NOT hit this menu (no terminator present)
    if cm.match_choice('done', menu) is not None:
        fails.append(('naz_done_scoped', 'matched without terminator'))


def main():
    tests = [test_window_commit, test_alternation_wakes_ai, test_undo_per_line,
             test_undo_mismatch_alarm, test_owed_draw_gate,
             test_fp_ok_and_roll,
             test_eye_allocation,
             test_window_rng_deferral, test_pending_phase_buffer,
             test_d6_side_binding, test_human_plain_pass,
             test_forced_rows_and_echo, test_capture_marker_sweep,
             test_defender_retreat_advance,
             test_human_chat_attack_declaration,
             test_restore_watch_wired,
             test_refuse_ineligible_arms_watch,
             test_illegal_window_not_rng_deferred,
             test_siege_withdraw_window,
             test_forced_downgrade_arrival_echo, test_cc_reveal_echo,
             test_fp_acts_first,
             test_route_error_isolated_per_line,
             test_undo_restores_driver_turn_state,
             test_human_fp_declaration,
             test_live_pending_never_forced,
             test_dragged_combat_card_commits_via_buffer,
             test_hidden_combat_card_commit_and_reveal,
             test_single_die_from_hunt_box_not_recover,
             test_window_before_action_phase_refused,
             test_unowed_draw_refused_by_client_undo,
             test_demanded_draw_routes_to_funnel_not_gesture,
             test_refusal_draw_reissue_gate,
             test_fs_window_with_cards_in_hand_defers_on_hunt_not_draw,
             test_siege_movein_banks_human_move_demand,
             test_named_minion_kill_claims_echo,
             test_decision_born_move_demand_not_rebased,
             test_reconcile_stale_dump_and_erase_witness,
             test_card_play_window_two_ticks,
             test_track_ops_on_shim,
             test_ai_companion_death_rendered_from_client_zone,
             test_blind_hand_discard_identity_from_dump,
             test_human_declaration_planner_form,
             test_livelock_breakers,
             test_human_reveal_placement_marker_flip,
             test_ambiguous_hold_is_voiced,
             test_fear_fire_foes_group_split_is_one_gesture,
             test_bl149_companion_death_removal_claims_echo,
             test_self_repair_offer_is_explicit_logged_and_declinable,
             test_self_repair_yes_kills_waits_resyncs_and_sets_turn,
             test_side_of_last_die_kill_rule,
             test_bl150_unvoiced_discard_leg_is_loud,
             test_paid_noop_sees_the_aragorn_crowning,
             test_bl152_in_window_leader_recruit_is_not_owed,
             test_human_eyes_judged_by_engine_menu,
             test_zero_eyes_chat_refused_under_minimum,
             test_ai_deck_draw_starve_ordered,
             test_special_hunt_tile_draw_feeds_tile_funnel,
             test_used_die_out_of_hunt_box_refused,
             test_nonhand_chosen_reveal_refused,
             test_lenient_mode_accepts_refused_window,
             test_lenient_mode_zero_eyes_and_reset,
             test_lenient_hand_facts,
             test_scouts_retreat_drag_in_reveal_gap,
             test_card_play_political_rider_not_required,
             test_engine_truth_reconcilers,
             test_card_play_region_chat_guard,
             test_nazgul_chain_chat_vocabulary,
             test_early_eye_keeps_ai_declare_open,
             test_post_terminator_draw_feeds_deferred_window,
             test_two_draws_in_window_match,
             test_two_army_moves_one_die,
             test_two_army_moves_vacated_source,
             test_card_leg_gate_wiring,
             test_progress_sweep_destination_aware,
             test_two_recruits_one_muster_die,
             test_fellowship_death_banks_named_demand,
             test_echo_move_drag_claims_ledger,
             test_move_row_fate,
             test_cc_reveal_claim_once]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', default=None)
    args = ap.parse_args()
    if args.k:
        tests = [t for t in tests if args.k in t.__name__]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
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
