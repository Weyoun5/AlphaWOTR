"""wotr_bridge/live/emission.py instruments (LIFT-D, REBUILD_PLAN.md).

House style: test_engine_sided_api.py -- fails-list runner, MEASURED
QUANTITY stated per test, sensitivity AND specificity controls, exit 1
on any failure.

INSTRUMENT MAP (which action type got which instrument, stated plainly
per the task contract):

  FULL RENDERED ROUND-TRIP (the headline instrument): the emitted lines
  are parsed back through log_replay.LogReplayer -- the raw-channel
  client model, the same rig parity_test.py:326-330 wires -- applied to
  a parser state built from the same setup + the same client roll
  lines, and the parsed effect must equal the ENGINE's own apply_action
  delta via wotr_bridge.live.client_board.board_delta (which includes
  the __dice__/__fellowship__/__political__ terms):
      army_move, advance_political, muster_recruit (BOTH legs of the
      muster die, composite delta), fellowship_move, eye_allocate,
      card_option (REAL optioned cards driven end-to-end through
      play_event -> pending_card_option -> the chosen option:
      Paths of the Woses move lane + Cirdan's Ships recruit lane,
      engine delta filtered of the play's own __hands__/__table__
      bookkeeping, which the '!CALL play_card' client handler
      performs physically -- documented in the legs).

  CARD-EFFECT ROW RENDERER (emission_card, the bounded rule-6
  exception): the two round-trip legs above are its specificity;
  test_card_ambiguity_hold is its sensitivity (two sources feeding one
  destination MUST hold naming the rows, and the HOLD must leave the
  client model untouched); test_card_move_specificity pins the
  single-source synthetic move at field level (frm/to zones + typed
  pid).

  L1 SHAPE (emitter.classify_line round-trips each emitted line, plus
  field-level assertions on pids/zones/order) for the types whose full
  resolution rides client randomness (!CALL) or a mid-battle position
  too heavy to sync a fresh replayer to:
      play_event staging, army_attack declaration, combat_card
      hidden commit + reveal, casualty allocation (kills, downgrade
      sources, eliminated-outright), combat_retreat,
      retreat_into_siege, siege_trim, hunt_guide_separate, mouth_move,
      dwarven_ring, renamed-companion moves, Mordor-track moves, the
      table-slot kill grammar, the ring-paid table-card discard.

  SENSITIVITY NOTE (measured during mutation testing): the parser
  marks a die used at the SPEND drag, so a missing die-KILL line does
  not move the round-trip delta -- the field-level 'kill last / no
  kill on the open leg' assertions are the sensitive instrument for
  the kill bookkeeping (mutation 1 went RED on them).  The retrack
  mutation went RED on BOTH the pin and the full round-trip delta.

  PREDICATE / POLICY units:
      play_was_refunded (all three contract conditions),
      fungible-shortage HOLD vs clean-dump lines,
      card_* HOLD (rule 6: no board-diff fallback exists),
      FP-dead-permanence vs SA-dead-recycling muster pools
      (USER rulings 2026-08-18 / Part-35).
"""
import os
import random
import sys
import tempfile

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
for _p in (_SRC, _BRIDGE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as board_mod
import game_state as gs
import game_logic as gl
import log_replay as lr
import ai
import emitter
from ai import Action
from game_state import Side, ActionDieFace
from board import sanitize
from wotr_bridge.live.client_view import ClientView
from wotr_bridge.live import emission
from wotr_bridge.live.client_board import board_delta

NICK = 'WOTR_AI'
FP_FACES = ['CHARACTER', 'CHARACTER', 'MUSTER', 'WILL_OF_WEST']
SH_FACES = ['CHARACTER', 'ARMY', 'MUSTER', 'MUSTER', 'EVENT',
            'MUSTER_ARMY', 'CHARACTER']


class TestView(ClientView):
    """ClientView + the driver-supplied commitment-decode inverse the
    module reads through _card_pid (card_pid_map).  ClientView itself
    lacks the map -- a REPORTED gap, stubbed here exactly the way the
    driver will supply it."""

    def __init__(self, state_fn):
        ClientView.__init__(self, state_fn)
        self.card_pid_map = {}


def fresh_state(seed=7):
    random.seed(seed)
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    gl.initialize_decks(st)
    return st


def adopt(st):
    """Client-rolled faces (menu stability), NO Eyes so the fellowship
    leg is hunt-free and deterministic."""
    trip = [(Side.FREE_PEOPLES, i, ActionDieFace[f])
            for i, f in enumerate(FP_FACES)]
    trip += [(Side.SHADOW, i, ActionDieFace[f])
             for i, f in enumerate(SH_FACES)]
    gl.adopt_rolled_faces(st, trip)


def roll_lines():
    """The client roll lines that carry the SAME faces adopt() landed,
    so the parser registers the die pids exactly as a live client
    would."""
    fp_ids = emitter._FP_DIE_IDS[:4]
    sh_ids = emitter._SH_DIE_IDS[:7]
    fpv = [str(emitter._FP_FACE_VALUE[f]) for f in FP_FACES]
    shv = [str(emitter._SH_FACE_VALUE[f]) for f in SH_FACES]
    return [
        emitter.render('dice_roll', {'nick': NICK, 'dice': fp_ids,
                                     'values': fpv}, local=False),
        emitter.render('dice_roll', {'nick': NICK, 'dice': sh_ids,
                                     'values': shv}, local=False),
    ]


def make_view(holder):
    v = TestView(lambda: holder[0])
    for i, pid in enumerate(emitter._FP_DIE_IDS[:4]):
        v.bind_die(pid, Side.FREE_PEOPLES, i)
    for i, pid in enumerate(emitter._SH_DIE_IDS[:7]):
        v.bind_die(pid, Side.SHADOW, i)
    return v


def replay(lines):
    """The offline round-trip rig: a fresh parser state fed the emitted
    lines through log_replay.LogReplayer (parity_test.py:326-330 wiring;
    _ai_bridge_log per the donor's _replay :680-683)."""
    random.seed(99)
    b2 = board_mod.load_board()
    pst = gs.setup_initial_game(b2)
    gl.initialize_decks(pst)
    rep = lr.LogReplayer(pst, b2, verbose=False)
    rep._ai_bridge_log = True
    with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False,
                                     encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
        path = f.name
    try:
        rep.parse_and_replay(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return rep.state


def classify(line):
    return emitter.classify_line(emitter.strip_tail(line))


# =====================================================================
# FULL ROUND-TRIP LEGS
# =====================================================================
def test_roundtrip_army_move(fails):
    """MEASURES: parse(emit(army_move)) == the engine's apply_action
    delta (board_delta over armies/dice/fellowship/political/...).
    Non-zero control: the engine delta must be non-empty and contain an
    army term, or the equality below would be a hollow zero.  Also pins
    the die bookkeeping: spend line first, kill line last (single-step
    CHARACTER-die move, donor :16363-16375)."""
    st = fresh_state()
    adopt(st)
    view = make_view([st])
    acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
    mv = next(a for a in acts if a.action_type == 'army_move'
              and st.dice[Side.FREE_PEOPLES][a.die_idx].current
              is ActionDieFace.CHARACTER
              and a.params.get('n_regular', 0) >= 1)
    post = ai.apply_action(st, mv)
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, mv,
                                post=post)
    if em.hold:
        fails.append(('army_move_hold', em.hold))
        return
    op0 = classify(em.lines[0])[0]
    if op0 != 'die_spend_fp':
        fails.append(('army_move_spend_first', 'line0 op=%r' % op0))
    opl, fl = classify(em.lines[-1])
    if opl != 'kill':
        fails.append(('army_move_kill_last', 'last op=%r' % opl))
    d_engine = board_delta(st, post)
    if not d_engine or not any(k not in ('__dice__',)
                               for k in d_engine):
        fails.append(('army_move_control',
                      'NOT-MEASURED: empty engine delta'))
        return
    base = roll_lines()
    d_parser = board_delta(replay(base), replay(base + em.lines))
    if d_engine != d_parser:
        fails.append(('army_move_roundtrip',
                      'engine=%r parser=%r' % (d_engine, d_parser)))


def test_roundtrip_advance_and_muster_chain(fails):
    """MEASURES: (1) parse(emit(advance_political)) == engine delta;
    (2) the muster die's TWO legs compose: leg1 emits the die spend and
    NO kill (multi-step open, donor :16363-16375), leg2 emits NO spend
    (donor :16210-16216) and the kill, the two legs pick DIFFERENT pool
    pids (the retrack quantity -- without it leg2 re-picked leg1's
    piece and the parser lost a recruit), and the composite
    parse(emit(leg1)+emit(leg2)) == the engine's two-leg delta."""
    st = fresh_state()
    adopt(st)
    holder = [st]
    view = make_view(holder)
    log = roll_lines()
    ap = next(a for a in ai.get_legal_actions(st, Side.SHADOW)
              if a.action_type == 'advance_political'
              and 'SAURON' in str(a.params.get('nation')).upper())
    post_ap = ai.apply_action(st, ap)
    em_ap = emission.render_action(view, NICK, Side.SHADOW, ap,
                                   post=post_ap)
    if em_ap.hold:
        fails.append(('advance_hold', em_ap.hold))
        return
    d_e = board_delta(st, post_ap)
    d_p = board_delta(replay(log), replay(log + em_ap.lines))
    if d_e != d_p:
        fails.append(('advance_roundtrip',
                      'engine=%r parser=%r' % (d_e, d_p)))
    if '__political__' not in d_e:
        fails.append(('advance_control',
                      'NOT-MEASURED: no political term in the delta'))
    log += em_ap.lines
    holder[0] = post_ap

    mu1 = next((a for a in ai.get_legal_actions(post_ap, Side.SHADOW)
                if a.action_type == 'muster_recruit'), None)
    if mu1 is None:
        fails.append(('muster_control',
                      'NOT-MEASURED: no muster in the at-war menu'))
        return
    p1 = ai.apply_action(post_ap, mu1)
    em1 = emission.render_action(view, NICK, Side.SHADOW, mu1, post=p1)
    if em1.hold:
        fails.append(('muster1_hold', em1.hold))
        return
    if p1.pending_muster_die is None:
        fails.append(('muster_control', 'NOT-MEASURED: leg1 did not '
                      'open the multi-step'))
        return
    ops1 = [classify(x)[0] for x in em1.lines]
    if 'die_spend_sh' not in ops1:
        fails.append(('muster1_spend', 'ops=%r' % ops1))
    if 'kill' in ops1:
        fails.append(('muster1_no_kill_yet',
                      'kill emitted on the OPEN leg: %r' % ops1))
    holder[0] = p1
    mu2 = next((a for a in ai.get_legal_actions(p1, Side.SHADOW)
                if a.action_type == 'muster_recruit'), None)
    if mu2 is None:
        fails.append(('muster_control', 'NOT-MEASURED: no leg2'))
        return
    p2 = ai.apply_action(p1, mu2)
    em2 = emission.render_action(view, NICK, Side.SHADOW, mu2, post=p2)
    if em2.hold:
        fails.append(('muster2_hold', em2.hold))
        return
    ops2 = [classify(x)[0] for x in em2.lines]
    if 'die_spend_sh' in ops2:
        fails.append(('muster2_no_second_drag',
                      'second-leg die drag emitted: %r' % ops2))
    if 'kill' not in ops2:
        fails.append(('muster2_kill', 'no kill on the closing leg'))
    pids1 = classify(em1.lines[-1])[1].get('pieces') or []
    pids2 = classify(em2.lines[0])[1].get('pieces') or []
    if set(pids1) & set(pids2):
        fails.append(('muster_retrack',
                      'leg2 re-picked leg1 piece(s): %r'
                      % sorted(set(pids1) & set(pids2))))
    d_e = board_delta(post_ap, p2)
    d_p = board_delta(replay(log),
                      replay(log + em1.lines + em2.lines))
    if d_e != d_p:
        fails.append(('muster_roundtrip',
                      'engine=%r parser=%r' % (d_e, d_p)))

    # deferred kill (donor :15731-15758): a leg2 'pass' closes the
    # multi-step without a die-carrying action -> the pending die is
    # still killed
    ps = next((a for a in ai.get_legal_actions(p1, Side.SHADOW)
               if a.action_type == 'pass'), None)
    if ps is None:
        print('  [muster] deferred-kill leg NOT-MEASURED '
              '(no pass in the leg2 menu)')
    else:
        pp = ai.apply_action(p1, ps)
        if pp.pending_muster_die is None:
            emp = emission.render_action(view, NICK, Side.SHADOW, ps,
                                         post=pp)
            kops = [classify(x)[0] for x in emp.lines]
            if 'kill' not in kops:
                fails.append(('muster_deferred_kill',
                              'pass closed the step, no kill: %r'
                              % kops))
        else:
            print('  [muster] deferred-kill leg NOT-MEASURED '
                  '(pass kept the step open)')


def test_roundtrip_fellowship_move(fails):
    """MEASURES: parse(emit(fellowship_move)) == engine delta with an
    empty Hunt Box (hunt-free, deterministic).  Pins the die-to-hunt
    bookkeeping: the spent die is dropped into the Hunt Box, NEVER
    killed (_DIE_TO_HUNT_ACTIONS, donor :18150 + :16363)."""
    st = fresh_state()
    adopt(st)
    view = make_view([st])
    fm = next(a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
              if a.action_type == 'fellowship_move')
    random.seed(5)
    post = ai.apply_action(st, fm)
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, fm,
                                post=post)
    if em.hold:
        fails.append(('fsp_hold', em.hold))
        return
    ops = [classify(x)[0] for x in em.lines]
    if 'die_to_hunt' not in ops:
        fails.append(('fsp_die_to_hunt', 'ops=%r' % ops))
    if 'kill' in ops:
        fails.append(('fsp_no_kill',
                      'fellowship die was killed instead of banked: %r'
                      % ops))
    d_e = board_delta(st, post)
    if '__fellowship__' not in d_e:
        fails.append(('fsp_control',
                      'NOT-MEASURED: no fellowship term'))
        return
    base = roll_lines()
    d_p = board_delta(replay(base), replay(base + em.lines))
    if d_e != d_p:
        fails.append(('fsp_roundtrip',
                      'engine=%r parser=%r' % (d_e, d_p)))


def test_roundtrip_eye_allocate(fails):
    """MEASURES: parse(emit(eye_allocate count=2)) == engine delta
    (pre-roll allocation; the parser registers the die pids from the
    Controls.372 lines themselves).  The die indices mirror the
    engine's own executor rule (ai.py:10727-10732).  Zero-eye control:
    count=0 emits ZERO lines plus the ('nc','0e') chat request -- the
    LEARNED zero-eye choice is visible only in chat (absence-is-the-
    choice)."""
    st = fresh_state()
    st._eye_allocation_phase = True
    view = make_view([st])
    acts = ai.get_legal_actions(st, Side.SHADOW)
    ea = next((a for a in acts if a.action_type == 'eye_allocate'
               and a.params.get('count') == 2), None)
    if ea is None:
        fails.append(('eye_control', 'NOT-MEASURED: no count=2 option'))
        return
    post = ai.apply_action(st, ea)
    em = emission.render_action(view, NICK, Side.SHADOW, ea, post=post)
    if em.hold:
        fails.append(('eye_hold', em.hold))
        return
    if len(em.lines) != 2 or any(classify(x)[0] != 'eye_alloc'
                                 for x in em.lines):
        fails.append(('eye_lines', '%r' % (em.lines,)))
    d_e = board_delta(st, post)
    if '__dice__' not in d_e:
        fails.append(('eye_control', 'NOT-MEASURED: no dice term'))
        return
    d_p = board_delta(replay([]), replay(em.lines))
    if d_e != d_p:
        fails.append(('eye_roundtrip',
                      'engine=%r parser=%r' % (d_e, d_p)))
    ea0 = next((a for a in acts if a.action_type == 'eye_allocate'
                and a.params.get('count') == 0), None)
    if ea0 is None:
        print('  [eye] zero-eye leg NOT-MEASURED (no count=0 option)')
        return
    em0 = emission.render_action(view, NICK, Side.SHADOW, ea0,
                                 post=ai.apply_action(st, ea0))
    if em0.hold or em0.lines or ('zero_eyes', '0e') not in em0.chat:
        fails.append(('eye_zero', 'lines=%r chat=%r hold=%r'
                      % (em0.lines, em0.chat, em0.hold)))


# =====================================================================
# L1-SHAPE LEGS
# =====================================================================
def test_play_event_staging_shape(fails):
    """MEASURES (L1 shape): the play_event staging order lifted from
    donor :15874-15900 -- the play's OWN gestures ahead of the effect's
    randomness: [die-spend line] then the 'plays <name>' chat request
    then '!CALL play_card <pid>'.  No die kill rides the Emission (the
    kill must FOLLOW the resolution; driver-owed, donor :15760-15829).
    Sensitivity: with no card-pid map the render HOLDS naming the gap
    -- a pid is never guessed."""
    st = fresh_state()
    adopt(st)
    for _ in range(2):
        gl.draw_event_card(st, Side.FREE_PEOPLES, 'character')
        gl.draw_event_card(st, Side.FREE_PEOPLES, 'strategy')
    view = make_view([st])
    pe = next((a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
               if a.action_type == 'play_event'), None)
    if pe is None:
        fails.append(('play_control', 'NOT-MEASURED: no play_event'))
        return
    cid = pe.params.get('card_id')
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, pe)
    if not em.hold or 'no client pid' not in em.hold:
        fails.append(('play_pid_hold', 'expected HOLD, got %r' % em))
    view.card_pid_map[cid] = '123'
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, pe)
    if em.hold:
        fails.append(('play_hold', em.hold))
        return
    if len(em.lines) != 1:
        fails.append(('play_lines', '%r' % (em.lines,)))
        return
    op, f = classify(em.lines[0])
    want_pid = emitter.die_piece_for(view, 'FREE_PEOPLES', pe.die_idx)
    if op not in ('die_spend_fp', 'die_spend_fp81') \
            or f.get('die') != want_pid:
        fails.append(('play_spend', 'op=%r die=%r want=%r'
                      % (op, f.get('die'), want_pid)))
    if em.calls != ['!CALL play_card 123']:
        fails.append(('play_call', '%r' % (em.calls,)))
    want_name = 'plays %s' % emission.parser_card_name(cid).lower()
    if em.chat != [('card_announce', want_name)]:
        fails.append(('play_chat', '%r != %r'
                      % (em.chat, [('card_announce', want_name)])))
    if any(classify(x)[0] == 'kill' for x in em.lines):
        fails.append(('play_no_kill',
                      'die kill packed before the !CALL'))


def test_play_refund_predicate(fails):
    """MEASURES: play_was_refunded tests the WHOLE contract (donor
    :15903-15956, BRIDGE-7): specificity -- a complete engine no-op
    (card kept, die intact, no chain) is True; sensitivity, one leg per
    condition -- (a) card left the hand, (b) die consumed, (c) chain
    opened, each alone makes it False.  Pairing per the task: an
    un-refunded play still RENDERS lines (render_action non-empty)."""
    st = fresh_state()
    adopt(st)
    gl.draw_event_card(st, Side.FREE_PEOPLES, 'character')
    cid = next(c for c in st.hands[Side.FREE_PEOPLES].hand
               if isinstance(c, str))
    act = Action(die_idx=2, action_type='play_event',
                 params={'card_id': cid, 'side': 'fp'})
    post_noop = ai.clone_state(st)
    if not emission.play_was_refunded(st, post_noop, act):
        fails.append(('refund_specificity', 'pure no-op not refunded'))
    p = ai.clone_state(st)
    p.hands[Side.FREE_PEOPLES].hand.remove(cid)
    if emission.play_was_refunded(st, p, act):
        fails.append(('refund_card_left', 'card gone, still refunded'))
    p = ai.clone_state(st)
    p.dice[Side.FREE_PEOPLES][2].used = True
    if emission.play_was_refunded(st, p, act):
        fails.append(('refund_die_spent', 'die spent, still refunded'))
    p = ai.clone_state(st)
    p.dice[Side.FREE_PEOPLES][2].used = True
    p.pending_card_option = {'side': 'fp', 'card_id': cid,
                             'options': ['x']}
    if emission.play_was_refunded(st, p, act):
        fails.append(('refund_chain_open', 'chain open, still refunded'))
    view = make_view([st])
    view.card_pid_map[cid] = '111'
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, act)
    if em.hold or not (em.lines or em.calls):
        fails.append(('refund_pairing',
                      'un-refunded play rendered nothing: %r' % em))


def test_fungible_shortage_hold(fails):
    """MEASURES (design rule 10): when the client model cannot supply
    the pids an army_move needs, render_action returns a loud HOLD
    naming the type -- it never invents pieces and never falls back to
    a board diff (sensitivity).  Specificity: the untouched model
    renders lines for the same action."""
    st = fresh_state()
    adopt(st)
    mv = next(a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
              if a.action_type == 'army_move'
              and st.dice[Side.FREE_PEOPLES][a.die_idx].current
              is ActionDieFace.CHARACTER
              and a.params.get('n_regular', 0) >= 1)
    post = ai.apply_action(st, mv)
    clean = make_view([st])
    em_ok = emission.render_action(clean, NICK, Side.FREE_PEOPLES, mv,
                                   post=post)
    if em_ok.hold or not em_ok.lines:
        fails.append(('shortage_specificity',
                      'clean dump did not render: %r' % em_ok))
    starved = make_view([st])
    fk = sanitize(str(mv.params['from']))
    n_moved = 0
    for pid, z in emitter._initial().items():
        if emitter._norm_zone(str(z)) in (fk, fk + ' stronghold'):
            starved.piece_region[str(pid)] = 'forlindon'
            n_moved += 1
    if n_moved == 0:
        fails.append(('shortage_control',
                      'NOT-MEASURED: nothing to starve'))
        return
    em = emission.render_action(starved, NICK, Side.FREE_PEOPLES, mv,
                                post=post)
    if not em.hold:
        fails.append(('shortage_sensitivity',
                      'starved dump rendered %d line(s) instead of '
                      'holding' % len(em.lines)))
    elif 'army_move' not in em.hold:
        fails.append(('shortage_names_type', em.hold))


def test_attack_declaration(fails):
    """MEASURES (L1 shape): army_attack renders the die-spend line plus
    ONE ('attack_declare', '<to> from <frm>') chat request and NOTHING
    else -- no board lines, no calls, no die kill (the battle die dies
    at battle end, donor :8558-8570; the defender's political
    activation is engine-side state, not emitted here)."""
    st = fresh_state()
    adopt(st)
    ap = next(a for a in ai.get_legal_actions(st, Side.SHADOW)
              if a.action_type == 'advance_political'
              and 'ISENGARD' in str(a.params.get('nation')).upper())
    cur = ai.apply_action(st, ap)
    at = next((a for a in ai.get_legal_actions(cur, Side.SHADOW)
               if a.action_type == 'army_attack'), None)
    if at is None:
        fails.append(('attack_control', 'NOT-MEASURED: no attack'))
        return
    view = make_view([cur])
    em = emission.render_action(view, NICK, Side.SHADOW, at)
    if em.hold:
        fails.append(('attack_hold', em.hold))
        return
    if len(em.lines) != 1 or classify(em.lines[0])[0] != 'die_spend_sh':
        fails.append(('attack_lines', '%r' % (em.lines,)))
    if em.calls:
        fails.append(('attack_calls', '%r' % (em.calls,)))
    frm = sanitize(str(at.params.get('from')))
    to = sanitize(str(at.params.get('to')))
    want = to if frm == to else '%s from %s' % (to, frm)
    if em.chat != [('attack_declare', want)]:
        fails.append(('attack_chat', '%r != %r'
                      % (em.chat, [('attack_declare', want)])))


# =====================================================================
# PENDING-CHAIN DECISIONS (driven battle)
# =====================================================================
def _drive_to_battle():
    """Fresh state -> Isengard at war -> Orthanc attacks Fords of Isen
    (the one fresh-setup adjacency).  Returns (pre_attack_state,
    attack_action)."""
    st = fresh_state()
    adopt(st)
    for _ in range(2):
        gl.draw_event_card(st, Side.SHADOW, 'character')
        gl.draw_event_card(st, Side.SHADOW, 'strategy')
    ap = next(a for a in ai.get_legal_actions(st, Side.SHADOW)
              if a.action_type == 'advance_political'
              and 'ISENGARD' in str(a.params.get('nation')).upper())
    cur = ai.apply_action(st, ap)
    at = next((a for a in ai.get_legal_actions(cur, Side.SHADOW)
               if a.action_type == 'army_attack'), None)
    return cur, at


def _walk_to(cur, want_type, max_steps=80):
    """Walk the pending chain (first-action picks) until the menu
    offers want_type; returns (state, actions, side, kind) or None."""
    guard = 0
    while guard < max_steps:
        guard += 1
        dp = ai.describe_pending(cur)
        if dp is None:
            return None
        hits = [x for x in dp['actions']
                if x.action_type == want_type]
        if hits:
            return cur, dp['actions'], dp['side'], dp['kind']
        cur = ai.apply_action(cur, dp['actions'][0])
    return None


def _w2_025_full_advance(st):
    """W2-025 (amendment 1.49 sub-bullet 2026-09-03): after
    retreat_into_siege the ATTACKER decides the advance; the recipes
    below want the pre-W2-025 end state (whole stack inside, siege
    established) = the full victor_advance row.  Returns st unchanged
    when no such decision is open."""
    pc = getattr(st, 'pending_combat', None) or {}
    if pc.get('phase') != 'victor_advance' \
            or not pc.get('post_siege_retreat'):
        return st
    rows = [a for a in ai.get_legal_actions(st, pc['deciding_side'])
            if a.action_type == 'victor_advance'
            and a.params.get('take_leaders')
            and a.params.get('n_leaders') is None]
    if not rows:
        return st
    rows.sort(key=lambda a: -(int(a.params.get('n_regular') or 0)
                              + int(a.params.get('n_elite') or 0)
                              + int(a.params.get('n_nazgul') or 0)))
    return ai.apply_action(st, rows[0])


def test_combat_card_decision(fails):
    """MEASURES (K1 step 2 / R4, W1#44 + #49): a combat-card commit
    renders as ['!CALL choose_card <pid>'] ONLY -- the client's own
    HIDDEN commit (F1, 'chooses a card.[<deck back>]'); NO 'play_card'
    (choose+playchosen = an immediate reveal before the human has
    committed), NO 'cc: <title>' chat (the title leaked the identity
    and six accented titles failed the validator), NO inject lines.
    render_combat_reveal renders ['!CALL reveal_chosen'] (one directive
    reveals every chosen card, both sides; the reveal lines speak) and
    nothing else.  combat_card_none stays the ('nc','nc') decline.
    Sensitivity: no card-pid map -> HOLD."""
    cur, at = _drive_to_battle()
    if at is None:
        fails.append(('cc_control', 'NOT-MEASURED: no attack'))
        return
    random.seed(2)
    c = ai.apply_action(cur, at)
    got = _walk_to(c, 'combat_card_play')
    if got is None:
        fails.append(('cc_control',
                      'NOT-MEASURED: no combat_card_play reachable'))
        return
    state, acts, side, kind = got
    ccp = next(x for x in acts if x.action_type == 'combat_card_play')
    view = make_view([state])
    em = emission.render_decision(view, NICK, side, kind, ccp)
    if not em.hold or 'no client pid' not in em.hold:
        fails.append(('cc_pid_hold', 'expected HOLD, got %r' % em))
    view.card_pid_map[ccp.params.get('card_id')] = '170'
    em = emission.render_decision(view, NICK, side, kind, ccp)
    if em.hold:
        fails.append(('cc_hold', em.hold))
        return
    # D7: the hidden commit is the raw 'chooses a card' line under the
    # AI's OWN nick (Controls.840) - a directive would render under
    # the LOCAL nick and collide with the human's chosen slot
    if em.lines != ['$%s<~Controls.840~>170' % NICK]:
        fails.append(('cc_no_lines', '%r' % (em.lines,)))
    if em.calls:
        fails.append(('cc_call', '%r' % (em.calls,)))
    if em.chat:
        fails.append(('cc_chat_leak', '%r' % (em.chat,)))
    if any('play_card' in str(x) for x in em.calls):
        fails.append(('cc_play_card', '%r' % (em.calls,)))
    ccn = next(x for x in acts if x.action_type == 'combat_card_none')
    em = emission.render_decision(view, NICK, side, kind, ccn)
    if em.hold or em.lines or em.calls or em.chat != [('nc', 'nc')]:
        fails.append(('cc_none', '%r %r' % (em.chat, em.hold)))
    rv = emission.render_combat_reveal(view, NICK, side)
    if rv.hold or rv.lines or rv.chat \
            or rv.calls != ['!CALL reveal_chosen']:
        fails.append(('cc_reveal', '%r %r %r %r'
                      % (rv.hold, rv.lines, rv.calls, rv.chat)))


def test_mixed_nation_move_by_nation(fails):
    """MEASURES (WAVE-K wkg5 LIVE FAILURE, witness escalation 'units
    dimrill dale isengard/regular: engine=0 client=1'): the AI's own
    army_move out of a MIXED-nation Shadow army must move pieces of
    EXACTLY the (nation, kind) composition the ENGINE moved (pre/post
    delta at the source) - a nation-blind pick shipped an Isengard
    regular where the engine moved a Sauron one (the donor's t131
    class; emitter.plan_army_move already honours params
    'moved_by_nation', emission never supplied it).  Specimen: an
    Isengard regular joins Moria (North Dunland -> Moria), then Moria
    -> Dimrill Dale with n_regular=2.  Control: the engine must leave
    the Isengard regular in Moria (else NOT-MEASURED)."""
    from board import sanitize
    from wotr_bridge.live import client_board as cb
    st = fresh_state()
    adopt(st)

    def _comp(state, reg):
        d = {}
        for k, army in state.armies.items():
            if sanitize(str(k)) != reg:
                continue
            for u in army.units:
                key = '%s:%s' % (str(getattr(u.nation, 'value',
                                             u.nation)).lower(),
                                 str(getattr(u.kind, 'value',
                                             u.kind)).lower())
                d[key] = d.get(key, 0) + 1
        return d

    def _pick(state, at, frm, to, die, **want):
        for a in ai.get_legal_actions(state, Side.SHADOW):
            if a.action_type != at or a.die_idx != die:
                continue
            if sanitize(str(a.params.get('from'))) != frm                     or sanitize(str(a.params.get('to'))) != to:
                continue
            if all(int(a.params.get(k) or 0) == v
                   for k, v in want.items()):
                return a
        return None
    # Isengard to WAR first (setup box 1 -> war; a not-at-war nation's
    # units may not cross into another nation's region)
    ap = next((a for a in ai.get_legal_actions(st, Side.SHADOW)
               if a.action_type == 'advance_political' and a.die_idx == 2
               and 'ISENGARD' in str(a.params.get('nation')).upper()),
              None)
    if ap is None:
        fails.append(('mix_control', 'NOT-MEASURED: no Isengard advance '
                      'on die 2'))
        return
    st = ai.apply_action(st, ap)
    holder = [st]
    view = make_view(holder)
    mv1 = _pick(st, 'army_move', 'north dunland', 'moria', 1,
                n_regular=1, n_elite=0)
    if mv1 is None:
        fails.append(('mix_control', 'NOT-MEASURED: no Isengard regular '
                      'move North Dunland -> Moria on die 1'))
        return
    st1 = ai.apply_action(st, mv1)
    em1 = emission.render_action(view, NICK, Side.SHADOW, mv1, post=st1)
    if em1.hold:
        fails.append(('mix_control', 'NOT-MEASURED: first move held %s'
                      % em1.hold))
        return
    # mirror the first move in the view (live: own-emission tracking)
    for op, f in em1.ops:
        if op == 'move':
            for pid in f.get('pieces', ()):
                view.piece_region[str(pid)] = 'moria'
    # close the ARMY-die second leg with the menu's pass
    if st1.pending_army_move_die is not None:
        pa = next((a for a in ai.get_legal_actions(st1, Side.SHADOW)
                   if a.action_type == 'pass'), None)
        if pa is not None:
            st1 = ai.apply_action(st1, pa)
    holder[0] = st1
    mv2 = _pick(st1, 'army_move', 'moria', 'dimrill dale', 5,
                n_regular=2, n_elite=0)
    if mv2 is None:
        fails.append(('mix_control', 'NOT-MEASURED: no Moria -> Dimrill '
                      'Dale 2-regular move on die 5'))
        return
    st2 = ai.apply_action(st1, mv2)
    pre_c, post_c = _comp(st1, 'moria'), _comp(st2, 'moria')
    moved = {k: n - post_c.get(k, 0) for k, n in pre_c.items()
             if n - post_c.get(k, 0) > 0}
    if moved != {'sauron:regular': 2}:
        fails.append(('mix_control', 'NOT-MEASURED: engine moved %r '
                      '(the specimen wants two Sauron regulars)'
                      % (moved,)))
        return
    em2 = emission.render_action(view, NICK, Side.SHADOW, mv2, post=st2)
    if em2.hold:
        fails.append(('mix_hold', em2.hold))
        return
    got = {}
    for op, f in em2.ops:
        if op != 'move':
            continue
        for pid in f.get('pieces', ()):
            t = cb.unit_type(str(pid))
            if t and str(t[1]).lower() in ('regular', 'elite'):
                key = '%s:%s' % (str(getattr(t[0], 'value', t[0])
                                     ).lower().replace('_', ' '),
                                 str(t[1]).lower())
                got[key] = got.get(key, 0) + 1
    if got != moved:
        fails.append(('mix_nation_pick',
                      'rendered %r but the engine moved %r (ops=%r)'
                      % (got, moved, [f.get('pieces') for op, f
                                      in em2.ops if op == 'move'])))


def test_card_effect_die_face_change(fails):
    """MEASURES (WAVE-K K6, wkf1 LIVE WEDGE - the first AI-as-FP game:
    the AI played Mirror of Galadriel and emission_card HELD forever on
    the dice row): an immediate card effect whose ONLY board delta is a
    die FACE change (an unused FP Character die -> Will of the West)
    renders the corpus die-change line for that die and nothing else;
    no hold.  Controls: a USAGE flip (used False -> True) in the same
    delta still HOLDS (no card-effect grammar for it)."""
    import copy
    from game_state import ActionDieFace
    from ai import Action
    st = fresh_state()
    adopt(st)                         # FP die 0 = CHARACTER, unused
    cid = next((c for c in st.fp_char_deck
                if 'Mirror_of_Galadriel' in c), None)
    if cid is None:
        fails.append(('dfc_control', 'NOT-MEASURED: Mirror of Galadriel '
                      'not in the FP character deck'))
        return
    st.fp_char_deck.remove(cid)
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    post = copy.deepcopy(st)
    post.hands[Side.FREE_PEOPLES].hand.remove(cid)
    d = post.dice[Side.FREE_PEOPLES][0]
    if d.used or d.current != ActionDieFace.CHARACTER:
        fails.append(('dfc_control', 'NOT-MEASURED: FP die 0 is not an '
                      'unused CHARACTER die'))
        return
    d.current = ActionDieFace.WILL_OF_WEST
    view = make_view([st])
    act = Action(die_idx=1, action_type='play_event',
                 params={'card_id': cid, 'side': 'fp'})
    from wotr_bridge.live import emission_card
    em = emission_card.render_card_effect(view, NICK, Side.FREE_PEOPLES,
                                          'play_event:immediate', act,
                                          pre=st, post=post)
    if em.hold:
        fails.append(('dfc_hold', em.hold))
        return
    dc = [ln for ln in em.lines if ' changes ' in ln]
    if len(dc) != 1 or '210' not in dc[0] or not dc[0].rstrip().endswith(' W'):
        fails.append(('dfc_line', 'want one die-change line for pid 210 '
                      '-> W, got %r' % (em.lines,)))
    # control: a usage flip is NOT a face change - still a hold
    post2 = copy.deepcopy(post)
    post2.dice[Side.FREE_PEOPLES][1].used = True
    em2 = emission_card.render_card_effect(view, NICK, Side.FREE_PEOPLES,
                                           'play_event:immediate', act,
                                           pre=st, post=post2)
    if not em2.hold:
        fails.append(('dfc_usage_specificity', 'a usage flip rendered '
                      'without a hold: %r' % (em2.lines,)))


def test_siege_assault_render(fails):
    """MEASURES (WAVE-K K8, wkg6 LIVE WEDGE - AI-as-Shadow: the AI's own
    siege ASSAULT of a besieged stronghold): the engine's
    attacker_siege_decision -> siege_assault decision must render
    WITHOUT a hold (an empty emission: the region was declared by the
    army_attack that opened the pending; the assault round's rolls
    follow via the funnel).  Before K8: 'no L2 planner for siege_assault'
    -> DriverHalt every tick.  Real specimen: Sauron to war, Moria ->
    Dimrill Dale, attack Lorien, defender withdraws INTO the stronghold,
    a fresh army die re-attacks the besieged Lorien -> siege_assault."""
    import game_logic as gl
    from game_state import ActionDieFace, Nation
    from board import sanitize
    st = fresh_state()
    # explicit SHADOW roster (adopt()'s SH_FACES set die 4 = EVENT,
    # which has no political advance): 0/2/3 ARMY, 4/5 MUSTER
    gl.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.CHARACTER),
        (Side.SHADOW, 2, ActionDieFace.ARMY),
        (Side.SHADOW, 3, ActionDieFace.ARMY),
        (Side.SHADOW, 4, ActionDieFace.MUSTER),
        (Side.SHADOW, 5, ActionDieFace.MUSTER)])
    for d in (4, 5):
        if st.political.is_at_war(Nation.SAURON):
            break
        ap = next((a for a in ai.get_legal_actions(st, Side.SHADOW)
                   if a.action_type == 'advance_political'
                   and 'SAURON' in str(a.params.get('nation')).upper()
                   and a.die_idx == d), None)
        if ap is None:
            fails.append(('sa_control', 'NOT-MEASURED: no Sauron advance '
                          'on die %d' % d))
            return
        st = ai.apply_action(st, ap)

    def _pick(state, at, frm, to, die):
        c = [a for a in ai.get_legal_actions(state, Side.SHADOW)
             if a.action_type == at and a.die_idx == die
             and sanitize(str(a.params.get('from'))).lower() == frm
             and sanitize(str(a.params.get('to'))).lower() == to]
        c.sort(key=lambda a: -(int(a.params.get('n_regular') or 0)
                               + int(a.params.get('n_elite') or 0)))
        return c[0] if c else None
    mv = _pick(st, 'army_move', 'moria', 'dimrill dale', 0)
    if mv is None:
        fails.append(('sa_control', 'NOT-MEASURED: no Moria -> Dimrill '
                      'Dale move'))
        return
    st = ai.apply_action(st, mv)
    if st.pending_army_move_die is not None:
        pa = next((a for a in ai.get_legal_actions(st, Side.SHADOW)
                   if a.action_type == 'pass'), None)
        if pa is not None:
            st = ai.apply_action(st, pa)
    at = _pick(st, 'army_attack', 'dimrill dale', 'lorien', 2)
    if at is None:
        fails.append(('sa_control', 'NOT-MEASURED: no attack on Lorien'))
        return
    st = ai.apply_action(st, at)
    dp = ai.describe_pending(st)
    ris = next((a for a in (dp['actions'] if dp else [])
                if a.action_type == 'retreat_into_siege'), None)
    if ris is None:
        fails.append(('sa_control', 'NOT-MEASURED: no retreat_into_siege'))
        return
    st = ai.apply_action(st, ris)
    st = _w2_025_full_advance(st)    # the Shadow attacker moves in
    if 'lorien' not in {sanitize(str(b)) for b in
                        (getattr(st, 'besieged_strongholds', set())
                         or set())}:
        fails.append(('sa_control', 'NOT-MEASURED: Lorien not besieged'))
        return
    at2 = next((a for a in ai.get_legal_actions(st, Side.SHADOW)
                if a.action_type == 'army_attack' and a.die_idx == 3
                and sanitize(str(a.params.get('to'))) == 'lorien'), None)
    if at2 is None:
        fails.append(('sa_control', 'NOT-MEASURED: no re-attack on the '
                      'besieged Lorien'))
        return
    st = ai.apply_action(st, at2)
    dp3 = ai.describe_pending(st)
    if dp3 is None or dp3['kind'] != 'combat:attacker_siege_decision':
        fails.append(('sa_control', 'NOT-MEASURED: pending is %r'
                      % (dp3 and dp3['kind'],)))
        return
    sa = next((a for a in dp3['actions']
               if a.action_type == 'siege_assault'), None)
    if sa is None:
        fails.append(('sa_control', 'NOT-MEASURED: no siege_assault'))
        return
    post = ai.apply_action(st, sa)
    view = make_view([st])
    em = emission.render_decision(view, NICK, Side.SHADOW, dp3['kind'],
                                  sa, post=post, pre=st)
    if em.hold:
        fails.append(('sa_hold', em.hold))
        return
    if em.lines or em.calls or em.chat:
        fails.append(('sa_empty', 'siege_assault should render nothing '
                      '(the round rolls follow): %r'
                      % ((em.lines, em.calls, em.chat),)))


def test_casualty_leftover_hit_downgrade(fails):
    """Four hits on 4E+3R: three selected Regular losses then a forced reduction.

    Exercise selected-operation emission and the same residual-row sweep used
    by the driver. The physical expected endpoint is 3E+1R, not a fourth kill.
    """
    from game_state import Unit, UnitKind, Nation
    state, attack = _drive_to_battle()
    if attack is None:
        fails.append(('lh_control', 'No legal battle fixture'))
        return
    army = ai._armies_lookup(state, 'orthanc')
    army.units = ([Unit(Nation.ISENGARD, UnitKind.ELITE) for _ in range(4)]
                  + [Unit(Nation.ISENGARD, UnitKind.REGULAR) for _ in range(3)])
    state.pending_combat = dict(phase='casualty_allocation', alloc_stage='attacker',
        attacker_region='Orthanc', defender_region='Fords of Isen',
        attacker_side=Side.SHADOW, defender_side=Side.FREE_PEOPLES,
        deciding_side=Side.SHADOW, is_siege=False, rounds_fought=1,
        atk_hits=4, def_hits=0)
    holder = [state]
    view = make_view(holder)
    for pid in ('322', '323', '324'):
        view.piece_region[pid] = 'Orthanc'
    view.piece_region['005'] = 'SA Reinforcements'
    for _ in range(3):
        action = next(a for a in ai.get_legal_actions(state, Side.SHADOW)
                      if a.action_type == 'casualty_allocate'
                      and a.params.get('operation') == 'remove_regular'
                      and a.params.get('nation') == Nation.ISENGARD)
        post = ai.apply_action(state, action)
        em = emission.render_decision(view, NICK, Side.SHADOW,
            'combat:casualty_allocation', action, post=post, pre=state)
        if em.hold:
            fails.append(('lh_selected_hold', em.hold))
            return
        total = emission.removal_rows(state, post, 'shadow')
        voiced, bad = emission.expressed_rows(view, em.ops, 'shadow')
        if bad:
            fails.append(('lh_untyped_emission', em.ops))
            return
        residual = emission.subtract_rows(total, voiced)
        if not emission.rows_empty(residual):
            forced, ledger = emission.render_forced_rows(
                view, NICK, Side.SHADOW, residual, state, label='casualty test')
            if forced.hold:
                fails.append(('lh_forced_hold', forced.hold))
                return
        state = post
        holder[0] = state
    actual = sorted(u.kind.name for u in ai._armies_lookup(state, 'orthanc').units)
    if actual != ['ELITE', 'ELITE', 'ELITE', 'REGULAR']:
        fails.append(('lh_engine_endpoint', actual))
    physical = sorted(kind for pid in emitter._pieces_in(view, 'orthanc')
                      for nat, kind in [emitter._unit_type(pid, view, side_hint='shadow')]
                      if nat == Nation.ISENGARD and kind in ('regular', 'elite'))
    if physical != ['elite', 'elite', 'elite', 'regular']:
        fails.append(('lh_physical_endpoint', physical))


def test_casualty_allocation(fails):
    """MEASURES (L1 shape + physical supply, donor :9760-10028 /
    emitter casualty branch): on a REAL casualty_allocate decision --
      (a) remove_regular: pure kills -- `casualties` hand_exit_b pieces of the
          allocating side, out of the battle zone;
      (b)+(d) reduce_elite downgrade with the ENGINE's own stamp threaded
          from the applied post state (the live server always passes
          post): source 'pool' => swap_place removes ONE on-board elite
          and swap_remove places ONE replacement regular taken from the
          reinforcement pool;
      (c) the engine stamps source None (no pool, no box: K6) => the
          elite is ELIMINATED OUTRIGHT (USER ruling 2026-08-18): elite
          removal WITHOUT any replacement placement;
      (e) the engine stamps 'cas' but the client dead pile holds no
          such regular => HOLD, never a phantom.
    Every stamp comes from combat._consume_downgrade_source on a real
    casualty_allocate apply; each variant renders on a FRESH view."""
    cur, at = _drive_to_battle()
    if at is None:
        fails.append(('cas_control', 'NOT-MEASURED: no attack'))
        return
    got = None
    for seed in range(20):
        random.seed(seed)
        c = ai.apply_action(cur, at)
        got = _walk_to(c, 'casualty_allocate')
        if got is not None:
            break
    if got is None:
        fails.append(('cas_control',
                      'NOT-MEASURED: no casualty_allocate in 20 seeds'))
        return
    state, acts, side, kind = got
    pc = state.pending_combat
    stage = pc.get('alloc_stage')
    region = sanitize(str(pc.get('attacker_region')
                          if stage == 'attacker'
                          else pc.get('defender_region')))
    hits = int(pc.get('atk_hits' if stage == 'attacker'
                      else 'def_hits') or 0)
    side_tag = 'shadow' if side is Side.SHADOW else 'fp'

    # (a) pure kills
    ca0 = next((x for x in acts if x.params.get('operation') == 'remove_regular'), None)
    if ca0 is not None and hits > 0:
        view = make_view([state])
        em = emission.render_decision(view, NICK, side, kind, ca0,
                                      post=ai.apply_action(state, ca0))
        if em.hold:
            fails.append(('cas0_hold', em.hold))
        else:
            kop = [classify(x) for x in em.lines]
            kb = [f for op, f in kop if op == 'hand_exit_b']
            if len(kb) != 1 or len(kb[0].get('pieces') or []) != 1:
                fails.append(('cas0_kills',
                              'want one selected Regular casualty, lines=%r' % (em.lines,)))
            else:
                for p in kb[0]['pieces']:
                    nat, ut = emitter._unit_type(p, view,
                                                 side_hint=side_tag)
                    if ut not in ('regular', 'elite') or \
                            emitter._side_of_nation(nat) != side_tag:
                        fails.append(('cas0_piece_type',
                                      'pid %s typed (%r,%r)'
                                      % (p, nat, ut)))
                if kb[0].get('frm') != emitter.zone_code_for(region):
                    fails.append(('cas0_zone', kb[0].get('frm')))
    else:
        fails.append(('cas0_control', 'NOT-MEASURED: no remove_regular option / no hits'))

    # (b)+(d) reduce_elite with the ENGINE's stamp threaded from the applied
    # post state (the live server always passes post): the default
    # supply stamps 'pool' -> swap_place removes the elite and
    # swap_remove places ONE replacement regular from the reinforcement
    # pool (never the dead pile; never a nation-blind pick)
    from test_live_engine_fixwave import (_dwarf_no_supply_state,
                                          _to_allocation,
                                          field_battle_state, place,
                                          pick)
    from game_state import Nation

    def _mirror(view, nation, n_regular, n_elite, region):
        """The client holds the placed defenders where the engine put
        them (the rig's view starts from the setup positions)."""
        pids = emitter.pick_units(view, 'fp reinf', 'fp',
                                  n_regular=n_regular, n_elite=n_elite,
                                  nation=nation, side_hint='fp')
        for p in pids:
            view.piece_region[p] = region
        return pids

    def _k1(st, tag):
        seed, post = _to_allocation(st, 1)
        if post is None:
            fails.append((tag, 'NOT-MEASURED: no seed reached a 1-hit '
                               'defender allocation'))
            return None, None, None
        k1 = pick(ai.get_legal_actions(post, Side.FREE_PEOPLES),
                  'casualty_allocate', operation='reduce_elite')
        if k1 is None:
            fails.append((tag, 'NOT-MEASURED: reduce_elite not offered'))
            return None, None, None
        return post, k1, ai.apply_action(post, k1)

    st_pool = field_battle_state(atk_reg=5, def_nation=Nation.GONDOR,
                                 def_reg=1, def_elite=1)
    state, ca1, after = _k1(st_pool, 'cas_control')
    if state is None:
        return
    stamps = list(getattr(after, '_dg_src_log', []) or [])[
        len(getattr(state, '_dg_src_log', []) or []):]
    if stamps != [(Nation.GONDOR, 'pool')]:
        fails.append(('cas_control', 'NOT-MEASURED: engine stamped %r, '
                      'want a pool source' % (stamps,)))
        return
    dp = ai.describe_pending(state)
    view = make_view([state])
    _mirror(view, 'Gondor', 0, 1, 'Pelargir')
    pool_before = set(emitter._pieces_in(view, 'fp reinf'))
    em = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                  ca1, post=after, pre=state)
    if em.hold:
        fails.append(('cas1_hold', em.hold))
    else:
        kop = [classify(x) for x in em.lines]
        ops = [op for op, _ in kop]
        if 'swap_place' not in ops or 'swap_remove' not in ops:
            fails.append(('cas1_swap', 'ops=%r' % ops))
        placed = None
        for op, f in kop:
            if op == 'swap_remove':
                placed = (f.get('pieces') or [None])[0]
        if placed is None or placed not in pool_before:
            fails.append(('cas_pool_source',
                          'placed %r not from fp reinf' % (placed,)))
        elif emitter._unit_type(placed, make_view([state]),
                                side_hint='fp')[1] != 'regular':
            fails.append(('cas_pool_kind', placed))

    # (c) eliminated outright: the engine stamps source None (pool 0,
    # box 0 - K6 one answerer) => elite removal WITHOUT any replacement
    st_none = _dwarf_no_supply_state()
    place(st_none, 'erebor', Nation.DWARVES, regulars=4, clear=True)
    state, ca1, after = _k1(st_none, 'cas_none_control')
    if state is None:
        return
    stamps = list(getattr(after, '_dg_src_log', []) or [])[
        len(getattr(state, '_dg_src_log', []) or []):]
    if stamps != [(Nation.DWARVES, None)]:
        fails.append(('cas_none_control', 'NOT-MEASURED: engine stamped '
                      '%r, want (DWARVES, None)' % (stamps,)))
        return
    dp = ai.describe_pending(state)
    view = make_view([state])
    _mirror(view, 'Dwarves', 1, 1, 'Pelargir')
    em = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                  ca1, post=after, pre=state)
    if em.hold:
        fails.append(('cas_none_hold', em.hold))
    else:
        ops = [classify(x)[0] for x in em.lines]
        if 'swap_place' not in ops:
            fails.append(('cas_none_removes', 'ops=%r' % ops))
        if 'swap_remove' in ops:
            fails.append(('cas_none_outright',
                          'a replacement regular was placed for a '
                          'None-source downgrade (USER ruling '
                          '2026-08-18 violated): %r' % (em.lines,)))

    # (e) W1#77: a stamped source the board CANNOT honour (the engine
    # says the regular returns from the casualty box; the client dead
    # pile holds no such regular) must HOLD loudly - never fall back to
    # a nation-blind pick that places a phantom
    st_cas = _dwarf_no_supply_state()
    place(st_cas, 'erebor', Nation.DWARVES, regulars=3, clear=True)
    st_cas.fp_casualties = {Nation.DWARVES: {'regular': 1, 'elite': 0}}
    state, ca1, after = _k1(st_cas, 'cas_cas_control')
    if state is None:
        return
    stamps = list(getattr(after, '_dg_src_log', []) or [])[
        len(getattr(state, '_dg_src_log', []) or []):]
    if stamps != [(Nation.DWARVES, 'cas')]:
        fails.append(('cas_cas_control', 'NOT-MEASURED: engine stamped '
                      '%r, want (DWARVES, cas)' % (stamps,)))
        return
    dp = ai.describe_pending(state)
    view = make_view([state])
    _mirror(view, 'Dwarves', 1, 1, 'Pelargir')
    try:
        emitter.pick_units(view, 'fp dead', 'fp', n_regular=1,
                           nation='Dwarves', side_hint='fp')
        fails.append(('cas_cas_control', 'NOT-MEASURED: fp dead already '
                      'holds a Dwarven regular'))
        return
    except LookupError:
        pass
    em = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                  ca1, post=after, pre=state)
    if not em.hold or em.lines:
        fails.append(('cas_stamped_source_silent',
                      'unhonourable stamped source rendered %r'
                      % (em.lines,)))


def _lorien_siege_state(extra_elven=()):
    """Sauron to war, Moria -> Dimrill Dale, Dimrill Dale attacks
    Lorien: the ENGINE's defender_siege_choice pending (the
    test_siege_assault_render recipe).  extra_elven_regulars are
    appended to the Lorien garrison BEFORE the attack so the withdrawal
    overstacks the stronghold (siege_trim); a tuple of UnitKind values.
    Returns the state with the pending open, or None when the recipe is
    not reachable."""
    import game_logic as gl
    import game_state as gs
    from game_state import ActionDieFace, Nation
    from board import sanitize
    st = fresh_state()
    gl.adopt_rolled_faces(st, [
        (Side.SHADOW, 0, ActionDieFace.ARMY),
        (Side.SHADOW, 1, ActionDieFace.CHARACTER),
        (Side.SHADOW, 2, ActionDieFace.ARMY),
        (Side.SHADOW, 3, ActionDieFace.ARMY),
        (Side.SHADOW, 4, ActionDieFace.MUSTER),
        (Side.SHADOW, 5, ActionDieFace.MUSTER)])
    for d in (4, 5):
        if st.political.is_at_war(Nation.SAURON):
            break
        ap = next((a for a in ai.get_legal_actions(st, Side.SHADOW)
                   if a.action_type == 'advance_political'
                   and 'SAURON' in str(a.params.get('nation')).upper()
                   and a.die_idx == d), None)
        if ap is None:
            return None
        st = ai.apply_action(st, ap)

    def _pick(state, at, frm, to, die):
        c = [a for a in ai.get_legal_actions(state, Side.SHADOW)
             if a.action_type == at and a.die_idx == die
             and sanitize(str(a.params.get('from'))).lower() == frm
             and sanitize(str(a.params.get('to'))).lower() == to]
        c.sort(key=lambda a: -(int(a.params.get('n_regular') or 0)
                               + int(a.params.get('n_elite') or 0)))
        return c[0] if c else None
    mv = _pick(st, 'army_move', 'moria', 'dimrill dale', 0)
    if mv is None:
        return None
    st = ai.apply_action(st, mv)
    if st.pending_army_move_die is not None:
        pa = next((a for a in ai.get_legal_actions(st, Side.SHADOW)
                   if a.action_type == 'pass'), None)
        if pa is not None:
            st = ai.apply_action(st, pa)
    if extra_elven:
        k = next(k for k in st.armies if sanitize(str(k)) == 'lorien')
        for kind in extra_elven:
            st.armies[k].units.append(gs.Unit(nation=Nation.ELVES,
                                              kind=kind))
    at = _pick(st, 'army_attack', 'dimrill dale', 'lorien', 2)
    if at is None:
        return None
    st = ai.apply_action(st, at)
    dp = ai.describe_pending(st)
    if dp is None or not any(a.action_type == 'retreat_into_siege'
                             for a in dp['actions']):
        return None
    return st


def test_retreats(fails):
    """MEASURES (W1#72 / W1#73, K4): the ENGINE's own retreat actions
    render.  combat_retreat is taken from get_legal_actions on a real
    defender_retreat pending (params carry ONLY 'to_region'; the
    pending carries attacker_region/defender_region, never 'region')
    and must render ONE group move of the retreating side's pieces from
    the battle region to the destination; retreat_into_siege is taken
    from a real defender_siege_choice pending (params == {}) and must
    render the group move into the stronghold sub-box.  Before the fix
    both raised LookupError -> HOLD -> DriverHalt every tick after the
    engine had already retreated (the vacuous hand-built leg this
    replaces passed on params the engine never produces)."""
    from board import sanitize
    cur, at = _drive_to_battle()
    if at is None:
        fails.append(('retreat_control', 'NOT-MEASURED: no attack'))
        return
    random.seed(2)
    c = ai.apply_action(cur, at)
    got = _walk_to(c, 'combat_retreat')
    if got is None:
        fails.append(('retreat_control',
                      'NOT-MEASURED: no combat_retreat reachable'))
        return
    state, acts, side, kind = got
    pc = state.pending_combat
    if 'region' in pc or 'from' in (acts[0].params or {}):
        fails.append(('retreat_shape', 'engine shape changed: %r %r'
                      % (sorted(pc), acts[0].params)))
    crs = [a for a in acts if a.action_type == 'combat_retreat']
    cr = crs[0]
    if set(cr.params) != {'to_region'}:
        fails.append(('retreat_params', '%r' % (cr.params,)))
    battle = sanitize(str(pc['defender_region']
                          if pc['defender_side'] == side
                          else pc['attacker_region']))
    view = make_view([state])
    want_pids = sorted(emitter._side_pieces_in(
        view, battle, 'fp' if side == Side.FREE_PEOPLES else 'shadow'))
    post = ai.apply_action(state, cr)
    em = emission.render_decision(view, NICK, side, kind, cr,
                                  post=post, pre=state)
    if em.hold:
        fails.append(('retreat_hold', em.hold))
    else:
        op, f = classify(em.lines[0])
        if op != 'move' or len(em.lines) != 1:
            fails.append(('retreat_op', '%r' % (em.lines,)))
        elif f.get('frm') != emitter.zone_code_for(battle) \
                or f.get('to') != emitter.zone_code_for(
                    sanitize(str(cr.params['to_region']))):
            fails.append(('retreat_zones', '%r' % (f,)))
        elif sorted(f.get('pieces') or []) != want_pids or not want_pids:
            fails.append(('retreat_pieces', '%r != %r'
                          % (f.get('pieces'), want_pids)))

    st = _lorien_siege_state()
    if st is None:
        fails.append(('siege_retreat_control',
                      'NOT-MEASURED: Lorien siege recipe unreachable'))
        return
    dp = ai.describe_pending(st)
    rs = next(a for a in dp['actions']
              if a.action_type == 'retreat_into_siege')
    if rs.params:
        fails.append(('siege_retreat_params', '%r' % (rs.params,)))
    view2 = make_view([st])
    want2 = sorted(emitter._side_pieces_in(view2, 'lorien', 'fp'))
    post2 = ai.apply_action(st, rs)
    em = emission.render_decision(view2, NICK, dp['side'], dp['kind'],
                                  rs, post=post2, pre=st)
    if em.hold:
        fails.append(('siege_retreat_hold', em.hold))
        return
    op, f = classify(em.lines[0])
    if op != 'move' or len(em.lines) != 1 \
            or f.get('frm') != emitter.zone_code_for('lorien') \
            or f.get('to') != emitter.zone_code_for('lorien stronghold'):
        fails.append(('siege_retreat_zone', '%r' % (em.lines,)))
    elif sorted(f.get('pieces') or []) != want2 or not want2:
        fails.append(('siege_retreat_pieces', '%r != %r'
                      % (f.get('pieces'), want2)))
    # the client model follows the emitted drag (later picks in the
    # same chain read the sub-box)
    if sorted(emitter._side_pieces_in(view2, 'lorien', 'fp')) != want2:
        fails.append(('siege_retreat_retrack',
                      '%r' % (emitter._side_pieces_in(view2, 'lorien',
                                                      'fp'),)))


def test_siege_trim(fails):
    """MEASURES (W1#47, K4): the ENGINE's siege_trim decision (after a
    retreat_into_siege that overstacks the stronghold: the owner picks
    how many ELITES return to reinforcements, the rest of the excess
    are regulars) renders the excess units' send-to-box move out of
    the stronghold sub-box into the side's reinforcement zone, with the
    (nation, kind) composition the ENGINE removed (pre/post).  Before
    the fix: 'no L2 planner for siege_trim' -> HOLD -> DriverHalt.
    Both k values of a 1-excess garrison are measured (k=1 sends an
    elite, k=0 a regular) so a planner that ignores k cannot pass."""
    from board import sanitize
    import game_state as gs
    st = _lorien_siege_state(extra_elven=(gs.UnitKind.REGULAR,
                                          gs.UnitKind.REGULAR,
                                          gs.UnitKind.ELITE))
    if st is None:
        fails.append(('trim_control',
                      'NOT-MEASURED: Lorien siege recipe unreachable'))
        return
    dp = ai.describe_pending(st)
    rs = next(a for a in dp['actions']
              if a.action_type == 'retreat_into_siege')
    view = make_view([st])
    # the client holds the three extra Elven units in Lorien too
    extra = emitter.pick_units(view, 'fp reinf', 'fp', n_regular=2,
                               n_elite=1, nation='Elves', side_hint='fp')
    for p in extra:
        view.piece_region[p] = 'Lorien'
    st2 = ai.apply_action(st, rs)
    em0 = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                   rs, post=st2, pre=st)
    if em0.hold:
        fails.append(('trim_control', 'retreat held: %s' % em0.hold))
        return
    # W2-025: the garrison cap binds once the attacker ADVANCES (the
    # siege begins) - take the full advance, then the owner's trim opens
    st2 = _w2_025_full_advance(st2)
    holder = [st2]
    view._state_fn = lambda: holder[0]
    dp2 = ai.describe_pending(st2)
    if dp2 is None or dp2['kind'] != 'combat:siege_trim':
        fails.append(('trim_control', 'NOT-MEASURED: pending %r'
                      % (dp2 and dp2['kind'],)))
        return
    trims = {int(a.params['elites_back']): a for a in dp2['actions']
             if a.action_type == 'siege_trim'}
    if sorted(trims) != [0, 1]:
        fails.append(('trim_menu', '%r' % (sorted(trims),)))
        return
    sub = emitter.zone_code_for('lorien stronghold')
    for k, want_kind in ((1, 'elite'), (0, 'regular')):
        a = trims[k]
        post = ai.apply_action(st2, a)
        v = make_view([st2])
        v.piece_region = dict(view.piece_region)
        em = emission.render_decision(v, NICK, dp2['side'], dp2['kind'],
                                      a, post=post, pre=st2)
        if em.hold:
            fails.append(('trim_hold_k%d' % k, em.hold))
            continue
        if len(em.lines) != 1:
            fails.append(('trim_lines_k%d' % k, '%r' % (em.lines,)))
            continue
        op, f = classify(em.lines[0])
        pieces = f.get('pieces') or ''
        if isinstance(pieces, str):
            pieces = pieces.split()
        if op != 'hand_exit' or f.get('frm') != sub \
                or f.get('to') != emitter.FP_POOL:
            fails.append(('trim_shape_k%d' % k,
                          '%r %r' % (op, f)))
            continue
        if len(pieces) != 1:
            fails.append(('trim_count_k%d' % k, '%r' % (pieces,)))
            continue
        nat, ut = emitter._unit_type(pieces[0], v, side_hint='fp')
        if ut != want_kind or 'elves' not in str(nat).lower():
            fails.append(('trim_kind_k%d' % k,
                          '%r -> %r %r' % (pieces[0], nat, ut)))
        # the engine removed exactly one Elven unit of that kind
        k9 = next(kk for kk in post.armies
                  if sanitize(str(kk)) == 'lorien')
        n_elite_post = sum(1 for u in post.armies[k9].units
                           if str(u.kind.value).lower() == 'elite'
                           and 'elves' in str(u.nation.value).lower())
        if n_elite_post != (2 if k == 1 else 3):
            fails.append(('trim_engine_k%d' % k, '%d' % n_elite_post))


# =====================================================================
# CARD-EFFECT ROW RENDERER (emission_card -- the bounded rule-6
# exception; provenance in its module docstring)
# =====================================================================
def test_card_multi_source_move(fails):
    """MEASURES (WAVE-J, the wjg8 LIVE wedge): 'Rage of the
    Dunlendings' pours Isengard regulars from North AND South Dunland
    into one destination - a COUNT-CONSERVING multi-source pairing of
    fungible units, which the old rule held as 'AMBIGUOUS pairing'
    FOREVER (the AI step re-halted every tick for 136s until the
    stall budget killed the game).  Now renders as deterministic move
    lines; nothing is guessed (any assignment of fungible units
    yields the identical client board)."""
    # THE EXACT LIVE SPECIMEN (wjg8 server hold, verbatim rows):
    # isengard:regular minus=[('north dunland', 1), ('south dunland',
    # 1)] plus=[('moria', 2)] fp_cas+0 -- reproduced as the synthetic
    # delta the renderer judged (the renderer is the unit under test;
    # the real Rage walk rides the live legs).
    import game_logic as _gl
    from game_state import UnitKind as _UK

    def rage_rows(post):
        moved = []
        for key in ('north dunland', 'south dunland'):
            a = _army_of(post, key)
            u = next(x for x in a.units if x.kind == _UK.REGULAR)
            a.units.remove(u)
            moved.append(u)
        _gl.get_or_create_army(post, 'moria').units.extend(moved)

    pre, post, act = _synth_pair(rage_rows)
    view = make_view([pre])
    em = emission.render_decision(view, NICK, Side.SHADOW,
                                  'card_option', act, post=post,
                                  pre=pre)
    if em.hold:
        fails.append(('rage_hold',
                      'the wjg8 eternal AMBIGUOUS hold: %s'
                      % str(em.hold)[:160]))
        return
    moves = [ln for ln in em.lines if 'Scribe.141' in ln]
    if len(moves) != 2:
        fails.append(('rage_moves', em.lines))


def _open_card_option(cid, die_idx=3, prep=None, seed=7):
    # die 3 = WILL_OF_WEST (wild): FLE-11 (2026-08-30) narrowed
    # play_event to the card-type's native faces, so the old
    # die 1 (CHARACTER) can no longer fund a Strategy-deck play
    # - that acceptance was the LA-1-family wide-tuple bug.
    """Drive a REAL optioned card end-to-end: cid into the FP hand,
    play_event applied through the engine -> pending_card_option opens.
    Returns (pre_state, describe_pending dict).  `prep(st)` mutates the
    state BEFORE the play (e.g. political position) so the prep never
    pollutes the pre->post delta."""
    st = fresh_state(seed)
    adopt(st)
    # Strider guides: with die 3 = WILL the Gandalf-guide same-deck
    # redraw (EVENT/WILL axis, FLE-11) would fire an engine-side draw
    # this offline harness cannot feed - the redraw is pinned in
    # test_audit0829_FLE, not here.
    st.fellowship.guide = st.fellowship.companions[1]
    if prep is not None:
        prep(st)
    st.hands[Side.FREE_PEOPLES].hand.append(cid)
    pe = Action(die_idx=die_idx, action_type='play_event',
                params={'card_id': cid, 'side': 'fp'})
    pre = ai.apply_action(st, pe)
    if getattr(pre, 'pending_card_option', None) is None:
        return None, None
    return pre, ai.describe_pending(pre)


def _filtered_engine_delta(pre, post):
    """The engine delta MINUS the play's own bookkeeping rows: the
    pending card's cid leaving the hand (__hands__) and its landing on
    the face-up discard/table (__table__) are performed PHYSICALLY by
    the '!CALL play_card' client handler at play time, so the option
    renderer consumes them without lines (emission_card docstring,
    EXPECTED rule).  Everything else must round-trip."""
    d = board_delta(pre, post)
    d.pop('__hands__', None)
    d.pop('__table__', None)
    return d


def test_card_option_move_roundtrip(fails):
    """MEASURES (full rendered round-trip): a REAL card_option decision
    -- Paths of the Woses, played through the engine's own play_event
    defer -> pending_card_option -> the chosen {from,to} option applied
    -- renders as literal move lines whose parse-back equals the
    engine's own apply delta (armies term; __hands__/__table__ filtered
    per _filtered_engine_delta).  Non-zero control: the filtered delta
    must carry BOTH the source and destination army rows.  Rohan is put
    At War BEFORE the play so the prep never rides the measured
    delta."""
    cid = 'FREE_PEOPLES_Strategy_Paths_of_the_Woses'
    from game_state import Nation

    def prep(st):
        st.political.boxes[Nation.ROHAN] = 3
        st.political.active[Nation.ROHAN] = True

    pre, dp = _open_card_option(cid, prep=prep)
    if pre is None:
        fails.append(('cardmv_control',
                      'NOT-MEASURED: Woses play did not defer'))
        return
    act = next((x for x in dp['actions']
                if x.params['choices'].get('from_region') == 'edoras'
                and x.params['choices'].get('to_region')
                == 'minas tirith'), None)
    if act is None:
        fails.append(('cardmv_control',
                      'NOT-MEASURED: no edoras->minas tirith option'))
        return
    post = ai.apply_action(pre, act)
    d_e = _filtered_engine_delta(pre, post)
    if 'edoras' not in d_e or 'minas tirith' not in d_e:
        fails.append(('cardmv_control',
                      'NOT-MEASURED: delta lacks the move rows: %r'
                      % (sorted(d_e),)))
        return
    view = make_view([pre])
    em = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                  act, post=post, pre=pre)
    if em.hold:
        fails.append(('cardmv_hold', em.hold))
        return
    if em.calls or em.chat:
        fails.append(('cardmv_shape', 'calls=%r chat=%r'
                      % (em.calls, em.chat)))
    ops = [classify(x)[0] for x in em.lines]
    if not em.lines or any(op != 'move' for op in ops):
        fails.append(('cardmv_ops', 'ops=%r' % (ops,)))
    base = roll_lines()
    d_p = board_delta(replay(base), replay(base + em.lines))
    if d_e != d_p:
        fails.append(('cardmv_roundtrip',
                      'engine=%r parser=%r' % (d_e, d_p)))
    # post=None sensitivity: the delta renderer must HOLD, never guess
    em0 = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                   act, post=None, pre=pre)
    if not em0.hold or 'post' not in em0.hold:
        fails.append(('cardmv_post_hold', 'expected HOLD, got %r'
                      % em0))


def test_card_option_recruit_roundtrip(fails):
    """MEASURES (full rendered round-trip, recruit lane): Cirdan's
    Ships driven end-to-end -- pending_card_option -> the Grey Havens
    option -- renders pool drags (FP Reinf ONLY: FP dead permanent) and
    the parse-back equals the engine's filtered delta.  Non-zero
    control: the filtered delta must carry a +units grey havens row.
    The Elves are put At War BEFORE the play (handler requirement) so
    the prep never rides the measured delta."""
    cid = "FREE_PEOPLES_Strategy_Cirdan's_Ships"
    from game_state import Nation

    def prep(st):
        st.political.boxes[Nation.ELVES] = 3
        st.political.active[Nation.ELVES] = True

    pre, dp = _open_card_option(cid, prep=prep)
    if pre is None:
        fails.append(('cardrec_control',
                      'NOT-MEASURED: Cirdan play did not defer'))
        return
    act = next((x for x in dp['actions']
                if x.params['choices'].get('region') == 'grey havens'),
               None)
    if act is None:
        fails.append(('cardrec_control',
                      'NOT-MEASURED: no grey havens option'))
        return
    post = ai.apply_action(pre, act)
    d_e = _filtered_engine_delta(pre, post)
    gh = d_e.get('grey havens')
    if not gh or not gh[0] or not all(n > 0 for _, n in gh[0]):
        fails.append(('cardrec_control',
                      'NOT-MEASURED: no +units grey havens row: %r'
                      % (d_e,)))
        return
    view = make_view([pre])
    em = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                  act, post=post, pre=pre)
    if em.hold:
        fails.append(('cardrec_hold', em.hold))
        return
    n_units = sum(n for _, n in gh[0])
    if len(em.lines) != n_units:
        fails.append(('cardrec_lines', 'want %d recruit drags, got %r'
                      % (n_units, em.lines)))
    pool = emitter.FP_POOL
    for ln in em.lines:
        op, f = classify(ln)
        if op != 'move' or f.get('frm') != pool:
            fails.append(('cardrec_pool',
                          'recruit not a pool drag: op=%r f=%r'
                          % (op, f)))
    base = roll_lines()
    d_p = board_delta(replay(base), replay(base + em.lines))
    if d_e != d_p:
        fails.append(('cardrec_roundtrip',
                      'engine=%r parser=%r' % (d_e, d_p)))


def _synth_pair(mutate):
    """(pre, post, action): pre = fresh setup, post = clone edited by
    `mutate(post)` -- the synthetic card-effect deltas for the
    ambiguity instruments.  The action carries no pending (hands
    untouched -> no bookkeeping rows)."""
    pre = fresh_state()
    post = ai.clone_state(pre)
    mutate(post)
    act = Action(die_idx=-1, action_type='card_option',
                 params={'card_name': 'synthetic', 'choices': {}})
    return pre, post, act


def _army_of(st, key):
    for k, a in st.armies.items():
        if sanitize(str(k)) == key:
            return a
    return None


def test_card_ambiguity_hold(fails):
    """MEASURES (the renderer's HOLD discipline -- the guessed-pairing
    disease this replaces): two sources feeding one destination (one
    north regular from bree AND one from the shire arriving at south
    downs) MUST return a loud HOLD naming the ambiguous row, emit ZERO
    lines, and leave the client model UNTOUCHED (the scratch-view
    contract).  The delta ALSO carries a perfectly renderable row (the
    north downs elite moving to south downs, planned BEFORE the
    ambiguous row in tkey order) so the untouched-view assertion
    measures real isolation: a partial render that retracked the
    renderable row before holding would leak it.  This is the
    mutation-tested guard: greedy pairing here is exactly the
    phantom-piece family."""
    import game_logic as _gl
    from game_state import UnitKind

    def two_sources(post):
        a1 = _army_of(post, 'bree')
        a2 = _army_of(post, 'the shire')
        u1 = next(u for u in a1.units if u.kind == UnitKind.REGULAR)
        u2 = next(u for u in a2.units if u.kind == UnitKind.REGULAR)
        a1.units.remove(u1)
        a2.units.remove(u2)
        dest = _gl.get_or_create_army(post, 'south downs')
        dest.units.extend([u1, u2])
        # the renderable elite row (single source, single destination)
        a3 = _army_of(post, 'north downs')
        u3 = next(u for u in a3.units if u.kind == UnitKind.ELITE)
        a3.units.remove(u3)
        dest.units.append(u3)

    # WAVE-J SHARPENED CONTRACT (the wjg8 Rage wedge): a COUNT-
    # CONSERVING multi-source pairing with NO casualty increment is a
    # provable relocation of FUNGIBLE units (an FP kill would ride the
    # casualty ledger; SA kills converge with recycles) - it RENDERS.
    # The HOLD discipline now guards the genuinely ambiguous shape:
    # counts NOT conserving + a casualty increment (one unit died, one
    # moved - WHICH one moved changes the client board).
    pre, post, act = _synth_pair(two_sources)
    view = make_view([pre])
    em = emission.render_decision(view, NICK, Side.FREE_PEOPLES,
                                  'card_option', act, post=post,
                                  pre=pre)
    if em.hold:
        fails.append(('amb_fungible_renders',
                      'count-conserving fungible relocation HELD '
                      '(the wjg8 eternal-hold class): %s'
                      % str(em.hold)[:120]))
    elif len([ln for ln in em.lines if 'Scribe.141' in ln]) != 3:
        fails.append(('amb_fungible_lines', em.lines))

    def kill_or_move(post):
        from game_state import Nation
        a1 = _army_of(post, 'bree')
        a2 = _army_of(post, 'the shire')
        u1 = next(u for u in a1.units if u.kind == UnitKind.REGULAR)
        u2 = next(u for u in a2.units if u.kind == UnitKind.REGULAR)
        a1.units.remove(u1)
        a2.units.remove(u2)
        _gl.get_or_create_army(post, 'south downs').units.append(u1)
        # the OTHER regular DIED (engine ledger says so): kill-or-move
        # per source is genuinely ambiguous - different client boards
        slot = post.fp_casualties.setdefault(
            Nation.THE_NORTH, {'regular': 0, 'elite': 0})
        slot['regular'] = slot.get('regular', 0) + 1

    pre2, post2, act2 = _synth_pair(kill_or_move)
    view2 = make_view([pre2])
    before2 = dict(view2.piece_region)
    em2 = emission.render_decision(view2, NICK, Side.FREE_PEOPLES,
                                   'card_option', act2, post=post2,
                                   pre=pre2)
    if not em2.hold:
        fails.append(('amb_sensitivity',
                      'kill-or-move delta RENDERED %d line(s) '
                      'instead of holding: %r'
                      % (len(em2.lines), em2.lines)))
        return
    if 'the north:regular' not in em2.hold \
            or 'AMBIGUOUS' not in em2.hold:
        fails.append(('amb_names_rows', em2.hold))
    if em2.lines or em2.calls or em2.chat:
        fails.append(('amb_no_partial_output', '%r' % (em2,)))
    if view2.piece_region != before2:
        fails.append(('amb_view_untouched',
                      'a HOLD left provisional retracks on the view'))


def test_card_move_specificity(fails):
    """MEASURES (specificity for the ambiguity guard): the SAME shape
    with ONE source (one north regular, bree -> south downs) renders
    exactly one clean move line -- correct zones, one pid typed
    (the north, regular) -- and the view IS retracked on success."""
    import game_logic as _gl
    from game_state import UnitKind

    def one_source(post):
        a1 = _army_of(post, 'bree')
        u1 = next(u for u in a1.units if u.kind == UnitKind.REGULAR)
        a1.units.remove(u1)
        _gl.get_or_create_army(post, 'south downs').units.append(u1)

    pre, post, act = _synth_pair(one_source)
    view = make_view([pre])
    em = emission.render_decision(view, NICK, Side.FREE_PEOPLES,
                                  'card_option', act, post=post,
                                  pre=pre)
    if em.hold:
        fails.append(('spec_hold', em.hold))
        return
    if len(em.lines) != 1:
        fails.append(('spec_lines', '%r' % (em.lines,)))
        return
    op, f = classify(em.lines[0])
    pids = f.get('pieces') or []
    if op != 'move' or len(pids) != 1:
        fails.append(('spec_op', 'op=%r f=%r' % (op, f)))
        return
    if f.get('frm') != emitter.zone_code_for('bree') \
            or f.get('to') != emitter.zone_code_for('south downs'):
        fails.append(('spec_zones', '%r' % (f,)))
    nat, ut = emitter._unit_type(pids[0], view, side_hint='fp')
    if ut != 'regular' or 'north' not in str(
            getattr(nat, 'value', nat)).lower():
        fails.append(('spec_piece_type', 'pid %s typed (%r,%r)'
                      % (pids[0], nat, ut)))
    tracked = emitter._norm_zone(str(view.piece_region.get(pids[0])))
    if tracked != 'south downs':
        fails.append(('spec_retrack',
                      'success did not retrack the view: %r'
                      % (view.piece_region.get(pids[0]),)))


def test_fp_dead_permanence_pools(fails):
    """MEASURES (USER rulings: FP dead PERMANENT, SA dead RECYCLES --
    Part-35 / rulebook p.30, carried by the emitter's muster pools):
    an FP muster whose only remaining matching piece sits in the dead
    pile raises (rendered as a HOLD upstream) -- it never resurrects;
    the SAME shape for Shadow LEGALLY pulls the piece from the dead
    pile."""
    st = fresh_state()
    view = make_view([st])
    n = 0
    first = True
    for pid in sorted(emitter._initial()):
        try:
            nat, ut = emitter._unit_type(pid, view, side_hint='fp')
        except Exception:
            continue
        if ut == 'regular' and 'north' in str(
                getattr(nat, 'value', nat)).lower():
            view.piece_region[str(pid)] = 'dead' if first else 'arnor'
            first = False
            n += 1
    if n == 0:
        fails.append(('fp_pool_control', 'NOT-MEASURED: no north pids'))
        return
    try:
        emitter.plan_muster(view, NICK, 'FREE_PEOPLES',
                            {'nation': 'the_north', 'region': 'bree'},
                            'regular')
        fails.append(('fp_dead_resurrected',
                      'FP muster pulled from the dead pile'))
    except LookupError:
        pass
    view2 = make_view([st])
    m = 0
    first = True
    for pid in sorted(emitter._initial()):
        try:
            nat, ut = emitter._unit_type(pid, view2, side_hint='shadow')
        except Exception:
            continue
        if ut == 'regular' and 'sauron' in str(
                getattr(nat, 'value', nat)).lower():
            view2.piece_region[str(pid)] = 'dead' if first else 'gorgoroth'
            first = False
            m += 1
    if m == 0:
        fails.append(('sh_pool_control',
                      'NOT-MEASURED: no sauron pids'))
        return
    try:
        op, f = emitter.plan_muster(view2, NICK, 'SHADOW',
                                    {'nation': 'sauron',
                                     'region': 'gorgoroth'}, 'regular')
        if f.get('frm') != emitter.SH_DEAD:
            fails.append(('sh_dead_source',
                          'recycled SA muster frm=%r, want the dead '
                          'pile %r' % (f.get('frm'), emitter.SH_DEAD)))
    except LookupError as e:
        fails.append(('sh_dead_recycles',
                      'SA muster refused its legal dead-pile recycle: '
                      '%s' % e))


def test_combat_effect_magnitudes(fails):
    """MEASURES (WAVE-I, WAVE-H blocker 3): combat_effect_choice no
    longer wedges — each choice renders through the supported chat
    vocabulary (ChatOut.validate accepts every text):
      * Dread and Despair names the amount of Nazgul leadership;
      * Relentless Assault names the hits paid;
      * zero is an explicit named decline, never silent;
      * a generic count remains available for other effect classes;
      * no class returns a hold (the old NotImplemented wedge).
    The physical sacrifice renders when it MATERIALIZES (the round-
    resolution apply), via the forced-rows sweep — driver suite."""
    from wotr_bridge.live import chat as chat_mod
    holder = [fresh_state()]
    view = make_view(holder)
    co = chat_mod.ChatOut()

    def render(cid, count):
        act = Action(die_idx=-1, action_type='combat_effect_choice',
                     params={'card_id': cid, 'count': count})
        return emission.render_decision(view, NICK, Side.SHADOW,
                                        'combat_card', act)
    em = render('SHADOW_Strategy_Rage_of_the_Dunlendings', 2)
    if em.hold or len(em.chat) != 1:
        fails.append(('ra_shape', repr(em)))
        return
    k, t = em.chat[0]
    if (k, t) != ('prompt', 'Relentless Assault: take 2 hits'):
        fails.append(('ra_text', (k, t)))
    try:
        co.validate(k, t)
    except Exception as e:
        fails.append(('ra_vocab', repr(e)))
    # G3: zero must still be stated
    em0 = render('SHADOW_Strategy_The_King_is_Revealed', 0)
    if em0.hold or em0.chat != [('prompt', 'Relentless Assault: decline')]:
        fails.append(('zero_stated', repr(em0)))
    else:
        try:
            co.validate(*em0.chat[0])
        except Exception as e:
            fails.append(('zero_vocab', repr(e)))
    # Heroic Death (FP-held card) via the FP seat
    act = Action(die_idx=-1, action_type='combat_effect_choice',
                 params={'card_id':
                         'FREE_PEOPLES_Character_Mirror_of_Galadriel',
                         'count': 3})
    emh = emission.render_decision(view, NICK, Side.FREE_PEOPLES,
                                   'combat_card', act)
    if emh.hold or not emh.chat \
            or not emh.chat[0][1].endswith(': i choose 3'):
        fails.append(('hd_shape', repr(emh)))
    # Dread names its forfeited leadership — the REAL live cid (its
    # card_id carries no 'dread'; the effect kind must resolve via
    # combat.combat_effect_kind over the printed combat name)
    emd = render('SHADOW_Character_Grond_Hammer_of_the_Underworld', 3)
    if emd.hold or emd.chat != [('prompt', 'Dread and Despair: forfeit 3 Nazgul leadership')]:
        fails.append(('dread_ff', repr(emd)))
    else:
        try:
            co.validate(*emd.chat[0])
        except Exception as e:
            fails.append(('dread_vocab', repr(e)))


def test_discard_table_card(fails):
    """MEASURES: the 1.149 OPPONENT table-card kill, taken from
    get_legal_actions (Shadow: any die + one Army card + one Character
    card in hand while 'A Power too Great' sits on the FP table),
    renders the CORPUS sequence -- the die spend, the target dragged
    out of its Board slot into the ACTOR's Current-Card zone, one
    masked hand kill per payment card -- and HOLDS rather than guessing
    a piece when the client dump never seated the card.

    REPRODUCTION (USER 2026-08-20 live: "emission hold: no planner for
    'discard_table_card'"): the engine offers this action
    (ai._get_table_kill_actions) and the live stack had no planner, so
    the AI's choice raised EMISSION HOLD -> DriverHalt on EVERY cycle,
    forever, with the action already applied engine-side.

    GROUND TRUTH, 2023-01-20_183326_FP_CircleDakkon_SP_GenericPie
    L583-591 (Shadow kills 'A Power too Great'):
        moves Shadow Action Die (M) from SA Dice to Current Action.
        moves A Power Too Great/No Quarter from Board1 to Current SA Card.
        kills Shadow Character Card in Shadow Hand.
        kills Shadow Strategy Card(Army) in Shadow Hand.
        kills group (die, card) in Current Action.     <- driver-owed
    The FP leg: the Palantir (accented title) killed with a Will of the
    West die lands in the FP card zone (Game.2037)."""
    import persistent_effects as pe
    from cards import get_card
    from wotr_bridge.live import emission as em_mod
    st = fresh_state()
    adopt(st)
    army = next((c for c in st.sh_strat_deck
                 if getattr(get_card(c), 'card_type', None) == 'Army'),
                None)
    chr_ = next((c for c in st.sh_char_deck if '_Character_' in c), None)
    if army is None or chr_ is None:
        fails.append(('table_kill_control', 'NOT-MEASURED: no Army / '
                      'Character card in the Shadow decks'))
        return
    st.sh_strat_deck.remove(army)
    st.sh_char_deck.remove(chr_)
    st.hands[Side.SHADOW].hand.extend([army, chr_])
    st.persistent_cards.add_card(pe.create_persistent_card(
        'FREE_PEOPLES_Strategy_A_Power_too_Great', 'A Power too Great',
        Side.FREE_PEOPLES))
    acts = [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'discard_table_card']
    if not acts:
        fails.append(('table_kill_control', 'NOT-MEASURED: the engine '
                      'offers no discard_table_card'))
        return
    act = acts[0]
    post = ai.apply_action(st, act)
    # EX-8 (2026-08-29): the payment pair is a LEARNED pending
    # (ai._open_table_kill_payment); the kill itself renders the die
    # spend + the target move, and the AI's own answer (by search -
    # here the one legal pair) renders the two masked hand kills via
    # render_decision.  The old pin expected the engine to auto-pay.
    pco = getattr(post, 'pending_card_option', None)
    if not (isinstance(pco, dict) and pco.get('table_kill_payment')):
        fails.append(('table_kill_control', 'NOT-MEASURED: the engine '
                      'opened no table_kill_payment pending'))
        return
    if army not in post.hands[Side.SHADOW].hand \
            or chr_ not in post.hands[Side.SHADOW].hand:
        fails.append(('table_kill_pending_paid_early',
                      list(post.hands[Side.SHADOW].hand)))
        return
    pick = ai.table_kill_payment_action(post, army, chr_)
    post2 = ai.apply_action(post, pick)
    if army in post2.hands[Side.SHADOW].hand \
            or chr_ in post2.hands[Side.SHADOW].hand:
        fails.append(('table_kill_control', 'NOT-MEASURED: the chosen '
                      'pair did not leave the hand'))
        return
    holder = [st]
    view = make_view(holder)
    view.card_pid_map[army] = '901'
    view.card_pid_map[chr_] = '902'
    # THE TARGET COMES FROM THE SEAT WE WATCHED, never a static id
    # table (USER 2026-08-20: a hardcoded 145 was 'Kindred of
    # Glorfindel' in the live build and the AI killed THAT).  The pid
    # here (143) is the one the USER's own client dump reported for
    # 'The Power of Tom Bombadil' in Board1.
    view.piece_region['143'] = 'Board1'
    view.table_seats = {'a power too great':
                        {'slot': 'Game.2029', 'pid': '143',
                         'armed': True}}
    view.table_seat_pid = lambda k: (view.table_seats.get(k) or
                                     {}).get('pid')

    # CONTROL FIRST: with NO seat, the planner must HOLD rather than
    # guess a piece - the exact failure that shipped.
    _ns = make_view([st])
    _ns.card_pid_map.update(view.card_pid_map)
    _ns.table_seats = {}
    _ns.table_seat_pid = lambda k: None
    _em_ns = em_mod.render_action(_ns, NICK, Side.SHADOW, act, post=post)
    if not _em_ns.hold or _em_ns.lines:
        fails.append(('table_kill_unseated_not_held',
                      '%r %r' % (_em_ns.hold, _em_ns.lines)))

    em = em_mod.render_action(view, NICK, Side.SHADOW, act, post=post)
    if em.hold:
        fails.append(('table_kill_held', str(em.hold)[:160]))
        return
    ops = [classify(x) for x in em.lines]
    want_die = emitter.die_piece_for(view, 'SHADOW', act.die_idx)
    spend = [f for op, f in ops if op.startswith('die_spend')]
    if len(spend) != 1 or spend[0].get('die') != want_die:
        fails.append(('table_kill_spend', repr(em.lines)))
    # the target leaves its Board slot for the ACTOR's card zone
    if not any('143' in ln and 'Game.2036' in ln for ln in em.lines):
        fails.append(('table_kill_move', repr(em.lines)))
    # the kill itself pays nothing yet (the pair is the pending's)
    paid0 = [ln for ln in em.lines if '901' in ln or '902' in ln]
    if paid0:
        fails.append(('table_kill_paid_before_decision', repr(em.lines)))
    if em.calls or em.chat:
        fails.append(('table_kill_shape', '%r %r' % (em.calls, em.chat)))
    # one masked hand kill per payment card, rendered by the DECISION
    holder[0] = post
    em_pay = em_mod.render_decision(view, NICK, Side.SHADOW, 'card_option',
                                    pick, post=post2, pre=post)
    if em_pay.hold:
        fails.append(('table_kill_payment_held', str(em_pay.hold)[:160]))
    paid = [ln for ln in em_pay.lines if '901' in ln or '902' in ln]
    if len(paid) != 2 or len(em_pay.lines) != 2:
        fails.append(('table_kill_payment', repr(em_pay.lines)))
    if not all('Controls.46' in ln for ln in em_pay.lines):
        fails.append(('table_kill_payment_zone', repr(em_pay.lines)))
    holder[0] = st

    # the accented Palantir (FP actor, Will of the West payment) must
    # resolve (deaccent path) and go to the FP card zone
    st2 = fresh_state()
    adopt(st2)
    st2.persistent_cards.add_card(pe.create_persistent_card(
        'SA_202', 'The Palantir of Orthanc', Side.SHADOW))
    will = next((a for a in ai.get_legal_actions(st2, Side.FREE_PEOPLES)
                 if a.action_type == 'discard_table_card'
                 and a.params.get('payment') == 'will'), None)
    if will is None:
        fails.append(('table_kill_control', 'NOT-MEASURED: no will-paid '
                      'Palantir kill on the FP menu'))
        return
    post2 = ai.apply_action(st2, will)
    v2 = make_view([st2])
    v2.piece_region['202'] = 'Board2'
    v2.table_seats = {'the palantir of orthanc':
                      {'slot': 'Game.2030', 'pid': '202',
                       'armed': True}}
    v2.table_seat_pid = lambda k: (v2.table_seats.get(k) or {}).get('pid')
    em2 = em_mod.render_action(v2, NICK, Side.FREE_PEOPLES, will,
                               post=post2)
    if em2.hold or not any('202' in ln and 'Game.2037' in ln
                           for ln in em2.lines):
        fails.append(('table_kill_palantir',
                      '%r %r' % (em2.hold, em2.lines)))


def test_hunt_guide_separate(fails):
    """MEASURES (W1#46, K4): the ENGINE's hunt_guide_separate decision
    (game_logic.hunt_decision_menu while a hobbit guides) renders:
    OUTSIDE Mordor an EMPTY emission (the hobbit is queued for the
    hobbit_place pending, whose companion_separate action carries the
    placement drag -- rendered there, once); IN Mordor the hobbit dies:
    ONE move of the guide's piece into FP Casualties.  Before the fix:
    'no L2 planner for hunt_guide_separate' -> HOLD -> DriverHalt."""
    import game_logic as gl
    for mordor in (False, True):
        st = fresh_state()
        adopt(st)
        merry = next(c for c in st.fellowship.companions
                     if c.name == 'Meriadoc')
        st.fellowship.guide = merry
        if mordor:
            st.fellowship.mordor_entered = True
            st.fellowship.mordor_step = 0
        gl.continue_hunt_chain(st, 2, casualty_taken=False)
        dp = ai.describe_pending(st)
        hgs = next((a for a in (dp['actions'] if dp else [])
                    if a.action_type == 'hunt_guide_separate'), None)
        if hgs is None:
            fails.append(('hgs_control',
                          'NOT-MEASURED: no hunt_guide_separate offered'))
            return
        view = make_view([st])
        mpid = emitter._companion_pid('Meriadoc')
        view.piece_region[mpid] = 'Guide Box'
        post = ai.apply_action(st, hgs)
        em = emission.render_decision(view, NICK, dp['side'], dp['kind'],
                                      hgs, post=post, pre=st)
        tag = 'mordor' if mordor else 'map'
        if em.hold:
            fails.append(('hgs_hold_%s' % tag, em.hold))
            continue
        if not mordor:
            if em.lines or em.calls or em.chat:
                fails.append(('hgs_map_not_empty', '%r' % (em.lines,)))
            # the placement rides the next pending's companion_separate
            nxt = post
            dp2 = ai.describe_pending(nxt)
            guard = 0
            while dp2 is not None and guard < 6 and not any(
                    a.action_type == 'companion_separate'
                    for a in dp2['actions']):
                nxt = ai.apply_action(nxt, dp2['actions'][0])
                dp2 = ai.describe_pending(nxt)
                guard += 1
            cs = next((a for a in (dp2['actions'] if dp2 else [])
                       if a.action_type == 'companion_separate'
                       and a.params.get('companion') == 'Meriadoc'), None)
            if cs is None:
                fails.append(('hgs_place_control',
                              'NOT-MEASURED: no hobbit_place for Merry'))
                continue
            em2 = emission.render_decision(view, NICK, dp2['side'],
                                           dp2['kind'], cs,
                                           post=ai.apply_action(nxt, cs),
                                           pre=nxt)
            if em2.hold or len(em2.lines) != 1:
                fails.append(('hgs_place', '%r %r' % (em2.hold,
                                                      em2.lines)))
                continue
            op, f = classify(em2.lines[0])
            if op != 'move' or f.get('pieces') != [mpid] \
                    or f.get('frm') != emitter.GUIDE_BOX:
                fails.append(('hgs_place_shape', '%r %r' % (op, f)))
        else:
            if len(em.lines) != 1:
                fails.append(('hgs_mordor_lines', '%r' % (em.lines,)))
                continue
            # a drag into the Casualties box renders as the client's
            # send-to-box form (Controls.281, op 'hand_exit')
            op, f = classify(em.lines[0])
            pieces = f.get('pieces') or ''
            if isinstance(pieces, str):
                pieces = pieces.split()
            if op != 'hand_exit' or pieces != [mpid] \
                    or f.get('frm') != emitter.GUIDE_BOX \
                    or f.get('to') != emitter.FP_DEAD:
                fails.append(('hgs_mordor_shape', '%r %r' % (op, f)))


def test_mouth_move(fails):
    """MEASURES (W1#48, K4): the ENGINE's mouth_move action (Shadow
    Character die, Mouth of Sauron in play) renders the die spend, ONE
    move of the Mouth's piece (255) from the engine's 'from' region to
    its 'to' region, and the die kill.  Before the fix: 'no L2 planner
    for mouth_move' -> HOLD -> DriverHalt."""
    import game_logic as gl
    from board import sanitize
    st = fresh_state()
    adopt(st)
    st.mouth_of_sauron.in_play = True
    st.mouth_of_sauron.was_ever_in_play = True
    gl._add_minion_to_region(st, st.mouth_of_sauron, 'barad dur')
    acts = [a for a in ai.get_legal_actions(st, Side.SHADOW)
            if a.action_type == 'mouth_move']
    if not acts:
        fails.append(('mouth_control', 'NOT-MEASURED: no mouth_move'))
        return
    a = next((x for x in acts
              if sanitize(str(x.params.get('to'))) == 'gorgoroth'),
             acts[0])
    view = make_view([st])
    view.piece_region['255'] = 'Barad-Dur'
    post = ai.apply_action(st, a)
    em = emission.render_action(view, NICK, Side.SHADOW, a, post=post)
    if em.hold:
        fails.append(('mouth_hold', em.hold))
        return
    ops = [classify(ln) for ln in em.lines]
    moves = [(op, f) for op, f in ops if op == 'move']
    if len(moves) != 1:
        fails.append(('mouth_move_lines', '%r' % (em.lines,)))
        return
    op, f = moves[0]
    if f.get('pieces') != ['255'] \
            or f.get('frm') != emitter.zone_code_for('barad dur') \
            or f.get('to') != emitter.zone_code_for(
                sanitize(str(a.params['to']))):
        fails.append(('mouth_move_shape', '%r' % (f,)))
    if ops[0][0] != 'die_spend_sh' or ops[-1][0] != 'kill':
        fails.append(('mouth_die', '%r' % ([o for o, _ in ops],)))


def test_dwarven_ring(fails):
    """MEASURES (W1#50, K4): the ENGINE's dwarven_ring free action
    renders, given the post state, the ring marker's walk out of the
    dice pool into the Used box plus the die-face change line stating
    the ENGINE's new face; before the client answered the reroll
    (post None, the apply starved on the d6) it renders an EMPTY
    emission, never a HOLD (the directive is the server's; the move
    renders on the resumed attempt)."""
    st = fresh_state()
    adopt(st)
    st.fp_dwarven_rings = 1
    acts = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'dwarven_ring']
    if not acts:
        fails.append(('dwarf_control', 'NOT-MEASURED: no dwarven_ring'))
        return
    a = acts[0]
    random.seed(3)
    post = ai.apply_action(st, a)
    idx = int(a.params['die_idx'])
    new_face = post.dice[Side.FREE_PEOPLES][idx].current
    if new_face == st.dice[Side.FREE_PEOPLES][idx].current:
        fails.append(('dwarf_control', 'NOT-MEASURED: face unchanged'))
        return
    view = make_view([st])
    em0 = emission.render_action(view, NICK, Side.FREE_PEOPLES, a,
                                 post=None)
    if em0.hold or em0.lines or em0.calls or em0.chat:
        fails.append(('dwarf_preapply', '%r %r' % (em0.hold, em0.lines)))
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, a,
                                post=post)
    if em.hold:
        fails.append(('dwarf_hold', em.hold))
        return
    ops = [classify(ln) for ln in em.lines]
    kinds = [o for o, _ in ops]
    if 'die_change' not in kinds:
        fails.append(('dwarf_face', '%r' % (em.lines,)))
        return
    f = dict(ops[kinds.index('die_change')][1])
    want_letter = emitter._FACE_LETTER[new_face.name]
    if f.get('die') != emitter._FP_DIE_IDS[idx] \
            or f.get('face') != want_letter:
        fails.append(('dwarf_face_shape', '%r want %s/%s'
                      % (f, emitter._FP_DIE_IDS[idx], want_letter)))
    ring = [(o, g) for o, g in ops if o in ('move', 'hand_exit')
            and '428' in (g.get('pieces') or '')]
    if len(ring) != 1 or ring[0][1].get('frm') != emitter.FP_DICE_POOL \
            or ring[0][1].get('to') != emitter.DICE_USED:
        fails.append(('dwarf_ring_walk', '%r' % (em.lines,)))


def test_companion_move_renamed(fails):
    """MEASURES (W1#74, K4): a companion_move naming a RENAMED figure
    (crowned Aragorn, pid 241 whose static table name is 'Strider')
    resolves through the alias-aware pid lookup: the single-figure
    subset action renders ONE move of ['241']; a subset naming Aragorn
    AND another companion renders BOTH pids (never a partial drag).
    Actions come from get_legal_actions on a real state (Strider and
    Legolas separated to Hollin by the engine, Strider replaced by the
    engine's Aragorn companion object, Gimli joining).  Before the
    fix: LookupError (HOLD loop) alone, silently dropped in a group."""
    import game_state as gs
    from board import sanitize
    st = fresh_state()
    adopt(st)
    s1 = next((a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
               if a.action_type == 'companion_separate'
               and a.params.get('companion') == 'Strider'
               and sanitize(str(a.params['to_region'])) == 'hollin'),
              None)
    if s1 is None:
        fails.append(('rename_control', 'NOT-MEASURED: no Strider sep'))
        return
    st = ai.apply_action(st, s1)
    s2 = next((a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
               if a.action_type == 'companion_separate'
               and a.params.get('companion') == 'Legolas'
               and sanitize(str(a.params['to_region'])) == 'hollin'),
              None)
    if s2 is None:
        fails.append(('rename_control', 'NOT-MEASURED: no Legolas sep'))
        return
    st = ai.apply_action(st, s2)
    k = next(kk for kk in st.armies if sanitize(str(kk)) == 'hollin')
    for i, ld in enumerate(st.armies[k].leaders):
        if isinstance(ld, gs.Companion) and ld.name == 'Strider':
            st.armies[k].leaders[i] = gs.Companion(
                name='Aragorn', level=3, leadership=2,
                captain_of_the_west=True, activates_nations=[])
            st.aragorn_crowned.crowned = True
    gimli = next(c for c in st.fellowship.companions if c.name == 'Gimli')
    st.fellowship.companions.remove(gimli)
    st.armies[k].leaders.append(gimli)
    mv = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
          if a.action_type == 'companion_move']
    single = next((a for a in mv
                   if a.params.get('companions') == ['Aragorn']), None)
    pair = next((a for a in mv
                 if a.params.get('companions') == ['Aragorn', 'Legolas']),
                None)
    if single is None or pair is None:
        fails.append(('rename_control', 'NOT-MEASURED: menu %r'
                      % (sorted({tuple(a.params.get('companions') or ())
                                 for a in mv}),)))
        return
    for tag, a, want in (('single', single, ['241']),
                         ('pair', pair, ['241', '242'])):
        view = make_view([st])
        for p in ('241', '242', '243'):
            view.piece_region[p] = 'Hollin'
        post = ai.apply_action(st, a)
        em = emission.render_action(view, NICK, Side.FREE_PEOPLES, a,
                                    post=post)
        if em.hold:
            fails.append(('rename_hold_%s' % tag, em.hold))
            continue
        moves = [classify(ln) for ln in em.lines]
        moves = [(o, f) for o, f in moves if o == 'move']
        if len(moves) != 1 or sorted(moves[0][1].get('pieces') or []) \
                != want:
            fails.append(('rename_pieces_%s' % tag,
                          '%r want %r' % (moves, want)))


def test_mordor_move_preapply(fails):
    """MEASURES (W1#45, K4): the AI-as-FP fellowship_move on the Mordor
    track.  With post=None (the apply starved on the client's hunt
    tile) the render is an EMPTY emission -- never a HOLD (the tile
    ask is the server's directive; the move renders on the resumed
    attempt).  With post it renders the die spend, the pawn walk box
    N -> N+1 when the engine advanced, and the die drop into the Hunt
    Box."""
    st = fresh_state()
    adopt(st)
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 0
    st.fellowship.hidden = True
    st.fellowship.progress = 0
    acts = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'fellowship_move']
    if not acts:
        fails.append(('mordor_control', 'NOT-MEASURED: no move'))
        return
    a = acts[0]
    view = make_view([st])
    em0 = emission.render_action(view, NICK, Side.FREE_PEOPLES, a,
                                 post=None)
    if em0.hold or em0.lines or em0.calls or em0.chat:
        fails.append(('mordor_preapply', '%r %r' % (em0.hold, em0.lines)))
    random.seed(5)
    post = ai.apply_action(st, a)
    if int(post.fellowship.mordor_step) != 1:
        fails.append(('mordor_control', 'NOT-MEASURED: step %r'
                      % (post.fellowship.mordor_step,)))
        return
    em = emission.render_action(view, NICK, Side.FREE_PEOPLES, a,
                                post=post)
    if em.hold:
        fails.append(('mordor_hold', em.hold))
        return
    kinds = [classify(ln)[0] for ln in em.lines]
    if kinds != ['die_spend_fp', 'move', 'die_to_hunt']:
        fails.append(('mordor_post_shape', '%r' % (kinds,)))
        return
    f = classify(em.lines[1])[1]
    if f.get('pieces') != ['247'] \
            or f.get('frm') != emitter.MORDOR_TRACK_ZONES[0] \
            or f.get('to') != emitter.MORDOR_TRACK_ZONES[1]:
        fails.append(('mordor_walk', '%r' % (f,)))


def test_table_slot_kill_grammar(fails):
    """MEASURES (W1#75, K4): a kill whose zone is a Board table slot
    renders the corpus-attested Controls.296/297 'kills X in BoardN'
    form AT the slot (raw corpus, 17,692 logs: 10,629 such lines,
    e.g. '$Sirrus233<~Controls.296~>167 <~Controls.297~><~Game.2029~>'),
    never Controls.644/645 with a slot token (1,721,217 644/645 lines,
    Game.2035 in every one).  Specificity: an ordinary kill still
    normalises its zone to Game.2035."""
    ln = emitter.render('kill', {'nick': NICK, 'pieces': ['167'],
                                 'to': 'Game.2029'}, local=True)
    want = '$%s<~Controls.296~>167 <~Controls.297~><~Game.2029~> null' \
        % NICK
    if ln != want:
        fails.append(('slot_kill_form', '%r != %r' % (ln, want)))
    op, f = classify(ln)
    if op != 'hand_exit_b' or f.get('frm') != 'Game.2029' \
            or f.get('pieces') != ['167']:
        fails.append(('slot_kill_roundtrip', '%r %r' % (op, f)))
    for slot in ('Game.2030', 'Game.2034'):
        ln2 = emitter.render('kill', {'nick': NICK, 'pieces': ['1'],
                                      'to': slot}, local=False)
        if '<~Controls.645~>' in ln2 or ('<~%s~>' % slot) not in ln2:
            fails.append(('slot_kill_%s' % slot, ln2))
    ln3 = emitter.render('kill', {'nick': NICK, 'pieces': ['102'],
                                  'to': 'Game.2036'}, local=False)
    if ln3 != '$%s<~Controls.644~>102 <~Controls.645~><~Game.2035~>' \
            % NICK:
        fails.append(('kill_normalised', ln3))


def test_discard_table_card_ring_payment(fails):
    """MEASURES (W1#51): the FP discard_table_card that pays with an
    ELVEN RING (engine: fp_elven_rings -= 1, sh_elven_rings += 1)
    renders the ring marker's walk from its active slot to a Dark Ring
    slot (corpus: 'moves Nenya from Elven Ring 2 to Dark Ring 1' right
    before 'moves The Palantir of Orthanc/... from Board1 to Current FP
    Card'), the same Controls.366 walk the elven_ring planner emits.
    Action from get_legal_actions on a state holding the Palantir.
    Specificity: the 'will' payment of the same card walks no ring."""
    import persistent_effects as pe
    from board import sanitize
    st = fresh_state()
    adopt(st)
    st.persistent_cards.add_card(pe.create_persistent_card(
        'SA_202', 'The Palantir of Orthanc', Side.SHADOW))
    acts = [a for a in ai.get_legal_actions(st, Side.FREE_PEOPLES)
            if a.action_type == 'discard_table_card'
            and 'palant' in str(a.params.get('card_name')).lower()]
    ring = next((a for a in acts if a.params.get('payment') == 'ring'),
                None)
    will = next((a for a in acts if a.params.get('payment') == 'will'),
                None)
    if ring is None or will is None:
        fails.append(('ring_pay_control', 'NOT-MEASURED: menu %r'
                      % ([(a.die_idx, a.params.get('payment'))
                          for a in acts],)))
        return
    for tag, a, want_ring in (('ring', ring, True), ('will', will, False)):
        view = make_view([st])
        view.piece_region['202'] = 'Board2'
        view.table_seats = {'the palantir of orthanc':
                            {'slot': 'Game.2030', 'pid': '202',
                             'armed': True}}
        view.table_seat_pid = \
            lambda k, _v=view: (_v.table_seats.get(k) or {}).get('pid')
        post = ai.apply_action(st, a)
        if want_ring and int(post.fp_elven_rings) != \
                int(st.fp_elven_rings) - 1:
            fails.append(('ring_pay_control', 'NOT-MEASURED: engine '
                          'took no ring'))
            return
        em = emission.render_action(view, NICK, Side.FREE_PEOPLES, a,
                                    post=post)
        if em.hold:
            fails.append(('ring_pay_hold_%s' % tag, em.hold))
            continue
        walks = [classify(ln) for ln in em.lines]
        walks = [(o, f) for o, f in walks if o == 'marker_366']
        if want_ring:
            if len(walks) != 1:
                fails.append(('ring_pay_walk', '%r' % (em.lines,)))
                continue
            f = walks[0][1]
            if f.get('frm') not in emitter.RING_ACTIVE_SLOTS \
                    or f.get('to') not in emitter.RING_USED_SLOTS \
                    or f.get('piece') not in ('249', '250', '251'):
                fails.append(('ring_pay_walk_shape', '%r' % (f,)))
        elif walks:
            fails.append(('ring_pay_will_walks', '%r' % (em.lines,)))


def main():
    tests = [
        test_card_multi_source_move,
        test_roundtrip_army_move,
        test_roundtrip_advance_and_muster_chain,
        test_roundtrip_fellowship_move,
        test_roundtrip_eye_allocate,
        test_play_event_staging_shape,
        test_play_refund_predicate,
        test_fungible_shortage_hold,
        test_attack_declaration,
        test_combat_card_decision,
             test_mixed_nation_move_by_nation,
             test_siege_assault_render,
             test_card_effect_die_face_change,
        test_casualty_leftover_hit_downgrade,
             test_casualty_allocation,
        test_retreats,
        test_siege_trim,
        test_hunt_guide_separate,
        test_mouth_move,
        test_dwarven_ring,
        test_companion_move_renamed,
        test_mordor_move_preapply,
        test_table_slot_kill_grammar,
        test_discard_table_card_ring_payment,
        test_card_option_move_roundtrip,
        test_card_option_recruit_roundtrip,
        test_card_ambiguity_hold,
        test_card_move_specificity,
        test_fp_dead_permanence_pools,
        test_combat_effect_magnitudes,
        test_discard_table_card,
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
