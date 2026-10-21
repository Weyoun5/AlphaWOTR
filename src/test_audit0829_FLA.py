"""Audit 2026-08-29/30 - FLA fix-wave pins (par5 fleet Families A, B, D
+ the coordinator's FLH add-ons).  Session evidence under
C:/Users/sarve/Downloads/WOTR_AI_training_lab/agent_seat/sessions/
{par5_g1..g5,seqack_fp1} and the temp client dirs named in the reports.

  FLA-1  (A2, G2#4/#7 + G5#7)  the chat-'siege' answer never demanded
         the garrison seat: post_commit_sweep's BL-76 lane compared
         str(Side.X).lower() ('side.free_peoples') against 'fp'/'sh'
         prefixes - it could NEVER fire, for either side.
  FLA-2  (A1, G5#2 + SEQVAL#10 family)  a companion SEPARATION leaves
         the Fellowship box (a named JOIN with no leave): the named
         lane iterated leaves only, so nothing was banked and the
         census aged it into a divergence with 'owed: none'.
  FLA-3  (A3/A4a, G3#6)  a residual named relocation (1 leave + 1
         join) banked ONLY under a 'chat:' label - the Return of the
         Witch-king play (window label) moved WK engine-side with no
         demand.  Residual rows are unexpressed by construction:
         bank under ANY label.  + (D5a, G3#4) a banked *_move-only
         sweep must not chat 'your destroyed pieces...'.
  FLA-4  (A6b, G5#5)  a record arriving while a segmenter WINDOW is
         open never reached _consume_mirror_echo - the echo stayed
         open a whole turn, froze the census, and the seat's
         re-perform doubled the piece.
  FLA-5  (A6a, G5#6 + G1#2)  the BL-101 arrival self-close never
         fired for ':leader' rows (_engine_tkey_count counted
         regular/elite only) - a leader the client already placed
         was banked again and re-performed (engine=1 client=2).
  FLA-6  (A4b, G3#5)  emitter._companion_pid knew only the 8
         companions - the seat's named_move/named echo performer
         raised LookupError for 'Witch King' and the owed drag was
         never auto-performed.
  FLA-7  (A5, G4#2)  an adoption ran while open owed: demands
         explained the rows and erased the engine's paid recruits;
         census rows printed 'owed: none' while demands existed.
  FLA-13 (B2, G2#6 + G1#3)  the hunt guide-sacrifice DRAG (move to
         FP Casualties) produced EMPTY window facts: _rec_inert
         dropped it, nothing buffered, the pending never closed -
         hunt menu picks executed but never committed.
  FLA-15 (B1, G1#10)  a bare number over a combat_effect_choice menu
         returned None when no count matched instead of falling
         through to the advertised say_pick index lane.
  FLA-16 (B4, G1#5 + G2#8)  the turn-start declare menu published
         n_total=null after the AI's own action cleared the engine's
         _declare_phase flag - status_menu now re-arms it.
  FLA-8  (D1i, G4#6)  the self-repair kill list swept EVERYTHING in
         Current Action / card slots - including combat cards staged
         two turns earlier.  Scoped to pieces first seen there this
         turn.
  FLA-18 (D1ii, G4#6)  a JOURNAL MISMATCH during a step replay was a
         sticky halt with a dead chat lane; it now abandons the step
         and degrades to the self-repair ASK; a step opened before a
         resync is abandoned, never replayed against the new world.
  FLA-19 (D1iii, G4#6)  chat 'reset' clears the sticky halt through
         the SAME panel lever (K3).
  FLA-9  (D3, G5#9)  the legality gate force-undid a fully-rendered
         gesture judged against a state a CONFIRMED divergence said
         was wrong ('Attack Lorien from Lorien').  While a confirmed
         divergence stands the gate holds + witnesses, never
         force-undoes.
  FLA-10 (D4, G3#6b)  the census 'no' says 'you move the pieces to
         match my engine' and _dieless_gesture then refused exactly
         those drags ('stage an action die first') and undid them.
  FLA-12 (D6, G4#3)  the chat-committed combat card's physical
         reveal drag (plain hand -> card slot, no '(chosen)') was
         never claimed - the nag re-fired and the reveal rode the
         next unrelated gesture.
  FLA-20 (FLH H2, G4#5)  the client watchdog's 'modal dialog open:
         Confirm Random Discard' line halted the server while it was
         the EXPECTED wwsat confirm; telemetry + one evidence-gated
         !CALL wwsat re-send instead.

ASCII only.  RED-first: every fix pin failed on the unfixed tree
(lines recorded in FLA_FIX_REPORT.md).
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
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import ai                                                  # noqa: E402
import game_logic as gl                                    # noqa: E402
import game_state as gs                                    # noqa: E402
from game_state import Side                                # noqa: E402
from board import sanitize                                 # noqa: E402
from test_live_driver import Rig, adopted_state            # noqa: E402


def _chats(items):
    return [str(i[2]) for i in items
            if isinstance(i, tuple) and len(i) > 2 and i[0] == 'chat']


def _lines(items):
    return [str(i[1]) for i in items
            if isinstance(i, tuple) and len(i) > 1 and i[0] == 'line']


def _findings(drv, kind):
    return [v for k, v in drv.findings if k == kind]


# ---------------------------------------------------------------------
# FLA-1 (A2): chat-siege must demand the garrison seat
# ---------------------------------------------------------------------
def test_fla1_chat_siege_demands_garrison_seat(fails):
    """G2 findings 4+7 / G5 finding 7 (2/2 chat-siege divergences vs
    3/3 clean menu-sieges): a NEW siege whose garrison is the HUMAN's
    must bank the seat demand.  The side comparison used
    str(Side.FREE_PEOPLES).lower() ('side.free_peoples') vs 'fp' -
    never equal, the lane never fired."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    post.besieged_strongholds = set(
        getattr(post, 'besieged_strongholds', set()) or set())
    post.besieged_strongholds.add('lorien')
    items = drv.post_commit_sweep(pre, post, label='chat:chat')
    if not _findings(drv, 'siege_seat_demanded'):
        fails.append(('fla1_no_demand',
                      'chat-siege banked no garrison-seat demand: '
                      'findings=%r items=%r'
                      % (drv.findings[-4:], [i[0] for i in items])))
        return
    if sanitize(str(getattr(drv, '_siege_seat_owed', '') or '')) \
            != 'lorien':
        fails.append(('fla1_owed_not_set',
                      repr(getattr(drv, '_siege_seat_owed', None))))
    if not any('garrison' in c.lower() or 'stronghold' in c.lower()
               for c in _chats(items)):
        fails.append(('fla1_no_chat', _chats(items)))
    # the demand holds the census (the G5 loop: rows confirmed while
    # nothing said the seat was owed)
    if not any('seat the garrison' in h
               for h in drv.quiescence_holds()):
        fails.append(('fla1_no_hold', drv.quiescence_holds()[:6]))
    # SPECIFICITY: the AI's own garrison (human = SHADOW attacker)
    # demands nothing - its withdrawals render via its planner
    rig2 = Rig(adopted_state(), Side.FREE_PEOPLES, phase='actions')
    pre2 = ai.clone_state(rig2.drv.es)
    post2 = ai.clone_state(rig2.drv.es)
    post2.besieged_strongholds = set(
        getattr(post2, 'besieged_strongholds', set()) or set())
    post2.besieged_strongholds.add('lorien')
    rig2.drv.post_commit_sweep(pre2, post2, label='chat:chat')
    if _findings(rig2.drv, 'siege_seat_demanded'):
        fails.append(('fla1_specificity',
                      'the AI-side garrison was demanded of the human'))


# ---------------------------------------------------------------------
# FLA-2 (A1): separation leaving the Fellowship box banks named_move
# ---------------------------------------------------------------------
def _separate_strider(post, region):
    comp = next(c for c in post.fellowship.companions
                if getattr(c, 'name', '') == 'Strider')
    post.fellowship.companions = [
        c for c in post.fellowship.companions if c is not comp]
    if getattr(getattr(post.fellowship, 'guide', None), 'name', '') \
            == 'Strider':
        post.fellowship.guide = next(
            iter(post.fellowship.companions), None)
    gl.get_or_create_army(post, region).leaders.append(comp)
    return comp


def test_fla2_separation_banks_named_move(fails):
    """G5 finding 2 (Gwaihir separation: Strider -> north downs
    committed engine-side, never rendered, no echo owed -> confirmed
    divergence).  A named JOIN with no leave whose figure left the
    Fellowship banks a named_move from the Fellowship box; the
    human's own box->region drag closes it."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    _separate_strider(post, 'north downs')
    items = drv.post_commit_sweep(pre, post, label='chat:pick')
    rows = [e for e in drv.echo_ledger
            if e.get('kind') == 'named_move'
            and e.get('tkey') == 'Strider']
    if not rows:
        fails.append(('fla2_not_banked',
                      'separation banked no named_move: ledger=%r '
                      'items=%r' % ([dict(e) for e in drv.echo_ledger],
                                    [i[0] for i in items])))
        return
    row = rows[0]
    if sanitize(str(row.get('to_region'))) != 'north downs':
        fails.append(('fla2_to_region', dict(row)))
    if sanitize(str(row.get('region'))) != 'fellowship of the ring':
        fails.append(('fla2_from_region', dict(row)))
    if not any('drag' in c.lower() or 'show it on the board' in c.lower()
               for c in _chats(items)):
        fails.append(('fla2_no_chat', _chats(items)))
    # the demand text names both ends (census hold readable)
    if not any('Strider' in h and 'north downs' in h
               for h in drv.quiescence_holds()):
        fails.append(('fla2_no_hold', drv.quiescence_holds()[:6]))
    # the human's own drag from the Fellowship box CLOSES it
    drv._snapshot([], 'test', post=post)
    drv.es = post
    consumed = drv._consume_mirror_echo({
        'op': 'move', 'obj': 'Strider',
        'from': 'Fellowship Of The Ring', 'to': 'North Downs',
        'player': 'Wey',
        'norm': 'Wey moves Strider from Fellowship Of The Ring to '
                'North Downs.'})
    if not consumed or any(e.get('kind') == 'named_move'
                           for e in drv.echo_ledger):
        fails.append(('fla2_not_consumed',
                      'box->region drag did not close the demand: '
                      'consumed=%r ledger=%r'
                      % (consumed,
                         [dict(e) for e in drv.echo_ledger])))
    # SELF-CLOSE: with the client pid already at the destination the
    # demand never banks (no double-perform)
    rig2 = Rig(adopted_state(), Side.SHADOW, phase='actions')
    rig2.drv.view.piece_region['241'] = 'North Downs'
    pre2 = ai.clone_state(rig2.drv.es)
    post2 = ai.clone_state(rig2.drv.es)
    _separate_strider(post2, 'north downs')
    rig2.drv.post_commit_sweep(pre2, post2, label='chat:pick')
    if any(e.get('kind') == 'named_move' for e in rig2.drv.echo_ledger):
        fails.append(('fla2_selfclose',
                      [dict(e) for e in rig2.drv.echo_ledger]))
    if not _findings(rig2.drv, 'echo_move_self_closed'):
        fails.append(('fla2_selfclose_finding', rig2.drv.findings[-4:]))


# ---------------------------------------------------------------------
# FLA-3 (A3/A4a + D5a): named relocation banks under ANY label;
# *_move-only sweeps never chat 'destroyed pieces'
# ---------------------------------------------------------------------
def test_fla3_named_relocation_banks_under_window_label(fails):
    """G3 finding 6 (Return of the Witch-king, a play_event WINDOW
    label): WK moved osgiliath -> angmar engine-side with no banked
    drag.  The 'chat:'-only gate is gone: a residual 1-leave+1-join
    named relocation is unexpressed by construction and banks the
    paired move under any label.  G3 finding 4 (D5a): the sweep's
    tail chat for *_move-only rows must not read 'your destroyed
    pieces are still on the board'."""
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    wk = drv.es.witch_king
    wk.in_play = True
    gl.get_or_create_army(drv.es, 'osgiliath').leaders.append(wk)
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    a = next(ar for k, ar in post.armies.items()
             if sanitize(str(k)) == 'osgiliath')
    wk2 = next(ld for ld in a.leaders
               if getattr(ld, 'name', '') == 'Witch King')
    a.leaders.remove(wk2)
    gl.get_or_create_army(post, 'angmar').leaders.append(wk2)
    items = drv.post_commit_sweep(pre, post, label='window')
    rows = [e for e in drv.echo_ledger
            if e.get('kind') == 'named_move'
            and e.get('tkey') == 'Witch King']
    if not rows:
        fails.append(('fla3_not_banked',
                      'window-label named relocation banked nothing: '
                      'ledger=%r' % ([dict(e) for e in
                                      drv.echo_ledger],)))
        return
    row = rows[0]
    if sanitize(str(row.get('region'))) != 'osgiliath' \
            or sanitize(str(row.get('to_region'))) != 'angmar':
        fails.append(('fla3_row', dict(row)))
    # D5a: a *_move-only bank says the MOVE wording, never the kill
    if any('destroyed pieces' in c for c in _chats(items)):
        fails.append(('fla3_kill_wording', _chats(items)))
    if not any('show it on the board' in c or 'drag' in c.lower()
               for c in _chats(items)):
        fails.append(('fla3_no_move_chat', _chats(items)))


def test_fla6_minion_named_pid_resolves(fails):
    """G3 finding 5 (the owed 'Witch King osgiliath -> minas tirith'
    drag never auto-performed while group drags were): the seat's
    performer resolves the pid via emitter._companion_pid, which knew
    only the 8 COMPANIONS - 'Witch King' raised LookupError.  Minions
    resolve now; _named_pid_side keeps its side attribution."""
    import emitter
    from wotr_bridge.live import emission_card as emc
    try:
        pid = emitter._companion_pid('Witch King')
    except LookupError as e:
        fails.append(('fla6_lookup',
                      '_companion_pid raised for the Witch King: %r'
                      % (e,)))
        return
    if str(pid) != '253':
        fails.append(('fla6_pid', repr(pid)))
    for nm, want in (('Witch King', ('253', 'SHADOW')),
                     ('Saruman', ('254', 'SHADOW')),
                     ('Strider', ('241', 'FREE_PEOPLES')),
                     ('Aragorn', ('241', 'FREE_PEOPLES')),
                     ('Gandalf the White', ('240', 'FREE_PEOPLES'))):
        got = emc._named_pid_side(nm)
        if (str(got[0]), got[1]) != want:
            fails.append(('fla6_side', '%s -> %r (want %r)'
                          % (nm, got, want)))
    if emc._named_pid_side('No Such Figure') != (None, None):
        fails.append(('fla6_unknown',
                      emc._named_pid_side('No Such Figure')))


# ---------------------------------------------------------------------
# FLA-5 (A6a): leader arrival self-close
# ---------------------------------------------------------------------
def test_fla5_leader_arrival_self_closes(fails):
    """G5 finding 6 / G1 finding 2 (leaders edoras/helm s deep fp:
    engine=1 client=2 pids=284 283): the client had ALREADY placed
    the leader, but _engine_tkey_count returned None for ':leader'
    tkeys so the BL-101 self-close never ran - the arrival banked and
    the seat re-performed a duplicate leader."""
    from wotr_bridge.live import client_board as cb
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    # a client dump already showing one Rohan Leader at Edoras
    nat, ut = cb.unit_type('283')
    if str(ut).lower() != 'leader' or 'rohan' not in str(nat).lower():
        fails.append(('fla5_control',
                      'NOT-MEASURED: pid 283 is not a Rohan Leader '
                      '(%r, %r)' % (nat, ut)))
        return
    drv.view.piece_region['283'] = 'Edoras'
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    ldr = gs.Leader(nation=gs.Nation.ROHAN)
    gl.get_or_create_army(post, 'edoras').leaders.append(ldr)
    drv.post_commit_sweep(pre, post, label='chat:chat')
    rows = [e for e in drv.echo_ledger
            if str(e.get('tkey')) == 'rohan:leader']
    if rows:
        fails.append(('fla5_banked_anyway',
                      'client-satisfied leader arrival banked: %r'
                      % ([dict(e) for e in rows],)))
    if not _findings(drv, 'echo_in_self_closed'):
        fails.append(('fla5_no_selfclose_finding', drv.findings[-4:]))
    # SENSITIVITY: with the client leader ELSEWHERE the arrival still
    # banks (the self-close must not eat real demands)
    rig2 = Rig(adopted_state(), Side.SHADOW, phase='actions')
    rig2.drv.view.piece_region['283'] = 'Westemnet'
    pre2 = ai.clone_state(rig2.drv.es)
    post2 = ai.clone_state(rig2.drv.es)
    gl.get_or_create_army(post2, 'edoras').leaders.append(
        gs.Leader(nation=gs.Nation.ROHAN))
    rig2.drv.post_commit_sweep(pre2, post2, label='chat:chat')
    if not any(str(e.get('tkey')) == 'rohan:leader'
               for e in rig2.drv.echo_ledger):
        fails.append(('fla5_sensitivity',
                      'real leader arrival no longer banks: %r'
                      % ([dict(e) for e in rig2.drv.echo_ledger],)))


# ---------------------------------------------------------------------
# FLA-4 (A6b): a window-collected record still closes an open echo
# ---------------------------------------------------------------------
def test_fla4_window_collected_record_closes_echo(fails):
    """G5 finding 5 (echo 'drag in 1 x rohan:leader at edoras' never
    closed although the feed showed the placement; census frozen
    'owed:' a whole turn): the placement arrived while a die window
    was open, was collected as window content and never reached
    _consume_mirror_echo."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    drv._echo_seq = 1
    drv.echo_ledger.append({'kind': 'unit_in', 'side': 'fp',
                            'region': 'edoras', 'tkey': 'rohan:leader',
                            'n': 1, 'echo_id': 1})
    rig.feed('$Wey moves Free Action Die (W) from FP Dice to '
             'Current Action.')
    if getattr(drv.seg, 'window', None) is None:
        fails.append(('fla4_control', 'NOT-MEASURED: no window opened'))
        return
    rig.feed('$Wey moves Rohan Leader from FP Reinforcements to '
             'Edoras.')
    if any(e.get('kind') == 'unit_in' for e in drv.echo_ledger):
        fails.append(('fla4_echo_stuck',
                      'the placement did not close the echo: '
                      'ledger=%r window_records=%r'
                      % ([dict(e) for e in drv.echo_ledger],
                         [r.get('norm') for r in
                          (getattr(drv.seg.window, 'records', None)
                           or [])])))
    # the record must NOT also sit in the window as gesture content
    w = getattr(drv.seg, 'window', None)
    if w is not None and any(
            'Rohan Leader' in str(r.get('norm', ''))
            for r in (w.records or [])):
        fails.append(('fla4_double_booked',
                      [r.get('norm') for r in w.records]))


# ---------------------------------------------------------------------
# FLA-7 (A5): adoption defers on open owed demands; census rows name
# the demand
# ---------------------------------------------------------------------
def test_fla7_adopt_defers_on_owed_demands(fails):
    """G4 finding 2 (Orcs Multiplying Again: engine owed 6 recruits;
    the adopt then deleted them and the row said 'owed: none'): an
    auto/consented adoption while open demands explain rows is
    DEFERRED loudly; census row lines name the demand for their
    zone."""
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    drv._echo_seq = 1
    drv.echo_ledger.append({'kind': 'unit_in', 'side': 'shadow',
                            'region': 'dol guldur',
                            'tkey': 'sauron:regular', 'n': 3,
                            'echo_id': 1})
    pre_units = {sanitize(str(k)): len(a.units)
                 for k, a in drv.es.armies.items()}
    items = drv._adopt_board_now('consented')
    notes = [str(i[1]) for i in items if i and i[0] == 'note']
    if not any('DEFERRED' in n and 'owed' in n.lower() for n in notes):
        fails.append(('fla7_adopted_over_demand',
                      'adoption ran over an open owed demand: %r'
                      % (notes,)))
    if not _findings(drv, 'adopt_deferred_owed_demands'):
        fails.append(('fla7_no_finding', drv.findings[-4:]))
    post_units = {sanitize(str(k)): len(a.units)
                  for k, a in drv.es.armies.items()}
    if pre_units != post_units:
        fails.append(('fla7_engine_touched', (pre_units, post_units)))
    # census row lines name the demand for that zone, never
    # 'owed: none'
    line = drv._census_row_lines(
        ['units dol guldur sauron/regular: engine=8 client=5 '
         'pids=015 014 013 012 011'], 11)[0]
    if 'owed: none' in line or 'drag in 3 x sauron:regular' not in line:
        fails.append(('fla7_owed_none', line))
    # a row with NO matching demand still prints 'owed: none'
    line2 = drv._census_row_lines(
        ['units gorgoroth sauron/regular: engine=2 client=1 pids=-'],
        11)[0]
    if 'owed: none' not in line2:
        fails.append(('fla7_owed_specificity', line2))
    # manual (BL-147: client AS-IS) is NOT deferred by demands
    drv2 = Rig(adopted_state(), Side.FREE_PEOPLES,
               phase='actions').drv
    drv2.echo_ledger.append({'kind': 'unit_in', 'side': 'shadow',
                             'region': 'dol guldur',
                             'tkey': 'sauron:regular', 'n': 3,
                             'echo_id': 1})
    items2 = drv2._adopt_board_now('manual')
    notes2 = [str(i[1]) for i in items2 if i and i[0] == 'note']
    if any('DEFERRED' in n and 'owed' in n.lower() for n in notes2):
        fails.append(('fla7_manual_blocked', notes2))


# ---------------------------------------------------------------------
# FLA-13 (B2): the hunt guide-sacrifice drag commits + is voiced
# ---------------------------------------------------------------------
def test_fla13_hunt_guide_drag_commits_and_voices(fails):
    """G2 finding 6 / G1 finding 3: the hunt menu's guide-sacrifice
    physical form ('moves Gandalf the Grey ... to FP Casualties')
    produced EMPTY window facts, so _rec_inert dropped it and the
    pending never closed - hunt decisions were chat-only in practice.
    A companion moving INTO a dead zone is the casualty gesture
    (comps_gone), the pending commits, and the commit is voiced."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    guide = getattr(drv.es.fellowship, 'guide', None)
    if getattr(guide, 'name', '') != 'Gandalf the Grey':
        fails.append(('fla13_control', 'NOT-MEASURED: guide is %r'
                      % (getattr(guide, 'name', None),)))
        return
    drv.es.pending_hunt = {'damage': 2, 'phase': 'hunt_sacrifice',
                           'casualty_taken': False}
    dp = ai.describe_pending(drv.es)
    if dp is None or dp['kind'] != 'hunt' \
            or not any(a.action_type == 'hunt_sacrifice_guide'
                       for a in dp['actions']):
        fails.append(('fla13_control', 'NOT-MEASURED: pending %r'
                      % (dp and dp['kind'],)))
        return
    out = rig.feed('$Wey moves Gandalf the Grey from Guide Of The '
                   'Fellowship to FP Casualties.')
    if getattr(drv.es, 'pending_hunt', None) is not None:
        fails.append(('fla13_pending_open',
                      'the guide-sacrifice drag did not commit: '
                      'pending=%r stats=%r findings=%r'
                      % (drv.es.pending_hunt,
                         dict(drv.pending_stats),
                         drv.findings[-4:])))
        return
    if any(getattr(c, 'name', '') == 'Gandalf the Grey'
           for c in drv.es.fellowship.companions):
        fails.append(('fla13_guide_alive', 'Gandalf still in the ring'))
    chats = _chats(out) + [str(i[2]) for i in
                           getattr(drv, '_out_queue', [])
                           if isinstance(i, tuple) and len(i) > 2
                           and i[0] == 'chat']
    if not any('committed' in c for c in chats):
        fails.append(('fla13_no_voice', chats))


# ---------------------------------------------------------------------
# FLA-15 (B1): bare number falls through to the say_pick index
# ---------------------------------------------------------------------
def test_fla15_bare_number_falls_through_to_index(fails):
    """G1 finding 10: bare '2' over a combat_effect_choice menu whose
    counts did not include 2 was refused although the published row
    advertised say_pick=2.  The magnitude lane keeps precedence when
    a count matches; otherwise the number resolves as the advertised
    1-based index."""
    from wotr_bridge.live import chat as chat_mod

    class _A(object):
        def __init__(self, at, **params):
            self.action_type = at
            self.params = params
            self.die_idx = -1
    menu = [_A('combat_effect_choice', count=0),
            _A('combat_effect_choice', count=1)]
    hit = chat_mod.match_choice('2', menu)
    if hit is not menu[1]:
        fails.append(('fla15_refused',
                      'bare 2 did not resolve to row #2: %r' % (hit,)))
    # magnitude precedence unchanged: '1' is the count-1 row
    hit1 = chat_mod.match_choice('1', menu)
    if hit1 is not menu[1]:
        fails.append(('fla15_magnitude', repr(hit1)))
    # '0' stays the count-0 row (state it even if it is 0)
    hit0 = chat_mod.match_choice('0', menu)
    if hit0 is not menu[0]:
        fails.append(('fla15_zero', repr(hit0)))


# ---------------------------------------------------------------------
# FLA-8 (D1i): the repair kill list is scoped to the current turn
# ---------------------------------------------------------------------
def test_fla8_repair_kills_scoped_to_current_turn(fails):
    """G4 finding 6: accepting the self-repair killed 4 staged pieces
    INCLUDING two combat cards staged TWO TURNS earlier (still in
    Current Action) - the journal replay then died on the changed
    world.  Pieces first seen in a repair slot in an earlier turn are
    excluded, loudly."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    drv.es.turn = 2
    drv.view.piece_region.update({'101': 'Current Action',
                                  '150': 'Current SA Card',
                                  '151': 'Current Action'})
    drv._note_repair_slots()
    drv.es.turn = 4
    drv.view.piece_region['102'] = 'Current Action'
    drv._note_repair_slots()
    targets = drv._repair_kill_targets()
    pids = sorted(p for p, _z in targets)
    if pids != ['102']:
        fails.append(('fla8_scope',
                      'repair kill list not scoped to this turn: %r'
                      % (targets,)))
    if not _findings(drv, 'repair_kill_scope_excluded'):
        fails.append(('fla8_silent', drv.findings[-4:]))
    # SPECIFICITY: same-turn pieces all die
    rig2 = Rig(adopted_state(), Side.SHADOW, phase='actions')
    rig2.drv.es.turn = 4
    rig2.drv.view.piece_region.update({'101': 'Current Action',
                                       '150': 'Current SA Card'})
    rig2.drv._note_repair_slots()
    if len(rig2.drv._repair_kill_targets()) != 2:
        fails.append(('fla8_specificity',
                      rig2.drv._repair_kill_targets()))


# ---------------------------------------------------------------------
# FLA-9 (D3): no force-undo while a confirmed divergence stands
# ---------------------------------------------------------------------
_BAD_WINDOW = (
    '$Wey moves Free Action Die (C) from FP Dice to Current Action.',
    '$Wey moves North Regular from Bree to Forlindon.',
    '$Wey kills Free Action Die (C) in Current Action.',
)


def test_fla9_gate_holds_instead_of_undo_during_divergence(fails):
    """G5 finding 9: a fully legal, fully rendered move was
    force-undone citing an impossible 'closest legal move: Attack
    Lorien from Lorien' WHILE divergence #7 stood confirmed.  The
    gate must hold + witness, never force-undo on the authority of a
    board it knows is wrong."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    drv._census_confirmed_sig = ('sig',)
    out = rig.feed(*_BAD_WINDOW)
    undo = [i for i in out if i[0] == 'line'
            and 'Controls.819' in str(i[1])]
    if undo:
        fails.append(('fla9_force_undo',
                      'gate force-undid during an open divergence: %r'
                      % ([str(i[1])[:60] for i in undo],)))
    if not _findings(drv, 'gesture_hold_divergence_open'):
        fails.append(('fla9_no_finding', drv.findings[-4:]))
    if not drv.holds:
        fails.append(('fla9_not_held', dict(drv.window_stats)))
    if not any('HELD' in c for c in _chats(out)):
        fails.append(('fla9_no_chat', _chats(out)))
    # CONTROL: without a divergence the gate still refuses + undoes
    rig2 = Rig(adopted_state(), Side.SHADOW, phase='actions')
    out2 = rig2.feed(*_BAD_WINDOW)
    if not any(i[0] == 'line' and 'Controls.819' in str(i[1])
               for i in out2):
        fails.append(('fla9_control',
                      'the clean-board refusal lost its undo: %r'
                      % ([i[0] for i in out2],)))


# ---------------------------------------------------------------------
# FLA-10 (D4): die-less drags are repair traffic during a divergence
# ---------------------------------------------------------------------
def test_fla10_dieless_repair_drag_accepted(fails):
    """G3 finding 6b: the repair drag toward the ENGINE state was
    refused ('stage an action die first') and undone - the census
    'no' route ('you move the pieces to match my engine') was a
    catch-22.  While a confirmed divergence stands, a die-less drag
    is accepted as repair traffic."""
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    drv._census_confirmed_sig = ('sig',)
    out = rig.feed('$Wey moves Sauron Regular from Gorgoroth to '
                   'Minas Morgul.')
    if any(i[0] == 'line' and 'Controls.819' in str(i[1])
           for i in out):
        fails.append(('fla10_undone',
                      'repair drag undone during divergence'))
    if not _findings(drv, 'repair_drag_accepted'):
        fails.append(('fla10_refused',
                      'repair drag not accepted: findings=%r out=%r'
                      % (drv.findings[-4:], [i[0] for i in out])))
    # CONTROL: without a divergence the die-less drag still refuses
    rig2 = Rig(adopted_state(), Side.FREE_PEOPLES, phase='actions')
    out2 = rig2.feed('$Wey moves Sauron Regular from Gorgoroth to '
                     'Minas Morgul.')
    if not _findings(rig2.drv, 'dieless_gesture_refused'):
        fails.append(('fla10_control',
                      'clean-board die-less drag no longer refused: '
                      '%r' % (rig2.drv.findings[-4:],)))


# ---------------------------------------------------------------------
# FLA-12 (D6): the plain hand->slot reveal drag claims the demand
# ---------------------------------------------------------------------
def test_fla12_plain_reveal_drag_claims_once(fails):
    """G4 finding 3b: 'reveal your committed combat card' nagged 3-4x
    and the reveal only rode the NEXT unrelated gesture - the
    chat-committed card's physical reveal is a PLAIN hand->card-slot
    drag (no '(chosen)') that nothing claimed.  Claimed the moment it
    lands; the nag stops."""
    from wotr_bridge.live import gesture_match as gm
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    title = 'The Fighting Uruk-hai/Onslaught'
    cid = gm.title_to_cid(title)
    if cid is None:
        fails.append(('fla12_control',
                      'NOT-MEASURED: title %r resolves to no cid'
                      % title))
        return
    drv._cc_reveal_owed = {str(cid): 'owed'}
    rig.feed('$Wey moves %s from SA Hand to Current SA Card.' % title)
    if drv._cc_reveal_owed.get(str(cid)) != 'claimed':
        fails.append(('fla12_unclaimed',
                      'plain reveal drag left the demand owed: %r '
                      'findings=%r'
                      % (drv._cc_reveal_owed, drv.findings[-4:])))
    if not _findings(drv, 'cc_reveal_physical'):
        fails.append(('fla12_no_finding', drv.findings[-4:]))


# ---------------------------------------------------------------------
# FLA-16 (B4): the declare menu never publishes n_total=null
# ---------------------------------------------------------------------
def test_fla16_declare_menu_republishes(fails):
    """G1 finding 5 / G2 finding 8: the turn-start declare window
    published NO menu rows (n_total=null) after the AI's own action
    cleared the engine's _declare_phase flag.  status_menu re-arms
    the flag (K2's one answerer) before reading the pending."""
    import tempfile as _tf
    from wotr_bridge.live.live_server import build
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv, _bridge = build(td, 'SA', load_nets=False, seed=7)
    drv = srv.driver
    drv.phase = 'declare'
    drv.es._declare_phase = False      # the cleared-flag state
    drv._declare_done_turn = None
    hm = (srv.status_menu() or {}).get('human_menu') or {}
    if hm.get('pending') != 'declare' \
            or not (hm.get('by_type') or {}).get('fellowship_declare',
                                                 0):
        fails.append(('fla16_menu_null',
                      'declare window published no menu: pending=%r '
                      'by_type=%r n_actions=%r'
                      % (hm.get('pending'), hm.get('by_type'),
                         hm.get('n_actions'))))


# ---------------------------------------------------------------------
# FLA-17 (D5b): an unrenderable AI card_option degrades, never loops
# ---------------------------------------------------------------------
def test_fla17_emission_hold_degrades_for_card_option(fails):
    """G3 finding 12b: 'EMISSION HOLD: card_option The Ents Awake:
    Entmoot ... Huorns' re-raised every tick (DriverHalt loop,
    cycle_errors 0->7, rollover wedged).  For card_option /
    card_followup the hold degrades: commit with no lines, SAY the
    pick, sweep the residue."""
    import tempfile as _tf
    from wotr_bridge.live import emission as em_mod
    from wotr_bridge.live.live_server import build, DriverHalt
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv, _bridge = build(td, 'FP', load_nets=False, seed=7)
    drv = srv.driver
    # a REAL AI-FP card_option pending (the Eomer play)
    eid = None
    for dk in ('fp_char_deck', 'fp_strat_deck'):
        for c in list(getattr(drv.es, dk)):
            if 'omer_son' in str(c):
                eid = c
                getattr(drv.es, dk).remove(c)
                break
    if eid is None:
        fails.append(('fla17_control', 'NOT-MEASURED: no Eomer card'))
        return
    drv.es.hands[Side.FREE_PEOPLES].hand.append(eid)
    # no Gandalf-the-Grey guide redraw in this pin's way (the grant
    # would starve the step on a client draw before the render)
    _str17 = next((c for c in drv.es.fellowship.companions
                   if getattr(c, 'name', '') == 'Strider'), None)
    if _str17 is not None:
        drv.es.fellowship.guide = _str17
    gl.adopt_rolled_faces(drv.es, [(Side.FREE_PEOPLES, 0,
                                    gs.ActionDieFace.EVENT)])
    import ai as _ai
    menu = _ai.get_legal_actions(drv.es, Side.FREE_PEOPLES)
    pe = [a for a in menu if a.action_type == 'play_event'
          and 'omer_son' in str(a.params.get('card_id', ''))]
    if not pe:
        fails.append(('fla17_control', 'NOT-MEASURED: no play row'))
        return
    drv.es = _ai.apply_action(drv.es, pe[0])
    drv.hands.es = drv.es
    dp = drv.ai_action_dp()
    if dp is None or str(dp['kind']) != 'card_option':
        fails.append(('fla17_control', 'NOT-MEASURED: dp %r'
                      % (dp and dp['kind'],)))
        return
    srv._render = lambda *a, **k: em_mod.Emission(
        hold='card_option unrenderable (pin)')
    try:
        out = srv._ai_decide(dp)
    except DriverHalt as e:
        fails.append(('fla17_halt_loop',
                      'the hold still raises DriverHalt: %r' % (e,)))
        return
    if not _findings(drv, 'emission_hold_degraded'):
        fails.append(('fla17_no_finding', drv.findings[-6:]))
    if getattr(drv.es, 'pending_card_option', None) is not None:
        fails.append(('fla17_not_committed',
                      'the pick did not commit engine-side'))
    # _say returns the rendered chat LINE (a raw string in the RESP)
    if not any('committed' in str(i) for i in out):
        fails.append(('fla17_not_voiced',
                      [str(i)[:60] for i in out][:6]))


# ---------------------------------------------------------------------
# FLA-18 (D1ii): journal mismatch degrades to abandon + ASK
# ---------------------------------------------------------------------
def test_fla18_journal_mismatch_abandons_step(fails):
    """G4 finding 6: 'LIVE HALT: JOURNAL MISMATCH ... [sticky]' froze
    every later cycle (chat dead, cycle_errors 0->80).  The step is
    dead, not the game: abandon it, clear the sticky, alarm; a repeat
    routes the self-repair ASK.  A step opened before a resync is
    abandoned before it can replay a stale journal."""
    import tempfile as _tf
    import time as _t
    from wotr_bridge.live.live_server import build
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv, _bridge = build(td, 'SA', load_nets=False, seed=7)
    drv = srv.driver

    def _boom():
        srv.rand.sticky_halt = ('LIVE HALT: JOURNAL MISMATCH: '
                                'recorded pick x not among 4')
        raise RuntimeError(srv.rand.sticky_halt)
    srv._ai_draws = _boom
    srv._open_step = ('ai_draws', None)
    srv._step_resync0 = 0.0
    try:
        srv._run_open_step()
    except RuntimeError as e:
        fails.append(('fla18_still_raises',
                      'journal mismatch still escapes the step: %r'
                      % (e,)))
        return
    if srv.rand.sticky_halt is not None:
        fails.append(('fla18_sticky', srv.rand.sticky_halt))
    if srv._open_step is not None:
        fails.append(('fla18_step_open', srv._open_step))
    if not _findings(drv, 'journal_mismatch_step_abandoned'):
        fails.append(('fla18_no_finding', drv.findings[-4:]))
    if not any('JOURNAL MISMATCH' in a for a in drv.alarms):
        fails.append(('fla18_no_alarm', drv.alarms[-3:]))
    # a SECOND mismatch routes the self-repair ASK
    srv._open_step = ('ai_draws', None)
    srv._step_resync0 = 0.0
    srv._run_open_step()
    if drv._repair is None:
        fails.append(('fla18_no_ask',
                      'second mismatch proposed no repair'))
    # the resync-abandon leg
    drv._repair = None
    srv._abandon_step('pin reset')
    srv._open_step = ('ai_draws', None)
    srv._step_resync0 = 1.0
    drv._resync_ts = _t.time()
    srv._run_open_step()
    if not _findings(drv, 'step_abandoned_after_resync'):
        fails.append(('fla18_resync_leg', drv.findings[-4:]))


# ---------------------------------------------------------------------
# FLA-19 (D1iii): chat 'reset' clears the sticky halt
# ---------------------------------------------------------------------
def test_fla19_chat_reset_clears_sticky_halt(fails):
    """G4 finding 6: RESET+RESYNC was GUI-only and the chat lane was
    dead - unrecoverable from the CLI.  A chat 'reset' pulls the SAME
    panel lever; the halt text names the route."""
    import tempfile as _tf
    from wotr_bridge.live.live_server import build
    from wotr_bridge.live.driver import DriverHalt
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv, _bridge = build(td, 'SA', load_nets=False, seed=7)
    srv.rand.sticky_halt = 'LIVE HALT: JOURNAL MISMATCH: pin'
    try:
        srv._handle_inner({'lines': []})
        fails.append(('fla19_not_sticky',
                      'the sticky halt no longer halts'))
        return
    except DriverHalt as e:
        if 'reset' not in str(e).lower() or 'chat' not in str(e):
            fails.append(('fla19_route_unnamed', str(e)[:160]))
    try:
        srv._handle_inner({'lines': ['<Wey> reset']})
    except DriverHalt as e:
        fails.append(('fla19_chat_reset_dead',
                      "chat 'reset' did not clear the halt: %r"
                      % (e,)))
        return
    if srv.rand.sticky_halt is not None:
        fails.append(('fla19_halt_kept', srv.rand.sticky_halt))


# ---------------------------------------------------------------------
# FLA-20 (FLH H2): the WWSAT confirm modal is telemetry + one re-send
# ---------------------------------------------------------------------
def test_fla20_wwsat_modal_telemetry_and_resend(fails):
    """G4 finding 5 + FLH_HANDOFF: the watchdog's 'modal dialog open:
    Confirm Random Discard' line became DriverHalt x2 while it was
    the EXPECTED wwsat confirm; the ask was never re-sent and the
    seat had no route.  The line is telemetry while a card_pick
    starves, and proves the confirm died unanswered -> ONE re-send;
    waiting_on names the recovery."""
    import tempfile as _tf
    import time as _t
    from wotr_bridge.live.live_server import build
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv, _bridge = build(td, 'SA', load_nets=False, seed=7)
    drv = srv.driver

    class _CP(object):
        kind = 'card_pick'
        kw = {'ctx': 'WWSAT hand discard'}
    srv.rand._pending = _CP()
    line = ("<game> AI: modal dialog open: title='Confirm Random "
            "Discard' text='This will discard a random character "
            "card from the FP hand. Continue?' (injecting=true)")
    if not drv._wwsat_modal_expected_line(line):
        fails.append(('fla20_not_telemetry',
                      'the expected confirm line is not telemetry'))
    if not _findings(drv, 'wwsat_modal_telemetry'):
        fails.append(('fla20_no_finding', drv.findings[-3:]))
    # specificity: a DIFFERENT modal stays a failure line
    if drv._wwsat_modal_expected_line(
            "<game> AI: modal dialog open: title='Choose' text='x'"):
        fails.append(('fla20_overbroad', 'other modal consumed'))
    srv.rand._pending = None
    if drv._wwsat_modal_expected_line(line):
        fails.append(('fla20_no_pending_gate',
                      'consumed with no card_pick starving'))
    # the one evidence-gated re-send
    srv.rand._pending = _CP()
    drv._wwsat_modal_ts = _t.time() - 11.0
    srv._ai_draws = lambda: []           # keep the step open
    srv._open_step = ('ai_draws', None)
    srv._step_resync0 = None
    out = srv._run_open_step()
    if '!CALL wwsat' not in [str(i) for i in out]:
        fails.append(('fla20_no_resend', [str(i)[:60] for i in out]))
    if not _findings(drv, 'wwsat_ask_resent'):
        fails.append(('fla20_no_resend_finding', drv.findings[-3:]))
    out2 = srv._run_open_step()
    if '!CALL wwsat' in [str(i) for i in out2]:
        fails.append(('fla20_double_resend', out2))
    txt = srv._waiting_on()
    if 'Confirm Random Discard' not in str(txt):
        fails.append(('fla20_waiting_text', str(txt)[:120]))


# ---------------------------------------------------------------------
# FLC H1: the political-activation tripwire
# ---------------------------------------------------------------------
def test_flc_h1_activation_flip_tripwire(fails):
    """FLC H1 (par5 g1#12, CRITICAL): active[Gondor] flipped True
    off-engine between the T4 and T5 menus with no trigger anywhere
    in the feed, and a published advance then walked a PASSIVE nation
    into At War.  The tripwire checksums the activation map per tick;
    a flip with NO commit in between is a divergence-class alarm (the
    next occurrence names its writer)."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')
    drv = rig.drv
    drv._political_activation_witness()
    drv.es.political.active[gs.Nation.GONDOR] = True
    drv._political_activation_witness()
    rows = _findings(drv, 'political_activation_changed')
    if not rows or 'GONDOR' not in rows[-1] or 'NONE' not in rows[-1]:
        fails.append(('flc_h1_no_finding', rows or drv.findings[-4:]))
    if not any('FLIPPED OUTSIDE ANY COMMIT' in a for a in drv.alarms):
        fails.append(('flc_h1_no_alarm', drv.alarms[-3:]))
    from wotr_bridge.live import taxonomy as tax
    if tax.classify(next((a for a in drv.alarms
                          if 'FLIPPED OUTSIDE' in a), '')) \
            != 'divergence':
        fails.append(('flc_h1_class', 'alarm not divergence-class'))
    # SPECIFICITY: a flip WITH a commit records, never alarms
    rig2 = Rig(adopted_state(), Side.SHADOW, phase='actions')
    d2 = rig2.drv
    d2._political_activation_witness()
    d2.es.political.active[gs.Nation.GONDOR] = True
    d2.n_commits_total += 1
    d2.commits.append((None, [], 'chat:test', {}, None))
    d2._political_activation_witness()
    if any('FLIPPED OUTSIDE' in a for a in d2.alarms):
        fails.append(('flc_h1_overtrip', d2.alarms[-2:]))
    if not _findings(d2, 'political_activation_committed'):
        fails.append(('flc_h1_commit_unrecorded', d2.findings[-3:]))


# ---------------------------------------------------------------------
# FLC H2: box-sourced downgrade replacements reach the client
# ---------------------------------------------------------------------
def test_flc_h2_downgrade_source_rides_the_row(fails):
    """FLC H2 (par5 g1#11 + g1#9 / fleet C2+C4): the engine revives
    the downgrade Regular from the CASUALTY box first (rulebook
    2760-2767, _dg_src_log stamps 'cas'|'pool'), but the seat's
    arrival renderer only ever drew FP pieces from 'fp reinf' -
    'need 1 regular (fp) in fp reinf, found 0' refused 5x and the
    tester read a legal round as a phantom die.  The sweep banks the
    stamp on the row; the renderer honors it."""
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='actions')   # human = FP
    drv = rig.drv
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    # one Elven elite downgraded at lorien: elite -1, regular +1,
    # engine sourced the replacement from the casualty box
    a = next(ar for k, ar in post.armies.items()
             if sanitize(str(k)) == 'lorien')
    el = next((u for u in a.units
               if str(getattr(u.kind, 'value', u.kind)).lower()
               == 'elite'), None)
    if el is None:
        fails.append(('flc_h2_control', 'NOT-MEASURED: no elite at '
                      'lorien'))
        return
    a.units.remove(el)
    a.units.append(gs.Unit(nation=gs.Nation.ELVES,
                           kind=gs.UnitKind.REGULAR))
    post._dg_src_log = list(getattr(pre, '_dg_src_log', None) or ()) \
        + [(gs.Nation.ELVES, 'cas')]
    drv.post_commit_sweep(pre, post, label='chat:chat')
    rows = [e for e in drv.echo_ledger
            if e.get('kind') == 'unit_in'
            and e.get('tkey') == 'elves:regular']
    if not rows:
        fails.append(('flc_h2_no_row',
                      [dict(e) for e in drv.echo_ledger]))
        return
    if rows[0].get('src') != 'cas':
        fails.append(('flc_h2_no_src', dict(rows[0])))
    # the seat renderer honors src='cas': the CASUALTY-box pid is
    # picked ahead of the reinforcement pid
    import json as _json
    import emitter as _em
    import wotr_bridge.live.client_board as _cb
    sys.modules.setdefault('hg_probe', None)
    lab = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                       'audit')
    if lab not in sys.path:
        sys.path.insert(0, lab)
    import human_gestures as hg
    with open(os.path.join(_ROOT, 'wotr_bridge',
                           'initial_positions.json'),
              encoding='utf-8') as f:
        init = dict(_json.load(f))
    for _i in range(900, 960):       # pad past the torn-dump floor
        init[str(_i)] = 'Game.2034'
    pool_code = _em.FP_POOL
    elv = [p for p, c in sorted(init.items())
           if c == pool_code
           and (_cb.unit_type(p) or (None, None))[1] == 'regular'
           and 'elves' in str((_cb.unit_type(p)
                               or ('', ''))[0]).lower()]
    if len(elv) < 2:
        elv = [p for p, c in sorted(init.items())
               if (_cb.unit_type(p) or (None, None))[1] == 'regular'
               and 'elves' in str((_cb.unit_type(p)
                                   or ('', ''))[0]).lower()][:2]
    if len(elv) < 2:
        fails.append(('flc_h2_control',
                      'NOT-MEASURED: <2 elven regular pids'))
        return
    init[elv[0]] = _em.FP_DEAD       # one in the casualty box
    init[elv[1]] = _em.FP_POOL       # one in reinforcements
    by = {}
    for p, c in init.items():
        by.setdefault(c, []).append(p)
    td = tempfile.mkdtemp()
    dp = os.path.join(td, 'ai_client_board.txt')
    with open(dp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(
            '%d\t<~%s~>\t%s\t%s' % (i, c, ' '.join(sorted(ps)),
                                    ' '.join('-' for _ in ps))
            for i, (c, ps) in enumerate(sorted(by.items()))) + '\n')

    class _Ctx(object):
        dump_path = dp
        human_side = 'FP'

        def local_nick(self):
            return 'Wey'
    row = {'tkey': 'elves:regular', 'region': 'lorien', 'd': 1,
           'src': 'cas'}
    lines = hg._render_unit_rows(_Ctx(), [row], label='pin')
    if not lines or _em.FP_DEAD not in str(lines[0]):
        fails.append(('flc_h2_wrong_source',
                      'src=cas arrival not drawn from the casualty '
                      'box: %r' % (lines,)))
    # control: no src -> the pool stays first
    row2 = {'tkey': 'elves:regular', 'region': 'lorien', 'd': 1}
    lines2 = hg._render_unit_rows(_Ctx(), [row2], label='pin')
    if not lines2 or _em.FP_POOL not in str(lines2[0]):
        fails.append(('flc_h2_default_order', repr(lines2)))


# ---------------------------------------------------------------------
# FLA-A5b: consecutive arrival performances never re-pick the same pids
# ---------------------------------------------------------------------
def test_fla_a5_no_same_pid_double_pick(fails):
    """G4 finding 2 (Orcs Multiplying Again): the two owed recruit
    drags were performed in consecutive calls, each re-reading a dump
    the client had not yet updated - BOTH picked pids 339-341, the
    second drag pulled the first drag's pieces onward, 3 recruits
    never landed and the consented adopt then erased them (inject
    files 18/19_echo_kills.cmd verbatim).  Recently-picked pids are
    excluded until the dump can have settled."""
    import json as _json
    import emitter as _em
    import wotr_bridge.live.client_board as _cb
    lab = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                       'audit')
    if lab not in sys.path:
        sys.path.insert(0, lab)
    import human_gestures as hg
    hg._RECENT_ARRIVAL_PICKS.clear()
    with open(os.path.join(_ROOT, 'wotr_bridge',
                           'initial_positions.json'),
              encoding='utf-8') as f:
        init = dict(_json.load(f))
    for _i in range(900, 960):
        init[str(_i)] = 'Game.2034'
    sau = [p for p, c in sorted(init.items())
           if (_cb.unit_type(p) or (None, None))[1] == 'regular'
           and 'sauron' in str((_cb.unit_type(p)
                                or ('', ''))[0]).lower()][:6]
    if len(sau) < 2:
        fails.append(('fla_a5_control', 'NOT-MEASURED: <2 sauron '
                      'regular pids'))
        return
    for p in sau:
        init[p] = _em.SH_POOL
    by = {}
    for p, c in init.items():
        by.setdefault(c, []).append(p)
    td = tempfile.mkdtemp()
    dp = os.path.join(td, 'ai_client_board.txt')
    with open(dp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(
            '%d\t<~%s~>\t%s\t%s' % (i, c, ' '.join(sorted(ps)),
                                    ' '.join('-' for _ in ps))
            for i, (c, ps) in enumerate(sorted(by.items()))) + '\n')

    class _Ctx(object):
        dump_path = dp
        human_side = 'SA'

        def local_nick(self):
            return 'Wey'
    l1 = hg._render_unit_rows(_Ctx(), [{'tkey': 'sauron:regular',
                                        'region': 'dol guldur',
                                        'd': 1, 'src': 'pool'}],
                              label='pin')
    l2 = hg._render_unit_rows(_Ctx(), [{'tkey': 'sauron:regular',
                                        'region': 'mount gundabad',
                                        'd': 1, 'src': 'pool'}],
                              label='pin')

    def _pids(line):
        import re as _re
        m = _re.search(r'~>([\d ]+)<~', str(line))
        return set((m.group(1) if m else '').split())
    p1, p2 = _pids(l1[0] if l1 else ''), _pids(l2[0] if l2 else '')
    if not p1 or not p2 or (p1 & p2):
        fails.append(('fla_a5_double_pick',
                      'consecutive arrivals picked overlapping pids: '
                      '%r %r' % (sorted(p1), sorted(p2))))
    hg._RECENT_ARRIVAL_PICKS.clear()


# ---------------------------------------------------------------------
# FLC H3: chat-committed political advances render at the action
# ---------------------------------------------------------------------
def test_flc_h3_chat_commit_walks_political_marker(fails):
    """FLC H1/H3 (par5 g1#13): the CHAT-committed attack's
    attacked-nation advance never rendered at the action (the sweep's
    political rider ran only for gesture windows), so the client
    marker lagged to the rollover recon and read as a double step."""
    import log_replay as lr
    st = adopted_state()
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')   # human = SH
    drv = rig.drv
    pre = ai.clone_state(drv.es)
    post = ai.clone_state(drv.es)
    nat = gs.Nation.NORTH if hasattr(gs.Nation, 'NORTH') \
        else next(n for n in gs.Nation if 'north' in
                  str(getattr(n, 'value', n)).lower())
    b0 = int(post.political.boxes.get(nat, 0) or 0)
    post.political.boxes[nat] = b0 + 1
    pid = next((str(p) for p, n in lr.POLITICAL_MARKERS.items()
                if getattr(n, 'name', str(n))
                == getattr(nat, 'name', str(nat))), None)
    if pid is not None:
        drv.view.piece_region[pid] = 'Political %d' % (3 - b0)
    items = drv.post_commit_sweep(pre, post, label='chat:chat')
    if not _findings(drv, 'political_sweep_emitted'):
        fails.append(('flc_h3_no_walk',
                      'chat-commit political delta not walked: '
                      'findings=%r items=%r'
                      % (drv.findings[-4:],
                         [i[0] if isinstance(i, tuple) else 'line'
                          for i in items])))


# ---------------------------------------------------------------------
# FLC H5: every roll ask carries a nonce; card battles announce
# ---------------------------------------------------------------------
def test_flc_h5_step_asks_nonced_and_card_battle_announced(fails):
    """FLC H5 (par5 g4#4 + g1#8): the AI-step directive lane printed
    'roll 3 dice ' with no #N nonce (the window lane nonces every
    ask), and the Fighting Uruk-hai card battle went straight to
    rolls with no declaration chat."""
    import tempfile as _tf
    from wotr_bridge.live.live_server import build
    td = _tf.mkdtemp()
    open(os.path.join(td, 'ai_feed.txt'), 'w').close()
    srv, _bridge = build(td, 'SA', load_nets=False, seed=7)

    class _Call(object):
        kind = 'd6'
        kw = {'need': 3, 'side': Side.FREE_PEOPLES,
              'reason': 'combat'}
    out = srv._dir_lines(_Call())
    said = [s for s in out if 'roll 3 dice' in str(s)]
    if not said:
        fails.append(('flc_h5_control',
                      'NOT-MEASURED: no human roll ask built: %r'
                      % (out,)))
    elif not any('#' in str(s) for s in said):
        fails.append(('flc_h5_no_nonce', [str(s)[:60] for s in said]))
    # the card-battle announce
    pre = adopted_state()
    post = ai.clone_state(pre)
    post.pending_combat = {'attacker_region': 'orthanc',
                           'defender_region': 'helm s deep',
                           'phase': 'attacker_combat_card'}

    class _Act(object):
        action_type = 'play_event'
        params = {}
    out2 = srv._battle_announce_if_new(pre, post, _Act())
    if not out2 or 'deep from orthanc' not in ' '.join(
            str(i) for i in out2):
        fails.append(('flc_h5_no_announce', [str(i)[:60]
                                             for i in out2]))
    if not _findings(srv.driver, 'card_battle_announced'):
        fails.append(('flc_h5_no_finding', srv.driver.findings[-3:]))

    class _Act2(object):
        action_type = 'army_attack'
        params = {}
    if srv._battle_announce_if_new(pre, post, _Act2()):
        fails.append(('flc_h5_double_announce',
                      'army_attack announced twice'))


def main():
    fails = []
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append((t.__name__, 'EXC %r' % (e,)))
        print('[%s] %s' % ('FAIL' if len(fails) > n0 else 'PASS',
                           t.__name__))
        for f in fails[n0:]:
            print('    %r' % (f,))
    print('TOTAL=%d FAIL=%d' % (len(tests), len(fails)))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
