"""Audit 2026-08-29 - SQF fix-wave pins (seqack_fp1 agent-seat live
validation findings, session evidence in
C:/Users/sarve/Downloads/WOTR_AI_training_lab/agent_seat/sessions/seqack_fp1).

  SQF-1  (findings 1+8)  the witness cadence's census ITEMS were
         extended RAW into the transport RESP line list - the tuple
         broke the transport's '\\n'.join with
         TypeError('sequence item 0: expected str instance, tuple
         found') and the held note was LOST.  Items must ride
         _dispatch like every driver item.
  SQF-10 (finding 10, the divergence)  a chat/pick-committed decision
         whose ENGINE effect is a pure NAMED-figure relocation
         (Gwaihir: separated Legolas high pass -> dimrill dale)
         banked NO drag demand and emitted NO line - three census
         rows with 'owed: none'.  The named lane gets the unit lane's
         chat-relocation rule: bank named_move + say the drag.
  SQF-9  (finding 9)  the act-by-index render probe held on 'hand
         row: FREE_PEOPLES gained HU_... with no client-drawn pid'
         after the demanded redraw was fed - a HUMAN-side placeholder
         gain IS the client's own (blinded) draw and is consumed.
  SQF-11 (finding 11)  the census adopt-ask's 'no' was a dead end:
         the decline reply named only GUI-panel routes, a later 'yes'
         fell to 'not a command', and no chat verb could ever accept
         the adoption.  The decline now names the standing chat
         routes and 'adopt' consents at any time.
  SQF-5  (finding 5)  a card_option answered PHYSICALLY (menu act /
         board gesture) committed with no 'your pick is committed'
         voice line - the pending commit lane now voices like the
         chat pick lane (one description builder).
  SQF-6  (findings 6+7 root)  the client's frecover parked a GRANTED,
         engine-unspent die in a reinforcements box (the Controls.784
         artifact, die 212 'from Minhiriath'); the driver consumed it
         silently and the next roll came up 3/4.  The dice-
         bookkeeping lane now demands the counter-drag by pid.
  SQF-23 (findings 2+3, VERDICT pin - regression-only, never RED)
         requirement-gating is the 1.230(a) legal gate: Challenge of
         the King is absent while Strider is not with an FP Army in
         Gondor/Rohan and present once he is.

ASCII only.  RED-first: every fix pin failed on the unfixed tree
(lines recorded in SQF_FIX_REPORT.md).
"""
import json
import os
import sys
import tempfile
import time

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
from test_live_driver import Rig, adopted_state, fresh_state  # noqa: E402


def _chats(items):
    return [str(i[2]) for i in items
            if isinstance(i, tuple) and i and i[0] == 'chat']


def _put_legolas_on_board(st, region):
    comp = next(c for c in st.fellowship.companions
                if getattr(c, 'name', '') == 'Legolas')
    st.fellowship.companions.remove(comp)
    gl.get_or_create_army(st, region).leaders.append(comp)
    return comp


def _move_named(state, name, frm, to):
    a = next(ar for k, ar in state.armies.items()
             if sanitize(str(k)) == sanitize(frm))
    c = next(l for l in a.leaders if getattr(l, 'name', '') == name)
    a.leaders.remove(c)
    gl.get_or_create_army(state, to).leaders.append(c)


# ---------------------------------------------------------------------
# SQF-1: cadence items must survive the transport join
# ---------------------------------------------------------------------
def test_sqf1_witness_cadence_items_ride_dispatch():
    from test_audit0829_BRCMP import _server_rig
    srv, drv, rows, path = _server_rig(seed=31)
    # walk the turn start until no AI step is open (the session's own
    # idle declare window): human draws -> AI draws fed -> declare
    with open(os.path.join(os.path.dirname(path), 'ai_feed.txt'),
              'a', encoding='utf-8') as f:
        f.write('$Wey draws Free Character Card.\n'
                '$Wey draws Free Strategy Card.\n')
    srv.handle({'lines': ()})
    if drv.es.sh_char_deck:
        srv.rand.feed('card', ('SHADOW', 'character',
                               drv.es.sh_char_deck[-1]))
    srv.handle({'lines': ()})
    if drv.es.sh_strat_deck:
        srv.rand.feed('card', ('SHADOW', 'strategy',
                               drv.es.sh_strat_deck[-1]))
    srv.handle({'lines': ()})
    srv._open_step = None
    srv._step_queue = []
    # a quiescence hold so the cadence yields a census-held ITEM
    drv.owed_die_kills.append(('212', 'Current Action'))
    drv._census_held_said = None
    srv._cycles = 29             # next cycle is the %30 cadence tick
    lines = srv.handle({'lines': ()})
    try:
        '\n'.join(lines)
    except TypeError as e:
        raise AssertionError(
            'SQF-1 RED: the transport join over the RESP raises %r '
            '(non-string items: %r)'
            % (e, [x for x in lines if not isinstance(x, str)][:2]))
    bad = [x for x in lines if not isinstance(x, str)]
    assert not bad, 'SQF-1: non-string RESP items: %r' % (bad[:2],)
    assert any('census held' in str(n) for n in srv.notes), \
        'SQF-1: the census-held note never reached the notes: %r' \
        % (srv.notes[-4:],)
    print('[PASS] test_sqf1_witness_cadence_items_ride_dispatch')


# ---------------------------------------------------------------------
# SQF-10: the chat-committed named relocation banks the drag
# ---------------------------------------------------------------------
def test_sqf10_chat_named_relocation_banks_the_drag():
    st = adopted_state()
    _put_legolas_on_board(st, 'High Pass')
    rig = Rig(st, Side.SHADOW, phase='actions')      # human = FP
    drv = rig.drv

    def _gwaihir_effect(clone):
        _move_named(clone, 'Legolas', 'High Pass', 'Dimrill Dale')

    items = drv._commit_human(_gwaihir_effect, via='chat')
    row = next((e for e in drv.echo_ledger
                if e.get('kind') == 'named_move'), None)
    assert row is not None, (
        'SQF-10 RED: chat-committed Legolas relocation banked no drag '
        'demand (ledger %r, items %r)'
        % (drv.echo_ledger, items[:4]))
    assert row.get('tkey') == 'Legolas', row
    assert sanitize(str(row.get('region'))) == 'high pass', row
    assert sanitize(str(row.get('to_region'))) == 'dimrill dale', row
    assert any('dimrill dale' in c.lower() for c in _chats(items)), (
        'SQF-10: the drag was banked silently (BL-43): %r'
        % (_chats(items),))
    holds = drv.quiescence_holds()
    assert any('Legolas' in h and 'dimrill dale' in h.lower()
               for h in holds), (
        'SQF-10: the census does not hold on the owed drag: %r'
        % (holds,))
    # the human's own drag is the performance - consumed as the echo,
    # never a fresh gesture (the session's missing follow-through)
    rec = {'op': 'move', 'from': 'High Pass', 'to': 'Dimrill Dale',
           'obj': 'Legolas', 'player': 'Wey',
           'norm': 'Wey moves Legolas from High Pass to Dimrill Dale.'}
    assert drv._consume_mirror_echo(rec) is True, \
        'SQF-10: the drag did not close the named_move demand'
    assert not [e for e in drv.echo_ledger
                if e.get('kind') == 'named_move'], drv.echo_ledger
    print('[PASS] test_sqf10_chat_named_relocation_banks_the_drag')


# ---------------------------------------------------------------------
# SQF-9: the render probe accepts the client's own blinded draw
# ---------------------------------------------------------------------
def test_sqf9_placeholder_hand_gain_never_holds_the_render():
    from wotr_bridge.live import emission_card as emc
    st = adopted_state()
    _put_legolas_on_board(st, 'High Pass')
    pre = ai.clone_state(st)
    pre.pending_card_option = {
        'card_id': 'FREE_PEOPLES_Character_Gwaihir_the_Windlord',
        'card_name': 'Gwaihir the Windlord', 'options': []}
    post = ai.clone_state(pre)
    _move_named(post, 'Legolas', 'High Pass', 'Dimrill Dale')
    # the engine drew the human a card: blinded placeholder identity
    post.hands[Side.FREE_PEOPLES].hand.append('HU_FR_CHAR_15')
    rig = Rig(fresh_state(), Side.SHADOW)
    view = rig.drv.view
    view.piece_region['242'] = 'High Pass'

    class _A(object):
        action_type = 'card_option'
        die_idx = -1
        params = {'card_id':
                  'FREE_PEOPLES_Character_Gwaihir_the_Windlord'}

    em = emc.render_card_effect(view, 'WOTR_AI_Shadow',
                                Side.FREE_PEOPLES, 'card_option',
                                _A(), pre, post)
    assert not em.hold, 'SQF-9 RED: the render held: %s' % (em.hold,)
    assert any('242' in str(l) for l in (em.lines or [])), (
        'SQF-9: no Legolas (pid 242) drag rendered: %r'
        % (em.lines,))
    # SENSITIVITY: a NON-placeholder gain with no client pid still
    # holds (a real identity cannot appear without a client draw)
    post2 = ai.clone_state(pre)
    post2.hands[Side.FREE_PEOPLES].hand.append(
        'FREE_PEOPLES_Character_Elven_Rope')
    em2 = emc.render_card_effect(view, 'WOTR_AI_Shadow',
                                 Side.FREE_PEOPLES, 'card_option',
                                 _A(), pre, post2)
    assert em2.hold and 'funnel draw' in str(em2.hold), (
        'SQF-9 sensitivity: a real-cid gain with no pid must still '
        'hold: %r' % (em2.hold,))
    print('[PASS] test_sqf9_placeholder_hand_gain_never_holds_the_render')


# ---------------------------------------------------------------------
# SQF-11: the declined census ask keeps a CLI-reachable route
# ---------------------------------------------------------------------
def test_sqf11_declined_census_keeps_a_chat_route():
    from test_audit0829_BRCMP import (_diverged_rig, _read, _age_streak,
                                      _kinds)
    rig, drv, rows, path, info = _diverged_rig()
    frm, to = info['frm'], info['to']
    _read(drv)
    _age_streak(drv)
    _read(drv)
    assert drv._repair is not None \
        and drv._repair.get('kind') == 'census', drv._repair
    out_no = rig.feed('<Wey> no')
    assert drv._repair is None
    assert any('adopt' in c for c in _chats(out_no)), (
        'SQF-11 RED: the census decline names no chat route: %r'
        % (_chats(out_no),))
    _read(drv)      # books the decline (census_adopt_declined)
    # a later bare 'yes' names the standing routes - never a dead end
    out_yes = rig.feed('<Wey> yes')
    assert any('adopt' in c for c in _chats(out_yes)), (
        'SQF-11 RED: a later yes is a dead end: %r'
        % (_chats(out_yes),))
    # 'adopt' consents at any time: the one transactional adoption
    out_ad = rig.feed('<Wey> adopt')
    assert 'census_adopt_consented' in _kinds(drv), (
        'SQF-11 RED: chat adopt did not consent: %r / %r'
        % (_kinds(drv)[-6:], _chats(out_ad)))
    assert 'board_adopt' in _kinds(drv) \
        or any('adopt' in k for k in _kinds(drv)), _kinds(drv)
    out_clean = _read(drv)
    lq = drv._last_quiescent_census or {}
    assert not (lq.get('board') or ()), (
        'SQF-11: rows remain after the consented adoption: %r / %r'
        % (lq, out_clean))
    # 'adopt' with agreeing boards is refused, never a silent no-op
    out_ad2 = rig.feed('<Wey> adopt')
    assert any('agree' in c for c in _chats(out_ad2)), _chats(out_ad2)
    print('[PASS] test_sqf11_declined_census_keeps_a_chat_route')


# ---------------------------------------------------------------------
# SQF-5: the physically answered card_option pick is voiced
# ---------------------------------------------------------------------
def test_sqf5_pending_card_option_commit_is_voiced():
    import event_effects as ee
    st = fresh_state()
    st.fellowship.progress = 1
    faces = [(Side.SHADOW, i, f) for i, f in enumerate(
        (gs.ActionDieFace.EVENT, gs.ActionDieFace.ARMY,
         gs.ActionDieFace.MUSTER, gs.ActionDieFace.CHARACTER,
         gs.ActionDieFace.EYE, gs.ActionDieFace.MUSTER_ARMY,
         gs.ActionDieFace.ARMY))]
    faces += [(Side.FREE_PEOPLES, i, f) for i, f in enumerate(
        (gs.ActionDieFace.CHARACTER, gs.ActionDieFace.WILL_OF_WEST,
         gs.ActionDieFace.MUSTER, gs.ActionDieFace.EVENT))]
    gl.adopt_rolled_faces(st, faces)
    cid = None
    for dk in ('sh_char_deck', 'sh_strat_deck'):
        for c in list(getattr(st, dk, [])):
            if 'Nazg' in str(c) and 'Search' in str(c):
                cid = c
                getattr(st, dk).remove(c)
        if cid:
            break
    assert cid, 'fixture: Nazgul Search not in the shadow decks'
    rig = Rig(st, Side.FREE_PEOPLES, phase='actions')  # human = SH
    drv = rig.drv
    drv.es.hands[Side.SHADOW].hand.append(cid)
    res = ee.play_event_card(drv.es, Side.SHADOW, cid)
    assert res.get('success'), res
    dp = ai.describe_pending(drv.es, Side.SHADOW)
    assert dp is not None and str(dp['kind']) == 'card_option', dp
    row = dp['actions'][0]
    n0 = len(drv._out_queue)
    ok = drv._commit_chain([row], [], 'pending:card_option')
    assert ok, 'fixture: the card_option row did not commit'
    voiced = [str(i[2]) for i in drv._out_queue[n0:]
              if isinstance(i, tuple) and i and i[0] == 'chat'
              and 'your pick is committed' in str(i[2])]
    assert voiced, (
        'SQF-5 RED: the physically answered card_option committed '
        'with no voice line (queue tail %r)'
        % (drv._out_queue[n0:],))
    assert 'card_option' in voiced[0], voiced
    print('[PASS] test_sqf5_pending_card_option_commit_is_voiced')


# ---------------------------------------------------------------------
# SQF-6: the parked-die recover artifact is loud + demanded back
# ---------------------------------------------------------------------
def test_sqf6_die_parked_in_reinforcements_is_demanded_back():
    st = adopted_state()
    rig = Rig(st, Side.SHADOW, phase='draws')        # human = FP
    drv = rig.drv
    drv.view.die_id_map['212'] = (Side.FREE_PEOPLES, 2)
    # the session's own line (seqack_fp1 feed line 117) + its sidecar
    # record (meta seq 117: Controls.784 Game.1883 -> Game.2038,
    # pids ['212'])
    from wotr_bridge.live.sidecar import SidecarReader
    mp = rig.path + '.meta.jsonl'
    with open(mp, 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'seq': 1, 'gen': 1, 'nick': 'Wey', 'pids': ['212'],
            'tags': ['Controls.784', 'Controls.785', 'Game.1883',
                     'Controls.786', 'Game.2038']}) + '\n')
    drv.sidecar = SidecarReader(mp)
    drv.sidecar.poll()
    out = rig.feed('$Wey moves Free Action Die (M) from Minhiriath '
                   'to FP Reinforcements.')
    kinds = [k for k, _ in drv.findings]
    assert 'die_parked_in_reinforcements' in kinds, (
        'SQF-6 RED: the parked live die was consumed silently '
        '(findings %r, out %r)' % (drv.findings[-4:], out[:4]))
    assert any('212' in c for c in _chats(out)), (
        'SQF-6: the counter-drag demand names no pid: %r'
        % (_chats(out),))
    # SPECIFICITY: a SPENT die parked in reinforcements stays silent
    # (legitimate client bookkeeping of a revoked/idle piece)
    drv.view.die_id_map['213'] = (Side.FREE_PEOPLES, 3)
    drv.es.dice[Side.FREE_PEOPLES][3].used = True
    with open(mp, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'seq': 2, 'gen': 1, 'nick': 'Wey', 'pids': ['213'],
            'tags': ['Controls.784', 'Controls.785', 'Game.1883',
                     'Controls.786', 'Game.2038']}) + '\n')
    drv.sidecar.poll()
    n_before = kinds.count('die_parked_in_reinforcements')
    rig.feed('$Wey moves Free Action Die (P) from Minhiriath '
             'to FP Reinforcements.')
    kinds2 = [k for k, _ in drv.findings]
    assert kinds2.count('die_parked_in_reinforcements') == n_before, (
        'SQF-6 specificity: a spent die tripped the demand: %r'
        % (drv.findings[-3:],))
    print('[PASS] test_sqf6_die_parked_in_reinforcements_is_demanded_back')


# ---------------------------------------------------------------------
# SQF-23 (VERDICT pin, regression-only - never RED): 1.230(a)
# requirement-gating is the one legal gate on play_event
# ---------------------------------------------------------------------
def test_sqf23_requirement_gate_is_the_1230_boundary():
    cid = 'FREE_PEOPLES_Character_Challenge_of_the_King'
    st = adopted_state()             # FP die0 = CHARACTER
    st.hands[Side.FREE_PEOPLES].hand.append(cid)

    def _rows(state):
        menu = ai.get_legal_actions(state, Side.FREE_PEOPLES)
        return [a for a in menu if a.action_type == 'play_event'
                and a.params.get('card_id') == cid]
    # Strider in the Fellowship: the card's own play condition
    # ('Strider/Aragorn is with a Free Peoples Army in Gondor or
    # Rohan') is unmet -> legally absent (1.230(a): only
    # check_event_requirement gates)
    assert not _rows(st), (
        'SQF-23: Challenge of the King offered with its requirement '
        'unmet: %r' % (_rows(st),))
    # Strider separated with the Minas Tirith garrison: present
    comp = next(c for c in st.fellowship.companions
                if getattr(c, 'name', '') == 'Strider')
    st.fellowship.companions.remove(comp)
    gl.get_or_create_army(st, 'Minas Tirith').leaders.append(comp)
    assert _rows(st), (
        'SQF-23: Challenge of the King absent with its requirement '
        'met (a real prune)')
    print('[PASS] test_sqf23_requirement_gate_is_the_1230_boundary')


def main():
    tests = [
        test_sqf1_witness_cadence_items_ride_dispatch,
        test_sqf10_chat_named_relocation_banks_the_drag,
        test_sqf9_placeholder_hand_gain_never_holds_the_render,
        test_sqf11_declined_census_keeps_a_chat_route,
        test_sqf5_pending_card_option_commit_is_voiced,
        test_sqf6_die_parked_in_reinforcements_is_demanded_back,
        test_sqf23_requirement_gate_is_the_1230_boundary,
    ]
    fail = 0
    for t in tests:
        try:
            t()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            fail += 1
            print('[FAIL] %s: %r' % (t.__name__, e))
    print('TOTAL=%d FAIL=%d' % (len(tests), fail))
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
