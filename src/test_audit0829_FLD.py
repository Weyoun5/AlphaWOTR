"""FLD regression pins (2026-08-30) - par5b acceptance-fleet BRIDGE fixes.

Charter: PAR5B_SCORECARD items 1-10 + the H1 political-activation
writer hunt (the fleet's only remaining CRITICAL).

Pattern per FIX_BRIEF: plain test_* functions, main() prints
TOTAL=<n> FAIL=<m>, exit 1 on any failure.  Every pin went RED on the
unfixed tree first (RED lines quoted in FLD_FIX_REPORT.md).
"""
import os
import sys
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')

import ai                                                   # noqa: E402
import game_logic as gl                                     # noqa: E402
import game_state as gs                                     # noqa: E402
from game_state import Side, Nation                         # noqa: E402
from board import sanitize                                  # noqa: E402
from test_live_driver import Rig, adopted_state, fresh_state  # noqa: E402
from wotr_bridge.live import client_board as cb             # noqa: E402

FIXTURE = os.path.join(_ROOT, 'data', 'live_fixtures',
                       'client_board_20260821_strider_guide.txt')


def _chats(items):
    return [str(i[2]) for i in items
            if isinstance(i, tuple) and len(i) > 2 and i[0] == 'chat']


def _lines(items):
    return [str(i[1]) for i in items
            if isinstance(i, tuple) and len(i) > 1 and i[0] == 'line']


def _notes(items):
    return [str(i[1]) for i in items
            if isinstance(i, tuple) and len(i) > 1 and i[0] == 'note']


def _findings(drv, kind):
    return [v for k, v in drv.findings if k == kind]


# ---------------------------------------------------------------------
# FLD-H1 (PRIORITY 0, the fleet CRITICAL): the resync adoption lane
# was the off-engine political-activation writer.
# Captured shape: par5b_g2 live_server.out:1524 'resync-adopt:
# political The North: passive (client as-is)' immediately followed by
# the tripwire 'POLITICAL ACTIVATION FLIPPED OUTSIDE ANY COMMIT:
# THE_NORTH' (stall log ai_stall_20260830_040305_T2_nopending.log:72).
# The client counter FACE is never walked by any emission (grep
# Controls.323 over the whole g2 feed = 0 political flips), so the
# dump flag column reads the SETUP face forever and every chat/panel
# resync de-activated the nations the engine had legitimately
# activated.  Invariant: activation is ONE-WAY (log_replay t142,
# rulebook p.33) - adoption may never de-activate, and an activation
# needs a corroborated engine trigger.
# ---------------------------------------------------------------------
def test_fld_h1_resync_never_deactivates(fails):
    st = fresh_state()
    # the engine's legitimate trigger (g2 T2: Shadow army entered Old
    # Forest Road, a North region -> entry activation)
    st.political.active[Nation.THE_NORTH] = True
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.dump_path = FIXTURE          # 258 flag 'U' in the dump
    items = rig.drv.resync_recover(drain_feed=False)
    if not rig.drv.es.political.active.get(Nation.THE_NORTH, False):
        fails.append(('fld_h1_deactivated',
                      'resync adopted the stale client counter and '
                      'DE-ACTIVATED THE_NORTH (the par5b_g2 H1 '
                      'no-commit flip): notes=%r'
                      % [n for n in _notes(items)
                         if 'political' in n.lower()][:4]))
        return
    if not _findings(rig.drv, 'political_activation_kept'):
        fails.append(('fld_h1_no_finding',
                      'the blocked de-activation filed no finding: %r'
                      % rig.drv.findings[-6:]))
    # the client counter is healed: the face walk emits the marker
    # flip (Controls.323 on pid 258) - the client-native activation
    # gesture (log_replay t142 parses it as SET-ACTIVE, idempotent)
    if not any('Controls.323' in l and '258' in l
               for l in _lines(items)):
        fails.append(('fld_h1_no_face_walk',
                      'no activation flip walked to the client: %r'
                      % [l for l in _lines(items)][:6]))


def test_fld_h1_resync_never_activates_untriggered(fails):
    """The par5_g1 direction (Gondor flipped ACTIVE off-engine): a
    client counter on the coloured face with NO engine trigger must
    not activate the engine on adoption."""
    st = fresh_state()
    # the fixture's 256 (Elves) flag is '-'; claim engine-passive to
    # capture the g1 no-trigger direction
    st.political.active[Nation.ELVES] = False
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.dump_path = FIXTURE
    rig.drv.resync_recover(drain_feed=False)
    if rig.drv.es.political.active.get(Nation.ELVES, False):
        fails.append(('fld_h1_activated',
                      'resync adopted an uncorroborated ACTIVE '
                      'counter (the par5_g1 Gondor shape)'))
        return
    if not _findings(rig.drv, 'political_activation_unadopted'):
        fails.append(('fld_h1_no_unadopted_finding',
                      repr(rig.drv.findings[-6:])))


def test_fld_h1_face_walk_is_once_and_one_way(fails):
    """The flip op is a TOGGLE client-side: the face walk must emit at
    most ONCE per nation (a re-emission would flip the counter BACK
    to passive) and never walk the de-activate direction."""
    st = fresh_state()
    st.political.active[Nation.THE_NORTH] = True
    rig = Rig(st, Side.SHADOW)
    rig.drv.view.dump_path = FIXTURE
    rig.drv.view.refresh_dump()
    items1 = rig.drv._political_recon()
    items2 = rig.drv._political_recon()   # dump still lags ('U')
    n1 = sum(1 for l in _lines(items1)
             if 'Controls.323' in l and '258' in l)
    n2 = sum(1 for l in _lines(items2)
             if 'Controls.323' in l and '258' in l)
    if n1 != 1 or n2 != 0:
        fails.append(('fld_h1_face_walk_repeat',
                      'flip emitted %d then %d times (must be 1 then '
                      '0 - a repeat toggles the counter back)'
                      % (n1, n2)))
    # one-way: an engine-passive nation with a client '-' face is
    # never walked back (a de-activate flip would be parsed by t142
    # as an ACTIVATION)
    st2 = fresh_state()
    st2.political.active[Nation.ELVES] = False
    rig2 = Rig(st2, Side.SHADOW)
    rig2.drv.view.dump_path = FIXTURE
    rig2.drv.view.refresh_dump()
    items3 = rig2.drv._political_recon()
    if any('Controls.323' in l and '256' in l for l in _lines(items3)):
        fails.append(('fld_h1_deactivate_walked',
                      'the face walk emitted a de-activate flip'))


# ---------------------------------------------------------------------
# FLD-5 (scorecard item 5, B3): the by-fields die key + the
# n_leaders/take_leaders render override.
# ---------------------------------------------------------------------
class _Row(object):
    def __init__(self, i, action_type, params, die_idx=None):
        self.i = i
        self.action_type = action_type
        self.params = params
        self.die_idx = die_idx


_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')


def _rs():
    lab = os.path.join(_LAB, 'scenarios')
    if lab not in sys.path:
        sys.path.insert(0, lab)
    import run_scenario
    return run_scenario


def test_fld5_by_fields_die_key_filters_die_idx(fails):
    """par5b_g2 re-test 3 (B3 REGRESSION): ANY act carrying a 'die'
    key was refused - the menu prints rows as '#12 army_move die2 ...'
    so the tester's natural by-fields form carries die: N, but the
    matcher filtered it against a.params (which never holds 'die') and
    emptied the candidate list."""
    rs = _rs()
    fn = rs._filter_menu_rows
    rows = [
        _Row(10, 'army_move', {'from': 'erebor', 'to': 'iron hills',
                               'n_regular': 1, 'side': 'fp'},
             die_idx=2),
        _Row(11, 'army_move', {'from': 'erebor', 'to': 'iron hills',
                               'n_regular': 1, 'side': 'fp'},
             die_idx=6),
    ]
    out = fn(list(rows), {'type': 'army_move', 'from': 'erebor',
                          'to': 'iron hills', 'die': 2})
    if [a.i for a in out] != [10]:
        fails.append(('fld5_die_key',
                      "die:2 resolved %r, wanted [10] (the die key "
                      "filters against params and empties the menu)"
                      % ([a.i for a in out],)))
    # die_idx spelt out works the same; a wrong die still refuses
    out2 = fn(list(rows), {'type': 'army_move', 'die_idx': 6})
    if [a.i for a in out2] != [11]:
        fails.append(('fld5_die_idx_key', repr([a.i for a in out2])))
    out3 = fn(list(rows), {'type': 'army_move', 'die': 4})
    if out3:
        fails.append(('fld5_wrong_die_matched',
                      repr([a.i for a in out3])))
    # a die-keyed fellowship_move resolves too (the g2 refusal list)
    rows_f = [_Row(20, 'fellowship_move', {}, die_idx=2)]
    outf = fn(list(rows_f), {'type': 'fellowship_move', 'die': 2})
    if [a.i for a in outf] != [20]:
        fails.append(('fld5_fmove_die', repr([a.i for a in outf])))


def test_fld5_n_leaders_render_parity(fails):
    """par5b_g5#13: the published row n_leaders=1 (take_leaders=True,
    LA-8 per-count variant) rendered a drag moving ALL (2) leaders -
    the planner read only the legacy singular 'n_leader' key, so the
    explicit count was invisible and the take_leaders sweep took over,
    while the engine executor moved exactly 1 (ai.py move_leaders=
    int(n_leaders)).  Render == engine."""
    import emitter
    import log_replay as lr

    class _Rep(object):
        # the erebor SETUP pieces are tracked away so the pool is
        # exactly: 088 regular, 094 elite, 098+309 the two leaders
        piece_region = {'088': 'erebor', '094': 'erebor',
                        '098': 'erebor', '309': 'erebor',
                        '060': 'fp reinf', '061': 'fp reinf',
                        '062': 'fp reinf', '312': 'fp reinf'}
        piece_types = {}
        dynamic_types = {}
        ai_drawn_pids = set()
        state = fresh_state()      # _garrison_dest reads the siege set

    rep = _Rep()

    def _n_leaders_of(pids):
        return sum(1 for p in pids
                   if emitter._unit_type(str(p), rep)[1] == 'leader')

    op, payload = emitter.plan_army_move(
        rep, 'Wey', 'FREE_PEOPLES',
        {'from': 'erebor', 'to': 'iron hills', 'n_regular': 0,
         'n_elite': 1, 'take_leaders': True, 'n_leaders': 1,
         'side': 'fp'})
    n_led = _n_leaders_of(payload['pieces'])
    if n_led != 1:
        fails.append(('fld5_render_overmove',
                      'n_leaders=1 row rendered %d leader(s): %r '
                      '(take_leaders override wins over the '
                      'n_leaders key)' % (n_led, payload['pieces'])))
    # CONTROL: the plain take-all row still sweeps both leaders
    op2, payload2 = emitter.plan_army_move(
        rep, 'Wey', 'FREE_PEOPLES',
        {'from': 'erebor', 'to': 'iron hills', 'n_regular': 0,
         'n_elite': 1, 'take_leaders': True, 'side': 'fp'})
    n_led2 = _n_leaders_of(payload2['pieces'])
    if n_led2 != 2:
        fails.append(('fld5_takeall_control',
                      'take-all rendered %d leader(s): %r'
                      % (n_led2, payload2['pieces'])))


# ---------------------------------------------------------------------
# FLD-2 (scorecard item 2): the stale owed table-card-seat ledger +
# the comma-truncated hand-kill line.
# ---------------------------------------------------------------------
def test_fld2_table_seat_owed_releases(fails):
    """par5b g1#5 / g4#13+#15 (3-for-3) / g5#11: a performed
    table-card seat (the pid visibly on Board1/3/4) never released
    '_table_seat_owed' - NO release path existed - so the census was
    gated blind ('owed: seat table card N on a board slot') for the
    rest of the session."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)          # human = FP
    drv = rig.drv
    drv._table_seat_owed = '204'
    drv._table_seat_said = {'204'}
    drv.view.piece_region = {'204': 'Board3'}
    drv.view.refresh_dump = lambda: True
    drv.reconcile_board_quiet()
    if getattr(drv, '_table_seat_owed', None) is not None:
        fails.append(('fld2_owed_stuck',
                      'the performed seat (Board3) never released '
                      'the owed row: holds=%r'
                      % [h for h in drv.quiescence_holds()
                         if 'table card' in h]))
    # CONTROL: a pid still parked in the transient Current-Card zone
    # stays owed
    drv2 = Rig(fresh_state(), Side.SHADOW).drv
    drv2._table_seat_owed = '204'
    drv2._table_seat_said = {'204'}
    drv2.view.piece_region = {'204': 'Current FP Card'}
    drv2.view.refresh_dump = lambda: True
    drv2.reconcile_board_quiet()
    if getattr(drv2, '_table_seat_owed', None) != '204':
        fails.append(('fld2_control_released',
                      'an unperformed seat was released'))


def test_fld2_comma_card_kill_is_a_hand_discard(fails):
    """par5b g4#6: acting the discard row for 'On, On They Went'
    rendered '$Wey kills On, Shadow card in Shadow Hand.' (the title
    truncated at its comma) - the identified-name hand kill went to
    facts.unmapped, the pending buffer matched nothing, and the AI
    declared itself stuck.  ANY card kill in a hand zone is the
    discard gesture; identity rides the sidecar pid
    (driver._discard_identity_filter), never the rendered name."""
    from wotr_bridge.live import gesture_match as gm

    class _Win(object):
        die_face = None
        die_side = None

        def body_records(self):
            return [{'op': 'kill', 'zone': 'Shadow Hand',
                     'pieces': ['On, Shadow card']}]

    f = gm.window_facts(_Win())
    n = sum(v for (sd, _dk), v in f.hand_discards.items()
            if sd == 'SHADOW')
    if n != 1 or f.unmapped:
        fails.append(('fld2_kill_unmapped',
                      'the truncated card kill produced '
                      'hand_discards=%r unmapped=%r (the pending '
                      'buffer can never match it)'
                      % (dict(f.hand_discards), f.unmapped)))
    # CONTROL: the identity-less deck-label form still counts with
    # its deck
    class _Win2(_Win):
        def body_records(self):
            return [{'op': 'kill', 'zone': 'Shadow Hand',
                     'pieces': ['Shadow Character Card']}]

    f2 = gm.window_facts(_Win2())
    if f2.hand_discards.get(('SHADOW', 'character')) != 1:
        fails.append(('fld2_control_deck_label',
                      repr(dict(f2.hand_discards))))


# ---------------------------------------------------------------------
# FLD-1 (scorecard item 1): the chat-siege seat release/claim
# interlock (g5#17) - the AI refused and force-undid its OWN demanded
# seat drag.
# ---------------------------------------------------------------------
def test_fld1_seat_drag_survives_early_release(fails):
    """par5b_g5#17 (live_server.out:3803-3812): the dump ran AHEAD of
    the feed by 1 line - the BL-76 release saw the sub-box occupied
    and cleared '_siege_seat_owed' BEFORE the seat drag's own feed
    line was consumed; that line then fell to the die-less matcher
    and was refused + force-undone ('no die-less action fits that
    drag'), so the Lorien garrison fought outside the walls.  A
    recently-released seat keeps claiming its drag."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)          # human = FP
    drv = rig.drv
    drv.phase = 'actions'
    drv._siege_seat_owed = 'lorien'
    # the dump already shows the seat performed (ahead of the feed)
    drv.view.piece_region = {'035': 'Lorien Stronghold'}
    drv.view.refresh_dump = lambda: True
    drv.reconcile_board_quiet()
    if getattr(drv, '_siege_seat_owed', None) is not None:
        fails.append(('fld1_precondition',
                      'BL-76 release did not fire on the occupied '
                      'sub-box'))
        return
    # g5's census was CLEAN at the refusal (the report's key fact) -
    # the FLA-10 repair lane must not mask the claim in this fixture
    drv._divergence_open = lambda: False
    # NOW the drag's own feed line arrives
    rig.feed('$Wey moves group (Elven Elite, Elven Regular) from '
             'Lorien to Lorien Stronghold.')
    refused = _findings(drv, 'dieless_gesture_refused')
    if refused:
        fails.append(('fld1_refused',
                      'the released seat drag was refused as a fresh '
                      'die-less gesture: %r' % refused))
    if not _findings(drv, 'siege_seat_performed'):
        fails.append(('fld1_not_claimed',
                      'the drag was not claimed as the seat '
                      'performance: findings=%r' % drv.findings[-4:]))


# ---------------------------------------------------------------------
# FLD-3 (scorecard item 3, B1): a say_pick row with NO physical form
# is acted by SENDING ITS CHAT WORD (the protocol-word lane
# combat_continue/stop rows already ride, g4#10), never a dead
# refusal.  Sites: siege_assault (g3#8/g4#4), combat_effect_choice
# (g4#9/g1#6), victor_stay (g4#10), fp_shine (g5#18),
# hunt_sacrifice_random (g1#6), per-card card_option (g4#12).
# ---------------------------------------------------------------------
def test_fld3_say_pick_act_sends_chat_word(fails):
    rs = _rs()
    hg = rs.hg

    class _Ctx(object):
        def local_nick(self):
            return 'Wey'

    class _Menu(object):
        besieged = ()
        combat = None
        dynamic_types = {}
        fellowship = None

    class _Sc(object):
        human_side_name = 'SHADOW'

    r = rs.Run.__new__(rs.Run)
    r.sc = _Sc()
    r.ctx = _Ctx()
    notes, injected = [], []
    r.note = lambda s: notes.append(str(s))
    r.inject = lambda label, lines: injected.append((label,
                                                     list(lines)))

    def _refuse(act, timeout=30.0):
        raise hg.GestureRefused(
            'row #%s (%s) has no physical form: needs client '
            'randomness first (d6)' % (act.i, act.action_type))
    r._perform_server_gesture = _refuse
    _rd, _dm = hg._read_dump, hg._die_map_from_status
    hg._read_dump = lambda ctx: {}
    hg._die_map_from_status = lambda *a, **k: {}
    try:
        act = _Row(8, 'siege_assault', {'say_pick': 3}, die_idx=None)
        try:
            lines = r._play_menu_action(act, _Menu())
        except hg.GestureRefused as e:
            fails.append(('fld3_refused',
                          'the no-physical-form say_pick row was '
                          'refused instead of sending its chat '
                          'word: %s' % e))
            return
        want = '<Wey> pick 3'
        got = [l for _lb, ls in injected for l in ls] + list(lines)
        if not any(want in str(l) for l in got):
            fails.append(('fld3_wrong_word',
                          'expected %r, injected %r' % (want, got)))
    finally:
        hg._read_dump, hg._die_map_from_status = _rd, _dm


# ---------------------------------------------------------------------
# FLD-4 (scorecard item 4, B2): rows whose OWN gesture cannot match
# their pending.
# ---------------------------------------------------------------------
def test_fld4_choose_guide_own_gesture_matches(fails):
    """par5b_g5#15 (stall ai_stall_20260830_043527_T5_guide_change):
    the engine seats a TEMPORARY guide = best[0] while the tie
    pending is open (game_logic.change_guide allow_pending), so
    picking exactly that candidate is a NO-OP delta and the row's own
    guide-box drag 'matches none of its 3 legal options'.  The
    choose_guide leg's identity IS the engine fact."""
    from wotr_bridge.live import gesture_match as gm
    st = fresh_state()
    st.fellowship.companions = [
        c for c in st.fellowship.companions
        if c.name in ('Legolas', 'Gimli', 'Boromir', 'Meriadoc',
                      'Peregrin')]
    gl.change_guide(st, allow_pending=True)   # temp guide = best[0]
    if getattr(st.fellowship.guide, 'name', '') != 'Legolas':
        fails.append(('fld4_fixture',
                      'temp guide is %r, wanted Legolas'
                      % getattr(st.fellowship.guide, 'name', None)))
        return
    dp = ai.describe_pending(st)

    class _Win(object):
        die_face = None
        die_side = None
        combat_cards_open = False
        records = [{'op': 'move', 'obj': 'Legolas',
                    'from': 'Fellowship Of The Ring',
                    'to': 'Guide Of The Fellowship', 'player': 'Wey',
                    'norm': 'Wey moves Legolas from Fellowship Of '
                            'The Ring to Guide Of The Fellowship.'}]

        def body_records(self):
            return list(self.records)

    res = gm.match_window(st, dp['side'], _Win(), menu=dp['actions'])
    if res.status != 'match':
        fails.append(('fld4_guide_unmatchable',
                      'the row\'s own guide drag matched %r of the '
                      '%d options (the g5 stall)'
                      % (res.status, len(dp['actions']))))
        return
    picked = [x for x in res.chain
              if getattr(x, 'action_type', '') == 'choose_guide']
    if not picked or picked[0].params.get('companion') != 'Legolas':
        fails.append(('fld4_wrong_pick',
                      repr([(getattr(x, 'action_type', x),
                             getattr(x, 'params', None))
                            for x in res.chain])))


def test_fld4_siege_trim_seat_drag_is_follow_through(fails):
    """par5b_g2 siege_trim stall (ai_stall_20260830_042436): the
    window carried the garrison's seat-into-the-box drag (sieges +1)
    for a siege the ENGINE already holds - the trim option moves no
    siege flag, so strict siege parity refused both options and the
    double-execution cascade followed (chat pick + echo re-perform =
    engine=4 client=3)."""
    from wotr_bridge.live import gesture_match as gm
    win = gm.WindowFacts()
    win.sieges['minas tirith'] += 1
    eng = gm.EngFacts()
    ftr = gm.follow_through_for(
        [_Row(0, 'siege_trim', {})], live=True)
    if 'sieges' not in (ftr or ()):
        fails.append(('fld4_no_ft_entry',
                      'siege_trim grants no sieges follow-through: '
                      '%r' % (ftr,)))
    if not gm._covers(win, eng, follow_through=frozenset(('sieges',))):
        fails.append(('fld4_siege_parity_refused',
                      'the seat-into-siege drag is refused even '
                      'under sieges follow-through'))
    # SENSITIVITY: without the follow-through grant the strict parity
    # stands (a real siege-forming gesture still needs its engine
    # siege)
    if gm._covers(win, eng, follow_through=frozenset()):
        fails.append(('fld4_sensitivity',
                      'siege parity vanished without the grant'))


# ---------------------------------------------------------------------
# FLD-6 (scorecard item 6): the D-new resource-integrity chain.
# ---------------------------------------------------------------------
def test_fld6a_expired_deferred_debits_the_killed_die(fails):
    """par5b_g2 ghost die #1: 'DEFERRED WINDOW EXPIRED
    (resync-recovery): FREE_PEOPLES die=P' left the ENGINE die
    unspent while the client had staged+killed it - the engine then
    offered a 5th action from a 4-die pool and the dice resync
    re-minted the client die.  Ruling 1.230: a paid play COMPLETES
    (die spent), never refunded."""
    from wotr_bridge.live.segmenter import Window
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)          # human = FP
    drv = rig.drv
    d = drv.es.dice[Side.FREE_PEOPLES][0]
    d.current = gs.ActionDieFace.EVENT
    d.used = False
    w = Window('FREE_PEOPLES', 'P',
               {'op': 'move', 'obj': 'Free Action Die (P)',
                'die_stage': True,
                'norm': 'Wey moves Free Action Die (P) from FP Dice '
                        'to Current Action.'})
    w.records.append({'op': 'kill',
                      'pieces': ['Free Action Die (P)'],
                      'zone': 'Current Action',
                      'norm': 'Wey kills Free Action Die (P) in '
                              'Current Action.'})
    w.closed = 'die_killed'
    drv.deferred = (w, Side.FREE_PEOPLES)
    drv._expire_deferred('resync-recovery')
    if not d.used:
        fails.append(('fld6a_die_refunded',
                      'the expired deferral left the client-killed '
                      'die UNSPENT engine-side (the 5th-action '
                      'mint): findings=%r' % drv.findings[-3:]))
    if not _findings(drv, 'deferred_die_spent'):
        fails.append(('fld6a_no_finding', repr(drv.findings[-4:])))
    # CONTROL: an expiry whose window did NOT kill its die refunds
    # as before (the die stays playable)
    st2 = fresh_state()
    drv2 = Rig(st2, Side.SHADOW).drv
    d2 = drv2.es.dice[Side.FREE_PEOPLES][0]
    d2.current = gs.ActionDieFace.EVENT
    d2.used = False
    w2 = Window('FREE_PEOPLES', 'P',
                {'op': 'move', 'obj': 'Free Action Die (P)',
                 'die_stage': True, 'norm': 'stage'})
    w2.closed = 'undone'
    drv2.deferred = (w2, Side.FREE_PEOPLES)
    drv2._expire_deferred('test')
    if d2.used:
        fails.append(('fld6a_control',
                      'an undone window spent the engine die'))


def test_fld6b_adopt_defers_while_a_hold_stands(fails):
    """par5b_g2 ghost #2 (T7 Bree, engine=2 client=1 elites): a HELD
    window's lines stand DISPUTED on the client - the consented
    adoption baked the disputed muster into the engine, and the
    gate's later client-side undo left the engine double-applied."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv

    class _W(object):
        records = [{'op': 'move', 'norm': 'Wey moves ...'}]
        _hold_cond = 'divergence'

    drv.holds = [(_W(), None)]
    items = drv._adopt_board_now('consented')
    if not _findings(drv, 'adopt_deferred_owed_demands'):
        fails.append(('fld6b_adopted_over_hold',
                      'the adoption ran over a held (disputed) '
                      'window: %r' % [i[1] for i in items][:3]))


def test_fld6c_stalled_buffer_releases_repair_traffic(fails):
    """par5b_g2 stale-line poisoning: repair drags (the census-'no'
    contract) and self-performed lines were consumed as ANSWERS by
    the next open decision, jamming fp_shine / defender_combat_card /
    casualty_allocation / discard / declare in sequence.  With a
    divergence open, a stalled pending buffer releases its unmatched
    records as repair traffic instead of jamming."""
    st = fresh_state()
    st.fellowship.companions = [
        c for c in st.fellowship.companions
        if c.name in ('Legolas', 'Gimli', 'Boromir', 'Meriadoc',
                      'Peregrin')]
    gl.change_guide(st, allow_pending=True)
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    drv._divergence_open = lambda: True
    # an unrelated repair drag lands in the guide pending's buffer
    drv.pending_buf = drv._new_phase_window()
    drv.pending_buf.records.append(
        {'op': 'move', 'obj': 'Sauron Regular',
         'from': 'Gorgoroth', 'to': 'Minas Morgul',
         'norm': 'Wey moves Sauron Regular from Gorgoroth to '
                 'Minas Morgul.'})
    for _ in range(30):
        drv.tick()
        if _findings(drv, 'repair_drag_accepted'):
            break
    if not _findings(drv, 'repair_drag_accepted'):
        fails.append(('fld6c_jammed',
                      'the stalled buffer kept the repair drag as a '
                      'pending answer: stall=%r buf=%r'
                      % (_findings(drv, 'pending_buffer_stall')[:1],
                         len(getattr(drv.pending_buf, 'records', ())
                             if drv.pending_buf is not None else ()))))
    if drv.pending_buf is not None and drv.pending_buf.records:
        fails.append(('fld6c_buffer_kept',
                      '%d record(s) still jam the pending'
                      % len(drv.pending_buf.records)))


def test_fld6d_bare_yes_with_standing_divergence_consents(fails):
    """par5b_g2#28: an adopt consent 'yes' was answered with the
    DECLINE-routes reply (the ask had already been consumed) while
    the divergence stood - the human's yes IS the consent; only the
    explicit word 'adopt' worked."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    drv._last_quiescent_census = {
        'board': ['units bree the north/elite: engine=2 client=1']}
    items = drv._on_chat({'nick': 'Wey', 'body': 'yes'})
    if not _findings(drv, 'census_adopt_consented'):
        fails.append(('fld6d_yes_declined',
                      "a bare 'yes' with a standing divergence did "
                      'not consent: findings=%r chats=%r'
                      % (drv.findings[-3:], _chats(items)[:2])))
    # CONTROL: 'no' keeps the standing-routes reminder
    drv2 = Rig(fresh_state(), Side.SHADOW).drv
    drv2._last_quiescent_census = {'board': ['units x: e=1 c=0']}
    drv2._on_chat({'nick': 'Wey', 'body': 'no'})
    if not _findings(drv2, 'census_route_reminder'):
        fails.append(('fld6d_no_control',
                      repr(drv2.findings[-3:])))


def test_fld6e_chat_resync_acks(fails):
    """par5b_g5#2: chat 'resync' ran SILENTLY (zero chat ack while
    the diagnostics showed resync_recovery completed)."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    items = drv._on_chat({'nick': 'Wey', 'body': 'resync'})
    if not _chats(items):
        fails.append(('fld6e_silent',
                      'chat resync returned no chat ack: %r'
                      % [i[0] for i in items][:6]))


def test_fld6f_undo_guidance_names_no_redo_button(fails):
    """par5b_g5#2: the unreassertable-undo guidance pointed at a redo
    button the seat lacks."""
    from wotr_bridge.live import chat as chat_mod
    txt = chat_mod.undo_took_ai_line()
    if 'redo' in txt.lower():
        fails.append(('fld6f_redo_button',
                      'guidance names a redo button the seat lacks: '
                      '%r' % txt))
    if 'resync' not in txt.lower():
        fails.append(('fld6f_no_route', repr(txt)))


# ---------------------------------------------------------------------
# FLD-7 (scorecard item 7): the reveal-demand once contract.
# ---------------------------------------------------------------------
def test_fld7_reveal_demand_once_per_30_cycles(fails):
    """par5b_g4#2/#5 (2-3x nags before the auto-reveal landed) /
    g1#10 (6x): _owed_reveal_nag deduped on cycles//30 BUCKETS, so a
    demand said at cycle 29 re-printed at cycle 30 (a bucket
    boundary, back-to-back) instead of 30 cycles later."""
    from wotr_bridge.live.live_server import LiveServer

    class _Drv(object):
        _cc_reveal_owed = {'SHADOW_Strategy_X': 'owed'}

    class _Srv(object):
        driver = _Drv()
        _cycles = 29
        _reveal_nag_key = None

        def _say(self, kind, text):
            return ('chat', kind, text)

    s = _Srv()
    out1 = LiveServer._owed_reveal_nag(s)
    s._cycles = 30
    out2 = LiveServer._owed_reveal_nag(s)
    if out1 and out2:
        fails.append(('fld7_back_to_back',
                      'the reveal demand printed twice one cycle '
                      'apart (the g4 2x nag)'))
    # a full 30 cycles later the fallback re-nag survives (E1: the
    # demand can genuinely go unseen)
    s._cycles = 65
    out3 = LiveServer._owed_reveal_nag(s)
    if not out3:
        fails.append(('fld7_no_fallback',
                      'the 30-cycle fallback re-nag died'))
    # claiming the entry silences it
    s.driver._cc_reveal_owed = {'SHADOW_Strategy_X': 'claimed'}
    s._cycles = 130
    if LiveServer._owed_reveal_nag(s):
        fails.append(('fld7_claimed_nags', 'a claimed reveal nagged'))


# ---------------------------------------------------------------------
# FLD-8 (scorecard item 8): political walks starved by a held census.
# ---------------------------------------------------------------------
def test_fld8_political_walk_survives_owed_hold(fails):
    """par5b_g2 (C3 softened): attack/capture advances applied
    engine-side at the action but the CLIENT marker walked only in
    end-of-turn batches - the census was held ('owed: seat table
    card...' frozen from T3, item 2) and the engine-truth political
    recon ran only on a fully clean census.  An owed PIECE demand
    does not touch the political track: the walk must ride."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    # an owed piece-demand hold (the g2 freezer) - pid parked in the
    # transient zone so the release does not clear it
    drv._table_seat_owed = '204'
    drv._table_seat_said = {'204'}
    eng_box = int(st.political.boxes[Nation.THE_NORTH])
    client_pol = 'Political %d' % (3 - (eng_box + 1))  # one box off
    drv.view.piece_region = {'204': 'Current FP Card',
                             '258': client_pol}
    drv.view.refresh_dump = lambda: True
    items = drv.reconcile_board_quiet()
    if not _findings(drv, 'political_recon'):
        fails.append(('fld8_walk_starved',
                      'the owed-demand hold starved the political '
                      'walk (the g2 end-of-turn batching): notes=%r'
                      % [i[1] for i in items if i[0] == 'note'][:2]))
    # CONTROL: an open DECISION hold still gates the walk (the F17
    # double-walk class)
    st2 = fresh_state()
    st2.fellowship.companions = [
        c for c in st2.fellowship.companions
        if c.name in ('Legolas', 'Gimli', 'Boromir')]
    gl.change_guide(st2, allow_pending=True)
    drv2 = Rig(st2, Side.SHADOW).drv
    drv2.view.piece_region = {'258': client_pol}
    drv2.view.refresh_dump = lambda: True
    drv2.reconcile_board_quiet()
    if _findings(drv2, 'political_recon'):
        fails.append(('fld8_control',
                      'the walk ran over an open decision hold'))


# ---------------------------------------------------------------------
# FLD-9 (scorecard item 9): the randomness-demand robustness set.
# ---------------------------------------------------------------------
def test_fld9_starved_randomness_watchdog(fails):
    """par5b_g2: the whole Woodland siege-assault chain (5+2+2+1
    dice) starved with NO voiced ask; the AI stalled on its OWN owed
    Shadow draw (cleared only by a raw !CALL); g1#12: the extra
    Moria hunt roll carried no nonce and the tile draw starved 90s;
    g5#4/#5: a FED draw stalled the deferred window (race) and the
    resync then half-lost Celeborn's Galadhrim.  A pending that
    outlives its ask event is re-dispatched through the ONE ask lane
    (nonce mint + dedupe) after a short retry."""
    from wotr_bridge.live.live_server import LiveServer

    class _P(object):
        kind = 'd6'
        kw = {'need': 5, 'reason': 'combat'}

    class _Rand(object):
        def __init__(self):
            self.p = _P()

        def pending(self):
            return self.p

    class _Drv(object):
        retried = 0

        def _retry_blocked(self):
            self.retried += 1

    class _Srv(object):
        rand = _Rand()
        driver = _Drv()
        _cycles = 10
        asked = None

        def _window_ask_lines(self, call):
            self.asked = call
            return [('chat', 'roll_request', 'roll 5 dice #9')]

    s = _Srv()
    fn = getattr(LiveServer, '_starved_randomness_watchdog', None)
    if fn is None:
        fails.append(('fld9_no_watchdog',
                      'no starved-randomness watchdog exists - a '
                      'pending with no ask starves forever (the g2 '
                      'siege chain)'))
        return
    out = []
    for i in range(6):
        s._cycles = 10 + i
        out.extend(fn(s) or [])
        if s.asked is not None:
            break
    if s.asked is None:
        fails.append(('fld9_never_asked',
                      'the starved d6 pending was never re-asked'))
    if not s.driver.retried:
        fails.append(('fld9_no_retry',
                      'the fed-randomness retry (_retry_blocked) '
                      'never ran - the g5 fed-draw race stays'))
    # a card_pick pending stays with its own FLA-20 lane
    s2 = _Srv()
    s2.rand = _Rand()
    s2.rand.p = type('X', (), {'kind': 'card_pick', 'kw': {}})()
    s2.driver = _Drv()
    for i in range(6):
        s2._cycles = 10 + i
        fn(s2)
    if s2.asked is not None:
        fails.append(('fld9_card_pick_covered',
                      'card_pick must stay with the FLA-20 lane'))


# ---------------------------------------------------------------------
# FLD-10 (scorecard item 10): announce/template set.
# ---------------------------------------------------------------------
def test_fld10_bare_number_rides_the_published_pickmap(fails):
    """par5b_g5#1: menu row #1 advertised 'pass say_pick=2'; chat '2'
    was SILENTLY consumed against the dp-order index (a different
    numbering than the published menu) with no commit voice - the
    bare number IS the advertised say_pick and must resolve through
    the SAME published identity map as 'pick N', and voice."""
    from wotr_bridge.live import chat as chat_mod
    st = fresh_state()
    st.fellowship.companions = [
        c for c in st.fellowship.companions
        if c.name in ('Legolas', 'Gimli', 'Boromir')]
    gl.change_guide(st, allow_pending=True)
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    dp = __import__('ai').describe_pending(drv.es, drv.human_side)
    acts = dp['actions']
    if [a.params.get('companion') for a in acts] \
            != ['Legolas', 'Gimli', 'Boromir']:
        fails.append(('fld10_fixture', repr(acts)))
        return
    # the PUBLISHED numbering differs from dp order (the g5 shape):
    # published say_pick=3 names the Gimli row
    drv._pick_pub96 = {3: chat_mod.action_identity(acts[1])}
    items = drv._on_chat({'nick': 'Wey', 'body': '3'})
    got = getattr(drv.es.fellowship.guide, 'name', None)
    if got != 'Gimli':
        fails.append(('fld10_wrong_row',
                      "bare '3' committed %r instead of the "
                      'published row #3 (Gimli) - the dp-order index '
                      'lane won' % got))
    if not any('committed' in c.lower() for c in _chats(items)):
        fails.append(('fld10_no_voice',
                      'the bare-number commit voiced nothing: %r'
                      % _chats(items)[:2]))


def test_fld10_companion_chain_template(fails):
    """par5b_g1#4: the FP Book of Mazarbul companion-move chain was
    announced as 'move nazgul on the board' - the template must
    follow the choices' own vocabulary."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    st.pending_card_option = {
        'card_id': '__companion_moves__',
        'card_name': '__companion_moves__',
        'side': 'fp', 'decider': 'fp', 'direct': True,
        'options': [
            {'companions': ['Legolas'], 'from_region': 'rivendell',
             'to_region': 'high pass'},
            {'_comp_done': True}]}
    items = drv._maybe_announce_pending()
    said = ' '.join(_chats(items)).lower()
    if not said:
        fails.append(('fld10_no_announce',
                      'the companion chain announced nothing'))
        return
    if 'nazgul' in said:
        fails.append(('fld10_nazgul_template',
                      'companion chain announced with the nazgul '
                      'template (g1#4): %r' % said[:80]))


def test_fld10_die_removal_renders(fails):
    """par5b_g5#20 (GtW death: engine 4, client 5, phantom die broke
    the next full-pool roll) / g3#14a (WK death: engine=8 client=9
    roll wedge): the recover recalc SHRINKS the engine pool silently
    - the client re-renders the dead character's die with no removal
    line. The engine removal must retain the exact retired membership
    for the native recovery scheduler, which returns it to Reinforcements.
    A kill alone lets Recover Dice resurrect the removed die."""
    st = fresh_state()
    rig = Rig(st, Side.SHADOW)
    drv = rig.drv
    d5 = gs.ActionDie(list(drv.es.dice[Side.FREE_PEOPLES][0].faces),
                      Side.FREE_PEOPLES)
    d5.used = True
    drv.es.dice[Side.FREE_PEOPLES].append(d5)
    for s in (Side.FREE_PEOPLES, Side.SHADOW):
        for d in drv.es.dice[s]:
            d.used = True
    drv.view.die_id_map['358'] = (Side.FREE_PEOPLES, 4)
    items = drv._on_recover({'op': 'recover', 'norm': 'recover'})
    if len(drv.es.dice[Side.FREE_PEOPLES]) != 4:
        fails.append(('fld10e_fixture',
                      'recalc kept %d FP dice'
                      % len(drv.es.dice[Side.FREE_PEOPLES])))
        return
    recovery = [item[1] for item in items if item[0] == 'recover_ai']
    if (len(recovery) != 1 or recovery[0].get('pools', {}).get(
            'FREE_PEOPLES', {}).get('pids', [])[-1:] != ['358']):
        fails.append(('fld10e_retired_membership_lost',
                      'the native recovery scheduler lost retired pid 358'))
    if any('Controls.644' in line and '358' in line for line in _lines(items)):
        fails.append(('fld10e_recoverable_retired_die',
                      'kill would put the retired die back in the recoverable pool'))
    if '358' in (drv.view.die_id_map or {}):
        fails.append(('fld10e_pid_still_bound',
                      'the removed die pid stays bound'))


def test_fld10_discard_by_name_and_guidance(fails):
    """par5b_g5#9 (exact G5 repeat): 'discarding card swords in
    eriador' - the AI's OWN help phrase - was refused for a
    hand-limit discard, and the guidance named only the board lane."""
    from wotr_bridge.live import chat as chat_mod
    rows = [
        _Row(0, 'discard_card',
             {'card_id': 'FREE_PEOPLES_Character_Swords_in_Eriador',
              'say_pick': 1}),
        _Row(1, 'discard_card',
             {'card_id': 'FREE_PEOPLES_Strategy_The_Red_Arrow',
              'say_pick': 2}),
    ]
    act = chat_mod.match_choice('discarding card swords in eriador',
                                rows)
    if act is None or act.i != 0:
        fails.append(('fld10c_refused',
                      "the AI's own help phrase resolved %r"
                      % (None if act is None else act.i)))
    # ambiguity (a fragment matching BOTH names) stays a non-match
    act2 = chat_mod.match_choice('discarding card r', rows)
    if act2 is not None and getattr(act2, 'action_type', '') \
            == 'discard_card':
        fails.append(('fld10c_ambiguous_guessed', repr(act2.i)))
    # the guidance names the chat lanes beside the board one
    st = fresh_state()
    drv = Rig(st, Side.SHADOW).drv
    words = drv._pending_words({'actions': rows})
    if 'pick' not in words or 'discard <card name>' not in words:
        fails.append(('fld10c_guidance', repr(words)))


def test_fld10_turnover_pass_refusal_prints_counts(fails):
    """par5b_g5#3: the turn-over pass refusal omitted the two die
    counts (spec: every pass refusal names the numbers it was judged
    on)."""
    st = fresh_state()
    drv = Rig(st, Side.SHADOW).drv
    drv.phase = 'actions'
    drv._human_out_of_turn = lambda: True
    drv._turn_over_pending = lambda: True
    items = drv._human_plain_pass()
    said = ' '.join(_chats(items))
    if 'unused dice' not in said:
        fails.append(('fld10d_no_counts',
                      'the turn-over pass refusal prints no die '
                      'counts: %r' % said[:100]))


# ---------------------------------------------------------------------
def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    total, failed = 0, 0
    for name, fn in tests:
        total += 1
        fails = []
        try:
            fn(fails)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append(('EXC', repr(e)))
        if fails:
            failed += 1
            print('[FAIL] %s' % name)
            for f in fails:
                print('    %r' % (f,))
        else:
            print('[PASS] %s' % name)
    print('TOTAL=%d FAIL=%d' % (total, failed))
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
