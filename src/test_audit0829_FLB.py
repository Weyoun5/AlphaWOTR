#!/usr/bin/env python3
# -*- coding: ascii -*-
"""FLB regression pins (2026-08-30) - FLA_HANDOFF actionable items.

FLB-1  (FLA_HANDOFF 1, par5 g2#3, ruling 1.230): the Scouts PRE-BATTLE
       defender retreat menu must publish the decline row
       (combat_no_retreat) exactly as the mid-battle retreat menus do -
       the chat lane's 'stand' resolves combat_no_retreat FROM the
       menu, and the published help says 'retreat <region> / stand'.
       A committed retreat card left standing is a zero-effect play
       (1.230: completes, never refunded).

FLB-2  (FLA_HANDOFF 2b, par5 g2#2): the seat's by-fields matcher
       (run_scenario._filter_menu_rows) accepts BOTH leader-count
       vocabularies the engine publishes by design (ai.py:5444-5485,
       encoding-pinned): n_leaders=0 == take_leaders=False, and
       n_leaders=K matches the take-all row (take_leaders=True, no
       explicit n_leaders) exactly when K is the count that row moves.

FLB-3  (FLA_HANDOFF 3, par5 g3 T6): agent_seat's per-turn roll ledger
       (_rolled_pids) is keyed by the PUBLISHED status turn (cleared
       whenever it changes - a rollover the seat did not drive through
       t_recover must not refuse the auto-pool), and a pid the table
       shows un-faced+un-used again AFTER having shown it faced/used
       is dropped (the client is the face authority; an undo un-faced
       it).  CONTROL: a rolled pid the table has NEVER shown faced
       stays ledgered (the shgame6 S1 table-lag guard).

FLB-4  (FLA_HANDOFF 4, par5 g3#6): adopt.adopt_minions' object finder
       recognizes BOTH board-minion type names ('Minion' - the
       game_logic wrapper - and 'ShadowMinion' - the character record
       some paths append directly); the WK dump row (pid 253) must
       relocate a ShadowMinion-typed engine object too, not silently
       stay report-only.

Run:  python test_audit0829_FLB.py     (exit 0 = all pass)
"""

import os
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
_LAB = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab')
for _p in (_SRC, _ROOT, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random

import ai                                    # noqa: E402
import board                                 # noqa: E402
import game_logic                            # noqa: E402
import game_state as gs                      # noqa: E402
from board import sanitize                   # noqa: E402
from game_state import (Side, Nation, UnitKind, Unit, Leader,  # noqa: E402
                        ActionDieFace as Face)

FP = Side.FREE_PEOPLES
SH = Side.SHADOW

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def setup_game():
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for attr in ('_declare_phase', '_eye_allocation_phase'):
        setattr(st, attr, False)
    st.pending_discard = None
    return st


def set_dice(st, side, faces):
    st.dice[side] = []
    for f in faces:
        d = gs.ActionDie(faces=[f], side=side)
        d.current = f
        st.dice[side].append(d)


def place(st, rk, nation, regulars=0, elites=0, leaders=0, clear=False):
    a = game_logic.get_or_create_army(st, rk)
    if clear:
        a.units.clear()
        a.leaders.clear()
    for _ in range(regulars):
        a.units.append(Unit(nation=nation, kind=UnitKind.REGULAR))
    for _ in range(elites):
        a.units.append(Unit(nation=nation, kind=UnitKind.ELITE))
    for _ in range(leaders):
        a.leaders.append(Leader(nation=nation))
    return a


def _gondor_units_by_region(st, skip_san):
    out = {}
    for k, a in st.armies.items():
        ks = sanitize(str(k))
        if ks == skip_san:
            continue
        n = sum(1 for u in a.units if u.nation == Nation.GONDOR)
        if n:
            out[ks] = n
    return out


# ---------------------------------------------------------------------
# FLB-1: Scouts pre-battle retreat menu publishes the decline row
# ---------------------------------------------------------------------
@test
def test_flb1_scouts_prebattle_decline_row():
    """G2#3 (T3 Osgiliath): the pre-battle Scouts retreat published
    ONLY combat_retreat rows while the help said 'retreat <region> /
    stand' - 'stand' resolves combat_no_retreat from THIS menu (chat.
    match_choice), so the decline row must be published exactly as the
    mid-battle defender_retreat menu publishes it.  Committing it
    stands the army (no retreat move) and the battle proceeds."""
    random.seed(11)
    st = setup_game()
    st.political.boxes[Nation.SAURON] = 3
    st.political.active[Nation.SAURON] = True
    place(st, 'North Ithilien', Nation.SAURON, regulars=5, elites=1,
          clear=True)
    place(st, 'Osgiliath', Nation.GONDOR, regulars=3, clear=True)
    ra = 'FREE_PEOPLES_Strategy_The_Red_Arrow'
    if ra in st.fp_strat_deck:
        st.fp_strat_deck.remove(ra)
    if ra not in st.hands[FP].hand:
        st.hands[FP].hand.append(ra)
    set_dice(st, SH, [Face.ARMY])
    set_dice(st, FP, [Face.ARMY])

    atk = [a for a in ai.get_legal_actions(st, SH)
           if a.action_type == 'army_attack'
           and sanitize(str(a.params.get('to', ''))) == 'osgiliath']
    assert atk, 'precondition: no army_attack on osgiliath enumerated'
    ai._execute_action(st, atk[0])
    assert (st.pending_combat or {}).get('phase') == 'attacker_combat_card', \
        'precondition: phase=%r' % ((st.pending_combat or {}).get('phase'),)
    dec = [a for a in ai.get_legal_actions(st, SH)
           if a.action_type == 'combat_card_none']
    assert dec, 'precondition: no attacker decline row'
    ai._execute_action(st, dec[0])
    play = [a for a in ai.get_legal_actions(st, FP)
            if a.action_type == 'combat_card_play'
            and a.params.get('card_id') == ra]
    assert play, 'precondition: Red Arrow (Scouts) not on the ' \
        'defender combat-card menu'
    ai._execute_action(st, play[0])

    pc = st.pending_combat
    assert pc is not None and pc.get('phase') == 'defender_retreat', \
        'precondition: Scouts pre-battle retreat did not open: %r' \
        % ((pc or {}).get('phase'),)
    assert pc.get('_card_retreat_mandatory'), \
        'precondition: the card-retreat window flag is missing'
    menu = ai.get_legal_actions(st, FP)
    types = [a.action_type for a in menu]
    assert any(t == 'combat_retreat' for t in types), \
        'precondition: no retreat destinations published: %r' % (types,)
    # THE PIN (RED on the unfixed tree): the decline row is published
    stands = [a for a in menu if a.action_type == 'combat_no_retreat']
    assert stands, \
        'FLB-1 RED: Scouts pre-battle retreat menu published NO ' \
        'combat_no_retreat decline row (only %r) - chat \'stand\' ' \
        'resolves nothing' % (sorted(set(types)),)

    # committing the stand: no retreat move, window consumed
    before = _gondor_units_by_region(st, 'osgiliath')
    ai._execute_action(st, stands[0])
    pc2 = st.pending_combat
    assert pc2 is None or pc2.get('phase') != 'defender_retreat', \
        'the stand did not consume the retreat window: %r' \
        % (pc2.get('phase'),)
    if pc2 is not None:
        assert not pc2.get('_card_retreat_mandatory'), \
            'the mandatory flag survived the stand'
    after = _gondor_units_by_region(st, 'osgiliath')
    assert after == before, \
        'the stand MOVED defender units (a retreat happened): ' \
        'before=%r after=%r' % (before, after)


# ---------------------------------------------------------------------
# FLB-2: by-fields matcher accepts both leader-count vocabularies
# ---------------------------------------------------------------------
class _Row(object):
    def __init__(self, i, action_type, params, die_idx=None):
        self.i = i
        self.action_type = action_type
        self.params = params
        self.die_idx = die_idx


def _rs():
    lab = os.path.join(_LAB, 'scenarios')
    if lab not in sys.path:
        sys.path.insert(0, lab)
    import run_scenario
    return run_scenario


@test
def test_flb2_n_leaders_take_leaders_equivalence():
    """G2#2: an army_move given by fields with n_leaders=1 must match
    the row carrying take_leaders=True when that row moves all (=1)
    leaders; take_leaders=False == n_leaders=0.  With per-count
    sibling rows published, the take-all row's count is max(sibling
    n_leaders)+1 - the spec resolves exactly ONE row, never both."""
    rs = _rs()
    fn = getattr(rs, '_filter_menu_rows', None)
    assert fn is not None, 'shared matcher missing (SQG-2 regressed)'
    # G2's shape: ONE leader at the source - the mover row is
    # take_leaders=True with NO explicit n_leaders
    rows1 = [
        _Row(10, 'army_move', {'from': 'edoras', 'to': 'westemnet',
                               'n_regular': 1, 'n_elite': 1,
                               'take_leaders': True, 'side': 'fp'}),
        _Row(11, 'army_move', {'from': 'edoras', 'to': 'westemnet',
                               'n_regular': 1, 'n_elite': 1,
                               'take_leaders': False, 'side': 'fp'}),
    ]
    spec = {'type': 'army_move', 'from': 'edoras', 'to': 'westemnet',
            'n_regular': 1, 'n_elite': 1, 'n_leaders': 1}
    out = fn(list(rows1), spec)
    assert [a.i for a in out] == [10], \
        'FLB-2 RED: n_leaders=1 resolved %r, wanted [10] (the ' \
        'take_leaders=True all-leaders row is invisible to the ' \
        'n_leaders vocabulary)' % ([a.i for a in out],)
    # vice versa: n_leaders=0 == take_leaders=False
    out0 = fn(list(rows1), {'type': 'army_move', 'from': 'edoras',
                            'to': 'westemnet', 'n_leaders': 0})
    assert [a.i for a in out0] == [11], \
        'FLB-2 RED: n_leaders=0 resolved %r, wanted [11]' \
        % ([a.i for a in out0],)


@test
def test_flb2_disambiguation_and_negative_legs():
    """Sensitivity: with 3 leaders the per-count rows (1, 2) exist and
    the take-all row implies 3 - each n_leaders value resolves exactly
    its one row; an impossible count resolves NOTHING (loud refusal
    preserved); explicit-n_leaders matching is unchanged."""
    rs = _rs()
    fn = rs._filter_menu_rows
    rows = [
        _Row(20, 'army_move', {'from': 'lorien', 'to': 'dimrill dale',
                               'n_regular': 2, 'n_elite': 0,
                               'take_leaders': True}),
        _Row(21, 'army_move', {'from': 'lorien', 'to': 'dimrill dale',
                               'n_regular': 2, 'n_elite': 0,
                               'take_leaders': True, 'n_leaders': 1}),
        _Row(22, 'army_move', {'from': 'lorien', 'to': 'dimrill dale',
                               'n_regular': 2, 'n_elite': 0,
                               'take_leaders': True, 'n_leaders': 2}),
        _Row(23, 'army_move', {'from': 'lorien', 'to': 'dimrill dale',
                               'n_regular': 2, 'n_elite': 0,
                               'take_leaders': False}),
    ]

    def ids(spec):
        return [a.i for a in fn(list(rows), spec)]

    base = {'type': 'army_move', 'from': 'lorien', 'to': 'dimrill dale'}
    assert ids(dict(base, n_leaders=1)) == [21], ids(dict(base,
                                                          n_leaders=1))
    assert ids(dict(base, n_leaders=2)) == [22], ids(dict(base,
                                                          n_leaders=2))
    assert ids(dict(base, n_leaders=3)) == [20], \
        'take-all row (implied 3) not resolved: %r' \
        % (ids(dict(base, n_leaders=3)),)
    assert ids(dict(base, n_leaders=0)) == [23], ids(dict(base,
                                                          n_leaders=0))
    assert ids(dict(base, n_leaders=4)) == [], \
        'an impossible leader count must resolve NOTHING: %r' \
        % (ids(dict(base, n_leaders=4)),)
    # take_leaders stays a plain top-level filter
    assert ids(dict(base, take_leaders=False)) == [23]


# ---------------------------------------------------------------------
# FLB-3: the seat roll ledger is turn-keyed + face-authority pruned
# ---------------------------------------------------------------------
class _FakeCtx(object):
    pass


class _FakeRun(object):
    def __init__(self):
        self.ctx = _FakeCtx()
        self.payload = {}

    def status(self):
        return self.payload


class _Refused(Exception):
    pass


class _FakeMenu(object):
    def __init__(self, dice):
        self.dice = dice


class _FakeHG(object):
    GestureRefused = _Refused

    def __init__(self, run):
        self._run = run
        self.calls = []

    def read_status_menu(self, ctx):
        return _FakeMenu((self._run.payload.get('human_menu') or {})
                         .get('dice') or [])

    def human_roll(self, ctx, pids=None):
        self.calls.append(list(pids) if pids else pids)
        return ['injected']


def _seat():
    seat_dir = os.path.join(_LAB, 'agent_seat')
    scen_dir = os.path.join(_LAB, 'scenarios')
    audit_dir = os.path.join(_LAB, 'audit')
    for p in (seat_dir, scen_dir, audit_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import agent_seat as AS
    s = AS.Seat.__new__(AS.Seat)
    run = _FakeRun()
    s.run = run
    s.hg = _FakeHG(run)
    return s, run


def _dice(*rows):
    """rows: (pid, face_or_None, used_bool)"""
    return [{'pid': p, 'idx': i, 'face': f, 'used': u}
            for i, (p, f, u) in enumerate(rows)]


@test
def test_flb3_roll_ledger_clears_on_published_turn_change():
    """par5 g3 T6: a turn boundary the seat did not drive through
    t_recover left stale pids and the auto-pool refused 'every
    visible unrolled die was already sent to a roll this turn' while
    the table showed unfaced dice.  The ledger keys on the published
    status turn: a change clears it and the pool rolls."""
    s, run = _seat()
    s._rolled_pids = {'101', '102', '103'}
    s._rolled_turn = 1
    run.payload = {'human_menu': {
        'turn': 2,
        'dice': _dice(('101', None, False), ('102', None, False),
                      ('103', None, False))}}
    try:
        out = s.t_roll([])
    except _Refused as e:
        raise AssertionError(
            'FLB-3 RED: stale ledger survived the published turn '
            'rollover - auto-pool refused: %s' % (e,))
    assert s.hg.calls and sorted(s.hg.calls[-1]) == ['101', '102',
                                                     '103'], \
        'wrong pool rolled: %r' % (s.hg.calls,)
    assert out.get('rolled_pids'), out


@test
def test_flb3_unfaced_reappearance_drops_ledgered_pid():
    """The client is the face authority: a pid the table showed FACED
    and later shows un-faced+un-used again (an undo un-faced it) was
    not consumed by any roll - it leaves the ledger and rolls again.
    CONTROL: a rolled pid the table has NEVER shown faced stays
    ledgered (the S1 table-lag guard refuses the re-roll)."""
    s, run = _seat()
    # roll BOTH dice (they enter the ledger)
    run.payload = {'human_menu': {
        'turn': 5,
        'dice': _dice(('101', None, False), ('102', None, False))}}
    s.t_roll([])
    assert sorted(s.hg.calls[-1]) == ['101', '102'], s.hg.calls
    # the table catches up: both show FACED (marks them seen-faced;
    # the empty pool correctly refuses the re-press)
    run.payload = {'human_menu': {
        'turn': 5,
        'dice': _dice(('101', 'C', False), ('102', 'A', True))}}
    try:
        s.t_roll([])
    except _Refused:
        pass
    # the undo: 101 shows un-faced+un-used again - it was NOT
    # consumed by any roll; it must leave the ledger and roll
    run.payload = {'human_menu': {
        'turn': 5,
        'dice': _dice(('101', None, False), ('102', 'A', True))}}
    try:
        s.t_roll([])
    except _Refused as e:
        raise AssertionError(
            'FLB-3 RED: the un-faced reappearance of a ledgered pid '
            'did not drop it: %s' % (e,))
    assert s.hg.calls[-1] == ['101'], \
        'the reappeared pid did not roll: %r' % (s.hg.calls,)
    # CONTROL (S1 lag guard): 101 was JUST re-rolled and the table
    # still lags (un-faced, not seen faced since that roll) - the
    # re-press REFUSES instead of re-rolling
    run.payload = {'human_menu': {
        'turn': 5,
        'dice': _dice(('101', None, False), ('102', 'A', True))}}
    try:
        s.t_roll([])
    except _Refused:
        pass
    else:
        raise AssertionError(
            'CONTROL broken: the table-lag guard no longer refuses '
            'a same-turn re-roll of a never-faced pid')


# ---------------------------------------------------------------------
# FLB-4: adopt_minions finds ShadowMinion-typed figures (WK dump row)
# ---------------------------------------------------------------------
@test
def test_flb4_adopt_minions_relocates_shadowminion_typed_wk():
    """G3#6 / FLA_HANDOFF 4: board minion figures carry EITHER type
    name - the game_logic.Minion wrapper (musters) or the raw
    gs.ShadowMinion record (paths that append state.witch_king
    directly; FLA-3's live shape).  The WK dump row (pid 253) must
    relocate a ShadowMinion-typed engine object; the old finder
    (type name == 'Minion' only) silently stayed report-only."""
    from wotr_bridge.live import adopt
    st = setup_game()
    wk = st.witch_king
    wk.in_play = True
    game_logic.get_or_create_army(st, 'eastemnet').leaders.append(wk)
    rep = adopt.adopt_minions(st, {'253': 'Westemnet'})
    west = next((a for k, a in st.armies.items()
                 if adopt._norm(k) == 'westemnet'), None)
    assert west is not None and wk in west.leaders, \
        'FLB-4 RED: adopt_minions is blind to the ShadowMinion type ' \
        '- the WK dump row adopted nothing: rep=%r' % (rep,)
    east = next((a for k, a in st.armies.items()
                 if adopt._norm(k) == 'eastemnet'), None)
    assert east is None or wk not in east.leaders, \
        'WK duplicated: still at eastemnet too'
    assert any('Witch' in r for r in rep), rep
    # CONTROL: the Minion-typed wrapper leg is unchanged
    st2 = setup_game()
    m = game_logic.Minion(nation=Nation.SAURON, name='Witch King')
    game_logic.get_or_create_army(st2, 'eastemnet').leaders.append(m)
    rep2 = adopt.adopt_minions(st2, {'253': 'Westemnet'})
    west2 = next((a for k, a in st2.armies.items()
                  if adopt._norm(k) == 'westemnet'), None)
    assert west2 is not None and m in west2.leaders, rep2
    # never fabricates: no object -> no write
    st3 = setup_game()
    rep3 = adopt.adopt_minions(st3, {'253': 'Westemnet'})
    assert rep3 == [], rep3


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
