# -*- coding: ascii -*-
"""Regression pins for the 2026-08-29 SELF-PLAY fix wave, owner SPA
(src/ai.py + src/run_td_v33sp.py).

One plain test_* function per finding; every pin was run RED against
the unfixed tree before the fix landed (SPA_FIX_REPORT.md quotes the
RED lines).  Findings pinned here:
  SPA-1  SPMC-2   chance-branch turn crossing starts with the FP
  SPA-2  SPMC-5   in-tree apply failures counted + strict re-raise
  SPA-3  native   MIRROR FINGERPRINT guard refuses the stale pyd
  SPA-4  SPEP-6   hunt-kernel error class is loud + strict
  SPA-5  SPMC-4   Great Host rounds kernel-expand (1.254 pause shape)
  SPA-6  SPEP-3   turn-start snapshot rows (no structural skew-drop)
  SPA-7  SPEP-8   declare decision recorded every turn start (decline
                  included - session-34 absence doctrine)
  SPA-8  SPEP-7   episode reward stream == trained replay stream
  SPA-9  SPEP-4   ring metric advances with mordor_step (1.143b/1.225)
  SPA-10 SPMC-7   _state_sig covers ring counts + waiver flags/faces
  SPA-11 SPMC-8   in-tree empty menu hands over to the opponent (MC6)
  SPA-12 SPTR-9   FP_NATIONS/SHADOW_NATIONS order is hashseed-free
  SPA-13 SPTR-16  fresh-run weight guard is conditional (session-34)
"""
import os
import subprocess
import sys
import traceback

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('WOTR_ACT_GEN', 'v41')
os.environ.setdefault('WOTR_TURN_HORIZON', '1')
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import random

import numpy as np

import board
import game_state as gs
import game_logic
import ai
from board import sanitize
from game_state import (Side, Nation, UnitKind, Unit, Leader, Nazgul,
                        ActionDieFace as Face)

FP = Side.FREE_PEOPLES
SH = Side.SHADOW
PY = sys.executable


# ---------------------------------------------------------------- helpers
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


def place(st, rk, nation, regulars=0, elites=0, leaders=0, nazgul=0,
          clear=False):
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
    for _ in range(nazgul):
        a.leaders.append(Nazgul(nation=Nation.SAURON))
    return a


def gondor_at_war(st):
    st.political.boxes[Nation.GONDOR] = 3
    st.political.active[Nation.GONDOR] = True


class net_stub(object):
    """Net-free searches: near-uniform priors (pass-actions biased 3x
    so declare roots argmax to the DECLINE - the absence case SPEP-8
    pins), 0.0 leaf values.  The stubbed functions are the ONLY net
    touch points of the search/episode path (the timeout bootstrap
    guards its own net call)."""

    def __enter__(self):
        self._gp = ai._get_priors_and_value
        self._ev = ai._mcts_eval_state

        def _priors(st9, sd9, nt9, acts9):
            w = np.array([3.0 if a.action_type == 'pass' else 1.0
                          for a in acts9], dtype=np.float32)
            return w / w.sum(), 0.0

        ai._get_priors_and_value = _priors
        ai._mcts_eval_state = lambda st9, sd9, f9, s9: 0.0
        return self

    def __exit__(self, *a):
        ai._get_priors_and_value = self._gp
        ai._mcts_eval_state = self._ev


def reset_kernel_stats():
    for k in list(ai.KERNEL_STATS.keys()):
        v = ai.KERNEL_STATS[k]
        if isinstance(v, dict):
            v.clear()
        elif isinstance(v, list):
            del v[:]
        else:
            ai.KERNEL_STATS[k] = 0
    ai.KERNEL_ERROR_FIRST = None


_EP_CACHE = {}


def run_short_episode():
    """One 3-turn net-free traced episode, shared by SPA-6/7/8."""
    if _EP_CACHE:
        return _EP_CACHE['traj'], _EP_CACHE['trace']
    import glob as _glob
    import tempfile
    random.seed(4242)
    np.random.seed(4242)
    tdir = tempfile.mkdtemp(prefix='spa_ep_')
    tp = os.path.join(tdir, 'spa.trace.gz')
    ai.PLAYER_PROFILES['fp'] = 'ring'
    ai.PLAYER_PROFILES['sh'] = 'military'
    ai._TREE_CACHE['child'] = None
    ai._TREE_CACHE['sig'] = None
    with net_stub():
        traj = ai._run_single_episode(None, None, {
            'max_turns': 3, 'mcts_sims': 8, 'shadow_vp': 10,
            'trace_path': tp})
    hits = _glob.glob(os.path.join(tdir, '*.trace.gz'))
    assert hits, 'episode saved no trace'
    from game_trace import load_trace
    tr = load_trace(hits[0])
    _EP_CACHE['traj'] = traj
    _EP_CACHE['trace'] = tr
    return traj, tr


# ------------------------------------------------------- SPA-1 (SPMC-2)
def test_spa1_chance_branch_turn_crossing_starts_fp():
    """A REAL in-tree crossing consumes RNG and materializes as an
    action=None chance BRANCH - its settle must still start the new
    turn with the Free Peoples (rulebook 1406-1407; EX-2 covered only
    the deterministic flagged path)."""
    random.seed(7)
    st = setup_game()
    ai._safe_next_turn(st)
    st.pending_discard = None
    game_logic.roll_action_dice(st, pool=FP)
    game_logic.roll_action_dice(st, pool=SH)
    for s in (FP, SH):
        for d in st.dice[s]:
            d.used = True
    ai._SIM_OVERLAY = None
    ai._SEARCH_NETS = (None, None)
    parent = ai.MCTSNode(state=st, side=FP)
    real = ai._expand_node
    ai._expand_node = lambda *a, **k: 0.0
    try:
        with net_stub():
            ai._settle_new_node(parent, FP, None, None, FP)
            et = parent.children[0]
            assert et.action.action_type == 'end_turn'
            new_state, used_rng = ai._apply_action_detect_rng(
                parent.state, et.action)
            assert used_rng, 'crossing must consume RNG (draws+rolls)'
            branch = ai.MCTSNode(state=new_state, side=et.side,
                                 parent=et, action=None, prior=1.0)
            ai._settle_new_node(branch, et.side, None, None, FP)
    finally:
        ai._expand_node = real
    want = ai._turn_start_side(new_state)
    assert branch.side == want == FP, \
        ('SPMC-2: chance-branch crossing handed the first action to %s '
         '(want FP)' % getattr(branch.side, 'name', branch.side))


# ------------------------------------------------------- SPA-2 (SPMC-5)
def test_spa2_intree_apply_failure_counted_and_strict():
    """An in-tree apply_action failure must be COUNTED (SEARCH_STATS +
    first-error text) and re-raise under WOTR_KERNEL_STRICT=1 - never
    only a silent fake -1 terminal."""
    random.seed(11)
    st = setup_game()
    game_logic.roll_action_dice(st, pool=FP)
    game_logic.roll_action_dice(st, pool=SH)

    real_apply = ai._apply_action_detect_rng

    def raising_apply(parent_state, action):
        raise RuntimeError('spa2 forced apply failure')

    ai.SEARCH_STATS['apply_fail'] = 0
    ai.SEARCH_APPLY_FAIL_FIRST = None
    ai._apply_action_detect_rng = raising_apply
    old_strict = os.environ.pop('WOTR_KERNEL_STRICT', None)
    try:
        with net_stub():
            res = ai.mcts_search(st, FP, None, None, num_simulations=5,
                                 temperature=0.0, add_noise=False)
        assert ai.SEARCH_STATS['apply_fail'] > 0, \
            'SPMC-5: failed applies were not counted'
        assert ai.SEARCH_APPLY_FAIL_FIRST is not None \
            and 'spa2 forced apply failure' in ai.SEARCH_APPLY_FAIL_FIRST, \
            'SPMC-5: first-error text not captured'
        # strict mode re-raises instead of faking a terminal
        os.environ['WOTR_KERNEL_STRICT'] = '1'
        raised = False
        try:
            with net_stub():
                ai.mcts_search(st, FP, None, None, num_simulations=5,
                               temperature=0.0, add_noise=False)
        except RuntimeError:
            raised = True
        assert raised, 'SPMC-5: strict mode did not re-raise'
    finally:
        ai._apply_action_detect_rng = real_apply
        if old_strict is None:
            os.environ.pop('WOTR_KERNEL_STRICT', None)
        else:
            os.environ['WOTR_KERNEL_STRICT'] = old_strict


# ------------------------------------------------------- SPA-3 (native)
def test_spa3_native_without_fingerprint_is_refused():
    """WOTR_NATIVE=1 with a pyd that carries no MIRROR_FINGERPRINT must
    NOT bind the five mirrored functions (a stale build silently
    reverts fix waves only in the loop's env - SPMC-1/SPEP-1/SPLP-1);
    the refusal is exposed via NATIVE_BOUND / NATIVE_REFUSED_REASON."""
    import glob as _glob
    pyds = _glob.glob(os.path.join(SRC, 'wotr_native*.pyd'))
    assert pyds, 'shipped wotr_native pyd not found - pin invalid'
    code = (
        "import os, sys\n"
        "os.environ['WOTR_NATIVE'] = '1'\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')\n"
        "os.environ.setdefault('WOTR_ACT_GEN', 'v41')\n"
        "sys.path.insert(0, r'%s')\n"
        "import ai\n"
        "mods = [ai.clone_state.__module__, ai.encode_state.__module__,\n"
        "        ai._state_sig.__module__, ai.encode_action.__module__,\n"
        "        ai._action_cache_key.__module__]\n"
        "print('MODS=' + ','.join(mods))\n"
        "print('BOUND=%%r' %% getattr(ai, 'NATIVE_BOUND', 'MISSING'))\n"
        "print('REFUSED=%%r' %% getattr(ai, 'NATIVE_REFUSED_REASON',\n"
        "                              'MISSING'))\n" % SRC)
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = '0'
    r = subprocess.run([PY, '-c', code], capture_output=True, text=True,
                       env=env, cwd=SRC, timeout=600)
    out = r.stdout
    assert r.returncode == 0, 'subprocess failed: %s' % (r.stderr[-800:])
    mods_line = [l for l in out.splitlines() if l.startswith('MODS=')]
    assert mods_line, 'no MODS line: %s' % out[-400:]
    mods = mods_line[0][len('MODS='):].split(',')
    assert all(m == 'ai' for m in mods), \
        ('native guard: mirrored functions still bound to %s' % mods)
    assert "BOUND=False" in out, 'NATIVE_BOUND not False: %s' % out[-400:]
    assert "REFUSED='MISSING'" not in out and 'REFUSED=None' not in out, \
        'NATIVE_REFUSED_REASON not set: %s' % out[-400:]


# ------------------------------------------------------- SPA-4 (SPEP-6)
def test_spa4_hunt_kernel_error_is_loud_and_strict():
    """The hunt kernel's error-class fallback gets the SAME loud
    treatment as the combat kernel's: counted, first text latched in
    KERNEL_ERROR_FIRST, re-raise under WOTR_KERNEL_STRICT=1 (the audit
    measured 96 silent hunt_error fallbacks under strict mode)."""
    st = setup_game()
    parent = ai.MCTSNode(state=st, side=FP)
    node = ai.MCTSNode(state=None, side=FP, parent=parent,
                       action=ai.Action(die_idx=0,
                                        action_type='fellowship_move',
                                        params={}))
    real_apply = ai.apply_action

    def raising_apply(parent_state, action):
        raise RuntimeError('spa4 forced hunt failure')

    reset_kernel_stats()
    ai.apply_action = raising_apply
    old_strict = os.environ.pop('WOTR_KERNEL_STRICT', None)
    try:
        kind, st_out = ai._hunt_expand_try(node)
        assert kind is None and st_out is None
        assert ai.KERNEL_STATS['fallback'].get('hunt_error') == 1
        assert ai.KERNEL_ERROR_FIRST is not None \
            and 'spa4 forced hunt failure' in ai.KERNEL_ERROR_FIRST, \
            ('SPEP-6: hunt_error did not latch KERNEL_ERROR_FIRST '
             '(got %r)' % (ai.KERNEL_ERROR_FIRST,))
        os.environ['WOTR_KERNEL_STRICT'] = '1'
        raised = False
        try:
            ai._hunt_expand_try(node)
        except RuntimeError:
            raised = True
        assert raised, 'SPEP-6: strict mode did not re-raise hunt_error'
    finally:
        ai.apply_action = real_apply
        if old_strict is None:
            os.environ.pop('WOTR_KERNEL_STRICT', None)
        else:
            os.environ['WOTR_KERNEL_STRICT'] = old_strict
        reset_kernel_stats()


# ------------------------------------------------------- SPA-5 (SPMC-4)
def test_spa5_great_host_round_kernel_expands():
    """A Great Host round must kernel-expand with ZERO branch_verify
    fallbacks: the 1.254 phase-9 pause carries the GH auto-hit as the
    ENEMY owner's allocation, which the old verify compared against the
    round's deferred hits (always 0 there) and failed."""
    from cards import get_card
    random.seed(13)
    st = setup_game()
    gondor_at_war(st)
    place(st, 'Minas Tirith', Nation.GONDOR, regulars=1, elites=1,
          leaders=0, clear=True)
    place(st, 'Osgiliath', Nation.SAURON, regulars=9, elites=1,
          clear=True)
    gh = None
    for c in list(st.sh_strat_deck):
        card = get_card(c)
        if (getattr(card, 'combat_name', '') or '').strip() == 'Great Host':
            gh = c
            break
    assert gh is not None, 'no Great Host card in the SH strategy deck'
    st.sh_strat_deck.remove(gh)
    st.hands[SH].hand.append(gh)
    set_dice(st, FP, [Face.ARMY])
    set_dice(st, SH, [Face.ARMY])

    atk = [a for a in ai.get_legal_actions(st, FP)
           if a.action_type == 'army_attack'
           and sanitize(str(a.params.get('to', ''))) == 'osgiliath']
    assert atk, 'no attack on osgiliath enumerated'
    ai._execute_action(st, atk[0])
    assert (st.pending_combat or {}).get('phase') == 'attacker_combat_card'
    dec = [a for a in ai.get_legal_actions(st, FP)
           if a.action_type == 'combat_card_none']
    assert dec, 'no attacker decline enumerated'
    ai._execute_action(st, dec[0])
    assert (st.pending_combat or {}).get('phase') == 'defender_combat_card'
    play = [a for a in ai.get_legal_actions(st, SH)
            if a.action_type == 'combat_card_play'
            and a.params.get('card_id') == gh]
    assert play, 'Great Host not on the defender combat-card menu'

    ai._SIM_OVERLAY = None
    reset_kernel_stats()
    parent = ai.MCTSNode(state=st, side=SH)
    node = ai.MCTSNode(state=None, side=SH, parent=parent,
                       action=play[0])
    node.is_chance = True
    node.outcomes = []
    new_state, used_rng = ai._apply_action_detect_rng(st, play[0])
    assert used_rng, 'the round apply must consume RNG'
    ok = ai._kernel_expand_combat(node, new_state)
    fb = dict(ai.KERNEL_STATS['fallback'])
    assert ok and not fb.get('branch_verify'), \
        ('SPMC-4: GH round did not kernel-expand (ok=%s fallback=%s '
         'detail=%s)' % (ok, fb,
                         ai.KERNEL_STATS.get('fallback_detail',
                                             [])[:2]))
    reset_kernel_stats()


# ------------------------------------------------------- SPA-6 (SPEP-3)
def test_spa6_turn_start_snapshot_rows():
    """Every turn's first FP delta row must pair with the state it was
    decided in: the OPS_V2 collector used to skew-drop it every turn
    (menu-now=0 on the pre-roll snapshot), and the eye row paired with
    a pre-entry state on entry turns.  declare_decline rows are
    stripped here - pairing them needs the SPB-owned one-line
    delta_trainer._flag_for addition (see SPA_HANDOFF.md)."""
    import copy
    import delta_trainer
    traj, tr = run_short_episode()
    tr2 = {'meta': tr['meta'], 'steps': []}
    for s in tr['steps']:
        if s.get('action_type') == 'declare_decline':
            s = dict(s)
            s.pop('search_stats', None)
        tr2['steps'].append(s)
    rows, stats = delta_trainer.collect_delta_samples(tr2, 'fp')
    assert stats['skew_dropped'] == 0, \
        ('SPEP-3: %d fp delta rows skew-dropped (structural per-turn '
         'drop)' % stats['skew_dropped'])
    assert stats['rows'] >= 3, 'too few fp delta rows: %s' % stats
    # structural pairing: each eye row's predecessor snapshot is the
    # post-entry turn_state row (the state the eye search ran on)
    steps = [s for s in tr['steps'] if s.get('state') is not None]
    eye_idx = [i for i, s in enumerate(steps)
               if s.get('action_type') == 'eye_allocate'
               and s.get('search_stats') is not None]
    assert eye_idx, 'no eye rows in trace'
    for i in eye_idx:
        prev = steps[i - 1]
        assert (prev.get('action_type') == 'turn_state'
                and (prev.get('params') or {}).get('ctx') == 'post_entry'), \
            ('SPEP-3: eye row pairs with %r, not the post_entry '
             'snapshot' % (prev.get('action_type'),))
    # ... and the turn's first main FP decision pairs with a rolled
    # state (the dice_rolled snapshot precedes it)
    roll_idx = [i for i, s in enumerate(steps)
                if s.get('action_type') == 'turn_state'
                and (s.get('params') or {}).get('ctx') == 'dice_rolled']
    assert roll_idx, 'SPEP-3: no dice_rolled snapshot rows in trace'


# ------------------------------------------------------- SPA-7 (SPEP-8)
def test_spa7_declare_family_row_every_turn_start():
    """1.230 offers the declare every turn start; the decision
    (INCLUDING the decline) is an absence-class sample and must leave
    one declare-family trace row per turn start (session-34
    absence-is-the-choice doctrine)."""
    traj, tr = run_short_episode()
    turns = int(traj['turns'])
    # declare opportunities: turn starts (initial + new_turn of a
    # PLAYED turn) whose snapshot shows a hidden, pre-Mordor
    # fellowship (the phase's own gate; auto-declare needs a gateway
    # and cannot fire in this short game)
    opps = 0
    for s in tr['steps']:
        if s.get('kind') not in ('initial', 'new_turn'):
            continue
        st9 = s.get('state')
        if st9 is None or int(s.get('turn', 0)) > turns:
            continue
        if st9.fellowship.hidden and not st9.fellowship.mordor_entered:
            opps += 1
    fam = [s for s in tr['steps']
           if s.get('action_type') in ('fellowship_declare',
                                       'declare_decline')
           and s.get('search_stats') is not None]
    assert opps >= 1, 'no declare opportunities in the probe episode'
    assert len(fam) >= opps, \
        ('SPEP-8: %d declare-family rows for %d declare opportunities'
         % (len(fam), opps))


# ------------------------------------------------------- SPA-8 (SPEP-7)
def test_spa8_reward_stream_twins_sum_equal():
    """The episode trajectory's per-side reward sums must equal the
    TRAINED stream's (_replay_reward_stream) - the episode used to
    record phase rows at literal 0.0 while the replay charged the
    per-call SH urgency constants on them."""
    import run_td_v33loop as trainer
    traj, tr = run_short_episode()
    sides = traj['sides']
    rewards = traj['rewards']
    ep = {'fp': float(rewards[sides == 0].sum()),
          'sh': float(rewards[sides == 1].sum())}
    profs = {'fp': 'ring', 'sh': 'military'}
    ai.PLAYER_PROFILES['fp'] = profs['fp']
    ai.PLAYER_PROFILES['sh'] = profs['sh']
    per = trainer._replay_reward_stream(tr, profs)
    rp = {t: float(sum(per[t][1])) for t in ('fp', 'sh')}
    for t in ('fp', 'sh'):
        assert abs(ep[t] - rp[t]) < 1e-4, \
            ('SPEP-7: %s stream sums differ: episode %.6f vs replay '
             '%.6f' % (t, ep[t], rp[t]))


# ------------------------------------------------------- SPA-9 (SPEP-4)
def test_spa9_ring_metric_advances_in_mordor():
    """1.143b ('the Mordor track continues the same scale') + 1.225
    ('1 step from destroying the ring is better than 10 steps'): the
    ring metric must be continuous at entry and strictly increase with
    mordor_step; closeness(step 4) > closeness(step 0)."""
    st = setup_game()
    st.fellowship.region = 'morannon'
    st.fellowship.progress = 0
    st.fellowship.hidden = True
    pre_entry = ai._ring_progress_metric(st)
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 0
    st.fellowship.progress = 0
    at_entry = ai._ring_progress_metric(st)
    assert abs(pre_entry - at_entry) < 1e-9, \
        ('SPEP-4: entry step discontinuous (%.3f -> %.3f)'
         % (pre_entry, at_entry))
    vals = []
    for k in range(0, 6):
        st.fellowship.mordor_step = k
        vals.append(ai._ring_progress_metric(st))
    assert all(vals[i + 1] > vals[i] for i in range(5)), \
        'SPEP-4: metric not strictly increasing across Mordor: %s' % vals
    st.fellowship.mordor_step = 0
    c0 = ai._closeness_to_victory('fp', 'ring', st)
    st.fellowship.mordor_step = 4
    c4 = ai._closeness_to_victory('fp', 'ring', st)
    assert c4 > c0, \
        'SPEP-4: closeness(step4)=%.4f <= closeness(step0)=%.4f' % (c4, c0)
    # step 5 = destroyed = full track
    st.fellowship.mordor_step = 5
    try:
        rt = ai._gate_distance(st, 'rivendell') + ai._RING_TRACK_TAIL
    except Exception:
        rt = 13.0
    assert abs(ai._ring_progress_metric(st) - rt) < 1e-9, \
        'SPEP-4: step 5 must reach the full ring track'


# ------------------------------------------------------ SPA-10 (SPMC-7)
def test_spa10_state_sig_rings_and_waiver():
    st = setup_game()
    a = ai.clone_state(st)
    b = ai.clone_state(st)
    b.fp_elven_rings = max(0, int(b.fp_elven_rings) - 1)
    assert int(a.fp_elven_rings) != int(b.fp_elven_rings)
    assert ai._state_sig(a) != ai._state_sig(b), \
        'SPMC-7: states differing only in elven-ring count share a sig'
    c = ai.clone_state(st)
    c.fp_ring_waiver = True
    c.fp_ring_waiver_face = Face.MUSTER
    assert ai._state_sig(a) != ai._state_sig(c), \
        'SPMC-7: a live ring waiver does not change the sig'
    d = ai.clone_state(c)
    d.fp_ring_waiver_face = Face.EVENT
    assert ai._state_sig(c) != ai._state_sig(d), \
        'SPMC-7: waiver FACE does not change the sig'


# ------------------------------------------------------ SPA-11 (SPMC-8)
def test_spa11_intree_empty_menu_hands_over():
    """MC6 in-tree twin (rulebook 1681-1683): a side with dice but an
    empty menu hands the rest of the round to its opponent inside the
    search too - not a dead-end leaf."""
    st = setup_game()
    game_logic.roll_action_dice(st, pool=FP)
    game_logic.roll_action_dice(st, pool=SH)
    ai._SIM_OVERLAY = None
    real_menu = ai.get_legal_actions

    def menu(state9, side9):
        if side9 == FP:
            return []
        return real_menu(state9, side9)

    node = ai.MCTSNode(state=ai.clone_state(st), side=FP)
    ai.get_legal_actions = menu
    try:
        with net_stub():
            ai._expand_node(node, None, None, FP)
    finally:
        ai.get_legal_actions = real_menu
    assert node.side == SH and node.children, \
        ('SPMC-8: empty FP menu did not hand over in-tree '
         '(side=%s children=%d)' % (getattr(node.side, 'name', '?'),
                                    len(node.children)))


# ------------------------------------------------------ SPA-12 (SPTR-9)
def test_spa12_nation_constants_hashseed_free():
    """FP_NATIONS/SHADOW_NATIONS must iterate in the same order under
    any PYTHONHASHSEED (set literals of id-hashed Enum members ordered
    the political menu by interpreter hash)."""
    assert isinstance(ai.FP_NATIONS, tuple) \
        and isinstance(ai.SHADOW_NATIONS, tuple), \
        'SPTR-9: nation constants are not fixed-order tuples'
    code = (
        "import sys\n"
        "sys.path.insert(0, r'%s')\n"
        "import os\n"
        "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')\n"
        "os.environ.setdefault('WOTR_ACT_GEN', 'v41')\n"
        "import ai\n"
        "print('ORDER=' + '|'.join(n.name for n in ai.FP_NATIONS)\n"
        "      + '#' + '|'.join(n.name for n in ai.SHADOW_NATIONS))\n"
        % SRC)
    orders = []
    for seed in ('1', '2'):
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = seed
        r = subprocess.run([PY, '-c', code], capture_output=True,
                           text=True, env=env, cwd=SRC, timeout=600)
        assert r.returncode == 0, r.stderr[-500:]
        line = [l for l in r.stdout.splitlines()
                if l.startswith('ORDER=')]
        assert line, r.stdout[-300:]
        orders.append(line[0])
    assert orders[0] == orders[1], \
        ('SPTR-9: nation iteration order depends on PYTHONHASHSEED:\n'
         '  seed1: %s\n  seed2: %s' % (orders[0], orders[1]))


# ----------------------------------------------------- SPA-13 (SPTR-16)
def test_spa13_fresh_run_guard_conditional():
    """Session-34 fresh-run guard, recorded design: '0 TD episodes on
    record -> delete _parallel_weights/*_weights_v*.pt; crash-resume
    with episodes present keeps them.'  The unconditional v33sp port
    deleted on every launch."""
    import tempfile
    import run_td_v33sp as sp
    tdir = tempfile.mkdtemp(prefix='spa_guard_')
    pw = os.path.join(tdir, '_parallel_weights')
    os.makedirs(pw)
    wf = os.path.join(pw, 'fp_ring_weights_v3.pt')
    glog = os.path.join(tdir, 'games.jsonl')
    old_root, old_log = sp.ROOT, sp.GAMES_LOG
    sp.ROOT, sp.GAMES_LOG = tdir, glog
    try:
        # branch 1: episodes on record -> checkpoints SURVIVE
        with open(wf, 'wb') as f:
            f.write(b'x')
        with open(glog, 'w') as f:
            f.write('{"winner": "fp"}\n')
        sp._fresh_run_weight_guard()
        assert os.path.exists(wf), \
            ('SPTR-16: the guard deleted checkpoints although TD '
             'episodes are on record')
        # branch 2: no episodes on record -> stale checkpoints cleared
        os.remove(glog)
        sp._fresh_run_weight_guard()
        assert not os.path.exists(wf), \
            'SPTR-16: fresh run did not clear the stale checkpoint'
    finally:
        sp.ROOT, sp.GAMES_LOG = old_root, old_log


# ------------------------------------------------ SPA-14 (SPB addendum)
def test_spa14_safe_defaults_resolve_bare_env():
    """The collapsed-regime operators must not be the code DEFAULTS:
    a bare env resolves to pi_mode=visit_tf, temp_earned=True,
    head_act=scaled2 (run-7 collapse operators only via explicit
    env)."""
    assert 'WOTR_SELECT_CFG' not in os.environ, \
        'pin invalid: WOTR_SELECT_CFG set in the test env'
    cfg = ai._select_cfg_from_env()
    assert cfg.pi_mode == 'visit_tf', \
        'default pi_mode is %r (run-7 collapse operator)' % cfg.pi_mode
    assert cfg.temp_earned is True, 'default temp_earned is off'
    old = os.environ.pop('WOTR_VALUE_HEAD_ACT', None)
    try:
        head = ai._value_head_act()
    finally:
        if old is not None:
            os.environ['WOTR_VALUE_HEAD_ACT'] = old
    assert type(head).__name__ == '_ScaledTanhHead', \
        'default head activation is %s (Tanh clips TD targets)' \
        % type(head).__name__


# ------------------------------------------------- SPA-15 (SPTR-11)
def test_spa15_state_dict_load_is_transactional():
    """A failed checkpoint load must leave the net UNTOUCHED - torch
    copies compatible tensors before raising, so validation must run
    BEFORE the copy (session-34 chimera class)."""
    import torch
    net = ai.WOTRValueNet(ai.get_feature_dim(), side='fp')
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    keys = [k for k, v in sd.items()
            if v.dtype.is_floating_point and v.numel() > 1]
    probe, brk = keys[0], keys[-1]
    assert probe != brk
    sd[probe] = torch.full_like(sd[probe], 999.0)
    sd[brk] = torch.zeros(tuple(s + 1 for s in sd[brk].shape))
    before = net.state_dict()[probe].clone()
    raised = False
    try:
        ai._load_state_dict_compat(net, sd, 'spa15')
    except RuntimeError:
        raised = True
    assert raised, 'mismatched checkpoint did not raise'
    after = net.state_dict()[probe]
    assert torch.equal(after, before), \
        ('SPTR-11: the failed load PARTIALLY SPLICED the net '
         '(probe tensor overwritten before the raise)')


# ------------------------------------------------- SPA-16 (SPTR-12)
def test_spa16_conf_autoreg_failure_is_warned():
    """The never-fatal conf auto-registration swallow must leave a
    visible WARN line (sanction covers non-fatality, not silence)."""
    import logging
    ckpt_path = os.path.join(os.path.dirname(SRC),
                             'wotr_fp_corpus_live.pt')
    assert os.path.exists(ckpt_path), 'pin needs the live fp net'
    import confidence
    real = confidence.register_from_checkpoint

    def boom(*a, **k):
        raise RuntimeError('spa16 malformed stamp')

    records = []

    class Grab(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    h = Grab()
    logging.getLogger('wotr.conf').addHandler(h)
    confidence.register_from_checkpoint = boom
    try:
        net = ai.load_value_net(ckpt_path, side='fp')
        assert net is not None, 'net load must stay non-fatal'
    finally:
        confidence.register_from_checkpoint = real
        logging.getLogger('wotr.conf').removeHandler(h)
    assert any('spa16 malformed stamp' in m for m in records), \
        ('SPTR-12: conf auto-registration failure left no WARN line '
         '(records=%s)' % records[:3])


# ----------------------------------------------------------------- main
TESTS = [
    test_spa1_chance_branch_turn_crossing_starts_fp,
    test_spa2_intree_apply_failure_counted_and_strict,
    test_spa3_native_without_fingerprint_is_refused,
    test_spa4_hunt_kernel_error_is_loud_and_strict,
    test_spa5_great_host_round_kernel_expands,
    test_spa6_turn_start_snapshot_rows,
    test_spa7_declare_family_row_every_turn_start,
    test_spa8_reward_stream_twins_sum_equal,
    test_spa9_ring_metric_advances_in_mordor,
    test_spa10_state_sig_rings_and_waiver,
    test_spa11_intree_empty_menu_hands_over,
    test_spa12_nation_constants_hashseed_free,
    test_spa13_fresh_run_guard_conditional,
    test_spa14_safe_defaults_resolve_bare_env,
    test_spa15_state_dict_load_is_transactional,
    test_spa16_conf_autoreg_failure_is_warned,
]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
            print('PASS %s' % t.__name__)
        except Exception:
            fails += 1
            print('FAIL %s' % t.__name__)
            traceback.print_exc()
    print('TOTAL=%d FAIL=%d' % (len(TESTS), fails))
    if fails:
        sys.exit(1)


if __name__ == '__main__':
    main()
