"""wotr_bridge/live/randomness.py - the client-randomness funnel
protocol lifted out of the monolith (REBUILD_PLAN Phase 2 lift-list).

Each test states its MEASURED QUANTITY first (instrument protocol,
design rule 11).  Every gate has a sensitivity control (the defect
class it must catch goes RED) and a specificity control (the
legitimate twin stays GREEN); every zero ships with a proven non-zero.

House style: fails-list runner, exit 1 on any failure, ASCII only.
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_BRIDGE = os.path.abspath(os.path.join(_SRC, '..', 'wotr_bridge'))
for _p in (_SRC, _BRIDGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import board as board_mod
import game_logic
import game_state as gs
from game_state import Side, ActionDieFace

from live import randomness as R


def fresh_state():
    b = board_mod.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    return st


def new_cr(**kw):
    """A fresh installed ClientRandomness (install is once-per-process;
    each call re-points the module's _ACTIVE dispatch at the new
    instance - the current-instance dispatch the old monolith needed
    a proxy class for)."""
    kw.setdefault('seed', 1234)
    cr = R.ClientRandomness(**kw)
    cr.install()
    return cr


class _StubAction(object):
    """Minimal engine-Action shape for act_key (action_type, die_idx,
    params)."""

    def __init__(self, action_type, die_idx=0, params=None):
        self.action_type = action_type
        self.die_idx = die_idx
        self.params = params or {}

    def __repr__(self):
        return '<A %s>' % self.action_type


# ---------------------------------------------------------------- (a)
def test_starved_d6_funnel(fails):
    """MEASURES: a real game_logic.roll_d6_batch call on the REAL
    trajectory (inside advance()) raises NeedClientCall with the right
    kind/need/side when the feed is dry, and after feed() the SAME
    call returns exactly the fed faces.  Also proves the BaseException
    contract: an engine-style `except Exception` guard must NOT eat
    the starve."""
    cr = new_cr()
    cr.begin_attempt()
    got = None
    with cr.advance():
        try:
            game_logic.roll_d6_batch(3, 'combat roll', side=Side.SHADOW)
        except R.NeedClientCall as e:
            got = e
    if got is None:
        fails.append(('d6_starve_raises', 'no NeedClientCall'))
        return
    if got.kind != 'd6' or got.kw.get('need') != 3 \
            or got.kw.get('side') is not Side.SHADOW:
        fails.append(('d6_starve_payload', 'kind=%r kw=%r'
                      % (got.kind, got.kw)))
    if cr.pending() is not got:
        fails.append(('d6_pending_recorded', 'pending()=%r'
                      % (cr.pending(),)))
    res = cr.feed('d6', [2, 5, 6])
    if res['kept'] != [2, 5, 6] or res['dropped']:
        fails.append(('d6_feed_banked', repr(res)))
    if cr.pending() is not None:
        fails.append(('d6_pending_cleared', 'still pending after feed'))
    cr.begin_attempt()                     # the retry
    with cr.advance():
        vals = game_logic.roll_d6_batch(3, 'combat roll',
                                        side=Side.SHADOW)
    if vals != [2, 5, 6]:
        fails.append(('d6_fed_values_returned', repr(vals)))
    if cr.unconsumed()['d6'] != 0:
        fails.append(('d6_all_consumed', repr(cr.unconsumed())))
    cr.step_commit()
    # BaseException contract: an engine-style broad `except Exception`
    # guard must NOT eat the starve (its own step, so the need-1
    # pending cannot bleed into other legs)
    cr.begin_attempt()
    eaten = True
    with cr.advance():
        try:
            try:
                game_logic.roll_d6_batch(1, 'guarded roll')
            except Exception:
                pass                       # a broad effect guard
        except R.NeedClientCall:
            eaten = False
    if eaten:
        fails.append(('d6_baseexception_contract',
                      'except Exception swallowed the client call'))
    cr.step_commit()


def test_hunt_reroll_decline_shim(fails):
    """MEASURES the engine-gap shim (2026-08-19): with
    reroll_decline_shim armed, a STARVED ctx='hunt reroll' consume
    returns natural 1s (byte-equivalent to DECLINING the optional p.40
    re-roll the frozen engine takes unconditionally), consumes any fed
    remainder first, and records the debt in shim_uses.
    SPECIFICITY: unarmed, the same call starves; a NON-reroll ctx
    starves even when armed."""
    cr = new_cr()
    cr.begin_attempt()
    # unarmed: starves
    starved = False
    with cr.advance():
        try:
            game_logic.roll_d6_batch(1, 'hunt reroll')
        except R.NeedClientCall:
            starved = True
    if not starved:
        fails.append(('shim_unarmed_starves', 'no NeedClientCall'))
    cr.step_commit()
    # armed: fed remainder first, then 1s; debt recorded
    cr.begin_attempt()
    cr.feed('d6', [5])
    cr.reroll_decline_shim = True
    try:
        with cr.advance():
            vals = game_logic.roll_d6_batch(3, 'hunt reroll')
    finally:
        cr.reroll_decline_shim = False
    if vals != [5, 1, 1]:
        fails.append(('shim_values', repr(vals)))
    if not cr.shim_uses or cr.shim_uses[-1] != ('hunt reroll', 2):
        fails.append(('shim_debt_recorded', repr(cr.shim_uses)))
    cr.step_commit()
    # armed but a different ctx: still starves (the shim is scoped to
    # the ONE named engine gap, never a general randomness fabricator)
    cr.begin_attempt()
    cr.reroll_decline_shim = True
    starved = False
    try:
        with cr.advance():
            try:
                game_logic.roll_d6_batch(1, 'combat roll')
            except R.NeedClientCall:
                starved = True
    finally:
        cr.reroll_decline_shim = False
    if not starved:
        fails.append(('shim_scope', 'non-reroll ctx did not starve'))
    cr.step_commit()


def test_starved_tile_funnel(fails):
    """MEASURES: pop_hunt_tile with a STOCKED pool starves on the real
    trajectory (NeedClientCall 'tile'); after feeding a face the pool
    really holds, the retry returns a tile of that face AND removes it
    from the pool (pop semantics).  pick_hunt_tile leaves the pool
    intact (pick semantics).  Sensitivity: a fed face the pool does
    NOT hold halts loudly as a pool desync."""
    st = fresh_state()
    cr = new_cr()
    if not st.hunt_pool:
        fails.append(('tile_precondition', 'fresh pool empty'))
        return
    n0 = len(st.hunt_pool)
    face = R.tile_face(st.hunt_pool[0])
    cr.begin_attempt()
    got = None
    with cr.advance():
        try:
            game_logic.pop_hunt_tile(st)
        except R.NeedClientCall as e:
            got = e
    if got is None or got.kind != 'tile':
        fails.append(('tile_starve', repr(got)))
        return
    cr.feed('tile', (face, '218', 'Game.2036'))
    cr.begin_attempt()
    with cr.advance():
        t = game_logic.pop_hunt_tile(st)
    if R.tile_face(t) != face:
        fails.append(('tile_fed_identity', '%r != %r'
                      % (R.tile_face(t), face)))
    if len(st.hunt_pool) != n0 - 1:
        fails.append(('tile_pop_removes', 'pool %d -> %d'
                      % (n0, len(st.hunt_pool))))
    cr.step_commit()
    # pick semantics: pool NOT shrunk
    face2 = R.tile_face(st.hunt_pool[0])
    n1 = len(st.hunt_pool)
    cr.begin_attempt()
    cr.feed('tile', face2)
    with cr.advance():
        t2 = game_logic.pick_hunt_tile(st)
    if R.tile_face(t2) != face2 or len(st.hunt_pool) != n1:
        fails.append(('tile_pick_keeps_pool', 'face %r pool %d -> %d'
                      % (R.tile_face(t2), n1, len(st.hunt_pool))))
    cr.step_commit()
    # sensitivity: pool-desync halt on a face the pool cannot hold
    cr.begin_attempt()
    cr.feed('tile', 'not a real tile face')
    halted = False
    try:
        with cr.advance():
            game_logic.pop_hunt_tile(st)
    except RuntimeError as e:
        halted = 'pool desync' in str(e) and 'LIVE HALT' in str(e)
    if not halted:
        fails.append(('tile_desync_halts', 'bogus face accepted'))
    if not (cr.sticky_halt and 'pool desync' in cr.sticky_halt):
        fails.append(('tile_desync_sticky', repr(cr.sticky_halt)))
    cr.reset_feed()
    cr.step_commit()


# ---------------------------------------------------------------- (b)
def test_mcts_clone_scoping(fails):
    """MEASURES (specificity): at advance-depth 0 - where MCTS
    simulation clones run - the patched funnels do NOT raise, do NOT
    touch the feed, and use the ENGINE's own global RNG (proven by
    seeded reproducibility, not by absence alone)."""
    import random as pyrandom
    st = fresh_state()
    cr = new_cr()                          # feed is empty on purpose
    before = cr.unconsumed().copy()
    pyrandom.seed(99)
    a = game_logic.roll_d6_batch(3, 'combat roll')
    pyrandom.seed(99)
    b = game_logic.roll_d6_batch(3, 'combat roll')
    if len(a) != 3 or any(not 1 <= v <= 6 for v in a):
        fails.append(('scope_d6_shape', repr(a)))
    if a != b:
        fails.append(('scope_engine_rng',
                      'seeded twins differ: %r %r (not the global '
                      'engine stream)' % (a, b)))
    comps = [c for c in st.fellowship.companions if c.name != 'Gollum']
    v = game_logic.pick_random_companion(comps, 'depth-0 probe')
    if v not in comps:
        fails.append(('scope_companion_engine_pick', repr(v)))
    t = game_logic.pick_hunt_tile(st)
    if t not in st.hunt_pool:
        fails.append(('scope_tile_engine_pick', repr(t)))
    # tripwires quiet at depth 0
    try:
        pyrandom.choice(['x', 'y'])
        pyrandom.randint(1, 6)
    except RuntimeError as e:
        fails.append(('scope_tripwire_quiet', repr(e)))
    if cr.unconsumed() != before:
        fails.append(('scope_feed_untouched', '%r -> %r'
                      % (before, cr.unconsumed())))


# ---------------------------------------------------------------- (c)
def test_unfunneled_tripwire(fails):
    """MEASURES (sensitivity): random.choice / random.randint reached
    on the REAL trajectory HALT LOUDLY with a named STICKY message.
    The classifiers name the defect class: a Companion-shaped pick
    names the bypassed game_logic.pick_random_companion funnel; a
    WWSAT-shaped pick names the unlanded Phase-1 funnel.  The sticky
    message survives reset_feed()."""
    import random as pyrandom
    st = fresh_state()
    cr = new_cr()

    def _expect_halt(fn, want_substrings, tag):
        try:
            with cr.advance():
                fn()
        except RuntimeError as e:
            msg = str(e)
            missing = [w for w in want_substrings if w not in msg]
            if missing:
                fails.append((tag, 'halt lacks %r in %r'
                              % (missing, msg[:120])))
            if cr.sticky_halt != msg:
                fails.append((tag + '_sticky', repr(cr.sticky_halt)))
            return
        fails.append((tag, 'no halt raised'))

    _expect_halt(lambda: pyrandom.choice(['x', 'y']),
                 ['LIVE HALT', 'unfunneled random.choice'],
                 'trip_choice_generic')
    _expect_halt(lambda: pyrandom.randint(1, 6),
                 ['LIVE HALT', 'unfunneled random.randint'],
                 'trip_randint')
    comps = [c for c in st.fellowship.companions if c.name != 'Gollum']
    _expect_halt(lambda: pyrandom.choice(comps),
                 ['LIVE HALT', 'pick_random_companion'],
                 'trip_choice_companion_named')
    wwsat_pop = ['FREE_PEOPLES_Character_23', 'FREE_PEOPLES_Character_04']
    _expect_halt(lambda: pyrandom.choice(wwsat_pop),
                 ['LIVE HALT', 'WWSAT'],
                 'trip_choice_wwsat_named')
    sticky = cr.sticky_halt
    cr.reset_feed()
    if cr.sticky_halt != sticky:
        fails.append(('trip_sticky_survives_reset',
                      repr(cr.sticky_halt)))


# ---------------------------------------------------------------- (d)
def test_rightmost_excess(fails):
    """MEASURES (client fact 19, G1 verbatim example '3 1 4 5 -> use
    3 1 4'): with a pending ask for 3 d6, a single roll line of 5
    faces [3,1,4,5,2] banks exactly the LEFTMOST 3 = [3,1,4] and
    drops [5,2] from the RIGHT; the retried step consumes [3,1,4].
    Specificity: an exact-count line drops nothing."""
    cr = new_cr()
    cr.begin_attempt()
    with cr.advance():
        try:
            game_logic.roll_d6_batch(3, 'combat roll')
        except R.NeedClientCall:
            pass
    res = cr.feed('d6', [3, 1, 4, 5, 2])
    if res['kept'] != [3, 1, 4]:
        fails.append(('excess_leftmost_kept', repr(res)))
    if res['dropped'] != [5, 2]:
        fails.append(('excess_rightmost_dropped', repr(res)))
    cr.begin_attempt()
    with cr.advance():
        vals = game_logic.roll_d6_batch(3, 'combat roll')
    if vals != [3, 1, 4]:
        fails.append(('excess_step_consumes_leftmost', repr(vals)))
    if cr.unconsumed()['d6'] != 0:
        fails.append(('excess_nothing_banked_extra',
                      repr(cr.unconsumed())))
    cr.step_commit()
    # specificity: exact-count line -> zero dropped
    cr.begin_attempt()
    with cr.advance():
        try:
            game_logic.roll_d6_batch(3, 'combat roll')
        except R.NeedClientCall:
            pass
    res2 = cr.feed('d6', [2, 2, 2])
    if res2['dropped']:
        fails.append(('excess_specificity', repr(res2)))
    cr.begin_attempt()
    with cr.advance():
        game_logic.roll_d6_batch(3, 'combat roll')
    cr.step_commit()


def test_cross_line_surplus(fails):
    """MEASURES (client fact 19, second clause): a WHOLE extra roll
    line arriving after the pending need is met is banked UNTRIMMED as
    cross-line surplus and feeds the SAME step's next consumer without
    a second directive (the human pre-produced the reroll)."""
    cr = new_cr()
    cr.begin_attempt()
    with cr.advance():
        try:
            game_logic.roll_d6_batch(1, 'hunt roll')
        except R.NeedClientCall:
            pass
    r1 = cr.feed('d6', [4])                # satisfies the ask
    if cr.pending() is not None:
        fails.append(('surplus_ask_satisfied', 'pending survived'))
    r2 = cr.feed('d6', [2, 5])             # pre-produced reroll line
    if r2['dropped'] or r2['kept'] != [2, 5]:
        fails.append(('surplus_kept_whole', repr(r2)))
    starved_again = False
    cr.begin_attempt()
    with cr.advance():
        first = game_logic.roll_d6_batch(1, 'hunt roll')
        try:
            second = game_logic.roll_d6_batch(2, 'hunt reroll')
        except R.NeedClientCall:
            starved_again = True
            second = None
    if first != [4]:
        fails.append(('surplus_first_consumer', repr(first)))
    if starved_again:
        fails.append(('surplus_no_second_directive',
                      'next consumer starved despite surplus'))
    elif second != [2, 5]:
        fails.append(('surplus_next_consumer_values', repr(second)))
    if cr.unconsumed()['d6'] != 0:
        fails.append(('surplus_honesty_zero', repr(cr.unconsumed())))
    cr.step_commit()
    _ = r1


# ---------------------------------------------------------------- (e)
def test_deterministic_retry_journal(fails):
    """MEASURES: a step that picks a decision, starves, is fed and
    retried REPLAYS the same pick from the journal without re-running
    the chooser - read off the product's OWN counters (_journal /
    _journal_ix and the chooser call count), not a proxy.
    Sensitivity: a retry whose candidates no longer contain the
    journaled pick halts as JOURNAL MISMATCH."""
    cr = new_cr()
    actions = [_StubAction('x', 0), _StubAction('y', 1),
               _StubAction('z', 2)]
    calls = {'n': 0}

    def chooser(acts):
        calls['n'] += 1
        return list(reversed(acts))        # deterministic pick: 'z'

    cr.begin_attempt()
    ranked1 = cr.journal_pick(actions, chooser)
    if ranked1[0].action_type != 'z' or calls['n'] != 1:
        fails.append(('journal_first_attempt', '%r calls=%d'
                      % (ranked1[0], calls['n'])))
    with cr.advance():
        try:
            game_logic.roll_d6_batch(2, 'combat roll')
        except R.NeedClientCall:
            pass
    cr.feed('d6', [1, 2])
    cr.begin_attempt()                     # the retry
    ranked2 = cr.journal_pick(actions, chooser)
    if calls['n'] != 1:
        fails.append(('journal_no_rerun',
                      'chooser ran %d times on retry' % calls['n']))
    if ranked2[0].action_type != 'z':
        fails.append(('journal_same_pick', repr(ranked2[0])))
    if len(cr._journal) != 1 or cr._journal_ix != 1:
        fails.append(('journal_own_counter', 'journal=%r ix=%d'
                      % (cr._journal, cr._journal_ix)))
    with cr.advance():
        vals = game_logic.roll_d6_batch(2, 'combat roll')
    if vals != [1, 2]:
        fails.append(('journal_step_completes', repr(vals)))
    cr.step_commit()
    if cr._journal or cr._journal_ix:
        fails.append(('journal_commit_clears', 'journal survived'))
    # sensitivity: journaled pick missing from the retry menu -> halt
    cr.begin_attempt()
    cr.journal_pick(actions, chooser)      # journals 'z'
    cr.begin_attempt()                     # simulate a retry
    halted = False
    try:
        cr.journal_pick([_StubAction('x', 0)], chooser)
    except RuntimeError as e:
        halted = 'JOURNAL MISMATCH' in str(e)
    if not halted:
        fails.append(('journal_mismatch_halts', 'divergence accepted'))
    cr.step_commit()


def test_per_step_rng_stream(fails):
    """MEASURES: residual (non-tripwired) global randomness inside
    advance() replays IDENTICALLY on a retry of the same step (the
    per-step stream resets), and a NEW step gets a different stream
    (specificity: the reset is per-step, not a global freeze)."""
    import random as pyrandom
    cr = new_cr()
    cr.begin_attempt()
    with cr.advance():
        r1 = pyrandom.random()
        try:
            game_logic.roll_d6_batch(1, 'combat roll')
        except R.NeedClientCall:
            pass
    cr.feed('d6', [6])
    cr.begin_attempt()
    with cr.advance():
        r2 = pyrandom.random()
        game_logic.roll_d6_batch(1, 'combat roll')
    if r1 != r2:
        fails.append(('rng_retry_replays', '%r != %r' % (r1, r2)))
    cr.step_commit()
    cr.begin_attempt()
    with cr.advance():
        r3 = pyrandom.random()
    if r3 == r1:
        fails.append(('rng_new_step_differs',
                      'two steps share a residual stream'))
    cr.step_commit()


def test_pin_determinism(fails):
    """MEASURES: pin() gives probe #N of a step the SAME RNG stream on
    every attempt (retries re-enumerate identical legal sets), while
    two probes of one attempt differ (the pin varies by probe index -
    non-vacuity)."""
    import random as pyrandom
    cr = new_cr()
    cr.begin_attempt()
    with cr.pin():
        p1 = pyrandom.random()
    with cr.pin():
        p2 = pyrandom.random()
    with cr.advance():
        try:
            game_logic.roll_d6_batch(1, 'combat roll')
        except R.NeedClientCall:
            pass
    cr.feed('d6', [3])
    cr.begin_attempt()                     # retry
    with cr.pin():
        q1 = pyrandom.random()
    with cr.pin():
        q2 = pyrandom.random()
    if (p1, p2) != (q1, q2):
        fails.append(('pin_retry_identical', '%r != %r'
                      % ((p1, p2), (q1, q2))))
    if p1 == p2:
        fails.append(('pin_probe_index_varies',
                      'probe 0 and 1 share a stream'))
    with cr.advance():
        game_logic.roll_d6_batch(1, 'combat roll')
    cr.step_commit()


# ---------------------------------------------------------------- (f)
def test_named_companion_funnel(fails):
    """MEASURES: the patched NAMED funnel game_logic.
    pick_random_companion starves on the real trajectory
    (NeedClientCall 'randomcomp' carrying ctx) and, once fed, returns
    the exact candidate object whose name matches ANY of the fed name
    faces (dual-sided acceptance).  Sensitivity: a fed name outside
    the candidate pool halts as a pool desync."""
    st = fresh_state()
    cr = new_cr()
    comps = [c for c in st.fellowship.companions if c.name != 'Gollum']
    if len(comps) < 2:
        fails.append(('comp_precondition', 'not enough companions'))
        return
    target = comps[1]
    cr.begin_attempt()
    got = None
    with cr.advance():
        try:
            game_logic.pick_random_companion(comps,
                                             'hunt casualty draw')
        except R.NeedClientCall as e:
            got = e
    if got is None or got.kind != 'randomcomp' \
            or got.kw.get('ctx') != 'hunt casualty draw':
        fails.append(('comp_starve', repr(got and (got.kind, got.kw))))
        return
    dirs = cr.directives_for(got)
    if [d.kind for d in dirs] != ['chat', 'call'] \
            or dirs[0].text != 'random companion draw?' \
            or dirs[1].text != '!CALL random_companion':
        fails.append(('comp_directives', repr(dirs)))
    # dual-face acceptance: one bogus face + the real name
    cr.feed('companion', ('Not A Real Face', target.name))
    cr.begin_attempt()
    with cr.advance():
        victim = game_logic.pick_random_companion(
            comps, 'hunt casualty draw')
    if victim is not target:
        fails.append(('comp_fed_pick_returned', '%r is not %r'
                      % (victim, target)))
    cr.step_commit()
    # sensitivity: fed name outside the pool -> desync halt
    cr.begin_attempt()
    cr.feed('companion', 'Sauron Himself')
    halted = False
    try:
        with cr.advance():
            game_logic.pick_random_companion(comps, 'hunt casualty draw')
    except RuntimeError as e:
        halted = 'pool desync' in str(e)
    if not halted:
        fails.append(('comp_desync_halts', 'bogus name accepted'))
    cr.reset_feed()
    cr.step_commit()


def test_card_funnel_and_grant(fails):
    """MEASURES: the patched draw_event_card starves with the exact
    (side, deck) item, and once fed lands the CLIENT's cid with exact
    draw semantics (deck loses it, hand gains it, provenance records
    it).  grant_draw integration (design change 3): the starve
    propagates THROUGH grant_draw, which stamps one unit of
    state._owed_draws itself - this funnel must not interfere - and
    the fed retry settles the debt to zero.  Deck-empty stays a
    RuntimeError (unfulfillable grant, no debt).  Sensitivity: a fed
    cid the deck does not hold halts as deck desync."""
    st = fresh_state()
    cr = new_cr()
    cid = st.sh_char_deck[5]
    n_deck = len(st.sh_char_deck)
    n_hand = len(st.hands[Side.SHADOW].hand)
    cr.begin_attempt()
    got = None
    with cr.advance():
        try:
            game_logic.draw_event_card(st, Side.SHADOW, 'character')
        except R.NeedClientCall as e:
            got = e
    if got is None or got.kind != 'draws' \
            or got.kw.get('items') != [('SHADOW', 'character')]:
        fails.append(('card_starve', repr(got and (got.kind, got.kw))))
        return
    dirs = cr.directives_for(got)
    if [(d.kind, d.text) for d in dirs] != [('call',
                                             '!CALL draw_sa_char')]:
        fails.append(('card_directive', repr(dirs)))
    cr.feed('card', ('SHADOW', 'character', cid, '123'))
    if cr.pending() is not None:
        fails.append(('card_pending_cleared', 'still pending'))
    cr.begin_attempt()
    with cr.advance():
        game_logic.draw_event_card(st, Side.SHADOW, 'character')
    hand = st.hands[Side.SHADOW].hand
    if len(hand) != n_hand + 1 or hand[-1] != cid:
        fails.append(('card_lands_fed_cid', repr(hand[-3:])))
    if cid in st.sh_char_deck or len(st.sh_char_deck) != n_deck - 1:
        fails.append(('card_leaves_deck', 'deck %d -> %d'
                      % (n_deck, len(st.sh_char_deck))))
    if str(cid) not in cr.cid_provenance:
        fails.append(('card_provenance', repr(cr.cid_provenance)))
    cr.step_commit()
    # grant_draw integration on a fresh state
    st2 = fresh_state()
    cr.begin_attempt()
    debt_key = (Side.SHADOW.name, 'strategy')
    propagated = False
    with cr.advance():
        try:
            game_logic.grant_draw(st2, Side.SHADOW, 'strategy')
        except R.NeedClientCall:
            propagated = True
    debt = (getattr(st2, '_owed_draws', None) or {}).get(debt_key, 0)
    if not propagated:
        fails.append(('grant_starve_propagates', 'starve swallowed'))
    if debt != 1:
        fails.append(('grant_stamps_debt', 'debt=%d, expected 1' % debt))
    cid2 = st2.sh_strat_deck[0]
    cr.feed('card', ('SHADOW', 'strategy', cid2))
    cr.begin_attempt()
    with cr.advance():
        game_logic.grant_draw(st2, Side.SHADOW, 'strategy')
    debt2 = (getattr(st2, '_owed_draws', None) or {}).get(debt_key, 0)
    if debt2 != 0:
        fails.append(('grant_retry_settles', 'debt=%d after retry'
                      % debt2))
    cr.step_commit()
    # deck-empty contract: RuntimeError, never a starve, no debt
    st3 = fresh_state()
    st3.sh_char_deck = []
    cr.begin_attempt()
    rt = False
    try:
        with cr.advance():
            game_logic.draw_event_card(st3, Side.SHADOW, 'character')
    except RuntimeError as e:
        rt = 'empty' in str(e)
    except R.NeedClientCall:
        rt = False
    if not rt:
        fails.append(('card_empty_deck_contract',
                      'empty deck did not RuntimeError'))
    cr.step_commit()
    # sensitivity: fed cid the deck does not hold -> desync halt
    st4 = fresh_state()
    cr.begin_attempt()
    cr.feed('card', ('SHADOW', 'character', 'BOGUS_CARD_ID'))
    halted = False
    try:
        with cr.advance():
            game_logic.draw_event_card(st4, Side.SHADOW, 'character')
    except RuntimeError as e:
        halted = 'deck desync' in str(e)
    if not halted:
        fails.append(('card_desync_halts', 'bogus cid accepted'))
    cr.reset_feed()
    cr.step_commit()


def test_dwarven_reroll_faces(fails):
    """MEASURES: a Dwarven Ring reroll on the real trajectory consumes
    CLIENT d6 values (ctx='dwarven_ring' on the starve) and repeats
    until the indexed face leaves the original's class - a same-class
    d6 is consumed and rejected, the next lands; the die shows
    faces[(v-1) % len(faces)] of the landing v and one ring is spent."""
    st = fresh_state()
    cr = new_cr()
    import game_logic as gl
    st.fp_dwarven_rings = 1
    gl.adopt_rolled_faces(
        st, [(Side.FREE_PEOPLES, 0, ActionDieFace.CHARACTER)])
    die = st.dice[Side.FREE_PEOPLES][0]
    faces = list(die.faces)
    orig_cls = gl._DWARVEN_FACE_CLASSES.get(die.current, frozenset())
    bad_v = good_v = None
    for i, f in enumerate(faces):
        shares = bool(gl._DWARVEN_FACE_CLASSES.get(f, frozenset())
                      & orig_cls)
        if shares and bad_v is None:
            bad_v = i + 1
        if not shares and good_v is None:
            good_v = i + 1
    if bad_v is None or good_v is None:
        fails.append(('dwarven_control',
                      'NOT-MEASURED: face table gives no bad/good pair'))
        return
    # starve first (sensitivity: the ask names the dwarven context)
    cr.begin_attempt()
    got = None
    try:
        with cr.advance():
            gl.use_dwarven_ring(st, Side.FREE_PEOPLES, die_idx=0)
    except R.NeedClientCall as e:
        got = e
    if got is None or got.kind != 'd6' \
            or got.kw.get('ctx') != 'dwarven_ring':
        fails.append(('dwarven_starve', repr(got and (got.kind,
                                                      got.kw))))
        return
    cr.feed('d6', [bad_v])
    cr.feed('d6', [good_v])
    cr.begin_attempt()
    with cr.advance():
        gl.use_dwarven_ring(st, Side.FREE_PEOPLES, die_idx=0)
    want_face = faces[(good_v - 1) % len(faces)]
    if die.current is not want_face:
        fails.append(('dwarven_lands_client_face', '%r != %r'
                      % (die.current, want_face)))
    if st.fp_dwarven_rings != 0:
        fails.append(('dwarven_ring_spent', 'rings=%d'
                      % st.fp_dwarven_rings))
    if cr.unconsumed()['d6'] != 0:
        fails.append(('dwarven_consumed_both',
                      'the same-class d6 was not consumed: %r'
                      % (cr.unconsumed(),)))
    cr.step_commit()


def test_directive_builders(fails):
    """MEASURES: directives_for renders the exact wire dialect
    (PROTOCOL.md table) - AI-side d6 -> '!CALL combat_roll N';
    HUMAN-side d6 -> a chat ASK with die/dice grammar (G1: each side
    rolls its own); draws -> per-deck '!CALL draw_*'; action_roll ->
    SHADOW pool first (G11); unknown kind halts."""
    cr = R.ClientRandomness(seed=7)                  # no install needed
    d = cr.directives_for(R.NeedClientCall('d6', need=3,
                                           ctx='combat roll',
                                           side=Side.SHADOW))
    if [(x.kind, x.text) for x in d] != [('call', '!CALL combat_roll 3')]:
        fails.append(('dir_d6_ai', repr(d)))
    hu = R.ClientRandomness(human_side=Side.SHADOW, seed=7)
    d = hu.directives_for(R.NeedClientCall('d6', need=3,
                                           ctx='combat roll',
                                           side=Side.SHADOW))
    if [(x.kind, x.text) for x in d] != [('chat', 'roll 3 dice')]:
        fails.append(('dir_d6_human_ask', repr(d)))
    d = hu.directives_for(R.NeedClientCall('d6', need=1, ctx='x',
                                           side='SHADOW'))
    if [(x.kind, x.text) for x in d] != [('chat', 'roll 1 die')]:
        fails.append(('dir_d6_singular', repr(d)))
    d = hu.directives_for(R.NeedClientCall('d6', need=2, ctx='x',
                                           side=Side.FREE_PEOPLES))
    if [(x.kind, x.text) for x in d] != [('call', '!CALL combat_roll 2')]:
        fails.append(('dir_d6_other_side_is_call', repr(d)))
    d = cr.directives_for(R.NeedClientCall('tile'))
    if [(x.kind, x.text) for x in d] != [('call', '!CALL hunt_tile')]:
        fails.append(('dir_tile', repr(d)))
    d = cr.directives_for(R.NeedClientCall(
        'draws', items=[('FREE_PEOPLES', 'character'),
                        ('SHADOW', 'strategy')]))
    if [x.text for x in d] != ['!CALL draw_fp_char',
                               '!CALL draw_sa_strat']:
        fails.append(('dir_draws', repr(d)))
    d = cr.directives_for(R.NeedClientCall(
        'action_roll',
        sides={'FREE_PEOPLES': ('Game.2042', ['210']),
               'SHADOW': ('Game.2043', ['099'])}))
    if [x.text for x in d] != ['!CALL roll_action_dice Game.2043',
                               '!CALL roll_action_dice Game.2042']:
        fails.append(('dir_action_roll_shadow_first', repr(d)))
    raised = False
    try:
        cr.directives_for(R.NeedClientCall('wwsat'))
    except RuntimeError:
        raised = True
    if not raised:
        fails.append(('dir_unknown_kind_halts',
                      'wwsat (no funnel landed) built a directive'))


def test_named_card_pick_funnel(fails):
    """MEASURES: the patched NAMED funnel game_logic.pick_random_card
    starves on the real trajectory (NeedClientCall 'card_pick' with
    ctx) and, once fed, the REAL product site (_wwsat_resolve
    source='hand') discards exactly the fed card — asserted on the
    product's own hand and FACE-DOWN discard pile (1.261), never a
    proxy.
    Sensitivity: a fed id outside the candidate pool halts as a hand
    desync."""
    st = fresh_state()
    cr = new_cr()
    for _ in range(3):
        try:
            game_logic.draw_event_card(st, Side.FREE_PEOPLES,
                                       'character')
        except Exception:
            break
    hand = st.hands[Side.FREE_PEOPLES].hand
    chars = [c for c in hand
             if isinstance(c, str) and 'Character' in c]
    if len(chars) < 2:
        fails.append(('cardpick_precondition',
                      'NOT-MEASURED: no Character cids in hand %r'
                      % hand[:4]))
        return
    target = chars[1]
    cr.begin_attempt()
    got = None
    with cr.advance():
        try:
            game_logic._wwsat_resolve(st, 'hand')
        except R.NeedClientCall as e:
            got = e
    if got is None or got.kind != 'card_pick' \
            or got.kw.get('ctx') != 'WWSAT hand discard':
        fails.append(('cardpick_starve',
                      repr(got and (got.kind, got.kw))))
        return
    if target not in hand:
        fails.append(('cardpick_aborted_attempt_mutated',
                      'starved attempt already removed the card'))
        return
    cr.feed('card_pick', target)
    cr.begin_attempt()
    with cr.advance():
        game_logic._wwsat_resolve(st, 'hand')
    if target in hand:
        fails.append(('cardpick_discards', 'fed card still in hand'))
    # 1.261: the WWSAT hand discard files FACE-DOWN (FP private pile);
    # a face-up appearance is a hidden-information leak
    if target not in st.hands[Side.FREE_PEOPLES].discarded_face_down:
        fails.append(('cardpick_facedown',
                      '1.261: fed card not in the FP face-down pile'))
    dfu = getattr(st, 'discard_face_up', None)
    if isinstance(dfu, dict) and target in dfu.get('fp_character', []):
        fails.append(('cardpick_faceup_leak',
                      '1.261: fed card leaked to the face-up pile'))
    cr.step_commit()
    # sensitivity: fed id outside the pool -> hand desync halt
    cr.begin_attempt()
    cr.feed('card_pick', 'No Such Card Id')
    halted = False
    try:
        with cr.advance():
            game_logic._wwsat_resolve(st, 'hand')
    except RuntimeError as e:
        halted = 'hand desync' in str(e)
    if not halted:
        fails.append(('cardpick_desync_halts', 'bogus id accepted'))
    cr.reset_feed()
    cr.step_commit()


def test_honesty_counter_and_reset(fails):
    """MEASURES: unconsumed() reports exactly what was fed and not
    consumed (non-zero control first, then the zero after reset);
    feed() rejects string d6 payloads and unknown kinds loudly."""
    cr = new_cr()
    cr.feed('d6', [1, 2])
    cr.feed('tile', '3')
    cr.feed('card', ('SHADOW', 'character', 'CID_X'))
    cr.feed('companion', 'Gimli')
    u = cr.unconsumed()
    if (u['d6'], u['tiles'], u['cards'], u['companion']) != (2, 1, 1, 1):
        fails.append(('honesty_nonzero_control', repr(u)))
    cr.reset_feed()
    u2 = cr.unconsumed()
    if any(u2.values()):
        fails.append(('honesty_zero_after_reset', repr(u2)))
    for bad_kind, payload in (('wwsat', ('p', 'c')),
                              ('action_roll', 'SHADOW')):
        try:
            cr.feed(bad_kind, payload)
            fails.append(('feed_rejects_' + bad_kind, 'accepted'))
        except ValueError:
            pass
    try:
        cr.feed('d6', '316')
        fails.append(('feed_rejects_string_d6', 'accepted'))
    except ValueError:
        pass


def test_abort_step(fails):
    """MEASURES (K3): an OPEN step with a journaled pick that is
    ABANDONED (no commit) must be closable - abort_step() discards the
    journal, clears the pending ask and the cursors, and the next
    step's chooser RUNS AGAIN (the product's own counters: step_open,
    _journal, the chooser call count).  Sensitivity control first: the
    same sequence WITHOUT abort replays the stale journal silently
    (same-keyed menu) - the defect class this pin closes."""
    cr = new_cr()
    actions = [_StubAction('x', 0), _StubAction('y', 1),
               _StubAction('z', 2)]
    calls = {'n': 0}

    def chooser(acts):
        calls['n'] += 1
        return list(reversed(acts))        # deterministic pick: 'z'

    # sensitivity control: abandon WITHOUT abort -> stale replay
    cr.begin_attempt()
    cr.journal_pick(actions, chooser)
    with cr.advance():
        try:
            game_logic.roll_d6_batch(1, 'combat roll')
        except R.NeedClientCall:
            pass
    cr.begin_attempt()
    cr.journal_pick(actions, chooser)
    if calls['n'] != 1:
        fails.append(('abort_control',
                      'NOT-MEASURED: the stale journal did not replay '
                      '(calls=%d) - the abort pin below is hollow'
                      % calls['n']))
    # the real thing: abort, then the chooser must run again
    if not hasattr(cr, 'abort_step'):
        fails.append(('abort_api', 'ClientRandomness.abort_step missing'))
        return
    cr.abort_step('test abandon')
    if cr.step_open:
        fails.append(('abort_closes_step', 'step still open'))
    if cr._journal or cr._journal_ix:
        fails.append(('abort_drops_journal',
                      'journal=%r ix=%d' % (cr._journal, cr._journal_ix)))
    if cr.pending() is not None:
        fails.append(('abort_clears_pending', repr(cr.pending())))
    cr.begin_attempt()
    ranked = cr.journal_pick(actions, chooser)
    if calls['n'] != 2:
        fails.append(('abort_chooser_reruns',
                      'chooser ran %d times after abort' % calls['n']))
    if ranked[0].action_type != 'z':
        fails.append(('abort_fresh_pick', repr(ranked[0])))
    # a fed outcome the abandoned step never consumed STAYS banked (the
    # next step's honesty gate reports it - never silently dropped)
    cr.feed('d6', [4])
    cr.abort_step('second abandon')
    if cr.unconsumed()['d6'] != 1:
        fails.append(('abort_keeps_feed', repr(cr.unconsumed())))
    cr.reset_feed()
    # idempotent on a closed step
    cr.abort_step('nothing open')


def test_card_pick_directive(fails):
    """MEASURES (W1#18/#64): the 'card_pick' kind (WWSAT random FP-hand
    discard, funnel game_logic.pick_random_card) has a directive - the
    client's OWN random-discard handler ('!CALL wwsat', AiOpp
    execDirective) - and EVERY kind the funnel contract lists is
    covered by directives_for (a starve that cannot be answered crashed
    the AI step every tick)."""
    cr = R.ClientRandomness(seed=7)
    try:
        d = cr.directives_for(R.NeedClientCall('card_pick',
                                               ctx='WWSAT hand discard'))
    except RuntimeError as e:
        fails.append(('dir_card_pick', 'RuntimeError %s' % e))
        return
    if [(x.kind, x.text) for x in d] != [('call', '!CALL wwsat')]:
        fails.append(('dir_card_pick_text', repr(d)))
    # kind coverage: every documented funnel kind builds SOMETHING
    samples = {
        'd6': dict(need=1, ctx='x', side=Side.SHADOW),
        'tile': {},
        'draws': dict(items=[('SHADOW', 'character')]),
        'action_roll': dict(sides={'SHADOW': ('Game.2043', [])}),
        'randomcomp': dict(ctx='hunt'),
        'card_pick': dict(ctx='WWSAT hand discard'),
    }
    for k, kw in samples.items():
        try:
            if not cr.directives_for(R.NeedClientCall(k, **kw)):
                fails.append(('dir_kind_empty', k))
        except RuntimeError as e:
            fails.append(('dir_kind_uncovered', '%s: %s' % (k, e)))


def main():
    tests = [
        test_abort_step,
        test_card_pick_directive,
        test_starved_d6_funnel,
        test_hunt_reroll_decline_shim,
        test_starved_tile_funnel,
        test_mcts_clone_scoping,
        test_unfunneled_tripwire,
        test_rightmost_excess,
        test_cross_line_surplus,
        test_deterministic_retry_journal,
        test_per_step_rng_stream,
        test_pin_determinism,
        test_named_companion_funnel,
        test_named_card_pick_funnel,
        test_card_funnel_and_grant,
        test_dwarven_reroll_faces,
        test_directive_builders,
        test_honesty_counter_and_reset,
    ]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception as e:
            fails.append((t.__name__, 'EXCEPTION %r' % (e,)))
        except R.NeedClientCall as e:
            fails.append((t.__name__, 'ESCAPED CLIENT CALL %r' % (e,)))
        print('  [%s] %s' % ('PASS' if len(fails) == n0 else 'FAIL',
                             t.__name__))
    print('=' * 60)
    print('TOTAL %d  FAIL %d' % (len(tests), len(fails)))
    for f in fails:
        print('  FAIL:', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
