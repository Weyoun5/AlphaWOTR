"""L0 ORACLES for the live-play lane (TEST_STRATEGY_SPEC.md section 3,
USER 2026-08-25: "your unit tests pass 100% of the time but never
catch any of the bugs I am actually encountering").

Three contracts every live-lane test should end on:

  assert_invariants(state)   the engine state is physically possible
  assert_lane_clean(driver)  the driver has nothing open, nothing
                             unconsumed, nothing alarmed, and (when a
                             census is supplied) agrees with the client
  assert_voiced(items, outcome)  every non-commit outcome spoke; a
                             commit never refused

Each `*_violations` function returns a list of (kind, detail) rows so a
test can assert on the exact defect; the `assert_*` wrappers raise
OracleFailure with the rows.  A test that calls none of these is a
smoke test, not a test (spec L0).

Sensitivity controls live in test_live_oracles.py: every oracle is
proven to FIRE on a broken input, not only to pass on a good one
(instrument discipline: "mutation-test the test").
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class OracleFailure(AssertionError):
    def __init__(self, oracle, rows):
        self.oracle = oracle
        self.rows = list(rows)
        AssertionError.__init__(
            self, '%s: %d violation(s): %s'
            % (oracle, len(self.rows),
               '; '.join('%s=%s' % (k, str(d)[:90]) for k, d in self.rows[:6])))


# ---------------------------------------------------------------------------
# assert_invariants(state)
# ---------------------------------------------------------------------------

def state_invariant_violations(state):
    """(kind, detail) rows.  Checks:
      units_over_components  deployed + casualties > the physical pieces
                             (USER 2026-08-18: never more units than pieces)
      card_in_two_zones      a card id present in two of deck / hand /
                             face-up discard / face-down discard
      hand_over_limit        > 6 cards with no discard pending
      companion_duplicate    a companion both in the Fellowship and on
                             the board, or twice on the board
      army_key_duplicate     two army entries for one region (the
                             'helm s deep' twin-key defect)

    NOT checked: the 6-card hand limit.  It is a TURN-END rule, not a
    state invariant - a player legally holds >6 mid-turn after a draw
    and discards down only at the end of their own turn (BL-148 probe
    2026-08-25: fpgame7 froze at a 'declare' pending with Shadow on 7
    cards, engine conservation clean; an unconditional hand>6 check
    fired on that legal state).
      political_range        box outside 0..3
      political_passive_cap  a passive nation past box 2 (p.34)
      pending_empty          an open pending whose option list is empty
    """
    import ai
    from board import sanitize
    from game_state import Side, Companion
    rows = []
    # -- units vs components
    try:
        dep = ai._count_deployed_all(state)
        fcas = getattr(state, 'fp_casualties', None) or {}
        for nat, mx in ai.NATION_MAX_FIGURES.items():
            d = dep.get(nat, {})
            c = (fcas.get(nat, {})
                 if ai._side_of_nation(nat) == Side.FREE_PEOPLES else {})
            for typ, m in mx.items():
                tot = d.get(typ, 0) + c.get(typ, 0)
                if tot > m:
                    rows.append(('units_over_components',
                                 '%s %s: %d deployed + %d casualties > %d'
                                 % (getattr(nat, 'value', nat), typ,
                                    d.get(typ, 0), c.get(typ, 0), m)))
    except Exception as e:                       # oracle must not hide
        rows.append(('units_check_unreadable', repr(e)))
    # -- cards: no id in two zones
    zones = []
    for attr in ('fp_char_deck', 'fp_strat_deck',
                 'sh_char_deck', 'sh_strat_deck'):
        zones.append((attr, list(getattr(state, attr, None) or ())))
    for side, pi in (getattr(state, 'hands', None) or {}).items():
        zones.append(('hand:%s' % getattr(side, 'name', side),
                      list(getattr(pi, 'hand', None) or ())))
    for k, v in (getattr(state, 'discard_face_up', None) or {}).items():
        zones.append(('discard_face_up:%s' % k, list(v or ())))
    zones.append(('discarded_face_down',
                  list(getattr(state, 'discarded_face_down', None) or ())))
    where = {}
    for zname, cids in zones:
        for cid in cids:
            if cid in where and where[cid] != zname:
                rows.append(('card_in_two_zones',
                             '%s in %s and %s' % (cid, where[cid], zname)))
            where.setdefault(cid, zname)
    # -- hand limit (6) unless a discard is pending
    try:
        dp = ai.describe_pending(state)
    except Exception as e:
        dp = None
        rows.append(('pending_unreadable', repr(e)))
    pend_kind = str((dp or {}).get('kind', '')) if isinstance(dp, dict) else ''
    # -- companions: at most one home
    seen = {}
    fel = getattr(getattr(state, 'fellowship', None), 'companions', None) or []
    for c in fel:
        nm = getattr(c, 'name', None)
        if nm:
            seen[nm] = 'fellowship'
    for key, army in (getattr(state, 'armies', None) or {}).items():
        for l in getattr(army, 'leaders', None) or ():
            if not isinstance(l, Companion):
                continue
            nm = getattr(l, 'name', None)
            if nm in seen:
                rows.append(('companion_duplicate',
                             '%s in %s and %s' % (nm, seen[nm], key)))
            else:
                seen[nm] = str(key)
    # -- army keys unique under sanitize
    keys = {}
    for key in (getattr(state, 'armies', None) or {}):
        s = sanitize(str(key))
        if s in keys:
            rows.append(('army_key_duplicate', '%r and %r' % (keys[s], key)))
        keys[s] = key
    # -- political
    pol = getattr(state, 'political', None)
    if pol is not None:
        for nat, box in (getattr(pol, 'boxes', None) or {}).items():
            if not (0 <= int(box) <= 3):
                rows.append(('political_range',
                             '%s at %r' % (getattr(nat, 'name', nat), box)))
            elif int(box) > 2 and not (getattr(pol, 'active', None)
                                      or {}).get(nat, False):
                rows.append(('political_passive_cap',
                             '%s passive at box %d'
                             % (getattr(nat, 'name', nat), int(box))))
    # -- pending with no options
    if isinstance(dp, dict):
        opts = dp.get('options')
        if isinstance(opts, (list, tuple)) and len(opts) == 0:
            rows.append(('pending_empty', pend_kind or repr(dp)[:80]))
    return rows


def assert_invariants(state):
    rows = state_invariant_violations(state)
    if rows:
        raise OracleFailure('assert_invariants', rows)


# ---------------------------------------------------------------------------
# assert_lane_clean(driver)
# ---------------------------------------------------------------------------

def lane_violations(drv, census_rows=None, dump_path=None):
    """(kind, detail) rows.  Checks on a LiveDriver / ScenarioDriver:
      alarm            any alarm raised
      deferred_window  a window still waiting on client randomness
      open_hold        a matcher hold still open
      pending_buffer   records buffered into an open pending
      unconsumed_rng   client randomness fed and never consumed
      census           rows from `census_rows` (or computed from
                       `dump_path` against the driver's engine state)
      state_invariant  assert_invariants on the driver's engine state
    """
    rows = []
    for a in list(getattr(drv, 'alarms', None) or ()):
        rows.append(('alarm', str(a)[:160]))
    if getattr(drv, 'deferred', None) is not None:
        w = drv.deferred[0] if isinstance(drv.deferred, tuple) else drv.deferred
        rows.append(('deferred_window',
                     '%d record(s)' % len(getattr(w, 'records', None) or ())))
    if getattr(drv, 'holds', None):
        rows.append(('open_hold', '%d hold(s)' % len(drv.holds)))
    pb = getattr(drv, 'pending_buf', None)
    if pb is not None and getattr(pb, 'records', None):
        rows.append(('pending_buffer', '%d record(s)' % len(pb.records)))
    rand = getattr(drv, 'rand', None)
    if rand is not None and hasattr(rand, 'unconsumed'):
        try:
            un = rand.unconsumed()
        except Exception as e:
            un = {'unreadable': repr(e)}
        if isinstance(un, dict):
            bad = {k: v for k, v in un.items() if v}
        else:
            bad = un if un else {}
        if bad:
            rows.append(('unconsumed_rng', str(bad)[:160]))
    if census_rows is None and dump_path:
        try:
            from wotr_bridge.live import client_board as _cb
            census_rows = _cb.diff_engine_vs_dump(
                drv.es, _cb.read_board(dump_path))
        except Exception as e:
            rows.append(('census_unreadable', repr(e)))
            census_rows = None
    for r in list(census_rows or ())[:20]:
        rows.append(('census', str(r)[:160]))
    es = getattr(drv, 'es', None)
    if es is not None:
        for k, d in state_invariant_violations(es):
            rows.append(('state_invariant:%s' % k, d))
    return rows


def assert_lane_clean(drv, census_rows=None, dump_path=None):
    rows = lane_violations(drv, census_rows=census_rows, dump_path=dump_path)
    if rows:
        raise OracleFailure('assert_lane_clean', rows)


# ---------------------------------------------------------------------------
# assert_voiced(items, outcome)
# ---------------------------------------------------------------------------

NON_COMMIT = ('refused', 'hold', 'ambiguous', 'none', 'deferred')


def voice_violations(items, outcome):
    """(kind, detail) rows.  `items` = the driver's out tuples for one
    tick (('chat', kind, text) / ('note', text) / ...); `outcome` in
    {'match'} | NON_COMMIT.
      silent_outcome   a non-commit outcome produced no chat line
      refusal_on_match a commit tick carried a refusal chat line
    """
    chats = [it for it in (items or ()) if it and it[0] == 'chat']
    refusals = [it for it in chats
                if len(it) > 1 and str(it[1]) in ('refusal', 'refuse')]
    if outcome in NON_COMMIT and not chats:
        return [('silent_outcome', '%s with %d item(s), no chat'
                 % (outcome, len(items or ())))]
    if outcome == 'match' and refusals:
        return [('refusal_on_match', str(refusals[0])[:160])]
    return []


def assert_voiced(items, outcome):
    rows = voice_violations(items, outcome)
    if rows:
        raise OracleFailure('assert_voiced', rows)
