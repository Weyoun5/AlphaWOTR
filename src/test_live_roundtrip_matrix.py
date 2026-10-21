"""L2 ROUND-TRIP MATRIX (TEST_STRATEGY_SPEC.md section 3 L2) - the
missing specification of the human-gesture lane.

For each corpus state x side x legal action: render the action as the
HUMAN would (the natural client-feed grammar a human's gesture logs -
NOT the emitter's inject tokens, which the client expands), feed it
through a LiveDriver whose randomness funnel is the production shape,
and assert the driver COMMITS a chain whose post-state fingerprint
equals apply(state, action); then the L0 lane oracle.

Failure is TYPED so the scoreboard is legible:
  pass          committed the state-identical chain, lane clean
  refused       matcher rejected a legal human form (a real bug)
  held          deferred / ambiguous with no progress
  wrong_action  committed a DIFFERENT state (fingerprint mismatch)
  diverged      committed but the L0 lane oracle found a break
  silent        no commit and no voice (the BL-146 class)

Deferrals (LOUD, never a silent cap - instrument discipline):
  needs_randomness  the action consumes a client roll/draw/tile; the
                    driver would need the same stream fed (Phase 1b)
  no_renderer       no exact human-feed template for this action type
                    yet (the renderer grows per type; a wrong render
                    would test nothing, so we defer instead)
  render_skip       the action produces no window (pass), or an
                    apply/precondition the harness cannot set up

COVERAGE TODAY: army_move (single unit-type, no in-group leaders).
This is where BL-77 / BL-136 / BL-146 lived.  Every other type is
counted and deferred by name until its renderer lands.

Run:  python test_live_roundtrip_matrix.py
      python test_live_roundtrip_matrix.py --full   (all corpus states)
"""
import os
import sys
from collections import Counter

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_SRC, _ROOT, os.path.join(_ROOT, 'wotr_bridge')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import ai                                                  # noqa: E402
import emitter                                             # noqa: E402
from board import sanitize                                 # noqa: E402
from game_state import Side                                # noqa: E402
from wotr_bridge.live import emission, gesture_match as gm  # noqa: E402
from wotr_bridge.live.randomness import (ClientRandomness,  # noqa: E402
                                         NeedClientCall)
from wotr_bridge.live.client_view import ClientView        # noqa: E402
import live_oracles as lo                                  # noqa: E402
from test_live_driver import Rig, fresh_state, adopted_state  # noqa: E402
from test_live_matcher import token_for                    # noqa: E402

NICK = 'Wey'
_SIDEWORD = {Side.SHADOW: 'Shadow', Side.FREE_PEOPLES: 'Free'}
_DICEZONE = {Side.SHADOW: 'SA Dice', Side.FREE_PEOPLES: 'FP Dice'}


def other(s):
    return Side.SHADOW if s is Side.FREE_PEOPLES else Side.FREE_PEOPLES


def board_fingerprint(state):
    """Board OUTCOME identity, EXCLUDING the action-dice pool.  The
    driver consumes the die as part of the human gesture (the die-kill
    line); apply_action's atomic die bookkeeping differs, so the pool
    is not a valid cross-check - the die-kill is verified by the driver
    committing at all.  Everything else (armies, political, fellowship,
    named leaders, rings, hands, table) + the open pending + VP IS the
    outcome the gesture must reproduce."""
    from wotr_bridge.live import client_board as _cb
    proj = dict(_cb.proj_board(state))
    proj.pop('dice', None)
    try:
        dp = ai.describe_pending(state)
        pend = None if dp is None else (str(dp.get('kind')),
                                        str(dp.get('side')))
    except Exception:
        pend = 'err'
    vp = getattr(state, 'vp', None)
    return (repr(sorted(proj.items(), key=lambda kv: str(kv[0]))), pend,
            (getattr(vp, 'free_peoples', None), getattr(vp, 'shadow', None)))


def make_view(holder):
    v = ClientView(lambda: holder[0])
    v.card_pid_map = {}
    for i, pid in enumerate(emitter._FP_DIE_IDS[:4]):
        v.bind_die(pid, Side.FREE_PEOPLES, i)
    for i, pid in enumerate(emitter._SH_DIE_IDS[:7]):
        v.bind_die(pid, Side.SHADOW, i)
    return v


def _die_letter(state, side, die_idx):
    try:
        face = state.dice[side][die_idx].current
        return emitter._FACE_LETTER.get(getattr(face, 'name', None))
    except Exception:
        return None


def _from_nation(state, region):
    for k, army in state.armies.items():
        if sanitize(k) == sanitize(region):
            nats = {u.nation for u in army.units}
            leaders = [l for l in army.leaders]
            return nats, leaders
    return set(), []


# ---------------------------------------------------------------------------
# human_feed_lines: the natural client-feed a human's gesture logs.
# Returns (lines, None) when rendered exactly, or (None, reason) to defer.
# ---------------------------------------------------------------------------

def _plain_nazgul_at(state, region):
    """(count, any_named) of Nazgul leaders at a region."""
    _, leaders = _from_nation(state, region)
    plain = [l for l in leaders if type(l).__name__ == 'Nazgul'
             and not str(getattr(l, 'name', '') or '')]
    named = [l for l in leaders if type(l).__name__ == 'Nazgul'
             and str(getattr(l, 'name', '') or '')]
    return len(plain), bool(named)


def _die_lines(side, letter):
    sw, dz = _SIDEWORD[side], _DICEZONE[side]
    return ('$%s moves %s Action Die (%s) from %s to Current Action.'
            % (NICK, sw, letter, dz),
            '$%s kills %s Action Die (%s) in Current Action.'
            % (NICK, sw, letter))


def human_feed_lines(state, side, action, post):
    at = action.action_type
    letter = _die_letter(state, side, action.die_idx)
    if at == 'nazgul_move':
        return _nazgul_move_lines(state, side, action, post, letter)
    if at != 'army_move':
        return None, 'no_renderer:%s' % at
    p = action.params or {}
    # DEFER any move that carries a Nazgul or a leader with the units -
    # the group-token render (e.g. 'group (Nazgul, Sauron Elite)') is
    # not built yet, and rendering the units ALONE under-specifies the
    # gesture (the driver would correctly leave the leader behind and
    # the board would diverge - a FALSE test).  Correctness over
    # coverage: render only what we render exactly.
    if p.get('n_nazgul') or p.get('n_leaders'):
        return None, 'no_renderer:army_move+leaders'
    if p.get('take_leaders') and _from_nation(state, p.get('from', ''))[1]:
        return None, 'no_renderer:army_move+leaders'
    n_reg = int(p.get('n_regular') or 0)
    n_eli = int(p.get('n_elite') or 0)
    if bool(n_reg) == bool(n_eli):            # need exactly one type
        return None, 'no_renderer:army_move_mixed'
    kind_word = 'Elite' if n_eli else 'Regular'
    count = n_eli or n_reg
    nats, _ = _from_nation(state, p.get('from', ''))
    if len(nats) != 1:
        return None, 'no_renderer:army_move_multination'
    nat = next(iter(nats))
    try:
        tok = token_for(nat, kind_word, count)
    except Exception as e:
        return None, 'no_renderer:token(%s)' % (repr(e)[:30])
    letter = _die_letter(state, side, action.die_idx)
    if letter is None:
        return None, 'no_renderer:die_letter'
    sw, dz = _SIDEWORD[side], _DICEZONE[side]
    stage, kill = _die_lines(side, letter)
    lines = [stage,
             '$%s moves %s from %s to %s.'
             % (NICK, tok, str(p['from']).title(), str(p['to']).title()),
             kill]
    return lines, None


def _nazgul_move_lines(state, side, action, post, letter):
    """Nazgul move: the plain-Nazgul count that actually LEFT the source
    (from the pre/post diff, so the count is exact), rendered as
    'Nazgul' / '[n]Nazgul'.  Named minions (Witch-king, Mouth) defer -
    they render by name, not built yet."""
    if letter is None:
        return None, 'no_renderer:die_letter'
    p = action.params or {}
    frm = p.get('from')
    to = p.get('to')
    if not frm or not to:
        return None, 'no_renderer:nazgul_noregion'
    pre_n, pre_named = _plain_nazgul_at(state, frm)
    post_n, _ = _plain_nazgul_at(post, frm)
    if pre_named:
        return None, 'no_renderer:nazgul_named'
    moved = pre_n - post_n
    if moved <= 0:
        return None, 'no_renderer:nazgul_zero'
    tok = 'Nazgul' if moved == 1 else '[%d]Nazgul' % moved
    stage, kill = _die_lines(side, letter)
    lines = [stage,
             '$%s moves %s from %s to %s.'
             % (NICK, tok, str(frm).title(), str(to).title()),
             kill]
    return lines, None


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------

def round_trip(state, side, action):
    r = ClientRandomness(seed=13)
    r.install()
    ref = ai.clone_state(state)
    try:
        with r.advance():
            post = ai.apply_action(ref, action)
    except NeedClientCall:
        return 'needs_randomness', 'apply'
    except Exception as e:
        return 'apply_error', repr(e)[:80]
    lines, why = human_feed_lines(state, side, action, post)
    if lines is None:
        return why.split(':', 1)[0], why
    drv_state = ai.clone_state(state)
    rig = Rig(drv_state, other(side), phase='actions')
    try:
        rig.feed(*lines)
    except Exception as e:
        return 'feed_error', repr(e)[:80]
    drv = rig.drv
    committed = bool(drv.commits) and drv.commits[-1][2] == 'window'
    if not committed:
        ws = drv.window_stats
        if ws.get('ambiguous'):
            return 'held', 'ambiguous %s' % ws
        if ws.get('hold') or drv.deferred is not None or drv.holds:
            return 'held', str(ws)
        # no commit, no hold: was anything voiced (a refusal)?
        voiced = any(it and it[0] == 'chat' for it in (drv._out_queue or []))
        return ('refused' if voiced else 'silent'), str(ws)
    if board_fingerprint(drv.es) != board_fingerprint(post):
        return 'wrong_action', 'board mismatch'
    lane = lo.lane_violations(drv)
    if lane:
        return 'diverged', str(lane[:2])[:120]
    return 'pass', None


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def _rollout(state, side, n, seed):
    """Apply up to n random legal deterministic actions (no randomness)
    to reach a varied non-pending action-phase state."""
    import random
    rng = random.Random(seed)
    st = ai.clone_state(state)
    for _ in range(n):
        try:
            if ai.describe_pending(st) is not None:
                break
        except Exception:
            break
        menu = [a for a in ai.get_legal_actions(st, side)
                if a.action_type in ('army_move', 'nazgul_move',
                                     'advance_political')]
        if not menu:
            break
        a = rng.choice(menu)
        r = ClientRandomness(seed=1)
        r.install()
        try:
            with r.advance():
                st = ai.apply_action(st, a)
        except Exception:
            break
    return st


def build_corpus(full=False):
    corpus = []
    base = adopted_state()
    corpus.append(('setup', base))
    n = 8 if full else 3
    for i in range(n):
        corpus.append(('rollout-sh-%d' % i,
                       _rollout(base, Side.SHADOW, 2 + i % 3, 100 + i)))
        corpus.append(('rollout-fp-%d' % i,
                       _rollout(base, Side.FREE_PEOPLES, 2 + i % 3, 200 + i)))
    return corpus


def main(argv):
    full = '--full' in argv
    corpus = build_corpus(full)
    board = Counter()          # (action_type, outcome) -> n
    per_type_total = Counter()
    examples = {}
    n_states = 0
    for label, st in corpus:
        try:
            if ai.describe_pending(st) is not None:
                continue           # Phase 1: non-pending action phase
        except Exception:
            continue
        n_states += 1
        for side in (Side.SHADOW, Side.FREE_PEOPLES):
            try:
                menu = ai.get_legal_actions(st, side)
            except Exception:
                continue
            for a in menu:
                per_type_total[a.action_type] += 1
                outcome, detail = round_trip(st, side, a)
                board[(a.action_type, outcome)] += 1
                if outcome in ('refused', 'silent', 'wrong_action',
                               'diverged') and (a.action_type, outcome) \
                        not in examples:
                    examples[(a.action_type, outcome)] = (label, side.name,
                                                          detail)
    # scoreboard
    print('L2 ROUND-TRIP MATRIX  (%d non-pending states, %s)'
          % (n_states, 'full' if full else 'sample'))
    print('=' * 66)
    RED = ('refused', 'silent', 'wrong_action', 'diverged')
    n_red = 0
    covered = 0
    for at in sorted(per_type_total):
        outs = {o: c for (t, o), c in board.items() if t == at}
        tot = per_type_total[at]
        passes = outs.get('pass', 0)
        red = sum(outs.get(o, 0) for o in RED)
        n_red += red
        covered += passes + red
        tag = 'RED %d' % red if red else ('  %d ok' % passes if passes
                                          else 'deferred')
        detail = ' '.join('%s=%d' % (o, c) for o, c in sorted(outs.items()))
        print('  %-22s n=%-5d %-10s %s' % (at, tot, tag, detail))
    print('-' * 66)
    print('COVERED (round-tripped): %d ; RED (bugs): %d' % (covered, n_red))
    for (at, o), (label, side, detail) in sorted(examples.items()):
        print('  RED %-18s %-12s %s/%s :: %s'
              % (at, o, label, side, str(detail)[:60]))
    # runner-parseable summary (covered = the tests that actually ran)
    print('TOTAL %d  FAIL %d' % (covered, n_red))
    return 1 if n_red else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
