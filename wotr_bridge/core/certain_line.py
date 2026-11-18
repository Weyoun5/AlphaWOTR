"""THE CERTAIN LINE.  "An opponent will only make a decision if it benefits them": a line that reaches a position
for certain is never worse than a line that reaches the same position only if the opponent chooses so.  Asked at every
decision of the AI's own die action, after its pick is made:

  the reference line   the pick, then each side's own net (one net call a decision, no search), dice from the engine,
                       to the end of the mover's NEXT die action (OWN_ACTIONS).  It is uncertain when the opponent had
                       a real choice INSIDE one of the mover's actions (withdraw into the siege or fight), or a die was
                       rolled in one.
  a certain line       another first move, then the mover's own steps with the same dice: no roll, no choice of the
                       opponent inside them; the opponent's OWN die action in between is played as in the reference
                       line.  The steps are searched toward the reference line's end: an option is tried when every
                       region it names still differs from that end and (an army move) it leaves its origin or fills its
                       destination with the end's count of Regulars and Elites; it is kept when it sets one of its
                       regions to its final content.  Bounds: APPLIES options (2 ms each), MENUS, SECONDS.
  the verdict          a certain line ends in the SAME position (ai._outcome: everything else equal, who acts next
                       included; the counts equal or better).  Of all certain lines found, the one whose first move
                       the mover's net rates highest:
                         - the reference line needed the opponent's choice -> that first move is PLAYED ("outright");
                         - it needed a ROLL -> no proof (the end compared is one roll of many): the first move joins
                           the turn search's candidates with the bar at its floor ("favoured");
                         - it was certain itself -> only a line strictly better on the counts wins.
                       No verdict when the pick itself starts a certain line.

THE MOVER'S PICK WITHIN A COUNTED SPLIT.  An army move names how many Regulars and Elites go, not which: the engine
takes the first of the army's roster, a player picks.  Where a step splits a kind by count out of an army holding that
kind of several nations, the comparison takes the pick that fits the end position (interchangeable FOR THAT STEP);
positions that differ by nations in any other way stay different.

LIMITS.  The opponent's own action in between is held the same in both lines (he may answer a certain line
differently); options that name no region (cards, the Hunt) are not searched; a reference line in which the opponent's
own action rolled, or the mover had to answer inside it, is not judged, nor one that needs a net this process has not.
"""
import random
import time
from collections import Counter

import numpy as np

OWN_ACTIONS = 2          # the reference line: to the end of the mover's next die action
STEPS = 40               # decisions of one line, at most
APPLIES = 2000           # options applied per decision in the search, at most
MENUS = 400              # menus built per decision in the search, at most
SECONDS = 15.0
DEADLINE = None          # TOURNAMENT-DEADLINE: a time.monotonic() value set by a caller on a clock (the protocol adapter's
                         # per-decision clock); past it the reference line gives no verdict and the search stops (cut)
REGION_KEYS = ('from', 'to', 'region')
LAST = {}                # the last decision's figures (the log line, reports)


def _same(a, b):
    return a.action_type == b.action_type and a.die_idx == b.die_idx and a.params == b.params


def _decider(ai, state, owner):
    """-> (the side that decides next, is it the start of a new die action); (None, False) at the end."""
    if getattr(state, 'game_over', False) or not getattr(state, 'dice_rolled', False):
        return None, False
    side = ai._pending_decision_side(state)
    if side is not None:
        return side, False
    if state.pending_army_move_die is not None or state.pending_muster_die is not None:
        return owner, False
    return ai._next_acting_side(state, owner), True


def _end(ai, state, side, owner):
    """A line's end as a position: (everything else + who acts next, the counts).  The turn-routing mark a resolved
    battle leaves (the next die belongs to the non-attacker) is replaced by what it decides."""
    plain = state
    if getattr(state, '_battle_resolved_attacker', None) is not None:
        plain = ai.clone_state(state)
        plain._battle_resolved_attacker = None
    out = ai._outcome(plain, side)
    return None if out is None else ((out[0], _decider(ai, state, owner)[0]), out[1])


def _regions(ai, action):
    p = action.params if isinstance(action.params, dict) else {}
    return [ai.sanitize(str(p[k])) for k in REGION_KEYS if p.get(k) is not None]


def _pick(ai, seats, state, side, menu):
    """A side's net's own first choice (None: this process has no net for that side)."""
    net = seats.net(side)
    return None if net is None else ai._greedy_prior_action(state, side, net, list(menu))


class _Search(object):
    def __init__(self, ai, seats, side, ref):
        self.ai, self.seats, self.side, self.ref = ai, seats, side, ref
        self.end_key, self.end_q = ref['key']
        self.end_own = self.end_q['own']
        self.strict = ref['kind'] is None
        self.applies = self.menus = 0
        self.began, self.cut = time.perf_counter(), False

    def out(self):
        if self.applies >= APPLIES or self.menus >= MENUS or time.perf_counter() - self.began > SECONDS or \
                (DEADLINE is not None and time.monotonic() > DEADLINE):
            self.cut = True
        return self.cut

    def own(self, state):
        return self.ai._outcome(state, self.side)[1]['own']

    def reached(self, state, owner):
        got = _end(self.ai, state, self.side, owner)
        if got is None or got[0] != self.end_key:
            return False
        return (not self.strict) if got[1] == self.end_q else self.ai._at_least_as_good(self.side, got[1], self.end_q)

    def picked(self, state, action, now):
        """`state` with the origin army's roster put so that a counted split takes the units the end position has at
        the option's destination; `state` itself when the split leaves no choice."""
        p = action.params if isinstance(action.params, dict) else {}
        if p.get('from') is None or p.get('to') is None:
            return state
        ai = self.ai
        src, dst = ai.sanitize(str(p['from'])), ai.sanitize(str(p['to']))
        key = next((k for k in state.armies if ai.sanitize(str(k)) == src), None)
        units = list(state.armies[key].units) if key is not None else []
        if not any(0 < int(p.get(count) or 0) < sum(1 for u in units if u.kind.name == kind)
                   and len({u.nation for u in units if u.kind.name == kind}) > 1
                   for kind, count in (('REGULAR', 'n_regular'), ('ELITE', 'n_elite'))):
            return state
        want = Counter(self.end_own.get(dst, {}))
        want.subtract(now.get(dst, {}))
        clone = ai.clone_state(state)
        army = clone.armies[key]
        first, rest = [], []
        for u in army.units:
            k = (u.nation.name, u.kind.name)
            if want.get(k, 0) > 0:
                want[k] -= 1
                first.append(u)
            else:
                rest.append(u)
        army.units = type(army.units)(first + rest)
        return clone

    def fits(self, act, regs, now):
        """Before an army move is applied: does it leave its origin, or fill its destination, with as many Regulars
        and Elites as the end position has there?  (Other options: yes.)"""
        p = act.params if isinstance(act.params, dict) else {}
        if p.get('from') is None or p.get('to') is None or not ('n_regular' in p or 'n_elite' in p):
            return True

        def count(board, region):
            units = board.get(region, {})
            return [sum(v for k, v in units.items() if isinstance(k, tuple) and k[1] == kind) for kind in ('REGULAR', 'ELITE')]
        moved = [int(p.get('n_regular') or 0), int(p.get('n_elite') or 0)]
        return ([a - m for a, m in zip(count(now, regs[0]), moved)] == count(self.end_own, regs[0])
                or [a + m for a, m in zip(count(now, regs[1]), moved)] == count(self.end_own, regs[1]))

    def steps(self, state, menu):
        """The own options worth a try here, each applied: [(option, the state after it)]."""
        ai, now = self.ai, self.own(state)
        differ = {r for r in set(now) | set(self.end_own) if now.get(r, {}) != self.end_own.get(r, {})}
        out = []
        for act in menu:
            regs = _regions(ai, act)
            if act.action_type != 'pass' and not (regs and all(r in differ for r in regs) and self.fits(act, regs, now)):
                continue
            if self.out():
                break
            self.applies += 1
            try:
                after, rolled = ai._apply_action_detect_rng(self.picked(state, act, now), act)
            except Exception:
                continue
            if rolled:
                continue
            if act.action_type != 'pass':
                board = self.own(after)
                if not any(board.get(r, {}) == self.end_own.get(r, {}) for r in regs):
                    continue                     # (it sets none of its regions to the end position's content)
            out.append((act, after))
        return out

    def walk(self, state, owner, begun, ri):
        """Does a certain line run from `state` to the reference line's end?"""
        ai, side, ref = self.ai, self.side, self.ref
        for _ in range(STEPS):
            if self.out():
                return False
            d, fresh = _decider(ai, state, owner)
            done = begun == ref['begun'] and ri == len(ref['replies'])
            if d is None or (fresh and done):
                return done and self.reached(state, owner)
            if fresh:
                if d == side:
                    if begun == ref['begun']:
                        return False
                    begun += 1
                elif ri == len(ref['replies']):
                    return False
                owner = d
            self.menus += 1
            menu = self.seats.menu(state, d)
            if not menu:
                return False
            if d == side:
                if owner != side:
                    return False                 # (the mover answering inside the opponent's action: not searched)
                return any(self.walk(after, owner, begun, ri) for _act, after in self.steps(state, menu))
            if owner == side:
                if len(menu) > 1:
                    return False                 # the opponent decides inside the mover's action: not certain
                act = menu[0]
            else:
                act = next((m for m in menu if _same(m, ref['replies'][ri])), None) if ri < len(ref['replies']) else None
                if act is None:
                    return False                 # his action of the reference line is not open to him here
                ri += 1
            self.applies += 1
            state, rolled = ai._apply_action_detect_rng(state, act)
            if rolled:
                return False
        return False


def _reference(ai, seats, state, side, own):
    """The pick, then each side's own net, to the end of the mover's next die action."""
    cur, rolled = ai._apply_action_detect_rng(state, own)
    kind, owner, begun, replies = ('roll' if rolled else None), side, 1, []
    for _ in range(STEPS):
        if DEADLINE is not None and time.monotonic() > DEADLINE:
            return None                          # (TOURNAMENT-DEADLINE: out of time, no verdict)
        d, fresh = _decider(ai, cur, owner)
        if d is None:
            break
        if fresh:
            if begun == OWN_ACTIONS:
                break
            begun += d == side
            owner = d
        menu = seats.menu(cur, d)
        act = menu[0] if len(menu) == 1 else _pick(ai, seats, cur, d, menu) if menu else None
        if act is None or (d == side and owner != side):
            return None                          # (no net for that side, or the mover answering inside his action)
        if d != side and owner != side:
            replies.append(act)
        elif d != side and len(menu) > 1 and kind is None:
            kind = 'decision'
        cur, rolled = ai._apply_action_detect_rng(cur, act)
        if rolled:
            if owner != side:
                return None                      # the opponent's own action rolled: it cannot be played again
            kind = 'roll'
    else:
        return None
    key = _end(ai, cur, side, owner)
    return None if key is None else dict(key=key, kind=kind, begun=begun, replies=replies)


def find(ai, seats, state, side, owner, menu, own):
    """own: the AI's pick (an entry of menu), at a decision of `side`'s own die action.
    seats: .menu(state, side) -> the options that side is offered; .net(side) -> its policy net or None.
    -> None, or ('outright' | 'favoured', the entry of menu that starts the certain line the net rates highest)."""
    LAST.clear()
    pending = ai._pending_decision_side(state)
    if len(menu) < 2 or pending not in (None, side) or (pending is not None and owner not in (None, side)):
        return None                              # (not a decision of the mover's own die action)
    began = time.perf_counter()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        ref = _reference(ai, seats, state, side, own)
        LAST.update(reference=None if ref is None else ref['kind'] or 'certain')
        if ref is None:
            return None
        search = _Search(ai, seats, side, ref)
        found = [act for act, after in search.steps(state, menu) if search.walk(after, side, 1, 0)]
        LAST.update(applies=search.applies, menus=search.menus, cut=search.cut, lines=len(found),
                    seconds=time.perf_counter() - began)
        if not found or any(_same(act, own) for act in found):
            return None
        priors = ai._get_priors_and_value(state, side, seats.net(side), list(menu))[0]
        rated = {id(a): float(p) for a, p in zip(menu, priors)}
        best = max(found, key=lambda act: rated.get(id(act), 0.0))
        LAST.update(verdict='favoured' if ref['kind'] == 'roll' else 'outright')
        return LAST['verdict'], best
    except Exception as error:                   # (a line the engine cannot play is no verdict)
        LAST.update(error='%s: %s' % (type(error).__name__, str(error)[:200]))
        return None
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
