"""The adapter's deadline on a per-decision clock (no pytest, no engine needed):

    python -I protocol/test_deadline.py

A fake search stands in for their OwnModelSearch: it records the deadline and the simulation budget it is given and
"runs" until the deadline.  Checked: the answer's deadline from the line's arrival (the clock less the 200 ms margin;
game clocks and correspondence get none), the search's deadline (the reserve for what follows the search, at least 30%
of the time left for the search), the budget from the measured rate (capped by --sims), what settle() learns, and a
whole decision through LeagueSearcher.choose answered before the deadline.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wotr_protocol_bot as P  # noqa: E402

FAILS = []


def check(cond, what):
    print('%s %s' % ('ok  ' if cond else 'FAIL', what))
    if not cond:
        FAILS.append(what)


def searcher(sims=3000):
    s = object.__new__(P.LeagueSearcher)
    s.sims, s.max_seconds = sims, None
    s.rates, s.recent = {}, {}
    s.deadline, s.loop_rates, s.tails, s.certains, s.pending = None, {}, {}, {}, None
    s.can_stop = True
    return s


def bot(kind='per_decision'):
    b = object.__new__(P.AlphaWotrBot)
    b.clock_kind = kind
    b.MARGIN_MS = 200
    return b


def test_answer_deadline():
    b = bot()
    check(abs(b._deadline({'left_ms': 2000}, 100.0) - 101.8) < 1e-9, 'per decision 2 s: answer by arrival + 1.8 s')
    check(abs(b._deadline({'left_ms': 150}, 100.0) - 100.0) < 1e-9, 'a clock under the margin: answer at once')
    check(bot('game')._deadline({'left_ms': 600000, 'opponent_ms': 1}, 1.0) is None, 'a game clock: no deadline')
    check(bot('correspondence')._deadline({'left_ms': 5000, 'deadline': 'x'}, 1.0) is None,
          'correspondence: no deadline')
    check(b._deadline({}, 1.0) is None, 'no clock: no deadline')
    b.MARGIN_MS = 500
    check(abs(b._deadline({'left_ms': 2000}, 0.0) - 1.5) < 1e-9, '--margin-ms 500: answer by 1.5 s')


def test_plan():
    s = searcher()
    s.deadline = 11.8                                # 1.8 s from now = 10.0
    n, until = s.deadline_plan('ACTION', now=10.0)
    first = s.FIRST_TAIL * 1.5 + s.FIRST_CERTAIN + s.AFTER
    check(abs(until - (11.8 - first)) < 1e-9, 'a kind not seen: the first reserve (%.3f s)' % first)
    check(n == int(s.FIRST_RATE * (until - 10.0) * s.SHAPE), 'its budget from FIRST_RATE x SHAPE (%d)' % n)
    s.loop_rates['ACTION'] = [100.0, 120.0]
    s.tails['ACTION'] = [0.02, 0.08, 0.05]
    s.certains['ACTION'] = [0.01, 0.03]
    n, until = s.deadline_plan('ACTION', now=10.0)
    reserve = 0.08 * 1.5 + 0.02 + s.AFTER
    check(abs(until - (11.8 - reserve)) < 1e-9, 'the reserve: 1.5 x the most tail + the certain mean + AFTER')
    check(n == int(110.0 * (until - 10.0) * s.SHAPE), 'the budget from the mean rate (%d)' % n)
    s.certains['ACTION'] = [1.0]
    n, until2 = s.deadline_plan('ACTION', now=10.0)
    check(abs((11.8 - until2) - (0.08 * 1.5 + 1.8 / 6 + s.AFTER)) < 1e-9, "the certain line's allowance at most a sixth")
    s.tails['ACTION'] = [5.0]
    n, until3 = s.deadline_plan('ACTION', now=10.0)
    check(abs(until3 - (10.0 + 0.3 * 1.8)) < 1e-9, 'a huge tail: the search still gets 30% of the time')
    s.sims = 50
    check(s.deadline_plan('ACTION', now=10.0)[0] == 50, 'the budget capped by --sims')
    s.sims = 3000
    s.deadline = 10.0
    n, until4 = s.deadline_plan('ACTION', now=10.5)
    check(until4 == 10.5 and n == 8, 'past the deadline: stop at once, the smallest budget (8)')
    s.deadline, s.max_seconds = 20.0, 1.0
    n, until5 = s.deadline_plan('ACTION', now=10.0)
    check(until5 < 11.0, '--max-seconds caps the time as well')


def test_settle():
    s = searcher()
    s.pending = {'kind': 'ARMY_MOVE', 'returned': 5.0, 'certain': 0.1,
                 'timing': {'simulations_run': 40, 'loop': 0.5, 'tail': 0.03, 'stop_to_end': 0.01}}
    s.settle(done=5.2)
    check(abs(s.loop_rates['ARMY_MOVE'][0] - 80.0) < 1e-9, 'settle: the loop rate (simulations / loop seconds)')
    check(abs(s.tails['ARMY_MOVE'][0] - (0.01 + 0.03 + 0.2 - 0.1)) < 1e-9,
          'settle: the tail = stop to end + search tail + after the search less the certain line')
    check(s.certains['ARMY_MOVE'] == [0.1] and s.pending is None, 'settle: the certain line kept, pending cleared')
    for i in range(20):
        s.pending = {'kind': 'ARMY_MOVE', 'returned': 0.0, 'timing': {'simulations_run': 10, 'loop': 0.1}}
        s.settle(done=0.0)
    check(len(s.tails['ARMY_MOVE']) == s.KEEP and len(s.loop_rates['ARMY_MOVE']) == 3, 'settle keeps the last few')
    s.settle()
    check(s.pending is None, 'settle without a search: nothing')


class FakeAction(object):
    def __init__(self, name):
        self.action_type, self.die_idx, self.params = name, 0, {}


class FakeSearch(object):
    """OwnModelSearch's interface: choose(state, side, num_simulations=, deadline=) runs until the deadline."""

    def __init__(self, acts, rate=100.0, tail=0.03):
        self.acts, self.rate, self.tail, self.calls = acts, rate, tail, []

    def choose(self, state, side, num_simulations=300, deadline=None):
        t0 = time.monotonic()
        self.calls.append((num_simulations, deadline))
        run = 0
        while run < num_simulations:
            if run and deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(1.0 / self.rate)
            run += 1
        loop = time.monotonic() - t0
        time.sleep(self.tail)
        return self.acts[0], {'q_values': [0.1] * len(self.acts), 'played': 0,
                              'search_timing': {'simulations_run': run, 'budget': num_simulations, 'loop': loop,
                                                'tail': self.tail, 'stopped': run < num_simulations,
                                                'stop_to_end': 0.0}}

    def raw(self, state, side, menu):
        return menu[0]


class FakeTurnSearch(object):
    class certain_line(object):
        DEADLINE = None

    seen = []

    @staticmethod
    def Seats(menu, seats):
        return None

    @classmethod
    def certain(cls, ai, seats, st, side, owner, menu, hit):
        cls.seen.append(cls.certain_line.DEADLINE)
        return None, None

    @staticmethod
    def improve(ai, st, side, owner, seed, menu, hit, setting, favoured=None):
        return hit


class FakeUnits(object):
    @staticmethod
    def pick(ai, st, side, pick):
        return pick


class FakeStrength(object):
    @staticmethod
    def plan(position, allowed, budget):
        return budget, None


class FakeAI(object):
    @staticmethod
    def get_legal_actions(st, side):
        raise AssertionError('the caller gave the menu')


def test_choose():
    import types
    acts = [FakeAction('a'), FakeAction('b'), FakeAction('c')]
    s = searcher()
    fake = FakeSearch(acts)
    s.searches = {'side': fake}
    s.seats = None
    s.turn_search, s.ai_units, s.strength = FakeTurnSearch, FakeUnits, FakeStrength
    s.ts, s.position = False, 1
    saved = P.ai
    P.ai = FakeAI()
    P.ai._TREE_CACHE = {}
    mod = types.ModuleType('ai_policies')
    mod.project_ai_actions = lambda st, side, legal: types.SimpleNamespace(actions=list(legal))
    mod.action_policy_key = lambda a: a.action_type
    sys.modules['ai_policies'] = mod
    try:
        for clock in (2.0, 1.0):
            arrived = time.monotonic()
            s.deadline = arrived + clock - 0.2
            pick, info = s.choose(object(), 'side', 'ACTION', set(), legal=acts)
            s.settle()
            took = time.monotonic() - arrived
            sims, deadline = fake.calls[-1]
            check(pick is acts[0] and deadline is not None and deadline < s.deadline,
                  '%.0f s: the search got a deadline before the answer\'s' % clock)
            check(took < clock - 0.2, '%.0f s: answered in %.3f s (by %.1f s)' % (clock, took, clock - 0.2))
            check(info.get('run') is not None and info.get('stopped') is not None, 'the info has run and stopped')
            check(FakeTurnSearch.seen[-1] == s.deadline - s.AFTER and FakeTurnSearch.certain_line.DEADLINE is None,
                  'the certain line had the deadline less AFTER, reset after')
        s.deadline = None
        s.max_seconds = None
        s.choose(object(), 'side', 'ACTION', set(), legal=acts, left_ms=None)
        check(fake.calls[-1][1] is None and fake.calls[-1][0] == 3000, 'no deadline: the whole budget, no deadline')
    finally:
        P.ai = saved
        sys.modules.pop('ai_policies', None)


if __name__ == '__main__':
    test_answer_deadline()
    test_plan()
    test_settle()
    test_choose()
    print('%d failed' % len(FAILS) if FAILS else 'all passed')
    sys.exit(1 if FAILS else 0)
