"""TOURNAMENT-DEADLINE: does the native search stop at its deadline?  (No pytest; their engine and the native library
built for this machine: python native_backend/live_search_oct19/build_portable.py.)

    python -s -P protocol/test_native_deadline.py [--src SRC] [--decisions N] [--seed S] [--ms 250]

Plays a game of their engine with random picks (EngineSession) and, at each of the first N decisions with three or more
options, runs the seat's league search (OwnModelSearch, the native backend) with 3,000 simulations and a deadline
MS ahead.  Checked per search: it stopped (fewer simulations than its budget, at least one), the loop ended within
STOP_MS of the deadline (the last simulation after the stop), and the whole search returned (its tail included) and
answered an action of the menu; and once without a deadline, a small budget runs to its end as before.
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wotr_protocol_bot as P  # noqa: E402

STOP_MS = 40.0       # the loop's end after the deadline: one simulation (about 10-20 ms on one M4 core)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=P.DEFAULT_SRC)
    ap.add_argument('--decisions', type=int, default=12)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--ms', type=float, default=250.0)
    a = ap.parse_args()
    P.import_engine(a.src, True)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'wotr_bridge'))
    from core import own_model
    from engine_play import EngineSession
    searches = {side: own_model.OwnModelSearch(side, own_model.load_own_net('mc200_fpwin_shwin', side), seed=1)
                for side in (P.FP, P.SH)}
    rng = random.Random(a.seed)
    session = EngineSession(seed=a.seed, mcts_sims=10, max_turns=6)
    fails, done, late = [], 0, []
    while not session.game_over() and done < a.decisions:
        dp = session.current()
        if len(dp.actions) >= 3:
            search = searches[dp.side]
            t0 = time.monotonic()
            deadline = t0 + a.ms / 1000.0
            act, info = search.choose(session.state, dp.side, num_simulations=3000, deadline=deadline)
            t1 = time.monotonic()
            timing = (info or {}).get('search_timing') or {}
            run, budget = timing.get('simulations_run'), timing.get('budget')
            loop_end = t0 + (timing.get('setup') or 0) + (timing.get('loop') or 0)
            over = 1000.0 * (loop_end - deadline)
            late.append(over)
            ok = (info is not None and timing.get('stopped') and run is not None and 1 <= run < budget
                  and over <= STOP_MS and act is not None)
            print('%s T%s %s %s: %d options, %s of %s simulations, loop end %+.1f ms from the deadline, '
                  'returned %+.1f ms (tail %.0f ms)' % (
                      'ok  ' if ok else 'FAIL', dp.turn, dp.side.name, dp.kind, len(dp.actions), run, budget, over,
                      1000.0 * (t1 - deadline), 1000.0 * (timing.get('tail') or 0)))
            if not ok:
                fails.append(dp.kind)
            done += 1
        session.submit(rng.randrange(len(dp.actions)), dp.side)
    if late:
        print('loop end after the deadline: most %.1f ms, mean %.1f ms over %d searches' % (
            max(late), sum(late) / len(late), len(late)))
    # without a deadline: the budget runs out as before
    while not session.game_over():
        dp = session.current()
        if len(dp.actions) >= 3:
            act, info = searches[dp.side].choose(session.state, dp.side, num_simulations=20)
            timing = (info or {}).get('search_timing') or {}
            ok = not timing.get('stopped') and timing.get('simulations_run', 0) >= 1
            print('%s no deadline: %s of %s simulations' % (
                'ok  ' if ok else 'FAIL', timing.get('simulations_run'), timing.get('budget')))
            if not ok:
                fails.append('no deadline')
            break
        session.submit(rng.randrange(len(dp.actions)), dp.side)
    print('%d failed' % len(fails) if fails else 'all passed (%d searches)' % done)
    return 1 if fails or not done else 0


if __name__ == '__main__':
    sys.exit(main())
