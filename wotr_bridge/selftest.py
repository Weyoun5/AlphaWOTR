"""Does this PC's build of the game think as it should?  Both searches on saved positions; no client, no game.

    python wotr_bridge/selftest.py [--limit N] [--sims 60] [--no-turn-search] [--out FILE]

The positions (selftest_positions.pkl beside this file) are decisions of real AI-vs-AI games: the state the server
handed its search, the side, the legal menu.  For each one:
  1  THE TREE SEARCH, twice with the same seed: by the native library this PC loads, and by the Python engine.  The two
     must pick the same move - that is what "the library computes the engine's numbers on this platform" means (on a
     Mac the library is a clang build with the Mac's own math library; another platform's picks are no yardstick) -
     and the move must be one of the menu.
  2  THE TURN SEARCH (the cheapest setting), through the bridge's own module and the game's own file: it must answer,
     with a move of the menu.
--out writes every pick as JSON; selftest_reference.json beside this file holds the picks of the Windows build, and
how many of this run's picks equal them is printed for information (a last-digit difference of a math library can
flip a close choice: it is not a failure).
Exit 0 = every position passed.
"""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'src'), HERE]
import launch_head as lh            # noqa: E402

SEED = 20261114
TURN_SEARCH_SETTING = 'wide'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--positions', default=os.path.join(HERE, 'selftest_positions.pkl'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sims', type=int, default=60)
    ap.add_argument('--no-turn-search', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    for key, val in lh.HEAD_ENV:        # the launchers' environment, before the engine is imported
        os.environ[key] = val
    out_path = os.path.abspath(a.out) if a.out else ''
    with open(a.positions, 'rb') as f:
        rows = pickle.load(f)
    if a.limit:
        rows = rows[:a.limit]
    os.chdir(os.path.join(ROOT, 'src'))
    import ai
    from ai_policies import action_policy_key
    from core import own_model, turn_search
    sides = {'FREE_PEOPLES': ai.Side.FREE_PEOPLES, 'SHADOW': ai.Side.SHADOW}
    nets = {s: own_model.load_own_net(lh.NET_TAG, s) for s in sides.values()}
    searches = {backend: {s: own_model.OwnModelSearch(s, nets[s], seed=SEED, backend=backend) for s in sides.values()}
                for backend in ('native', 'python')}
    identity = searches['native'][ai.Side.FREE_PEOPLES].backend_identity
    print('native library: %s' % identity.get('controller_binary_sha256'), flush=True)

    def in_menu(act, menu):
        key = action_policy_key(act)
        return next((m for m in menu if m.action_type == act.action_type and m.die_idx == act.die_idx
                     and m.params == act.params), None) or next((m for m in menu if action_policy_key(m) == key), None)

    def shown(act):
        return [act.die_idx, act.action_type, json.dumps(act.params, sort_keys=True, default=str)]

    if not a.no_turn_search:
        turn_search.start(lh.NET_TAG)
    results, bad = [], 0
    for n, r in enumerate(rows):
        side, owner = sides[r['side']], sides.get(r['owner'])
        menu = [ai.Action(die_idx=d, action_type=k, params=p) for d, k, p in r['menu']]
        rec = dict(n=n, side=r['side'], turn=r['turn'], kind=r['kind'], menu=len(menu))
        picks = {}
        for backend in ('native', 'python'):
            state = pickle.loads(r['handed'])
            t0 = time.perf_counter()
            with ai.search_turn_owner(owner):
                act, _policy = searches[backend][side].choose(state, side, num_simulations=a.sims)
            picks[backend] = in_menu(act, menu)
            rec[backend] = shown(act)
            rec[backend + '_seconds'] = round(time.perf_counter() - t0, 1)
        rec['same'] = rec['native'] == rec['python']
        # the simulations knob at its lowest: the net's own choice, no search
        raw = searches['native'][side].raw(pickle.loads(r['handed']), side, menu)
        rec['raw'], rec['raw_in_menu'] = raw and shown(raw), raw is not None and in_menu(raw, menu) is not None
        rec['in_menu'] = picks['native'] is not None
        ok = rec['same'] and rec['in_menu'] and rec['raw_in_menu']
        if not a.no_turn_search and picks['native'] is not None:
            asked = turn_search._S['asked']
            t0 = time.perf_counter()
            got = turn_search.improve(ai, pickle.loads(r['handed']), side, owner, r['seed'], menu, picks['native'],
                                      TURN_SEARCH_SETTING)
            rec['turn_search'] = dict(answered=turn_search._S['asked'] > asked, in_menu=any(got is m for m in menu),
                                      pick=shown(got), seconds=round(time.perf_counter() - t0, 1),
                                      cannot_run=bool(turn_search._S.get('cannot_run')))
            # a PC with too little free memory for one worker runs no turn search, by design: said, not failed
            ok = ok and rec['turn_search']['in_menu'] and (rec['turn_search']['answered'] or rec['turn_search']['cannot_run'])
        rec['ok'] = ok
        bad += not ok
        results.append(rec)
        print('[%s] %d %s turn %s, %d choices: native %s (%.1f s) | python %s (%.1f s)%s' % (
            'PASS' if ok else 'FAIL', n, r['side'], r['turn'], len(menu), rec['native'][1], rec['native_seconds'],
            'the same' if rec['same'] else rec['python'][1], rec['python_seconds'],
            '' if 'turn_search' not in rec else ' | turn search %s (%.1f s)' % (
                'answered ' + rec['turn_search']['pick'][1] if rec['turn_search']['answered'] else
                'not run: too little free memory' if rec['turn_search']['cannot_run'] else 'DID NOT ANSWER',
                rec['turn_search']['seconds'])), flush=True)
    if not a.no_turn_search:
        turn_search.close()
    reference = os.path.join(HERE, 'selftest_reference.json')
    if os.path.isfile(reference):
        with open(reference, encoding='utf-8') as f:
            known = {x['n']: x for x in json.load(f)['results']}
        same = sum(1 for x in results if x['n'] in known and known[x['n']]['native'] == x['native'])
        print('tree-search picks equal to the Windows build\'s: %d of %d (for information)' % (same, len(results)))
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(dict(platform=sys.platform, sims=a.sims, library=identity.get('controller_binary_sha256'),
                           results=results), f, indent=1)
    print('SELFTEST: %s (%d positions, %d failed)' % ('ALL PASS' if not bad else 'FAILED', len(results), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
