"""Find and reproduce the decisions their native search refuses (harvested states, as the adapter handed them).

    venv/bin/python -I wasm/tools/repro_refusals.py --alpha ALPHAWOTR_CHECKOUT --positions FILE.pkl
        [--only N,N,...] [--out FILE.pkl] [--sims-from-record]

Each decision is searched as record_searches.py searches it (OwnModelSearch on the native backend, the adapter's
simulations, seed 20261114); a decision the search refuses is printed with the exception, its pending decision
(pending_combat's phase and effect_choice, pending_card_option, ...) and the same decision searched by their Python
backend for comparison. --out keeps the refused decisions (record_searches.py --positions shape) for the fix's test.
"""
import argparse
import os
import pickle
import sys
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', required=True)
    ap.add_argument('--positions', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--no-python', action='store_true')
    a = ap.parse_args()
    alpha = os.path.abspath(a.alpha)
    positions = os.path.abspath(a.positions)
    out = os.path.abspath(a.out) if a.out else ''
    sys.path[:0] = [os.path.join(alpha, 'src'), os.path.join(alpha, 'wotr_bridge')]
    import launch_head as lh
    for key, val in lh.HEAD_ENV:
        os.environ[key] = val
    os.chdir(os.path.join(alpha, 'src'))
    import ai
    from core import own_model
    sides = {'FREE_PEOPLES': ai.Side.FREE_PEOPLES, 'SHADOW': ai.Side.SHADOW}
    nets = {s: own_model.load_own_net(lh.NET_TAG, s) for s in sides.values()}
    native = {s: own_model.OwnModelSearch(s, nets[s], seed=20261114, backend='native') for s in sides.values()}
    python = {s: own_model.OwnModelSearch(s, nets[s], seed=20261114, backend='python') for s in sides.values()}
    with open(positions, 'rb') as f:
        rows = pickle.load(f)
    only = {int(x) for x in a.only.split(',') if x}
    refused = []
    for n, r in enumerate(rows):
        if only and n not in only:
            continue
        side = sides[r['side']]
        sims = int(r.get('sims') or 300)
        try:
            act, _ = native[side].choose(pickle.loads(r['handed']), side, num_simulations=sims)
            if only:
                print('%d: searched, %s' % (n, act.action_type), flush=True)
            continue
        except Exception as error:
            st = pickle.loads(r['handed'])
            pc = getattr(st, 'pending_combat', None) or {}
            print('%d (%s, turn %s, %s): REFUSED %s: %s' % (n, r['side'], r.get('turn'), r.get('source'),
                                                         type(error).__name__, str(error)[:300]), flush=True)
            print('   pending_combat phase=%r effect_choice=%r' % (pc.get('phase'), pc.get('effect_choice')))
            print('   attacker %r -> defender %r, cards %r / %r, rounds %r' % (
                pc.get('attacker_region'), pc.get('defender_region'), pc.get('atk_combat_card'),
                pc.get('def_combat_card'), pc.get('rounds_fought')))
            print('   pending_card_option=%r' % (getattr(st, 'pending_card_option', None),))
            refused.append(r)
            if not a.no_python:
                try:
                    act, _ = python[side].choose(pickle.loads(r['handed']), side, num_simulations=sims)
                    print('   python backend: searched, picks %s %r' % (act.action_type, act.params))
                except Exception as e2:
                    print('   python backend: refused too: %s: %s' % (type(e2).__name__, str(e2)[:200]))
            tb = traceback.format_exception(type(error), error, error.__traceback__)
            print('   ' + ''.join(tb[-3:]).replace('\n', '\n   ')[:800])
    if out:
        with open(out, 'wb') as f:
            pickle.dump(refused, f)
    print('%d refused' % len(refused))


if __name__ == '__main__':
    main()
