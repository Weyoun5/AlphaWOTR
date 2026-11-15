# -*- coding: utf-8 -*-
"""Differential fuzz: engine legal-action surface vs clean-room
rules_oracle (P3, rules-assurance program, session 35).

At EVERY decision point of self-play episodes (low-sim, high-temp for
state diversity), the engine's offered action list is checked by the
oracle:
  - over-generation: oracle says an offered action is ILLEGAL
  - under-generation: oracle says a rules-mandated option is MISSING
Each flag records (episode seed, decision idx, turn, side, action,
reason, whether the engine actually PLAYED it) and the episode trace
is kept on disk (GUI-viewable) so the user can watch the situation.

Usage:
  python src/oracle_fuzz.py --episodes 48 --workers 12 --sims 8
"""
import argparse
import glob as _glob
import json
import os
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONHASHSEED', '0')


def _state_summary(state):
    f = getattr(state, 'fellowship', None)
    pend = [k for k in dir(state) if k.startswith('pending')
            and getattr(state, k, None)]
    return {
        'turn': getattr(state, 'turn', None),
        'fp_hidden': getattr(f, 'hidden', None) if f else None,
        'progress': getattr(f, 'progress', None) if f else None,
        'corruption': getattr(f, 'corruption', None) if f else None,
        'mordor': getattr(f, 'mordor_entered', None) if f else None,
        'pending': pend[:6],
    }


def _fuzz_worker(wid, ep_seeds, sims, temp, fp_path, sh_path,
                 out_dir, flag_q):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['WOTR_NET_ARCH'] = 'base'
    sys.path.insert(0, SRC)
    import random

    import numpy as np
    import torch
    torch.set_num_threads(1)

    import ai
    import game_state as gs
    import rules_oracle as oracle
    ai.DEVICE = torch.device('cpu')

    def _load(path, side):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            raise FileNotFoundError('fuzz net missing: %s' % full)
        net = ai.load_value_net(full, side=side)
        if net is None:
            raise RuntimeError('fuzz net failed to load: %s' % full)
        net.eval()
        return net
    fpn = _load(fp_path, 'fp')
    shn = _load(sh_path, 'sh')

    orig = ai.mcts_action_with_policy
    ctr = {'True': 0, 'False': 0, 'None': 0, 'dec': 0,
           'oracle_err': 0, 'undergen': 0}
    cur = {'seed': None, 'dec': 0, 'flags': 0}

    def wrapped(state, side, fp_net, sh_net, *a, **k):
        flags_here = []
        try:
            acts = ai.get_legal_actions(state, side)
        except Exception as e:
            # queued IMMEDIATELY (the search call below will likely
            # re-raise the same error and skip the flags_here flush)
            # and counted so the episode trace is KEPT on disk
            flag_q.put(('flag', {
                'kind': 'engine_error', 'seed': cur['seed'],
                'dec': cur['dec'], 'side': str(side),
                'reason': repr(e)[:160],
                'state': _state_summary(state)}))
            cur['flags'] += 1
            acts = None
        if acts is not None:
            # NOTE: an EMPTY menu still runs required_offers — the
            # most extreme under-generation (rules mandate an option,
            # engine offers nothing) must be checked, not skipped
            for act in acts:
                try:
                    v, reason = oracle.check_action(state, side, act)
                except Exception as e:
                    v, reason = None, 'ORACLE_ERR ' + repr(e)[:100]
                    ctr['oracle_err'] += 1
                ctr[str(v)] = ctr.get(str(v), 0) + 1
                if v is None and str(reason).startswith(
                        'oracle-internal'):
                    # check_action never raises — internal crashes
                    # come back as this reason string and must not
                    # hide in the 'unchecked' bucket
                    ctr['oracle_err'] += 1
                    flags_here.append({
                        'kind': 'oracle_error',
                        'action': repr(act)[:200],
                        'action_type': act.action_type,
                        'reason': str(reason)[:200]})
                if v is False:
                    flags_here.append({
                        'kind': 'overgen', 'action': repr(act)[:200],
                        'action_type': act.action_type,
                        'die_idx': act.die_idx, 'reason': reason})
            try:
                missing = oracle.required_offers(state, side, acts)
            except Exception as e:
                missing = []
                ctr['oracle_err'] += 1
                flags_here.append({
                    'kind': 'oracle_error', 'action': None,
                    'action_type': None,
                    'reason': 'required_offers %s: %s'
                              % (type(e).__name__, str(e)[:160])})
            for msg in missing:
                ctr['undergen'] += 1
                flags_here.append({'kind': 'undergen',
                                   'action': None,
                                   'action_type': None,
                                   'reason': str(msg)[:200]})
        # Piece-conservation tripwire (USER 2026-07-20, P4-class
        # invariant): deployed + tracked FP casualties may never
        # exceed a nation's total figures.  One-sided by design:
        # catches DUPLICATION-side leaks (over-mustered / re-mustered
        # figures wrongly kept as casualties); vanished figures have
        # no lower-bound check here.  Checked at EVERY decision.
        try:
            dep = ai._count_deployed_all(state)
            fcas = getattr(state, 'fp_casualties', None) or {}
            for nat, maxes in ai.NATION_MAX_FIGURES.items():
                d = dep.get(nat, {})
                nc = (fcas.get(nat, {})
                      if ai._side_of_nation(nat) == gs.Side.FREE_PEOPLES
                      else {})
                for typ, mx in maxes.items():
                    tot = d.get(typ, 0) + nc.get(typ, 0)
                    if tot > mx:
                        ctr['conserve'] = ctr.get('conserve', 0) + 1
                        flags_here.append({
                            'kind': 'conservation', 'action': None,
                            'action_type': None,
                            'reason': '%s %s: deployed %d + cas %d > max %d'
                                      % (nat.value, typ, d.get(typ, 0),
                                         nc.get(typ, 0), mx)})
        except Exception as _oe9:
            # review P2 (2026-07-27): keep the fuzz alive but never
            # count an oracle failure without saying WHAT failed.
            # Routed through flags_here so it gets seed/dec/side/state
            # enrichment AND keeps the trace on disk.
            ctr['oracle_err'] += 1
            flags_here.append({
                'kind': 'oracle_error', 'action': None,
                'action_type': None,
                'reason': 'conservation %s: %s'
                          % (type(_oe9).__name__, str(_oe9)[:160])})
        res = orig(state, side, fp_net, sh_net, *a, **k)
        chosen = res[0] if isinstance(res, tuple) else res
        if flags_here:
            summ = _state_summary(state)
            for f in flags_here:
                f.update({'seed': cur['seed'], 'dec': cur['dec'],
                          'side': str(side),
                          'played': (f['action'] is not None
                                     and chosen is not None
                                     and repr(chosen)[:200] == f['action']),
                          'state': summ})
                flag_q.put(('flag', f))
            cur['flags'] += len(flags_here)
        cur['dec'] += 1
        ctr['dec'] += 1
        return res
    ai.mcts_action_with_policy = wrapped

    tr_dir = os.path.join(out_dir, 'traces')
    for seed in ep_seeds:
        random.seed(seed)
        np.random.seed(seed)
        ai.PLAYER_PROFILES['fp'] = random.choice(ai.FP_PROFILES)
        ai.PLAYER_PROFILES['sh'] = random.choice(ai.SH_PROFILES)
        cur['seed'], cur['dec'], cur['flags'] = seed, 0, 0
        tp = os.path.join(tr_dir, 'fuzz_s%05d.trace.gz' % seed)
        cfg = {'max_turns': 40, 'mcts_sims': sims, 'shadow_vp': 10,
               'temperature_override': temp, 'trace_path': tp}
        t0 = time.time()
        err = None
        try:
            ai._run_single_episode(fpn, shn, cfg)
        except Exception as e:
            err = repr(e)[:200]
        if cur['flags'] == 0 and err is None:
            # '.' anchor after the seed: a bare '<stem>*' prefix glob
            # crosses seeds at decimal-width boundaries (s10000*
            # matches s100000)
            for g in ([tp] + _glob.glob(
                    tp.replace('.trace.gz', '.') + '*.trace.gz')):
                try:
                    os.remove(g)
                except OSError:
                    pass
        flag_q.put(('ep', {'seed': seed, 'wid': wid,
                           'decisions': cur['dec'],
                           'flags': cur['flags'], 'err': err,
                           'secs': round(time.time() - t0, 1)}))
    flag_q.put(('done', {'wid': wid, 'ctr': ctr}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=48)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--sims', type=int, default=8)
    ap.add_argument('--temp', type=float, default=1.5)
    ap.add_argument('--seed0', type=int, default=9000)
    ap.add_argument('--fp', default='wotr_fp_corpus_v37td.pt')
    ap.add_argument('--sh', default='wotr_sh_corpus_v37td.pt')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data',
                                                  'oracle_fuzz_out'))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'traces'), exist_ok=True)

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    flag_q = ctx.Queue()
    seeds = [args.seed0 + i for i in range(args.episodes)]
    chunks = [seeds[w::args.workers] for w in range(args.workers)]
    procs = []
    for w in range(args.workers):
        if not chunks[w]:
            continue
        p = ctx.Process(target=_fuzz_worker,
                        args=(w, chunks[w], args.sims, args.temp,
                              args.fp, args.sh, args.out, flag_q))
        p.start()
        procs.append(p)

    flags = []
    eps = []
    ctrs = []
    n_done = 0
    fpath = os.path.join(args.out, 'flags.jsonl')
    with open(fpath, 'w', encoding='utf-8') as fh:
        while n_done < len(procs):
            try:
                kind, rec = flag_q.get(timeout=60)
            except Exception:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            if kind == 'flag':
                flags.append(rec)
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
            elif kind == 'ep':
                eps.append(rec)
                if len(eps) % 8 == 0:
                    print('  [%d/%d eps] flags so far: %d'
                          % (len(eps), args.episodes, len(flags)),
                          flush=True)
            elif kind == 'done':
                ctrs.append(rec['ctr'])
                n_done += 1
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    tot = {}
    for c in ctrs:
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
    print('=== ORACLE FUZZ: %d episodes, %d decisions ===' %
          (len(eps), tot.get('dec', 0)))
    print('  verdicts: legal %d  ILLEGAL %d  unchecked %d  '
          'oracle_errors %d  undergen %d  conserve %d' %
          (tot.get('True', 0), tot.get('False', 0),
           tot.get('None', 0), tot.get('oracle_err', 0),
           tot.get('undergen', 0), tot.get('conserve', 0)))
    if n_done < len(procs) or (not eps and args.episodes > 0):
        print('=== ORACLE FUZZ FAILED: %d/%d workers reported, '
              '%d episodes ===' % (n_done, len(procs), len(eps)))
        raise SystemExit(1)
    ep_err = [e for e in eps if e.get('err')]
    print('  episode errors: %d' % len(ep_err))
    for e in ep_err[:5]:
        print('    seed %s: %s' % (e['seed'], e['err']))
    by = {}
    for f in flags:
        key = (f['kind'], f.get('action_type'), f['reason'][:80])
        by[key] = by.get(key, 0) + 1
    print('  --- flag groups (count | kind | action_type | reason) ---')
    for key, n in sorted(by.items(), key=lambda kv: -kv[1])[:40]:
        print('  %4d | %s | %s | %s' % (n, key[0], key[1], key[2]))
    played = sum(1 for f in flags if f.get('played'))
    print('  flags actually PLAYED by engine: %d / %d' %
          (played, len(flags)))
    print('  flags file: %s' % fpath)
    print('  traces kept: %d' %
          len(_glob.glob(os.path.join(args.out, 'traces',
                                      '*.trace.gz'))))


if __name__ == '__main__':
    main()
