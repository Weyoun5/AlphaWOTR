# -*- coding: utf-8 -*-
"""Parallel cross-architecture arena via dual inference servers
(SCALE, session 34).  Nets of DIFFERENT WOTR_NET_ARCHs cannot coexist
in one process — each side is served by its own infer_server; workers
hold RemoteNet handles with per-net clients.

Default: 96 games (8/worker x 12), half A-as-FP half B-as-FP, 60-sim
searches, temperature 0.1, no root noise, profiles off.

Usage:
  python src/arena_scale.py --a-arch wide --a-fp wotr_fp_corpus_v31.pt
      --a-sh wotr_sh_corpus_v31.pt --b-arch xformer
      --b-fp wotr_fp_corpus_v31x.pt --b-sh wotr_sh_corpus_v31x.pt
"""
import argparse
import json
import os
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np


def _server_entry(conns, net_paths, arch):
    # Preserve the GPU assignment inherited from the scheduler.
    sys.path.insert(0, SRC)
    import infer_server
    infer_server.server_main(conns, net_paths, arch)


def _episode_randomness(run_seed, worker_id, local_index, worker_stride):
    """Stable global arena episodes and A/B side assignment in isolated mode."""
    from rng_streams import current_contract, LEGACY, served_episode_seed
    if current_contract() == LEGACY:
        return {}, (worker_id + local_index) % 2 == 0
    index = worker_id + local_index * worker_stride
    seed = served_episode_seed(run_seed, index)
    return {'rng_episode_seed': seed,
            'rng_episode_identity': {'run_seed': run_seed, 'episode_index': index}}, index % 2 == 0


def _arena_worker(wid, conn_a, conn_b, n_games, sims, out_q, seed0=4000, episode_stride=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    sys.path.insert(0, SRC)
    import random

    import torch
    torch.set_num_threads(1)

    import ai
    import infer_server
    ai.DEVICE = torch.device('cpu')

    def _attach(conn):
        # Delta-blindness fix (USER 2026-08-02): base-arch sides pass
        # {'local': {fp,sh paths}} instead of a server pipe.  Local
        # nets run the FULL local pv path including the 1.226a
        # policy_delta head (the remote pv is base-logits-only), so
        # the gate measures exactly what self-play deploys — and base
        # nets are faster locally anyway (IPC > compute).
        if isinstance(conn, dict) and 'local' in conn:
            nets = {'fp': ai.load_value_net(conn['local']['fp'],
                                            side='fp'),
                    'sh': ai.load_value_net(conn['local']['sh'],
                                            side='sh')}
            # a missing checkpoint loads as None and would silently
            # play heuristic-leaf MCTS under a plausible Elo — refuse
            _miss9 = [k for k, v in nets.items() if v is None]
            if _miss9:
                raise FileNotFoundError(
                    'arena local net missing/unloadable: %s (%r)'
                    % (_miss9, conn['local']))
            return nets
        return infer_server.attach_client(conn, ['fp', 'sh'],
                                          set_global=False, require_metadata=True)

    a = _attach(conn_a)
    b = _attach(conn_b)
    # 1.224c follow-through (USER 2026-07-30 leak hunt): the search
    # terminal + leaf shaping are now PROFILE-AWARE, so the gate must
    # measure each net in its DEPLOYED mission regime — a blind arena
    # would score fp_ring with a military-chasing search it never
    # trains under.  Both A and B games get the same pins (symmetric).
    ai.PLAYER_PROFILES['fp'] = \
        os.environ.get('WOTR_ARENA_FP_PROFILE') or None
    ai.PLAYER_PROFILES['sh'] = \
        os.environ.get('WOTR_ARENA_SH_PROFILE') or None
    random.seed(seed0 + wid)
    np.random.seed(seed0 + wid)
    cfg = dict(max_turns=40, mcts_sims=sims, shadow_vp=10,
               temperature_override=0.1, add_noise=False)
    for g in range(n_games):
        rng_config, a_is_fp = _episode_randomness(seed0, wid, g, episode_stride)
        fp = a['fp'] if a_is_fp else b['fp']
        sh = b['sh'] if a_is_fp else a['sh']
        t0 = time.time()
        try:
            traj = ai._run_single_episode(fp, sh, {**cfg, **rng_config})
            w = traj.get('winner')
            gs = dict(traj.get('game_stats') or {})
            gs['wall_s'] = round(time.time() - t0, 1)
            gs['a_is_fp'] = bool(a_is_fp)
            out_q.put(('g', wid, a_is_fp, w, gs.get('turns'),
                       time.time() - t0, gs))
        except Exception as e:
            if rng_config:
                from rng_streams import ISOLATED
                detail = {'rng_contract': ISOLATED, **rng_config,
                          'profiles': dict(ai.PLAYER_PROFILES), 'a_is_fp': a_is_fp}
                out_q.put(('e', wid, a_is_fp, repr(e)[:120], None, None, detail))
            else:
                out_q.put(('e', wid, a_is_fp, repr(e)[:120], None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a-arch', required=True)
    ap.add_argument('--a-fp', required=True)
    ap.add_argument('--a-sh', required=True)
    ap.add_argument('--b-arch', required=True)
    ap.add_argument('--b-fp', required=True)
    ap.add_argument('--b-sh', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--games-per-worker', type=int, default=8)
    ap.add_argument('--sims', type=int, default=60)
    ap.add_argument('--seed0', type=int, default=4000)
    ap.add_argument('--games-jsonl', default=None,
                    help='append one game_stats record per arena game '
                         '(GUI feed)')
    args = ap.parse_args()
    if args.seed0 < 0 or args.seed0 + args.workers - 1 >= 2 ** 32:
        ap.error('seed0 plus worker IDs must fit unsigned 32-bit seeds')
    if args.workers < 1 or args.games_per_worker < 1 or args.sims < 1:
        ap.error('workers, games-per-worker, and sims must be positive')

    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    a_paths = {'fp': os.path.join(ROOT, args.a_fp),
               'sh': os.path.join(ROOT, args.a_sh)}
    b_paths = {'fp': os.path.join(ROOT, args.b_fp),
               'sh': os.path.join(ROOT, args.b_sh)}
    # base arch -> LOCAL nets in workers (delta-aware, no server);
    # big arches (wide/xformer) keep the batched-GPU server path
    a_local = args.a_arch == 'base'
    b_local = args.b_arch == 'base'
    pairs_a = None if a_local else \
        [ctx.Pipe() for _ in range(args.workers)]
    pairs_b = None if b_local else \
        [ctx.Pipe() for _ in range(args.workers)]
    control_a = None if a_local else ctx.Pipe()
    control_b = None if b_local else ctx.Pipe()
    sa = sb = None
    procs = []
    out_q = ctx.Queue()
    from served_lifecycle import cleanup
    try:
        if not a_local:
            sa = ctx.Process(target=_server_entry, args=(
                [p[0] for p in pairs_a] + [control_a[0]], a_paths, args.a_arch))
            sa.start()
        if not b_local:
            sb = ctx.Process(target=_server_entry, args=(
                [p[0] for p in pairs_b] + [control_b[0]], b_paths, args.b_arch))
            sb.start()
        if sa is not None or sb is not None:
            time.sleep(4)
            # a server that died at startup leaves every worker blocked in
            # recv() for the whole wall clock — fail loudly instead
            if (sa is not None and not sa.is_alive()) or \
                    (sb is not None and not sb.is_alive()):
                raise SystemExit('arena_scale: infer server died at '
                                 'startup (check net paths/arch)')

        for w in range(args.workers):
            conn_a = {'local': a_paths} if a_local else pairs_a[w][1]
            conn_b = {'local': b_paths} if b_local else pairs_b[w][1]
            p = ctx.Process(target=_arena_worker,
                            args=(w, conn_a, conn_b,
                                  args.games_per_worker, args.sims, out_q, args.seed0, args.workers))
            p.start()
            procs.append(p)

        total = args.workers * args.games_per_worker
        res = {'a_fp_w': 0, 'a_sh_w': 0, 'b_fp_w': 0, 'b_sh_w': 0,
               'timeout': 0, 'err': 0}
        a_score = 0.0
        n = 0
        t0 = time.time()
        while n < total and time.time() - t0 < 3600 * 3:
            try:
                m = out_q.get(timeout=1)
            except Exception:
                if not any(p.is_alive() for p in procs):
                    break
                continue
            n += 1
            if m[0] == 'e':
                res['err'] += 1
                # surface the first few error payloads (never just
                # 'errors N' with no cause)
                if res['err'] <= 5:
                    print('  [err w%s] %s' % (m[1], m[3]), flush=True)
                continue
            _, wid, a_is_fp, w, turns, secs = m[:6]
            if args.games_jsonl and len(m) > 6 and m[6]:
                try:
                    with open(args.games_jsonl, 'a',
                              encoding='utf-8') as jf:
                        jf.write(json.dumps(m[6]) + '\n')
                except OSError:
                    pass
            if w == 'timeout':
                res['timeout'] += 1
                a_score += 0.5
            elif w == 'fp':
                a_score += 1.0 if a_is_fp else 0.0
                res['a_fp_w' if a_is_fp else 'b_fp_w'] += 1
            elif w == 'sh':
                a_score += 0.0 if a_is_fp else 1.0
                res['b_sh_w' if a_is_fp else 'a_sh_w'] += 1
            if n % 8 == 0:
                print('  [%d/%d] A-score so far %.2f' % (n, total,
                                                         a_score / n),
                      flush=True)

    finally:
        cleanup_a = cleanup(list(enumerate(procs)), sa, control_a[1] if control_a else None)
        cleanup_b = cleanup([], sb, control_b[1] if control_b else None)
        for pair in [*(pairs_a or []), *(pairs_b or []),
                     *([control_a] if control_a else []), *([control_b] if control_b else [])]:
            for connection in pair:
                connection.close()
        out_q.close()
        out_q.join_thread()

    played = n - res['err']
    worker_errors = {wid: code for wid, code in cleanup_a['worker_exitcodes'].items()
                     if code != 0}
    cleanup_failed = bool(cleanup_a['alive'] or cleanup_b['alive'] or
                          cleanup_a['errors'] or cleanup_b['errors'] or
                          any(code not in (None, 0) for code in (
                              cleanup_a['server_exitcode'], cleanup_b['server_exitcode'])))
    if played <= 0 or n != total or res['err'] or worker_errors or cleanup_failed:
        print('=== ARENA INCOMPLETE: %d/%d games completed; %d reported errors; %d unreported ==='
              % (played, total, res['err'], total - n), flush=True)
        raise SystemExit('arena_scale: evaluation incomplete; no score or Elo certified')
    s = a_score / max(1, played)
    elo = (400.0 * np.log10(s / (1.0 - s))
           if 0.0 < s < 1.0 else float('inf') * (1 if s >= 1 else -1))
    print('=== ARENA %s(A) vs %s(B): %d games ===' %
          (args.a_arch, args.b_arch, played))
    print('  A as FP: %d W   A as SH: %d W   B as FP: %d W   '
          'B as SH: %d W   timeouts %d   errors %d' %
          (res['a_fp_w'], res['a_sh_w'], res['b_fp_w'], res['b_sh_w'],
           res['timeout'], res['err']))
    print('  A score = %.3f   Elo(A-B) = %+.0f' % (s, elo))


if __name__ == '__main__':
    main()
