# -*- coding: utf-8 -*-
"""Throughput benchmark: local-CPU inference vs batched-GPU inference
server, base vs wide nets (SCALE, USER 2026-07-19).

Modes:
  local        12 workers, per-worker CPU nets (production baseline)
  server-base  12 workers -> infer_server on CUDA, base-arch nets
  server-wide  12 workers -> infer_server on CUDA, wide-arch nets

Each mode runs a fixed wall-clock window of continuous self-play
episodes (production config: 1000 sims, max_turns 40, profiles).
Metrics: MCTS decisions/sec (aggregate), mean/median search latency,
completed games, decisions/game, eps/hr extrapolation.

Usage:
  python src/bench_scale.py --parity              # correctness gate
  python src/bench_scale.py --mode local --minutes 10
  python src/bench_scale.py --mode server-base --minutes 10
  python src/bench_scale.py --mode server-wide --minutes 10 \
      --fp wotr_fp_corpus_v31jt.pt --sh wotr_sh_corpus_v31jt.pt
"""
import argparse
import os
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np


def _bench_worker(wid, mode, conn, arch, sims, fp_path, sh_path,
                  stop_evt, stats_q):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['WOTR_NET_ARCH'] = 'base' if mode == 'local' else arch
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    sys.path.insert(0, SRC)
    import random

    import torch
    torch.set_num_threads(1)

    import ai
    ai.DEVICE = torch.device('cpu')

    try:
        if mode == 'local':
            # canonical loader (head_act reconstruction + generation
            # refusal): a hand-rolled load benchmarked a Tanh head
            # against scaled2 checkpoints
            fpn = _canon_load(fp_path, 'fp', ai)
            shn = _canon_load(sh_path, 'sh', ai)
        else:
            import infer_server
            handles = infer_server.attach_client(conn, ['fp', 'sh'], require_metadata=True)
            fpn, shn = handles['fp'], handles['sh']
    except Exception as e:
        stats_q.put(('e', wid, time.time(),
                     'worker init failed: %s' % repr(e)[:120]))
        return

    orig = ai.mcts_action_with_policy
    counter = {'n': 0}

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        counter['n'] += 1
        stats_q.put(('d', wid, time.time(),
                     time.perf_counter() - t0))
        return r
    ai.mcts_action_with_policy = timed

    random.seed(1000 + wid)
    np.random.seed(1000 + wid)
    while not stop_evt.is_set():
        ai.PLAYER_PROFILES['fp'] = random.choice(ai.FP_PROFILES)
        ai.PLAYER_PROFILES['sh'] = random.choice(ai.SH_PROFILES)
        n0 = counter['n']
        t0 = time.time()
        try:
            traj = ai._run_single_episode(
                fpn, shn, {'max_turns': 40, 'mcts_sims': sims,
                           'shadow_vp': 10})
            gs9 = traj.get('game_stats') or {}
            stats_q.put(('g', wid, time.time(),
                         gs9.get('turns'), gs9.get('winner'),
                         counter['n'] - n0, time.time() - t0))
        except Exception as e:
            stats_q.put(('e', wid, time.time(), repr(e)[:120]))


def _canon_load(path, side, ai):
    net = ai.load_value_net(os.path.join(ROOT, path), side=side)
    if net is None:
        raise RuntimeError('net failed to load: %s' % path)
    net.eval()
    return net


def run_mode(mode, arch, fp_path, sh_path, workers, minutes, sims):
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    stats_q = ctx.Queue()
    stop_evt = ctx.Event()
    server = None
    pairs = []
    if mode != 'local':
        pairs = [ctx.Pipe() for _ in range(workers)]
        import infer_server  # noqa: F401 (path check)
        net_paths = {'fp': os.path.join(ROOT, fp_path),
                     'sh': os.path.join(ROOT, sh_path)}
        server = ctx.Process(
            target=_server_entry,
            args=([p[0] for p in pairs], net_paths, arch))
        server.start()
        time.sleep(2)
    procs = []
    for w in range(workers):
        conn = pairs[w][1] if pairs else None
        p = ctx.Process(target=_bench_worker,
                        args=(w, mode, conn, arch, sims, fp_path,
                              sh_path, stop_evt, stats_q))
        p.start()
        procs.append(p)

    t_end = time.time() + minutes * 60
    decisions = []
    games = []
    errors = 0
    err_msgs = []
    import queue as _qmod
    warmup_until = time.time() + 90   # exclude model-load / first-turn warmup
    while time.time() < t_end:
        try:
            m = stats_q.get(timeout=2.0)
        except _qmod.Empty:
            if not any(p.is_alive() for p in procs):
                print('  ALL WORKERS DEAD — aborting bench window',
                      flush=True)
                break
            continue
        if m[0] == 'd' and m[2] >= warmup_until:
            decisions.append(m[3])
        elif m[0] == 'g':
            games.append(m)
        elif m[0] == 'e':
            errors += 1
            if len(err_msgs) < 5:
                err_msgs.append(str(m[3]))
    stop_evt.set()
    time.sleep(1)
    for p in procs:
        p.terminate()
    if server is not None:
        server.terminate()

    window = minutes * 60 - 90
    dps = len(decisions) / max(1.0, window)
    lat = np.array(decisions) if decisions else np.array([0.0])
    dpg = (np.mean([g[5] for g in games]) if games else float('nan'))
    eph = (dps * 3600.0 / dpg) if games and dpg > 0 else float('nan')
    print('=== %s (%s): %d workers, sims=%d, window %.0f min' %
          (mode, arch if mode != 'local' else 'base', workers, sims,
           minutes))
    print('  decisions: %d  (%.1f/sec aggregate)' % (len(decisions),
                                                     dps))
    print('  search latency: mean %.2fs  p50 %.2fs  p90 %.2fs' %
          (lat.mean(), np.percentile(lat, 50), np.percentile(lat, 90)))
    print('  games completed: %d  decisions/game %.0f  errors %d' %
          (len(games), dpg, errors))
    for _em in err_msgs:
        print('    error: %s' % _em)
    print('  => extrapolated eps/hr: %.1f' % eph)
    return {'mode': mode, 'dps': dps, 'lat_mean': float(lat.mean()),
            'games': len(games), 'dpg': float(dpg), 'eph': float(eph)}


def _server_entry(conns, net_paths, arch):
    # Preserve the GPU assignment inherited from the scheduler.
    sys.path.insert(0, SRC)
    import infer_server
    infer_server.server_main(conns, net_paths, arch)


def parity_check(fp_path, sh_path):
    """Local vs server priors/value on fixed states must agree."""
    import multiprocessing as mp

    import torch

    import ai
    import board
    import game_logic
    import game_state as gs
    import infer_server
    ai.DEVICE = torch.device('cpu')
    b = board.load_board()
    st = gs.setup_initial_game(b)
    game_logic.initialize_decks(st)
    for _s in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        for _d in ('character', 'strategy'):
            game_logic.draw_event_card(st, _s, _d)
    game_logic.roll_action_dice(st, pool=gs.Side.FREE_PEOPLES)
    game_logic.roll_action_dice(st, pool=gs.Side.SHADOW)

    sh_local = _canon_load(sh_path, 'sh', ai)
    acts = ai.get_legal_actions(st, gs.Side.SHADOW)
    pri_l, v_l = ai._get_priors_and_value(st, gs.Side.SHADOW, sh_local,
                                          acts)

    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    control_parent, control_child = ctx.Pipe()
    server = ctx.Process(target=_server_entry,
                         args=([child, control_child],
                               {'fp': os.path.join(ROOT, fp_path),
                                'sh': os.path.join(ROOT, sh_path)},
                               os.environ.get('WOTR_NET_ARCH',
                                              'base')))
    previous_client = getattr(ai, 'INFER_CLIENT', None)
    from served_lifecycle import cleanup
    try:
        server.start()
        child.close()
        control_child.close()
        handles = infer_server.attach_client(parent, ['fp', 'sh'], require_metadata=True)
        pri_r, v_r = ai._get_priors_and_value(st, gs.Side.SHADOW, handles['sh'], acts)
    finally:
        ai.INFER_CLIENT = previous_client
        cleanup([], server, control_parent)
        for connection in (parent, child, control_parent, control_child):
            connection.close()
    dp = float(np.max(np.abs(np.asarray(pri_l) - np.asarray(pri_r))))
    dv = abs(v_l - v_r)
    print('parity: max |prior diff| = %.2e   |value diff| = %.2e'
          % (dp, dv))
    ok = dp < 1e-3 and dv < 1e-3
    print('PARITY ' + ('OK' if ok else 'FAILED'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['local', 'server-base',
                                       'server-wide'])
    ap.add_argument('--parity', action='store_true')
    ap.add_argument('--minutes', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--fp', default='wotr_fp_corpus_v37td.pt')
    ap.add_argument('--sh', default='wotr_sh_corpus_v37td.pt')
    ap.add_argument('--arch', default=None,
                    help='server net arch (overrides mode default)')
    args = ap.parse_args()
    if args.parity:
        ok = parity_check(args.fp, args.sh)
        sys.exit(0 if ok else 1)
    if args.mode is None:
        ap.error('--mode is required unless --parity')
    arch = args.arch or ('wide' if args.mode == 'server-wide'
                         else 'base')
    run_mode(args.mode, arch, args.fp, args.sh, args.workers,
             args.minutes, args.sims)


if __name__ == '__main__':
    main()
