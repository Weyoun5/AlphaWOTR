# -*- coding: utf-8 -*-
"""Local-inference self-play runner for the v33 training loop.

The CLI uses CPU workers. The worker API also accepts an explicit CUDA
device and verifies availability before loading models. Shared inference
uses a separate served operator; which configuration is faster depends
on the search workload and concurrency and must be measured.

Nets: the loop passes WOTR_SP_NET_DIR (per-profile live_*.pt roster);
ad-hoc runs must pass WOTR_SP_FP_NET / WOTR_SP_SH_NET explicitly.
There is NO filename fallback (WIRING 2026-08-11): the old
v31jt/v31td/v31 chain silently loaded ancient Tanh-era nets on this
box whenever the env was missing — a no-net run must refuse, not
play a superseded generation.

Standard config: 1000 MCTS sims, max_turns 40, Shadow VP 10, temps
0.6/0.3/0.15, tree reuse, root determinization, noise-free eye +
declare picks, random FP/SH profile per game, CROSS-TURN SEARCH
(WOTR_TURN_HORIZON=1; simulations cross one boundary through a chance
node sampling next-turn rolls).

Outputs:
  data/td_v33_traces/           per-game replayable traces
  data/td_v33_games.jsonl       td_dashboard feed
  data/td_v33_report.md         per-window markdown report

Usage:
  python run_td_v33sp.py                       # 500 games, 12 workers
  python run_td_v33sp.py --games N --workers W --sims S
  python run_td_v33sp.py --smoke               # 1 game / 1 worker
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import uuid

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('WOTR_TURN_HORIZON', '1')


def _net_path(tag):
    """Explicit env override or None — NO generation fallback.  The
    retired v31jt/v31td/v31 chain was a trap: those files still exist
    on disk, so an ad-hoc `python run_td_v33sp.py` silently played
    stale Tanh-generation nets instead of the loop roster.  main()
    refuses loudly when no net source is configured."""
    return os.environ.get('WOTR_SP_%s_NET' % tag.upper()) or None


FP_NET = _net_path('fp')
SH_NET = _net_path('sh')
GAMES_LOG = os.environ.get('WOTR_SP_GAMES_LOG') or os.path.join(ROOT, 'data', 'td_v33_games.jsonl')
REPORT = os.environ.get('WOTR_SP_REPORT') or os.path.join(ROOT, 'data', 'td_v33_report.md')
TRACE_DIR = os.environ.get('WOTR_SP_TRACE_DIR') or os.path.join(ROOT, 'data', 'td_v33_traces')
SMOKE_TRACE_DIR = os.environ.get('WOTR_SP_SMOKE_TRACE_DIR') or os.path.join(ROOT, 'data', 'td_v33_smoke_traces')
_CHECKPOINT_IDS = {}


def _report_paths():
    """Attempt/coverage outputs follow the caller's isolated output stem."""
    source = (os.environ.get('WOTR_SP_REPORT')
              or os.environ.get('WOTR_SP_GAMES_LOG') or REPORT)
    stem = os.path.splitext(source)[0]
    return (os.environ.get('WOTR_SP_ATTEMPTS_LOG') or stem + '.attempts.jsonl',
            os.environ.get('WOTR_SP_PHASE_SUMMARY') or stem + '.phase.json')


def _checkpoint_id(path):
    """Hash each checkpoint path once during this worker's load phase."""
    key = os.path.normcase(os.path.abspath(path))
    if key not in _CHECKPOINT_IDS:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        _CHECKPOINT_IDS[key] = digest.hexdigest()
    return _CHECKPOINT_IDS[key]


def _append_attempt(path, record):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')


def _message_record(message, phase_id, label):
    """Return (game attempt or None, worker failure or None).

    Legacy error tuples lack proof that an episode started. Keep their
    original diagnostic as a worker failure, without inventing a game.
    """
    if message[0] == 'g':
        record = dict(message[2])
        record.update(record_type='game_attempt', status='completed')
    elif (len(message) >= 4 and isinstance(message[3], dict)
          and message[3].get('record_type') == 'game_attempt'):
        record = dict(message[3])
        record.update(status='error')
    else:
        failure = dict(message[3]) if len(message) >= 4 and isinstance(message[3], dict) else {}
        failure.update(record_type='worker_error', worker_id=message[1],
                       error=failure.get('error') or str(message[2]),
                       phase_id=phase_id, batch=label)
        failure.setdefault('stage', 'unclassified_worker_error')
        return None, failure
    record.update(worker_id=message[1], phase_id=phase_id, batch=label)
    return record, None


def _write_phase_summary(path, phase_id, label, planned, attempts,
                         worker_failures, attempts_path, finished=False):
    from profile_metrics import summarize_profile_results
    observed = len(attempts)
    summary = {
        'phase_id': phase_id, 'batch': label, 'finished': finished,
        'attempts_log': os.path.abspath(attempts_path),
        'planned': planned, 'observed': observed,
        'unreported': max(0, planned - observed),
        'completed': sum(row.get('status') == 'completed' for row in attempts),
        'error_attempts': sum(row.get('status') == 'error' for row in attempts),
        'coverage_complete': observed == planned,
        'worker_failures': worker_failures,
        'profile_metrics': {side: summarize_profile_results(attempts, side=side)
                            for side in ('fp', 'sh')},
        'coverage_note': ('Rates use observed attempts only. Unreported planned games '
                          'have no invented result, profile, or checkpoint identity; '
                          'incomplete coverage prevents a complete per-attempt claim.'),
    }
    _publish_phase_summary(path, summary)
    return summary


def _publish_phase_summary(path, summary):
    """Publish one complete phase snapshot, tolerating brief Windows readers."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    # Windows readers can briefly hold the destination without delete sharing.
    # Keep the prior complete JSON visible until atomic replacement succeeds.
    # Permanent denial still raises, retaining the unpublished temporary file.
    for attempt in range(25):
        try:
            os.replace(path + '.tmp', path)
            break
        except PermissionError:
            if attempt == 24:
                raise
            time.sleep(.02)


def _fresh_run_weight_guard():
    """Session-34 TD-chimera fresh-run guard, CONDITION RESTORED
    (SPTR-16/SPEP-13): the recorded design is '0 TD episodes on
    record -> delete _parallel_weights/*_weights_v*.pt; crash-resume
    with episodes present keeps them'.  The v33sp port had dropped
    the condition - an unconditional wildcard delete on EVERY launch
    (the standing no-destructive-patterns class)."""
    from reward_contract import uses_predecision
    if uses_predecision():
        # An isolated new experiment does not own the legacy league files.
        return
    try:
        if os.path.getsize(GAMES_LOG) > 0:
            # TD episodes on record: a crash-resume keeps checkpoints
            return
    except OSError:
        pass  # no games log = fresh run
    pw = os.path.join(ROOT, '_parallel_weights')
    if os.path.isdir(pw):
        stale = glob.glob(os.path.join(pw, '*_weights_v*.pt'))
        for p in stale:
            try:
                os.remove(p)
            except OSError:
                pass
        if stale:
            print('fresh-run guard: cleared %d stale checkpoint(s)'
                  % len(stale), flush=True)


def _frozen_env():
    """1.259 frozen-opponent anchor config (set by
    run_td_v33loop.play_batch from loop_config.json; the loop
    generation-checks the pair at boot).  Returns (p, fp_path,
    sh_path) or None.  RuntimeError (worker-catchable, reported via
    out_q) on a half-configured pair or a bad p - a misconfigured
    anchor must refuse loudly, never run half-frozen."""
    fp9 = os.environ.get('WOTR_SP_FROZEN_FP') or None
    sh9 = os.environ.get('WOTR_SP_FROZEN_SH') or None
    if bool(fp9) != bool(sh9):
        raise RuntimeError(
            'frozen-opponent anchor (1.259): WOTR_SP_FROZEN_FP/'
            'WOTR_SP_FROZEN_SH must be set as a PAIR; got fp=%r sh=%r'
            % (fp9, sh9))
    if not fp9:
        return None
    raw9 = os.environ.get('WOTR_SP_FROZEN_P') or '0.30'
    try:
        p9 = float(raw9)
    except ValueError:
        raise RuntimeError('frozen-opponent anchor (1.259): bad '
                           'WOTR_SP_FROZEN_P=%r' % raw9)
    return (p9, fp9, sh9)


def _frozen_rng(seed0, wid):
    """1.259 selection stream: DEDICATED and worker-seeded so the
    anchor draw never perturbs the profile/episode RNG sequence.
    Integer seed only (no hash dependence); the SPD pin replays these
    exact calls."""
    import random
    return random.Random(1259 + seed0 * 4099 + wid * 2)


def _frozen_pick(rng, p):
    """1.259: ONE rng.random() draw decides frozen-vs-live (u < p), a
    SECOND draw picks the frozen SIDE uniformly.  Returns 'fp'/'sh'/
    None.  Keep the draw shape stable - the pin replays it."""
    if rng.random() >= p:
        return None
    return 'fp' if rng.random() < 0.5 else 'sh'


def _load_net(path, tag):
    # ONE loader: ai.load_value_net is the single loader every consumer
    # uses (checkpoint-described head_act + confidence-provider
    # auto-registration).
    import ai
    net = ai.load_value_net(path, side=tag)
    net.eval()
    net._checkpoint_sha256 = _checkpoint_id(path)
    return net


def _episode_randomness(run_seed, episode_index):
    """Keep profile/anchor choices independent of gameplay and worker count."""
    import random
    from rng_streams import current_contract, LEGACY, served_episode_seed, derive_seed
    if current_contract() == LEGACY:
        return {}, random, None
    seed = served_episode_seed(run_seed, episode_index)
    config = {'rng_episode_seed': seed,
              'rng_episode_identity': {'run_seed': run_seed, 'episode_index': episode_index}}
    return (config, random.Random(derive_seed(seed, 'profiles')),
            random.Random(derive_seed(seed, 'frozen-side')))


def _worker(wid, n_games, sims, trace_dir, seed0, fp_path, sh_path,
            out_q, net_dir=None, local_device='cpu', episode_stride=1):
    if local_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['WOTR_NET_ARCH'] = 'base'
    os.environ['WOTR_TREE_REUSE'] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    sys.path.insert(0, SRC)
    import random

    import numpy as np
    import torch
    torch.set_num_threads(1)

    import ai
    _CHECKPOINT_IDS.clear()
    try:
        if local_device not in ('cpu', 'cuda'):
            raise ValueError('Unsupported local worker device: %r' % local_device)
        if local_device == 'cuda' and (not torch.cuda.is_available()
                                       or torch.cuda.device_count() < 1):
            raise RuntimeError('CUDA worker requested but no assigned CUDA device is available')
        ai.DEVICE = torch.device(local_device)
        from reward_contract import current_contract
        contract = current_contract()
        # profile pins (2026-08-02 focus run): the loop narrows the
        # sampled profiles via env; default = full ai lists
        fp_profs = [p for p in (os.environ.get('WOTR_SP_FP_PROFILES')
                                or ','.join(ai.FP_PROFILES)).split(',')
                    if p]
        sh_profs = [p for p in (os.environ.get('WOTR_SP_SH_PROFILES')
                                or ','.join(ai.SH_PROFILES)).split(',')
                    if p]
        if net_dir:
            # v33 PER-PROFILE MODE (USER 2026-07-29: FULLY SEPARATE
            # nets, one per profile, absolutely nothing shared) — each
            # variant is an independent load; per game the assigned
            # profiles select which net PAIR plays.
            fp_nets = {p: _load_net(os.path.join(
                net_dir, 'live_fp_%s.pt' % p), 'fp')
                for p in fp_profs}
            sh_nets = {p: _load_net(os.path.join(
                net_dir, 'live_sh_%s.pt' % p), 'sh')
                for p in sh_profs}
        else:
            fpn = _load_net(fp_path, 'fp')
            shn = _load_net(sh_path, 'sh')
        # 1.259: the frozen pair loads through the SAME ai.load_value_
        # net path (generation-refused loudly here if the loop's boot
        # preflight was bypassed)
        fz9 = _frozen_env()
        frz_nets = None
        if fz9:
            frz_nets = {'fp': _load_net(fz9[1], 'fp'),
                        'sh': _load_net(fz9[2], 'sh')}
    except Exception as e:
        # a corrupt/incompatible ckpt must be reported, never kill the
        # worker silently (the collector would hang forever)
        detail = 'net load failed: %s' % repr(e)
        out_q.put(('e', wid, detail[:180], {
            'record_type': 'worker_error', 'stage': 'net_load',
            'worker_id': wid, 'assigned_games': n_games, 'error': detail}))
        return

    random.seed(seed0 + wid)
    np.random.seed(seed0 + wid)
    frz_rng = _frozen_rng(seed0, wid) if frz_nets else None
    os.makedirs(trace_dir, exist_ok=True)
    for k in range(n_games):
        rng_config, profile_rng, anchor_rng = _episode_randomness(seed0, wid + k * episode_stride)
        ai.PLAYER_PROFILES['fp'] = profile_rng.choice(fp_profs)
        ai.PLAYER_PROFILES['sh'] = profile_rng.choice(sh_profs)
        if net_dir:
            fpn = fp_nets[ai.PLAYER_PROFILES['fp']]
            shn = sh_nets[ai.PLAYER_PROFILES['sh']]
        # 1.259: with probability p the frozen pair's net plays ONE
        # side (uniform); that side is stamped and never trained on
        fs9 = (_frozen_pick(anchor_rng if anchor_rng is not None else frz_rng, fz9[0])
               if frz_rng is not None else None)
        fpn9 = frz_nets['fp'] if fs9 == 'fp' else fpn
        shn9 = frz_nets['sh'] if fs9 == 'sh' else shn
        checkpoint_ids = {'fp': getattr(fpn9, '_checkpoint_sha256', None),
                          'sh': getattr(shn9, '_checkpoint_sha256', None)}
        tp = os.path.join(trace_dir, 'w%d_g%d.trace.gz' % (wid, k))
        t0 = time.time()
        attempt = {'record_type': 'game_attempt', 'worker_id': wid, 'episode': k,
                   'profiles': dict(ai.PLAYER_PROFILES), 'checkpoint_ids': checkpoint_ids,
                   'reward_contract': contract, 'trace_path': tp, 'frozen_side': fs9}
        if rng_config:
            from rng_streams import ISOLATED
            attempt.update(rng_contract=ISOLATED, **rng_config)
        try:
            traj = ai._run_single_episode(
                fpn9, shn9, {'max_turns': 40, 'mcts_sims': sims,
                             'shadow_vp': 10, 'trace_path': tp,
                             'frozen_side': fs9, 'checkpoint_ids': dict(checkpoint_ids), **rng_config})
            gs9 = dict(traj.get('game_stats') or {})
            gs9.update(checkpoint_ids=dict(checkpoint_ids), reward_contract=contract,
                       worker_id=wid, episode=k, status='completed')
            gs9['wall_s'] = round(time.time() - t0, 1)
            gs9.setdefault('trace_path', tp)
            gs9.setdefault('profiles',
                           {'fp': ai.PLAYER_PROFILES['fp'],
                            'sh': ai.PLAYER_PROFILES['sh']})
            out_q.put(('g', wid, gs9))
        except Exception as e:
            attempt.update(status='error', stage='episode', error=repr(e),
                           error_type=type(e).__name__, wall_s=round(time.time() - t0, 1))
            out_q.put(('e', wid, repr(e)[:200], attempt))


def run_phase(label, n_games, workers, sims, trace_dir, seed0,
              net_dir=None):
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    out_q = ctx.Queue()
    attempts_path, summary_path = _report_paths()
    phase_id = uuid.uuid4().hex
    attempts, worker_failures = [], []
    _write_phase_summary(summary_path, phase_id, label, n_games, attempts,
                         worker_failures, attempts_path)

    per = n_games // workers
    extra = n_games - per * workers
    procs = []
    assignments = []
    for w in range(workers):
        n_w = per + (1 if w < extra else 0)
        if n_w <= 0:
            continue
        p = ctx.Process(target=_worker,
                        args=(w, n_w, sims, trace_dir, seed0,
                              FP_NET, SH_NET, out_q, net_dir, 'cpu', workers))
        p.start()
        procs.append(p)
        assignments.append((w, n_w, p))

    stats, errors = [], []
    t0 = time.time()
    import ai
    import queue as _qmod
    while len(attempts) < n_games:
        try:
            m = out_q.get(timeout=5)
        except _qmod.Empty:
            # dead-worker hang guard (the
            # arena_scale liveness pattern) — if every worker died
            # without reporting, stop collecting instead of blocking
            # on out_q.get() forever
            if not any(p.is_alive() for p in procs):
                print('all workers dead with %d/%d games collected — '
                      'aborting collection' % (len(stats), n_games),
                      flush=True)
                break
            continue
        attempt, failure = _message_record(m, phase_id, label)
        if attempt is not None:
            _append_attempt(attempts_path, attempt)
            attempts.append(attempt)
        if failure is not None:
            worker_failures.append(failure)
        _write_phase_summary(summary_path, phase_id, label, n_games, attempts,
                             worker_failures, attempts_path)
        if m[0] == 'g':
            stats.append(m[2])
            ai.append_game_log(GAMES_LOG, m[2], label)
            print('[%s] %d/%d  w%d  winner=%s turns=%s (%.0fs, cum %.0f min)'
                  % (label, len(stats), n_games, m[1],
                     m[2].get('winner'), m[2].get('turns'),
                     m[2].get('wall_s', 0), (time.time() - t0) / 60),
                  flush=True)
        else:
            errors.append(m)
            print('[%s] worker %d ERROR: %s' % (label, m[1], m[2]),
                  flush=True)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    for wid, planned, process in assignments:
        observed = sum(row.get('worker_id') == wid for row in attempts)
        if observed < planned:
            worker_failures.append({
                'record_type': 'worker_error', 'stage': 'reporting_gap',
                'worker_id': wid, 'assigned_games': planned,
                'observed': observed, 'unreported': planned - observed,
                'exitcode': process.exitcode})
    _write_phase_summary(summary_path, phase_id, label, n_games, attempts,
                         worker_failures, attempts_path, finished=True)
    try:
        ai.write_td_window_report(stats, label, REPORT)
    except Exception as e:
        print('report write skipped: %r' % e, flush=True)
    n_fp = sum(1 for g in stats if str(g.get('winner', '')).lower()
               in ('fp', 'free_peoples', 'freepeoples'))
    print('=== %s DONE: %d games, FP wins %d, errors %d, wall %.0f min'
          % (label, len(stats), n_fp, len(errors),
             (time.time() - t0) / 60), flush=True)
    return stats, errors


def _require_complete(stats, errors, expected, label):
    """A completed batch has every requested result and no worker errors."""
    if errors or len(stats) != expected:
        print('=== %s INCOMPLETE: %d/%d games collected (%d errors) ==='
              % (label, len(stats), expected, len(errors)), flush=True)
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=500)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    net_dir = os.environ.get('WOTR_SP_NET_DIR') or None
    if net_dir:
        import ai as _ai_chk
        _fp_p = [p for p in (os.environ.get('WOTR_SP_FP_PROFILES')
                             or ','.join(_ai_chk.FP_PROFILES)
                             ).split(',') if p]
        _sh_p = [p for p in (os.environ.get('WOTR_SP_SH_PROFILES')
                             or ','.join(_ai_chk.SH_PROFILES)
                             ).split(',') if p]
        want = (['live_fp_%s.pt' % p for p in _fp_p]
                + ['live_sh_%s.pt' % p for p in _sh_p])
        missing = [f for f in want
                   if not os.path.exists(os.path.join(net_dir, f))]
        if missing:
            sys.exit('per-profile mode: missing %s in %s'
                     % (missing, net_dir))
        print('nets (%d-net v33, per-profile): %s'
              % (len(want), ', '.join(want)),
              flush=True)
    else:
        if not (FP_NET and SH_NET):
            sys.exit(
                'no nets configured: set WOTR_SP_NET_DIR (loop '
                'per-profile roster) or WOTR_SP_FP_NET + '
                'WOTR_SP_SH_NET (explicit pair).  The silent '
                'v31-generation filename fallback was removed '
                '2026-08-11 — it loaded superseded nets.')
        for p in (FP_NET, SH_NET):
            if not os.path.exists(p):
                sys.exit('missing net: %s' % p)
        print('nets (BASE, local CPU): FP=%s  SH=%s'
              % (os.path.basename(FP_NET), os.path.basename(SH_NET)),
              flush=True)
    print('WOTR_TURN_HORIZON=%s' % os.environ.get('WOTR_TURN_HORIZON'),
          flush=True)
    # 1.259: validate the frozen-opponent anchor at BOOT (pair +
    # existence; the loop's preflight already generation-checked it)
    try:
        fz9 = _frozen_env()
    except RuntimeError as e:
        sys.exit(str(e))
    if fz9:
        for p9 in (fz9[1], fz9[2]):
            if not os.path.exists(p9):
                sys.exit('frozen-opponent anchor (1.259): missing '
                         'frozen checkpoint %s' % p9)
        print('frozen-opponent anchor (1.259): p=%.2f fp=%s sh=%s'
              % (fz9[0], os.path.basename(fz9[1]),
                 os.path.basename(fz9[2])), flush=True)

    _fresh_run_weight_guard()

    if args.smoke:
        print('=== v33sp SMOKE: 1 game, 1 worker, %d sims ===' % args.sims,
              flush=True)
        stats, errors = run_phase(
            'v33sp smoke', 1, 1, args.sims, SMOKE_TRACE_DIR,
            seed0=8000, net_dir=net_dir)
        _require_complete(stats, errors, 1, 'v33sp SMOKE')
        print('=== SMOKE COMPLETE ===', flush=True)
        return

    print('=== v33sp SELF-PLAY: %d games, %d workers, %d sims, '
          'BASE local CPU ===' % (args.games, args.workers, args.sims),
          flush=True)
    stats, errors = run_phase(
        'v33sp selfplay', args.games, args.workers, args.sims,
        TRACE_DIR,
        seed0=int(os.environ.get('WOTR_SP_SEED0', 30000)),
        net_dir=net_dir)
    _require_complete(stats, errors, args.games, 'v33sp')
    print('=== v33sp ALL GAMES COMPLETE ===', flush=True)


if __name__ == '__main__':
    main()
