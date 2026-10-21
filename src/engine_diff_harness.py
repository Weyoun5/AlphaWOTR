"""Engine validation harness (TEACHER RETIRED 2026-07-27).

The raw-log closed-loop teacher and its replay-parity/canary gates are
deleted (USER order) — the rendered pipeline is the sole extraction
path.  Two gates remain:

  GATE 1  FIXED-SEED SELF-PLAY PARITY
      Play M self-play games at fixed seeds twice each; assert
      identical winner, turn count, and decision count (engine
      determinism).  SKIPs cleanly when no nets are trained.

  GATE 2  RULE SUITES
      src/test_rules.py + src/test_comprehensive.py (must be 107/107)
      + src/test_conservation.py.

Companion gates outside this file: test_clause_matrix (217 clauses),
skew_gate.py (train/serve encoding parity), trace_audit.py (self-play
trace invariants), and the rendered pipeline's own censuses
(telemetry-only since 2026-08-11).

Usage:
  python src/engine_diff_harness.py            # both gates
  python src/engine_diff_harness.py --only 2
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time

# Absolute paths resolved BEFORE importing board, which os.chdir()s
# into src/ at import — relative paths would then misresolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

# (no canary gate: campaign_survey is RETIRED)

# Default self-play nets: checkpoints only load under their own
# encoding generation, so gate 1 must run on CURRENT-generation nets
# (td > plain within the pinned version, newest corpus glob as a
# fallback).  When nothing resolves, the fabricated name fails main()'s
# existence check and gate 1 records a SKIP that counts as PASS -- a
# deliberate soft-skip for net-less trees, NOT an error.


def _default_net(tag):
    for sfx in ('v37td', 'v37'):
        name = 'wotr_%s_corpus_%s.pt' % (tag, sfx)
        if os.path.exists(os.path.join(ROOT, name)):
            return name
    import glob
    cands = sorted(glob.glob(os.path.join(ROOT,
                                          'wotr_%s_corpus_*.pt' % tag)))
    if cands:
        return os.path.basename(cands[-1])
    return 'wotr_%s_corpus_v37td.pt' % tag   # -> SKIP in main()


DEFAULT_FP_NET = _default_net('fp')
DEFAULT_SH_NET = _default_net('sh')

# --------------------------------------------------------------------------
def gate_selfplay(seeds, sims, max_turns, fp_net_path, sh_net_path):
    """GATE 1: fixed-seed self-play determinism. Returns (passed, detail_dict)."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import ai
    ai.DEVICE = torch.device('cpu')

    def load(path, side):
        # canonical loader: head_act reconstruction + eval() +
        # cross-generation refusal (a hand-rolled load here evaluated
        # a DIFFERENT value function than production)
        net = ai.load_value_net(os.path.join(ROOT, path), side=side)
        if net is None:
            raise RuntimeError('gate-1 net failed to load: %s' % path)
        return net

    fp = load(fp_net_path, 'fp')
    sh = load(sh_net_path, 'sh')

    def play(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        r = ai._run_single_episode(fp, sh, {
            'max_turns': max_turns, 'mcts_sims': sims, 'shadow_vp': 10,
            'add_noise': False, 'temperature_override': 0.15})
        return [r['winner'], int(r['turns']), int(len(r['sides']))]

    all_ok = True
    detail_rows = []
    for seed in seeds:
        r1 = play(seed)
        r2 = play(seed)
        identical = (r1 == r2)
        all_ok = all_ok and identical
        detail_rows.append(
            f"seed={seed} run1={r1} run2={r2} identical={identical}")
        print(f"  [gate1] seed={seed} run1={r1} run2={r2} "
              f"identical={identical}", flush=True)

    return all_ok, {'rows': detail_rows}


# --------------------------------------------------------------------------
def _run_py(script):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', PYTHONHASHSEED='0',
               PYTHONIOENCODING='utf-8')
    p = subprocess.run([sys.executable, os.path.join(_HERE, script)],
                       cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=1800)
    return p.stdout.decode('utf-8', 'replace')


def gate_suites():
    """GATE 2: rule suites. Returns (passed, detail_dict)."""
    out_c = _run_py('test_comprehensive.py')
    m_run = re.search(r'Tests run:\s*(\d+)', out_c)
    m_fail = re.search(r'Failures:\s*(\d+)', out_c)
    m_err = re.search(r'Errors:\s*(\d+)', out_c)
    runs = int(m_run.group(1)) if m_run else -1
    fails = int(m_fail.group(1)) if m_fail else -1
    errs = int(m_err.group(1)) if m_err else -1
    comp_ok = (runs == 107 and fails == 0 and errs == 0)
    print(f"  [gate2] test_comprehensive: run={runs} fail={fails} "
          f"err={errs}", flush=True)

    out_r = _run_py('test_rules.py')
    rules_ok = 'All tests passed!' in out_r
    print(f"  [gate2] test_rules: {'PASS' if rules_ok else 'FAIL'}",
          flush=True)

    # Piece-conservation invariant (2026-07-20): unit-level check that
    # every nation's deployed+casualties <= max after unit ops and
    # after EVERY event card. Locks the casualty/downgrade/card mint
    # cluster (amendments 1.187-1.190).
    out_cons = _run_py('test_conservation.py')
    cons_ok = 'RESULT: PASS' in out_cons
    print(f"  [gate2] test_conservation: {'PASS' if cons_ok else 'FAIL'}",
          flush=True)

    return (comp_ok and rules_ok and cons_ok), {
        'comprehensive': f"run={runs} fail={fails} err={errs}",
        'rules_all_passed': rules_ok,
        'conservation': cons_ok}


# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='1,2',
                    help='comma list of gate numbers to run')
    ap.add_argument('--sp-seeds', default='7,13',
                    help='gate-1 self-play seeds')
    ap.add_argument('--sp-sims', type=int, default=60)
    ap.add_argument('--sp-max-turns', type=int, default=12)
    ap.add_argument('--fp-net', default=DEFAULT_FP_NET)
    ap.add_argument('--sh-net', default=DEFAULT_SH_NET)
    args = ap.parse_args()

    only = {int(x) for x in args.only.split(',') if x.strip()}
    unknown = only - {1, 2}
    if unknown or not only:
        sys.exit('unknown gate(s) %s -- valid gates are 1,2'
                 % sorted(unknown))
    results = []
    t0 = time.time()

    if 1 in only:
        print('=== GATE 1: FIXED-SEED SELF-PLAY PARITY ===', flush=True)
        _fp2 = os.path.join(ROOT, args.fp_net)
        _sh2 = os.path.join(ROOT, args.sh_net)
        if not (os.path.exists(_fp2) and os.path.exists(_sh2)):
            print("  SKIP: nets not found (%s / %s) — train first"
                  % (args.fp_net, args.sh_net), flush=True)
            results.append(('GATE 1 self-play (SKIPPED: no nets)', True,
                            {'skipped': 'nets missing'}))
        else:
            seeds = [int(x) for x in args.sp_seeds.split(',')
                     if x.strip()]
            p, d = gate_selfplay(seeds, args.sp_sims, args.sp_max_turns,
                                 args.fp_net, args.sh_net)
            results.append(('GATE 1 self-play', p, d))

    if 2 in only:
        print('=== GATE 2: RULE SUITES ===', flush=True)
        p, d = gate_suites()
        results.append(('GATE 2 rule-suites', p, d))

    print('=' * 64)
    print("HARNESS RESULTS  (%.0fs)" % (time.time() - t0))
    print("=" * 64)
    all_pass = True
    for name, p, d in results:
        all_pass = all_pass and p
        print(f"  {'PASS' if p else 'FAIL'}  {name}")
        for k, v in d.items():
            print(f"          {k}: {v}")
    print("-" * 64)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
