"""Engine validation harness (TEACHER RETIRED 2026-07-27).

The raw-log closed-loop teacher and its replay-parity/canary gates are
deleted (USER order) — the rendered pipeline is the sole extraction
path.  Two gates remain:

  GATE 1  FIXED-SEED SELF-PLAY PARITY
      Play M self-play games at fixed seeds twice each; assert
      identical winner, turn count, and decision count (engine
      determinism).  Runs on CURRENT-generation nets: under the
      caller's WOTR_ACT_GEN, or, when that is unset, under the
      deployed head's generation (wotr_bridge/launch_head.py
      HEAD_ENV); the default pair is that generation's net tag
      (src/net_tags.py).  SKIPs cleanly only when no candidate pair
      is on disk; a pair the loader refuses is a FAIL, and gate 2
      still runs.

  GATE 2  RULE SUITES
      src/test_rules.py + src/test_comprehensive.py (must be 107/107)
      + src/test_conservation.py, under the caller's environment.

Companion gates outside this file: test_clause_matrix (217 clauses),
skew_gate.py (train/serve encoding parity), trace_audit.py (self-play
trace invariants), and the rendered pipeline's own censuses
(telemetry-only since 2026-08-11).

Usage:
  python src/engine_diff_harness.py            # both gates
  python src/engine_diff_harness.py --only 2
  python src/engine_diff_harness.py --fp-net wotr_fp_X.pt --sh-net wotr_sh_X.pt
"""
import argparse
import importlib.util
import os
import random
import re
import subprocess
import sys
import time
import traceback

# Absolute paths resolved BEFORE importing board, which os.chdir()s
# into src/ at import — relative paths would then misresolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

# (no canary gate: campaign_survey is RETIRED)

# Gate-1 nets: a checkpoint loads only under its own encoding generation
# (ai.load_value_net refuses any other), so gate 1 must run on
# CURRENT-generation nets under that generation.  R17-HARNESS-LOOP: the
# default used to be the v37 names with a lexically-last
# 'wotr_*_corpus_*.pt' fallback; with v37 gone that picked the retired
# v41-generation corpus_v42td pair, never the deployed v53 pair, and its
# load refusal killed the run before gate 2 under any generation but v41.
# The generation now comes from the head (launch_head.HEAD_ENV) unless the
# caller pins WOTR_ACT_GEN, and the default pair from net_tags.auto_tags,
# the table engine_play and the live own-model loader read.
GENERATION_KEYS = ('WOTR_ACT_GEN', 'WOTR_REWARD_CONTRACT')


def _head_generation():
    """WOTR_ACT_GEN and WOTR_REWARD_CONTRACT of the deployed head, from
    wotr_bridge/launch_head.py HEAD_ENV (stdlib only; loaded from its
    file without writing bytecode into wotr_bridge)."""
    path = os.path.join(ROOT, 'wotr_bridge', 'launch_head.py')
    spec = importlib.util.spec_from_file_location('_harness_launch_head', path)
    module = importlib.util.module_from_spec(spec)
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = saved
    head = dict(module.HEAD_ENV)
    return {key: head[key] for key in GENERATION_KEYS}


def gate1_environment(environ=None):
    """The generation variables gate 1 sets for its run.  Empty when the
    caller pinned WOTR_ACT_GEN (the caller's choice stands, its reward
    contract included); otherwise the head's generation, with the head's
    reward contract unless the caller set one."""
    environ = os.environ if environ is None else environ
    if environ.get('WOTR_ACT_GEN', '').strip():
        return {}
    head = _head_generation()
    if environ.get('WOTR_REWARD_CONTRACT', '').strip():
        del head['WOTR_REWARD_CONTRACT']
    return head


def _generation_text():
    return ' '.join('%s=%s' % (key, os.environ.get(key) or '(unset)')
                    for key in GENERATION_KEYS)


class _Refused(Exception):
    pass


def resolve_gate1_nets(ai, fp_net=None, sh_net=None):
    """Load the gate-1 pair under the running generation.

    An explicit --fp-net / --sh-net names that side; an unnamed side takes
    net_tags.auto_tags(<running generation>) in order, and the first
    candidate pair that is on disk AND loads wins.  Returns (fp_name,
    sh_name, fp, sh) with the loaded nets, or None when no candidate pair
    is on disk (the net-less soft skip).  Pairs that are on disk but all
    refused raise, naming every refusal: never a SKIP."""
    import net_tags
    if fp_net and sh_net:
        candidates = [(fp_net, sh_net)]
    else:
        candidates = [(fp_net or 'wotr_fp_%s.pt' % tag,
                       sh_net or 'wotr_sh_%s.pt' % tag)
                      for tag in net_tags.auto_tags(ai._OBSERVATION_GENERATION)]
    present = [pair for pair in candidates
               if all(os.path.exists(os.path.join(ROOT, name)) for name in pair)]
    if not present:
        return None

    def load(name, side):
        # canonical loader: head_act reconstruction + eval() +
        # cross-generation refusal (a hand-rolled load here evaluated
        # a DIFFERENT value function than production)
        try:
            net = ai.load_value_net(os.path.join(ROOT, name), side=side)
        except (RuntimeError, ValueError) as exc:
            raise _Refused('%s: %s' % (name, exc))
        if net is None:
            raise RuntimeError('gate-1 net failed to load: %s' % name)
        return net

    refusals = []
    for fp_name, sh_name in present:
        try:
            return fp_name, sh_name, load(fp_name, 'fp'), load(sh_name, 'sh')
        except _Refused as exc:
            refusals.append(str(exc))
    raise RuntimeError(
        'no gate-1 net pair loads under %s (refused: %s). Pass --fp-net/'
        '--sh-net nets of this generation, or set WOTR_ACT_GEN (and '
        'WOTR_REWARD_CONTRACT) to the generation of the nets.'
        % (_generation_text(), ' | '.join(refusals)))


# --------------------------------------------------------------------------
def gate_selfplay(seeds, sims, max_turns, fp_net_path=None, sh_net_path=None):
    """GATE 1: fixed-seed self-play determinism. Returns (passed,
    detail_dict); detail 'skipped' when no candidate pair is on disk.
    The gate-1 generation variables are set for this gate only: gate 2
    runs under the caller's environment."""
    overrides = gate1_environment()
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        return _gate_selfplay(seeds, sims, max_turns, fp_net_path,
                              sh_net_path, sorted(overrides))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _gate_selfplay(seeds, sims, max_turns, fp_net_path, sh_net_path,
                   from_head):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import ai
    ai.DEVICE = torch.device('cpu')
    wanted = os.environ.get('WOTR_ACT_GEN', '').strip().lower()
    if ai._OBSERVATION_GENERATION != wanted:
        raise RuntimeError(
            'ai was imported under WOTR_ACT_GEN=%r before gate 1 set %r; '
            'run the harness in a fresh process'
            % (ai._OBSERVATION_GENERATION, wanted))
    print('  [gate1] generation %s%s' % (
        _generation_text(),
        ' (from wotr_bridge/launch_head.py: %s)' % ', '.join(from_head)
        if from_head else ' (caller environment)'), flush=True)
    pair = resolve_gate1_nets(ai, fp_net_path, sh_net_path)
    if pair is None:
        print('  SKIP: no gate-1 net pair on disk under %s -- train first'
              % _generation_text(), flush=True)
        return True, {'skipped': 'nets missing (%s)' % _generation_text()}
    fp_name, sh_name, fp, sh = pair
    print('  [gate1] nets %s / %s' % (fp_name, sh_name), flush=True)

    # the objective each net was trained for (engine_play.load_nets): a
    # profile-contract episode validates every net against its profile
    from reward_contract import current_contract, PROFILE, validate_net_profile
    saved_profiles = dict(ai.PLAYER_PROFILES)
    if current_contract() == PROFILE:
        profiles = {}
        for side, net in (('fp', fp), ('sh', sh)):
            profile = getattr(net, 'reward_profile', None)
            validate_net_profile(net, side, profile)
            profiles[side] = profile
        ai.PLAYER_PROFILES.update(profiles)

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
    try:
        for seed in seeds:
            r1 = play(seed)
            r2 = play(seed)
            identical = (r1 == r2)
            all_ok = all_ok and identical
            detail_rows.append(
                f"seed={seed} run1={r1} run2={r2} identical={identical}")
            print(f"  [gate1] seed={seed} run1={r1} run2={r2} "
                  f"identical={identical}", flush=True)
    finally:
        ai.PLAYER_PROFILES.clear()
        ai.PLAYER_PROFILES.update(saved_profiles)

    return all_ok, {'generation': _generation_text(),
                    'nets': '%s / %s' % (fp_name, sh_name),
                    'rows': detail_rows}


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
    ap.add_argument('--fp-net', default=None,
                    help='gate-1 FP net (default: the net tag of the '
                         'gate-1 generation, src/net_tags.py)')
    ap.add_argument('--sh-net', default=None,
                    help='gate-1 SH net (default as --fp-net)')
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
        seeds = [int(x) for x in args.sp_seeds.split(',') if x.strip()]
        try:
            p, d = gate_selfplay(seeds, args.sp_sims, args.sp_max_turns,
                                 args.fp_net, args.sh_net)
        except Exception as exc:
            # R17-HARNESS-LOOP: an error in gate 1 is gate 1's FAIL; gate 2
            # still runs and the summary still prints (exit 1)
            traceback.print_exc()
            p, d = False, {'error': '%s: %s' % (type(exc).__name__, exc)}
        results.append(('GATE 1 self-play (SKIPPED: no nets)'
                        if d.get('skipped') else 'GATE 1 self-play', p, d))

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
