# -*- coding: utf-8 -*-
"""Closed-loop self-play with per-side value and residual-policy updates.

Games must satisfy trace integrity and reward-contract checks before entering
replay. Strength probes and arena scores remain measurements, not promotion
or style gates. The legacy profile objective remains the compatibility
default; WOTR_REWARD_CONTRACT=predecision_winloss_v1 selects a separate,
versioned win/loss experiment. Existing buffers are never relabeled.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from process_tree import ProcessTree

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# Preserve default CPU execution without overriding scheduler-owned masks.
# Explicit GPU selection is validated before model loads or training writes.
from training_device import (requested_training_device, activate_training_device,
                             VISIBILITY_KEYS)
_LOOP_INSERTED_CPU_MASK = (requested_training_device() == 'cpu' and
                           not any(key in os.environ for key in VISIBILITY_KEYS))
if _LOOP_INSERTED_CPU_MASK:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from reward_contract import current_contract, WINLOSS, PROFILE, PREDECISION, uses_predecision
if __name__ == '__main__':
    # New diagnostic runs use corrected profiles; imports never change the
    # contract of a dashboard or training process that happens to import us.
    os.environ.setdefault('WOTR_REWARD_CONTRACT', PROFILE)
LOOP = os.environ.get('WOTR_LOOP_DIR') or os.path.join(
    ROOT, 'data', {WINLOSS: 'winloss_loop_v1', PROFILE: 'profile_loop_v1'}.get(current_contract(), 'v33loop'))
TRACE_DIR = (os.path.join(LOOP, 'incoming_traces') if uses_predecision()
             else os.path.join(ROOT, 'data', 'td_v33_traces'))
GAMES_JSONL = (os.path.join(LOOP, 'incoming_games.jsonl') if uses_predecision()
               else os.path.join(ROOT, 'data', 'td_v33_games.jsonl'))
# v33jt seeds: ring-inventory + declare-staging + pool-staging fixes,
# outcome-anchored value heads (gates a/c/e/f/h/i all green).
JT = {'fp': os.path.join(ROOT, 'wotr_fp_corpus_v33jt.pt'),
      'sh': os.path.join(ROOT, 'wotr_sh_corpus_v33jt.pt')}
for _seed_side in ('fp', 'sh'):
    JT[_seed_side] = os.environ.get('WOTR_LOOP_SEED_' + _seed_side.upper()) or JT[_seed_side]
PY = sys.executable

# USER 2026-08-02 ("for speed, only train FP ring vs SH military;
# when I'm satisfied we look at the other profiles"): 2-net focus run
# — every game trains BOTH nets (~3x data/net/batch vs the 6-net
# split).  Full roster to restore later:
#   (('fp','mixed'), ('sh','mixed'), ('fp','ring'), ('fp','military'),
#    ('sh','corruption'), ('sh','military'))
NET_KEYS = (('fp', 'ring'), ('sh', 'military'))
# fixed arena partners (baseline nets; identical on both sides of the
# pairing so the measured delta isolates the one live net under test)
ARENA_PARTNER = {'fp': ('sh', 'military'), 'sh': ('fp', 'ring')}

BATCH_GAMES = int(os.environ.get('WOTR_BATCH_GAMES', '50') or 50)
ARENA_EVERY_BATCHES = 4          # = every 200 games
LAM = 0.8
LR = 1e-4
EPOCHS = 2
# 1.226b: residual policy-delta training — Adam weight decay ON DELTA
# PARAMS ONLY (shrinkage toward zero = the BC prior stays the anchor)
DELTA_WD = 1e-4

# ---- GUI-tunable config (USER 2026-08-11: dashboard settings panel)
# LOOP\loop_config.json overrides the constants above at STARTUP —
# the dashboard writes it, a loop (re)start reads it.  Unknown keys
# are rejected loudly (a typo must never silently run defaults).
ARENA_ENABLED = True
ARENA_WORKERS = 12               # arena games = workers x per-worker
ARENA_GAMES_PER_WORKER = 8
ARENA_SIMS = 60
# 1.259 (USER 2026-08-29, reaffirmed): 30% of worker episodes play one
# side with a FROZEN opponent net pair; that side is never trained on.
# DISABLED-LOUD until a current-generation pair is configured (the
# intended anchor = the v42 BC baseline once the re-seed exists).
FROZEN_OPPONENT_P = 0.30
FROZEN_CKPT_FP = None
FROZEN_CKPT_SH = None
_CFG_KEYS = ('arena_enabled', 'arena_every_batches', 'arena_workers',
             'arena_games_per_worker', 'arena_sims', 'batch_games',
             'lr', 'epochs', 'lam', 'delta_wd', 'buffer_k',
             'workers', 'sims', 'total_games', 'frozen_opponent_p',
             'frozen_ckpt_fp', 'frozen_ckpt_sh')


def _training_controls(cfg):
    values = [cfg.get(key, default) for key, default in
              (('lr', LR), ('epochs', EPOCHS), ('lam', LAM), ('delta_wd', DELTA_WD))]
    if any(isinstance(value, bool) for value in values):
        raise ValueError('Training controls must be numbers, not booleans')
    lr, epochs, lam, wd = map(float, values)
    if not all(math.isfinite(value) for value in (lr, epochs, lam, wd)):
        raise ValueError('Training controls must be finite')
    if lr <= 0 or epochs < 1 or not epochs.is_integer() or not 0 <= lam <= 1 or wd < 0:
        raise ValueError('Training requires lr > 0, integral epochs >= 1, lam in [0,1], delta_wd >= 0')
    return lr, int(epochs), lam, wd


def _apply_loop_config():
    """Read LOOP\\loop_config.json into the module constants; returns
    the dict of overrides for the startup log (empty if no file).
    workers/sims/total_games are consumed by main() as argparse
    DEFAULTS (explicit CLI flags still win)."""
    global ARENA_ENABLED, ARENA_EVERY_BATCHES, ARENA_WORKERS, \
        ARENA_GAMES_PER_WORKER, ARENA_SIMS, BATCH_GAMES, LR, EPOCHS, \
        LAM, DELTA_WD, BUFFER_K, FROZEN_OPPONENT_P, FROZEN_CKPT_FP, \
        FROZEN_CKPT_SH
    p = os.path.join(LOOP, 'loop_config.json')
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as stream:
        cfg = json.load(stream)
    if not isinstance(cfg, dict):
        raise ValueError('loop_config.json must contain an object')
    bad = [k for k in cfg if k not in _CFG_KEYS]
    if bad:
        raise SystemExit('loop_config.json: unknown key(s) %s '
                         '(valid: %s)' % (bad, list(_CFG_KEYS)))
    # Validate training controls before mutating any effective settings.
    training = _training_controls(cfg)
    ARENA_ENABLED = bool(cfg.get('arena_enabled', ARENA_ENABLED))
    ARENA_EVERY_BATCHES = int(cfg.get('arena_every_batches',
                                      ARENA_EVERY_BATCHES))
    ARENA_WORKERS = int(cfg.get('arena_workers', ARENA_WORKERS))
    ARENA_GAMES_PER_WORKER = int(cfg.get('arena_games_per_worker',
                                         ARENA_GAMES_PER_WORKER))
    ARENA_SIMS = int(cfg.get('arena_sims', ARENA_SIMS))
    BATCH_GAMES = int(cfg.get('batch_games', BATCH_GAMES))
    LR, EPOCHS, LAM, DELTA_WD = training
    if 'buffer_k' in cfg:
        BUFFER_K = int(cfg['buffer_k'])
    FROZEN_OPPONENT_P = float(cfg.get('frozen_opponent_p',
                                      FROZEN_OPPONENT_P))
    FROZEN_CKPT_FP = cfg.get('frozen_ckpt_fp') or None
    FROZEN_CKPT_SH = cfg.get('frozen_ckpt_sh') or None
    # 1.259: the anchor is a NET PAIR - a half-configured pair must
    # refuse at boot, never run half-frozen silently
    if bool(FROZEN_CKPT_FP) != bool(FROZEN_CKPT_SH):
        raise SystemExit(
            'loop_config.json: frozen_ckpt_fp/frozen_ckpt_sh must be '
            'set as a PAIR (1.259 anchor is a frozen net pair); got '
            'fp=%r sh=%r' % (FROZEN_CKPT_FP, FROZEN_CKPT_SH))
    return cfg


def _frozen_cfg():
    """1.259: (p, fp_path, sh_path) when the frozen-opponent anchor is
    configured (both checkpoints set), else None - the DISABLED case
    is loud per batch via _log_frozen_telemetry, never silent."""
    if FROZEN_CKPT_FP and FROZEN_CKPT_SH:
        return (float(FROZEN_OPPONENT_P), FROZEN_CKPT_FP,
                FROZEN_CKPT_SH)
    return None


def _nm(tag, side, prof):
    return os.path.join(LOOP, '%s_%s_%s.pt' % (tag, side, prof))


def log_line(text):
    os.makedirs(LOOP, exist_ok=True)
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(LOOP, 'loop_log.md'), 'a',
              encoding='utf-8') as f:
        f.write('- %s %s\n' % (stamp, text))
    print(text, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(obj, path):
    import torch
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_net(net, side, path, profile=None):
    import ai
    from trace_decisions import current_learning_context
    from selector_contract import training_identity
    if current_contract() == PROFILE:
        from reward_contract import validate_profile
        validate_profile(side, profile)
    ck = {'model_state': net.state_dict(),
          'input_dim': ai.get_feature_dim(),
          'action_dim': ai.ACTION_ENCODING_DIM,
          # head activation is checkpoint-described: the
          # ScaledTanh module is parameter-free, so WITHOUT
          # this field a reload silently reconstructs plain
          # Tanh (VALUE interop trap; battery A6 — bit us live
          # in leg-2v3 before this hunk landed)
          'head_act': ('scaled2'
                       if getattr(net.value_head[-1], 'scale',
                                  None) else 'tanh'),
           'side': side,
           'reward_contract': current_contract()}
    ck.update(current_learning_context())
    from trace_decisions import checkpoint_learning_lineage
    ck.update(checkpoint_learning_lineage(net))
    ck.update(training_identity())
    if current_contract() == PROFILE:
        ck['reward_profile'] = profile
    from optimizer_continuity import checkpoint_payload
    ck.update(checkpoint_payload(net, ck))
    # conf_* fields are checkpoint-described the same way head_act is
    # (ultrareview bug_006): a from-scratch dict here stripped the
    # confidence stamp from live_*.pt on the FIRST accepted batch,
    # de-registering the provider for every later load (CONF-4 DARK
    # from batch 2 on).  Round-trip the stamp from the file being
    # overwritten.
    try:
        import torch as _t
        if os.path.exists(path):
            _old = _t.load(path, map_location='cpu',
                           weights_only=False)
            for _f in ('conf_mech', 'conf_state', 'conf_calib',
                       'conf_meta'):
                if _f in _old:
                    ck[_f] = _old[_f]
    except Exception as _e:
        log_line('save_net: conf_* round-trip FAILED for %s (%r) — '
                 'saving without stamp' % (os.path.basename(path), _e))
    atomic_save(ck, path)


def load_net(path, side, profile=None):
    activate_training_device()
    import ai
    net = ai.load_value_net(path, side=side, profile=profile)
    if net is not None:
        from optimizer_continuity import load_training_state
        load_training_state(net, path)
    return net


def preflight_generation():
    """Boot-time generation pin (audit 2026-08-29 SPTR-1/SPCP-4/
    SPLP-2): the resolved action-encoding generation and the roster
    about to be adopted must agree BEFORE any training happens --
    the loop must never train a lineage whose action_dim differs
    from the generation it encodes under.  Reads only the checkpoint
    stamps (no net construction); refuses loudly with the exact
    facts on any mismatch or unverifiable stamp."""
    import pickle
    import torch
    import ai
    gen = os.environ.get('WOTR_ACT_GEN') or '(unset)'
    want = int(ai.ACTION_ENCODING_DIM)
    for tag in ('baseline', 'live', 'last_good'):
        for key in NET_KEYS:
            p = _nm(tag, *key)
            if not os.path.exists(p):
                continue
            try:
                ck = torch.load(p, map_location='cpu',
                                weights_only=False)
            except (OSError, RuntimeError, EOFError, ValueError,
                    pickle.UnpicklingError) as e:
                raise SystemExit(
                    'GEN PREFLIGHT: unreadable roster checkpoint '
                    '%s (%r) -- refusing to boot' % (p, e))
            got = ck.get('action_dim') if isinstance(ck, dict) \
                else None
            if isinstance(ck, dict):
                from trace_decisions import validate_action_encoding_contract
                validate_action_encoding_contract(ck, required=(
                    ck.get('reward_contract') in PREDECISION or
                    'observation_contract' in ck or 'selector_contract' in ck))
            if isinstance(ck, dict) and uses_predecision() and (
                    uses_predecision(ck.get('reward_contract')) or 'selector_contract' in ck):
                from selector_contract import current_identity, validate_identity
                validate_identity(ck, expected=current_identity())
            if got is None:
                raise SystemExit(
                    'GEN PREFLIGHT: %s carries no action_dim stamp '
                    '-- cannot verify the generation (env '
                    'WOTR_ACT_GEN=%s wants action_dim=%d); refusing '
                    'an unverifiable roster' % (p, gen, want))
            if int(got) != want:
                msg = ('GEN PREFLIGHT MISMATCH: env WOTR_ACT_GEN=%s '
                       'resolves ACTION_ENCODING_DIM=%d but roster '
                       'net %s stamps action_dim=%d. Refusing to '
                       'train the wrong lineage. The Aug-15 roster '
                       'predates the v41 switch: RE-SEED '
                       'deliberately (move baseline_/live_/'
                       'last_good_*.pt aside and re-run setup from '
                       'a %d-dim seed pair, e.g. the corpus_live/'
                       'v41td nets) before booting under this '
                       'generation.'
                       % (gen, want, p, int(got), want))
                log_line(msg)
                raise SystemExit(msg)
    # 1.259: the preflight covers the FROZEN opponent checkpoints too
    # -- a wrong-generation frozen net is refused at BOOT, not per
    # episode (the workers load them through ai.load_value_net).
    for lbl, fz9 in (('frozen_ckpt_fp', FROZEN_CKPT_FP),
                     ('frozen_ckpt_sh', FROZEN_CKPT_SH)):
        if not fz9:
            continue
        if not os.path.exists(fz9):
            raise SystemExit(
                'GEN PREFLIGHT: frozen checkpoint %s=%s does not '
                'exist (1.259 anchor) -- refusing to boot'
                % (lbl, fz9))
        try:
            ck = torch.load(fz9, map_location='cpu',
                            weights_only=False)
        except (OSError, RuntimeError, EOFError, ValueError,
                pickle.UnpicklingError) as e:
            raise SystemExit(
                'GEN PREFLIGHT: unreadable frozen checkpoint %s=%s '
                '(%r) -- refusing to boot' % (lbl, fz9, e))
        got = ck.get('action_dim') if isinstance(ck, dict) else None
        if isinstance(ck, dict):
            from trace_decisions import validate_action_encoding_contract
            validate_action_encoding_contract(ck, required=(
                ck.get('reward_contract') in PREDECISION or
                'observation_contract' in ck or 'selector_contract' in ck))
        if got is None:
            raise SystemExit(
                'GEN PREFLIGHT: frozen checkpoint %s=%s carries no '
                'action_dim stamp -- cannot verify the generation '
                '(env WOTR_ACT_GEN=%s wants action_dim=%d); refusing '
                'an unverifiable frozen anchor (1.259)'
                % (lbl, fz9, gen, want))
        if int(got) != want:
            msg = ('GEN PREFLIGHT MISMATCH: frozen checkpoint %s=%s '
                   'stamps action_dim=%d but env WOTR_ACT_GEN=%s '
                   'resolves ACTION_ENCODING_DIM=%d. A '
                   'wrong-generation frozen anchor is refused at '
                   'BOOT (1.259) -- configure a current-generation '
                   'pair (intended: the v42 BC baseline).'
                   % (lbl, fz9, int(got), gen, want))
            log_line(msg)
            raise SystemExit(msg)


def setup():
    """Clone seeds -> 3 x len(NET_KEYS) physical files; write the
    baseline sha256 manifest; assert full isolation."""
    os.makedirs(LOOP, exist_ok=True)
    from trace_decisions import current_learning_context
    from selector_contract import training_identity
    context_contracts = dict(current_learning_context(), **training_identity())
    contract_path = os.path.join(LOOP, 'learning_contract.json')
    if os.path.exists(contract_path):
        with open(contract_path, encoding='utf-8') as handle:
            recorded = json.load(handle)
        if recorded.get('reward_contract') != current_contract():
            raise ValueError('Loop reward contract mismatch; select a fresh WOTR_LOOP_DIR')
        for field, expected in context_contracts.items():
            if recorded.get(field) != expected:
                raise ValueError('Loop %s contract mismatch; select a fresh WOTR_LOOP_DIR' % field)
    elif (
            glob.glob(os.path.join(LOOP, 'live_*.pt'))
            or glob.glob(os.path.join(LOOP, 'buffer', '*', 'b*.pkl.gz'))):
        raise ValueError('Unversioned existing loop cannot be relabeled; select a fresh WOTR_LOOP_DIR')
    for side, prof in NET_KEYS:
        seed = JT[side]
        for tag in ('baseline', 'live', 'last_good'):
            dst = _nm(tag, side, prof)
            if not os.path.exists(dst):
                if not os.path.exists(seed):
                    raise SystemExit('missing seed net: %s' % seed)
                shutil.copy2(seed, dst)
    with open(contract_path + '.tmp', 'w', encoding='utf-8') as handle:
        json.dump({'reward_contract': current_contract(), 'seed_paths': JT,
                   **context_contracts}, handle, indent=2)
    os.replace(contract_path + '.tmp', contract_path)
    man_path = os.path.join(LOOP, 'baseline_manifest.json')
    if not os.path.exists(man_path):
        man = {os.path.basename(_nm('baseline', s, p)):
               sha256(_nm('baseline', s, p)) for s, p in NET_KEYS}
        with open(man_path, 'w', encoding='utf-8') as f:
            json.dump(man, f, indent=1)
        log_line('baseline manifest written (%d nets)' % len(man))
    # isolation assertions: 3 tiers x len(NET_KEYS) distinct real files
    paths = [_nm(t, s, p) for t in ('baseline', 'live', 'last_good')
             for s, p in NET_KEYS]
    assert len(set(map(os.path.abspath, paths))) == 3 * len(NET_KEYS)
    for p in paths:
        assert os.path.isfile(p), p
    # (stray pre-run traces are handled by quarantine_stale_traces at
    # loop start)


def integrity_check():
    """Baselines must NEVER change: verify sha256 vs the manifest."""
    man = json.load(open(os.path.join(LOOP, 'baseline_manifest.json'),
                         encoding='utf-8'))
    for name, want in man.items():
        got = sha256(os.path.join(LOOP, name))
        if got != want:
            log_line('BASELINE INTEGRITY VIOLATION: %s changed!' % name)
            raise SystemExit(3)


def quarantine_stale_traces():
    """Traces predating THIS loop invocation must never enter a batch
    (batch-0 audit: an aborted prior run's strays in the shared
    TRACE_DIR were swept into the next run's batch by play_batch's
    unconditional move).  Startup sweep: move any pre-existing
    self-play traces + runner games.jsonl into a quarantine dir."""
    stale = glob.glob(os.path.join(TRACE_DIR, '*.trace.gz'))
    if os.path.exists(GAMES_JSONL):
        stale.append(GAMES_JSONL)
    if not stale:
        return
    qdir = os.path.join(LOOP, 'stale_quarantine_%s'
                        % time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(qdir, exist_ok=True)
    for f in stale:
        shutil.move(f, qdir)
    log_line('quarantined %d stale pre-run trace file(s) -> %s'
             % (len(stale), os.path.basename(qdir)))


# The legacy local backend retains its historical wall limit. Served runs
# detect per-worker loss of concrete progress and child/RPC failures instead;
# a healthy batch may take any total duration.
BATCH_WALL_SEC = 8 * 3600
_PRE_READY_ATTEMPT = None

# ---- graceful per-game failure + native route (package Y, 2026-09-11) ----
# USER: "don't crash fully on 1 break, just handle it gracefully and we can
# note it in the log for later root cause detection".  A served batch whose
# runner recorded status=error attempts (fixtures under <output>/failures/)
# is still archived, ACCEPTED and trained on when it holds at least
# WOTR_LOOP_MIN_BATCH_GAMES completed games (default 1); the shortfall and
# every failure entry go to the loop log and LOOP/failures_ledger.jsonl.
MIN_BATCH_GAMES_ENV = 'WOTR_LOOP_MIN_BATCH_GAMES'
FAILURES_LEDGER = 'failures_ledger.jsonl'
# Operational knobs (never learning controls) and the native-route selectors are
# excluded from the journal's environment identity; the native route itself is
# recorded under the journal 'sources' (a search implementation identity that a
# reviewed_runtime_source_transitions_v1 envelope can approve).
JOURNAL_ENVIRONMENT_EXCLUDED = ('WOTR_LOOP_DIR', 'WOTR_TOTAL_GAMES', 'WOTR_WORKERS',
                                MIN_BATCH_GAMES_ENV, 'WOTR_SP_MAX_GAME_FAILURES',
                                'WOTR_NATIVE_SEARCH', 'WOTR_NATIVE_SEARCH_BOX',
                                'WOTR_NATIVE_SEARCH_REFERENCE', 'WOTR_NATIVE_SEARCH_CORE',
                                # Y2 per-search budget guard (runner/orchestrator operational
                                # knobs; a breach is a recorded game failure, never a learning control)
                                'WOTR_SEARCH_MAX_SECONDS', 'WOTR_SEARCH_MAX_PRIVATE_GB',
                                'WOTR_SEARCH_MAX_ADVANCES', 'WOTR_SEARCH_WATCHDOG_INTERRUPT',
                                'WOTR_SEARCH_BUDGET_EXPORT')
# Mirror of run_sp_served.SEARCH_BUDGET_DEFAULTS (the runner is never imported here).
SEARCH_BUDGET_DEFAULTS = {'WOTR_SEARCH_MAX_SECONDS': '300', 'WOTR_SEARCH_MAX_PRIVATE_GB': '4.0',
                          'WOTR_SEARCH_MAX_ADVANCES': '20', 'WOTR_SEARCH_WATCHDOG_INTERRUPT': '1'}
_SHORTFALL_TOTAL = {'planned': 0, 'completed': 0, 'game_failures': 0, 'batches_short': 0}


def search_budget_policy(env):
    """Loop-log text of the per-search budget the runner will apply (env or defaults)."""
    values = {name: (env.get(name) or '').strip() or default for name, default in SEARCH_BUDGET_DEFAULTS.items()}
    return ('max_seconds=%s max_private_gb=%s max_advances=%s x sims watchdog_interrupt=%s (0 = off; a breach is '
            'recorded as one status=error game with fixture + ledger row, the batch continues)'
            % (values['WOTR_SEARCH_MAX_SECONDS'], values['WOTR_SEARCH_MAX_PRIVATE_GB'],
               values['WOTR_SEARCH_MAX_ADVANCES'], values['WOTR_SEARCH_WATCHDOG_INTERRUPT']))


def min_batch_games():
    raw = os.environ.get(MIN_BATCH_GAMES_ENV)
    if raw is None or not raw.strip():
        return 1
    value = int(raw)
    if value < 1:
        raise ValueError('%s must be a positive integer' % MIN_BATCH_GAMES_ENV)
    return value


def native_search_enabled():
    return os.environ.get('WOTR_NATIVE_SEARCH', '0') == '1'


def _native_search_source_identity():
    """Use the runner's real root-only adapter identity without loading a DLL."""
    from run_sp_served import native_search_source_identity
    return native_search_source_identity()


def _append_ledger(path, row):
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(row) + '\n')


def _record_batch_shortfall(idx, bdir, phase, failure_rows, completed, planned, rc, walled):
    """Loop log + LOOP/failures_ledger.jsonl: one row per failed game (fixture, trace, error,
    decision/turn/side) and one batch_shortfall summary row - the material for later root cause."""
    ledger = os.path.join(LOOP, FAILURES_LEDGER)
    fdir = os.path.join(bdir, 'failures')
    for row in failure_rows:
        detail = row.get('failure') or {}
        fixture = detail.get('fixture_path')
        if fixture:
            fixture = os.path.join(fdir, os.path.basename(str(fixture)))
        entry = {'record_type': 'game_failure', 'batch': idx, 'batch_dir': bdir,
                 'phase_id': phase.get('phase_id'), 'worker_id': row.get('worker_id'),
                 'episode': row.get('episode'), 'error': row.get('error'), 'error_type': row.get('error_type'),
                 'decision_index': detail.get('decision_index'), 'side': detail.get('side'),
                 'turn': detail.get('turn'), 'phase_flags_at_search': detail.get('phase_flags_at_search'),
                 'fixture_path': fixture, 'fixture_sha256': detail.get('fixture_sha256'),
                 'search_failure_error': detail.get('search_failure_error'),
                 'trace_path': os.path.join(fdir, os.path.basename(str(row.get('trace_path') or ''))) if row.get('trace_path') else None,
                 'search_route': row.get('search_route', 'python'), 'native_refusal': row.get('native_refusal'),
                 'rng_episode_identity': row.get('rng_episode_identity'), 'wall_s': row.get('wall_s'),
                 # Y2: per-search budget breach (orchestrator guard / watchdog) when that was the failure
                 'budget': detail.get('budget') or row.get('search_budget'),
                 'traceback_tail': (row.get('traceback') or '')[-1500:], 'recorded_at': time.time()}
        _append_ledger(ledger, entry)
        budget = entry['budget'] or {}
        log_line('GAME FAILURE batch %03d worker %s episode %s (decision %s, turn %s, side %s): %s -> fixture %s '
                 '(ROOT-CAUSE LATER; the batch continues)%s'
                 % (idx, row.get('worker_id'), row.get('episode'), detail.get('decision_index'),
                    detail.get('turn'), detail.get('side'), row.get('error'), fixture,
                    (' [SEARCH BUDGET %s: %s=%s limit %s]' % (budget.get('source'), budget.get('which'),
                                                             budget.get('value'), budget.get('limit')))
                    if budget else ''))
    _SHORTFALL_TOTAL['planned'] += planned
    _SHORTFALL_TOTAL['completed'] += completed
    _SHORTFALL_TOTAL['game_failures'] += len(failure_rows)
    _SHORTFALL_TOTAL['batches_short'] += int(completed < planned)
    summary = {'record_type': 'batch_shortfall', 'batch': idx, 'batch_dir': bdir, 'phase_id': phase.get('phase_id'),
               'planned': planned, 'completed': completed, 'game_failures': len(failure_rows),
               'runner_returncode': rc, 'wall_cap_reached': walled, 'min_batch_games': min_batch_games(),
               'cumulative': dict(_SHORTFALL_TOTAL), 'recorded_at': time.time()}
    _append_ledger(ledger, summary)
    log_line('BATCH %03d SHORT: %d/%d games completed, %d game failure(s) recorded; accepted for training when '
             '>= %s=%d (ledger %s)' % (idx, completed, planned, len(failure_rows), MIN_BATCH_GAMES_ENV,
                                        min_batch_games(), ledger))


def _archive_served_failures(output, bdir, phase, attempts):
    """Keep the runner's fixtures + ledger beside the batch and move failed games' traces out
    of the batch root (extraction/audit glob the root; failed traces are diagnostics, never
    training rows).  Returns the phase's error attempt rows."""
    fdir = os.path.join(bdir, 'failures')
    source_dir = os.path.join(output, 'failures')
    source_ledger = os.path.join(output, 'failures.jsonl')
    if os.path.isdir(source_dir):
        shutil.copytree(source_dir, fdir, dirs_exist_ok=True)
    if os.path.isfile(source_ledger):
        os.makedirs(fdir, exist_ok=True)
        shutil.copyfile(source_ledger, os.path.join(fdir, 'failures.jsonl'))
    rows = []
    if phase is None:
        return rows
    for row in attempts:
        if row.get('phase_id') != phase.get('phase_id') or row.get('status') != 'error':
            continue
        rows.append(row)
        name = os.path.basename(str(row.get('trace_path') or ''))
        source = os.path.join(bdir, name)
        if name.endswith('.trace.gz') and os.path.isfile(source):
            os.makedirs(fdir, exist_ok=True)
            shutil.move(source, os.path.join(fdir, name))
    return rows


def play_batch(idx, games, workers, sims):
    backend = os.environ.get('WOTR_LOOP_SELFPLAY', 'standard_v1')
    if backend not in ('standard_v1', 'served_v1'):
        raise ValueError('Unknown loop self-play backend: ' + backend)
    source_trace_dir, source_games_path = TRACE_DIR, GAMES_JSONL
    served_output = None
    attempts_path = os.path.join(LOOP, 'incoming_b%03d.attempts.jsonl' % idx)
    phase_path = os.path.join(LOOP, 'incoming_b%03d.phase.json' % idx)
    env = dict(os.environ,
               KMP_DUPLICATE_LIB_OK='TRUE', PYTHONHASHSEED='0',
               WOTR_SP_NET_DIR=LOOP,
               WOTR_SP_TRACE_DIR=TRACE_DIR,
               WOTR_SP_GAMES_LOG=GAMES_JSONL,
               WOTR_SP_REPORT=os.path.join(LOOP, 'selfplay_report.md'),
               WOTR_SP_ATTEMPTS_LOG=attempts_path,
               WOTR_SP_PHASE_SUMMARY=phase_path,
               WOTR_SP_SEED0=str(30000 + idx * 1000),
               # profile pins derived from NET_KEYS: workers sample
               # ONLY trained profiles (2-net focus run)
               WOTR_SP_FP_PROFILES=','.join(
                   p for s, p in NET_KEYS if s == 'fp'),
               WOTR_SP_SH_PROFILES=','.join(
                   p for s, p in NET_KEYS if s == 'sh'))
    fz9 = _frozen_cfg()
    if fz9:
        # 1.259: the anchor rides the existing loop->selfplay env
        # channel; the pair was generation-checked at boot
        env['WOTR_SP_FROZEN_P'] = str(fz9[0])
        env['WOTR_SP_FROZEN_FP'] = fz9[1]
        env['WOTR_SP_FROZEN_SH'] = fz9[2]
    args = [PY, '-u', os.path.join(SRC, 'run_td_v33sp.py'),
            '--games', str(games), '--workers', str(workers)]
    if sims:
        args += ['--sims', str(sims)]
    if backend == 'served_v1':
        args, served_output = _served_batch_launch(idx, games, workers, sims, env, fz9)
        source_trace_dir = os.path.join(served_output, 'traces')
        source_games_path = os.path.join(served_output, 'games.jsonl')
        attempts_path = os.path.join(served_output, 'attempts.jsonl')
        phase_path = os.path.join(served_output, 'phase.json')
    if _PRE_READY_ATTEMPT is not None:
        if served_output is None:
            raise RuntimeError("Pre-ready recovery requires served output")
        _PRE_READY_ATTEMPT.launching(idx, served_output, workers, games)
    p = ProcessTree(args, cwd=ROOT, env=env)
    walled = False
    supervisor_cleanup = None
    try:
        rc = p.wait(timeout=None if served_output is not None else BATCH_WALL_SEC)
    except subprocess.TimeoutExpired:
        walled = True
        log_line('BATCH WALL CAP: batch %03d exceeded %d s -- '
                 'killing the selfplay tree and proceeding with the '
                 'traces collected so far (PARTIAL batch)'
                 % (idx, BATCH_WALL_SEC))
        # Every batch owns its subprocess tree. Cleanup must finish before
        # traces can be archived or admitted; a cleanup failure propagates.
        cleanup = p.terminate()
        supervisor_cleanup = cleanup
        rc = cleanup['root_returncode']
        log_line('BATCH CLEANUP: %s' % json.dumps(cleanup, sort_keys=True))
    except BaseException:
        # Interruptions and caller errors must not leave active game workers.
        cleanup = p.terminate()
        if _PRE_READY_ATTEMPT is not None:
            _PRE_READY_ATTEMPT.quiescent(p, cleanup['root_returncode'], cleanup)
        raise
    if _PRE_READY_ATTEMPT is not None:
        _PRE_READY_ATTEMPT.quiescent(p, rc, supervisor_cleanup)
    if served_output is not None:
        return _archive_served_batch(idx, served_output, rc, walled, supervisor_cleanup)
    bdir = os.path.join(LOOP, 'batch_%03d' % idx)
    os.makedirs(bdir, exist_ok=True)
    if os.path.exists(phase_path):
        with open(phase_path, encoding='utf-8') as handle:
            phase = json.load(handle)
        phase.update(runner_returncode=rc, wall_cap_reached=walled)
        with open(os.path.join(bdir, 'phase.json'), 'w', encoding='utf-8') as handle:
            json.dump(phase, handle, indent=2)
        if os.path.exists(attempts_path):
            with open(attempts_path, encoding='utf-8') as source, open(
                    os.path.join(bdir, 'attempts.jsonl'), 'w', encoding='utf-8') as target:
                for line in source:
                    if line.strip() and json.loads(line).get('phase_id') == phase['phase_id']:
                        target.write(line)
    for f in glob.glob(os.path.join(source_trace_dir, '*.trace.gz')):
        shutil.move(f, bdir)
    if os.path.exists(source_games_path):
        shutil.move(source_games_path, os.path.join(bdir, 'games.jsonl'))
        # USER 2026-07-29 ("it got cleared"): keep the dashboard feed
        # CUMULATIVE — re-append this batch's rows with trace_path
        # rewritten to the archived batch location.
        _feed_append(os.path.join(bdir, 'games.jsonl'), bdir)
    if rc != 0 and not walled:
        raise SystemExit('batch %d selfplay exit %d; evidence saved in %s' % (idx, rc, bdir))
    return bdir


FEED_JSONL = os.path.join(LOOP, 'all_games.jsonl')


def _feed_append(batch_jsonl, bdir):
    """Append batch rows (trace paths fixed to the archive) to the
    CUMULATIVE dashboard feed — its own file, never the runner's
    per-batch GAMES_JSONL (those basenames collide with baseline
    games; mixing them would mis-attribute profiles in the TD meta
    lookup)."""
    try:
        with open(FEED_JSONL, 'a', encoding='utf-8') as out, open(batch_jsonl, encoding='utf-8') as source:
            for ln in source:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    tp = str(r.get('trace_path', ''))
                    if tp:
                        r['trace_path'] = os.path.join(
                            bdir, os.path.basename(tp))
                    # USER 2026-07-29: dashboard groups by 'batch' —
                    # stamp the archive name so batches separate
                    r['batch'] = os.path.basename(bdir)
                    out.write(json.dumps(r) + '\n')
                except ValueError:
                    continue
    except OSError as e:
        log_line('feed append failed (non-fatal): %r' % e)


def _log_frozen_telemetry(bdir):
    """1.259 batch telemetry: when the anchor is unconfigured the loop
    says so LOUDLY once per batch (never silent); when configured it
    reports the realized frozen-episode split from the batch's
    games.jsonl (the episode meta carries 'frozen_side').  Returns the
    logged line."""
    fz9 = _frozen_cfg()
    if fz9 is None:
        line = ('FROZEN-OPPONENT ANCHOR DISABLED: no frozen '
                'checkpoint configured (1.259; intended anchor = '
                'the v42 BC baseline)')
        log_line(line)
        return line
    n = {'fp': 0, 'sh': 0}
    m = 0
    gj = os.path.join(bdir, 'games.jsonl')
    if os.path.exists(gj):
        for ln in open(gj, encoding='utf-8'):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            m += 1
            fs = r.get('frozen_side')
            if fs in n:
                n[fs] += 1
    line = ('FROZEN-OPPONENT ANCHOR (1.259, p=%.2f): '
            'frozen_episodes=%d/%d side_split=fp:%d sh:%d'
            % (fz9[0], n['fp'] + n['sh'], m, n['fp'], n['sh']))
    log_line(line)
    return line


def audit_batch(bdir):
    r = subprocess.run([PY, os.path.join(SRC, 'trace_audit.py'), bdir],
                       cwd=ROOT, env=dict(os.environ,
                                          KMP_DUPLICATE_LIB_OK='TRUE'),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', 'replace')
    ok = r.returncode == 0 and 'AUDIT PASS' in out
    if not ok:
        log_line('TRACE AUDIT (telemetry) FAILED for %s — violations '
                 'logged, loop continues (v2)' % bdir)
        print(out[-2000:])
    return ok


def _replay_reward_stream(tr, profs, detail=None, *, for_learning=True):
    """Read recorded inputs; fresh learning additionally requires current schemas.

    Read-only attribution can inspect an older recorded predecision stream.
    That opt-out never regenerates or admits its vectors to training.
    """
    import numpy as np
    from trace_decisions import (is_learning_decision, trace_ai_policy_contract,
                                 trace_observation_contract, trace_action_encoding_contract)
    from reward_contract import uses_predecision, PREDECISION, LEGACY, trace_contract
    contract = trace_contract(tr)
    if for_learning:
        trace_ai_policy_contract(tr)
        trace_observation_contract(tr)
        trace_action_encoding_contract(tr)
    if contract == LEGACY:
        from reward_contract import current_contract
        if current_contract() != LEGACY:
            raise ValueError('Legacy replay requires WOTR_REWARD_CONTRACT=legacy_profile_v1')
        return _replay_reward_stream_legacy(tr, profs, detail)
    if for_learning:
        trace_action_encoding_contract(tr, required=(uses_predecision() or contract in PREDECISION))
    if contract not in PREDECISION:
        raise ValueError('Unsupported trace reward contract: %r' % contract)
    per = {tag: ([], [], []) for tag in ('fp', 'sh')}
    for index, row in enumerate(tr['steps']):
        if not is_learning_decision(row):
            continue
        context = row.get('decision_context') or {}
        if context.get('contract') != contract:
            raise ValueError('Decision %d lacks matching pre-decision context' % index)
        enc = np.asarray(context['encoding'], dtype=np.float32)
        if enc.ndim != 1 or not np.isfinite(enc).all():
            raise ValueError('Decision %d has invalid encoding' % index)
        tag = row['side'].lower()
        encs, rewards, phis = per[tag]
        encs.append(enc.copy())
        rewards.append(0.0)
        phis.append(0.0)
        if detail is not None:
            detail.append({'step_idx': index, 'tag': tag, 'k': len(encs)-1,
                           'paid': 0.0, 'reward_contract': contract})
    return per


def _replay_reward_stream_legacy(tr, profs, detail=None):
    """Per-side (encodings, rewards, phis) with interval semantics +
    military bank.  THE canonical reward/phi stream: the v2 trainer
    (extract_batch_streams) consumes it and td_attribution renders it
    (1.224f refactor — the
    view used to hand-mirror this logic and silently drifted; a single
    implementation makes that bug class impossible).  `detail`: pass a
    list and one dict per emitted sample is appended
    ({'step_idx','tag','k','paid'}) so a consumer can map samples back
    to trace rows without re-deriving any semantics."""
    import ai
    from trace_decisions import forced_policy_reason
    from game_state import Side
    per = {'fp': ([], [], []), 'sh': ([], [], [])}
    last_stats = {'fp': None, 'sh': None}
    mil_bank = {'fp': 0.0, 'sh': 0.0}
    prev_state = None
    for si, s in enumerate(tr['steps']):
        st = s.get('state')
        sd = s.get('side')
        at = str(s.get('action_type'))
        if st is None:
            continue
        if forced_policy_reason(s) is not None:
            # Preserve its resulting state for interval/tail accounting, but
            # charge no decision cost and do not move a learned anchor.
            prev_state = st
            continue
        if sd in ('FP', 'SH') and at != 'None':
            tag = sd.lower()
            side = Side.FREE_PEOPLES if tag == 'fp' else Side.SHADOW
            pre_int = last_stats[tag] or ai._reward_stats(
                prev_state if prev_state is not None else st)
            die_face = None
            try:
                di = s.get('die_idx')
                if (isinstance(di, int) and di >= 0
                        and prev_state is not None):
                    _c = prev_state.dice[side][di].current
                    die_face = _c.value if _c is not None else None
            except Exception:
                die_face = None
            # Encode FIRST: a failed encode drops the SAMPLE but must
            # not consume the interval or the military bank - leaving
            # last_stats untouched folds this interval into the next
            # emitted sample, so no reward mass is silently lost.
            try:
                _enc9 = ai.encode_state(st, side)
            except Exception as _enc_err9:
                log_line('[reward-stream] encode failed step %d %s: %r'
                         % (si, tag, _enc_err9))
                prev_state = st
                continue
            axis_r, mil_r = ai._compute_step_reward_split(
                pre_int, st, side)
            mil_bank[tag] += mil_r
            reward = axis_r
            paid = 0.0
            if ai._is_military_action(at, s.get('params'), die_face) \
                    and abs(mil_bank[tag]) > 1e-9:
                paid = mil_bank[tag]
                reward += paid
                mil_bank[tag] = 0.0
            last_stats[tag] = ai._reward_stats(st)
            # RESIDUAL VALUE LEARNING (USER 2026-07-30): the net
            # is trained on G - phi and evaluated as net + phi
            # (_mcts_eval_state) — the hand-written mission
            # scaffold enters the search exactly once, and the
            # net can cancel it with a negative residual wherever
            # experience says the scaffold is wrong.  phi is the
            # SIDE-SIGNED leaf bonus, same sign convention as the
            # leaf eval.
            _sgn9 = 1.0 if tag == 'fp' else -1.0
            _phi9 = _sgn9 * float(ai._profile_shaping(st))
            per[tag][0].append(_enc9)
            per[tag][1].append(float(reward))
            per[tag][2].append(_phi9)
            if detail is not None:
                detail.append({'step_idx': si, 'tag': tag,
                               'k': len(per[tag][0]) - 1,
                               'paid': float(paid)})
        prev_state = st
    # The last opponent action (or an annotation/turn-end effect) can
    # change the score after this side has stopped acting.  Settle that
    # previously unobserved interval before flushing the military bank.
    # This is not a new decision, so do not charge another urgency cost.
    if prev_state is not None:
        for tag in ('fp', 'sh'):
            if last_stats[tag] is None or not per[tag][1]:
                continue
            side = Side.FREE_PEOPLES if tag == 'fp' else Side.SHADOW
            tail_axis, tail_mil = ai._compute_step_reward_split(
                last_stats[tag], prev_state, side,
                include_decision_cost=False)
            per[tag][1][-1] += tail_axis
            mil_bank[tag] += tail_mil
            if detail is not None:
                for d in reversed(detail):
                    if d['tag'] == tag:
                        d['tail_axis'] = float(tail_axis)
                        d['tail_military'] = float(tail_mil)
                        break

    # 1.143b TERMINAL FLUSH (USER 2026-08-02): residual mil_bank at
    # stream end (a stronghold captured after the side's last military
    # action) is added to that side's LAST emitted sample's reward; in
    # detail mode the flush shows up in that sample's 'paid' (the
    # view's bank label) plus an explicit 'flushed' amount.  Twin of
    # the episode loop's end-of-game flush in ai._run_single_episode.
    for tag in ('fp', 'sh'):
        resid = mil_bank[tag]
        if abs(resid) > 1e-9 and per[tag][1]:
            per[tag][1][-1] += resid
            mil_bank[tag] = 0.0
            if detail is not None:
                for d in reversed(detail):
                    if d['tag'] == tag:
                        d['paid'] = float(d.get('paid', 0.0) + resid)
                        d['flushed'] = float(resid)
                        break
    return per


def _collect_search_samples(tr, tag):
    """1.226b: (state_encoding, search_stats) pairs for `tag`'s stored
    per-decision root stats.  The encoding is computed on the
    PRE-decision state (the previous snapshot — trace rows store the
    post-state) with the transient phase flag restored for declare/eye
    rows, since the search encoded the flag set.  Rows with K < 2
    candidates teach nothing and are skipped."""
    import ai
    from trace_decisions import (forced_policy_reason, trace_observation_contract,
                                 trace_ai_policy_contract)
    trace_observation_contract(tr)
    trace_ai_policy_contract(tr)
    from trace_decisions import trace_action_encoding_contract
    trace_action_encoding_contract(tr, required=(uses_predecision() or
        (tr.get('meta') or {}).get('reward_contract') in PREDECISION))
    import delta_trainer as _dt9
    from game_state import Side
    out = []
    # 1.259: ONE frozen predicate (delta_trainer.frozen_side_of) for
    # both collectors; drops count into the shared DRIVER_STATS so the
    # per-batch DRIVER_STATS: line reports them under either operator
    fs9 = _dt9.frozen_side_of(tr)
    prev = None
    for s in tr['steps']:
        st = s.get('state')
        if st is None:
            continue
        if forced_policy_reason(s) is not None:
            if s.get('side', '').lower() == tag:
                _dt9.DRIVER_STATS['forced_dropped'] = _dt9.DRIVER_STATS.get('forced_dropped', 0) + 1
            prev = st
            continue
        ss = s.get('search_stats')
        sd = s.get('side')
        if (ss is not None and sd in ('FP', 'SH')
                and sd.lower() == tag and prev is not None
                and len(ss.get('p', ())) >= 2):
            if fs9 == tag or str(s.get('frozen_side') or '') == tag:
                _dt9.DRIVER_STATS['frozen_dropped'] = \
                    _dt9.DRIVER_STATS.get('frozen_dropped', 0) + 1
                prev = st
                continue
            side = Side.FREE_PEOPLES if tag == 'fp' else Side.SHADOW
            at = str(s.get('action_type'))
            # SPB 1b2 (audit 2026-08-29): 'declare_decline' rows were
            # searched under _declare_phase too (SPEP-8 absence
            # sample) -- inline twin of delta_trainer._flag_for; the
            # two must never diverge (pinned in test_audit0829_SPC2).
            flag = ('_declare_phase'
                    if at in ('fellowship_declare', 'declare_decline')
                    else '_eye_allocation_phase'
                    if at == 'eye_allocate' else None)
            enc = None
            try:
                if flag:
                    setattr(prev, flag, True)
                context = s.get('decision_context') or {}
                enc = (context['encoding'] if 'encoding' in context
                       else ai.encode_state(prev, side))
            except Exception:
                enc = None
            finally:
                if flag:
                    try:
                        setattr(prev, flag, False)
                    except Exception:
                        pass
            if enc is not None:
                out.append((enc, ss))
        prev = st
    return out


def _arena_gpu_environment():
    env = dict(os.environ)
    if _LOOP_INSERTED_CPU_MASK and env.get('CUDA_VISIBLE_DEVICES') == '-1':
        env.pop('CUDA_VISIBLE_DEVICES')
    return env


def run_arena_one(side, prof, idx=None):
    """Score ONE net (v2 scoreboard): live vs its own baseline, identical baseline
    partner on the other side in both pairings.  Every arena game is
    appended to the GUI feed (USER 2026-07-30) with batch label
    'arena_<idx>_<net>' and orientation fields."""
    ps, pp = ARENA_PARTNER[side]
    partner = _nm('baseline', ps, pp)
    if side == 'fp':
        a = ['--a-fp', _nm('live', side, prof), '--a-sh', partner]
        b = ['--b-fp', _nm('baseline', side, prof), '--b-sh', partner]
    else:
        a = ['--a-fp', partner, '--a-sh', _nm('live', side, prof)]
        b = ['--b-fp', partner, '--b-sh', _nm('baseline', side, prof)]
    name = '%s_%s' % (side, prof)
    gj = os.path.join(LOOP, 'arena_games_tmp.jsonl')
    try:
        os.remove(gj)
    except OSError:
        pass
    args = [PY, '-u', os.path.join(SRC, 'arena_scale.py'),
            '--a-arch', 'base', '--b-arch', 'base',
            '--workers', str(ARENA_WORKERS),
            '--games-per-worker', str(ARENA_GAMES_PER_WORKER),
            '--sims', str(ARENA_SIMS),
            '--games-jsonl', gj] + a + b
    # mission-regime pins (1.224c): the gated net plays under ITS
    # profile, the fixed partner under its own
    fp_prof = prof if side == 'fp' else pp
    sh_prof = prof if side == 'sh' else pp
    # HUNT-B F8 (WIRING 2026-08-11): the arena is the SCOREBOARD (v2:
    # no promotion/revert) and must play the SAME regime the sp
    # workers hard-pin — without these pins a launcher override
    # (e.g. WOTR_TREE_REUSE=0 A/B) silently split the arena's visit
    # dynamics from every training game.  Identical triple: NET_ARCH +
    # TREE_REUSE hard-pinned, TURN_HORIZON setdefault (mirrors
    # run_td_v33sp / launch_smoke_gate exactly).
    env9 = dict(_arena_gpu_environment(), KMP_DUPLICATE_LIB_OK='TRUE',
                PYTHONHASHSEED='0',
                WOTR_NET_ARCH='base',
                WOTR_TREE_REUSE='1',
                WOTR_ARENA_FP_PROFILE=str(fp_prof),
                WOTR_ARENA_SH_PROFILE=str(sh_prof))
    env9.setdefault('WOTR_TURN_HORIZON', '1')
    r = subprocess.run(args, cwd=ROOT, env=env9,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', 'replace')
    score = None
    n_played = None
    for ln in out.splitlines():
        if 'A score =' in ln:
            try:
                score = float(ln.split('A score =')[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif '=== ARENA' in ln and 'games ===' in ln:
            # '=== ARENA a(A) vs b(B): N games ===' (arena_scale)
            try:
                n_played = int(ln.rsplit(':', 1)[1].split()[0])
            except (ValueError, IndexError):
                pass
    if score is not None:
        # SPLP-6 (audit 2026-08-29): the 3 h arena wall cap scores
        # PARTIAL games -- the scoreboard row must say how many games
        # the score stands on, never pass a partial off as full.
        expected = ARENA_WORKERS * ARENA_GAMES_PER_WORKER
        log_line('arena %s: n_played=%s of %d%s'
                 % (name,
                    n_played if n_played is not None else '?',
                    expected,
                    '' if n_played == expected else
                    ' -- PARTIAL/short score (wall cap, errors or '
                    'unparsed count)'))
    if score is None:
        # a crashed arena must be distinguishable from a format change
        log_line('arena %s: no score parsed (exit %d); output tail: %s'
                 % (name, r.returncode, out[-400:].replace('\n', ' | ')))
    # feed the arena games to the GUI: batch = arena_<idx>_<net>;
    # arena_live_in_game says whether the LIVE net (vs its baseline
    # copy) played this game — pairing A always hosts the live net,
    # and each game takes fp from one pairing and sh from the other
    try:
        if os.path.exists(gj):
            with open(FEED_JSONL, 'a', encoding='utf-8') as ff:
                for ln in open(gj, encoding='utf-8'):
                    try:
                        rec = json.loads(ln)
                    except ValueError:
                        continue
                    aif = bool(rec.get('a_is_fp'))
                    rec['batch'] = 'arena_%s_%s' % (
                        '%03d' % idx if idx is not None else 'x', name)
                    rec['profiles'] = {'fp': str(fp_prof),
                                       'sh': str(sh_prof)}
                    rec['arena_net'] = name
                    rec['arena_live_in_game'] = (
                        (side == 'fp' and aif)
                        or (side == 'sh' and not aif))
                    if score is not None:
                        rec['arena_score'] = score
                    ff.write(json.dumps(rec) + '\n')
            os.remove(gj)
    except OSError as e:
        log_line('arena feed append failed: %r' % (e,))
    return score


def _pid_lock():
    """Single-instance guard (2026-07-29: overlapping restarts made
    the night's timeline unreadable) — refuse to start if another
    loop process is alive."""
    lock = os.path.join(LOOP, 'loop.pid')
    if os.path.exists(lock):
        try:
            other = int(open(lock).read().strip())
        except (ValueError, OSError):
            other = None
        if other:
            try:
                import psutil                     # optional
                alive = psutil.pid_exists(other)
            except ImportError:
                if sys.platform == 'win32':
                    # os.kill(pid, 0) TERMINATES the process on
                    # Windows (TerminateProcess) — never probe with
                    # it; require psutil loudly instead.
                    raise SystemExit(
                        'psutil required for the single-instance '
                        'guard on Windows (os.kill would TERMINATE '
                        'the running loop)')
                try:
                    os.kill(other, 0)
                    alive = True
                except OSError:
                    alive = False
            if alive and other != os.getpid():
                raise SystemExit(
                    'another loop is running (pid %d) — refusing '
                    'to double-run' % other)
    os.makedirs(LOOP, exist_ok=True)
    with open(lock, 'w') as f:
        f.write(str(os.getpid()))


def main():
    cfg = _apply_loop_config()
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-games', type=int,
                    default=int(cfg.get('total_games', 2000)))
    ap.add_argument('--workers', type=int,
                    default=int(cfg.get('workers', 12)))
    ap.add_argument('--sims', type=int,
                    default=(int(cfg['sims'])
                             if cfg.get('sims') else None))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if cfg:
        log_line('loop_config.json applied: %s' % json.dumps(cfg))
    # v2 is the ONLY path; WOTR_LOOP_V2 is not consulted - a plain
    # launch runs the shipped regime.
    if a.smoke:
        return _smoke_v2(a)
    return main_v2(a)


def _smoke_v2(a):
    """Machinery + isolation smoke on the SHIPPED v2 operator stack
    (ship-discipline: validate the shipped branch, never a sibling):
    extract + train on LOOP\baseline_games, verify per-net isolation,
    then restore live weights from a pre-smoke snapshot (last_good is
    not advanced under v2 and may be the stale seed)."""
    _resolve_ops()
    setup()
    preflight_generation()
    _pid_lock()          # never run concurrently with a live loop
    integrity_check()
    bg = os.path.join(LOOP, 'baseline_games')
    games = glob.glob(os.path.join(bg, '*.trace.gz'))
    if not games:
        raise SystemExit(
            'SMOKE: no games in %s - generate a baseline batch first '
            '(a smoke with nothing to train on must never pass)' % bg)
    snaps = {}
    for key in NET_KEYS:
        live = _nm('live', *key)
        snaps[key] = live + '.smokepre'
        shutil.copy2(live, snaps[key])
    try:
        pre = {k: sha256(_nm('live', *k)) for k in NET_KEYS}
        data = extract_batch_streams(bg)
        losses = train_nets_v2(data, {('%s_%s' % k): LR
                                      for k in NET_KEYS})
        post = {k: sha256(_nm('live', *k)) for k in NET_KEYS}
        changed = [k for k in pre if pre[k] != post[k]]
        log_line('SMOKE: %d games, losses %s' % (len(games), losses))
        log_line('SMOKE: %d/%d live nets changed independently: %s'
                 % (len(changed), len(NET_KEYS),
                    ['%s_%s' % k for k in changed]))
        # the updated nets must all DIFFER from one another
        log_line('SMOKE: distinct live hashes: %d/%d'
                 % (len(set(post.values())), len(NET_KEYS)))
        integrity_check()
    finally:
        for k, sp in snaps.items():
            shutil.copy2(sp, _nm('live', *k))
            try:
                os.remove(sp)
            except OSError:
                pass
    log_line('SMOKE done (live restored from pre-smoke snapshot)')


# ===================================================================
# LOOP V2 - the ONLY path.
#   R1 ReplayBuffer: last WOTR_BUFFER_K batches per net.
#   Census / probe / audit are TELEMETRY ONLY and the arena is a pure
#   SCOREBOARD (USER orders 2026-08-11: gates stripped): every batch
#   trains, every update is kept, nothing reverts, halts, or blocks.
# ===================================================================

BUFFER_K = int(os.environ.get('WOTR_BUFFER_K', '6'))
DELTA_SAMPLE_CAP = int(os.environ.get('WOTR_DELTA_CAP', '8000'))


def _probeset_path():
    return os.environ.get('WOTR_PROBESET') or os.path.join(
        LOOP, 'probeset_v1.pkl.gz')


def _net_names():
    return ['%s_%s' % k for k in NET_KEYS]


def _side_key(side):
    return next(k for k in NET_KEYS if k[0] == side)


# Delta-sample collector hook: v1 default is _collect_search_samples,
# which pairs each turn's FIRST decision with the pre-turn-boundary
# snapshot (9.2% skew-rows, SEARCH comms\06 finding 1.3).  The adopted
# package points this at the full-support prep collector, which
# regenerates menus and drops non-matching rows at ingestion.
DELTA_COLLECTOR = _collect_search_samples


def _join_attempt_records(trace_records, attempts):
    """Count each played game once, even if its trace is missing or partial."""
    import ntpath
    import re
    def key(record):
        name = ntpath.basename(str(record.get('trace_path') or ''))
        stem = name.removesuffix('.trace.gz')
        return re.sub(r'_(?:(?:fp|sh)_(?:military|ring_destroyed|ring_corruption)|timeout_timeout)$', '', stem)
    result = [dict(row) for row in trace_records]
    known = {key(row): row for row in result if key(row)}
    for attempt in attempts:
        match = known.get(key(attempt)) if key(attempt) else None
        if match is not None:
            # Keep trace/admission facts, but preserve the failed execution
            # even when a diagnostic partial trace also exists.
            if attempt.get('status') == 'error':
                match.update(status='error', error=attempt.get('error'))
            continue
        record = dict(attempt)
        if record.get('status') != 'error':
            record['admission'] = {'accepted': False, 'valid': False,
                                   'errors': [{'code': 'trace_missing'}]}
        result.append(record)
    return result


def extract_batch_streams(bdir):
    """Admit intact games individually, then build versioned training data.

    Strength and style metrics do not decide admission. Source execution
    errors, malformed decisions and incompatible contracts do.
    """
    import numpy as np
    import ai
    import game_trace
    from trace_admission import validate_trace
    from reward_contract import current_contract, trace_contract, uses_predecision, terminal
    from trace_decisions import current_learning_context
    from selector_contract import training_identity
    contract = current_contract()
    context_contracts = dict(current_learning_context(), **training_identity())
    admission = {'reward_contract': contract, 'accepted': 0, 'rejected': 0,
                 'traces': [], **context_contracts}
    metric_records = []
    meta = {}
    gj = os.path.join(bdir, 'games.jsonl')
    if os.path.exists(gj):
        for ln in open(gj, encoding='utf-8'):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            meta[os.path.basename(str(r.get('trace_path', '')))] = r
    out = {k: {'value': [], 'delta': [], 'reward_contract': contract,
               **context_contracts}
           for k in NET_KEYS}
    for tp in sorted(glob.glob(os.path.join(bdir, '*.trace.gz'))):
        base = os.path.basename(tp)
        rec = meta.get(base, {})
        tr = None
        try:
            tr = game_trace.load_trace(tp)
            report = validate_trace(
                tr, decision_contract=contract if uses_predecision(contract) else None,
                **context_contracts)
        except Exception as exc:
            report = {'accepted': False, 'errors': [
                {'code': 'trace_unreadable', 'detail': repr(exc)}]}
        report['file'] = base
        admission['traces'].append(report)
        observed = dict(rec)
        observed.update((tr or {}).get('meta') or {})
        observed['admission'] = report
        observed['trace_path'] = tp
        metric_records.append(observed)
        if not report['accepted']:
            admission['rejected'] += 1
            continue
        tm = tr.get('meta') or {}
        profs = tm.get('profiles') or rec.get('profiles') or {}
        winner = tm.get('winner')
        victory = str(tm.get('victory_type') or '')
        reason = None
        if trace_contract(tr) != contract:
            reason = 'reward_contract_mismatch'
        elif not (profs.get('fp') and profs.get('sh')):
            reason = 'profiles_missing'
        elif any(rec.get(k) is not None and rec[k] != tm.get(k)
                 for k in ('winner', 'victory_type')):
            reason = 'result_metadata_mismatch'
        if reason:
            report['accepted'] = False
            report.setdefault('errors', []).append({'code': reason})
            admission['rejected'] += 1
            continue
        admission['accepted'] += 1
        old_profiles = dict(ai.PLAYER_PROFILES)
        ai.PLAYER_PROFILES['fp'] = profs.get('fp')
        ai.PLAYER_PROFILES['sh'] = profs.get('sh')
        try:
            per = _replay_reward_stream(tr, profs)
            # 1.259: the frozen side's own decisions are excluded from
            # value targets (its value sequence never enters training);
            # the LIVE side's rows from the same episode still train.
            # The collector call still runs for the frozen tag so its
            # frozen_dropped counter proves the policy rows were
            # dropped, not lost.
            import delta_trainer as _dt9
            fs9 = _dt9.frozen_side_of(tr)
            final_state = None
            for s9 in reversed(tr['steps']):
                if s9.get('state') is not None:
                    final_state = s9['state']
                    break
            for tag in ('fp', 'sh'):
                encs, rews, phis = per[tag]
                key = (tag, str(profs.get(tag)))
                if key in out and len(encs) >= (1 if uses_predecision(contract) else 3):
                    if tag != fs9:
                        term = (terminal(winner, tag, profile=profs.get(tag),
                                         victory_type=victory, contract=contract)
                                if uses_predecision(contract)
                                else float(ai._terminal_for(
                                    tag, profs.get(tag), victory,
                                    winner == tag, final_state)))
                        out[key]['value'].append(
                            (np.asarray(encs, dtype=np.float32),
                             np.asarray(rews, dtype=np.float32),
                             np.asarray(phis, dtype=np.float32),
                             term))
                    out[key]['delta'].extend(
                        (_dt9.collect_for_driver if uses_predecision(contract)
                         else DELTA_COLLECTOR)(tr, tag))
        finally:
            ai.PLAYER_PROFILES.clear()
            ai.PLAYER_PROFILES.update(old_profiles)
    path = os.path.join(bdir, 'training_admission.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as handle:
        json.dump(admission, handle, indent=2)
    os.replace(path + '.tmp', path)
    # Errors have no finished trace. Keep them in attempted-game accounting
    # without turning them into invented wins/losses or training examples.
    attempts_path = os.path.join(bdir, 'attempts.jsonl')
    attempts = []
    if os.path.exists(attempts_path):
        with open(attempts_path, encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    attempt = json.loads(line)
                except ValueError as exc:
                    raise ValueError('Malformed attempt record in %s' % attempts_path) from exc
                attempts.append(attempt)
    metric_records = _join_attempt_records(metric_records, attempts)
    from profile_metrics import summarize_profile_results
    metrics = {tag: summarize_profile_results(metric_records, side=tag)
               for tag in ('fp', 'sh')}
    phase_path = os.path.join(bdir, 'phase.json')
    if os.path.exists(phase_path):
        with open(phase_path, encoding='utf-8') as handle:
            metrics['coverage'] = json.load(handle)
    else:
        metrics['coverage'] = {'status': 'unknown',
                               'reason': 'No worker attempt coverage report recorded'}
    metrics_path = os.path.join(bdir, 'profile_metrics.json')
    with open(metrics_path + '.tmp', 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    os.replace(metrics_path + '.tmp', metrics_path)
    return out


class ReplayBuffer(object):
    """Per-net FIFO of admitted, contract-compatible training batches."""

    def __init__(self, root, net_keys, k=None):
        from reward_contract import current_contract
        from trace_decisions import current_learning_context
        from selector_contract import training_identity
        self.root = root
        self.reward_contract = current_contract()
        self.context_contracts = dict(current_learning_context(), **training_identity())
        self.k = k or BUFFER_K
        self.net_keys = list(net_keys)
        os.makedirs(root, exist_ok=True)

    def _dir(self, key):
        d = os.path.join(self.root, '%s_%s' % key)
        os.makedirs(d, exist_ok=True)
        return d

    def add(self, idx, streams):
        import gzip as _gz
        import pickle as _pk
        from reward_contract import LEGACY
        for key in self.net_keys:
            if streams[key].get('reward_contract', LEGACY) != self.reward_contract:
                raise ValueError('Replay buffer reward contract mismatch; use a fresh loop directory')
            for field, expected in self.context_contracts.items():
                if streams[key].get(field) != expected:
                    raise ValueError('Replay buffer %s contract mismatch; use a fresh loop directory' % field)
            _validate_selector_stream(streams[key])
        streams = _buffer_pack(streams)
        for key in self.net_keys:
            path = os.path.join(self._dir(key), 'b%03d.pkl.gz' % idx)
            # SPLP-4 (audit 2026-08-29): tmp + os.replace -- the ONE
            # non-atomic persist in the loop; a kill mid-dump left a
            # truncated member that killed every later boot.
            tmp = path + '.tmp'
            with _gz.open(tmp, 'wb') as f:
                _pk.dump(streams[key], f, protocol=4)
            os.replace(tmp, path)
        self.prune()

    def prune(self):
        for key in self.net_keys:
            fs = sorted(glob.glob(os.path.join(self._dir(key),
                                               'b*.pkl.gz')))
            for p in fs[:-self.k]:
                os.remove(p)

    def batches(self, key):
        return sorted(glob.glob(os.path.join(self._dir(key),
                                             'b*.pkl.gz')))

    def load(self, key):
        import gzip as _gz
        import pickle as _pk
        import zlib as _zl
        out = {'value': [], 'delta': [], 'reward_contract': self.reward_contract,
               **self.context_contracts}
        for p in self.batches(key):
            try:
                with _gz.open(p, 'rb') as f:
                    d = _pk.load(f)
            except (OSError, EOFError, _zl.error,
                    _pk.UnpicklingError) as e:
                # SPLP-4 (audit 2026-08-29): an unreadable member
                # (truncated by a crash predating the atomic add)
                # is quarantined LOUDLY -- it must never be a
                # die-at-every-boot poison pill.  (.corrupt falls
                # outside the b*.pkl.gz glob, so batches()/prune()
                # never see it again.)
                q = p + '.corrupt'
                try:
                    os.replace(p, q)
                    where = os.path.basename(q)
                except OSError:
                    where = 'LEFT IN PLACE (rename failed)'
                log_line('BUFFER CORRUPT: %s unreadable (%r) -- '
                         'quarantined: %s; training continues '
                         'without that batch'
                         % (os.path.basename(p), e, where))
                continue
            from reward_contract import LEGACY
            if d.get('reward_contract', LEGACY) != self.reward_contract:
                raise ValueError('Replay buffer reward contract mismatch in %s; use a fresh loop directory' % p)
            for field, expected in self.context_contracts.items():
                if d.get(field) != expected:
                    raise ValueError('Replay buffer %s contract mismatch in %s; use a fresh loop directory' % (field, p))
            _validate_selector_stream(d)
            out['value'].extend(d['value'])
            out['delta'].extend(d['delta'])
        return out


def _value_update_op(net, seqs, lr):
    """Value-head operator on explicit sequences.  Body = the CURRENT
    shipped operator (per-game Adam steps, EPOCHS epochs, trunk eval /
    head train); the driver contract (net, seqs, lr) -> loss is what
    VALUE's adopted operator plugs into."""
    import numpy as np
    import torch
    with torch.no_grad():
        targets = []
        for enc, rews, phis, term in seqs:
            v = net(torch.from_numpy(enc)).numpy().reshape(-1) + phis
            g = np.zeros(len(enc), dtype=np.float32)
            g[-1] = rews[-1] + term
            for t in range(len(enc) - 2, -1, -1):
                g[t] = rews[t] + (1 - LAM) * v[t + 1] + LAM * g[t + 1]
            targets.append(np.clip(g - phis, -2.0, 2.0))
    for p in net.parameters():
        p.requires_grad = False
    head = [p for k, p in net.named_parameters()
            if k.startswith('value_head')]
    for p in head:
        p.requires_grad = True
    opt = torch.optim.Adam(head, lr=lr)
    net.eval()
    net.value_head.train()
    tot = cnt = 0.0
    for _ in range(EPOCHS):
        for (enc, _r, _p, _t), g in zip(seqs, targets):
            pred = net(torch.from_numpy(enc)).reshape(-1)
            loss = torch.nn.functional.mse_loss(
                pred, torch.from_numpy(g))
            opt.zero_grad()
            loss.backward()
            opt.step()
            from trace_decisions import record_warm_start_update
            record_warm_start_update(net, 'value')
            tot += float(loss.item())
            cnt += 1
    net.eval()
    return round(tot / max(1.0, cnt), 4)


def _delta_update_op(net, samples, lr):
    """Policy-delta operator on explicit samples.  Body = the CURRENT
    shipped operator (per-sample CE toward select_root_action's pi
    target, eligible-arm masking); SEARCH's adopted target/operator
    plugs into the same (net, samples, lr) -> losses contract."""
    import numpy as np
    import torch
    import ai
    if uses_predecision():
        raise ValueError('Corrected selector targets require delta_trainer, not legacy delta operator')
    for p in net.parameters():
        p.requires_grad = False
    dpars = [p for k2, p in net.named_parameters()
             if k2.startswith('policy_delta')]
    for p in dpars:
        p.requires_grad = True
    opt_d = torch.optim.Adam(dpars, lr=lr, weight_decay=DELTA_WD)
    net.eval()
    d_losses = []
    err_d = 0
    skip_d = 0
    first_err = None
    trained_total = 0
    for _ in range(EPOCHS):
        tot_d = cnt_d = 0.0
        for enc9, ss9 in samples:
            try:
                x9 = torch.from_numpy(np.asarray(
                    [enc9], dtype=np.float32))
                with torch.no_grad():
                    h9, ne9 = net._process_input(x9)
                ae9 = torch.from_numpy(np.asarray(
                    ss9['ae'], dtype=np.float32))
                ri9 = torch.from_numpy(np.asarray(
                    ss9['ri'], dtype=np.int64))
                logits9 = net.action_scores(
                    h9, ne9, ae9, ri9, with_delta=True)
                cs9 = {'p': np.asarray(ss9['p'], dtype=np.float64),
                       'q': np.asarray(ss9['q'], dtype=np.float64),
                       'n': np.asarray(ss9['n'], dtype=np.float64)}
                if 'tf' in ss9:
                    cs9['tf'] = np.asarray(ss9['tf'],
                                           dtype=np.float64)
                _i9, pi9, dg9 = ai.select_root_action(cs9, 0.0)
                el9 = dg9['elig']
                idx9 = np.nonzero(el9)[0]
                if len(idx9) < 2:
                    skip_d += 1
                    continue
                tgt9 = torch.from_numpy(pi9[idx9].astype(np.float32))
                loss9 = -(tgt9 * torch.log_softmax(
                    logits9[torch.from_numpy(idx9)], dim=0)).sum()
                opt_d.zero_grad()
                loss9.backward()
                opt_d.step()
                tot_d += float(loss9.item())
                cnt_d += 1
            except Exception as e9:
                # SPTR-6 (audit 2026-08-29): count + keep the first
                # error -- the old bare continue could train 0
                # samples and still report a 0.0 loss as success.
                err_d += 1
                if first_err is None:
                    first_err = '%s: %s' % (type(e9).__name__, e9)
                continue
        trained_total += int(cnt_d)
        d_losses.append(round(tot_d / max(1.0, cnt_d), 4))
    net.eval()
    if samples and trained_total == 0:
        raise RuntimeError(
            'v1 delta op: 0 of %d samples trained (errors=%d '
            'first=%s skipped_elig=%d) -- refusing to report a 0.0 '
            'loss as success (SPTR-6; contract mirrors '
            'delta_trainer.train_policy_delta)'
            % (len(samples), err_d, first_err, skip_d))
    if err_d:
        log_line('v1 delta op: %d/%d sample-errors swallowed '
                 '(first=%s), %d trained' % (err_d,
                                             len(samples) * EPOCHS,
                                             first_err,
                                             trained_total))
    return d_losses


def _resolve_ops():
    """Adopted-operator resolution.  The v2 operators
    (delta_trainer / value_trainer_v2) are the DEFAULT (lead order
    2026-08-29, SPTR-5/SPLP-3: an UNSET env silently ran the run-7
    legacy operators -- the silent-fallback doctrine forbids that;
    the default-OFF rollback design is the same class that retired
    v33_loop.ps1).  The legacy v1 ops stay reachable ONLY via an
    explicit WOTR_OPS_V2=0, and EITHER path prints one loud line
    naming the operators chosen.  A missing adopted op still fails
    the import loudly -- never a silent fall back to the
    proven-broken one."""
    global DELTA_COLLECTOR, VALUE_UPDATE_OP, DELTA_UPDATE_OP, _OPS_V2_ACTIVE
    from optimizer_continuity import mode as optimizer_mode, VERSION as persistent_optimizer_mode
    if os.environ.get('WOTR_OPS_V2') == '0' and optimizer_mode() == persistent_optimizer_mode:
        raise ValueError('Persistent optimizer state requires the v2 training operators')
    if (os.environ.get('WOTR_OPS_V2') == '0'
            and requested_training_device() != 'cpu'):
        raise ValueError('Legacy training operators require WOTR_TRAINING_DEVICE=cpu')
    device = activate_training_device()
    if os.environ.get('WOTR_OPS_V2') == '0':
        if uses_predecision():
            raise ValueError('%s requires the v2 training operators' % current_contract())
        _OPS_V2_ACTIVE = False
        DELTA_COLLECTOR = _collect_search_samples
        VALUE_UPDATE_OP = _value_update_op
        DELTA_UPDATE_OP = _delta_update_op
        log_line('OPS: LEGACY v1 operators (explicit WOTR_OPS_V2=0):'
                 ' value=_value_update_op delta=_delta_update_op'
                 ' collector=_collect_search_samples -- the run-7'
                 ' thrash channel; A/B forensics only')
        return
    import delta_trainer
    DELTA_COLLECTOR = delta_trainer.collect_for_driver
    DELTA_UPDATE_OP = delta_trainer.delta_update_for_driver
    import value_trainer_v2 as _vt
    VALUE_UPDATE_OP = _vt.value_update_for_driver
    _OPS_V2_ACTIVE = True
    log_line('TRAINING DEVICE: %s (requested=%s)' % (device, requested_training_device()))
    log_line('OPTIMIZER STATE: %s' % optimizer_mode())
    log_line('TRAINING: value_epochs=%d policy_epochs=%d lambda=%g delta_wd=%g'
             % (EPOCHS, EPOCHS, LAM, DELTA_WD))
    log_line('OPS: v2 adopted operators (default; WOTR_OPS_V2=%s):'
             ' value=value_trainer_v2.value_update_for_driver'
             ' delta=delta_trainer.delta_update_for_driver'
             ' collector=delta_trainer.collect_for_driver'
             % (os.environ.get('WOTR_OPS_V2') or 'unset'))


_OPS_V2_ACTIVE = False
VALUE_UPDATE_OP = _value_update_op
DELTA_UPDATE_OP = _delta_update_op


def _buffer_pack(streams):
    """Downcast prep-format delta rows for buffer storage (f16 action
    encodings; upcast at load) — full-support rows carry the whole
    regenerated menu and would multiply into GBs at f32."""
    import numpy as np
    for key, d in streams.items():
        for row in d.get('delta', ()):
            if isinstance(row, dict):
                for f in ('ae', 'extra_ae'):
                    if f in row and getattr(row[f], 'dtype', None) \
                            is not None:
                        row[f] = row[f].astype(np.float16)
    return streams


def _validate_selector_stream(data):
    """Refuse incompatible replay before model or FIFO mutation."""
    from selector_contract import training_identity, validate_identity, validate_training_row
    expected = training_identity()
    from trace_decisions import validate_action_encoding_contract
    validate_action_encoding_contract(data, required=(uses_predecision() or
        data.get('reward_contract') in PREDECISION))
    if expected or 'selector_contract' in data:
        validate_identity(data, expected=expected or None)
    for row in data.get('delta', ()):
        if isinstance(row, dict):
            validate_action_encoding_contract(row, required=(uses_predecision() or
                row.get('selector_reward_contract') in PREDECISION))
            validate_training_row(row)
        elif expected:
            raise ValueError('Selector contract requires versioned policy rows')


def _buffer_unpack_delta(rows):
    import numpy as np
    for row in rows:
        if isinstance(row, dict):
            for f in ('ae', 'extra_ae'):
                if f in row and getattr(row[f], 'dtype', None) \
                        is not None:
                    row[f] = row[f].astype(np.float32)
    return rows


def _log_driver_stats():
    """SPTR-8 driver half (audit 2026-08-29): per-batch telemetry of
    the v2 collector's drop/mask counters
    (delta_trainer.DRIVER_STATS: rows/skew_dropped/error/stale_arms),
    then reset -- silent-drop counters must land in the driver log,
    never only in per-trace prints.  No-op under the v1 collector
    (delta_trainer not loaded = no DRIVER_STATS contract).  Returns
    the logged line, or None."""
    dt = sys.modules.get('delta_trainer')
    st = getattr(dt, 'DRIVER_STATS', None) if dt else None
    if not isinstance(st, dict) or not st:
        return None
    line = ('DRIVER_STATS: '
            + ' '.join('%s=%s' % (k, st[k]) for k in sorted(st)))
    log_line(line)
    for k in st:
        st[k] = 0
    return line


def train_nets_v2(data_by_net, lr_by_net):
    """R1/R5 driver: per-net update from BUFFERED data at LoopState-
    scaled LR; delta rows get a fixed-seed shuffle then a
    deterministic cap subsample.  (The B5 delta-holdout CE meter was
    DELETED 2026-08-11, USER de-bloat order: it was dark since birth —
    `delta_trainer.eval_target_ce` never existed, so the hasattr-
    guarded call never fired and the 20%% carve silently discarded
    training rows for a meter that could not report.)"""
    import random
    # Check every cohort before the first side's value/policy update. A later
    # incompatible cohort must not leave a partially updated checkpoint pair.
    for key in NET_KEYS:
        if key in data_by_net:
            _validate_selector_stream(data_by_net[key])
    losses = {}
    for key in NET_KEYS:
        side, prof = key
        name = '%s_%s' % key
        data = data_by_net.get(key) or {'value': [], 'delta': []}
        if not data['value']:
            losses[name] = None
            continue
        net = load_net(_nm('live', side, prof), side, profile=prof)
        value_options = {'epochs': EPOCHS, 'lam': LAM} if _OPS_V2_ACTIVE else {}
        policy_options = {'epochs': EPOCHS, 'wd': DELTA_WD} if _OPS_V2_ACTIVE else {}
        vloss = VALUE_UPDATE_OP(net, data['value'], lr_by_net[name], **value_options)
        dsamp = _buffer_unpack_delta(list(data['delta']))
        if dsamp:
            rng = random.Random(1234)
            rng.shuffle(dsamp)
            if len(dsamp) > DELTA_SAMPLE_CAP:
                dsamp = dsamp[:DELTA_SAMPLE_CAP]
        dloss = (DELTA_UPDATE_OP(net, dsamp, lr_by_net[name], **policy_options)
                 if dsamp else None)
        losses[name] = (vloss, len(data['value']),
                        {'delta': dloss, 'n_delta': len(dsamp)})
        save_net(net, side, _nm('live', side, prof), profile=prof)
        del net
    # SPTR-8: surface + reset the collector's per-batch drop/mask
    # counters (accumulated by collect_for_driver during extraction)
    _log_driver_stats()
    return losses


# CONF-4 once-per-run DARK latch (flags-off case only; in-process
# state, reset each loop invocation — no env knob)
_CONF4_DARK_LOGGED = [False]


def stage_a_gate(bdir):
    """Stage A (TELEMETRY): census verdict on the just-played batch;
    census JSON left inside the batch dir for the record."""
    import loop_meters
    agg, _rows = loop_meters.batch_census(bdir)
    ok, fails = loop_meters.stage_a_verdict(agg)
    # CONF-4 trend: DARK-never-silent + WARN on >0.2 conf_mean drop
    # vs the previous batch's stored census
    if agg.get('conf_meter_state') == 'DARK':
        # USER de-bloat 2026-08-11: with the conf flags OFF, DARK is
        # the EXPECTED state — log it ONCE per run, not per batch.
        # With WOTR_CONF_TRACK=1 a DARK census means the provider is
        # absent while the flag demands payloads — that stays loud
        # per batch.
        _flags_on = os.environ.get('WOTR_CONF_TRACK') == '1'
        if _flags_on or not _CONF4_DARK_LOGGED[0]:
            log_line('CONF-4 @%s: DARK (no cf payload; %s) — %d rows '
                     'uncovered%s'
                     % (os.path.basename(bdir),
                        ('provider absent under WOTR_CONF_TRACK=1'
                         if _flags_on else 'conf flags off'),
                        int(agg.get('conf_dark_rows') or 0),
                        ('' if _flags_on else
                         '; further DARK logs suppressed this run')))
            _CONF4_DARK_LOGGED[0] = True
    else:
        prev_cm = None
        try:
            import re as _re9
            m9 = _re9.search(r'batch_(\d+)', os.path.basename(bdir))
            if m9 and int(m9.group(1)) > 0:
                prev_p = os.path.join(
                    os.path.dirname(bdir),
                    'batch_%03d' % (int(m9.group(1)) - 1),
                    'stage_a_census.json')
                if os.path.exists(prev_p):
                    with open(prev_p, encoding='utf-8') as f9:
                        prev_cm = (json.load(f9).get('census') or
                                   {}).get('conf_mean')
        except Exception:
            prev_cm = None
        cm = agg.get('conf_mean')
        drop = (float(prev_cm) - float(cm)
                if prev_cm is not None and cm is not None else 0.0)
        log_line('CONF-4 @%s: conf_mean %s uniform_novelty %.3f%s'
                 % (os.path.basename(bdir),
                    ('%.3f' % cm) if cm is not None else 'n/a',
                    float(agg.get('conf_uniform_novelty_rate') or 0),
                    (' WARN conf-mean drop %.3f > 0.2' % drop)
                    if drop > 0.2 else ''))
    try:
        with open(os.path.join(bdir, 'stage_a_census.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'ok': ok,
                       'fails': [[x[0], x[1]] for x in fails],
                       'census': agg}, f, indent=1, default=float)
    except OSError:
        pass
    return ok, fails, agg


def _live_hashes():
    return {'%s_%s' % k: sha256(_nm('live', *k)) for k in NET_KEYS}


def _probe_live_rows():
    """Probe the CURRENT live pair; cached by live-net hashes so the
    pre-update probe is free whenever weights are unchanged since the
    last accepted update."""
    import loop_meters
    cache_p = os.path.join(LOOP, 'probe_cache.json')
    h = _live_hashes()
    if os.path.exists(cache_p):
        try:
            with open(cache_p, encoding='utf-8') as f:
                c = json.load(f)
            if c.get('hashes') == h:
                return c['rows']
        except (OSError, ValueError):
            pass
    fp_key, sh_key = _side_key('fp'), _side_key('sh')
    rows = loop_meters.probe_rows(
        _nm('live', *fp_key), _nm('live', *sh_key), _probeset_path(),
        profiles=(fp_key[1], sh_key[1]))
    _write_probe_cache(rows, h)
    return rows


def _write_probe_cache(rows, hashes=None):
    cache_p = os.path.join(LOOP, 'probe_cache.json')
    try:
        tmp = cache_p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'hashes': hashes or _live_hashes(),
                       'rows': rows}, f)
        os.replace(tmp, cache_p)
    except OSError:
        pass


def _anchor_t1_from_buffer(data_by_net):
    """B3' calibration anchor: realized MC return (minus phi, the
    residual currency the probe's v_raw speaks) at each game's first
    3 samples, per side, over the buffered healthy batches."""
    import numpy as np
    anchor = {}
    for key in NET_KEYS:
        side = key[0]
        vals = []
        for seq in (data_by_net.get(key) or {}).get('value', ()):
            if not (isinstance(seq, tuple) and len(seq) == 4):
                continue           # anchor is advisory: non-stream
            enc, rews, phis, term = seq   # entries are just skipped
            if len(rews) < 3:
                continue
            total = float(np.sum(rews)) + float(term)
            g = total
            for t in range(3):
                vals.append(g - float(phis[t]))
                g -= float(rews[t])
        if vals:
            anchor[side] = float(np.mean(vals))
    return anchor


def stage_b_gate(state, idx, pre_rows=None, anchor_t1=None):
    """R3 Stage B: probe the just-updated live pair vs the seed
    reference; per-update T1 swing uses LoopState's stored t1;
    paired-KL trust region vs the pre-update rows; B3' toward-anchor
    escape vs the buffer's realized returns."""
    import loop_meters
    fp_key, sh_key = _side_key('fp'), _side_key('sh')
    with open(os.path.join(LOOP, 'probe_ref_seed.json'),
              encoding='utf-8') as f:
        ref = json.load(f)
    rows = loop_meters.probe_rows(
        _nm('live', *fp_key), _nm('live', *sh_key), _probeset_path(),
        profiles=(fp_key[1], sh_key[1]))
    prev = {'fp': state.d['%s_%s' % fp_key].get('t1_prev'),
            'sh': state.d['%s_%s' % sh_key].get('t1_prev')}
    # B4 anchor = the seed probe reference.  (The promoted-checkpoint
    # anchor file died with arena promotion: nothing writes
    # probe_anchor_promoted.json anymore.)
    anchor_rows = ref['rows']
    verdict = loop_meters.stage_b_compare(ref['rows'], rows,
                                          prev_t1=prev,
                                          pre_rows=pre_rows,
                                          anchor_t1=anchor_t1,
                                          anchor_rows=anchor_rows)
    verdict['rows'] = rows
    try:
        with open(os.path.join(LOOP,
                               'probe_after_batch_%03d.json' % idx),
                  'w', encoding='utf-8') as f:
            json.dump({'verdict': {'ok': verdict['ok'],
                                   'fails': verdict['fails'],
                                   'warns': verdict['warns'],
                                   'metrics': verdict['metrics']},
                       'rows': rows}, f)
    except OSError:
        pass
    return verdict




def _batch_journal():
    mode = os.environ.get('WOTR_BATCH_RECOVERY', 'disabled')
    if mode == 'disabled':
        return None
    if mode != 'single_writer_journal_v1':
        raise ValueError('Unknown batch recovery contract')
    if ARENA_ENABLED or not uses_predecision() or os.environ.get('WOTR_NATIVE') != '0':
        raise ValueError('Initial batch recovery requires arena/native disabled and predecision traces')
    import batch_recovery
    from pathlib import Path
    import importlib
    import ai, board, cards, torch, numpy as np
    from selector_contract import training_identity
    from trace_decisions import current_learning_context
    modules = ['ai', 'loop_meters', 'value_trainer_v2', 'delta_trainer',
               'selector_contract', 'reward_contract', 'trace_decisions', 'batch_recovery']
    sources = {name: batch_recovery.digest(importlib.import_module(name).__file__)
               for name in modules}
    sources['run_td_v33loop'] = batch_recovery.digest(__file__)
    # Conservative complete Python source identity, with active overlay modules
    # taking precedence over the shared runtime they import.
    for directory in dict.fromkeys([Path(ai.__file__).parent, Path(__file__).parent]):
        for path in directory.rglob('*.py'):
            sources['tree/' + path.relative_to(directory).as_posix()] = batch_recovery.digest(path)
    if os.environ.get('WOTR_OPTIMIZER_STATE') == 'persistent_adam_v1':
        sources['optimizer_continuity'] = batch_recovery.digest(
            importlib.import_module('optimizer_continuity').__file__)
    if native_search_enabled():
        # The native search route is a search-implementation identity (pin file,
        # controller build, orchestrator), recorded like any other source.
        sources['native_search'] = _native_search_source_identity()
    controls = {'sources': sources,
        'data': {'board': batch_recovery.digest(board.MAP_JSON),
                 'cards': batch_recovery.digest(Path(cards.__file__).parent.parent / 'data/card_list.csv')},
        'resolved': {'reward': current_contract(), 'selector': training_identity(),
                     'context': current_learning_context(), 'net_keys': [list(k) for k in NET_KEYS],
                     'device': requested_training_device()},
        'runtime': {'python': list(sys.version_info[:3]), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'threads': torch.get_num_threads(),
                    'interop_threads': torch.get_num_interop_threads(),
                    'deterministic': torch.are_deterministic_algorithms_enabled(),
                    'matmul_precision': torch.get_float32_matmul_precision(),
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'matmul_tf32': torch.backends.cuda.matmul.allow_tf32},
        'environment': {
        k: v for k, v in os.environ.items() if k.startswith('WOTR_')
        and k not in JOURNAL_ENVIRONMENT_EXCLUDED},
        'globals': {k: globals()[k] for k in
                    ('BATCH_GAMES', 'LR', 'EPOCHS', 'LAM', 'DELTA_WD',
                     'BUFFER_K', 'DELTA_SAMPLE_CAP')},
        'probes_sha256': batch_recovery.digest(_probeset_path())}
    return batch_recovery.BatchJournal(LOOP,
        [_nm('live', *k) for k in NET_KEYS], NET_KEYS, controls)


def run_batch_cycle(idx, ctx):
    """Play, admit valid traces, train, and measure the resulting update.

    The extractor excludes invalid/incompatible games individually. An empty
    admitted batch cannot trigger an update on old data. Census and strength
    probes remain telemetry; weak play is not an integrity failure.
    """
    global _PRE_READY_ATTEMPT
    state = ctx['state']
    names = _net_names()
    journal = ctx.get('journal') or _batch_journal()
    from pre_ready_recovery import for_loop
    pre_ready = for_loop(journal, os.environ.get('WOTR_LOOP_SELFPLAY', 'standard_v1'))
    if journal is not None:
        from pathlib import Path
        if (Path(ctx['buffer'].root).resolve() != journal.path('buffer')
                or Path(state.path).resolve() != journal.path('loop_state.json')
                or ctx['buffer'].k != BUFFER_K
                or ctx['buffer'].net_keys != list(NET_KEYS)):
            raise RuntimeError('Recovery state/replay configuration differs from loop workspace')
        journal.recover()
        journal.bootstrap()
        if pre_ready is not None:
            pre_ready.recover()
        bdir = journal.ready(idx)
        if bdir is not None:
            import loop_meters
            # Replace in-memory bookkeeping with the restored before image.
            state.d = loop_meters.LoopState(state.path, names).d
        else:
            if pre_ready is None:
                bdir = ctx['play'](idx)
            else:
                if _PRE_READY_ATTEMPT is not None:
                    raise RuntimeError('Nested pre-ready play is unsupported')
                pre_ready.start(idx)
                _PRE_READY_ATTEMPT = pre_ready
                try:
                    bdir = ctx['play'](idx)
                finally:
                    _PRE_READY_ATTEMPT = None
    else:
        bdir = ctx['play'](idx)
    # 1.259: one loud anchor line per batch (DISABLED notice or the
    # realized frozen-episode split) - never silent
    _log_frozen_telemetry(bdir)
    if not ctx['audit'](bdir):
        ctx['log']('AUDIT batch %03d: violations found; per-trace '
                   'admission will exclude invalid games. See audit output.' % idx)
    ok_a, fails, _agg = ctx['census'](bdir)
    if not ok_a:
        ctx['log']('CENSUS (telemetry) batch %03d: %s'
                   % (idx, [f[0] for f in fails]))
    streams = ctx['extract'](bdir)
    if not any(d.get('value') or d.get('delta') for d in streams.values()):
        raise RuntimeError('Batch %03d has no admitted training data; see training_admission.json' % idx)
    if journal is not None:
        journal.begin(idx, bdir)
    ctx['buffer'].add(idx, streams)
    data = {k: ctx['buffer'].load(k) for k in NET_KEYS}
    lr = {n: state.lr(n, LR) for n in names}
    pre_rows = ctx['probe_pre']()
    losses = ctx['train'](data, lr)
    anchor = _anchor_t1_from_buffer(data)
    verdict = ctx['probe'](idx, pre_rows, anchor)
    if not verdict['ok']:
        ctx['log']('PROBE (telemetry) after batch %03d: %s — update '
                   'KEPT' % (idx, verdict['fails'][:4]))
    ctx['accept_probe'](verdict)
    for n in names:
        state.batch_ok(n)
    m = verdict.get('metrics', {})
    for side in ('fp', 'sh'):
        t1 = m.get('t1_%s' % side)
        if t1 is not None:
            state.d['%s_%s' % _side_key(side)]['t1_prev'] = t1
    state.save()
    ctx['log']('batch %03d v2 trained: %s (lr %s)'
               % (idx, losses, {k: round(v, 7)
                                for k, v in lr.items()}))
    if ctx['arena_due'](idx + 1):
        if journal is not None:
            raise RuntimeError('Arena is outside initial batch recovery scope')
        ctx['arena'](idx + 1)
    if journal is not None:
        journal.commit(idx)
        if pre_ready is not None:
            pre_ready.recover()
    return 'ok'


def gate_all_v2(idx, state):
    """Arena = SCOREBOARD ONLY (USER order 2026-08-11 gate removal):
    the vs-baseline games still play every 4 batches and the score is
    logged — promotion copies, reverts, and LR changes are all
    removed.  live_*.pt is the one continuously-training lineage; the
    user judges by the score trend."""
    for side, prof in NET_KEYS:
        name = '%s_%s' % (side, prof)
        score = run_arena_one(side, prof, idx=idx)
        if score is None:
            log_line('arena@%03d %s: unparseable (scoreboard only, '
                     'no action)' % (idx, name))
            continue
        state.d[name]['history'].append(['ARENA_SCORE', score])
        state.save()
        log_line('arena@%03d %s: score %.3f vs frozen baseline '
                 '(scoreboard)' % (idx, name, score))
    # DEFECT-1 leg-D WATCH: learned ring-scarcity price must not
    # erode (v33-corpus baseline dV -0.056; band [-0.15, -0.01]).
    # VALUE's probe module plugs in when landed; absent = logged
    # DARK, never silent.
    try:
        import spend_sensitivity
        dv = spend_sensitivity.probe(
            _nm('live', *_side_key('fp')), n_pairs=200)
        band_ok = -0.15 <= float(dv) <= -0.01
        log_line('spend-sensitivity WATCH @%03d: dV %.4f %s'
                 % (idx, float(dv),
                    'ok' if band_ok else
                    'WARN outside [-0.15,-0.01] — pre-registered '
                    'response: corpus-anchor value epochs'))
    except ImportError:
        log_line('spend-sensitivity WATCH @%03d: DARK (probe module '
                 'not landed)' % idx)
    except Exception as e9:
        log_line('spend-sensitivity WATCH @%03d: ERROR %r' % (idx, e9))
    integrity_check()


def _next_batch_idx(done):
    """SPLP-7 (audit 2026-08-29): the next batch index is max
    existing + 1 -- len(glob) collides with an existing batch dir
    after any gap (a deleted or renamed dir) and play_batch would
    then MIX two batches' traces in one dir.  Quarantined renames
    (batch_NNN_quarantined_*) still occupy their number."""
    import re as _re7
    nums = []
    for d in done:
        m7 = _re7.match(r'batch_(\d+)', os.path.basename(d))
        if m7:
            nums.append(int(m7.group(1)))
    return (max(nums) + 1) if nums else 0


def _unbuffered_last_batch(done):
    """SPLP-7 (audit 2026-08-29): a crash between play_batch and
    buffer.add leaves a played batch that advances the index but
    never enters training -- surface it at boot instead of dropping
    it silently.  Returns the batch number, or None."""
    import re as _re7
    nums = []
    for d in done:
        m7 = _re7.match(r'batch_(\d+)$', os.path.basename(d))
        if m7:
            nums.append(int(m7.group(1)))
    if not nums:
        return None
    b7 = max(nums)
    for k in NET_KEYS:
        if os.path.exists(os.path.join(
                LOOP, 'buffer', '%s_%s' % k, 'b%03d.pkl.gz' % b7)):
            return None
    return b7


def main_v2(a):
    import loop_meters
    _resolve_ops()
    journal = _batch_journal()
    if journal is not None:
        # Validate configured paths before lock/setup mutate the workspace.
        _pid_lock()
        journal.recover()
    setup()
    preflight_generation()
    if journal is None:
        _pid_lock()
    integrity_check()
    # No pre-flight smoke gate (USER order 2026-08-11): the loop
    # launches straight into batch play.  launch_smoke_gate.py is a
    # MANUAL tool (python src\launch_smoke_gate.py); nothing invokes it
    # automatically.
    log_line('v2 batch loop (no pre-flight gate; smoke gate is '
             'manual-only by user order)')
    quarantine_stale_traces()
    ps = _probeset_path()
    if not os.path.exists(ps):
        raise SystemExit('LOOP V2: probe set missing: %s (build via '
                         'experiments regime_probeset)' % ps)
    ref_p = os.path.join(LOOP, 'probe_ref_seed.json')
    fp_key, sh_key = _side_key('fp'), _side_key('sh')
    if not os.path.exists(ref_p):
        log_line('v2: generating seed probe reference (one-time)')
        rows = loop_meters.probe_rows(
            _nm('baseline', *fp_key), _nm('baseline', *sh_key), ps,
            profiles=(fp_key[1], sh_key[1]))
        loop_meters.validate_probe_reference(rows)
        tmp = ref_p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'rows': rows}, f)
        os.replace(tmp, ref_p)
    else:
        with open(ref_p, encoding='utf-8') as source:
            reference = json.load(source)
        loop_meters.validate_probe_reference(reference.get('rows'))
    state = loop_meters.LoopState(
        os.path.join(LOOP, 'loop_state.json'), _net_names())
    buffer = ReplayBuffer(os.path.join(LOOP, 'buffer'), NET_KEYS)
    ctx = {
        'state': state,
        'buffer': buffer,
        'play': lambda i: play_batch(i, BATCH_GAMES, a.workers,
                                     a.sims),
        'audit': audit_batch,
        'census': stage_a_gate,
        'extract': extract_batch_streams,
        'train': train_nets_v2,
        'probe_pre': _probe_live_rows,
        'probe': lambda i, pre, anch: stage_b_gate(
            state, i, pre_rows=pre, anchor_t1=anch),
        'accept_probe': lambda v: _write_probe_cache(
            v.get('rows') or []),
        'arena_due': lambda nxt: (ARENA_ENABLED
                                  and nxt % ARENA_EVERY_BATCHES == 0),
        'arena': lambda nxt: gate_all_v2(nxt, state),
        'log': log_line,
    }
    if journal is not None:
        journal.bootstrap()
        from pre_ready_recovery import for_loop
        recovery = for_loop(journal, os.environ.get('WOTR_LOOP_SELFPLAY', 'standard_v1'))
        if recovery is not None:
            recovery.recover()
        ctx['journal'] = journal
    done = sorted(glob.glob(os.path.join(LOOP, 'batch_*')))
    idx = journal.next_index() if journal is not None else _next_batch_idx(done)
    b7 = None if journal is not None else _unbuffered_last_batch(done)
    if b7 is not None:
        log_line('WARNING: batch %03d was played but never buffered '
                 '(crash between play_batch and buffer.add) -- its '
                 'games are EXCLUDED from training and the index '
                 'moves on (SPLP-7)' % b7)
    log_line('LOOP V2 START at batch %d (%d-net, buffer K=%d, '
             'target %d games)'
             % (idx, len(NET_KEYS), BUFFER_K, a.total_games))
    while idx * BATCH_GAMES < a.total_games:
        t0 = time.time()
        res = run_batch_cycle(idx, ctx)
        log_line('batch %03d cycle result: %s (%.0f min)'
                 % (idx, res, (time.time() - t0) / 60))
        idx += 1
        integrity_check()
    log_line('LOOP V2 DONE: %d batches' % idx)


def _served_batch_launch(idx, games, workers, sims, env, frozen):
    """Fixed-pair served backend; the controller owns its fresh output tree."""
    keys = {side: [p for s, p in NET_KEYS if s == side] for side in ('fp', 'sh')}
    if len(NET_KEYS) != 2 or any(len(v) != 1 for v in keys.values()):
        raise ValueError('Served loop backend requires one pinned profile per side')
    if frozen is not None and frozen[0] > 0:
        raise ValueError('Served loop backend does not yet support frozen opponents')
    inherited_fp = env.get('WOTR_SP_FROZEN_FP') or None
    inherited_sh = env.get('WOTR_SP_FROZEN_SH') or None
    if bool(inherited_fp) != bool(inherited_sh):
        raise ValueError('Served loop refuses a partial inherited frozen checkpoint pair')
    raw_probability = env.get('WOTR_SP_FROZEN_P')
    if raw_probability is not None or inherited_fp:
        try:
            probability = float(raw_probability or '0.30')
        except (ValueError, TypeError):
            raise ValueError('Served loop has an invalid inherited frozen probability') from None
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Served loop has an invalid inherited frozen probability')
        if probability > 0:
            raise ValueError('Served loop does not support inherited active frozen settings')
    if games < 1 or workers < 1 or not sims or sims < 1:
        raise ValueError('Served loop games, workers and sims must be positive')
    base = os.path.join(LOOP, 'incoming_served_b%03d' % idx)
    output = base
    shard = env.get('WOTR_LOOP_SHARD_ID')
    if 'FLUX_TASK_RANK' in env and shard is None:
        raise ValueError('Flux served loop requires explicit WOTR_LOOP_SHARD_ID')
    if shard is not None:
        if not shard.isdigit():
            raise ValueError('WOTR_LOOP_SHARD_ID must be a nonnegative integer')
        output = os.path.join(base, 'shard_%06d' % int(shard))
    if os.path.exists(output) or os.path.exists(os.path.join(LOOP, 'batch_%03d' % idx)):
        raise ValueError('Served batch output already exists; recovery must select a ready batch')
    paths = {side: _nm('live', side, keys[side][0]) for side in keys}
    if any(not os.path.isfile(path) for path in paths.values()):
        raise FileNotFoundError('Served loop live checkpoint pair is incomplete')
    args = [PY, '-u', os.path.join(SRC, 'run_sp_served.py'), '--arch', 'base',
            '--fp-net', paths['fp'], '--sh-net', paths['sh'],
            '--games', str(games), '--workers', str(workers), '--sims', str(sims),
            '--seed0', env['WOTR_SP_SEED0'], '--out-dir', base]
    if shard is not None:
        args += ['--shard-id', str(int(shard))]
    reuse_options = {'WOTR_SERVED_REUSE_MANIFEST': '--reuse-completed-manifest',
                     'WOTR_SERVED_REUSE_SHA256': '--reuse-completed-sha256',
                     'WOTR_SERVED_REUSE_REVIEW': '--reuse-review',
                     'WOTR_SERVED_REUSE_REVIEW_SHA256': '--reuse-review-sha256'}
    # Reviewed recovery of a later uncommitted collection must retain its
    # completed identities while using the already committed live pair.
    # The pointer is generation-specific; the ordinary reuse loader checks
    # the exact seed, checkpoint pair, budget, receipts and admission again.
    recovery_pointer = os.path.join(LOOP, '.batch_journal',
                                    'completed_reuse_b%03d.json' % idx)
    if idx > 0 and os.path.isfile(recovery_pointer):
        with open(recovery_pointer, encoding='utf-8') as stream:
            reuse = json.load(stream)
        if (set(reuse) != {'schema', 'index', 'environment'}
                or reuse['schema'] != 'reviewed_generation_reuse_v1'
                or type(reuse['index']) is not int or reuse['index'] != idx
                or set(reuse['environment']) != set(reuse_options)
                or not all(isinstance(v, str) and v for v in reuse['environment'].values())):
            raise ValueError('Invalid generation-specific completed-game reuse')
        for key, option in reuse_options.items():
            args += [option, reuse['environment'][key]]
    if any(env.get(key) for key in reuse_options):
        if not all(env.get(key) for key in reuse_options):
            raise ValueError('Incomplete explicit completed-game reuse configuration')
        # This bounded repair retains only generation-zero games. Later
        # generations use the ordinary fresh collection/checkpoint pipeline.
        if idx == 0:
            for key, option in reuse_options.items():
                args += [option, env[key]]
    if native_search_enabled():
        identity = _native_search_source_identity()
        log_line('SEARCH ROUTE batch %03d: native (WOTR_NATIVE_SEARCH=1; pin %s; controller binary %s)'
                 % (idx, identity['reference_path'], identity['controller_binary_sha256']))
    else:
        log_line('SEARCH ROUTE batch %03d: python (WOTR_NATIVE_SEARCH unset/0)' % idx)
    log_line('GAME-FAILURE POLICY batch %03d: failed games are recorded (fixture + ledger) and the batch '
             'continues (WOTR_SP_MAX_GAME_FAILURES=%s, default = batch size); accepted for training with '
             '>= %s=%d completed games'
             % (idx, env.get('WOTR_SP_MAX_GAME_FAILURES') or 'unset', MIN_BATCH_GAMES_ENV, min_batch_games()))
    log_line('SEARCH BUDGET batch %03d: %s' % (idx, search_budget_policy(env)))
    return args, output



def _validate_served_archive_coverage(output, bdir, phase, attempts, games, rc, walled, errors):
    """Validate observed records; never infer an unreported game's outcome."""
    if phase is None:
        return
    owned = [row for row in attempts if row.get('phase_id') == phase['phase_id']]
    strict = any(key in phase for key in ('planned', 'completed', 'error_attempts',
                                          'unreported', 'coverage_complete', 'finished'))
    def error(message):
        errors.append({'file': 'served coverage', 'error': message})
    claims_results = bool(phase.get('observed') or phase.get('completed')
                          or phase.get('coverage_complete'))
    if claims_results:
        if not os.path.isfile(os.path.join(bdir, 'served_attempts.raw.jsonl')):
            error('Phase claims results but attempts metadata is absent')
        if (phase.get('completed', phase.get('observed', 0))
                and not os.path.isfile(os.path.join(bdir, 'served_games.raw.jsonl'))):
            error('Phase claims completed games but game metadata is absent')
    paths = {}
    completed_paths = []
    game_paths = []
    def trace_name(row, require_existing, require_path):
        source = row.get('trace_path')
        if not source:
            if require_path:
                error('Observed completed record has no trace_path')
            return None
        if not isinstance(source, str):
            error('trace_path must be a string')
            return None
        canonical = os.path.normcase(os.path.abspath(source))
        name = os.path.basename(source)
        normalized_name = os.path.normcase(name)
        previous = paths.setdefault(normalized_name, canonical)
        if previous != canonical:
            error('Different source trace paths collapse to one archive basename: ' + name)
        expected = os.path.normcase(os.path.abspath(os.path.join(output, 'traces', name)))
        if not os.path.isabs(source) or canonical != expected or not name.endswith('.trace.gz'):
            error('Trace target is outside the served trace directory: ' + source)
        if require_existing and not os.path.isfile(os.path.join(bdir, name)):
            error('Completed trace target is absent from archive: ' + name)
        return normalized_name
    for row in owned:
        completed = row.get('status') == 'completed'
        name = trace_name(row, completed, strict and completed)
        if completed:
            completed_paths.append(name)
    for row in games:
        game_paths.append(trace_name(row, True, strict))
    if not strict:
        return
    fields = ('planned', 'observed', 'unreported', 'completed', 'error_attempts')
    if any(type(phase.get(key)) is not int or phase[key] < 0 for key in fields):
        error('Phase coverage counters must be complete nonnegative integers')
        return
    if type(phase.get('coverage_complete')) is not bool or type(phase.get('finished')) is not bool:
        error('Phase completion flags must be explicit booleans')
    if phase['planned'] < 1 or phase['observed'] > phase['planned']:
        error('Invalid planned/observed phase coverage')
    if phase['observed'] != len(owned):
        error('Observed phase counter differs from owned attempt records')
    if phase['unreported'] != phase['planned'] - phase['observed']:
        error('Unreported phase counter differs from planned minus observed')
    if phase.get('coverage_complete') != (phase['observed'] == phase['planned']):
        error('Phase coverage_complete flag contradicts observed coverage')
    if phase['completed'] != sum(row.get('status') == 'completed' for row in owned):
        error('Completed phase counter differs from attempt records')
    if phase['error_attempts'] != sum(row.get('status') == 'error' for row in owned):
        error('Error-attempt phase counter differs from attempt records')
    if any(row.get('status') not in ('completed', 'error') for row in owned):
        error('Attempt record has an unsupported status')
    identities = set()
    for row in owned:
        identity = (row.get('worker_id'), row.get('episode'))
        if any(type(value) is not int or value < 0 for value in identity):
            error('Attempt requires nonnegative worker/episode identity')
        elif identity in identities:
            error('Duplicate worker/episode attempt identity')
        else:
            identities.add(identity)
    if (len(games) != phase['completed'] or len(set(game_paths)) != len(game_paths)
            or sorted(str(p) for p in game_paths) != sorted(str(p) for p in completed_paths)):
        error('Completed attempt traces and game records do not correspond one to one')
    if rc == 0 and not walled:
        # Graceful per-game failure: error attempts are allowed on a successful
        # exit; every assigned episode must still be REPORTED and no worker-level
        # failure may exist (their traces were moved to <batch>/failures/ first).
        if (phase.get('finished') is not True or phase.get('coverage_complete') is not True
                or phase.get('worker_failures') or phase.get('unreported_episodes')
                or phase.get('parent_error')):
            error('Successful runner exit does not have complete phase evidence '
                  '(every assigned episode reported, no worker failure)')
        archived_traces = {os.path.normcase(os.path.basename(path))
                           for path in glob.glob(os.path.join(bdir, '*.trace.gz'))}
        if archived_traces != set(completed_paths):
            error('Successful phase trace inventory contains unreported or missing traces')



def _archive_served_batch(idx, output, rc, walled, supervisor_cleanup):
    """Keep raw evidence before parsing; only validated records reach handoff."""
    bdir = os.path.join(LOOP, 'batch_%03d' % idx)
    os.makedirs(bdir, exist_ok=False)
    raw_names = {'phase.json': 'served_phase.raw.json',
                 'attempts.jsonl': 'served_attempts.raw.jsonl',
                 'games.jsonl': 'served_games.raw.jsonl',
                 'launch.json': 'served_launch.json', 'report.md': 'served_report.md'}
    for name, archive_name in raw_names.items():
        source = os.path.join(output, name)
        if os.path.isfile(source):
            shutil.copyfile(source, os.path.join(bdir, archive_name))
    for source in glob.glob(os.path.join(output, 'traces', '*.trace.gz')):
        shutil.move(source, bdir)
    supervisor = {'runner_returncode': rc, 'wall_cap_reached': walled,
                  'cleanup': supervisor_cleanup, 'served_output': output}
    with open(os.path.join(bdir, 'supervisor.json'), 'w', encoding='utf-8') as handle:
        json.dump(supervisor, handle, indent=2)
    errors = []
    phase = None
    try:
        with open(os.path.join(bdir, 'served_phase.raw.json'), encoding='utf-8') as handle:
            phase = json.load(handle)
        if (not isinstance(phase, dict) or not isinstance(phase.get('phase_id'), str)
                or not phase['phase_id']):
            raise ValueError('phase must have a nonempty phase_id')
    except (OSError, ValueError) as exc:
        errors.append({'file': 'phase.json', 'error': str(exc)})
        phase = None

    def records(raw_name, original_name):
        path = os.path.join(bdir, raw_name)
        if not os.path.exists(path):
            return []
        rows = []
        try:
            with open(path, encoding='utf-8') as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError('record must be an object')
                        rows.append(row)
                    except ValueError as exc:
                        errors.append({'file': original_name, 'line': number, 'error': str(exc)})
        except (OSError, UnicodeError) as exc:
            errors.append({'file': original_name, 'error': str(exc)})
        return rows

    attempts = records('served_attempts.raw.jsonl', 'attempts.jsonl')
    games = records('served_games.raw.jsonl', 'games.jsonl')
    failure_rows = _archive_served_failures(output, bdir, phase, attempts)
    _validate_served_archive_coverage(output, bdir, phase, attempts, games, rc, walled, errors)
    if phase is not None:
        phase.update(runner_returncode=rc, wall_cap_reached=walled)
        # The runner's own cleanup result is preserved. The parent's actual
        # tree termination result is separate in supervisor.json.
        with open(os.path.join(bdir, 'phase.json'), 'w', encoding='utf-8') as handle:
            json.dump(phase, handle, indent=2)
    if errors:
        with open(os.path.join(bdir, 'served_archive_errors.json'), 'w', encoding='utf-8') as handle:
            json.dump({'errors': errors, 'handoff': 'refused'}, handle, indent=2)
        raise ValueError('Served archive metadata is incomplete/malformed; evidence retained in %s' % bdir)

    def write_rows(name, rows):
        with open(os.path.join(bdir, name), 'w', encoding='utf-8') as handle:
            for original in rows:
                row = dict(original)
                trace = row.get('trace_path')
                if trace:
                    # Failed games' traces live under <batch>/failures/ (diagnostics only).
                    folder = os.path.join(bdir, 'failures') if row.get('status') == 'error' else bdir
                    row['trace_path'] = os.path.join(folder, os.path.basename(str(trace)))
                handle.write(json.dumps(row) + '\n')
    owned = [row for row in attempts if row.get('phase_id') == phase['phase_id']]
    write_rows('attempts.jsonl', owned)
    write_rows('games.jsonl', games)
    if os.environ.get('WOTR_PRE_READY_RECOVERY', '') != 'served_quiescent_v1':
        _feed_append(os.path.join(bdir, 'games.jsonl'), bdir)
    completed = sum(row.get('status') == 'completed' for row in owned)
    planned = int(phase.get('planned') or 0)
    if failure_rows or completed < planned:
        _record_batch_shortfall(idx, bdir, phase, failure_rows, completed, planned, rc, walled)
    if rc != 0 and not walled:
        raise SystemExit('batch %d selfplay exit %d; evidence saved in %s' % (idx, rc, bdir))
    minimum = min_batch_games()
    if completed < minimum:
        log_line('BATCH %03d REFUSED: %d completed game(s) < %s=%d; evidence saved in %s'
                 % (idx, completed, MIN_BATCH_GAMES_ENV, minimum, bdir))
        raise SystemExit('batch %d has %d completed game(s) < %s=%d; evidence saved in %s'
                         % (idx, completed, MIN_BATCH_GAMES_ENV, minimum, bdir))
    return bdir



if __name__ == '__main__':
    main()
