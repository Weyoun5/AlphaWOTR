# -*- coding: utf-8 -*-
"""Train values from per-player TD(lambda) sequences.

The default operator updates the value head and appended observation columns.
Explicit full_network_v1 trains the complete value pathway using eval-mode
modules with autograd enabled: running statistics and dropout stay fixed for
singleton safety. Legacy/appended modes retain their existing behavior.
"""
import os
from training_device import model_device

VALUE_BS = int(os.environ.get('WOTR_VALUE_BS', '256'))
VALUE_EPOCHS = int(os.environ.get('WOTR_VALUE_EPOCHS', '1'))
VALUE_GRAD_CLIP = float(os.environ.get('WOTR_VALUE_GRAD_CLIP', '1.0'))
LAM = 0.8   # default; build_targets resolves the DRIVER's effective
            # LAM at call time (loop_config.json can override it)


def _effective_lam():
    """The driver's runtime LAM (run_td_v33loop applies loop_config
    overrides); falls back to the local default when the driver module
    is absent (standalone use)."""
    try:
        import run_td_v33loop as _drv9
        return float(_drv9.LAM)
    except Exception:
        return LAM


def _head_clip(net):
    """Target clip matched to the head's reachable range: 0.98*S for
    ScaledTanh, 0.98 for legacy Tanh (reachability is the invariant —
    battery A2)."""
    scale = getattr(net.value_head[-1], 'scale', None)
    if scale:
        return 0.98 * float(scale)
    print('value_trainer_v2: LEGACY Tanh head detected — clipping '
          'targets to +-0.98 (tail grading lost).  Relaunch with '
          'SWAP-ONLY scaled-head seeds (V-G-prime).', flush=True)
    return 0.98


def _freeze_scope(net):
    """Head-only training scope with V-E BN freeze."""
    import torch
    for p in net.parameters():
        p.requires_grad = False
    head = [p for k, p in net.named_parameters()
            if k.startswith('value_head')]
    for p in head:
        p.requires_grad = True
    net.eval()
    net.value_head.train()
    for m in net.value_head.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()
    return head


def build_targets(net, seqs, lam=None, clip=None):
    """TD(lambda) at pre-decision states with outgoing interval rewards.

    seqs: [(enc float32 (N,D), rewards (N,), potentials (N,), terminal)].
    The last reward covers the interval to termination. Under the win/loss
    contract rewards and potentials are zero and terminal is +1 or -1.
    """
    import numpy as np
    import torch
    if lam is None:
        lam = _effective_lam()
    if clip is None:
        clip = _head_clip(net)
    from reward_contract import uses_predecision
    bounded = uses_predecision()
    if bounded:
        clip = min(float(clip), 1.0)
    if not 0 <= lam <= 1 or not np.isfinite(clip) or clip <= 0:
        raise ValueError('TD lambda must be in [0,1] and clip positive/finite')
    targets = []
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            for enc, rews, phis, term in seqs:
                enc, rews, phis = (np.asarray(x, dtype=np.float32)
                                   for x in (enc, rews, phis))
                if (enc.ndim != 2 or len(enc) == 0
                        or rews.shape != (len(enc),) or phis.shape != (len(enc),)
                        or not all(np.isfinite(x).all() for x in (enc, rews, phis))
                        or not np.isfinite(term)):
                    raise ValueError('Malformed or nonfinite value trajectory')
                v = net(torch.from_numpy(enc).to(model_device(net))).detach().cpu().numpy().reshape(-1) + phis
                if v.shape != (len(enc),) or not np.isfinite(v).all():
                    raise ValueError('Invalid value bootstrap predictions')
                if bounded:
                    if np.any(rews != 0) or np.any(phis != 0) or not -1 <= term <= 1:
                        raise ValueError('Predecision contract requires zero rewards/potentials and bounded terminals')
                    v = np.clip(v, -1.0, 1.0)
                g = np.zeros(len(enc), dtype=np.float32)
                g[-1] = rews[-1] + term
                for t in range(len(enc) - 2, -1, -1):
                    g[t] = rews[t] + (1 - lam) * v[t + 1] + lam * g[t + 1]
                targets.append(np.clip(g - phis, -clip, clip))
    finally:
        net.train(was_training)
    return targets


def value_update_for_driver(net, seqs, lr, *, epochs=None, lam=None):
    """Driver contract (net, seqs, lr) -> mean loss (float).
    Deterministic given (weights, seqs): fixed shuffle seed AND a
    fixed torch seed here (dropout in the head consumes torch RNG —
    the driver does not seed it)."""
    import numpy as np
    import torch
    if epochs is not None and (type(epochs) is not int or epochs <= 0):
        raise ValueError('Value epochs must be a positive integer')
    effective_epochs = VALUE_EPOCHS if epochs is None else epochs
    if not seqs:
        return None
    torch.manual_seed(0)
    targets = build_targets(net, seqs, lam=lam)
    enc_all = np.concatenate([s[0] for s in seqs], axis=0)
    tgt_all = np.concatenate(targets).astype(np.float32)
    head = _freeze_scope(net)
    from observation_training import ObservationTrainingScope
    with ObservationTrainingScope(net) as scope:
        from optimizer_continuity import optimizer_for, updated
        opt = optimizer_for(net, 'value', scope.parameter_groups(head), scope, lr)
        rng = np.random.RandomState(0)
        n = len(enc_all)
        # BatchNorm stays in eval, so a final singleton batch is valid.
        bs = min(VALUE_BS, n)
        tot = cnt = 0.0
        for _ in range(effective_epochs):
            order = rng.permutation(n)
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                pred = net(torch.from_numpy(enc_all[idx]).to(model_device(net))).reshape(-1)
                loss = torch.nn.functional.mse_loss(
                    pred, torch.from_numpy(tgt_all[idx]).to(pred.device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']], VALUE_GRAD_CLIP)
                opt.step()
                updated(net, 'value', opt)
                scope.record_step('value')
                from trace_decisions import record_warm_start_update
                record_warm_start_update(net, 'value')
                tot += float(loss.item())
                cnt += 1
        net.eval()
        return round(tot / max(1.0, cnt), 4)


# ---------------------------------------------------------------
# V-G recalibration (ADOPTION/rollout step, user-gated; never called
# by the loop)
# ---------------------------------------------------------------
def recalibrate_head(net, seqs, lr=1e-3, epochs=3, seed=0):
    """REJECTED-FOR-ADOPTION (D-V2 register): full-head MC refit
    repriced the state-value landscape (erebor survival line erased,
    6/6 smoke losses) — kept ONLY as the rejection-evidence
    implementation.  Adoption seeds are SWAP-ONLY (V-G'); the level
    bias converges in the gated loop."""
    import numpy as np
    import torch
    clip = _head_clip(net)
    encs, tgts = [], []
    for enc, rews, phis, term in seqs:
        g = np.zeros(len(enc), dtype=np.float64)
        g[-1] = rews[-1] + term
        for t in range(len(enc) - 2, -1, -1):
            g[t] = rews[t] + g[t + 1]
        encs.append(enc)
        tgts.append(np.clip(g - phis, -clip, clip))
    enc_all = np.concatenate(encs, axis=0)
    tgt_all = np.concatenate(tgts).astype(np.float32)
    head = _freeze_scope(net)
    opt = torch.optim.Adam(head, lr=lr)
    rng = np.random.RandomState(seed)
    n = len(enc_all)
    # same B5-adjacent guard as value_update_for_driver: fewer samples
    # than one minibatch must not silently no-op with loss 0.0
    bs = min(VALUE_BS, n)
    if bs < 2:
        raise RuntimeError(
            'recalibrate_head: only %d samples (< 2) - nothing to fit'
            % n)
    tot = cnt = 0.0
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            idx = order[i:i + bs]
            pred = net(torch.from_numpy(enc_all[idx]).to(model_device(net))).reshape(-1)
            loss = torch.nn.functional.mse_loss(
                pred, torch.from_numpy(tgt_all[idx]).to(pred.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head, VALUE_GRAD_CLIP)
            opt.step()
            from trace_decisions import record_warm_start_update
            record_warm_start_update(net, 'value')
            tot += float(loss.item())
            cnt += 1
    net.eval()
    return {'loss': round(tot / max(1.0, cnt), 4),
            'steps': int(cnt), 'samples': int(n)}


def save_v2_checkpoint(net, side, path, profile=None):
    """Checkpoint with the head_act field (battery A6: without it a
    reload silently reconstructs plain Tanh — the interop trap).
    conf_* fields round-trip from any file being overwritten
    (ultrareview bug_006: a from-scratch dict strips the confidence
    stamp and silently de-registers the provider on later loads)."""
    import torch
    import ai
    from reward_contract import current_contract, PROFILE, validate_profile
    from trace_decisions import current_learning_context
    from selector_contract import training_identity
    if current_contract() == PROFILE:
        profile = profile or getattr(net, 'reward_profile', None)
        validate_profile(side, profile)
    ck = {'model_state': net.state_dict(),
          'input_dim': ai.get_feature_dim(),
          'action_dim': ai.ACTION_ENCODING_DIM,
          'head_act': ('scaled2'
                       if getattr(net.value_head[-1], 'scale',
                                  None) else 'tanh'),
          'side': side, 'reward_contract': current_contract()}
    ck.update(current_learning_context())
    from trace_decisions import checkpoint_learning_lineage
    ck.update(checkpoint_learning_lineage(net))
    ck.update(training_identity())
    if current_contract() == PROFILE:
        ck['reward_profile'] = profile
    from optimizer_continuity import checkpoint_payload
    ck.update(checkpoint_payload(net, ck))
    try:
        if os.path.exists(path):
            _old = torch.load(path, map_location='cpu',
                              weights_only=False)
            for _f in ('conf_mech', 'conf_state', 'conf_calib',
                       'conf_meta'):
                if _f in _old:
                    ck[_f] = _old[_f]
    except Exception:
        pass  # stamp restored by the next --stage conf-stamp
    tmp = path + '.tmp'
    torch.save(ck, tmp)
    os.replace(tmp, path)


def swap_head_scaled2(net):
    """In-place head swap to ScaledTanh S=2 (parameter-free)."""
    import ai
    net.value_head[-1] = ai._ScaledTanhHead(2.0)
    return net
