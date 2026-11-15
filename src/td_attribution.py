# -*- coding: utf-8 -*-
"""Inspect rewards and temporal advantages with explicit value provenance.

Win/loss traces use the behavior values saved before each decision. Missing
values stay unavailable. An explicitly selected checkpoint can instead
provide a retrospective analysis; legacy traces also use retrospective
checkpoint estimates. Advantages describe realized versus predicted
returns, not proof that an individual move caused the game's outcome.

Usage: python td_attribution.py <trace.gz> [fp_profile] [sh_profile]
Also callable from the dashboard (Attribution button).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_net(ai, torch, profile_key):
    """Choose a recent compatible net for retrospective legacy analysis.

    A current live file is not evidence of the model used in an old game.
    Generation checks are delegated to the normal value-net loader.
    """
    import glob
    import re
    side = profile_key[:2]

    def _newest(pattern):
        best_v, best_p = -1, None
        for p in glob.glob(os.path.join(ROOT, pattern)):
            m = re.search(r'_v(\d+)', os.path.basename(p))
            v = int(m.group(1)) if m else 0
            if v > best_v:
                best_v, best_p = v, p
        return best_p

    # prefer the value-refit chain explicitly (corpus_vNNtd naming; the
    # bare corpus glob cannot rank v30jt over v30 - the _v(\d+) regex
    # reads 30 for all three suffixes)
    cands = [
        # Current live weights provide reanalysis, not historical credit.
        os.path.join(ROOT, 'data', 'v33loop',
                     'live_%s.pt' % profile_key),
        _newest(os.path.join('_parallel_weights',
                             profile_key + '_weights_v*.pt')),
        _newest('wotr_%s_corpus_v*jt.pt' % side),
        _newest('wotr_%s_corpus_v*td.pt' % side),
        _newest('wotr_%s_corpus_v*.pt' % side),
    ]
    # ONE-loader discipline (HUNT-B F5): ai.load_value_net is the single
    # law (head_act from the checkpoint, generation refusal, conf
    # auto-reg).  A cross-generation candidate is SKIPPED LOUDLY, never
    # zero-padded or faked.
    for p in cands:
        if not p or not os.path.exists(p):
            continue
        try:
            net = ai.load_value_net(p, side=side)
            if net is None:
                continue
            net.eval()
            return net, os.path.basename(p)
        except Exception as e:
            print('attribution: skipping %s (%s)'
                  % (os.path.basename(p), repr(e)[:160]))
            continue
    return None, None


def _finite_value(value):
    """Accept numeric predictions, without inventing values for missing data."""
    import math
    from numbers import Real
    if isinstance(value, Real) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _temporal_advantages(steps, outcomes, lam):
    """Use the next decision of the same player, preserving missing-value gaps.

    Gamma is one, as in the trainer. A missing future prediction invalidates
    the affected TD edge and any earlier complete GAE sum, while leaving
    independently computable one-step deltas visible. Undecided games use
    the last observed value as a provisional bootstrap, not a result label.
    """
    for tag in ('fp', 'sh'):
        seq = [s for s in steps if s['side'] == tag.upper()]
        nxt = outcomes[tag]
        if nxt is None:
            nxt = seq[-1]['v'] if seq else None
        gae = 0.0
        for row in reversed(seq):
            value = row['v']
            delta = (row['reward'] + nxt - value
                     if value is not None and nxt is not None else None)
            gae = (delta + lam * (gae if gae is not None else 0.0)
                   if delta is not None and (gae is not None or lam == 0) else None)
            row['delta'] = delta
            row['adv'] = gae
            nxt = value


def compute_attribution(trace_path, fp_profile=None,
                        sh_profile=None, *, checkpoint_paths=None):
    """Return per-decision rewards, value provenance, and temporal advantages.

    For the win/loss contract, recorded decision_context['value'] is the
    default and already includes the recorded potential. Missing predictions
    are unavailable, even if a newer checkpoint exists. checkpoint_paths may
    explicitly map 'fp' and/or 'sh' to files for retrospective reanalysis;
    unspecified win/loss sides still use recorded predictions. Legacy traces
    use recent checkpoints because their historical predictions are absent.

    Lambda comes from the current loop configuration, not historical metadata.
    Environment and existing player profiles are restored on every exit.
    """
    import numpy as np
    import torch
    import ai
    import game_logic
    import game_trace
    from reward_contract import uses_predecision, current_contract, trace_contract, terminal
    from trace_decisions import forced_policy_reason, is_learning_decision

    selected = dict(checkpoint_paths or {})
    if set(selected) - {'fp', 'sh'}:
        raise ValueError("checkpoint_paths keys must be 'fp' or 'sh'")
    tr = game_trace.load_trace(trace_path)
    contract = trace_contract(tr)
    old_contract = os.environ.get('WOTR_REWARD_CONTRACT')
    old_profiles = dict(ai.PLAYER_PROFILES)
    recorded_profiles = (tr.get('meta') or {}).get('profiles') or {}
    profiles = {'fp': fp_profile or recorded_profiles.get('fp') or 'ring',
                'sh': sh_profile or recorded_profiles.get('sh') or 'military'}
    if uses_predecision(contract) and any(
            recorded_profiles.get(tag) and recorded_profiles[tag] != profiles[tag]
            for tag in ('fp', 'sh')):
        raise ValueError('Attribution profiles differ from the recorded reward objective')
    try:
        os.environ['WOTR_REWARD_CONTRACT'] = contract
        current_contract()  # Refuse unknown trace semantics before replay.
        ai.PLAYER_PROFILES.update(profiles)
        all_rows = [s for s in tr['steps'] if s.get('state') is not None
                    and s.get('kind') != 'new_turn']
        rows = [s for s in all_rows if is_learning_decision(s)]
        forced_steps = [
            {'trace_step': i, 'turn': s.get('turn'), 'side': s.get('side'),
             'action_type': s.get('action_type'), 'params': dict(s.get('params') or {}),
             'forced_policy_reason': forced_policy_reason(s)}
            for i, s in enumerate(tr['steps']) if forced_policy_reason(s) is not None]
        if not rows:
            raise RuntimeError('trace has no learned decisions with state snapshots')
        prev_of = {id(all_rows[k]): all_rows[k - 1]
                   for k in range(1, len(all_rows))}

        import run_td_v33loop as trainer
        trainer._apply_loop_config()
        detail = []
        per = trainer._replay_reward_stream(tr, profiles, detail=detail, for_learning=False)
        by_step = {d['step_idx']: d for d in detail}
        step_index = {id(s): i for i, s in enumerate(tr['steps'])}
        values, sources, nets = {}, {}, {}
        for tag in ('fp', 'sh'):
            encs, _rewards, phis = per[tag]
            values[tag] = [None] * len(encs)
            sources[tag] = ['unavailable'] * len(encs)
            net = None
            if tag in selected:
                path = os.fspath(selected[tag])
                net = ai.load_value_net(path, side=tag)
                if net is None:
                    raise ValueError('Cannot load selected checkpoint: %s' % path)
                net.eval()
                nets[tag] = path
            elif not uses_predecision(contract):
                net, nets[tag] = _load_net(ai, torch, tag + '_' + profiles[tag])
            else:
                nets[tag] = None
                for d in detail:
                    if d['tag'] != tag:
                        continue
                    context = tr['steps'][d['step_idx']].get('decision_context') or {}
                    value = _finite_value(context.get('value'))
                    values[tag][d['k']] = value
                    if value is not None:
                        sources[tag][d['k']] = 'recorded_behavior'
            if net is not None and encs:
                with torch.no_grad():
                    x = torch.from_numpy(np.asarray(encs, dtype=np.float32)).to(ai.DEVICE)
                    predictions = net(x).cpu().numpy().reshape(-1)
                if len(predictions) != len(encs):
                    raise ValueError('Checkpoint returned the wrong number of values')
                for k, prediction in enumerate(predictions):
                    value = _finite_value(float(prediction) + float(phis[k]))
                    values[tag][k] = value
                    if value is not None:
                        sources[tag][k] = 'retrospective_checkpoint'

        steps, unmapped = [], 0
        for j, row in enumerate(rows):
            state = row['state']
            context = row.get('decision_context') or {}
            pre_state = context.get('state')
            event_source = 'recorded_predecision'
            if pre_state is None:
                prev_row = prev_of.get(id(row))
                pre_state = prev_row['state'] if prev_row is not None else None
                event_source = 'previous_trace_row' if pre_state is not None else 'unavailable'
            events = []
            if pre_state is not None:
                vp0 = game_logic.calculate_victory_points(pre_state)
                vp1 = game_logic.calculate_victory_points(state)
                for name, delta in (
                        ('FP VP', vp1['free_peoples'] - vp0['free_peoples']),
                        ('SH VP', vp1['shadow'] - vp0['shadow']),
                        ('corr', state.fellowship.corruption - pre_state.fellowship.corruption),
                        ('prog', state.fellowship.progress - pre_state.fellowship.progress)):
                    if delta:
                        events.append('%+d %s' % (delta, name))
            d = by_step.get(step_index[id(row)])
            if d is None:
                unmapped += 1
                reward, paid, value, value_source = 0.0, 0.0, None, 'unavailable'
            else:
                reward = per[d['tag']][1][d['k']]
                paid = d['paid']
                value = values[d['tag']][d['k']]
                value_source = sources[d['tag']][d['k']]
            if abs(paid) > 1e-9:
                events.append('bank%+.2f' % paid)
            params = row.get('params') or {}
            label = row.get('action_type', '?')
            if label == 'card_option' and params.get('card_id') == '__gandalf_draw__':
                choice = (params.get('choices') or {}).get('gandalf_draw')
                verb = ('take' if choice is True else 'decline'
                        if choice is False else 'unknown choice')
                deck = params.get('deck')
                deck = deck if deck in ('character', 'strategy') else 'unknown deck'
                label = 'Gandalf draw: %s (%s)' % (verb, deck)
            else:
                for key in ('to', 'region', 'card_id', 'count'):
                    if params.get(key):
                        short = str(params[key]).replace('FREE_PEOPLES_', '').replace('SHADOW_', '')
                        label += ' %s' % short[:22]
                        break
            steps.append({'i': j, 'turn': row.get('turn'), 'side': row['side'],
                          'label': label, 'reward': float(reward), 'events': events,
                          'event_source': event_source,
                          'regret': _finite_value(row.get('regret')),
                          'v': value, 'value_source': value_source})

        meta = tr.get('meta') or {}
        winner = meta.get('winner')
        undecided = winner not in ('fp', 'sh')
        final_state = all_rows[-1]['state']
        outcomes = {}
        for tag in ('fp', 'sh'):
            if undecided:
                outcomes[tag] = None
            elif uses_predecision(contract):
                outcomes[tag] = terminal(winner, tag, profile=profiles[tag],
                                         victory_type=meta.get('victory_type'),
                                         contract=contract)
            else:
                outcomes[tag] = float(ai._terminal_for(
                    tag, profiles[tag], str(meta.get('victory_type') or ''),
                    winner == tag, final_state))
        lam = trainer.LAM
        _temporal_advantages(steps, outcomes, lam)
        scored = [s for s in steps if s['adv'] is not None]
        modes = {s['value_source'] for s in steps if s['v'] is not None}
        value_mode = next(iter(modes)) if len(modes) == 1 else ('mixed' if modes else 'unavailable')
        return {'steps': steps, 'forced_steps': forced_steps,
                'top_credit': sorted([s for s in scored if s['adv'] > 0],
                                     key=lambda x: -x['adv'])[:10],
                'top_blame': sorted([s for s in scored if s['adv'] < 0],
                                    key=lambda x: x['adv'])[:10],
                'top_regret': sorted([s for s in steps if s['regret'] is not None],
                                     key=lambda x: -x['regret'])[:10],
                'meta': meta, 'nets': nets, 'profiles': profiles,
                'reward_contract': contract, 'value_mode': value_mode,
                'unavailable_values': sum(s['v'] is None for s in steps),
                'lambda': lam, 'lambda_source': 'current_loop_configuration',
                'outcomes': outcomes, 'unmapped': unmapped,
                'untrained_timeout': undecided,
                'stream_extra': len(detail) - (len(rows) - unmapped)}
    finally:
        ai.PLAYER_PROFILES.clear()
        ai.PLAYER_PROFILES.update(old_profiles)
        if old_contract is None:
            os.environ.pop('WOTR_REWARD_CONTRACT', None)
        else:
            os.environ['WOTR_REWARD_CONTRACT'] = old_contract


def show(trace_path, fp_profile=None, sh_profile=None, *, checkpoint_paths=None):
    # Render to PNG + OS viewer: the interactive TkAgg backend silently
    # falls back to Agg in this environment ('does nothing' bug) - a
    # saved image is deterministic and keeps the credit/blame lists
    # visible inside the figure itself.
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    att = compute_attribution(trace_path, fp_profile, sh_profile,
                              checkpoint_paths=checkpoint_paths)
    fp_profile, sh_profile = att['profiles']['fp'], att['profiles']['sh']
    steps = att['steps']
    xs = [s['i'] for s in steps]

    fig, axes = plt.subplots(4, 1, figsize=(15, 13),
                             gridspec_kw={'height_ratios': [2, 2, 2, 3]})
    title = 'Hindsight attribution - %s\nFP=%s (%s)   SH=%s (%s)' \
            '   winner=%s %s' % (
                os.path.basename(trace_path),
                att['profiles']['fp'], att['nets']['fp'],
                att['profiles']['sh'], att['nets']['sh'],
                att['meta'].get('winner'),
                att['meta'].get('victory_type'))
    title += ('\n%s; values=%s; unavailable=%d; lambda=%g (current configuration)'
              % (att['reward_contract'], att['value_mode'],
                 att['unavailable_values'], att['lambda']))
    if att['untrained_timeout']:
        title += '\nUndecided game: provisional bootstrap; no terminal outcome label'
    if att.get('unmapped') or att.get('stream_extra'):
        title += ('\nWARN: view/trainer row filters disagree '
                  '(%d unmapped, %d extra) - semantics drift, fix it'
                  % (att.get('unmapped', 0), att.get('stream_extra', 0)))
    fig.suptitle(title, fontsize=10)

    for tag, col in (('FP', 'tab:blue'), ('SH', 'tab:red')):
        seq = [s for s in steps if s['side'] == tag and s['v'] is not None]
        axes[0].plot([s['i'] for s in seq], [s['v'] for s in seq],
                     col, label='V(%s own view)' % tag, lw=1.2)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].set_ylabel('value')
    axes[0].legend(loc='upper left', fontsize=8)

    for tag, col in (('FP', 'tab:blue'), ('SH', 'tab:red')):
        seq = [s for s in steps if s['side'] == tag]
        axes[1].bar([s['i'] for s in seq], [s['reward'] for s in seq],
                    color=col, alpha=0.7, width=1.0)
    for s in steps:
        if s['events']:
            axes[1].annotate(','.join(s['events']),
                             (s['i'], s['reward']),
                             fontsize=6, rotation=90,
                             ha='center', va='bottom')
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].set_ylabel('step reward')

    for tag, col in (('FP', 'tab:blue'), ('SH', 'tab:red')):
        seq = [s for s in steps if s['side'] == tag and s['adv'] is not None]
        axes[2].bar([s['i'] for s in seq], [s['adv'] for s in seq],
                    color=col, alpha=0.7, width=1.0)
    axes[2].axhline(0, color='gray', lw=0.5)
    axes[2].set_ylabel('advantage\n(credit / blame)')
    axes[2].set_xlabel('decision #')

    # turn boundaries
    last_t = None
    for s in steps:
        if s['turn'] != last_t:
            last_t = s['turn']
            for ax in axes:
                ax.axvline(s['i'], color='lightgray', lw=0.5, zorder=0)

    xmax = max(x['i'] for x in steps) + 1
    for ax in axes[:3]:
        ax.set_xlim(-1, xmax)

    # 4th panel: the credit/blame lists rendered INTO the image
    axes[3].axis('off')
    lines = ['TOP CREDIT (better-than-expected paths)']
    for s2 in att['top_credit']:
        lines.append('  T%-2s #%-3d %-3s adv=%+.3f  %s %s' % (
            s2['turn'], s2['i'], s2['side'], s2['adv'], s2['label'][:44],
            ('[' + ','.join(s2['events']) + ']') if s2['events'] else ''))
    lines.append('')
    lines.append('TOP BLAME (worse-than-expected paths)')
    for s2 in att['top_blame']:
        lines.append('  T%-2s #%-3d %-3s adv=%+.3f  %s %s' % (
            s2['turn'], s2['i'], s2['side'], s2['adv'], s2['label'][:44],
            ('[' + ','.join(s2['events']) + ']') if s2['events'] else ''))
    if att.get('top_regret'):
        lines.append('')
        lines.append("TOP REGRET (worst picks vs search's own best)")
        for s2 in att['top_regret']:
            lines.append('  T%-2s #%-3d %-3s regret=%.3f  %s %s' % (
                s2['turn'], s2['i'], s2['side'], s2['regret'],
                s2['label'][:44],
                ('[' + ','.join(s2['events']) + ']')
                if s2['events'] else ''))
    axes[3].text(0.01, 0.98, '\n'.join(lines), fontsize=8,
                 family='monospace', va='top',
                 transform=axes[3].transAxes)

    plt.tight_layout()
    out_dir = os.path.join(ROOT, 'data', 'attribution')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(trace_path).replace('.trace.gz', '')
    out_png = os.path.join(out_dir, 'attr_%s_fp-%s_sh-%s.png'
                           % (base, fp_profile, sh_profile))
    fig.savefig(out_png, dpi=125)
    plt.close(fig)
    print('attribution image:', out_png)
    try:
        os.startfile(out_png)
    except Exception as e:
        print('open failed (%s) - image is on disk' % e)
    return out_png


if __name__ == '__main__':
    tp = sys.argv[1]
    fp_p = sys.argv[2] if len(sys.argv) > 2 else None
    sh_p = sys.argv[3] if len(sys.argv) > 3 else None
    show(tp, fp_p, sh_p)
