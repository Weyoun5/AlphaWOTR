"""Offline v52 v5/v6 weights -> exact-cargo v7/v3, with explicit fresh Adam.

This migrates checkpoint initialization only. Historical data is never relabeled.
Run with the source architecture, selector, reward and value-head environment.
"""
import argparse
import copy
import hashlib
import io
from pathlib import Path

import torch

SOURCE_STATE = 'v52_observed_combat_round_v2'
SOURCE_ACTIONS = ('observed_combat_choices_v5', 'observed_combat_choices_v6')
SOURCE_STATE_DIM, SOURCE_ACTION_DIM = 4481, 133
VERSION = 'v52_card_choices_v7_cargo_v3_explicit_adam_reset_v1'


def reset_action_columns(contract):
    """Conservative reset of shared scalar lanes whose uses changed in v6.

    Card embedding identities are still identities; keep the shared embedding.
    Rage's elite counts and Lidless' histogram reuse columns 93..100. Worn's
    three sources and Guards' unit choice add meanings to the option scalars.
    """
    import ai
    if contract not in SOURCE_ACTIONS:
        raise ValueError('Unrecognized source action semantics')
    return sorted({ai._ACT_OPTVAL_OFF, ai._ACT_OPTKIND_OFF, *range(93, 101)}) if contract.endswith('_v5') else []


def expand_tensor(value, name, target, source_contract):
    import ai
    out = value.detach().cpu().clone()
    if name in ('backbone.0.weight', 'tok_global.weight'):
        return torch.cat((out, out.new_zeros((out.shape[0], 166))), dim=1)
    if name in ('spatial_action_scorer.0.weight', 'policy_delta.0.weight'):
        card_dim = target.card_embedding.embedding_dim
        cut = out.shape[1] - card_dim
        start = cut - (SOURCE_ACTION_DIM - 1)
        for raw_column in reset_action_columns(source_contract):
            assert raw_column != ai._ACTION_CARD_IDX_DIM
            out[:, start + raw_column - (raw_column > ai._ACTION_CARD_IDX_DIM)] = 0
        return torch.cat((out[:, :cut], out.new_zeros((out.shape[0], 24)), out[:, cut:]), dim=1)
    return out


def convert_checkpoint(checkpoint, source_sha256, *, reset_optimizer=False):
    import ai
    import optimizer_continuity as oc
    from trace_decisions import current_learning_context, checkpoint_learning_lineage
    from warmstart_combat_schema import _attach
    if not reset_optimizer:
        raise ValueError('Schema migration requires explicit reset_optimizer=True')
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('A complete source SHA256 is required')
    if (not isinstance(checkpoint, dict) or checkpoint.get('observation_contract') != SOURCE_STATE
            or checkpoint.get('action_encoding_contract') not in SOURCE_ACTIONS
            or checkpoint.get('input_dim') != SOURCE_STATE_DIM
            or checkpoint.get('action_dim') != SOURCE_ACTION_DIM
            or 'model_state_dict' in checkpoint or not isinstance(checkpoint.get('model_state'), dict)):
        raise ValueError('Source must explicitly identify v52/v2 and action v5 or v6 at 4481/133')
    if (ai.ACT_GEN != 'v52' or ai.OBSERVATION_CONTRACT != 'v52_observed_combat_round_v3'
            or ai.get_feature_dim() != 4647 or ai.ACTION_ENCODING_DIM != 157):
        raise ValueError('Explicit v7/v3 4647/157 runtime required')
    if checkpoint.get('side') not in ('fp', 'sh'):
        raise ValueError('Explicit source side required')
    if any(checkpoint.get(k, ai.NET_ARCH_NAME) != ai.NET_ARCH_NAME for k in ('arch', 'arch_name')):
        raise ValueError('Select source architecture explicitly')
    # Offline, single-threaded shape construction. Restore width even on error;
    # ordinary checkpoint/data loaders continue enforcing current contracts.
    with torch.random.fork_rng(devices=[]):
        target = ai.WOTRValueNet(4647, side=checkpoint['side'])
        try:
            ai.ACTION_ENCODING_DIM = SOURCE_ACTION_DIM
            reference = ai.WOTRValueNet(SOURCE_STATE_DIM, side=checkpoint['side'])
        finally:
            ai.ACTION_ENCODING_DIM = 157
    source = checkpoint['model_state']
    expected = reference.state_dict()
    if set(source) != set(expected):
        raise ValueError('Source architecture parameter keys differ')
    for name, template in expected.items():
        value = source[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype != template.dtype or value.shape != template.shape
                or (value.is_floating_point() and not torch.isfinite(value).all())):
            raise ValueError('Invalid source tensor: ' + name)
    _attach(reference, checkpoint)
    checkpoint_learning_lineage(reference)
    metadata = copy.deepcopy(checkpoint)
    metadata.update(current_learning_context())
    _attach(target, metadata)
    # Training scope describes historical optimizer work; keep it only in the
    # lineage. Fresh steps on the new schema must start from zero.
    if hasattr(target, 'training_scope'):
        del target.training_scope
    identity = oc._identity(target)
    for key in ('side', 'head_act', 'reward_contract', 'selector_contract', 'selector_config', 'selector_config_sha256'):
        if checkpoint.get(key) != identity[key]:
            raise ValueError('Source runtime contract mismatch: ' + key)
    if checkpoint.get('reward_profile') != identity['profile']:
        raise ValueError('Source reward profile differs')
    if checkpoint.get('ai_policy_contract') not in (identity['ai_policy_contract'], 'elite_and_bonus_draw_v1', 'elite_bonus_draw_combat_victim_v2'):
        raise ValueError('Unknown source fixed-policy contract')
    old_optimizer = checkpoint.get(oc.FIELD)
    if old_optimizer is not None:
        if (not isinstance(old_optimizer, dict) or set(old_optimizer) != {'version', 'sha256', 'blob'}
                or old_optimizer['version'] != oc.VERSION or not isinstance(old_optimizer['blob'], bytes)
                or hashlib.sha256(old_optimizer['blob']).hexdigest() != old_optimizer['sha256']):
            raise ValueError('Invalid source optimizer envelope')
        inner = torch.load(io.BytesIO(old_optimizer['blob']), map_location='cpu', weights_only=True)
        if (not isinstance(inner, dict) or set(inner) != {'version', 'model_sha256', 'record'}
                or inner.get('version') != oc.VERSION or inner.get('model_sha256') != oc._model_hash(source)):
            raise ValueError('Source optimizer model binding mismatch')
    converted = {name: expand_tensor(value, name, target, checkpoint['action_encoding_contract']) for name, value in source.items()}
    target.load_state_dict(converted, strict=True)
    result = copy.deepcopy(checkpoint)
    result.pop(oc.FIELD, None)
    result.pop('training_scope', None)
    result.update(current_learning_context(), input_dim=4647, action_dim=157, model_state=converted,
        training_status='warm_start_not_retrained', warm_start_updates={'value': 0, 'policy': 0},
        warm_start={'conversion': VERSION, 'source_sha256': source_sha256,
            'source_observation_contract': SOURCE_STATE,
            'source_action_encoding_contract': checkpoint['action_encoding_contract'],
            'source_ai_policy_contract': checkpoint.get('ai_policy_contract'),
            'source_training_scope': copy.deepcopy(checkpoint.get('training_scope')),
            'source_training_status': checkpoint.get('training_status'),
            'source_warm_start_updates': copy.deepcopy(checkpoint.get('warm_start_updates')),
            'parent_warm_start': copy.deepcopy(checkpoint.get('warm_start')),
            'observation_columns_added': 166, 'action_columns_added': 24,
            'action_columns_reset': reset_action_columns(checkpoint['action_encoding_contract']),
            'new_columns_initialization': 'zero', 'new_context_requires_training': True,
            'optimizer_migrated': False, 'optimizer_reset': 'explicit_fresh_optimizer',
            'source_optimizer_sha256': old_optimizer['sha256'] if old_optimizer else None,
            'historical_rows_migrated': False})
    _attach(target, result)
    target._optimizer_continuity = {'identity': oc._identity(target), 'start': 'explicit_fresh_optimizer',
                                   'components': {'value': None, 'policy': None}}
    oc._validate_record(target, target._optimizer_continuity)
    # Always persist the audited fresh record, even when the offline process's
    # optimizer mode is reset_each_call. A resumed persistent run is unambiguous.
    inner = {'version': oc.VERSION, 'model_sha256': oc._model_hash(converted), 'record': target._optimizer_continuity}
    stream = io.BytesIO(); torch.save(inner, stream); blob = stream.getvalue()
    result[oc.FIELD] = {'version': oc.VERSION, 'sha256': hashlib.sha256(blob).hexdigest(), 'blob': blob}
    checkpoint_learning_lineage(target)
    return result


def convert_file(source, destination, expected_sha256, *, reset_optimizer=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Destination must be a new separate checkpoint')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Source bytes differ from explicit expected SHA256')
    result = convert_checkpoint(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), digest, reset_optimizer=reset_optimizer)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during conversion')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(result, stream)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--reset-optimizer', required=True, action='store_true')
    args = parser.parse_args()
    convert_file(args.source, args.destination, args.source_sha256, reset_optimizer=args.reset_optimizer)
    print('Created v7/v3 warm-start checkpoint with fresh Adam; fresh extraction and retraining required.')
