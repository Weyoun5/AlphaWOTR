"""Explicit, non-overwriting v51 -> v52 weights and persistent Adam migration.

Run offline in a v52 runtime with the source architecture, reward, selector,
head activation and optimizer scope selected. Historical arrays are untouched.
"""
import argparse
import copy
import hashlib
import io
import os
from pathlib import Path
import torch
from combat_round_observation import (OBSERVATION_CONTRACT, ACTION_CONTRACT,
    SOURCE_STATE_DIM, SOURCE_ACTION_DIM, TARGET_STATE_DIM, TARGET_ACTION_DIM,
    STATE_TAIL_DIM, ACTION_TAIL_DIM)

SOURCE_CONTRACT = 'v51_breaking_hunt_context_v1'
SOURCE_ACTION_CONTRACT = 'casualty_card_choices_v3'
SOURCE_POLICY_CONTRACT = 'elite_and_bonus_draw_v1'
VERSION = 'v51_to_v52_observed_combat_adam_v1'

def _expand(tensor, name, card_dim):
    if name in ('backbone.0.weight', 'tok_global.weight'):
        cut, width = tensor.shape[1], STATE_TAIL_DIM
    elif name in ('spatial_action_scorer.0.weight', 'policy_delta.0.weight'):
        cut, width = tensor.shape[1] - card_dim, ACTION_TAIL_DIM
    else:
        return tensor.detach().cpu().clone()
    return torch.cat((tensor[:, :cut], tensor.new_zeros((tensor.shape[0], width)),
                      tensor[:, cut:]), dim=1).detach().cpu()

def _attach(net, checkpoint):
    for key in ('observation_contract', 'action_encoding_contract', 'reward_contract',
                'reward_profile', 'training_scope', 'warm_start', 'training_status',
                'warm_start_updates'):
        if key in checkpoint:
            setattr(net, key, copy.deepcopy(checkpoint[key]))

def _source_optimizer(checkpoint, reference, target):
    import optimizer_continuity as oc
    outer = checkpoint.get(oc.FIELD)
    if outer is None:
        return None
    if (not isinstance(outer, dict) or set(outer) != {'version', 'sha256', 'blob'}
            or outer['version'] != oc.VERSION or not isinstance(outer['blob'], bytes)
            or hashlib.sha256(outer['blob']).hexdigest() != outer['sha256']):
        raise ValueError('Invalid source Adam envelope')
    inner = torch.load(io.BytesIO(outer['blob']), map_location='cpu', weights_only=True)
    if (set(inner) != {'version', 'model_sha256', 'record'} or inner['version'] != oc.VERSION
            or inner['model_sha256'] != oc._model_hash(checkpoint['model_state'])):
        raise ValueError('Source Adam model binding mismatch')
    record = inner['record']
    expected = oc._identity(target)
    expected.update(input_dim=SOURCE_STATE_DIM, action_dim=SOURCE_ACTION_DIM,
                    ai_policy_contract=SOURCE_POLICY_CONTRACT,
                    observation_contract=SOURCE_CONTRACT,
                    action_encoding_contract=SOURCE_ACTION_CONTRACT,
                    model_layout=[[n, list(p.shape), str(p.dtype)] for n,p in reference.named_parameters()])
    if (not isinstance(record, dict) or set(record) != {'identity','start','components'}
            or record['identity'] != expected or record['start'] != 'explicit_fresh_optimizer'
            or set(record['components']) != {'value','policy'}):
        raise ValueError('Source Adam identity or component mismatch')
    for component, entry in record['components'].items():
        if entry is None:
            continue
        layout = oc._layout(reference, oc._groups(reference, component))
        if (set(entry) != {'layout','state','steps','initialized'} or entry['layout'] != layout
                or type(entry['steps']) is not int or entry['steps'] < 0):
            raise ValueError('Source Adam layout/counter mismatch')
        oc._validate_state(entry['state'], layout)
        states = entry['state']['state']
        if (entry['initialized'] != sorted(states) or bool(states) != bool(entry['steps'])
                or any(float(s['step']) > entry['steps'] for s in states.values())):
            raise ValueError('Source Adam initialized/step mismatch')
    return copy.deepcopy(record)

def convert_checkpoint(checkpoint, source_sha256):
    if (TARGET_STATE_DIM, TARGET_ACTION_DIM) != (4481, 133):
        raise ValueError('This historical v51-to-v52 migration does not cover exact cargo or revised card choices; initialize a fresh current-contract model or use an explicitly audited migration')
    import ai
    import optimizer_continuity as oc
    from trace_decisions import current_learning_context, checkpoint_learning_lineage
    from observation_training import validate_training_scope, full_parameter_layout
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('A complete source SHA256 is required')
    if (not isinstance(checkpoint, dict) or checkpoint.get('observation_contract') != SOURCE_CONTRACT
            or checkpoint.get('action_encoding_contract') != SOURCE_ACTION_CONTRACT
            or checkpoint.get('input_dim') != SOURCE_STATE_DIM
            or checkpoint.get('action_dim') != SOURCE_ACTION_DIM
            or 'model_state_dict' in checkpoint or not isinstance(checkpoint.get('model_state'), dict)):
        raise ValueError('Source must explicitly identify the v51 model/state/action contracts')
    if (ai.ACT_GEN != 'v52' or ai.OBSERVATION_CONTRACT != OBSERVATION_CONTRACT
            or ai.get_feature_dim() != TARGET_STATE_DIM or ai.ACTION_ENCODING_DIM != TARGET_ACTION_DIM):
        raise ValueError('Explicit v52 runtime required')
    if checkpoint.get('side') not in ('fp','sh') or checkpoint.get('head_act') not in ('scaled2','tanh'):
        raise ValueError('Unsupported source side/head')
    if checkpoint['head_act'] != os.environ.get('WOTR_VALUE_HEAD_ACT', 'scaled2'):
        raise ValueError('Select source value head activation explicitly')
    if any(checkpoint.get(k, ai.NET_ARCH_NAME) != ai.NET_ARCH_NAME for k in ('arch','arch_name')):
        raise ValueError('Select source architecture explicitly')
    if checkpoint.get('ai_policy_contract') != SOURCE_POLICY_CONTRACT:
        raise ValueError('Source fixed-policy contract differs')
    # Offline shape reference only. No loader is relaxed; the temporary width
    # is restored before any model validation and this command is single-threaded.
    with torch.random.fork_rng(devices=[]):
        target = ai.WOTRValueNet(TARGET_STATE_DIM, side=checkpoint['side'])
        try:
            ai.ACTION_ENCODING_DIM = SOURCE_ACTION_DIM
            reference = ai.WOTRValueNet(SOURCE_STATE_DIM, side=checkpoint['side'])
        finally:
            ai.ACTION_ENCODING_DIM = TARGET_ACTION_DIM
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
    reference.load_state_dict(source, strict=True)
    _attach(reference, checkpoint)
    checkpoint_learning_lineage(reference)
    target_metadata = copy.deepcopy(checkpoint)
    target_metadata.update(current_learning_context(), input_dim=TARGET_STATE_DIM,
                           action_dim=TARGET_ACTION_DIM)
    _attach(target, target_metadata)
    # Strict identity additionally verifies reward/profile/selector/architecture
    # against the selected runtime, whether or not Adam was persisted.
    identity = oc._identity(target)
    for key in ('side','head_act','reward_contract',
                'selector_contract','selector_config','selector_config_sha256'):
        if checkpoint.get(key) != identity[key]:
            raise ValueError('Source runtime contract mismatch: ' + key)
    if checkpoint.get('reward_profile') != identity['profile']:
        raise ValueError('Source reward profile differs')
    record = _source_optimizer(checkpoint, reference, target)
    card_dim = target.card_embedding.embedding_dim
    converted = {name:_expand(value,name,card_dim) for name,value in source.items()}
    target.load_state_dict(converted, strict=True)
    result = copy.deepcopy(checkpoint)
    result.update(current_learning_context(), model_state=converted,
                  input_dim=TARGET_STATE_DIM, action_dim=TARGET_ACTION_DIM,
                  training_status='warm_start_not_retrained', warm_start_updates={'value':0,'policy':0},
                  warm_start={'conversion':VERSION, 'source_sha256':source_sha256,
                      'source_observation_contract':SOURCE_CONTRACT,
                      'source_action_encoding_contract':SOURCE_ACTION_CONTRACT,
                      'source_ai_policy_contract':SOURCE_POLICY_CONTRACT,
                      'source_training_scope':copy.deepcopy(checkpoint.get('training_scope')),
                      'source_training_status':checkpoint.get('training_status'),
                      'source_warm_start_updates':copy.deepcopy(checkpoint.get('warm_start_updates')),
                      'parent_warm_start':copy.deepcopy(checkpoint.get('warm_start')),
                      'observation_columns_added':STATE_TAIL_DIM,'action_columns_added':ACTION_TAIL_DIM,
                      'new_columns_initialization':'zero','new_context_requires_training':True,
                      'optimizer_migrated':record is not None,'historical_rows_migrated':False})
    scope = result.get('training_scope')
    if scope is not None:
        scope.update(observation_contract=OBSERVATION_CONTRACT, input_dim=TARGET_STATE_DIM)
        if 'full_network_v1' in scope['modes']:
            scope['modes']['full_network_v1'].update(full_parameter_layout(target.named_parameters()))
        if 'appended_columns_v1' in scope['modes']:
            scope['modes']['appended_columns_v1']['raw_end'] += STATE_TAIL_DIM
            scope['modes']['appended_columns_v1']['column_end'] += STATE_TAIL_DIM
        target.training_scope = scope
        validate_training_scope(scope, net=target)
    _attach(target, result)
    if record is not None:
        if oc.mode() != oc.VERSION:
            raise ValueError('Persistent Adam source requires explicit persistent_adam_v1 mode')
        record['identity'] = oc._identity(target)
        for component, entry in record['components'].items():
            if entry is None:
                continue
            names = [item[0] for group in entry['layout'] for item in group]
            for index, state in entry['state']['state'].items():
                for moment in ('exp_avg','exp_avg_sq'):
                    state[moment] = _expand(state[moment], names[index], card_dim)
            entry['layout'] = oc._layout(target, oc._groups(target,component))
        target._optimizer_continuity = record
        oc._validate_record(target, record)
        result.pop(oc.FIELD,None)
        result.update(oc.checkpoint_payload(target,result))
        oc.restore_checkpoint(target,result)
    checkpoint_learning_lineage(target)
    return result

def convert_file(source, destination, expected_sha256):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Destination must be a new separate checkpoint')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Source bytes differ from explicit expected SHA256')
    result = convert_checkpoint(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),digest)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during conversion')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(result,stream)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args()
    convert_file(args.source,args.destination,args.source_sha256)
    print('Created explicit v52 warm start; new combat inputs still require training.')

if __name__ == '__main__':
    main()
