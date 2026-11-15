"""Explicit v50-to-v51 observation initialization; existing weights stay exact."""
import argparse
import copy
import hashlib
import io
import os
from pathlib import Path
import torch

SOURCE_CONTRACT = 'v50_route_event_context_v1'
TARGET_CONTRACT = 'v51_breaking_hunt_context_v1'
SOURCE_DIM = 4347
TARGET_DIM = 4350
ACTION_CONTRACT = 'casualty_card_choices_v3'

def convert_checkpoint(checkpoint, source_sha256):
    import ai
    from trace_decisions import current_learning_context, checkpoint_learning_lineage
    from reward_contract import LEGACY, PREDECISION, PROFILE, validate_profile
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('Source SHA256 must be a complete lowercase digest')
    if not isinstance(checkpoint, dict):
        raise ValueError('Source checkpoint must be a mapping')
    if (checkpoint.get('observation_contract') != SOURCE_CONTRACT or
            checkpoint.get('action_encoding_contract') != ACTION_CONTRACT or
            type(checkpoint.get('input_dim')) is not int or checkpoint['input_dim'] != SOURCE_DIM or
            type(checkpoint.get('action_dim')) is not int or checkpoint['action_dim'] != 105):
        raise ValueError('Source must explicitly identify v50 observations and 105 actions')
    if (ai.ACT_GEN != 'v51' or ai.OBSERVATION_CONTRACT != TARGET_CONTRACT or
            ai.get_feature_dim() != TARGET_DIM or ai.ACTION_ENCODING_DIM != 105):
        raise ValueError('Select explicit v51 runtime with 4350 observations and 105 actions')
    side = checkpoint.get('side')
    if side not in ('fp', 'sh') or checkpoint.get('head_act') not in ('scaled2', 'tanh'):
        raise ValueError('Unsupported source side or value head')
    reward = checkpoint.get('reward_contract')
    if reward not in (LEGACY, *PREDECISION):
        raise ValueError('Missing or unsupported source reward contract')
    if reward == PROFILE or checkpoint.get('reward_profile') is not None:
        validate_profile(side, checkpoint.get('reward_profile'))
    if checkpoint.get('ai_policy_contract') != current_learning_context()['ai_policy_contract']:
        raise ValueError('Source fixed-policy contract differs')
    for name in ('arch', 'arch_name'):
        if name in checkpoint and checkpoint[name] != ai.NET_ARCH_NAME:
            raise ValueError('Select the original source architecture')
    states = [checkpoint[k] for k in ('model_state','model_state_dict') if k in checkpoint]
    if len(states) != 1 or not isinstance(states[0], dict):
        raise ValueError('Source must have one model-state mapping')
    source_state = states[0]
    target = ai.WOTRValueNet(TARGET_DIM, side=side)
    reference = ai.WOTRValueNet(SOURCE_DIM, side=side)
    expected = reference.state_dict()
    if set(source_state) != set(expected):
        raise ValueError('Source model keys differ from selected architecture')
    for key, tensor in expected.items():
        actual = source_state[key]
        if (not isinstance(actual, torch.Tensor) or actual.layout != torch.strided or
                actual.shape != tensor.shape or actual.dtype != tensor.dtype or
                (actual.is_floating_point() and not torch.isfinite(actual).all())):
            raise ValueError('Invalid source parameter: ' + key)
    reference.load_state_dict(source_state, strict=True)
    reference.observation_contract = SOURCE_CONTRACT
    reference.warm_start = copy.deepcopy(checkpoint.get('warm_start'))
    reference.training_status = checkpoint.get('training_status')
    reference.warm_start_updates = copy.deepcopy(checkpoint.get('warm_start_updates') or {})
    reference.training_scope = copy.deepcopy(checkpoint.get('training_scope'))
    checkpoint_learning_lineage(reference)
    projection = 'tok_global.weight' if target.arch_kind == 'transformer' else 'backbone.0.weight'
    converted = {key: tensor.detach().cpu().clone() for key, tensor in source_state.items()}
    weight = converted[projection]
    converted[projection] = torch.cat((weight, weight.new_zeros((weight.shape[0],3))), dim=1)
    target.load_state_dict(converted, strict=True)
    result = {key:copy.deepcopy(checkpoint[key]) for key in ('side','head_act','reward_contract','reward_profile') if key in checkpoint}
    result.update(current_learning_context(), model_state=converted, input_dim=TARGET_DIM,
                  action_dim=105, arch_name=target.arch_name, training_status='warm_start_not_retrained',
                  warm_start_updates={'value':0,'policy':0}, warm_start={
                      'conversion':'v50_4347_to_v51_4350_weights_only_v1', 'source_sha256':source_sha256,
                      'source_observation_contract':SOURCE_CONTRACT,
                      'source_training_scope':copy.deepcopy(checkpoint.get('training_scope')),
                      'source_training_status':checkpoint.get('training_status'),
                      'source_selector_identity': {key: copy.deepcopy(checkpoint[key]) for key in
                          ('selector_contract', 'selector_config', 'selector_config_sha256') if key in checkpoint},
                      'source_warm_start_updates':copy.deepcopy(checkpoint.get('warm_start_updates') or {}),
                      'parent_warm_start':copy.deepcopy(checkpoint.get('warm_start')),
                      'expanded_parameter':projection,'insertion_column':weight.shape[1],
                      'new_observation_columns':3,'new_columns_initialization':'zero',
                      'new_observation_context_requires_training':True,'replay_or_optimizer_migrated':False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--arch', choices=('base', 'wide', 'xformer'))
    args = parser.parse_args()
    os.environ.setdefault('WOTR_ACT_GEN', 'v51')
    if args.arch:
        os.environ['WOTR_NET_ARCH'] = args.arch
    convert_file(args.source, args.destination)
    print('Created explicit v51 initialization; Breaking protection source features still require training.')


def convert_file(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Destination must be a new separate checkpoint')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    result = convert_checkpoint(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),digest)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during conversion')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(result,stream)
    return result


if __name__ == '__main__':
    main()
