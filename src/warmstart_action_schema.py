"""Initialize final v45 weights from an explicit v44/v2/93 checkpoint.

Twelve zero columns precede the card-embedding suffix in both policy
projections; 434 zero columns extend the observation projection. All original
weights are preserved. This is
weights-only initialization: optimizer, replay, and validation claims are
not migrated, and the new action semantics still require training.
"""
import argparse
import copy
import hashlib
import io
import os
from pathlib import Path

import torch


SOURCE_OBSERVATION = 'v44_fellowship_tail_v1'
SOURCE_ACTION_CONTRACT = 'casualty_operations_v2'
TARGET_ACTION_CONTRACT = 'casualty_card_choices_v3'
TARGET_OBSERVATION = 'v45_deferred_card_context_v1'
TARGET_INPUT_DIM = 4180
SOURCE_INPUT_DIM = 3746
SOURCE_ACTION_DIM = 93
TARGET_ACTION_DIM = 105
EXPANDED_PARAMETERS = ('spatial_action_scorer.0.weight', 'policy_delta.0.weight')


def convert_checkpoint(checkpoint, source_sha256):
    """Return an independent weights-only artifact; never mutate the source."""
    import ai
    from reward_contract import LEGACY, PREDECISION, PROFILE, validate_profile
    from trace_decisions import current_learning_context

    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(character not in '0123456789abcdef' for character in source_sha256)):
        raise ValueError('Source SHA256 must be a complete lowercase hexadecimal digest')
    if not isinstance(checkpoint, dict):
        raise ValueError('Source checkpoint must be a mapping')
    if (checkpoint.get('observation_contract') != SOURCE_OBSERVATION
            or checkpoint.get('action_encoding_contract') != SOURCE_ACTION_CONTRACT):
        raise ValueError('Source must explicitly identify v44 observations and casualty_operations_v2 actions')
    if (type(checkpoint.get('input_dim')) is not int or checkpoint['input_dim'] != SOURCE_INPUT_DIM
            or type(checkpoint.get('action_dim')) is not int or checkpoint['action_dim'] != SOURCE_ACTION_DIM):
        raise ValueError('Source must have input_dim=3746 and action_dim=93')
    context = current_learning_context()
    if (ai.ACT_GEN != 'v45' or ai.ACTION_ENCODING_DIM != TARGET_ACTION_DIM
            or ai.get_feature_dim() != TARGET_INPUT_DIM
            or context.get('observation_contract') != TARGET_OBSERVATION
            or context.get('action_encoding_contract') != TARGET_ACTION_CONTRACT):
        raise ValueError('Select WOTR_ACT_GEN=v45 with the 4180-observation/105-action target schema')
    side = checkpoint.get('side')
    if side not in ('fp', 'sh'):
        raise ValueError('Source checkpoint must identify side fp or sh')
    head_act = checkpoint.get('head_act')
    if head_act not in ('scaled2', 'tanh'):
        raise ValueError('Source must explicitly identify a supported head_act')
    reward = checkpoint.get('reward_contract')
    if reward not in (LEGACY, *PREDECISION):
        raise ValueError('Source must explicitly identify its reward contract')
    source_scope = checkpoint.get('training_scope')
    if source_scope is not None:
        from observation_training import validate_training_scope
        validate_training_scope(source_scope)
        if (source_scope['observation_contract'] != SOURCE_OBSERVATION or
                source_scope['input_dim'] != SOURCE_INPUT_DIM):
            raise ValueError('Source training scope differs from its checkpoint schema')
    profile = checkpoint.get('reward_profile')
    if reward == PROFILE or profile is not None:
        validate_profile(side, profile)
    for field in ('arch', 'arch_name'):
        if field in checkpoint and checkpoint[field] != ai.NET_ARCH_NAME:
            raise ValueError('Source architecture differs; select its WOTR_NET_ARCH')
    states = [checkpoint[key] for key in ('model_state', 'model_state_dict') if key in checkpoint]
    if len(states) != 1 or not isinstance(states[0], dict) or not states[0]:
        raise ValueError('Source must contain exactly one nonempty model state mapping')
    state = states[0]
    net = ai.WOTRValueNet(TARGET_INPUT_DIM, side=side)
    expected = net.state_dict()
    if set(state) != set(expected):
        raise ValueError('Source architecture keys differ from the selected model')
    card_dim = net.card_embedding.embedding_dim
    added = TARGET_ACTION_DIM - SOURCE_ACTION_DIM
    observation_added = TARGET_INPUT_DIM - SOURCE_INPUT_DIM
    observation_parameter = ('tok_global.weight' if net.arch_kind == 'transformer'
                             else 'backbone.0.weight')
    # Validate every shape before allocating converted tensors. Unexpected
    # input shifts, missing delta parameters, or partial migrations fail.
    for key, target in expected.items():
        source = state[key]
        if not isinstance(source, torch.Tensor) or source.layout != torch.strided:
            raise ValueError('Unsupported parameter tensor: %s' % key)
        want = list(target.shape)
        if key in EXPANDED_PARAMETERS:
            if len(want) != 2:
                raise ValueError('Unsupported policy projection shape: %s' % key)
            want[1] -= added
        elif key == observation_parameter:
            want[1] -= observation_added
        if list(source.shape) != want or source.dtype != target.dtype:
            raise ValueError('Unsupported parameter shape or dtype: %s' % key)
        if source.is_floating_point() and not torch.isfinite(source).all():
            raise ValueError('Nonfinite source parameter: %s' % key)
    if source_scope is not None and 'appended_columns_v1' in source_scope['modes']:
        recorded = source_scope['modes']['appended_columns_v1']
        width = state[observation_parameter].shape[1]
        if (recorded['parameter'] != observation_parameter or
                recorded['column_end'] != width or
                recorded['column_start'] != width - (SOURCE_INPUT_DIM - 3742)):
            raise ValueError('Source training scope differs from its projection layout')
    if source_scope is not None and 'full_network_v1' in source_scope['modes']:
        from observation_training import validate_full_parameter_layout
        # Parameter names survive conversion; validate the original widths,
        # not the expanded target's layout. Buffers are not optimizer weights.
        validate_full_parameter_layout(source_scope['modes']['full_network_v1'],
            ((name, state[name]) for name, _ in net.named_parameters()))
    converted, insertions = {}, {}
    for key, source in state.items():
        source = source.detach().cpu()
        if key in EXPANDED_PARAMETERS:
            cut = source.shape[1] - card_dim
            if cut <= SOURCE_ACTION_DIM - 1:
                raise ValueError('Unsupported action feature/card suffix layout')
            converted[key] = torch.cat((source[:, :cut], source.new_zeros((source.shape[0], added)),
                                        source[:, cut:]), dim=1)
            insertions[key] = cut
        elif key == observation_parameter:
            converted[key] = torch.cat((source, source.new_zeros(
                (source.shape[0], observation_added))), dim=1)
            insertions[key] = source.shape[1]
        else:
            converted[key] = source.clone()
    net.load_state_dict(converted, strict=True)
    result = {key: copy.deepcopy(checkpoint[key]) for key in
              ('side', 'head_act', 'reward_contract', 'reward_profile') if key in checkpoint}
    result.update(context)
    result.update(model_state=converted, input_dim=TARGET_INPUT_DIM,
                  action_dim=TARGET_ACTION_DIM, arch_name=net.arch_name,
                  training_status='warm_start_not_retrained', warm_start_updates={'value': 0, 'policy': 0},
                  warm_start={
                      'conversion': 'v44_3746_93_to_v45_4180_105_weights_only_v1',
                      'source_sha256': source_sha256,
                      'source_observation_contract': SOURCE_OBSERVATION,
                      'source_action_encoding_contract': SOURCE_ACTION_CONTRACT,
                      'source_reward_contract': reward,
                      'source_architecture': net.arch_name,
                      'source_training_status': checkpoint.get('training_status'),
                      'source_training_scope': copy.deepcopy(source_scope),
                      'source_warm_start_updates': copy.deepcopy(checkpoint.get('warm_start_updates') or {}),
                      'source_selector_identity': {key: copy.deepcopy(checkpoint[key]) for key in
                          ('selector_contract', 'selector_config', 'selector_config_sha256') if key in checkpoint},
                      'parent_warm_start': copy.deepcopy(checkpoint.get('warm_start')),
                      'expanded_parameters': [*EXPANDED_PARAMETERS, observation_parameter],
                      'insertion_columns': insertions,
                      'new_action_columns': added,
                      'new_observation_columns': observation_added,
                      'new_columns_initialization': 'zero',
                      'card_embedding_suffix_columns': card_dim,
                      'new_action_policy_requires_training': True,
                      'new_observation_context_requires_training': True,
                      'replay_or_optimizer_migrated': False,
                  })
    return result


def convert_file(source, destination):
    """Read one immutable source and create a new file without overwrites."""
    # Resolve before importing ai/board, which can change the process cwd.
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Destination must be a new file separate from the source')
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    converted = convert_checkpoint(checkpoint, digest)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during conversion')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(converted, stream)
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--arch', choices=('base', 'wide', 'xformer'))
    args = parser.parse_args()
    os.environ.setdefault('WOTR_ACT_GEN', 'v45')
    if args.arch:
        os.environ['WOTR_NET_ARCH'] = args.arch
    convert_file(args.source, args.destination)
    print('Created explicit v45 initialization; new action/context features still require training.')


if __name__ == '__main__':
    main()
