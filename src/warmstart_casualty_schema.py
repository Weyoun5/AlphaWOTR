"""Explicit v41/v42/v43 weight initialization for current decision semantics.

This is an initialization artifact, not evidence that the new casualty policy
has been trained. Existing checkpoints and optimizer/replay state stay intact.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import torch

OLD_OBSERVATION = 'v41_ring_rearguard_state_v1'
COMBAT_OBSERVATION = 'v42_casualty_operations_v1'
CARD_OBSERVATION = 'v43_card_casualty_operations_v1'


def convert_checkpoint(checkpoint, source_sha256):
    import ai
    from trace_decisions import current_learning_context
    observation = checkpoint.get('observation_contract')
    historical_dimensions = {OLD_OBSERVATION: 3742, COMBAT_OBSERVATION: 3745,
                             CARD_OBSERVATION: 3745}
    if observation not in historical_dimensions:
        raise ValueError('Warm start requires an explicitly identified v41/v42/v43 observation checkpoint')
    if observation == COMBAT_OBSERVATION and checkpoint.get('action_encoding_contract') != 'casualty_operations_v1':
        raise ValueError('v42 warm start requires its original action contract')
    if observation == CARD_OBSERVATION and checkpoint.get('action_encoding_contract') != 'casualty_operations_v2':
        raise ValueError('v43 warm start requires its original action contract')
    old_dim = checkpoint.get('input_dim')
    new_dim = ai.get_feature_dim()
    added = new_dim - historical_dimensions[observation]
    if added <= 0 or old_dim != historical_dimensions[observation] or checkpoint.get('action_dim') != ai.ACTION_ENCODING_DIM:
        raise ValueError('Unexpected source input/action dimensions')
    state = checkpoint.get('model_state') or checkpoint.get('model_state_dict')
    if not isinstance(state, dict):
        raise ValueError('Missing model state')
    net = ai.WOTRValueNet(new_dim, side=checkpoint.get('side'))
    expected = net.state_dict()
    if set(state) != set(expected):
        raise ValueError('Checkpoint architecture differs; select its WOTR_NET_ARCH')
    expandable = 'tok_global.weight' if net.arch_kind == 'transformer' else 'backbone.0.weight'
    converted = {}
    expanded = []
    for key, target in expected.items():
        source = state[key]
        if source.shape == target.shape:
            converted[key] = source.detach().clone()
        elif (key == expandable and source.ndim == 2 and
              source.shape[0] == target.shape[0] and source.shape[1] + added == target.shape[1]):
            converted[key] = torch.cat([source.detach().clone(), source.new_zeros((source.shape[0], added))], dim=1)
            expanded.append(key)
        else:
            raise ValueError('Unsupported parameter shape change: %s' % key)
    if expanded != ([expandable] if added else []):
        raise ValueError('Unexpected expanded input projections')
    net.load_state_dict(converted, strict=True)
    # Explicit allowlist excludes old optimizer state, replay, training steps,
    # selector-target claims, and validation scores from the new artifact.
    result = {key: checkpoint[key] for key in
              ('side', 'arch', 'arch_name', 'head_act', 'reward_contract', 'reward_profile')
              if key in checkpoint}
    result.update(current_learning_context())
    result.update(input_dim=new_dim, action_dim=ai.ACTION_ENCODING_DIM,
                  model_state=converted, training_status='warm_start_not_retrained',
                  warm_start={
                      'source_sha256': source_sha256,
                      'source_observation_contract': checkpoint['observation_contract'],
                      'source_action_encoding_contract': checkpoint.get('action_encoding_contract', 'legacy'),
                      'source_reward_contract': checkpoint.get('reward_contract'),
                      'expanded_parameter': expandable if added else None,
                      'new_input_columns': added,
                      'new_columns_initialization': 'zero' if added else 'not_applicable',
                      'casualty_policy_requires_training': True,
                      'fellowship_tail_policy_requires_training': True,
                      'replay_or_optimizer_migrated': False,
                      'parent_warm_start': copy.deepcopy(checkpoint.get('warm_start')),
                      'source_training_status': checkpoint.get('training_status'),
                      'source_warm_start_updates': copy.deepcopy(checkpoint.get('warm_start_updates') or {}),
                  })
    return result


def main():
    os.environ.setdefault('WOTR_ACT_GEN', 'v41')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or destination.exists():
        parser.error('Destination must be a new file, separate from the source')
    data = source.read_bytes()
    checkpoint = torch.load(source, weights_only=True, map_location='cpu')
    result = convert_checkpoint(checkpoint, hashlib.sha256(data).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save(result, stream)
    print('Created warm-start weights; new casualty policy has not been retrained: %s' % destination)


if __name__ == '__main__':
    main()
