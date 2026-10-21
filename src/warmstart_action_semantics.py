"""Explicit v52 action-v4 to action-v5 initialization, without relabeling data.

No parameter widths change. All weights and Adam moments remain exact, while
the new semantics are explicitly marked untrained. Run in the selected v52
architecture/reward/selector/optimizer runtime; never overwrite a checkpoint.
"""
import argparse
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import torch

SOURCE_ACTION = 'observed_combat_choices_v4'
TARGET_ACTION = 'observed_combat_choices_v5'
SOURCE_POLICY = 'elite_bonus_draw_combat_victim_v2'
TARGET_POLICY = 'elite_bonus_draw_combat_victim_mighty_v3'
VERSION = 'v52_action_v4_to_v5_same_width_warmstart_v1'


def convert_checkpoint(checkpoint, source_sha256):
    import ai
    import optimizer_continuity as oc
    from trace_decisions import current_learning_context, checkpoint_learning_lineage
    from warmstart_combat_schema import _attach
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('Explicit complete source SHA256 required')
    context = current_learning_context()
    if (ai.ACT_GEN != 'v52' or ai.get_feature_dim() != 4481 or ai.ACTION_ENCODING_DIM != 133
            or context['action_encoding_contract'] != TARGET_ACTION
            or context['ai_policy_contract'] != TARGET_POLICY):
        raise ValueError('Select current v52 action-v5 runtime')
    if (not isinstance(checkpoint, dict) or checkpoint.get('input_dim') != 4481
            or checkpoint.get('action_dim') != 133
            or checkpoint.get('action_encoding_contract') != SOURCE_ACTION
            or checkpoint.get('ai_policy_contract') != SOURCE_POLICY
            or checkpoint.get('observation_contract') != context['observation_contract']
            or checkpoint.get('side') not in ('fp', 'sh')
            or 'model_state_dict' in checkpoint
            or not isinstance(checkpoint.get('model_state'), dict)):
        raise ValueError('Source must explicitly identify v52 action-v4/policy-v2 checkpoint')
    with torch.random.fork_rng(devices=[]):
        net = ai.WOTRValueNet(4481, side=checkpoint['side'])
    state = checkpoint['model_state']
    template = net.state_dict()
    if set(state) != set(template):
        raise ValueError('Source architecture keys differ')
    for key, expected in template.items():
        tensor = state[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != expected.shape or tensor.dtype != expected.dtype
                or tensor.is_floating_point() and not torch.isfinite(tensor).all()):
            raise ValueError('Invalid source tensor: ' + key)
    net.load_state_dict(state, strict=True)
    _attach(net, checkpoint)
    checkpoint_learning_lineage(net)
    result = copy.deepcopy(checkpoint)
    result.update(context)
    _attach(net, result)
    identity = oc._identity(net)
    for key in ('side','head_act','reward_contract','selector_contract',
                'selector_config','selector_config_sha256'):
        if checkpoint.get(key) != identity[key]:
            raise ValueError('Source runtime identity differs: ' + key)
    if checkpoint.get('reward_profile') != identity['profile']:
        raise ValueError('Source reward profile differs')
    for key in ('arch','arch_name'):
        if key in checkpoint and checkpoint[key] != net.arch_name:
            raise ValueError('Source architecture differs')
    record = None
    envelope = checkpoint.get(oc.FIELD)
    if envelope is not None:
        if (not isinstance(envelope,dict) or set(envelope) != {'version','sha256','blob'}
                or envelope['version'] != oc.VERSION or not isinstance(envelope['blob'],bytes)
                or hashlib.sha256(envelope['blob']).hexdigest() != envelope['sha256']):
            raise ValueError('Invalid source Adam envelope')
        payload = torch.load(io.BytesIO(envelope['blob']),map_location='cpu',weights_only=True)
        if (set(payload) != {'version','model_sha256','record'} or payload['version'] != oc.VERSION
                or payload['model_sha256'] != oc._model_hash(state)):
            raise ValueError('Source Adam model binding differs')
        record = payload['record']
        old_identity = identity | {'action_encoding_contract':SOURCE_ACTION,'ai_policy_contract':SOURCE_POLICY}
        if not isinstance(record,dict) or record.get('identity') != old_identity:
            raise ValueError('Source Adam identity differs')
        record = copy.deepcopy(record)
        record['identity'] = identity
        oc._validate_record(net,record)
        if oc.mode() != oc.VERSION:
            raise ValueError('Select persistent_adam_v1 for an Adam-bearing source')
    result.update(training_status='warm_start_not_retrained',
        warm_start_updates={'value':0,'policy':0},
        warm_start={'conversion':VERSION,'source_sha256':source_sha256,
            'source_action_encoding_contract':SOURCE_ACTION,'source_ai_policy_contract':SOURCE_POLICY,
            'source_observation_contract':checkpoint['observation_contract'],
            'source_training_status':checkpoint.get('training_status'),
            'source_warm_start_updates':copy.deepcopy(checkpoint.get('warm_start_updates')),
            'source_training_scope':copy.deepcopy(checkpoint.get('training_scope')),
            'parent_warm_start':copy.deepcopy(checkpoint.get('warm_start')),
            'weights_unchanged':True,'observation_columns_added':0,'action_columns_added':0,
            'new_semantics_require_training':True,'historical_rows_migrated':False,
            'optimizer_migrated':record is not None,'optimizer_moments_unchanged':record is not None,
            'semantic_changes':['WWSAT explicit source/decline action identity','Mighty fixed legal payer policy']})
    if result.get('training_scope'):
        for entry in result['training_scope']['modes'].values():
            for key in ('value_steps','policy_steps'):
                if key in entry:
                    entry[key] = 0
    _attach(net,result)
    checkpoint_learning_lineage(net)
    if record is not None:
        net._optimizer_continuity = record
        result.pop(oc.FIELD,None)
        result.update(oc.checkpoint_payload(net,result))
        oc.restore_checkpoint(net,result)
    return result


def convert_file(source,destination,expected_sha256):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError('Destination must be new and separate')
    raw=source.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Source differs from explicit SHA256')
    result=convert_checkpoint(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True),digest)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Source changed during conversion')
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=destination.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(result,stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive, atomic publication on the same filesystem; no overwrite.
        os.link(temp,destination)
    finally:
        Path(temp).unlink(missing_ok=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args()
    convert_file(args.source,args.destination,args.source_sha256)
    print('Created explicit action-v5 warm start; weights unchanged, new semantics untrained.')
