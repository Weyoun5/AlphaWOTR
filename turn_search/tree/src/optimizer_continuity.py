"""Opt-in per-model Adam continuation, separate from paired-batch recovery.

Inference reads the embedded bytes but never decodes optimizer tensors. Only
training loads decode the CPU payload, after hash/version checks. No sidecars.
"""
import copy
import hashlib
import io
import math
import os

VERSION = 'persistent_adam_v1'
RESET = 'reset_per_call_v1'
FIELD = 'optimizer_continuity'


def mode():
    value = os.environ.get('WOTR_OPTIMIZER_STATE', RESET)
    if value not in (RESET, VERSION):
        raise ValueError('Unknown optimizer continuity mode: %r' % value)
    return value


def _fresh_allowed():
    value = os.environ.get('WOTR_OPTIMIZER_ALLOW_FRESH', '0')
    if value not in ('0', '1'):
        raise ValueError('WOTR_OPTIMIZER_ALLOW_FRESH must be 0 or 1')
    return value == '1'


def _number(value, name, *, positive=False):
    import numbers
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError('Invalid optimizer ' + name)
    return float(value)


def _identity(net):
    import ai
    from reward_contract import current_contract, PROFILE, validate_profile
    from trace_decisions import current_learning_context
    from selector_contract import current_identity
    side = getattr(net, 'side', None)
    if side not in ('fp', 'sh'):
        raise ValueError('Persistent optimizer requires explicit model side')
    context = current_learning_context()
    for key in ('observation_contract', 'action_encoding_contract'):
        if getattr(net, key, None) != context[key]:
            raise ValueError('Persistent optimizer model ' + key + ' mismatch')
    reward = current_contract()
    if getattr(net, 'reward_contract', None) != reward:
        raise ValueError('Persistent optimizer model reward contract mismatch')
    profile = getattr(net, 'reward_profile', None)
    if reward == PROFILE:
        validate_profile(side, profile)
        # The explicit model/checkpoint profile owns training identity.
        # Probe/search globals can be unset outside their temporary scope.
    scope = os.environ.get('WOTR_OBSERVATION_TRAINING', 'appended_columns_v1')
    if scope not in ('head_only_legacy', 'appended_columns_v1', 'full_network_v1'):
        raise ValueError('Unknown persistent optimizer training scope')
    return dict(version=VERSION, side=side, profile=profile, reward_contract=reward,
                input_dim=net.input_dim, action_dim=ai.ACTION_ENCODING_DIM,
                architecture=[type(net).__name__, net.arch_name, net.arch_kind],
                model_layout=[[n, list(p.shape), str(p.dtype)] for n,p in net.named_parameters()],
                head_act='scaled2' if getattr(net.value_head[-1], 'scale', None) else 'tanh',
                scope=scope, **context, **current_identity())


def _layout(net, groups):
    names = {id(p):n for n,p in net.named_parameters()}
    seen = set(); result = []
    for group in groups:
        items = []
        for parameter in group['params']:
            key = id(parameter)
            if key not in names or key in seen:
                raise ValueError('Optimizer groups contain duplicate or foreign parameter')
            seen.add(key)
            items.append([names[key], list(parameter.shape), str(parameter.dtype)])
        if not items:
            raise ValueError('Empty optimizer group')
        result.append(items)
    return result


def _groups(net, component):
    from observation_training import ObservationTrainingScope
    prefix = 'value_head' if component == 'value' else 'policy_delta'
    head = [p for name,p in net.named_parameters() if name.startswith(prefix)]
    return ObservationTrainingScope(net).parameter_groups(head)


def _validate_state(state, layout):
    import torch
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'}:
        raise ValueError('Malformed Adam state dictionary')
    groups = state['param_groups']
    if not isinstance(groups, list) or len(groups) != len(layout) or not isinstance(state['state'], dict):
        raise ValueError('Adam group count mismatch')
    index_layout = {}; all_ids = []
    for group, expected in zip(groups, layout):
        if not isinstance(group, dict) or not isinstance(group.get('params'), list) or len(group['params']) != len(expected):
            raise ValueError('Adam ordered parameter count mismatch')
        # Fixed algorithm settings are part of this version; LR/decay may change.
        if (tuple(group.get('betas', ())) != (.9, .999) or group.get('eps') != 1e-8
                or any(group.get(k, False) for k in ('amsgrad','maximize','capturable','differentiable','fused','decoupled_weight_decay'))
                or group.get('foreach') not in (None, False)):
            raise ValueError('Unsupported Adam algorithm settings')
        _number(group.get('lr'), 'saved learning rate', positive=True)
        _number(group.get('weight_decay'), 'saved weight decay')
        for index, spec in zip(group['params'], expected):
            if type(index) is not int or index < 0 or index in index_layout:
                raise ValueError('Invalid or repeated optimizer parameter index')
            index_layout[index] = spec; all_ids.append(index)
    if all_ids != list(range(len(all_ids))):
        raise ValueError('Adam parameter indices must retain canonical order')
    if any(type(index) is not int or index not in index_layout for index in state['state']):
        raise ValueError('Adam state contains unknown parameter index')
    for index, values in state['state'].items():
        if not isinstance(values, dict) or set(values) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('Malformed Adam parameter state')
        step = values['step']
        if (not isinstance(step, torch.Tensor) or step.ndim != 0
                or step.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(step).item() or step.item() < 1
                or step.item() != int(step.item())):
            raise ValueError('Invalid Adam step scalar')
        _, shape, dtype = index_layout[index]
        for name in ('exp_avg','exp_avg_sq'):
            tensor = values[name]
            if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shape
                    or str(tensor.dtype) != dtype or not torch.isfinite(tensor).all().item()
                    or (name == 'exp_avg_sq' and torch.any(tensor < 0).item())):
                raise ValueError('Invalid Adam moment ' + name)


def _validate_record(net, record):
    if (not isinstance(record, dict) or set(record) != {'identity','start','components'}
            or record['identity'] != _identity(net)
            or record['start'] != 'explicit_fresh_optimizer'
            or not isinstance(record['components'], dict)
            or set(record['components']) != {'value','policy'}):
        raise ValueError('Optimizer identity/scope/start contract mismatch')
    for component, entry in record['components'].items():
        if entry is None:
            continue
        if not isinstance(entry, dict) or set(entry) != {'layout','state','steps','initialized'}:
            raise ValueError('Malformed optimizer component')
        expected = _layout(net, _groups(net, component))
        if entry['layout'] != expected:
            raise ValueError('Optimizer ordered group layout mismatch: ' + component)
        _validate_state(entry['state'], expected)
        if (type(entry['steps']) is not int or entry['steps'] < 0
                or entry['initialized'] != sorted(entry['state']['state'])
                or bool(entry['state']['state']) != bool(entry['steps'])
                or any(s['step'].item() > entry['steps'] for s in entry['state']['state'].values())):
            raise ValueError('Optimizer initialized-parameter/step metadata mismatch')


def _record(net):
    current = getattr(net, '_optimizer_continuity', None)
    if current is None:
        if not _fresh_allowed():
            raise ValueError('Optimizer state absent; explicitly allow a fresh optimizer start')
        current = {'identity':_identity(net), 'start':'explicit_fresh_optimizer',
                   'components':{'value':None,'policy':None}}
        net._optimizer_continuity = current
        net._persistent_adam = {}
    # Validate both components before any call can update either pathway.
    _validate_record(net, current)
    return current


def optimizer_for(net, component, groups, scope, lr):
    import torch
    if mode() == RESET:
        for name in ('_optimizer_continuity','_persistent_adam'):
            if hasattr(net, name):delattr(net, name)
        return torch.optim.Adam(groups, lr=lr)
    _number(lr, 'learning rate', positive=True)
    for group in groups:_number(group.get('weight_decay', 0), 'weight decay')
    if component not in ('value','policy') or scope.mode != _identity(net)['scope']:
        raise ValueError('Optimizer component/scope mismatch')
    record = _record(net)
    layout = _layout(net, groups)
    if layout != _layout(net, _groups(net, component)):
        raise ValueError('Unexpected supplied optimizer group order')
    cache = getattr(net, '_persistent_adam', {})
    optimizer = cache.get(component)
    entry = record['components'][component]
    if optimizer is None:
        optimizer = torch.optim.Adam(groups, lr=lr)
        if entry is not None:
            optimizer.load_state_dict(copy.deepcopy(entry['state']))
        cache[component] = optimizer; net._persistent_adam = cache
    elif any(a is not b for supplied, actual in zip(groups, optimizer.param_groups)
             for a,b in zip(supplied['params'], actual['params'])):
        raise ValueError('Cached Adam is not bound to current model parameters')
    # Saved settings must never override current validated driver controls.
    for actual, supplied in zip(optimizer.param_groups, groups):
        actual['lr'] = lr
        actual['weight_decay'] = supplied.get('weight_decay', 0)
    state=optimizer.state_dict()
    record['components'][component] = {'layout':layout, 'state':state,
        'steps':entry['steps'] if entry is not None else 0, 'initialized':sorted(state['state'])}
    return optimizer


def updated(net, component, optimizer):
    if mode() == VERSION:
        entry=net._optimizer_continuity['components'][component]
        entry['state'] = optimizer.state_dict()
        entry['initialized'] = sorted(entry['state']['state'])
        entry['steps'] += 1


def _model_hash(state):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
            raise ValueError('Nonfinite or invalid model state for optimizer binding')
        tensor = value.detach().cpu().contiguous()
        digest.update(repr((name,list(tensor.shape),str(tensor.dtype))).encode('utf8'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_payload(net, checkpoint):
    """Embed a CPU-only opaque blob bound to the model saved in this object."""
    import torch
    if mode() == RESET:return {}
    record = _record(net)
    # Legacy reward checkpoints normally omit selector provenance. Opting
    # into optimizer continuation binds even these to the active experiment.
    for key in ('selector_contract','selector_config','selector_config_sha256'):
        checkpoint.setdefault(key, copy.deepcopy(record['identity'][key]))
    _validate_checkpoint_identity(net, checkpoint)
    def cpu(value):
        if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value, dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value, list):return [cpu(v) for v in value]
        if isinstance(value, tuple):return tuple(cpu(v) for v in value)
        return value
    inner = {'version':VERSION, 'model_sha256':_model_hash(checkpoint['model_state']), 'record':cpu(record)}
    stream=io.BytesIO();torch.save(inner,stream);blob=stream.getvalue()
    return {FIELD:{'version':VERSION,'sha256':hashlib.sha256(blob).hexdigest(),'blob':blob}}


def restore_checkpoint(net, checkpoint):
    """Validate both optimizers before attaching any training continuation."""
    import torch
    if mode() == RESET:return net
    outer = checkpoint.get(FIELD)
    _validate_checkpoint_identity(net, checkpoint, fresh=outer is None)
    if outer is None:
        _record(net)
        return net
    if (not isinstance(outer,dict) or set(outer) != {'version','sha256','blob'}
            or outer['version'] != VERSION or not isinstance(outer['blob'],bytes)
            or hashlib.sha256(outer['blob']).hexdigest() != outer['sha256']):
        raise ValueError('Optimizer blob version/hash mismatch')
    inner=torch.load(io.BytesIO(outer['blob']),map_location='cpu',weights_only=True)
    if (not isinstance(inner,dict) or set(inner) != {'version','model_sha256','record'}
            or inner['version'] != VERSION
            or inner['model_sha256'] != _model_hash(checkpoint['model_state'])
            or inner['model_sha256'] != _model_hash(net.state_dict())):
        raise ValueError('Optimizer payload is not bound to these model weights')
    _validate_record(net, inner['record'])
    net._optimizer_continuity=inner['record'];net._persistent_adam={}
    return net


def load_training_state(net, path):
    import torch
    if mode() == RESET:return net
    return restore_checkpoint(net,torch.load(path,map_location='cpu',weights_only=True))


def _validate_checkpoint_identity(net, checkpoint, fresh=False):
    expected = _identity(net)
    for key in ('side','reward_contract','input_dim','action_dim','head_act',
                'observation_contract','action_encoding_contract','ai_policy_contract',
                'selector_contract','selector_config','selector_config_sha256'):
        if fresh and key.startswith('selector_') and not any(k.startswith('selector_') for k in checkpoint):
            # Converted weight-only seeds do not claim an inherited selector.
            # The explicit fresh optimizer records the current experiment.
            continue
        if checkpoint.get(key) != expected[key]:
            raise ValueError('Optimizer checkpoint contract mismatch: ' + key)
    if checkpoint.get('reward_profile') != expected['profile']:
        raise ValueError('Optimizer checkpoint profile mismatch')
