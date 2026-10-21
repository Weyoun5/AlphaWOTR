"""Explicit legacy, appended-column, and full-network training scopes.

Full-network mode uses evaluation-mode modules with autograd enabled. This
keeps dropout deterministic and BatchNorm running statistics fixed, including
singleton policy rows. Its metadata records optimizer scope and completed
steps, not a claim that every eligible parameter changed or learned skill.
"""
import os
import copy
import hashlib
import json

BASE_INPUT_DIM = 3742
CONTRACT_DIMENSIONS = {
    'v42_casualty_operations_v1': 3745,
    'v43_card_casualty_operations_v1': 3745,
    'v44_fellowship_tail_v1': 3746,
    'v45_deferred_card_context_v1': 4180,
    'v46_companion_movement_context_v1': 4190,
    'v47_fellowship_move_timing_v1': 4191,
    'v48_hunt_protection_context_v1': 4216,
    'v49_reveal_route_context_v1': 4280,
    'v50_route_event_context_v1': 4347,
    'v51_breaking_hunt_context_v1': 4350,
    'v52_observed_combat_round_v2': 4481,
    'v52_observed_combat_round_v3': 4647,
}

FULL_NETWORK_MODE = 'full_network_v1'
FULL_MODULE_MODE = 'eval_autograd_v1'


def full_parameter_layout(named_parameters):
    """Fingerprint names/shapes, excluding buffers and parameter values."""
    parameters = list(named_parameters)
    if not parameters or len({name for name, _ in parameters}) != len(parameters):
        raise ValueError('Full-network scope requires unique named parameters')
    layout = sorted((name, list(parameter.shape)) for name, parameter in parameters)
    encoded = json.dumps(layout, separators=(',', ':')).encode('utf-8')
    return {'parameter_layout_sha256': hashlib.sha256(encoded).hexdigest(),
            'parameter_count': sum(parameter.numel() for _, parameter in parameters),
            'module_mode': FULL_MODULE_MODE}


def validate_full_parameter_layout(record, named_parameters):
    expected = full_parameter_layout(named_parameters)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Recorded full-network layout differs from model parameters')


def validate_training_scope(metadata, net=None):
    if (not isinstance(metadata, dict) or metadata.get('version') != 'observation_training_scope_v1'
            or set(metadata) != {'version', 'observation_contract', 'input_dim', 'modes'}
            or not isinstance(metadata['modes'], dict) or not metadata['modes']):
        raise ValueError('Malformed observation training scope metadata')
    for mode, record in metadata['modes'].items():
        if mode not in ('head_only_legacy', 'appended_columns_v1', FULL_NETWORK_MODE) or not isinstance(record, dict):
            raise ValueError('Unknown recorded observation training mode')
        if mode == FULL_NETWORK_MODE:
            fields = {'value_steps', 'policy_steps', 'parameter_layout_sha256',
                      'parameter_count', 'module_mode'}
            contract, dimension = metadata['observation_contract'], metadata['input_dim']
            if (set(record) != fields or contract not in CONTRACT_DIMENSIONS
                    or type(dimension) is not int or dimension != CONTRACT_DIMENSIONS[contract]
                    or any(type(record[key]) is not int or record[key] < 0
                           for key in ('value_steps', 'policy_steps'))
                    or type(record['parameter_count']) is not int or record['parameter_count'] <= 0
                    or record['module_mode'] != FULL_MODULE_MODE
                    or not isinstance(record['parameter_layout_sha256'], str)
                    or len(record['parameter_layout_sha256']) != 64
                    or any(char not in '0123456789abcdef' for char in record['parameter_layout_sha256'])):
                raise ValueError('Invalid recorded full-network training scope')
            continue
        counts = {'value_steps', 'policy_steps', 'value_projection_changes', 'policy_projection_changes'}
        layout = {'parameter', 'column_start', 'column_end', 'raw_start', 'raw_end'}
        if set(record) != counts | (layout if mode == 'appended_columns_v1' else set()):
            raise ValueError('Unexpected recorded observation training fields')
        if any(type(record[key]) is not int or record[key] < 0 for key in counts):
            raise ValueError('Invalid observation optimizer counts')
        for component in ('value', 'policy'):
            changes = record[component + '_projection_changes']
            if changes > record[component + '_steps'] or (mode == 'head_only_legacy' and changes):
                raise ValueError('Invalid observation projection change count')
        if mode == 'appended_columns_v1':
            contract = metadata['observation_contract']
            dimension = metadata['input_dim']
            if contract not in CONTRACT_DIMENSIONS or type(dimension) is not int or dimension != CONTRACT_DIMENSIONS[contract]:
                raise ValueError('Recorded observation schema is incompatible')
            if (record['parameter'] not in ('backbone.0.weight', 'tok_global.weight') or
                    any(type(record[key]) is not int for key in layout - {'parameter'}) or
                    record['raw_start'] != BASE_INPUT_DIM or record['raw_end'] != dimension or
                    record['column_start'] <= 0 or
                    record['column_end'] - record['column_start'] != dimension - BASE_INPUT_DIM):
                raise ValueError('Invalid recorded observation projection range')
    if net is not None:
        if (metadata['observation_contract'] != getattr(net, 'observation_contract', None) or
                metadata['input_dim'] != getattr(net, 'input_dim', None)):
            raise ValueError('Recorded training scope differs from the model schema')
        if 'appended_columns_v1' in metadata['modes']:
            name, parameter, cut = _projection_layout(net)
            record = metadata['modes']['appended_columns_v1']
            if (record['parameter'] != name or record['column_start'] != cut or
                    record['column_end'] != parameter.shape[1]):
                raise ValueError('Recorded training projection differs from the model layout')
        if FULL_NETWORK_MODE in metadata['modes']:
            _projection_layout(net)  # Supported architecture and observation schema.
            validate_full_parameter_layout(metadata['modes'][FULL_NETWORK_MODE], net.named_parameters())
    return metadata


def _projection_layout(net):
    contract = getattr(net, 'observation_contract', None)
    dimension = getattr(net, 'input_dim', None)
    if contract not in CONTRACT_DIMENSIONS or type(dimension) is not int or dimension != CONTRACT_DIMENSIONS[contract]:
        raise ValueError('Observation training requires an explicit recognized contract/dimension')
    import ai
    if not isinstance(net, ai.WOTRValueNet):
        raise ValueError('Observation training requires a supported WOTRValueNet architecture')
    if (net._spatial_dim != ai.NUM_REGIONS * ai.NODE_FEAT_DIM or
            net._global_dim != dimension - net._spatial_dim - ai.MAX_HAND_SIZE):
        raise ValueError('Observation projection layout differs from the supported architecture')
    if net.arch_kind == 'gnn':
        name = 'backbone.0.weight'
        projection = net.backbone[0]
        expected = net.board_gnn.output_proj.out_features + net.card_proj[0].out_features + net._global_dim
    elif net.arch_kind == 'transformer':
        name = 'tok_global.weight'
        projection = net.tok_global
        expected = net._global_dim
    else:
        raise ValueError('Unknown observation projection architecture: %r' % net.arch_kind)
    parameter = dict(net.named_parameters()).get(name)
    if (parameter is None or parameter.ndim != 2 or parameter.shape[1] != expected or
            projection.in_features != expected):
        raise ValueError('Unexpected observation projection shape')
    cut = expected - (dimension - BASE_INPUT_DIM)
    if cut <= 0:
        raise ValueError('Invalid appended-observation projection offset')
    return name, parameter, cut


class ObservationTrainingScope:
    def __init__(self, net):
        self.net = net
        self.mode = os.environ.get('WOTR_OBSERVATION_TRAINING', 'appended_columns_v1')
        self.recompute_features = self.mode != 'head_only_legacy'
        self.full_parameters = None
        self.parameter = None
        self.cut = None
        self.hook = None
        self.last_projection = None
        prior = getattr(net, 'training_scope', None)
        if prior is not None:
            validate_training_scope(prior, net=net)
        if self.mode == 'head_only_legacy':
            return
        if self.mode == FULL_NETWORK_MODE:
            _projection_layout(net)
            self.full_parameters = list(net.parameters())
            self.full_layout = full_parameter_layout(net.named_parameters())
            return
        if self.mode != 'appended_columns_v1':
            raise ValueError('Unknown observation training scope: %r' % self.mode)
        self.name, self.parameter, self.cut = _projection_layout(net)

    def __enter__(self):
        if self.full_parameters is not None:
            for parameter in self.full_parameters:
                parameter.requires_grad = True
            self.net.eval()
        if self.parameter is not None:
            self.parameter.requires_grad = True
            self.last_projection = self.parameter[:, self.cut:].detach().clone()
            def mask(gradient):
                masked = gradient.clone()
                masked[:, :self.cut] = 0
                return masked
            self.hook = self.parameter.register_hook(mask)
        return self

    def parameter_groups(self, head_parameters, head_decay=0):
        if self.full_parameters is not None:
            # DELTA_WD retains its existing meaning: residual-head shrinkage.
            # New shared/base weights are trainable without silently opting
            # them into a previously head-specific regularization setting.
            head_ids = {id(parameter) for parameter in head_parameters}
            if not head_ids <= {id(parameter) for parameter in self.full_parameters}:
                raise ValueError('Training head contains parameters outside the model')
            heads = [parameter for parameter in self.full_parameters if id(parameter) in head_ids]
            other = [parameter for parameter in self.full_parameters if id(parameter) not in head_ids]
            return ([{'params': heads, 'weight_decay': head_decay}] if heads else []) + (
                [{'params': other, 'weight_decay': 0.0}] if other else [])
        groups = [{'params': head_parameters, 'weight_decay': head_decay}]
        if self.parameter is not None:
            groups.append({'params': [self.parameter], 'weight_decay': 0.0})
        return groups

    def record_step(self, component):
        import torch
        if component not in ('value', 'policy'):
            raise ValueError('Unknown observation optimizer component')
        metadata = copy.deepcopy(getattr(self.net, 'training_scope', None))
        if metadata is None:
            metadata = {'version': 'observation_training_scope_v1',
                        'observation_contract': getattr(self.net, 'observation_contract', None),
                        'input_dim': getattr(self.net, 'input_dim', None), 'modes': {}}
        if self.full_parameters is not None:
            record = metadata['modes'].setdefault(self.mode,
                dict(value_steps=0, policy_steps=0, **self.full_layout))
            record[component + '_steps'] += 1
            self.net.training_scope = validate_training_scope(metadata, net=self.net)
            return
        record = metadata['modes'].setdefault(self.mode, dict(value_steps=0, policy_steps=0,
                    value_projection_changes=0, policy_projection_changes=0))
        changed = False
        if self.parameter is not None:
            changed = not torch.equal(self.last_projection, self.parameter[:, self.cut:])
            self.last_projection = self.parameter[:, self.cut:].detach().clone()
            record.update(parameter=self.name, column_start=self.cut,
                          column_end=self.parameter.shape[1], raw_start=BASE_INPUT_DIM,
                          raw_end=self.net.input_dim)
        record[component + '_steps'] += 1
        record[component + '_projection_changes'] += int(changed)
        self.net.training_scope = validate_training_scope(metadata, net=self.net)

    def __exit__(self, *exception):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
