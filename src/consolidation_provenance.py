"""Validate and preserve fixed-kind policy labels through corpus consolidation."""
import hashlib
import json
import os

import numpy as np

FIELD = 'policy_target_projection'
SIDECAR = 'policy_target_projection.jsonl'
SCHEMA = 'consolidated_policy_projection_v1'


def _validate(value, side):
    from ai_policies import _freeze, _recruit_choice
    if not isinstance(value, dict):
        raise ValueError('projection must be a mapping or null')
    if (value.get('schema') != 'fixed_recruit_target_projection_v1'
            or value.get('transformation') != 'maximum_elites_same_non_kind_choices'
            or value.get('literal_human_kind_claimed') is not False
            or value.get('replay_action') != 'source_human_action'):
        raise ValueError('unsupported or incomplete fixed recruitment provenance')
    source, target = value.get('source_human_action'), value.get('projected_action')
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError('source_human_action and projected_action are required')
    records = []
    scores = []
    for action in (source, target):
        if action.get('action_type') != 'card_option' or type(action.get('die_idx')) is not int:
            raise ValueError('projection must describe card_option actions with die indices')
        params = action.get('params')
        if not isinstance(params, dict) or not isinstance(params.get('card_id'), str) or not params['card_id']:
            raise ValueError('projection action needs identified card params')
        owner = params.get('side')
        if owner is not None and owner not in (
                ('fp', 'FREE_PEOPLES', 'Side.FREE_PEOPLES') if side == 0
                else ('sh', 'SHADOW', 'Side.SHADOW')):
            raise ValueError('projection action side differs from packed sample')
        choice = _recruit_choice(params.get('choices'))
        if choice is None:
            raise ValueError('projection has no recognized recruitment kind')
        outer = {k: v for k, v in params.items() if k not in ('choices', 'idx', 'card_name')}
        records.append((action['action_type'], action['die_idx'], _freeze(outer), choice[0]))
        scores.append(choice[1])
    if records[0] != records[1] or scores[1] <= scores[0]:
        raise ValueError('projection changes an independent target/payment or does not increase Elites')
    return value


def read_rows(data, source):
    sides = data['side']
    n = len(sides)
    if FIELD not in data.files:
        return [None] * n
    values = data[FIELD]
    if values.shape != (n,) or values.dtype.kind not in ('U', 'S'):
        raise ValueError(f'{source}: {FIELD} must be a string array with shape ({n},)')
    phases = data['decision_phase']
    if sides.shape != (n,) or phases.shape != (n,):
        raise ValueError(f'{source}: projection side/phase arrays must have shape ({n},)')
    result = []
    for index, raw in enumerate(values.astype(str)):
        try:
            value = json.loads(raw)
            if value is not None:
                if int(sides[index]) not in (0, 1) or str(phases[index]) != 'card_option':
                    raise ValueError('projection needs a card_option row and valid packed side')
                value = _validate(value, int(sides[index]))
            result.append(value)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{source}: malformed {FIELD} at sample {index}: {error}') from error
    return result


def record(source, sample, side, row, metadata, source_size, source_target,
           keep=None, target=None, *, partition='train'):
    if metadata is None:
        return None
    kept = [int(index) for index in keep] if keep is not None else None
    identity = kept is not None and kept == list(range(source_size))
    mapping = ({'kind': 'identity'} if identity else
               {'kind': 'retained_source_indices', 'indices': kept}) if kept is not None else None
    value = {'schema': SCHEMA, 'source_npz': source, 'source_sample': int(sample),
             'partition': partition, 'side': 'fp' if int(side) == 0 else 'sh',
             'consolidated_row': None if row is None else int(row),
             'source_menu_size': int(source_size), 'source_menu_target_index': int(source_target),
             'consolidated_menu_size': None if kept is None else len(kept),
             'consolidated_target_index': None if target is None else int(target),
             'candidate_mapping': mapping, FIELD: metadata}
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n'


def publish(directory, lines):
    path = os.path.join(directory, SIDECAR)
    digest = hashlib.sha256()
    counts = {'train': 0, 'holdout': 0, 'fp': 0, 'sh': 0}
    with open(path + '.tmp', 'wb') as stream:
        for line in lines:
            payload = line.encode('utf-8')
            stream.write(payload)
            digest.update(payload)
            row = json.loads(line)
            counts[row['partition']] += 1
            if row['partition'] == 'train':
                counts[row['side']] += 1
    os.replace(path + '.tmp', path)
    return {'schema': SCHEMA, 'file': SIDECAR, 'sha256': digest.hexdigest(),
            'records': counts['train'] + counts['holdout'], 'train_records': counts['train'],
            'holdout_records': counts['holdout'], 'train_by_side': {'fp': counts['fp'], 'sh': counts['sh']},
            'source_field': FIELD, 'value_trajectory': 'literal_human_replay'}


def verify(directory, metadata):
    if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA or metadata.get('file') != SIDECAR:
        raise ValueError('Invalid consolidated projection sidecar binding')
    digest = hashlib.sha256()
    count = 0
    with open(os.path.join(directory, SIDECAR), 'rb') as stream:
        for line in stream:
            digest.update(line)
            count += 1
    if digest.hexdigest() != metadata.get('sha256') or count != metadata.get('records'):
        raise ValueError('Consolidated projection sidecar hash/count mismatch')
