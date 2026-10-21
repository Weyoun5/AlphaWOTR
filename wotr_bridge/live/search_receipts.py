"""Bounded local evidence for a real AI choice, before board execution.

Keep state and search scores together so a reported bad move can be inspected
without replaying an entire feed. These files are diagnostics, not training rows.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time


def capture(state):
    import numpy as np
    return pickle.dumps({'state': state, 'python_rng': random.getstate(),
                         'numpy_rng': np.random.get_state()}, protocol=5)


def save(client_dir, sequence, side, kind, simulations, before, action, stats,
         *, capacity=64):
    if capacity < 1:
        raise ValueError('Search receipt capacity must be positive')
    rows = []
    stats = stats or {}
    roots = stats.get('root_actions', ())
    for i, root in enumerate(roots):
        row = {'action_type': root.action_type, 'die_idx': root.die_idx,
               'params': root.params}
        for source, name in [('priors_clean', 'prior'), ('visit_counts', 'visits'),
                             ('q_values', 'q'), ('term_frac', 'terminal_fraction')]:
            values = stats.get(source)
            if values is not None:
                row[name] = float(values[i])
        rows.append(row)
    receipt = {'schema': 'bridge_search_receipt_v1', 'sequence': int(sequence),
               'created_at': time.time(), 'side': str(side), 'kind': str(kind),
               'requested_simulations': int(simulations),
               'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
               'state_sha256': hashlib.sha256(before).hexdigest(),
               'selected': {'action_type': action.action_type,
                            'die_idx': action.die_idx, 'params': action.params},
               'root_actions': rows,
               'own_value_search': stats.get('own_value_search')}
    folder = Path(client_dir) / 'ai_search_receipts'
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / ('decision_%02d' % (int(sequence) % capacity))
    for suffix, payload in [('.pkl', before),
                            ('.json', json.dumps(receipt, default=str,
                                                indent=2).encode('utf-8'))]:
        target = stem.with_suffix(suffix)
        temporary = target.with_suffix(suffix + '.tmp')
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    return stem.with_suffix('.json')
