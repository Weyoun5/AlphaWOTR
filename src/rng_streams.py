"""Opt-in persistent game/search RNG isolation for synchronous CPU workers.

Legacy mode never reseeds or swaps RNG state. Isolated episodes seed gameplay
explicitly and carry a separate advancing Python/NumPy state across searches.
This adapter is process-local, supports nested searches on one thread, and
rejects using its isolated search stream from another thread. It is not a
general adapter for concurrent games within one process.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
import random
import threading

import numpy as np

LEGACY = 'shared_worker_rng_v1'
ISOLATED = 'isolated_game_search_v1'
_episode = None
_search_python = None
_search_numpy = None
_depth = 0
_owner_thread = None
_search_calls = 0


def current_contract():
    contract = os.environ.get('WOTR_RNG_CONTRACT', LEGACY)
    if contract not in (LEGACY, ISOLATED):
        raise ValueError('Unknown RNG contract: %r' % contract)
    return contract


def _integer(value, name, upper=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (upper is not None and value >= upper):
        raise ValueError('%s must be a nonnegative integer%s' % (name, ' below %d' % upper if upper else ''))
    return value


def derive_seed(seed, domain, *, bits=64):
    _integer(seed, 'episode seed', 2 ** 64)
    if bits not in (32, 64) or not isinstance(domain, str) or not domain:
        raise ValueError('Seed derivation requires a domain and 32 or 64 bits')
    payload = ('%s\0%d\0%s' % (ISOLATED, seed, domain)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:bits // 8], 'big')


def served_episode_seed(run_seed, episode_index):
    """Derive from run/shard seed and global episode index, never worker count."""
    _integer(run_seed, 'run seed', 2 ** 32)
    _integer(episode_index, 'episode index')
    return run_seed if episode_index == 0 else derive_seed(run_seed, 'served-episode:%d' % episode_index)


def start_episode(config):
    """Reset both domains once, before setup/deck shuffles, never per decision."""
    global _episode, _search_python, _search_numpy, _owner_thread, _search_calls
    contract = current_contract()
    if _depth:
        raise RuntimeError('Cannot begin an episode while a search scope is active')
    if contract == LEGACY:
        if 'rng_episode_seed' in config:
            raise ValueError('An explicit episode seed requires the isolated RNG contract')
        _episode = _search_python = _search_numpy = _owner_thread = None
        _search_calls = 0
        return None
    if _episode is not None and threading.get_ident() != _owner_thread:
        raise RuntimeError('Isolated episode reset belongs to its synchronous worker thread')
    seed = _integer(config.get('rng_episode_seed'), 'rng_episode_seed', 2 ** 64)
    identity = config.get('rng_episode_identity')
    # Validate/copy identity before altering any stream; JSON records remain portable.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    seeds = {'game_python': seed, 'game_numpy': derive_seed(seed, 'game-numpy', bits=32),
             'search_python': derive_seed(seed, 'search-python'),
             'search_numpy': derive_seed(seed, 'search-numpy', bits=32)}
    search_python = random.Random(seeds['search_python']).getstate()
    search_numpy = np.random.RandomState(seeds['search_numpy']).get_state()
    random.seed(seeds['game_python'])
    np.random.seed(seeds['game_numpy'])
    _search_python, _search_numpy = search_python, search_numpy
    _owner_thread = threading.get_ident()
    _search_calls = 0
    _episode = {'contract': ISOLATED, 'episode_seed': seed, 'episode_identity': identity,
                'seed_derivation': 'sha256-domain-prefix-big-endian-v1', 'seeds': seeds,
                'python_random_state_version': random.getstate()[0], 'numpy_rng': 'RandomState-MT19937'}
    return episode_metadata()


def episode_metadata():
    if current_contract() == LEGACY:
        return None
    if _episode is None:
        raise RuntimeError('Isolated RNG mode requires an explicit episode initialization')
    return copy.deepcopy(_episode)


def search_call_count():
    return _search_calls


@contextmanager
def search_scope():
    """Swap once at the outer boundary and retain advancing search randomness."""
    global _depth, _search_python, _search_numpy, _search_calls
    contract = current_contract()
    if contract == LEGACY:
        yield
        return
    if _episode is None:
        raise RuntimeError('Isolated search requires start_episode with an explicit seed')
    if threading.get_ident() != _owner_thread:
        raise RuntimeError('Isolated RNG streams belong to one synchronous worker thread')
    outermost = _depth == 0
    if outermost:
        game_python, game_numpy = random.getstate(), np.random.get_state()
        random.setstate(_search_python)
        np.random.set_state(_search_numpy)
        _search_calls += 1
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if outermost:
            try:
                _search_python = random.getstate()
                _search_numpy = np.random.get_state()
            finally:
                random.setstate(game_python)
                np.random.set_state(game_numpy)
