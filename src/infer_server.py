# -*- coding: utf-8 -*-
"""Batched GPU inference server for CPU self-play workers.

One GPU process owns the nets and batches requests from concurrent CPU
workers. The throughput benefit depends on model, search budget, worker
count and hardware; compare admitted whole-game throughput with local CPU
inference before choosing a deployment.

Protocol (numpy over multiprocessing Pipes, one duplex pipe/worker):
  ('eval',  rid, key, X[B,D])                  -> (rid, values[B])
  ('pv',    rid, key, vec[D], encs[N,A], regs[N]) -> (rid, priors[N], v)
  ('reload', paths_dict)                       -> no reply (weight sync)
  ('stop',)                                    -> server exits

Workers hold RemoteNet handles (attribute remote_key) and set
ai.INFER_CLIENT — the MCTS helpers (_get_priors_and_value /
_mcts_eval_state) route through the server when they see the handle.
Requests are synchronous per worker (MCTS is sequential), so batching
comes from cross-worker concurrency: the server drains every ready
pipe, groups by (op, key), and runs one forward per group.
"""
import os
import sys
import hashlib
import numbers
import math
import time
from inference_transport import exchange as _exchange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np


class RemoteNet:
    """Stand-in net object carried by workers; MCTS helpers detect
    remote_key and route through the attached client (falling back to
    the process-global ai.INFER_CLIENT).  A per-net _client allows one
    process to talk to SEVERAL servers — e.g. an arena between nets of
    different architectures, each hosted by its own server."""

    def __init__(self, key, client=None):
        self.remote_key = key
        self.side = key[:2]
        self._client = client

    def eval(self):
        return self

    def __bool__(self):
        return True


class InferClient:
    """Synchronous client with one deadline for request and complete response.

    WOTR_INFER_TIMEOUT_SEC defaults to 60 positive finite seconds. The deadline
    includes blocked request writes and incomplete reply frames. A failed
    exchange closes the connection; no helper thread retains ownership.
    """

    def __init__(self, conn, response_timeout=None):
        self.conn = conn
        self._rid = 0
        self._closed = False
        self.response_timeout = self._timeout(
            os.environ.get('WOTR_INFER_TIMEOUT_SEC', '60')
            if response_timeout is None else response_timeout)

    @staticmethod
    def _timeout(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('Inference timeout must be a positive finite number of seconds')
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError('Inference timeout must be a positive finite number of seconds') from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Inference timeout must be a positive finite number of seconds')
        return seconds

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.conn.close()

    def _roundtrip(self, msg, timeout=None):
        seconds = self.response_timeout if timeout is None else min(self.response_timeout, self._timeout(timeout))
        if self._closed:
            raise RuntimeError('Inference connection is closed after a previous failure or close')
        try:
            try:
                reply = _exchange(self.conn, msg, time.monotonic() + seconds)
            except TimeoutError as exc:
                failure = TimeoutError('Inference %s request %s exceeded %.6g seconds; connection invalidated'
                                       % (msg[0], msg[1], seconds))
                if hasattr(failure, 'add_note'):
                    for note in getattr(exc, '__notes__', ()):
                        failure.add_note(note)
                raise failure from exc
            if (not isinstance(reply, (tuple, list)) or len(reply) < 2
                    or isinstance(reply[0], (bool, np.bool_))
                    or not isinstance(reply[0], numbers.Integral) or reply[0] != msg[1]):
                raise ValueError('Inference reply does not match request %s' % msg[1])
            return reply
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                if hasattr(exc, 'add_note'):
                    exc.add_note('Inference connection cleanup failed: %r' % cleanup_error)
            raise

    def describe(self, timeout=60):
        self._rid += 1
        _, description = self._roundtrip(('describe', self._rid), timeout=timeout)
        if not isinstance(description, dict):
            raise ValueError('Inference server checkpoint description must be a mapping')
        return description

    def eval(self, key, X):
        self._rid += 1
        _, values = self._roundtrip(('eval', self._rid, key,
                                     np.ascontiguousarray(X)))
        return values

    def pv(self, key, vec, encs, regs):
        self._rid += 1
        _, priors, v = self._roundtrip(
            ('pv', self._rid, key, np.ascontiguousarray(vec),
             np.ascontiguousarray(encs), np.ascontiguousarray(regs)))
        return priors, float(v)

    def pv_many(self, key, items):
        """S-003 leaf batching: K (vec, encs, regs) requests in ONE
        message, ONE reply [(priors, v), ...] in order.  One message each
        way (never K interleaved sends) so a full pipe buffer can never
        deadlock against the server's replies."""
        self._rid += 1
        _, out = self._roundtrip(
            ('pvm', self._rid, key,
             [(np.ascontiguousarray(v), np.ascontiguousarray(e),
               np.ascontiguousarray(r)) for v, e, r in items]))
        return [(p, float(v)) for p, v in out]


def attach_client(conn, keys, set_global=True, require_metadata=False):
    """Called inside a worker process: build handles for one server
    connection.  set_global also installs the client as
    ai.INFER_CLIENT (single-server case)."""
    import ai
    cli = InferClient(conn)
    try:
        description = cli.describe() if require_metadata else None
        handles = {}
        for key in keys:
            handle = RemoteNet(key, client=cli)
            if description is not None:
                metadata = description.get(key)
                if not isinstance(metadata, dict) or any(field not in metadata for field in (
                        'checkpoint_id', 'reward_profile', 'reward_contract')):
                    raise ValueError('Inference checkpoint metadata missing for %s' % key)
                checkpoint_id = metadata['checkpoint_id']
                if (not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64
                        or any(c not in '0123456789abcdef' for c in checkpoint_id)):
                    raise ValueError('Invalid inference checkpoint identity for %s' % key)
                from reward_contract import (LEGACY, PREDECISION, PROFILE,
                                             current_contract, validate_profile)
                if metadata['reward_contract'] not in (None, LEGACY, *PREDECISION):
                    raise ValueError('Invalid inference reward contract for %s' % key)
                if metadata['reward_contract'] not in (None, current_contract()):
                    raise ValueError('Inference reward contract differs from worker for %s' % key)
                if metadata['reward_contract'] == PROFILE and metadata['reward_profile'] is None:
                    raise ValueError('Profile-trained inference checkpoint lacks its profile for %s' % key)
                if metadata['reward_profile'] is not None:
                    validate_profile(key[:2], metadata['reward_profile'])
                from trace_decisions import validate_action_encoding_contract
                handle.action_encoding_contract = validate_action_encoding_contract(
                    metadata, required=(metadata['reward_contract'] in PREDECISION or
                        'observation_contract' in metadata or 'selector_contract' in metadata))
                modern = (metadata['reward_contract'] in PREDECISION or
                          any(field in metadata for field in (
                              'observation_contract', 'selector_contract',
                              'action_encoding_contract')))
                observation_contract = metadata.get('observation_contract')
                if modern and observation_contract != ai.OBSERVATION_CONTRACT:
                    raise ValueError('Inference observation contract differs from worker for %s: %r != %r'
                                     % (key, observation_contract, ai.OBSERVATION_CONTRACT))
                for field, expected in (('input_dim', ai.get_feature_dim()),
                                        ('action_dim', ai.ACTION_ENCODING_DIM)):
                    if field not in metadata and not modern:
                        continue
                    value = metadata.get(field)
                    if (isinstance(value, (bool, np.bool_)) or
                            not isinstance(value, numbers.Integral) or value != expected):
                        raise ValueError('Inference %s differs from worker for %s: %r != %r'
                                         % (field, key, value, expected))
                    setattr(handle, field, int(value))
                handle.observation_contract = observation_contract
                handle.checkpoint_id = checkpoint_id
                handle.reward_profile = metadata['reward_profile']
                handle.reward_contract = metadata['reward_contract']
            handles[key] = handle
        if set_global:
            ai.INFER_CLIENT = cli
        return handles
    except BaseException as exc:
        try:
            cli.close()
        except Exception as cleanup_error:
            if hasattr(exc, 'add_note'):
                exc.add_note('Inference metadata cleanup failed: %r' % cleanup_error)
        raise



def _checkpoint_id(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_net(ai, torch, path, side):
    """ONE-loader discipline (HUNT-B F4, 2026-08-11): the hand-rolled
    load here skipped the checkpoint contract — head_act was never
    read (a scaled2 checkpoint served Tanh-squashed values to every
    remote consumer), no confidence auto-registration, no generation
    refusal (a stale file died as an opaque shape error at best).
    ai.load_value_net applies the whole law; it constructs on
    ai.DEVICE, which server_main sets before loading.  NOTE: the
    remote pv path below scores via net.spatial_action_scorer and,
    behind ai._policy_delta_enabled(), ADDS the 1.226a residual
    policy delta -- mirroring the local score_actions_spatial
    composition (the old BASE-ONLY remote pv silently dropped a
    trained delta; audit 2026-08-29 SPLP-8 doc fix -- code was
    already right)."""
    path = ai._resolve_model_path(path)
    checkpoint_id = _checkpoint_id(path)
    net = ai.load_value_net(path, side=side)
    if net is None:
        raise FileNotFoundError(
            'infer_server: net checkpoint missing: %s' % path)
    if _checkpoint_id(path) != checkpoint_id:
        raise ValueError('Inference checkpoint changed while loading: %s' % path)
    net.checkpoint_id = checkpoint_id
    net.eval()
    return net


def server_main(conns, net_paths, arch, device='cuda'):
    """Server process entry.  conns: list of Pipe ends (one per worker);
    net_paths: {key: checkpoint_path}; arch: WOTR_NET_ARCH value."""
    os.environ['WOTR_NET_ARCH'] = arch
    import torch

    requested = torch.device(device)
    if requested.type not in ('cpu', 'cuda'):
        raise ValueError('Unsupported inference server device: %s' % requested)
    if requested.type == 'cuda':
        count = torch.cuda.device_count()
        if (not torch.cuda.is_available() or count < 1
                or (requested.index is not None and requested.index >= count)):
            raise RuntimeError('Inference server requested %s but no matching assigned GPU is available'
                               % requested)
    import ai
    ai.DEVICE = requested
    torch.set_num_threads(2)
    graph_enabled = os.environ.get('WOTR_INFER_CUDA_GRAPH', '0') == '1'
    graph_options = {}
    if graph_enabled:
        graph_options = dict(max_graphs=int(os.environ.get('WOTR_INFER_GRAPH_MAX_GRAPHS', '32')),
                             max_cached_bytes=int(os.environ.get('WOTR_INFER_GRAPH_CACHE_BYTES', str(512*1024*1024))),
                             max_padded_rows=int(os.environ.get('WOTR_INFER_GRAPH_MAX_ROWS', '32768')))
        if min(graph_options.values()) < 1:
            raise ValueError('CUDA Graph cache limits must be positive')
    graph_caches = {}
    frozen_weights = os.environ.get('WOTR_INFER_FROZEN', '0') == '1'
    nets = {k: _load_net(ai, torch, p, k[:2])
            for k, p in net_paths.items()}
    print('[infer_server] up: %d nets (%s), %d workers, device %s'
          % (len(nets), arch, len(conns), ai.DEVICE), flush=True)

    from multiprocessing.connection import wait
    live = list(conns)
    n_req = 0
    n_batch = 0
    while live:
        ready = wait(live, timeout=5.0)
        if not ready:
            continue
        evals = {}    # key -> list of (conn, rid, X)
        pvs = {}      # key -> list of (conn, rid, vec, encs, regs)
        stop = False
        for c in ready:
            try:
                while c.poll():
                    m = c.recv()
                    op = m[0]
                    if op == 'stop':
                        stop = True
                    elif op == 'describe':
                        # Sent only after all startup loads succeed. Workers
                        # retain the actual served profile and content identity.
                        c.send((m[1], {key: {
                            'checkpoint_id': net.checkpoint_id,
                            'immutable_weights': frozen_weights,
                            'input_dim': int(net.input_dim),
                            'action_dim': int(ai.ACTION_ENCODING_DIM),
                            'reward_profile': getattr(net, 'reward_profile', None),
                            'reward_contract': getattr(net, 'reward_contract', None),
                            **({'observation_contract': net.observation_contract}
                               if getattr(net, 'observation_contract', None) is not None else {}),
                            **({'action_encoding_contract': net.action_encoding_contract}
                               if getattr(net, 'action_encoding_contract', None) is not None else {}),
                        } for key, net in nets.items()}))
                    elif op == 'reload':
                        if frozen_weights:
                            raise RuntimeError('Immutable inference session refuses checkpoint reload')
                        for k, p in m[1].items():
                            try:
                                nets[k] = _load_net(ai, torch, p, k[:2])
                            except Exception as e:
                                print('[infer_server] reload %s failed:'
                                      ' %r' % (k, e), flush=True)
                    elif op == 'eval':
                        evals.setdefault(m[2], []).append(
                            (c, m[1], m[3]))
                        n_req += 1
                    elif op == 'pv':
                        pvs.setdefault(m[2], []).append(
                            (c, m[1], m[3], m[4], m[5], None))
                        n_req += 1
                    elif op == 'pvm':
                        if not m[3]:
                            c.send((m[1], []))
                            n_req += 1
                            continue
                        # S-003: K requests in one message -> the same
                        # batched pv path, ONE reply once every item of
                        # the group has its (priors, v)
                        grp = [m[1], len(m[3]), [None] * len(m[3])]
                        for j, (vv, ee, rr) in enumerate(m[3]):
                            pvs.setdefault(m[2], []).append(
                                (c, m[1], vv, ee, rr, (grp, j)))
                        n_req += 1
            except (EOFError, OSError):
                live.remove(c)
        # NOTE: stop is honored AFTER the processing block below —
        # requests drained in the same cycle must still be answered
        # (a worker blocked in recv() would otherwise EOF).
        with torch.no_grad():
            for key, items in evals.items():
                net = nets[key]
                X = np.concatenate([it[2] for it in items], axis=0)
                graph_values = (graph_enabled and requested.type == 'cuda'
                                and getattr(net, 'arch_name', None) == 'base'
                                and getattr(net, 'arch_kind', None) == 'gnn'
                                and len(X) <= graph_options['max_padded_rows'])
                if graph_values:
                    from graph_pv import GraphPV, GraphCacheBudgetExceeded
                    cached = graph_caches.get(key)
                    if cached is None or cached.net is not net:
                        if cached is not None: cached.close()
                        cached = graph_caches[key] = GraphPV(net, **graph_options)
                    try:
                        vals = cached.eval_values(np.asarray(X, dtype=np.float32))
                    except GraphCacheBudgetExceeded:
                        xt = torch.from_numpy(X).float().to(ai.DEVICE)
                        vals = net(xt).detach().cpu().numpy().reshape(-1)
                else:
                    xt = torch.from_numpy(X).float().to(ai.DEVICE)
                    vals = net(xt).detach().cpu().numpy().reshape(-1)
                pos = 0
                for c, rid, Xi in items:
                    k = Xi.shape[0]
                    try:
                        c.send((rid, vals[pos:pos + k].copy()))
                    except (EOFError, OSError):
                        pass
                    pos += k
                n_batch += 1
            for key, items in pvs.items():
                net = nets[key]
                from inference_group import policy_value_group
                graph_items = [(it[2], it[3], it[4]) for it in items]
                max_actions = max((len(it[1]) for it in graph_items), default=0)
                padded = len(graph_items) * (1 << max(0, max_actions-1).bit_length())
                valid_rows = sum(len(it[1]) for it in graph_items)
                if (graph_enabled and requested.type == 'cuda' and getattr(net, 'arch_name', None) == 'base'
                        and getattr(net, 'arch_kind', None) == 'gnn'
                        and padded <= graph_options['max_padded_rows'] and padded <= 2*max(1, valid_rows)):
                    from graph_pv import GraphPV, GraphCacheBudgetExceeded
                    cached = graph_caches.get(key)
                    if cached is None or cached.net is not net:
                        if cached is not None: cached.close()
                        cached = graph_caches[key] = GraphPV(net, **graph_options)
                    try:
                        results = cached.pv_many(graph_items, ai._policy_delta_enabled())
                    except GraphCacheBudgetExceeded:
                        results = policy_value_group(net, graph_items, ai.DEVICE,
                                                     ai._policy_delta_enabled())
                else:
                    results = policy_value_group(net, graph_items, ai.DEVICE,
                                                 ai._policy_delta_enabled())
                for i, (c, rid, _v, e_i, _r, grp) in enumerate(items):
                    pr, value = results[i]
                    try:
                        if grp is None:
                            c.send((rid, pr, value))
                        else:
                            g, j = grp
                            g[2][j] = (pr, value)
                            g[1] -= 1
                            if g[1] == 0:
                                c.send((rid, g[2]))
                    except (EOFError, OSError):
                        pass
                n_batch += 1
        if stop:
            break
    import json
    if graph_enabled:
        print('[cuda_graph] ' + json.dumps({key: value.stats() for key, value in graph_caches.items()}), flush=True)
    print('[infer_server] down after %d requests / %d batches'
          % (n_req, n_batch), flush=True)


if __name__ == '__main__':
    print('infer_server is a library; start via bench_scale.py or the '
          'self-play driver')
