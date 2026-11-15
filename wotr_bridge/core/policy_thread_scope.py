"""Temporary CPU threading for large policy heads of a serial live search."""
from contextlib import contextmanager
import os


@contextmanager
def policy_thread_scope(net, *, enabled=True, threads=None, minimum_rows=1024):
    """Retain original model operations and restore threads/instance attributes.

    The live search serializes inference in one process. Board/backbone work
    runs at its existing thread count; only sufficiently large policy heads
    enter this scope. Set WOTR_LIVE_POLICY_THREADS=1 to disable the override.
    """
    stats = {}
    count = int(os.environ.get('WOTR_LIVE_POLICY_THREADS', '8') if threads is None else threads)
    if (not enabled or count <= 1 or not hasattr(net, 'action_scores')
            or not hasattr(net, 'spatial_action_scorer')
            or any(module.training for module in net.modules())):
        yield stats
        return
    import torch
    owner = net.spatial_action_scorer
    delta = getattr(net, 'policy_delta', None)
    original_scores = net.action_scores
    original_scorer = owner.forward
    originals = [(obj, name, name in obj.__dict__, obj.__dict__.get(name))
                 for obj, name in ((net, 'action_scores'), (owner, 'forward'))]
    frames = []
    stats.update(policy_threads=count, policy_thread_calls=0, policy_thread_rows=0)

    def scorer(x):
        if (frames and frames[-1]['restore'] is None and not torch.is_grad_enabled()
                and not net.training and not owner.training
                and (delta is None or not delta.training)
                and x.device.type == 'cpu' and x.shape[0] >= minimum_rows):
            previous = torch.get_num_threads()
            if previous == 1:
                frames[-1]['restore'] = previous
                torch.set_num_threads(count)
                stats['policy_thread_calls'] += 1
                stats['policy_thread_rows'] += int(x.shape[0])
        return original_scorer(x)

    def action_scores(*args, **kwargs):
        frame = {'restore': None}
        frames.append(frame)
        try:
            return original_scores(*args, **kwargs)
        finally:
            frames.pop()
            if frame['restore'] is not None:
                torch.set_num_threads(frame['restore'])

    net.action_scores = action_scores
    owner.forward = scorer
    try:
        yield stats
    finally:
        for obj, name, existed, value in reversed(originals):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
