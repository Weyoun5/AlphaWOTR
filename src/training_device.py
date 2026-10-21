"""Explicit loop device selection; operators follow their actual model device.

The loop configures its process-wide ai loader once before use, just as the
self-play worker does. Scheduler visibility masks remain authoritative.
PyTorch ROCm uses the same cuda device names; hardware acceptance is separate.
"""
import os
import re

VISIBILITY_KEYS = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')


def requested_training_device():
    name = os.environ.get('WOTR_TRAINING_DEVICE', 'cpu')
    if name != 'cpu' and re.fullmatch(r'cuda(?::(?:0|[1-9][0-9]*))?', name) is None:
        raise ValueError('WOTR_TRAINING_DEVICE must be cpu, cuda, or cuda:<visible index>')
    return name


def resolve_training_device():
    import torch
    name = requested_training_device()
    if name == 'cpu':
        return torch.device('cpu')
    index = int(name.split(':')[1]) if ':' in name else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError('Explicit training device %s is unavailable within the assigned GPU visibility masks' % name)
    return torch.device('cuda', index)


def activate_training_device():
    """Set the loop's model-loader device before models/probes are loaded."""
    device = resolve_training_device()
    import ai
    ai.DEVICE = device
    return device


def model_device(net):
    """No global-device assumption in standalone operators or probes."""
    parameter = next(net.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(net.buffers(), None)
    if buffer is not None:
        return buffer.device
    # Stateless evaluators (for example nn.Identity) consume CPU inputs from
    # the NumPy trajectory, without needing a trainable parameter.
    import torch
    return torch.device('cpu')
