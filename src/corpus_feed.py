"""Read-ahead for train_corpus.py's budget-batch loop.

The training step is dominated by cold page faults on the feed memmaps
(warm-cache assembly ~15ms vs ~103ms/step observed). Numpy-only spawn
workers touch upcoming batches' pages a window ahead; the main process
then assembles inline at warm speed. Workers return nothing (payload
IPC measured 3x slower than warm inline assembly), so the tensor data
path is the same in-process code with or without workers.
"""
import os
from collections import deque

import numpy as np

_G = {}


def init(ds_dir, tag):
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    _G.clear()
    _G['ds_dir'] = ds_dir
    _G['tag'] = tag


def _arrays():
    if 'X' not in _G:
        ds, tag = _G['ds_dir'], _G['tag']
        _G['X'] = np.load(os.path.join(ds, f'{tag}_X.npy'),
                          mmap_mode='r')
        _G['cands'] = np.load(os.path.join(ds, f'{tag}_cands.npy'),
                              mmap_mode='r')
        _G['regs'] = np.load(os.path.join(ds, f'{tag}_regs.npy'),
                             mmap_mode='r')
        _G['pos'] = np.load(os.path.join(ds, f'{tag}_pos.npy'),
                            mmap_mode='r')
        _G['off'] = np.load(os.path.join(ds, f'{tag}_off.npy'))
    return _G['X'], _G['cands'], _G['regs'], _G['pos'], _G['off']


def assemble(batch):
    """Sample indices -> (bi, xb, cb, rb, pb, mb); the single data path.
    cb/xb keep the on-disk dtype (f32 or f16); consumers cast on-GPU."""
    X, cands, regs, pos, off = _arrays()
    bi = np.sort(np.asarray(batch))
    B = len(bi)
    lens = (off[bi + 1] - off[bi]).astype(np.int64)
    Kb = int(lens.max())
    cb = np.zeros((B, Kb, cands.shape[1]), dtype=cands.dtype)
    rb = np.full((B, Kb), -1, dtype=np.int64)
    pb = np.zeros((B, Kb), dtype=bool)
    mb = np.zeros((B, Kb), dtype=bool)
    for j, si in enumerate(bi):
        lo, hi = int(off[si]), int(off[si + 1])
        k = hi - lo
        cb[j, :k] = cands[lo:hi]
        rb[j, :k] = regs[lo:hi]
        pb[j, :k] = pos[lo:hi].astype(bool)
        mb[j, :k] = True
    xb = np.ascontiguousarray(X[bi])
    return bi, xb, cb, rb, pb, mb


def warm(batch):
    """Fault a batch's memmap pages into the OS cache; returns nothing."""
    X, cands, regs, pos, off = _arrays()
    bi = np.sort(np.asarray(batch))
    X[bi].sum(dtype=np.float64)
    for si in bi:
        lo, hi = int(off[si]), int(off[si + 1])
        cands[lo:hi].sum(dtype=np.float64)
        regs[lo:hi].sum()
        pos[lo:hi].sum()
    return None


def warm_iter(pool, batches, window):
    """Yield assemble(b) for each batch in order, with workers warming
    the cache up to `window` batches ahead."""
    batches = list(batches)
    pending = deque()
    for b in batches[:window]:
        pending.append(pool.apply_async(warm, (b,)))
    for i, b in enumerate(batches):
        if pending:
            pending.popleft().get()
        k = i + window
        if k < len(batches):
            pending.append(pool.apply_async(warm, (batches[k],)))
        yield assemble(b)


def _convert_file(src, dst, chunk_bytes=1 << 29):
    a = np.load(src, mmap_mode='r')
    if a.dtype == np.float16:
        # already f16 at build time (current-generation cands arrays):
        # a chunked re-encode would be a slow identity copy
        del a
        import shutil
        shutil.copy2(src, dst + '.tmp')
        os.replace(dst + '.tmp', dst)
        return
    tmp = dst + '.tmp'
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16,
                                    shape=a.shape)
    rows = max(1, chunk_bytes // max(1, a.strides[0]))
    for lo in range(0, a.shape[0], rows):
        hi = min(lo + rows, a.shape[0])
        out[lo:hi] = a[lo:hi].astype(np.float16)
    out.flush()
    del out
    os.replace(tmp, dst)


def convert_to_f16(ds_dir, out_dir):
    """f16 twin of a feed dir: X/cands converted, everything else copied.
    Resumable (skips completed files; partials land as .tmp)."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(os.listdir(ds_dir),
                   key=lambda s: (not s.startswith('sh'), s))
    for name in names:
        if name.endswith('.tmp'):
            continue
        src = os.path.join(ds_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst):
            print('skip (exists): ' + name, flush=True)
            continue
        if name.endswith('_X.npy') or name.endswith('_cands.npy'):
            print('f16: ' + name, flush=True)
            _convert_file(src, dst)
        else:
            print('copy: ' + name, flush=True)
            shutil.copy2(src, dst + '.tmp')
            os.replace(dst + '.tmp', dst)
    print('CONVERT DONE', flush=True)


if __name__ == '__main__':
    import sys
    convert_to_f16(sys.argv[1], sys.argv[2])
