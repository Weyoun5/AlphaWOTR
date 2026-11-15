"""Exact, search-local memoization for an immutable CPU inference model.

Different hidden-card samples often have identical board inputs. Cache the
board encoder; keys contain all input bytes, shape, stride and dtype. Complete
backbone/value inputs rarely repeat and cost more memory to retain than they
save. Nothing survives a search or a checkpoint replacement.
"""
from collections import OrderedDict
from contextlib import contextmanager


@contextmanager
def cached_inference(net, *, enabled=True, capacity_bytes=128*1024*1024):
    stats={}
    if (not enabled or getattr(net,'arch_kind',None) == 'transformer'
            or not hasattr(net,'board_gnn')
            or any(module.training for module in net.modules())):
        yield stats
        return
    import torch
    restorations=[]
    # The live search owns an immutable inference model for this entire scope.
    # Its frozen numeric helpers call eval() thousands of times, recursively
    # resetting the same module flags. Establish that mode once, then retain it.
    # Training models remain untouched and use the uncached path below.
    existed='eval' in net.__dict__
    previous=net.__dict__.get('eval')
    net.eval=lambda:net
    restorations.append((net,'eval',existed,previous))
    def wrap(owner,name,label):
        existed=name in owner.__dict__
        previous=owner.__dict__.get(name)
        original=getattr(owner,name)
        cache=OrderedDict()
        used=0
        stats[label+'_hits']=stats[label+'_misses']=0
        def forward(x):
            nonlocal used
            if torch.is_grad_enabled() or net.training or x.device.type!='cpu':
                return original(x)
            key=(str(x.dtype),tuple(x.shape),tuple(x.stride()),x.detach().numpy().tobytes())
            hit=cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                stats[label+'_hits']+=1
                return hit[0]
            stats[label+'_misses']+=1
            result=original(x)
            tensors=result if isinstance(result,tuple) else (result,)
            size=len(key[3])+sum(t.numel()*t.element_size() for t in tensors)+256
            if size<=capacity_bytes:
                while cache and used+size>capacity_bytes:
                    _,(_,old_size)=cache.popitem(last=False)
                    used-=old_size
                cache[key]=(result,size)
                used+=size
            return result
        setattr(owner,name,forward)
        restorations.append((owner,name,existed,previous))
    wrap(net.board_gnn,'forward','board')
    try:
        yield stats
    finally:
        for owner,name,existed,previous in reversed(restorations):
            if existed:setattr(owner,name,previous)
            else:delattr(owner,name)
