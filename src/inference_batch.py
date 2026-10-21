"""Bounded spatial scoring across local GPU search leaves.

Each leaf retains its own action order and softmax. This inference-only helper
uses the existing scorer and residual head; it does not change training math.
"""
import numpy as np
import torch


def spatial_policy_value_batch(net, vecs, aes, ris, device, with_delta,
                               max_rows=16384):
    if len(aes) != len(vecs) or len(ris) != len(vecs):
        raise ValueError('Policy batch counts differ')
    if not vecs:
        return []
    if max_rows < 1:
        raise ValueError('max_rows must be positive')
    counts = [len(a) for a in aes]
    for a, r in zip(aes, ris):
        if a.ndim != 2 or r.ndim != 1 or len(a) != len(r):
            raise ValueError('Action and region shapes differ')
        if len(a) == 0:
            raise ValueError('Policy leaf has no actions')
    with torch.no_grad():
        x = torch.from_numpy(np.stack(vecs).astype(np.float32, copy=False)).to(device)
        bb, ne = net.forward_backbone(x)
        vals = net.value_head(bb).squeeze(-1).detach().cpu().numpy().reshape(-1)
        if vals.shape != (len(vecs),):
            raise ValueError('Policy value batch shape differs')
        encs = np.concatenate(aes)
        regs = np.concatenate(ris)
        owners = np.repeat(np.arange(len(vecs), dtype=np.int64), counts)
        mean = ne.mean(dim=1)
        pieces = []
        for start in range(0, len(encs), max_rows):
            end = start + max_rows
            ae = torch.from_numpy(encs[start:end]).to(device)
            ri = torch.from_numpy(regs[start:end]).to(device)
            own = torch.from_numpy(owners[start:end]).to(device)
            features, cards = net._embed_action_encs(ae)
            region = torch.where((ri < 0).unsqueeze(-1), mean[own], ne[own, ri.clamp(min=0)])
            combined = torch.cat([bb[own], region, features, cards], dim=-1)
            logits = net.spatial_action_scorer(combined).squeeze(-1)
            if with_delta and getattr(net, 'policy_delta', None) is not None:
                logits = logits + net.policy_delta(combined).squeeze(-1)
            if logits.shape != (len(ae),):
                raise ValueError('Policy scorer row shape differs')
            pieces.append(logits)
        logits = torch.cat(pieces)
        probabilities = [torch.softmax(part, dim=0) for part in torch.split(logits, counts)]
        host = torch.cat(probabilities).detach().cpu().numpy()
        result = []
        pos = 0
        for i, count in enumerate(counts):
            result.append((host[pos:pos + count].copy(), float(vals[i])))
            pos += count
        return result
