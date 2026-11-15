"""Inference-only PV grouping with one GPU-to-host synchronization.

The scorer, residual policy head, per-menu softmax and float32 arithmetic are
unchanged. Owners are known on CPU, avoiding CUDA repeat_interleave's dynamic
output-size synchronization. No waiting window or search allocation changes.
"""
import numpy as np
import torch


def policy_value_group(net, items, device, with_delta):
    if not items:
        return []
    counts = [len(item[1]) for item in items]
    with torch.no_grad():
        vecs = np.stack([item[0] for item in items])
        xt = torch.from_numpy(vecs).float().to(device)
        backbone, nodes = net.forward_backbone(xt)
        values = net.value_head(backbone).squeeze(-1).reshape(-1)
        encs = np.concatenate([item[1] for item in items])
        regs = np.concatenate([item[2] for item in items])
        owners = np.repeat(np.arange(len(items), dtype=np.int64), counts)
        owner = torch.from_numpy(owners).to(device)
        actions = torch.from_numpy(encs).float().to(device)
        regions = torch.from_numpy(regs).long().to(device)
        features, cards = net._embed_action_encs(actions)
        mean = nodes.mean(dim=1)
        region_rows = torch.where((regions < 0).unsqueeze(-1),
                                  mean[owner], nodes[owner, regions.clamp(min=0)])
        combined = torch.cat([backbone[owner], region_rows, features, cards], dim=-1)
        logits = net.spatial_action_scorer(combined).squeeze(-1)
        if with_delta and getattr(net, 'policy_delta', None) is not None:
            logits = logits + net.policy_delta(combined).squeeze(-1)
        # Empty menus preserve the server's existing empty-prior behavior.
        probabilities = [torch.softmax(part, dim=0)
                         for part in torch.split(logits, counts)]
        host = torch.cat([values, *probabilities]).detach().cpu().numpy()
        offset = len(items)
        result = []
        for index, count in enumerate(counts):
            result.append((host[offset:offset+count].astype(np.float32, copy=True),
                           float(host[index])))
            offset += count
        return result
