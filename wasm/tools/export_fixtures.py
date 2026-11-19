"""Export what the WebAssembly port needs from the Python side, and the references it is checked against.

    venv/bin/python -I wasm/tools/export_fixtures.py --alpha ALPHAWOTR_CHECKOUT --out wasm/fixtures [--positions N]
        [--search-positions K --sims S --max-cases M]

ALPHAWOTR_CHECKOUT is a checkout of this repository with the native library built for this machine
(python native_backend/live_search_oct19/build_portable.py); the engine, the nets and the native library are taken
from there, exactly as the AI server loads them (wotr_bridge/launch_head.py HEAD_ENV, own_model.load_own_net).

Writes into --out:
  net_fp.awn, net_sh.awn          the seats' league nets (launch_head.NET_TAG), weights only   (awn format below)
  value_fp.awn, value_sh.awn      the value nets bound to them (own_model.SEARCH_VALUE_NETS)
  tables.awg                      the static native tables as typed graphs (awg format below):
                                    observation  (observation_tables.table_data, before prepare_observation_tables)
                                    actions_v53  (native_action_features.action_tables_v53, before prepare)
  positions.awg                   the selftest positions' handed states as typed graphs (snapshot.encode_graph)
  positions.ref                   per position: side, the native full menu's row count, the native v53
                                  observation, the native v53 action rows and regions of that menu, and torch's
                                  value (head and value net), logits and priors on them               (awr format)
  nn_cases.ref                    leaf requests recorded from native searches (NumericInferenceBridge.run; PV, PVM and value-only EVAL) with
                                  torch's outputs on each leaf, one leaf at a time                    (awr format)

awn: b'AWN1', u32 header bytes, header (UTF-8 JSON: {"meta":{...},"tensors":[{"name","shape","dtype","offset"}]}),
     then the tensor data, little-endian, float32 or int32, at 'offset' bytes after the header.
awg: b'AWG1', u32 count, per graph: u32 name length, name, u64 values, u64 edges, u64 bytes, u64 root,
     values * 40 bytes (struct WotrValue), edges * u64, bytes.
awr: b'AWR1', u32 count, per record: u32 side (0 FP, 1 SH), u32 rows, u32 menu_rows (positions only, else 0),
     f32 obs[4745], f32 actions[rows*199], i32 regions[rows], f32 head_value, f32 vnet_value,
     f32 logits[rows], f32 priors[rows] (softmax; the bridge then divides by sum+1e-10), f32 backbone[256].
"""
import argparse
import ctypes as C
import json
import os
import pickle
import struct
import sys

import numpy as np


def write_awn(path, net, meta):
    sd = net.state_dict()
    header_t, blobs, offset = [], [], 0
    for name, tensor in sd.items():
        if name.endswith('num_batches_tracked'):
            continue
        t = tensor.detach().cpu().contiguous()
        if t.dtype.is_floating_point:
            arr, dtype = t.numpy().astype('<f4'), 'f32'
        else:
            arr, dtype = t.numpy().astype('<i4'), 'i32'
        header_t.append(dict(name=name, shape=list(arr.shape), dtype=dtype, offset=offset))
        b = arr.tobytes()
        blobs.append(b)
        offset += len(b)
    header = json.dumps(dict(meta=meta, tensors=header_t), sort_keys=True).encode('utf-8')
    header += b' ' * ((-len(header)) % 4)
    with open(path, 'wb') as f:
        f.write(b'AWN1' + struct.pack('<I', len(header)) + header)
        for b in blobs:
            f.write(b)


def graph_bytes(name, graph):
    values = bytes(graph.values)
    edges = bytes(graph.edges)
    assert len(values) == 40 * len(graph.values)
    nb = name.encode('utf-8')
    return (struct.pack('<I', len(nb)) + nb + struct.pack('<QQQQ', len(graph.values), len(graph.edges), len(graph.data),
                                                          graph.root) + values + edges + graph.data)


def write_awg(path, graphs):
    with open(path, 'wb') as f:
        f.write(b'AWG1' + struct.pack('<I', len(graphs)))
        for name, graph in graphs:
            f.write(graph_bytes(name, graph))


def torch_reference(ai, net, obs, actions, regions):
    """One leaf as numeric_bridge._pv computes it (forward_backbone, value head or value net, scorer + delta)."""
    torch = ai.torch
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.array([obs], dtype=np.float32))
        bb, ne = net.forward_backbone(x)
        head = float(net.value_head(bb).squeeze(-1).item())
        vm = getattr(net, 'value_model', None)
        vnet = float(vm.values([obs])[0]) if vm is not None else float('nan')
        if len(actions):
            ae = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
            ri = torch.from_numpy(np.ascontiguousarray(regions, dtype=np.int64))
            logits = net.action_scores(bb, ne, ae, ri).cpu().numpy().astype(np.float32)
            priors = torch.softmax(torch.from_numpy(logits), dim=0).numpy().astype(np.float32)
        else:
            logits = priors = np.zeros(0, np.float32)
    return head, vnet, logits, priors, bb[0].cpu().numpy().astype(np.float32)


def awr_record(side, obs, actions, regions, menu_rows, ref):
    head, vnet, logits, priors, bb = ref
    rows = len(regions)
    out = struct.pack('<III', side, rows, menu_rows)
    out += np.asarray(obs, '<f4').tobytes() + np.asarray(actions, '<f4').reshape(-1).tobytes()
    out += np.asarray(regions, '<i4').tobytes() + struct.pack('<ff', head, vnet)
    out += np.asarray(logits, '<f4').tobytes() + np.asarray(priors, '<f4').tobytes() + np.asarray(bb, '<f4').tobytes()
    return out


def write_awr(path, records):
    with open(path, 'wb') as f:
        f.write(b'AWR1' + struct.pack('<I', len(records)))
        for r in records:
            f.write(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--positions', type=int, default=0, help='0 = all selftest positions')
    ap.add_argument('--search-positions', type=int, default=6)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--max-cases', type=int, default=150, help='per search')
    a = ap.parse_args()
    alpha = os.path.abspath(a.alpha)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    bridge_dir = os.path.join(alpha, 'wotr_bridge')
    sys.path[:0] = [os.path.join(alpha, 'src'), bridge_dir]
    import launch_head as lh
    for key, val in lh.HEAD_ENV:
        os.environ[key] = val
    os.chdir(os.path.join(alpha, 'src'))
    import ai
    from core import own_model
    sides = {'FREE_PEOPLES': ai.Side.FREE_PEOPLES, 'SHADOW': ai.Side.SHADOW}
    nets = {s: own_model.load_own_net(lh.NET_TAG, s) for s in sides.values()}
    seed = 20261114
    searches = {s: own_model.OwnModelSearch(s, nets[s], seed=seed, backend='native') for s in sides.values()}

    # 1. weights
    consts = dict(obs_dim=int(nets[ai.Side.FREE_PEOPLES].input_dim), act_dim=int(ai.ACTION_ENCODING_DIM),
                  regions=int(ai.NUM_REGIONS), node_feat=int(ai.NODE_FEAT_DIM), card_idx_start=int(ai._CARD_IDX_START),
                  hand=int(ai.MAX_HAND_SIZE), card_pad=int(ai.CARD_PAD_IDX), action_card_col=int(ai._ACTION_CARD_IDX_DIM))
    for s, tag in ((ai.Side.FREE_PEOPLES, 'fp'), (ai.Side.SHADOW, 'sh')):
        net = nets[s]
        head_act = 'scaled2' if isinstance(net.value_head[-1], ai._ScaledTanhHead) else 'tanh'
        write_awn(os.path.join(out, 'net_%s.awn' % tag), net, dict(consts, role='policy', side=tag, net_tag=lh.NET_TAG,
                                                                    head_act=head_act))
        vm = getattr(net, 'value_model', None)
        if vm is not None:
            write_awn(os.path.join(out, 'value_%s.awn' % tag), vm.model,
                      dict(consts, role='value', side=tag, head_act='tanh', sha256=vm.sha256, name=str(vm.name)))
        print('weights %s: head %s, value net %s' % (tag, head_act, vm and vm.sha256[:12]), flush=True)

    # 2. static tables (as their graphs before the native prepare step; the wasm side prepares them itself)
    native_dir = os.path.join(alpha, 'native_backend')
    sys.path[:0] = [native_dir]
    import snapshot, observation_tables, native_action_features
    obs_graph = snapshot.encode_graph(ai, observation_tables.table_data(ai))
    act_graph = snapshot.encode_graph(ai, native_action_features.action_tables_v53(ai))
    write_awg(os.path.join(out, 'tables.awg'), [('observation', obs_graph), ('actions_v53', act_graph)])
    print('tables: observation %d values, actions_v53 %d values' % (len(obs_graph.values), len(act_graph.values)))

    # 3. positions: graphs + native menu/encodings + torch references
    with open(os.path.join(alpha, 'wotr_bridge', 'selftest_positions.pkl'), 'rb') as f:
        rows = pickle.load(f)
    if a.positions:
        rows = rows[:a.positions]
    tables = observation_tables.native_tables(ai)
    features = native_action_features.ActionFeaturesV53(ai)
    lib = snapshot.library()
    enc = lib.wotr_snapshot_encode_state_v53
    enc.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.POINTER(C.c_float), C.c_size_t]
    enc.restype = C.c_int
    graphs, refs = [], []
    for n, r in enumerate(rows):
        side = sides[r['side']]
        si = 0 if side is ai.Side.FREE_PEOPLES else 1
        state = pickle.loads(r['handed'])
        graph = snapshot.encode_graph(ai, state)
        graphs.append(('position%d' % n, graph))
        handle = snapshot.NativeSnapshot.from_graph(graph)
        obs = np.zeros(consts['obs_dim'], np.float32)
        if enc(handle._pointer, tables._pointer, si, obs.ctypes.data_as(C.POINTER(C.c_float)), obs.size):
            raise snapshot._error(lib)
        py_obs = np.asarray(ai.encode_state(state, side), np.float32)
        with handle.menu_handle(ai, si, 'full') as menu:
            count = menu.counts[0] and int(menu.export_graph().values[menu.root].count)
            feats, regions = features.encode(handle, menu, si, count)
        ref = torch_reference(ai, nets[side], obs, feats, regions)
        refs.append(awr_record(si, obs, feats, regions, count, ref))
        print('position %d %s turn %s: menu %d rows, native obs == python obs: %s, top prior %.4f' % (
            n, r['side'], r['turn'], count, bool(np.array_equal(obs, py_obs)), float(ref[3].max()) if count else 0),
            flush=True)
        handle.close()
    write_awg(os.path.join(out, 'positions.awg'), graphs)
    write_awr(os.path.join(out, 'positions.ref'), refs)

    # 4. leaf requests recorded from native searches
    sys.path[:0] = [os.path.join(native_dir, 'experiments/native_search_core/inference')]
    import numeric_bridge
    cases = []
    original = numeric_bridge.NumericInferenceBridge.run
    limit = [0]

    def recording(self, request):
        if len(cases) < limit[0] and int(request.operation) in (1, 2, 3):
            net = self.nets[int(request.model_side)]
            bounds = request.bounds() if int(request.operation) != 3 else [(0, 0)] * len(request.leaf_ids)
            for i, (lo, hi) in enumerate(bounds):
                if len(cases) >= limit[0]:
                    break
                obs = np.array(request.observations[i], np.float32)
                acts = np.array(request.actions[lo:hi], np.float32)
                regs = np.array(request.regions[lo:hi], np.int64)
                cases.append(awr_record(int(request.model_side), obs, acts, regs, 0,
                                        torch_reference(ai, net, obs, acts, regs)))
        return original(self, request)

    numeric_bridge.NumericInferenceBridge.run = recording
    try:
        for n, r in enumerate(rows[:a.search_positions]):
            side, owner = sides[r['side']], sides.get(r['owner'])
            before = len(cases)
            limit[0] = before + a.max_cases
            with ai.search_turn_owner(owner):
                act, _ = searches[side].choose(pickle.loads(r['handed']), side, num_simulations=a.sims)
            print('search %d: %s, %d leaf cases recorded' % (n, act.action_type, len(cases) - before), flush=True)
    finally:
        numeric_bridge.NumericInferenceBridge.run = original
    write_awr(os.path.join(out, 'nn_cases.ref'), cases)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
