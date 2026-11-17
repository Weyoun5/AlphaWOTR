"""Record whole native searches as the live AI server runs them, for the module's replay and parity tests.

    venv/bin/python -I wasm/tools/record_searches.py --alpha ALPHAWOTR_CHECKOUT --out wasm/fixtures
        [--seeds 3] [--sims 300] [--positions FILE.pkl ...]

Each decision of the selftest positions (wotr_bridge/selftest_positions.pkl, and any --positions file of the same
shape: dicts with 'handed' (a pickled GameState), 'side', 'owner', 'menu') is searched by OwnModelSearch on the
native backend with simulations from strength.plan(1, menu, --sims), once per seed. Hooks on the controller binding
record what crosses the boundary, in order, and the root's final pick; the search itself is unchanged.

Writes into --out:
  tables.awg     extended with the menu-category and Hunt chance tables ('menu', 'hunt'), next to 'observation'
                 and 'actions_v53' (export_fixtures.py writes those two; this script rewrites the file with all four)
  SET/NNNN.aws   one search a file (format below; count 1); SET is --set (default searches)
  or, with --verdicts FILE, only the same-but-better test records:
  awv: b'AWV1', u32 count, per search: u32 side, u32 position, graph: root state, graph: root menu,
       i64 better[K], u32 has_better

aws: b'AWS1', u32 count, then per search:
  u32 position, u32 seed index, u32 side (0 FP, 1 SH), u32 simulations
  u32 n, config bytes (struct WotrSearchConfig as sent);  u32 n, own config bytes (struct WotrOwnValueConfig)
  rng: u32 words[624], u32 index, i32 has_gauss, u64 gauss bits             (the search stream at creation)
  graph: the root state                                                     (awg graph record: name, counts, arrays)
  u32 events, each:
    u8 kind = 1 numeric: u64 request_id, i32 op, i32 model_side, i32 purpose, u64 batch, u64 rows,
        u64 leaf_ids[batch], u64 offsets[batch+1], f32 obs[batch*4745], f32 actions[rows*199], i64 regions[rows],
        reply: f64 values[batch], f64 conf[batch], and when op != 3: f32 priors[rows], f64 mixed[rows]
    u8 kind = 2 combat distribution: u64 request_id, i32 model_side, u64 leaf_id,
        graph: the request payload, graph: the reply
  root view: u64 K, u64 visits[K], u64 term[K], u64 forced[K], f64 value_sum[K], f64 value_sq[K], f64 conf_sum[K],
        f64 phi_sum[K], f32 priors_clean[K], i8 proven[K], f64 proven_value[K], i32 side[K], i32 root_side,
        u64 simulations_run, u64 budget
  rng after: as above
  graph: the root menu (descriptor rows)
  i64 better[K] (ai._strictly_better's verdicts; all -1 when it returned None), u32 has_better
  u32 pick (build_visit_info's index), f64 seconds (the whole native search, Python included)
  u64 seat seed, u32 search index (OwnModelSearch: the own-value seed is derive_seed(seat seed,
      'bridge-own-reply:INDEX')), u64 search seed (random.seed before the search, 0 = not seeded: --seeded),
  i32 turn owner (0 none, 1 FP, 2 SH)
"""
import argparse
import ctypes as C
import os
import pickle
import random
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rng_bytes(state):
    words = state[1][:624]
    index = state[1][624]
    gauss = state[2]
    bits = 0 if gauss is None else struct.unpack('<Q', struct.pack('<d', gauss))[0]
    return struct.pack('<624I', *words) + struct.pack('<IiQ', index, int(gauss is not None), bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--positions', nargs='*', default=[])
    ap.add_argument('--limit', type=int, default=0, help='positions per file (0 = all)')
    ap.add_argument('--set', default='searches', help='the folder under --out for the .aws files')
    ap.add_argument('--seeded', action='store_true', help='seed the search stream (random.seed) before each search '
                    'and record the seeds, so the module can set the search up from the state alone')
    ap.add_argument('--verdicts', default='', help='write only the same-but-better test records (awv) to this file, '
                    'no .aws (the root state, the root menu, the verdicts; a few hundred kB a search)')
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import export_fixtures as ef
    alpha = os.path.abspath(a.alpha)
    out = os.path.abspath(a.out)
    verdicts_path = os.path.abspath(a.verdicts) if a.verdicts else ''
    files = [os.path.abspath(p) for p in a.positions] or [os.path.join(alpha, 'wotr_bridge', 'selftest_positions.pkl')]
    os.makedirs(out, exist_ok=True)
    sys.path[:0] = [os.path.join(alpha, 'src'), os.path.join(alpha, 'wotr_bridge')]
    import launch_head as lh
    for key, val in lh.HEAD_ENV:
        os.environ[key] = val
    os.chdir(os.path.join(alpha, 'src'))
    import ai
    from core import own_model, strength
    sides = {'FREE_PEOPLES': ai.Side.FREE_PEOPLES, 'SHADOW': ai.Side.SHADOW}
    nets = {s: own_model.load_own_net(lh.NET_TAG, s) for s in sides.values()}
    seeds = [20261114 + 7919 * i for i in range(a.seeds)]
    searches = {(s, i): own_model.OwnModelSearch(s, nets[s], seed=seed, backend='native')
                for s in sides.values() for i, seed in enumerate(seeds)}
    native_dir = os.path.join(alpha, 'native_backend')
    import snapshot, observation_tables, native_action_features
    import controller_binding as base
    import own_binding
    native = searches[(ai.Side.FREE_PEOPLES, 0)].native          # the orchestrator module (native_search.load)
    res = native.resources(ai)

    # 1. tables: the two of export_fixtures plus the menu categories and the Hunt chance table
    graphs = [('observation', snapshot.encode_graph(ai, observation_tables.table_data(ai))),
              ('actions_v53', snapshot.encode_graph(ai, native_action_features.action_tables_v53(ai))),
              ('menu', res.menu.tables.export_graph()), ('hunt', res.chance.export_graph())]
    ef.write_awg(os.path.join(out, 'tables.awg'), graphs)

    # 2. hooks: what crosses the boundary, in order
    rec = {}

    def graph(name, g):
        return ef.graph_bytes(name, g)

    init0 = own_binding.OwnController.__init__

    def init(self, lib, root_state, observation_tables_, action_tables, menu_tables, hunt_tables, rng, config, **kw):
        init0(self, lib, root_state, observation_tables_, action_tables, menu_tables, hunt_tables, rng, config, **kw)
        rec.update(config=bytes(self._config), own=bytes(self._own_config), rng=rng_bytes(rng.getstate()),
                   root=graph('root', root_state.export_graph()), events=[])
    own_binding.OwnController.__init__ = init

    request0, submit0 = base.Controller.request, base.Controller.submit
    payload0, submit_payload0 = base.Controller.request_payload, base.Controller.submit_payload
    root0, rng0, menu0 = base.Controller.root, base.Controller.rng_state, base.Controller.root_menu

    def request(self):
        view = request0(self)
        rec['pending'] = view
        return view

    def submit(self, reply):
        v = rec.pop('pending')
        b = struct.pack('<BQiiiQQ', 1, v.request_id, v.operation, v.model_side, v.purpose, v.batch, v.rows)
        b += np.asarray(v.leaf_ids, '<u8').tobytes() + np.asarray(v.offsets, '<u8').tobytes()
        b += np.asarray(v.observations, '<f4').tobytes() + np.asarray(v.actions, '<f4').tobytes()
        b += np.asarray(v.regions, '<i8').tobytes()
        b += np.asarray(reply.values, '<f8').tobytes() + np.asarray(reply.conf, '<f8').tobytes()
        if v.operation != 3:
            b += np.asarray(reply.priors, '<f4').tobytes() + np.asarray(reply.mixed, '<f8').tobytes()
        rec['events'].append(b)
        return submit0(self, reply)

    def request_payload(self, ai_):
        pointer = self._lib.wotr_search_controller_request_payload(self._pointer)
        with snapshot.NativeSnapshot(self._lib, pointer) as p:
            rec['payload'] = graph('payload', p.export_graph())
        return payload0(self, ai_)

    def submit_payload(self, reply_snapshot):
        v = rec.pop('pending')
        b = struct.pack('<BQiQ', 2, v.request_id, v.model_side, int(v.leaf_ids[0]))
        b += rec.pop('payload') + graph('reply', reply_snapshot.export_graph())
        rec['events'].append(b)
        return submit_payload0(self, reply_snapshot)

    def root(self):
        r = root0(self)
        b = struct.pack('<Q', r.children)
        for name, dt in (('visits', '<u8'), ('term_visits', '<u8'), ('forced_visits', '<u8'), ('value_sum', '<f8'),
                         ('value_sq', '<f8'), ('conf_sum', '<f8'), ('phi_sum', '<f8'), ('priors_clean', '<f4'),
                         ('proven', '<i1'), ('proven_value', '<f8'), ('side', '<i4')):
            b += np.asarray(getattr(r, name), dt).tobytes()
        rec['root_view'] = b + struct.pack('<iQQ', r.root_side, r.simulations_run, r.budget)
        return r

    def rng_state(self):
        st = rng0(self)
        rec['rng_after'] = rng_bytes(st)
        return st

    def root_menu(self):
        m = menu0(self)
        rec['menu'] = graph('menu', m.export_graph())
        return m

    base.Controller.request, base.Controller.submit = request, submit
    base.Controller.request_payload, base.Controller.submit_payload = request_payload, submit_payload
    base.Controller.root, base.Controller.rng_state, base.Controller.root_menu = root, rng_state, root_menu
    visit0 = native.build_visit_info

    def build_visit_info(ai_, root_, root_actions, action_encs, region_indices, temperature, better=None):
        idx, info = visit0(ai_, root_, root_actions, action_encs, region_indices, temperature, better)
        k = int(root_.children)
        rec['better'] = struct.pack('<%dq' % k, *([-1] * k if better is None else [int(x) for x in better]))
        rec['better'] += struct.pack('<I', int(better is not None))
        rec['pick'] = int(idx)
        return idx, info
    native.build_visit_info = build_visit_info

    # 3. the searches
    records = []
    for fi, path in enumerate(files):
        with open(path, 'rb') as f:
            rows = pickle.load(f)
        if a.limit:
            rows = rows[:a.limit]
        for n, r in enumerate(rows):
            side, owner = sides[r['side']], sides.get(r['owner'])
            menu = [ai.Action(die_idx=d, action_type=k, params=p) for d, k, p in r['menu']]
            if r.get('sims'):                  # harvested decisions carry the simulations the adapter asked for
                sims = int(r['sims'])
            else:
                sims, _setting = strength.plan(1, menu, a.sims)
                if len(menu) < 2:
                    continue
            if not sims:
                continue
            for si in range(len(seeds)):
                rec.clear()
                search = searches[(side, si)]
                own_index = search.index
                search_seed = (7000000 + 1000 * (1000 * fi + n) + si) if a.seeded else 0
                if a.seeded:
                    random.seed(search_seed)
                t0 = time.perf_counter()
                try:
                    with ai.search_turn_owner(owner):
                        act, _ = searches[(side, si)].choose(pickle.loads(r['handed']), side, num_simulations=sims)
                except Exception as error:      # (their search refuses a few positions: 'Unsupported ...')
                    print('position %d.%d seed %d: the native search refused: %s: %s' % (
                        fi, n, si, type(error).__name__, str(error)[:120]), flush=True)
                    continue
                seconds = time.perf_counter() - t0
                if 'pick' not in rec:
                    print('position %d.%d seed %d: no native search ran (forced or refused)' % (fi, n, si))
                    continue
                si_side = 0 if side is ai.Side.FREE_PEOPLES else 1
                b = struct.pack('<IIII', 1000 * fi + n, si, si_side, sims)
                b += struct.pack('<I', len(rec['config'])) + rec['config']
                b += struct.pack('<I', len(rec['own'])) + rec['own'] + rec['rng'] + rec['root']
                b += struct.pack('<I', len(rec['events'])) + b''.join(rec['events'])
                b += rec['root_view'] + rec['rng_after'] + rec['menu'] + rec['better']
                b += struct.pack('<Id', rec['pick'], seconds)
                owner_code = 0 if owner is None else (1 if owner is ai.Side.FREE_PEOPLES else 2)
                b += struct.pack('<QIQi', seeds[si], own_index, search_seed, owner_code)
                if a.verdicts:
                    v = struct.pack('<II', si_side, 1000 * fi + n) + rec['root'] + rec['menu'] + rec['better']
                    records.append(v)
                else:
                    records.append(b)
                kinds = [e[0] for e in rec['events']]
                print('position %d.%d (%s, turn %s) seed %d: %d sims, %d requests (%d combat), pick %d %s, %.2f s'
                      % (fi, n, r['side'], r.get('turn'), si, sims, len(kinds), kinds.count(2), rec['pick'],
                         act.action_type, seconds), flush=True)
    if a.verdicts:
        with open(verdicts_path, 'wb') as f:
            f.write(b'AWV1' + struct.pack('<I', len(records)))
            for v in records:
                f.write(v)
        print('wrote %d verdict records to %s' % (len(records), a.verdicts))
        return
    folder = os.path.join(out, a.set)
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        if name.endswith('.aws'):
            os.remove(os.path.join(folder, name))
    for i, b in enumerate(records):
        with open(os.path.join(folder, '%04d.aws' % i), 'wb') as f:
            f.write(b'AWS1' + struct.pack('<I', 1) + b)
    print('wrote %d searches to %s' % (len(records), folder))


if __name__ == '__main__':
    main()
