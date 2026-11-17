// Phase-1 smoke test of the WebAssembly build in Node: their rules engine and encoders, and our network, on the
// selftest positions, against the native library and torch (wasm/tools/export_fixtures.py wrote the references).
//
//   node wasm/smoke/smoke.mjs [wasm/build/alphawotr.mjs] [wasm/fixtures]
//
// Per position: the handed state's typed graph becomes a snapshot in the module; the full legal menu
// (wotr_snapshot_full_actions) must have the native library's row count; the v53 observation
// (wotr_snapshot_encode_state_v53) and the menu's v53 action rows and regions (wotr_snapshot_encode_actions_v53) must
// equal the native library's bit for bit; the network's priors and value on them (awnet_forward, the value net's
// value) must be within 1e-5 of torch's, with the same top action. Exit 1 on any difference.
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const modulePath = resolve(process.argv[2] || resolve(here, '../build/alphawotr.mjs'));
const fixtures = resolve(process.argv[3] || resolve(here, '../fixtures'));
const factory = (await import(pathToFileURL(modulePath).href)).default;
const M = await factory();
const is64 = modulePath.includes('64');
// pointers and size_t are Numbers on wasm32; on wasm64 (MEMORY64) the exports take and return BigInt
const P = is64 ? (x) => BigInt(x) : (x) => x;
const N = is64 ? (x) => Number(x) : (x) => x;
const heap = () => M.HEAPU8;

function copyIn(bytes) {
  const p = N(M._malloc(P(bytes.length)));
  heap().set(bytes, p);
  return p;
}
function check(handle, what) {
  if (!N(handle)) throw new Error(what + ': ' + M.UTF8ToString(M._wotr_snapshot_last_error()));
  return handle;
}

// awg: graphs by name -> pointer to their counts inside the module's memory (kept for the run)
function loadGraphs(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (b.toString('latin1', 0, 4) !== 'AWG1') throw new Error(path + ': not awg');
  const count = dv.getUint32(4, true);
  let off = 8;
  const graphs = new Map();
  for (let i = 0; i < count; i++) {
    const nl = dv.getUint32(off, true);
    const name = b.toString('utf8', off + 4, off + 4 + nl);
    off += 4 + nl;
    const nv = Number(dv.getBigUint64(off, true)), ne = Number(dv.getBigUint64(off + 8, true));
    const nb = Number(dv.getBigUint64(off + 16, true));
    const size = 32 + 40 * nv + 8 * ne + nb;
    graphs.set(name, copyIn(b.subarray(off, off + size)));
    off += size;
  }
  return graphs;
}

function loadRecords(path) {
  const b = readFileSync(path);
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (b.toString('latin1', 0, 4) !== 'AWR1') throw new Error(path + ': not awr');
  const count = dv.getUint32(4, true);
  let off = 8;
  const f32 = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = dv.getFloat32(off + 4 * i, true); off += 4 * n; return a; };
  const i32 = (n) => { const a = new Int32Array(n); for (let i = 0; i < n; i++) a[i] = dv.getInt32(off + 4 * i, true); off += 4 * n; return a; };
  const out = [];
  for (let c = 0; c < count; c++) {
    const side = dv.getUint32(off, true), rows = dv.getUint32(off + 4, true), menuRows = dv.getUint32(off + 8, true);
    off += 12;
    const obs = f32(4745), actions = f32(rows * 199), regions = i32(rows);
    const [head, vnet] = f32(2);
    const logits = f32(rows), priors = f32(rows), backbone = f32(256);
    out.push({ side, rows, menuRows, obs, actions, regions, head, vnet, logits, priors, backbone });
  }
  return out;
}

function loadNet(path) {
  const b = readFileSync(path);
  const p = copyIn(b);
  const net = M._awnet_load(P(p), P(b.length));
  if (!N(net)) throw new Error(path + ': ' + M.UTF8ToString(M._awnet_last_error()));
  return net;
}

console.log('module %s: abi %d, snapshot abi %d, controller abi %d', modulePath.split('/').pop(),
  M._wotr_abi_version(), M._wotr_snapshot_abi_version(), M._wotr_search_controller_abi_version());
const t0 = performance.now();
const tables = loadGraphs(resolve(fixtures, 'tables.awg'));
const obsTables = check(M._awasm_graph_create(P(tables.get('observation'))), 'observation tables');
if (M._wotr_snapshot_prepare_observation_tables(obsTables)) check(0, 'prepare observation tables');
const actTables = check(M._awasm_graph_create(P(tables.get('actions_v53'))), 'action tables');
if (M._wotr_snapshot_prepare_action_tables_v53(actTables)) check(0, 'prepare action tables');
const nets = [loadNet(resolve(fixtures, 'net_fp.awn')), loadNet(resolve(fixtures, 'net_sh.awn'))];
const vnets = [loadNet(resolve(fixtures, 'value_fp.awn')), loadNet(resolve(fixtures, 'value_sh.awn'))];
console.log('tables and 4 nets loaded in %d ms', Math.round(performance.now() - t0));

const positions = loadGraphs(resolve(fixtures, 'positions.awg'));
const refs = loadRecords(resolve(fixtures, 'positions.ref'));
let failures = 0;
refs.forEach((ref, n) => {
  const t1 = performance.now();
  const state = check(M._awasm_graph_create(P(positions.get('position' + n))), 'position ' + n);
  const menu = check(M._wotr_snapshot_full_actions(state, obsTables, ref.side), 'full actions ' + n);
  const rows = Number(M._awasm_snapshot_root_count(menu));   // int64: a BigInt on every target
  const obsP = N(M._malloc(P(4745 * 4)));
  if (M._wotr_snapshot_encode_state_v53(state, obsTables, ref.side, P(obsP), P(4745))) check(0, 'encode state');
  const obs = new Float32Array(M.HEAPU8.buffer, obsP, 4745).slice();
  const actP = N(M._malloc(P(Math.max(1, rows) * 199 * 4))), regP = N(M._malloc(P(Math.max(1, rows) * 8)));
  if (M._wotr_snapshot_encode_actions_v53(state, obsTables, menu, actTables, ref.side, P(actP), P(rows * 199), P(regP), P(rows)))
    check(0, 'encode actions');
  const acts = new Float32Array(M.HEAPU8.buffer, actP, rows * 199).slice();
  const regs = Array.from(new BigInt64Array(M.HEAPU8.buffer, regP, rows), Number);
  const priP = N(M._malloc(P(Math.max(1, rows) * 4))), valP = N(M._malloc(P(4)));
  const t2 = performance.now();
  M._awnet_forward(nets[ref.side], vnets[ref.side], P(obsP), P(actP), P(regP), P(rows), P(priP), P(valP));
  const t3 = performance.now();
  const priors = new Float32Array(M.HEAPU8.buffer, priP, rows).slice();
  const value = new Float32Array(M.HEAPU8.buffer, valP, 1)[0];
  const sameObs = obs.every((x, i) => Object.is(x, ref.obs[i]));
  const sameActs = rows === ref.rows && acts.every((x, i) => Object.is(x, ref.actions[i])) &&
    regs.every((x, i) => x === ref.regions[i]);
  let dp = 0, top = 0, topRef = 0;
  for (let i = 0; i < rows; i++) {
    dp = Math.max(dp, Math.abs(priors[i] - ref.priors[i]));
    if (priors[i] > priors[top]) top = i;
    if (ref.priors[i] > ref.priors[topRef]) topRef = i;
  }
  const dv = Math.abs(value - ref.vnet);
  const ok = rows === ref.menuRows && sameObs && sameActs && dp <= 1e-5 && dv <= 1e-5 && top === topRef;
  failures += !ok;
  console.log('[%s] position %d (%s): menu %d rows (native %d), observation %s, action rows %s, priors max diff %s, ' +
    'value %s (torch %s), top action %s; engine %d ms, network %d ms',
    ok ? 'PASS' : 'FAIL', n, ref.side ? 'SH' : 'FP', rows, ref.menuRows, sameObs ? 'identical' : 'DIFFERENT',
    sameActs ? 'identical' : 'DIFFERENT', dp.toExponential(2), value.toFixed(6), ref.vnet.toFixed(6),
    top === topRef ? 'same' : 'DIFFERENT', Math.round(t2 - t1), Math.round(t3 - t2));
  for (const p of [obsP, actP, regP, priP, valP]) M._free(P(p));
  M._wotr_snapshot_destroy(menu);
  M._wotr_snapshot_destroy(state);
});
console.log(failures ? 'FAIL (%d positions)' : 'PASS (%d positions)', failures || refs.length);
process.exit(failures ? 1 : 0);
