// The AlphaWOTR worker (docs/AlphaWOTR-Wasm-Plan.md section 2): a protocol bot in a module worker. The page posts
// {init: {base, sims}} once, then {line} for each referee message; the worker posts {ready} (or {error}) and {line}
// for each reply. The data files are fetched from `base` (cached by the browser).
import createModule from './alphawotr.mjs';
import { createBot, DATA_FILES } from './alphawotr-core.mjs';

let bot = null;
const queue = [];

async function init({ base, sims }) {
  const files = {};
  await Promise.all(DATA_FILES.map(async (name) => {
    const r = await fetch(new URL(name, base));
    if (!r.ok) throw new Error(name + ': HTTP ' + r.status);
    files[name] = new Uint8Array(await r.arrayBuffer());
  }));
  bot = await createBot(createModule, files, { sims });
  postMessage({ ready: true });
  while (queue.length) handle(queue.shift());
}

function handle(line) {
  const out = bot.message(line);
  if (out) postMessage({ line: out });
}

onmessage = (e) => {
  const d = e.data || {};
  if (d.init) { init(d.init).catch((err) => postMessage({ error: String(err && err.message || err) })); return; }
  if (typeof d.line === 'string') { if (bot) handle(d.line); else queue.push(d.line); }
};
