// AlphaWOTR (MIT, see the repository's LICENSE) as a WotR Bot Protocol bot on WebAssembly: the module's bot
// (aw_create / aw_message) over data files handed in as bytes. Shared by the browser worker and the Node wrapper.
//
//   const bot = await createBot(Module, files, {sims: 300});   // files: {name: Uint8Array}
//   const reply = bot.message(line);                           // '' when the message has no reply
export const DATA_FILES = ['adapter.json', 'prototypes.awg', 'tables.awg',
                           'net_fp.awn', 'net_sh.awn', 'value_fp.awn', 'value_sh.awn'];

export async function createBot(moduleFactory, files, options = {}) {
  const M = await moduleFactory();
  try { M.FS.mkdir('/aw'); } catch (e) { /* exists */ }
  for (const name of DATA_FILES) {
    if (!files[name]) throw new Error('AlphaWOTR: missing data file ' + name);
    M.FS.writeFile('/aw/' + name, files[name]);
  }
  const str = (s) => { const n = M.lengthBytesUTF8(s) + 1; const p = M._malloc(n); M.stringToUTF8(s, p, n); return p; };
  const cfg = str(JSON.stringify({dir: '/aw', sims: options.sims ?? 300, max_seconds: options.maxSeconds ?? null}));
  const handle = M._aw_create(cfg);
  M._free(cfg);
  if (!handle) throw new Error('AlphaWOTR: ' + M.UTF8ToString(M._aw_last_error()));
  // the network's tables are in the module now: the files need not stay in its file system
  return {
    message(line) {
      const p = str(line);
      const out = M.UTF8ToString(M._aw_message(handle, p));
      M._free(p);
      return out;
    },
    lastError() { return M.UTF8ToString(M._aw_last_error()); },
    destroy() { M._aw_destroy(handle); },
  };
}
