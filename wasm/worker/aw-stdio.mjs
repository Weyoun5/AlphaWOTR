#!/usr/bin/env node
// AlphaWOTR on WebAssembly as a stdio protocol bot (one JSON message a line in, one reply a line out), for
// `cli referee` (proc:) and the transcript checks.   node aw-stdio.mjs DATA_DIR [--sims N]
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { createBot, DATA_FILES } from './alphawotr-core.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const dir = args[0];
const si = args.indexOf('--sims');
const sims = si >= 0 ? Number(args[si + 1]) : 300;
const files = {};
for (const name of DATA_FILES) files[name] = new Uint8Array(fs.readFileSync(path.join(dir, name)));
const { default: createModule } = await import(path.join(here, '..', 'build', 'alphawotr.mjs'));
const bot = await createBot(createModule, files, { sims });
const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  if (!line.trim()) continue;
  const out = bot.message(line);
  if (out) process.stdout.write(out + '\n');
  try { if (JSON.parse(line).type === 'quit') break; } catch (e) { /* not JSON */ }
}
bot.destroy();
