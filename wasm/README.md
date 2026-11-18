AlphaWOTR in WebAssembly
========================

AlphaWOTR's strength-1 decision running in a web browser: the league nets with their value nets, the native tree
search (`native_backend/live_search_oct19`), the certain line and the unit rules, compiled with Emscripten and run in
a Web Worker. The worker is a client of the WotR Bot Protocol: it speaks the same messages as
`protocol/wotr_protocol_bot.py`, so a web page, or a referee through `worker/aw-stdio.mjs`, can play against it.

Everything is in this folder; nothing outside it is changed. AlphaWOTR's C++ is not edited in place: `build.sh`
copies it into `build/` and applies the patches of `patches/` to the copy.


Layout
------

    build.sh               builds the module (wasm32, optionally with SIMD), the 64-bit variant and a native build of
                           the same patched sources (the parity reference); also the test programs
    patches/               sed scripts applied to the copy of the sources, each documented at its top:
      0001                   pointer-width asserts that hold on 32-bit
      0003, 0004             a single-threaded search controller (a provider answers each request inside advance())
                             and its export
      0006                   a cached name lookup (speed)
                           A patch that changes nothing fails the build, so a release that moved the code is noticed.
    nn/                    the network's forward pass in C++ (no torch): the value net 'base' in eval mode, a direct
                           and a factored policy path; SIMD with --simd
    driver/                the module's C++ around the search: the controller's ABI, the combat replies computed in
                           the module, the root selection, the certain line, the unit rules, "strictly better", the
                           live search's set-up and seeds, the decision as a whole (decision.cpp), the C API
                           (wasm_api.cpp), the replay of recorded searches (replay.cpp)
    adapter/               the protocol adapter's Python ported to C++: the state built from the protocol's view
                           (build_state.cpp, on a model of the engine's object graph, pyobj.h), matching the
                           protocol's options to actions (matching.cpp), plans and chains (plans.cpp), the bot
                           (bot.cpp: the protocol messages in and out)
    worker/                alphawotr-worker.js (the Web Worker), alphawotr-core.mjs (shared with Node),
                           aw-stdio.mjs (a stdio protocol bot), test.html + serve.py (a browser check)
    tools/                 Python that exports the data and records the references from a checkout of AlphaWOTR
    smoke/                 a Node smoke test of the module against the native library and torch

Each ported Python function names its source file and line.


Requirements
------------

- Emscripten (tested with emsdk 7.0.0); `EMSDK` pointing at emsdk's folder.
- Node 24 (for the tests and `aw-stdio.mjs`).
- For the data and the references: a checkout of AlphaWOTR with its Python environment and the native library built
  (`native_backend/live_search_oct19/build_portable.py`).


Build
-----

    wasm/build.sh wasm32 --simd     build/alphawotr-simd.mjs + .wasm: the module the browser loads
    wasm/build.sh wasm32            the same without SIMD
    wasm/build.sh wasm64            -sMEMORY64 (64-bit, as on the hosts the selftest certifies; narrower browser support)
    wasm/build.sh native            build/native/: the same patched sources with the host's compiler (parity reference)

The numeric flags are the native build's own (`build_portable.py`): no FMA contraction, no fast-math, libm pow/log.
Single-threaded, no mimalloc.


The data
--------

The worker loads seven data files from one folder (about 24 MB with the module, 17 MB compressed):

    net_fp.awn, net_sh.awn          the seats' league nets (weights only)
    value_fp.awn, value_sh.awn      the value nets bound to them
    tables.awg                      the static native tables (observation, actions)
    adapter.json, prototypes.awg    the adapter's name tables and the objects build_state creates

Exported from a checkout of AlphaWOTR (ALPHA below):

    python -I wasm/tools/export_fixtures.py --alpha ALPHA --out DATA
    PYTHONHASHSEED=0 python -s wasm/tools/record_build_state.py --alpha ALPHA --out DATA [TRANSCRIPTS ...]

`record_build_state.py` runs without `-I` and with `PYTHONHASHSEED=0`, as the launcher runs Python (the set order of
the object graph depends on it); given protocol transcripts it also records build_state references for the parity
check. `export_fixtures.py` also writes test positions and references. The file formats are described at the top of
the two scripts.


Run
---

A protocol bot on stdin/stdout (one JSON message a line), e.g. under a referee:

    node wasm/worker/aw-stdio.mjs DATA --sims 300 [--module build/alphawotr-simd.mjs]

In a page: start `worker/alphawotr-worker.js` as a module worker, post `{init: {base: URL, sims}}` once (URL: the
folder with the module and the data files, fetched from there and cached by the browser), then `{line: MESSAGE}` for
each referee message; the worker posts `{ready}` (or `{error}`), then `{line: REPLY}` for each reply. The SIMD build
needs WebAssembly SIMD (every current browser).

A browser check: in a folder with the worker's files, `data/` (the module and the data) and a protocol transcript
`t1.jsonl`, run `python3 serve.py 8000` and open `http://localhost:8000/test.html` (`?t=FILE` for another
transcript): the page plays the transcript's decisions through the worker and posts the result to `result.json`.


Parity with the Python
----------------------

    wasm/build/native/nn_parity DATA         the network against torch on recorded leaves (also node build/nn_parity.js)
    node wasm/smoke/smoke.mjs                the engine, the encoders and the network against the native library
    wasm/build/native/replay DATA [--kernel|--live] [FILES.aws]    recorded native searches replayed in the module
    wasm/build/native/bs_parity DATA         build_state against the Python adapter's, byte for byte

Recording searches and real decisions for these: `tools/record_searches.py`, `tools/harvest_decisions.py`.

Results with v0.2.0 (2026-11-17):

- the network: 906 leaves, 19,586 action rows; values within 7e-7, priors within 5.5e-6 of torch (torch's own float32
  error is about 1e-5 relative); the same top action on every leaf;
- searches: identical with recorded replies; with the module's own network and combat replies, the same picks;
- decisions: 2,729 of 2,729 real decisions from protocol transcripts the same as the Python adapter's;
- games: 98 paired games through a referee against the Python adapter at the same settings (strength 1, 100
  simulations): 55.1% +- 9.1 for the module, 0 illegal answers and 0 timeouts in about 17,500 decisions;
- speed: about 1.5 s a decision at 300 simulations in Chrome.

`wasm/build/native/bot_parity FIXTURES/p4 FIXTURES` plays recorded protocol transcripts through the bot and compares
every answer with the Python adapter's (`--timing FILE` writes one line a decision; `--before` runs it as before the
speed fixes below; `--sims N` times the bot at its own simulations).

**The certain line's time and the menu (2026-11-18).** As the Python adapter of branch `ours` does, a decision's menu
is worked out once (the search takes the bot's projection, `awdec::Projection`), and the certain line is bounded by
time: their `SECONDS` (15 s), on a per-decision clock the answer's deadline less 40 ms (`certain_line.DEADLINE`), and
the module's own cap `certain_seconds` (1 s; `aw_create`'s config), so it cannot hold a decision up for seconds. A line
the cap cuts gives no verdict, which can only keep the search's own pick.

A known difference: Emscripten's libm can differ from another platform's in the last bit of `pow`, `log` or `exp`,
which can flip a very close choice, as between the native platforms themselves.


A new release of AlphaWOTR
--------------------------

1. Rebase this branch (`ours`, the one branch every build of ours follows; CHANGES-OURS.md) on the release.
2. `wasm/build.sh`: a patch that no longer applies fails here.
3. Export the data again and run the parity checks above.
4. Compare the Python on the decision path since the last release (`orchestrator.py`, `own_model.py`,
   `certain_line.py`, `ai_units.py`, `strength.py`, the protocol adapter) and port what changed. An unknown
   controller request kind, table schema or checkpoint contract is refused loudly rather than guessed.
5. Whole games through a referee and a short match against the Python adapter before the module is used.

License: AlphaWOTR's (MIT).
