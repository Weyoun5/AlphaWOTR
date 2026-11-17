AlphaWOTR in WebAssembly (branch wasm-port)
===========================================

AlphaWOTR's strength-1 decision (the league nets with their value nets, the native tree search, the certain line,
the unit rules) running inside a browser Web Worker, as a client of the WotR Bot Protocol (the protocol adapter,
protocol/wotr_protocol_bot.py, ported). The plan, the phases and the decisions: docs/AlphaWOTR-Wasm-Plan.md in the
War of the Ring (behindmedia) repository. This folder holds the port; nothing outside it is changed.

Phase 1 (2026-11-17): the foundation
------------------------------------
- build.sh             builds the box (native_backend/live_search_oct19, copied and patched in build/) with
                       Emscripten: wasm32 (patches/*.sed applied) or wasm64 (-sMEMORY64, unchanged sources); and the
                       network parity test (native or wasm)
- patches/             sed scripts applied to the copy, each documented at its top
- nn/awnet.{h,cpp}     the network forward pass in C++ (no torch): WOTRValueNet 'base' in eval mode, a direct and a
                       factored ("fast") policy path
- nn/nn_parity.cpp     the C++ forward pass against torch on recorded leaves
- driver/wasm_api.cpp  the module's own C surface (graph loading, menu size, the network)
- smoke/smoke.mjs      Node smoke test: their engine and encoders and our network in wasm against the native library
                       and torch on the selftest positions
- tools/export_fixtures.py   weights (awn), tables and positions (awg), references (awr), from a checkout with the
                       native library built (formats documented at its top)
- tools/torch_f64_error.py   torch's own float32 error against float64 (the scale of the parity bound)

Run (on a Mac with the venv of wotr-alpha and emsdk in ~/Projects/BehindMedia/emsdk):

    venv/bin/python -I wasm/tools/export_fixtures.py --alpha ~/Projects/BehindMedia/wotr-alpha/alphawotr-041313c --out wasm/fixtures
    wasm/build.sh wasm32 && wasm/build.sh wasm64 && wasm/build.sh native
    wasm/build/native/nn_parity wasm/fixtures
    node wasm/build/nn_parity.js wasm/fixtures
    node wasm/smoke/smoke.mjs                              # wasm32
    node wasm/smoke/smoke.mjs wasm/build/alphawotr64.mjs   # wasm64

Results of 2026-11-17 (emsdk 7.0.0, Node 24.19):
- smoke: 6 of 6 positions PASS on wasm32 and wasm64: the full menu's row count equals the native library's, the v53
  observation and every action row and region are bit-identical, priors within 3e-7 of torch, the value net's value
  within 1e-6, the same top action.
- network parity (906 leaves, 19,586 action rows; native and wasm alike): backbone 1.3e-6, values 7e-7, priors 5.5e-6,
  logits 1.6e-5 relative (torch's own float32 logits differ from float64 by 1.05e-5 relative); the same top action
  on every leaf. wasm (no SIMD yet): 1.3 ms a leaf body, 5.4 us an action row on the fast path.

Phase 2 (2026-11-17): the search in the module
----------------------------------------------
- patches/0003, 0004     the single-thread controller: a provider answers each request inside advance()
- driver/aw_controller_abi.h   the controller's structs and exports for the module's C++
- driver/box_unity.cpp   the live box's snapshot.cpp + the turn search box's combat kernel + driver/combat_reply.inc
                         (the COMBAT_DIST replies computed in the module: awasm_combat_reply)
- driver/select.h        build_visit_info + ai.select_root_action (temperature 0) in C++
- driver/replay.cpp      replays recorded searches: recorded replies (bit-identical search), --kernel (the module's
                         combat distributions), --live (the module's network and value nets; same pick?)
- tools/record_searches.py   records whole native searches (live settings) through hooks on the controller binding
- tools/harvest_decisions.py real decisions from protocol transcripts (the Python adapter run offline)

    venv/bin/python -I wasm/tools/record_searches.py --alpha CHECKOUT --out wasm/fixtures            # selftest, 3 seeds
    venv/bin/python -I wasm/tools/harvest_decisions.py --alpha CHECKOUT --out wasm/fixtures/harvest.pkl T1.protocol.jsonl ...
    venv/bin/python -I wasm/tools/record_searches.py --alpha CHECKOUT --out wasm/fixtures --set harvest --seeds 1 --positions wasm/fixtures/harvest.pkl
    wasm/build.sh native && wasm/build.sh wasm32 && wasm/build.sh wasm32 --simd
    wasm/build/native/replay wasm/fixtures [--kernel|--live] [FILES.aws]
    node wasm/build/replay[-simd].js wasm/fixtures [--kernel|--live] [FILES.aws]

The recordings are large (about 70 MB a search: every request's observations and action rows); they stay out of git.
Results and the wasm32 decision: docs/AlphaWOTR-Wasm-Plan.md section 6b (behindmedia repository).

Next (phase 2, rest): ai._strictly_better in C++ (the replays use the recorded verdicts); the driver's own
build_config and seeds (own_model's seat seeds, rng_streams.derive_seed, the search stream) so a decision needs no
recording; then phase 3 (the state from the protocol view).
