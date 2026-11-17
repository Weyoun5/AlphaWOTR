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

Next (phase 2): the search in the module. The controller runs its search on a std::thread and waits on a condition
variable for each network evaluation (cpp/search_controller.inc, await_reply); single-threaded wasm needs a patch that
calls an inference callback in place of the wait. Then the orchestrator's root (build_config, the own-value context,
build_visit_info, select_root_action, ai._strictly_better) and the combat-distribution replies (the turn search's box
has them in C++: turn_search/native_w2x/box/cpp/combat_kernel.inc, kernel_expand.inc), checked by replaying Python's
recorded replies (bit-identical root visits) and then with our network (same picks on the selftest and a larger set).
