#!/bin/bash
# Build AlphaWOTR's native search box (native_backend/live_search_oct19) for WebAssembly, plus our parts.
#
#   wasm/build.sh [wasm32|wasm64|native]     (default wasm32)
#
#   wasm32  build/alphawotr.mjs + .wasm        their sources with the patches in patches/ (32-bit size_t and pointers)
#   wasm64  build/alphawotr64.mjs + .wasm      their sources UNCHANGED, -sMEMORY64 (64-bit size_t and pointers, as on
#                                              the 64-bit hosts their selftest certifies); browser support is narrower
#   native  build/native/nn_parity             the network parity test with this Mac's clang
#   each wasm target also builds build/nn_parity[64].js, the network parity test as a Node program.
#
# Their sources are copied into build/src-<target> and patches are applied to the copy (sed scripts, documented at
# their top, the file named in a '# file:' line), so a new release of theirs is taken by re-running this script; a
# patch that changes nothing fails the build loudly.
# Numeric flags are the box's own (build_portable.py NUMERIC): no FMA contraction, no fast-math, libm pow/log.
# Single-threaded, no mimalloc (the allocator unit is not compiled, as on the Mac). EMSDK: emsdk's folder.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
BOX="$ROOT/native_backend/live_search_oct19"
OUT="$HERE/build"
MODE="${1:-wasm32}"
NUMERIC="-ffp-contract=off -fno-fast-math -fno-builtin-pow -fno-builtin-log"

if [ "$MODE" = native ]; then
  mkdir -p "$OUT/native"
  c++ -std=c++17 -O2 $NUMERIC -I"$HERE/nn" -o "$OUT/native/nn_parity" "$HERE/nn/awnet.cpp" "$HERE/nn/nn_parity.cpp"
  echo "built $OUT/native/nn_parity"
  exit 0
fi

case "$MODE" in
  wasm32) SUFFIX=""; ARCH=""; PATCHES=("$HERE"/patches/*.sed) ;;
  wasm64) SUFFIX="64"; ARCH="-sMEMORY64=1"; PATCHES=() ;;
  *) echo "unknown target $MODE" >&2; exit 2 ;;
esac
SRC="$OUT/src-$MODE"
rm -rf "$SRC" && mkdir -p "$SRC" && cp -R "$BOX/cpp" "$BOX/include" "$SRC/"
for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
  f=$(sed -n 's/^# file: //p' "$p")
  before=$(shasum "$SRC/$f" | cut -d' ' -f1)
  sed -i '' -f "$p" "$SRC/$f"
  [ "$before" != "$(shasum "$SRC/$f" | cut -d' ' -f1)" ] || { echo "patch $(basename "$p") changed nothing in $f" >&2; exit 1; }
done

source "${EMSDK:-$HOME/Projects/BehindMedia/emsdk}/emsdk_env.sh" >/dev/null 2>&1
INC="-I$SRC/include -I$HERE/nn -I$HERE/driver"
OBJ="$OUT/obj-$MODE"
mkdir -p "$OBJ"
EMXX="em++ -std=c++17 -O2 -fwasm-exceptions $ARCH $NUMERIC $INC"
pids=()
for u in "$SRC/cpp/position.cpp" "$SRC/cpp/snapshot.cpp" "$SRC/cpp/random_stream.cpp" "$HERE/nn/awnet.cpp" \
         "$HERE/driver/wasm_api.cpp"; do
  $EMXX -c "$u" -o "$OBJ/$(basename "${u%.cpp}").o" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
EXPORTS='["_malloc","_free","_wotr_abi_version","_wotr_snapshot_abi_version","_wotr_search_controller_abi_version",
"_wotr_snapshot_destroy","_wotr_snapshot_last_error","_wotr_snapshot_prepare_observation_tables",
"_wotr_snapshot_prepare_action_tables_v53","_wotr_snapshot_full_actions","_wotr_snapshot_encode_state_v53",
"_wotr_snapshot_encode_actions_v53","_awasm_graph_create","_awasm_snapshot_root_count","_awnet_load","_awnet_free",
"_awnet_last_error","_awnet_forward","_awnet_obs_dim","_awnet_act_dim"]'
EXPORTS=$(echo "$EXPORTS" | tr -d '\n')
em++ -O2 -fwasm-exceptions $ARCH -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,worker,web \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sSTACK_SIZE=4MB \
  -sEXPORTED_FUNCTIONS="$EXPORTS" \
  -sEXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32","HEAP32","HEAP64","UTF8ToString"]' \
  "$OBJ"/*.o -o "$OUT/alphawotr$SUFFIX.mjs"
# the network parity test as a Node program (reads the fixtures from the host file system)
em++ -O2 -fwasm-exceptions $ARCH -sALLOW_MEMORY_GROWTH=1 -sNODERAWFS=1 -sENVIRONMENT=node "$OBJ/awnet.o" \
  "$HERE/nn/nn_parity.cpp" $INC $NUMERIC -std=c++17 -o "$OUT/nn_parity$SUFFIX.js"
ls -la "$OUT/alphawotr$SUFFIX.mjs" "$OUT/alphawotr$SUFFIX.wasm" "$OUT/nn_parity$SUFFIX.js" "$OUT/nn_parity$SUFFIX.wasm"
