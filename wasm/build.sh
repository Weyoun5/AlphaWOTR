#!/bin/bash
# Build AlphaWOTR's native search box (native_backend/live_search_oct19) for WebAssembly, plus our parts.
#
#   wasm/build.sh [wasm32|wasm64|native] [--simd]    (default wasm32)
#
#   wasm32  build/alphawotr.mjs + .wasm       32-bit size_t and pointers
#   wasm64  build/alphawotr64.mjs + .wasm     -sMEMORY64: 64-bit size_t and pointers, as on the hosts their selftest
#                                             certifies; browser support is narrower
#   native  build/native/                     the same patched sources with this Mac's clang (parity reference,
#                                             and the threaded controller beside the single-thread one)
#   each target also builds the test programs: nn_parity (network against torch), replay (recorded searches),
#   bs_parity (the adapter's build_state against Python's);
#   for wasm they are Node programs (build/nn_parity[64].js, build/replay[64].js) reading the host's files.
#   --simd: the network with wasm SIMD (-msimd128 for nn/awnet.cpp: its explicit SIMD path, the scalar build's bits)
#           (wasm targets; output names get -simd). The engine stays scalar: compiled with -msimd128 it replayed the
#           same searches but was no faster (2026-11-17: 72.7 s against 70.1 s for the 18 selftest searches)
#
# Their sources are copied into build/src-<target> and the patches in patches/ are applied to the copy (sed scripts
# or unified diffs, documented at their top; '# file:' names a sed script's file, '# targets:' the targets), so a new release of
# theirs is taken by re-running this script; a patch that changes nothing fails the build loudly.
# Numeric flags are the box's own (build_portable.py NUMERIC): no FMA contraction, no fast-math, libm pow/log.
# Single-threaded, no mimalloc (the allocator unit is not compiled, as on the Mac). EMSDK: emsdk's folder.
# At most 7 compilers at once (the machine is shared).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
BOX="$ROOT/native_backend/live_search_oct19"
OUT="$HERE/build"
MODE="${1:-wasm32}"
SIMD=0; [ "${2:-}" = "--simd" ] && SIMD=1
NNFLAGS=""
NUMERIC="-ffp-contract=off -fno-fast-math -fno-builtin-pow -fno-builtin-log"

case "$MODE" in
  wasm32) SUFFIX="" ; ARCH="" ;;
  wasm64) SUFFIX="64"; ARCH="-sMEMORY64=1" ;;
  native) SUFFIX="" ; ARCH="" ;;
  *) echo "unknown target $MODE" >&2; exit 2 ;;
esac
if [ $SIMD = 1 ]; then
  [ "$MODE" = native ] && { echo "--simd is for the wasm targets" >&2; exit 2; }
  NNFLAGS="-msimd128"; SUFFIX="$SUFFIX-simd"
fi
SRC="$OUT/src-$MODE"
rm -rf "$SRC" && mkdir -p "$SRC" && cp -R "$BOX/cpp" "$BOX/include" "$SRC/"
# the turn search box's C++ port of combat_dist.exact_distribution (same release), for the COMBAT_DIST replies
cp "$ROOT/turn_search/native_w2x/box/cpp/combat_kernel.inc" "$ROOT/turn_search/native_w2x/box/cpp/chance_driver.inc" "$SRC/cpp/"
for p in "$HERE"/patches/*; do
  grep -Eq "^# targets:.* $MODE( |$)" "$p" || continue
  case "$p" in
    *.sed)
      f=$(sed -n 's/^# file: //p' "$p")
      before=$(shasum "$SRC/$f" | cut -d' ' -f1)
      sed -i '' -f "$p" "$SRC/$f"
      [ "$before" != "$(shasum "$SRC/$f" | cut -d' ' -f1)" ] || { echo "patch $(basename "$p") changed nothing in $f" >&2; exit 1; } ;;
    *.diff)   # a unified diff of the box's files (paths native_backend/live_search_oct19/...): patch -p3 on the copy
      patch -s -p3 -d "$SRC" --forward --no-backup-if-mismatch < "$p" || { echo "patch $(basename "$p") does not apply" >&2; exit 1; } ;;
  esac
done

INC="-I$SRC/include -I$SRC/cpp -I$HERE/nn -I$HERE/driver -I$HERE/adapter"
OBJ="$OUT/obj-$MODE$SUFFIX"
mkdir -p "$OBJ"
if [ "$MODE" = native ]; then
  CXX="c++ -std=c++17 -O2 -fexceptions $NUMERIC $INC"
  LINK="c++"
else
  source "${EMSDK:-$HOME/Projects/BehindMedia/emsdk}/emsdk_env.sh" >/dev/null 2>&1
  CXX="em++ -std=c++17 -O2 -fwasm-exceptions $ARCH $NUMERIC $INC"
  LINK="em++ -O2 -fwasm-exceptions $ARCH"
fi
pids=()
for u in "$SRC/cpp/position.cpp" "$HERE/driver/box_unity.cpp" "$SRC/cpp/random_stream.cpp" "$HERE/driver/wasm_api.cpp" \
         "$HERE/driver/replay.cpp" "$HERE/nn/nn_parity.cpp" "$HERE/adapter/build_state.cpp" "$HERE/adapter/matching.cpp" "$HERE/adapter/bs_parity.cpp"; do
  $CXX -c "$u" -o "$OBJ/$(basename "${u%.cpp}").o" & pids+=($!)
done
$CXX $NNFLAGS -c "$HERE/nn/awnet.cpp" -o "$OBJ/awnet.o" & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
BOXOBJ="$OBJ/position.o $OBJ/box_unity.o $OBJ/random_stream.o $OBJ/wasm_api.o $OBJ/awnet.o $OBJ/build_state.o $OBJ/matching.o"

if [ "$MODE" = native ]; then
  mkdir -p "$OUT/native"
  $LINK -o "$OUT/native/nn_parity" "$OBJ/awnet.o" "$OBJ/nn_parity.o"
  $LINK -o "$OUT/native/replay" $BOXOBJ "$OBJ/replay.o"
  $LINK -o "$OUT/native/bs_parity" $BOXOBJ "$OBJ/bs_parity.o"
  ls -la "$OUT/native/nn_parity" "$OUT/native/replay" "$OUT/native/bs_parity"
  exit 0
fi
EXPORTS='["_malloc","_free","_wotr_abi_version","_wotr_snapshot_abi_version","_wotr_search_controller_abi_version",
"_wotr_snapshot_destroy","_wotr_snapshot_last_error","_wotr_snapshot_prepare_observation_tables",
"_wotr_snapshot_prepare_action_tables_v53","_wotr_snapshot_full_actions","_wotr_snapshot_encode_state_v53",
"_wotr_snapshot_encode_actions_v53","_awasm_graph_create","_awasm_snapshot_root_count","_awnet_load","_awnet_free",
"_awnet_last_error","_awnet_forward","_awnet_obs_dim","_awnet_act_dim","_awasm_combat_reply"]'
EXPORTS=$(echo "$EXPORTS" | tr -d '\n')
$LINK -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,worker,web \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sSTACK_SIZE=4MB \
  -sEXPORTED_FUNCTIONS="$EXPORTS" \
  -sEXPORTED_RUNTIME_METHODS='["HEAPU8","HEAPF32","HEAP32","HEAP64","UTF8ToString"]' \
  $BOXOBJ -o "$OUT/alphawotr$SUFFIX.mjs"
# the test programs as Node programs (they read the fixtures from the host file system)
NODEPROG="-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB -sNODERAWFS=1 -sENVIRONMENT=node"
$LINK $NODEPROG "$OBJ/awnet.o" "$OBJ/nn_parity.o" -o "$OUT/nn_parity$SUFFIX.js"
$LINK $NODEPROG $BOXOBJ "$OBJ/replay.o" -o "$OUT/replay$SUFFIX.js"
$LINK $NODEPROG $BOXOBJ "$OBJ/bs_parity.o" -o "$OUT/bs_parity$SUFFIX.js"
ls -la "$OUT/alphawotr$SUFFIX.wasm" "$OUT/nn_parity$SUFFIX.wasm" "$OUT/replay$SUFFIX.wasm"
