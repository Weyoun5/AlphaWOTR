# file: cpp/search_controller_exports.inc
# targets: wasm32
# wasm32 has 4-byte pointers: the controller's extern "C" request/reply/root structs (which carry pointers) are
# smaller than on the 64-bit Python ctypes ABI. The asserts guard that ctypes layout, which wasm never uses (the
# driver is compiled with the same headers). Keep them on 64-bit hosts only.
s|^static_assert(sizeof(WotrSearchRequest)==88,|#if UINTPTR_MAX > 0xFFFFFFFFu\
static_assert(sizeof(WotrSearchRequest)==88,|
s|^\(static_assert(sizeof(WotrSearchRoot)==128,.*\)$|\1\
#endif|
