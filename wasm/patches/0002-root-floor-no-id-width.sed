# file: cpp/search_controller.inc
# targets: wasm32
# wasm32 only: the root free-force floor keeps "no child yet" as size_t fp9=NO_ID, and NO_ID is the uint64 maximum.
# With a 32-bit size_t it becomes 0xFFFFFFFF and "fp9==NO_ID" is never true (clang: -Wconstant-conversion). Hold it
# in a uint64_t, as every other NO_ID is held. No change on 64-bit hosts.
s|size_t fp9=NO_ID;|uint64_t fp9=NO_ID;|
