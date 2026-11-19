// The WebAssembly module's own C surface (phase 1): graph loading in the awg layout, the menu size, and the network.
// Everything else the smoke test calls is the box's own extern "C" API (wotr_snapshot_*), exported unchanged.
#include "wotr_snapshot.h"
#include "awnet.h"

#include <cstring>
#include <string>
#include <vector>

static std::string g_error;

extern "C" {

// One graph of an awg file, starting at its u64 counts (after the name): values, edges, bytes, root, then the arrays.
// Returns a snapshot handle (wotr_snapshot_destroy), or null (wotr_snapshot_last_error).
WOTR_API void* awasm_graph_create(const uint8_t* p) {
    uint64_t nv, ne, nb, root;
    std::memcpy(&nv, p, 8); std::memcpy(&ne, p + 8, 8); std::memcpy(&nb, p + 16, 8); std::memcpy(&root, p + 24, 8);
    const uint8_t* values = p + 32;
    const uint8_t* edges = values + 40 * nv;
    const uint8_t* bytes = edges + 8 * ne;
    // the arrays of an awg file are only 4-byte aligned: copy them into aligned storage
    std::vector<WotrValue> v(nv);
    std::vector<uint64_t> e(ne);
    std::memcpy(v.data(), values, 40 * nv);
    std::memcpy(e.data(), edges, 8 * ne);
    return wotr_snapshot_create(v.data(), size_t(nv), e.data(), size_t(ne), bytes, size_t(nb), root);
}

// The number of entries of a snapshot's root container (a menu: its rows).
WOTR_API int64_t awasm_snapshot_root_count(const void* handle) {
    const size_t nv = wotr_snapshot_node_count(handle), ne = wotr_snapshot_edge_count(handle),
                 nb = wotr_snapshot_byte_count(handle);
    std::vector<WotrValue> v(nv);
    std::vector<uint64_t> e(ne);
    std::vector<uint8_t> b(nb);
    if (wotr_snapshot_export(handle, v.data(), nv, e.data(), ne, b.data(), nb)) return -1;
    return int64_t(v[wotr_snapshot_root(handle)].count);
}

WOTR_API void* awnet_load(const uint8_t* data, size_t size) {
    auto* net = new awnet::Net();
    if (!net->load(data, size, g_error)) { delete net; return nullptr; }
    return net;
}

WOTR_API void awnet_free(void* net) { delete static_cast<awnet::Net*>(net); }
WOTR_API const char* awnet_last_error() { return g_error.c_str(); }
WOTR_API int awnet_obs_dim(const void* net) { return static_cast<const awnet::Net*>(net)->obs_dim(); }
WOTR_API int awnet_act_dim(const void* net) { return static_cast<const awnet::Net*>(net)->act_dim(); }

// One leaf: the value of `value_net` (or of `net`'s own head when value_net is null) and the softmax priors of
// `net`'s scorer over the rows (fast path). Returns 0.
WOTR_API int awnet_forward(const void* net, const void* value_net, const float* obs, const float* actions,
                           const int64_t* regions, size_t rows, float* priors, float* value) {
    const auto& n = *static_cast<const awnet::Net*>(net);
    awnet::Body b;
    n.body(obs, b, rows != 0);
    if (value_net) {
        awnet::Body bv;
        const auto& vn = *static_cast<const awnet::Net*>(value_net);
        vn.body(obs, bv, false);
        *value = vn.value(bv);
    } else {
        *value = n.value(b);
    }
    if (rows) {
        std::vector<float> logits(rows);
        n.logits(b, actions, regions, rows, logits.data(), true);
        awnet::Net::softmax(logits.data(), rows, priors);
    }
    return 0;
}

}  // extern "C"
