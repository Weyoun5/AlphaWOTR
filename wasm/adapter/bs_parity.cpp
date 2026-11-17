// The C++ build_state against the Python adapter's (record_build_state.py's build_state.awb).
//
//   bs_parity FIXTURES_DIR [--verbose]
//
// Per case: the view and the memo as the Python had them; the state built here, encoded as encode_graph encodes,
// against the Python state's graph: byte for byte ("identical"), else field by field with sets compared as sets
// ("equal": Python's set iteration order is its hash table's), else the first differing field's path.
#include "adapter.h"
#include "wotr_random.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>

namespace {

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct G {
    const pyobj::Graph& g;
    std::string text(uint64_t n) const { const auto& v = g.values[n]; return std::string(reinterpret_cast<const char*>(&g.bytes[v.offset]), v.count); }
    std::vector<uint64_t> kids(uint64_t n) const { const auto& v = g.values[n]; return std::vector<uint64_t>(g.edges.begin() + v.offset, g.edges.begin() + v.offset + v.count); }
};

std::string canon(const G& a, uint64_t n, int depth = 0) {
    const auto& v = a.g.values[n];
    if (depth > 60) return "...";
    switch (v.tag) {
    case WOTR_NONE: return "None";
    case WOTR_BOOL: return v.integer ? "True" : "False";
    case WOTR_INT: return std::to_string(v.integer);
    case WOTR_FLOAT: { char b[40]; std::snprintf(b, sizeof b, "%.17g", v.real); return b; }
    case WOTR_STRING: return "'" + a.text(n) + "'";
    default: {
        std::vector<std::string> parts;
        for (auto c : a.kids(n)) parts.push_back(canon(a, c, depth + 1));
        if (v.tag == WOTR_SET || v.tag == WOTR_FROZENSET) std::sort(parts.begin(), parts.end());
        std::string o = "<" + std::to_string(v.tag) + ">(";
        for (auto& p : parts) o += p + ",";
        return o + ")";
    }
    }
}

// the first difference as a path, "" when equal (sets as sets)
std::string diff(const G& a, uint64_t x, const G& b, uint64_t y, const std::string& path, int depth = 0) {
    const auto& u = a.g.values[x]; const auto& v = b.g.values[y];
    if (depth > 200) return "";
    if (u.tag != v.tag) return path + ": tag " + std::to_string(u.tag) + " (C++) vs " + std::to_string(v.tag) + " (Python): " + canon(a, x).substr(0, 120) + " vs " + canon(b, y).substr(0, 120);
    switch (u.tag) {
    case WOTR_NONE: return "";
    case WOTR_BOOL: case WOTR_INT: return u.integer == v.integer ? "" : path + ": " + std::to_string(u.integer) + " (C++) vs " + std::to_string(v.integer) + " (Python)";
    case WOTR_FLOAT: return u.real == v.real ? "" : path + ": float differs";
    case WOTR_STRING: case WOTR_BYTES: case WOTR_BIGINT: return a.text(x) == b.text(y) ? "" : path + ": '" + a.text(x) + "' (C++) vs '" + b.text(y) + "' (Python)";
    case WOTR_SET: case WOTR_FROZENSET: {
        std::vector<std::string> p, q;
        for (auto c : a.kids(x)) p.push_back(canon(a, c));
        for (auto c : b.kids(y)) q.push_back(canon(b, c));
        std::sort(p.begin(), p.end()); std::sort(q.begin(), q.end());
        return p == q ? "" : path + ": set differs: " + std::to_string(p.size()) + " vs " + std::to_string(q.size()) + " items";
    }
    default: {
        const auto ka = a.kids(x), kb = b.kids(y);
        if (u.tag == WOTR_OBJECT && a.text(ka[0]) != b.text(kb[0])) return path + ": type " + a.text(ka[0]) + " vs " + b.text(kb[0]);
        if (u.tag == WOTR_DICT || u.tag == WOTR_OBJECT) {
            const size_t start = u.tag == WOTR_OBJECT ? 1 : 0;
            for (size_t i = start; i + 1 < std::max(ka.size(), kb.size()) + 1; i += 2) {
                if (i + 1 >= ka.size() || i + 1 >= kb.size()) return path + ": " + std::to_string((ka.size() - start) / 2) + " vs " + std::to_string((kb.size() - start) / 2) + " entries";
                const std::string key = canon(a, ka[i]);
                if (key != canon(b, kb[i])) return path + ": key " + key + " (C++) vs " + canon(b, kb[i]) + " (Python) at entry " + std::to_string(i / 2);
                const std::string r = diff(a, ka[i + 1], b, kb[i + 1], path + (u.tag == WOTR_OBJECT && key == "'__dict__'" ? "" : "[" + key + "]"), depth + 1);
                if (!r.empty()) return r;
            }
            return "";
        }
        if (ka.size() != kb.size()) return path + ": " + std::to_string(ka.size()) + " (C++) vs " + std::to_string(kb.size()) + " (Python) items";
        for (size_t i = 0; i < ka.size(); ++i) {
            const std::string r = diff(a, ka[i], b, kb[i], path + "[" + std::to_string(i) + "]", depth + 1);
            if (!r.empty()) return r;
        }
        return "";
    }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bs_parity FIXTURES_DIR [--verbose]\n"); return 2; }
    const std::string dir = argv[1];
    const bool verbose = argc > 2 && std::string(argv[2]) == "--verbose";
    auto tj = slurp(dir + "/adapter.json");
    awad::Tables tables;
    tables.load(awjson::parse(std::string(tj.begin(), tj.end())));
    auto pg = slurp(dir + "/prototypes.awg");
    uint32_t nl; std::memcpy(&nl, &pg[8], 4);
    const pyobj::Graph pgraph = pyobj::read_graph(&pg[12 + nl]);
    pyobj::Model model;
    pyobj::Obj* protos = model.decode(pgraph);
    awad::Adapter adapter(model, tables, protos);
    auto data = slurp(dir + "/build_state.awb");
    uint32_t count; std::memcpy(&count, &data[4], 4);
    size_t off = 8, identical = 0, equal = 0;
    double secs = 0;
    for (uint32_t c = 0; c < count; ++c) {
        uint32_t len; std::memcpy(&len, &data[off], 4); off += 4;
        const std::string text(reinterpret_cast<const char*>(&data[off]), len); off += len;
        awad::Memo memo;
        std::memcpy(&memo.side, &data[off], 4); off += 4;
        std::vector<uint32_t> words(624); std::memcpy(words.data(), &data[off], 4 * 624); off += 4 * 624;
        uint32_t index; int32_t hg; uint64_t bits;
        std::memcpy(&index, &data[off], 4); std::memcpy(&hg, &data[off + 4], 4); std::memcpy(&bits, &data[off + 8], 8); off += 16;
        memo.messenger_used = data[off]; memo.fellowship_acted = data[off + 1]; memo.in_mordor = data[off + 2]; off += 3;
        uint32_t gl; std::memcpy(&gl, &data[off], 4); off += 4 + gl;
        const pyobj::Graph want = pyobj::read_graph(&data[off]);
        uint64_t nv, ne, nb; std::memcpy(&nv, &data[off], 8); std::memcpy(&ne, &data[off + 8], 8); std::memcpy(&nb, &data[off + 16], 8);
        off += 32 + 40 * nv + 8 * ne + nb;
        memo.rng = wotr_random_create(words.data(), 624, index, hg, bits);
        std::string why;
        bool same = false, eq = false;
        try {
            const awjson::Value view = awjson::parse(text);
            const auto t0 = std::chrono::steady_clock::now();
            pyobj::Obj* st = adapter.build_state(view, memo);
            const pyobj::Graph got = model.encode(st);
            secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            same = got.values.size() == want.values.size() && got.edges == want.edges && got.bytes == want.bytes &&
                   std::memcmp(got.values.data(), want.values.data(), 40 * got.values.size()) == 0 && got.root == want.root;
            if (!same) {
                why = diff(G{got}, got.root, G{want}, want.root, "state"); eq = why.empty();
                if (eq && verbose) {   // the first node that differs, for the set order
                    for (size_t n = 0; n < std::min(got.values.size(), want.values.size()); ++n)
                        if (std::memcmp(&got.values[n], &want.values[n], 40) != 0) {
                            why = "first differing node " + std::to_string(n) + ": " + canon(G{got}, n).substr(0, 200) + " vs " + canon(G{want}, n).substr(0, 200);
                            break;
                        }
                    if (why.empty()) {
                        why = "values " + std::to_string(got.values.size()) + "/" + std::to_string(want.values.size()) + ", edges " +
                              std::to_string(got.edges.size()) + "/" + std::to_string(want.edges.size()) + ", bytes " + std::to_string(got.bytes.size()) + "/" + std::to_string(want.bytes.size());
                        for (size_t n = 0; n < std::min(got.edges.size(), want.edges.size()); ++n)
                            if (got.edges[n] != want.edges[n]) { why += ", first edge differing " + std::to_string(n) + ": " + canon(G{got}, got.edges[n]).substr(0, 150) + " vs " + canon(G{want}, want.edges[n]).substr(0, 150); break; }
                    }
                }
            }
        } catch (const std::exception& e) { why = std::string("error: ") + e.what(); }
        wotr_random_destroy(memo.rng);
        identical += same; equal += same || eq;
        if (!same && (verbose || !eq)) std::printf("case %u: %s%s\n", c, eq ? "equal up to set order (not byte-identical) " : "", why.c_str());
    }
    std::printf("%s: %u cases, %zu byte-identical, %zu equal field by field (sets as sets); %.1f ms a state\n",
                equal == count ? "PASS" : "FAIL", count, identical, equal, count ? 1e3 * secs / count : 0.0);
    return equal == count ? 0 : 1;
}
