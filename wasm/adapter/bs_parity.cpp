// The C++ build_state against the Python adapter's (record_build_state.py's build_state.awb).
//
//   bs_parity FIXTURES_DIR [--verbose]
//
// Per case: the view and the memo as the Python had them; the state built here, encoded as encode_graph encodes,
// against the Python state's graph: byte for byte ("identical"), else field by field with sets compared as sets
// ("equal": Python's set iteration order is its hash table's), else the first differing field's path.
#include "adapter.h"
#include "decision.h"
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

// a graph value as record_build_state.py's canon() writes a Python value
std::string pcanon(const G& a, uint64_t n) {
    const auto& v = a.g.values[n];
    switch (v.tag) {
    case WOTR_NONE: return "N";
    case WOTR_BOOL: return v.integer ? "T" : "F";
    case WOTR_INT: return "i" + std::to_string(v.integer);
    case WOTR_FLOAT: { char b[40]; std::snprintf(b, sizeof b, "f%.17g", v.real); return b; }
    case WOTR_STRING: return "s" + std::to_string(v.count) + ":" + a.text(n);
    case WOTR_ENUM: return "e" + a.text(a.kids(n)[1]);
    case WOTR_LIST: case WOTR_TUPLE: { std::string o = "["; bool first = true; for (auto c : a.kids(n)) { if (!first) o += ","; first = false; o += pcanon(a, c); } return o + "]"; }
    case WOTR_SET: case WOTR_FROZENSET: case WOTR_DICT: {
        std::vector<std::string> parts; const auto ks = a.kids(n);
        if (v.tag == WOTR_DICT) for (size_t i = 0; i + 1 < ks.size(); i += 2) parts.push_back(pcanon(a, ks[i]) + ":" + pcanon(a, ks[i + 1]));
        else for (auto c : ks) parts.push_back(pcanon(a, c));
        std::sort(parts.begin(), parts.end());
        std::string o = v.tag == WOTR_DICT ? "{" : "<"; for (size_t i = 0; i < parts.size(); ++i) { if (i) o += ","; o += parts[i]; }
        return o + (v.tag == WOTR_DICT ? "}" : ">");
    }
    case WOTR_OBJECT: {
        const auto ks = a.kids(n);
        for (size_t i = 1; i + 1 < ks.size(); i += 2) if (a.text(ks[i]) == "__dict__") {
            const auto d = a.kids(ks[i + 1]);
            for (size_t j = 0; j + 1 < d.size(); j += 2) if (a.g.values[d[j]].tag == WOTR_STRING && a.text(d[j]) == "name") return "o" + a.text(d[j + 1]);
        }
        const std::string t = a.text(ks[0]); return "o" + t.substr(t.rfind('.') + 1);
    }
    default: return "?";
    }
}
// a graph value as the adapter's Python sees it, in JSON terms (an Enum as "\x01" "enum:MEMBER", an object as its name)
awjson::Value to_value(const G& a, uint64_t n) {
    const auto& v = a.g.values[n];
    awjson::Value out;
    switch (v.tag) {
    case WOTR_NONE: break;
    case WOTR_BOOL: out.kind = awjson::Value::Bool; out.b = v.integer != 0; break;
    case WOTR_INT: out.kind = awjson::Value::Int; out.i = v.integer; break;
    case WOTR_FLOAT: out.kind = awjson::Value::Real; out.d = v.real; break;
    case WOTR_STRING: out.kind = awjson::Value::Str; out.s = a.text(n); break;
    case WOTR_ENUM: out.kind = awjson::Value::Str; out.s = "\x01" "enum:" + a.text(a.kids(n)[1]); break;
    case WOTR_LIST: case WOTR_TUPLE: case WOTR_SET: case WOTR_FROZENSET:
        out.kind = awjson::Value::Arr; for (auto c : a.kids(n)) out.arr.push_back(to_value(a, c)); break;
    case WOTR_DICT: {
        out.kind = awjson::Value::Obj; const auto ks = a.kids(n);
        for (size_t i = 0; i + 1 < ks.size(); i += 2) {
            const auto& kv = a.g.values[ks[i]];
            const std::string key = kv.tag == WOTR_STRING ? a.text(ks[i]) : (kv.tag == WOTR_INT ? std::to_string(kv.integer) : pcanon(a, ks[i]));
            out.obj.emplace_back(key, to_value(a, ks[i + 1]));
        }
        break;
    }
    default: out.kind = awjson::Value::Str; out.s = pcanon(a, n).substr(1); break;
    }
    return out;
}
pyobj::Graph export_graph(const void* h) {
    pyobj::Graph g;
    const size_t nv = wotr_snapshot_node_count(h), ne = wotr_snapshot_edge_count(h), nb = wotr_snapshot_byte_count(h);
    g.values.resize(nv); g.edges.resize(ne); g.bytes.resize(nb); g.root = wotr_snapshot_root(h);
    wotr_snapshot_export(h, g.values.data(), nv, g.edges.data(), ne, g.bytes.data(), nb);
    return g;
}
std::vector<std::string> menu_rows(const void* menu) {
    const pyobj::Graph g = export_graph(menu);
    G a{g};
    std::vector<std::string> rows;
    for (auto row : a.kids(g.root)) {
        const auto parts = a.kids(row);
        rows.push_back(std::to_string(g.values[parts[0]].integer) + "|" + a.text(parts[1]) + "|" + pcanon(a, parts[2]));
    }
    return rows;
}

}  // namespace

extern "C" void* awasm_graph_create(const uint8_t* p);
extern "C" int64_t awasm_snapshot_root_count(const void* handle);

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
    const bool v5 = std::memcmp(data.data(), "AWB5", 4) == 0;   // AWB5: and the stages of the adapter's decision
    const bool v4 = v5 || std::memcmp(data.data(), "AWB4", 4) == 0;   // AWB4: and the option matching, the answer, the search
    const bool v3 = v4 || std::memcmp(data.data(), "AWB3", 4) == 0;   // AWB3: and the engine's menus
    const bool v2 = v3 || std::memcmp(data.data(), "AWB2", 4) == 0;   // AWB2: each case also carries the set-up (phase 3b)
    // the observation tables for the box's menus (tables.awg beside the fixtures)
    void* obs_tables = nullptr;
    if (v3) {
        auto tb = slurp(dir + "/tables.awg");
        uint32_t tc; std::memcpy(&tc, &tb[4], 4);
        size_t q = 8;
        for (uint32_t i = 0; i < tc; ++i) {
            uint32_t n; std::memcpy(&n, &tb[q], 4); const std::string name(reinterpret_cast<const char*>(&tb[q + 4]), n);
            const uint8_t* counts = &tb[q + 4 + n];
            uint64_t nv, ne, nb; std::memcpy(&nv, counts, 8); std::memcpy(&ne, counts + 8, 8); std::memcpy(&nb, counts + 16, 8);
            if (name == "observation") { obs_tables = awasm_graph_create(counts); wotr_snapshot_prepare_observation_tables(obs_tables); }
            q += 4 + n + 32 + 40 * nv + 8 * ne + nb;
        }
    }
    adapter.obs_tables = obs_tables;
    size_t menu_checked = 0, menu_same = 0, proj_same = 0, menu_rows_total = 0;
    uint32_t count; std::memcpy(&count, &data[4], 4);
    size_t searched = 0, setup_ported = 0, setup_same = 0;
    size_t off = 8, identical = 0, equal = 0;
    struct MatchRef { int64_t o; bool plan, chain; };
    struct SetupRef { std::string dec, label, flag; pyobj::Graph graph; std::vector<std::string> legal, projected; std::vector<MatchRef> match; int64_t answer = -1; std::string how;
                      uint64_t seat = 0; uint32_t index = 0; uint64_t seed = 0; int64_t sims = -1; awjson::Value stages; };
    // the decision on the module (driver/decision.h), against the adapter's answer and its stages
    awdec::Box box;
    bool decide = false;
    if (v5) {
        std::string err;
        const std::string nets = std::ifstream(dir + "/net_fp.awn").good() ? dir : dir + "/..";
        decide = box.load(dir + "/tables.awg", nets, err);
        if (!decide) std::printf("decisions not checked: %s\n", err.c_str());
    }
    std::map<std::string, size_t> stage_same, stage_total;
    size_t dec_total = 0, dec_same = 0, dec_skipped = 0; double dec_secs = 0;
    size_t match_rows = 0, match_same = 0, match_unported = 0, match_cases = 0, match_cases_same = 0;
    std::vector<SetupRef> setups;
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
        if (v2) {
            uint32_t has; std::memcpy(&has, &data[off], 4); off += 4;
            if (has) {
                ++searched;
                std::string part[3];
                for (int q = 0; q < 3; ++q) { uint32_t n; std::memcpy(&n, &data[off], 4); part[q].assign(reinterpret_cast<const char*>(&data[off + 4]), n); off += 4 + n; }
                uint32_t sl; std::memcpy(&sl, &data[off], 4); off += 4 + sl;
                const pyobj::Graph want_setup = pyobj::read_graph(&data[off]);
                uint64_t a, b2, c2; std::memcpy(&a, &data[off], 8); std::memcpy(&b2, &data[off + 8], 8); std::memcpy(&c2, &data[off + 16], 8);
                off += 32 + 40 * a + 8 * b2 + c2;
                std::vector<std::string> menus[2];
                if (v3) for (int q = 0; q < 2; ++q) {
                    uint32_t rows; std::memcpy(&rows, &data[off], 4); off += 4;
                    for (uint32_t r = 0; r < rows; ++r) { uint32_t n; std::memcpy(&n, &data[off], 4); menus[q].emplace_back(reinterpret_cast<const char*>(&data[off + 4]), n); off += 4 + n; }
                }
                std::vector<MatchRef> match; int64_t answer = -1; std::string how;
                uint64_t sr_seat = 0, sr_seed = 0; uint32_t sr_index = 0; int64_t sr_sims = -1; awjson::Value stages;
                if (v4) {   // the option matching of each legal row, the answer, the search
                    uint32_t rows; std::memcpy(&rows, &data[off], 4); off += 4;
                    for (uint32_t r = 0; r < rows; ++r) { MatchRef mr; std::memcpy(&mr.o, &data[off], 8); mr.plan = data[off + 8]; mr.chain = data[off + 9]; off += 10; match.push_back(mr); }
                    std::memcpy(&answer, &data[off], 8); off += 8; uint32_t hl; std::memcpy(&hl, &data[off], 4); how.assign(reinterpret_cast<const char*>(&data[off + 4]), hl); off += 4 + hl;
                    std::memcpy(&sr_seat, &data[off], 8); std::memcpy(&sr_index, &data[off + 8], 4); std::memcpy(&sr_seed, &data[off + 12], 8); std::memcpy(&sr_sims, &data[off + 20], 8);
                    off += 8 + 4 + 8 + 8;
                    if (v5) { uint32_t jl; std::memcpy(&jl, &data[off], 4); stages = awjson::parse(std::string(reinterpret_cast<const char*>(&data[off + 4]), jl)); off += 4 + jl; }
                }
                setups.push_back({part[0], part[1], part[2], want_setup, menus[0], menus[1], match, answer, how, sr_seat, sr_index, sr_seed, sr_sims, stages});
            } else setups.push_back({});
        }
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
            if (v2 && !setups.empty() && !setups.back().dec.empty()) {   // the set-up, where its kind is ported
                const SetupRef& ref = setups.back();
                const awjson::Value dec = awjson::parse(ref.dec);
                const auto su = adapter.setup(dec.get("kind") ? dec.get("kind")->s : "", st, dec, view, memo);
                if (su.ported) {
                    ++setup_ported;
                    const pyobj::Graph gs = model.encode(st);
                    const bool sg = gs.edges == ref.graph.edges && gs.bytes == ref.graph.bytes && gs.values.size() == ref.graph.values.size() &&
                                    std::memcmp(gs.values.data(), ref.graph.values.data(), 40 * gs.values.size()) == 0;
                    const bool sl = su.label == ref.label && su.flag == ref.flag;
                    setup_same += sg && sl;
                    if (!(sg && sl)) std::printf("case %u: set-up of %s differs: %s%s\n", c, dec.get("kind")->s.c_str(),
                                                 sl ? "" : "label/flag; ", sg ? "" : diff(G{gs}, gs.root, G{ref.graph}, ref.graph.root, "state").c_str());
                    if (v3 && sg && sl && !su.label.empty()) {   // the box's menu on the state as set up, under the phase flag
                        if (!su.flag.empty()) model.set_attr(st, su.flag, model.boolean(true));
                        const pyobj::Graph gf = model.encode(st);
                        void* sn = wotr_snapshot_create(gf.values.data(), gf.values.size(), gf.edges.data(), gf.edges.size(), gf.bytes.data(), gf.bytes.size(), gf.root);
                        void* full = sn ? wotr_snapshot_full_actions(sn, obs_tables, memo.side) : nullptr;
                        ++menu_checked;
                        if (!full) std::printf("case %u (%s): the box's menu failed: %s\n", c, dec.get("kind")->s.c_str(), wotr_snapshot_last_error());
                        else {
                            const auto rows = menu_rows(full);
                            menu_rows_total += rows.size();
                            void* meta = nullptr;
                            void* sel = wotr_snapshot_project_policy(sn, full, memo.side, &meta);
                            std::vector<std::string> prow = sel ? menu_rows(sel) : std::vector<std::string>{};
                            if (prow.empty()) prow = rows;
                            const bool ms = rows == ref.legal, ps = prow == ref.projected;
                            menu_same += ms; proj_same += ps;
                            if (!ms || !ps) {
                                std::printf("case %u (%s, flag %s): menu %s (%zu rows, Python %zu), projected %s (%zu, Python %zu)\n", c,
                                            dec.get("kind")->s.c_str(), su.flag.c_str(), ms ? "same" : "DIFFERENT", rows.size(), ref.legal.size(),
                                            ps ? "same" : "DIFFERENT", prow.size(), ref.projected.size());
                                if (verbose && !ms) {
                                    for (size_t i = 0; i < std::max(rows.size(), ref.legal.size()) && i < 400; ++i) {
                                        const std::string x = i < rows.size() ? rows[i] : "-", y = i < ref.legal.size() ? ref.legal[i] : "-";
                                        if (x != y) { std::printf("   row %zu: box %s\n          python %s\n", i, x.substr(0, 300).c_str(), y.substr(0, 300).c_str()); break; }
                                    }
                                }
                            }
                            if (v4 && ms) {   // the option matching, row by row, against the adapter's map_action
                                const pyobj::Graph mg = export_graph(full);
                                G ga{mg};
                                const auto rowsn = ga.kids(mg.root);
                                bool all_same = true;
                                for (size_t ri = 0; ri < rowsn.size() && ri < ref.match.size(); ++ri) {
                                    const auto parts = ga.kids(rowsn[ri]);
                                    awad::Adapter::Row row;
                                    row.die = mg.values[parts[0]].integer; row.type = ga.text(parts[1]); row.params = to_value(ga, parts[2]);
                                    row.menu = full; row.index = int64_t(ri);
                                    const auto mt = adapter.map_action(dec.get("kind")->s, st, dec, row, memo);
                                    ++match_rows;
                                    if (mt.unported) { ++match_unported; all_same = false; continue; }
                                    const MatchRef& want = ref.match[ri];
                                    const int64_t got_o = (mt.ok && !mt.o_none) ? mt.o : -1;
                                    const bool same = got_o == want.o && (mt.ok ? mt.has_plan : false) == want.plan && (mt.ok ? mt.chain : false) == want.chain;
                                    match_same += same; all_same &= same;
                                    if (!same && verbose)
                                        std::printf("case %u (%s) row %zu %s: C++ o=%lld plan=%d chain=%d, Python o=%lld plan=%d chain=%d\n", c, dec.get("kind")->s.c_str(), ri,
                                                    rows[ri].substr(0, 160).c_str(), (long long)got_o, int(mt.has_plan), int(mt.chain), (long long)want.o, int(want.plan), int(want.chain));
                                }
                                ++match_cases; match_cases_same += all_same;
                                if (decide && all_same) {   // LeagueSearcher.choose and _search_and_map's tail on the module
                                    std::vector<awad::Adapter::Matched> mts;
                                    for (size_t ri = 0; ri < rowsn.size(); ++ri) {
                                        const auto parts = ga.kids(rowsn[ri]);
                                        awad::Adapter::Row row; row.die = mg.values[parts[0]].integer; row.type = ga.text(parts[1]);
                                        row.params = to_value(ga, parts[2]); row.menu = full; row.index = int64_t(ri);
                                        mts.push_back(adapter.map_action(dec.get("kind")->s, st, dec, row, memo));
                                    }
                                    std::set<std::string> banned;
                                    std::vector<size_t> mapped;
                                    for (size_t ri = 0; ri < mts.size(); ++ri) { if (mts[ri].ok) mapped.push_back(ri); else banned.insert(rows[ri]); }
                                    std::string how_cpp; int64_t answer = -1; std::map<std::string, std::string> got;
                                    const auto t0 = std::chrono::steady_clock::now();
                                    if (mapped.empty()) { how_cpp = "default"; }
                                    else {
                                        bool single = true;
                                        for (size_t q : mapped) single &= mts[q].o == mts[mapped[0]].o && mts[q].o_none == mts[mapped[0]].o_none && !mts[q].has_plan && !mts[q].chain;
                                        if (single) { how_cpp = "single"; answer = mts[mapped[0]].o_none ? -1 : mts[mapped[0]].o; }
                                        else {
                                            std::string pick;
                                            if (mapped.size() == 1) pick = rows[mapped[0]];
                                            else {
                                                // allowed: the projected menu's rows not banned (else the legal menu's)
                                                std::vector<int64_t> allowed_idx;   // indices into `rows`
                                                for (const auto& pr : prow) if (!banned.count(pr)) { auto it = std::find(rows.begin(), rows.end(), pr); if (it != rows.end()) allowed_idx.push_back(int64_t(it - rows.begin())); }
                                                if (allowed_idx.empty()) for (size_t ri = 0; ri < rows.size(); ++ri) if (!banned.count(rows[ri])) allowed_idx.push_back(int64_t(ri));
                                                if (allowed_idx.size() == 1) pick = rows[size_t(allowed_idx[0])];
                                                else if (!allowed_idx.empty()) {
                                                    const auto pk = awdec::search(box, sn, memo.side, ref.sims > 0 ? ref.sims : 8, ref.seat, ref.index, ref.seed, 0);
                                                    std::string act;
                                                    if (!pk.error.empty()) act = "ERROR " + pk.error;
                                                    else { const auto mr = menu_rows(pk.menu); act = pk.index >= 0 && size_t(pk.index) < mr.size() ? mr[size_t(pk.index)] : "?"; wotr_snapshot_destroy(pk.menu); }
                                                    got["search"] = act;
                                                    std::string hit;
                                                    for (auto ai : allowed_idx) if (rows[size_t(ai)] == act) { hit = act; break; }
                                                    if (hit.empty()) {   // the search's pick has no referee counterpart: the net's own choice among the allowed
                                                        void* sub = awdec::menu_rows(full, allowed_idx);
                                                        const int64_t ri = sub ? awdec::raw(box, sn, memo.side, sub) : -1;
                                                        if (sub) wotr_snapshot_destroy(sub);
                                                        if (ri >= 0) hit = rows[size_t(allowed_idx[size_t(ri)])];
                                                        got["raw"] = hit;
                                                    }
                                                    pick = hit;
                                                    // THE CERTAIN LINE over the projected menu (the adapter's `menu`), then the ban check
                                                    void* pmenu = (sel && awasm_snapshot_root_count(sel) > 0) ? sel : full;
                                                    if (!hit.empty()) {
                                                        int64_t own_i = -1;
                                                        for (size_t q = 0; q < prow.size(); ++q) if (prow[q] == hit) { own_i = int64_t(q); break; }
                                                        if (own_i >= 0) {
                                                            const int64_t outright = awdec::certain_line(box, sn, memo.side, pmenu, own_i);
                                                            if (outright >= 0) { got["certain_outright"] = prow[size_t(outright)]; if (!banned.count(prow[size_t(outright)])) pick = prow[size_t(outright)]; }
                                                            else if (outright == -2) got["certain_error"] = wotr_snapshot_last_error();
                                                        }
                                                    }
                                                    // ai_units: not ported yet
                                                }
                                            }
                                            got["pick"] = pick;
                                            how_cpp = "search";
                                            size_t at_i = mapped[0];
                                            for (size_t q : mapped) if (rows[q] == pick) { at_i = q; break; }
                                            answer = mts[at_i].o_none ? -1 : mts[at_i].o;
                                        }
                                    }
                                    dec_secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                                    ++dec_total;
                                    const bool same = answer == ref.answer;
                                    dec_same += same;
                                    for (const char* k : {"search", "raw", "pick", "certain_outright"}) {
                                        if (!got.count(k)) continue;
                                        const awjson::Value* w = ref.stages.get(k);
                                        ++stage_total[k];
                                        stage_same[k] += w && w->kind == awjson::Value::Str && w->s == got[k];
                                    }
                                    if (!same || verbose) {
                                        const awjson::Value* co = ref.stages.get("certain_outright");
                                        const awjson::Value* un = ref.stages.get("units");
                                        const awjson::Value* pp = ref.stages.get("pick");
                                        std::printf("case %u (%s): answer C++ %lld, Python %lld%s%s%s%s\n", c, dec.get("kind")->s.c_str(), (long long)answer, (long long)ref.answer,
                                                    (co && co->kind == awjson::Value::Str) ? " [Python: certain line outright]" : "",
                                                    (un && pp && un->kind == awjson::Value::Str && pp->kind == awjson::Value::Str && un->s == pp->s && got.count("pick") && got["pick"] != pp->s) ? " [Python: unit rule]" : "",
                                                    got.count("search") && ref.stages.get("search") && ref.stages.get("search")->s != got["search"] ? " [search differs]" : "",
                                                    got.count("raw") ? " [raw]" : "");
                                        if (!same) {
                                            const awjson::Value* cl = ref.stages.get("certain_last");
                                            std::string last;
                                            if (cl) for (const auto& kv : cl->obj) last += kv.first + "=" + (kv.second.kind == awjson::Value::Str ? kv.second.s : std::to_string(kv.second.kind == awjson::Value::Int ? kv.second.i : -9)) + " ";
                                            std::printf("   C++ certain %s%s\n   Python certain %s; %s\n   hit/search %s\n", got.count("certain_outright") ? got["certain_outright"].c_str() : "-",
                                                        got.count("certain_error") ? (" error " + got["certain_error"]).c_str() : "",
                                                        co && co->kind == awjson::Value::Str ? co->s.c_str() : "-", last.c_str(), got.count("search") ? got["search"].c_str() : "-");
                                        }
                                    }
                                }
                            }
                            if (sel) wotr_snapshot_destroy(sel);
                            if (meta) wotr_snapshot_destroy(meta);
                            wotr_snapshot_destroy(full);
                        }
                        if (sn) wotr_snapshot_destroy(sn);
                    }
                }
            }
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
    std::printf("%s: %u cases, %zu byte-identical, %zu equal field by field (sets as sets); %.1f ms a state%s\n",
                equal == count ? "PASS" : "FAIL", count, identical, equal, count ? 1e3 * secs / count : 0.0,
                v2 ? (" (" + std::to_string(searched) + " with the set-up recorded; " + std::to_string(setup_ported) + " of a ported kind, " +
                      std::to_string(setup_same) + " set up byte-identically" +
                      (v3 ? "; menus: " + std::to_string(menu_checked) + " checked, " + std::to_string(menu_same) + " row for row as Python's get_legal_actions, " +
                            std::to_string(proj_same) + " projected as project_ai_actions; " + std::to_string(menu_rows_total) + " rows" : std::string()) +
                      (v5 ? "; decisions: " + std::to_string(dec_same) + " of " + std::to_string(dec_total) + " the adapter's answer (search pick " +
                            std::to_string(stage_same["search"]) + "/" + std::to_string(stage_total["search"]) + ", final pick " + std::to_string(stage_same["pick"]) + "/" +
                            std::to_string(stage_total["pick"]) + ", " + std::to_string(dec_total ? dec_secs / dec_total * 1e3 : 0.0) + " ms a decision)" : std::string()) +
                      (v4 ? "; matching: " + std::to_string(match_same) + " of " + std::to_string(match_rows) + " rows as map_action (" + std::to_string(match_unported) +
                            " of a kind not ported), " + std::to_string(match_cases_same) + " of " + std::to_string(match_cases) + " decisions entirely" : std::string()) + ")").c_str() : "");
    return equal == count ? 0 : 1;
}
