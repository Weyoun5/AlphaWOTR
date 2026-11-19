// Graph texts shared by the harnesses and the bot: a box graph's canonical text (as record_build_state.py's
// canon_action, a menu row's identity), its JSON value (an Enum as "\x01enum:MEMBER"), a snapshot's export.
#pragma once
#include <cstring>
#include <string>
#include <vector>
#include "json.h"
#include "pyobj.h"
#include "wotr_snapshot.h"

namespace awgt {
struct G {
    const pyobj::Graph& g;
    std::string text(uint64_t n) const { const auto& v = g.values[n]; return std::string(reinterpret_cast<const char*>(&g.bytes[v.offset]), v.count); }
    std::vector<uint64_t> kids(uint64_t n) const { const auto& v = g.values[n]; return std::vector<uint64_t>(g.edges.begin() + v.offset, g.edges.begin() + v.offset + v.count); }
};

inline std::string canon(const G& a, uint64_t n, int depth = 0) {
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
inline std::string pcanon(const G& a, uint64_t n) {
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
inline awjson::Value to_value(const G& a, uint64_t n) {
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
inline pyobj::Graph export_graph(const void* h) {
    pyobj::Graph g;
    const size_t nv = wotr_snapshot_node_count(h), ne = wotr_snapshot_edge_count(h), nb = wotr_snapshot_byte_count(h);
    g.values.resize(nv); g.edges.resize(ne); g.bytes.resize(nb); g.root = wotr_snapshot_root(h);
    wotr_snapshot_export(h, g.values.data(), nv, g.edges.data(), ne, g.bytes.data(), nb);
    return g;
}
inline std::vector<std::string> menu_rows(const void* menu) {
    const pyobj::Graph g = export_graph(menu);
    G a{g};
    std::vector<std::string> rows;
    for (auto row : a.kids(g.root)) {
        const auto parts = a.kids(row);
        rows.push_back(std::to_string(g.values[parts[0]].integer) + "|" + a.text(parts[1]) + "|" + pcanon(a, parts[2]));
    }
    return rows;
}

}  // namespace awgt
