// A mutable model of the engine's Python object graph, for the protocol adapter's port (phase 3).
//
// The adapter builds a GameState of their Python engine from the protocol's view; the box works on that state packed
// as a typed graph (native_backend/snapshot.py encode_graph). This model holds such a graph as linked objects with
// identity (a list held by two objects is one object), is decoded from a graph (the prototypes Python's constructors
// make) and encoded back exactly as encode_graph does: a node per compound object (by identity), one node per
// distinct string (by value), scalars every time, children after their container's node, a registered global list's
// binding name before its items.
//   tag: the WotrValue tags (WOTR_NONE ... WOTR_BIGINT). OBJECT kids: [type name, attribute name, value, ...] as
//   encode_graph writes them ('__dict__' then the attribute dict, then the class's slots); ENUM kids: [type, member];
//   DICT kids: key, value, key, value, ... in insertion order.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "wotr_snapshot.h"

namespace pyobj {

// CPython's str hash under PYTHONHASHSEED=0 (the AI server's launch_head.HEAD_ENV): SipHash-1-3 with a zero key over
// the string's bytes (1-byte kind: ASCII and Latin-1 text), -1 mapped to -2; 0 for the empty string
inline int64_t str_hash(const std::string& s) {
    if (s.empty()) return 0;
    auto rotl = [](uint64_t x, int b) { return (x << b) | (x >> (64 - b)); };
    uint64_t v0 = 0x736f6d6570736575ULL, v1 = 0x646f72616e646f6dULL, v2 = 0x6c7967656e657261ULL, v3 = 0x7465646279746573ULL;
    auto round = [&]() {
        v0 += v1; v1 = rotl(v1, 13); v1 ^= v0; v0 = rotl(v0, 32);
        v2 += v3; v3 = rotl(v3, 16); v3 ^= v2;
        v0 += v3; v3 = rotl(v3, 21); v3 ^= v0;
        v2 += v1; v1 = rotl(v1, 17); v1 ^= v2; v2 = rotl(v2, 32);
    };
    const uint8_t* in = reinterpret_cast<const uint8_t*>(s.data());
    const size_t len = s.size();
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t m; std::memcpy(&m, in + i, 8);
        v3 ^= m; round(); v0 ^= m;
    }
    uint64_t b = uint64_t(len) << 56;
    for (size_t j = 0; j < len - i; ++j) b |= uint64_t(in[i + j]) << (8 * j);
    v3 ^= b; round(); v0 ^= b;
    v2 ^= 0xff; round(); round(); round();
    int64_t h = int64_t(v0 ^ v1 ^ v2 ^ v3);
    return h == -1 ? -2 : h;
}

// The iteration order of a Python set made empty and then given these strings by add() in this order
// (Objects/setobject.c: set_add_entry with LINEAR_PROBES 9 and PERTURB_SHIFT 5, resizing at fill*5 >= mask*3 to
// used*4, re-inserting in table order).
inline std::vector<size_t> cpython_set_order(const std::vector<std::string>& items) {
    struct Slot { long item = -1; int64_t hash = 0; };
    std::vector<Slot> table(8);
    size_t mask = 7, fill = 0, used = 0;
    auto insert_clean = [](std::vector<Slot>& t, size_t m, long item, int64_t hash) {
        size_t perturb = size_t(hash), i = size_t(hash) & m;
        for (;;) {
            if (t[i].item < 0) { t[i] = {item, hash}; return; }
            if (i + 9 <= m) for (size_t j = 1; j <= 9; ++j) if (t[i + j].item < 0) { t[i + j] = {item, hash}; return; }
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & m;
        }
    };
    for (size_t n = 0; n < items.size(); ++n) {
        const int64_t hash = str_hash(items[n]);
        size_t perturb = size_t(hash), i = size_t(hash) & mask;
        bool present = false, placed = false;
        while (!placed && !present) {
            size_t probes = (i + 9 <= mask) ? 9 : 0;
            size_t e = i;
            for (;;) {
                if (table[e].item < 0) { table[e] = {long(n), hash}; placed = true; break; }
                if (table[e].hash == hash && items[size_t(table[e].item)] == items[n]) { present = true; break; }
                if (probes-- == 0) break;
                ++e;
            }
            if (placed || present) break;
            perturb >>= 5;
            i = (i * 5 + 1 + perturb) & mask;
        }
        if (present) continue;
        ++fill; ++used;
        if (fill * 5 >= mask * 3) {
            size_t minused = used > 50000 ? used * 2 : used * 4, size = 8;
            while (size <= minused) size <<= 1;
            std::vector<Slot> fresh(size);
            for (const auto& sl : table) if (sl.item >= 0) insert_clean(fresh, size - 1, sl.item, sl.hash);
            table.swap(fresh); mask = size - 1; fill = used;
        }
    }
    std::vector<size_t> order;
    for (const auto& sl : table) if (sl.item >= 0) order.push_back(size_t(sl.item));
    return order;
}

struct Obj {
    uint32_t tag = WOTR_NONE;
    bool binding = false;              // a registered global list (flags 1); its name in `s`
    bool fresh_set = false;            // a set made here by set() + add: encoded in CPython's table order
    int64_t i = 0; double d = 0.0;
    std::string s;                     // STRING / BYTES / BIGINT text, the binding name
    std::vector<Obj*> k;               // children
};

struct Graph {                         // a flat graph, as wotr_snapshot_create takes it
    std::vector<WotrValue> values; std::vector<uint64_t> edges; std::vector<uint8_t> bytes; uint64_t root = 0;
};

class Model {
public:
    Obj* make(uint32_t tag) { arena_.emplace_back(); arena_.back().tag = tag; return &arena_.back(); }
    Obj* none() { return make(WOTR_NONE); }
    Obj* boolean(bool b) { Obj* o = make(WOTR_BOOL); o->i = b; return o; }
    Obj* integer(int64_t v) { Obj* o = make(WOTR_INT); o->i = v; return o; }
    Obj* str(const std::string& v) { Obj* o = make(WOTR_STRING); o->s = v; return o; }
    Obj* list(std::vector<Obj*> items = {}) { Obj* o = make(WOTR_LIST); o->k = std::move(items); return o; }
    Obj* tuple(std::vector<Obj*> items = {}) { Obj* o = make(WOTR_TUPLE); o->k = std::move(items); return o; }
    Obj* set() { Obj* o = make(WOTR_SET); o->fresh_set = true; return o; }
    Obj* dict() { return make(WOTR_DICT); }
    // an enum member: one object per (type, member), as Python's enum members are singletons
    Obj* enumv(const std::string& type, const std::string& member) {
        const std::string key = type + "." + member;
        auto it = enums_.find(key);
        if (it != enums_.end()) return it->second;
        Obj* o = make(WOTR_ENUM); o->k = {str(type), str(member)};
        enums_[key] = o;
        return o;
    }

    // ---- decoding a graph (memo by node: shared objects stay shared)
    Obj* decode(const Graph& g) {
        std::vector<Obj*> memo(g.values.size(), nullptr);
        return decode_node(g, g.root, memo);
    }

    // ---- encoding (snapshot.encode_graph)
    Graph encode(Obj* root) const {
        Graph g;
        std::unordered_map<const Obj*, uint64_t> memo;
        std::unordered_map<std::string, uint64_t> strings;
        g.root = visit(g, root, memo, strings);
        return g;
    }

    // ---- a deep copy of a prototype: new objects, except what the state shares with every other (enums, the
    //      registered global lists, and the objects of `keep`: the board and its regions)
    Obj* clone(Obj* o, std::unordered_map<Obj*, Obj*>& memo, const std::vector<std::string>& keep_types) {
        if (o->tag == WOTR_ENUM || o->binding) return o;
        if (o->tag <= WOTR_FLOAT || o->tag == WOTR_STRING || o->tag == WOTR_BYTES || o->tag == WOTR_BIGINT) {
            Obj* c = make(o->tag); *c = *o; return c;
        }
        auto it = memo.find(o);
        if (it != memo.end()) return it->second;
        if (o->tag == WOTR_OBJECT) for (const auto& t : keep_types) if (o->k[0]->s == t) return o;
        Obj* c = make(o->tag);
        memo[o] = c;
        c->i = o->i; c->d = o->d; c->s = o->s;
        for (Obj* x : o->k) c->k.push_back(clone(x, memo, keep_types));
        return c;
    }
    Obj* clone(Obj* o, const std::vector<std::string>& keep_types = {}) {
        std::unordered_map<Obj*, Obj*> memo;
        return clone(o, memo, keep_types);
    }
    // copy.copy of an object: a new object and a new attribute dict holding the same values
    Obj* shallow(Obj* o) {
        require_object(o);
        Obj* c = make(WOTR_OBJECT);
        c->k = o->k;
        for (size_t j = 1; j + 1 < c->k.size(); j += 2)
            if (c->k[j]->s == "__dict__") { Obj* d = make(WOTR_DICT); d->k = c->k[j + 1]->k; c->k[j + 1] = d; }
        return c;
    }

    // ---- objects and dicts
    static void require_object(const Obj* o) { if (!o || o->tag != WOTR_OBJECT) throw std::runtime_error("an engine object is expected"); }
    static const std::string& type(const Obj* o) { require_object(o); return o->k[0]->s; }
    static bool key_equal(const Obj* a, const Obj* b) {
        if (a == b) return true;
        if (a->tag == WOTR_STRING && b->tag == WOTR_STRING) return a->s == b->s;
        if ((a->tag == WOTR_INT || a->tag == WOTR_BOOL) && (b->tag == WOTR_INT || b->tag == WOTR_BOOL)) return a->i == b->i;
        return false;
    }
    // the slot of attribute `name`: in __dict__, else a slot; nullptr when absent
    static Obj** attr_slot(Obj* o, const std::string& name) {
        require_object(o);
        for (size_t j = 1; j + 1 < o->k.size(); j += 2) {
            if (o->k[j]->s == "__dict__") {
                Obj* d = o->k[j + 1];
                for (size_t m = 0; m + 1 < d->k.size(); m += 2) if (d->k[m]->tag == WOTR_STRING && d->k[m]->s == name) return &d->k[m + 1];
            } else if (o->k[j]->s == name) return &o->k[j + 1];
        }
        return nullptr;
    }
    static Obj* attr(Obj* o, const std::string& name) {
        Obj** s = attr_slot(o, name);
        if (!s) throw std::runtime_error("engine object " + type(o) + " has no attribute " + name);
        return *s;
    }
    // setattr: the existing attribute, else a new one at the end of __dict__ (Python's dict insertion)
    void set_attr(Obj* o, const std::string& name, Obj* v) {
        if (Obj** s = attr_slot(o, name)) { *s = v; return; }
        for (size_t j = 1; j + 1 < o->k.size(); j += 2)
            if (o->k[j]->s == "__dict__") { o->k[j + 1]->k.push_back(str(name)); o->k[j + 1]->k.push_back(v); return; }
        throw std::runtime_error("engine object " + type(o) + " takes no attribute " + name);
    }
    static Obj* dict_get(Obj* d, const Obj* key) {
        for (size_t m = 0; m + 1 < d->k.size(); m += 2) if (key_equal(d->k[m], key)) return d->k[m + 1];
        return nullptr;
    }
    static Obj* dict_get(Obj* d, const std::string& key) {
        for (size_t m = 0; m + 1 < d->k.size(); m += 2) if (d->k[m]->tag == WOTR_STRING && d->k[m]->s == key) return d->k[m + 1];
        return nullptr;
    }
    void dict_set(Obj* d, Obj* key, Obj* v) {
        for (size_t m = 0; m + 1 < d->k.size(); m += 2) if (key_equal(d->k[m], key)) { d->k[m + 1] = v; return; }
        d->k.push_back(key); d->k.push_back(v);
    }
    // set.add (the iteration order of a Python set is its hash table's; this model keeps insertion order, and the
    // parity test compares sets as sets)
    static void set_add(Obj* s, Obj* v) {
        for (Obj* x : s->k) if (key_equal(x, v)) return;
        s->k.push_back(v);
    }
    static bool truthy(const Obj* o) {
        switch (o->tag) {
        case WOTR_NONE: return false;
        case WOTR_BOOL: case WOTR_INT: return o->i != 0;
        case WOTR_FLOAT: return o->d != 0.0;
        case WOTR_STRING: case WOTR_BYTES: return !o->s.empty();
        case WOTR_LIST: case WOTR_TUPLE: case WOTR_SET: case WOTR_FROZENSET: return !o->k.empty();
        case WOTR_DICT: return !o->k.empty();
        default: return true;
        }
    }

private:
    std::deque<Obj> arena_;
    std::unordered_map<std::string, Obj*> enums_;

    Obj* decode_node(const Graph& g, uint64_t n, std::vector<Obj*>& memo) {
        if (memo[n]) return memo[n];
        const WotrValue& v = g.values[n];
        if (v.tag == WOTR_ENUM) {
            const auto& t = g.values[g.edges[v.offset]]; const auto& m = g.values[g.edges[v.offset + 1]];
            Obj* e = enumv(std::string(reinterpret_cast<const char*>(&g.bytes[t.offset]), t.count),
                           std::string(reinterpret_cast<const char*>(&g.bytes[m.offset]), m.count));
            memo[n] = e;
            return e;
        }
        Obj* o = make(v.tag);
        memo[n] = o;
        o->i = v.integer; o->d = v.real;
        if (v.tag == WOTR_STRING || v.tag == WOTR_BYTES || v.tag == WOTR_BIGINT) {
            o->s.assign(reinterpret_cast<const char*>(g.bytes.data() + v.offset), size_t(v.count));
            o->i = 0;
            return o;
        }
        if (v.tag >= WOTR_LIST) {
            if (v.flags == 1) {
                o->binding = true;
                const auto& nm = g.values[v.integer];
                o->s.assign(reinterpret_cast<const char*>(&g.bytes[nm.offset]), size_t(nm.count));
                o->i = 0;
            }
            for (uint64_t j = 0; j < v.count; ++j) o->k.push_back(decode_node(g, g.edges[v.offset + j], memo));
        }
        return o;
    }
    static uint64_t visit(Graph& g, const Obj* o, std::unordered_map<const Obj*, uint64_t>& memo,
                          std::unordered_map<std::string, uint64_t>& strings) {
        if (o->tag == WOTR_STRING) { auto it = strings.find(o->s); if (it != strings.end()) return it->second; }
        const bool compound = o->tag >= WOTR_LIST && o->tag != WOTR_BIGINT;
        // CPython's empty tuple is one object: every () of a state is one node
        static const Obj empty_tuple_key{};
        if (o->tag == WOTR_TUPLE && o->k.empty()) {
            auto it = memo.find(&empty_tuple_key);
            if (it != memo.end()) return it->second;
            memo[&empty_tuple_key] = g.values.size();
        }
        if (compound) { auto it = memo.find(o); if (it != memo.end()) return it->second; }
        const uint64_t index = g.values.size();
        g.values.push_back(WotrValue{});
        if (compound) memo[o] = index;
        WotrValue node{};
        node.tag = o->tag;
        switch (o->tag) {
        case WOTR_NONE: break;
        case WOTR_BOOL: case WOTR_INT: node.integer = o->i; break;
        case WOTR_FLOAT: node.real = o->d; break;
        case WOTR_STRING: case WOTR_BYTES: case WOTR_BIGINT:
            node.offset = g.bytes.size(); node.count = o->s.size();
            g.bytes.insert(g.bytes.end(), o->s.begin(), o->s.end());
            if (o->tag == WOTR_STRING) strings[o->s] = index;
            break;
        default: {
            if (o->binding) {
                node.flags = 1;
                Obj name; name.tag = WOTR_STRING; name.s = o->s;
                node.integer = int64_t(visit(g, &name, memo, strings));
            }
            std::vector<uint64_t> kids;
            std::vector<const Obj*> items(o->k.begin(), o->k.end());
            if (o->fresh_set && !items.empty() &&
                std::all_of(items.begin(), items.end(), [](const Obj* x) { return x->tag == WOTR_STRING; })) {
                std::vector<std::string> texts;
                for (const Obj* x : items) texts.push_back(x->s);
                std::vector<const Obj*> ordered;
                for (size_t j : cpython_set_order(texts)) ordered.push_back(items[j]);
                items.swap(ordered);
            }
            for (const Obj* x : items) kids.push_back(visit(g, x, memo, strings));
            node.offset = g.edges.size(); node.count = kids.size();
            g.edges.insert(g.edges.end(), kids.begin(), kids.end());
        }
        }
        g.values[index] = node;
        return index;
    }
};

// Graph <-> the awg record layout (u64 values, edges, bytes, root, then the arrays), as written by the tools
inline Graph read_graph(const uint8_t* counts) {
    Graph g; uint64_t nv, ne, nb;
    std::memcpy(&nv, counts, 8); std::memcpy(&ne, counts + 8, 8); std::memcpy(&nb, counts + 16, 8); std::memcpy(&g.root, counts + 24, 8);
    g.values.resize(nv); g.edges.resize(ne); g.bytes.resize(nb);
    std::memcpy(g.values.data(), counts + 32, 40 * nv);
    std::memcpy(g.edges.data(), counts + 32 + 40 * nv, 8 * ne);
    if (nb) std::memcpy(g.bytes.data(), counts + 32 + 40 * nv + 8 * ne, nb);
    return g;
}

}  // namespace pyobj
