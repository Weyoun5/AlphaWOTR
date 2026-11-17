// The adapter's option helpers (protocol/wotr_protocol_bot.py opts, first, a field compared as Python's o.get(k) != v)
#pragma once
#include <string>
#include <vector>
#include "json.h"

namespace awad::opt {
using awjson::Value;
inline const Value NULLV;
inline const Value& at(const Value& v, const std::string& k) { const Value* x = v.get(k); return x ? *x : NULLV; }
inline const Value& or_empty(const Value& v, const std::string& k) { const Value& x = at(v, k); return x.truthy() ? x : NULLV; }
inline Value S(const std::string& s) { Value v; v.kind = Value::Str; v.s = s; return v; }
inline Value I(int64_t i) { Value v; v.kind = Value::Int; v.i = i; return v; }
inline Value B(bool b) { Value v; v.kind = Value::Bool; v.b = b; return v; }
// a field value as Python compares o.get(k) != v: None for a missing slug
struct F { std::string k; bool none; Value v; };
inline F f(const std::string& k, const Value& v) { return {k, false, v}; }
inline F fnone(const std::string& k) { return {k, true, Value()}; }
inline F fstr(const std::string& k, const std::string& s, bool present) { return present ? f(k, S(s)) : fnone(k); }
inline bool equal(const Value& a, const Value& b) {
    if ((a.kind == Value::Int || a.kind == Value::Bool) && (b.kind == Value::Int || b.kind == Value::Bool))
        return (a.kind == Value::Int ? a.i : a.b) == (b.kind == Value::Int ? b.i : b.b);
    if (a.kind != b.kind) return false;
    if (a.kind == Value::Str) return a.s == b.s;
    if (a.kind == Value::Null) return true;
    if (a.kind == Value::Real) return a.d == b.d;
    return false;
}
inline std::vector<const Value*> opts(const Value& dec, const std::string& kind, const std::vector<F>& fields = {}) {
    std::vector<const Value*> out;
    for (const Value& o : or_empty(dec, "options").arr) {
        if (!kind.empty() && at(o, "kind").s != kind) continue;
        bool ok = true;
        for (const auto& fl : fields) {
            const Value& x = at(o, fl.k);
            if (fl.none ? x.kind != Value::Null : !equal(x, fl.v)) { ok = false; break; }
        }
        if (ok) out.push_back(&o);
    }
    return out;
}
inline bool has_o(const Value* o) { return o && at(*o, "o").kind != Value::Null; }
// first(lst): the first option's 'o' (none: false)
inline bool first(const std::vector<const Value*>& c, int64_t& o) { if (c.empty() || !has_o(c[0])) return false; o = at(*c[0], "o").as_int(); return true; }
}  // namespace awad::opt
