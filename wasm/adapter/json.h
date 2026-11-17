// A small JSON reader for the protocol's messages and the adapter's tables: objects keep their key order (Python's
// json.loads gives dicts in document order, and the adapter iterates them in that order).
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace awjson {

struct Value {
    enum Kind { Null, Bool, Int, Real, Str, Arr, Obj } kind = Null;
    bool b = false; int64_t i = 0; double d = 0; std::string s;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    bool is_null() const { return kind == Null; }
    const Value* get(const std::string& k) const {
        if (kind != Obj) return nullptr;
        for (const auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    // Python truthiness
    bool truthy() const {
        switch (kind) {
        case Null: return false; case Bool: return b; case Int: return i != 0; case Real: return d != 0.0;
        case Str: return !s.empty(); case Arr: return !arr.empty(); case Obj: return !obj.empty();
        }
        return false;
    }
    // int(x) of a number or a numeric string
    int64_t as_int() const {
        if (kind == Int) return i;
        if (kind == Bool) return b;
        if (kind == Real) return int64_t(d);
        if (kind == Str) return std::stoll(s);
        throw std::runtime_error("JSON value is not a number");
    }
};

struct Parser {
    const char* p; const char* e;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    static void utf8(std::string& o, uint32_t c) {
        if (c < 0x80) o += char(c);
        else if (c < 0x800) { o += char(0xc0 | (c >> 6)); o += char(0x80 | (c & 0x3f)); }
        else if (c < 0x10000) { o += char(0xe0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 0x3f)); o += char(0x80 | (c & 0x3f)); }
        else { o += char(0xf0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 0x3f)); o += char(0x80 | ((c >> 6) & 0x3f)); o += char(0x80 | (c & 0x3f)); }
    }
    uint32_t hex4() {
        if (e - p < 4) throw std::runtime_error("JSON: short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = *p++; v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else throw std::runtime_error("JSON: bad \\u escape");
        }
        return v;
    }
    std::string string() {
        if (p >= e || *p != '"') throw std::runtime_error("JSON: string expected");
        ++p;
        std::string o;
        while (p < e && *p != '"') {
            if (*p != '\\') { o += *p++; continue; }
            ++p;
            const char c = *p++;
            switch (c) {
            case '"': o += '"'; break; case '\\': o += '\\'; break; case '/': o += '/'; break;
            case 'b': o += '\b'; break; case 'f': o += '\f'; break; case 'n': o += '\n'; break;
            case 'r': o += '\r'; break; case 't': o += '\t'; break;
            case 'u': {
                uint32_t cp = hex4();
                if (cp >= 0xd800 && cp < 0xdc00 && e - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                    p += 2; const uint32_t lo = hex4(); cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                }
                utf8(o, cp); break;
            }
            default: throw std::runtime_error("JSON: bad escape");
            }
        }
        if (p >= e) throw std::runtime_error("JSON: unterminated string");
        ++p;
        return o;
    }
    Value value(int depth = 0) {
        if (depth > 200) throw std::runtime_error("JSON nested too deeply");
        ws();
        if (p >= e) throw std::runtime_error("JSON ends early");
        Value v;
        if (*p == '{') {
            v.kind = Value::Obj; ++p; ws();
            if (*p == '}') { ++p; return v; }
            for (;;) {
                ws(); std::string k = string(); ws();
                if (*p++ != ':') throw std::runtime_error("JSON: ':' expected");
                Value x = value(depth + 1);
                bool replaced = false;   // a repeated key keeps its first position and takes the last value
                for (auto& kv : v.obj) if (kv.first == k) { kv.second = std::move(x); replaced = true; break; }
                if (!replaced) v.obj.emplace_back(std::move(k), std::move(x));
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != '}') throw std::runtime_error("JSON: '}' expected");
                return v;
            }
        }
        if (*p == '[') {
            v.kind = Value::Arr; ++p; ws();
            if (*p == ']') { ++p; return v; }
            for (;;) {
                v.arr.push_back(value(depth + 1)); ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != ']') throw std::runtime_error("JSON: ']' expected");
                return v;
            }
        }
        if (*p == '"') { v.kind = Value::Str; v.s = string(); return v; }
        if (e - p >= 4 && !std::strncmp(p, "null", 4)) { p += 4; return v; }
        if (e - p >= 4 && !std::strncmp(p, "true", 4)) { p += 4; v.kind = Value::Bool; v.b = true; return v; }
        if (e - p >= 5 && !std::strncmp(p, "false", 5)) { p += 5; v.kind = Value::Bool; v.b = false; return v; }
        const char* start = p;
        bool real = false;
        if (*p == '-') ++p;
        while (p < e && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) {
            if (*p == '.' || *p == 'e' || *p == 'E') real = true;
            ++p;
        }
        if (p == start) throw std::runtime_error("JSON: bad value");
        const std::string num(start, p);
        if (real) { v.kind = Value::Real; v.d = std::strtod(num.c_str(), nullptr); }
        else { v.kind = Value::Int; try { v.i = std::stoll(num); } catch (const std::out_of_range&) { v.i = int64_t(std::stoull(num)); } }   // (a u64 seed: its bits)
        return v;
    }
};

inline Value parse(const std::string& text) {
    Parser ps{text.data(), text.data() + text.size()};
    Value v = ps.value();
    ps.ws();
    if (ps.p != ps.e) throw std::runtime_error("JSON: trailing characters");
    return v;
}

}  // namespace awjson
