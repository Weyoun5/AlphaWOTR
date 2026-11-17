// The AlphaWOTR bot on the module (see bot.h), as protocol/wotr_protocol_bot.py reads.
#include "bot.h"
#include "graph_text.h"
#include "opts.h"
#include "wotr_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>

extern "C" int64_t awasm_snapshot_root_count(const void* handle);

using awjson::Value;
using namespace awad::opt;

namespace awbot {
namespace {
const std::set<std::string> CHAIN_KINDS = {"ARMY_MOVE", "RECRUIT", "MOVE_COMPANIONS", "MOVE_MINIONS", "CARD_CHOICE",
                                           "JOIN_GROUP", "ARMY_SELECTION", "CHOOSE_ATTACK", "SEPARATE_COMPANIONS"};
const std::set<std::string> SMALL_KINDS = {"discard_card", "discard_table_card", "eye_allocate", "fellowship_declare", "declare_decline",
                                           "combat_retreat", "combat_no_retreat", "retreat_into_siege", "casualty_allocate",
                                           "casualty_preserve_units", "casualty_preserve_elites"};
constexpr int64_t SMALL_SIMS = 100;
const std::set<std::string> SETUP_KINDS = {"CARD_CHOICE", "ACTION", "FELLOWSHIP_PHASE", "HUNT_ALLOCATION", "DISCARD_EXCESS_CARD", "COMBAT_CARD",
                                           "CASUALTY", "CEASE_ATTACK", "EXTEND_SIEGE", "RETREAT", "FIELD_OR_SIEGE", "ADVANCE", "WHITE_RIDER",
                                           "HUNT_CASUALTY", "GUIDE_ABILITY", "REVEAL_FELLOWSHIP", "CHOOSE_GUIDE"};
std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += char(c);
    }
    return o + "\"";
}
std::string json_of(const Value& v) {
    switch (v.kind) {
    case Value::Null: return "null";
    case Value::Bool: return v.b ? "true" : "false";
    case Value::Int: return std::to_string(v.i);
    case Value::Real: { char b[32]; std::snprintf(b, sizeof b, "%.17g", v.d); return b; }
    case Value::Str: return json_str(v.s);
    case Value::Arr: { std::string o = "["; for (size_t i = 0; i < v.arr.size(); ++i) o += (i ? "," : "") + json_of(v.arr[i]); return o + "]"; }
    case Value::Obj: { std::string o = "{"; for (size_t i = 0; i < v.obj.size(); ++i) o += (i ? "," : "") + json_str(v.obj[i].first) + ":" + json_of(v.obj[i].second); return o + "}"; }
    }
    return "null";
}
std::string row_type(const pyobj::Graph& g, uint64_t row) {
    awgt::G a{g};
    return a.text(a.kids(row)[1]);
}
}  // namespace

std::optional<double> decision_ms(const Value& clock, const std::string& clock_kind) {
    const Value* left = clock.get("left_ms");
    if (!left || left->is_null()) return std::nullopt;
    if (clock_kind == "correspondence" || clock.get("deadline")) return std::nullopt;
    const double l = left->kind == Value::Real ? left->d : double(left->as_int());
    if (clock_kind == "game" || clock.get("opponent_ms")) {
        const Value* inc = clock.get("increment_ms");
        const double i = (inc && inc->truthy()) ? (inc->kind == Value::Real ? inc->d : double(inc->as_int())) : 0.0;
        return std::min(l / 16.0 + i / 2.0, l / 4.0);
    }
    return l;
}

Bot::Bot(pyobj::Model& model, awad::Adapter& adapter, const awad::Tables& tables, awdec::Box& b)
    : m(model), ad(adapter), t(tables), box(b) {}
Bot::~Bot() { if (memo.rng) wotr_random_destroy(memo.rng); }

void Bot::set_memo_rng(const std::vector<uint32_t>& words, uint32_t index, bool has_gauss, double gauss) {
    uint64_t bits = 0; std::memcpy(&bits, &gauss, 8);
    if (memo.rng) wotr_random_destroy(memo.rng);
    memo.rng = wotr_random_create(words.data(), words.size(), index, has_gauss ? 1 : 0, bits);
}

int64_t Bot::budget(const std::string& kind, std::optional<double> left_ms) const {
    int64_t n = sims;
    std::optional<double> limit = max_seconds;
    if (left_ms) {
        const double room = std::max(0.5, (*left_ms / 1000.0) * 0.4 - 0.2);
        limit = limit ? std::min(*limit, room) : room;
    }
    if (limit && !rates.empty()) {
        auto it = rates.find(kind);
        double rate = (it != rates.end() && it->second) ? it->second : std::min_element(rates.begin(), rates.end(), [](auto& a, auto& b) { return a.second < b.second; })->second;
        n = std::min<int64_t>(n, std::max<int64_t>(8, int64_t(rate * *limit)));
    }
    return n;
}

std::string Bot::handle(const Value& msg) {
    const std::string ty = at(msg, "type").kind == Value::Str ? at(msg, "type").s : "";
    if (ty == "new_game") {
        const Value& sb = at(msg, "seed_bot");
        const uint64_t seed = sb.truthy() ? uint64_t(std::llabs(sb.as_int())) : 0;
        if (memo.rng) wotr_random_destroy(memo.rng);
        memo = awad::Memo();
        memo.side = at(msg, "side").kind == Value::Str && at(msg, "side").s == "SHADOW" ? 1 : 0;
        const uint32_t limbs[2] = {uint32_t(seed), uint32_t(seed >> 32)};
        memo.rng = wotr_random_create_seeded(limbs, (seed >> 32) ? 2 : 1);   // random.Random(seed)
        own_seed = seed;
        plan.reset(); work = false; in_game = true;
        clock_kind = at(or_empty(msg, "clock"), "kind").kind == Value::Str ? at(or_empty(msg, "clock"), "kind").s : "";
        return "{\"type\":\"ready\",\"id\":" + json_of(at(msg, "id")) + "}";
    }
    if (ty == "event") {   // GameMemo.event
        if (!in_game) return "";
        const Value& ev = or_empty(msg, "event");
        const std::string k = at(ev, "kind").kind == Value::Str ? at(ev, "kind").s : "";
        if (k == "turn_started") { memo.messenger_used = false; memo.fellowship_acted = false; }
        else if (k == "fellowship_entered_mordor") memo.in_mordor = true;
        else if (k == "fellowship_moved" || k == "fellowship_hidden") { if (memo.in_mordor) memo.fellowship_acted = true; }
        else if (k == "die_used") {
            if (at(ev, "side").kind == Value::Str && at(ev, "side").s == "SHADOW" && at(ev, "face").kind == Value::Str && at(ev, "face").s == "MUSTER" &&
                at(ev, "action").kind == Value::Str && (at(ev, "action").s == "move_armies" || at(ev, "action").s == "attack"))
                memo.messenger_used = true;
        }
        return "";
    }
    if (ty == "decide") return decide(msg);
    if (ty == "game_over") { in_game = false; return ""; }
    return "";
}

std::string Bot::decide(const Value& msg) {
    const Value& dec = at(msg, "decision");
    last = Trace();
    const auto left = decision_ms(or_empty(msg, "clock"), clock_kind);
    std::optional<int64_t> o; std::string how;
    try { std::tie(o, how) = decide_(dec, or_empty(msg, "view"), left); }
    catch (const std::exception& e) { how = "error"; last.note = e.what(); o.reset(); }
    std::set<int64_t> legal;
    for (const Value& x : or_empty(dec, "options").arr) if (x.get("o") && !at(x, "o").is_null()) legal.insert(at(x, "o").as_int());
    if (!o || !legal.count(*o)) {
        const auto d = awad::default_option(dec);
        o = d.first;
        how = how != "error" ? "default" : "error-default";
        plan.reset();
    }
    last.o = o; last.how = how;
    return "{\"type\":\"choice\",\"id\":" + json_of(at(msg, "id")) + ",\"o\":" + std::to_string(*o) + ",\"fingerprint\":" +
           json_of(at(dec, "fingerprint")) + ",\"info\":{\"how\":" + json_str(how) + "}}";
}

std::pair<std::optional<int64_t>, std::string> Bot::decide_(const Value& dec, const Value& view, std::optional<double> left) {
    const std::string kind = at(dec, "kind").s;
    const Value& options = or_empty(dec, "options");
    if (options.arr.size() == 1) {
        if (plan && plan->takes(kind)) { try { plan->answer(dec); } catch (...) { plan.reset(); } }   // _continue_plan_quietly
        return {at(options.arr[0], "o").is_null() ? std::nullopt : std::optional<int64_t>(at(options.arr[0], "o").as_int()), "only"};
    }
    if (plan && plan->takes(kind)) {   // 1. the plan of the engine's last action
        const auto o = plan->answer(dec);
        if (o) { last.note = plan->name; return {o, "plan"}; }
    }
    plan.reset();
    if (work && CHAIN_KINDS.count(kind)) {   // 2. a chain of the engine's last action: not ported yet
        last.unported = "chain";
        return {std::nullopt, "unported"};
    }
    if (!CHAIN_KINDS.count(kind)) work = false;
    // 3. the state rebuilt from the view, the question set up
    pyobj::Obj* st = ad.build_state(view, memo);
    awad::Adapter::Setup su;
    if (SETUP_KINDS.count(kind)) {
        su = ad.setup(kind, st, dec, view, memo);
        if (!su.ported) { last.unported = "setup " + kind; return {std::nullopt, "unported"}; }
    }
    if (su.label.empty() && !su.has_match) {   // ready None: a rule, else the default
        bool has = false;
        const auto o = awad::rule(t, kind, dec, view, memo.side, work, &has);
        if (o) return {o, "rule"};
        return {std::nullopt, "default"};
    }
    if (su.has_match) return {su.match_none ? std::nullopt : std::optional<int64_t>(su.match), "rule"};
    return search_and_map(st, dec, su.label, su.flag, left);
}

std::pair<std::optional<int64_t>, std::string> Bot::search_and_map(pyobj::Obj* st, const Value& dec, const std::string& label,
                                                                   const std::string& flag, std::optional<double> left) {
    using Matched = awad::Adapter::Matched;
    const std::string kind = at(dec, "kind").s;
    const auto t0 = std::chrono::steady_clock::now();
    if (!flag.empty()) m.set_attr(st, flag, m.boolean(true));
    const pyobj::Graph gf = m.encode(st);
    void* sn = wotr_snapshot_create(gf.values.data(), gf.values.size(), gf.edges.data(), gf.edges.size(), gf.bytes.data(), gf.bytes.size(), gf.root);
    if (!sn) throw std::runtime_error(std::string("state: ") + wotr_snapshot_last_error());
    void* full = wotr_snapshot_full_actions(sn, ad.obs_tables, memo.side);
    struct Guard { void* p; ~Guard() { if (p) wotr_snapshot_destroy(p); } } g1{sn}, g2{full};
    if (!full) throw std::runtime_error(std::string("menu: ") + wotr_snapshot_last_error());
    const auto rows = awgt::menu_rows(full);
    if (rows.empty()) return {std::nullopt, "default"};
    void* meta = nullptr;
    void* sel = wotr_snapshot_project_policy(sn, full, memo.side, &meta);
    Guard g3{sel}, g4{meta};
    std::vector<std::string> prow = sel ? awgt::menu_rows(sel) : std::vector<std::string>{};
    if (prow.empty()) prow = rows;
    // every row matched to an option (map_action), the rest banned
    const pyobj::Graph mg = awgt::export_graph(full);
    awgt::G ga{mg};
    const auto rowsn = ga.kids(mg.root);
    std::vector<Matched> mts;
    for (size_t ri = 0; ri < rowsn.size(); ++ri) {
        const auto parts = ga.kids(rowsn[ri]);
        awad::Adapter::Row row; row.die = mg.values[parts[0]].integer; row.type = ga.text(parts[1]);
        row.params = awgt::to_value(ga, parts[2]); row.menu = full; row.index = int64_t(ri);
        Matched mt;
        try { mt = ad.map_action(kind, st, dec, row, memo); } catch (...) { mt = Matched(); }
        if (mt.unported) last.unported = "matching " + kind;
        mts.push_back(std::move(mt));
    }
    std::set<std::string> banned;
    std::vector<size_t> mapped;
    for (size_t ri = 0; ri < mts.size(); ++ri) { if (mts[ri].ok) mapped.push_back(ri); else banned.insert(rows[ri]); }
    if (mapped.empty()) return {std::nullopt, "default"};
    bool single = true;
    for (size_t q : mapped) single &= mts[q].o == mts[mapped[0]].o && mts[q].o_none == mts[mapped[0]].o_none && !mts[q].has_plan && !mts[q].chain;
    if (single) { last.engine = rows[mapped[0]]; return {mts[mapped[0]].o_none ? std::nullopt : std::optional<int64_t>(mts[mapped[0]].o), "single"}; }
    Matched* chosen = nullptr;
    Matched units_mt;
    void* units_menu = nullptr;
    Guard g5{nullptr};
    std::string how = "single";
    if (mapped.size() == 1) chosen = &mts[mapped[0]];
    else {
        // LeagueSearcher.choose: the projected menu's rows not banned (else the legal menu's)
        std::vector<int64_t> allowed;
        for (const auto& pr : prow) if (!banned.count(pr)) { auto it = std::find(rows.begin(), rows.end(), pr); if (it != rows.end()) allowed.push_back(int64_t(it - rows.begin())); }
        if (allowed.empty()) for (size_t ri = 0; ri < rows.size(); ++ri) if (!banned.count(rows[ri])) allowed.push_back(int64_t(ri));
        std::string pick;
        if (allowed.empty()) last.note = "no allowed action";
        else if (allowed.size() == 1) { pick = rows[size_t(allowed[0])]; how = "search"; }
        else {
            how = "search";
            int64_t n = budget(label, left);
            bool small = true;
            for (auto ai : allowed) small &= SMALL_KINDS.count(row_type(mg, rowsn[size_t(ai)])) != 0;
            if (n && small) n = std::min(n, SMALL_SIMS);
            std::string act;
            if (n) {
                const SearchArgs sa = next_search ? next_search(n) : SearchArgs{own_seed, uint32_t(last.searches), own_seed ^ 0x9e3779b97f4a7c15ULL, n};
                ++last.searches;
                const auto ts = std::chrono::steady_clock::now();
                const auto pk = awdec::search(box, sn, memo.side, sa.sims, sa.seat, sa.index, sa.seed, 0);
                if (!pk.error.empty()) throw std::runtime_error("search: " + pk.error);
                const auto mr = awgt::menu_rows(pk.menu);
                act = pk.index >= 0 && size_t(pk.index) < mr.size() ? mr[size_t(pk.index)] : "";
                wotr_snapshot_destroy(pk.menu);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
                if (sa.sims >= 8 && secs > 0.05) {   // the rate: the slowest of the last three searches of this kind
                    auto& rc = recent[label];
                    if (rc.size() > 2) rc.erase(rc.begin(), rc.end() - 2);
                    rc.push_back(sa.sims / secs);
                    rates[label] = *std::min_element(rc.begin(), rc.end());
                }
            }
            std::string hit;
            for (auto ai : allowed) if (rows[size_t(ai)] == act) { hit = act; break; }
            if (hit.empty()) {   // the search's pick has no referee counterpart: the net's own choice among the allowed
                void* sub = awdec::menu_rows(full, allowed);
                const int64_t ri = sub ? awdec::raw(box, sn, memo.side, sub) : -1;
                if (sub) wotr_snapshot_destroy(sub);
                if (ri >= 0) hit = rows[size_t(allowed[size_t(ri)])];
            }
            pick = hit;
            // THE CERTAIN LINE over the projected menu, then the ban check
            void* pmenu = (sel && awasm_snapshot_root_count(sel) > 0) ? sel : full;
            if (!hit.empty()) {
                int64_t own_i = -1;
                for (size_t q = 0; q < prow.size(); ++q) if (prow[q] == hit) { own_i = int64_t(q); break; }
                if (own_i >= 0) {
                    const int64_t outright = awdec::certain_line(box, sn, memo.side, pmenu, own_i);
                    if (outright >= 0 && !banned.count(prow[size_t(outright)])) pick = prow[size_t(outright)];
                }
            }
            // the seat's unit rule
            if (!pick.empty()) {
                auto it = std::find(rows.begin(), rows.end(), pick);
                int changed = 0;
                void* um = it == rows.end() ? nullptr : awdec::units(box, sn, memo.side, full, int64_t(it - rows.begin()), &changed);
                if (um) {
                    const std::string ur = awgt::menu_rows(um)[0];
                    if (!banned.count(ur)) { pick = ur; units_menu = um; g5.p = um; } else wotr_snapshot_destroy(um);
                }
            }
        }
        if (!pick.empty()) for (size_t q : mapped) if (rows[q] == pick) { chosen = &mts[q]; last.engine = pick; break; }
        if (!chosen && !pick.empty() && units_menu && !banned.count(pick)) {   // matched by itself
            const pyobj::Graph ug = awgt::export_graph(units_menu);
            awgt::G gu{ug};
            const auto parts = gu.kids(gu.kids(ug.root)[0]);
            awad::Adapter::Row row; row.die = ug.values[parts[0]].integer; row.type = gu.text(parts[1]);
            row.params = awgt::to_value(gu, parts[2]); row.menu = units_menu; row.index = 0;
            try { units_mt = ad.map_action(kind, st, dec, row, memo); } catch (...) { units_mt = Matched(); }
            if (units_mt.ok) { chosen = &units_mt; last.engine = pick; last.note = "the seat's unit rule"; }
        }
        if (!chosen) { chosen = &mts[mapped[0]]; last.engine = rows[mapped[0]]; last.note = "fallback: first matched"; }
    }
    const auto o = chosen->option();
    plan = chosen->plan();
    if (chosen->chain) work = true;
    (void)t0;
    return {o, how};
}

}  // namespace awbot
