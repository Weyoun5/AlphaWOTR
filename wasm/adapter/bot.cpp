// The AlphaWOTR bot on the module (see bot.h), as protocol/wotr_protocol_bot.py reads.
#include "bot.h"
#include "graph_text.h"
#include "opts.h"
#include "wotr_random.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <set>

extern "C" int64_t awasm_snapshot_root_count(const void* handle);

using awjson::Value;
using namespace awad::opt;

#ifndef AW_COMMIT
#define AW_COMMIT "unknown"
#endif

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
Bot::~Bot() { if (memo.rng) wotr_random_destroy(memo.rng); if (own_rng) wotr_random_destroy(own_rng); }

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
        if (own_rng) wotr_random_destroy(own_rng);
        own_rng = wotr_random_create_seeded(limbs, (seed >> 32) ? 2 : 1);   // random.seed(seed_bot): the global stream
        plan.reset(); drop_work(); in_game = true;
        clock_kind = at(or_empty(msg, "clock"), "kind").kind == Value::Str ? at(or_empty(msg, "clock"), "kind").s : "";
        return "{\"type\":\"ready\",\"id\":" + json_of(at(msg, "id")) + "}";
    }
    if (ty == "hello")
        return "{\"type\":\"hello\",\"id\":" + json_of(at(msg, "id")) + ",\"protocol\":\"wotr-bot/0.1\",\"name\":\"AlphaWOTR\","
               "\"version\":\"v0.2.0/mc200_fpwin_shwin-s1/protocol-2/wasm-" AW_COMMIT "\",\"author\":\"AlphaWOTR (protocol adapter, WebAssembly)\",\"events\":true,"
               "\"options\":[{\"name\":\"sims\",\"type\":\"int\",\"default\":" + std::to_string(sims) + ",\"min\":1,\"max\":100000}]}";
    if (ty == "options") {
        const Value& v = or_empty(msg, "values");
        if (v.get("sims")) { try { sims = std::max<int64_t>(1, at(v, "sims").as_int()); } catch (...) {} }
        return "";
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


// the engine's menu on a state: get_legal_actions (the box's full menu), project_ai_actions, each row's text
struct Menu {
    void* sn = nullptr; void* full = nullptr; void* sel = nullptr; void* meta = nullptr;
    pyobj::Graph mg;
    std::vector<uint64_t> rowsn;
    std::vector<std::string> rows, prow;
    ~Menu() { for (void* p : {sn, full, sel, meta}) if (p) wotr_snapshot_destroy(p); }
};

namespace {
std::unique_ptr<Menu> menu_of(pyobj::Model& m, pyobj::Obj* st, const void* obs, int side) {
    auto mn = std::make_unique<Menu>();
    const pyobj::Graph gf = m.encode(st);
    mn->sn = wotr_snapshot_create(gf.values.data(), gf.values.size(), gf.edges.data(), gf.edges.size(), gf.bytes.data(), gf.bytes.size(), gf.root);
    if (!mn->sn) throw std::runtime_error(std::string("state: ") + wotr_snapshot_last_error());
    mn->full = wotr_snapshot_full_actions(mn->sn, obs, side);
    if (!mn->full) throw std::runtime_error(std::string("menu: ") + wotr_snapshot_last_error());
    mn->rows = awgt::menu_rows(mn->full);
    mn->sel = wotr_snapshot_project_policy(mn->sn, mn->full, side, &mn->meta);
    if (mn->sel) mn->prow = awgt::menu_rows(mn->sel);
    if (mn->prow.empty()) mn->prow = mn->rows;
    mn->mg = awgt::export_graph(mn->full);
    awgt::G ga{mn->mg};
    mn->rowsn = ga.kids(mn->mg.root);
    return mn;
}
awad::Adapter::Row row_of(const Menu& mn, size_t ri) {
    awgt::G ga{mn.mg};
    const auto parts = ga.kids(mn.rowsn[ri]);
    awad::Adapter::Row row; row.die = mn.mg.values[parts[0]].integer; row.type = ga.text(parts[1]);
    row.params = awgt::to_value(ga, parts[2]); row.menu = mn.full; row.index = int64_t(ri);
    return row;
}
bool is_none(pyobj::Obj* w, const char* name) {
    pyobj::Obj** v = pyobj::Model::attr_slot(w, name);
    return !v || (*v)->tag == WOTR_NONE;
}
std::string text_of(const Value& v) { return v.kind == Value::Str ? v.s : v.kind == Value::Null ? std::string("\x01") : v.kind == Value::Int ? std::to_string(v.i) : std::string("\x02"); }
}  // namespace

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
    if (work && CHAIN_KINDS.count(kind)) {   // 2. a chain of the engine's last action, on its copy of the state
        if (auto r = chain(dec, left)) return *r;
        if (auto o = diff_answer(dec)) { last.note = "as the engine resolved it"; return {o, "board"}; }
        if ((kind == "ARMY_MOVE" || kind == "RECRUIT" || kind == "MOVE_COMPANIONS" || kind == "MOVE_MINIONS") &&
            !opts(dec, "end").empty() && chain_over(work, kind)) {
            last.note = "chain over";
            const auto e = opts(dec, "end");
            return {has_o(e[0]) ? std::optional<int64_t>(at(*e[0], "o").as_int()) : std::nullopt, "chain"};
        }
    }
    if (!CHAIN_KINDS.count(kind)) drop_work();
    // 3. the state rebuilt from the view, the question set up
    pyobj::Obj* st = ad.build_state(view, memo);
    awad::Adapter::Setup su;
    if (SETUP_KINDS.count(kind)) {
        su = ad.setup(kind, st, dec, view, memo);
        if (!su.ported) { last.unported = "setup " + kind; return {std::nullopt, "unported"}; }
    }
    if (su.label.empty() && !su.has_match) {   // ready None: a rule, else the default
        bool has = false;
        const auto o = awad::rule(t, kind, dec, view, memo.side, work != nullptr, &has);
        if (o) return {o, "rule"};
        return {std::nullopt, "default"};
    }
    if (su.has_match) return {su.match_none ? std::nullopt : std::optional<int64_t>(su.match), "rule"};
    return search_and_map(st, dec, su.label, su.flag, left);
}

// LeagueSearcher.choose on a menu: the pick's row text ('' none); has_sims: its info carries 'sims';
// units_menu: the unit rule's action as a one-row menu (owned by the caller) when it named units
std::string Bot::choose(Menu& mn, const std::set<std::string>& banned, const std::string& label, std::optional<double> left,
                        bool& has_sims, void*& units_menu) {
    has_sims = false; units_menu = nullptr;
    const auto& rows = mn.rows;
    std::vector<int64_t> allowed;   // the projected menu's rows not banned (else the legal menu's)
    for (const auto& pr : mn.prow) if (!banned.count(pr)) { auto it = std::find(rows.begin(), rows.end(), pr); if (it != rows.end()) allowed.push_back(int64_t(it - rows.begin())); }
    if (allowed.empty()) for (size_t ri = 0; ri < rows.size(); ++ri) if (!banned.count(rows[ri])) allowed.push_back(int64_t(ri));
    if (allowed.empty()) { last.note = "no allowed action"; return ""; }
    has_sims = true;
    if (allowed.size() == 1) return rows[size_t(allowed[0])];
    int64_t n = budget(label, left);
    bool small = true;
    for (auto ai : allowed) small &= SMALL_KINDS.count(row_type(mn.mg, mn.rowsn[size_t(ai)])) != 0;
    if (n && small) n = std::min(n, SMALL_SIMS);
    std::string act;
    if (n) {
        const SearchArgs sa = next_search ? next_search(n) : SearchArgs{own_seed, uint32_t(last.searches), own_seed ^ 0x9e3779b97f4a7c15ULL, n};
        ++last.searches;
        const auto ts = std::chrono::steady_clock::now();
        const auto pk = awdec::search(box, mn.sn, memo.side, sa.sims, sa.seat, sa.index, sa.seed, 0);
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
        void* sub = awdec::menu_rows(mn.full, allowed);
        const int64_t ri = sub ? awdec::raw(box, mn.sn, memo.side, sub) : -1;
        if (sub) wotr_snapshot_destroy(sub);
        if (ri >= 0) hit = rows[size_t(allowed[size_t(ri)])];
    }
    std::string pick = hit;
    // THE CERTAIN LINE over the projected menu, then the ban check
    void* pmenu = (mn.sel && awasm_snapshot_root_count(mn.sel) > 0) ? mn.sel : mn.full;
    if (!hit.empty()) {
        int64_t own_i = -1;
        for (size_t q = 0; q < mn.prow.size(); ++q) if (mn.prow[q] == hit) { own_i = int64_t(q); break; }
        if (own_i >= 0) {
            const int64_t outright = awdec::certain_line(box, mn.sn, memo.side, pmenu, own_i);
            if (outright >= 0 && !banned.count(mn.prow[size_t(outright)])) pick = mn.prow[size_t(outright)];
        }
    }
    // the seat's unit rule
    if (!pick.empty()) {
        auto it = std::find(rows.begin(), rows.end(), pick);
        int changed = 0;
        void* um = it == rows.end() ? nullptr : awdec::units(box, mn.sn, memo.side, mn.full, int64_t(it - rows.begin()), &changed);
        if (um) {
            const std::string ur = awgt::menu_rows(um)[0];
            if (!banned.count(ur)) { pick = ur; units_menu = um; } else wotr_snapshot_destroy(um);
        }
    }
    return pick;
}

// _apply_copy(st, a, flag): the action applied on a copy (sn: the state's snapshot, with the flag set when one is
// given); its draws from the global Random. Null when the engine refused it.
pyobj::Obj* Bot::apply_copy(const void* sn, const void* menu, int64_t index, const std::string& flag) {
    ++last.applies;
    void* rng = next_apply_rng ? next_apply_rng() : nullptr;
    const bool own = rng == nullptr;
    if (own) { if (!own_rng) { const uint32_t z = 0; own_rng = wotr_random_create_seeded(&z, 1); } rng = own_rng; }
    void* rng_out = nullptr; void* events = nullptr;
    void* after = wotr_snapshot_apply_descriptor_random(sn, ad.obs_tables, menu, rng, memo.side, index, &rng_out, &events);
    if (events) wotr_snapshot_destroy(events);
    if (rng_out) { if (own) { wotr_random_destroy(own_rng); own_rng = rng_out; } else wotr_random_destroy(rng_out); }
    if (!own) wotr_random_destroy(rng);
    if (!after) { last.note = std::string("apply failed: ") + wotr_snapshot_last_error(); return nullptr; }
    const pyobj::Graph g = awgt::export_graph(after);
    wotr_snapshot_destroy(after);
    pyobj::Obj* w = m.decode(g);
    if (!flag.empty()) m.set_attr(w, flag, m.boolean(false));
    return w;
}

std::pair<std::optional<int64_t>, std::string> Bot::search_and_map(pyobj::Obj* st, const Value& dec, const std::string& label,
                                                                   const std::string& flag, std::optional<double> left) {
    using Matched = awad::Adapter::Matched;
    const std::string kind = at(dec, "kind").s;
    if (!flag.empty()) m.set_attr(st, flag, m.boolean(true));
    auto mn = menu_of(m, st, ad.obs_tables, memo.side);
    if (mn->rows.empty()) return {std::nullopt, "default"};
    std::vector<Matched> mts;   // every row matched to an option (map_action), the rest banned
    for (size_t ri = 0; ri < mn->rowsn.size(); ++ri) {
        Matched mt;
        try { mt = ad.map_action(kind, st, dec, row_of(*mn, ri), memo); } catch (...) { mt = Matched(); }
        if (mt.unported) last.unported = "matching " + kind;
        mts.push_back(std::move(mt));
    }
    std::set<std::string> banned;
    std::vector<size_t> mapped;
    for (size_t ri = 0; ri < mts.size(); ++ri) { if (mts[ri].ok) mapped.push_back(ri); else banned.insert(mn->rows[ri]); }
    if (mapped.empty()) { if (!flag.empty()) m.set_attr(st, flag, m.boolean(false)); return {std::nullopt, "default"}; }
    bool single = true;
    for (size_t q : mapped) single &= mts[q].o == mts[mapped[0]].o && mts[q].o_none == mts[mapped[0]].o_none && !mts[q].has_plan && !mts[q].chain;
    if (single) {
        if (!flag.empty()) m.set_attr(st, flag, m.boolean(false));
        last.engine = mn->rows[mapped[0]];
        return {mts[mapped[0]].o_none ? std::nullopt : std::optional<int64_t>(mts[mapped[0]].o), "single"};
    }
    Matched* chosen = nullptr;
    Matched units_mt;
    void* units_menu = nullptr;
    struct Guard { void* p; ~Guard() { if (p) wotr_snapshot_destroy(p); } } g5{nullptr};
    const void* a_menu = mn->full; int64_t a_index = -1;
    std::string how = "single";
    if (mapped.size() == 1) { chosen = &mts[mapped[0]]; a_index = int64_t(mapped[0]); }
    else {
        bool has_sims = false;
        const std::string pick = choose(*mn, banned, label, left, has_sims, units_menu);
        g5.p = units_menu;
        if (has_sims) how = "search";
        if (!pick.empty()) for (size_t q : mapped) if (mn->rows[q] == pick) { chosen = &mts[q]; a_index = int64_t(q); last.engine = pick; break; }
        if (!chosen && !pick.empty() && units_menu && !banned.count(pick)) {   // matched by itself
            const pyobj::Graph ug = awgt::export_graph(units_menu);
            awgt::G gu{ug};
            const auto parts = gu.kids(gu.kids(ug.root)[0]);
            awad::Adapter::Row row; row.die = ug.values[parts[0]].integer; row.type = gu.text(parts[1]);
            row.params = awgt::to_value(gu, parts[2]); row.menu = units_menu; row.index = 0;
            try { units_mt = ad.map_action(kind, st, dec, row, memo); } catch (...) { units_mt = Matched(); }
            if (units_mt.ok) { chosen = &units_mt; a_menu = units_menu; a_index = 0; last.engine = pick; last.note = "the seat's unit rule"; }
        }
        if (!chosen) { chosen = &mts[mapped[0]]; a_index = int64_t(mapped[0]); last.engine = mn->rows[mapped[0]]; last.note = "fallback: first matched"; }
    }
    const auto o = chosen->option();
    plan = chosen->plan();
    if (std::getenv("AW_BOT_DEBUG")) std::fprintf(stderr, "[bot] %s: %s -> plan %s, chain %d\n", kind.c_str(), last.engine.substr(0, 200).c_str(), plan ? plan->name.c_str() : "none", int(chosen->chain));
    if (chosen->chain) {   // the engine's chain goes on, on a copy of the state (st under the flag: mn->sn)
        work = apply_copy(mn->sn, a_menu, a_index, flag);
        {   // the action's type
            const pyobj::Graph ag = a_menu == mn->full ? pyobj::Graph() : awgt::export_graph(a_menu);
            const pyobj::Graph& gg = a_menu == mn->full ? mn->mg : ag;
            awgt::G ga{gg};
            const auto rr = ga.kids(gg.root);
            work_action = ga.text(ga.kids(rr[size_t(a_menu == mn->full ? a_index : 0)])[1]);
        }
        work_base = st;
        diff.reset();
    }
    if (!flag.empty()) m.set_attr(st, flag, m.boolean(false));
    return {o, how};
}

// _chain: the next link of the engine's chain, on its copy of the state (none: it has no answer)
std::optional<std::pair<std::optional<int64_t>, std::string>> Bot::chain(const Value& dec, std::optional<double> left) {
    using Matched = awad::Adapter::Matched;
    const std::string kind = at(dec, "kind").s;
    std::unique_ptr<Menu> mn;
    try { mn = menu_of(m, work, ad.obs_tables, memo.side); } catch (...) { return std::nullopt; }
    if (mn->rows.empty() || !chain_fits(work, dec, *mn)) return std::nullopt;
    std::vector<Matched> mts;
    for (size_t ri = 0; ri < mn->rowsn.size(); ++ri) {
        Matched mt;
        try { mt = ad.map_action(kind, work, dec, row_of(*mn, ri), memo); } catch (...) { mt = Matched(); }
        if (mt.unported) last.unported = "matching " + kind;
        mts.push_back(std::move(mt));
    }
    std::set<std::string> banned;
    std::vector<size_t> mapped;
    for (size_t ri = 0; ri < mts.size(); ++ri) { if (mts[ri].ok) mapped.push_back(ri); else banned.insert(mn->rows[ri]); }
    if (mapped.empty()) return std::nullopt;
    last.note = "chain";
    size_t at_i = mapped[0];
    if (mapped.size() > 1) {
        bool has_sims = false; void* units_menu = nullptr;
        const std::string pick = choose(*mn, banned, "action", left, has_sims, units_menu);
        if (units_menu) wotr_snapshot_destroy(units_menu);
        bool found = false;
        if (!pick.empty()) for (size_t q : mapped) if (mn->rows[q] == pick) { at_i = q; found = true; break; }
        if (!found) last.note = "chain, fallback: first matched";
    }
    last.engine = mn->rows[at_i];
    Matched& mt = mts[at_i];
    const auto o = mt.option();
    plan = mt.plan();
    work = apply_copy(mn->sn, mn->full, int64_t(at_i), "");
    return std::make_pair(o, std::string("chain"));
}

bool Bot::chain_fits(pyobj::Obj* w, const Value& dec, const Menu& mn) {
    using M = pyobj::Model;
    const std::string kind = at(dec, "kind").s;
    pyobj::Obj** pco_s = M::attr_slot(w, "pending_card_option");
    pyobj::Obj* pco = (pco_s && (*pco_s)->tag != WOTR_NONE) ? *pco_s : nullptr;
    auto card_name = [&]() -> std::string { if (!pco || pco->tag != WOTR_DICT) return "\x01"; pyobj::Obj* n = M::dict_get(pco, "card_name"); return n && n->tag == WOTR_STRING ? n->s : "\x01"; };
    if (kind == "ARMY_MOVE")
        return !is_none(w, "pending_army_move_die") || !is_none(w, "pending_card_followup") ||
               (pco && card_name() != "__companion_moves__" && card_name() != "__nazgul_moves__");
    if (kind == "RECRUIT") return !is_none(w, "pending_muster_die") || pco;
    if (kind == "MOVE_COMPANIONS") return pco && card_name() == "__companion_moves__";
    if (kind == "MOVE_MINIONS") return pco && card_name() == "__nazgul_moves__";
    if (kind == "CARD_CHOICE") return pco || !is_none(w, "pending_card_followup");
    for (auto rn : mn.rowsn) { const std::string ty = row_type(mn.mg, rn); if (ty == "card_option" || ty == "army_move" || ty == "army_attack") return true; }
    return false;
}

bool Bot::chain_over(pyobj::Obj* w, const std::string& kind) {
    using M = pyobj::Model;
    pyobj::Obj** pco_s = M::attr_slot(w, "pending_card_option");
    pyobj::Obj* pco = (pco_s && (*pco_s)->tag != WOTR_NONE) ? *pco_s : nullptr;
    auto card_name = [&]() -> std::string { if (!pco || pco->tag != WOTR_DICT) return "\x01"; pyobj::Obj* n = M::dict_get(pco, "card_name"); return n && n->tag == WOTR_STRING ? n->s : "\x01"; };
    if (kind == "ARMY_MOVE") return is_none(w, "pending_army_move_die") && is_none(w, "pending_card_followup");
    if (kind == "RECRUIT") return is_none(w, "pending_muster_die") && !pco;
    if (kind == "MOVE_COMPANIONS") return !pco || card_name() != "__companion_moves__";
    if (kind == "MOVE_MINIONS") return !pco || card_name() != "__nazgul_moves__";
    return false;
}

// figure_counts(st, names): {(region slug, nation, unit): n} in the map's order, and {(region slug, character)}
void Bot::figure_counts(pyobj::Obj* st, std::vector<std::pair<Key, int64_t>>& out, std::set<std::pair<std::string, std::string>>& chars) {
    using M = pyobj::Model;
    pyobj::Obj* armies = M::attr(st, "armies");
    for (size_t j = 0; j + 1 < armies->k.size(); j += 2) {
        const std::string k = armies->k[j]->s;
        std::string sl = ad.slug(k);
        if (sl.empty()) sl = "\x01";
        for (int side : {0, 1}) {
            const auto f = ad.figures(st, k, side);
            for (const auto& [nu, n] : f.units) {
                const Key key{sl, nu.first, nu.second};
                auto it = std::find_if(out.begin(), out.end(), [&](auto& e) { return e.first == key; });
                if (it == out.end()) out.push_back({key, n}); else it->second += n;
            }
            for (const auto& c : f.chars) chars.insert({sl, c});
        }
    }
}

std::optional<int64_t> Bot::diff_answer(const Value& dec) {
    const std::string kind = at(dec, "kind").s;
    if (!work || !work_base || (kind != "CARD_CHOICE" && kind != "ARMY_SELECTION" && kind != "ARMY_MOVE")) return std::nullopt;
    if (!diff) {
        std::vector<std::pair<Key, int64_t>> b, w;
        std::set<std::pair<std::string, std::string>> bc, wc;
        figure_counts(work_base, b, bc);
        figure_counts(work, w, wc);
        Diff d;
        std::vector<Key> keys;   // set(b) | set(w) (here: b's keys, then w's new ones)
        for (auto& e : b) keys.push_back(e.first);
        for (auto& e : w) if (std::find(keys.begin(), keys.end(), e.first) == keys.end()) keys.push_back(e.first);
        auto get = [](const std::vector<std::pair<Key, int64_t>>& v, const Key& k) { for (auto& e : v) if (e.first == k) return e.second; return int64_t(0); };
        for (const auto& k : keys) {
            const int64_t dd = get(w, k) - get(b, k);
            if (dd > 0) d.added.push_back({k, dd}); else if (dd < 0) d.removed.push_back({k, -dd});
        }
        for (auto& c : wc) if (!bc.count(c)) d.chars_to.insert(c);
        for (auto& c : bc) if (!wc.count(c)) d.chars_from.insert(c);
        diff = d;
    }
    Diff& d = *diff;
    const Value& ctx = or_empty(dec, "context");
    auto any_region = [](const std::vector<std::pair<Key, int64_t>>& v, const std::string& r, bool pos) {
        for (auto& e : v) if (std::get<0>(e.first) == r && (!pos || e.second > 0)) return true; return false; };
    if (kind == "ARMY_MOVE") {   // a card's Army moves the engine made inside the card
        if (work_action != "play_event") return std::nullopt;
        int best_s = -1; std::optional<int64_t> best_o; std::pair<std::string, std::string> best_m;
        for (auto* o : opts(dec, "army_move")) {
            const std::string f = text_of(at(*o, "from")), to = text_of(at(*o, "to"));
            if (d.moves_done.count({f, to})) continue;
            const int s = 2 * any_region(d.removed, f, true) + 2 * any_region(d.added, to, true);
            if (s >= 4 && (best_s < 0 || s > best_s)) { best_s = s; best_o = has_o(o) ? std::optional<int64_t>(at(*o, "o").as_int()) : std::nullopt; best_m = {f, to}; }
        }
        if (best_s >= 0) { d.moves_done.insert(best_m); return best_o; }
        const auto e = opts(dec, "end");
        return (!e.empty() && has_o(e[0])) ? std::optional<int64_t>(at(*e[0], "o").as_int()) : std::nullopt;
    }
    if (kind == "ARMY_SELECTION") {   // a card's Army move: as many as left the region in the engine
        const std::string frm = text_of(at(ctx, "from"));
        const Value* best = nullptr; int64_t bd = 0;
        auto nat_of = [&](const Value& o) { const Value& x = at(o, "nation"); return text_of(x.truthy() ? x : at(ctx, "nation")); };
        auto unit_of = [&](const Value& o) { const Value& x = at(o, "unit"); return text_of(x.truthy() ? x : at(ctx, "unit")); };
        for (auto* o : opts(dec, "count")) {
            int64_t want = 0;
            if (at(*o, "character").truthy()) want = d.chars_from.count({frm, text_of(at(*o, "character"))}) ? 1 : 0;
            else { const Key k{frm, nat_of(*o), unit_of(*o)}; for (auto& e : d.removed) if (e.first == k) want = e.second; }
            const int64_t dd = std::llabs((o->get("count") ? at(*o, "count").as_int() : 0) - want);
            if (!best || dd < bd) { best = o; bd = dd; }
        }
        if (!best) return std::nullopt;
        if (!at(*best, "character").truthy()) {
            const Key k{frm, nat_of(*best), unit_of(*best)};
            for (auto& e : d.removed) if (e.first == k) e.second = std::max<int64_t>(0, e.second - (best->get("count") ? at(*best, "count").as_int() : 0));
        }
        return has_o(best) ? std::optional<int64_t>(at(*best, "o").as_int()) : std::nullopt;
    }
    // a card's question: the words of its question, the answers' words
    std::string q = at(ctx, "question").kind == Value::Str ? at(ctx, "question").s : (at(ctx, "question").is_null() ? "" : "?");
    if (!ctx.get("question")) q = "";
    for (char& ch : q) if (ch == '?') ch = ' ';
    std::vector<std::string> words;
    { size_t i = 0; while (i < q.size()) { while (i < q.size() && std::isspace(static_cast<unsigned char>(q[i]))) ++i; size_t j = i; while (j < q.size() && !std::isspace(static_cast<unsigned char>(q[j]))) ++j; if (j > i) words.push_back(q.substr(i, j - i)); i = j; } }
    std::string region = "\x03", nation = "\x03";   // \x03: None (any)
    for (auto& wd : words) if (t.region_in.count(wd)) { region = wd; break; }
    for (auto& wd : words) if (t.nation_in.count(wd)) { nation = wd; break; }
    static const std::map<std::string, std::string> UNIT_ANSWERS = {{"REGULAR", "regular"}, {"ELITE", "elite"}, {"LEADER", "leader"}, {"NAZGUL", "nazgul"}};
    auto upper = [](std::string x) { for (char& c : x) c = char(std::toupper(static_cast<unsigned char>(c))); return x; };
    auto answer_text = [](const Value& o) { const Value* a = o.get("answer"); return !a ? std::string() : a->kind == Value::Str ? a->s : a->is_null() ? std::string("None") : a->kind == Value::Int ? std::to_string(a->i) : a->kind == Value::Bool ? std::string(a->b ? "True" : "False") : std::string("?"); };
    std::vector<const Value*> unit_opts;
    for (auto* o : opts(dec, "card_choice")) if (UNIT_ANSWERS.count(upper(answer_text(*o)))) unit_opts.push_back(o);
    if (!unit_opts.empty()) {
        for (auto* o : unit_opts) {
            const std::string unit = UNIT_ANSWERS.at(upper(answer_text(*o)));
            for (auto& e : d.added) {
                if (e.second > 0 && std::get<2>(e.first) == unit && (region == "\x03" || region == std::get<0>(e.first)) &&
                    (nation == "\x03" || nation == std::get<1>(e.first))) {
                    --e.second;
                    return has_o(o) ? std::optional<int64_t>(at(*o, "o").as_int()) : std::nullopt;
                }
            }
        }
        return std::nullopt;
    }
    int64_t best_s = 0; std::optional<int64_t> best_o; bool found = false;
    for (auto* o : opts(dec, "card_choice")) {
        int64_t s = 0;
        const Value& rv = at(*o, "region");
        if (!rv.is_null()) {
            const std::string r = text_of(rv);
            for (auto& e : d.added) if (std::get<0>(e.first) == r) s += e.second;
            for (auto& e : d.removed) if (std::get<0>(e.first) == r) s += e.second;
            for (auto& c : d.chars_to) if (c.first == r) s += 1;
        }
        Value fv = at(*o, "from"), tv = at(*o, "to");
        std::vector<std::string> path;
        { const std::string a = answer_text(*o); size_t i = 0; while (true) { size_t j = a.find('>', i); path.push_back(a.substr(i, j == std::string::npos ? std::string::npos : j - i)); if (j == std::string::npos) break; i = j + 1; } }
        if (fv.is_null() && path.size() > 1) { fv = S(path.front()); tv = S(path.back()); }
        if (!fv.is_null() && !tv.is_null()) {
            if (any_region(d.removed, text_of(fv), false)) s += 2;
            if (any_region(d.added, text_of(tv), false)) s += 2;
            for (size_t i = 1; i + 1 < path.size(); ++i) if (any_region(d.added, path[i], false)) s += 1;
        }
        const Value& cv = at(*o, "character");
        if (!cv.is_null()) {
            const std::string c = text_of(cv);
            for (auto& e : d.chars_to) if (e.second == c) s += 2;
            for (auto& e : d.chars_from) if (e.second == c) s += 1;
        }
        if (s > 0 && (!found || s > best_s)) { found = true; best_s = s; best_o = has_o(o) ? std::optional<int64_t>(at(*o, "o").as_int()) : std::nullopt; }
    }
    return found ? best_o : std::nullopt;
}

}  // namespace awbot
