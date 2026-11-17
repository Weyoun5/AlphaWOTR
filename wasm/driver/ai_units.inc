// WHICH units an AI seat takes (wotr_bridge/core/ai_units.py pick: _move, _press), on the box's graphs (compiled into
#include <functional>
// driver/box_unity.cpp). The pieces of the engine it reads are their Python's, ported here as they are written:
//   army_move_participants.selected / preexisting / pick, army_card_cargo.decode / key, ai._border_pred9,
//   game_logic.leader_must_follow / _leader_side / _is_saruman, PoliticalTrack.is_at_war (box >= 3),
//   ai._alloc_side_units with the battle's participants (combat_allocation.inc ca_partition).
// The result is the action's new parameter: an army move's unit_set, or a pressed siege's paying nation.
namespace awau {

const std::vector<std::string> SHADOW = {"SAURON", "ISENGARD", "SOUTHRON_EASTERLINGS"};
constexpr int64_t PAIR = 2;
const std::map<std::string, std::vector<std::string>> GIVEN_FIRST = {
    {"REGULAR", {"SOUTHRON_EASTERLINGS", "ISENGARD", "SAURON"}}, {"ELITE", {"SAURON", "SOUTHRON_EASTERLINGS", "ISENGARD"}}};
using Key = std::pair<std::string, std::string>;   // (NATION, KIND)
using Count = std::map<Key, int64_t>;
struct Fail {};

std::string nat(const Snapshot& s, uint64_t u) { return std::string(enum_member(s, obs_attr(s, u, "nation"), "game_state.Nation")); }
std::string kind(const Snapshot& s, uint64_t u) { return std::string(enum_member(s, obs_attr(s, u, "kind"), "game_state.UnitKind")); }
bool shadow_nation(const std::string& n) { return std::find(SHADOW.begin(), SHADOW.end(), n) != SHADOW.end(); }
int nation_side(const std::string& n) { return shadow_nation(n) ? 1 : 0; }
std::string leader_type(const Snapshot& s, uint64_t l) { return std::string(object_type(s, l)); }
std::string leader_name(const Snapshot& s, uint64_t l) {
    const auto n = s.nodes[l].tag == WOTR_OBJECT ? obs_attr(s, l, "name", false) : missing;
    return (n != missing && s.nodes[n].tag == WOTR_STRING) ? std::string(text(s, n)) : "";
}
// game_logic._leader_side
int leader_side(const Snapshot& s, uint64_t l) {
    const auto t = leader_type(s, l);
    if (t == "game_state.Companion") return 0;
    if (t == "game_state.Nazgul" || t == "game_logic.Minion") return 1;
    const auto n = obs_attr(s, l, "nation", false);
    if (n == missing || ctx_none(s, n)) return 0;
    return nation_side(std::string(enum_member(s, n, "game_state.Nation")));
}
// game_logic.leader_must_follow: a plain Leader (not Nazgul, Companion or a named Minion)
bool must_follow(const Snapshot& s, uint64_t l) {
    const auto t = leader_type(s, l);
    if (t == "game_state.Nazgul" || t == "game_state.Companion") return false;
    const auto nm = leader_name(s, l);
    if (nm == "Saruman" || nm == "Witch King" || nm == "Mouth of Sauron") return false;
    return t == "game_state.Leader" || t == "game_logic.Minion";
}
bool saruman(const Snapshot& s, uint64_t l) {
    if (leader_type(s, l) != "game_logic.Minion") return false;
    std::string n = leader_name(s, l); for (char& ch : n) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return n.find("saruman") != std::string::npos;
}
bool at_war(const Snapshot& s, const std::string& nation) {
    const auto boxes = obs_attr(s, obs_attr(s, s.root, "political"), "boxes");
    const auto b = ctx_nation_value(s, boxes, nation);
    return !ctx_none(s, b) && s.nodes[b].integer >= 3;
}
// army_card_cargo.key (the box's cag_key)
std::tuple<std::string, std::string, std::string> cargo_key(const Snapshot& s, uint64_t f, bool unit) { return cag_key(s, f, unit); }

struct Selection { std::vector<uint64_t> units, leaders; };
// army_move_participants.selected(army(units, leaders), side, params, nation_ok)
Selection selected(const Snapshot& s, const std::vector<uint64_t>& army_units, const std::vector<uint64_t>& army_leaders, int side,
                   const Snapshot& ps, uint64_t p, const std::function<bool(const std::string&)>* may,
                   const std::vector<std::tuple<std::string, std::string, int64_t>>* unit_set) {
    auto param = [&](const char* k) -> std::optional<int64_t> {
        const auto v = field(ps, p, k, false);
        if (v == missing || ctx_none(ps, v)) return std::nullopt;
        return ps.nodes[v].integer;
    };
    std::vector<uint64_t> units;
    for (auto u : army_units) if (ctx_unit_side(s, u) == side && (!may || (*may)(nat(s, u)))) units.push_back(u);
    Selection out;
    std::map<std::string, int64_t> chosen_kinds;
    for (const auto& [k, name] : std::vector<std::pair<std::string, const char*>>{{"ELITE", "n_elite"}, {"REGULAR", "n_regular"}}) {
        std::vector<uint64_t> pool;
        for (auto u : units) if (kind(s, u) == k) pool.push_back(u);
        const auto c = param(name);
        const int64_t count = c ? *c : int64_t(pool.size());
        if (count < 0 || count > int64_t(pool.size())) throw Fail{};
        for (int64_t i = 0; i < count; ++i) out.units.push_back(pool[size_t(i)]);
        chosen_kinds[k] += count;
    }
    if (unit_set) {   // pick(units, unit_set, unit=True), first matches in list order
        std::map<std::tuple<std::string, std::string, std::string>, int64_t> due;
        for (const auto& [n, k, c] : *unit_set) due[{"unit", n, k}] += c;
        std::vector<uint64_t> named;
        for (auto u : units) { auto it = due.find(cargo_key(s, u, true)); if (it != due.end() && it->second > 0) { named.push_back(u); --it->second; } }
        for (const auto& [k, v] : due) if (v) throw Fail{};
        std::map<std::string, int64_t> nk;
        for (auto u : named) ++nk[kind(s, u)];
        for (const auto& k : {"ELITE", "REGULAR"}) if (nk[k] != chosen_kinds[k]) throw Fail{};
        out.units = named;
    }
    if (out.units.empty()) throw Fail{};
    std::vector<uint64_t> leaders;
    for (auto l : army_leaders) if (leader_side(s, l) == side && !saruman(s, l) && (!may || !must_follow(s, l) || (*may)(std::string(enum_member(s, obs_attr(s, l, "nation"), "game_state.Nation")))))
        leaders.push_back(l);
    const auto nn = param("n_nazgul");
    const auto nl = param("n_leaders");
    const auto tl = field(ps, p, "take_leaders", false);
    const bool take = tl == missing || ctx_none(ps, tl) ? true : truth(ps, tl);
    const int64_t count = nl ? *nl : (take ? int64_t(leaders.size()) : 0);
    if (count < 0 || (nn && *nn < 0)) throw Fail{};
    std::vector<uint64_t> ordered;
    for (auto l : leaders) if (must_follow(s, l)) ordered.push_back(l);
    for (auto l : leaders) if (!must_follow(s, l)) ordered.push_back(l);
    if (field(ps, p, "leader_set", false) != missing && !ctx_none(ps, field(ps, p, "leader_set", false))) throw Fail{};   // not reached by an army move row here
    if (!nn) { for (int64_t i = 0; i < count && size_t(i) < ordered.size(); ++i) out.leaders.push_back(ordered[size_t(i)]); }
    else {
        int64_t taken = 0;
        for (auto l : leaders) if (leader_type(s, l) == "game_state.Nazgul" && taken < *nn) { out.leaders.push_back(l); ++taken; }
        int64_t others = 0;
        for (auto l : ordered) if (leader_type(s, l) != "game_state.Nazgul" && others < count) { out.leaders.push_back(l); ++others; }
    }
    return out;
}

// give_up(pool, n_regular, n_elite, lost): the units that go, one at a time by THE RULE
Count give_up(Count left, int64_t n_regular, int64_t n_elite, bool lost = true) {
    Count gone;
    for (const auto& [k, n] : std::vector<std::pair<std::string, int64_t>>{{"REGULAR", n_regular}, {"ELITE", n_elite}}) {
        for (int64_t i = 0; i < n; ++i) {
            std::vector<std::string> nations;
            for (const auto& x : GIVEN_FIRST.at(k)) if (left[{x, k}] > 0) nations.push_back(x);
            if (nations.empty()) break;
            int64_t isengard = 0;
            for (const auto& [key, v] : left) if (key.first == "ISENGARD") isengard += v;
            const auto& order = GIVEN_FIRST.at(k);
            auto rank = [&](const std::string& x) {
                return std::make_tuple(lost && x == "ISENGARD" && isengard <= 1,
                                       lost && k == "REGULAR" && x == "SAURON" && left[{"SAURON", "REGULAR"}] <= PAIR,
                                       std::find(order.begin(), order.end(), x) - order.begin());
            };
            std::string pick = nations[0];
            for (const auto& x : nations) if (rank(x) < rank(pick)) pick = x;
            --left[{pick, k}]; ++gone[{pick, k}];
        }
    }
    return gone;
}
Count detached(const Count& pool, int64_t n_regular, int64_t n_elite, bool main_goes) {
    if (!main_goes) return give_up(pool, n_regular, n_elite);
    int64_t regulars = 0, total = 0;
    for (const auto& [k, v] : pool) { total += v; if (k.second == "REGULAR") regulars += v; }
    const Count stay = give_up(pool, regulars - n_regular, total - regulars - n_elite);
    Count out;
    for (const auto& [k, v] : pool) { auto it = stay.find(k); const int64_t d = v - (it == stay.end() ? 0 : it->second); if (d > 0) out[k] = d; }
    return out;
}
Count nonzero(Count c) { for (auto it = c.begin(); it != c.end();) if (it->second == 0) it = c.erase(it); else ++it; return c; }

// _move: -> the unit_set, or empty (none)
std::vector<std::tuple<std::string, std::string, int64_t>> move(const Snapshot& s, const Snapshot& t, const Snapshot& ps, uint64_t row, int side) {
    const auto parts = child_edges(ps, row);
    if (text(ps, parts[1]) != "army_move") return {};
    const auto p = parts[2];
    auto has = [&](const char* k) { const auto v = field(ps, p, k, false); return v != missing && !ctx_none(ps, v); };
    if (has("unit_set") || !has("from") || !has("to") || (!has("n_regular") && !has("n_elite"))) return {};
    const std::string from = sanitize_name(text(ps, field(ps, p, "from"))), to = sanitize_name(text(ps, field(ps, p, "to")));
    const auto entries = child_edges(s, obs_attr(s, s.root, "armies"));
    uint64_t army = missing;
    for (size_t j = 0; j + 1 < entries.size(); j += 2) if (sanitize_name(text(s, entries[j])) == from) { army = entries[j + 1]; break; }
    if (army == missing) return {};
    std::vector<uint64_t> units = child_edges(s, obs_attr(s, army, "units")), leaders = child_edges(s, obs_attr(s, army, "leaders"));
    // ai._border_pred9
    std::function<bool(const std::string&)> may_fn;
    const std::function<bool(const std::string&)>* may = nullptr;
    if (from != to) {
        std::set<std::string> nats;
        for (auto u : units) if (ctx_unit_side(s, u) == side) nats.insert(nat(s, u));
        for (auto l : leaders) if (must_follow(s, l) && leader_side(s, l) == side) nats.insert(std::string(enum_member(s, obs_attr(s, l, "nation"), "game_state.Nation")));
        bool all = true;
        for (const auto& n : nats) all &= at_war(s, n);
        const auto dest = ctx_get(t, field(t, t.root, "region_nations"), to);
        if (!all && !ctx_none(t, dest)) {
            const std::string d(enum_member(t, dest, "game_state.Nation"));
            may_fn = [&s, d](const std::string& n) { return at_war(s, n) || n == d; };
            may = &may_fn;
        }
    }
    // the second leg of an Army die: only the figures present before the first leg (preexisting)
    const auto pdie = obs_attr(s, s.root, "pending_army_move_die", false);
    const auto pfrom = obs_attr(s, s.root, "pending_army_move_from", false);
    if (pdie != missing && !ctx_none(s, pdie) && from == sanitize_name(pfrom != missing && !ctx_none(s, pfrom) ? text(s, pfrom) : "")) {
        const auto snap = obs_attr(s, s.root, "pending_army_move_preexist", false);
        const auto figs = (snap != missing && !ctx_none(s, snap)) ? field(s, snap, "figures", false) : missing;
        auto due = (figs != missing && !ctx_none(s, figs)) ? cag_decode(s, figs) : std::map<ca_CargoKey, int64_t>{};
        std::vector<uint64_t> u2, l2;
        for (auto u : units) { auto it = due.find(cargo_key(s, u, true)); if (it != due.end() && it->second > 0) { u2.push_back(u); --it->second; } }
        for (auto l : leaders) { auto it = due.find(cargo_key(s, l, false)); if (it != due.end() && it->second > 0) { l2.push_back(l); --it->second; } }
        for (const auto& [k, v] : due) if (v) throw Fail{};
        units = u2; leaders = l2;
    }
    Selection listed;
    try { listed = selected(s, units, leaders, side, ps, p, may, nullptr); } catch (const Fail&) { return {}; }
    std::vector<uint64_t> eligible;
    for (auto u : units) if (ctx_unit_side(s, u) == side && (!may || (*may)(nat(s, u)))) eligible.push_back(u);
    int64_t n_r = 0, n_e = 0;
    for (auto u : listed.units) (kind(s, u) == "REGULAR" ? n_r : n_e) += 1;
    if (!(0 < n_r + n_e && n_r + n_e < int64_t(eligible.size()))) return {};
    Count named;
    if (side == 1) {
        Count pool;
        for (auto u : eligible) if (shadow_nation(nat(s, u))) ++pool[{nat(s, u), kind(s, u)}];
        int64_t total = 0; for (const auto& [k, v] : pool) total += v;
        if (total != int64_t(eligible.size())) return {};
        bool any_leader = false;
        for (auto l : leaders) if (leader_side(s, l) == side) any_leader = true;
        const bool main_goes = any_leader ? !listed.leaders.empty() : n_r + n_e > int64_t(eligible.size()) - n_r - n_e;
        named = detached(pool, n_r, n_e, main_goes);
    } else {   // at_war_first: per kind, the nations At War first, the roster's order otherwise
        for (const auto& [k, n] : std::vector<std::pair<std::string, int64_t>>{{"REGULAR", n_r}, {"ELITE", n_e}}) {
            std::vector<uint64_t> of_kind;
            for (auto u : eligible) if (kind(s, u) == k) of_kind.push_back(u);
            std::stable_sort(of_kind.begin(), of_kind.end(), [&](uint64_t a, uint64_t b) { return int(!at_war(s, nat(s, a))) < int(!at_war(s, nat(s, b))); });
            for (int64_t i = 0; i < n && size_t(i) < of_kind.size(); ++i) ++named[{nat(s, of_kind[size_t(i)]), k}];
        }
    }
    Count as_listed;
    for (auto u : listed.units) ++as_listed[{nat(s, u), kind(s, u)}];
    if (nonzero(named) == nonzero(as_listed)) return {};
    std::vector<std::tuple<std::string, std::string, int64_t>> unit_set;
    for (const auto& [k, v] : named) if (v) unit_set.push_back({k.first, k.second, v});
    try { selected(s, units, leaders, side, ps, p, may, &unit_set); } catch (const Fail&) { return {}; }
    return unit_set;
}

// _press: -> the paying nation, or ""
std::string press(const Snapshot& s, const Snapshot& ps, uint64_t row, int side) {
    const auto parts = child_edges(ps, row);
    if (text(ps, parts[1]) != "siege_press" || side != 1) return "";
    const auto p = parts[2];
    auto has = [&](const char* k) { const auto v = ps.nodes[p].tag == WOTR_DICT ? field(ps, p, k, false) : missing; return v != missing && !ctx_none(ps, v); };
    if (has("nation")) return "";
    { const auto v = ps.nodes[p].tag == WOTR_DICT ? field(ps, p, "no_reduce", false) : missing; if (v != missing && truth(ps, v)) return ""; }
    const auto pc = obs_attr(s, s.root, "pending_combat", false);
    if (pc == missing || ctx_none(s, pc)) return "";
    const auto ar = field(s, pc, "attacker_region", false), as = field(s, pc, "attacker_side", false);
    if (ar == missing || !truth(s, ar) || as == missing || ctx_none(s, as) || !enum_is(s, as, "game_state.Side", "SHADOW")) return "";
    const std::string region = sanitize_name(text(s, ar));
    // ai._alloc_side_units(state, region, side, ELITE): the battle's participants (a card casualty's eligible nations
    // do not apply to a pressed siege)
    std::vector<uint64_t> elites;
    for (auto u : ca_units(s, region, side)) if (kind(s, u) == "ELITE" && shadow_nation(nat(s, u))) elites.push_back(u);
    std::set<std::string> nations;
    for (auto u : elites) nations.insert(nat(s, u));
    if (nations.size() < 2) return "";
    Count pool;
    for (auto u : elites) ++pool[{nat(s, u), "ELITE"}];
    const Count g = give_up(pool, 0, 1, false);
    std::string nation;
    for (const auto& [k, v] : g) if (v) { nation = k.first; break; }
    if (nation.empty()) return "";
    uint64_t payer = missing;
    for (auto u : elites) if (nat(s, u) == nation) { payer = u; break; }
    if (payer == elites[0]) return "";
    return nation;
}

}  // namespace awau

namespace awau {
// the menu `menu` row `index` with the parameter set: a one-row menu (native_menu_rows, then the dict's entry put)
std::unique_ptr<Snapshot> with_param(const Snapshot& menu, int64_t index, const std::function<void(Snapshot&, uint64_t)>& put) {
    Snapshot idx; idx.root = 0;
    idx.root = cp_container(idx, WOTR_LIST, {rr_scalar(idx, WOTR_INT, index)});
    auto out = native_menu_rows(menu, idx);
    Snapshot& m = *out;
    const auto row = child_edges(m, m.root)[0];
    auto parts = child_edges(m, row);
    const auto params = rr_scalar(m, WOTR_DICT);
    set_edges(m, params, child_edges(m, parts[2]));
    put(m, params);
    parts[2] = params;
    const auto nrow = cp_container(m, m.nodes[row].tag, parts);
    m.root = cp_container(m, WOTR_LIST, {nrow});
    return ac_compact(m);
}
}  // namespace awau

extern "C" {
// ai_units.pick for the row `index` of `menu`: the action with the units its rule names, as a one-row menu (owned),
// or null when it is played as it came (*changed 0) or on error (*changed -1, wotr_snapshot_last_error)
WOTR_API void* awasm_ai_units(const void* state, const void* tables, const void* menu, int32_t side, int64_t index, int32_t* changed) {
    *changed = -1;
    return checked<void*>([&]() -> void* {
        const Snapshot& s = snapshot(state); const Snapshot& ps = snapshot(menu);
        const auto rows = child_edges(ps, ps.root);
        require(index >= 0 && size_t(index) < rows.size(), "ai_units: the row is not in the menu");
        try {
            const auto us = awau::move(s, snapshot(tables), ps, rows[size_t(index)], side);
            if (!us.empty()) {
                *changed = 1;
                return awau::with_param(ps, index, [&](Snapshot& m, uint64_t params) {
                    std::vector<uint64_t> entries;
                    for (const auto& [n, k, v] : us) {
                        const auto c = rr_scalar(m, WOTR_INT, v);
                        entries.push_back(cp_container(m, WOTR_TUPLE, {rr_string(m, "unit"), rr_string(m, n), rr_string(m, k), c}));
                    }
                    cp_put(m, params, "unit_set", cp_container(m, WOTR_TUPLE, entries));
                }).release();
            }
            const auto n = awau::press(s, ps, rows[size_t(index)], side);
            if (!n.empty()) {
                *changed = 1;
                return awau::with_param(ps, index, [&](Snapshot& m, uint64_t params) { cp_put(m, params, "nation", mu_enum(m, "game_state.Nation", n)); }).release();
            }
        } catch (const awau::Fail&) {}   // (the rule raised: LeagueSearcher logs it and keeps the pick)
        *changed = 0;
        return nullptr;
    }, nullptr);
}

// The engine's action applied with the global Random (wotr_snapshot_apply_descriptor_random), and an army move that
// names its units (the unit rule's unit_set, which the box's executor does not take): the army's roster ordered so
// that the counted move takes the named units (army_move_participants.pick: the first matches in roster order), the
// move applied without unit_set. Python's _execute_action takes the same units.
WOTR_API void* awasm_apply_descriptor_random(const void* state, const void* tables, const void* menu, const void* random, int32_t side,
                                             int64_t index, void** random_out, void** events) {
    std::unique_ptr<Snapshot> copy, one;
    bool named = false;
    {
        const bool ok = checked<bool>([&]() -> bool {
            const Snapshot& ps = snapshot(menu);
            const auto rows = child_edges(ps, ps.root);
            require(index >= 0 && size_t(index) < rows.size(), "apply: the row is not in the menu");
            const auto parts = child_edges(ps, rows[size_t(index)]);
            if (text(ps, parts[1]) != "army_move" || ps.nodes[parts[2]].tag != WOTR_DICT) return true;
            const auto us = field(ps, parts[2], "unit_set", false);
            if (us == missing || ctx_none(ps, us)) return true;
            named = true;
            const Snapshot& s = snapshot(state);
            copy = std::make_unique<Snapshot>(s);
            Snapshot& c = *copy;
            const std::string from = sanitize_name(text(ps, field(ps, parts[2], "from")));
            const auto entries = child_edges(c, obs_attr(c, c.root, "armies"));
            uint64_t army = missing;
            for (size_t j = 0; j + 1 < entries.size(); j += 2) if (sanitize_name(text(c, entries[j])) == from) { army = entries[j + 1]; break; }
            require(army != missing, "apply: the moving army is missing");
            std::map<std::pair<std::string, std::string>, int64_t> due;
            for (auto e : child_edges(ps, us)) { const auto k = child_edges(ps, e); due[{std::string(text(ps, k[1])), std::string(text(ps, k[2]))}] += ps.nodes[k[3]].integer; }
            const auto list = obs_attr(c, army, "units");
            const auto units = child_edges(c, list);
            std::vector<uint64_t> first, rest;
            for (auto u : units) { auto it = due.find({awau::nat(c, u), awau::kind(c, u)}); if (it != due.end() && it->second > 0) { --it->second; first.push_back(u); } else rest.push_back(u); }
            for (const auto& [k, v] : due) require(v == 0, "A chosen army figure is not available to move");
            first.insert(first.end(), rest.begin(), rest.end());
            const auto& node = c.nodes[list];
            for (size_t j = 0; j < first.size(); ++j) c.edges[node.offset + j] = first[j];
            one = awau::with_param(ps, index, [&](Snapshot& m, uint64_t params) {
                auto kv = child_edges(m, params);
                std::vector<uint64_t> keep;
                for (size_t j = 0; j + 1 < kv.size(); j += 2) if (!text_is(m, kv[j], "unit_set")) { keep.push_back(kv[j]); keep.push_back(kv[j + 1]); }
                set_edges(m, params, keep);
            });
            return true;
        }, false);
        if (!ok) return nullptr;
    }
    if (!named) return wotr_snapshot_apply_descriptor_random(state, tables, menu, random, side, index, random_out, events);
    return wotr_snapshot_apply_descriptor_random(copy.get(), tables, one.get(), random, side, 0, random_out, events);
}
}
