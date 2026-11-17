// The option matching: protocol/wotr_protocol_bot.py map_action and the MAPPERS (the adapter's lines 2272-2936),
// for the questions the adapter searches. An engine row (die, action type, parameters) of the box's menu is matched
// to the referee's option; the result says the option, whether a plan follows (Match.has_plan) and whether the
// engine's chain goes on (Match.chain). Not ported yet: m_advance and the move / attack answers of m_card_choice
// (their plans apply the action on a copy of the state first): they return `unported`.
#include "adapter.h"
#include "wotr_random.h"

#include <algorithm>
#include <cctype>
#include <set>

using pyobj::Obj;
using awjson::Value;

namespace awad {
namespace {
const Value NULLV;
const Value& at(const Value& v, const std::string& k) { const Value* x = v.get(k); return x ? *x : NULLV; }
const Value& or_empty(const Value& v, const std::string& k) { const Value& x = at(v, k); return x.truthy() ? x : NULLV; }
Value S(const std::string& s) { Value v; v.kind = Value::Str; v.s = s; return v; }
Value I(int64_t i) { Value v; v.kind = Value::Int; v.i = i; return v; }
Value B(bool b) { Value v; v.kind = Value::Bool; v.b = b; return v; }
// a field value as Python compares o.get(k) != v: None for a missing slug
struct F { std::string k; bool none; Value v; };
F f(const std::string& k, const Value& v) { return {k, false, v}; }
F fnone(const std::string& k) { return {k, true, Value()}; }
F fstr(const std::string& k, const std::string& s, bool present) { return present ? f(k, S(s)) : fnone(k); }
bool equal(const Value& a, const Value& b) {
    if ((a.kind == Value::Int || a.kind == Value::Bool) && (b.kind == Value::Int || b.kind == Value::Bool))
        return (a.kind == Value::Int ? a.i : a.b) == (b.kind == Value::Int ? b.i : b.b);
    if (a.kind != b.kind) return false;
    if (a.kind == Value::Str) return a.s == b.s;
    if (a.kind == Value::Null) return true;
    if (a.kind == Value::Real) return a.d == b.d;
    return false;
}
std::vector<const Value*> opts(const Value& dec, const std::string& kind, const std::vector<F>& fields = {}) {
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
bool has_o(const Value* o) { return o && at(*o, "o").kind != Value::Null; }
std::string plain(const std::string& x) { std::string o; for (unsigned char c : x) if (std::isalnum(c) && c < 128) o += char(std::tolower(c)); return o; }
std::string lower(std::string x) { for (char& c : x) c = char(std::tolower(static_cast<unsigned char>(c))); return x; }
const std::map<std::string, std::vector<std::string>> DIE_ACTIONS = {
    {"draw_card", {"draw_card"}}, {"play_event", {"play_card"}}, {"army_attack", {"attack", "attack_with_leader"}},
    {"army_move", {"move_armies", "move_army_with_leader"}}, {"advance_political", {"diplomacy"}},
    {"muster_recruit", {"recruit"}}, {"muster_elite", {"recruit"}}, {"muster_minion", {"play_minion"}},
    {"transform_gandalf", {"play_companion"}}, {"crown_aragorn", {"play_companion"}},
    {"fellowship_move", {"move_fellowship"}}, {"fellowship_hide", {"hide_fellowship", "hide_fellowship_strider"}},
    {"companion_move", {"move_companions"}}, {"companion_separate", {"separate_companions"}},
    {"nazgul_move", {"move_minions"}}, {"mouth_move", {"move_minions"}}, {"discard_table_card", {"force_discard"}}};
}  // namespace

std::string Adapter::slug(const std::string& key) const {
    const std::string san = awad_sanitize(key);
    for (const auto& kv : t.region_in) if (kv.second == san) return kv.first;
    return "";
}

Adapter::Matched Adapter::map_action(const std::string& kind, Obj* st, const Value& dec, const Row& a, const Memo& memo) {
    using M = pyobj::Model;
    Matched r;
    const Value& p = a.params;
    auto pstr = [&](const char* k) -> std::string { return pstr_of(p, k); };
    auto pslug = [&](const char* k) { const std::string s = pstr(k); return s.empty() ? std::string() : slug(s); };
    auto first_of = [&](const std::vector<const Value*>& c) { if (!c.empty() && has_o(c[0])) { r.ok = true; r.o = at(*c[0], "o").as_int(); } };
    // _m(cand, plan, chain)
    auto mm = [&](const std::vector<const Value*>& c, bool plan = false, bool chain = false) {
        if (c.empty()) return r;
        r.ok = true; r.o_none = !has_o(c[0]); if (!r.o_none) r.o = at(*c[0], "o").as_int();
        r.has_plan = plan; r.chain = chain; return r;
    };
    auto die_options = [&](const std::vector<std::string>& actions, const std::vector<F>& fields = {}) {
        std::vector<const Value*> c;
        for (const auto& name : actions) {
            std::vector<F> fl = {f("die", I(a.die)), f("action", S(name))};
            fl.insert(fl.end(), fields.begin(), fields.end());
            for (auto* o : opts(dec, "use_die", fl)) c.push_back(o);
        }
        return c;
    };
    auto char_out = [&](const std::string& name) -> std::string { for (const auto& kv : t.char_in) if (kv.second == name) return kv.first; return ""; };
    auto card_key = [&](const std::string& cid) -> std::string { auto it = t.card_out.find(cid); return it == t.card_out.end() ? "" : it->second; };
    const std::string& ty = a.type;

    if (kind == "ACTION") {                                   // m_action
        if (ty == "pass") { first_of(opts(dec, "pass")); return r; }
        if (ty == "elven_ring") {
            const std::string face = t.face_by_value.count(pstr("new_face")) ? t.face_by_value.at(pstr("new_face")) : "";
            std::string to;
            for (const auto& kv : t.face_in) if (kv.second == face) to = kv.first;
            first_of(opts(dec, "elven_ring", {f("die", at(p, "target_die_idx")), fstr("to", to, !to.empty())}));
            return r;
        }
        auto da = DIE_ACTIONS.find(ty);
        if (da == DIE_ACTIONS.end()) return r;
        std::vector<std::string> acts = da->second;
        if (ty == "draw_card") return mm(die_options(acts, {f("deck", at(p, "deck"))}));
        if (ty == "play_event") { const std::string k = card_key(pstr("card_id")); return mm(die_options(acts, {fstr("card", k, !k.empty())}), false, true); }
        if (ty == "advance_political") {
            auto ni = t.nation_out.find(pstr("nation"));
            const F nat = ni == t.nation_out.end() ? fnone("nation") : f("nation", S(ni->second));
            std::vector<const Value*> c;
            if (at(p, "companion_ability").truthy()) {
                const std::string ck = char_out(pstr("companion_ability"));
                c = die_options({"companion_politics"}, {fstr("character", ck, !ck.empty())});
                if (c.empty()) c = die_options({"diplomacy"}, {nat});
            } else c = die_options({"diplomacy"}, {nat});
            return mm(c);
        }
        if (ty == "muster_minion") {
            const std::string ck = char_out(pstr("minion")), rg = pslug("region");
            return mm(die_options(acts, {fstr("character", ck, !ck.empty()), fstr("region", rg, !rg.empty())}));
        }
        if (ty == "transform_gandalf") { const std::string rg = pslug("region"); return mm(die_options(acts, {f("character", S("gandalf-the-white")), fstr("region", rg, !rg.empty())})); }
        if (ty == "crown_aragorn") return mm(die_options(acts, {f("character", S("aragorn"))}));
        if (ty == "fellowship_move" || ty == "fellowship_hide") {
            if (ty == "fellowship_hide" && at(p, "strider_ability").truthy()) acts = {"hide_fellowship_strider", "hide_fellowship"};
            return mm(die_options(acts));
        }
        if (ty == "discard_table_card") {
            const std::string k = card_key(pstr("card_id"));
            const auto cand = die_options(acts, {fstr("card", k, !k.empty())});
            const bool ring = pstr("payment") == "ring";
            std::vector<const Value*> pick;
            for (auto* o : cand) if (at(*o, "elven_ring").truthy() == ring) pick.push_back(o);
            return mm(pick.empty() ? cand : pick);
        }
        // the die's face: st.dice[me][die_idx].current
        std::string face;
        Obj* dice = M::dict_get(M::attr(st, "dice"), this->m.enumv("game_state.Side", memo.side ? "SHADOW" : "FREE_PEOPLES"));
        if (dice && a.die >= 0 && size_t(a.die) < dice->k.size()) {
            Obj* cur = M::attr(dice->k[size_t(a.die)], "current");
            if (cur->tag == WOTR_ENUM) face = cur->k[1]->s;
        }
        auto ordered = [&]() {
            std::vector<std::string> order = acts;
            if (face == "CHARACTER") std::reverse(order.begin(), order.end());
            auto c = die_options({order[0]});
            if (c.empty()) c = die_options(std::vector<std::string>(order.begin() + 1, order.end()));
            return c;
        };
        if (ty == "army_move") { const auto c = ordered(); if (c.empty()) return r; return mm(c, true, true); }   // the plan: a function
        if (ty == "army_attack") { const auto c = ordered(); if (c.empty()) return r; return mm(c, true, false); }
        if (ty == "muster_recruit" || ty == "muster_elite" || ty == "companion_move" || ty == "nazgul_move" || ty == "mouth_move") {
            const auto c = die_options(acts); if (c.empty()) return r; return mm(c, true, true);
        }
        if (ty == "companion_separate") { const auto c = die_options(acts); if (c.empty()) return r; return mm(c, true, false); }
        return r;
    }
    if (kind == "FELLOWSHIP_PHASE") {
        if (ty == "fellowship_declare") { const std::string rg = pslug("region"); return mm(opts(dec, "declare", {fstr("region", rg, !rg.empty())})); }
        if (ty == "pass") return mm(opts(dec, "done"));
        return r;
    }
    if (kind == "HUNT_ALLOCATION") {
        if (ty == "eye_allocate") return mm(opts(dec, "hunt_dice", {f("count", I(at(p, "count").kind == Value::Null ? 0 : at(p, "count").as_int()))}));
        return r;
    }
    if (kind == "DISCARD_EXCESS_CARD") {
        if (ty == "discard_card") { const std::string k = card_key(pstr("card_id")); return mm(opts(dec, "discard", {fstr("card", k, !k.empty())})); }
        return r;
    }
    if (kind == "COMBAT_CARD") {
        if (ty == "combat_card_play") { const std::string k = card_key(pstr("card_id")); return mm(opts(dec, "combat_card", {fstr("card", k, !k.empty())})); }
        if (ty == "combat_card_none") return mm(opts(dec, "none"));
        return r;
    }
    if (kind == "CASUALTY") {                                 // m_casualty: CasualtyPlan + casualty_pick
        if (ty != "casualty_allocate") return r;
        const int64_t k = at(p, "count").kind == Value::Null ? 0 : at(p, "count").as_int();
        const auto red = opts(dec, "casualty", {f("unit", S("elite")), f("effect", S("reduce"))});
        const auto rem_r = opts(dec, "casualty", {f("unit", S("regular"))});
        const auto rem_e = opts(dec, "casualty", {f("unit", S("elite")), f("effect", S("remove"))});
        const Value& ctx = or_empty(dec, "context");
        const int64_t hits = ctx.get("hits") ? at(ctx, "hits").as_int() : 1;
        const Value* pick = nullptr;
        if (k > 0 && !red.empty()) pick = red[0];
        else if (!rem_r.empty()) pick = rem_r[0];
        else if (!rem_e.empty() && hits >= 2) pick = rem_e[0];
        else if (!red.empty()) pick = red[0];
        else if (!rem_e.empty()) pick = rem_e[0];
        if (!has_o(pick)) return r;
        r.ok = true; r.o = at(*pick, "o").as_int(); r.has_plan = true; return r;
    }
    if (kind == "CEASE_ATTACK") {
        if (ty == "combat_continue" || ty == "siege_press") return mm(opts(dec, "continue"));
        if (ty == "combat_cease" || ty == "siege_no_press") return mm(opts(dec, "cease"));
        return r;
    }
    if (kind == "EXTEND_SIEGE") {
        if (ty == "siege_press") return mm(opts(dec, "extend"));
        if (ty == "siege_no_press") return mm(opts(dec, "stop"));
        return r;
    }
    if (kind == "RETREAT") {
        if (ty == "combat_retreat") { const std::string rg = pslug("to_region"); return mm(opts(dec, "retreat", {fstr("region", rg, !rg.empty())})); }
        if (ty == "combat_no_retreat") return mm(opts(dec, "stand"));
        return r;
    }
    if (kind == "FIELD_OR_SIEGE") {
        if (ty == "retreat_into_siege") return mm(opts(dec, "siege"));
        if (ty == "combat_no_retreat" || ty == "combat_retreat") return mm(opts(dec, "field"), true);
        return r;
    }
    if (kind == "WHITE_RIDER") {
        if (ty == "shine") return mm(opts(dec, "use"));
        if (ty == "shine_none") return mm(opts(dec, "skip"));
        return r;
    }
    if (kind == "HUNT_CASUALTY") {
        if (ty == "hunt_take_corruption") return mm(opts(dec, "corruption"));
        if (ty == "hunt_sacrifice_guide") return mm(opts(dec, "guide"));
        if (ty == "hunt_sacrifice_random") return mm(opts(dec, "random"));
        return r;
    }
    if (kind == "GUIDE_ABILITY") {
        if (ty == "hunt_gollum_reveal" || ty == "hunt_guide_separate") return mm(opts(dec, "use"));
        if (ty == "hunt_gollum_no_reveal") return mm(opts(dec, "skip"));
        if (ty == "hunt_take_corruption" || ty == "hunt_sacrifice_guide" || ty == "hunt_sacrifice_random") return mm(opts(dec, "skip"), true);
        return r;
    }
    if (kind == "REVEAL_FELLOWSHIP") {
        if (ty == "fellowship_reveal_place") { const std::string rg = pslug("region"); return mm(opts(dec, "reveal", {fstr("region", rg, !rg.empty())})); }
        return r;
    }
    if (kind == "CHOOSE_GUIDE") {
        if (ty == "choose_guide") { const std::string ck = char_out(pstr("companion")); return mm(opts(dec, "guide", {fstr("character", ck, !ck.empty())})); }
        return r;
    }
    if (kind == "ADVANCE") {
        if (ty == "victor_stay") return mm(opts(dec, "stay"));
        if (ty != "victor_advance") return r;
        const MovePlan plan = army_move_plan(st, a, memo);    // m_advance: ArmyMovePlan(advance=True).answer(dec)
        if (!plan.ok) return r;
        const auto cand = opts(dec, "advance");
        if (cand.empty()) return r;
        std::vector<const Value*> pick;
        for (auto* o : cand) if (at(*o, "all").truthy() == plan.everything) pick.push_back(o);
        const Value* o = pick.empty() ? cand[0] : pick[0];
        if (!has_o(o)) return r;
        r.ok = true; r.o = at(*o, "o").as_int(); r.has_plan = true; return r;
    }
    if (kind == "CARD_CHOICE") {                              // m_card_choice
        if (ty == "card_option") {
            const Value& ch = or_empty(p, "choices");
            if (ch.get("palantir_deck")) return mm(opts(dec, "draw", {f("deck", at(ch, "palantir_deck"))}));
            if (ch.get("use_hunt_reduction")) {
                if (at(ch, "use_hunt_reduction").truthy()) {
                    std::vector<const Value*> c;
                    for (auto* o : opts(dec, "card_choice")) {
                        const Value& tc = at(*o, "target_card");
                        const std::string cid = tc.kind == Value::Str ? t.card(tc.s) : "";
                        if (!cid.empty() ? cid == pstr_of(ch, "card_id") : at(ch, "card_id").kind == Value::Null) c.push_back(o);
                    }
                    return mm(c);
                }
                return mm(opts(dec, "card_choice", {f("skip", B(true))}));
            }
            if (at(ch, "_comp_done").truthy() || at(ch, "_naz_done").truthy()) {
                auto e = opts(dec, "end");
                const Value* o = (!e.empty() && has_o(e[0])) ? e[0] : nullptr;
                if (!o) { auto s = opts(dec, "card_choice", {f("skip", B(true))}); if (!s.empty() && has_o(s[0])) o = s[0]; }
                if (!o) return r;
                r.ok = true; r.o = at(*o, "o").as_int(); r.chain = true; return r;
            }
            std::set<std::pair<std::string, std::string>> toks;
            choice_tokens(ch, toks);
            const int64_t best = score_options(dec, toks, !ch.truthy());
            if (best < 0) return r;
            r.ok = true; r.o = best; r.chain = true; return r;
        }
        if (ty == "followup_done") {
            auto s = opts(dec, "card_choice", {f("skip", B(true))});
            const Value* o = (!s.empty() && has_o(s[0])) ? s[0] : nullptr;
            if (!o) { auto e = opts(dec, "end"); if (!e.empty() && has_o(e[0])) o = e[0]; }
            if (!o) return r;
            r.ok = true; r.o = at(*o, "o").as_int(); r.chain = true; return r;
        }
        if (ty == "combat_effect_choice") return mm(opts(dec, "card_choice", {f("count", I(at(p, "count").kind == Value::Null ? 0 : at(p, "count").as_int()))}));
        if (ty == "combat_retreat") { const std::string rg = pslug("to_region"); return mm(opts(dec, "card_choice", {fstr("region", rg, !rg.empty())})); }
        if (ty == "army_move" || ty == "army_attack") {
            const std::string fr = pslug("from"), to = pslug("to");
            auto c = opts(dec, "card_choice", {fstr("from", fr, !fr.empty()), fstr("to", to, !to.empty())});
            if (!c.empty()) { r.ok = true; r.o_none = !has_o(c[0]); if (!r.o_none) r.o = at(*c[0], "o").as_int(); r.chain = true; return r; }
            const std::string word = ty == "army_move" ? "move" : "attack";
            for (auto* o : opts(dec, "card_choice")) if (lower(at(*o, "answer").kind == Value::Str ? at(*o, "answer").s : "") == word) {
                if (ty == "army_move") {   // Match(o, army_move_plan(...), chain=True): a plan unless the move fails
                    r.ok = true; r.o = at(*o, "o").as_int(); r.has_plan = army_move_plan(st, a, memo).ok; r.chain = true; return r;
                }
                r.ok = true; r.o = at(*o, "o").as_int(); r.has_plan = true; r.chain = true; return r;
            }
            return r;
        }
        return r;
    }
    r.unported = !(kind == "ACTION");
    return r;
}

// a parameter's text (an Enum's member name)
// figures(st, key, side): {(nation, unit): n} and the Character keys of `side` in a region
Adapter::Figures Adapter::figures(Obj* st, const std::string& key, int side) {
    using M = pyobj::Model;
    Figures out;
    Obj* a = army_at(st, key);
    if (!a) return out;
    static const std::set<std::string> FP = {"ELVES", "DWARVES", "GONDOR", "THE_NORTH", "ROHAN"};
    for (Obj* u : M::attr(a, "units")->k) {
        const std::string nat = M::attr(u, "nation")->k[1]->s;
        if ((FP.count(nat) ? 0 : 1) != side) continue;
        ++out.units[{t.nation_out.at(nat), M::attr(u, "kind")->k[1]->s == "REGULAR" ? "regular" : "elite"}];
    }
    for (Obj* ld : M::attr(a, "leaders")->k) {
        const std::string tn = M::type(ld);
        if (tn == "game_state.Nazgul") { if (side == 1) ++out.units[{"SAURON", "nazgul"}]; }
        else if (tn == "game_state.Leader") {
            const std::string nat = M::attr(ld, "nation")->k[1]->s;
            if ((FP.count(nat) ? 0 : 1) == side) ++out.units[{t.nation_out.at(nat), "leader"}];
        } else {
            Obj** n = M::attr_slot(ld, "name");
            if (!n || (*n)->tag != WOTR_STRING) continue;
            std::string ck;
            for (const auto& kv : t.char_in) if (kv.second == (*n)->s) ck = kv.first;
            if (ck.empty()) continue;
            const bool minion = std::find(t.minion_keys.begin(), t.minion_keys.end(), ck) != t.minion_keys.end();
            if (minion == (side == 1)) out.chars.insert(ck);
        }
    }
    return out;
}

// army_move_plan: the action applied on a copy (the box's apply_native_descriptor, as ai._execute_action), the
// figures that left its region, whether that was everything that can move (Saruman never moves)
Adapter::MovePlan Adapter::army_move_plan(Obj* st, const Row& a, const Memo& memo) {
    MovePlan plan;
    const std::string frm = pstr_of(a.params, "from");
    const pyobj::Graph g = m.encode(st);
    void* sn = wotr_snapshot_create(g.values.data(), g.values.size(), g.edges.data(), g.edges.size(), g.bytes.data(), g.bytes.size(), g.root);
    if (!sn) return plan;
    // apply_native_random_descriptor (the native _execute_action) with a stream of its own: the actions planned
    // here (army moves, victor advances) draw no random number
    const uint32_t limb = 1;
    void* rng = wotr_random_create_seeded(&limb, 1);
    void* rng_out = nullptr; void* events = nullptr;
    void* after = (a.menu && obs_tables && rng) ? wotr_snapshot_apply_descriptor_random(sn, obs_tables, a.menu, rng, memo.side, a.index, &rng_out, &events) : nullptr;
    if (rng_out) wotr_random_destroy(rng_out);
    if (events) wotr_snapshot_destroy(events);
    if (rng) wotr_random_destroy(rng);
    wotr_snapshot_destroy(sn);
    if (!after) return plan;                                  // ai._execute_action raised: no plan
    pyobj::Graph ag;
    const size_t nv = wotr_snapshot_node_count(after), ne = wotr_snapshot_edge_count(after), nb = wotr_snapshot_byte_count(after);
    ag.values.resize(nv); ag.edges.resize(ne); ag.bytes.resize(nb); ag.root = wotr_snapshot_root(after);
    wotr_snapshot_export(after, ag.values.data(), nv, ag.edges.data(), ne, ag.bytes.data(), nb);
    wotr_snapshot_destroy(after);
    Obj* w = m.decode(ag);
    const Figures before = figures(st, frm, memo.side), now = figures(w, frm, memo.side);
    std::map<std::pair<std::string, std::string>, int64_t> moved;
    for (const auto& [k, n] : before.units) { auto it = now.units.find(k); const int64_t d = n - (it == now.units.end() ? 0 : it->second); if (d > 0) moved[k] = d; }
    std::set<std::string> chars;
    for (const auto& c : before.chars) if (!now.chars.count(c)) chars.insert(c);
    if (moved.empty() && chars.empty()) return plan;
    std::set<std::string> movable = before.chars;
    movable.erase("saruman");
    plan.ok = true;
    plan.everything = moved == before.units && chars == movable;
    return plan;
}

std::string Adapter::pstr_of(const Value& v, const char* k) {
    const Value& x = at(v, k);
    if (x.kind != Value::Str) return "";
    return x.s.rfind("\x01" "enum:", 0) == 0 ? x.s.substr(6) : x.s;
}

// choice_tokens: the protocol names a card option of the engine mentions
void Adapter::choice_tokens(const Value& v, std::set<std::pair<std::string, std::string>>& out) {
    if (v.kind == Value::Obj) { for (const auto& kv : v.obj) if (kv.first.rfind("_", 0) != 0) choice_tokens(kv.second, out); return; }
    if (v.kind == Value::Arr) { for (const auto& x : v.arr) choice_tokens(x, out); return; }
    if (v.kind == Value::Bool) { out.insert({"bool", v.b ? "1" : "0"}); return; }
    if (v.kind == Value::Int) { out.insert({"int", std::to_string(v.i)}); return; }
    if (v.kind == Value::Str && v.s.rfind("\x01" "enum:", 0) == 0) {          // an Enum: its protocol nation, if any
        const std::string member = v.s.substr(6);
        auto it = t.nation_out.find(member);
        if (it != t.nation_out.end()) out.insert({"nation", it->second});
        return;
    }
    const std::string s = v.kind == Value::Str ? v.s : (v.kind == Value::Null ? "None" : (v.kind == Value::Real ? std::to_string(v.d) : ""));
    const std::string sl = slug(s);
    if (!sl.empty()) out.insert({"region", sl});
    for (const auto& kv : t.char_in) if (kv.second == s) out.insert({"character", kv.first});
    auto ck = t.card_out.find(s);
    if (ck != t.card_out.end()) out.insert({"card", ck->second});
    for (const auto& kv : t.nation_in) {
        const std::string ps = plain(s);
        if (ps == plain(kv.first) || ps == plain(t.nation_value.count(kv.second) ? t.nation_value.at(kv.second) : "") || ps == plain(kv.second))
            out.insert({"nation", kv.first});
    }
    static const std::set<std::string> DECLINE = {"none", "skip", "decline", "no", "pass", "done", "stay"};
    if (DECLINE.count(lower(s))) out.insert({"decline", "1"});
    out.insert({"word", plain(s)});
}

// score_options: the referee's card answer that best fits the engine's choice (-1: none)
int64_t Adapter::score_options(const Value& dec, const std::set<std::pair<std::string, std::string>>& toks, bool empty) {
    bool have = false; int64_t best_s = 0, best_o = -1;
    const bool decline = toks.count({"decline", "1"}) || empty;
    for (const Value& o : or_empty(dec, "options").arr) {
        int64_t s = 0;
        for (const auto& [field, tk] : std::vector<std::pair<std::string, std::string>>{
                 {"region", "region"}, {"from", "region"}, {"to", "region"}, {"character", "character"}, {"nation", "nation"}, {"target_card", "card"}}) {
            const Value& x = at(o, field);
            if (x.kind != Value::Null) s += toks.count({tk, x.kind == Value::Str ? x.s : std::to_string(x.i)}) ? 3 : -1;
        }
        if (at(o, "count").kind != Value::Null) s += toks.count({"int", std::to_string(at(o, "count").as_int())}) ? 3 : -1;
        if (o.get("yes")) s += (at(o, "yes").truthy() != decline) ? 1 : -1;
        if (at(o, "skip").truthy()) s += decline ? 2 : -2;
        std::string ans = at(o, "answer").kind == Value::Str ? at(o, "answer").s : "";
        for (char& c : ans) if (c == '>' || c == ':') c = ' ';
        size_t i = 0;
        while (i < ans.size()) {
            while (i < ans.size() && std::isspace(static_cast<unsigned char>(ans[i]))) ++i;
            size_t j = i;
            while (j < ans.size() && !std::isspace(static_cast<unsigned char>(ans[j]))) ++j;
            if (j > i) {
                const std::string part = ans.substr(i, j - i);
                if (toks.count({"region", part}) || toks.count({"card", part})) s += 1;
                if (toks.count({"word", plain(part)})) s += 1;
            }
            i = j;
        }
        if (!have || s > best_s) { have = true; best_s = s; best_o = at(o, "o").kind == Value::Null ? -1 : at(o, "o").as_int(); }
    }
    if (!have || best_s <= 0) return -1;
    return best_o;
}

}  // namespace awad
