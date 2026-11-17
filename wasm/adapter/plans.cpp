// The plans, RULES and default_option of protocol/wotr_protocol_bot.py (see plans.h), as the Python reads.
#include "plans.h"
#include "opts.h"

#include <algorithm>
#include <cctype>

using awjson::Value;

namespace awad {
using namespace awad::opt;
namespace {
const Value& ctx(const Value& dec) { return or_empty(dec, "context"); }
// o.get(k) or fallback (a falsy value falls back, as Python's `or`)
const Value& or_ctx(const Value& o, const Value& c, const char* k) { const Value& x = at(o, k); return x.truthy() ? x : at(c, k); }
std::string key_text(const Value& v) { return v.kind == Value::Str ? v.s : v.kind == Value::Null ? std::string("\x01none") : v.kind == Value::Int ? std::to_string(v.i) : "\x01?"; }
// int(o.get(k, dflt)) for an int-like field
int64_t int_of(const Value& o, const char* k, int64_t dflt) { const Value* x = o.get(k); return x ? x->as_int() : dflt; }
Answer o_of(const Value* o) { if (!has_o(o)) return std::nullopt; return at(*o, "o").as_int(); }
Answer first_of(const std::vector<const Value*>& c) { return c.empty() ? std::nullopt : o_of(c[0]); }
std::string lower(std::string x) { for (char& c : x) c = char(std::tolower(static_cast<unsigned char>(c))); return x; }
}  // namespace

int region_distance(const Tables& t, const Value& a, const Value& b, int limit) {
    if (equal(a, b) && a.kind == b.kind) return 0;
    auto region = [&](const Value& s) -> std::string {
        if (s.kind != Value::Str) return "";
        auto it = t.region_in.find(s.s); return it == t.region_in.end() ? "" : it->second;
    };
    const std::string ka = region(a), kb = region(b);
    if (ka.empty() || kb.empty()) return limit + 1;
    std::set<std::string> seen{ka};
    std::vector<std::string> frontier{ka};
    for (int d = 1; d <= limit; ++d) {
        std::vector<std::string> nxt;
        for (const auto& r : frontier) {
            auto it = t.adjacent.find(r);
            if (it == t.adjacent.end()) continue;
            for (const auto& n0 : it->second) {
                const std::string n = awad_sanitize(n0);
                if (n == kb) return d;
                if (seen.insert(n).second) nxt.push_back(n);
            }
        }
        frontier = nxt;
    }
    return limit + 1;
}
int region_distance(const Tables& t, const std::string& a, const std::string& b, int limit) { return region_distance(t, S(a), S(b), limit); }

Answer nearest(const Tables& t, const std::vector<const Value*>& options, const Value& to, const std::string& field) {
    if (options.empty()) return std::nullopt;
    const Value* best = nullptr; int bd = 0;
    for (auto* o : options) { const int d = region_distance(t, at(*o, field), to, 12); if (!best || d < bd) { best = o; bd = d; } }
    return o_of(best);
}
Answer nearest(const Tables& t, const std::vector<const Value*>& options, const std::string& to, const std::string& field) {
    return nearest(t, options, S(to), field);
}

// ---- ArmyMovePlan ----
ArmyMovePlan::ArmyMovePlan(const Tables& t_, Value f, Value to_, std::map<std::pair<std::string, std::string>, int64_t> m,
                           std::set<std::string> c, bool all, bool adv)
    : t(t_), frm(std::move(f)), to(std::move(to_)), moved(std::move(m)), chars(std::move(c)), everything(all), advance(adv) {
    name = "army-move";
    kinds = advance ? std::vector<std::string>{"ADVANCE", "ARMY_SELECTION"} : std::vector<std::string>{"ARMY_MOVE", "ARMY_SELECTION"};
}
Answer ArmyMovePlan::answer(const Value& dec) {
    const std::string k = at(dec, "kind").s;
    auto pick_all = [&](const std::vector<const Value*>& cand) {
        for (auto* o : cand) if (at(*o, "all").truthy() == everything) return o_of(o);
        return o_of(cand[0]);
    };
    if (k == "ARMY_MOVE") {
        if (started) return std::nullopt;
        const auto cand = opts(dec, "army_move", {frm.is_null() ? fnone("from") : f("from", frm), to.is_null() ? fnone("to") : f("to", to)});
        if (cand.empty()) {
            const auto nr = opts(dec, "army_move", {frm.is_null() ? fnone("from") : f("from", frm)});
            if (nr.empty()) return std::nullopt;
            started = true; near = true;
            return nearest(t, nr, to);
        }
        started = true;
        return pick_all(cand);
    }
    if (k == "ADVANCE") {
        const auto cand = opts(dec, "advance");
        if (cand.empty()) return std::nullopt;
        started = true;
        return pick_all(cand);
    }
    if (k == "ARMY_SELECTION" && started) {
        const Value& c = ctx(dec);
        auto want_of = [&](const Value& o) -> int64_t {
            if (at(o, "character").truthy()) return chars.count(at(o, "character").s) ? 1 : 0;
            const Value& n = or_ctx(o, c, "nation"); const Value& u = or_ctx(o, c, "unit");
            if (n.kind != Value::Str || u.kind != Value::Str) return 0;
            auto it = moved.find({n.s, u.s}); return it == moved.end() ? 0 : it->second;
        };
        Answer chosen;
        for (auto* o : opts(dec, "count")) if (int_of(*o, "count", -1) == want_of(*o)) chosen = o_of(o);
        if (!chosen) {
            const Value* best = nullptr; int64_t bd = 0;
            for (auto* o : opts(dec, "count")) {
                const int64_t d = std::llabs(int_of(*o, "count", 0) - want_of(*o));
                if (!best || d < bd) { best = o; bd = d; }
            }
            chosen = best ? o_of(best) : std::nullopt;
        }
        return chosen;
    }
    return std::nullopt;
}

// ---- AttackPlan ----
AttackPlan::AttackPlan(Value f, Value to_) : frm(std::move(f)), to(std::move(to_)) { kinds = {"CHOOSE_ATTACK", "ARMY_SELECTION"}; name = "attack"; }
Answer AttackPlan::answer(const Value& dec) {
    if (at(dec, "kind").s == "CHOOSE_ATTACK") {
        const auto cand = opts(dec, "attack", {frm.is_null() ? fnone("from") : f("from", frm), to.is_null() ? fnone("to") : f("to", to)});
        std::vector<const Value*> pick;
        for (auto* o : cand) { const Value* a = o->get("all"); if (!a || a->truthy()) pick.push_back(o); }
        return first_of(pick.empty() ? cand : pick);
    }
    const Value& c = ctx(dec);
    const auto cand = opts(dec, "count");
    if (cand.empty()) return std::nullopt;
    const Value* best = cand[0];
    for (auto* o : cand) if (int_of(*o, "count", 0) > int_of(*best, "count", 0)) best = o;
    const Value* avail = c.get("available");
    if (avail && !avail->is_null())
        for (auto* o : cand) if (int_of(*o, "count", 0) == avail->as_int()) return o_of(o);
    return o_of(best);
}

// ---- RecruitPlan ----
RecruitPlan::RecruitPlan(Value r, Value n, Value u) : region(std::move(r)), nation(std::move(n)), unit(std::move(u)) { kinds = {"RECRUIT"}; name = "recruit"; }
Answer RecruitPlan::answer(const Value& dec) {
    if (done) return std::nullopt;
    auto fv = [](const char* k, const Value& v) { return v.is_null() ? fnone(k) : f(k, v); };
    const auto cand = opts(dec, "recruit", {fv("region", region), fv("nation", nation), fv("unit", unit)});
    if (!cand.empty()) { done = true; return o_of(cand[0]); }
    return std::nullopt;
}

// ---- CompanionMovePlan ----
CompanionMovePlan::CompanionMovePlan(std::vector<std::string> names, Value f_, Value to_) : frm(std::move(f_)), to(std::move(to_)) {
    group.insert(names.begin(), names.end()); kinds = {"MOVE_COMPANIONS", "JOIN_GROUP"}; name = "companions";
}
Answer CompanionMovePlan::answer(const Value& dec) {
    if (at(dec, "kind").s == "MOVE_COMPANIONS") {
        if (started) return std::nullopt;
        for (auto* o : opts(dec, "move_companion")) {
            const Value& ch = at(*o, "character");
            const Value* fr = o->get("from");
            if (ch.kind == Value::Str && group.count(ch.s) && equal(at(*o, "to"), to) && at(*o, "to").kind == to.kind &&
                (fr ? (equal(*fr, frm) && fr->kind == frm.kind) : true)) {
                started = true; return o_of(o);
            }
        }
        return std::nullopt;
    }
    const Value& who = at(ctx(dec), "character");
    auto mine = [&](const std::string& kind) {
        std::vector<const Value*> out;
        for (auto* o : opts(dec, kind)) { const Value* ch = o->get("character"); const Value& c = ch ? *ch : who; if (equal(c, who) && c.kind == who.kind) out.push_back(o); }
        return out;
    };
    const auto join = mine("join"), stay = mine("stay");
    if (who.kind == Value::Str && group.count(who.s)) return !join.empty() ? first_of(join) : first_of(stay);
    return !stay.empty() ? first_of(stay) : first_of(join);
}

// ---- SeparatePlan ----
SeparatePlan::SeparatePlan(std::vector<std::string> names, Value to_) : left(std::move(names)), to(std::move(to_)) { kinds = {"SEPARATE_COMPANIONS"}; name = "separate"; }
Answer SeparatePlan::answer(const Value& dec) {
    for (auto* o : opts(dec, "separate")) {
        const Value& ch = at(*o, "character");
        if (ch.kind != Value::Str) continue;
        auto it = std::find(left.begin(), left.end(), ch.s);
        if (it != left.end()) { left.erase(it); return o_of(o); }
    }
    const auto cand = opts(dec, "separate_to", {to.is_null() ? fnone("to") : f("to", to)});
    if (!cand.empty()) return o_of(cand[0]);
    return std::nullopt;
}

// ---- MinionPlan ----
MinionPlan::MinionPlan(const Tables& t_, std::vector<std::tuple<std::string, Value, Value>> fl) : t(t_), flights(std::move(fl)) { kinds = {"MOVE_MINIONS"}; name = "minions"; }
Answer MinionPlan::answer(const Value& dec) {
    if (flights.empty()) return std::nullopt;
    const auto [fig, frm, to] = flights[0];
    auto fv = [](const char* k, const Value& v) { return v.is_null() ? fnone(k) : f(k, v); };
    std::vector<const Value*> moves, select;
    if (fig == "nazgul") { moves = opts(dec, "move_nazgul", {fv("from", frm)}); select = opts(dec, "select_nazgul", {fv("from", frm)}); }
    else { moves = opts(dec, "move_minion", {f("character", S(fig))}); select = opts(dec, "select_minion", {f("character", S(fig))}); }
    if (!moves.empty()) {
        flights.erase(flights.begin());
        for (auto* o : moves) if (equal(at(*o, "to"), to) && at(*o, "to").kind == to.kind) return o_of(o);
        near = true;
        return nearest(t, moves, to);
    }
    return first_of(select);
}

// ---- casualties ----
std::string casualty_key(const Value& dec) {
    const Value& c = ctx(dec);
    return key_text(at(c, "region")) + "|" + key_text(at(c, "round")) + "|" + key_text(at(c, "attacker")) + "|" + key_text(at(dec, "side"));
}
Answer casualty_pick(const Value& dec, bool reduce_first, CasualtyPlan* plan) {
    const auto red = opts(dec, "casualty", {f("unit", S("elite")), f("effect", S("reduce"))});
    const auto rem_r = opts(dec, "casualty", {f("unit", S("regular"))});
    const auto rem_e = opts(dec, "casualty", {f("unit", S("elite")), f("effect", S("remove"))});
    if (reduce_first && !red.empty()) { if (plan) --plan->k; return o_of(red[0]); }
    if (!rem_r.empty()) return o_of(rem_r[0]);
    const Value* h = ctx(dec).get("hits");
    const int64_t hits = h ? h->as_int() : 1;
    if (!rem_e.empty() && hits >= 2) return o_of(rem_e[0]);
    if (!red.empty()) { if (plan) --plan->k; return o_of(red[0]); }
    if (!rem_e.empty()) return o_of(rem_e[0]);
    return std::nullopt;
}
CasualtyPlan::CasualtyPlan(std::string key_, int64_t k_) : key(std::move(key_)), k(k_) { kinds = {"CASUALTY"}; name = "casualties"; }
Answer CasualtyPlan::answer(const Value& dec) {
    if (casualty_key(dec) != key) return std::nullopt;
    return casualty_pick(dec, k > 0, this);
}

// ---- RetreatPlan, AnswerPlan ----
RetreatPlan::RetreatPlan(std::optional<Value> to_) : to(std::move(to_)) { kinds = {"RETREAT"}; name = "retreat"; }
Answer RetreatPlan::answer(const Value& dec) {
    if (!to) return first_of(opts(dec, "stand"));
    return first_of(opts(dec, "retreat", {to->is_null() ? fnone("region") : f("region", *to)}));
}
AnswerPlan::AnswerPlan(const std::string& kind, std::string okind_) : okind(std::move(okind_)) { kinds = {kind}; name = "answer"; }
Answer AnswerPlan::answer(const Value& dec) { return first_of(opts(dec, okind)); }

// ---- RULES ----
namespace {
Answer rule_card_choice(const Value& dec, const Value& view, int side, bool work) {
    const Value& c = ctx(dec);
    std::vector<const Value*> use;
    for (auto* o : opts(dec, "card_choice")) if (!at(*o, "skip").truthy()) use.push_back(o);
    const Value& cardv = at(c, "card");
    const std::string card = cardv.kind == Value::Str ? cardv.s : (cardv.truthy() ? "?" : "");
    const bool mine = card.rfind(side == 0 ? "fp-" : "sh-", 0) == 0;
    if (mine && !work && !opts(dec, "card_choice", {f("skip", B(true))}).empty()) return first_of(use);
    const Value& qv = at(c, "question");
    const std::string q = lower(qv.kind == Value::Str ? qv.s : "");
    if (c.get("damage") || c.get("tile") || q.find("hunt") != std::string::npos) return first_of(use);
    std::map<std::string, int64_t> answers;
    for (auto* o : opts(dec, "card_choice")) {
        const Value& a = at(*o, "answer");
        const std::string k = lower(a.kind == Value::Str ? a.s : a.is_null() ? "" : std::to_string(a.as_int()));
        if (has_o(o)) answers[k] = at(*o, "o").as_int();   // (dict(): the last one wins)
    }
    const Value& fs = or_empty(view, "fellowship");
    if (answers.count("corruption") && answers.count("eliminate")) {
        int64_t n = 1;
        { size_t i = 0; bool found = false;
          while (i < q.size() && !found) { while (i < q.size() && q[i] == ' ') ++i; size_t j = i; while (j < q.size() && q[j] != ' ') ++j;
            const std::string w = q.substr(i, j - i); if (!w.empty() && std::all_of(w.begin(), w.end(), ::isdigit)) { n = std::stoll(w); found = true; } i = j; } }
        const Value* cor = fs.get("corruption");
        if ((cor ? cor->as_int() : 0) + n >= 10) return answers["eliminate"];
        return answers["corruption"];
    }
    if (answers.count("hand") && answers.count("skip")) {
        const Value& hand = at(or_empty(view, "cards"), "hand");
        return (hand.kind == Value::Arr ? hand.arr.size() : 0) >= 3 ? answers["hand"] : answers["skip"];
    }
    std::vector<const Value*> chars;
    for (auto* o : opts(dec, "card_choice")) if (at(*o, "character").truthy()) chars.push_back(o);
    if (!chars.empty() && q.find("separate") != std::string::npos) {
        static const std::map<std::string, int> level = {{"gandalf-the-grey", 3}, {"strider", 3}, {"boromir", 2}, {"legolas", 2}, {"gimli", 2}, {"meriadoc", 1}, {"peregrin", 1}};
        const Value& guide = at(fs, "guide");
        auto rank = [&](const Value* o) { const std::string ch = at(*o, "character").s; auto it = level.find(ch);
            return std::make_pair(guide.kind == Value::Str && ch == guide.s, it == level.end() ? 2 : it->second); };
        const Value* best = chars[0];
        for (auto* o : chars) if (rank(o) < rank(best)) best = o;
        return o_of(best);
    }
    const auto draws = opts(dec, "draw");
    if (!draws.empty()) {
        const Value& decks = at(or_empty(or_empty(view, "cards"), "decks"), side == 0 ? "FREE_PEOPLES" : "SHADOW");
        const Value* best = draws[0]; int64_t bn = -1;
        for (auto* o : draws) { const Value& d = at(*o, "deck"); const Value* n = d.kind == Value::Str ? decks.get(d.s) : nullptr; const int64_t v = n ? n->as_int() : 0;
            if (bn < 0 || v > bn) { best = o; bn = v; } }
        return o_of(best);
    }
    return std::nullopt;
}
}  // namespace

Answer rule(const Tables& t, const std::string& kind, const Value& dec, const Value& view, int side, bool work, bool* has_rule) {
    if (has_rule) *has_rule = true;
    if (kind == "CASUALTY") return casualty_pick(dec, false);
    if (kind == "CHOOSE_ATTACK") {
        const auto cand = opts(dec, "attack");
        std::set<std::pair<std::string, std::string>> pairs;
        for (auto* o : cand) pairs.insert({key_text(at(*o, "from")), key_text(at(*o, "to"))});
        if (pairs.size() != 1) return std::nullopt;
        std::vector<const Value*> all;
        for (auto* o : cand) { const Value* a = o->get("all"); if (!a || a->truthy()) all.push_back(o); }
        return first_of(all.empty() ? cand : all);
    }
    if (kind == "SEPARATE_COMPANIONS") {
        static const std::set<std::string> FPN = {"DWARVES", "ELVES", "GONDOR", "NORTH", "ROHAN"};
        std::set<std::string> fp;
        for (const auto& kv : t.nation_in) if (FPN.count(kv.first)) fp.insert(kv.first);
        const Value& regions = or_empty(view, "regions");
        const auto cand = opts(dec, "separate_to");
        for (auto* o : cand) {
            const Value& to = at(*o, "to");
            const Value* r = to.kind == Value::Str ? regions.get(to.s) : nullptr;
            for (const Value& u : (r ? or_empty(*r, "units") : NULLV).arr) {
                const Value& n = at(u, "nation"); const Value& un = at(u, "unit");
                if (n.kind == Value::Str && fp.count(n.s) && un.kind == Value::Str && (un.s == "regular" || un.s == "elite")) return o_of(o);
            }
        }
        if (cand.empty()) return std::nullopt;
        return nearest(t, cand, at(or_empty(view, "fellowship"), "region"));
    }
    if (kind == "REMOVE_EXCESS_UNIT") {
        const Answer a = first_of(opts(dec, "remove_unit", {f("unit", S("regular"))}));
        return (a && *a != 0) ? a : first_of(opts(dec, "remove_unit"));   // (`or`: an 'o' of 0 falls through, as in Python)
    }
    if (kind == "CARD_CHOICE") return rule_card_choice(dec, view, side, work);
    if (kind == "MOVE_MINIONS" || kind == "MOVE_COMPANIONS") return first_of(opts(dec, "end"));
    if (kind == "JOIN_GROUP") return first_of(opts(dec, "stay"));
    if (has_rule) *has_rule = false;
    return std::nullopt;
}

std::pair<int64_t, std::string> default_option(const Value& dec) {
    const std::string kind = at(dec, "kind").s;
    const Value& options = or_empty(dec, "options");
    if (options.arr.empty()) return {0, "none"};
    auto pick = [&](std::initializer_list<const char*> ks) -> Answer { for (auto k : ks) { auto c = opts(dec, k); if (!c.empty()) return o_of(c[0]); } return std::nullopt; };
    const int64_t o0 = at(options.arr[0], "o").as_int();
    if (kind == "ACTION") { auto o = pick({"use_die"}); return o ? std::make_pair(*o, std::string("first die action")) : std::make_pair(o0, std::string("first")); }
    if (kind == "REMOVE_EXCESS_UNIT") { auto c = opts(dec, "remove_unit", {f("unit", S("regular"))}); if (!c.empty()) return {*o_of(c[0]), "a Regular first"}; }
    if (kind == "ARMY_MOVE" || kind == "MOVE_COMPANIONS" || kind == "MOVE_MINIONS" || kind == "RECRUIT") { auto o = pick({"end"}); if (o) return {*o, "end"}; }
    if (kind == "ARMY_SELECTION") {
        auto c = opts(dec, "count");
        if (!c.empty()) { const Value* b = c[0]; for (auto* x : c) if (int_of(*x, "count", 0) > int_of(*b, "count", 0)) b = x; return {*o_of(b), "all of them"}; }
    }
    if (kind == "JOIN_GROUP") { auto o = pick({"stay"}); if (o) return {*o, "stay"}; }
    if (kind == "CASUALTY") { auto o = casualty_pick(dec, false); if (o) return {*o, "a Regular first"}; }
    if (kind == "COMBAT_CARD") { auto o = pick({"none"}); if (o) return {*o, "no card"}; }
    if (kind == "CARD_CHOICE") { for (auto* x : opts(dec, "card_choice")) if (!at(*x, "skip").truthy()) return {*o_of(x), "first answer"}; }
    if (kind == "GUIDE_ABILITY") { auto o = pick({"skip"}); if (o) return {*o, "skip"}; }
    return {o0, "option 0"};
}

}  // namespace awad
