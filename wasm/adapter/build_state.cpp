// protocol/wotr_protocol_bot.py build_state and _build_cards (lines 460-690), line for line, on the object model.
// Every object the Python constructs comes from a prototype its constructor made (record_build_state.py), so the
// attributes, their order and their defaults are the engine's own.
#include "adapter.h"
#include "wotr_random.h"

#include <algorithm>
#include <cctype>
#include <tuple>

using pyobj::Obj;
using awjson::Value;

namespace awad {
namespace {
const std::set<std::string> FP_NATIONS = {"ELVES", "DWARVES", "GONDOR", "THE_NORTH", "ROHAN"};
const Value NULL_VALUE;
const Value& at(const Value& v, const std::string& k) { const Value* x = v.get(k); return x ? *x : NULL_VALUE; }
// v.get(k) or {} / [] : the value when truthy, else an empty one of no kind (iterations see nothing)
const Value& or_empty(const Value& v, const std::string& k) { const Value& x = at(v, k); return x.truthy() ? x : NULL_VALUE; }
std::string side_out(int side) { return side ? "SHADOW" : "FREE_PEOPLES"; }
std::string tag(int side) { return side ? "sh" : "fp"; }
int side_in(const std::string& name) { return name == "FREE_PEOPLES" ? 0 : 1; }
}  // namespace

void Tables::load(const Value& j) {
    auto strmap = [&](const char* key, std::map<std::string, std::string>& out) {
        for (const auto& kv : at(j, key).obj) out[kv.first] = kv.second.kind == Value::Null ? "" : kv.second.s;
    };
    strmap("region_in", region_in); strmap("nation_in", nation_in); strmap("face_in", face_in);
    strmap("card_in", card_in); strmap("char_in", char_in); strmap("board_keys", board_keys);
    strmap("region_nation", region_nation);
    for (const auto& kv : at(j, "cards_of").obj) for (const auto& c : kv.second.arr) cards_of[kv.first].push_back(c.s);
    for (const auto& x : at(j, "vp_fp_settlements").arr) vp_fp.push_back({x.arr[0].s, x.arr[1].as_int()});
    for (const auto& x : at(j, "vp_sh_settlements").arr) vp_sh.push_back({x.arr[0].s, x.arr[1].as_int()});
    for (const auto& x : at(j, "minion_keys").arr) minion_keys.push_back(x.s);
    for (const auto& x : at(j, "decks").arr) decks.push_back(x.s);
    for (const auto& x : at(j, "special_tiles").arr) special_tiles.insert(x.s);
    for (const auto& kv : card_in) card_out[kv.second] = kv.first;
    for (const auto& kv : nation_in) nation_out[kv.second] = kv.first;
    for (const auto& kv : at(j, "stronghold").obj) stronghold[kv.first] = kv.second.truthy();
    for (const auto& kv : at(j, "fp_region").obj) fp_region[kv.first] = kv.second.truthy();
    for (const auto& kv : at(j, "adjacent").obj) for (const auto& x : kv.second.arr) adjacent[kv.first].insert(x.s);
    strmap("card_title", card_title); strmap("combat_effect_kind", combat_effect_kind);
    hidden_cc = at(j, "hidden_cc").s;
}

Adapter::Adapter(pyobj::Model& model, const Tables& tables, Obj* prototypes) : m(model), t(tables), protos(prototypes) {}

Obj* Adapter::proto(const std::string& name) const {
    Obj* p = pyobj::Model::dict_get(protos, name);
    if (!p) throw std::runtime_error("no prototype " + name);
    return p;
}

// board.Board.region(name): by sanitised name, 'the ' added or dropped
Obj* Adapter::region_object(Obj* st, const std::string& name) {
    Obj* regions = pyobj::Model::attr(pyobj::Model::attr(st, "board"), "_regions");
    const std::string key = awad_sanitize(name);
    if (Obj* r = pyobj::Model::dict_get(regions, key)) return r;
    const std::string alt = key.rfind("the ", 0) == 0 ? key.substr(4) : "the " + key;
    if (Obj* r = pyobj::Model::dict_get(regions, alt)) return r;
    throw std::runtime_error("unknown region " + name);
}

// game_logic.get_or_create_army
Obj* Adapter::get_or_create_army(Obj* st, const std::string& region_key) {
    Obj* armies = pyobj::Model::attr(st, "armies");
    const std::string san = awad_sanitize(region_key);
    for (size_t j = 0; j + 1 < armies->k.size(); j += 2) if (awad_sanitize(armies->k[j]->s) == san) return armies->k[j + 1];
    Obj* region = region_object(st, san);
    Obj** rk = pyobj::Model::attr_slot(region, "key");
    const std::string key = (rk && (*rk)->tag == WOTR_STRING && !(*rk)->s.empty()) ? (*rk)->s : san;
    for (size_t j = 0; j + 1 < armies->k.size(); j += 2) if (awad_sanitize(armies->k[j]->s) == key) return armies->k[j + 1];
    Obj* army = m.clone(proto("army"), {"board.Region", "board.Board"});
    m.set_attr(army, "region", region);
    m.dict_set(armies, m.str(key), army);
    return army;
}

// character_figure(key, by_name)
Obj* Adapter::character_figure(const std::string& key, const std::map<std::string, Obj*>& by_name) {
    auto it = t.char_in.find(key);
    if (it == t.char_in.end()) return nullptr;
    const std::string& name = it->second;
    auto b = by_name.find(name);
    if (b != by_name.end()) return b->second;
    if (name == "Gandalf the White") return m.clone(proto("figure:gandalf-the-white"));
    if (name == "Aragorn") return m.clone(proto("figure:aragorn"));
    if (std::find(t.minion_keys.begin(), t.minion_keys.end(), key) != t.minion_keys.end()) return m.clone(proto("figure:" + key));
    return nullptr;
}

// Names.tile(key)
Obj* Adapter::tile(const std::string& key) {
    if (t.special_tiles.count(key)) return m.shallow(proto("special:" + key));
    Obj* tl = m.clone(proto("tile"));
    if (key.rfind("eye", 0) == 0) {
        m.set_attr(tl, "is_eye", m.boolean(true));
        m.set_attr(tl, "reveal", m.boolean(true));     // HuntTile.__post_init__: an Eye always reveals
        return tl;
    }
    std::string head = key.substr(0, key.find('-'));
    const bool reveal = !head.empty() && head.back() == 'r';
    while (!head.empty() && head.back() == 'r') head.pop_back();
    m.set_attr(tl, "damage", m.integer(std::stoll(head)));
    m.set_attr(tl, "reveal", m.boolean(reveal));
    return tl;
}

// memo.rng.shuffle(items) (CPython random.shuffle: the box's twin, wotr_random_shuffle)
std::vector<std::string> Adapter::shuffled(Memo& memo, std::vector<std::string> items) {
    std::vector<uint64_t> idx(items.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    if (!idx.empty() && wotr_random_shuffle(memo.rng, idx.data(), idx.size()))
        throw std::runtime_error(std::string("shuffle: ") + wotr_random_last_error());
    std::vector<std::string> out;
    for (auto i : idx) out.push_back(items[i]);
    return out;
}

Obj* Adapter::build_state(const Value& view, Memo& memo) {
    using M = pyobj::Model;
    Obj* st = m.clone(proto("initial"), {"board.Region", "board.Board"});
    std::map<std::string, Obj*> by_name;
    Obj* fs = M::attr(st, "fellowship");
    for (Obj* c : M::attr(fs, "companions")->k) by_name[M::attr(c, "name")->s] = c;
    by_name["Gollum"] = M::attr(st, "gollum");

    m.set_attr(st, "armies", m.dict());
    m.set_attr(st, "besieged_strongholds", m.set());
    Obj* ec = m.dict();
    m.dict_set(ec, side_enum(0), m.set());
    m.dict_set(ec, side_enum(1), m.set());
    m.set_attr(st, "enemy_controlled", ec);
    for (const auto& [slug, r] : or_empty(view, "regions").obj) {
        auto rk = t.region_in.find(slug);
        if (rk == t.region_in.end()) continue;
        const std::string& key = rk->second;
        const Value& units = or_empty(r, "units");
        const Value& chars = or_empty(r, "characters");
        if (units.truthy() || chars.truthy()) {
            Obj* army = get_or_create_army(st, key);
            Obj* aunits = M::attr(army, "units"); Obj* aleaders = M::attr(army, "leaders");
            for (const Value& u : units.arr) {
                auto nk = t.nation_in.find(at(u, "nation").s);
                Obj* nat = (at(u, "nation").kind == Value::Str && nk != t.nation_in.end()) ? nation_enum(nk->second) : m.none();
                const int64_t n = at(u, "n").kind == Value::Null ? 0 : at(u, "n").as_int();
                const std::string kind = at(u, "unit").s;
                for (int64_t q = 0; q < n; ++q) {
                    if (kind == "regular" || kind == "elite") {
                        Obj* unit = m.clone(proto("unit"));
                        m.set_attr(unit, "nation", nat);
                        m.set_attr(unit, "kind", m.enumv("game_state.UnitKind", kind == "regular" ? "REGULAR" : "ELITE"));
                        aunits->k.push_back(unit);
                    } else if (kind == "leader") {
                        Obj* leader = m.clone(proto("leader"));
                        m.set_attr(leader, "nation", nat);
                        aleaders->k.push_back(leader);
                    } else if (kind == "nazgul") {
                        aleaders->k.push_back(m.clone(proto("nazgul")));
                    }
                }
            }
            for (const Value& ck : chars.arr) {
                Obj* fig = character_figure(ck.s, by_name);
                if (fig) aleaders->k.push_back(fig);
            }
        }
        const Value& siege = at(r, "siege");
        const bool besieged = !(siege.kind == Value::Null || (siege.kind == Value::Bool && !siege.b) ||
                                (siege.kind == Value::Str && (siege.s.empty() || siege.s == "false")));
        if (besieged) M::set_add(M::attr(st, "besieged_strongholds"), m.str(key));
        const Value& control = at(r, "control");
        auto rn = t.region_nation.find(key);
        if (control.truthy() && rn != t.region_nation.end() && !rn->second.empty()) {
            const int ctl = side_in(control.s);
            const int native = FP_NATIONS.count(rn->second) ? 0 : 1;
            if (ctl != native) M::set_add(M::dict_get(ec, side_enum(ctl)), m.str(key));
        }
    }
    {   // st.armies = dict((k, a) for k, a in st.armies.items() if a.units or a.leaders)
        Obj* armies = M::attr(st, "armies");
        Obj* kept = m.dict();
        for (size_t j = 0; j + 1 < armies->k.size(); j += 2) {
            Obj* a = armies->k[j + 1];
            if (M::truthy(M::attr(a, "units")) || M::truthy(M::attr(a, "leaders"))) { kept->k.push_back(armies->k[j]); kept->k.push_back(a); }
        }
        m.set_attr(st, "armies", kept);
    }

    const Value& chars = or_empty(view, "characters");
    auto where = [&](const std::string& key) -> std::string {
        const Value& c = or_empty(chars, key);
        const Value& w = at(c, "where");
        return w.kind == Value::Null && !c.get("where") ? "not_in_play" : w.s;
    };

    const Value& f = or_empty(view, "fellowship");
    std::string fkey = "rivendell";
    if (at(f, "region").kind == Value::Str) { auto it = t.region_in.find(at(f, "region").s); if (it != t.region_in.end() && !it->second.empty()) fkey = it->second; }
    m.set_attr(fs, "region", region_object(st, fkey));
    m.set_attr(fs, "progress", m.integer(f.get("progress") ? at(f, "progress").as_int() : 0));
    m.set_attr(fs, "corruption", m.integer(f.get("corruption") ? at(f, "corruption").as_int() : 0));
    m.set_attr(fs, "hidden", m.boolean(f.get("hidden") ? at(f, "hidden").truthy() : true));
    const Value& mordor = at(f, "mordor");
    m.set_attr(fs, "mordor_entered", m.boolean(mordor.kind != Value::Null));
    m.set_attr(fs, "mordor_step", m.integer(mordor.truthy() ? mordor.as_int() : 0));
    {
        Obj* companions = m.list();
        for (const Value& k : or_empty(f, "companions").arr) {
            if (k.s == "gollum") continue;
            auto ci = t.char_in.find(k.s);
            if (ci == t.char_in.end() || !by_name.count(ci->second)) continue;
            companions->k.push_back(by_name[ci->second]);
        }
        m.set_attr(fs, "companions", companions);
        const Value& guide = at(f, "guide");
        auto gi = guide.kind == Value::Str ? t.char_in.find(guide.s) : t.char_in.end();
        if (guide.kind == Value::Str && guide.s == "gollum") m.set_attr(fs, "guide", M::attr(st, "gollum"));
        else if (gi != t.char_in.end() && by_name.count(gi->second)) m.set_attr(fs, "guide", by_name[gi->second]);
        else if (!companions->k.empty()) m.set_attr(fs, "guide", companions->k[0]);
        else m.set_attr(fs, "guide", M::attr(st, "gollum"));
    }
    memo.in_mordor = mordor.kind != Value::Null;

    for (const auto& [key, attr] : std::vector<std::pair<std::string, std::string>>{
             {"witch-king", "witch_king"}, {"saruman", "saruman"}, {"mouth-of-sauron", "mouth_of_sauron"}}) {
        const std::string w = where(key);
        Obj* minion = M::attr(st, attr);
        m.set_attr(minion, "in_play", m.boolean(w == "region"));
        m.set_attr(minion, "was_ever_in_play", m.boolean(w == "region" || w == "eliminated"));
    }
    m.set_attr(M::attr(st, "gandalf_the_white"), "in_play", m.boolean(where("gandalf-the-white") != "not_in_play"));
    m.set_attr(M::attr(st, "aragorn_crowned"), "crowned", m.boolean(where("aragorn") != "not_in_play"));

    Obj* political = M::attr(st, "political");
    for (const auto& [n, p] : or_empty(view, "politics").obj) {
        auto nk = t.nation_in.find(n);
        if (nk == t.nation_in.end()) continue;
        Obj* nat = nation_enum(nk->second);
        m.dict_set(M::attr(political, "boxes"), nat, m.integer(3 - (p.get("step") ? at(p, "step").as_int() : 3)));
        m.dict_set(M::attr(political, "active"), nat, m.boolean(at(p, "active").truthy()));
    }
    {
        Obj* cas = m.dict();
        for (const auto& [n, c] : or_empty(view, "casualties").obj) {
            auto nk = t.nation_in.find(n);
            if (nk == t.nation_in.end() || !FP_NATIONS.count(nk->second)) continue;
            Obj* d = m.dict();
            for (const char* kind : {"regular", "elite", "leader"})
                m.dict_set(d, m.str(kind), m.integer(c.get(kind) ? at(c, kind).as_int() : 0));
            m.dict_set(cas, nation_enum(nk->second), d);
        }
        m.set_attr(st, "fp_casualties", cas);
    }

    Obj* dice = m.dict();
    m.dict_set(dice, side_enum(0), m.list());
    m.dict_set(dice, side_enum(1), m.list());
    m.set_attr(st, "dice", dice);
    Obj* hunt_box = m.list();
    m.set_attr(st, "hunt_box", hunt_box);
    int64_t eyes_allocated = 0;
    bool rolled = false;
    for (int side = 0; side < 2; ++side) {
        const Value& dv = or_empty(view, "dice");
        for (const Value& d : or_empty(dv, side_out(side)).arr) {
            Obj* die = m.clone(proto(side ? "die_sh" : "die_fp"));
            const Value& face = at(d, "face");
            Obj* current = m.none();
            if (face.truthy()) { auto fi = t.face_in.find(face.s); if (fi != t.face_in.end()) current = m.enumv("game_state.ActionDieFace", fi->second); }
            m.set_attr(die, "current", current);
            const std::string status = at(d, "status").kind == Value::Str ? at(d, "status").s : "";
            if (face.truthy()) rolled = true;
            if (status == "used") m.set_attr(die, "used", m.boolean(true));
            else if (status == "hunt_box") {
                m.set_attr(die, "used", m.boolean(true));
                if (side == 1) {
                    if (M::attr(die, "current")->tag == WOTR_NONE) eyes_allocated += 1;
                    m.set_attr(die, "current", m.enumv("game_state.ActionDieFace", "EYE"));
                }
                hunt_box->k.push_back(die);
            }
            M::dict_get(dice, side_enum(side))->k.push_back(die);
        }
    }
    m.set_attr(st, "eyes_allocated", m.integer(eyes_allocated));
    m.set_attr(st, "dice_rolled", m.boolean(rolled));
    const int64_t nfp = int64_t(M::dict_get(dice, side_enum(0))->k.size()), nsh = int64_t(M::dict_get(dice, side_enum(1))->k.size());
    m.set_attr(st, "fp_bonus_dice", m.integer(std::max<int64_t>(0, nfp - 4)));
    m.set_attr(st, "sh_bonus_dice", m.integer(std::max<int64_t>(0, nsh - 7)));

    const Value& hunt = or_empty(view, "hunt");
    auto tiles = [&](std::initializer_list<const char*> keys) {
        Obj* l = m.list();
        for (const char* k : keys) for (const Value& x : or_empty(hunt, k).arr) l->k.push_back(tile(x.s));
        return l;
    };
    m.set_attr(st, "hunt_pool", tiles({"pool"}));
    m.set_attr(st, "hunt_discard", tiles({"drawn", "ready"}));
    m.set_attr(st, "hunt_removed", tiles({"removed"}));
    m.set_attr(st, "_fp_used_hunt_box_last_turn", m.boolean(at(hunt, "shadow_must_hunt").truthy()));

    build_cards(st, view, memo);

    const Value& rings = or_empty(view, "elven_rings");
    const Value& used = or_empty(view, "elven_ring_used");
    m.set_attr(st, "fp_elven_rings", m.integer(rings.get("FREE_PEOPLES") ? at(rings, "FREE_PEOPLES").as_int() : 0));
    m.set_attr(st, "sh_elven_rings", m.integer(rings.get("SHADOW") ? at(rings, "SHADOW").as_int() : 0));
    m.set_attr(st, "fp_ring_used_this_turn", m.boolean(at(used, "FREE_PEOPLES").truthy()));
    m.set_attr(st, "sh_ring_used_this_turn", m.boolean(at(used, "SHADOW").truthy()));
    m.set_attr(st, "turn", m.integer(view.get("turn") ? at(view, "turn").as_int() : 1));
    m.set_attr(st, "sh_mouth_muster_used_this_turn", m.boolean(memo.messenger_used));
    m.set_attr(st, "fellowship_acted_in_mordor", m.boolean(memo.fellowship_acted));
    {   // game_logic.calculate_victory_points
        auto captured = [&](int side, const std::string& key) {
            for (Obj* x : M::dict_get(ec, side_enum(side))->k) if (x->s == key) return true;
            return false;
        };
        int64_t sh_vp = 0, fp_vp = 0;
        for (const auto& [k, v] : t.vp_fp) if (captured(1, k)) sh_vp += v;
        for (const auto& [k, v] : t.vp_sh) if (captured(0, k)) fp_vp += v;
        Obj* vp = M::attr(st, "vp");
        m.set_attr(vp, "free_peoples", m.integer(fp_vp));
        m.set_attr(vp, "shadow", m.integer(sh_vp));
    }
    m.set_attr(st, "pending_discard", m.none());
    return st;
}

void Adapter::build_cards(Obj* st, const Value& view, Memo& memo) {
    using M = pyobj::Model;
    const int me = memo.side, opp = 1 - me;
    const Value& c = or_empty(view, "cards");
    const Value& discards = or_empty(c, "discards");
    const Value& table = or_empty(c, "on_table");
    const Value& battle = or_empty(view, "battle");
    std::vector<std::string> in_battle;
    for (const char* k : {"attacker_card", "defender_card", "event_card"}) if (at(battle, k).truthy()) in_battle.push_back(at(battle, k).s);
    auto deck_of = [](const std::string& key) { return key.find("-char-") != std::string::npos ? "character" : "strategy"; };
    auto cards = [&](const Value& v) { std::vector<std::string> out; for (const Value& k : v.arr) out.push_back(k.s); return out; };

    Obj* dfu = M::attr(st, "discard_face_up");
    for (int side = 0; side < 2; ++side)
        for (const auto& deck : t.decks) {
            Obj* l = m.list();
            for (const auto& k : cards(or_empty(discards, side_out(side)))) {
                const std::string cid = t.card(k);
                if (deck_of(k) == deck && !cid.empty()) l->k.push_back(m.str(cid));
            }
            m.dict_set(dfu, m.str(tag(side) + "_" + deck), l);
        }

    auto ids = [&](const Value& v) { std::vector<std::string> out; for (const auto& k : cards(v)) { auto cid = t.card(k); if (!cid.empty()) out.push_back(cid); } return out; };
    const std::vector<std::string> hand = ids(or_empty(c, "hand")), fdown = ids(or_empty(c, "discarded_face_down"));
    std::set<std::string> seen(hand.begin(), hand.end());
    seen.insert(fdown.begin(), fdown.end());
    for (const auto& k : in_battle) seen.insert(t.card(k));
    for (const auto& k : cards(or_empty(discards, side_out(me)))) seen.insert(t.card(k));
    for (const auto& k : cards(or_empty(table, side_out(me)))) seen.insert(t.card(k));
    const Value& decks = or_empty(or_empty(c, "decks"), side_out(me));
    auto deck_attr = [&](int side, const std::string& deck) { return tag(side) + "_" + (deck == "character" ? "char" : "strat") + "_deck"; };
    auto strings = [&](const std::vector<std::string>& xs) { Obj* l = m.list(); for (const auto& x : xs) l->k.push_back(m.str(x)); return l; };
    for (const auto& deck : t.decks) {
        std::vector<std::string> rest;
        for (const auto& cid : t.cards_of.at(side_out(me) + "|" + deck)) if (!seen.count(cid)) rest.push_back(cid);
        rest = shuffled(memo, rest);
        const int64_t want = decks.get(deck) ? at(decks, deck).as_int() : int64_t(rest.size());
        rest.resize(size_t(std::max<int64_t>(0, std::min<int64_t>(want, int64_t(rest.size())))));
        m.set_attr(st, deck_attr(me, deck), strings(rest));
    }
    Obj* hands = M::attr(st, "hands");
    Obj* mine = M::dict_get(hands, side_enum(me));
    m.set_attr(mine, "hand", strings(hand));
    m.set_attr(mine, "discarded_face_down", strings(fdown));
    m.set_attr(mine, "needs_discard", m.boolean(hand.size() > 6));

    std::set<std::string> oseen;
    for (const auto& k : in_battle) oseen.insert(t.card(k));
    for (const auto& k : cards(or_empty(discards, side_out(opp)))) oseen.insert(t.card(k));
    for (const auto& k : cards(or_empty(table, side_out(opp)))) oseen.insert(t.card(k));
    const Value& by_deck = or_empty(c, "opponent_hand_by_deck");
    const Value& odecks = or_empty(or_empty(c, "decks"), side_out(opp));
    const int64_t total_fd = c.get("opponent_discarded_face_down") ? at(c, "opponent_discarded_face_down").as_int() : 0;
    std::vector<std::string> ohand, ofd;
    for (const auto& deck : t.decks) {
        std::vector<std::string> unseen;
        for (const auto& cid : t.cards_of.at(side_out(opp) + "|" + deck)) if (!oseen.count(cid)) unseen.push_back(cid);
        unseen = shuffled(memo, unseen);
        const int64_t size = int64_t(unseen.size());
        const int64_t nh = std::min<int64_t>(by_deck.get(deck) ? at(by_deck, deck).as_int() : 0, size);
        const int64_t nd = std::min<int64_t>(odecks.get(deck) ? at(odecks, deck).as_int() : 0, size - nh);
        std::vector<std::string> rest(unseen.begin() + std::max<int64_t>(0, nh + nd), unseen.end());
        const int64_t take_fd = std::min<int64_t>(int64_t(rest.size()), std::max<int64_t>(0, total_fd - int64_t(ofd.size())));
        ohand.insert(ohand.end(), unseen.begin(), unseen.begin() + std::max<int64_t>(0, nh));
        ofd.insert(ofd.end(), rest.begin(), rest.begin() + take_fd);
        m.set_attr(st, deck_attr(opp, deck), strings(std::vector<std::string>(unseen.begin() + std::max<int64_t>(0, nh),
                                                                               unseen.begin() + std::max<int64_t>(0, nh + nd))));
    }
    Obj* theirs = M::dict_get(hands, side_enum(opp));
    m.set_attr(theirs, "hand", strings(ohand));
    m.set_attr(theirs, "discarded_face_down", strings(ofd));
    m.set_attr(theirs, "needs_discard", m.boolean(ohand.size() > 6));
    Obj* ds = m.clone(proto("deck_status"));
    for (const auto& [field, attr] : std::vector<std::pair<std::string, std::string>>{
             {"fp_character", "fp_char_deck"}, {"fp_strategy", "fp_strat_deck"}, {"sh_character", "sh_char_deck"}, {"sh_strategy", "sh_strat_deck"}})
        m.set_attr(ds, field, m.integer(int64_t(M::attr(st, attr)->k.size())));
    m.set_attr(st, "decks", ds);

    Obj* pcs = m.clone(proto("persistent_state"));
    for (int side = 0; side < 2; ++side)
        for (const auto& k : cards(or_empty(table, side_out(side)))) {
            const std::string cid = t.card(k);
            if (cid.empty()) continue;
            Obj* pc = m.clone(proto("persistent:" + cid));
            m.set_attr(pc, "side", side_enum(side));
            // PersistentCardState.add_card: by the card's side
            const bool fp = M::attr(pc, "side") == side_enum(0);
            M::attr(pcs, fp ? "fp_cards" : "sh_cards")->k.push_back(pc);
        }
    m.set_attr(st, "persistent_cards", pcs);
}

// ---------------------------------------------------------------- SETUPS (the adapter's lines 1879-2272)
namespace {
// opts(dec, kind, **fields): the options of `kind` whose fields equal the given ones (a field given as None: absent
// or null)
std::vector<const Value*> opts(const Value& dec, const std::string& kind, std::vector<std::pair<std::string, const Value*>> fields = {}) {
    std::vector<const Value*> out;
    for (const Value& o : or_empty(dec, "options").arr) {
        if (!kind.empty() && at(o, "kind").s != kind) continue;
        bool ok = true;
        for (const auto& [k, v] : fields) {
            const Value& x = at(o, k);
            if (v == nullptr) { if (x.kind != Value::Null) { ok = false; break; } continue; }
            if (x.kind != v->kind || x.s != v->s || x.i != v->i || x.b != v->b) { ok = false; break; }
        }
        if (ok) out.push_back(&o);
    }
    return out;
}
std::string lower(std::string x) { for (char& c : x) c = char(std::tolower(static_cast<unsigned char>(c))); return x; }
// plain(s): lower-case letters and digits (the titles compared here are ASCII)
std::string plain(const std::string& x) { std::string o; for (unsigned char c : x) if (std::isalnum(c) && c < 128) o += char(std::tolower(c)); return o; }
}  // namespace

// army_at(st, key): the army whose key sanitises as `key` does
Obj* Adapter::army_at(Obj* st, const std::string& key) {
    Obj* armies = pyobj::Model::attr(st, "armies");
    const std::string san = awad_sanitize(key);
    for (size_t j = 0; j + 1 < armies->k.size(); j += 2) if (awad_sanitize(armies->k[j]->s) == san) return armies->k[j + 1];
    return nullptr;
}

// game_logic._leader_side: Companions FP, Nazgul and Minions SH, plain leaders by nation (None: FP)
int Adapter::leader_side(Obj* ld) {
    const std::string& type = pyobj::Model::type(ld);
    if (type == "game_state.Companion") return 0;
    if (type == "game_state.Nazgul" || type == "game_logic.Minion") return 1;
    Obj** nat = pyobj::Model::attr_slot(ld, "nation");
    if (!nat || (*nat)->tag != WOTR_ENUM) return 0;
    return FP_NATIONS.count((*nat)->k[1]->s) ? 0 : 1;
}

// _sail_in: Corsairs of Umbar's landing (the adapter's lines 1921-1981)
Obj* Adapter::sail_in(Obj* st, const Value& b, const std::string& frm, const std::string& reg, int atk) {
    using M = pyobj::Model;
    Obj* src = army_at(st, frm);
    if (!src) return nullptr;
    auto nation_of = [](Obj* x) { return M::attr(x, "nation")->k[1]->s; };
    auto unit_side = [&](Obj* u) { return FP_NATIONS.count(nation_of(u)) ? 0 : 1; };
    auto saruman = [](Obj* ld) {
        if (M::type(ld) != "game_logic.Minion") return false;
        Obj** n = M::attr_slot(ld, "name");
        return n && (*n)->tag == WOTR_STRING && lower((*n)->s).find("saruman") != std::string::npos;
    };
    Obj* relief = nullptr;
    bool besieged = false;
    for (Obj* x : M::attr(st, "besieged_strongholds")->k) if (x->s == reg) besieged = true;
    // combat.garrison_side_for
    auto garrison_side = [&](const std::string& region) {
        Obj* ec = M::attr(st, "enemy_controlled");
        const std::string san = awad_sanitize(region);
        for (int side : {1, 0}) {
            Obj* set = M::dict_get(ec, side_enum(side));
            if (set) for (Obj* x : set->k) if (x->s == san) return side;
        }
        auto f = t.fp_region.find(san);
        return (f != t.fp_region.end() && f->second) ? 0 : 1;
    };
    if (besieged && garrison_side(reg) == atk && army_at(st, reg)) {
        // army_card_cargo.freeze(inventory(st, reg)): the Shadow figures there, by category, sorted
        Obj* garrison = army_at(st, reg);
        std::map<std::tuple<std::string, std::string, std::string>, int64_t> inv;
        for (Obj* u : M::attr(garrison, "units")->k)
            if (unit_side(u) == 1) ++inv[{"unit", nation_of(u), M::attr(u, "kind")->k[1]->s}];
        for (Obj* l : M::attr(garrison, "leaders")->k) {
            if (leader_side(l) != 1 || saruman(l)) continue;
            if (M::type(l) == "game_state.Nazgul") { ++inv[{"nazgul", "SAURON", ""}]; continue; }
            Obj** n = M::attr_slot(l, "name");
            if (n && M::truthy(*n)) ++inv[{"named", (*n)->s, ""}];
            else ++inv[{"leader", nation_of(l), ""}];
        }
        relief = m.tuple();
        for (const auto& [key, count] : inv)
            if (count) relief->k.push_back(m.tuple({m.str(std::get<0>(key)), m.str(std::get<1>(key)), m.str(std::get<2>(key)), m.integer(count)}));
    }
    std::map<std::pair<std::string, std::string>, int64_t> want;
    const bool listed = at(b, "attackers").truthy();
    for (const Value& p : or_empty(b, "attackers").arr) want[{at(p, "nation").s, at(p, "unit").s}] += at(p, "n").kind == Value::Null ? 0 : at(p, "n").as_int();
    auto take = [&](const std::pair<std::string, std::string>& k) {
        if (!listed) return true;
        auto it = want.find(k);
        if (it != want.end() && it->second > 0) { --it->second; return true; }
        return false;
    };
    Obj* dst = get_or_create_army(st, reg);
    Obj* keep_units = m.list();
    for (Obj* u : M::attr(src, "units")->k) {
        const std::string unit = M::attr(u, "kind")->k[1]->s == "REGULAR" ? "regular" : "elite";
        if (unit_side(u) == atk && take({t.nation_out.at(nation_of(u)), unit})) M::attr(dst, "units")->k.push_back(u);
        else keep_units->k.push_back(u);
    }
    Obj* keep_leaders = m.list();
    for (Obj* ld : M::attr(src, "leaders")->k) {
        const std::string& tn = M::type(ld);
        bool go;
        if (leader_side(ld) != atk || saruman(ld)) go = false;
        else if (tn == "game_state.Nazgul") go = take({"SAURON", "nazgul"});
        else if (tn == "game_state.Leader") go = take({t.nation_out.at(nation_of(ld)), "leader"});
        else go = true;
        (go ? M::attr(dst, "leaders") : keep_leaders)->k.push_back(ld);
    }
    m.set_attr(src, "units", keep_units);
    m.set_attr(src, "leaders", keep_leaders);
    if (keep_units->k.empty() && keep_leaders->k.empty()) {
        Obj* armies = M::attr(st, "armies");
        Obj* kept = m.dict();
        for (size_t j = 0; j + 1 < armies->k.size(); j += 2)
            if (armies->k[j + 1] != src) { kept->k.push_back(armies->k[j]); kept->k.push_back(armies->k[j + 1]); }
        m.set_attr(st, "armies", kept);
    }
    return relief;
}

// _battle: the engine's pending_combat for the referee's battle (the adapter's lines 1983-2046)
Obj* Adapter::battle(Obj* st, const Value& view, const Memo& memo, const std::string& phase, int64_t rounds_fought) {
    using M = pyobj::Model;
    const Value& b = at(view, "battle");
    if (!b.truthy()) return nullptr;
    const int atk = side_in(at(b, "attacker").s);
    auto fi = t.region_in.find(at(b, "from").s), ri = t.region_in.find(at(b, "region").s);
    if (at(b, "from").kind != Value::Str || at(b, "region").kind != Value::Str || fi == t.region_in.end() || ri == t.region_in.end()) return nullptr;
    std::string frm = fi->second;
    const std::string reg = ri->second;
    const std::string kind = at(b, "battle").kind == Value::Str ? at(b, "battle").s : "";
    auto sh = t.stronghold.find(reg);
    const bool stronghold = sh != t.stronghold.end() && sh->second;
    auto adj = t.adjacent.find(frm);
    const bool sailed = frm != reg && !(adj != t.adjacent.end() && adj->second.count(reg));
    Obj* relief = nullptr;
    bool pre_siege = false;
    if (sailed) {
        relief = sail_in(st, b, frm, reg, atk);
        for (Obj* x : M::attr(st, "besieged_strongholds")->k) if (x->s == reg) pre_siege = true;
        frm = reg;
    }
    Obj* pc = m.dict();
    auto put = [&](const char* k, Obj* v) { m.dict_set(pc, m.str(k), v); };
    put("attacker_region", m.str(frm)); put("defender_region", m.str(reg));
    put("attacker_side", side_enum(atk)); put("defender_side", side_enum(1 - atk));
    put("is_siege", m.boolean(kind == "siege")); put("stronghold_region", m.boolean(stronghold));
    put("phase", m.str(phase)); put("deciding_side", side_enum(memo.side));
    put("rounds_fought", m.integer(rounds_fought)); put("shine_asked", m.boolean(true));
    put("atk_casualty_policy", m.str("preserve_elites")); put("def_casualty_policy", m.str("preserve_elites"));
    if (at(b, "white_rider").truthy()) put("gtw_shine", m.boolean(true));
    if (sailed) {
        put("attacker_no_retreat", m.boolean(!pre_siege || relief != nullptr));
        if (relief) { put("is_siege", m.boolean(false)); put("corsairs_relief_garrison", relief); }
    }
    // the rearguard as v0.2.0 records it (the adapter's 2cc5f6d): pc['attacker_rearguard'] = (region, side,
    // ('forced_rearguard_v1', rear units by (nation, kind), rear plain Leaders by nation)); the rear stays put
    if (frm != reg && at(b, "attackers").truthy()) {
        std::map<std::pair<std::string, std::string>, int64_t> want;
        for (const Value& p : at(b, "attackers").arr) want[{at(p, "nation").s, at(p, "unit").s}] += at(p, "n").kind == Value::Null ? 0 : at(p, "n").as_int();
        if (Obj* army = army_at(st, frm)) {
            std::map<std::pair<std::string, std::string>, int64_t> units;
            std::map<std::string, int64_t> leaders;
            for (Obj* u : M::attr(army, "units")->k) {
                const std::string nat = M::attr(u, "nation")->k[1]->s;
                if ((FP_NATIONS.count(nat) ? 0 : 1) != atk) continue;
                const std::string kind = M::attr(u, "kind")->k[1]->s;
                auto it = want.find({t.nation_out.at(nat), kind == "REGULAR" ? "regular" : "elite"});
                if (it != want.end() && it->second > 0) --it->second;
                else ++units[{nat, kind}];
            }
            for (Obj* ld : M::attr(army, "leaders")->k) {
                if (M::type(ld) != "game_state.Leader" || leader_side(ld) != atk) continue;
                const std::string nat = M::attr(ld, "nation")->k[1]->s;
                auto it = want.find({t.nation_out.at(nat), "leader"});
                if (it != want.end() && it->second > 0) --it->second;
                else ++leaders[nat];
            }
            Obj* us = m.tuple();
            for (const auto& [k, c] : units) us->k.push_back(m.tuple({m.str(k.first), m.str(k.second), m.integer(c)}));
            Obj* ls = m.tuple();
            for (const auto& [k, c] : leaders) ls->k.push_back(m.tuple({m.str(k), m.integer(c)}));
            put("attacker_rearguard", m.tuple({m.str(awad_sanitize(frm)), m.str(atk ? "SHADOW" : "FREE_PEOPLES"),
                                                m.tuple({m.str("forced_rearguard_v1"), us, ls})}));
        }
    }
    m.set_attr(st, "pending_combat", pc);
    return pc;
}

Adapter::Setup Adapter::setup_card_choice(Obj* st, const Value& dec, const Value& view, const Memo& memo) {
    using M = pyobj::Model;
    Setup r;
    r.ported = true;                                           // a None set-up is a port result too
    const Value& ctx = or_empty(dec, "context");
    const int me = memo.side;
    const Value& b = at(view, "battle");
    std::vector<const Value*> regions;
    for (const Value* o : opts(dec, "card_choice")) if (at(*o, "region").truthy()) regions.push_back(o);
    auto question = [&]() { const Value& q = at(ctx, "question"); return q.kind == Value::Str ? q.s : (q.kind == Value::Null ? "None" : ""); };
    if (b.truthy() && !regions.empty() && lower(question()).find("retreat") != std::string::npos) {
        const bool attacker = side_in(at(b, "attacker").s) == me;
        const int64_t rnd = b.get("round") ? at(b, "round").as_int() : 1;
        Obj* pc = battle(st, view, memo, attacker ? "attacker_card_retreat" : "defender_retreat", rnd - 1);
        if (pc) {
            m.dict_set(pc, m.str("_card_retreat_offered"), m.boolean(true));
            m.dict_set(pc, m.str("_card_retreat_mandatory"), m.boolean(true));
            r.label = "action"; return r;
        }
    }
    const auto counts = opts(dec, "card_choice");
    const std::string card = at(ctx, "card").kind == Value::Str ? t.card(at(ctx, "card").s) : "";
    bool all_counts = !counts.empty();
    for (const Value* o : counts) if (at(*o, "count").kind == Value::Null) all_counts = false;
    if (b.truthy() && all_counts && !card.empty()) {
        auto ki = t.combat_effect_kind.find(card);
        const std::string kind = ki == t.combat_effect_kind.end() ? "" : ki->second;
        const bool attacker = side_in(at(b, "attacker").s) == me;
        if (!kind.empty()) {
            const int64_t rnd = b.get("round") ? at(b, "round").as_int() : 1;
            Obj* pc = battle(st, view, memo, "combat_effect_choice", rnd - 1);
            if (pc) {
                const std::string role = attacker ? "atk" : "def";
                Obj* ec = m.dict();
                Obj* cl = m.list();
                int64_t mx = 0; bool first = true;
                for (const Value* o : counts) { const int64_t c = at(*o, "count").as_int(); cl->k.push_back(m.integer(c)); if (first || c > mx) mx = c; first = false; }
                m.dict_set(ec, m.str("stage"), m.str(kind == "onslaught" ? "post" : "pre"));
                m.dict_set(ec, m.str("owner"), m.str(role));
                m.dict_set(ec, m.str("effect"), m.str(kind));
                m.dict_set(ec, m.str("card_id"), m.str(card));
                m.dict_set(ec, m.str("counts"), cl);
                m.dict_set(ec, m.str("max_count"), m.integer(mx));
                m.dict_set(pc, m.str("effect_choice"), ec);
                m.dict_set(pc, m.str(role + "_combat_card"), m.str(card));
                r.label = "action"; return r;
            }
        }
    }
    const auto draws = opts(dec, "draw");
    if (!draws.empty() && me == 1) {   // the Palantir of Orthanc's bonus draw (ai._open_palantir_deck_choice)
        Obj* opt = m.dict();
        auto put = [&](const char* k, Obj* v) { m.dict_set(opt, m.str(k), v); };
        put("card_id", m.str("__palantir_draw__")); put("card_name", m.str("Palantir draw"));
        put("side", m.str("sh")); put("decider", m.str("sh")); put("direct", m.boolean(true)); put("palantir_draw", m.boolean(true));
        Obj* options = m.list();
        for (const Value* o : draws) {
            Obj* d = m.dict();
            const Value& deck = at(*o, "deck");
            m.dict_set(d, m.str("palantir_deck"), deck.kind == Value::Str ? m.str(deck.s) : m.none());
            options->k.push_back(d);
        }
        put("options", options);
        m.set_attr(st, "pending_card_option", opt);
        r.label = "action"; return r;
    }
    Value skip_true; skip_true.kind = Value::Bool; skip_true.b = true;
    if (me == 0 && ctx.get("damage") && !opts(dec, "card_choice", {{"skip", &skip_true}}).empty()) {
        for (const Value* o : counts) {   // Axe and Bow / Horn of Gondor against the Hunt damage
            const Value& tc = at(*o, "target_card");
            const std::string cid = tc.kind == Value::Str ? t.card(tc.s) : "";
            if (cid.empty()) continue;
            const std::string title = t.card_title.at(cid);
            if (plain(title) == plain("Axe and Bow") || plain(title) == plain("Horn of Gondor")) {
                m.set_attr(st, "_pending_hunt_damage", m.integer(at(ctx, "damage").as_int()));
                Obj* opt = m.dict();
                auto put = [&](const char* k, Obj* v) { m.dict_set(opt, m.str(k), v); };
                put("card_id", m.str(cid)); put("card_name", m.str(title)); put("side", m.str("fp")); put("decider", m.str("fp"));
                put("direct", m.boolean(true)); put("hunt_reduction", m.boolean(true));
                Obj* use = m.dict(); m.dict_set(use, m.str("use_hunt_reduction"), m.integer(1)); m.dict_set(use, m.str("card_id"), m.str(cid));
                Obj* no = m.dict(); m.dict_set(no, m.str("use_hunt_reduction"), m.integer(0));
                put("options", m.list({use, no}));
                m.set_attr(st, "pending_card_option", opt);
                r.label = "action"; return r;
            }
        }
    }
    r.label = ""; return r;                                    // None: no engine decision
}

// game_logic.fellowship_region: the Fellowship's region key, sanitised
std::string Adapter::fellowship_region(Obj* st) {
    using M = pyobj::Model;
    Obj* r = M::attr(M::attr(st, "fellowship"), "region");
    if (r->tag == WOTR_STRING) return awad_sanitize(r->s);
    for (const char* a : {"key", "name", "orig_name"}) {
        Obj** v = M::attr_slot(r, a);
        if (v && (*v)->tag == WOTR_STRING) return awad_sanitize((*v)->s);
    }
    throw std::runtime_error("fellowship region without a key");
}

// ai.auto_mordor_gateway: the gateway the Fellowship's progress reaches (hidden, not in Mordor), else ""
std::string Adapter::auto_mordor_gateway(Obj* st) {
    using M = pyobj::Model;
    Obj* f = M::attr(st, "fellowship");
    const int64_t progress = M::attr(f, "progress")->i;
    if (M::truthy(M::attr(f, "mordor_entered")) || !M::truthy(M::attr(f, "hidden")) || progress <= 0) return "";
    const std::string cur = fellowship_region(st);
    for (const std::string gw : {"morannon", "minas morgul"}) {
        if (!t.stronghold.count(gw)) continue;                      // board.regions
        std::set<std::string> seen = {cur};
        std::vector<std::pair<std::string, int64_t>> q = {{cur, 0}};
        int64_t dist = -1;
        for (size_t h = 0; h < q.size(); ++h) {
            const auto [r, d] = q[h];
            if (r == gw) { dist = d; break; }
            if (d >= progress) continue;
            auto adj = t.adjacent.find(r);
            if (adj == t.adjacent.end()) continue;
            for (const auto& x : adj->second) {
                const std::string a = awad_sanitize(x);
                if (seen.insert(a).second) q.push_back({a, d + 1});
            }
        }
        if (dist >= 0 && dist <= progress) return gw;
    }
    return "";
}

Adapter::Setup Adapter::setup(const std::string& kind, Obj* st, const Value& dec, const Value& view, const Memo& memo) {
    using M = pyobj::Model;
    Setup r;
    const Value& b = or_empty(view, "battle");
    const Value& ctx = or_empty(dec, "context");
    const int me = memo.side;
    const int64_t rnd = b.get("round") ? at(b, "round").as_int() : 1;
    auto done = [&](Obj* pc, const char* label = "action") { r.ported = true; r.label = pc ? label : ""; return r; };
    if (kind == "ACTION") {                       // setup_action
        m.set_attr(st, "pending_combat", m.none());
        m.set_attr(st, "dice_rolled", m.boolean(true));
        r.ported = true; r.label = "action";
    } else if (kind == "HUNT_ALLOCATION") {       // setup_hunt_allocation
        r.ported = true; r.label = "eye"; r.flag = "_eye_allocation_phase";
    } else if (kind == "DISCARD_EXCESS_CARD") {   // setup_discard
        Obj* d = m.dict();
        m.dict_set(d, m.str("side"), side_enum(me));
        m.set_attr(st, "pending_discard", d);
        r.ported = true; r.label = "discard";
    } else if (kind == "COMBAT_CARD") {           // setup_combat_card
        const bool attacker = side_in(at(b, "attacker").s) == me;
        Obj* pc = battle(st, view, memo, attacker ? "attacker_combat_card" : "defender_combat_card", rnd - 1);
        if (pc && !attacker) {
            const Value& back = at(b, "attacker_card_back");
            const Value& shown = at(b, "attacker_card");
            if (shown.truthy()) {
                const std::string cid = t.card(shown.s);
                m.dict_set(pc, m.str("atk_combat_card"), cid.empty() ? m.none() : m.str(cid));
            } else if (back.truthy()) {
                m.dict_set(pc, m.str("atk_combat_card"), m.str(t.hidden_cc));
                Obj* h = m.dict(); m.dict_set(h, m.str("deck"), m.str(back.s));
                m.dict_set(pc, m.str("atk_hidden_cc"), h);
            } else m.dict_set(pc, m.str("atk_combat_card"), m.none());
        }
        return done(pc);
    } else if (kind == "CASUALTY") {              // setup_casualty
        const bool attacker = side_in(at(b, "attacker").s) == me;
        Obj* pc = battle(st, view, memo, "casualty_allocation", rnd);
        if (pc) {
            const int64_t hits = ctx.get("hits") ? at(ctx, "hits").as_int() : 1;
            m.dict_set(pc, m.str("alloc_stage"), m.str(attacker ? "attacker" : "defender"));
            m.dict_set(pc, m.str("atk_hits"), m.integer(attacker ? hits : 0));
            m.dict_set(pc, m.str("def_hits"), m.integer(attacker ? 0 : hits));
        }
        return done(pc);
    } else if (kind == "CEASE_ATTACK") {
        return done(battle(st, view, memo, at(b, "battle").kind == Value::Str && at(b, "battle").s == "siege" ? "combat_round_result" : "attacker_continue", rnd));
    } else if (kind == "EXTEND_SIEGE") {
        return done(battle(st, view, memo, "combat_round_result", rnd));
    } else if (kind == "RETREAT") {
        return done(battle(st, view, memo, "defender_retreat", rnd));
    } else if (kind == "FIELD_OR_SIEGE") {
        return done(rnd <= 1 ? battle(st, view, memo, "defender_siege_choice", 0) : battle(st, view, memo, "defender_retreat", rnd - 1));
    } else if (kind == "ADVANCE") {
        Obj* pc = battle(st, view, memo, "victor_advance", rnd);
        if (pc) m.dict_set(pc, m.str("is_siege"), m.boolean(false));
        return done(pc);
    } else if (kind == "WHITE_RIDER") {
        Obj* pc = battle(st, view, memo, "fp_shine", rnd - 1);
        if (pc) {   // pc.pop('shine_asked', None)
            for (size_t j = 0; j + 1 < pc->k.size(); j += 2)
                if (pc->k[j]->s == "shine_asked") { pc->k.erase(pc->k.begin() + long(j), pc->k.begin() + long(j) + 2); break; }
        }
        return done(pc);
    } else if (kind == "HUNT_CASUALTY") {
        Obj* h = m.dict();
        m.dict_set(h, m.str("damage"), m.integer(ctx.get("damage") ? at(ctx, "damage").as_int() : 1));
        m.dict_set(h, m.str("phase"), m.str("hunt_sacrifice"));
        m.dict_set(h, m.str("casualty_taken"), m.boolean(false));
        m.set_attr(st, "pending_hunt", h);
        m.set_attr(st, "_pending_hunt_damage", m.integer(0));
        r.ported = true; r.label = "action";
    } else if (kind == "GUIDE_ABILITY") {
        Obj* g = M::attr(M::attr(st, "fellowship"), "guide");
        Obj** nm = M::attr_slot(g, "name");
        Obj** ability = M::attr_slot(g, "guide_ability");
        std::string phase;
        if (nm && (*nm)->tag == WOTR_STRING && (*nm)->s == "Gollum") phase = "hunt_gollum";
        else if (ability && (*ability)->tag == WOTR_STRING && (*ability)->s == "sacrifice_reduce_hunt") phase = "hunt_sacrifice";
        r.ported = true;
        if (phase.empty()) return r;
        Obj* h = m.dict();
        m.dict_set(h, m.str("damage"), m.integer(ctx.get("damage") ? at(ctx, "damage").as_int() : 1));
        m.dict_set(h, m.str("phase"), m.str(phase));
        m.dict_set(h, m.str("casualty_taken"), m.boolean(false));
        m.set_attr(st, "pending_hunt", h);
        m.set_attr(st, "_pending_hunt_damage", m.integer(0));
        r.label = "action";
    } else if (kind == "REVEAL_FELLOWSHIP") {
        m.set_attr(M::attr(st, "fellowship"), "hidden", m.boolean(false));
        m.set_attr(st, "_reveal_place_due", m.boolean(true));
        m.set_attr(st, "_pending_hunt_damage", m.integer(0));
        m.set_attr(st, "_sh_tile_drawn_regions", m.set());
        r.ported = true; r.label = "action";
    } else if (kind == "CHOOSE_GUIDE") {
        Obj* cands = m.list();
        for (const Value* o : opts(dec, "guide")) {
            auto ci = at(*o, "character").kind == Value::Str ? t.char_in.find(at(*o, "character").s) : t.char_in.end();
            if (ci != t.char_in.end() && !ci->second.empty()) cands->k.push_back(m.str(ci->second));
        }
        r.ported = true;
        if (cands->k.empty()) return r;
        Obj* d = m.dict(); m.dict_set(d, m.str("candidates"), cands);
        m.set_attr(st, "pending_guide_change", d);
        r.label = "action";
    } else if (kind == "CARD_CHOICE") {
        return setup_card_choice(st, dec, view, memo);
    } else if (kind == "FELLOWSHIP_PHASE") {      // setup_fellowship
        r.ported = true;
        auto match = [&](const std::vector<const Value*>& o) {
            r.has_match = true;
            if (o.empty() || at(*o[0], "o").kind == Value::Null) r.match_none = true; else r.match = at(*o[0], "o").as_int();
            return r;
        };
        Obj* fs = M::attr(st, "fellowship");
        const auto enter = opts(dec, "enter_mordor");
        const std::string here = fellowship_region(st);
        if (!enter.empty() && (here == "morannon" || here == "minas morgul")) return match(enter);
        // ai._auto_enter_mordor on a copy: it declares into the gateway its progress reaches
        const std::string gw = auto_mordor_gateway(st);
        if (!gw.empty()) {
            std::string slug;
            for (const auto& kv : t.region_in) if (kv.second == awad_sanitize(gw)) slug = kv.first;
            Value sv; sv.kind = Value::Str; sv.s = slug;
            const auto o = opts(dec, "declare", {{"region", &sv}});
            if (!o.empty() && at(*o[0], "o").kind != Value::Null) return match(o);
        }
        const auto declares = opts(dec, "declare");
        if (declares.empty() || !M::truthy(M::attr(fs, "hidden"))) return match(opts(dec, "done"));
        // ai._get_declare_actions(st, allow_in_place=True) is empty only for a revealed Fellowship, one in Mordor, or a
        // negative progress (otherwise its own region is reachable)
        if (M::truthy(M::attr(fs, "mordor_entered")) || M::attr(fs, "progress")->i < 0) return match(opts(dec, "done"));
        r.label = "declare"; r.flag = "_declare_phase";
    }
    return r;
}

}  // namespace awad
