// protocol/wotr_protocol_bot.py build_state and _build_cards (lines 460-690), line for line, on the object model.
// Every object the Python constructs comes from a prototype its constructor made (record_build_state.py), so the
// attributes, their order and their defaults are the engine's own.
#include "adapter.h"
#include "wotr_random.h"

#include <algorithm>

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

// SETUPS (the adapter's lines 1879-2272), ported kind by kind
Adapter::Setup Adapter::setup(const std::string& kind, Obj* st, const Value& dec, const Value& view, const Memo& memo) {
    (void)dec; (void)view;
    Setup r;
    if (kind == "ACTION") {                       // setup_action
        m.set_attr(st, "pending_combat", m.none());
        m.set_attr(st, "dice_rolled", m.boolean(true));
        r.ported = true; r.label = "action";
    } else if (kind == "HUNT_ALLOCATION") {       // setup_hunt_allocation
        r.ported = true; r.label = "eye"; r.flag = "_eye_allocation_phase";
    } else if (kind == "DISCARD_EXCESS_CARD") {   // setup_discard
        Obj* d = m.dict();
        m.dict_set(d, m.str("side"), side_enum(memo.side));
        m.set_attr(st, "pending_discard", d);
        r.ported = true; r.label = "discard";
    }
    return r;
}

}  // namespace awad
