// The protocol adapter (protocol/wotr_protocol_bot.py) in C++: phase 3, the state from the referee's view.
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>
#include "json.h"
#include "pyobj.h"

std::string awad_sanitize(const std::string& name);   // board.sanitize (the box's sanitize_name; driver/box_unity.cpp)

namespace awad {

// The adapter's name tables (Names), from wasm/tools/record_build_state.py's adapter.json
struct Tables {
    std::map<std::string, std::string> region_in, nation_in, face_in, card_in, char_in, board_keys;
    std::map<std::string, std::vector<std::string>> cards_of;          // "FREE_PEOPLES|character" -> engine card ids
    std::map<std::string, std::string> region_nation;                   // engine region key -> Nation member ("" None)
    std::vector<std::pair<std::string, int64_t>> vp_fp, vp_sh;           // game_logic._vp_settlement_tables
    std::vector<std::string> minion_keys, decks;
    std::set<std::string> special_tiles;
    std::map<std::string, std::string> card_out;                        // engine card id -> protocol key
    std::map<std::string, std::string> nation_out;                      // Nation member -> protocol nation
    std::map<std::string, bool> stronghold, fp_region;                  // board.regions is_stronghold, gl.is_fp_region
    std::map<std::string, std::set<std::string>> adjacent;              // BOARD.adjacent
    std::map<std::string, std::string> card_title, combat_effect_kind;  // card id -> title, combat.combat_effect_kind
    std::string hidden_cc;                                              // ai._HIDDEN_CC
    std::map<std::string, std::string> face_by_value, nation_value;     // ActionDieFace value -> member; Nation member -> value
    void load(const awjson::Value& j);
    std::string card(const std::string& key) const { auto it = card_in.find(key); return it == card_in.end() ? "" : it->second; }
};

// GameMemo: what the bot remembers of a game beyond the view
struct Memo {
    int side = 0;                       // 0 FP, 1 SH
    void* rng = nullptr;                // random.Random(seed) (wotr_random handle; owned by the caller)
    bool messenger_used = false, fellowship_acted = false, in_mordor = false;
};

class Adapter {
public:
    // prototypes: the decoded prototypes graph (record_build_state.py) in `model`
    Adapter(pyobj::Model& model, const Tables& tables, pyobj::Obj* prototypes);
    // build_state(view, memo, names): a GameState for the view (the model's object)
    pyobj::Obj* build_state(const awjson::Value& view, Memo& memo);
    // SETUPS[kind](bot, st, dec, view): the question set up on the state; false when the kind is not ported yet.
    // label / flag: the search label and the phase flag ('' none) the Python returns
    // match: the set-up answered by a rule (Match(o)): the option's 'o' (or none when has_match and o is None)
    struct Setup { bool ported = false; std::string label, flag; bool has_match = false; bool match_none = false; int64_t match = 0; };
    Setup setup(const std::string& kind, pyobj::Obj* st, const awjson::Value& dec, const awjson::Value& view, const Memo& memo);
    // an engine row of the box's menu: die, action type, parameters (an Enum as "\x01enum:MEMBER")
    // menu, index: the box's menu snapshot and the row's index in it (the plans that apply the action on a copy)
    struct Row { int64_t die = -1; std::string type; awjson::Value params; const void* menu = nullptr; int64_t index = -1; };
    const void* obs_tables = nullptr;   // the box's prepared observation tables (for the plans that apply an action)
    // map_action's Match: ok (an option matched; o_none: its 'o' is None), has_plan, chain; unported: not yet
    struct Matched { bool ok = false, o_none = false, has_plan = false, chain = false, unported = false; int64_t o = -1; };
    Matched map_action(const std::string& kind, pyobj::Obj* st, const awjson::Value& dec, const Row& row, const Memo& memo);
    std::string slug(const std::string& key) const;

private:
    pyobj::Model& m;
    const Tables& t;
    pyobj::Obj* protos;
    pyobj::Obj* proto(const std::string& name) const;
    pyobj::Obj* side_enum(int side) { return m.enumv("game_state.Side", side ? "SHADOW" : "FREE_PEOPLES"); }
    pyobj::Obj* nation_enum(const std::string& member) { return m.enumv("game_state.Nation", member); }
    pyobj::Obj* region_object(pyobj::Obj* st, const std::string& name);
    pyobj::Obj* get_or_create_army(pyobj::Obj* st, const std::string& key);
    pyobj::Obj* character_figure(const std::string& key, const std::map<std::string, pyobj::Obj*>& by_name);
    pyobj::Obj* tile(const std::string& key);
    void build_cards(pyobj::Obj* st, const awjson::Value& view, Memo& memo);
    std::vector<std::string> shuffled(Memo& memo, std::vector<std::string> items);
    pyobj::Obj* army_at(pyobj::Obj* st, const std::string& key);
    std::string fellowship_region(pyobj::Obj* st);
    std::string auto_mordor_gateway(pyobj::Obj* st);
    int leader_side(pyobj::Obj* leader);
    pyobj::Obj* sail_in(pyobj::Obj* st, const awjson::Value& b, const std::string& frm, const std::string& reg, int atk);
    pyobj::Obj* battle(pyobj::Obj* st, const awjson::Value& view, const Memo& memo, const std::string& phase, int64_t rounds_fought);
    static std::string pstr_of(const awjson::Value& v, const char* k);
    struct Figures { std::map<std::pair<std::string, std::string>, int64_t> units; std::set<std::string> chars; };
    Figures figures(pyobj::Obj* st, const std::string& key, int side);
    // army_move_plan: (ok, everything, from slug, to slug); ok false = None
    struct MovePlan { bool ok = false, everything = false; };
    MovePlan army_move_plan(pyobj::Obj* st, const Row& a, const Memo& memo);
    void choice_tokens(const awjson::Value& v, std::set<std::pair<std::string, std::string>>& out);
    int64_t score_options(const awjson::Value& dec, const std::set<std::pair<std::string, std::string>>& toks, bool empty);
    Setup setup_card_choice(pyobj::Obj* st, const awjson::Value& dec, const awjson::Value& view, const Memo& memo);
};

}  // namespace awad
