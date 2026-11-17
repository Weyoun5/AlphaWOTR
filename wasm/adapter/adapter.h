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
    struct Setup { bool ported = false; std::string label, flag; };
    Setup setup(const std::string& kind, pyobj::Obj* st, const awjson::Value& dec, const awjson::Value& view, const Memo& memo);

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
};

}  // namespace awad
