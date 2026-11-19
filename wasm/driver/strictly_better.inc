// THE SAME BUT BETTER and THE SIEGE BAN at the root, line for line with src/ai.py:
//   _strictly_better (18431), _outcome (18306), _at_least_as_good (18360), _sieges_held (18400), _siege_left (18418),
//   _apply_action_detect_rng (15406)
// on the live box's graphs (compiled into driver/box_unity.cpp after snapshot.cpp, whose helpers it uses):
//   apply_action + _resume_preaction_for_search  = apply_native_random_descriptor + sr_resume_preaction (the step the
//                                                  native search itself makes for a root child)
//   "consumed a random number"                   = the stream's state differs after the step (random.getstate() !=)
//   _state_sig(state, parts=True)                = sm_Builder::state_key(false) (search_metadata.inc, its native twin)
// The positions' "rest" is compared by a canonical text of its graph values; the quantities as Python's Counters.
namespace awsb {

constexpr int64_t BANNED = -2;
constexpr int SIEGE_BAN_ARMY = 5;                                         // ai._SIEGE_BAN_ARMY
const std::set<size_t> SB_BLANK = {1, 2, 7, 8, 11, 14, 15, 23, 34, 35, 36, 37};   // ai._SB_BLANK

// a canonical text of a graph value (equal text <=> equal value as Python compares the key's tuples)
void canon(const Snapshot& s, uint64_t n, std::string& o, int depth = 0) {
    require(depth < 400, "same-but-better key nested too deeply");
    if (n == missing) { o += 'N'; return; }
    const auto& v = s.nodes[n];
    switch (v.tag) {
    case WOTR_NONE: o += 'N'; return;
    case WOTR_BOOL: case WOTR_INT: o += 'i'; o += std::to_string(v.integer); return;   // True == 1 in Python
    case WOTR_FLOAT: { char b[40]; std::snprintf(b, sizeof b, "f%.17g", v.real); o += b; return; }
    case WOTR_STRING: case WOTR_BYTES: case WOTR_BIGINT:
        o += v.tag == WOTR_STRING ? 's' : v.tag == WOTR_BYTES ? 'b' : 'i';
        o += std::to_string(v.count); o += ':';
        o.append(reinterpret_cast<const char*>(s.bytes.data() + v.offset), size_t(v.count)); return;
    default: {
        o += 'c'; o += std::to_string(v.tag); o += '(';
        for (uint64_t k = 0; k < v.count; ++k) { canon(s, s.edges[v.offset + k], o, depth + 1); o += ','; }
        o += ')';
    }
    }
}

int nation_side(const Snapshot& s, uint64_t nation) {
    return fp_nation(validate_nation_name(enum_member(s, nation, "game_state.Nation"))) ? 0 : 1;
}
std::string member(const Snapshot& s, uint64_t n) {             // an enum's member name, or the text of a str
    if (ctx_none(s, n)) return "None";
    if (s.nodes[n].tag == WOTR_ENUM) return std::string(text(s, s.edges[s.nodes[n].offset + 1]));
    if (s.nodes[n].tag == WOTR_STRING) return std::string(text(s, n));
    if (s.nodes[n].tag == WOTR_INT || s.nodes[n].tag == WOTR_BOOL) return std::to_string(s.nodes[n].integer);
    std::string o; canon(s, n, o); return o;
}
int64_t number(const Snapshot& s, uint64_t n) {
    if (ctx_none(s, n)) return 0;
    const auto& v = s.nodes[n];
    require(v.tag == WOTR_INT || v.tag == WOTR_BOOL, "same-but-better quantity must be an integer");
    return v.integer;
}

using Counter = std::map<std::string, int64_t>;
using Board = std::map<std::string, Counter>;
struct Quantities {
    Board own, enemy;
    Counter own_dice, enemy_dice, own_hand, gone;
    int64_t own_dice_all = 0, enemy_dice_all = 0, own_vp = 0, enemy_vp = 0, corruption = 0;
    int64_t progress[2] = {0, 0}, rings[4] = {0, 0, 0, 0};
    bool operator==(const Quantities& b) const {
        return own == b.own && enemy == b.enemy && own_dice == b.own_dice && enemy_dice == b.enemy_dice &&
               own_hand == b.own_hand && gone == b.gone && own_dice_all == b.own_dice_all &&
               enemy_dice_all == b.enemy_dice_all && own_vp == b.own_vp && enemy_vp == b.enemy_vp &&
               corruption == b.corruption && progress[0] == b.progress[0] && progress[1] == b.progress[1] &&
               rings[0] == b.rings[0] && rings[1] == b.rings[1] && rings[2] == b.rings[2] && rings[3] == b.rings[3];
    }
};

uint64_t hands_of(const Snapshot& s, int side) {
    const auto e = enum_value_edge(s, obs_attr(s, s.root, "hands"), "game_state.Side", side ? "SHADOW" : "FREE_PEOPLES");
    require(e != missing, "same-but-better: hand missing");
    return s.edges[e];
}

// ai._outcome -> false when the engine cannot key the position (None)
bool outcome(const Snapshot& s, int side, std::string& rest, Quantities& q) {
    sm_Builder b(s);
    const auto key = b.state_key(false);
    if (ctx_none(b.out, key)) return false;
    const auto parts = child_edges(b.out, key);
    require(parts.size() >= 38, "_state_sig key layout changed: _SB_BLANK must follow it");
    const int other = 1 - side;
    rest.clear();
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        if (SB_BLANK.count(i)) rest += 'N'; else canon(b.out, parts[i], rest);
        rest += ';';
    }
    const auto pub = child_edges(b.out, parts.back());
    rest += '|';
    for (size_t i = 0; i < pub.size(); ++i) if (i != 5) { canon(b.out, pub[i], rest); rest += ';'; }
    rest += '|';
    std::vector<std::string> other_hand;
    for (auto c : child_edges(s, obs_attr(s, hands_of(s, other), "hand"))) other_hand.push_back(b.string(c));
    std::sort(other_hand.begin(), other_hand.end());
    for (auto& c : other_hand) { rest += std::to_string(c.size()); rest += ':'; rest += c; }
    rest += '|';
    for (int sd = 0; sd < 2; ++sd) {
        const auto fd = obs_attr(s, hands_of(s, sd), "discarded_face_down", false);
        rest += std::to_string(ctx_none(s, fd) ? 0 : child_edges(s, fd).size()); rest += ';';
    }
    // the quantities
    q = Quantities();
    Board board[2];
    const auto armies = obs_attr(s, s.root, "armies");
    const auto entries = child_edges(s, armies);
    for (size_t i = 0; i + 1 < entries.size(); i += 2) {
        const std::string where = sanitize_name(b.string(entries[i]));
        const auto army = entries[i + 1];
        for (auto u : child_edges(s, obs_attr(s, army, "units"))) {
            const auto nation = obs_attr(s, u, "nation");
            board[nation_side(s, nation)][where][member(s, nation) + "/" + member(s, obs_attr(s, u, "kind"))] += 1;
        }
        for (auto l : child_edges(s, obs_attr(s, army, "leaders"))) {
            const auto nation = s.nodes[l].tag == WOTR_OBJECT ? obs_attr(s, l, "nation", false) : missing;
            const int owner = !ctx_none(s, nation) ? nation_side(s, nation)
                              : (object_type(s, l) == "game_state.Companion" ? 0 : 1);
            board[owner][where][b.leader(l)] += 1;
        }
    }
    q.own = board[side]; q.enemy = board[other];
    std::set<uint64_t> in_hunt;
    const auto hunt_box = obs_attr(s, s.root, "hunt_box", false);
    if (!ctx_none(s, hunt_box)) for (auto d : child_edges(s, hunt_box)) in_hunt.insert(d);
    auto to_use = [&](int sd, Counter& c, int64_t& all) {
        const auto dice_dict = obs_attr(s, s.root, "dice");
        const auto e = enum_value_edge(s, dice_dict, "game_state.Side", sd ? "SHADOW" : "FREE_PEOPLES");
        if (e == missing) { all = 0; return; }
        const auto list = child_edges(s, s.edges[e]);
        all = int64_t(list.size());
        for (auto d : list) {
            const auto current = obs_attr(s, d, "current", false);
            if (truth(s, obs_attr(s, d, "used")) || ctx_none(s, current) || in_hunt.count(d)) continue;
            c[member(s, current)] += 1;
        }
    };
    to_use(side, q.own_dice, q.own_dice_all);
    to_use(other, q.enemy_dice, q.enemy_dice_all);
    for (auto c : child_edges(s, obs_attr(s, hands_of(s, side), "hand"))) q.own_hand[b.string(c)] += 1;
    const auto vp = obs_attr(s, s.root, "vp");
    const int64_t vfp = number(s, obs_attr(s, vp, "free_peoples")), vsh = number(s, obs_attr(s, vp, "shadow"));
    q.own_vp = side == 0 ? vfp : vsh; q.enemy_vp = side == 0 ? vsh : vfp;
    const auto cas = obs_attr(s, s.root, "fp_casualties", false);
    if (truth(s, cas)) {
        const auto ce = child_edges(s, cas);
        for (size_t i = 0; i + 1 < ce.size(); i += 2) {
            if (!truth(s, ce[i + 1])) continue;
            const auto kinds = child_edges(s, ce[i + 1]);
            for (size_t j = 0; j + 1 < kinds.size(); j += 2) {
                const int64_t v = number(s, kinds[j + 1]);
                if (v) q.gone[member(s, ce[i]) + "/" + member(s, kinds[j])] = v;
            }
        }
    }
    const auto f = obs_attr(s, s.root, "fellowship");
    q.corruption = number(s, obs_attr(s, f, "corruption"));
    q.progress[0] = number(s, obs_attr(s, f, "progress")); q.progress[1] = number(s, obs_attr(s, f, "mordor_step"));
    const int64_t fpr = number(s, obs_attr(s, s.root, "fp_elven_rings")), shr = number(s, obs_attr(s, s.root, "sh_elven_rings"));
    const bool fpu = truth(s, obs_attr(s, s.root, "fp_ring_used_this_turn")), shu = truth(s, obs_attr(s, s.root, "sh_ring_used_this_turn"));
    const bool fp = side == 0;
    q.rings[0] = fp ? fpr : shr; q.rings[1] = -(fp ? shr : fpr);
    q.rings[2] = !(fp ? fpu : shu); q.rings[3] = fp ? shu : fpu;
    return true;
}

bool within(const Counter& x, const Counter& y) {
    for (auto& [k, v] : x) { auto it = y.find(k); if ((it == y.end() ? 0 : it->second) < v) return false; }
    return true;
}
bool board_within(const Board& x, const Board& y) {
    static const Counter empty;
    for (auto& [r, c] : x) { auto it = y.find(r); if (!within(c, it == y.end() ? empty : it->second)) return false; }
    return true;
}
// ai._at_least_as_good: a at least as good as b on every quantity and better on one
bool at_least_as_good(int side, const Quantities& a, const Quantities& b) {
    const bool fp = side == 0;
    bool ok = board_within(b.own, a.own) && board_within(a.enemy, b.enemy) && within(b.own_dice, a.own_dice) &&
              within(a.enemy_dice, b.enemy_dice) && a.own_dice_all >= b.own_dice_all &&
              a.enemy_dice_all <= b.enemy_dice_all && within(b.own_hand, a.own_hand) && a.own_vp >= b.own_vp &&
              a.enemy_vp <= b.enemy_vp && (fp ? within(a.gone, b.gone) : within(b.gone, a.gone)) &&
              (fp ? a.corruption <= b.corruption : a.corruption >= b.corruption);
    for (int i = 0; ok && i < 2; ++i) ok = fp ? a.progress[i] >= b.progress[i] : a.progress[i] <= b.progress[i];
    for (int i = 0; ok && i < 4; ++i) ok = a.rings[i] >= b.rings[i];
    return ok && !(a == b);
}

std::set<std::string> besieged(const Snapshot& s) {
    std::set<std::string> out;
    const auto bs = obs_attr(s, s.root, "besieged_strongholds", false);
    if (!ctx_none(s, bs)) for (auto c : child_edges(s, bs)) out.insert(sanitize_name(text(s, c)));
    return out;
}
// ai._sieges_held (the Shadow's sieges it may not walk away from)
std::vector<std::string> sieges_held(const Snapshot& s, const Snapshot& t, int side) {
    std::vector<std::string> held;
    const auto sieged = besieged(s);
    if (side != 1 || sieged.empty()) return held;
    const auto entries = child_edges(s, obs_attr(s, s.root, "armies"));
    for (size_t i = 0; i + 1 < entries.size(); i += 2) {
        const std::string where = sanitize_name(text(s, entries[i]));
        if (!sieged.count(where) || crl_garrison_side(s, t, where) == side) continue;
        const auto army = entries[i + 1];
        std::vector<uint64_t> units;
        for (auto u : child_edges(s, obs_attr(s, army, "units"))) if (nation_side(s, obs_attr(s, u, "nation")) == side) units.push_back(u);
        size_t figures = 0;
        for (auto l : child_edges(s, obs_attr(s, army, "leaders"))) {
            const auto nation = s.nodes[l].tag == WOTR_OBJECT ? obs_attr(s, l, "nation", false) : missing;
            const bool mine = !ctx_none(s, nation) ? nation_side(s, nation) == side : object_type(s, l) != "game_state.Companion";
            figures += mine;
        }
        if (units.size() >= size_t(SIEGE_BAN_ARMY) ||
            (units.size() == 1 && figures == 0 && enum_is(s, obs_attr(s, units[0], "kind"), "game_state.UnitKind", "REGULAR")))
            held.push_back(where);
    }
    return held;
}
// ai._siege_left
bool siege_left(const Snapshot& after, int side, const std::vector<std::string>& held) {
    const auto sieged = besieged(after);
    const auto entries = child_edges(after, obs_attr(after, after.root, "armies"));
    for (const auto& where : held) {
        if (sieged.count(where)) continue;
        bool stands = false;
        for (size_t i = 0; i + 1 < entries.size() && !stands; i += 2) {
            if (sanitize_name(text(after, entries[i])) != where) continue;
            for (auto u : child_edges(after, obs_attr(after, entries[i + 1], "units")))
                if (nation_side(after, obs_attr(after, u, "nation")) == side) { stands = true; break; }
        }
        if (!stands) return true;
    }
    return false;
}

std::vector<uint32_t> rng_words(const void* rng) {
    std::vector<uint32_t> w(627);
    uint32_t index; int32_t g; uint64_t bits;
    wotr_random_export(rng, w.data(), 624, &index, &g, &bits);
    w[624] = index; w[625] = uint32_t(g); w[626] = uint32_t(bits ^ (bits >> 32));
    return w;
}

// ai._strictly_better over the rows of `menu` (the root children). Returns false for None.
bool strictly_better(const Snapshot& state, const Snapshot& t, const Snapshot& menu, int side, std::vector<int64_t>& better) {
    const auto rows = child_edges(menu, menu.root);
    const size_t k = rows.size();
    better.assign(k, -1);
    std::string root_rest; Quantities root_q;
    if (k < 2 || !outcome(state, side, root_rest, root_q)) return false;
    const auto held = sieges_held(state, t, side);
    std::vector<std::pair<std::string, std::vector<std::pair<size_t, Quantities>>>> seen;   // insertion order
    std::set<size_t> banned;
    const uint32_t seed_limb = 20261117u;
    for (size_t i = 0; i < k; ++i) {
        try {
            std::unique_ptr<void, void (*)(void*)> rng(wotr_random_create_seeded(&seed_limb, 1), wotr_random_destroy);
            const auto before = rng_words(rng.get());
            const std::string kind(text(menu, child_edges(menu, rows[i])[1]));
            auto applied = apply_native_random_descriptor(state, t, menu, rng.get(), side, int64_t(i));
            auto settled = sr_resume_preaction(state, *applied.state, t, applied.random.get(), kind);
            if (rng_words(settled.random.get()) != before) continue;               // rolled: no part of any verdict
            const Snapshot& after = *settled.state;
            if (!held.empty() && siege_left(after, side, held)) { banned.insert(i); continue; }
            std::string rest; Quantities q;
            if (!outcome(after, side, rest, q)) continue;
            auto it = std::find_if(seen.begin(), seen.end(), [&](auto& g) { return g.first == rest; });
            if (it == seen.end()) { seen.push_back({rest, {}}); it = seen.end() - 1; }
            it->second.push_back({i, std::move(q)});
        } catch (...) {
            continue;                    // an option that cannot be applied here is no part of any verdict
        }
    }
    if (banned.size() < k) for (size_t i : banned) better[i] = BANNED;
    for (auto& [rest, group] : seen) {
        if (group.size() < 2) continue;
        for (auto& [i, qb] : group) {
            std::vector<size_t> above;   // indices into group
            for (size_t g = 0; g < group.size(); ++g) if (group[g].first != i && at_least_as_good(side, group[g].second, qb)) above.push_back(g);
            if (above.empty()) continue;
            std::vector<size_t> top;
            for (size_t g : above) {
                bool beaten = false;
                for (size_t h = 0; h < group.size() && !beaten; ++h)
                    if (h != g && at_least_as_good(side, group[h].second, group[g].second)) beaten = true;
                if (!beaten) top.push_back(g);
            }
            better[i] = int64_t(group[(top.empty() ? above : top)[0]].first);
        }
    }
    for (auto x : better) if (x != -1) return true;
    return false;
}

}  // namespace awsb

extern "C" {
// ai._strictly_better on a root: the state, the observation tables, the root menu (rows in child order), the side.
// 1 = verdicts written to out[k] (k = the menu's rows), 0 = None, -1 = error (wotr_snapshot_last_error).
WOTR_API int awasm_strictly_better(const void* state, const void* tables, const void* menu, int32_t side, int64_t* out, size_t k) {
    return checked<int>([&]() {
        std::vector<int64_t> better;
        const bool any = awsb::strictly_better(snapshot(state), snapshot(tables), snapshot(menu), side, better);
        require(better.size() == k, "strictly_better: the output size differs from the menu");
        for (size_t i = 0; i < k; ++i) out[i] = better[i];
        return any ? 1 : 0;
    }, -1);
}
}
