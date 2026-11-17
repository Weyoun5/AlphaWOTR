// THE CERTAIN LINE (wotr_bridge/core/certain_line.py find, turn_search.certain + _same_but_better), line for line on
// the box's graphs (compiled into driver/box_unity.cpp after strictly_better.inc, whose outcome it uses):
//   ai._apply_action_detect_rng     apply_native_random_descriptor + sr_resume_preaction, the stream compared
//   ai._pending_decision_side        native_pending_actor(state, -1).side
//   ai._next_acting_side             native_next_actor
//   seats.menu (LeagueSearcher._menu) native_full_actions + project_native_ai_actions (the full menu when empty)
//   _pick (_greedy_prior_action)     the projection's first row when forced, else the highest prior (first of ties)
//   ai._outcome / _at_least_as_good  awsb::outcome / awsb::at_least_as_good (strictly_better.inc)
// The priors come from the module's network through a callback (the box carries no network).
// Bounds: APPLIES 2000 and MENUS 400 as in the Python; its 15-second wall-clock cut is not reproduced.
extern "C" {
// priors of the rows of `menu` (no projection) for `side`: out[n]; 0 on success
typedef int (*aw_priors_fn)(void* user, const void* state, int32_t side, const void* menu, float* out, size_t n);
}
namespace awcl {

constexpr int OWN_ACTIONS = 2, STEPS = 40;
constexpr uint64_t APPLIES = 2000, MENUS = 400;
using Snap = std::shared_ptr<const Snapshot>;

struct Ctx { const Snapshot& t; aw_priors_fn priors; void* user; };

struct Menu { Snap snap; size_t n = 0; };
size_t rows_of(const Snapshot& m) { return child_edges(m, m.root).size(); }
uint64_t row_node(const Menu& m, size_t i) { return child_edges(*m.snap, m.snap->root)[i]; }
std::string row_text(const Menu& m, size_t i) { std::string o; awsb::canon(*m.snap, row_node(m, i), o); return o; }
std::string row_kind(const Menu& m, size_t i) { return std::string(text(*m.snap, child_edges(*m.snap, row_node(m, i))[1])); }
uint64_t row_params(const Menu& m, size_t i) { return child_edges(*m.snap, row_node(m, i))[2]; }

// LeagueSearcher._menu: the projection of the legal menu, or the legal menu when the projection is empty
Menu menu_for(const Ctx& c, const Snapshot& s, int side) {
    auto full = native_full_actions(s, c.t, side);
    auto proj = project_native_ai_actions(s, *full, side);
    Menu m;
    if (proj.actions && rows_of(*proj.actions) > 0) m.snap = Snap(std::move(proj.actions)); else m.snap = Snap(std::move(full));
    m.n = rows_of(*m.snap);
    return m;
}

// _greedy_prior_action(state, side, net, menu) (decision=False): (the menu of the row, the row), row -1 when none
std::pair<Menu, int64_t> pick(const Ctx& c, const Snapshot& s, int side, const Menu& menu) {
    auto proj = project_native_ai_actions(s, *menu.snap, side);
    Menu sel; sel.snap = Snap(std::move(proj.actions)); sel.n = rows_of(*sel.snap);
    if (!proj.forced_reason.empty()) return {sel, sel.n ? 0 : -1};
    if (sel.n == 0) return {sel, -1};
    std::vector<float> p(sel.n);
    if (c.priors(c.user, &s, side, sel.snap.get(), p.data(), sel.n)) throw std::runtime_error("certain line: no priors");
    size_t best = 0;
    for (size_t i = 1; i < sel.n; ++i) if (p[i] > p[best]) best = i;
    return {sel, int64_t(best)};
}

// ai._apply_action_detect_rng: (the state after, rolled)
std::pair<Snap, bool> apply(const Ctx& c, const Snapshot& s, const Menu& m, size_t row, int side) {
    const uint32_t limb = 7;
    std::unique_ptr<void, void (*)(void*)> rng(wotr_random_create_seeded(&limb, 1), wotr_random_destroy);
    const auto before = awsb::rng_words(rng.get());
    auto applied = apply_native_random_descriptor(s, c.t, *m.snap, rng.get(), side, int64_t(row));
    auto settled = sr_resume_preaction(s, *applied.state, c.t, applied.random.get(), row_kind(m, row));
    const bool rolled = awsb::rng_words(settled.random.get()) != before;
    return {Snap(std::move(settled.state)), rolled};
}

// _decider: (the side that decides next, whether a new die action starts); side -1 at the end
std::pair<int, bool> decider(const Snapshot& s, int owner) {
    if (truth(s, obs_attr(s, s.root, "game_over", false)) || !truth(s, obs_attr(s, s.root, "dice_rolled", false))) return {-1, false};
    const auto p = native_pending_actor(s, -1);
    if (p.side >= 0) return {p.side, false};
    if (!ctx_none(s, obs_attr(s, s.root, "pending_army_move_die", false)) || !ctx_none(s, obs_attr(s, s.root, "pending_muster_die", false)))
        return {owner, false};
    return {owner < 0 ? -1 : native_next_actor(s, owner), true};
}

struct End { bool ok = false; std::string key; awsb::Quantities q; };
// _end: the line's end as a position (the rest with who decides next, the counts); the routing mark of a resolved
// battle replaced by what it decides
End end(const Snapshot& s, int side, int owner) {
    End e;
    const auto mark = obs_attr(s, s.root, "_battle_resolved_attacker", false);
    std::string rest;
    if (!ctx_none(s, mark)) {
        Snapshot plain(s);
        rr_set_attr(plain, plain.root, "_battle_resolved_attacker", rr_scalar(plain, WOTR_NONE));
        if (!awsb::outcome(plain, side, rest, e.q)) return e;
    } else if (!awsb::outcome(s, side, rest, e.q)) return e;
    e.key = rest + "#" + std::to_string(decider(s, owner).first);
    e.ok = true;
    return e;
}

std::vector<std::string> regions(const Menu& m, size_t i) {
    const auto& s = *m.snap;
    const auto p = row_params(m, i);
    std::vector<std::string> out;
    if (s.nodes[p].tag != WOTR_DICT) return out;
    for (const char* k : {"from", "to", "region"}) {
        const auto v = field(s, p, k, false);
        if (v == missing || ctx_none(s, v)) continue;
        out.push_back(sanitize_name(s.nodes[v].tag == WOTR_STRING ? std::string(text(s, v)) : awsb::member(s, v)));
    }
    return out;
}
int64_t param_int(const Menu& m, size_t i, const char* k, bool* present = nullptr) {
    const auto& s = *m.snap;
    const auto p = row_params(m, i);
    const auto v = s.nodes[p].tag == WOTR_DICT ? field(s, p, k, false) : missing;
    if (present) *present = v != missing;
    if (v == missing || ctx_none(s, v)) return 0;
    return s.nodes[v].tag == WOTR_INT || s.nodes[v].tag == WOTR_BOOL ? s.nodes[v].integer : 0;
}

struct Reference { bool ok = false; End key; bool roll = false, decision = false; int begun = 1; std::vector<std::string> replies; };

// _reference: the pick, then each side's own net, to the end of the mover's next die action
Reference reference(const Ctx& c, const Snapshot& state, int side, const Menu& menu, size_t own) {
    Reference r;
    auto [cur, rolled] = apply(c, state, menu, own, side);
    r.roll = rolled;
    int owner = side, begun = 1;
    bool finished = false;
    for (int step = 0; step < STEPS; ++step) {
        const auto [d, fresh] = decider(*cur, owner);
        if (d < 0) { finished = true; break; }
        if (fresh) {
            if (begun == OWN_ACTIONS) { finished = true; break; }
            begun += d == side;
            owner = d;
        }
        const Menu m = menu_for(c, *cur, d);
        Menu am = m; int64_t act = -1;
        if (m.n == 1) act = 0;
        else if (m.n > 1) { auto pk = pick(c, *cur, d, m); am = pk.first; act = pk.second; }
        if (act < 0 || (d == side && owner != side)) return r;
        if (d != side && owner != side) r.replies.push_back(row_text(am, size_t(act)));
        else if (d != side && m.n > 1 && !r.roll && !r.decision) r.decision = true;
        auto next = apply(c, *cur, am, size_t(act), d);
        cur = next.first;
        if (next.second) {
            if (owner != side) return r;
            r.roll = true; r.decision = false;
        }
    }
    if (!finished) return r;
    r.key = end(*cur, side, owner);
    r.begun = begun;
    r.ok = r.key.ok;
    return r;
}

struct Search {
    const Ctx& c; int side; const Reference& ref;
    const awsb::Board& end_own; bool strict;
    uint64_t applies = 0, menus = 0; bool cut = false;
    Search(const Ctx& c_, int side_, const Reference& ref_) : c(c_), side(side_), ref(ref_), end_own(ref_.key.q.own),
        strict(!ref_.roll && !ref_.decision) {}
    bool out() { if (applies >= APPLIES || menus >= MENUS) cut = true; return cut; }
    awsb::Board own(const Snapshot& s) { std::string rest; awsb::Quantities q; if (!awsb::outcome(s, side, rest, q)) throw std::runtime_error("unkeyed"); return q.own; }
    bool reached(const Snapshot& s, int owner) {
        const End got = end(s, side, owner);
        if (!got.ok || got.key != ref.key.key) return false;
        return got.q == ref.key.q ? !strict : awsb::at_least_as_good(side, got.q, ref.key.q);
    }
    static int64_t kind_count(const awsb::Counter& c, const std::string& kind) {
        int64_t n = 0;
        for (const auto& [k, v] : c) { const auto slash = k.find('/'); if (slash != std::string::npos && k.substr(slash + 1) == kind) n += v; }
        return n;
    }
    // picked: the origin army's roster ordered so that a counted split takes the units the end has at the destination
    Snap picked(const Snap& state, const Menu& m, size_t i, const awsb::Board& now) {
        const auto& ps = *m.snap; const auto p = row_params(m, i);
        if (ps.nodes[p].tag != WOTR_DICT) return state;
        const auto fv = field(ps, p, "from", false), tv = field(ps, p, "to", false);
        if (fv == missing || tv == missing || ctx_none(ps, fv) || ctx_none(ps, tv)) return state;
        const std::string src = sanitize_name(text(ps, fv)), dst = sanitize_name(text(ps, tv));
        const Snapshot& s = *state;
        const auto entries = child_edges(s, obs_attr(s, s.root, "armies"));
        uint64_t army = missing;
        for (size_t j = 0; j + 1 < entries.size(); j += 2) if (sanitize_name(text(s, entries[j])) == src) { army = entries[j + 1]; break; }
        std::vector<uint64_t> units = army == missing ? std::vector<uint64_t>{} : child_edges(s, obs_attr(s, army, "units"));
        bool split = false;
        for (const auto& [kind, count] : std::vector<std::pair<std::string, const char*>>{{"REGULAR", "n_regular"}, {"ELITE", "n_elite"}}) {
            const int64_t n = param_int(m, i, count);
            int64_t of_kind = 0; std::set<std::string> nations;
            for (auto u : units) if (awsb::member(s, obs_attr(s, u, "kind")) == kind) { ++of_kind; nations.insert(awsb::member(s, obs_attr(s, u, "nation"))); }
            if (0 < n && n < of_kind && nations.size() > 1) split = true;
        }
        if (!split) return state;
        std::map<std::string, int64_t> want;
        auto e = end_own.find(dst); if (e != end_own.end()) for (const auto& [k, v] : e->second) want[k] += v;
        auto w = now.find(dst); if (w != now.end()) for (const auto& [k, v] : w->second) want[k] -= v;
        auto clone = std::make_shared<Snapshot>(s);
        const auto list = obs_attr(*clone, army, "units");
        std::vector<uint64_t> first, rest;
        for (auto u : units) {
            const std::string k = awsb::member(s, obs_attr(s, u, "nation")) + "/" + awsb::member(s, obs_attr(s, u, "kind"));
            auto it = want.find(k);
            if (it != want.end() && it->second > 0) { --it->second; first.push_back(u); } else rest.push_back(u);
        }
        first.insert(first.end(), rest.begin(), rest.end());
        const auto& node = clone->nodes[list];
        for (size_t j = 0; j < first.size(); ++j) clone->edges[node.offset + j] = first[j];
        return clone;
    }
    bool fits(const Menu& m, size_t i, const std::vector<std::string>& regs, const awsb::Board& now) {
        bool hr = false, he = false;
        param_int(m, i, "n_regular", &hr); param_int(m, i, "n_elite", &he);
        const auto& ps = *m.snap; const auto p = row_params(m, i);
        const bool has_from = ps.nodes[p].tag == WOTR_DICT && field(ps, p, "from", false) != missing && !ctx_none(ps, field(ps, p, "from", false));
        const bool has_to = ps.nodes[p].tag == WOTR_DICT && field(ps, p, "to", false) != missing && !ctx_none(ps, field(ps, p, "to", false));
        if (!has_from || !has_to || !(hr || he)) return true;
        auto count = [&](const awsb::Board& b, const std::string& r) {
            auto it = b.find(r);
            if (it == b.end()) return std::array<int64_t, 2>{0, 0};
            return std::array<int64_t, 2>{kind_count(it->second, "REGULAR"), kind_count(it->second, "ELITE")};
        };
        const std::array<int64_t, 2> moved{param_int(m, i, "n_regular"), param_int(m, i, "n_elite")};
        const auto a0 = count(now, regs[0]), e0 = count(end_own, regs[0]);
        const auto a1 = count(now, regs[1]), e1 = count(end_own, regs[1]);
        return (a0[0] - moved[0] == e0[0] && a0[1] - moved[1] == e0[1]) || (a1[0] + moved[0] == e1[0] && a1[1] + moved[1] == e1[1]);
    }
    static const awsb::Counter& at_region(const awsb::Board& b, const std::string& r) {
        static const awsb::Counter empty;
        auto it = b.find(r); return it == b.end() ? empty : it->second;
    }
    // steps: the own options worth a try here, each applied
    std::vector<std::pair<size_t, Snap>> steps(const Snap& state, const Menu& menu) {
        const awsb::Board now = own(*state);
        std::set<std::string> differ;
        for (const auto& [r, c2] : now) if (c2 != at_region(end_own, r)) differ.insert(r);
        for (const auto& [r, c2] : end_own) if (c2 != at_region(now, r)) differ.insert(r);
        std::vector<std::pair<size_t, Snap>> out;
        for (size_t i = 0; i < menu.n; ++i) {
            const auto regs = regions(menu, i);
            const bool pass = row_kind(menu, i) == "pass";
            if (!pass) {
                bool all_differ = !regs.empty();
                for (const auto& r : regs) all_differ &= differ.count(r) != 0;
                if (!(all_differ && fits(menu, i, regs, now))) continue;
            }
            if (out_()) break;
            ++applies;
            std::pair<Snap, bool> after;
            try { after = apply(c, *picked(state, menu, i, now), menu, i, side); } catch (...) { continue; }
            if (after.second) continue;
            if (!pass) {
                const awsb::Board board = own(*after.first);
                bool sets = false;
                for (const auto& r : regs) sets |= at_region(board, r) == at_region(end_own, r);
                if (!sets) continue;
            }
            out.push_back({i, after.first});
        }
        return out;
    }
    bool out_() { return out(); }
    bool walk(Snap state, int owner, int begun, size_t ri) {
        for (int step = 0; step < STEPS; ++step) {
            if (out()) return false;
            const auto [d, fresh] = decider(*state, owner);
            const bool done = begun == ref.begun && ri == ref.replies.size();
            if (d < 0 || (fresh && done)) return done && reached(*state, owner);
            if (fresh) {
                if (d == side) { if (begun == ref.begun) return false; ++begun; }
                else if (ri == ref.replies.size()) return false;
                owner = d;
            }
            ++menus;
            const Menu menu = menu_for(c, *state, d);
            if (menu.n == 0) return false;
            if (d == side) {
                if (owner != side) return false;
                for (auto& [i, after] : steps(state, menu)) { (void)i; if (walk(after, owner, begun, ri)) return true; }
                return false;
            }
            size_t act = 0;
            if (owner == side) { if (menu.n > 1) return false; act = 0; }
            else {
                if (ri >= ref.replies.size()) return false;
                bool found = false;
                for (size_t i = 0; i < menu.n; ++i) if (row_text(menu, i) == ref.replies[ri]) { act = i; found = true; break; }
                if (!found) return false;
                ++ri;
            }
            ++applies;
            auto next = apply(c, *state, menu, act, d);
            if (next.second) return false;
            state = next.first;
        }
        return false;
    }
};

// certain_line.find + turn_search.certain's outright verdict with _same_but_better: -> the row of `menu` to play
// outright, or -1
int64_t certain(const Ctx& c, const Snapshot& state, int side, const Menu& menu, size_t own) {
    if (menu.n < 2 || !truth(state, obs_attr(state, state.root, "dice_rolled", false))) return -1;
    const int pending = native_pending_actor(state, -1).side;
    if (pending != -1 && pending != side) return -1;
    try {
        const Reference ref = reference(c, state, side, menu, own);
        if (!ref.ok) return -1;
        Search search(c, side, ref);
        auto root = std::make_shared<Snapshot>(state);
        std::vector<size_t> found;
        for (auto& [i, after] : search.steps(root, menu)) if (search.walk(after, side, 1, 0)) found.push_back(i);
        if (found.empty()) return -1;
        const std::string own_text = row_text(menu, own);
        for (size_t i : found) if (row_text(menu, i) == own_text) return -1;
        if (ref.roll) return -1;                                  // 'favoured': the turn search's (not at strength 1)
        std::vector<float> p(menu.n);
        if (c.priors(c.user, &state, side, menu.snap.get(), p.data(), menu.n)) return -1;
        size_t best = found[0];
        for (size_t i : found) if (p[i] > p[best]) best = i;
        // _same_but_better(state, side, menu, hit, own): the whole menu (up to 400 rows), else [hit, own]
        std::vector<int64_t> verdicts;
        size_t at = best;
        std::vector<int64_t> better;
        bool any = false;
        if (menu.n <= 400) any = awsb::strictly_better(state, c.t, *menu.snap, side, better);
        else {
            Snapshot pair(*menu.snap);
            const auto rows = child_edges(*menu.snap, menu.snap->root);
            const uint64_t list = pair.nodes.size();
            WotrValue v{}; v.tag = WOTR_LIST; v.offset = pair.edges.size(); v.count = 2;
            pair.nodes.push_back(v); pair.edges.push_back(rows[best]); pair.edges.push_back(rows[own]); pair.root = list;
            any = awsb::strictly_better(state, c.t, pair, side, better);
            at = 0;
        }
        if (any && better[at] == awsb::BANNED) return -1;         // keeps the own pick
        size_t moved = at;
        if (any) { std::set<size_t> walked; while (moved < better.size() && better[moved] >= 0 && !walked.count(moved)) { walked.insert(moved); moved = size_t(better[moved]); } }
        const size_t row = menu.n <= 400 ? moved : (moved == 0 ? best : own);
        if (row == own) return -1;
        return int64_t(row);
    } catch (...) {
        return -1;                                                // a line the engine cannot play is no verdict
    }
}

}  // namespace awcl

extern "C" {
// THE CERTAIN LINE at strength 1: -> the row of `menu` (the adapter's projected menu) to play outright instead of
// `own`, -1 for none, -2 on error (wotr_snapshot_last_error)
WOTR_API int64_t awasm_certain_line(const void* state, const void* tables, const void* menu, int32_t side, int64_t own,
                                    aw_priors_fn priors, void* user) {
    return checked<int64_t>([&]() -> int64_t {
        awcl::Ctx c{snapshot(tables), priors, user};
        awcl::Menu m;
        m.snap = awcl::Snap(std::make_shared<Snapshot>(snapshot(menu)));
        m.n = awcl::rows_of(*m.snap);
        require(own >= 0 && size_t(own) < m.n, "certain line: the own pick is not a row of the menu");
        return awcl::certain(c, snapshot(state), side, m, size_t(own));
    }, -2);
}
}
