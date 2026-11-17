#include "decision.h"
#include "aw_controller_abi.h"
#include "select.h"
#include "setup.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>

extern "C" void* awasm_graph_create(const uint8_t* p);
extern "C" void* awasm_combat_reply(const void* tables, const void* payload);
extern "C" int awasm_strictly_better(const void* state, const void* tables, const void* menu, int32_t side, int64_t* out, size_t k);
extern "C" int64_t awasm_snapshot_root_count(const void* handle);
extern "C" int64_t awasm_certain_line(const void* state, const void* tables, const void* menu, int32_t side, int64_t own,
                                      int (*priors)(void*, const void*, int32_t, const void*, float*, size_t), void* user);

extern "C" void* awasm_ai_units(const void* state, const void* tables, const void* menu, int32_t side, int64_t index, int32_t* changed);

namespace awdec {
namespace {

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

constexpr size_t OBS = 4745, ACT = 199;

// the live numeric bridge (numeric_bridge.NumericInferenceBridge.run) and the combat service, for the provider
struct Live {
    Box* box; int side; double eps; std::string error;
};

int provide(void* user, void* ctl) {
    Live& c = *static_cast<Live*>(user);
    WotrSearchRequest q{}; q.struct_size = sizeof(q);
    if (wotr_search_controller_request(ctl, &q)) { c.error = std::string("request: ") + wotr_snapshot_last_error(); return 1; }
    if (q.operation == awabi::OP_DIST) {
        void* payload = wotr_search_controller_request_payload(ctl);
        if (!payload) { c.error = std::string("request_payload: ") + wotr_snapshot_last_error(); return 1; }
        void* reply = awasm_combat_reply(c.box->obs, payload);
        wotr_snapshot_destroy(payload);
        if (!reply) { c.error = std::string("combat kernel: ") + wotr_snapshot_last_error(); return 1; }
        const int rc = wotr_search_controller_submit_payload(ctl, q.request_id, reply);
        wotr_snapshot_destroy(reply);
        if (rc) { c.error = std::string("submit_payload: ") + wotr_snapshot_last_error(); return 1; }
        return 0;
    }
    const awnet::Net& net = c.box->net[c.side];
    const awnet::Net& vnet = c.box->vnet[c.side];
    const size_t B = q.batch, R = q.rows;
    std::vector<double> values(B), conf(B, 1.0), mixed(R);
    std::vector<float> priors(R);
    for (size_t i = 0; i < B; ++i) {
        const float* obs = q.observations + i * OBS;
        awnet::Body bv;
        vnet.body(obs, bv, false);
        values[i] = double(vnet.value(bv));
        if (q.operation == awabi::OP_EVAL) continue;
        const size_t lo = q.offsets[i], hi = q.offsets[i + 1], n = hi - lo;
        awnet::Body b;
        net.body(obs, b, true);
        std::vector<float> logits(n);
        net.logits(b, q.actions + lo * ACT, q.regions + lo, n, logits.data(), true);
        awnet::Net::softmax(logits.data(), n, &priors[lo]);
        float sum = 0.0f;
        for (size_t j = lo; j < hi; ++j) sum += priors[j];
        const float denom = sum + 1e-10f;
        double total = 0.0;
        for (size_t j = lo; j < hi; ++j) { priors[j] = priors[j] / denom; total += double(priors[j]); }
        for (size_t j = lo; j < hi; ++j) mixed[j] = (1.0 - c.eps) * (double(priors[j]) / total) + c.eps / double(n);
    }
    WotrSearchReply r{}; r.struct_size = sizeof(r); r.request_id = q.request_id;
    r.values = values.data(); r.conf = conf.data();
    if (q.operation != awabi::OP_EVAL) { r.priors = priors.data(); r.mixed = mixed.data(); }
    if (wotr_search_controller_submit(ctl, &r)) { c.error = std::string("submit: ") + wotr_snapshot_last_error(); return 1; }
    return 0;
}

// a list of integers as a snapshot (native_menu_rows' index list)
void* int_list(const std::vector<int64_t>& xs) {
    std::vector<WotrValue> v(1 + xs.size());
    std::vector<uint64_t> e;
    std::memset(v.data(), 0, sizeof(WotrValue) * v.size());
    v[0].tag = WOTR_LIST; v[0].offset = 0; v[0].count = xs.size();
    for (size_t i = 0; i < xs.size(); ++i) { v[1 + i].tag = WOTR_INT; v[1 + i].integer = xs[i]; e.push_back(1 + i); }
    return wotr_snapshot_create(v.data(), v.size(), e.data(), e.size(), nullptr, 0, 0);
}

// project_ai_actions on a menu: (the selected rows, forced)
std::pair<void*, bool> project(const void* state, const void* menu, int side) {
    void* meta = nullptr;
    void* sel = wotr_snapshot_project_policy(state, menu, side, &meta);
    bool forced = false;
    if (meta) {
        const size_t nv = wotr_snapshot_node_count(meta);
        std::vector<WotrValue> v(nv); std::vector<uint64_t> e(wotr_snapshot_edge_count(meta)); std::vector<uint8_t> b(wotr_snapshot_byte_count(meta));
        wotr_snapshot_export(meta, v.data(), nv, e.data(), e.size(), b.data(), b.size());
        const auto& root = v[wotr_snapshot_root(meta)];
        forced = root.count >= 1 && v[e[root.offset]].tag != WOTR_NONE;   // (forced_reason, reasons)
        wotr_snapshot_destroy(meta);
    }
    return {sel, forced};
}

}  // namespace

bool Box::load(const std::string& tables_awg, const std::string& nets_dir, std::string& error) {
    auto tb = slurp(tables_awg);
    if (tb.size() < 8) { error = "cannot read " + tables_awg; return false; }
    uint32_t tc; std::memcpy(&tc, &tb[4], 4);
    size_t q = 8;
    for (uint32_t i = 0; i < tc; ++i) {
        uint32_t n; std::memcpy(&n, &tb[q], 4);
        const std::string name(reinterpret_cast<const char*>(&tb[q + 4]), n);
        const uint8_t* counts = &tb[q + 4 + n];
        uint64_t nv, ne, nb; std::memcpy(&nv, counts, 8); std::memcpy(&ne, counts + 8, 8); std::memcpy(&nb, counts + 16, 8);
        void* h = awasm_graph_create(counts);
        if (name == "observation") { obs = h; wotr_snapshot_prepare_observation_tables(obs); }
        else if (name == "actions_v53") { act = h; wotr_snapshot_prepare_action_tables_v53(act); }
        else if (name == "menu") menu_tables = h;
        else if (name == "hunt") hunt = h;
        q += 4 + n + 32 + 40 * nv + 8 * ne + nb;
    }
    if (!obs || !act || !menu_tables || !hunt) { error = "tables.awg lacks a table"; return false; }
    const char* tag[2] = {"fp", "sh"};
    for (int s = 0; s < 2; ++s) {
        auto p = slurp(nets_dir + "/net_" + tag[s] + ".awn"), v = slurp(nets_dir + "/value_" + tag[s] + ".awn");
        if (!net[s].load(p.data(), p.size(), error) || !vnet[s].load(v.data(), v.size(), error)) return false;
    }
    return true;
}

void* menu_rows(const void* menu, const std::vector<int64_t>& indices) {
    void* idx = int_list(indices);
    void* out = wotr_snapshot_search_menu_rows(menu, idx);
    wotr_snapshot_destroy(idx);
    return out;
}

Pick search(Box& box, const void* state, int side, int64_t sims, uint64_t seat_seed, uint32_t index, uint64_t search_seed,
            int turn_owner) {
    Pick pk;
    const auto t0 = std::chrono::steady_clock::now();
    // orchestrator.native_search: the policy projection of the full menu; forced -> its first row, no search
    void* full = wotr_snapshot_full_actions(state, box.obs, side);
    if (!full) { pk.error = std::string("full menu: ") + wotr_snapshot_last_error(); return pk; }
    auto [sel, forced] = project(state, full, side);
    wotr_snapshot_destroy(full);
    if (!sel) { pk.error = std::string("projection: ") + wotr_snapshot_last_error(); return pk; }
    if (forced) { pk.menu = sel; pk.index = 0; pk.forced = true; return pk; }
    wotr_snapshot_destroy(sel);
    WotrSearchConfig cfg = awsetup::search_config(side, sims, turn_owner);
    WotrOwnValueConfig own = awsetup::own_config(seat_seed, index);
    uint32_t limbs[2] = {uint32_t(search_seed), uint32_t(search_seed >> 32)};
    void* rng = wotr_random_create_seeded(limbs, 2);
    void* ctl = wotr_search_controller_create_own_value(state, box.obs, box.act, box.menu_tables, box.hunt, rng, &cfg, &own);
    if (!ctl) { pk.error = std::string("create: ") + wotr_snapshot_last_error(); wotr_random_destroy(rng); return pk; }
    Live live{&box, side, cfg.opponent_policy_epsilon, ""};
    int32_t status = -1;
    if (wotr_search_controller_set_sync_provider(ctl, provide, &live) || wotr_search_controller_advance(ctl, &status) ||
        status != awabi::STATUS_COMPLETE) {
        pk.error = live.error.empty() ? std::string("search: ") + wotr_snapshot_last_error() : live.error;
        wotr_search_controller_destroy(ctl); wotr_random_destroy(rng); return pk;
    }
    WotrSearchRoot rv{}; rv.struct_size = sizeof(rv);
    wotr_search_controller_root(ctl, &rv);
    const size_t k = rv.children;
    awselect::Stats st;
    st.k = k;
    for (size_t i = 0; i < k; ++i) {
        st.visits.push_back(rv.visits[i]); st.term.push_back(rv.term_visits[i]); st.forced.push_back(rv.forced_visits[i]);
        st.value_sum.push_back(rv.value_sum[i]); st.value_sq.push_back(rv.value_sq[i]); st.conf_sum.push_back(rv.conf_sum[i]);
        st.phi_sum.push_back(rv.phi_sum[i]); st.priors.push_back(rv.priors_clean[i]); st.proven.push_back(rv.proven[i]);
        st.proven_value.push_back(rv.proven_value[i]); st.side.push_back(rv.side[i]);
    }
    st.root_side = rv.root_side;
    st.conf_on = false;
    pk.menu = wotr_search_controller_root_menu(ctl);
    std::vector<int64_t> verdicts(k, -1);
    if (pk.menu && awasm_strictly_better(state, box.obs, pk.menu, side, verdicts.data(), k) == 1) st.better = verdicts;
    pk.index = int64_t(awselect::pick(st));
    wotr_search_controller_destroy(ctl);
    wotr_random_destroy(rng);
    pk.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return pk;
}

int64_t raw(Box& box, const void* state, int side, const void* menu) {
    // project_ai_actions over the given menu: forced -> its first row
    auto [sel, forced] = project(state, menu, side);
    if (!sel) return -1;
    const int64_t n = awasm_snapshot_root_count(sel);
    // the selected rows' positions in `menu` (project_policy keeps rows; matched by their exported content)
    auto rows_of = [](const void* m) {
        const size_t nv = wotr_snapshot_node_count(m);
        std::vector<WotrValue> v(nv); std::vector<uint64_t> e(wotr_snapshot_edge_count(m)); std::vector<uint8_t> b(wotr_snapshot_byte_count(m));
        wotr_snapshot_export(m, v.data(), nv, e.data(), e.size(), b.data(), b.size());
        // a row's identity: its canonical bytes (a depth-first dump)
        std::vector<std::string> out;
        std::function<void(uint64_t, std::string&)> dump = [&](uint64_t i, std::string& o) {
            const auto& x = v[i];
            o += char('A' + x.tag);
            if (x.tag == WOTR_BOOL || x.tag == WOTR_INT) o += std::to_string(x.integer);
            else if (x.tag == WOTR_FLOAT) o.append(reinterpret_cast<const char*>(&x.real), 8);
            else if (x.tag == WOTR_STRING || x.tag == WOTR_BYTES || x.tag == WOTR_BIGINT) { o += std::to_string(x.count) + ":"; o.append(reinterpret_cast<const char*>(&b[x.offset]), x.count); }
            else if (x.tag >= WOTR_LIST) { o += "("; for (uint64_t j = 0; j < x.count; ++j) dump(e[x.offset + j], o); o += ")"; }
        };
        const auto& root = v[wotr_snapshot_root(m)];
        for (uint64_t j = 0; j < root.count; ++j) { std::string s; dump(e[root.offset + j], s); out.push_back(s); }
        return out;
    };
    const auto all = rows_of(menu), chosen = rows_of(sel);
    std::vector<int64_t> pos;
    for (const auto& r : chosen) { auto it = std::find(all.begin(), all.end(), r); pos.push_back(it == all.end() ? -1 : int64_t(it - all.begin())); }
    if (forced) { wotr_snapshot_destroy(sel); return pos.empty() ? -1 : pos[0]; }
    if (n <= 0) { wotr_snapshot_destroy(sel); return -1; }
    // _get_priors_and_value's priors over the selected rows
    std::vector<float> obs(OBS), acts(static_cast<size_t>(n) * ACT); std::vector<int64_t> regs(static_cast<size_t>(n));
    if (wotr_snapshot_encode_state_v53(state, box.obs, side, obs.data(), OBS) ||
        wotr_snapshot_encode_actions_v53(state, box.obs, sel, box.act, side, acts.data(), acts.size(), regs.data(), size_t(n))) {
        wotr_snapshot_destroy(sel); return -1;
    }
    awnet::Body b;
    box.net[side].body(obs.data(), b, true);
    std::vector<float> logits(static_cast<size_t>(n)), priors(static_cast<size_t>(n));
    box.net[side].logits(b, acts.data(), regs.data(), size_t(n), logits.data(), true);
    awnet::Net::softmax(logits.data(), size_t(n), priors.data());
    float sum = 0.0f; for (float p : priors) sum += p;
    for (float& p : priors) p = p / (sum + 1e-10f);
    // decision=True: THE SAME BUT BETTER over the 40 highest priors
    std::vector<int64_t> order(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) order[size_t(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t c) { return priors[size_t(a)] > priors[size_t(c)]; });
    if (order.size() > 40) order.resize(40);
    void* top = menu_rows(sel, order);
    std::vector<int64_t> better(order.size(), -1);
    const int sb = top ? awasm_strictly_better(state, box.obs, top, side, better.data(), order.size()) : 0;
    if (top) wotr_snapshot_destroy(top);
    if (sb != 1) better.clear();
    size_t first = 0;
    if (!better.empty()) { first = order.size(); for (size_t kk = 0; kk < order.size(); ++kk) if (better[kk] != awselect::BANNED) { first = kk; break; } if (first == order.size()) first = 0; }
    const size_t chosen_i = awselect::move_to_better(first, better);
    wotr_snapshot_destroy(sel);
    return pos[size_t(order[chosen_i])];
}

int priors(void* user, const void* state, int32_t side, const void* menu, float* out, size_t n) {
    Box& box = *static_cast<Box*>(user);
    std::vector<float> obs(OBS), acts(n * ACT);
    std::vector<int64_t> regs(n);
    if (wotr_snapshot_encode_state_v53(state, box.obs, side, obs.data(), OBS) ||
        wotr_snapshot_encode_actions_v53(state, box.obs, menu, box.act, side, acts.data(), acts.size(), regs.data(), n)) return 1;
    awnet::Body b;
    box.net[side].body(obs.data(), b, true);
    std::vector<float> logits(n);
    box.net[side].logits(b, acts.data(), regs.data(), n, logits.data(), true);
    awnet::Net::softmax(logits.data(), n, out);
    float sum = 0.0f; for (size_t i = 0; i < n; ++i) sum += out[i];
    for (size_t i = 0; i < n; ++i) out[i] = out[i] / (sum + 1e-10f);
    return 0;
}

int64_t certain_line(Box& box, const void* state, int side, const void* menu, int64_t own) {
    return awasm_certain_line(state, box.obs, menu, side, own, priors, &box);
}

void* units(Box& box, const void* state, int side, const void* menu, int64_t index, int* changed) {
    int32_t c = -1;
    void* out = awasm_ai_units(state, box.obs, menu, side, index, &c);
    *changed = c;
    return out;
}

}  // namespace awdec
