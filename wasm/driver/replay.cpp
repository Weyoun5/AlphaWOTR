// Replays native searches recorded by wasm/tools/record_searches.py in this build of the box.
//
//   replay FIXTURES_DIR [--threaded] [--live] [--limit N] [FILE.aws ...]
//
// recorded (default): the controller gets the recorded root state, tables, configuration and search RNG; every
//   request it makes must equal the recorded one bit for bit (operation, side, purpose, leaf ids, offsets,
//   observations, action rows, regions), and it is answered with the recorded reply (torch's numbers, or the Python
//   combat service's graph). At the end the root view (visits, value sums, ...), the search RNG's state and the root
//   menu must equal the recorded ones bit for bit, and the pick (select.h, from the root view and the recorded
//   "same but better" verdicts) must be the recorded one.
// live: the replies come from this module's network (awnet) and value net; only the pick is compared (a reply that
//   differs in the last bits may change the tree), with the visit share of the recorded pick.
// --threaded: the box's own threaded controller (native only); default: the single-thread provider (wasm patch 0003).
#include "aw_controller_abi.h"
#include "awnet.h"
#include "select.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <dirent.h>

extern "C" void* awasm_graph_create(const uint8_t* p);

namespace {

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Reader {
    const uint8_t* p; const uint8_t* e;
    template <class T> T get() { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
    template <class T> std::vector<T> vec(size_t n) { std::vector<T> v(n); std::memcpy(v.data(), p, n * sizeof(T)); p += n * sizeof(T); return v; }
    const uint8_t* skip(size_t n) { const uint8_t* q = p; p += n; return q; }
    // an awg graph record: -> pointer to its counts (what awasm_graph_create takes), and its whole size
    const uint8_t* graph(size_t* size = nullptr) {
        const uint32_t nl = get<uint32_t>(); p += nl;
        const uint8_t* counts = p;
        uint64_t nv, ne, nb; std::memcpy(&nv, p, 8); std::memcpy(&ne, p + 8, 8); std::memcpy(&nb, p + 16, 8);
        const size_t total = 32 + 40 * nv + 8 * ne + nb;
        p += total;
        if (size) *size = total;
        return counts;
    }
};

struct Rng { std::vector<uint32_t> words; uint32_t index; int32_t has_gauss; uint64_t bits; };
Rng read_rng(Reader& r) { Rng g; g.words = r.vec<uint32_t>(624); g.index = r.get<uint32_t>(); g.has_gauss = r.get<int32_t>(); g.bits = r.get<uint64_t>(); return g; }

struct NumericEvent {
    uint64_t id; int32_t op, side, purpose; uint64_t batch, rows;
    std::vector<uint64_t> leaves, offsets; std::vector<float> obs, actions; std::vector<int64_t> regions;
    std::vector<double> values, conf, mixed; std::vector<float> priors;
};
struct DistEvent { uint64_t id; int32_t side; uint64_t leaf; const uint8_t* payload; const uint8_t* reply; };
struct Event { int kind; NumericEvent n; DistEvent d; };

struct RootView {
    uint64_t k = 0; std::vector<uint64_t> visits, term, forced; std::vector<double> vsum, vsq, csum, psum, pval;
    std::vector<float> priors; std::vector<int8_t> proven; std::vector<int32_t> side; int32_t root_side = 0;
    uint64_t sims = 0, budget = 0;
};

struct Search {
    uint32_t position, seed, side, sims;
    std::vector<uint8_t> config, own;
    Rng rng, rng_after;
    const uint8_t* root;
    std::vector<Event> events;
    RootView view;
    const uint8_t* menu; size_t menu_size;
    std::vector<int64_t> better; bool has_better;
    uint32_t pick; double seconds;
};

Search parse(Reader& r, size_t obs_dim, size_t act_dim) {
    Search s;
    s.position = r.get<uint32_t>(); s.seed = r.get<uint32_t>(); s.side = r.get<uint32_t>(); s.sims = r.get<uint32_t>();
    s.config = r.vec<uint8_t>(r.get<uint32_t>());
    s.own = r.vec<uint8_t>(r.get<uint32_t>());
    s.rng = read_rng(r);
    s.root = r.graph();
    const uint32_t n = r.get<uint32_t>();
    s.events.resize(n);
    for (auto& ev : s.events) {
        ev.kind = r.get<uint8_t>();
        if (ev.kind == 1) {
            auto& e = ev.n;
            e.id = r.get<uint64_t>(); e.op = r.get<int32_t>(); e.side = r.get<int32_t>(); e.purpose = r.get<int32_t>();
            e.batch = r.get<uint64_t>(); e.rows = r.get<uint64_t>();
            e.leaves = r.vec<uint64_t>(e.batch); e.offsets = r.vec<uint64_t>(e.batch + 1);
            e.obs = r.vec<float>(e.batch * obs_dim); e.actions = r.vec<float>(e.rows * act_dim);
            e.regions = r.vec<int64_t>(e.rows);
            e.values = r.vec<double>(e.batch); e.conf = r.vec<double>(e.batch);
            if (e.op != awabi::OP_EVAL) { e.priors = r.vec<float>(e.rows); e.mixed = r.vec<double>(e.rows); }
        } else {
            auto& d = ev.d;
            d.id = r.get<uint64_t>(); d.side = r.get<int32_t>(); d.leaf = r.get<uint64_t>();
            d.payload = r.graph(); d.reply = r.graph();
        }
    }
    auto& v = s.view;
    v.k = r.get<uint64_t>();
    v.visits = r.vec<uint64_t>(v.k); v.term = r.vec<uint64_t>(v.k); v.forced = r.vec<uint64_t>(v.k);
    v.vsum = r.vec<double>(v.k); v.vsq = r.vec<double>(v.k); v.csum = r.vec<double>(v.k); v.psum = r.vec<double>(v.k);
    v.priors = r.vec<float>(v.k); v.proven = r.vec<int8_t>(v.k); v.pval = r.vec<double>(v.k); v.side = r.vec<int32_t>(v.k);
    v.root_side = r.get<int32_t>(); v.sims = r.get<uint64_t>(); v.budget = r.get<uint64_t>();
    s.rng_after = read_rng(r);
    s.menu = r.graph(&s.menu_size);
    s.better = r.vec<int64_t>(v.k); s.has_better = r.get<uint32_t>() != 0;
    s.pick = r.get<uint32_t>(); s.seconds = r.get<double>();
    return s;
}

template <class T> bool same(const T* a, const T* b, size_t n) { return n == 0 || std::memcmp(a, b, n * sizeof(T)) == 0; }

struct Context {
    const Search* s; size_t next = 0; std::string mismatch;
    bool live = false; size_t obs_dim = 0, act_dim = 0;
    const awnet::Net* net = nullptr; const awnet::Net* vnet = nullptr; double eps = 0.0;
    double nn_seconds = 0; uint64_t leaves = 0, rows = 0, requests = 0;
    std::vector<std::unique_ptr<void, void (*)(void*)>> keep;
};

void fail(Context& c, const std::string& why) { if (c.mismatch.empty()) c.mismatch = why; }

// Answer the pending request (the Python driver's body between two advance() calls). 0 = answered.
int answer(void* user, void* ctl) {
    Context& c = *static_cast<Context*>(user);
    WotrSearchRequest q{}; q.struct_size = sizeof(q);
    if (wotr_search_controller_request(ctl, &q)) { fail(c, std::string("request: ") + wotr_snapshot_last_error()); return 1; }
    ++c.requests;
    if (q.operation == awabi::OP_DIST) {
        if (c.live) { fail(c, "live combat distributions are not wired yet"); return 1; }
        if (c.next >= c.s->events.size() || c.s->events[c.next].kind != 2) { fail(c, "unexpected COMBAT_DIST request"); return 1; }
        const DistEvent& d = c.s->events[c.next++].d;
        if (q.leaf_ids[0] != d.leaf || q.model_side != d.side) fail(c, "COMBAT_DIST request differs");
        void* reply = awasm_graph_create(d.reply);
        if (!reply) { fail(c, "reply graph"); return 1; }
        const int rc = wotr_search_controller_submit_payload(ctl, q.request_id, reply);
        wotr_snapshot_destroy(reply);
        if (rc) { fail(c, std::string("submit_payload: ") + wotr_snapshot_last_error()); return 1; }
        return 0;
    }
    if (!c.live) {
        if (c.next >= c.s->events.size() || c.s->events[c.next].kind != 1) { fail(c, "unexpected numeric request"); return 1; }
        const NumericEvent& e = c.s->events[c.next++].n;
        const bool ok = q.operation == e.op && q.model_side == e.side && q.purpose == e.purpose && q.batch == e.batch &&
                        q.rows == e.rows && same(q.leaf_ids, e.leaves.data(), e.batch) &&
                        same(q.offsets, e.offsets.data(), e.batch + 1) &&
                        same(q.observations, e.obs.data(), e.obs.size()) &&
                        same(q.actions, e.actions.data(), e.actions.size()) && same(q.regions, e.regions.data(), e.rows);
        if (!ok) fail(c, "request " + std::to_string(c.next - 1) + " differs from the recorded one");
        WotrSearchReply r{}; r.struct_size = sizeof(r); r.request_id = q.request_id;
        r.values = e.values.data(); r.conf = e.conf.data();
        r.priors = e.priors.empty() ? nullptr : e.priors.data(); r.mixed = e.mixed.empty() ? nullptr : e.mixed.data();
        if (wotr_search_controller_submit(ctl, &r)) { fail(c, std::string("submit: ") + wotr_snapshot_last_error()); return 1; }
        return 0;
    }
    // live: numeric_bridge.NumericInferenceBridge.run with this module's network
    const auto t0 = std::chrono::steady_clock::now();
    const size_t B = q.batch, R = q.rows;
    std::vector<double> values(B), conf(B, 1.0), mixed(R);
    std::vector<float> priors(R);
    for (size_t i = 0; i < B; ++i) {
        const float* obs = q.observations + i * c.obs_dim;
        awnet::Body bv;
        c.vnet->body(obs, bv, false);
        values[i] = double(c.vnet->value(bv));
        if (q.operation == awabi::OP_EVAL) continue;
        const size_t lo = q.offsets[i], hi = q.offsets[i + 1], n = hi - lo;
        awnet::Body b;
        c.net->body(obs, b, true);
        std::vector<float> logits(n);
        c.net->logits(b, q.actions + lo * c.act_dim, q.regions + lo, n, logits.data(), true);
        awnet::Net::softmax(logits.data(), n, &priors[lo]);
        float sum = 0.0f;
        for (size_t j = lo; j < hi; ++j) sum += priors[j];
        const float denom = sum + 1e-10f;
        double total = 0.0;
        for (size_t j = lo; j < hi; ++j) { priors[j] = priors[j] / denom; total += double(priors[j]); }
        for (size_t j = lo; j < hi; ++j) mixed[j] = (1.0 - c.eps) * (double(priors[j]) / total) + c.eps / double(n);
        c.rows += n;
    }
    c.leaves += B;
    c.nn_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    WotrSearchReply r{}; r.struct_size = sizeof(r); r.request_id = q.request_id;
    r.values = values.data(); r.conf = conf.data();
    if (q.operation != awabi::OP_EVAL) { r.priors = priors.data(); r.mixed = mixed.data(); }
    if (wotr_search_controller_submit(ctl, &r)) { fail(c, std::string("submit: ") + wotr_snapshot_last_error()); return 1; }
    return 0;
}

std::vector<uint8_t> export_graph(const void* h) {
    const size_t nv = wotr_snapshot_node_count(h), ne = wotr_snapshot_edge_count(h), nb = wotr_snapshot_byte_count(h);
    std::vector<uint8_t> out(32 + 40 * nv + 8 * ne + nb);
    const uint64_t counts[4] = {nv, ne, nb, wotr_snapshot_root(h)};
    std::memcpy(out.data(), counts, 32);
    std::vector<WotrValue> v(nv); std::vector<uint64_t> e(ne); std::vector<uint8_t> b(nb);
    wotr_snapshot_export(h, v.data(), nv, e.data(), ne, b.data(), nb);
    std::memcpy(&out[32], v.data(), 40 * nv); std::memcpy(&out[32 + 40 * nv], e.data(), 8 * ne);
    if (nb) std::memcpy(&out[32 + 40 * nv + 8 * ne], b.data(), nb);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: replay FIXTURES_DIR [--threaded] [--live] [--limit N] [FILE.aws ...]\n"); return 2; }
    const std::string dir = argv[1];
    bool threaded = false, live = false; size_t limit = 0;
    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threaded") threaded = true;
        else if (a == "--live") live = true;
        else if (a == "--limit" && i + 1 < argc) limit = std::stoul(argv[++i]);
        else files.push_back(a);
    }
    if (files.empty()) {
        if (DIR* d = opendir((dir + "/searches").c_str())) {
            while (dirent* e = readdir(d)) {
                const std::string n = e->d_name;
                if (n.size() > 4 && n.substr(n.size() - 4) == ".aws") files.push_back(dir + "/searches/" + n);
            }
            closedir(d);
        }
        std::sort(files.begin(), files.end());
    }
    if (limit && files.size() > limit) files.resize(limit);
    // tables
    auto tables = slurp(dir + "/tables.awg");
    Reader tr{tables.data() + 8, tables.data() + tables.size()};
    void* obs_t = nullptr; void* act_t = nullptr; void* menu_t = nullptr; void* hunt_t = nullptr;
    uint32_t tcount; std::memcpy(&tcount, tables.data() + 4, 4);
    for (uint32_t i = 0; i < tcount; ++i) {
        const uint32_t nl = tr.get<uint32_t>(); const std::string name(reinterpret_cast<const char*>(tr.p), nl);
        tr.p -= 4;
        void* h = awasm_graph_create(tr.graph());
        if (!h) { std::fprintf(stderr, "table %s: %s\n", name.c_str(), wotr_snapshot_last_error()); return 2; }
        if (name == "observation") { obs_t = h; if (wotr_snapshot_prepare_observation_tables(h)) return 2; }
        else if (name == "actions_v53") { act_t = h; if (wotr_snapshot_prepare_action_tables_v53(h)) return 2; }
        else if (name == "menu") menu_t = h;
        else if (name == "hunt") hunt_t = h;
    }
    if (!obs_t || !act_t || !menu_t || !hunt_t) { std::fprintf(stderr, "tables.awg lacks a table (run record_searches.py)\n"); return 2; }
    awnet::Net nets[2], vnets[2];
    if (live) {
        const char* tag[2] = {"fp", "sh"};
        for (int s = 0; s < 2; ++s) {
            std::string err;
            auto p = slurp(dir + "/net_" + tag[s] + ".awn"), v = slurp(dir + "/value_" + tag[s] + ".awn");
            if (!nets[s].load(p.data(), p.size(), err) || !vnets[s].load(v.data(), v.size(), err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        }
    }
    const size_t OBS = 4745, ACT = 199;
    int bad = 0; size_t same_pick = 0; double t_total = 0, t_py = 0, t_nn = 0;
    for (const auto& path : files) {
        auto data = slurp(path);
        Reader r{data.data() + 8, data.data() + data.size()};
        Search s = parse(r, OBS, ACT);
        WotrSearchConfig cfg{}; std::memcpy(&cfg, s.config.data(), std::min(sizeof(cfg), s.config.size()));
        WotrOwnValueConfig own{}; std::memcpy(&own, s.own.data(), std::min(sizeof(own), s.own.size()));
        void* root = awasm_graph_create(s.root);
        void* rng = wotr_random_create(s.rng.words.data(), 624, s.rng.index, s.rng.has_gauss, s.rng.bits);
        Context c; c.s = &s; c.live = live; c.obs_dim = OBS; c.act_dim = ACT;
        c.net = &nets[s.side]; c.vnet = &vnets[s.side]; c.eps = cfg.opponent_policy_epsilon;
        const auto t0 = std::chrono::steady_clock::now();
        void* ctl = wotr_search_controller_create_own_value(root, obs_t, act_t, menu_t, hunt_t, rng, &cfg, &own);
        if (!ctl) { std::printf("[FAIL] %s: create: %s\n", path.c_str(), wotr_snapshot_last_error()); ++bad; continue; }
        int32_t status = -1;
        if (!threaded) {
            if (wotr_search_controller_set_sync_provider(ctl, answer, &c)) { std::printf("set provider: %s\n", wotr_snapshot_last_error()); return 2; }
            if (wotr_search_controller_advance(ctl, &status)) fail(c, std::string("advance: ") + wotr_snapshot_last_error());
        } else {
            for (;;) {
                if (wotr_search_controller_advance(ctl, &status)) { fail(c, std::string("advance: ") + wotr_snapshot_last_error()); break; }
                if (status != awabi::STATUS_REQUEST) break;
                if (answer(&c, ctl)) break;
            }
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        t_total += secs; t_py += s.seconds; t_nn += c.nn_seconds;
        bool ok = c.mismatch.empty() && status == awabi::STATUS_COMPLETE;
        std::string detail;
        uint32_t pick = 0xffffffff; double share = 0;
        if (ok) {
            WotrSearchRoot rv{}; rv.struct_size = sizeof(rv);
            wotr_search_controller_root(ctl, &rv);
            const auto& v = s.view;
            const size_t k = rv.children;
            const bool view_same = k == v.k && same(rv.visits, v.visits.data(), k) && same(rv.term_visits, v.term.data(), k) &&
                same(rv.forced_visits, v.forced.data(), k) && same(rv.value_sum, v.vsum.data(), k) && same(rv.value_sq, v.vsq.data(), k) &&
                same(rv.conf_sum, v.csum.data(), k) && same(rv.phi_sum, v.psum.data(), k) && same(rv.priors_clean, v.priors.data(), k) &&
                same(rv.proven, v.proven.data(), k) && same(rv.proven_value, v.pval.data(), k) && same(rv.side, v.side.data(), k) &&
                rv.root_side == v.root_side && rv.simulations_run == v.sims && rv.budget == v.budget;
            // the pick: select.h on the module's root view, with the recorded verdicts of ai._strictly_better
            awselect::Stats st;
            st.k = k;
            for (size_t i = 0; i < k; ++i) {
                st.visits.push_back(rv.visits[i]); st.term.push_back(rv.term_visits[i]); st.forced.push_back(rv.forced_visits[i]);
                st.value_sum.push_back(rv.value_sum[i]); st.value_sq.push_back(rv.value_sq[i]); st.conf_sum.push_back(rv.conf_sum[i]);
                st.phi_sum.push_back(rv.phi_sum[i]); st.priors.push_back(rv.priors_clean[i]); st.proven.push_back(rv.proven[i]);
                st.proven_value.push_back(rv.proven_value[i]); st.side.push_back(rv.side[i]);
            }
            st.root_side = rv.root_side;
            st.conf_on = cfg.conf_leaf || cfg.conf_track;
            if (s.has_better) st.better = s.better;
            pick = awselect::pick(st);
            uint64_t total = 0; for (size_t i = 0; i < k; ++i) total += rv.visits[i];
            if (s.pick < k && total) share = double(rv.visits[s.pick]) / double(total);
            if (!live) {
                void* rr = wotr_search_controller_rng(ctl);
                std::vector<uint32_t> words(624); uint32_t index; int32_t hg; uint64_t bits;
                wotr_random_export(rr, words.data(), 624, &index, &hg, &bits);
                wotr_random_destroy(rr);
                const bool rng_same = words == s.rng_after.words && index == s.rng_after.index && hg == s.rng_after.has_gauss && bits == s.rng_after.bits;
                void* m = wotr_search_controller_root_menu(ctl);
                const auto mg = export_graph(m);
                wotr_snapshot_destroy(m);
                const bool menu_same = mg.size() == s.menu_size && std::memcmp(mg.data(), s.menu, s.menu_size) == 0;
                ok = view_same && rng_same && menu_same && pick == s.pick && c.next == s.events.size();
                detail = std::string("root view ") + (view_same ? "identical" : "DIFFERENT") + ", rng " + (rng_same ? "identical" : "DIFFERENT") +
                         ", root menu " + (menu_same ? "identical" : "DIFFERENT") + ", " + std::to_string(c.next) + "/" + std::to_string(s.events.size()) + " requests";
            } else {
                ok = pick == s.pick;
                char buf[160];
                std::snprintf(buf, sizeof buf, "%llu requests, %llu leaves, %llu rows, network %.2f s", (unsigned long long)c.requests,
                              (unsigned long long)c.leaves, (unsigned long long)c.rows, c.nn_seconds);
                detail = buf;
            }
        } else {
            detail = c.mismatch.empty() ? std::string("status ") + std::to_string(status) : c.mismatch;
        }
        same_pick += pick == s.pick;
        bad += !ok;
        std::printf("[%s] position %u seed %u (%s, %u sims): pick %u (recorded %u, its visit share %.2f), %s; %.2f s (Python native %.2f s)\n",
                    ok ? "PASS" : "FAIL", s.position, s.seed, s.side ? "SH" : "FP", s.sims, pick, s.pick, share, detail.c_str(), secs, s.seconds);
        std::fflush(stdout);
        wotr_search_controller_destroy(ctl);
        wotr_random_destroy(rng);
        wotr_snapshot_destroy(root);
    }
    std::printf("%s: %zu searches, %zu same pick; %.1f s (network %.1f s) against Python native %.1f s\n",
                bad ? "FAIL" : "PASS", files.size(), same_pick, t_total, t_nn, t_py);
    return bad ? 1 : 0;
}
