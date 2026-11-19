// The C++ forward pass against torch on recorded leaves (wasm/tools/export_fixtures.py: positions.ref, nn_cases.ref).
//
//   nn_parity FIXTURES_DIR            (native build: wasm/build.sh native; wasm: node wasm/build/nn_parity.mjs DIR)
//
// For every record: the body, the policy net's value head, the bound value net's value, the logits of the action
// rows by the direct path and by the factored fast path, their softmax; the largest absolute difference from torch
// of each, whether the top action is the same, and the time per leaf. Exit 1 when a value, backbone number or prior differs by more than 1e-5, or a logit by more than 5e-5
// relative to max(1, |torch|): torch's own float32 logits differ from a float64 forward by up to 1.05e-5 relative on
// these leaves (wasm/tools/torch_f64_error.py), so a tighter bound would test torch's rounding, not ours.
#include "awnet.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// abs: |c - torch|; rel: |c - torch| / max(1, |torch|) (logits reach |x| ~ 100, where float32 itself has ~1e-5 steps)
struct Stat {
    const char* name; bool relative = false; double limit = 1e-5, max = 0, ref = 0; size_t n = 0;
    void add(double a, double b) {
        double d = std::fabs(a - b);
        if (relative) d /= std::fmax(1.0, std::fabs(b));
        if (d > max || std::isnan(d)) max = d;
        if (std::fabs(b) > ref) ref = std::fabs(b);
        ++n;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: nn_parity FIXTURES_DIR\n"); return 2; }
    const std::string dir = argv[1];
    awnet::Net policy[2], value[2];
    const char* tag[2] = {"fp", "sh"};
    for (int s = 0; s < 2; ++s) {
        std::string err;
        auto p = slurp(dir + "/net_" + tag[s] + ".awn");
        auto v = slurp(dir + "/value_" + tag[s] + ".awn");
        if (!policy[s].load(p.data(), p.size(), err) || !value[s].load(v.data(), v.size(), err)) {
            std::fprintf(stderr, "load %s: %s\n", tag[s], err.c_str());
            return 2;
        }
    }
    int bad = 0;
    for (const char* file : {"positions.ref", "nn_cases.ref"}) {
        auto data = slurp(dir + "/" + file);
        if (data.size() < 8 || std::memcmp(data.data(), "AWR1", 4)) { std::fprintf(stderr, "%s: not awr\n", file); return 2; }
        uint32_t count; std::memcpy(&count, &data[4], 4);
        size_t off = 8;
        auto take = [&](void* dst, size_t bytes) { std::memcpy(dst, &data[off], bytes); off += bytes; };
        Stat body{"backbone"}, head{"head value"}, vnet{"value net"}, slow{"logits direct", true, 5e-5}, fast{"logits fast", true, 5e-5},
             pslow{"priors direct"}, pfast{"priors fast"};
        size_t top_same = 0, top_n = 0, rows_total = 0;
        double t_body = 0, t_slow = 0, t_fast = 0;
        for (uint32_t c = 0; c < count; ++c) {
            uint32_t side, rows, menu_rows;
            take(&side, 4); take(&rows, 4); take(&menu_rows, 4);
            const int od = policy[side].obs_dim(), ad = policy[side].act_dim();
            std::vector<float> obs(od), acts(size_t(rows) * ad), lg(rows), pr(rows), bb(256);
            std::vector<int32_t> reg32(rows);
            take(obs.data(), 4 * obs.size()); take(acts.data(), 4 * acts.size()); take(reg32.data(), 4 * rows);
            float hv, vv; take(&hv, 4); take(&vv, 4);
            take(lg.data(), 4 * rows); take(pr.data(), 4 * rows); take(bb.data(), 4 * 256);
            std::vector<int64_t> reg(reg32.begin(), reg32.end());
            awnet::Body b, bv;
            auto t0 = std::chrono::steady_clock::now();
            policy[side].body(obs.data(), b, true);
            auto t1 = std::chrono::steady_clock::now();
            t_body += std::chrono::duration<double>(t1 - t0).count();
            for (int k = 0; k < 256; ++k) body.add(b.h[k], bb[k]);
            head.add(policy[side].value(b), hv);
            value[side].body(obs.data(), bv, false);
            vnet.add(value[side].value(bv), vv);
            if (!rows) continue;
            std::vector<float> o1(rows), o2(rows), p1(rows), p2(rows);
            t0 = std::chrono::steady_clock::now();
            policy[side].logits(b, acts.data(), reg.data(), rows, o1.data(), false);
            t1 = std::chrono::steady_clock::now();
            policy[side].logits(b, acts.data(), reg.data(), rows, o2.data(), true);
            auto t2 = std::chrono::steady_clock::now();
            t_slow += std::chrono::duration<double>(t1 - t0).count();
            t_fast += std::chrono::duration<double>(t2 - t1).count();
            awnet::Net::softmax(o1.data(), rows, p1.data());
            awnet::Net::softmax(o2.data(), rows, p2.data());
            size_t a_ref = 0, a_fast = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                slow.add(o1[r], lg[r]); fast.add(o2[r], lg[r]); pslow.add(p1[r], pr[r]); pfast.add(p2[r], pr[r]);
                if (pr[r] > pr[a_ref]) a_ref = r;
                if (p2[r] > p2[a_fast]) a_fast = r;
            }
            top_same += a_ref == a_fast; ++top_n; rows_total += rows;
        }
        std::printf("%s: %u leaves, %zu with action rows (%zu rows)\n", file, count, top_n, rows_total);
        for (Stat* s : {&body, &head, &vnet, &slow, &fast, &pslow, &pfast}) {
            std::printf("  %-14s max %s %.3g over %zu numbers (max |torch| %.3g)\n", s->name,
                        s->relative ? "rel diff" : "|diff|  ", s->max, s->n, s->ref);
            if (!(s->max <= s->limit)) ++bad;
        }
        std::printf("  same top action (fast path vs torch): %zu of %zu\n", top_same, top_n);
        std::printf("  time: body %.3f ms/leaf, rows direct %.2f us/row, rows fast %.2f us/row\n",
                    1e3 * t_body / (count ? count : 1), 1e6 * t_slow / (rows_total ? rows_total : 1),
                    1e6 * t_fast / (rows_total ? rows_total : 1));
    }
    std::printf(bad ? "FAIL\n" : "PASS (values, backbone, priors within 1e-5; logits within 5e-5 relative)\n");
    return bad ? 1 : 0;
}
