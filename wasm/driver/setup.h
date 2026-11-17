// The live search's set-up, so the module searches a position from its state alone (no recording):
//   the search configuration          orchestrator.build_config with the live settings (the AI server's
//                                     launch_head.HEAD_ENV, ai.py's constants; the same in every recorded search)
//   the own-value configuration       own_binding.own_configuration(seed, 2, 2, reply_shared=False), the seed
//                                     own_model.OwnModelSearch.choose: rng_streams.derive_seed(seat seed,
//                                     'bridge-own-reply:INDEX') (SHA-256 of 'isolated_game_search_v1\0SEED\0DOMAIN')
//   the search stream                 the Python `random` state the controller is created from; the module's
//                                     convention is Random(search seed) (CPython random.seed(int)), the parity
//                                     harness seeds Python's the same way (record_searches.py --seeded)
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "aw_controller_abi.h"

namespace awsetup {

// FIPS 180-4 SHA-256
inline void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,
        0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
        0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
        0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,
        0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<uint8_t> m(data, data + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    const uint64_t bits = uint64_t(len) * 8;
    for (int i = 7; i >= 0; --i) m.push_back(uint8_t(bits >> (8 * i)));
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t c = 0; c < m.size(); c += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = uint32_t(m[c + 4*i]) << 24 | uint32_t(m[c + 4*i + 1]) << 16 | uint32_t(m[c + 4*i + 2]) << 8 | m[c + 4*i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3), s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h[0], b = h[1], cc = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
            hh = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += cc; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) out[4*i + j] = uint8_t(h[i] >> (24 - 8*j));
}

// rng_streams.derive_seed(seed, domain) (64 bits): the first 8 bytes of the digest, big-endian
inline uint64_t derive_seed(uint64_t seed, const std::string& domain) {
    std::string payload = std::string("isolated_game_search_v1") + '\0' + std::to_string(seed) + '\0' + domain;
    uint8_t d[32];
    sha256(reinterpret_cast<const uint8_t*>(payload.data()), payload.size(), d);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | d[i];
    return v;
}

// own_binding.own_configuration(seed, reply_panels=2, belief_worlds=2, reply_shared=False)
inline WotrOwnValueConfig own_config(uint64_t seat_seed, uint32_t index) {
    const uint64_t seed = derive_seed(seat_seed, "bridge-own-reply:" + std::to_string(index));
    WotrOwnValueConfig c;
    std::memset(&c, 0, sizeof c);
    c.struct_size = sizeof c; c.version = 1; c.reply_panels = 2; c.belief_worlds = 2; c.flags = 0;
    int bits = 0; for (uint64_t x = seed; x; x >>= 1) ++bits;
    c.seed_words = uint32_t((bits + 31) / 32);
    for (uint32_t i = 0; i < c.seed_words; ++i) c.seed[i] = uint32_t(seed >> (32 * i));
    return c;
}

// orchestrator.build_config with the live settings. turn_owner: 0 not given, 1 FP, 2 SH (ai.search_turn_owner)
inline WotrSearchConfig search_config(int side, int64_t simulations, int turn_owner) {
    WotrSearchConfig c;
    std::memset(&c, 0, sizeof c);
    c.struct_size = 192;                 // controller_binding.config_wire_size: 192 while fpu_parent is 0
    c.root_side = side; c.num_simulations = simulations;
    c.sims_menu_scale = 0; c.sims_per_child = 60; c.sims_floor = 120;
    c.c_puct = 1.5;                      // the WIN seats' bound value nets (own_model.SEARCH_VALUE_NETS)
    c.leaf_batch = 1; c.turn_horizon = 1; c.max_children = 40;
    c.chance_k_max = 24; c.kernel_k_max = 24; c.chance_draws = 16;
    c.free_cands = 1; c.free_cands_n = 3; c.free_force = 12;
    c.opponent_policy_mode = 2; c.opponent_policy_epsilon = 0.02;
    c.conf_leaf = 0; c.conf_track = 0; c.conf_ref = 0; c.determinize = 1; c.add_noise = 0; c.overlay_assert = 0;
    c.reward_contract = 2;               // predecision_profile_v1
    c.fp_profile = 3; c.sh_profile = 3;  // 'mixed' (the league nets' reward profile)
    c.phi_rings_rho = 0.0; c.trace = 0; c.root_turn_owner = turn_owner; c.fpu_parent = 0;
    return c;
}

}  // namespace awsetup
