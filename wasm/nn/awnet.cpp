#include "awnet.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>

namespace awnet {
namespace {

// A minimal JSON reader for the awn header (objects, arrays, strings without escapes beyond \" \\, numbers).
struct J {
    enum Kind { Null, Num, Str, Arr, Obj } kind = Null;
    double num = 0;
    std::string str;
    std::vector<J> arr;
    std::map<std::string, J> obj;
    const J& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("awn header lacks " + k);
        return it->second;
    }
    bool has(const std::string& k) const { return obj.count(k) != 0; }
};

struct Reader {
    const char* p;
    const char* e;
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    J value() {
        ws();
        if (p >= e) throw std::runtime_error("awn header ends early");
        J v;
        if (*p == '{') {
            v.kind = J::Obj; ++p; ws();
            if (*p == '}') { ++p; return v; }
            for (;;) {
                ws(); std::string k = string(); ws();
                if (*p++ != ':') throw std::runtime_error("awn header: ':' expected");
                v.obj[k] = value(); ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != '}') throw std::runtime_error("awn header: '}' expected");
                return v;
            }
        }
        if (*p == '[') {
            v.kind = J::Arr; ++p; ws();
            if (*p == ']') { ++p; return v; }
            for (;;) {
                v.arr.push_back(value()); ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != ']') throw std::runtime_error("awn header: ']' expected");
                return v;
            }
        }
        if (*p == '"') { v.kind = J::Str; v.str = string(); return v; }
        if (!std::strncmp(p, "null", 4)) { p += 4; return v; }
        if (!std::strncmp(p, "true", 4)) { p += 4; v.kind = J::Num; v.num = 1; return v; }
        if (!std::strncmp(p, "false", 5)) { p += 5; v.kind = J::Num; v.num = 0; return v; }
        char* end = nullptr;
        v.kind = J::Num; v.num = std::strtod(p, &end);
        if (end == p) throw std::runtime_error("awn header: bad value");
        p = end;
        return v;
    }
    std::string string() {
        if (*p++ != '"') throw std::runtime_error("awn header: string expected");
        std::string s;
        while (p < e && *p != '"') {
            if (*p == '\\') { ++p; }
            s.push_back(*p++);
        }
        ++p;
        return s;
    }
};

// y[o] = b[o] + sum_i W[o][i] x[i], float accumulation in 8 partial sums (vectorisable without reassociation flags).
inline float dot(const float* w, const float* x, int n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) acc[k] += w[i + k] * x[i + k];
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; i < n; ++i) s += w[i] * x[i];
    return s;
}

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

}  // namespace

const Tensor& Net::t(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("awn file lacks tensor " + name);
    return it->second;
}

bool Net::load(const uint8_t* data, size_t size, std::string& error) {
    try {
        if (size < 8 || std::memcmp(data, "AWN1", 4) != 0) throw std::runtime_error("not an awn file");
        uint32_t hlen;
        std::memcpy(&hlen, data + 4, 4);
        if (8 + size_t(hlen) > size) throw std::runtime_error("awn header out of range");
        Reader r{reinterpret_cast<const char*>(data + 8), reinterpret_cast<const char*>(data + 8 + hlen)};
        J header = r.value();
        const uint8_t* blob = data + 8 + hlen;
        const size_t blob_size = size - 8 - hlen;
        tensors_.clear();
        for (const J& d : header.at("tensors").arr) {
            Tensor tensor;
            size_t count = 1;
            for (const J& s : d.at("shape").arr) { tensor.shape.push_back(int64_t(s.num)); count *= size_t(s.num); }
            const size_t offset = size_t(d.at("offset").num);
            if (offset + 4 * count > blob_size) throw std::runtime_error("awn tensor out of range: " + d.at("name").str);
            if (d.at("dtype").str == "f32") {
                tensor.f.resize(count);
                std::memcpy(tensor.f.data(), blob + offset, 4 * count);
            } else {
                tensor.i.resize(count);
                std::memcpy(tensor.i.data(), blob + offset, 4 * count);
            }
            tensors_[d.at("name").str] = std::move(tensor);
        }
        const J& m = header.at("meta");
        obs_dim_ = int(m.at("obs_dim").num); act_dim_ = int(m.at("act_dim").num); regions_ = int(m.at("regions").num);
        node_feat_ = int(m.at("node_feat").num); card_start_ = int(m.at("card_idx_start").num);
        hand_ = int(m.at("hand").num); card_pad_ = int(m.at("card_pad").num); card_col_ = int(m.at("action_card_col").num);
        scaled2_ = m.at("head_act").str == "scaled2";
        meta_ = std::string(reinterpret_cast<const char*>(data + 8), hlen > 400 ? 400 : hlen);
        esrc_ = t("board_gnn.edge_src").i;
        edst_ = t("board_gnn.edge_dst").i;
        dis_ = t("board_gnn.deg_inv_sqrt").f;
        // shape checks for the 'base' architecture
        if (t("backbone.0.weight").shape != std::vector<int64_t>{512, 64 + 32 + (obs_dim_ - regions_ * node_feat_ - hand_)})
            throw std::runtime_error("backbone width differs from the observation layout");
        if (t("spatial_action_scorer.0.weight").shape[1] != 256 + 64 + (act_dim_ - 1) + 16)
            throw std::runtime_error("scorer width differs from the action layout");
        // card parts of the first policy layers, once per net (fast path)
        const int in = 256 + 64 + (act_dim_ - 1) + 16, coff = 256 + 64 + (act_dim_ - 1);
        const auto& emb = t("card_embedding.weight").f;
        const auto& w1 = t("spatial_action_scorer.0.weight").f;
        const auto& wd = t("policy_delta.0.weight").f;
        card1_.assign(97 * 192, 0.0f); card_d_.assign(97 * 128, 0.0f);
        for (int c = 0; c < 97; ++c) {
            for (int o = 0; o < 192; ++o) card1_[c * 192 + o] = dot(&w1[size_t(o) * in + coff], &emb[c * 16], 16);
            for (int o = 0; o < 128; ++o) card_d_[c * 128 + o] = dot(&wd[size_t(o) * in + coff], &emb[c * 16], 16);
        }
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

void Net::linear(const std::string& prefix, const float* x, int in, int out, float* y) const {
    const auto& w = t(prefix + ".weight").f;
    auto bit = tensors_.find(prefix + ".bias");
    for (int o = 0; o < out; ++o) {
        float s = dot(&w[size_t(o) * in], x, in);
        if (bit != tensors_.end()) s += bit->second.f[o];
        y[o] = s;
    }
}

// torch's CPU batch_norm in eval: alpha = w / sqrt(var + eps), beta = b - mean * alpha, y = x * alpha + beta
void Net::batchnorm(const std::string& prefix, float* x, int n) const {
    const auto& w = t(prefix + ".weight").f;
    const auto& b = t(prefix + ".bias").f;
    const auto& mean = t(prefix + ".running_mean").f;
    const auto& var = t(prefix + ".running_var").f;
    for (int i = 0; i < n; ++i) {
        const float invstd = 1.0f / std::sqrt(var[i] + 1e-5f);
        const float alpha = w[i] * invstd;
        const float beta = b[i] - mean[i] * alpha;
        x[i] = x[i] * alpha + beta;
    }
}

int Net::card_index(float raw) const {
    // torch: raw.long() (truncation), negative -> padding, clamp(0, pad)
    int64_t idx = int64_t(raw);
    if (idx < 0) idx = card_pad_;
    if (idx > card_pad_) idx = card_pad_;
    return int(idx);
}

void Net::body(const float* obs, Body& out, bool fast_policy) const {
    const int N = regions_, F = node_feat_, H = 64;
    // GCN
    std::vector<float> x(obs, obs + size_t(N) * F), agg, lin(size_t(N) * H), res(size_t(N) * H);
    int in = F;
    for (int layer = 0; layer < 2; ++layer) {
        const std::string p = "board_gnn.layers." + std::to_string(layer);
        agg.assign(size_t(N) * in, 0.0f);
        for (size_t e = 0; e < esrc_.size(); ++e) {
            const int s = esrc_[e], d = edst_[e];
            const float norm_s = dis_[s], norm_d = dis_[d];
            for (int k = 0; k < in; ++k) agg[size_t(d) * in + k] += x[size_t(s) * in + k] * norm_s * norm_d;
        }
        for (int n = 0; n < N; ++n) {
            linear(p + ".linear", &agg[size_t(n) * in], in, H, &lin[size_t(n) * H]);
            batchnorm(p + ".bn", &lin[size_t(n) * H], H);
            if (in != H) linear(p + ".residual", &x[size_t(n) * in], in, H, &res[size_t(n) * H]);
            else std::memcpy(&res[size_t(n) * H], &x[size_t(n) * in], sizeof(float) * H);
            for (int k = 0; k < H; ++k) lin[size_t(n) * H + k] = relu(lin[size_t(n) * H + k] + res[size_t(n) * H + k]);
        }
        x = lin;
        in = H;
    }
    out.nodes = x;
    out.mean.assign(H, 0.0f);
    for (int k = 0; k < H; ++k) {
        float s = 0.0f;
        for (int n = 0; n < N; ++n) s += x[size_t(n) * H + k];
        out.mean[k] = s / float(N);
    }
    const int glob = obs_dim_ - N * F - hand_;
    std::vector<float> combined(64 + 32 + glob);
    linear("board_gnn.output_proj", out.mean.data(), H, 64, combined.data());
    // cards
    float sum[16] = {0}, count = 0.0f;
    const auto& emb = t("card_embedding.weight").f;
    for (int j = 0; j < hand_; ++j) {
        const float raw = obs[card_start_ + j];
        const float mask = raw >= 0.0f ? 1.0f : 0.0f;
        const int idx = card_index(raw);
        for (int k = 0; k < 16; ++k) sum[k] += emb[idx * 16 + k] * mask;
        count += mask;
    }
    if (count < 1.0f) count = 1.0f;
    float pooled[16];
    for (int k = 0; k < 16; ++k) pooled[k] = sum[k] / count;
    linear("card_proj.0", pooled, 16, 32, &combined[64]);
    for (int k = 0; k < 32; ++k) combined[64 + k] = relu(combined[64 + k]);
    // global features: everything but the regions and the card slots
    const int pre = card_start_ - N * F;
    std::memcpy(&combined[96], obs + N * F, sizeof(float) * pre);
    std::memcpy(&combined[96 + pre], obs + card_start_ + hand_, sizeof(float) * (glob - pre));
    std::vector<float> h1(512);
    linear("backbone.0", combined.data(), int(combined.size()), 512, h1.data());
    batchnorm("backbone.1", h1.data(), 512);
    for (float& v : h1) v = relu(v);
    out.h.assign(256, 0.0f);
    linear("backbone.4", h1.data(), 512, 256, out.h.data());
    batchnorm("backbone.5", out.h.data(), 256);
    for (float& v : out.h) v = relu(v);
    if (fast_policy) {
        const int rin = 256 + 64 + (act_dim_ - 1) + 16;
        const auto& w1 = t("spatial_action_scorer.0.weight").f;
        const auto& b1 = t("spatial_action_scorer.0.bias").f;
        const auto& wd = t("policy_delta.0.weight").f;
        const auto& bd = t("policy_delta.0.bias").f;
        out.base1.resize(192); out.base_d.resize(128);
        for (int o = 0; o < 192; ++o) out.base1[o] = dot(&w1[size_t(o) * rin], out.h.data(), 256) + b1[o];
        for (int o = 0; o < 128; ++o) out.base_d[o] = dot(&wd[size_t(o) * rin], out.h.data(), 256) + bd[o];
        out.reg1.resize(size_t(N + 1) * 192); out.reg_d.resize(size_t(N + 1) * 128);
        for (int n = 0; n <= N; ++n) {
            const float* node = n < N ? &out.nodes[size_t(n) * H] : out.mean.data();
            for (int o = 0; o < 192; ++o) out.reg1[size_t(n) * 192 + o] = dot(&w1[size_t(o) * rin + 256], node, 64);
            for (int o = 0; o < 128; ++o) out.reg_d[size_t(n) * 128 + o] = dot(&wd[size_t(o) * rin + 256], node, 64);
        }
    }
}

float Net::value(const Body& b) const {
    float v1[128], v;
    linear("value_head.0", b.h.data(), 256, 128, v1);
    batchnorm("value_head.1", v1, 128);
    for (float& x : v1) x = relu(x);
    linear("value_head.4", v1, 128, 1, &v);
    return scaled2_ ? 2.0f * std::tanh(v / 2.0f) : std::tanh(v);
}

void Net::logits(const Body& b, const float* actions, const int64_t* regions, size_t rows, float* out, bool fast) const {
    const int A = act_dim_, F = A - 1, rin = 256 + 64 + F + 16;
    const auto& emb = t("card_embedding.weight").f;
    std::vector<float> combined(rin), z1(192), z2(64), zd(128);
    const auto& w1 = t("spatial_action_scorer.0.weight").f;
    const auto& wd = t("policy_delta.0.weight").f;
    for (size_t r = 0; r < rows; ++r) {
        const float* row = actions + r * A;
        const int64_t reg = regions[r];
        const int card = card_index(row[card_col_]);
        if (!fast) {
            std::memcpy(&combined[0], b.h.data(), sizeof(float) * 256);
            std::memcpy(&combined[256], reg >= 0 ? &b.nodes[size_t(reg) * 64] : b.mean.data(), sizeof(float) * 64);
            std::memcpy(&combined[320], row, sizeof(float) * card_col_);
            std::memcpy(&combined[320 + card_col_], row + card_col_ + 1, sizeof(float) * (F - card_col_));
            std::memcpy(&combined[320 + F], &emb[card * 16], sizeof(float) * 16);
            linear("spatial_action_scorer.0", combined.data(), rin, 192, z1.data());
            linear("policy_delta.0", combined.data(), rin, 128, zd.data());
        } else {
            const size_t rr = size_t(reg >= 0 ? reg : regions_);
            for (int o = 0; o < 192; ++o) z1[o] = b.base1[o] + b.reg1[rr * 192 + o] + card1_[card * 192 + o];
            for (int o = 0; o < 128; ++o) zd[o] = b.base_d[o] + b.reg_d[rr * 128 + o] + card_d_[card * 128 + o];
            for (int k = 0; k < A; ++k) {
                if (k == card_col_) continue;
                const float v = row[k];
                if (v == 0.0f) continue;
                const int col = 320 + (k < card_col_ ? k : k - 1);
                for (int o = 0; o < 192; ++o) z1[o] += w1[size_t(o) * rin + col] * v;
                for (int o = 0; o < 128; ++o) zd[o] += wd[size_t(o) * rin + col] * v;
            }
        }
        for (float& v : z1) v = relu(v);
        linear("spatial_action_scorer.3", z1.data(), 192, 64, z2.data());
        for (float& v : z2) v = relu(v);
        float s, d;
        linear("spatial_action_scorer.5", z2.data(), 64, 1, &s);
        for (float& v : zd) v = relu(v);
        linear("policy_delta.2", zd.data(), 128, 1, &d);
        out[r] = s + d;
    }
}

void Net::softmax(const float* logits, size_t n, float* out) {
    if (!n) return;
    float m = logits[0];
    for (size_t i = 1; i < n; ++i) m = logits[i] > m ? logits[i] : m;
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { out[i] = std::exp(logits[i] - m); sum += out[i]; }
    for (size_t i = 0; i < n; ++i) out[i] /= sum;
}

}  // namespace awnet
