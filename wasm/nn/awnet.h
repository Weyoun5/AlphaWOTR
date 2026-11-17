// AlphaWOTR's network forward pass without torch: src/ai.py WOTRValueNet (arch 'base') in eval mode.
//
//   board GCN 2 x (30|64 -> 64, D^-1/2 A D^-1/2, BatchNorm, residual, ReLU), mean pool, projection 64
//   cards Embedding(97,16) over the six hand slots, masked mean, Linear 16->32 + ReLU
//   body concat(64 + 32 + 1589) -> 512 -> 256 (BatchNorm, ReLU; dropout is the identity in eval)
//   value head 256 -> 128 (BatchNorm, ReLU) -> 1, tanh or 2*tanh(x/2) ('scaled2')
//   policy per action row: concat(body 256, region node embedding 64 | mean of the nodes, row without column 33 (198),
//          card embedding of column 33 (16)) = 534 -> 192 -> 64 -> 1, plus the residual delta 534 -> 128 -> 1
//
// Weights come from wasm/tools/export_fixtures.py (the 'awn' format, documented there). The same class serves the
// seats' league nets (priors) and the separately trained value nets (WIN seats' leaf values).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace awnet {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> f;      // f32 tensors
    std::vector<int32_t> i;    // i32 tensors (the GCN's edge lists)
};

// What the body computes once per leaf; the action rows of that leaf reuse it.
struct Body {
    std::vector<float> h;        // [256] backbone output
    std::vector<float> nodes;    // [105 * 64] GCN node embeddings (layer 2 output)
    std::vector<float> mean;     // [64] mean of the node embeddings (non-regional rows)
    // fast policy path: the first layers' parts that do not depend on the row
    std::vector<float> base1, base_d;   // [192], [128]: W_bb . h + bias
    std::vector<float> reg1, reg_d;     // [106 * 192], [106 * 128]: W_region . node (row 105 = the mean)
};

class Net {
public:
    bool load(const uint8_t* data, size_t size, std::string& error);
    int obs_dim() const { return obs_dim_; }
    int act_dim() const { return act_dim_; }
    const std::string& meta_text() const { return meta_; }

    void body(const float* obs, Body& out, bool fast_policy) const;
    float value(const Body& b) const;
    // logits of `rows` action rows (rows x act_dim floats, regions -1 = non-regional). fast = the factored first layer
    // (row-independent parts computed once per leaf, card parts once per net, zero features skipped).
    void logits(const Body& b, const float* actions, const int64_t* regions, size_t rows, float* out, bool fast) const;
    static void softmax(const float* logits, size_t n, float* out);

private:
    const Tensor& t(const std::string& name) const;
    void linear(const std::string& prefix, const float* x, int in, int out, float* y) const;
    void batchnorm(const std::string& prefix, float* x, int n) const;
    int card_index(float raw) const;

    std::unordered_map<std::string, Tensor> tensors_;
    std::string meta_;
    int obs_dim_ = 0, act_dim_ = 0, regions_ = 0, node_feat_ = 0, card_start_ = 0, hand_ = 0, card_pad_ = 0, card_col_ = 0;
    bool scaled2_ = false;
    std::vector<int32_t> esrc_, edst_;
    std::vector<float> dis_;
    std::vector<float> card1_, card_d_;   // [97 * 192], [97 * 128]: card-embedding parts of the first policy layers
};

}  // namespace awnet
