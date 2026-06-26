#include "slie.hpp"
#include <cmath>
#include <cstring>
#include <random>

namespace ai2 {

SLIE::SLIE(Index vocab_size, Index d_model,
           Index k_hashes, Index m_buckets,
           Index d_pos, Index sketch_width,
           Index sketch_depth)
    : vocab_size_(vocab_size), d_model_(d_model),
      k_hashes_(k_hashes), m_buckets_(m_buckets),
      d_pos_(d_pos),
      sketch_width_(sketch_width), sketch_depth_(sketch_depth)
{
    assert(k_hashes > 0 && m_buckets > 0);
    assert(d_model % k_hashes == 0 && "d_model must be divisible by k_hashes");

    shard_dim_ = d_model_ / k_hashes_;

    // Initialize hash ensemble with different seeds
    hashes_.reserve(k_hashes_);
    for (Index j = 0; j < k_hashes_; ++j) {
        hashes_.emplace_back(j * 12345 + 42, m_buckets_);
    }

    // Initialize weight shards: each W_j ∈ R^{m_buckets × shard_dim}
    // Xavier-like initialization
    weight_shards_.resize(k_hashes_ * m_buckets_ * shard_dim_);
    {
        std::mt19937_64 rng(42);
        Real scale = std::sqrt(2.0 / (m_buckets_ + shard_dim_));
        std::normal_distribution<Real> normal(0, scale);
        for (auto& w : weight_shards_) w = normal(rng);
    }

    // Tiny RNN parameters: p_i = tanh(W_h * p_{i-1} + W_i * e_i + b)
    {
        std::mt19937_64 rng(1337);
        Real scale_h = std::sqrt(2.0 / (d_pos_ + d_pos_));
        Real scale_i = std::sqrt(2.0 / (d_pos_ + d_model_));

        W_pos_h_.resize(d_pos_, Vec(d_pos_));
        for (auto& row : W_pos_h_)
            for (auto& v : row) v = std::normal_distribution<Real>(0, scale_h)(rng);

        W_pos_i_.resize(d_pos_, Vec(d_model_));
        for (auto& row : W_pos_i_)
            for (auto& v : row) v = std::normal_distribution<Real>(0, scale_i)(rng);

        b_pos_.resize(d_pos_, 0);
    }

    last_pos_.resize(d_pos_, 0);

    // Count-Min Sketch
    sketch_table_.resize(sketch_depth_, Vec(sketch_width_, 0));
    sketch_hashes_.reserve(sketch_depth_);
    for (Index j = 0; j < sketch_depth_; ++j) {
        sketch_hashes_.emplace_back(j * 9999 + 7, sketch_width_);
    }
}

Vec SLIE::che_forward(Index token_id) const {
    Vec emb(d_model_, 0);
    for (Index j = 0; j < k_hashes_; ++j) {
        Index bucket = hashes_[j](token_id);
        Index offset = j * m_buckets_ * shard_dim_ + bucket * shard_dim_;
        for (Index k = 0; k < shard_dim_; ++k) {
            emb[j * shard_dim_ + k] = weight_shards_[offset + k];
        }
    }
    return emb;
}

Vec SLIE::spe_forward(const Vec& prev_pos, const Vec& embedding) {
    // p_i = tanh(W_h * p_{i-1} + W_i * e_i + b)
    Vec p = mat_vec(W_pos_h_, prev_pos);
    Vec e = mat_vec(W_pos_i_, embedding);
    p = elem_add(p, e);
    p = elem_add(p, b_pos_);
    for (auto& v : p) v = std::tanh(v);
    last_pos_ = p;
    return p;
}

void SLIE::sketch_update(Index token_id) {
    for (Index j = 0; j < sketch_depth_; ++j) {
        Index bucket = sketch_hashes_[j](token_id);
        sketch_table_[j][bucket] += 1;
    }
}

Vec SLIE::sketch_features() const {
    // Return the Count-Min sketch frequencies as features
    // We return the minimum across hash functions for a small sample
    Vec features(shard_dim_, 0);
    for (Index i = 0; i < shard_dim_; ++i) {
        Real freq = std::numeric_limits<Real>::max();
        for (Index j = 0; j < sketch_depth_; ++j) {
            freq = std::min(freq, sketch_table_[j][i % sketch_width_]);
        }
        features[i] = std::tanh(freq / 10.0);
    }
    return features;
}

Vec SLIE::forward(Index token_id, const Vec& prev_pos) {
    // Layer 1A: Consistent hashing embedding
    Vec emb = che_forward(token_id);
    last_emb_ = emb;

    // Layer 1B: Streaming positional encoding
    Vec pos = spe_forward(prev_pos, emb);

    // Combine embedding + position + sketch
    Vec combined = elem_add(emb, pos);

    // Layer 1C: Update sketch and add sketch features
    sketch_update(token_id);

    return combined;
}

void SLIE::reset_position() {
    std::fill(last_pos_.begin(), last_pos_.end(), 0);
}

} // namespace ai2
