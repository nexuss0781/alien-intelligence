#pragma once
#include "types.hpp"
#include <array>
#include <vector>
#include <memory>
#include <unordered_map>

namespace ai2 {

// Sub-Linear Input Encoding (SLIE) — Component 1
// Provides O(1) per-token encoding via:
//   1A: Consistent Hashing Embedding (CHE)
//   1B: Streaming Positional Encoding (SPE)
//   1C: Sketch-Based Feature Extraction (Count-Min Sketch)
class SLIE {
public:
    // vocab_size: |V|, d_model: embedding dimension
    // k_hashes: number of hash functions in ensemble
    // m_buckets: number of buckets per hash table
    // d_pos: tiny RNN hidden size for positional encoding
    // sketch_width, sketch_depth: Count-Min sketch parameters
    SLIE(Index vocab_size, Index d_model,
         Index k_hashes = 4, Index m_buckets = 65536,
         Index d_pos = 8, Index sketch_width = 2048,
         Index sketch_depth = 4);

    // Layer 1A: Consistent Hashing Embedding — O(1)
    // Maps token_id -> d_model-dimensional embedding via hash ensemble
    Vec che_forward(Index token_id) const;

    // Layer 1B: Streaming Positional Encoding — O(1)
    // Updates tiny RNN state given previous position encoding and current embedding
    Vec spe_forward(const Vec& prev_pos, const Vec& embedding);

    // Layer 1C: Count-Min Sketch update + query — O(1)
    void sketch_update(Index token_id);
    Vec sketch_features() const;

    // Full forward: combines all three sub-components — O(1) per token
    Vec forward(Index token_id, const Vec& prev_pos);

    // Accessors
    Index d_model() const { return d_model_; }
    Index d_pos() const { return d_pos_; }
    const Vec& last_embedding() const { return last_emb_; }
    const Vec& last_position() const { return last_pos_; }

    // Reset positional state
    void reset_position();

private:
    Index vocab_size_;
    Index d_model_;
    Index k_hashes_;
    Index m_buckets_;
    Index d_pos_;

    // Hash ensemble (fixed, O(1) evaluation)
    std::vector<UniversalHash> hashes_;

    // Learnable weight shards W_j ∈ R^{m_buckets × (d_model / k_hashes)}
    Vec weight_shards_;  // flattened, shard j is at [j * (d_model/k_hashes) * m_buckets ..]

    Index shard_dim_;  // d_model / k_hashes

    // Tiny RNN for positional encoding
    // p_i = tanh(W_pos * [p_{i-1}; e_i] + b_pos)
    Mat W_pos_h_;  // hidden->hidden d_pos x d_pos
    Mat W_pos_i_;  // input->hidden d_pos x d_model
    Vec b_pos_;    // bias
    Vec last_pos_; // current positional state p_i

    Vec last_emb_;

    // Count-Min Sketch
    // sketch_depth rows, each of sketch_width counters
    Index sketch_width_;
    Index sketch_depth_;
    Mat sketch_table_;
    std::vector<UniversalHash> sketch_hashes_;
};

} // namespace ai2
