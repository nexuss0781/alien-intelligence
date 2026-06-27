#pragma once
#include "model.hpp"
#include "dataloader.hpp"
#include <vector>

namespace ai2 {

// Pre-computes and caches hidden states (after SLIE+LSSC+STRE) for all tokens in RAM.
// With d_model=256 and 1.3M tokens: ~1.3GB as float.
// With d_model=256 and 1.3M tokens: ~2.6GB as double.
// Also caches SSOG routing info (expert indices + weights) per position.
// Total: ~3-4GB RAM usage.
class HiddenCache {
public:
    // Build cache by running the full forward pipeline on all batches
    void build(Model& model, DataLoader& loader);

    Index num_batches() const { return n_batches_; }
    Index batch_size() const { return batch_size_; }
    Index seq_len() const { return seq_len_; }
    Index d_model() const { return d_model_; }
    Index n_experts() const { return n_experts_; }
    Index k_experts() const { return k_experts_; }

    // Get flattened hidden states for a batch (as float, GPU-ready)
    // Output: [batch_size * seq_len * d_model] floats
    // Returns pointer into internal buffer (valid until next get_batch call)
    const float* get_hidden_batch(Index batch_idx) const;

    // Get routing info for a batch
    // expert_idxs: [batch_size * seq_len * k_experts] ints
    // expert_wgts: [batch_size * seq_len * k_experts] floats
    void get_routing_batch(Index batch_idx,
                           const int*& expert_idxs,
                           const float*& expert_wgts) const;

    // Get targets for a batch
    const Mat& get_targets(Index batch_idx) const { return target_cache_[batch_idx]; }

private:
    Index n_batches_ = 0;
    Index batch_size_ = 0;
    Index seq_len_ = 0;
    Index d_model_ = 0;
    Index n_experts_ = 0;
    Index k_experts_ = 0;

    // Flat buffer: [n_batches][batch_size * seq_len][d_model]
    std::vector<std::vector<float>> hidden_cache_;

    // Routing cache: [n_batches][position][k_experts]
    std::vector<std::vector<int>> routing_idx_cache_;
    std::vector<std::vector<float>> routing_wgt_cache_;

    // Target cache: [n_batches][batch_size][seq_len]
    std::vector<Mat> target_cache_;
};

} // namespace ai2
