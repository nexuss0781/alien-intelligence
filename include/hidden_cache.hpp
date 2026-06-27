#pragma once
#include "model.hpp"
#include "dataloader.hpp"
#include <vector>
#include <string>

namespace ai2 {

class HiddenCache {
public:
    void build(Model& model, DataLoader& loader);

    // Save/load cache to/from disk (~3GB binary file)
    void save(const std::string& path) const;
    bool load(const std::string& path);

    Index num_batches() const { return n_batches_; }
    Index batch_size() const { return batch_size_; }
    Index seq_len() const { return seq_len_; }
    Index d_model() const { return d_model_; }
    Index n_experts() const { return n_experts_; }
    Index k_experts() const { return k_experts_; }

    const float* get_hidden_batch(Index batch_idx) const;
    const float* get_mixture_batch(Index batch_idx) const;
    void get_routing_batch(Index batch_idx,
                           const int*& expert_idxs,
                           const float*& expert_wgts) const;
    const Mat& get_targets(Index batch_idx) const { return target_cache_[batch_idx]; }

private:
    Index n_batches_ = 0;
    Index batch_size_ = 0;
    Index seq_len_ = 0;
    Index d_model_ = 0;
    Index n_experts_ = 0;
    Index k_experts_ = 0;

    std::vector<std::vector<float>> hidden_cache_;
    std::vector<std::vector<float>> mixture_cache_;
    std::vector<std::vector<int>> routing_idx_cache_;
    std::vector<std::vector<float>> routing_wgt_cache_;
    std::vector<Mat> target_cache_;
};

} // namespace ai2
