#include "hidden_cache.hpp"
#include <iostream>
#include <cstring>

namespace ai2 {

void HiddenCache::build(Model& model, DataLoader& loader) {
    batch_size_ = loader.batch_size();
    seq_len_ = loader.seq_len();
    d_model_ = model.config().d_model;
    n_experts_ = model.ssog().n_experts();
    k_experts_ = model.ssog().k_experts();

    n_batches_ = loader.batches_per_epoch();
    std::size_t hidden_per_batch = batch_size_ * seq_len_ * d_model_;
    Index pos_per_batch = batch_size_ * seq_len_;

    hidden_cache_.resize(n_batches_);
    mixture_cache_.resize(n_batches_);
    routing_idx_cache_.resize(n_batches_);
    routing_wgt_cache_.resize(n_batches_);
    target_cache_.resize(n_batches_);

    Index total_positions = 0;
    Index vocab_size = model.config().vocab_size;
    Index d_node = model.config().d_node;

    bool have_gpu = model.gpu_ctx() != nullptr;

    std::cout << "  Pre-computing hidden states + mixture for " << n_batches_
              << " batches (" << (n_batches_ * pos_per_batch) << " positions, ~"
              << (n_batches_ * hidden_per_batch * 4 * 2 / (1024*1024)) << "MB total)..."
              << std::endl;

    loader.reset();

    // Per-batch GPU buffers for mixture computation
    std::vector<float> gpu_hidden_buf(pos_per_batch * d_model_);
    std::vector<int> gpu_idx_buf(pos_per_batch * k_experts_);
    std::vector<float> gpu_wgt_buf(pos_per_batch * k_experts_);

    for (Index bi = 0; bi < n_batches_; ++bi) {
        Batch batch = loader.next();
        if (batch.tokens.empty()) break;

        Mat tokens = batch.tokens;
        Mat& targets = batch.targets;
        target_cache_[bi] = targets;

        Index buf_sz = hidden_per_batch;
        hidden_cache_[bi].resize(buf_sz, 0.0f);
        mixture_cache_[bi].resize(buf_sz, 0.0f);
        routing_idx_cache_[bi].resize(pos_per_batch * k_experts_, 0);
        routing_wgt_cache_[bi].resize(pos_per_batch * k_experts_, 0.0f);

        Index pos_offset = 0;

        for (Index b = 0; b < batch_size_; ++b) {
            // SLIE
            Mat embeddings(seq_len_, Vec(d_model_, 0));
            Vec prev_pos(model.slie().d_pos(), 0);
            model.slie().reset_position();

            for (Index t = 0; t < seq_len_; ++t) {
                Index token = tokens[b][t];
                if (token >= vocab_size) token = vocab_size - 1;
                embeddings[t] = model.slie().forward(token, prev_pos);
                prev_pos = model.slie().last_position();
            }

            // LSSC
            Mat hidden = model.lssc().forward(embeddings);

            // STRE
            auto stre_features = model.stre().forward(hidden, model.config().n_layers);
            for (Index t = 0; t < seq_len_; ++t) {
                Vec& h = hidden[t];
                Index node_idx = t < stre_features.size() ? t : stre_features.size() - 1;
                if (node_idx < stre_features.size()) {
                    for (Index i = 0; i < std::min<Index>(d_node, d_model_); ++i) {
                        h[i] += 0.1 * stre_features[node_idx][i % d_node];
                    }
                }
            }

            // Store hidden + routing, compute mixture via CPU fallback or GPU
            for (Index t = 0; t < seq_len_; ++t) {
                const Vec& h = hidden[t];
                for (Index j = 0; j < d_model_; ++j)
                    hidden_cache_[bi][pos_offset * d_model_ + j] = static_cast<float>(h[j]);

                // SSOG routing
                std::vector<Index> experts = model.ssog().route(h);
                Vec weights = model.ssog().gating_weights(h, experts);

                for (Index k = 0; k < k_experts_; ++k) {
                    Index idx = (pos_offset * k_experts_) + k;
                    if (k < experts.size()) {
                        routing_idx_cache_[bi][idx] = static_cast<int>(experts[k]);
                        routing_wgt_cache_[bi][idx] = static_cast<float>(weights[k]);
                    } else {
                        routing_idx_cache_[bi][idx] = 0;
                        routing_wgt_cache_[bi][idx] = 0.0f;
                    }
                }

                pos_offset++;
            }
        }

        // Compute mixture on GPU
        if (have_gpu) {
            gpu::compute_mixture_batch(
                model.gpu_ctx(),
                hidden_cache_[bi].data(),
                routing_idx_cache_[bi].data(),
                routing_wgt_cache_[bi].data(),
                mixture_cache_[bi].data(),
                static_cast<int>(pos_offset));
        } else {
            // CPU fallback: compute mixture via SSOG
            pos_offset = 0;
            for (Index b = 0; b < batch_size_; ++b) {
                for (Index t = 0; t < seq_len_; ++t) {
                    Vec h(d_model_);
                    for (Index j = 0; j < d_model_; ++j)
                        h[j] = static_cast<Real>(hidden_cache_[bi][pos_offset * d_model_ + j]);
                    Vec mix = model.ssog().sparse_mixture(h);
                    for (Index j = 0; j < d_model_; ++j)
                        mixture_cache_[bi][pos_offset * d_model_ + j] = static_cast<float>(mix[j]);
                    pos_offset++;
                }
            }
        }

        total_positions += pos_offset;

        if ((bi + 1) % 20 == 0 || bi == 0 || bi == n_batches_ - 1) {
            std::size_t mb = (total_positions * d_model_ * 4 * 2) / (1024*1024);
            std::cout << "    Batch " << (bi + 1) << "/" << n_batches_
                      << "  ~" << mb << "MB cached" << std::endl;
        }
    }

    std::size_t total_mb = total_positions * d_model_ * 4 * 2 / (1024*1024);
    std::cout << "  Cache done: " << total_positions << " positions, ~"
              << total_mb << "MB RAM (hidden + mixture)" << std::endl;
}

const float* HiddenCache::get_hidden_batch(Index batch_idx) const {
    if (batch_idx >= n_batches_) return nullptr;
    return hidden_cache_[batch_idx].data();
}

const float* HiddenCache::get_mixture_batch(Index batch_idx) const {
    if (batch_idx >= n_batches_) return nullptr;
    return mixture_cache_[batch_idx].data();
}

void HiddenCache::get_routing_batch(Index batch_idx,
                                     const int*& expert_idxs,
                                     const float*& expert_wgts) const {
    if (batch_idx >= n_batches_) {
        expert_idxs = nullptr;
        expert_wgts = nullptr;
        return;
    }
    expert_idxs = routing_idx_cache_[batch_idx].data();
    expert_wgts = routing_wgt_cache_[batch_idx].data();
}

} // namespace ai2
