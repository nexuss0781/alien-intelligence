#include "hidden_cache.hpp"
#include <iostream>

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
    routing_idx_cache_.resize(n_batches_);
    routing_wgt_cache_.resize(n_batches_);
    target_cache_.resize(n_batches_);

    Index total_positions = 0;

    std::cout << "  Pre-computing hidden states for " << n_batches_
              << " batches (" << (n_batches_ * pos_per_batch) << " positions, "
              << (n_batches_ * hidden_per_batch * 4 / (1024*1024)) << "MB)..."
              << std::endl;

    loader.reset();
    auto& slie = model.slie();
    auto& lssc = model.lssc();
    auto& stre = model.stre();
    auto& ssog = model.ssog();
    Index vocab_size = model.config().vocab_size;
    Index d_node = model.config().d_node;

    for (Index bi = 0; bi < n_batches_; ++bi) {
        Batch batch = loader.next();
        if (batch.tokens.empty()) break;

        Mat tokens = batch.tokens;
        Mat& targets = batch.targets;
        target_cache_[bi] = targets;

        hidden_cache_[bi].resize(hidden_per_batch, 0.0f);
        routing_idx_cache_[bi].resize(pos_per_batch * k_experts_, 0);
        routing_wgt_cache_[bi].resize(pos_per_batch * k_experts_, 0.0f);

        Index pos_offset = 0;

        for (Index b = 0; b < batch_size_; ++b) {
            // SLIE
            Mat embeddings(seq_len_, Vec(d_model_, 0));
            Vec prev_pos(slie.d_pos(), 0);
            slie.reset_position();

            for (Index t = 0; t < seq_len_; ++t) {
                Index token = tokens[b][t];
                if (token >= vocab_size) token = vocab_size - 1;
                embeddings[t] = slie.forward(token, prev_pos);
                prev_pos = slie.last_position();
            }

            // LSSC
            Mat hidden = lssc.forward(embeddings);

            // STRE
            auto stre_features = stre.forward(hidden, model.config().n_layers);
            for (Index t = 0; t < seq_len_; ++t) {
                Vec& h = hidden[t];
                Index node_idx = t < stre_features.size() ? t : stre_features.size() - 1;
                if (node_idx < stre_features.size()) {
                    for (Index i = 0; i < std::min<Index>(d_node, d_model_); ++i) {
                        h[i] += 0.1 * stre_features[node_idx][i % d_node];
                    }
                }
            }

            // Store hidden + routing
            for (Index t = 0; t < seq_len_; ++t) {
                const Vec& h = hidden[t];

                // Store hidden as float
                for (Index j = 0; j < d_model_; ++j)
                    hidden_cache_[bi][pos_offset * d_model_ + j] = static_cast<float>(h[j]);

                // SSOG routing
                std::vector<Index> experts = ssog.route(h);
                Vec weights = ssog.gating_weights(h, experts);

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

        total_positions += pos_offset;

        if ((bi + 1) % 20 == 0 || bi == 0 || bi == n_batches_ - 1) {
            std::size_t mb = hidden_cache_[bi].size() * sizeof(float) / (1024*1024);
            std::cout << "    Batch " << (bi + 1) << "/" << n_batches_
                      << "  cache: ~" << (total_positions * d_model_ * 4 / (1024*1024))
                      << "MB so far" << std::endl;
        }
    }

    std::size_t total_mb = total_positions * d_model_ * 4 / (1024*1024);
    std::cout << "  Hidden cache done: " << total_positions << " positions, ~"
              << total_mb << "MB in RAM" << std::endl;
}

const float* HiddenCache::get_hidden_batch(Index batch_idx) const {
    if (batch_idx >= n_batches_) return nullptr;
    return hidden_cache_[batch_idx].data();
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
