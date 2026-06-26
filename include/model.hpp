#pragma once
#include "types.hpp"
#include "slie.hpp"
#include "lssc.hpp"
#include "stre.hpp"
#include "uq.hpp"
#include "ataa.hpp"
#include "ssog.hpp"
#include "tokenizer.hpp"
#include <memory>

namespace ai2 {

struct TrainingMetrics {
    Real loss = 0;
    Real accuracy = 0;
    Real perplexity = 0;
    Real sheaf_conflict = 0;
    Real conformal_set_size = 0;
};

struct ModelConfig {
    Index vocab_size = 256;
    Index d_model = 128;
    Index d_state = 64;
    Index n_rf = 128;
    Index d_node = 64;
    Index max_degree = 8;
    Index n_lsh_tables = 4;
    Index n_ensemble = 3;
    Index n_experts = 32;
    Index k_experts = 4;
    Index d_task = 32;
    Index r_lora = 8;
    Index reservoir_size = 1000;
    Index sketch_rank = 16;
    Index n_layers = 3;
    Index k_hashes = 4;
    Index m_buckets = 16384;
    Index d_pos = 8;
    Index sketch_width = 1024;
    Index sketch_depth = 4;
    Index lsh_dim = 16;
};

class Model {
public:
    Model(const ModelConfig& cfg);

    // Forward pass for a batch of sequences
    // Input: tokens [batch_size x seq_len]
    // Output: logits [batch_size x seq_len x vocab_size]
    // Returns training metrics
    TrainingMetrics forward(const Mat& tokens, const Mat& targets);

    // Get logits from last forward pass
    std::vector<std::vector<Vec>> logits() const { return logits_; }

    // Access components for gradient updates
    SLIE& slie() { return *slie_; }
    LSSC& lssc() { return *lssc_; }
    STRE& stre() { return *stre_; }
    UQ& uq() { return *uq_; }
    ATAA& ataa() { return *ataa_; }
    SSOG& ssog() { return *ssog_; }

    const ModelConfig& config() const { return cfg_; }

    // Parameter count (trainable)
    Index trainable_params() const;

    // Reset model state for new sequence
    void reset_state();

private:
    ModelConfig cfg_;

    std::unique_ptr<SLIE> slie_;
    std::unique_ptr<LSSC> lssc_;
    std::unique_ptr<STRE> stre_;
    std::unique_ptr<UQ> uq_;
    std::unique_ptr<ATAA> ataa_;
    std::unique_ptr<SSOG> ssog_;

    // Cached outputs from forward pass
    std::vector<std::vector<Vec>> logits_;  // [batch x seq_len x vocab]
    std::vector<Mat> hidden_states_;        // [batch x seq_len x d_model]

    // Loss computation helpers
    Real cross_entropy_loss(const Vec& logits, Index target) const;
    Real compute_accuracy(const Vec& logits, Index target) const;
};

} // namespace ai2
