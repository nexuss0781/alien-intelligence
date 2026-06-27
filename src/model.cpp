#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ai2 {

Model::Model(const ModelConfig& cfg) : cfg_(cfg) {
    slie_ = std::make_unique<SLIE>(
        cfg.vocab_size, cfg.d_model, cfg.k_hashes, cfg.m_buckets,
        cfg.d_pos, cfg.sketch_width, cfg.sketch_depth);

    lssc_ = std::make_unique<LSSC>(cfg.d_model, cfg.d_state, cfg.n_rf);

    stre_ = std::make_unique<STRE>(cfg.d_node, cfg.max_degree,
                                   cfg.lsh_dim, cfg.n_lsh_tables);

    uq_ = std::make_unique<UQ>(cfg.vocab_size, cfg.reservoir_size,
                               cfg.n_ensemble, cfg.d_model);

    ataa_ = std::make_unique<ATAA>(cfg.d_model, cfg.d_task, cfg.r_lora,
                                   cfg.n_examples, cfg.sketch_rank);

    ssog_ = std::make_unique<SSOG>(cfg.d_model, cfg.vocab_size,
                                   cfg.n_experts, cfg.k_experts,
                                   cfg.n_lsh_tables);

    Index n_vocab = cfg.vocab_size;
    Index d_model = cfg.d_model;
    param_W_out_.resize(n_vocab * d_model, 0);
    grad_W_out_.resize(n_vocab * d_model, 0);
    param_b_out_.resize(n_vocab, 0);
    grad_b_out_.resize(n_vocab, 0);
    sync_params_from_ssog();

    gpu_ctx_ = gpu::init(ssog_->W_out(), ssog_->b_out());
    if (gpu_ctx_) {
        std::cout << "  GPU acceleration enabled (gradient computation)" << std::endl;
    } else {
        std::cout << "  GPU not available, using CPU fallback" << std::endl;
    }
}

Model::~Model() {
    gpu::destroy(gpu_ctx_);
}

TrainingMetrics Model::forward(const Mat& tokens, const Mat& targets) {
    Index batch_size = tokens.size();
    Index seq_len = batch_size > 0 ? tokens[0].size() : 0;

    logits_.clear();
    hidden_out_.clear();

    TrainingMetrics metrics;
    Real total_loss = 0;
    Real total_correct = 0;
    Real total_conflict = 0;
    Real total_tokens_valid = 0;

    for (Index b = 0; b < batch_size; ++b) {
        Mat embeddings(seq_len, Vec(cfg_.d_model, 0));
        Vec prev_pos(cfg_.d_pos, 0);
        slie_->reset_position();

        for (Index t = 0; t < seq_len; ++t) {
            Index token = tokens[b][t];
            if (token >= cfg_.vocab_size) token = cfg_.vocab_size - 1;
            embeddings[t] = slie_->forward(token, prev_pos);
            prev_pos = slie_->last_position();
        }

        Mat hidden = lssc_->forward(embeddings);

        auto stre_features = stre_->forward(hidden, cfg_.n_layers);
        Real conflict = 0;
        if (!stre_features.empty()) {
            conflict = stre_->compute_conflict(stre_features);
        }
        total_conflict += conflict;

        for (Index t = 0; t < seq_len; ++t) {
            Vec& h = hidden[t];
            Index node_idx = t < stre_features.size() ? t : stre_features.size() - 1;
            if (node_idx < stre_features.size()) {
                for (Index i = 0; i < std::min(cfg_.d_node, cfg_.d_model); ++i) {
                    h[i] += 0.1 * stre_features[node_idx][i % cfg_.d_node];
                }
            }
        }

        // Cache hidden states
        for (Index t = 0; t < seq_len; ++t) {
            if (hidden_out_.size() <= b * seq_len + t)
                hidden_out_.resize((b + 1) * seq_len);
            hidden_out_[b * seq_len + t] = hidden[t];
        }

        std::vector<Vec> seq_logits(seq_len, Vec(cfg_.vocab_size, 0));

        for (Index t = 0; t < seq_len; ++t) {
            const Vec& h = hidden[t];
            Index target = targets[b][t];
            if (target == 0) continue;

            Vec dummy_probs(cfg_.vocab_size, 1.0 / cfg_.vocab_size);
            auto uq_result = uq_->step(dummy_probs, conflict, 0.1);

            auto output = ssog_->forward(h,
                                          uq_result.conformal_uncertainty,
                                          uq_result.conformal_set);
            for (Index i = 0; i < cfg_.vocab_size; ++i) {
                seq_logits[t][i] = std::log(output.calibrated_probs[i] + EPS);
            }

            Real loss = cross_entropy_loss(seq_logits[t], target);
            total_loss += loss;
            total_correct += compute_accuracy(seq_logits[t], target);
            total_tokens_valid += 1;
        }

        logits_.push_back(seq_logits);
    }

    if (total_tokens_valid > 0) {
        metrics.loss = total_loss / total_tokens_valid;
        metrics.accuracy = total_correct / total_tokens_valid;
        metrics.perplexity = std::exp(metrics.loss);
        metrics.sheaf_conflict = total_conflict / batch_size;
    }

    return metrics;
}

void Model::compute_gradients(const Mat& targets) {
    Index batch_size = targets.size();
    Index seq_len = batch_size > 0 ? targets[0].size() : 0;

    if (gpu_ctx_) {
        gpu::batched_gradient(gpu_ctx_, hidden_out_, logits_, targets,
                               grad_W_out_, grad_b_out_,
                               hidden_out_.size(),
                               cfg_.vocab_size, cfg_.d_model,
                               batch_size, seq_len);
        return;
    }

    // CPU fallback
    for (Index b = 0; b < batch_size; ++b) {
        for (Index t = 0; t < seq_len; ++t) {
            Index target = targets[b][t];
            if (target == 0) continue;
            if (b * seq_len + t >= hidden_out_.size()) continue;
            if (b >= logits_.size() || t >= logits_[b].size()) continue;

            const Vec& h = hidden_out_[b * seq_len + t];
            const Vec& logits = logits_[b][t];
            Index n_vocab = cfg_.vocab_size;
            Index d_model = cfg_.d_model;

            Vec p = softmax(logits);
            for (Index i = 0; i < n_vocab; ++i) {
                Real dL_dz = p[i] - (i == target ? 1.0 : 0.0);
                grad_b_out_[i] += dL_dz;
                for (Index j = 0; j < d_model; ++j) {
                    grad_W_out_[i * d_model + j] += dL_dz * h[j];
                }
            }
        }
    }
}

void Model::zero_gradients() {
    std::fill(grad_W_out_.begin(), grad_W_out_.end(), 0);
    std::fill(grad_b_out_.begin(), grad_b_out_.end(), 0);
}

void Model::sync_params_to_ssog() {
    Index n_vocab = cfg_.vocab_size;
    Index d_model = cfg_.d_model;
    auto& W_out = ssog_->W_out();
    auto& b_out = ssog_->b_out();
    for (Index i = 0; i < n_vocab; ++i) {
        b_out[i] = param_b_out_[i];
        for (Index j = 0; j < d_model; ++j) {
            W_out[i][j] = param_W_out_[i * d_model + j];
        }
    }
    if (gpu_ctx_) {
        gpu::sync_weights(gpu_ctx_, param_W_out_, param_b_out_,
                          n_vocab, d_model);
    }
}

void Model::sync_params_from_ssog() {
    Index n_vocab = cfg_.vocab_size;
    Index d_model = cfg_.d_model;
    const auto& W_out = ssog_->W_out();
    const auto& b_out = ssog_->b_out();
    for (Index i = 0; i < n_vocab; ++i) {
        param_b_out_[i] = b_out[i];
        for (Index j = 0; j < d_model; ++j) {
            param_W_out_[i * d_model + j] = W_out[i][j];
        }
    }
}

Real Model::cross_entropy_loss(const Vec& logits, Index target) const {
    if (target >= logits.size()) return 0;
    Real max_logit = *std::max_element(logits.begin(), logits.end());
    Real sum_exp = 0;
    for (auto& l : logits) sum_exp += std::exp(l - max_logit);
    Real log_sum_exp = max_logit + std::log(sum_exp);
    return log_sum_exp - logits[target];
}

Real Model::compute_accuracy(const Vec& logits, Index target) const {
    if (target >= logits.size()) return 0;
    Index pred = std::max_element(logits.begin(), logits.end()) - logits.begin();
    return (pred == target) ? 1.0 : 0.0;
}

Index Model::trainable_params() const {
    Index count = 0;
    count += cfg_.vocab_size * cfg_.d_model;
    count += cfg_.vocab_size;
    for (Index e = 0; e < cfg_.n_experts; ++e) {
        count += cfg_.d_model * cfg_.d_model;
        count += cfg_.d_model;
    }
    count += cfg_.d_model * cfg_.d_model;
    return count;
}

void Model::reset_state() {
    slie_->reset_position();
}

} // namespace ai2
