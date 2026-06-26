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
}

TrainingMetrics Model::forward(const Mat& tokens, const Mat& targets) {
    Index batch_size = tokens.size();
    Index seq_len = tokens[0].size();

    // Clear cached outputs
    logits_.clear();
    hidden_states_.clear();

    TrainingMetrics metrics;
    Real total_loss = 0;
    Real total_correct = 0;
    Real total_conflict = 0;
    Real total_conf_size = 0;
    Index total_tokens = 0;

    // Process each sequence in the batch
    for (Index b = 0; b < batch_size; ++b) {
        // Step 1: SLIE — encode each token
        Mat embeddings(seq_len, Vec(cfg_.d_model, 0));
        Vec prev_pos(cfg_.d_pos, 0);
        slie_->reset_position();

        for (Index t = 0; t < seq_len; ++t) {
            Index token = tokens[b][t];
            if (token >= cfg_.vocab_size) token = cfg_.vocab_size - 1;
            embeddings[t] = slie_->forward(token, prev_pos);
            prev_pos = slie_->last_position();
        }

        // Step 2: LSSC — process sequence
        Mat hidden = lssc_->forward(embeddings);

        // Step 3: STRE — build reasoning graph
        auto features = stre_->forward(hidden, cfg_.n_layers);

        // Compute sheaf conflict
        Real conflict = 0;
        if (!features.empty()) {
            conflict = stre_->compute_conflict(features);
        }
        total_conflict += conflict;

        // Step 4 & 6: For each position, compute UQ + SSOG output
        // Collect hidden states for this batch
        std::vector<Vec> seq_logits(seq_len, Vec(cfg_.vocab_size, 0));

        for (Index t = 0; t < seq_len; ++t) {
            const Vec& h = hidden[t];

            // UQ — uncertainty quantification
            // Use a placeholder distribution for conformal scoring
            // In full training, this would use ensemble predictions
            Vec dummy_probs(cfg_.vocab_size, 1.0 / cfg_.vocab_size);
            auto uq_result = uq_->step(dummy_probs, conflict, 0.1);

            // STRE feature enhancement
            Vec enhanced_h = h;
            if (t < features.size()) {
                for (Index i = 0; i < std::min(cfg_.d_node, cfg_.d_model); ++i) {
                    enhanced_h[i] += 0.1 * features[t][i % cfg_.d_node];
                }
            }

            // SSOG — output distribution
            auto output = ssog_->forward(enhanced_h,
                                          uq_result.conformal_uncertainty,
                                          uq_result.conformal_set);

            // Store logits (use log of calibrated probs for numerical stability)
            for (Index i = 0; i < cfg_.vocab_size; ++i) {
                seq_logits[t][i] = std::log(output.calibrated_probs[i] + EPS);
            }

            // Compute metrics
            Index target = targets[b][t];
            Real loss = cross_entropy_loss(seq_logits[t], target);
            total_loss += loss;
            total_correct += compute_accuracy(seq_logits[t], target);
            total_conf_size += uq_result.conformal_set.size();
            total_tokens++;
        }

        logits_.push_back(seq_logits);
    }

    // Average metrics
    if (total_tokens > 0) {
        metrics.loss = total_loss / total_tokens;
        metrics.accuracy = total_correct / total_tokens;
        metrics.perplexity = std::exp(metrics.loss);
        metrics.sheaf_conflict = total_conflict / batch_size;
        metrics.conformal_set_size = total_conf_size / total_tokens;
    }

    return metrics;
}

Real Model::cross_entropy_loss(const Vec& logits, Index target) const {
    if (target >= logits.size()) return 0;
    // Cross-entropy: -log(p_target) where p = softmax(logits)
    // Numerically stable: -logits[target] + log(sum(exp(logits)))
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
    // Count parameters in SSOG (output layer) and UQ
    Index count = 0;

    // SSOG: expert weights + biases
    for (Index e = 0; e < cfg_.n_experts; ++e) {
        count += cfg_.d_model * cfg_.d_model;  // expert weights
        count += cfg_.d_model;                  // expert bias
    }

    // SSOG: output projection
    count += cfg_.vocab_size * cfg_.d_model;  // W_out
    count += cfg_.vocab_size;                 // b_out

    // SSOG: gating
    count += cfg_.d_model * cfg_.d_model;  // W_gate

    return count;
}

void Model::reset_state() {
    slie_->reset_position();
}

} // namespace ai2
