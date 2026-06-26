#include "ssog.hpp"
#include <random>
#include <cmath>
#include <unordered_set>
#include <numeric>

namespace ai2 {

SSOG::SSOG(Index d_model, Index n_vocab, Index n_experts,
           Index k_experts, Index n_lsh_tables)
    : d_model_(d_model), n_vocab_(n_vocab),
      n_experts_(n_experts), k_experts_(k_experts),
      n_lsh_tables_(n_lsh_tables)
{
    std::mt19937_64 rng(2025);

    // Initialize LSH functions for routing
    lsh_funcs_.reserve(n_lsh_tables_);
    for (Index t = 0; t < n_lsh_tables_; ++t) {
        lsh_funcs_.emplace_back(d_model_, t * 5555 + 33);
    }

    // Initialize expert weights and biases
    expert_weights_.resize(n_experts_);
    expert_biases_.resize(n_experts_);
    Real scale_exp = std::sqrt(2.0 / (d_model_ + d_model_));
    for (Index e = 0; e < n_experts_; ++e) {
        expert_weights_[e].resize(d_model_, Vec(d_model_));
        for (auto& row : expert_weights_[e])
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale_exp)(rng);

        expert_biases_[e].resize(d_model_, 0);
    }

    // Initialize output projection
    Real scale_out = std::sqrt(2.0 / (d_model_ + n_vocab_));
    W_out_.resize(n_vocab_, Vec(d_model_));
    for (auto& row : W_out_)
        for (auto& v : row)
            v = std::normal_distribution<Real>(0, scale_out)(rng);
    b_out_.resize(n_vocab_, 0);

    // Gating projection
    Real scale_gate = std::sqrt(2.0 / (d_model_ + d_model_));
    W_gate_.resize(d_model_, Vec(d_model_));
    for (auto& row : W_gate_)
        for (auto& v : row)
            v = std::normal_distribution<Real>(0, scale_gate)(rng);
}

std::vector<Index> SSOG::route(const Vec& h) const {
    // Use LSH to route to expert buckets
    std::unordered_set<Index> expert_set;
    for (Index t = 0; t < n_lsh_tables_; ++t) {
        Index bucket = lsh_funcs_[t](h, n_experts_);
        expert_set.insert(bucket);
    }

    // If we got too many, truncate to k_experts
    // If too few, pad with nearest neighbors
    std::vector<Index> experts(expert_set.begin(), expert_set.end());

    // Ensure exactly k_experts (or as many as available)
    if (experts.size() > k_experts_) {
        experts.resize(k_experts_);
    } else if (experts.size() < k_experts_) {
        // Fill remaining with round-robin from remaining experts
        for (Index i = 0; i < n_experts_ && experts.size() < k_experts_; ++i) {
            if (std::find(experts.begin(), experts.end(), i) == experts.end()) {
                experts.push_back(i);
            }
        }
    }

    return experts;
}

Vec SSOG::gating_weights(const Vec& h, const std::vector<Index>& experts) const {
    // Compute gating scores via learned projection
    Vec scores(experts.size());
    Vec gate_proj = mat_vec(W_gate_, h);

    for (Index i = 0; i < experts.size(); ++i) {
        // Score = W_gate[expert_i] · h  (simplified — just use a linear score)
        scores[i] = dot(gate_proj, expert_weights_[experts[i]][0]);
    }

    return softmax(scores);
}

Vec SSOG::expert_forward(Index expert_id, const Vec& h) const {
    // E_i(h) = W_i · h + b_i
    Vec result = mat_vec(expert_weights_[expert_id], h);
    return elem_add(result, expert_biases_[expert_id]);
}

Vec SSOG::sparse_mixture(const Vec& h) const {
    std::vector<Index> experts = route(h);
    Vec weights = gating_weights(h, experts);

    Vec result(d_model_, 0);
    for (Index i = 0; i < experts.size(); ++i) {
        Vec expert_out = expert_forward(experts[i], h);
        for (Index j = 0; j < d_model_; ++j) {
            result[j] += weights[i] * expert_out[j];
        }
    }
    return result;
}

Vec SSOG::base_distribution(const Vec& mixture) const {
    // p(y) = softmax(W_out · mixture + b_out)
    Vec logits = mat_vec(W_out_, mixture);
    logits = elem_add(logits, b_out_);
    return softmax(logits);
}

Vec SSOG::calibrated_distribution(const Vec& base_probs,
                                   Real conformal_uncertainty,
                                   const std::vector<Index>& conformal_set) const
{
    if (conformal_uncertainty < EPS || conformal_set.empty()) {
        return base_probs;
    }

    // p̃(y) = (1-C) * p(y) + C * 1_C(y) / |C|
    // This is self-normalizing: sum = (1-C)*1 + C*1 = 1
    Vec cal = base_probs;

    Real inv_set_size = 1.0 / conformal_set.size();
    Real c = conformal_uncertainty;
    Real one_minus_c = 1.0 - c;

    for (Index i = 0; i < n_vocab_ && i < base_probs.size(); ++i) {
        cal[i] = base_probs[i] * one_minus_c;
    }

    for (Index idx : conformal_set) {
        cal[idx] += c * inv_set_size;
    }

    return cal;
}

SSOG::Output SSOG::forward(const Vec& h, Real conformal_uncertainty,
                            const std::vector<Index>& conformal_set) const
{
    Output out;
    out.mixture = sparse_mixture(h);
    out.base_probs = base_distribution(out.mixture);
    out.calibrated_probs = calibrated_distribution(
        out.base_probs, conformal_uncertainty, conformal_set);
    return out;
}

} // namespace ai2
