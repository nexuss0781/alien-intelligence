#include "uq.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace ai2 {

UQ::UQ(Index n_vocab, Index reservoir_size,
       Index n_ensemble, Index d_model)
    : n_vocab_(n_vocab), reservoir_size_(reservoir_size),
      n_ensemble_(n_ensemble), d_model_(d_model),
      reservoir_sum_(0), ensemble_count_(0)
{
    // Initialize fusion parameters
    std::mt19937_64 rng(9876);
    W_u_.resize(3);
    for (auto& v : W_u_) v = std::normal_distribution<Real>(0, 0.1)(rng);
    b_u_ = 0;
}

Real UQ::conformal_score(Real prob_true) const {
    // s_i = 1 - p(true_i | context)
    return std::clamp(1.0 - prob_true, 0.0, 1.0);
}

void UQ::update_reservoir(Real score) {
    reservoir_.push_back(score);
    reservoir_sum_ += score;
    if (reservoir_.size() > reservoir_size_) {
        reservoir_sum_ -= reservoir_.front();
        reservoir_.pop_front();
    }
}

Real UQ::quantile_threshold(Real alpha) const {
    if (reservoir_.empty()) return 0.9;  // default conservative threshold

    std::vector<Real> scores(reservoir_.begin(), reservoir_.end());
    std::sort(scores.begin(), scores.end());

    Index idx = static_cast<Index>((1.0 - alpha) * scores.size());
    idx = std::min(idx, (Index)(scores.size() - 1));
    return scores[idx];
}

std::vector<Index> UQ::prediction_set(const Vec& probs, Real alpha) const {
    Real q = quantile_threshold(alpha);
    std::vector<Index> pred_set;
    pred_set.reserve(probs.size());
    for (Index i = 0; i < probs.size() && i < n_vocab_; ++i) {
        if (1.0 - probs[i] <= q) {
            pred_set.push_back(i);
        }
    }
    return pred_set;
}

void UQ::add_ensemble_prediction(const Vec& probs, Index member_id) {
    if (ensemble_probs_.empty()) {
        ensemble_probs_.resize(n_ensemble_, Vec(n_vocab_, 0));
    }
    if (member_id < n_ensemble_) {
        ensemble_probs_[member_id] = probs;
        ensemble_count_ = std::max(ensemble_count_, (Index)(member_id + 1));
    }
}

void UQ::clear_ensemble() {
    ensemble_probs_.clear();
    ensemble_count_ = 0;
}

Vec UQ::ensemble_distribution() const {
    if (ensemble_count_ == 0) return Vec(n_vocab_, 1.0 / n_vocab_);
    Vec avg(n_vocab_, 0);
    for (Index k = 0; k < ensemble_count_; ++k) {
        for (Index i = 0; i < n_vocab_; ++i) {
            avg[i] += ensemble_probs_[k][i];
        }
    }
    Real inv = 1.0 / ensemble_count_;
    for (auto& v : avg) v *= inv;
    return avg;
}

Real UQ::epistemic_uncertainty() const {
    if (ensemble_count_ < 2) return 0.5;
    // Var_k[E_{y~p_θ_k}[y]]
    // For a categorical, we compute variance of the mean predictions
    Vec means(n_vocab_, 0);
    for (Index k = 0; k < ensemble_count_; ++k) {
        Real mean_k = 0;
        for (Index i = 0; i < n_vocab_; ++i) {
            mean_k += i * ensemble_probs_[k][i];
        }
        means[k] = mean_k;
    }
    return variance(means) / (n_vocab_ * n_vocab_);
}

Real UQ::aleatoric_uncertainty() const {
    if (ensemble_count_ == 0) return 0.5;
    // E_k[Var_{y~p_θ_k}[y]]
    Real total = 0;
    for (Index k = 0; k < ensemble_count_; ++k) {
        Real mean_k = 0;
        Real var_k = 0;
        for (Index i = 0; i < n_vocab_; ++i) {
            mean_k += i * ensemble_probs_[k][i];
        }
        for (Index i = 0; i < n_vocab_; ++i) {
            Real diff = (Real)i - mean_k;
            var_k += diff * diff * ensemble_probs_[k][i];
        }
        total += var_k;
    }
    return total / (ensemble_count_ * n_vocab_ * n_vocab_);
}

Real UQ::fuse_uncertainty(Real conflict) const {
    // C_i = σ(W_u[Var_epistemic; Var_aleatoric; conflict] + b_u)
    Real epi = epistemic_uncertainty();
    Real ale = aleatoric_uncertainty();
    Real val = W_u_[0] * epi + W_u_[1] * ale + W_u_[2] * conflict + b_u_;
    return sigmoid(val);
}

UQ::UncertaintyResult UQ::step(const Vec& probs, Real conflict_score,
                                Real alpha) {
    UncertaintyResult res;

    // Conformal: compute non-conformity score and prediction set
    Real max_prob = 0;
    for (Index i = 0; i < probs.size() && i < n_vocab_; ++i) {
        if (probs[i] > max_prob) { max_prob = probs[i]; }
    }

    Real score = conformal_score(max_prob);
    update_reservoir(score);
    res.conformal_set = prediction_set(probs, alpha);

    // Conformal uncertainty: size of prediction set normalized
    res.conformal_uncertainty = std::clamp(
        (Real)res.conformal_set.size() / n_vocab_, 0.0, 1.0);

    // Bayesian
    res.epistemic = epistemic_uncertainty();
    res.aleatoric = aleatoric_uncertainty();

    // Fused
    res.fused = fuse_uncertainty(conflict_score);

    return res;
}

} // namespace ai2
