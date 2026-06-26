#pragma once
#include "types.hpp"
#include <vector>
#include <deque>

namespace ai2 {

// Uncertainty Quantification (UQ) — Component 4
// Dual-track uncertainty via conformal prediction + Bayesian ensemble.
// Complexity: O(1) per token, O(n) total.
class UQ {
public:
    // n_vocab: vocabulary size for prediction sets
    // reservoir_size: size of score reservoir for conformal quantile
    // n_ensemble: K = number of ensemble members
    // d_model: feature dimension
    UQ(Index n_vocab, Index reservoir_size = 1000,
       Index n_ensemble = 5, Index d_model = 256);

    // --------------- Track A: Conformal Prediction ---------------

    // Compute non-conformity score: s_i = 1 - p(true_i | context)
    // O(1)
    Real conformal_score(Real prob_true) const;

    // Update reservoir with new score
    // O(1) amortized
    void update_reservoir(Real score);

    // Get current quantile threshold: q̂ = quantile({s_j}, 1-α)
    // O(m) where m = reservoir_size = O(1)
    Real quantile_threshold(Real alpha = 0.1) const;

    // Build prediction set: C_i = {v : 1 - p(v) ≤ q̂}
    // O(|V|) but we can use sparse version with O(k) where k = set size
    std::vector<Index> prediction_set(const Vec& probs, Real alpha = 0.1) const;

    // --------------- Track B: Bayesian Ensemble ---------------

    // Register ensemble member prediction
    // Each member provides a probability distribution over vocab
    // O(1) per member
    void add_ensemble_prediction(const Vec& probs, Index member_id);
    void clear_ensemble();

    // Compute ensemble predictive distribution: p_ens = (1/K) Σ p_θ_k
    // O(|V|)
    Vec ensemble_distribution() const;

    // Epistemic uncertainty: Var_k[E_{y~p_θ_k}[y]]
    // O(K·|V|) = O(1) since K = O(1)
    Real epistemic_uncertainty() const;

    // Aleatoric uncertainty: E_k[Var_{y~p_θ_k}[y]]
    // O(K·|V|) = O(1)
    Real aleatoric_uncertainty() const;

    // --------------- Fusion ---------------

    // Fusion field: C_i = σ(W_u [Var_epistemic; Var_aleatoric; conflict_i] + b_u)
    // O(1)
    Real fuse_uncertainty(Real conflict = 0) const;

    // Full forward for one token position
    struct UncertaintyResult {
        Real conformal_uncertainty;  // C_uncertain
        Real epistemic;
        Real aleatoric;
        Real fused;
        std::vector<Index> conformal_set;
    };

    // Process one step: O(1)
    UncertaintyResult step(const Vec& probs, Real conflict_score = 0,
                           Real alpha = 0.1);

    // Accessors
    Index n_vocab() const { return n_vocab_; }
    Index n_ensemble() const { return n_ensemble_; }

private:
    Index n_vocab_;
    Index reservoir_size_;
    Index n_ensemble_;
    Index d_model_;

    // Conformal reservoir (fixed-size deque)
    std::deque<Real> reservoir_;
    Real reservoir_sum_;  // for efficient mean if needed

    // Ensemble storage: K probability distributions
    std::vector<Vec> ensemble_probs_;
    Index ensemble_count_;

    // Fusion parameters
    Vec W_u_;  // W_u ∈ R^{3} (epistemic, aleatoric, conflict -> scalar)
    Real b_u_;
};

} // namespace ai2
