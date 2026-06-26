#pragma once
#include "types.hpp"
#include <complex>
#include <vector>

namespace ai2 {

// Linear State-Space Core (LSSC) — Component 2
// Hybrid SSM / Random-Feature Attention with learned gating
// Complexity: O(n) for full sequence, O(1) per step
class LSSC {
public:
    // d_model: input/output dimension
    // d_state: SSM latent state dimension (d_s, typically 64-128)
    // n_rf: number of random features for RFA
    LSSC(Index d_model, Index d_state = 64, Index n_rf = 256);

    // --------------- Sub-Component 2A: Diagonal SSM (S4D) ---------------
    struct S4DState {
        CVec x;  // complex state vector, size d_state
    };

    // Initialize SSM state (all zeros)
    S4DState ssm_init() const;

    // Single SSM step: x_{k+1} = Λ x_k + B u_k
    // Returns y_k = Re(C x_k) + D u_k
    // O(d_state) = O(1)
    Vec ssm_step(S4DState& state, const Vec& u_k);

    // Parallel scan over full sequence: O(n)
    Mat ssm_scan(const Mat& U);

    // --------------- Sub-Component 2B: Random Feature Attention (RFA) ---------------
    // φ(x, w) = exp(w^T x - ||x||^2/2)
    // Attention(Q,K,V) = φ(Q)(φ(K)^T V) / (φ(Q) φ(K)^T)

    // Compute random features for a given input vector: O(1)
    Vec rfa_feature(const Vec& x) const;

    // Causal RFA over sequence: O(n) via prefix sums
    Mat rfa_causal(const Mat& Q, const Mat& K, const Mat& V);

    // Non-causal RFA: O(n)
    Mat rfa_bidirectional(const Mat& Q, const Mat& K, const Mat& V);

    // --------------- Sub-Component 2C: Gated Linear Recurrence (GLR) ---------------
    // α_k = σ(W_g z_k + b_g)
    // h_k = α_k ⊙ y_k^{SSM} + (1 - α_k) ⊙ y_k^{RFA}

    // Single step gated fusion: O(d_model) = O(1)
    Vec gated_step(const Vec& z_k, const Vec& y_ssm, const Vec& y_rfa);

    // Full forward: alternate SSM, RFA, and gated fusion over sequence
    // Input: sequence of embeddings Z ∈ R^{n × d_model}
    // Output: encoded sequence H ∈ R^{n × d_model}
    // O(n) total
    Mat forward(const Mat& Z);

    // Accessors
    Index d_model() const { return d_model_; }
    Index d_state() const { return d_state_; }
    Index n_rf() const { return n_rf_; }

private:
    Index d_model_;
    Index d_state_;
    Index n_rf_;

    // S4D parameters
    CVec lambda_;        // Λ diagonal: λ_j = exp(ω_j + i·φ_j)
    Vec omega_;          // real exponents
    Vec phi_;            // imaginary phases
    Mat B_;              // B ∈ R^{d_state × d_model}
    Mat C_;              // C ∈ R^{d_model × d_state}
    Vec D_;              // D ∈ R^{d_model} (skip connection)

    // RFA parameters
    Mat W_rf_;           // random projection matrix ∈ R^{n_rf × d_model}
                         // each row w ~ N(0, I)

    // Gating parameters
    Mat W_g_;            // W_g ∈ R^{d_model × d_model}
    Vec b_g_;            // b_g ∈ R^{d_model}
};

} // namespace ai2
