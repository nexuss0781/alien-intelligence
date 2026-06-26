#pragma once
#include "types.hpp"
#include <vector>
#include <unordered_map>

namespace ai2 {

// Sparse Synthesis & Output Generation (SSOG) — Component 6
// LSH-routed sparse MoE with calibrated output distribution.
// Complexity: O(1) per token, O(n) total.
class SSOG {
public:
    // d_model: input feature dimension
    // n_vocab: vocabulary size
    // n_experts: total number of experts
    // k_experts: number of active experts per token (k = O(1))
    // n_lsh_tables: number of LSH tables for routing
    SSOG(Index d_model, Index n_vocab, Index n_experts = 32,
         Index k_experts = 4, Index n_lsh_tables = 4);

    // --------------- LSH-Based Expert Routing ---------------

    // Route hidden state to k experts via LSH
    // gate(h) = {i : h ∈ bucket_j(h)}
    // O(n_lsh_tables) = O(1)
    std::vector<Index> route(const Vec& h) const;

    // Compute gating weights for selected experts
    // g_i(h) = softmax over selected experts' scores
    // O(k) = O(1)
    Vec gating_weights(const Vec& h, const std::vector<Index>& experts) const;

    // --------------- Expert Evaluation ---------------

    // Evaluate an expert on hidden state h
    // E_i(h) = W_i · h  (linear expert, O(d_model) = O(1))
    Vec expert_forward(Index expert_id, const Vec& h) const;

    // Sparse mixture output:
    // output = Σ_{i ∈ gate(h)} g_i(h) · E_i(h)
    // O(k · d_model) = O(1)
    Vec sparse_mixture(const Vec& h) const;

    // --------------- Output Distribution ---------------

    // Base probability from sparse mixture + output projection
    // p(y) = softmax(W_out · mixture)
    // O(|V|) — this is the vocab projection, unavoidable
    Vec base_distribution(const Vec& mixture) const;

    // Calibrated output distribution incorporating uncertainty:
    // p̃(y) = [p(y)·(1-C_uncertain) / Z] + [1_C(y)·C_uncertain / |C|]
    // where C_uncertain is the conformal uncertainty score
    // O(|V|) worst case, but can be sparse
    Vec calibrated_distribution(const Vec& base_probs,
                                 Real conformal_uncertainty,
                                 const std::vector<Index>& conformal_set) const;

    // Full forward pass
    // Returns (mixture, base_probs, calibrated_probs)
    struct Output {
        Vec mixture;
        Vec base_probs;
        Vec calibrated_probs;
    };
    Output forward(const Vec& h, Real conformal_uncertainty = 0,
                   const std::vector<Index>& conformal_set = {}) const;

    // Accessors
    Index n_experts() const { return n_experts_; }
    Index k_experts() const { return k_experts_; }
    Index n_vocab() const { return n_vocab_; }

private:
    Index d_model_;
    Index n_vocab_;
    Index n_experts_;
    Index k_experts_;
    Index n_lsh_tables_;

    // LSH functions for routing
    std::vector<LSHFunction> lsh_funcs_;

    // Expert weights: W_i ∈ R^{d_model × d_model} for each expert i
    std::vector<Mat> expert_weights_;

    // Expert biases: b_i ∈ R^{d_model} for each expert i
    std::vector<Vec> expert_biases_;

    // Output projection: W_out ∈ R^{n_vocab × d_model}
    Mat W_out_;
    Vec b_out_;

    // Gating projection (for softmax scores): W_gate ∈ R^{d_model × d_model}
    Mat W_gate_;
};

} // namespace ai2
