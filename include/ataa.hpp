#pragma once
#include "types.hpp"
#include <vector>
#include <deque>

namespace ai2 {

// Agility & Task-Agnostic Adaptation (ATAA) — Component 5
// Hypernetwork-mediated O(1) task adaptation with continual learning.
// Complexity: O(n) adaptation (dominated by example encoding), O(1) per step.
class ATAA {
public:
    // d_model: base model dimension
    // d_task: task embedding dimension
    // r_lora: LoRA rank (r = O(1), typically 8-16)
    // n_examples: number of examples for task encoding (k)
    // sketch_rank: rank for sketched SVD in OGD
    ATAA(Index d_model, Index d_task = 32, Index r_lora = 8,
         Index n_examples = 8, Index sketch_rank = 16);

    // --------------- Task Embedding ---------------

    // Encode task from k example prompts using tiny LSSC
    // τ = Pool(LSSC_tiny(examples))
    // O(k · d_model) = O(n) with k = O(1) -> O(1) per example
    Vec encode_task(const std::vector<Vec>& examples);

    // Set current task
    void set_current_task(const Vec& task_embedding);

    // --------------- Hypernetwork ---------------

    // Hypernetwork H: T → Θ generates LoRA adapters
    // H(τ; φ) = [A(τ); B(τ)] where A ∈ R^{r × d}, B ∈ R^{d × r}
    // O(1) hypernetwork evaluation

    // Generate LoRA adapter for a given layer
    // Returns (B, A) such that W_task = W_0 + BA
    std::pair<Mat, Mat> generate_adapter(const Vec& task_emb) const;

    // Apply LoRA adaptation to a weight matrix
    // W_task = W_0 + BA — O(d^2) in general but we apply on-the-fly
    Vec apply_lora(const Vec& x, const Mat& W0,
                   const Mat& B, const Mat& A) const;

    // --------------- Continual Learning (OGD) ---------------

    // Update sketched gradient subspace with new gradient
    // O(rank · d) = O(1)
    void update_gradient_subspace(const Mat& gradient);

    // Project gradient onto null space of previous tasks
    // ∇_proj = P_t ∇L_t where P_t is projection onto null space
    // O(sketch_rank · d) = O(1)
    Mat project_gradient(const Mat& gradient) const;

    // --------------- Full Adaptation ---------------

    // Adapt to a new task given examples
    // Returns (B, A) LoRA adapters
    std::pair<Mat, Mat> adapt(const std::vector<Vec>& examples);

    // Accessors
    Index d_model() const { return d_model_; }
    Index r_lora() const { return r_lora_; }
    const Vec& current_task() const { return current_task_; }

private:
    Index d_model_;
    Index d_task_;
    Index r_lora_;
    Index n_examples_;
    Index sketch_rank_;

    // Hypernetwork parameters
    // H(τ) = W_h2 · σ(W_h1 · τ + b_h1) + b_h2
    // Outputs flat vector of size 2 * d_model * r_lora
    Mat W_h1_;  // d_task -> 4*d_task
    Vec b_h1_;
    Mat W_h2_;  // 4*d_task -> 2*d_model*r_lora
    Vec b_h2_;

    // Task embedding
    Vec current_task_;
    bool has_task_ = false;

    // OGD subspace (sketched SVD)
    // Maintains orthonormal basis U ∈ R^{d_model × sketch_rank}
    Mat ogd_basis_;
    Index ogd_count_ = 0;

    // Tiny LSSC for task encoding
    // Simplified: just a 2-layer MLP with pooling
    Mat enc_W1_, enc_W2_;
    Vec enc_b1_, enc_b2_;
};

} // namespace ai2
