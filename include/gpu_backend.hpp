#pragma once
#include "types.hpp"
#include <cstddef>
#include <vector>

namespace ai2 { namespace gpu {

struct GPUContext {
    // Persistent model params
    float* d_W_out = nullptr;
    float* d_b_out = nullptr;
    float* d_W_expert = nullptr;
    float* d_b_expert = nullptr;

    // Scratch/work buffers
    float* d_hidden = nullptr;
    int*   d_expert_idxs = nullptr;
    float* d_expert_wgts = nullptr;
    int*   d_targets = nullptr;
    float* d_logits = nullptr;
    float* d_loss = nullptr;
    float* d_mixture = nullptr;

    std::size_t vocab = 0;
    std::size_t d_model = 0;
    std::size_t n_experts = 0;
    std::size_t k_experts = 0;
    std::size_t max_positions = 0;
};

GPUContext* init(const Mat& W_out, const Vec& b_out,
                 const std::vector<Mat>& expert_weights,
                 const std::vector<Vec>& expert_biases,
                 std::size_t max_positions,
                 std::size_t n_experts, std::size_t k_experts);

void destroy(GPUContext* ctx);

void sync_weights(GPUContext* ctx,
                  const float* h_W_out, const float* h_b_out,
                  const float* h_W_expert, const float* h_b_expert,
                  std::size_t vocab, std::size_t d_model,
                  std::size_t n_experts);

// Build mixture on GPU: for each position, compute weighted expert sum
// All data in device memory
void build_mixture(GPUContext* ctx,
                   const float* d_hidden,
                   const int* d_expert_idxs,
                   const float* d_expert_wgts,
                   float* d_mixture,
                   int n_positions);

// Full host→device→kernel→device→host pipeline for one batch of mixture
void compute_mixture_batch(GPUContext* ctx,
                           const float* host_hidden,
                           const int* host_expert_idxs,
                           const float* host_expert_wgts,
                           float* host_mixture,
                           int n_positions);

// Combined forward+backward using pre-computed mixture (no expert eval in forward):
//  - Forward: W_out @ mixture + b_out → softmax → cross-entropy
//  - Backward: gradient of W_out, b_out using raw hidden states
//  - Returns total loss
float forward_backward(GPUContext* ctx,
                       const float* mixture,
                       const float* hidden,
                       const int* targets,
                       int n_positions,
                       Vec& grad_W_out, Vec& grad_b_out,
                       std::size_t vocab, std::size_t d_model);

}} // namespace ai2::gpu
