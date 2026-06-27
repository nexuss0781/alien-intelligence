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

    // Scratch/work buffers (reused across steps)
    float* d_hidden = nullptr;
    int*   d_expert_idxs = nullptr;
    float* d_expert_wgts = nullptr;
    int*   d_targets = nullptr;
    float* d_logits = nullptr;
    float* d_loss = nullptr;

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

// Combined forward+backward for one batch:
//  - Copies host data to GPU
//  - Runs forward kernel (SSOG eval + output proj + softmax + cross-entropy)
//  - Runs backward kernel (gradient of W_out, b_out)
//  - Copies gradients back to host
//  - Returns total loss
float forward_backward(GPUContext* ctx,
                       const float* hidden,
                       const int*   expert_idxs,
                       const float* expert_wgts,
                       const int*   targets,
                       int n_positions,
                       Vec& grad_W_out, Vec& grad_b_out,
                       std::size_t vocab, std::size_t d_model);

}} // namespace ai2::gpu
