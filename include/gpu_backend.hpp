#pragma once
#include "types.hpp"
#include <cstddef>
#include <vector>

namespace ai2 { namespace gpu {

struct GPUContext {
    float* d_W_out = nullptr;      // [vocab x d_model]
    float* d_b_out = nullptr;      // [vocab]
    float* d_scratch = nullptr;
    std::size_t scratch_sz = 0;
    std::size_t vocab = 0;
    std::size_t d_model = 0;
};

GPUContext* init(const Mat& W_out, const Vec& b_out);
void destroy(GPUContext* ctx);

void sync_weights(GPUContext* ctx, const Vec& param_W_out, const Vec& param_b_out,
                  std::size_t vocab, std::size_t d_model);

// Batched gradient: computes dL/dW_out and dL/db_out for all positions on GPU
void batched_gradient(GPUContext* ctx,
                      const std::vector<Vec>& hidden_host,
                      const std::vector<std::vector<Vec>>& logits_host,
                      const Mat& targets,
                      Vec& grad_W_out, Vec& grad_b_out,
                      std::size_t n_positions,
                      std::size_t vocab, std::size_t d_model,
                      std::size_t batch_size, std::size_t seq_len);

}} // namespace ai2::gpu
