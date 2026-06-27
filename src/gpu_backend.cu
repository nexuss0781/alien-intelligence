#include "gpu_backend.hpp"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

namespace ai2 { namespace gpu {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// Gradient kernel: one block per (vocab_i), vocab blocks, d_model threads per block.
// Each thread handles one d_model dimension for its vocab entry,
// iterating over all positions to accumulate gradient.
__global__ void batched_gradient_kernel(
    const float* __restrict__ logits,
    const int*   __restrict__ targets,
    const float* __restrict__ hidden,
    float* __restrict__ grad_W_out,
    float* __restrict__ grad_b_out,
    int n_positions, int vocab, int d_model)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i >= vocab || j >= d_model) return;

    float accum_w = 0.0f;
    float accum_b = 0.0f;

    for (int pos = 0; pos < n_positions; ++pos) {
        float p_i = logits[pos * vocab + i];
        int t = targets[pos];
        float dL_dz = p_i - (i == t ? 1.0f : 0.0f);
        accum_w += dL_dz * hidden[pos * d_model + j];
        if (j == 0) accum_b += dL_dz;
    }

    grad_W_out[i * d_model + j] = accum_w;
    if (j == 0) grad_b_out[i] = accum_b;
}

GPUContext* init(const Mat& W_out, const Vec& b_out) {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::cerr << "  No CUDA devices found." << std::endl;
        return nullptr;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  GPU: " << prop.name
              << "  " << (prop.totalGlobalMem / (1024*1024)) << "MB"
              << "  Cores: " << prop.multiProcessorCount
              << std::endl;

    auto* ctx = new GPUContext;
    ctx->vocab = W_out.size();
    ctx->d_model = W_out.empty() ? 0 : W_out[0].size();

    std::size_t n_vocab = ctx->vocab;
    std::size_t d_model = ctx->d_model;

    CUDA_CHECK(cudaMalloc(&ctx->d_W_out, n_vocab * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_b_out, n_vocab * sizeof(float)));

    std::vector<float> h_W(n_vocab * d_model);
    std::vector<float> h_b(n_vocab);
    for (std::size_t i = 0; i < n_vocab; ++i) {
        h_b[i] = static_cast<float>(b_out[i]);
        for (std::size_t j = 0; j < d_model; ++j) {
            h_W[i * d_model + j] = static_cast<float>(W_out[i][j]);
        }
    }
    CUDA_CHECK(cudaMemcpy(ctx->d_W_out, h_W.data(), n_vocab * d_model * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_b_out, h_b.data(), n_vocab * sizeof(float),
                           cudaMemcpyHostToDevice));

    ctx->scratch_sz = n_vocab * 16384;
    CUDA_CHECK(cudaMalloc(&ctx->d_scratch, ctx->scratch_sz * sizeof(float)));

    return ctx;
}

void destroy(GPUContext* ctx) {
    if (!ctx) return;
    if (ctx->d_W_out) cudaFree(ctx->d_W_out);
    if (ctx->d_b_out) cudaFree(ctx->d_b_out);
    if (ctx->d_scratch) cudaFree(ctx->d_scratch);
    delete ctx;
}

void sync_weights(GPUContext* ctx, const Vec& param_W_out, const Vec& param_b_out,
                  std::size_t vocab, std::size_t d_model) {
    if (!ctx) return;
    std::vector<float> h_W(vocab * d_model);
    for (std::size_t i = 0; i < vocab; ++i)
        for (std::size_t j = 0; j < d_model; ++j)
            h_W[i * d_model + j] = static_cast<float>(param_W_out[i * d_model + j]);

    std::vector<float> h_b(vocab);
    for (std::size_t i = 0; i < vocab; ++i)
        h_b[i] = static_cast<float>(param_b_out[i]);

    CUDA_CHECK(cudaMemcpy(ctx->d_W_out, h_W.data(), vocab * d_model * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_b_out, h_b.data(), vocab * sizeof(float),
                           cudaMemcpyHostToDevice));
}

void batched_gradient(GPUContext* ctx,
                      const std::vector<Vec>& hidden_host,
                      const std::vector<std::vector<Vec>>& logits_host,
                      const Mat& targets,
                      Vec& grad_W_out, Vec& grad_b_out,
                      std::size_t n_positions,
                      std::size_t vocab, std::size_t d_model,
                      std::size_t batch_size, std::size_t seq_len)
{
    if (!ctx) return;
    int v = static_cast<int>(vocab);
    int d = static_cast<int>(d_model);

    // Count valid (non-padding) positions
    std::size_t valid_count = 0;
    for (std::size_t b = 0; b < batch_size; ++b)
        for (std::size_t t = 0; t < seq_len; ++t)
            if (b < targets.size() && t < targets[b].size() && targets[b][t] != 0)
                valid_count++;

    if (valid_count == 0) return;

    // Build flat arrays for valid positions
    std::vector<float> h_probs(valid_count * v);
    std::vector<int> h_targets(valid_count);
    std::vector<float> h_hidden(valid_count * d);

    std::size_t idx = 0;
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t t = 0; t < seq_len; ++t) {
            if (b >= targets.size() || t >= targets[b].size()) continue;
            if (targets[b][t] == 0) continue;
            if (b >= logits_host.size() || t >= logits_host[b].size()) continue;

            Vec p = softmax(logits_host[b][t]);
            for (std::size_t i = 0; i < vocab; ++i)
                h_probs[idx * v + i] = static_cast<float>(p[i]);

            h_targets[idx] = static_cast<int>(targets[b][t]);
            std::size_t hidx = b * seq_len + t;
            if (hidx < hidden_host.size()) {
                for (std::size_t j = 0; j < d_model; ++j)
                    h_hidden[idx * d + j] = static_cast<float>(hidden_host[hidx][j]);
            }
            idx++;
        }
    }

    int n = static_cast<int>(valid_count);

    float* d_probs = nullptr;
    int* d_targets = nullptr;
    float* d_hidden = nullptr;
    float* d_grad_W = nullptr;
    float* d_grad_b = nullptr;

    CUDA_CHECK(cudaMalloc(&d_probs, n * v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hidden, n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W, v * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, v * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_probs, h_probs.data(), n * v * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), n * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, h_hidden.data(), n * d * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grad_W, 0, v * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_b, 0, v * sizeof(float)));

    int threads = std::min(d, 256);
    batched_gradient_kernel<<<v, threads, 0>>>(
        d_probs, d_targets, d_hidden, d_grad_W, d_grad_b, n, v, d);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_grad_W(v * d);
    std::vector<float> h_grad_b(v);
    CUDA_CHECK(cudaMemcpy(h_grad_W.data(), d_grad_W, v * d * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_b.data(), d_grad_b, v * sizeof(float),
                           cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < vocab; ++i) {
        grad_b_out[i] += static_cast<Real>(h_grad_b[i]);
        for (std::size_t j = 0; j < d_model; ++j) {
            grad_W_out[i * d_model + j] += static_cast<Real>(h_grad_W[i * d_model + j]);
        }
    }

    CUDA_CHECK(cudaFree(d_probs));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_grad_W));
    CUDA_CHECK(cudaFree(d_grad_b));
}

}} // namespace ai2::gpu
