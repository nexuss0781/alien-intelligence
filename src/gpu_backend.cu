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

// ---------------------------------------------------------------------------
// Compute mixture: weighted sum of expert outputs (cache-building kernel)
// One block per position, d_model threads per block.
// ---------------------------------------------------------------------------
__global__ void compute_mixture_kernel(
    const float* __restrict__ hidden,
    const int*   __restrict__ expert_idxs,
    const float* __restrict__ expert_wgts,
    float* __restrict__ mixture_out,
    const float* __restrict__ W_expert,
    const float* __restrict__ b_expert,
    int n_positions, int d_model, int n_experts, int k_experts)
{
    int pos = blockIdx.x;
    if (pos >= n_positions) return;

    extern __shared__ float s_hidden[];

    int j = threadIdx.x;
    if (j >= d_model) return;

    s_hidden[j] = hidden[pos * d_model + j];
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < k_experts; ++k) {
        int e = expert_idxs[pos * k_experts + k];
        float w = expert_wgts[pos * k_experts + k];
        if (e < 0 || e >= n_experts || w == 0.0f) continue;

        float expert_out = 0.0f;
        for (int l = 0; l < d_model; ++l)
            expert_out += W_expert[e * d_model * d_model + j * d_model + l] * s_hidden[l];
        expert_out += b_expert[e * d_model + j];
        acc += w * expert_out;
    }

    mixture_out[pos * d_model + j] = acc;
}

// ---------------------------------------------------------------------------
// Output projection forward: W_out @ mixture + b_out → logits
// One block per position, max(vocab, d_model) threads per block.
// ---------------------------------------------------------------------------
__global__ void output_proj_kernel(
    const float* __restrict__ mixture,
    const float* __restrict__ W_out,
    const float* __restrict__ b_out,
    float* __restrict__ logits_out,
    float* __restrict__ loss_out,
    const int*   __restrict__ targets,
    int n_positions, int vocab, int d_model)
{
    int pos = blockIdx.x;
    if (pos >= n_positions) return;

    extern __shared__ float s_mix[];
    int j = threadIdx.x;

    if (j < d_model) s_mix[j] = mixture[pos * d_model + j];
    __syncthreads();

    if (j < vocab) {
        float logit = b_out[j];
        for (int l = 0; l < d_model; ++l)
            logit += W_out[j * d_model + l] * s_mix[l];
        logits_out[pos * vocab + j] = logit;
    }
    __syncthreads();

    // Thread 0 computes loss
    if (loss_out && j == 0) {
        int base = pos * vocab;
        float mx = logits_out[base];
        for (int i = 1; i < vocab; ++i)
            if (logits_out[base + i] > mx) mx = logits_out[base + i];
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab; ++i)
            sum_exp += expf(logits_out[base + i] - mx);
        int t = targets[pos];
        float p_target = expf((t >= 0 && t < vocab) ? logits_out[base + t] - mx : 0.0f);
        loss_out[pos] = -logf(fmaxf(p_target / (sum_exp + 1e-10f), 1e-10f));
    }
}

// ---------------------------------------------------------------------------
// Gradient kernel (uses raw hidden, not mixture)
// ---------------------------------------------------------------------------
__global__ void gradient_kernel(
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
        int base = pos * vocab;
        float mx = logits[base];
        for (int v = 1; v < vocab; ++v)
            if (logits[base + v] > mx) mx = logits[base + v];
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; ++v)
            sum_exp += expf(logits[base + v] - mx);
        float p_i = expf(logits[base + i] - mx) / (sum_exp + 1e-10f);

        int t = targets[pos];
        float dL_dz = p_i - (i == t ? 1.0f : 0.0f);
        accum_w += dL_dz * hidden[pos * d_model + j];
        if (j == 0) accum_b += dL_dz;
    }

    grad_W_out[i * d_model + j] = accum_w;
    if (j == 0) grad_b_out[i] = accum_b;
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------
GPUContext* init(const Mat& W_out, const Vec& b_out,
                 const std::vector<Mat>& expert_weights,
                 const std::vector<Vec>& expert_biases,
                 std::size_t max_positions,
                 std::size_t n_experts, std::size_t k_experts)
{
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) return nullptr;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  GPU: " << prop.name
              << "  " << (prop.totalGlobalMem / (1024*1024)) << "MB"
              << "  SMs: " << prop.multiProcessorCount
              << std::endl;

    auto* ctx = new GPUContext;
    ctx->vocab = W_out.size();
    ctx->d_model = W_out.empty() ? 0 : W_out[0].size();
    ctx->n_experts = n_experts;
    ctx->k_experts = k_experts;
    ctx->max_positions = max_positions;

    std::size_t v = ctx->vocab;
    std::size_t d = ctx->d_model;
    std::size_t ne = ctx->n_experts;
    std::size_t k = ctx->k_experts;
    std::size_t np = ctx->max_positions;

    // Model params
    CUDA_CHECK(cudaMalloc(&ctx->d_W_out, v * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_b_out, v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_W_expert, ne * d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_b_expert, ne * d * sizeof(float)));

    {
        std::vector<float> h_W(v * d);
        std::vector<float> h_b(v);
        for (std::size_t i = 0; i < v; ++i) {
            h_b[i] = static_cast<float>(b_out[i]);
            for (std::size_t j = 0; j < d; ++j)
                h_W[i * d + j] = static_cast<float>(W_out[i][j]);
        }
        CUDA_CHECK(cudaMemcpy(ctx->d_W_out, h_W.data(), v * d * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->d_b_out, h_b.data(), v * sizeof(float),
                               cudaMemcpyHostToDevice));
    }
    {
        std::vector<float> h_W_expert(ne * d * d);
        std::vector<float> h_b_expert(ne * d);
        for (std::size_t e = 0; e < ne; ++e) {
            for (std::size_t j = 0; j < d; ++j) {
                h_b_expert[e * d + j] = static_cast<float>(expert_biases[e][j]);
                for (std::size_t l = 0; l < d; ++l)
                    h_W_expert[e * d * d + j * d + l] = static_cast<float>(expert_weights[e][j][l]);
            }
        }
        CUDA_CHECK(cudaMemcpy(ctx->d_W_expert, h_W_expert.data(), ne * d * d * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->d_b_expert, h_b_expert.data(), ne * d * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    // Scratch buffers
    CUDA_CHECK(cudaMalloc(&ctx->d_hidden, np * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_mixture, np * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_expert_idxs, np * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_expert_wgts, np * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_targets, np * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_logits, np * v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx->d_loss, np * sizeof(float)));

    return ctx;
}

void destroy(GPUContext* ctx) {
    if (!ctx) return;
    if (ctx->d_W_out) cudaFree(ctx->d_W_out);
    if (ctx->d_b_out) cudaFree(ctx->d_b_out);
    if (ctx->d_W_expert) cudaFree(ctx->d_W_expert);
    if (ctx->d_b_expert) cudaFree(ctx->d_b_expert);
    if (ctx->d_hidden) cudaFree(ctx->d_hidden);
    if (ctx->d_mixture) cudaFree(ctx->d_mixture);
    if (ctx->d_expert_idxs) cudaFree(ctx->d_expert_idxs);
    if (ctx->d_expert_wgts) cudaFree(ctx->d_expert_wgts);
    if (ctx->d_targets) cudaFree(ctx->d_targets);
    if (ctx->d_logits) cudaFree(ctx->d_logits);
    if (ctx->d_loss) cudaFree(ctx->d_loss);
    delete ctx;
}

void sync_weights(GPUContext* ctx,
                  const float* h_W_out, const float* h_b_out,
                  const float* h_W_expert, const float* h_b_expert,
                  std::size_t vocab, std::size_t d_model,
                  std::size_t n_experts)
{
    if (!ctx) return;
    CUDA_CHECK(cudaMemcpy(ctx->d_W_out, h_W_out, vocab * d_model * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_b_out, h_b_out, vocab * sizeof(float),
                           cudaMemcpyHostToDevice));
    if (h_W_expert && h_b_expert && n_experts > 0) {
        CUDA_CHECK(cudaMemcpy(ctx->d_W_expert, h_W_expert,
                               n_experts * d_model * d_model * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ctx->d_b_expert, h_b_expert,
                               n_experts * d_model * sizeof(float),
                               cudaMemcpyHostToDevice));
    }
}

void build_mixture(GPUContext* ctx,
                   const float* d_hidden,
                   const int* d_expert_idxs,
                   const float* d_expert_wgts,
                   float* d_mixture,
                   int n_positions)
{
    if (!ctx || n_positions <= 0) return;

    int d = static_cast<int>(ctx->d_model);
    int ne = static_cast<int>(ctx->n_experts);
    int k = static_cast<int>(ctx->k_experts);

    int smem = d * sizeof(float);
    compute_mixture_kernel<<<n_positions, d, smem>>>(
        d_hidden, d_expert_idxs, d_expert_wgts, d_mixture,
        ctx->d_W_expert, ctx->d_b_expert,
        n_positions, d, ne, k);
    CUDA_CHECK(cudaGetLastError());
}

void compute_mixture_batch(GPUContext* ctx,
                           const float* host_hidden,
                           const int* host_expert_idxs,
                           const float* host_expert_wgts,
                           float* host_mixture,
                           int n_positions)
{
    if (!ctx || n_positions <= 0) return;
    int d = static_cast<int>(ctx->d_model);
    int k = static_cast<int>(ctx->k_experts);

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(ctx->d_hidden, host_hidden, n_positions * d * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_expert_idxs, host_expert_idxs,
                           n_positions * k * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_expert_wgts, host_expert_wgts,
                           n_positions * k * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Launch mixture kernel
    int smem = d * sizeof(float);
    compute_mixture_kernel<<<n_positions, d, smem>>>(
        ctx->d_hidden, ctx->d_expert_idxs, ctx->d_expert_wgts, ctx->d_mixture,
        ctx->d_W_expert, ctx->d_b_expert,
        n_positions, d, static_cast<int>(ctx->n_experts), k);
    CUDA_CHECK(cudaGetLastError());

    // Copy mixture back
    CUDA_CHECK(cudaMemcpy(host_mixture, ctx->d_mixture,
                           n_positions * d * sizeof(float),
                           cudaMemcpyDeviceToHost));
}

float forward_backward(GPUContext* ctx,
                       const float* mixture,
                       const float* hidden,
                       const int* targets,
                       int n_positions,
                       Vec& grad_W_out, Vec& grad_b_out,
                       std::size_t vocab, std::size_t d_model)
{
    if (!ctx || n_positions <= 0) return 0.0f;

    int v = static_cast<int>(vocab);
    int d = static_cast<int>(d_model);
    int n = n_positions;

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(ctx->d_mixture, mixture, n * d * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_hidden, hidden, n * d * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_targets, targets, n * sizeof(int),
                           cudaMemcpyHostToDevice));

    // Forward: output projection + softmax + loss
    int smem = d * sizeof(float);
    int threads = d > v ? d : v;
    output_proj_kernel<<<n, threads, smem>>>(
        ctx->d_mixture, ctx->d_W_out, ctx->d_b_out,
        ctx->d_logits, ctx->d_loss, ctx->d_targets,
        n, v, d);
    CUDA_CHECK(cudaGetLastError());

    // Copy loss back
    std::vector<float> h_loss(n);
    CUDA_CHECK(cudaMemcpy(h_loss.data(), ctx->d_loss, n * sizeof(float),
                           cudaMemcpyDeviceToHost));
    float total_loss = 0.0f;
    for (float l : h_loss) total_loss += l;

    // Backward: gradient of W_out, b_out
    float* d_grad_W = nullptr;
    float* d_grad_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grad_W, v * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b, v * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_W, 0, v * d * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_b, 0, v * sizeof(float)));

    int gthreads = d < 256 ? d : 256;
    gradient_kernel<<<v, gthreads, 0>>>(
        ctx->d_logits, ctx->d_targets, ctx->d_hidden,
        d_grad_W, d_grad_b, n, v, d);
    CUDA_CHECK(cudaGetLastError());

    // Copy gradients back
    std::vector<float> h_grad_W(v * d);
    std::vector<float> h_grad_b(v);
    CUDA_CHECK(cudaMemcpy(h_grad_W.data(), d_grad_W, v * d * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_b.data(), d_grad_b, v * sizeof(float),
                           cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < vocab; ++i) {
        grad_b_out[i] += static_cast<Real>(h_grad_b[i]);
        for (std::size_t j = 0; j < d_model; ++j)
            grad_W_out[i * d_model + j] += static_cast<Real>(h_grad_W[i * d_model + j]);
    }

    CUDA_CHECK(cudaFree(d_grad_W));
    CUDA_CHECK(cudaFree(d_grad_b));

    return total_loss;
}

}} // namespace ai2::gpu
