#include "ataa.hpp"
#include <random>
#include <cmath>
#include <numeric>

namespace ai2 {

ATAA::ATAA(Index d_model, Index d_task, Index r_lora,
           Index n_examples, Index sketch_rank)
    : d_model_(d_model), d_task_(d_task), r_lora_(r_lora),
      n_examples_(n_examples), sketch_rank_(sketch_rank)
{
    std::mt19937_64 rng(4242);

    // Hypernetwork: H(τ) = W_h2 · σ(W_h1 · τ + b_h1) + b_h2
    {
        Real scale1 = std::sqrt(2.0 / (d_task_ + 4 * d_task_));
        W_h1_.resize(4 * d_task_, Vec(d_task_));
        for (auto& row : W_h1_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale1)(rng);
        b_h1_.resize(4 * d_task_, 0);

        Index out_dim = 2 * d_model_ * r_lora_;
        Real scale2 = std::sqrt(2.0 / (4 * d_task_ + out_dim));
        W_h2_.resize(out_dim, Vec(4 * d_task_));
        for (auto& row : W_h2_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale2)(rng);
        b_h2_.resize(out_dim, 0);
    }

    // Task encoding network: tiny MLP
    {
        Real scale1 = std::sqrt(2.0 / (d_model_ + d_task_));
        enc_W1_.resize(d_task_, Vec(d_model_));
        for (auto& row : enc_W1_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale1)(rng);
        enc_b1_.resize(d_task_, 0);

        Real scale2 = std::sqrt(2.0 / (d_task_ + d_task_));
        enc_W2_.resize(d_task_, Vec(d_task_));
        for (auto& row : enc_W2_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale2)(rng);
        enc_b2_.resize(d_task_, 0);
    }

    // OGD basis (start empty)
    ogd_basis_.resize(d_model_, Vec(sketch_rank_, 0));
}

Vec ATAA::encode_task(const std::vector<Vec>& examples) {
    // τ = Pool(MLP_tiny(examples))
    // Simple: average of encoded examples
    Vec task_emb(d_task_, 0);
    Index count = std::min(n_examples_, (Index)examples.size());

    for (Index i = 0; i < count; ++i) {
        Vec h = mat_vec(enc_W1_, examples[i]);
        h = elem_add(h, enc_b1_);
        for (auto& v : h) v = std::tanh(v);
        h = mat_vec(enc_W2_, h);
        h = elem_add(h, enc_b2_);
        for (auto& v : h) v = std::tanh(v);
        task_emb = elem_add(task_emb, h);
    }

    if (count > 0) {
        Real inv = 1.0 / count;
        for (auto& v : task_emb) v *= inv;
    }

    return task_emb;
}

void ATAA::set_current_task(const Vec& task_embedding) {
    current_task_ = task_embedding;
    has_task_ = true;
}

std::pair<Mat, Mat> ATAA::generate_adapter(const Vec& task_emb) const {
    // Hypernetwork: H(τ) = W_h2 · ReLU(W_h1 · τ + b_h1) + b_h2
    Vec h = mat_vec(W_h1_, task_emb);
    h = elem_add(h, b_h1_);
    for (auto& v : h) v = std::max<Real>(0, v);  // ReLU
    h = mat_vec(W_h2_, h);
    h = elem_add(h, b_h2_);

    // Reshape into B ∈ R^{d_model × r_lora} and A ∈ R^{r_lora × d_model}
    Index half = d_model_ * r_lora_;
    Mat B(d_model_, Vec(r_lora_));
    Mat A(r_lora_, Vec(d_model_));

    for (Index i = 0; i < d_model_; ++i) {
        for (Index j = 0; j < r_lora_; ++j) {
            B[i][j] = h[i * r_lora_ + j];
            A[j][i] = h[half + i * r_lora_ + j];
        }
    }

    return {B, A};
}

Vec ATAA::apply_lora(const Vec& x, const Mat& W0,
                      const Mat& B, const Mat& A) const
{
    // W_task x = W_0 x + B A x
    Vec result = mat_vec(W0, x);
    Vec ax = mat_vec(A, x);
    Vec bax = mat_vec(B, ax);
    return elem_add(result, bax);
}

void ATAA::update_gradient_subspace(const Mat& gradient) {
    // Simple power iteration-based subspace update
    if (ogd_count_ == 0) {
        // Initialize with normalized gradient columns
        for (Index j = 0; j < sketch_rank_; ++j) {
            Index col = j % gradient[0].size();
            for (Index i = 0; i < d_model_; ++i) {
                ogd_basis_[i][j] = gradient[i][col];
            }
        }
        ogd_count_++;
        return;
    }

    // Gram-Schmidt update to maintain orthonormal basis
    for (Index s = 0; s < sketch_rank_; ++s) {
        // Get gradient direction for this basis vector
        Vec g(d_model_);
        for (Index i = 0; i < d_model_; ++i) {
            g[i] = gradient[i][s % gradient[0].size()];
        }

        // Orthogonalize against existing basis
        for (Index j = 0; j < sketch_rank_; ++j) {
            Real proj = 0;
            for (Index i = 0; i < d_model_; ++i) {
                proj += ogd_basis_[i][j] * g[i];
            }
            for (Index i = 0; i < d_model_; ++i) {
                g[i] -= proj * ogd_basis_[i][j];
            }
        }

        // Normalize
        Real n = std::sqrt(norm2(g));
        if (n > EPS) {
            for (Index i = 0; i < d_model_; ++i) {
                ogd_basis_[i][s] = g[i] / n;
            }
        }
    }
    ogd_count_++;
}

Mat ATAA::project_gradient(const Mat& gradient) const {
    if (ogd_count_ == 0) return gradient;

    // P_t = I - U U^T where U is the basis of previous task gradients
    // ∇_proj = ∇ - U U^T ∇
    Mat proj = gradient;

    for (Index s = 0; s < sketch_rank_; ++s) {
        // Compute U_s^T ∇ (row vector)
        for (Index j = 0; j < (Index)gradient[0].size(); ++j) {
            Real coeff = 0;
            for (Index i = 0; i < d_model_; ++i) {
                coeff += ogd_basis_[i][s] * gradient[i][j];
            }
            for (Index i = 0; i < d_model_; ++i) {
                proj[i][j] -= coeff * ogd_basis_[i][s];
            }
        }
    }

    return proj;
}

std::pair<Mat, Mat> ATAA::adapt(const std::vector<Vec>& examples) {
    // Step 1: Encode task
    Vec task_emb = encode_task(examples);
    set_current_task(task_emb);

    // Step 2: Generate LoRA adapter via hypernetwork
    return generate_adapter(task_emb);
}

} // namespace ai2
