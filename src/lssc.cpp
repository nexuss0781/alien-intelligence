#include "lssc.hpp"
#include <random>
#include <complex>

namespace ai2 {

LSSC::LSSC(Index d_model, Index d_state, Index n_rf)
    : d_model_(d_model), d_state_(d_state), n_rf_(n_rf)
{
    std::mt19937_64 rng(2024);

    // S4D: Initialize Λ = diag(λ_j) with λ_j = exp(ω_j + i·φ_j)
    {
        Real log_scale = -0.5;
        std::normal_distribution<Real> omega_dist(log_scale, 0.1);
        std::uniform_real_distribution<Real> phi_dist(0, 2 * PI);

        lambda_.resize(d_state_);
        omega_.resize(d_state_);
        phi_.resize(d_state_);

        for (Index j = 0; j < d_state_; ++j) {
            omega_[j] = omega_dist(rng);
            phi_[j] = phi_dist(rng);
            lambda_[j] = std::exp(std::complex<Real>(omega_[j], phi_[j]));
        }

        Real scale_b = std::sqrt(2.0 / d_state_);
        B_.resize(d_state_, Vec(d_model_));
        for (auto& row : B_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale_b)(rng);

        Real scale_c = std::sqrt(2.0 / d_model_);
        C_.resize(d_model_, Vec(d_state_));
        for (auto& row : C_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale_c)(rng);

        D_.resize(d_model_);
        std::fill(D_.begin(), D_.end(), 0.1);
    }

    // RFA: Random projection matrix W_rf ∈ R^{n_rf × d_model}
    {
        W_rf_.resize(n_rf_, Vec(d_model_));
        for (auto& row : W_rf_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, 1)(rng);
    }

    // Gating parameters
    {
        Real scale_g = std::sqrt(2.0 / (d_model_ + d_model_));
        W_g_.resize(d_model_, Vec(d_model_));
        for (auto& row : W_g_)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale_g)(rng);
        b_g_.resize(d_model_, 0);
    }
}

LSSC::S4DState LSSC::ssm_init() const {
    S4DState s;
    s.x.resize(d_state_, std::complex<Real>(0, 0));
    return s;
}

Vec LSSC::ssm_step(S4DState& state, const Vec& u_k) {
    // x_{k+1} = Λ · x_k + B · u_k
    CVec x_next(d_state_);
    for (Index j = 0; j < d_state_; ++j) {
        x_next[j] = lambda_[j] * state.x[j];
        std::complex<Real> bu(0, 0);
        for (Index i = 0; i < u_k.size(); ++i) {
            bu += B_[j][i] * u_k[i];
        }
        x_next[j] += bu;
    }
    state.x = x_next;

    // y_k = Re(C · x_k) + D · u_k
    Vec y(d_model_, 0);
    for (Index i = 0; i < d_model_; ++i) {
        for (Index j = 0; j < d_state_; ++j) {
            y[i] += C_[i][j] * state.x[j].real();
            y[i] -= C_[i][j] * state.x[j].imag();  // Re(C * conj is same as)
        }
        // Actually: y = Re(C · x) + D ⊙ u
        y[i] += D_[i] * u_k[i];
    }
    return y;
}

Mat LSSC::ssm_scan(const Mat& U) {
    Index n = U.size();
    Mat Y(n);
    S4DState state = ssm_init();
    for (Index i = 0; i < n; ++i) {
        Y[i] = ssm_step(state, U[i]);
    }
    return Y;
}

Vec LSSC::rfa_feature(const Vec& x) const {
    // φ(x, w) = exp(w^T x - ||x||^2 / 2)
    Vec phi(n_rf_);
    Real n2 = norm2(x) / 2;
    for (Index j = 0; j < n_rf_; ++j) {
        Real dot_val = dot(W_rf_[j], x);
        phi[j] = std::exp(dot_val - n2);
    }
    return phi;
}

Mat LSSC::rfa_causal(const Mat& Q, const Mat& K, const Mat& V) {
    Index n = Q.size();
    assert(n == K.size() && n == V.size());

    // Compute random features for all positions
    Mat phi_Q(n), phi_K(n);
    for (Index i = 0; i < n; ++i) {
        phi_Q[i] = rfa_feature(Q[i]);
        phi_K[i] = rfa_feature(K[i]);
    }

    // Causal: Attention_i = (φ(Q_i) · Σ_{j≤i} φ(K_j)^T V_j) / (φ(Q_i) · Σ_{j≤i} φ(K_j)^T)
    // Using prefix sums: O(n)
    Mat Y(n, Vec(d_model_, 0));

    // Prefix sum of φ(K)^T V ∈ R^{n_rf × d_model}
    Mat KV_prefix(n_rf_, Vec(d_model_, 0));
    // Prefix sum of φ(K) ∈ R^{n_rf}
    Vec K_prefix(n_rf_, 0);

    for (Index i = 0; i < n; ++i) {
        // Update prefix sums
        for (Index j = 0; j < n_rf_; ++j) {
            K_prefix[j] += phi_K[i][j];
            for (Index k = 0; k < d_model_; ++k) {
                KV_prefix[j][k] += phi_K[i][j] * V[i][k];
            }
        }

        // Compute numerator: φ(Q_i) · KV_prefix
        Vec num(d_model_, 0);
        Real den = 0;
        for (Index j = 0; j < n_rf_; ++j) {
            den += phi_Q[i][j] * K_prefix[j];
            for (Index k = 0; k < d_model_; ++k) {
                num[k] += phi_Q[i][j] * KV_prefix[j][k];
            }
        }

        // Normalize
        if (den > EPS) {
            for (Index k = 0; k < d_model_; ++k) {
                Y[i][k] = num[k] / den;
            }
        }
    }
    return Y;
}

Mat LSSC::rfa_bidirectional(const Mat& Q, const Mat& K, const Mat& V) {
    Index n = Q.size();
    // Full (non-causal): compute φ(Q) · (φ(K)^T V) / (φ(Q) · φ(K)^T)
    // φ(Q) ∈ R^{n × n_rf}, φ(K) ∈ R^{n × n_rf}
    // φ(K)^T V ∈ R^{n_rf × d_model}
    // φ(Q) · (φ(K)^T V) ∈ R^{n × d_model}
    // denom: φ(Q) · φ(K)^T_1 ∈ R^{n}

    Mat phi_Q(n), phi_K(n);
    for (Index i = 0; i < n; ++i) {
        phi_Q[i] = rfa_feature(Q[i]);
        phi_K[i] = rfa_feature(K[i]);
    }

    // Compute KV = φ(K)^T V
    Mat KV(n_rf_, Vec(d_model_, 0));
    for (Index j = 0; j < n_rf_; ++j) {
        for (Index i = 0; i < n; ++i) {
            for (Index k = 0; k < d_model_; ++k) {
                KV[j][k] += phi_K[i][j] * V[i][k];
            }
        }
    }

    // Compute K_sum = Σ_i φ(K_i)
    Vec K_sum(n_rf_, 0);
    for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < n_rf_; ++j)
            K_sum[j] += phi_K[i][j];

    Mat Y(n, Vec(d_model_, 0));
    for (Index i = 0; i < n; ++i) {
        Real den = 0;
        for (Index j = 0; j < n_rf_; ++j) {
            den += phi_Q[i][j] * K_sum[j];
            for (Index k = 0; k < d_model_; ++k) {
                Y[i][k] += phi_Q[i][j] * KV[j][k];
            }
        }
        if (den > EPS) {
            for (Index k = 0; k < d_model_; ++k) {
                Y[i][k] /= den;
            }
        }
    }
    return Y;
}

Vec LSSC::gated_step(const Vec& z_k, const Vec& y_ssm, const Vec& y_rfa) {
    // α_k = σ(W_g · z_k + b_g)
    Vec gate = mat_vec(W_g_, z_k);
    gate = elem_add(gate, b_g_);
    for (auto& v : gate) v = sigmoid(v);

    // h_k = α ⊙ y_ssm + (1 - α) ⊙ y_rfa
    Vec h(d_model_);
    for (Index i = 0; i < d_model_; ++i) {
        h[i] = gate[i] * y_ssm[i] + (1 - gate[i]) * y_rfa[i];
    }
    return h;
}

Mat LSSC::forward(const Mat& Z) {
    Index n = Z.size();
    if (n == 0) return {};

    // Extract Q, K, V from Z (in practice, these would be learned projections)
    // For this implementation, we use Z directly as input to both pathways
    Mat Q = Z, K = Z, V = Z;

    // SSM pathway (parallel scan)
    Mat Y_ssm = ssm_scan(Z);

    // RFA pathway (bidirectional for full context, causal for autoregressive)
    Mat Y_rfa = rfa_causal(Q, K, V);

    // Gated fusion
    Mat H(n);
    for (Index i = 0; i < n; ++i) {
        H[i] = gated_step(Z[i], Y_ssm[i], Y_rfa[i]);
    }
    return H;
}

} // namespace ai2
