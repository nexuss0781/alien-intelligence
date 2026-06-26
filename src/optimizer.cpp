#include "optimizer.hpp"
#include <cmath>
#include <algorithm>

namespace ai2 {

Optimizer::Optimizer(Type type, Real lr, Real beta1, Real beta2,
                     Real eps, Real weight_decay)
    : type_(type), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {}

void Optimizer::add_param(const std::string& name, Vec* param, Vec* gradient) {
    ParamState state;
    state.param = param;
    state.gradient = gradient;
    state.m.resize(param->size(), 0);
    state.v.resize(param->size(), 0);
    params_.push_back(state);
    (void)name;
}

void Optimizer::step() {
    step_++;

    for (auto& state : params_) {
        Vec& p = *state.param;
        Vec& g = *state.gradient;
        Index n = p.size();

        if (g.size() != n) continue;

        if (type_ == SGD) {
            for (Index i = 0; i < n; ++i) {
                Real grad = g[i] + weight_decay_ * p[i];
                // Gradient clipping
                grad = std::clamp(grad, -1.0, 1.0);
                p[i] -= lr_ * grad;
            }
        } else if (type_ == ADAM) {
            for (Index i = 0; i < n; ++i) {
                Real grad = g[i] + weight_decay_ * p[i];

                state.m[i] = beta1_ * state.m[i] + (1 - beta1_) * grad;
                state.v[i] = beta2_ * state.v[i] + (1 - beta2_) * grad * grad;

                Real m_hat = state.m[i] / (1 - std::pow(beta1_, step_));
                Real v_hat = state.v[i] / (1 - std::pow(beta2_, step_));

                Real update = m_hat / (std::sqrt(v_hat) + eps_);
                // Gradient clipping
                update = std::clamp(update, -1.0, 1.0);
                p[i] -= lr_ * update;
            }
        }
    }
}

void Optimizer::zero_grad() {
    for (auto& state : params_) {
        if (state.gradient) {
            std::fill(state.gradient->begin(), state.gradient->end(), 0);
        }
    }
}

Index Optimizer::num_params() const {
    Index count = 0;
    for (auto& state : params_) {
        count += state.param->size();
    }
    return count;
}

} // namespace ai2
