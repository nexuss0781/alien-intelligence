#pragma once
#include "types.hpp"
#include <unordered_map>
#include <string>
#include <vector>

namespace ai2 {

class Optimizer {
public:
    enum Type { SGD, ADAM };
    Optimizer(Type type = ADAM, Real lr = 0.001,
              Real beta1 = 0.9, Real beta2 = 0.999,
              Real eps = 1e-8, Real weight_decay = 0.01);

    // Register a parameter tensor for optimization
    void add_param(const std::string& name, Vec* param, Vec* gradient);

    // Step: update all parameters
    void step();

    // Zero all gradients
    void zero_grad();

    // Set learning rate
    void set_lr(Real lr) { lr_ = lr; }
    Real lr() const { return lr_; }

    // Learning rate scheduling
    void set_step(Index step) { step_ = step; }
    Index step() const { return step_; }

    // Get total parameter count
    Index num_params() const;

private:
    Type type_;
    Real lr_;
    Real beta1_, beta2_, eps_, weight_decay_;
    Index step_ = 0;

    struct ParamState {
        Vec* param;
        Vec* gradient;
        Vec m;  // first moment (Adam)
        Vec v;  // second moment (Adam)
    };

    std::vector<ParamState> params_;
};

} // namespace ai2
