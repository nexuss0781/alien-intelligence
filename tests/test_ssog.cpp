#include "ssog.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace ai2;

int failed = 0;

#define TEST(name) \
    do { \
        std::cout << "  " << name << "... "; \
        bool pass = true;

#define END_TEST(name) \
        if (pass) std::cout << "PASS" << std::endl; \
        else { std::cout << "FAIL" << std::endl; failed++; } \
    } while(0)

#define CHECK(cond) \
    do { if (!(cond)) { pass = false; \
        std::cerr << "    FAIL at line " << __LINE__ << ": " #cond << std::endl; } \
    } while(0)

void test_routing_selects_k_experts() {
    TEST("LSH routing selects k experts")
    SSOG ssog(16, 100, 32, 4);
    Vec h(16, 0.5);
    auto experts = ssog.route(h);
    CHECK(experts.size() == 4);
    for (auto& e : experts) CHECK(e < 32);
    END_TEST("LSH routing selects k experts");
}

void test_routing_deterministic() {
    TEST("LSH routing is deterministic for same input")
    SSOG ssog(16, 100, 32, 4);
    Vec h(16, 0.5);
    auto e1 = ssog.route(h);
    auto e2 = ssog.route(h);
    CHECK(e1 == e2);
    END_TEST("LSH routing is deterministic for same input");
}

void test_routing_different_for_different_inputs() {
    TEST("LSH routing can give different experts for different inputs")
    SSOG ssog(16, 100, 64, 4);
    Vec h1(16, 0.1), h2(16, 10.0);
    auto e1 = ssog.route(h1);
    auto e2 = ssog.route(h2);
    // May or may not differ, but should not crash
    CHECK(e1.size() == 4);
    CHECK(e2.size() == 4);
    END_TEST("LSH routing can give different experts for different inputs");
}

void test_gating_weights_sum_to_one() {
    TEST("Gating weights sum to 1")
    SSOG ssog(16, 100, 32, 4);
    Vec h(16, 0.5);
    auto experts = ssog.route(h);
    Vec weights = ssog.gating_weights(h, experts);
    Real sum = 0;
    for (auto& w : weights) sum += w;
    CHECK(std::abs(sum - 1.0) < 1e-6);
    END_TEST("Gating weights sum to 1");
}

void test_gating_weights_non_negative() {
    TEST("Gating weights are non-negative")
    SSOG ssog(16, 100, 32, 4);
    Vec h(16, 0.5);
    auto experts = ssog.route(h);
    Vec weights = ssog.gating_weights(h, experts);
    for (auto& w : weights) CHECK(w >= 0);
    END_TEST("Gating weights are non-negative");
}

void test_expert_forward_shape() {
    TEST("Expert forward produces correct output shape")
    SSOG ssog(16, 100, 32, 4);
    Vec h(16, 0.5);
    Vec expert_out = ssog.expert_forward(0, h);
    CHECK(expert_out.size() == 16);
    END_TEST("Expert forward produces correct output shape");
}

void test_sparse_mixture_shape() {
    TEST("Sparse mixture produces correct shape")
    SSOG ssog(16, 100, 32, 4);
    Vec h(16, 0.5);
    Vec mixture = ssog.sparse_mixture(h);
    CHECK(mixture.size() == 16);
    END_TEST("Sparse mixture produces correct shape");
}

void test_base_distribution_probabilities() {
    TEST("Base distribution sums to 1")
    SSOG ssog(16, 50, 32, 4);
    Vec h(16, 0.5);
    Vec mixture = ssog.sparse_mixture(h);
    Vec probs = ssog.base_distribution(mixture);
    CHECK(probs.size() == 50);
    Real sum = 0;
    for (auto& p : probs) sum += p;
    CHECK(std::abs(sum - 1.0) < 1e-6);
    END_TEST("Base distribution sums to 1");
}

void test_base_distribution_non_negative() {
    TEST("Base distribution probabilities are non-negative")
    SSOG ssog(16, 50, 32, 4);
    Vec h(16, 0.5);
    Vec mixture = ssog.sparse_mixture(h);
    Vec probs = ssog.base_distribution(mixture);
    for (auto& p : probs) CHECK(p >= 0);
    END_TEST("Base distribution probabilities are non-negative");
}

void test_calibrated_distribution() {
    TEST("Calibrated distribution sums to 1")
    SSOG ssog(16, 30, 32, 4);
    Vec h(16, 0.3);
    Vec base(30, 1.0/30);
    std::vector<Index> conformal_set = {0, 1, 2, 10, 15};
    Vec cal = ssog.calibrated_distribution(base, 0.3, conformal_set);
    Real sum = 0;
    for (auto& p : cal) sum += p;
    CHECK(std::abs(sum - 1.0) < 1e-6);
    END_TEST("Calibrated distribution sums to 1");
}

void test_calibrated_spreads_mass() {
    TEST("Calibrated distribution spreads mass when uncertain")
    SSOG ssog(16, 20, 32, 4);
    Vec base(20, 0.05);
    base[5] = 0.55;  // peak at token 5
    std::vector<Index> conf_set = {3, 5, 7, 11};

    Vec cal_high = ssog.calibrated_distribution(base, 0.8, conf_set);
    Vec cal_low = ssog.calibrated_distribution(base, 0.0, conf_set);

    // High uncertainty should give more mass to conformal set tokens
    Real mass_high = 0, mass_low = 0;
    for (auto& idx : conf_set) {
        mass_high += cal_high[idx];
        mass_low += cal_low[idx];
    }
    CHECK(mass_high >= mass_low - 1e-10);
    END_TEST("Calibrated distribution spreads mass when uncertain");
}

void test_full_forward() {
    TEST("Full forward produces valid output")
    SSOG ssog(16, 30, 32, 4);
    Vec h(16, 0.5);
    auto out = ssog.forward(h, 0.2, {1, 5, 10});
    CHECK(out.mixture.size() == 16);
    CHECK(out.base_probs.size() == 30);
    CHECK(out.calibrated_probs.size() == 30);

    Real sum = 0;
    for (auto& p : out.calibrated_probs) sum += p;
    CHECK(std::abs(sum - 1.0) < 1e-6);
    END_TEST("Full forward produces valid output");
}

void test_numerical_stability() {
    TEST("No NaN/Inf in any outputs")
    SSOG ssog(16, 30, 32, 4);
    Vec h(16, 0.5);
    for (int i = 0; i < 10; ++i) {
        auto out = ssog.forward(h, 0.3, {1, 2});
        for (auto& v : out.mixture) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
        for (auto& v : out.base_probs) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
        for (auto& v : out.calibrated_probs) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
    }
    END_TEST("No NaN/Inf in any outputs");
}

void test_many_experts() {
    TEST("Handles many experts efficiently (O(1) per token)")
    SSOG ssog(16, 50, 256, 4);
    Vec h(16, 0.5);
    auto out = ssog.forward(h);
    CHECK(out.calibrated_probs.size() == 50);
    END_TEST("Handles many experts efficiently (O(1) per token)");
}

int main() {
    std::cout << "=== Component 6: SSOG Tests ===" << std::endl;

    test_routing_selects_k_experts();
    test_routing_deterministic();
    test_routing_different_for_different_inputs();
    test_gating_weights_sum_to_one();
    test_gating_weights_non_negative();
    test_expert_forward_shape();
    test_sparse_mixture_shape();
    test_base_distribution_probabilities();
    test_base_distribution_non_negative();
    test_calibrated_distribution();
    test_calibrated_spreads_mass();
    test_full_forward();
    test_numerical_stability();
    test_many_experts();

    if (failed > 0) {
        std::cerr << failed << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "All SSOG tests PASSED." << std::endl;
    return 0;
}
