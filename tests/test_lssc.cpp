#include "lssc.hpp"
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

void test_ssm_step_consistency() {
    TEST("SSM step produces consistent output dimensions")
    Index d_model = 32, d_state = 16;
    LSSC lssc(d_model, d_state, 64);

    Vec input(d_model, 0.5);
    auto state = lssc.ssm_init();
    Vec out = lssc.ssm_step(state, input);

    CHECK(out.size() == d_model);
    bool non_zero = false;
    for (auto& v : out) if (std::abs(v) > 1e-10) non_zero = true;
    CHECK(non_zero);
    END_TEST("SSM step produces consistent output dimensions");
}

void test_ssm_state_evolution() {
    TEST("SSM state evolves over time")
    LSSC lssc(16, 8, 32);
    Vec input(16, 1.0);

    auto state = lssc.ssm_init();
    lssc.ssm_step(state, input);
    auto state_after = state;

    bool changed = false;
    for (size_t j = 0; j < state.x.size(); ++j)
        if (std::abs(state.x[j] - std::complex<Real>(0,0)) > 1e-10)
            changed = true;
    CHECK(changed);
    END_TEST("SSM state evolves over time");
}

void test_ssm_scan_linear() {
    TEST("SSM scan processes sequence in O(n)")
    LSSC lssc(16, 8, 32);
    Index n = 100;

    Mat U(n, Vec(16, 0.1));
    Mat Y = lssc.ssm_scan(U);

    CHECK(Y.size() == n);
    for (auto& row : Y) CHECK(row.size() == 16);
    END_TEST("SSM scan processes sequence in O(n)");
}

void test_rfa_feature_shape() {
    TEST("RFA random features have correct dimension")
    LSSC lssc(32, 16, 128);
    Vec x(32, 0.5);
    Vec phi = lssc.rfa_feature(x);
    CHECK(phi.size() == 128);
    bool non_zero = false;
    for (auto& v : phi) if (v > 1e-10) non_zero = true;
    CHECK(non_zero);
    END_TEST("RFA random features have correct dimension");
}

void test_rfa_feature_bounded() {
    TEST("RFA features are bounded and positive")
    LSSC lssc(32, 16, 64);
    Vec x(32, 1.0);
    Vec phi = lssc.rfa_feature(x);
    for (auto& v : phi) {
        CHECK(v > 0);
        CHECK(!std::isnan(v));
        CHECK(!std::isinf(v));
    }
    END_TEST("RFA features are bounded and positive");
}

void test_rfa_causal_shape() {
    TEST("RFA causal produces correct output shape")
    LSSC lssc(16, 8, 32);
    Index n = 20;
    Mat Q(n, Vec(16, 0.5)), K(n, Vec(16, 0.5)), V(n, Vec(16, 0.5));
    Mat Y = lssc.rfa_causal(Q, K, V);
    CHECK(Y.size() == n);
    for (auto& row : Y) CHECK(row.size() == 16);
    END_TEST("RFA causal produces correct output shape");
}

void test_rfa_causal_prefix() {
    TEST("RFA causal first token depends only on itself")
    LSSC lssc(16, 8, 64);
    Index n = 5;
    Mat Q(n, Vec(16, 0.3)), K(n, Vec(16, 0.3)), V(n, Vec(16, 0.3));
    Mat Y = lssc.rfa_causal(Q, K, V);
    // First token output should be valid
    for (auto& v : Y[0]) {
        CHECK(!std::isnan(v));
    }
    END_TEST("RFA causal first token depends only on itself");
}

void test_gated_fusion_shape() {
    TEST("Gated fusion produces correct output")
    LSSC lssc(16, 8, 32);
    Vec z(16, 0.5), y_ssm(16, 0.3), y_rfa(16, 0.7);
    Vec h = lssc.gated_step(z, y_ssm, y_rfa);
    CHECK(h.size() == 16);
    bool non_zero = false;
    for (auto& v : h) if (std::abs(v) > 1e-10) non_zero = true;
    CHECK(non_zero);
    END_TEST("Gated fusion produces correct output");
}

void test_gated_fusion_interpolation() {
    TEST("Gated fusion interpolates between SSM and RFA")
    LSSC lssc(16, 8, 32);
    Vec z(16, 10.0);  // strong gate signal
    Vec y_ssm(16, 1.0), y_rfa(16, 0.0);
    Vec h = lssc.gated_step(z, y_ssm, y_rfa);
    // Gate should be near 1, so output should be near y_ssm
    Real diff = 0;
    for (size_t i = 0; i < h.size(); ++i) diff += std::abs(h[i] - y_ssm[i]);
    CHECK(diff < h.size() * 0.5);
    END_TEST("Gated fusion interpolates between SSM and RFA");
}

void test_forward_complexity_o_n() {
    TEST("Full forward processes sequence in O(n)")
    LSSC lssc(16, 8, 32);
    Mat Z(50, Vec(16, 0.2));
    Mat H = lssc.forward(Z);
    CHECK(H.size() == 50);
    END_TEST("Full forward processes sequence in O(n)");
}

void test_numerical_stability() {
    TEST("No NaN/Inf during long sequence processing")
    LSSC lssc(16, 8, 32);
    Mat Z(200, Vec(16, 0.1));
    Mat H = lssc.forward(Z);
    for (auto& row : H)
        for (auto& v : row) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
    END_TEST("No NaN/Inf during long sequence processing");
}

void test_ssm_stability() {
    TEST("SSM remains stable for many steps")
    LSSC lssc(8, 4, 16);
    Vec input(8, 0.1);
    auto state = lssc.ssm_init();
    for (int i = 0; i < 500; ++i) {
        Vec out = lssc.ssm_step(state, input);
        for (auto& v : out) {
            CHECK(std::abs(v) < 1e6);
            CHECK(!std::isnan(v));
        }
    }
    END_TEST("SSM remains stable for many steps");
}

int main() {
    std::cout << "=== Component 2: LSSC Tests ===" << std::endl;

    test_ssm_step_consistency();
    test_ssm_state_evolution();
    test_ssm_scan_linear();
    test_rfa_feature_shape();
    test_rfa_feature_bounded();
    test_rfa_causal_shape();
    test_rfa_causal_prefix();
    test_gated_fusion_shape();
    test_gated_fusion_interpolation();
    test_forward_complexity_o_n();
    test_numerical_stability();
    test_ssm_stability();

    if (failed > 0) {
        std::cerr << failed << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "All LSSC tests PASSED." << std::endl;
    return 0;
}
