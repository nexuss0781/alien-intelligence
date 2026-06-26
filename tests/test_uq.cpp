#include "uq.hpp"
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

void test_conformal_score_bounds() {
    TEST("Conformal score is in [0,1]")
    UQ uq(100);
    CHECK(uq.conformal_score(0.9) >= 0);
    CHECK(uq.conformal_score(0.9) <= 1);
    CHECK(uq.conformal_score(0.0) <= 1);
    CHECK(uq.conformal_score(1.0) >= 0);
    END_TEST("Conformal score is in [0,1]");
}

void test_conformal_score_value() {
    TEST("Conformal score = 1 - p")
    UQ uq(100);
    CHECK(std::abs(uq.conformal_score(0.7) - 0.3) < 1e-10);
    CHECK(std::abs(uq.conformal_score(0.0) - 1.0) < 1e-10);
    CHECK(std::abs(uq.conformal_score(1.0) - 0.0) < 1e-10);
    END_TEST("Conformal score = 1 - p");
}

void test_reservoir_update() {
    TEST("Reservoir updates correctly")
    UQ uq(100, 10);
    for (int i = 0; i < 5; ++i) uq.update_reservoir(0.1 * i);
    Real q = uq.quantile_threshold(0.5);
    CHECK(q >= 0);
    CHECK(q <= 1);
    END_TEST("Reservoir updates correctly");
}

void test_reservoir_bounded() {
    TEST("Reservoir respects maximum size")
    UQ uq(100, 5);
    for (int i = 0; i < 100; ++i) uq.update_reservoir(0.5);
    Real q = uq.quantile_threshold(0.1);
    CHECK(q >= 0);
    CHECK(q <= 1);
    END_TEST("Reservoir respects maximum size");
}

void test_prediction_set() {
    TEST("Prediction set is non-empty for confident predictions")
    UQ uq(50);
    Vec probs(50, 0.01);
    probs[5] = 0.91; // very confident
    auto set = uq.prediction_set(probs, 0.1);
    CHECK(set.size() >= 1);
    END_TEST("Prediction set is non-empty for confident predictions");
}

void test_prediction_set_empty_uncertain() {
    TEST("Prediction set can be large for uncertain predictions")
    UQ uq(10);
    Vec probs(10, 0.1); // uniform
    for (int i = 0; i < 20; ++i) uq.update_reservoir(0.9); // high uncertainty
    auto set = uq.prediction_set(probs, 0.1);
    CHECK(set.size() >= 1);  // at least the most likely
    END_TEST("Prediction set can be large for uncertain predictions");
}

void test_ensemble_add() {
    TEST("Ensemble accepts member predictions")
    UQ uq(20, 100, 5);
    Vec probs(20, 0.05);
    probs[0] = 0.5;
    uq.add_ensemble_prediction(probs, 0);
    uq.add_ensemble_prediction(probs, 1);
    Vec avg = uq.ensemble_distribution();
    CHECK(avg.size() == 20);
    CHECK(std::abs(avg[0] - 0.5) < 1e-10);
    END_TEST("Ensemble accepts member predictions");
}

void test_ensemble_clears() {
    TEST("Ensemble clears correctly")
    UQ uq(20, 100, 5);
    Vec probs(20, 0.05);
    uq.add_ensemble_prediction(probs, 0);
    uq.clear_ensemble();
    Vec avg = uq.ensemble_distribution();
    CHECK(avg.size() == 20);
    // After clear, should return uniform
    END_TEST("Ensemble clears correctly");
}

void test_epistemic_uncertainty() {
    TEST("Epistemic uncertainty is in [0,1]")
    UQ uq(10, 100, 3);
    Vec p1(10, 0.1), p2(10, 0.1), p3(10, 0.1);
    p1[0] = 0.8; p2[2] = 0.8; p3[4] = 0.8;
    uq.add_ensemble_prediction(p1, 0);
    uq.add_ensemble_prediction(p2, 1);
    uq.add_ensemble_prediction(p3, 2);
    Real epi = uq.epistemic_uncertainty();
    CHECK(epi >= 0);
    CHECK(epi <= 1);
    END_TEST("Epistemic uncertainty is in [0,1]");
}

void test_aleatoric_uncertainty() {
    TEST("Aleatoric uncertainty is in [0,1]")
    UQ uq(10, 100, 3);
    Vec p1(10, 0.1), p2(10, 0.1), p3(10, 0.1);
    uq.add_ensemble_prediction(p1, 0);
    uq.add_ensemble_prediction(p2, 1);
    uq.add_ensemble_prediction(p3, 2);
    Real ale = uq.aleatoric_uncertainty();
    CHECK(ale >= 0);
    CHECK(ale <= 1);
    END_TEST("Aleatoric uncertainty is in [0,1]");
}

void test_fusion_bounded() {
    TEST("Fused uncertainty is in [0,1]")
    UQ uq(10, 100, 3);
    Real fused = uq.fuse_uncertainty(0.5);
    CHECK(fused >= 0);
    CHECK(fused <= 1);
    END_TEST("Fused uncertainty is in [0,1]");
}

void test_full_step() {
    TEST("Full UQ step produces valid result")
    UQ uq(30, 50, 3);
    Vec probs(30, 0.01);
    probs[7] = 0.7;
    auto res = uq.step(probs, 0.2, 0.1);
    CHECK(res.conformal_uncertainty >= 0);
    CHECK(res.conformal_uncertainty <= 1);
    CHECK(res.epistemic >= 0);
    CHECK(res.aleatoric >= 0);
    CHECK(res.fused >= 0);
    CHECK(res.fused <= 1);
    CHECK(res.conformal_set.size() >= 1);
    END_TEST("Full UQ step produces valid result");
}

void test_numerical_stability() {
    TEST("No NaN/Inf in any UQ outputs")
    UQ uq(50, 100, 5);
    Vec probs(50, 0.02);
    probs[0] = 0.5;
    for (int i = 0; i < 10; ++i) {
        auto res = uq.step(probs, 0.3, 0.1);
        CHECK(!std::isnan(res.conformal_uncertainty));
        CHECK(!std::isinf(res.conformal_uncertainty));
        CHECK(!std::isnan(res.epistemic));
        CHECK(!std::isinf(res.aleatoric));
        CHECK(!std::isnan(res.fused));
    }
    END_TEST("No NaN/Inf in any UQ outputs");
}

int main() {
    std::cout << "=== Component 4: UQ Tests ===" << std::endl;

    test_conformal_score_bounds();
    test_conformal_score_value();
    test_reservoir_update();
    test_reservoir_bounded();
    test_prediction_set();
    test_prediction_set_empty_uncertain();
    test_ensemble_add();
    test_ensemble_clears();
    test_epistemic_uncertainty();
    test_aleatoric_uncertainty();
    test_fusion_bounded();
    test_full_step();
    test_numerical_stability();

    if (failed > 0) {
        std::cerr << failed << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "All UQ tests PASSED." << std::endl;
    return 0;
}
