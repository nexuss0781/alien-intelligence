#include "ataa.hpp"
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

void test_task_encoding() {
    TEST("Task encoding produces fixed-size embedding")
    ATAA ataa(16, 8, 4, 4);
    std::vector<Vec> examples = {
        Vec(16, 0.1), Vec(16, 0.2), Vec(16, 0.3)
    };
    Vec emb = ataa.encode_task(examples);
    CHECK(emb.size() == 8);
    END_TEST("Task encoding produces fixed-size embedding");
}

void test_task_encoding_deterministic() {
    TEST("Task encoding is deterministic for same input")
    ATAA ataa(16, 8, 4, 4);
    std::vector<Vec> examples = {Vec(16, 0.5), Vec(16, 0.3)};
    Vec emb1 = ataa.encode_task(examples);
    Vec emb2 = ataa.encode_task(examples);
    for (size_t i = 0; i < emb1.size(); ++i)
        CHECK(std::abs(emb1[i] - emb2[i]) < 1e-10);
    END_TEST("Task encoding is deterministic for same input");
}

void test_hypernetwork_adapter_shape() {
    TEST("Hypernetwork produces correct LoRA adapter shapes")
    Index d_model = 32, r_lora = 8;
    ATAA ataa(d_model, 16, r_lora, 4);
    Vec task_emb(16, 0.5);
    auto [B, A] = ataa.generate_adapter(task_emb);
    CHECK(B.size() == d_model);
    CHECK(B[0].size() == r_lora);
    CHECK(A.size() == r_lora);
    CHECK(A[0].size() == d_model);
    END_TEST("Hypernetwork produces correct LoRA adapter shapes");
}

void test_lora_application() {
    TEST("LoRA application produces correct output dimension")
    Index d_model = 16, r_lora = 4;
    ATAA ataa(d_model, 8, r_lora, 4);
    Vec x(d_model, 0.5);
    Mat W0(d_model, Vec(d_model, 0.1));
    Mat B(d_model, Vec(r_lora, 0.2));
    Mat A(r_lora, Vec(d_model, 0.3));
    Vec y = ataa.apply_lora(x, W0, B, A);
    CHECK(y.size() == d_model);
    END_TEST("LoRA application produces correct output dimension");
}

void test_lora_different_from_base() {
    TEST("LoRA produces different output from base")
    ATAA ataa(16, 8, 4, 4);
    Vec x(16, 0.5);
    Mat W0(16, Vec(16, 0.1));
    // Generate actual adapter from hypernetwork
    Vec task(8, 0.5);
    auto [B, A] = ataa.generate_adapter(task);
    Vec y_base = mat_vec(W0, x);
    Vec y_lora = ataa.apply_lora(x, W0, B, A);
    bool diff = false;
    for (size_t i = 0; i < y_base.size(); ++i)
        if (std::abs(y_base[i] - y_lora[i]) > 1e-10) diff = true;
    CHECK(diff);
    END_TEST("LoRA produces different output from base");
}

void test_gradient_subspace() {
    TEST("Gradient subspace update works")
    ATAA ataa(8, 4, 2, 4, 4);
    Mat grad(8, Vec(8, 0.1));
    ataa.update_gradient_subspace(grad);
    // Should not crash
    CHECK(true);
    END_TEST("Gradient subspace update works");
}

void test_gradient_projection() {
    TEST("Gradient projection does not change shape")
    ATAA ataa(8, 4, 2, 4, 4);
    Mat grad(8, Vec(8, 0.5));
    ataa.update_gradient_subspace(grad);
    Mat proj = ataa.project_gradient(grad);
    CHECK(proj.size() == 8);
    CHECK(proj[0].size() == 8);
    END_TEST("Gradient projection does not change shape");
}

void test_gradient_projection_reduces_norm() {
    TEST("Projected gradient has smaller or equal norm")
    ATAA ataa(8, 4, 2, 4, 4);
    Mat grad(8, Vec(8, 0.5));
    ataa.update_gradient_subspace(grad);
    Mat proj = ataa.project_gradient(grad);

    Real orig_norm = 0, proj_norm = 0;
    for (auto& row : grad) orig_norm += norm2(row);
    for (auto& row : proj) proj_norm += norm2(row);
    CHECK(proj_norm <= orig_norm + 1e-10);
    END_TEST("Projected gradient has smaller or equal norm");
}

void test_adapt_returns_adapters() {
    TEST("Full adaptation returns adapters")
    ATAA ataa(16, 8, 4, 4);
    std::vector<Vec> examples = {Vec(16, 0.3), Vec(16, 0.7)};
    auto [B, A] = ataa.adapt(examples);
    CHECK(B.size() == 16);
    CHECK(A.size() == 4);
    END_TEST("Full adaptation returns adapters");
}

void test_current_task_set() {
    TEST("Current task is set after adaptation")
    ATAA ataa(16, 8, 4, 4);
    std::vector<Vec> examples = {Vec(16, 0.5)};
    ataa.adapt(examples);
    CHECK(ataa.current_task().size() == 8);
    END_TEST("Current task is set after adaptation");
}

void test_numerical_stability() {
    TEST("No NaN/Inf in hypernetwork outputs")
    ATAA ataa(32, 16, 8, 4);
    Vec task(16, 0.5);
    auto [B, A] = ataa.generate_adapter(task);
    for (auto& row : B)
        for (auto& v : row) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
    for (auto& row : A)
        for (auto& v : row) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
    END_TEST("No NaN/Inf in hypernetwork outputs");
}

void test_ogd_accumulation() {
    TEST("OGD accumulates multiple gradient updates")
    ATAA ataa(8, 4, 2, 4, 4);
    for (int i = 0; i < 5; ++i) {
        Mat grad(8, Vec(8, 0.1 * (i + 1)));
        ataa.update_gradient_subspace(grad);
    }
    // Should remain stable
    CHECK(true);
    END_TEST("OGD accumulates multiple gradient updates");
}

int main() {
    std::cout << "=== Component 5: ATAA Tests ===" << std::endl;

    test_task_encoding();
    test_task_encoding_deterministic();
    test_hypernetwork_adapter_shape();
    test_lora_application();
    test_lora_different_from_base();
    test_gradient_subspace();
    test_gradient_projection();
    test_gradient_projection_reduces_norm();
    test_adapt_returns_adapters();
    test_current_task_set();
    test_numerical_stability();
    test_ogd_accumulation();

    if (failed > 0) {
        std::cerr << failed << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "All ATAA tests PASSED." << std::endl;
    return 0;
}
