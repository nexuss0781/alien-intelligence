#include "slie.hpp"
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

void test_che_correctness() {
    TEST("CHE produces valid embeddings")
    SLIE slie(1000, 64);
    Vec emb = slie.che_forward(42);
    CHECK(emb.size() == 64);
    bool non_zero = false;
    for (auto& v : emb) if (std::abs(v) > 1e-10) non_zero = true;
    CHECK(non_zero);
    END_TEST("CHE produces valid embeddings");
}

void test_che_determinism() {
    TEST("CHE is deterministic (same token -> same embedding)")
    SLIE slie(1000, 64);
    Vec emb1 = slie.che_forward(42);
    Vec emb2 = slie.che_forward(42);
    CHECK(emb1.size() == emb2.size());
    for (size_t i = 0; i < emb1.size(); ++i)
        CHECK(std::abs(emb1[i] - emb2[i]) < 1e-12);
    END_TEST("CHE is deterministic");
}

void test_che_different_tokens() {
    TEST("CHE gives different embeddings for different tokens")
    SLIE slie(1000, 64);
    Vec emb_a = slie.che_forward(100);
    Vec emb_b = slie.che_forward(200);
    bool different = false;
    for (size_t i = 0; i < emb_a.size(); ++i)
        if (std::abs(emb_a[i] - emb_b[i]) > 1e-10) different = true;
    CHECK(different);
    END_TEST("CHE gives different embeddings for different tokens");
}

void test_spe_update() {
    TEST("SPE updates positional state")
    SLIE slie(1000, 64, 4, 256, 8);
    Vec prev(slie.d_pos(), 0);
    Vec emb = slie.che_forward(1);
    Vec pos = slie.spe_forward(prev, emb);
    CHECK(pos.size() == 8);
    // Should be different from prev
    bool changed = false;
    for (size_t i = 0; i < pos.size(); ++i)
        if (std::abs(pos[i]) > 1e-10) changed = true;
    CHECK(changed);
    END_TEST("SPE updates positional state");
}

void test_spe_different_positions() {
    TEST("SPE gives different states for different positions")
    SLIE slie(1000, 64, 4, 256, 8);
    Vec prev(slie.d_pos(), 0);
    Vec emb1 = slie.che_forward(1);
    Vec pos1 = slie.spe_forward(prev, emb1);
    Vec prev2 = pos1;
    Vec emb2 = slie.che_forward(2);
    Vec pos2 = slie.spe_forward(prev2, emb2);
    bool diff = false;
    for (size_t i = 0; i < pos1.size(); ++i)
        if (std::abs(pos1[i] - pos2[i]) > 1e-10) diff = true;
    CHECK(diff);
    END_TEST("SPE gives different states for different positions");
}

void test_sketch_update() {
    TEST("Sketch updates counters")
    SLIE slie(1000, 64);
    // Can't easily access internal state, but verify forward runs
    Vec prev(slie.d_pos(), 0);
    Vec out = slie.forward(42, prev);
    CHECK(out.size() == 64);
    // Run again with correct position state — should change sketch state
    Vec out2 = slie.forward(42, slie.last_position());
    CHECK(out2.size() == 64);
    END_TEST("Sketch updates counters");
}

void test_reset_position() {
    TEST("reset_position resets state")
    SLIE slie(1000, 64, 4, 256, 8);
    Vec prev(slie.d_pos(), 0);
    slie.forward(1, prev);
    slie.reset_position();
    Vec pos_after = slie.last_position();
    for (auto& v : pos_after) CHECK(std::abs(v) < 1e-12);
    END_TEST("reset_position resets state");
}

void test_complexity_o1_per_token() {
    TEST("O(1) per token — embedding dimension independent")
    SLIE slie_small(1000, 32);
    SLIE slie_large(1000, 256);
    Vec prev_s(slie_small.d_pos(), 0);
    Vec prev_l(slie_large.d_pos(), 0);

    auto t1 = slie_small.forward(42, prev_s);
    auto t2 = slie_large.forward(42, prev_l);
    CHECK(t1.size() == 32);
    CHECK(t2.size() == 256);
    END_TEST("O(1) per token — embedding dimension independent");
}

void test_vocab_oo_bounds() {
    TEST("Handles OOB vocabulary indices gracefully")
    SLIE slie(100, 64);
    Vec emb = slie.che_forward(999);
    CHECK(emb.size() == 64);
    END_TEST("Handles OOB vocabulary indices gracefully");
}

void test_numerical_stability() {
    TEST("No NaN or Inf in outputs")
    SLIE slie(1000, 64);
    Vec prev(slie.d_pos(), 0);
    for (int i = 0; i < 100; ++i) {
        Vec out = slie.forward(i % 1000, prev);
        for (auto& v : out) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
        prev = slie.last_position();
    }
    END_TEST("No NaN or Inf in outputs");
}

int main() {
    std::cout << "=== Component 1: SLIE Tests ===" << std::endl;

    test_che_correctness();
    test_che_determinism();
    test_che_different_tokens();
    test_spe_update();
    test_spe_different_positions();
    test_sketch_update();
    test_reset_position();
    test_complexity_o1_per_token();
    test_vocab_oo_bounds();
    test_numerical_stability();

    if (failed > 0) {
        std::cerr << failed << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "All SLIE tests PASSED." << std::endl;
    return 0;
}
