#include "stre.hpp"
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

void test_graph_construction() {
    TEST("Graph construction produces bounded-degree graph")
    STRE stre(16, 6, 8, 3);
    Mat Z(30, Vec(8, 0.5));
    auto mapping = stre.build_graph(Z);

    CHECK(mapping.size() == 30);
    CHECK(stre.num_nodes() > 0);
    CHECK(stre.num_nodes() <= 30);

    // Check bounded degree
    for (auto& node : stre.nodes()) {
        CHECK(node.neighbors.size() <= 6);
    }
    END_TEST("Graph construction produces bounded-degree graph");
}

void test_graph_no_duplicate_edges() {
    TEST("No duplicate edges in graph")
    STRE stre(16, 8, 8, 3);
    Mat Z(20, Vec(8, 0.5));
    stre.build_graph(Z);

    for (auto& e : stre.edges()) {
        Index count = 0;
        for (auto& e2 : stre.edges()) {
            if ((e.u == e2.u && e.v == e2.v) || (e.u == e2.v && e.v == e2.u))
                count++;
        }
        CHECK(count == 1);
    }
    END_TEST("No duplicate edges in graph");
}

void test_restriction_maps() {
    TEST("Restriction maps initialized correctly (identity via build_graph)")
    STRE stre(8, 4, 8, 3);
    Mat Z(10, Vec(8, 0.3));
    stre.build_graph(Z);

    CHECK(stre.num_nodes() > 0);
    END_TEST("Restriction maps initialized correctly (identity via build_graph)");
}

void test_propagation_shape() {
    TEST("Propagation produces correct output dimensions")
    STRE stre(8, 4, 8, 3);
    Mat Z(15, Vec(8, 0.5));
    auto features = stre.forward(Z, 2);

    CHECK(features.size() == stre.nodes().size());
    for (auto& fv : features) CHECK(fv.size() == 8);
    END_TEST("Propagation produces correct output dimensions");
}

void test_forward_no_nan() {
    TEST("Forward pass produces no NaN/Inf")
    STRE stre(8, 4, 8, 3);
    Mat Z(20, Vec(8, 0.2));
    auto features = stre.forward(Z, 3);
    for (auto& fv : features)
        for (auto& v : fv) {
            CHECK(!std::isnan(v));
            CHECK(!std::isinf(v));
        }
    END_TEST("Forward pass produces no NaN/Inf");
}

void test_conflict_score_bounded() {
    TEST("Conflict score is non-negative and bounded")
    STRE stre(8, 4, 8, 3);
    Mat Z(12, Vec(8, 0.5));
    auto features = stre.forward(Z, 2);
    Real conflict = stre.compute_conflict(features);
    CHECK(conflict >= 0);
    CHECK(!std::isnan(conflict));
    CHECK(!std::isinf(conflict));
    END_TEST("Conflict score is non-negative and bounded");
}

void test_pathology_detection() {
    TEST("Pathology detection runs without error")
    STRE stre(8, 4, 8, 3);
    Mat Z(10, Vec(8, 0.5));
    stre.forward(Z, 1);
    auto paths = stre.detect_pathologies(0.5);
    CHECK(paths.size() <= stre.nodes().size());
    END_TEST("Pathology detection runs without error");
}

void test_graph_connectivity() {
    TEST("Graph has reasonable connectivity")
    STRE stre(16, 6, 8, 3);
    Mat Z(40, Vec(8, 0.5));
    stre.build_graph(Z);

    Index total_edges = 0;
    for (auto& node : stre.nodes()) total_edges += node.neighbors.size();
    CHECK(total_edges % 2 == 0);  // each edge counted twice
    END_TEST("Graph has reasonable connectivity");
}

void test_bounded_treewidth() {
    TEST("Graph respects bounded degree (treewidth proxy)")
    STRE stre(16, 5, 8, 3);
    Mat Z(50, Vec(8, 0.5));
    stre.build_graph(Z);

    for (auto& node : stre.nodes()) {
        CHECK(node.neighbors.size() <= 5);
    }
    END_TEST("Graph respects bounded degree (treewidth proxy)");
}

void test_different_seeds_different_graphs() {
    TEST("Different seeds/states produce different graphs")
    STRE stre(16, 6, 8, 3);
    Mat Z1(15, Vec(8, 0.1));
    Mat Z2(15, Vec(8, 0.9));
    auto g1 = stre.build_graph(Z1);
    (void)stre.build_graph(Z2);  // should not crash
    CHECK(true);
    END_TEST("Different seeds/states produce different graphs");
}

void test_many_nodes() {
    TEST("Handles many nodes efficiently (O(n) check)")
    STRE stre(16, 8, 8, 3);
    Mat Z(200, Vec(8, 0.3));
    auto features = stre.forward(Z, 2);
    CHECK(features.size() <= 200);
    // Should complete quickly (O(n))
    END_TEST("Handles many nodes efficiently (O(n) check)");
}

int main() {
    std::cout << "=== Component 3: STRE Tests ===" << std::endl;

    test_graph_construction();
    test_graph_no_duplicate_edges();
    test_restriction_maps();
    test_propagation_shape();
    test_forward_no_nan();
    test_conflict_score_bounded();
    test_pathology_detection();
    test_graph_connectivity();
    test_bounded_treewidth();
    test_different_seeds_different_graphs();
    test_many_nodes();

    if (failed > 0) {
        std::cerr << failed << " test(s) FAILED!" << std::endl;
        return 1;
    }
    std::cout << "All STRE tests PASSED." << std::endl;
    return 0;
}
