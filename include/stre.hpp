#pragma once
#include "types.hpp"
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>

namespace ai2 {

// Sheaf-Theoretic Reasoning Engine (STRE) — Component 3
// Models reasoning as a sheaf over a bounded-degree graph.
// Provides O(n) propagation and cohomological consistency checking.
class STRE {
public:
    // d_node: node feature dimension (d_r)
    // max_degree: maximum node degree Δ = O(1)
    // lsh_dim: dimension for LSH clustering
    // n_lsh_tables: number of LSH tables for robust bucketing
    STRE(Index d_node = 64, Index max_degree = 8,
         Index lsh_dim = 16, Index n_lsh_tables = 4);

    // --------------- Graph Construction ---------------

    struct Node {
        Index id;
        Vec features;
        std::vector<Index> neighbors;
    };

    struct Edge {
        Index u, v;
    };

    // Build reasoning graph from encoded sequence Z ∈ R^{n × d_model}
    // Uses LSH clustering for node assignment and locality-based edges
    // Complexity: O(n) with O(1) per node
    // Returns node indices for each input position
    std::vector<Index> build_graph(const Mat& Z);

    // Access graph
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<Edge>& edges() const { return edges_; }

    // --------------- Sheaf Neural Network ---------------

    // Restriction maps F_{u->v}: linear transformations
    // Stored per directed edge: F_{v->u} ∈ R^{d_node × d_node}
    void init_restriction_maps(Index seed = 42);

    // Sheaf Laplacian: Δ_F(x)_v = Σ_{u~v} F_{v->u}^T (F_{v->u} x_v - F_{u->v} x_u)
    // O(|E|) = O(n) since Δ = O(1)
    Vec sheaf_laplacian(const Vec& node_features, Index node_idx) const;
    Mat sheaf_laplacian_all(const std::vector<Vec>& features) const;

    // One layer of sheaf propagation:
    // x_v^{(l+1)} = σ(W^{(l)} x_v^{(l)} + Σ_{u~v} F_{v->u}^{(l)} x_u^{(l)})
    // O(n)
    std::vector<Vec> propagate_layer(const std::vector<Vec>& features,
                                     const Mat& W, Index layer) const;

    // Full forward pass through L layers: O(L · n) = O(n)
    std::vector<Vec> forward(const Mat& Z, Index n_layers = 3);

    // --------------- Pathological Consistency ---------------

    // conflict(G) = ||Δ_F x||^2 — cohomological consistency detector
    // Higher values indicate local inconsistencies
    Real compute_conflict(const std::vector<Vec>& features) const;

    // Detect and resolve pathologies via specialist MLP
    // Returns positions flagged as pathological
    std::vector<Index> detect_pathologies(Real threshold = 1.0) const;

    // Accessors
    Index d_node() const { return d_node_; }
    Index num_nodes() const { return nodes_.size(); }

private:
    Index d_node_;
    Index max_degree_;
    Index lsh_dim_;
    Index n_lsh_tables_;

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;

    // LSH functions for graph construction
    std::vector<LSHFunction> lsh_funcs_;

    // Restriction maps F_{v->u} per directed edge
    // Indexed by (u, v) pairs stored in map
    // Using flat storage: for each node v, store maps from each neighbor u
    std::vector<std::vector<Mat>> restriction_maps_; // [node_idx][neighbor_idx] ∈ R^{d_node × d_node}

    // Position -> node mapping (for position i, which node does it belong to?)
    std::vector<Index> pos_to_node_;

    // Specialist MLP for pathology resolution (2-layer: d_node -> 4*d_node -> d_node)
    Mat path_W1_, path_W2_;
    Vec path_b1_, path_b2_;
};

} // namespace ai2
