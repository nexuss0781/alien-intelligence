#include "stre.hpp"
#include <random>
#include <map>
#include <queue>

namespace ai2 {

STRE::STRE(Index d_node, Index max_degree,
           Index lsh_dim, Index n_lsh_tables)
    : d_node_(d_node), max_degree_(max_degree),
      lsh_dim_(lsh_dim), n_lsh_tables_(n_lsh_tables)
{
    std::mt19937_64 rng(31415);

    // Initialize LSH functions for graph construction
    lsh_funcs_.reserve(n_lsh_tables_);
    for (Index t = 0; t < n_lsh_tables_; ++t) {
        lsh_funcs_.emplace_back(lsh_dim_, t * 7777 + 11);
    }

    // Initialize per-layer propagation weights (stored once, not re-randomized)
    Real scale = std::sqrt(2.0 / (d_node_ + d_node_));
    for (Index l = 0; l < 6; ++l) {
        Mat W(d_node_, Vec(d_node_));
        for (auto& row : W)
            for (auto& v : row)
                v = std::normal_distribution<Real>(0, scale)(rng);
        layer_W_.push_back(W);
    }

    // Initialize specialist MLP
    Real scale1 = std::sqrt(2.0 / (d_node_ + 4 * d_node_));
    Real scale2 = std::sqrt(2.0 / (4 * d_node_ + d_node_));

    path_W1_.resize(4 * d_node_, Vec(d_node_));
    for (auto& row : path_W1_)
        for (auto& v : row)
            v = std::normal_distribution<Real>(0, scale1)(rng);

    path_b1_.resize(4 * d_node_, 0);

    path_W2_.resize(d_node_, Vec(4 * d_node_));
    for (auto& row : path_W2_)
        for (auto& v : row)
            v = std::normal_distribution<Real>(0, scale2)(rng);

    path_b2_.resize(d_node_, 0);
}

std::vector<Index> STRE::build_graph(const Mat& Z) {
    Index n = Z.size();
    pos_to_node_.resize(n);
    nodes_.clear();
    edges_.clear();

    std::map<std::vector<Index>, std::vector<Index>> bucket_groups;

    for (Index i = 0; i < n; ++i) {
        std::vector<Index> sig(n_lsh_tables_);
        for (Index t = 0; t < n_lsh_tables_; ++t) {
            Vec proj(lsh_dim_);
            for (Index j = 0; j < lsh_dim_; ++j) {
                proj[j] = Z[i][j % Z[i].size()];
            }
            sig[t] = lsh_funcs_[t](proj, 256);
        }
        bucket_groups[sig].push_back(i);
    }

    Index node_id = 0;
    for (auto& [sig, positions] : bucket_groups) {
        (void)sig;
        Node node;
        node.id = node_id;
        node.features.resize(d_node_, 0);

        for (Index pos : positions) {
            const Vec& z = Z[pos % Z.size()];
            for (Index j = 0; j < std::min(d_node_, (Index)z.size()); ++j) {
                node.features[j] += z[j];
            }
        }
        Real inv = 1.0 / std::max(positions.size(), (size_t)1);
        for (auto& f : node.features) f *= inv;

        for (Index pos : positions) {
            pos_to_node_[pos] = node_id;
        }

        nodes_.push_back(std::move(node));
        node_id++;
    }

    // Build edges with bounded degree
    for (Index i = 0; i < n; ++i) {
        Index u = pos_to_node_[i];
        for (Index j = i + 1; j < std::min(n, i + max_degree_ + 1); ++j) {
            Index v = pos_to_node_[j];
            if (u != v) {
                bool exists = false;
                for (auto& nb : nodes_[u].neighbors) {
                    if (nb == v) { exists = true; break; }
                }
                if (!exists && nodes_[u].neighbors.size() < max_degree_
                          && nodes_[v].neighbors.size() < max_degree_) {
                    nodes_[u].neighbors.push_back(v);
                    nodes_[v].neighbors.push_back(u);
                    edges_.push_back({u, v});
                }
            }
        }
    }

    // Build identity restriction maps (stable, deterministic)
    restriction_maps_.resize(nodes_.size());
    for (Index v = 0; v < nodes_.size(); ++v) {
        restriction_maps_[v].resize(nodes_[v].neighbors.size());
        for (Index ni = 0; ni < nodes_[v].neighbors.size(); ++ni) {
            Mat I(d_node_, Vec(d_node_, 0));
            for (Index i = 0; i < d_node_; ++i) I[i][i] = 1.0;
            restriction_maps_[v][ni] = I;
        }
    }

    return pos_to_node_;
}

Vec STRE::sheaf_laplacian(const Vec& node_features, Index node_idx) const {
    Vec lap(d_node_, 0);
    Index v = node_idx;
    const auto& nbs = nodes_[v].neighbors;
    for (Index ni = 0; ni < nbs.size(); ++ni) {
        Index u = nbs[ni];
        const Mat& F_vu = restriction_maps_[v][ni];
        Index u_ni = 0;
        for (; u_ni < nodes_[u].neighbors.size(); ++u_ni) {
            if (nodes_[u].neighbors[u_ni] == v) break;
        }
        if (u_ni < nodes_[u].neighbors.size()) {
            const Mat& F_uv = restriction_maps_[u][u_ni];
            Vec F_vu_xv = mat_vec(F_vu, node_features);
            Vec F_uv_xu = mat_vec(F_uv, node_features);
            Vec diff = axpy(1, F_vu_xv, scale(-1, F_uv_xu));
            for (Index i = 0; i < d_node_; ++i)
                for (Index j = 0; j < d_node_; ++j)
                    lap[i] += F_vu[j][i] * diff[j];
        }
    }
    return lap;
}

Mat STRE::sheaf_laplacian_all(const std::vector<Vec>& features) const {
    Index n = features.size();
    Mat result(n, Vec(d_node_, 0));

    std::cout << "    [debug stre] sheaf_laplacian_all: n=" << n
              << " nodes_.size()=" << nodes_.size()
              << " rest_maps_.size()=" << restriction_maps_.size()
              << " d_node=" << d_node_
              << " first_feat[0]=" << (n > 0 && features[0].size() > 0 ? features[0][0] : -999)
              << " result[0][0]=" << (result.size() > 0 && result[0].size() > 0 ? result[0][0] : -999)
              << std::endl;

    for (Index v = 0; v < n; ++v) {
        const auto& nbs = nodes_[v].neighbors;
        for (Index ni = 0; ni < nbs.size(); ++ni) {
            Index u = nbs[ni];
            const Mat& F_vu = restriction_maps_[v][ni];

            Index u_ni = 0;
            for (; u_ni < nodes_[u].neighbors.size(); ++u_ni) {
                if (nodes_[u].neighbors[u_ni] == v) break;
            }

            if (u_ni < nodes_[u].neighbors.size()) {
                const Mat& F_uv = restriction_maps_[u][u_ni];

                Vec F_vu_xv = mat_vec(F_vu, features[v]);
                Vec F_uv_xu = mat_vec(F_uv, features[u]);

                Vec diff = axpy(1, F_vu_xv, scale(-1, F_uv_xu));

                for (Index i = 0; i < d_node_; ++i) {
                    for (Index j = 0; j < d_node_; ++j) {
                        result[v][i] += F_vu[j][i] * diff[j];
                    }
                }
            }
        }
    }
    return result;
}

std::vector<Vec> STRE::propagate_layer(const std::vector<Vec>& features,
                                         const Mat& W, Index /*layer*/) const
{
    Index n = features.size();
    std::vector<Vec> next(n, Vec(d_node_, 0));

    for (Index v = 0; v < n; ++v) {
        Vec agg = mat_vec(W, features[v]);

        const auto& nbs = nodes_[v].neighbors;
        for (Index ni = 0; ni < nbs.size(); ++ni) {
            Index u = nbs[ni];
            const Mat& F_vu = restriction_maps_[v][ni];

            Vec msg = mat_vec(F_vu, features[u]);
            agg = elem_add(agg, msg);
        }

        for (Index i = 0; i < d_node_; ++i) {
            agg[i] = std::tanh(agg[i]);
        }
        next[v] = agg;
    }
    return next;
}

std::vector<Vec> STRE::forward(const Mat& Z, Index n_layers) {
    // Build graph from input (creates nodes, edges, identity restriction maps)
    build_graph(Z);

    Index n_nodes = nodes_.size();

    // Initialize node features from Z (pooled per node)
    std::vector<Vec> features(n_nodes);
    for (Index v = 0; v < n_nodes; ++v) {
        features[v] = nodes_[v].features;
    }

    // Propagate through stored layer weights (stable, deterministic)
    for (Index l = 0; l < n_layers && l < layer_W_.size(); ++l) {
        features = propagate_layer(features, layer_W_[l], l);
    }

    return features;
}

Real STRE::compute_conflict(const std::vector<Vec>& features) const {
    Mat lap = sheaf_laplacian_all(features);
    Real conflict = 0;
    for (const auto& row : lap) {
        conflict += norm2(row);
    }
    return conflict / std::max(features.size(), (size_t)1);
}

std::vector<Index> STRE::detect_pathologies(Real /*threshold*/) const {
    std::vector<Index> pathological;
    for (Index v = 0; v < nodes_.size(); ++v) {
        Real fnorm = norm2(nodes_[v].features);
        if (fnorm < 0.01 || nodes_[v].neighbors.size() >= max_degree_) {
            pathological.push_back(v);
        }
    }
    return pathological;
}

} // namespace ai2
