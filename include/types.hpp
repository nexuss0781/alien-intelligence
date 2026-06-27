#pragma once
#include <vector>
#include <complex>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <numeric>
#include <iostream>
#include <limits>
#include <memory>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ai2 {

using Real = double;
using Vec = std::vector<Real>;
using Mat = std::vector<Vec>;
using CVec = std::vector<std::complex<Real>>;
using CMat = std::vector<CVec>;
using Index = std::size_t;

constexpr Real PI = 3.14159265358979323846;
constexpr Real EPS = 1e-12;

inline Real sigmoid(Real x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline Vec softmax(const Vec& x) {
    Vec y(x.size());
    Real m = *std::max_element(x.begin(), x.end());
    Real s = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:s)
#endif
    for (Index i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - m);
        s += y[i];
    }
    if (s < EPS) s = EPS;
    for (auto& v : y) v /= s;
    return y;
}

inline Real dot(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Real r = 0;
    for (Index i = 0; i < a.size(); ++i) r += a[i] * b[i];
    return r;
}

inline Real norm2(const Vec& a) {
    Real r = 0;
    for (auto& v : a) r += v * v;
    return r;
}

inline Vec axpy(Real alpha, const Vec& x, const Vec& y) {
    assert(x.size() == y.size());
    Vec r(x.size());
    for (Index i = 0; i < x.size(); ++i) r[i] = alpha * x[i] + y[i];
    return r;
}

inline Vec mat_vec(const Mat& A, const Vec& x) {
    Index m = A.size();
    assert(m > 0 && A[0].size() == x.size());
    Vec y(m, 0);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (Index i = 0; i < m; ++i)
        for (Index j = 0; j < x.size(); ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

inline Mat mat_mul(const Mat& A, const Mat& B) {
    Index m = A.size(), n = A[0].size(), p = B[0].size();
    assert(n == B.size());
    Mat C(m, Vec(p, 0));
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (Index i = 0; i < m; ++i)
        for (Index k = 0; k < n; ++k)
            for (Index j = 0; j < p; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

inline Vec elem_mul(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec r(a.size());
    for (Index i = 0; i < a.size(); ++i) r[i] = a[i] * b[i];
    return r;
}

inline Vec elem_add(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec r(a.size());
    for (Index i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

inline Vec scale(Real s, const Vec& a) {
    Vec r(a.size());
    for (Index i = 0; i < a.size(); ++i) r[i] = s * a[i];
    return r;
}

inline Real mean(const Vec& a) {
    return std::accumulate(a.begin(), a.end(), 0.0) / a.size();
}

inline Real variance(const Vec& a) {
    Real m = mean(a);
    Real v = 0;
    for (auto& x : a) v += (x - m) * (x - m);
    return v / a.size();
}

// Universal hash function family (Carter-Wegman)
class UniversalHash {
    Index a_, b_, m_, p_;
public:
    UniversalHash(Index seed, Index modulus)
        : m_(modulus), p_(2147483647) {
        std::mt19937_64 rng(seed);
        a_ = (rng() % (p_ - 1)) + 1;
        b_ = rng() % p_;
    }
    Index operator()(Index x) const {
        Index h = (a_ * x + b_) % p_;
        return h % m_;
    }
};

// MurmurHash3-inspired fast hash for LSH
inline Index hash_value(Index x, Index seed) {
    x ^= seed;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

// Simple random projection for LSH
class LSHFunction {
    Vec a_;
    Real b_;
public:
    LSHFunction(Index dim, Index seed) : a_(dim) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<Real> norm(0, 1);
        for (auto& v : a_) v = norm(rng);
        std::uniform_real_distribution<Real> uni(0, 1);
        b_ = uni(rng);
    }
    Index operator()(const Vec& x, Index num_buckets) const {
        return static_cast<Index>(std::abs(dot(a_, x) + b_)) % num_buckets;
    }
};

} // namespace ai2
