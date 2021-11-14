#ifndef MNNCORRECT_UTILS_HPP
#define MNNCORRECT_UTILS_HPP

#include <deque>
#include <vector>
#include <limits>
#include <type_traits>
#include <set>
#include <cmath>

namespace mnncorrect {

template<typename Index>
struct MnnPairs {
    size_t size() const { 
        return left.size(); 
    }

    void clear() {
        left.clear();
        right.clear();
        return;
    }

    std::deque<Index> left, right;
};

template<typename Float>
Float normalize_vector(int ndim, Float* ptr) {
    Float l2norm = 0;
    for (int d = 0; d < ndim; ++d) {
        l2norm += ptr[d] * ptr[d];
    }

    if (l2norm) {
        l2norm = std::sqrt(l2norm);
        for (int d = 0; d < ndim; ++d) {
            ptr[d] /= l2norm;
        }
    }

    return l2norm;
}

template<typename Index, typename Dist>
using NeighborSet = std::vector<std::vector<std::pair<Index, Dist> > >;

template<class Vector>
auto unique(const Vector& input) {
    typedef typename std::remove_const<typename std::remove_reference<decltype(*input.begin())>::type>::type Value;
    std::set<Value> collected(input.begin(), input.end());
    return std::vector<Value>(collected.begin(), collected.end());
}

template<typename Index>
std::vector<Index> invert_index(size_t n, const std::vector<Index>& uniq, Index placeholder = 0) {
    std::vector<Index> output(n, placeholder);
    for (size_t u = 0; u < uniq.size(); ++u) {
        output[uniq[u]] = u;        
    }
    return output;
}

template<typename Float>
Float median(size_t n, Float* ptr) {
    if (!n) {
        return std::numeric_limits<Float>::quiet_NaN();
    }
    size_t half = n / 2;
    bool is_even = n % 2 == 0;

    std::nth_element(ptr, ptr + half, ptr + n);
    Float mid = *(ptr + half);
    if (!is_even) {
        return mid;
    }

    std::nth_element(ptr, ptr + half - 1, ptr + n);
    return (mid + *(ptr + half - 1)) / 2;
}

constexpr double mad2sigma = 1.4826;

template<typename Float>
void median_distance_from_center(int ndim, size_t nobs, const Float* data, size_t nref, const Float* centers, const Index* assignments, Float* output) {
    std::vector<std::vector<Float> > collected(nref);
    for (size_t o = 0; o < nobs; ++o) {
        auto rptr = data + o * ndim;
        auto cptr = centers + assignments[o] * ndim;

        Float dist = 0;
        for (int d = 0; d < ndim; ++d) {
            Float diff = rptr[d] - cptr[d];
            dist += diff * diff;
        }

        collected[assignments[o]].push_back(dist);
    }

    #pragma omp parallel for
    for (size_t r = 0; r < nref; ++r) {
        auto& current = collected[r];
        if (current.size()) {
            output[r] = median(current.size(), current.data());
        } else {
            output[r] = 0; // shouldn't be possible, but whatever, just in case.
        }
    }

    return;
}

}


#endif
