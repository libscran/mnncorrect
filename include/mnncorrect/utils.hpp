#ifndef MNNCORRECT_UTILS_HPP
#define MNNCORRECT_UTILS_HPP

#include <vector>
#include <limits>
#include <type_traits>
#include <algorithm>

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Distance_>
using NeighborSet = std::vector<std::vector<std::pair<Index_, Distance_> > >;

template<typename Index_, typename Distance_>
std::vector<std::vector<Index_> > invert_neighbors(size_t n, const NeighborSet<Index_, Distance_>& neighbors, Distance_ limit) {
    std::vector<std::vector<Index_> > output(n);
    const Index_ num_neighbors = neighbors.size();
    for (Index_ i = 0; i < num_neighbors; ++i) {
        for (const auto& x : neighbors[i]) {
            if (x.second <= limit) {
                output[x.first].push_back(i);
            }
        }
    }
    return output;
}

template<typename Index_>
std::vector<Index_> invert_indices(size_t n, const std::vector<Index_>& uniq) {
    std::vector<Index_> output(n, static_cast<Index_>(-1)); // we don't check this anyway.
    Index_ num_uniq = uniq.size();
    for (Index_ u = 0; u < num_uniq; ++u) {
        output[uniq[u]] = u;
    }
    return output;
}

template<typename Float_>
Float_ median(size_t n, Float_* ptr) {
    if (!n) {
        return std::numeric_limits<Float_>::quiet_NaN();
    }
    size_t half = n / 2;
    bool is_even = n % 2 == 0;

    std::nth_element(ptr, ptr + half, ptr + n);
    Float_ mid = *(ptr + half);
    if (!is_even) {
        return mid;
    }

    std::nth_element(ptr, ptr + half - 1, ptr + n);
    return (mid + *(ptr + half - 1)) / 2;
}

constexpr double mad2sigma = 1.4826;

}

}

#endif
