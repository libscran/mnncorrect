#ifndef MNNCORRECT_FIND_LOCAL_CENTERS_HPP
#define MNNCORRECT_FIND_LOCAL_CENTERS_HPP

#include "utils.hpp"

#include <vector>

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Distance_>
void find_local_centers(const NeighborSet<Index_, Distance_>& neighbors, std::vector<Index_>& centers) {
    centers.clear();

    auto nobs = neighbors.size();
    for (decltype(nobs) o = 0; o < nobs; ++o) {
        const auto& values = neighbors[o];
        auto min = values.back().second;
        bool found_better = false;

        // We define a 'local center' as any observation where its distance to the
        // k-th nearest neighbor is the lowest of all of its neighbors. This means
        // that this observation lies in a local maxima of density.
        for (const auto& v : values) {
            if (neighbors[v.first].back().second < min) {
                found_better = true;
                break;
            }
        }
        if (!found_better) {
            centers.push_back(o);
        }
    }
}

}

}

#endif
