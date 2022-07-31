#ifndef MNNCORRECT_FUSE_NN_RESULTS_HPP
#define MNNCORRECT_FUSE_NN_RESULTS_HPP

#include <vector>
#include "knncolle/knncolle.hpp"
#include "utils.hpp"

namespace mnncorrect {

template<typename Index, typename Dist>
NeighborSet<Index, Dist> quick_find_nns(size_t n, const Dist* query, const knncolle::Base<Index, Dist>* index, int k, int nthreads) {
    NeighborSet<Index, Dist> output(n);
    int ndim = index->ndim();

#ifndef MNNCORRECT_CUSTOM_PARALLEL
    #pragma omp parallel for num_threads(nthreads)
    for (size_t l = 0; l < n; ++l) {
#else
    MNNCORRECT_CUSTOM_PARALLEL(n, [&](size_t start, size_t end) -> void {
    for (size_t l = start; l < end; ++l) {
#endif

        output[l] = index->find_nearest_neighbors(query + ndim * l, k);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
    }
#else
    }
    }, nthreads);
#endif

    return output;
}

template<typename Index, typename Dist>
void fuse_nn_results(std::vector<std::pair<Index, Dist> >& base, const std::vector<std::pair<Index, Dist> >& alt, size_t num_neighbors, Index offset = 0) {
    auto last = base;
    base.clear();
    auto lIt = last.begin();
    auto aIt = alt.begin();
    while (base.size() < num_neighbors) {
        if (lIt != last.end() && aIt != alt.end()) {
            if (lIt->second > aIt->second) {
                base.push_back(*aIt);
                base.back().first += offset;
                ++aIt;
            } else {
                base.push_back(*lIt);
                ++lIt;
            }
        } else if (lIt != last.end()) {
            base.push_back(*lIt);
            ++lIt;
        } else if (aIt != alt.end()) {
            base.push_back(*aIt);
            base.back().first += offset;
            ++aIt;
        } else {
            break;
        }
    }
}

}

#endif

