#ifndef MNNCORRECT_FUSE_NN_RESULTS_HPP
#define MNNCORRECT_FUSE_NN_RESULTS_HPP

#include <vector>
#include "knncolle/knncolle.hpp"
#include "utils.hpp"

namespace mnncorrect {

template<typename Index, typename Dist>
NeighborSet<Index, Dist> quick_find_nns(size_t n, const Dist* query, const knncolle::Base<Index, Dist>* index, int k) {
    NeighborSet<Index, Dist> output(n);
    int ndim = index->ndim();
    #pragma omp parallel for
    for (size_t l = 0; l < n; ++l) {
        output[l] = index->find_nearest_neighbors(query + ndim * l, k);
    }
    return output;
}

template<typename Index, typename Dist>
void fuse_nn_results(std::vector<std::pair<Index, Dist> >& base, const std::vector<std::pair<Index, Dist> >& alt, size_t num_neighbors, Index offset = 0) {
    auto last = base;
    base.clear();
    auto lIt = last.begin(), aIt = alt.begin();
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

