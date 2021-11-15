#ifndef MNNCORRECT_FIND_MUTUAL_NN_HPP
#define MNNCORRECT_FIND_MUTUAL_NN_HPP

#include <vector>
#include <deque>
#include <algorithm>
#include "utils.hpp"

namespace mnncorrect {

template<typename Index>
struct MnnPairs {
    size_t size() const { 
        return ref.size(); 
    }

    void clear() {
        ref.clear();
        target.clear();
        return;
    }

    std::deque<Index> ref, target;
};

template<typename Index, typename Float>
MnnPairs<Index> find_mutual_nns(const NeighborSet<Index, Float>& ref, const Index* target) {
    MnnPairs<Index> output;
    for (size_t l = 0; l < ref.size(); ++l) {
        for (const auto& f : ref[l]) {
            if (static_cast<Index>(l) == target[f.first]) {
                output.ref.push_back(l);
                output.target.push_back(f.first);
            }
        }
    }
    return output;
}

template<typename Index, typename Float, class Searcher>
MnnPairs<Index> find_mutual_nns(const Float* ref, const Float* target, const Searcher* ref_index, const Searcher* target_index, int k, NeighborSet<Index, Float>& ref_neighbors, Index* target_closest) {
    int ndim = ref_index->ndim();
    size_t nref = ref_index->nobs();
    size_t ntarget = target_index->nobs();

    #pragma omp parallel for
    for (size_t r = 0; r < ntarget; ++r) {
        auto current = target + ndim * r;
        auto found = ref_index->find_nearest_neighbors(current, 1);
        target_closest[r] = found.front().first;
    }

    #pragma omp parallel for
    for (size_t l = 0; l < nref; ++l) {
        auto current = ref + ndim * l;
        ref_neighbors[l] = target_index->find_nearest_neighbors(current, k);
    }

    return find_mutual_nns(ref_neighbors, target_closest);
}

}

#endif
