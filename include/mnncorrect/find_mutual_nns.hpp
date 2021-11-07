#ifndef MNNCORRECT_FIND_MUTUAL_NN_HPP
#define MNNCORRECT_FIND_MUTUAL_NN_HPP

#include <vector>
#include <algorithm>
#include "knncolle/knncolle.hpp"
#include "utils.hpp"

namespace mnncorrect {

template<typename Index, typename Float>
MnnPairs<Index> find_mutual_nns(const NeighborSet<Index, Float>& left, const NeighborSet<Index, Float>& right) {
    size_t nleft = left.size();
    size_t nright = right.size();

    std::vector<std::vector<Index> > neighbors_of_left(nleft);
    for (size_t l = 0; l < nleft; ++l) {
        auto& storage = neighbors_of_left[l];
        for (const auto& f : left[l]) {
            storage.push_back(f.first);
        }
        std::sort(storage.begin(), storage.end());
    }

    MnnPairs<Index> output;
    std::vector<size_t> last(nleft);
    for (size_t r = 0; r < nright; ++r) {
        const auto& mine = right[r];

        for (auto left_pair : mine) {
            auto left_neighbor = left_pair.first;
            const auto& other = neighbors_of_left[left_neighbor];
            auto& position = last[left_neighbor];
            for (; position < other.size(); ++position) {
                if (other[position] >= r) {
                    if (other[position] == r) {
                        output.left.push_back(left_neighbor);
                        output.right.push_back(r);
                    }
                    break;
                }
            }
        }
    }

    return output;

}


template<typename Index, typename Float, class Searcher>
MnnPairs<Index> find_mutual_nns(
    const Float* left, 
    const Float* right, 
    const Searcher* left_index, 
    const Searcher* right_index, 
    int k_left, 
    int k_right)
{
    int ndim = left_index->ndim();
    size_t nleft = left_index->nobs();
    size_t nright = right_index->nobs();

    NeighborSet<Index, Float> neighbors_of_left(nleft);
    #pragma omp parallel for
    for (size_t l = 0; l < nleft; ++l) {
        auto current = left + ndim * l;
        neighbors_of_left[l] = right_index->find_nearest_neighbors(current, k_left);
    }

    NeighborSet<Index, Float> neighbors_of_right(nright);
    #pragma omp parallel for
    for (size_t r = 0; r < nright; ++r) {
        auto current = right + ndim * r;
        neighbors_of_right[r] = left_index->find_nearest_neighbors(current, k_right);
    }

    return find_mutual_nns<Index, Float>(neighbors_of_left, neighbors_of_right);
}

}

#endif
