#ifndef MNNCORRECT_FIND_MUTUAL_NN_HPP
#define MNNCORRECT_FIND_MUTUAL_NN_HPP

#include <vector>
#include <deque>
#include <algorithm>
#include "knncolle/knncolle.hpp"

namespace mnncorrect {

struct MnnPairs {
    std::deque<size_t> left, right;
};

template<typename T, class Searcher>
MnnPairs find_mutual_nns(const T* left, const T* right, const Searcher* left_index, const Searcher* right_index, int k_left, int k_right) {
    int ndim = left_index->ndim();
    size_t nleft = left_index->nobs();
    size_t nright = right_index->nobs();

    std::vector<std::vector<size_t> > neighbors_of_left(nleft);
    #pragma omp for parallel
    for (size_t l = 0; l < nleft; ++l) {
        auto& storage = neighbors_of_left[l]; 
        storage.reserve(k_left);

        auto current = left + ndim * l;
        auto found = right_index->find_nearest_neighbors(current, k_left);
        for (const auto& f : found) {
            storage.push_back(f.first);
        }
        std::sort(storage.begin(), storage.end());
    }

    std::vector<std::vector<size_t> > neighbors_of_right(nright);
    #pragma omp for parallel
    for (size_t r = 0; r < nright; ++r) {
        auto& storage = neighbors_of_right[r]; 
        storage.reserve(k_right);

        auto current = right + ndim * r;
        auto found = left_index->find_nearest_neighbors(current, k_right);
        for (const auto& f : found) {
            storage.push_back(f.first);
        }
    }

    // Identifying the mutual neighbors.
    MnnPairs output;
    std::vector<size_t> last(nleft);
    for (size_t r = 0; r < nright; ++r) {
        const auto& mine = neighbors_of_right[r];

        for (auto left_neighbor : mine) {
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

}

#endif
