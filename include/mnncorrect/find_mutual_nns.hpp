#ifndef MNNCORRECT_FIND_MUTUAL_NN_HPP
#define MNNCORRECT_FIND_MUTUAL_NN_HPP

#include <vector>
#include <algorithm>
#include "knncolle/knncolle.hpp"
#include "utils.hpp"

namespace mnncorrect {

template<typename Index>
struct MnnPairs {
    MnnPairs(size_t nright=0) : matches(nright) {}
    std::vector<std::vector<Index> > matches;
    size_t num_pairs = 0;
};

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

    MnnPairs<Index> output(nright);
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
                        output.matches[r].push_back(left_neighbor);
                        ++output.num_pairs;
                    }
                    break;
                }
            }
        }
    }

    return output;
}

template<typename Index, typename Dist>
NeighborSet<Index, Dist> find_nns(size_t n, const Dist* query, const knncolle::Base<Index, Dist>* index, int k) {
    NeighborSet<Index, Dist> output(n);
    int ndim = index->ndim();
    #pragma omp parallel for
    for (size_t l = 0; l < n; ++l) {
        output[l] = index->find_nearest_neighbors(query + ndim * l, k);
    }
    return output;
}

}

#endif
