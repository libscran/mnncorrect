#ifndef UTILS_H
#define UTILS_H

#include "knncolle/knncolle.hpp"
#include "mnncorrect/utils.hpp"

#include <algorithm>
#include <numeric>
#include <vector>
#include <cstddef>

template<typename Index_, typename Float_>
void find_neighbors(
    std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* data,
    const knncolle::Prebuilt<Index_, Float_, Float_>& index,
    const std::vector<Index_>& index_ids,
    int num_neighbors,
    mnncorrect::NeighborSet<Index_, Float_>& output
) { 
    std::vector<int> indices;
    std::vector<double> distances;
    auto searcher = index.initialize();
    for (auto i : ids) {
        searcher->search(data + static_cast<std::size_t>(i) * num_dim, num_neighbors, &indices, &distances);
        auto found = indices.size();
        for (decltype(found) j = 0; j < found; ++j) {
            output[i].emplace_back(index_ids[indices[j]], distances[j]);
        }
    }
}

#endif
