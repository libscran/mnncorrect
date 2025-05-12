#ifndef UTILS_H
#define UTILS_H

#include "knncolle/knncolle.hpp"
#include "mnncorrect/utils.hpp"

template<typename Index_, typename Float_>
std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > subset_and_index(
    std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* data,
    const knncolle::Builder<Index_, Float_, Float_>& builder,
    std::vector<Float_>& buffer)
{ 
    auto num_obs = ids.size();
    buffer.resize(static_cast<std::size_t>(num_obs) * num_dim);
    for (decltype(num_obs) e = 0; e < num_obs; ++e) {
        std::copy_n(
            data + static_cast<std::size_t>(ids[e]) * num_dim,
            num_dim,
            buffer.begin() + static_cast<std::size_t>(e) * num_dim
        );
    }
    return builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_obs, buffer.data()));
}

template<typename Index_, typename Float_>
void find_neighbors(
    std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* data,
    const knncolle::Prebuilt<Index_, Float_, Float_>& index,
    const std::vector<Index_>& index_ids,
    int num_neighbors,
    mnncorrect::internal::NeighborSet<Index_, Float_>& output)
{ 
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
