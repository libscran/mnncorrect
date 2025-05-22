#ifndef MNNCORRECT_FIND_BATCH_NEIGHBORS_HPP
#define MNNCORRECT_FIND_BATCH_NEIGHBORS_HPP

#include <vector>
#include <utility>
#include <cstddef>

#include "fuse_nn_results.hpp"
#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
void fill_pair_vector(const std::vector<Index_>& indices, const std::vector<Float_>& distances, std::vector<std::pair<Index_, Float_> >& output) {
    auto found = indices.size();
    output.clear();
    output.reserve(found);
    for (decltype(found) i = 0; i < found; ++i) {
        output.emplace_back(indices[i], distances[i]);
    }
}

template<typename Index_, class GetId_, typename Float_>
void find_batch_neighbors(
    std::size_t num_dim,
    Index_ num_obs,
    GetId_ get_data_id,
    const Float_* data,
    const BatchInfo<Index_, Float_>& batch,
    int num_neighbors,
    bool fuse_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    parallelize(num_threads, num_obs, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        std::vector<std::pair<Index_, Float_> > fuse_buffer1, fuse_buffer2;

        auto searcher = batch.index->initialize();
        auto capped_k = knncolle::cap_k_query(num_neighbors, batch.index->num_observations());

        for (Index_ l = start, end = start + length; l < end; ++l) {
            auto k = get_data_id(l);
            auto ptr = data + static_cast<std::size_t>(k) * num_dim;
            searcher->search(ptr, capped_k, &indices, &distances);
            for (auto& i : indices) {
                i += batch.offset;
            }

            auto& curnn = output[k];
            if (!fuse_neighbors) {
                fill_pair_vector(indices, distances, curnn);
            } else {
                fuse_buffer1.swap(curnn);
                fill_pair_vector(indices, distances, fuse_buffer2);
                fuse_nn_results(fuse_buffer1, fuse_buffer2, num_neighbors, curnn);
            }
        }

        for (const auto& extra : batch.extras) {
            auto searcher = extra.index->initialize();
            auto capped_k = knncolle::cap_k_query(num_neighbors, extra.index->num_observations());

            for (Index_ l = start, end = start + length; l < end; ++l) {
                auto k = get_data_id(l);
                auto ptr = data + static_cast<std::size_t>(k) * num_dim;
                searcher->search(ptr, capped_k, &indices, &distances);
                for (auto& i : indices) {
                    i = extra.ids[i];
                }
                auto& curnn = output[k];
                fuse_buffer1.swap(curnn);
                fill_pair_vector(indices, distances, fuse_buffer2);
                fuse_nn_results(fuse_buffer1, fuse_buffer2, num_neighbors, curnn);
            }
        }
    });
}

template<typename Index_, typename Float_>
void find_batch_neighbors(
    std::size_t num_dim,
    const BatchInfo<Index_, Float_>& ref,
    const BatchInfo<Index_, Float_>& target,
    const Float_* data,
    int num_neighbors,
    bool fuse_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    find_batch_neighbors(
        num_dim,
        ref.num_obs,
        [&](Index_ l) -> Index_ { return l + ref.offset; },
        data,
        target,
        num_neighbors,
        fuse_neighbors,
        num_threads,
        output 
    );

    for (const auto& extra : ref.extras) {
        find_batch_neighbors(
            num_dim,
            static_cast<Index_>(extra.ids.size()),
            [&](Index_ l) -> Index_ { return extra.ids[l]; },
            data,
            target,
            num_neighbors,
            fuse_neighbors,
            num_threads,
            output
        );
    }
}

template<typename Index_, typename Float_>
struct FindBatchNeighborsResults {
    NeighborSet<Index_, Float_> neighbors;
};

template<typename Index_, typename Float_>
void find_batch_neighbors(
    std::size_t num_dim,
    Index_ num_total,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const BatchInfo<Index_, Float_>& target,
    const Float_* data,
    int num_neighbors,
    int num_threads,
    FindBatchNeighborsResults<Index_, Float_>& output)
{
    output.neighbors.resize(num_total);
    for (decltype(references.size()) b = 0, end = references.size(); b < end; ++b) {
        const auto& curref = references[b];
        find_batch_neighbors(num_dim, curref, target, data, num_neighbors, false, num_threads, output.neighbors);
        find_batch_neighbors(num_dim, target, curref, data, num_neighbors, b > 0, num_threads, output.neighbors);
    }
}

}

}

#endif
