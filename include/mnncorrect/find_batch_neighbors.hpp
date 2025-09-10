#ifndef MNNCORRECT_FIND_BATCH_NEIGHBORS_HPP
#define MNNCORRECT_FIND_BATCH_NEIGHBORS_HPP

#include <vector>
#include <utility>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "fuse_nn_results.hpp"
#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
void fill_pair_vector(const std::vector<Index_>& indices, const std::vector<Float_>& distances, std::vector<std::pair<Index_, Float_> >& output) {
    const auto found = indices.size();
    output.clear();
    output.reserve(found);
    for (decltype(I(found)) i = 0; i < found; ++i) {
        output.emplace_back(indices[i], distances[i]);
    }
}

template<typename Index_, class GetId_, typename Float_>
void find_batch_neighbors(
    const std::size_t num_dim,
    const Index_ num_obs,
    const GetId_ get_data_id,
    const Float_* const data,
    const BatchInfo<Index_, Float_>& batch,
    const int num_neighbors,
    const bool fuse_neighbors,
    const int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    parallelize(num_threads, num_obs, [&](const int, const Index_ start, const Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        std::vector<std::pair<Index_, Float_> > fuse_buffer1, fuse_buffer2;

        { // scoped to prevent confusion from variable aliasing with batch.extras.
            auto searcher = batch.index->initialize();
            const auto capped_neighbors = knncolle::cap_k_query(num_neighbors, batch.index->num_observations());

            for (Index_ l = start, end = start + length; l < end; ++l) {
                const auto k = get_data_id(l);
                const auto ptr = data + sanisizer::product_unsafe<std::size_t>(k, num_dim);
                searcher->search(ptr, capped_neighbors, &indices, &distances);
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
        }

        for (const auto& extra : batch.extras) {
            auto searcher = extra.index->initialize();
            const auto capped_neighbors = knncolle::cap_k_query(num_neighbors, extra.index->num_observations());

            for (Index_ l = start, end = start + length; l < end; ++l) {
                const auto k = get_data_id(l);
                const auto ptr = data + sanisizer::product_unsafe<std::size_t>(k, num_dim);
                searcher->search(ptr, capped_neighbors, &indices, &distances);
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
    const std::size_t num_dim,
    const BatchInfo<Index_, Float_>& ref,
    const BatchInfo<Index_, Float_>& target,
    const Float_* const data,
    const int num_neighbors,
    const bool fuse_neighbors,
    const int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    find_batch_neighbors(
        num_dim,
        ref.num_obs,
        [&](const Index_ l) -> Index_ { return l + ref.offset; },
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
            [&](const Index_ l) -> Index_ { return extra.ids[l]; },
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
    const std::size_t num_dim,
    const Index_ num_total,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const BatchInfo<Index_, Float_>& target,
    const Float_* const data,
    const int num_neighbors,
    const int num_threads,
    FindBatchNeighborsResults<Index_, Float_>& output)
{
    sanisizer::resize(output.neighbors, num_total);
    const auto num_refs = references.size();
    for (decltype(I(num_refs)) b = 0; b < num_refs; ++b) {
        const auto& curref = references[b];
        find_batch_neighbors(num_dim, curref, target, data, num_neighbors, false, num_threads, output.neighbors);
        find_batch_neighbors(num_dim, target, curref, data, num_neighbors, b > 0, num_threads, output.neighbors);
    }
}

}

}

#endif
