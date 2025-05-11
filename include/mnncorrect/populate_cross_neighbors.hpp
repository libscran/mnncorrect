#ifndef MNNCORRECT_POPULATE_NEIGHBORS_HPP
#define MNNCORRECT_POPULATE_NEIGHBORS_HPP

#include <vector>
#include <utility>
#include <cstddef>

#include "parallelize.hpp"
#include "BatchInfo.hpp"
#include "fuse_nn_results.hpp"
#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct PopulateCrossNeighborsWorkspace {
    NeighborSet<Index_, Float_> neighbors;
    std::vector<Index_> ref_ids, target_ids;
    std::vector<BatchIndex> batch;
};

template<typename Index_, class GetId_, typename Float_>
void populate_batch_neighbors(
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
        auto searcher = batch.index->initialize();

        std::pair<std::pair<Index_, Float_> > fuse_buffer1, fuse_buffer2;
        auto store_nn = [&](Index_ k) -> void {
            auto& curnn = output[k];
            if (!fuse_neighbors) {
                fill_pair_vector(indices, distances, curnn);
            } else {
                fuse_buffer1.swap(curnn);
                fill_pair_vector(indices, distances, fuse_buffer2);
                fuse_nn_results(fuse_buffer1, fuse_buffer2, num_neighbors, curnn);
            }
        };

        for (Index_ l = start, end = start + length; l < end; ++l) {
            auto k = get_data_id(l);
            auto ptr = data + static_cast<std::size_t>(k) * num_dim;
            searcher->search(ptr, num_neighbors, &indices, &distances);
            for (auto& i : indices) {
                i += batch.offset;
            }
            store_nn(k);
        }

        for (auto extra : batch.extras) {
            auto searcher = extra.index->initialize();
            for (Index_ l = start, end = start + length; l < end; ++l) {
                auto k = get_data_id(l);
                auto ptr = data + static_cast<std::size_t>(k) * num_dim;
                searcher->search(ptr, num_neighbors, &indices, &distances);
                for (auto& i : indices) {
                    i = extra.ids[i];
                }
                store_nn(k);
            }
        }
    });
}

template<typename Index_, typename Float_>
void populate_cross_neighbors(
    std::size_t num_dim,
    const BatchInfo<Index_, Float_>& ref,
    const BatchInfo<Index_, Float_>& target,
    const Float_* data,
    int num_neighbors,
    bool fuse_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    populate_batch_neighbors(
        ref.num_obs,
        [&](Index_ l) -> Index_ { return l + ref.obs; },
        data,
        target,
        num_neighbors,
        fuse_neighbors,
        num_threads,
        output 
    );

    for (const auto& extra : ref.extras) {
        populate_batch_neighbors(
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
void populate_cross_neighbors(
    std::size_t num_dim,
    Index_ num_total,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const BatchInfo<Index_, Float_>& target,
    int num_neighbors,
    int num_threads,
    ConsolidatedNeighborWorkspace<Index_, Float_>& workspace)
{
    workspace.batch.resize(num_total);
    workspace.ref_ids.clear();
    workspace.neighbors.resize(num_total);

    for (decltype(references.size()) b = 0, end = references.size(); b < end; ++b) {
        const auto& curref = references[b];
        populate_cross_neighbors(curref, corrected, target, num_neighbors, false, num_threads, workspace.neighbors);
        populate_cross_neighbors(target, corrected, curref, num_neighbors, b > 0, num_threads, workspace.neighbors);

        // Adding all the details about which observations are in the reference metabatch,
        // and which of the inner batches they belonged to.
        workspace.ref_ids.reserve(workspace.ref_ids.size() + curref.num_obs);
        for (Index_ i = 0; i < curref.num_obs; ++i) {
            workspace.ref_ids.push_back(i + curref.offset);
        }
        std::fill(batch.begin() + curref.offset, curref.num_obs, b);

        for (const auto& extra : curref.extras) {
            ref_ids.insert(ref_ids.end(), extra.ids.begin(), extra.ids.end());
            for (auto e : extra.ids) {
                batch[e] = b;
            }
        }
    }

    workspace.target_ids.clear();
    workspace.ref_ids.reserve(workspace.ref_ids.size() + target.num_obs);
    for (Index_ i = 0; i < target.num_obs; ++i) {
        workspace.target_ids.push_back(i + target.offset);
    }
    for (const auto& extra : target.extras) {
        workspace.target_ids.insert(workspace.target_ids.end(), extra.ids.begin(), extra.ids.end());
    }

    std::sort(workspace.ref_ids.begin(), workspace.ref_ids.end());
    std::sort(workspace.target_ids.begin(), workspace.target_ids.end());
}

}

}

#endif
