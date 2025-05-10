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

template<typename Index_, class GetId_, typename Float_>
void populate_cross_neighbors(
    Index_ num_obs,
    GetId_ get_id,
    const Float_* data,
    const BatchInfo<Index_, Float_>& target,
    int num_neighbors,
    bool fuse_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& neighbors)
{
    parallelize(num_threads, num_obs, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = target.index->initialize();

        std::pair<std::pair<Index_, Float_> > fuse_buffer1, fuse_buffer2;
        auto store_nn = [&](Index_ k) -> void {
            auto& curnn = neighbors[k];
            if (!fuse_neighbors) {
                fill_pair_vector(indices, distances, curnn);
            } else {
                fuse_buffer1.swap(curnn);
                fill_pair_vector(indices, distances, fuse_buffer2);
                fuse_nn_results(fuse_buffer1, fuse_buffer2, num_neighbors, curnn, 0);
            }
        };

        for (Index_ l = start, end = start + length; l < end; ++l) {
            auto k = get_id(l);
            auto ptr = data + static_cast<std::size_t>(k) * ndim;
            searcher->search(ptr, num_neighbors, &indices, &distances);
            for (auto& i : indices) {
                i += target.offset;
            }
            store_nn(k);
        }

        for (auto extra : target.extras) {
            auto searcher = extra.index->initialize();
            for (Index_ l = start, end = start + length; l < end; ++l) {
                auto k = get_id(l);
                auto ptr = data + static_cast<std::size_t>(k) * ndim;
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
    const BatchInfo<Index_, Float_>& ref,
    const Float_* data,
    const BatchInfo<Index_, Float_>& target,
    int num_neighbors,
    bool fuse_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& neighbors)
{
    populate_cross_neighbors(
        ref.num_obs,
        [&](Index_ l) -> Index_ { return l + ref.obs; },
        data,
        target,
        num_neighbors,
        fuse_neighbors,
        num_threads,
        neighbors
    );

    for (const auto& extra : ref.extras) {
        populate_cross_neighbors(
            static_cast<Index_>(extra.ids.size()),
            [&](Index_ l) -> Index_ { return extra.ids[l]; },
            data,
            target,
            num_neighbors,
            fuse_neighbors,
            num_threads,
            neighbors
        );
    }
}

template<typename Index_, typename Float_>
struct ConsolidatedNeighborInfo {
    std::vector<Index_> ref_ids, target_ids;
    std::vector<BatchIndex> batch;
};

template<typename Index_, typename Float_>
void populate_neighbor_info(const std::vector<BatchInfo<Index_, Float_> >& references, const BatchInfo<Index_, Float_>& target, ConsolidatedNeighborInfo<Index_, Float_>& info) {
    auto fill_batch = [&](const BatchInfo<Index_, Float_>& batch) {
        for (Index_ i = 0; i < batch.nobs; ++i) {
            info.ref_ids.push_back(i + batch.offset);
        }
        std::fill(info.batch.begin() + batch.offset, batch.nobs, b);

        for (const auto& extra : batch.extras) {
            info.ref_ids.insert(info.ref_ids.end(), extra.ids.begin(), extra.ids.end());
            for (auto e : extra.ids) {
                info.batch[e] = b;
            }
        }
    }

    info.ref_ids.clear();
    for (BatchIndex b = 0; b < my_unmerged; ++b) {
        fill_batch(references[b]);
    }
    std::sort(info.ref_ids.begin(), info.ref_ids.end());

    info.target_ids.clear();
    fill_batch(target);
    std::sort(info.target_ids.begin(), info.target_ids.end());
}


}

}

#endif
