#ifndef MNNCORRECT_AUTOMATIC_ORDER_HPP
#define MNNCORRECT_AUTOMATIC_ORDER_HPP

#include <algorithm>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstddef>

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_mutual_nns.hpp"
#include "fuse_nn_results.hpp"
#include "correct_target.hpp"
#include "parallelize.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct Corrected {
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index;
    std::vector<Index_> ids;
};

template<typename Index_, typename Float_>
struct BatchInfo {
    Index_ original_offset;
    Index_ original_nobs;
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > original_index;
    std::vector<Corrected<Index_, Float_> > extras;
};

template<typename Index_, typename Float_, typename Matrix_>
class AutomaticOrder {
public:
    AutomaticOrder(
        std::size_t ndim,
        const std::vector<Index_>& nobs,
        const std::vector<const Float_*>& batches, 
        Float_* corrected,
        const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
        int num_neighbors, 
        ReferencePolicy ref_policy, 
        int nthreads)
    :
        my_ndim(ndim), 
        my_builder(builder),
        my_corrected(corrected),
        my_num_neighbors(num_neighbors),
        my_nthreads(nthreads)
    {
        BatchIndex nbatches = my_nobs.size();
        if (nbatches != my_batches.size()) {
            throw std::runtime_error("length of 'nobs' and 'batches' must be equal");
        }
        if (nbatches == 0) {
            return;
        }

        // Different policies to choose the batch order. 'my_order' is filled
        // in reverse order of batches to merge, with the first batch being unchanged. 
        if (ref_policy == ReferencePolicy::MAX_SIZE) {
            define_order(my_nobs, my_order);
        } else if (ref_policy == ReferencePolicy::MAX_VARIANCE || ref_policy == ReferencePolicy::MAX_RSS) {
            bool as_rss = ref_policy == ReferencePolicy::MAX_RSS;
            std::vector<Float_> vars = compute_total_variances(my_ndim, my_nobs, my_batches, as_rss, my_nthreads);
            define_order(vars, my_order);
        } else { // i.e., ref_policy = INPUT.
            my_order.resize(nbatches);
            std::iota(my_order.begin(), my_order.end(), 0);
        }

        my_batches.resize(nbatches);
        parallelize(nthreads, nbatches, [&](int, BatchIndex start, BatchIndex length) -> void {
            for (BatchIndex b = start, end = start + length; b < end; ++b) {
                auto& curbatch = my_batches[b];
                curbatch.original_nobs = nobs[b];
                curbatch.original_values = batches[b];
                curbatch.original_index = my_builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(ndim, my_nobs[b], my_batches[b]));
            }
        });

        Index_ sofar = 0;
        for (BatchIndex b = 0; b < nbatches; ++b) {
            my_batches[b].original_offset = sofar;
            std::copy_n(batches[b], static_cast<std::size_t>(my_nobs[b]) * ndim, my_corrected + static_cast<std::size_t>(sofar) * ndim); // cast to size_t to avoid overflow.
            sofar += my_nobs[b];
        }

        my_neighbors.resize(sofar);
        my_ref_ids.reserve(sofar);
        my_target_ids.reserve(sofar);
        my_target = nbatches - 1;
    }

protected:
    std::size_t my_ndim;
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<BatchInfo<Index_, Float_> > my_batches;

    Float_* my_corrected;
    std::vector<BatchIndex> my_order;
    BatchIndex my_target;

    NeighborSet<Index_, Float_> my_neighbors;
    std::vector<Index_> my_ref_ids, my_target_ids;
    MnnPairs<Index_, Float_> my_mnns;

    int my_num_neighbors;
    int my_nthreads;

private:
    template<class GetId_>
    void populate_neighbors(Index_ nobs, GetId_ get_id, const BatchInfo<Index_, Float_>& target, bool fuse) {
        parallelize(my_nthreads, nobs, [&](int, Index_ start, Index_ length) -> void {
            std::vector<Index_> indices;
            std::vector<Float_> distances;
            auto searcher = target.original_index->initialize();

            std::pair<std::pair<Index_, Float_> > fuse_buffer1, fuse_buffer2;
            auto store_nn = [&](Index_ k) -> void {
                auto& curnn = my_neighbors[k];
                if (fuse) {
                    fill_pair_vector(indices, distances, curnn);
                } else {
                    fuse_buffer1.swap(curnn);
                    fill_pair_vector(indices, distances, fuse_buffer2);
                    fuse_nn_results(fuse_buffer1, fuse_buffer2, my_num_neighbors, curnn, 0);
                }
            };

            for (Index_ l = start, end = start + length; l < end; ++l) {
                auto k = get_id(l);
                auto ptr = my_corrected + static_cast<std::size_t>(k) * ndim;
                searcher->search(ptr, my_num_neighbors, &indices, &distances);
                for (auto& i : indices) {
                    i += target.original_offset;
                }
                store_nn(k);
            }

            for (auto extra : target.extras) {
                auto searcher = extra.index->initialize();
                for (Index_ l = start, end = start + length; l < end; ++l) {
                    auto k = get_id(l);
                    auto ptr = my_corrected + static_cast<std::size_t>(k) * ndim;
                    searcher->search(ptr, my_num_neighbors, &indices, &distances);
                    for (auto& i : indices) {
                        i = extra.ids[i];
                    }
                    store_nn(k);
                }
            }
        });
    }

    void populate_neighbors(const BatchInfo<Index_, Float_>& ref, const BatchInfo<Index_, Float_>& target, bool fuse) {
        populate_neighbors(ref.original_nobs, [&](Index_ l) -> Index_ { return l + ref.original_obs; }, target, fuse);
        for (const auto& extra : ref.extras) {
            populate_neighbors(extra.ids.size(), [&](Index_ l) -> Index_ { return extra.ids[l]; }, target, true);
        }
    }

    static void fill_ids(const BatchInfo<Index_, Float_>& batch, std::vector<Index_>& ids) {
        for (Index_ i = 0; i < batch.original_nobs; ++i) {
            ids.push_back(i + batch.original_offset);
        }
        for (const auto& extra : batch.extras) {
            ids.insert(ids.end(), extra.ids.begin(), extra.ids.end());
        }
    }

public:
    void next(Float_ num_sds) {
        // Finding all of the neighbors.
        const auto& target_batch = my_batches[my_unmerged];
        for (BatchIndex b = 0; b < my_unmerged; ++b) {
            populate_neighbors(my_batches[b], target_batch, true);
            populate_neighbors(target_batch, my_batches[b], b > 0);
        }

        // Prefilling the IDs.
        my_ref_ids.clear();
        for (BatchIndex b = 0; b < my_unmerged; ++b) {
            fill_ids(curbatch, my_ref_ids);
        }
        std::sort(my_ref_ids.begin(), my_ref_ids.end());

        my_target_ids.clear();
        fill_ids(curbatch, my_target_ids);
        std::sort(my_target_ids.begin(), my_target_ids.end());

        // Find MNN pairs.
        find_mutual_nns(my_neighbors, my_ref_ids, my_target_ids, my_mnns);

        // Perform correction.
        correct_target(
            my_ndim, 
            my_ref_ids,
            my_target_ids,
            my_corrected, 
            my_mnns,
            my_corrected_batches,
            my_builder,
            my_num_neighbors,
            num_sds,
            my_nthreads
        );

        --my_unmerged;
    }
};

}

}

#endif
