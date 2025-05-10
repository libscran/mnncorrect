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
#include "populate_neighbors.hpp"
#include "correct_target.hpp"
#include "parallelize.hpp"

namespace mnncorrect {

namespace internal {

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
        my_ids.batch.reserve(sofar);
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
    ConsolidatedNeighborInfo<Index_, Float_> my_ids;
    FindMnnWorkspace<Index_, Float_> my_mnns;

    int my_num_neighbors;
    int my_nthreads;

public:
    void next(Float_ num_sds) {
        BatchInfo<Index_, Float_> target_batch(std::move(my_batches[my_target]));
        my_batches.pop_back();

        for (decltype(my_batches.size()) b = 0, end = my_batches.size(); b < end; ++b) {
            populate_cross_neighbors(my_batches[b], my_corrected, target_batch, my_num_neighbors, false, my_num_threads, my_neighbors);
            populate_cross_neighbors(target_batch, my_corrected, my_batches[b], my_num_neighbors, b > 0, my_num_threads, my_neighbors);
        }

        populate_neighbor_info(my_batches, target_batch, my_ids);

        find_closest_mnn(my_neighbors, my_ids.ref_ids, my_ids.target_ids, my_mnns);

        // TODO:
        // - Build indices for the MNN-involved cells.
        // - Search for the closest neighbors of the MNN-involved cells in each batch.
        // - Search each batch for the closest MNN-involved cells.
        // - Calculate the center of mass based on the nearby cells.
        // - Calculate the correction in the target based on its closest MNN-involved cell.
        // - Assign each cell to the batch of the other cell in the pair.
        // - Build indices for the newly reassigned cells in each remaining batch. 

        --my_unmerged;
    }
};

}

}

#endif
