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
        double tolerance,
        ReferencePolicy ref_policy, 
        int nthreads)
    :
        my_ndim(ndim), 
        my_builder(builder),
        my_corrected(corrected),
        my_num_neighbors(num_neighbors),
        my_tolerance(tolerance)
        my_nthreads(nthreads),
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

        my_num_total = 0;
        for (BatchIndex b = 0; b < nbatches; ++b) {
            my_batches[b].original_offset = my_num_total;
            std::copy_n(batches[b], static_cast<std::size_t>(my_nobs[b]) * ndim, my_corrected + static_cast<std::size_t>(my_num_total) * ndim); // cast to size_t to avoid overflow.
            my_num_total += my_nobs[b];
        }

        my_target = nbatches - 1;
    }

protected:
    std::size_t my_ndim;
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<BatchInfo<Index_, Float_> > my_batches;

    Float_* my_corrected;
    Index_ my_num_total;
    std::vector<BatchIndex> my_order;
    BatchIndex my_target;

    PopulateCrossNeighborsWorkspace<Index_, Float_> my_pop_workspace;
    FindMnnWorkspace<Index_, Float_> my_mnn_workspace;
    CorrectTargetWorkspace<Index_, Float_> my_correct_workspace;

    int my_num_neighbors;
    int my_nthreads;
    double my_tolerance;

public:
    void next(Float_ tolerance) {
        // Here, we denote 'my_batches' and 'target_batch' as "metabatches",
        // because they are agglomerations of the original batches. The idea is
        // to always merge two metabatches at each call to 'next()'.
        BatchInfo<Index_, Float_> target_batch(std::move(my_batches[my_target]));
        my_batches.pop_back();

        populate_cross_neighbors(
            my_num_total,
            my_batches,
            target_batch,
            my_corrected,
            my_num_neighbors,
            my_num_threads,
            my_pop_workspace
        );

        find_closest_mnn(my_pop_workspace, my_mnn_workspace);

        correct_target(
            my_ndim,
            my_num_total,
            my_batches,
            my_target,
            my_pop_workspace,
            my_mnn_workspace,
            my_builder,
            my_num_neighbors,
            my_num_threads,
            my_tolerance,
            my_corrected,
            my_correct_workspace
        );

        // TODO:
        // - Build indices for the newly reassigned cells in each remaining batch. 

        --my_unmerged;
    }
};

}

}

#endif
