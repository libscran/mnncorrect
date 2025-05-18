#ifndef MNNCORRECT_AUTOMATIC_ORDER_HPP
#define MNNCORRECT_AUTOMATIC_ORDER_HPP

#include <algorithm>
#include <stdexcept>
#include <memory>
#include <vector>
#include <cstddef>
#include <numeric>

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_closest_mnn.hpp"
#include "find_batch_neighbors.hpp"
#include "correct_target.hpp"
#include "define_merge_order.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
void fill_batch_ids(const BatchInfo<Index_, Float_>& batch, std::vector<Index_>& ids) {
    ids.resize(batch.num_obs);
    std::iota(ids.begin(), ids.end(), batch.offset);
    for (const auto& extra : batch.extras) {
        ids.insert(ids.end(), extra.ids.begin(), extra.ids.end());
    }
    std::sort(ids.begin(), ids.end());
}

template<typename Index_, typename Float_>
struct RedistributeCorrectedObservationsWorkspace {
    std::vector<Index_> offsets;
    std::vector<Float_> buffer;
};

template<typename Index_, typename Float_>
void redistribute_corrected_observations(
    std::size_t num_dim,
    CorrectTargetResults<Index_> correct_info,
    const Float_* data,
    const knncolle::Builder<Index_, Float_, Float_>& builder,
    int num_threads,
    RedistributeCorrectedObservationsWorkspace<Index_, Float_>& workspace,
    std::vector<BatchInfo<Index_, Float_> >& batches,
    std::vector<BatchIndex>& assigned_batch)
{
    // The idea with the workspace is to do one big allocation and then operate
    // on contiguous chunks of that allocation within each thread. This allows
    // us to use the upper bound of required space to create an allocation that
    // can be reused across all calls to this function.
    BatchIndex num_remaining = correct_info.reassignments.size();
    workspace.offsets.clear();
    workspace.offsets.reserve(num_remaining);
    Index_ sofar = 0;
    for (BatchIndex b = 0; b < num_remaining; ++b) {
        const auto& rem = correct_info.reassignments[b];
        workspace.offsets.push_back(sofar);
        sofar += rem.size();
        for (auto r : rem) {
            assigned_batch[r] = b;
        }
    }

    // Technically this is not necessary as we do a big allocation in the
    // AutomaticOrder constructor... but we just resize it here for safety.
    workspace.buffer.resize(static_cast<std::size_t>(sofar) * num_dim); // cast to avoid overflow.

    parallelize(num_threads, num_remaining, [&](int, BatchIndex start, BatchIndex length) -> void {
        for (BatchIndex b = start, end = start + length; b < end; ++b) {
            auto storage = workspace.buffer.data() + static_cast<std::size_t>(workspace.offsets[b]) * num_dim; // cast to avoid overflow.

            auto& reass = correct_info.reassignments[b];
            auto num_reass = reass.size();
            for (decltype(num_reass) i = 0; i < num_reass; ++i) {
                std::copy_n(
                    data + static_cast<std::size_t>(reass[i]) * num_dim, // ditto.
                    num_dim, 
                    storage + static_cast<std::size_t>(i) * num_dim
                );
            }

            batches[b].extras.emplace_back(
                builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_reass, storage)),
                std::move(reass)
            );
        }
    });
}

template<typename Index_, typename Float_, typename Matrix_>
class AutomaticOrder {
public:
    AutomaticOrder(
        std::size_t num_dim,
        const std::vector<Index_>& num_obs,
        const std::vector<const Float_*>& batches, 
        Float_* corrected,
        const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
        int num_neighbors, 
        int num_steps,
        MergePolicy merge_policy, 
        int num_threads)
    :
        my_num_dim(num_dim), 
        my_builder(builder),
        my_corrected(corrected),
        my_num_neighbors(num_neighbors),
        my_num_steps(num_steps),
        my_num_threads(num_threads)
    {
        BatchIndex nbatches = num_obs.size();
        if (nbatches != batches.size()) {
            throw std::runtime_error("length of 'num_obs' and 'batches' must be equal");
        }
        if (nbatches == 0) {
            return;
        }

        my_num_total = 0;
        std::vector<BatchInfo<Index_, Float_> > tmp_batches(nbatches);
        for (BatchIndex b = 0; b < nbatches; ++b) {
            tmp_batches[b].offset = my_num_total;
            auto cur_num_obs = num_obs[b];
            tmp_batches[b].num_obs = cur_num_obs;
            std::copy_n(
                batches[b],
                static_cast<std::size_t>(cur_num_obs) * num_dim, // cast to size_t to avoid overflow.
                my_corrected + static_cast<std::size_t>(my_num_total) * num_dim // ditto.
            );
            my_num_total += cur_num_obs;
        }

        parallelize(num_threads, nbatches, [&](int, BatchIndex start, BatchIndex length) -> void {
            for (BatchIndex b = start, end = start + length; b < end; ++b) {
                auto& curbatch = tmp_batches[b];
                curbatch.index = my_builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_obs[b], batches[b]));
            }
        });

        // Different policies to choose the batch order. 'order' is filled
        // in reverse order of batches to merge, with the first batch being unchanged. 
        std::vector<BatchIndex> order;
        if (merge_policy == MergePolicy::SIZE) {
            define_merge_order(num_obs, order);
        } else if (merge_policy == MergePolicy::VARIANCE || merge_policy == MergePolicy::RSS) {
            bool as_rss = merge_policy == MergePolicy::RSS;
            std::vector<Float_> vars = compute_total_variances(num_dim, num_obs, batches, as_rss, num_threads);
            define_merge_order(vars, order);
        } else { // i.e., merge_policy = INPUT.
            order.resize(nbatches);
            std::iota(order.begin(), order.end(), static_cast<BatchIndex>(0));
        }

        my_batches.reserve(nbatches);
        for (auto o : order) {
            my_batches.emplace_back(std::move(tmp_batches[o]));
        }

        // Do this after re-ordering so that we can index into 'my_batches'.
        my_batch_assignment.resize(my_num_total);
        for (BatchIndex b = 0; b < nbatches; ++b) {
            const auto& curbatch = my_batches[b];
            std::fill_n(my_batch_assignment.begin() + curbatch.offset, curbatch.num_obs, b);
        }

        // Allocate one big space for index construction once, so that we don't
        // have to reallocate within each redistribute_corrected_observations() call.
        my_build_workspace.buffer.resize(my_num_dim * static_cast<std::size_t>(my_num_total));
    }

protected:
    std::size_t my_num_dim;
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<BatchInfo<Index_, Float_> > my_batches;

    Float_* my_corrected;
    Index_ my_num_total;

    std::vector<Index_> my_target_ids;
    std::vector<BatchIndex> my_batch_assignment;

    FindBatchNeighborsResults<Index_, Float_> my_batch_nns;
    FindClosestMnnResults<Index_> my_mnns;
    FindClosestMnnWorkspace<Index_> my_mnn_workspace;
    CorrectTargetWorkspace<Index_, Float_> my_correct_workspace;
    RedistributeCorrectedObservationsWorkspace<Index_, Float_> my_build_workspace;

    int my_num_neighbors;
    double my_num_steps;
    int my_num_threads;

protected:
    bool next(bool test) {
        // Here, we denote 'my_batches' and 'target_batch' as "metabatches",
        // because they are agglomerations of the original batches. The idea is
        // to always merge two metabatches at each call to 'next()'.
        BatchInfo<Index_, Float_> target_batch(std::move(my_batches.back()));
        my_batches.pop_back();

        fill_batch_ids(target_batch, my_target_ids);

        find_batch_neighbors(
            my_num_dim,
            my_num_total,
            my_batches,
            target_batch,
            my_corrected,
            my_num_neighbors,
            my_num_threads,
            my_batch_nns
        );

        find_closest_mnn(
            my_target_ids,
            my_batch_nns.neighbors,
            my_mnn_workspace,
            my_mnns
        );

        auto correct_info = correct_target(
            my_num_dim,
            my_num_total,
            my_batches,
            target_batch,
            my_target_ids,
            my_batch_assignment,
            my_mnns,
            my_builder,
            my_num_neighbors,
            my_num_steps,
            my_num_threads,
            my_corrected,
            my_correct_workspace
        );

        // We don't need to do this at the last step.
        bool remaining = my_batches.size() > 1;
        if (remaining || test) {
            redistribute_corrected_observations(
                my_num_dim,
                std::move(correct_info),
                my_corrected,
                my_builder,
                my_num_threads,
                my_build_workspace,
                my_batches,
                my_batch_assignment
            );
        }

        return remaining;
    }

public:
    void merge() {
        while (next(false)) {}
    }
};

}

}

#endif
