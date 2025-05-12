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
#include "find_closest_mnn.hpp"
#include "find_batch_neighbors.hpp"
#include "correct_target.hpp"
#include "define_merge_order.hpp"

namespace mnncorrect {

namespace internal {

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
        double tolerance,
        ReferencePolicy ref_policy, 
        int num_threads)
    :
        my_num_dim(num_dim), 
        my_builder(builder),
        my_corrected(corrected),
        my_num_neighbors(num_neighbors),
        my_tolerance(tolerance),
        my_num_threads(num_threads)
    {
        BatchIndex nbatches = num_obs.size();
        if (nbatches != batches.size()) {
            throw std::runtime_error("length of 'num_obs' and 'batches' must be equal");
        }
        if (nbatches == 0) {
            return;
        }

        // Different policies to choose the batch order. 'my_order' is filled
        // in reverse order of batches to merge, with the first batch being unchanged. 
        if (ref_policy == ReferencePolicy::MAX_SIZE) {
            define_merge_order(num_obs, my_order);
        } else if (ref_policy == ReferencePolicy::MAX_VARIANCE || ref_policy == ReferencePolicy::MAX_RSS) {
            bool as_rss = ref_policy == ReferencePolicy::MAX_RSS;
            std::vector<Float_> vars = compute_total_variances(my_num_dim, num_obs, batches, as_rss, my_num_threads);
            define_merge_order(vars, my_order);
        } else { // i.e., ref_policy = INPUT.
            my_order.resize(nbatches);
            std::iota(my_order.begin(), my_order.end(), 0);
        }

        my_batches.resize(nbatches);
        parallelize(num_threads, nbatches, [&](int, BatchIndex start, BatchIndex length) -> void {
            for (BatchIndex b = start, end = start + length; b < end; ++b) {
                auto& curbatch = my_batches[b];
                curbatch.num_obs = num_obs[b];
                curbatch.index = my_builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_obs[b], batches[b]));
            }
        });

        my_num_total = 0;
        for (BatchIndex b = 0; b < nbatches; ++b) {
            my_batches[b].offset = my_num_total;
            std::copy_n(batches[b], static_cast<std::size_t>(num_obs[b]) * num_dim, my_corrected + static_cast<std::size_t>(my_num_total) * num_dim); // cast to size_t to avoid overflow.
            my_num_total += num_obs[b];
        }

        my_target = nbatches - 1;
    }

protected:
    std::size_t my_num_dim;
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<BatchInfo<Index_, Float_> > my_batches;

    Float_* my_corrected;
    Index_ my_num_total;
    std::vector<BatchIndex> my_order;
    BatchIndex my_target;

    FindBatchNeighborsResults<Index_, Float_> my_batch_nns;
    FindClosestMnnResults<Index_> my_mnns;
    FindClosestMnnWorkspace<Index_> my_mnn_workspace;
    CorrectTargetWorkspace<Index_, Float_> my_correct_workspace;

    int my_num_neighbors;
    double my_tolerance;
    int my_num_threads;

public:
    bool next() {
        // Here, we denote 'my_batches' and 'target_batch' as "metabatches",
        // because they are agglomerations of the original batches. The idea is
        // to always merge two metabatches at each call to 'next()'.
        BatchInfo<Index_, Float_> target_batch(std::move(my_batches[my_target]));
        my_batches.pop_back();

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
            my_batch_nns,
            my_mnn_workspace,
            my_mnns
        );

        correct_target(
            my_num_dim,
            my_num_total,
            my_batches,
            target_batch,
            my_batch_nns,
            my_mnns,
            my_builder,
            my_num_neighbors,
            my_num_threads,
            my_tolerance,
            my_corrected,
            my_correct_workspace
        );

        // Reassigning the target batch's observations to the various reference batches.
        BatchIndex num_remaining = my_batches.size();
        std::vector<std::vector<Index_> > reassigned(num_remaining);
        for (auto t : my_batch_nns.target_ids) {
            reassigned[my_correct_workspace.chosen_batch[t]].push_back(t);
        }

        parallelize(my_num_threads, num_remaining, [&](int, BatchIndex start, BatchIndex length) -> void {
            std::vector<Float_> buffer;
            for (BatchIndex b = start, end = start + length; b < end; ++b) {
                auto& reass = reassigned[b];
                buffer.resize(static_cast<std::size_t>(reass.size()) * my_num_dim); // cast to avoid overflow.

                auto num_reass = reass.size();
                for (decltype(num_reass) i = 0; i < num_reass; ++i) {
                    std::copy_n(
                        my_corrected + static_cast<std::size_t>(reass[i]) * my_num_dim, // ditto.
                        my_num_dim, 
                        buffer.begin() + static_cast<std::size_t>(i) * my_num_dim
                    );
                }

                my_batches[b].extras.emplace_back(
                    my_builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(my_num_dim, num_reass, buffer.data())),
                    std::move(reass)
                );
            }
        });

        --my_target;
        return my_target > 0;
    }

    void merge() {
        while (next()) {}
    }
};

}

}

#endif
