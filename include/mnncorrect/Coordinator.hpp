#ifndef MNNCORRECT_COORDINATOR_HPP
#define MNNCORRECT_COORDINATOR_HPP

#include <algorithm>
#include <memory>
#include <vector>
#include <cstddef>
#include <numeric>
#include <optional>

#include "sanisizer/sanisizer.hpp"
#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_closest_mnn.hpp"
#include "find_neighbors.hpp"
#include "correct_target.hpp"
#include "define_merge_order.hpp"

namespace mnncorrect {

template<typename Index_, typename Float_>
void fill_batch_ids(const MetaBatch<Index_, Float_>& meta_batch, std::vector<Index_>& ids) {
    ids.resize(meta_batch.original_ids.size); // this is known to not overflow as Coordinator's constructor already reserved the maximum space.
    std::iota(ids.begin(), ids.end(), meta_batch.original_ids.start);
    for (const auto& corrected : meta_batch.corrected) {
        ids.insert(ids.end(), corrected.ids.begin(), corrected.ids.end());
    }
    std::sort(ids.begin(), ids.end());
}

template<typename Index_, typename Float_>
struct RedistributeCorrectedObservationsWorkspace {
    std::vector<Index_> offsets;
    std::vector<Float_> buffer;
};

template<typename Index_, typename Float_, typename Matrix_>
void redistribute_corrected_observations(
    const std::size_t num_dim,
    CorrectTargetResults<Index_> correct_info,
    const Float_* const data,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    const int num_threads,
    RedistributeCorrectedObservationsWorkspace<Index_, Float_>& workspace,
    std::vector<MetaBatch<Index_, Float_> >& meta_batches,
    std::vector<BatchIndex>& meta_batch_assignments
) {
    // The idea with the workspace is to do one big allocation and then operate on contiguous chunks of that allocation within each thread.
    // This allows us to use the upper bound of required space to create an allocation that can be reused across all calls to this function.
    // False sharing should not be a major issue as there aren't many boundaries between threads at which contention could occur.
    const auto num_remaining = correct_info.reassignments.size();
    workspace.offsets.clear();
    workspace.offsets.reserve(num_remaining);
    Index_ sofar = 0;
    for (I<decltype(num_remaining)> b = 0; b < num_remaining; ++b) {
        const auto& rem = correct_info.reassignments[b];
        workspace.offsets.push_back(sofar);
        sofar += rem.size(); // known to NOT overflow, see Coordinator's constructor.
        for (const auto r : rem) {
            meta_batch_assignments[r] = b;
        }
    }

    parallelize(num_threads, num_remaining, [&](const int, const I<decltype(num_remaining)> start, const I<decltype(num_remaining)> length) -> void {
        for (BatchIndex b = start, end = start + length; b < end; ++b) {
            // workspace.buffer was already allocated in the Coordinator constructor so this pointer arithmetic is fine.
            const auto storage = workspace.buffer.data() + sanisizer::product_unsafe<std::size_t>(workspace.offsets[b], num_dim);

            auto& reass = correct_info.reassignments[b];
            const auto num_reass = reass.size();
            for (I<decltype(num_reass)> i = 0; i < num_reass; ++i) {
                std::copy_n(
                    data + sanisizer::product_unsafe<std::size_t>(reass[i], num_dim),
                    num_dim, 
                    storage + sanisizer::product_unsafe<std::size_t>(i, num_dim)
                );
            }

            meta_batches[b].corrected.emplace_back(
                builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_reass, storage)),
                std::move(reass)
            );
        }
    });
}

template<typename Index_, typename Float_, typename Matrix_>
class Coordinator {
public:
    Coordinator(
        const std::size_t num_dim,
        const std::vector<Batch<Index_> >& all_batches,
        Float_* const corrected,
        const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
        const int num_neighbors, 
        const int num_steps,
        const MergePolicy merge_policy, 
        const int num_threads
    ) :
        my_num_dim(num_dim), 
        my_builder(builder),
        my_corrected(corrected),
        my_num_neighbors(num_neighbors),
        my_num_steps(num_steps),
        my_num_threads(num_threads)
    {
        BatchIndex num_batches = sanisizer::cast<BatchIndex>(all_batches.size());

        // Filtering out empty batches.
        BatchIndex num_empty = 0;
        for (auto& batch : all_batches) {
            num_empty += (batch.size == 0);
        }

        num_batches -= num_empty;
        if (num_batches == 0) {
            return;
        }

        std::optional<std::vector<Batch<Index_> > > non_empty_batches;
        const auto& batches = [&]() -> const std::vector<Batch<Index_> >& {
            if (num_empty == 0) {
                return all_batches;
            }
            non_empty_batches.emplace();
            non_empty_batches->reserve(num_batches);
            for (auto& batch : all_batches) {
                if (batch.size) {
                    non_empty_batches->push_back(batch);
                }
            }
            return *non_empty_batches;
        }();

        // Different policies to choose the order in which batches are merged.
        // Note that 'order' is filled in reverse order of batches to merge, i.e., the batch at 'order.back()' is merged first.
        // The batch at `order.front()` is never merged and its values will never be corrected.
        std::vector<BatchIndex> order;
        if (merge_policy == MergePolicy::SIZE) {
            define_size_merge_order(batches, order);
        } else if (merge_policy == MergePolicy::VARIANCE || merge_policy == MergePolicy::RSS) {
            const bool as_rss = merge_policy == MergePolicy::RSS;
            const auto vars = compute_total_variances(num_dim, batches, corrected, as_rss, num_threads);
            define_variance_merge_order(vars, order);
        } else { // i.e., merge_policy = INPUT.
            sanisizer::resize(order, num_batches);
            std::iota(order.begin(), order.end(), static_cast<BatchIndex>(0));
        }

        // Each metabatch is an agglomeration of multiple original batches.
        // Initially, each metabatch just contains one of the original batches, but two metabatches will be merged at a time in each call to 'next()'.
        sanisizer::resize(my_meta_batches, num_batches);
        parallelize(num_threads, num_batches, [&](const int, const BatchIndex start, const BatchIndex length) -> void {
            for (BatchIndex b = start, end = start + length; b < end; ++b) {
                const auto& src = batches[order[b]];
                auto& dest = my_meta_batches[b];
                dest.original_ids = src;
                dest.original_index = my_builder.build_unique(
                    knncolle::SimpleMatrix<Index_, Float_>(
                        num_dim,
                        src.size,
                        corrected + sanisizer::product_unsafe<std::size_t>(src.start, num_dim)
                    )
                );
            }
        });

        Index_ num_total = 0;
        for (BatchIndex b = 0; b < num_batches; ++b) {
            const auto cur_size = my_meta_batches[b].original_ids.size;
            num_total = sanisizer::sum<Index_>(num_total, cur_size);
        }
        my_correct_workspace = CorrectTargetWorkspace<Index_, Float_>(num_total);

        // Do this after re-ordering so that we can index into 'my_meta_batches'.
        sanisizer::resize(my_meta_batch_assignment, num_total);
        for (BatchIndex b = 0; b < num_batches; ++b) {
            const auto& curbatch = my_meta_batches[b].original_ids;
            std::fill_n(my_meta_batch_assignment.begin() + curbatch.start, curbatch.size, b);
        }

        // Allocate one big space for index construction once, so that we don't have to reallocate within each redistribute_corrected_observations() call.
        my_build_workspace.buffer.resize(sanisizer::product<I<decltype(my_build_workspace.buffer.size())> >(my_num_dim, num_total));

        // Avoid repeated allocations in fill_batch_ids(). 
        my_target_ids.reserve(sanisizer::cast<I<decltype(my_target_ids.size())> >(num_total));

        // Avoid repeated allocations in find_neighbors().
        sanisizer::resize(my_neighbors, num_total);
    }

protected:
    std::size_t my_num_dim;
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<MetaBatch<Index_, Float_> > my_meta_batches;

    Float_* my_corrected;

    std::vector<Index_> my_target_ids;
    std::vector<BatchIndex> my_meta_batch_assignment;

    NeighborSet<Index_, Float_> my_neighbors;
    FindClosestMnnResults<Index_> my_mnns;
    FindClosestMnnWorkspace<Index_> my_mnn_workspace;
    CorrectTargetWorkspace<Index_, Float_> my_correct_workspace;
    RedistributeCorrectedObservationsWorkspace<Index_, Float_> my_build_workspace;

    int my_num_neighbors;
    double my_num_steps;
    int my_num_threads;

protected:
    bool next(bool test) {
        MetaBatch<Index_, Float_> target_meta_batch(std::move(my_meta_batches.back()));
        my_meta_batches.pop_back();

        fill_batch_ids(target_meta_batch, my_target_ids);

        find_neighbors(
            my_num_dim,
            my_meta_batches,
            target_meta_batch,
            my_corrected,
            my_num_neighbors,
            my_num_threads,
            my_neighbors 
        );

        find_closest_mnn(
            my_target_ids,
            my_neighbors,
            my_mnn_workspace,
            my_mnns
        );

        auto correct_info = correct_target(
            my_num_dim,
            my_meta_batches,
            target_meta_batch,
            my_meta_batch_assignment,
            my_target_ids,
            my_mnns,
            my_builder,
            my_num_neighbors,
            my_num_steps,
            my_num_threads,
            my_corrected,
            my_correct_workspace
        );

        // We don't need to do this at the last step.
        const bool remaining = my_meta_batches.size() > 1;
        if (remaining || test) {
            redistribute_corrected_observations(
                my_num_dim,
                std::move(correct_info),
                my_corrected,
                my_builder,
                my_num_threads,
                my_build_workspace,
                my_meta_batches,
                my_meta_batch_assignment
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

#endif
