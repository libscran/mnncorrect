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
#include "compute_center_of_mass.hpp"
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

template<typename Index_, typename Float_, typename Matrix_>
class Coordinator {
public:
    Coordinator(
        const std::size_t num_dim,
        const Index_ num_total,
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

        // Do this after re-ordering so that we can index into 'my_meta_batches'.
        sanisizer::resize(my_meta_batch_assignments, num_total);
        for (BatchIndex b = 0; b < num_batches; ++b) {
            const auto& curbatch = my_meta_batches[b].original_ids;
            std::fill_n(my_meta_batch_assignments.begin() + curbatch.start, curbatch.size, b);
        }

        // Populating the remaining buffers to be re-used across next() calls.
        my_reassignments.reserve(sanisizer::cast<I<decltype(my_reassignments.size())> >(num_batches - 1));

        sanisizer::resize(my_neighbors, num_total);

        my_target_ids.reserve(sanisizer::cast<I<decltype(my_target_ids.size())> >(num_total));
        my_mnn_workspace = FindClosestMnnWorkspace<Index_>(num_total);

        my_walk_workspace = NeighborhoodWalkWorkspace<Index_>(num_total);

        my_big_buffer.resize(sanisizer::product<I<decltype(my_big_buffer.size())> >(my_num_dim, num_total));
        sanisizer::resize(my_target_meta_batch, num_total);

        my_redist_offsets.reserve(num_batches - 1);
    }

protected:
    std::size_t my_num_dim;
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<MetaBatch<Index_, Float_> > my_meta_batches;

    Float_* my_corrected;

    std::vector<BatchIndex> my_meta_batch_assignments;
    std::vector<std::vector<Index_> > my_reassignments;

    NeighborSet<Index_, Float_> my_neighbors;

    std::vector<Index_> my_target_ids;
    FindClosestMnnResults<Index_> my_mnns;
    FindClosestMnnWorkspace<Index_> my_mnn_workspace;

    NeighborhoodWalkWorkspace<Index_> my_walk_workspace;

    std::vector<Float_> my_big_buffer;
    std::vector<BatchIndex> my_target_meta_batch;

    std::vector<Index_> my_redist_offsets;

    int my_num_neighbors;
    double my_num_steps;
    int my_num_threads;

protected:
    bool next(bool test) {
        MetaBatch<Index_, Float_> target_meta_batch(std::move(my_meta_batches.back()));
        my_meta_batches.pop_back();

        find_neighbors(
            my_num_dim,
            my_meta_batches,
            target_meta_batch,
            my_corrected,
            my_num_neighbors,
            my_num_threads,
            my_neighbors 
        );

        fill_batch_ids(target_meta_batch, my_target_ids);
        find_closest_mnn(
            my_target_ids,
            my_neighbors,
            my_mnn_workspace,
            my_mnns
        );

        // Build this first so that we can re-use the big buffer for the center of mass calculations.
        const auto target_mnn_index = subset_and_index(
            my_num_dim,
            my_mnns.target_mnns,
            my_corrected,
            my_builder,
            my_big_buffer.data()
        );

        const auto num_refs = my_meta_batches.size();
        my_reassignments.resize(num_refs); // known to be safe as we allocated in the constructor.

        // Split MNN-involved reference cells back into their meta-batches of origin.
        // Here we use 'my_reassignments' as a temporary place to put this information; we will overwrite it later.
        // We also abuse members of 'my_walk_workspace' as a proxy for a hashmap.
        my_walk_workspace.all_ids.clear();
        assert(std::accumulate(my_walk_workspace.visited.begin(), my_walk_workspace.visited.end(), static_cast<Index_>(0)) == 0);
        for (auto& reass : my_reassignments) {
            reass.clear();
        }
        for (const auto r : my_mnns.ref_mnns) {
            if (my_walk_workspace.visited[r]) {
                continue;
            }
            my_reassignments[my_meta_batch_assignments[r]].push_back(r);
            my_walk_workspace.visited[r] = true;
            my_walk_workspace.all_ids.push_back(r);
        }
        for (const auto r : my_walk_workspace.all_ids) {
            my_walk_workspace.visited[r] = false;
        }

        // Compute the center of mass for each MNN-involved cell in the reference meta-batches.
        for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
            compute_center_of_mass(
                my_num_dim,
                my_reassignments[r],
                my_meta_batches[r],
                my_corrected,
                my_num_neighbors,
                my_num_steps,
                my_num_threads,
                my_walk_workspace,
                my_neighbors, // used as workspace only, should be ignored on output.
                my_big_buffer.data()
            );
        }

        // Now computing the correction vector for each MNN pair.
        compute_center_of_mass(
            my_num_dim,
            my_mnns.target_mnns,
            target_meta_batch,
            my_corrected,
            my_num_neighbors,
            my_num_steps,
            my_num_threads,
            my_walk_workspace,
            my_neighbors, // used as workspace only, should be ignored on output.
            my_big_buffer.data()
        );

#ifndef NDEBUG
        {
            // Double-checking that 'target_mnns' are unique and do not overlap with 'ref_mnns'.
            // This is important for the validity of the next step where we store the correction vector in the same 'big_buffer'.
            auto crosscheck = sanisizer::create<std::vector<char> >(my_neighbors.size());
            for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
                for (const auto rx : my_reassignments[r]) {
                    assert(crosscheck[rx] == 0);
                    crosscheck[rx] = 1;
                }
            }
            {
                const auto num_pairs = my_mnns.target_mnns.size();
                for (I<decltype(num_pairs)> p = 0; p < num_pairs; ++p) {
                    const auto t = my_mnns.target_mnns[p];
                    assert(crosscheck[t] == 0);
                    crosscheck[t] = 1;
                }
            }
        }
#endif

        // Replacing each MNN-involved target cell's center of mass with the MNN-derived correction vector.
        // This saves us a subtraction in the inner loop when correcting all cells in tthe target metabatch.
        // Again, recall that 'target_mnns' is unique so this step will never modify each target cell's values in 'my_big_buffer' more than once.
        const auto num_pairs = my_mnns.target_mnns.size();
        for (I<decltype(num_pairs)> p = 0; p < num_pairs; ++p) {
            for (std::size_t d = 0; d < my_num_dim; ++d) {
                auto& correction = my_big_buffer[sanisizer::nd_offset<std::size_t>(d, my_num_dim, my_mnns.target_mnns[p])];
                correction = my_big_buffer[sanisizer::nd_offset<std::size_t>(d, my_num_dim, my_mnns.ref_mnns[p])] - correction;
            }
        }

        // Apply the correction to each cell in the target meta-batch based on its closest MNN-involved cell in the same meta-batch.
        const Index_ num_target = my_target_ids.size();
        parallelize(my_num_threads, num_target, [&](const int, const Index_ start, const Index_ length) -> void {
            auto searcher = target_mnn_index->initialize();
            std::vector<Index_> indices;
            assert(target_mnn_index->num_observations() > 0);

            for (Index_ i = start, end = start + length; i < end; ++i) {
                const auto tptr = my_corrected + sanisizer::product_unsafe<std::size_t>(my_target_ids[i], my_num_dim);
                // No need to cap the number of neighbors to a value below 1.
                // Each batch is expected to be non-empty at this point, see the assert above.
                searcher->search(tptr, 1, &indices, NULL);

                const auto mnn_i = indices.front();
                const auto correct_ptr = my_big_buffer.data() + sanisizer::product_unsafe<std::size_t>(my_num_dim, my_mnns.target_mnns[mnn_i]);
                for (std::size_t d = 0; d < my_num_dim; ++d) {
                    tptr[d] += correct_ptr[d];
                }

                my_target_meta_batch[i] = my_meta_batch_assignments[my_mnns.ref_mnns[mnn_i]];
            }
        });

        // Distributing observations of the target metabatch to one of the reference metabatches, based on its closest MNN pair.
        // We don't need to do this at the last step, unless we're testing and we want to check the output.
        if (num_refs > 1 || test) {
            for (auto& reass : my_reassignments) {
                reass.clear();
            }
            for (I<decltype(num_target)> i = 0; i < num_target; ++i) {
                my_reassignments[my_target_meta_batch[i]].push_back(my_target_ids[i]);
            }

            // We need to create the NN indices of the redistributed observations in each remaining metabatch. 
            // We organize our redistributed observations so that we can operate on contiguous chunks of 'big_buffer' within each thread.
            // False sharing should not be a major issue as there aren't many boundaries between threads at which contention could occur.
            my_redist_offsets.clear();
            Index_ sofar = 0;
            for (I<decltype(num_refs)> b = 0; b < num_refs; ++b) {
                const auto& rem = my_reassignments[b];
                my_redist_offsets.push_back(sofar);
                sofar += rem.size(); // known to NOT overflow as sofar <= num_total.
                for (const auto r : rem) {
                    my_meta_batch_assignments[r] = b;
                }
            }

            parallelize(my_num_threads, num_refs, [&](const int, const I<decltype(num_refs)> start, const I<decltype(num_refs)> length) -> void {
                for (BatchIndex b = start, end = start + length; b < end; ++b) {
                    auto& reass = my_reassignments[b];
                    if (reass.empty()) {
                        continue;
                    }
                    // Construct 'subdex' first before moving it into the metabatch, to avoid problems from also moving 'reass' in the same call.
                    auto storage = my_big_buffer.data() + sanisizer::product_unsafe<std::size_t>(my_redist_offsets[b], my_num_dim);
                    auto subdex = subset_and_index(my_num_dim, reass, my_corrected, my_builder, storage);
                    my_meta_batches[b].corrected.emplace_back(std::move(subdex), std::move(reass));
                }
            });
        }

        return num_refs > 1;
    }

public:
    void merge() {
        while (next(false)) {}
    }
};

}

#endif
