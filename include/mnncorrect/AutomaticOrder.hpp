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
#include "find_local_centers.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
Float_ compute_total_variance(std::size_t ndim, Index_ nobs, const Float_* values, std::vector<Float_>& mbuffer, bool as_rss) {
    std::fill(mbuffer.begin(), mbuffer.end(), 0);

    Float_ total = 0;
    for (Index_ i = 0; i < nobs; ++i) {
        for (std::size_t d = 0; d < ndim; ++d) {
            auto curval = values[d];
            auto& curmean = mbuffer[d];
            Float_ delta = curval - curmean;
            curmean += delta/(i + 1);
            total += delta * (curval -  curmean);
        }
        values += ndim;
    }

    if (!as_rss) {
        total /= nobs - 1;
    }
    return total;
}

template<typename Index_, typename Float_>
std::vector<Float_> compute_total_variances(std::size_t ndim, const std::vector<Index_>& nobs, const std::vector<const Float_*>& batches, bool as_rss, int nthreads) {
    BatchIndex nbatches = nobs.size();
    std::vector<Float_> vars(nbatches);
    parallelize(nthreads, nbatches, [&](int, BatchIndex start, BatchIndex length) -> void {
        std::vector<Float_> mean_buffer(ndim);
        for (BatchIndex b = start, end = start + length; b < end; ++b) {
            vars[b] = compute_total_variance<Float_>(ndim, nobs[b], batches[b], mean_buffer, as_rss);
        }
    });

    return vars;
}

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
        my_nobs(nobs), 
        my_batches(batches),
        my_builder(builder),
        my_indices(my_batches.size()),
        my_num_neighbors(num_neighbors),
        my_neighbors_ref(my_batches.size()), 
        my_neighbors_target(my_batches.size()), 
        my_corrected(corrected),
        my_nthreads(nthreads)
    {
        auto nbatches = my_nobs.size();
        if (nbatches != my_batches.size()) {
            throw std::runtime_error("length of 'nobs' and 'batches' must be equal");
        }
        if (nbatches == 0) {
            return;
        }

        parallelize(nthreads, nbatches, [&](int, BatchIndex start, BatchIndex length) -> void {
            for (BatchIndex b = start, end = start + length; b < end; ++b) {
                my_indices[b] = my_builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(ndim, my_nobs[b], my_batches[b]));
            }
        });

        // Different policies to pick the first batch. The default is to use
        // the first input batch, so first == Input is already covered.
        BatchIndex ref = 0;
        if (ref_policy == ReferencePolicy::MAX_SIZE) {
            ref = std::max_element(my_nobs.begin(), my_nobs.end()) - my_nobs.begin();
        } else if (ref_policy == ReferencePolicy::MAX_VARIANCE || ref_policy == ReferencePolicy::MAX_RSS) {
            bool as_rss = ref_policy == ReferencePolicy::MAX_RSS;
            std::vector<Float_> vars = compute_total_variances(my_ndim, my_nobs, my_batches, as_rss, my_nthreads);
            ref = std::max_element(vars.begin(), vars.end()) - vars.begin();
        }

        const Index_ refnum = my_nobs[ref];
        const Float_* refdata = my_batches[ref];
        std::copy_n(refdata, ndim * static_cast<std::size_t>(refnum), my_corrected); // cast to size_t to avoid overflow.
        my_ncorrected += refnum;
        my_order.push_back(ref);

        for (BatchIndex b = 0; b < nbatches; ++b) {
            if (b == ref) {
                continue;
            }
            my_remaining.insert(b);
            my_neighbors_target[b] = quick_find_nns(my_nobs[b], my_batches[b], *my_indices[ref], my_num_neighbors, my_nthreads);
            my_neighbors_ref[b] = quick_find_nns(refnum, refdata, *my_indices[b], my_num_neighbors, my_nthreads);
        }
    }

protected:
    std::size_t my_ndim;
    const std::vector<Index_>& my_nobs;
    const std::vector<const Float_*>& my_batches;

    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& my_builder;
    std::vector<std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > > my_indices;

    int my_num_neighbors;
    std::vector<NeighborSet<Index_, Float_> > my_neighbors_ref, my_neighbors_target;

    Float_* my_corrected;
    Index_ my_ncorrected = 0;
    NeighborSet<Index_, Float_> my_corrected_neighbors;
    std::vector<Index_> my_corrected_centers;

    std::vector<BatchIndex> my_order;
    std::unordered_set<BatchIndex> my_remaining;
    std::vector<unsigned long long> my_num_pairs; // at least 64 bits to guarantee storage of many pairs.

    int my_nthreads;

protected:
    template<bool purge_ = true>
    void update(BatchIndex latest) {
        my_order.push_back(latest);

        auto lat_num = my_nobs[latest]; 
        Index_ previous_ncorrected = my_ncorrected;
        my_ncorrected += lat_num;

        if constexpr(purge_) { // try to free some memory if there are many batches.
            my_neighbors_ref[latest].clear();
            my_neighbors_ref[latest].shrink_to_fit(); 
        }

        my_remaining.erase(latest);
        if (my_remaining.empty()) {
            return;
        }

        auto& lat_index = my_indices[latest];
        const Float_* lat_data = my_corrected + static_cast<std::size_t>(previous_ncorrected) * my_ndim; // cast to avoid overflow.
        lat_index = my_builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(my_ndim, lat_num, lat_data));

        // Updating cross-batch NN hits for the remaining (unprocessed) batches.
        for (auto b : my_remaining) {
            auto& rem_ref_neighbors = my_neighbors_ref[b];
            rem_ref_neighbors.resize(my_ncorrected);
            const auto& rem_index = my_indices[b];
            quick_find_nns(lat_num, lat_data, *rem_index, my_num_neighbors, my_nthreads, rem_ref_neighbors, previous_ncorrected);
            quick_fuse_nns(my_neighbors_target[b], my_batches[b], *lat_index, my_num_neighbors, my_nthreads, previous_ncorrected);
        }

        // Updating self-batch NN hits for the processed batches, in order to find new centers.
        my_corrected_neighbors.resize(my_ncorrected);
        quick_find_nns(*lat_index, my_num_neighbors, my_nthreads, my_corrected_neighbors, previous_ncorrected);
        Index_ sofar = 0;
        for (decltype(my_order.size()) i = 0, end = my_order.size() - 1; i < end; ++i) {
            auto b = my_order[i];
            auto batch_num = my_nobs[b];
            quick_fuse_nns(sofar, batch_num, my_corrected_neighbors, my_corrected, *lat_index, my_num_neighbors, my_nthreads, previous_ncorrected);
            quick_fuse_nns(previous_ncorrected, lat_num, my_corrected_neighbors, my_corrected, *(my_indices[b]), my_num_neighbors, my_nthreads, sofar);
            sofar += batch_num;
        }
        find_local_centers(my_corrected_neighbors, my_corrected_centers);
    }

protected:
    std::pair<BatchIndex, MnnPairs<Index_> > choose() {
        // Splitting up the remaining batches across threads. The idea is that
        // each thread reports the maximum among its assigned batches, and then
        // we compare the number of MNN pairs across the per-thread maxima.
        auto nremaining = my_remaining.size();
        BatchIndex per_thread = (nremaining / my_nthreads) + (nremaining % my_nthreads > 0);

        auto it = my_remaining.begin();
        std::vector<decltype(it)> partitions;
        partitions.reserve(my_nthreads + 1);

        BatchIndex counter = 0;
        for (auto it = my_remaining.begin(); it != my_remaining.end(); ++it) { // hashsets don't have random access iterators, so we ned to manually iterate.
            if (counter == 0) {
                partitions.push_back(it);
                if (partitions.size() == static_cast<decltype(partitions.size())>(my_nthreads)) {
                    break;
                }
            }
            ++counter;
            if (counter == per_thread) {
                counter = 0;
            }
        }

        int actual_nthreads = partitions.size(); // avoid having to check for threads that don't do any work.
        std::vector<MnnPairs<Index_> > collected(actual_nthreads);
        std::vector<BatchIndex> best(actual_nthreads);

        partitions.push_back(my_remaining.end()); // to easily check for the terminator in the last thread.

        // This should be a trivial allocation when njobs = nthreads.
        parallelize(actual_nthreads, actual_nthreads, [&](int, int start, int length) -> void {
            for (int t = start, end = start + length; t < end; ++t) {
                // Within each thread, scanning for the maximum among the allocated batches.
                auto startIt = partitions[t], endIt = partitions[t + 1];

                MnnPairs<Index_> best_pairs;
                BatchIndex chosen = *startIt;

                while (startIt != endIt) {
                    auto b = *startIt;
                    auto& nnref = my_neighbors_ref[b];
                    auto tmp = find_mutual_nns(nnref, my_neighbors_target[b]);

                    /* If a cell in the reference set is not in an MNN pair
                     * with an unmerged batch at iteration X, it can never be
                     * in an MNN pair at iteration X+1 or later. This is based
                     * on the fact that the corrected coordinates of that cell
                     * will not change across iterations, nor do the
                     * coordinates of the uncorrected batch; and the only
                     * change across iterations is the addition of cells from
                     * the newly corrected batches to the reference, which
                     * would not cause the non-MNN cell in the existing
                     * reference to suddenly become an MNN (and if anything,
                     * would compete in the NN search).
                     *
                     * As such, we can free the memory stores for those
                     * never-MNN cells, which should substantially lower memory
                     * consumption when there are many batches. 
                     */
                    {
                        auto bsize = nnref.size();
                        std::vector<unsigned char> present(bsize);
                        for (const auto& x : tmp.matches) {
                            for (auto y : x.second) {
                                present[y] = 1;
                            }
                        }

                        for (decltype(bsize) i = 0; i < bsize; ++i) {
                            auto& current = nnref[i];
                            if (!present[i] && !current.empty()) {
                                current.clear();
                                current.shrink_to_fit();
                            }
                        }
                    }

                    // Now, deciding if it's the best.
                    if (tmp.num_pairs > best_pairs.num_pairs) {
                        best_pairs = std::move(tmp);
                        chosen = b;
                    }
                    ++startIt;
                }

                collected[t] = std::move(best_pairs);
                best[t] = chosen;
            }
        });

        // Scanning across threads for the maximum. (We assume that results
        // from at least one thread are available.) 
        BatchIndex best_index = 0;
        for (int t = 1; t < actual_nthreads; ++t) {
            if (collected[t].num_pairs > collected[best_index].num_pairs) {
                best_index = t;
            }
        }

        return std::pair<BatchIndex, MnnPairs<Index_> >(best[best_index], std::move(collected[best_index]));
    }

public:
    void run(Float_ nmads, int robust_iterations, double robust_trim) {
        std::vector<Index_> target_centers;
        NeighborSet<Index_, Float_> target_neighbors;

        while (my_remaining.size()) {
            auto output = choose();
            auto target = output.first;
            auto target_num = my_nobs[target];
            auto target_data = my_batches[target];

            target_neighbors.resize(target_num);
            quick_find_nns<Index_, Float_>(*(my_indices[target]), my_num_neighbors, my_nthreads, target_neighbors, 0);
            find_local_centers(target_neighbors, target_centers);

            correct_target(
                my_ndim, 
                my_ncorrected, 
                my_corrected, 
                my_corrected_centers,
                target_num, 
                target_data, 
                target_centers,
                output.second, 
                my_builder,
                my_num_neighbors,
                nmads,
                robust_iterations,
                robust_trim,
                my_corrected + static_cast<std::size_t>(my_ncorrected) * my_ndim, // cast to size_t to avoid overflow.
                my_nthreads
            );

            update(output.first);
            my_num_pairs.push_back(output.second.num_pairs);
        }
    }

    const auto& get_order() const { return my_order; }

    const auto& get_num_pairs() const { return my_num_pairs; }
};

}

}

#endif
