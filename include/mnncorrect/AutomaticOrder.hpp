#ifndef MNNCORRECT_AUTOMATIC_ORDER_HPP
#define MNNCORRECT_AUTOMATIC_ORDER_HPP

#include <algorithm>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <vector>

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_mutual_nns.hpp"
#include "fuse_nn_results.hpp"
#include "correct_target.hpp"
#include "ReferencePolicy.hpp"

namespace mnncorrect {

namespace internal {

template<typename Float_>
Float_ compute_total_variance(size_t ndim, size_t nobs, const Float_* values, std::vector<Float_>& mbuffer, bool as_rss) {
    std::fill(mbuffer.begin(), mbuffer.end(), 0);

    Float_ total = 0;
    for (size_t i = 0; i < nobs; ++i) {
        for (size_t d = 0; d < ndim; ++d) {
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

template<typename Dim_, typename Index_, typename Float_>
class AutomaticOrder {
public:
    AutomaticOrder(
        size_t ndim,
        std::vector<size_t> nobs,
        std::vector<const Float_*> batches, 
        Float_* corrected,
        std::unique_ptr<knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_> > builder,
        int num_neighbors, 
        ReferencePolicy ref_policy, 
        size_t nobs_cap, 
        int nthreads)
    :
        my_ndim(ndim), 
        my_nobs(std::move(nobs)), 
        my_batches(std::move(batches)),
        my_builder(std::move(builder)),
        my_indices(my_batches.size()),
        my_num_neighbors(num_neighbors),
        my_neighbors_ref(my_batches.size()), 
        my_neighbors_target(my_batches.size()), 
        my_corrected(corrected),
        my_nobs_cap(nobs_cap),
        my_nthreads(nthreads)
    {
        size_t nbatches = my_nobs.size();
        if (nbatches != my_batches.size()) {
            throw std::runtime_error("length of 'nobs' and 'batches' must be equal");
        }
        if (nbatches == 0) {
            return;
        }

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel num_threads(my_nthreads)
#endif
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for (size_t b = 0; b < nbatches; ++b) {
#else
        MNNCORRECT_CUSTOM_PARALLEL(nbatches, [&](size_t start, size_t end) -> void {
            for (size_t b = start; b < end; ++b) {
#endif

                my_indices[b] = my_builder->build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(ndim, my_nobs[b], my_batches[b]));

#ifndef MNNCORRECT_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, my_nthreads);
#endif

        // Different policies to pick the first batch. The default is to use
        // the first input batch, so first == Input is already covered.
        size_t ref = 0;
        if (ref_policy == ReferencePolicy::MAX_SIZE) {
            ref = std::max_element(my_nobs.begin(), my_nobs.end()) - my_nobs.begin();

        } else if (ref_policy == ReferencePolicy::MAX_VARIANCE || ref_policy == ReferencePolicy::MAX_RSS) {
            bool as_rss = ref_policy == ReferencePolicy::MAX_RSS;
            std::vector<Float_> vars(nbatches);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
            #pragma omp parallel num_threads(my_nthreads)
#endif
            {
#else
            MNNCORRECT_CUSTOM_PARALLEL(nbatches, [&](size_t start, size_t end) -> void {
#endif

                std::vector<Float_> mean_buffer(my_ndim);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
                #pragma omp for
#endif
                for (size_t b = 0; b < nbatches; ++b) {
#else
                for (size_t b = start; b < end; ++b) {
#endif

                    vars[b] = compute_total_variance<Float_>(my_ndim, my_nobs[b], my_batches[b], mean_buffer, as_rss);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, my_nthreads);
#endif

            ref = std::max_element(vars.begin(), vars.end()) - vars.begin();
        }

        const size_t refnum = my_nobs[ref];
        const Float_* refdata = my_batches[ref];
        std::copy_n(refdata, ndim * refnum, my_corrected);
        my_ncorrected += refnum;
        my_order.push_back(ref);

        for (size_t b = 0; b < nbatches; ++b) {
            if (b == ref) {
                continue;
            }
            my_remaining.insert(b);
            my_neighbors_target[b] = quick_find_nns(my_nobs[b], my_batches[b], *my_indices[ref], my_num_neighbors, my_nthreads);
            my_neighbors_ref[b] = quick_find_nns(refnum, refdata, *my_indices[b], my_num_neighbors, my_nthreads);
        }
        return;
    }

protected:
    int my_ndim;
    std::vector<size_t> my_nobs;
    std::vector<const Float_*> my_batches;

    std::unique_ptr<knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_> > my_builder;
    std::vector<std::shared_ptr<knncolle::Prebuilt<Dim_, Index_, Float_> > > my_indices;

    int my_num_neighbors;
    std::vector<NeighborSet<Index_, Float_> > my_neighbors_ref;
    std::vector<NeighborSet<Index_, Float_> > my_neighbors_target;

    Float_* my_corrected;
    size_t my_ncorrected = 0;

    std::vector<size_t> my_order;
    std::unordered_set<size_t> my_remaining;
    std::vector<size_t> my_num_pairs;

    size_t my_nobs_cap;
    int my_nthreads;

protected:
    template<bool purge_ = true>
    void update(size_t latest) {
        size_t lat_num = my_nobs[latest]; 
        const Float_* lat_data = my_corrected + my_ncorrected * my_ndim; // these are already size_t's, so no need to cast to avoid overflow.

        my_order.push_back(latest);
        auto previous_ncorrected = my_ncorrected;
        my_ncorrected += lat_num;

        if constexpr(purge_) { // try to free some memory if there are many batches.
            my_neighbors_ref[latest].clear();
            my_neighbors_ref[latest].shrink_to_fit(); 
            my_indices[latest].reset();
        }

        my_remaining.erase(latest);
        if (my_remaining.empty()) {
            return;
        }

        auto lat_index = my_builder->build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(my_ndim, lat_num, lat_data));
        for (auto b : my_remaining) {
            auto& rem_ref_neighbors = my_neighbors_ref[b];
            rem_ref_neighbors.resize(my_ncorrected);
            const auto& rem_index = my_indices[b];

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
            #pragma omp parallel num_threads(my_nthreads)
#endif
            {
#else
            MNNCORRECT_CUSTOM_PARALLEL(lat_num, [&](size_t start, size_t end) -> void {
#endif

                std::vector<Index_> indices;
                std::vector<Float_> distances;
                auto rem_searcher = rem_index->initialize();

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
                #pragma omp for
#endif
                for (size_t l = 0; l < lat_num; ++l) {
#else
                for (size_t l = start; l < end; ++l) {
#endif

                    auto lat_ptr = lat_data + my_ndim * l; // already size_t's, so no need to cast to avoid overflow.
                    rem_searcher->search(lat_ptr, my_num_neighbors, &indices, &distances);
                    fill_pair_vector(indices, distances, rem_ref_neighbors[previous_ncorrected + l]);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, my_nthreads);
#endif 

            const size_t rem_num = my_nobs[b];
            const Float_* rem_data = my_batches[b];
            auto& rem_target_neighbors = my_neighbors_target[b];

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
            #pragma omp parallel num_threads(my_nthreads)
#endif
            {
#else
            MNNCORRECT_CUSTOM_PARALLEL(rem_num, [&](size_t start, size_t end) -> void {
#endif

                std::vector<Index_> indices;
                std::vector<Float_> distances;
                auto lat_searcher = lat_index->initialize();
                std::vector<std::pair<Index_, Float_> > search_buffer, fuse_buffer;

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
                #pragma omp for
#endif
                for (size_t r = 0; r < rem_num; ++r) {
#else
                for (size_t r = start; r < end; ++r) {
#endif

                    auto rem_ptr = rem_data + my_ndim * r; // already size_t's, no need to cast to avoid overflow.
                    lat_searcher->search(rem_ptr, my_num_neighbors, &indices, &distances);
                    fill_pair_vector(indices, distances, search_buffer);

                    auto& current_target_neighbors = rem_target_neighbors[r];
                    fuse_nn_results(current_target_neighbors, search_buffer, my_num_neighbors, fuse_buffer, static_cast<Index_>(previous_ncorrected));
                    fuse_buffer.swap(current_target_neighbors);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, my_nthreads);
#endif
        }

        return;
    }

protected:
    std::pair<size_t, MnnPairs<Index_> > choose() {
        // Splitting up the remaining batches across threads. The idea is that
        // each thread reports the maximum among its assigned batches, and then
        // we compare the number of MNN pairs across the per-thread maxima.
        size_t nremaining = my_remaining.size();
        size_t per_thread = (nremaining / my_nthreads) + (nremaining % my_nthreads > 0);

        auto it = my_remaining.begin();
        std::vector<decltype(it)> partitions;
        partitions.reserve(my_nthreads + 1);

        size_t counter = 0;
        for (auto it = my_remaining.begin(); it != my_remaining.end(); ++it) { // hashsets don't have random access iterators, so we ned to manually iterate.
            if (counter == 0) {
                partitions.push_back(it);
                if (partitions.size() == static_cast<size_t>(my_nthreads)) {
                    break;
                }
            }
            ++counter;
            if (counter == per_thread) {
                counter = 0;
            }
        }

        size_t actual_nthreads = partitions.size(); // avoid having to check for threads that don't do any work.
        std::vector<MnnPairs<Index_> > collected(actual_nthreads);
        std::vector<size_t> best(actual_nthreads);

        partitions.push_back(my_remaining.end()); // to easily check for the terminator in the last thread.

        // This should be a trivial allocation when njobs = nthreads.
#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel num_threads(actual_nthreads)
#endif
        {
#ifdef _OPENMP
            #pragma omp for 
#endif
            for (size_t t = 0; t < actual_nthreads; ++t) {
#else
        MNNCORRECT_CUSTOM_PARALLEL(actual_nthreads, [&](size_t start, size_t end) -> void {
            for (size_t t = start; t < end; ++t) {
#endif

                // Within each thread, scanning for the maximum among the allocated batches.
                auto startIt = partitions[t], endIt = partitions[t + 1];

                MnnPairs<Index_> best_pairs;
                size_t chosen = *startIt;

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
                        std::vector<unsigned char> present(nnref.size());
                        for (const auto& x : tmp.matches) {
                            for (auto y : x.second) {
                                present[y] = 1;
                            }
                        }

                        for (size_t i = 0, end = nnref.size(); i < end; ++i) {
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

#ifndef MNNCORRECT_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, actual_nthreads);
#endif

        // Scanning across threads for the maximum. (We assume that results
        // from at least one thread are available.) 
        size_t best_index = 0;
        for (size_t t = 1; t < actual_nthreads; ++t) {
            if (collected[t].num_pairs > collected[best_index].num_pairs) {
                best_index = t;
            }
        }

        return std::pair<size_t, MnnPairs<Index_> >(best[best_index], std::move(collected[best_index]));
    }

public:
    void run(Float_ nmads, int robust_iterations, double robust_trim) {
        while (my_remaining.size()) {
            auto output = choose();
            auto target = output.first;
            auto target_num = my_nobs[target];
            auto target_data = my_batches[target];

            correct_target(
                my_ndim, 
                my_ncorrected, 
                my_corrected, 
                target_num, 
                target_data, 
                output.second, 
                my_builder,
                my_num_neighbors,
                nmads,
                robust_iterations,
                robust_trim,
                my_corrected + my_ncorrected * my_ndim, // already size_t's, so no need to coerce.
                my_nobs_cap,
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
