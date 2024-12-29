#ifndef MNNCORRECT_COMPUTE_HPP
#define MNNCORRECT_COMPUTE_HPP

#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstdint>

#include "knncolle/knncolle.hpp"

#include "AutomaticOrder.hpp"
#include "CustomOrder.hpp"
#include "Options.hpp"
#include "restore_order.hpp"

/**
 * @file compute.hpp
 *
 * @brief Compute the MNN correction. 
 */

namespace mnncorrect {

/**
 * @brief Correction details from `compute()`.
 */
struct Details {
    /**
     * @cond
     */
    Details() = default;

    Details(std::vector<size_t> merge_order, std::vector<size_t> num_pairs) : merge_order(std::move(merge_order)), num_pairs(std::move(num_pairs)) {}
    /**
     * @endcond
     */

    /**
     * Order in which batches are merged.
     * The first entry is the index/ID of the batch used as the reference,
     * and the remaining entries are merged to the reference in the listed order.
     */
    std::vector<size_t> merge_order;

    /**
     * Number of MNN pairs identified at each merge step.
     * This is of length one less than `merge_order`.
     */
    std::vector<size_t> num_pairs;
};

/**
 * @cond
 */
namespace internal {

template<typename Dim_, typename Index_, typename Float_>
Details compute(size_t num_dim, const std::vector<size_t>& num_obs, const std::vector<const Float_*>& batches, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    auto builder = options.builder;
    if (!builder) {
        builder.reset(new knncolle::VptreeBuilder<knncolle::EuclideanDistance, knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_>);
    }

    if (!options.order.empty()) {
        CustomOrder<Dim_, Index_, Float_> runner(num_dim, num_obs, batches, output, *builder, options.num_neighbors, options.order, options.mass_cap, options.num_threads);
        runner.run(options.num_mads, options.robust_iterations, options.robust_trim);
        return Details(runner.get_order(), runner.get_num_pairs());

    } else if (options.automatic_order) {
        AutomaticOrder<Dim_, Index_, Float_> runner(num_dim, num_obs, batches, output, *builder, options.num_neighbors, options.reference_policy, options.mass_cap, options.num_threads);
        runner.run(options.num_mads, options.robust_iterations, options.robust_trim);
        return Details(runner.get_order(), runner.get_num_pairs());

    } else {
        std::vector<size_t> trivial_order(num_obs.size());
        std::iota(trivial_order.begin(), trivial_order.end(), 0);
        CustomOrder<Dim_, Index_, Float_> runner(num_dim, num_obs, batches, output, *builder, options.num_neighbors, trivial_order, options.mass_cap, options.num_threads);
        runner.run(options.num_mads, options.robust_iterations, options.robust_trim);
        return Details(std::move(trivial_order), runner.get_num_pairs());
    }
}

}
/**
 * @endcond
 */

/**
 * Batch correction using mutual nearest neighbors.
 *
 * This function implements a variant of the MNN correction method described by Haghverdi _et al._ (2018).
 * Two cells from different batches can form an MNN pair if they each belong in each other's set of nearest neighbors.
 * The MNN pairs are assumed to represent cells from corresponding subpopulations across the two batches.
 * Any differences in location between the paired cells can be interpreted as the batch effect and targeted for removal.
 *
 * We consider one batch to be the "reference" and the other to be the "target", where the aim is to correct the latter to the (unchanged) former. 
 * For each observation in the target batch, we find the closest MNN pairs (based on the locations of the paired observation in the same batch)
 * and we compute a robust average of the correction vectors involving those pairs.
 * This average is used to obtain a single correction vector that is applied to the target observation to obtain corrected values.
 *
 * Each MNN pair's correction vector is computed between the "center of mass" locations for the paired observations.
 * The center of mass for each observation is defined as a robust average of a subset of neighboring observations from the same batch.
 * Robustification is performed by iterations of trimming of observations that are furthest from the mean.
 * In addition, we explicitly remove observations that are more than a certain distance from the observation in the MNN pair.
 *
 * @see
 * Haghverdi L et al. (2018).
 * Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors.
 * _Nature Biotech._ 36, 421-427
 *
 * @tparam Dim_ Integer type for the dimensions of the neighbor search. 
 * @tparam Index_ Integer type for the observation index of the neighbor search. 
 * @tparam Float_ Floating-point type for the distances in the neighbor search.
 *
 * @param num_dim Number of dimensions.
 * @param num_obs Vector of length equal to the number of batches.
 * The `i`-th entry contains the number of observations in batch `i`.
 * @param[in] batches Vector of length equal to the number of batches.
 * The `i`-th entry points to a column-major dimension-by-observation array containing the uncorrected data for batch `i`,
 * where the number of rows is equal to `num_dim` and the number of columns is equal to `num_obs[i]`.
 * @param[out] output Pointer to an array containing a column-major matrix with number of rows equal to `num_dim` and number of columns equal to the sum of `num_obs`.
 * On output, the first `num_obs[0]` columns contain the corrected values of the first batch, 
 * the second `num_obs[1]` columns contain the corrected values of the second batch, and so on.
 * @param options Further options.
 *
 * @return Statistics about the merge process.
 */
template<typename Dim_, typename Index_, typename Float_>
Details compute(size_t num_dim, const std::vector<size_t>& num_obs, const std::vector<const Float_*>& batches, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    auto stats = internal::compute(num_dim, num_obs, batches, output, options);
    internal::restore_order(num_dim, stats.merge_order, num_obs, output);
    return stats;
}

/**
 * A convenience overload to merge contiguous batches contained in the same array.
 *
 * @tparam Dim_ Integer type for the dimensions of the neighbor search. 
 * @tparam Index_ Integer type for the observation index of the neighbor search. 
 * @tparam Float_ Floating-point type for the distances in the neighbor search.
 *
 * @param num_dim Number of dimensions.
 * @param num_obs Vector of length equal to the number of batches.
 * The `i`-th entry contains the number of observations in batch `i`.
 * @param[in] input Pointer to an array containing a column-major matrix of uncorrected values from all batches.
 * The number of rows is equal to `num_dim` and the number of columns is equal to the sum of `num_obs`.
 * The first `num_obs[0]` columns contain the uncorrected data for the first batch,
 * the next `num_obs[1]` columns contain observations for the second batch, and so on.
 * @param[out] output Pointer to an array containing a column-major matrix of the same dimensions as that in `input`, where the corrected values for all batches are stored.
 * On output, the first `num_obs[0]` columns contain the corrected values of the first batch, 
 * the second `num_obs[1]` columns contain the corrected values of the second batch, and so on.
 * @param options Further options.
 *
 * @return Statistics about the merge process.
 */
template<typename Dim_, typename Index_, typename Float_>
Details compute(size_t num_dim, const std::vector<size_t>& num_obs, const Float_* input, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    std::vector<const Float_*> batches;
    batches.reserve(num_obs.size());
    for (auto n : num_obs) {
        batches.push_back(input);
        input += n * num_dim; // already size_t's, so no need to worry about overflow.
    }
    return compute(num_dim, num_obs, batches, output, options);
}

/**
 * Merge batches where observations are arbitrarily ordered in the same array.
 *
 * @tparam Dim_ Integer type for the dimensions of the neighbor search. 
 * @tparam Index_ Integer type for the observation index of the neighbor search. 
 * @tparam Float_ Floating-point type for the distances in the neighbor search.
 * @tparam Batch_ Integer type for the batch IDs.
 *
 * @param num_dim Number of dimensions.
 * @param num_obs Number of observations across all batches.
 * @param[in] input Pointer to an array containing a column-major matrix of uncorrected values from all batches.
 * The number of rows is equal to `num_dim` and the number of columns is equal to `num_obs`.
 * Observations from the same batch do not need to be stored in adjacent columns.
 * @param[in] batch Pointer to an array of length `num_obs` containing the batch identity for each observation.
 * IDs should be zero-indexed and lie within \f$[0, N)\f$ where \f$N\f$ is the number of unique batches.
 * @param[out] output Pointer to an array containing a column-major matrix of the same dimensions as that in `input`, where the corrected values for all batches are stored.
 * The order of observations in `output` is the same as that in the `input`. 
 * @param options Further options.
 *
 * @return Statistics about the merge process.
 */
template<typename Dim_, typename Index_, typename Float_, typename Batch_>
Details compute(size_t num_dim, size_t num_obs, const Float_* input, const Batch_* batch, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    const size_t nbatches = (num_obs ? static_cast<size_t>(*std::max_element(batch, batch + num_obs)) + 1 : 0);
    std::vector<size_t> sizes(nbatches);
    for (size_t o = 0; o < num_obs; ++o) {
        ++sizes[batch[o]];
    }

    // Avoiding the need to allocate a temporary buffer
    // if we're already dealing with contiguous batches.
    bool already_sorted = true;
    for (size_t o = 1; o < num_obs; ++o) {
       if (batch[o] < batch[o-1]) {
           already_sorted = false;
           break;
       }
    }
    if (already_sorted) {
        return compute(num_dim, sizes, input, output, options);
    }

    size_t accumulated = 0;
    std::vector<size_t> offsets(nbatches);
    for (size_t b = 0; b < nbatches; ++b) {
        offsets[b] = accumulated;
        accumulated += sizes[b];
    }

    // Dumping everything by order into another vector.
    std::vector<Float_> tmp(num_dim * num_obs);
    std::vector<const Float_*> ptrs(nbatches);
    for (size_t b = 0; b < nbatches; ++b) {
        ptrs[b] = tmp.data() + offsets[b] * num_dim;
    }

    for (size_t o = 0; o < num_obs; ++o) {
        auto current = input + o * num_dim;
        auto& offset = offsets[batch[o]];
        auto destination = tmp.data() + num_dim * offset; // already size_t's, so no need to cast to avoid overflow.
        std::copy_n(current, num_dim, destination);
        ++offset;
    }

    auto stats = internal::compute(num_dim, sizes, ptrs, output, options);
    internal::restore_order(num_dim, stats.merge_order, sizes, batch, output);
    return stats;
}

}

#endif
