#ifndef MNNCORRECT_COMPUTE_HPP
#define MNNCORRECT_COMPUTE_HPP

#include <algorithm>
#include <vector>
#include <numeric>

#include "knncolle/knncolle.hpp"

#include "AutomaticOrder.hpp"
#include "CustomOrder.hpp"
#include "Options.hpp"
#include "restore_order.hpp"

/**
 * @file MnnCorrect.hpp
 *
 * @brief Batch correction using mutual nearest neighbors.
 */

namespace mnncorrect {

/**
 * @brief Correction details from `compute()`.
 */
struct Details {
    /**
     * @cond
     */
    Details() {}
    Details(std::vector<size_t> mo, std::vector<size_t> np) : merge_order(std::move(mo)), num_pairs(std::move(np)) {}
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
Details compute(size_t ndim, const std::vector<size_t>& nobs, const std::vector<const Float_*>& batches, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    auto reference_policy = options.reference_policy;
    auto num_neighbors = options.num_neighbors;
    auto nobs_cap = options.nobs_cap;
    auto nthreads = options.num_threads;

    auto builder = options.builder;
    if (!builder) {
        builder.reset(new knncolle::VptreeBuilder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_>);
    }

    if (!options.order.empty()) {
        CustomOrder<Index, Float, decltype(builder)> runner(ndim, nobs, batches, output, std::move(builder), num_neighbors, options.order.data(), nobs_cap, nthreads);
        runner.run(num_mads, robust_iterations, robust_trim);
        return Details(runner.get_order(), runner.get_num_pairs());

    } else if (automatic_order) {
        AutomaticOrder<Index, Float, decltype(builder)> runner(ndim, nobs, batches, output, std::move(builder), num_neighbors, reference_policy, nobs_cap, nthreads);
        runner.run(num_mads, robust_iterations, robust_trim);
        return Details(runner.get_order(), runner.get_num_pairs());

    } else {
        std::vector<size_t> trivial_order(nobs.size());
        std::iota(trivial_order.begin(), trivial_order.end(), 0);
        CustomOrder<Index, Float, decltype(builder)> runner(ndim, nobs, batches, std::move(output), builder, num_neighbors, trivial_order.data(), nobs_cap, nthreads);
        runner.run(num_mads, robust_iterations, robust_trim);
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
 * @tparam Index Integer type for the observation indices.
 * @tparam Float Floating point type for the data and distances.
 *
 * @param ndim Number of dimensions.
 * @param nobs Vector of length equal to the number of batches.
 * Each entry contains the number of observations in each batch.
 * @param[in] batches Vector of length equal to the number of batches.
 * Each entry points to a column-major dimension-by-observation array containing the uncorrected data for each batch.
 * @param[out] output Pointer to an array of length equal to the product of `ndim` with the sum of `nobs`.
 * This is used to store the corrected values from all batches.
 * @param[in] order Pointer to an array of indices specifying the merge order.
 * For example, the first entry contains the index of the batch in `batches` to be used as the reference,
 * the second entry specifies the batch to be merged first, and so on.
 * All entries should be unique and lie in $[0, N)$ where $N$ is the number of batches.
 * If omitted, the setting of `set_automatic_order()` is used.
 *
 * @return `output` is filled contiguously with the corrected values from successive batches,
 * i.e., the first batch takes `nobs[0] * ndim` elements, the second batch takes the next `nobs[1] * ndim` elements and so on.
 * Filling is done column-major, i.e., values for the same observations are adjacent.
 * A `Details` object is returned containing statistics about the merge process.
 */
template<typename Dim_, typename Index_, typename Float_>
Details compute(size_t ndim, const std::vector<size_t>& nobs, const std::vector<const Float_*>& batches, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    auto stats = internal::compute(ndim, nobs, batches, output, order);
    restore_order(ndim, stats.merge_order, nobs, output);
    return stats;
}

/**
 * A convenience overload to merge contiguous batches contained in the same array.
 *
 * @param ndim Number of dimensions.
 * @param nobs Vector of length equal to the number of batches.
 * Each entry contains the number of observations in each batch.
 * @param[in] input Pointer to a column-major dimension-by-observation array containing the uncorrected data for all batches.
 * Observations from the same batch are assumed to be contiguous,
 * i.e., the first `nobs[0]` columns contain observations from the first batch,
 * the next `nobs[1]` columns contain observations for the second batch, and so on.
 * @param[out] output Pointer to an array of length equal to the product of `ndim` with the sum of `nobs`.
 * This is used to store the corrected values from all batches.
 * @param[in] order Pointer to an array of indices specifying the merge order.
 * For example, the first entry contains the index of the batch in `nobs` to be used as the reference,
 * the second entry specifies the batch to be merged first, and so on.
 * All entries should be unique and lie in $[0, N)$ where $N$ is the number of batches.
 * If omitted, the setting of `set_automatic_order()` is used.
 *
 * @return `output` is filled contiguously with the corrected values from successive batches.
 * A `Details` object is returned containing statistics about the merge process.
 */
template<typename Dim_, typename Index_, typename Float_>
Details compute(size_t ndim, const std::vector<size_t>& nobs, const Float* input, Float* output, const Options<Dim_, Index_, Float_>& options) {
    std::vector<const Float_*> batches;
    batches.reserve(nobs.size());
    for (auto n : nobs) {
        batches.push_back(input);
        input += n * ndim; // already size_t's, so no need to worry about overflow.
    }
    return compute(ndim, nobs, batches, output, order);
}

/**
 * Merge batches where observations are arbitrarily ordered in the same array.
 *
 * @tparam Batch Integer type for the batch IDs.
 *
 * @param ndim Number of dimensions.
 * @param nobs Number of observations across all batches.
 * @param[in] input Pointer to a column-major dimension-by-observation array containing the uncorrected data for all batches.
 * @param[in] batch Pointer to an array of length `nobs` containing the batch ID for each observation.
 * IDs should be zero-indexed and lie within $[0, N)$ where $N$ is the number of unique batches.
 * @param[out] output Pointer to an array of length equal to the product of `ndim` with the sum of `nobs`.
 * This is used to store the corrected values from all batches.
 * @param[in] order Pointer to an array specifying the merge order.
 * Entries should correspond to levels of `batch`; the first entry specifies the batch to use as the reference,
 * the second entry specifies the first batch to merge, and so on.
 * All entries should be unique and lie in $[0, N)$ where $N$ is the number of batches.
 * If omitted, the setting of `set_automatic_order()` is used.
 *
 * @return `output` is filled with the corrected values from successive batches.
 * The order of observations in `output` is the same as that in the `input` (i.e., not necessarily contiguous).
 * A `Details` object is returned containing statistics about the merge process.
 */
template<typename Dim_, typename Index_, typename Float_, typename Batch_>
Details compute(size_t ndim, size_t nobs, const Float_* input, const Batch_* batch, Float_* output, const Options<Dim_, Index_, Float_>& options) {
    const size_t nbatches = (nobs ? static_cast<size_t>(*std::max_element(batch, batch + nobs)) + 1 : 0);
    std::vector<size_t> sizes(nbatches);
    for (size_t o = 0; o < nobs; ++o) {
        ++sizes[batch[o]];
    }

    // Avoiding the need to allocate a temporary buffer
    // if we're already dealing with contiguous batches.
    bool already_sorted = true;
    for (size_t o = 1; o < nobs; ++o) {
       if (batch[o] < batch[o-1]) {
           already_sorted = false;
           break;
       }
    }
    if (already_sorted) {
        return compute(ndim, sizes, input, output, options);
    }

    size_t accumulated = 0;
    std::vector<size_t> offsets(nbatches);
    for (size_t b = 0; b < nbatches; ++b) {
        offsets[b] = accumulated;
        accumulated += sizes[b];
    }

    // Dumping everything by order into another vector.
    std::vector<Float_> tmp(ndim * nobs);
    std::vector<const Float_*> ptrs(nbatches, tmp.data());
    for (size_t b = 0; b < nbatches; ++b) {
        ptrs[b] += offsets[b] * ndim;
    }
    for (size_t o = 0; o < nobs; ++o) {
        auto current = input + o * ndim;
        auto& offset = offsets[batch[o]];
        auto destination = tmp.data() + ndim * offset; // already size_t's, so no need to cast to avoid overflow.
        std::copy_n(current, ndim, destination);
        ++offset;
    }

    auto stats = internal::compute(ndim, sizes, ptrs, output, options);
    restore_order(ndim, stats.merge_order, sizes, batch, output);
    return stats;
}

}

#endif
