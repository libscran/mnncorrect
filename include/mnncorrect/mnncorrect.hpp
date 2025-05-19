#ifndef MNNCORRECT_HPP
#define MNNCORRECT_HPP

#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstddef>

#include "knncolle/knncolle.hpp"

#include "AutomaticOrder.hpp"
#include "restore_input_order.hpp"
#include "utils.hpp"

/**
 * @file mnncorrect.hpp
 * @brief Batch correction with mutual nearest neighbors.
 */

/**
 * @namespace mnncorrect
 * @brief Batch correction with mutual nearest neighbors.
 */
namespace mnncorrect {

/**
 * @brief Options for `compute()`.
 * @tparam Index_ Integer type for the observation indices.
 * @tparam Float_ Floating-point type for the input/output data.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 * Alternatively, it may be a `knncolle::SimpleMatrix`.
 */
template<typename Index_, typename Float_, class Matrix_ = knncolle::Matrix<Index_, Float_> >
struct Options {
    /**
     * Number of neighbors for the various search steps, primarily to identify MNN pairs.
     * This can also be interpreted as the lower bound on the number of observations in each "subpopulation". 
     * Larger values increase improve the stability of the correction, at the cost of reduced resolution when matching subpopulations across batches.
     */
    int num_neighbors = 15;

    /**
     * Number of steps for the recursive neighbor search to compute the center of mass.
     * Larger values mitigate the kissing effect but increase the risk of including inappropriately distant subpopulations into the center of mass.
     */
    int num_steps = 1;

    /**
     * Algorithm to use for building the nearest-neighbor search indices.
     * If NULL, defaults to an exact search via `knncolle::VptreeBuilder` with Euclidean distances.
     */
    std::shared_ptr<knncolle::Builder<Index_, Float_, Float_, Matrix_> > builder;

    /**
     * Policy to use to choose the merge order.
     */
    MergePolicy merge_policy = MergePolicy::RSS;

    /**
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

/**
 * @cond
 */
namespace internal {

template<typename Index_, typename Float_, class Matrix_>
void compute(std::size_t num_dim, const std::vector<Index_>& num_obs, const std::vector<const Float_*>& batches, Float_* output, const Options<Index_, Float_, Matrix_>& options) {
    auto builder = options.builder;
    if (!builder) {
        typedef knncolle::EuclideanDistance<Float_, Float_> Euclidean;
        builder.reset(new knncolle::VptreeBuilder<Index_, Float_, Float_, Matrix_, Euclidean>(std::make_shared<Euclidean>()));
    }

    AutomaticOrder<Index_, Float_, Matrix_> runner(
        num_dim,
        num_obs,
        batches,
        output,
        *builder,
        options.num_neighbors,
        options.num_steps,
        options.merge_policy,
        options.num_threads
    );

    runner.merge();
}

}
/**
 * @endcond
 */

/**
 * This function implements a variant of the mutual nearest neighbors (MNN) method for batch correction (Haghverdi _et al._, 2018).
 * Two cells from different batches can form an MNN pair if they each belong in each other's set of nearest neighbors.
 * The MNN pairs are assumed to represent cells from corresponding subpopulations across the two batches.
 * Any differences in location between the paired cells represents an estimate of the batch effect in that part of the high-dimensional space.
 *
 * We consider one batch to be the "reference" and the other to be the "target", where the aim is to correct the latter to the (unchanged) former. 
 * For each 
 * Each MNN pair is used to define a correction vector 
 * For each observation in the target batch, we find the closest MNN pairs (based on the locations of the paired observation in the same batch)
 * and we compute a robust average of the correction vectors involving those pairs.
 * This average is used to obtain a single correction vector that is applied to the target observation to obtain corrected values.
 *
 * Each MNN pair's correction vector is computed between the "center of mass" locations for the paired observations.
 * The center of mass for each observation is defined by recursively searching the neighbors of each MNN-involved observation
 * (and then the neighbors of those neighbors, up to a recursion depth of `Options::num_steps`) and computing the mean of their coordinates.
 * This improves the correction by mitigating the "kissing effect", i.e., where the correction vectors only form between the surfaces of the mass of points in each batch.
 *
 * @see
 * Haghverdi L et al. (2018).
 * Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors.
 * _Nature Biotech._ 36, 421-427
 *
 * @tparam Index_ Integer type for the observation index. 
 * @tparam Float_ Floating-point type for the input/output data.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 * Alternatively, it may be a `knncolle::SimpleMatrix`.
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
 */
template<typename Index_, typename Float_, class Matrix_>
void compute(std::size_t num_dim, const std::vector<Index_>& num_obs, const std::vector<const Float_*>& batches, Float_* output, const Options<Index_, Float_, Matrix_>& options) {
    internal::compute(num_dim, num_obs, batches, output, options);
}

/**
 * A convenience overload to merge contiguous batches contained in the same array.
 *
 * @tparam Index_ Integer type for the observation index. 
 * @tparam Float_ Floating-point type for the input/output data.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 * Alternatively, it may be a `knncolle::SimpleMatrix`.
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
 */
template<typename Index_, typename Float_, class Matrix_>
void compute(std::size_t num_dim, const std::vector<Index_>& num_obs, const Float_* input, Float_* output, const Options<Index_, Float_, Matrix_>& options) {
    std::vector<const Float_*> batches;
    batches.reserve(num_obs.size());
    for (auto n : num_obs) {
        batches.push_back(input);
        input += static_cast<std::size_t>(n) * num_dim; // cast to size_t's to avoid overflow.
    }
    compute(num_dim, num_obs, batches, output, options);
}

/**
 * Merge batches where observations are arbitrarily ordered in the same array.
 *
 * @tparam Index_ Integer type for the observation index. 
 * @tparam Float_ Floating-point type for the input/output data.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 * Alternatively, it may be a `knncolle::SimpleMatrix`.
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
 */
template<typename Index_, typename Float_, typename Batch_, class Matrix_>
void compute(std::size_t num_dim, Index_ num_obs, const Float_* input, const Batch_* batch, Float_* output, const Options<Index_, Float_, Matrix_>& options) {
    const BatchIndex nbatches = (num_obs ? static_cast<BatchIndex>(*std::max_element(batch, batch + num_obs)) + 1 : 0);
    std::vector<Index_> sizes(nbatches);
    for (Index_ o = 0; o < num_obs; ++o) {
        ++sizes[batch[o]];
    }

    // Avoiding the need to allocate a temporary buffer
    // if we're already dealing with contiguous batches.
    bool already_sorted = true;
    for (Index_ o = 1; o < num_obs; ++o) {
       if (batch[o] < batch[o-1]) {
           already_sorted = false;
           break;
       }
    }
    if (already_sorted) {
        compute(num_dim, sizes, input, output, options);
        return;
    }

    std::size_t accumulated = 0; // use size_t to avoid overflow issues during later multiplication.
    std::vector<std::size_t> offsets(nbatches);
    for (BatchIndex b = 0; b < nbatches; ++b) {
        offsets[b] = accumulated;
        accumulated += sizes[b];
    }

    // Dumping everything by order into another vector.
    std::vector<Float_> tmp(num_dim * static_cast<std::size_t>(num_obs)); // cast to size_t to avoid overflow.
    std::vector<const Float_*> ptrs(nbatches);
    for (BatchIndex b = 0; b < nbatches; ++b) {
        ptrs[b] = tmp.data() + offsets[b] * num_dim; // already size_t's, so no need to cast to avoid overflow.
    }

    for (Index_ o = 0; o < num_obs; ++o) {
        auto current = input + static_cast<std::size_t>(o) * num_dim; // cast to size_t to avoid overflow.
        auto& offset = offsets[batch[o]];
        auto destination = tmp.data() + num_dim * offset; // already size_t's, so no need to cast to avoid overflow.
        std::copy_n(current, num_dim, destination);
        ++offset;
    }

    internal::compute(num_dim, sizes, ptrs, output, options);
    internal::restore_input_order(num_dim, sizes, batch, output);
}

}

#endif
