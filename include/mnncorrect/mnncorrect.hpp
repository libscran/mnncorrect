#ifndef MNNCORRECT_HPP
#define MNNCORRECT_HPP

#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

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
 * @tparam Index_ Integer type of the observation indices.
 * @tparam Float_ Floating-point type of the input/output data.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 * Alternatively, it may be a `knncolle::SimpleMatrix`.
 */
template<typename Index_, typename Float_, class Matrix_ = knncolle::Matrix<Index_, Float_> >
struct Options {
    /**
     * Number of neighbors to use in the various search steps - specifically, identification of MNN pairs and calculation of the centers of mass. 
     * It can be interpreted as the lower bound on the number of observations in each "subpopulation". 
     *
     * Larger values improve the stability of the correction by increasing the number of MNN pairs and including more observations in each center of mass.
     * However, this comes at the cost of reduced resolution when matching subpopulations across batches.
     */
    int num_neighbors = 15;

    /**
     * Number of steps for the recursive neighbor search to compute the center of mass for each MNN-involved observationc.
     * Larger values mitigate the kissing effect but increase the risk of including inappropriately distant subpopulations into the center of mass.
     */
    int num_steps = 1;

    /**
     * Algorithm to use for building the nearest-neighbor search indices.
     * If NULL, defaults to an exact search via `knncolle::VptreeBuilder` with Euclidean distances.
     */
    std::shared_ptr<knncolle::Builder<Index_, Float_, Float_, Matrix_> > builder;

    /**
     * Policy for choosing the merge order.
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
void compute(const std::size_t num_dim, const std::vector<Index_>& num_obs, const std::vector<const Float_*>& batches, Float_* const output, const Options<Index_, Float_, Matrix_>& options) {
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
 * Two observations from different batches can form an MNN pair if they each belong in each other's set of nearest neighbors.
 * The MNN pairs are assumed to represent observations from corresponding subpopulations across the two batches.
 * Any differences in location between the paired observations represents an estimate of the batch effect in that part of the high-dimensional space.
 *
 * We consider one batch to be the "reference" and the other to be the "target", where the aim is to correct the latter to the former. 
 * Each MNN pair defines a correction vector that moves the target observation towards its paired reference observation.
 * For each observation in the target batch, we identify the closest observation in the same batch that is part of a MNN pair (i.e., "MNN-involved observations").
 * We apply that pair's correction vector to the observation to obtain its corrected coordinates.
 *
 * Each MNN pair's correction vector is computed between the "center of mass" locations for the paired observations.
 * The center of mass for each observation is defined by recursively searching the neighbors of each MNN-involved observation
 * (and then the neighbors of those neighbors, up to a recursion depth of `Options::num_steps`) and computing the mean of their coordinates.
 * This improves the correction by mitigating the "kissing effect", i.e., where the correction vectors only form between the surfaces of the mass of points in each batch.
 *
 * In the case of >2 batches, we define a merge order based on `Options::merge_policy`.
 * For the first batch to be merged, we identify MNN pairs to all other batches at once.
 * The subsequent correction effectively distributes the first batch's observations to all other batches.
 * This process is repeated for all remaining batches until only one batch remains that contains all observations.
 *
 * @see
 * Haghverdi L et al. (2018).
 * Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors.
 * _Nature Biotech._ 36, 421-427
 *
 * @tparam Index_ Integer type of the observation index. 
 * @tparam Float_ Floating-point type of the input/output data.
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
void compute(const std::size_t num_dim, const std::vector<Index_>& num_obs, const std::vector<const Float_*>& batches, Float_* const output, const Options<Index_, Float_, Matrix_>& options) {
    internal::compute(num_dim, num_obs, batches, output, options);
}

/**
 * Overload of `compute()` to merge contiguous batches contained in the same array.
 *
 * @tparam Index_ Integer type of the observation index. 
 * @tparam Float_ Floating-point type of the input/output data.
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
void compute(const std::size_t num_dim, const std::vector<Index_>& num_obs, const Float_* const input, Float_* const output, const Options<Index_, Float_, Matrix_>& options) {
    std::vector<const Float_*> batches;
    batches.reserve(num_obs.size());

    Index_ accumulated = 0;
    for (const auto n : num_obs) {
        batches.push_back(input + sanisizer::product_unsafe<std::size_t>(accumulated, num_dim));

        // After this check, all internal functions may assume that the total number of observations fits in an Index_.
        accumulated = sanisizer::sum<decltype(I(accumulated))>(accumulated, n);
    }

    compute(num_dim, num_obs, batches, output, options);
}

/**
 * Overload of `compute()` to merge batches where observations are arbitrarily ordered in the same array.
 *
 * @tparam Index_ Integer type of the observation index. 
 * @tparam Float_ Floating-point type of the input/output data.
 * @tparam Matrix_ Class of the input data matrix for the neighbor search.
 * This should satisfy the `knncolle::Matrix` interface.
 * Alternatively, it may be a `knncolle::SimpleMatrix`.
 * @tparam Batch_ Integer type of the batch IDs.
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
void compute(const std::size_t num_dim, const Index_ num_obs, const Float_* const input, const Batch_* const batch, Float_* const output, const Options<Index_, Float_, Matrix_>& options) {
    const BatchIndex nbatches = (num_obs ? sanisizer::sum<BatchIndex>(*std::max_element(batch, batch + num_obs), 1) : static_cast<BatchIndex>(0));
    auto sizes = sanisizer::create<std::vector<Index_> >(nbatches);
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

    Index_ accumulated = 0;
    auto offsets = sanisizer::create<std::vector<Index_> >(nbatches);
    std::vector<Float_> tmp(sanisizer::product<typename std::vector<Float_>::size_type>(num_dim, num_obs));
    auto ptrs = sanisizer::create<std::vector<const Float_*> >(nbatches);
    for (BatchIndex b = 0; b < nbatches; ++b) {
        ptrs[b] = tmp.data() + sanisizer::product_unsafe<std::size_t>(accumulated, num_dim);
        offsets[b] = accumulated;
        accumulated += sizes[b]; // this won't overflow as know that num_obs fits in an Index_.
    }

    for (Index_ o = 0; o < num_obs; ++o) {
        auto& offset = offsets[batch[o]];
        std::copy_n(
            input + sanisizer::product_unsafe<std::size_t>(o, num_dim),
            num_dim,
            tmp.data() + sanisizer::product_unsafe<std::size_t>(offset, num_dim)
        );
        ++offset;
    }

    internal::compute(num_dim, sizes, ptrs, output, options);
    internal::restore_input_order(num_dim, sizes, batch, output);
}

}

#endif
