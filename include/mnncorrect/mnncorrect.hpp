#ifndef MNNCORRECT_HPP
#define MNNCORRECT_HPP

#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

#include "Coordinator.hpp"
#include "reorder_matrix_in_place.hpp"
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
template<typename Index_, typename Float_, class Matrix_>
void compute_internal(
    const std::size_t num_dim,
    const Index_ num_total,
    const std::vector<Batch<Index_> >& batches,
    Float_* const data,
    const Options<Index_, Float_, Matrix_>& options
) {
    auto builder = options.builder;
    if (!builder) {
        typedef knncolle::EuclideanDistance<Float_, Float_> Euclidean;
        builder.reset(new knncolle::VptreeBuilder<Index_, Float_, Float_, Matrix_, Euclidean>(std::make_shared<Euclidean>()));
    }

    Coordinator<Index_, Float_, Matrix_> runner(
        num_dim,
        num_total,
        batches,
        data,
        *builder,
        options.num_neighbors,
        options.num_steps,
        options.merge_policy,
        options.num_threads
    );

    runner.merge();
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
 * @param[in] batches Vector of length equal to the number of batches.
 * The `i`-th entry contains the starting position and size of batch `i` in `data`.
 * Batches should be contiguous and non-overlapping, i.e., each observation in `data` should be assigned to exactly one batch in `batches`.
 * @param[in,out] data Pointer to an array containing a column-major matrix with number of rows equal to `num_dim` and number of columns equal to the sum of sizes in `batches`.
 * On input, it contains the uncorrected data for all observations from all batches.
 * Observations from the same batch should be stored in adjacent columns, where `batches[i]` specifies the first such column and number of columns in batch `i`.
 * On output, this contains the corrected values for all observations.
 * @param options Further options.
 */
template<typename Index_, typename Float_, class Matrix_>
void compute(const std::size_t num_dim, const std::vector<Batch<Index_> >& batches, Float_* const data, const Options<Index_, Float_, Matrix_>& options) {
    Index_ num_total = 0;
    for (const auto& batch : batches) {
        num_total = sanisizer::sum<Index_>(num_total, batch.size);
    }
    compute_internal(num_dim, num_total, batches, data, options);
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
 * @param[in,out] data Pointer to an array containing a column-major matrix with number of rows equal to `num_dim` and number of columns equal to `num_obs`.
 * On input, it contains the uncorrected data for all observations. 
 * (For this overload, observations from the same batch need not be in adjacent colmns.)
 * On output, this contains the corrected values for all observations.
 * @param[in] batch Pointer to an array of length `num_obs` containing the batch identity for each observation.
 * IDs should be zero-indexed and lie within `[0, num_batches)`.
 * @param num_batches Number of batches in `batch`.
 * @param options Further options.
 */
template<typename Index_, typename Float_, typename Batch_, class Matrix_>
void compute(
    const std::size_t num_dim,
    const Index_ num_obs,
    Float_* const data,
    const Batch_* const batch,
    const BatchIndex num_batches,
    const Options<Index_, Float_, Matrix_>& options
) {
    // Avoiding allocation of a temporary buffer if we're already dealing with contiguous batches.
    auto batches = sanisizer::create<std::vector<Batch<Index_> > >(num_batches);
    Index_ non_contiguous = 0;
    for (Index_ o = 0; o < num_obs; ++o) {
        auto& curbatch = batches[batch[o]];
        if (curbatch.size == 0) {
            curbatch.start = o;
            curbatch.size = 1;
        } else {
            non_contiguous += (o != curbatch.start + curbatch.size);
            ++curbatch.size;
        }
    }

    if (non_contiguous == 0) {
        compute_internal(num_dim, num_obs, batches, data, options);
        return;
    }

    // Otherwise, we reorganize the data so that observations from the same batch are in a single block.
    Index_ accumulated = 0;
    auto offsets = sanisizer::create<std::vector<Index_> >(num_batches);
    for (BatchIndex b = 0; b < num_batches; ++b) {
        offsets[b] = accumulated;
        batches[b].start = accumulated;
        accumulated += batches[b].size; // this won't overflow as know that num_obs fits in an Index_.
    }

    auto reordered = sanisizer::create<std::vector<Index_> >(num_obs);
    for (Index_ o = 0; o < num_obs; ++o) {
        auto& offset = offsets[batch[o]];
        reordered[offset] = o;
        ++offset;
    }
    auto mbuffer = sanisizer::create<std::vector<Float_> >(num_dim); 
    reorder_matrix_in_place(num_dim, num_obs, reordered, data, mbuffer);

    compute_internal(num_dim, num_obs, batches, data, options);

    // Reorganizing back to the original ordering.
    for (BatchIndex b = 0; b < num_batches; ++b) {
        offsets[b] = batches[b].start;
    }
    for (Index_ o = 0; o < num_obs; ++o) {
        auto& offset = offsets[batch[o]];
        reordered[o] = offset;
        ++offset;
    }
    reorder_matrix_in_place(num_dim, num_obs, reordered, data, mbuffer);
}

}

#endif
