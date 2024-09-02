#ifndef MNNCORRECT_OPTIONS_HPP
#define MNNCORRECT_OPTIONS_HPP

#include "knncolle/knncolle.hpp"
#include "ReferencePolicy.hpp"

/**
 * @file Options.hpp
 * @brief Options for MNN correction.
 */

namespace mnncorrect {

/**
 * @brief Options for `compute()`.
 * @tparam Dim_ Integer type for the dimensions of the neighbor search. 
 * @tparam Index_ Integer type for the observation index of the neighbor search. 
 * @tparam Float_ Floating-point type for the distances in the neighbor search.
 */
template<typename Dim_ = int, typename Index_ = int, typename Float_ = double>
struct Options {
    /**
     * Number of neighbors used in various search steps, primarily to identify MNN pairs.
     * Larger values increase the number of MNN pairs and improve the stability of the correction, 
     * at the cost of reduced resolution of matching subpopulations across batches.
     *
     * The number of neighbors is also used to identify the closest MNN pairs when computing the average correction vector for each target observation.
     * Again, this improves stability at the cost of resolution for local variations in the correction vectors.
     */
    int num_neighbors = 15;

    /**
     * Number of median absolute deviations to use to define the distance threshold for the center of mass calculations.
     * Larger values reduce biases from the kissing effect but increase the risk of including inappropriately distant subpopulations into the center of mass.
     */
    double num_mads = 3;

    /**
     * Algorithm to use for building the nearest-neighbor search indices.
     * If NULL, defaults to an exact search via `knncolle::VptreeBuilder` with Euclidean distances.
     */
    std::shared_ptr<knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_> > builder;

    /**
     * Manually specified merge order for the batches.
     * This should contain a permutation of all integers in \f${0, 1, 2, ..., N-1}\f$ where \f$N\f$ is the number of batches.
     * Each entry of this vector corresponds to a batch.
     *
     * At the first merge step, the `order[0]` batch is considered to be the reference.
     * `order[1]` is corrected against the reference and merged to form a new reference.
     * This is repeated for each remaining batch in the order specified by `order`.
     *
     * If this is empty and `Options::automatic_order = false`, batches are merged in the order that they were supplied in `compute()`.
     * If a `batch` array was supplied, the batches are merged in order of their identifiers, i.e., batch 0 is the reference.
     */
    std::vector<size_t> order;

    /**
     * Should batches be merged in an automatically-determined order?
     *
     * If `true` and `Options::order` is empty, the largest batch is used as the reference and other batches are successively merged onto it.
     * At each merge step, we choose the batch that forms the largest number of MNNs with the current reference, and the merged dataset is defined as the new reference.
     *
     * If this is empty and `Options::automatic_order = false`, batches are merged in the order that they were supplied in `compute()`.
     * If a `batch` array was supplied, the batches are merged in order of their identifiers, i.e., batch 0 is the reference.
     *
     * If `Options::order` is non-empty, this setting is ignored and the manually specified order is always used.
     */
    bool automatic_order = true;

    /**
     * Number of iterations to use for robustification when computing the center of mass for each MNN-involved cell.
     * At each iteration, the observations furthest from the center are removed, and the center is recomputed with the remaining observations.
     */
    int robust_iterations = 2;

    /**
     * Trimming proportion to use for robustification when computing the center of mass.
     * The proportion of observations with the largest distances from the center are removed for the next iteration of the center calculation.
     */
    double robust_trim = 0.25;

    /**
     * Policy to use to choose the reference batch when `Options::automatic_order = true`.
     */
    ReferencePolicy reference_policy = ReferencePolicy::MAX_RSS;

    /**
     * Cap on the number of observations used to compute the center of mass for each MNN-involved observation in the reference dataset.
     * The reference dataset is effectively downsampled to `mass_cap` observations for this specific calculation,
     * which speeds up multiple correction iterations at the cost of some precision.
     * If -1, no cap is used.
     */
    size_t mass_cap = -1;

    /**
     * Number of threads to use.
     * The parallelization scheme is defined by `parallelize()`.
     */
    int num_threads = 1;
};

}

#endif
