#ifndef MNNCORRECT_MNNCORRECT_HPP
#define MNNCORRECT_MNNCORRECT_HPP

#include <algorithm>
#include <vector>
#include "AutomaticOrder.hpp"
#include "CustomOrder.hpp"
#include "restore_order.hpp"
#include "knncolle/knncolle.hpp"

/**
 * @file MnnCorrect.hpp
 *
 * @brief Batch correction using mutual nearest neighbors.
 */

namespace mnncorrect {

/**
 * @brief Batch correction using mutual nearest neighbors.
 *
 * @tparam Index Integer type for the observation indices.
 * @tparam Float Floating point type for the data and distances.
 *
 * This class implements a variant of the MNN correction method described by Haghverdi _et al._ (2018).
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
 */
template<typename Index = int, typename Float = double>
class MnnCorrect {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_num_neighbors()` for more details.
         */
        static constexpr int num_neighbors = 15;

        /**
         * See `set_num_mads()` for more details.
         */
        static constexpr Float num_mads = 3;

        /**
         * See `set_approximate()` for more details.
         */
        static constexpr bool approximate = false;

        /**
         * See `set_automatic_order()` for more details.
         */
        static constexpr bool automatic_order = true;

        /**
         * See `set_robust_iterations()` for more details.
         */
        static constexpr int robust_iterations = 2;

        /**
         * See `set_robust_trim()` for more details.
         */
        static constexpr double robust_trim = 0.25;
    };

private:
    int num_neighbors = Defaults::num_neighbors;

    Float num_mads = Defaults::num_mads;

    bool approximate = Defaults::approximate;

    bool automatic_order = Defaults::automatic_order;

    int robust_iterations = Defaults::robust_iterations;

    double robust_trim = Defaults::robust_trim;

public:
    /**
     * @param n Number of nearest neighbors to use for the searches.
     *
     * @return A reference to this `MnnCorrect` object.
     *
     * This parameter is used to define the MNN pairs at the start.
     * Larger values increase the number of MNN pairs and improve the stability of the correction, 
     * at the cost of reduced resolution of matching subpopulations across batches.
     * The number of neighbors is also used to identify the closest MNN pairs when computing the average correction vector for each target observation.
     * Again, this improves stability at the cost of resolution for local variations in the correction vectors.
     */
    MnnCorrect& set_num_neighbors(int n = Defaults::num_neighbors) {
        num_neighbors = n;
        return *this;
    }

    /**
     * @param n Number of median absolute deviations to use to define the distance threshold for the center of mass calculations.
     * Larger values reduce biases from the kissing effect but increase the risk of including inappropriately distant subpopulations into the center of mass.
     *
     * @return A reference to this `MnnCorrect` object.
     */
    MnnCorrect& set_num_mads(Float n = Defaults::num_mads) {
        num_mads = n;
        return *this;
    }

    /**
     * @param a Should an approximate nearest neighbor search be performed with Annoy?
     *
     * @return A reference to this `MnnCorrect` object.
     */
    MnnCorrect& set_approximate(bool a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    /**
     * @param a Should batches be ordered automatically?
     *
     * @return A reference to this `MnnCorrect` object.
     *
     * If `true` and `order` is not supplied in `run()`, the largest batch is used as the reference and other batches are successively merged onto it.
     * At each merge step, we choose the batch that forms the largest number of MNNs with the current reference, and the merged dataset is defined as the new reference.
     *
     * If `false` and `order` is not supplied, the supplied order of batches (or order of batch IDs) is used directly.
     *
     * If `order` is supplied, this setting is ignored and the specified order is always used.
     */
    MnnCorrect& set_automatic_order(bool a = Defaults::automatic_order) {
        automatic_order = a;
        return *this;
    }

    /**
     * @param i Number of iterations to use for robustification.
     * At each iteration, the observations furthest from the mean are removed, and the mean is recomputed with the remaining observations.
     *
     * @return A reference to this `MnnCorrect` object.
     */
    MnnCorrect& set_robust_iterations(int i = Defaults::robust_iterations) {
        robust_iterations = i;
        return *this;
    }

    /**
     * @param t Trimming proportion to use for robustification when computing the center of mass.
     * The `t` proportion of observations with the largest distances from the mean vector are removed for the next iteration of the mean calculation.
     *
     * @return A reference to this `MnnCorrect` object.
     */
    MnnCorrect& set_robust_trim(double t = Defaults::robust_trim) {
        robust_trim = t;
        return *this;
    }

public:
    /**
     * @brief Correction details.
     */
    struct Details {
        /**
         * @cond
         */
        Details() {}
        Details(std::vector<int> mo, std::vector<int> np) : merge_order(std::move(mo)), num_pairs(std::move(np)) {}
        /**
         * @endcond
         */

        /**
         * Order in which batches are merged.
         * The first entry is the index/ID of the batch used as the reference,
         * and the remaining entries are merged to the reference in the listed order.
         */
        std::vector<int> merge_order;

        /**
         * Number of MNN pairs identified at each merge step.
         * This is of length one less than `merge_order`.
         */
        std::vector<int> num_pairs;
    };

private:
    typedef knncolle::Base<Index, Float> knncolleBase; 

    static std::shared_ptr<knncolleBase> approximate_builder(int nd, size_t no, const Float* d) {
        return std::shared_ptr<knncolleBase>(new knncolle::AnnoyEuclidean<Index, Float>(nd, no, d)); 
    }

    static std::shared_ptr<knncolleBase> exact_builder(int nd, size_t no, const Float* d) {
        return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d)); 
    }

    Details run_internal(int ndim, const std::vector<size_t>& nobs, const std::vector<const Float*>& batches, Float* output, const int* order) {
        // Function name decays to a function pointer, should be callable by just doing builder(). 
        auto builder = (approximate ? approximate_builder : exact_builder);

        if (order == NULL) {
            if (automatic_order) {
                AutomaticOrder<Index, Float, decltype(builder)> runner(ndim, nobs, batches, output, builder, num_neighbors);
                runner.run(num_mads, robust_iterations, robust_trim);
                return Details(runner.get_order(), runner.get_num_pairs());
            } else {
                std::vector<int> trivial_order(nobs.size());
                std::iota(trivial_order.begin(), trivial_order.end(), 0);
                CustomOrder<Index, Float, decltype(builder)> runner(ndim, nobs, batches, output, builder, num_neighbors, trivial_order.data());
                runner.run(num_mads, robust_iterations, robust_trim);
                return Details(std::move(trivial_order), runner.get_num_pairs());
            }
        } else {
            CustomOrder<Index, Float, decltype(builder)> runner(ndim, nobs, batches, output, builder, num_neighbors, order);
            runner.run(num_mads, robust_iterations, robust_trim);
            return Details(runner.get_order(), runner.get_num_pairs());
        }
    }

public:
    /**
     * Merge batches contained in separate arrays.
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
    Details run(int ndim, const std::vector<size_t>& nobs, const std::vector<const Float*>& batches, Float* output, const int* order = NULL) {
        auto stats = run_internal(ndim, nobs, batches, output, order);
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
    Details run(int ndim, const std::vector<size_t>& nobs, const Float* input, Float* output, const int* order = NULL) {
        std::vector<const Float*> batches;
        batches.reserve(nobs.size());
        for (auto n : nobs) {
            batches.push_back(input);
            input += n * ndim;
        }
        return run(ndim, nobs, batches, output, order);
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
    template<typename Batch>
    Details run(int ndim, size_t nobs, const Float* input, const Batch* batch, Float* output, const int* order = NULL) {
        const Batch nbatches = (nobs ? *std::max_element(batch, batch + nobs) + 1 : 0);
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
            return run(ndim, sizes, input, output, order);
        }

        size_t accumulated = 0;
        std::vector<size_t> offsets(nbatches);
        for (size_t b = 0; b < nbatches; ++b) {
            offsets[b] = accumulated;
            accumulated += sizes[b];
        }

        // Dumping everything by order into another vector.
        std::vector<Float> tmp(ndim * nobs);
        std::vector<const Float*> ptrs(nbatches, tmp.data());
        for (size_t b = 0; b < nbatches; ++b) {
            ptrs[b] += offsets[b] * ndim;
        }
        for (size_t o = 0; o < nobs; ++o) {
            auto current = input + o * ndim;
            auto& offset = offsets[batch[o]];
            auto destination = tmp.data() + ndim * offset;
            std::copy(current, current + ndim, destination);
            ++offset;
        }

        auto stats = run_internal(ndim, sizes, ptrs, output, order);
        restore_order(ndim, stats.merge_order, sizes, batch, output);
        return stats;
    }
};

}

#endif
