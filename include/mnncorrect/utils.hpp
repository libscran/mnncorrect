#ifndef MNNCORRECT_UTILS_HPP
#define MNNCORRECT_UTILS_HPP

#include <vector>
#include <algorithm>
#include <memory>
#include <cstddef>
#include <type_traits>

#include "knncolle/knncolle.hpp"

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

/**
 * @file utils.hpp
 * @brief Utilities for MNN correction.
 */

namespace mnncorrect {

/**
 * Integer type of the batch indices.
 */
typedef std::size_t BatchIndex;

/**
 * Policy for choosing the order of batches to merge.
 * 
 * - `INPUT` will use the input order of the batches.
 *   Observations in the last batch are corrected first, and then the second-last batch, and so on.
 *   This allows users to control the merge order by simply changing the inputs.
 * - `SIZE` will merge batches in order of increasing size (i.e., the number of observations).
 *   So, the smallest batch is corrected first while the largest batch is unchanged.
 *   The aim is to lower compute time by reducing the number of observations that need to be reprocessed in later merge steps.
 * - `VARIANCE` will merge batches in order of increasing variance between observations. 
 *   So, the batch with the lowest variance is corrected first while the batch with the highest variance is unchanged.
 *   The aim is to lower compute time by encouraging more observations to be corrected to the most variable batch, thus avoid reprocessing in later merge steps.
 * - `RSS` will merge batches in order of increasing residual sum of squares (RSS).
 *   This is effectively a compromise between `VARIANCE` and `SIZE`.
 */
enum class MergePolicy : char { INPUT, SIZE, VARIANCE, RSS };

/**
 * @tparam Task_ Integer type for the number of tasks.
 * @tparam Run_ Function to execute a range of tasks.
 *
 * @param num_workers Number of workers.
 * @param num_tasks Number of tasks.
 * @param run_task_range Function to iterate over a range of tasks within a worker.
 *
 * By default, this is an alias to `subpar::parallelize_range()`.
 * However, if the `MNNCORRECT_CUSTOM_PARALLEL` function-like macro is defined, it is called instead. 
 * Any user-defined macro should accept the same arguments as `subpar::parallelize_range()`.
 */
template<typename Task_, class Run_>
void parallelize(const int num_workers, const Task_ num_tasks, Run_ run_task_range) {
#ifndef MNNCORRECT_CUSTOM_PARALLEL
    // Methods could allocate or throw, so nothrow_ = false is safest.
    subpar::parallelize_range<false>(num_workers, num_tasks, std::move(run_task_range));
#else
    MNNCORRECT_CUSTOM_PARALLEL(num_workers, num_tasks, run_task_range);
#endif
}

/**
 * @brief Start and size of each batch.
 * @tparam Index_ Integer type of the observation indices.
 */
template<typename Index_>
struct Batch {
    /**
     * Starting index of each batch, i.e., the index of the first observation in the batch.
     */
    Index_ start = 0;

    /**
     * Size of each batch, i.e., the number of observations in the batch.
     */
    Index_ size = 0;
};

/**
 * @cond
 */
template<typename Index_, typename Distance_>
using NeighborSet = std::vector<std::vector<std::pair<Index_, Distance_> > >;

template<typename Index_, typename Float_>
struct MetaBatch {
    // Each uncorrected metabatch has an original contiguous set of observations.
    // Once corrected, the metabatch ceases to exist as it becomes part of the destination metabatch.
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > original_index;
    Batch<Index_> original_ids;

    // Corrected observations from other (meta)batches that have been redistributed into this meta batch.
    struct CorrectedBatch {
        CorrectedBatch() = default;
        CorrectedBatch(std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index, std::vector<Index_> ids) : index(std::move(index)), ids(std::move(ids)) {}
        std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index;
        std::vector<Index_> ids;
    };
    std::vector<CorrectedBatch> corrected;
};

template<typename Input_>
using I = std::remove_cv_t<std::remove_reference_t<Input_> >;

// Putting this here so that we can re-use it in the tests.
template<typename Index_, typename Float_, class Matrix_>
std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > subset_and_index(
    const std::size_t num_dim,
    const std::vector<Index_>& subset,
    const Float_* const data,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    Float_* const buffer
) {
    const auto num_subset = subset.size();
    for (I<decltype(num_subset)> f = 0; f < num_subset; ++f) {
        const auto curdata = data + sanisizer::product_unsafe<std::size_t>(subset[f], num_dim);
        std::copy_n(curdata, num_dim, buffer + sanisizer::product_unsafe<std::size_t>(f, num_dim));
    }
    return builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_subset, buffer));
}
/**
 * @endcond
 */

}

#endif
