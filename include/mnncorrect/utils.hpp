#ifndef MNNCORRECT_UTILS_HPP
#define MNNCORRECT_UTILS_HPP

#include <vector>
#include <algorithm>
#include <memory>
#include <cstddef>

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
void parallelize(int num_workers, Task_ num_tasks, Run_ run_task_range) {
#ifndef MNNCORRECT_CUSTOM_PARALLEL
    // Methods could allocate or throw, so nothrow_ = false is safest.
    subpar::parallelize_range<false>(num_workers, num_tasks, std::move(run_task_range));
#else
    MNNCORRECT_CUSTOM_PARALLEL(num_workers, num_tasks, run_task_range);
#endif
}

/**
 * @cond
 */
namespace internal {

template<typename Index_, typename Distance_>
using NeighborSet = std::vector<std::vector<std::pair<Index_, Distance_> > >;

template<typename Index_, typename Float_>
struct Corrected {
    Corrected() = default;
    Corrected(std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index, std::vector<Index_> ids) : index(std::move(index)), ids(std::move(ids)) {}
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index;
    std::vector<Index_> ids;
};

template<typename Index_, typename Float_>
struct BatchInfo {
    Index_ offset, num_obs;
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index;
    std::vector<Corrected<Index_, Float_> > extras;
};

}
/**
 * @endcond
 */

}

#endif
