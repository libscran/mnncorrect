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
 * Policy for choosing the first reference batch with the automatic merging procedure.
 * 
 * - `INPUT` will use the first supplied batch in the input order.
 *   This is useful in cases where one batch is known to contain most subpopulations and should be used as the reference,
 *   but there is no obvious ordering for the other batches.
 * - `MAX_SIZE` will use the largest batch (i.e., with the most observations).
 *   This is simple to compute and was the previous default;
 *   it does, at least, ensure that the initial reference has enough cells for stable correction.
 * - `MAX_VARIANCE` will use the batch with the greatest variance.
 *   This improves the likelihood of obtaining an reference that contains a diversity of subpopulations
 *   and thus is more likely to form sensible MNN pairs with subsequent batches.
 * - `MAX_RSS` will use the batch with the greatest residual sum of squares (RSS).
 *   This is similar to `MAX_VARIANCE` but it puts more weight on batches with more cells,
 *   so as to avoid picking small batches with few cells and unstable population strcuture.
 */
enum class ReferencePolicy : char { INPUT, MAX_SIZE, MAX_VARIANCE, MAX_RSS };

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

template<typename Index_, typename Float_>
struct SortBySecond {
    bool operator()(const std::pair<Index_, Float_>& left, const std::pair<Index_, Float_>& right) const {
        if (left.second == right.second) {
            return left.first < right.first;
        } else {
            return left.second < right.second;
        }
    }
};

}
/**
 * @endcond
 */

}

#endif
