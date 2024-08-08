#ifndef MNNCORRECT_REFERENCE_POLICY_HPP
#define MNNCORRECT_REFERENCE_POLICY_HPP

/**
 * @file ReferencePolicy.hpp
 *
 * @brief Automatic choice for the reference batch.
 */

namespace mnncorrect {

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

}

#endif
