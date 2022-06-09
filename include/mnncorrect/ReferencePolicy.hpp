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
 * - `Input` will use the first supplied batch in the input order.
 *   This is useful in cases where one batch is known to contain most subpopulations and should be used as the reference,
 *   but there is no obvious ordering for the other batches.
 * - `MaxSize` will use the largest batch (i.e., with the most observations).
 *   For historical reasons, this is the default - it does, at least, ensure that the initial reference has enough cells.
 * - `MaxVariance` will use the batch with the greatest variance.
 *   This improves the likelihood of obtaining an reference that contains a diversity of subpopulations
 *   and thus is more likely to form sensible MNN pairs with subsequent batches.
 * - `MaxRss` will use the batch with the greatest residual sum of squares (RSS).
 *   This is similar to `MaxVariance` but it puts more weight on batches with more cells,
 *   so as to avoid picking small batches with few cells and unstable population strcuture.
 */
enum ReferencePolicy { Input, MaxSize, MaxVariance, MaxRss };

}

#endif
