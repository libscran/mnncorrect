#ifndef MNNCORRECT_ROBUST_AVERAGE_HPP
#define MNNCORRECT_ROBUST_AVERAGE_HPP

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cstddef>

namespace mnncorrect {

namespace internal {

/* 
 * This computes a robust average after trimming away observations that are
 * furthest from the mean observation. Specifically, we keep all observations
 * that are less than or equal to the '1 - trim' quantile, given a trimming
 * proportion between 0 and 1. The trimmed set is used to compute a new mean
 * and this process is repeated for the specified number of iterations. 
 * 
 * The use of the quantile here is approximately equivalent to removing 'trim'
 * furthest observations, but with the following subtleties:
 *
 * - If there are observations at tied distances at the trim boundary,
 *   all ties are retained. This avoids arbitrary changes in the resullts
 *   depending on the input order of observations.
 * - If trim = 1, the point closest to the mean is always retained,
 *   ensuring that the mean calculation in the next iteration is defined.
 */

struct RobustAverageOptions {
    RobustAverageOptions(int iterations, double trim) : iterations(iterations), trim(trim) {
        if (trim < 0 || trim > 1) {
            throw std::runtime_error("trimming proportion must be in [0, 1]");
        }
        if (iterations < 0) {
            throw std::runtime_error("number of iterations must be non-negative");
        }
    }

    int iterations;
    double trim;
};

template<typename Float_>
struct RobustAverageWorkspace {
    std::vector<Float_> deltas, copy;
};

template<typename Float_>
Float_ quantile(std::vector<Float_>& x, double quantile) {
    auto num = x.size(); // should be positive at this point.
    const Float_ cut = (num - 1) * quantile;
    decltype(num) lower = cut; // floor.

    std::nth_element(x.begin(), x.begin() + lower, x.end());
    auto lval = x[lower];
    if (lower + 1 == num) { // just in case we're dealing with quantile == 1 
        return lval;
    }

    std::nth_element(x.begin() + lower, x.begin() + lower + 1, x.end());
    auto uval = x[lower + 1];
    Float_ gap = cut - lower;
    return lval + gap * (uval - lval); // i.e., (1 - gap) * lval + gap * uval, equivalent to quantile() in R.
}

template<typename Number_, class Function_, typename Float_>
void robust_average(std::size_t num_dim, Number_ num_vec, Function_ indfun, const Float_* data, Float_* output, RobustAverageWorkspace<Float_>& work, const RobustAverageOptions& options) {
    std::fill_n(output, num_dim, 0);
    if (num_vec == 0) {
        return;
    }

    for (Number_ i = 0; i < num_vec; ++i) {
        const auto dptr = data + static_cast<std::size_t>(indfun(i)) * num_dim; // cast to size_t to avoid overflow.
        for (std::size_t d = 0; d < num_dim; ++d) {
            output[d] += dptr[d];
        }
    }
    const double denom = 1.0 / num_vec;
    for (std::size_t d = 0; d < num_dim; ++d) {
        output[d] *= denom;
    }
    if (options.trim == 0) {
        return;
    }

    // The 'num_vec - 1' reflects the fact that we're comparing to a quantile.
    // The closest point is at 0%, while the furthest point is at 100%,
    // so we already spent one observation defining the boundaries.

    work.deltas.reserve(num_vec);
    for (int it = 0; it < options.iterations; ++it) {
        work.deltas.clear();

        for (Number_ i = 0; i < num_vec; ++i) {
            const auto dptr = data + static_cast<std::size_t>(indfun(i)) * num_dim; // cast to avoid overflow.
            Float_ d2 = 0;
            for (std::size_t d = 0; d < num_dim; ++d) {
                Float_ diff = output[d] - dptr[d];
                d2 += diff * diff;
            }
            work.deltas.push_back(d2);
        }

        work.copy.clear();
        work.copy.insert(work.copy.end(), work.deltas.begin(), work.deltas.end());
        const Float_ q = quantile(work.copy, 1.0 - options.trim);

        // When considering ties, we need to account for numerical imprecision
        // in the distance calculations. We do so by allowing a tolerance in
        // the comparison - in this case, of 1e-10.
        constexpr Float_ tol = 1.0000000001;
        const Float_ threshold = q * tol; 

        auto sum = [&](Float_ threshold) -> Number_ {
            Number_ counter = 0;
            std::fill_n(output, num_dim, 0);
            for (Number_ i = 0; i < num_vec; ++i) {
                if (work.deltas[i] <= threshold) {
                    const auto dptr = data + static_cast<std::size_t>(indfun(i)) * num_dim; // again, cast to avoid overflow.
                    for (std::size_t d = 0; d < num_dim; ++d) {
                        output[d] += dptr[d];
                    }
                    ++counter;
                }
            }
            return counter;
        };
        auto counter = sum(threshold);

        if (counter == 0) {
            // Failsafe in case numerical imprecision causes the quantile to be smaller than the minimum.
            auto min = *std::min_element(work.deltas.begin(), work.deltas.end());
            counter = sum(min);
        }

        const double denom = 1.0/counter;
        for (std::size_t d = 0; d < num_dim; ++d) {
            output[d] *= denom;
        }
    }
}

template<typename Number_, typename Float_>
void robust_average(std::size_t num_dim, Number_ num_vec, const Float_* data, Float_* output, RobustAverageWorkspace<Float_>& deltas, const RobustAverageOptions& options) {
    // Templating the number of points as this function is called in
    // 'correct_target()' with the number of MNN pairs being used as the
    // 'num_vec', and that might exceed the actual number of observations.
    robust_average(num_dim, num_vec, [](Number_ i) -> Number_ { return i; }, data, output, deltas, options);
}

template<typename Index_, typename Float_>
void robust_average(size_t num_dim, const std::vector<Index_>& indices, const Float_* data, Float_* output, RobustAverageWorkspace<Float_>& deltas, const RobustAverageOptions& options) {
    // Using the size_type for 'num_obs', as 'indices' may contain duplicates
    // from the inverted neighbors; this causes 'indices.size()' to possibly
    // exceed the capacity of the 'Index_' type.
    auto n_indices = indices.size();
    typedef decltype(n_indices) Counter;
    robust_average(num_dim, n_indices, [&](Counter i) -> Index_ { return indices[i]; }, data, output, deltas, options);
}

}

}

#endif
