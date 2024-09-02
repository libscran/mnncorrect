#ifndef MNNCORRECT_ROBUST_AVERAGE_HPP
#define MNNCORRECT_ROBUST_AVERAGE_HPP

#include <stdexcept>
#include <vector>
#include <algorithm>

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
 * - If trim > 0, the furthest point is always removed. This ensures that
 *   some trimming is always performed (unless, of course, there was only
 *   one point, in which case that point is just retained).
 */

class RobustAverageOptions {
public:
    RobustAverageOptions(int iterations, double trim) : iterations(iterations), trim(trim) {
        if (trim < 0 || trim > 1) {
            throw std::runtime_error("trimming proportion must be in [0, 1]");
        }
        if (iterations < 0) {
            throw std::runtime_error("number of iterations must be non-negative");
        }
    }

public:
    int get_iterations() const {
        return iterations;
    }

    double get_trim() const {
        return trim;
    }

private:
    int iterations;
    double trim;
};

template<class Function_, typename Float_>
void robust_average(size_t num_dim, size_t num_pts, Function_ indfun, const Float_* data, Float_* output, std::vector<std::pair<Float_, size_t> >& deltas, const RobustAverageOptions& options) {
    const auto iterations = options.get_iterations();
    const auto trim = options.get_trim();

    const size_t long_ndim = num_dim;
    std::fill_n(output, long_ndim, 0);
    if (num_pts == 0) {
        return;
    }

    for (size_t i = 0; i < num_pts; ++i) {
        const auto dptr = data + static_cast<size_t>(indfun(i)) * long_ndim; // cast to size_t to avoid overflow.
        for (size_t d = 0; d < num_dim; ++d) {
            output[d] += dptr[d];
        }
    }

    const double denom = 1.0 / num_pts;
    for (size_t d = 0; d < num_dim; ++d) {
        output[d] *= denom;
    }

    // The 'num_pts - 1' reflects the fact that we're comparing to a quantile.
    // The closest point is at 0%, while the furthest point is at 100%,
    // so we already spent one observation defining the boundaries.
    const Float_ threshold = (num_pts - 1) * (1.0 - trim);

    deltas.reserve(num_pts);
    for (int it = 0; it < iterations; ++it) {
        deltas.clear();

        for (size_t i = 0; i < num_pts; ++i) {
            auto j = indfun(i);
            const auto dptr = data + static_cast<size_t>(j) * long_ndim; // cast to avoid overflow.

            Float_ d2 = 0;
            for (size_t d = 0; d < num_dim; ++d) {
                Float_ diff = output[d] - dptr[d];
                d2 += diff * diff;
            }

            deltas.emplace_back(d2, j);
        }

        std::sort(deltas.begin(), deltas.end());

        // We always keep at least the closest observation.
        const auto first_ptr = data + static_cast<size_t>(deltas.front().second) * long_ndim; // cast to avoid overflow.
        std::copy_n(first_ptr, num_dim, output);
        Float_ counter = 1;
        Float_ last = deltas.front().first;

        // Checking if we can add another observation without cutting into
        // the specified trim proportion - 'counter/(npt - 1)' is the
        // quantile of the current observation in the loop. The exception
        // is if the threshold interrupts some ties, in which case all of
        // them are retained to avoid arbitrary ordering effects.
        for (size_t x = 1; x < num_pts; ++x) {

            // When considering ties, we need to account for numerical
            // imprecision in the distance calculations. We do so by
            // allowing a tolerance in the comparison - in this case, of
            // 1e-10. To avoid a sliding slope of inclusion, we fix our
            // comparisons to the first element of a tied run and only
            // consider subsequent elements to be tied if they are within
            // the tolerance of the first element.
            constexpr Float_ tol = 1.0000000001;
            if (deltas[x].first > last * tol) { // i.e., not tied.
                last = deltas[x].first;
                if (counter > threshold) {
                    break;
                }
            }

            const auto dptr = data + static_cast<size_t>(deltas[x].second) * long_ndim; // both are already size_t's to avoid overflow.
            for (size_t d = 0; d < num_dim; ++d) {
                output[d] += dptr[d];
            }

            ++counter;
        }

        const double denom = 1.0/counter;
        for (size_t d = 0; d < num_dim; ++d) {
            output[d] *= denom;
        }
    }
}

// Using 'size_t' as this function is called in 'correct_target()' with the
// number of MNN pairs being used as the 'num_pts', and that might exceed the
// actual number of observations.
template<typename Float_>
void robust_average(size_t num_dim, size_t num_pts, const Float_* data, Float_* output, std::vector<std::pair<Float_, size_t> >& deltas, const RobustAverageOptions& options) {
    robust_average(num_dim, num_pts, [](size_t i) -> size_t { return i; }, data, output, deltas, options);
}

// Using size_t for 'num_obs', as 'indices' may contain duplicates from the
// inverted neighbors; this causes 'indices.size()' to possibly exceed the
// capacity of the 'Index_' type.
template<typename Index_, typename Float_>
void robust_average(size_t num_dim, const std::vector<Index_>& indices, const Float_* data, Float_* output, std::vector<std::pair<Float_, size_t> >& deltas, const RobustAverageOptions& options) {
    robust_average(num_dim, indices.size(), [&](size_t i) -> size_t { return indices[i]; }, data, output, deltas, options);
}

}

}

#endif
