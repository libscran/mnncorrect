#ifndef MNNCORRECT_ROBUST_AVERAGE_HPP
#define MNNCORRECT_ROBUST_AVERAGE_HPP

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "utils.hpp"

namespace mnncorrect {

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
 *   some trimming is always performed when requested.
 */ 
template<typename Index, typename Float>
class RobustAverage {
public:
    RobustAverage(int it, double tr) : iterations(it), trim(tr) {
        if (trim < 0 || trim > 1) {
            throw std::runtime_error("trimming proportion must be in [0, 1]");
        }
        if (iterations < 0) {
            throw std::runtime_error("number of iterations must be non-negative");
        }
    }

private:
    int iterations;
    double trim;

    template<class Function>
    void run(int ndim, size_t npts, Function indfun, const Float* data, Float* output) {
        std::fill(output, output + ndim, 0);
        for (size_t i = 0; i < npts; ++i) {
            auto dptr = data + indfun(i) * ndim;
            for (int d = 0; d < ndim; ++d) {
                output[d] += dptr[d];
            }
        }
        for (int d = 0; d < ndim; ++d) {
            output[d] /= npts;
        }

        // The 'npts - 1' reflects the fact that we're comparing to a quantile.
        // The closest point is at 0%, while the furthest point is at 100%,
        // so we already spent one observation defining the boundaries.
        const double threshold = static_cast<double>(npts - 1) * (1 - trim);

        // And now iterating.
        deltas.reserve(npts);

        for (int it = 0; it < iterations; ++it) {
            deltas.clear();
            for (size_t i = 0; i < npts; ++i) {
                auto j = indfun(i);
                auto dptr = data + j * ndim;

                Float d2 = 0;
                for (int d = 0; d < ndim; ++d) {
                    Float diff = output[d] - dptr[d];
                    d2 += diff * diff;
                }
                deltas.emplace_back(d2, j);
            }

            std::sort(deltas.begin(), deltas.end());

            // We always keep at least the closest observation.
            auto first_ptr = data + deltas.front().second * ndim;
            std::copy(first_ptr, first_ptr + ndim, output);
            double counter = 1;
            Float last = deltas.front().first;

            // Checking if we can add another observation without cutting into
            // the specified trim proportion - 'counter/(npt - 1)' is the
            // quantile of the current observation in the loop. The exception
            // is if the threshold interrupts some ties, in which case all of
            // them are retained to avoid arbitrary ordering effects.
            for (size_t x = 1; x < npts; ++x) {

                // When considering ties, we need to account for numerical
                // precision by allowing a tolerance - in this case, of 1e-10.
                // To avoid a sliding slope of inclusion, we fix our
                // comparisons to the first element of a tied run and only
                // consider subsequent elements to be tied if they are within
                // the tolerance of the first element.
                constexpr Float tol = 1.0000000001;
                if (deltas[x].first > last * tol) { // i.e., not tied.
                    last = deltas[x].first;
                    if (counter > threshold) {
                        break;
                    }
                }

                auto dptr = data + deltas[x].second * ndim;
                for (int d = 0; d < ndim; ++d) {
                    output[d] += dptr[d];
                }

                ++counter;
            }

            for (int d = 0; d < ndim; ++d) {
                output[d] /= counter;
            }
        }
    }

public:
    void run(int ndim, size_t npts, const Float* data, Float* output) {
        run(ndim, npts, [](size_t i) -> size_t { return i; }, data, output);
        return;
    }

    void run(int ndim, const std::vector<Index>& indices, const Float* data, Float* output) {
        run(ndim, indices.size(), [&](size_t i) -> size_t { return indices[i]; }, data, output);
        return;
    }

private:
    std::vector<std::pair<Float, Index> > deltas;
};

}

#endif
