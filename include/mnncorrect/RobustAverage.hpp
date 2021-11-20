#ifndef MNNCORRECT_ROBUST_AVERAGE_HPP
#define MNNCORRECT_ROBUST_AVERAGE_HPP

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "utils.hpp"

namespace mnncorrect {

template<typename Index, typename Float>
class RobustAverage {
public:
    RobustAverage(int it, double tr, Float lim) : iterations(it), trim(tr), limit2(lim * lim) {
        check_args();
        return;
    }

    RobustAverage(int it, double tr) : iterations(it), trim(tr), limit2(std::numeric_limits<Float>::infinity()) {
        check_args();
        return;
    }

private:
    int iterations;
    double trim;
    Float limit2;

    void check_args() {
        if (trim < 0 || trim >= 1) {
            throw std::runtime_error("trimming proportion must be in [0, 1)");
        }
        if (iterations < 0) {
            throw std::runtime_error("number of iterations must be non-negative");
        }
    }

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
                if (d2 < limit2) {
                    deltas.emplace_back(d2, j);
                }
            }
            
            // Sort rather than nth_element; avoid machine-dependent
            // differences in order that affect numerical precision, as well as
            // problems when trim = 0 such that the nth element is the end
            // iterator (and thus not de-referenceable). 
            std::sort(deltas.begin(), deltas.end());

            std::fill(output, output + ndim, 0);
            size_t limit = std::ceil((1.0 - trim) * static_cast<double>(deltas.size()));

            for (size_t x = 0; x < limit; ++x) {
                auto dptr = data + deltas[x].second * ndim;
                for (int d = 0; d < ndim; ++d) {
                    output[d] += dptr[d];
                }
            }
            for (int d = 0; d < ndim; ++d) {
                output[d] /= limit;
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
