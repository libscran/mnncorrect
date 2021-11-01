#ifndef MNNCORRECT_DETERMINE_LIMIT_HPP
#define MNNCORRECT_DETERMINE_LIMIT_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "utils.hpp"

namespace mnncorrect {

template<typename Index, typename Float>
std::vector<Float> average_batch_vector(int ndim, size_t nref, const Float* ref, size_t ntarget, const Float* target, const MnnPairs<Index>& pairings) {
    std::vector<Float> weights(ntarget);
    for (auto x : pairings.right) {
        ++weights[x];
    }

    std::vector<Float> output(ndim);
    for (size_t p = 0; p < pairings.size(); ++p) {
        auto l = ref + ndim * pairings.left[p];
        auto r = target + ndim * pairings.right[p];
        auto w = 1 / weights[pairings.right[p]];
        for (int d = 0; d < ndim; ++d) {
            output[d] += (r[d] - l[d]) * w;
        }
    }

    // L2 normalize.
    Float l2 = 0;
    for (const auto& o : output) {
        l2 += o * o;
    }
    l2 = std::sqrt(l2);
    if (l2) {
        for (auto& o : output) {
            o /= l2;
        }
    }

    return output;
}

template<typename Index, typename Float>
Float limit_from_batch_vector(int ndim, size_t nobs, const Float* data, const std::vector<Float>& average, const std::vector<Index>& in_mnn, Float nmads = 3) {
    std::vector<Float> locations(nobs);
    for (size_t o = 0; o < nobs; ++o) {
        locations[o] = std::inner_product(average.begin(), average.end(), data + o * ndim, static_cast<Float>(0));
    }

    std::vector<Float> mnn_loc(in_mnn.size());
    for (size_t p = 0; p < in_mnn.size(); ++p) {
        mnn_loc[p] = locations[in_mnn[p]];
    }

    // Computing median + MAD of all points. Note that the median function will
    // mutate 'locations', though it doesn't make a difference to the results here.
    Float med = median(nobs, locations.data());
    for (auto& l : locations) {
        l = std::abs(l - med);
    }
    Float mad = median(nobs, locations.data());

    // Getting the median of the MNN-paired points.
    Float mnn_med = median(mnn_loc.size(), mnn_loc.data());

    // Under normality, most of the distribution should be obtained
    // within 3 sigma of the correction vector. We use this to define
    // the boundaries of the distribution as med +/- delta.
    Float delta = mad * nmads * static_cast<Float>(mad2sigma);

    // Computing the sigma based on the distance from the median
    // location to either one of the effective distribution boundaries.
    return std::max(med + delta - mnn_med, mnn_med - (med - delta));
}

template<typename Index, typename Float, class Builder>
NeighborSet<Index, Float> identify_closest_mnn(int ndim, size_t nobs, const Float* data, const std::vector<Index>& in_mnn, Builder bfun, int k, Float* buffer) {
    for (size_t f = 0; f < in_mnn.size(); ++f) {
        auto current = in_mnn[f];
        auto curdata = data + current * ndim;
        std::copy(curdata, curdata + ndim, buffer + f * ndim);
    }

    auto index = bfun(ndim, in_mnn.size(), buffer);
    NeighborSet<Index, Float> output(nobs);
    #pragma omp parallel for
    for (size_t o = 0; o < nobs; ++o) {
        output[o] = index->find_nearest_neighbors(data + o * ndim, k);
    }

    return output;
}

template<typename Index, typename Float>
Float limit_from_closest_distances(const NeighborSet<Index, Float>& found, Float nmads = 3) {
    assert(found.size() > 0);

    // Pooling all distances together.
    std::vector<Float> all_distances;
    all_distances.reserve(found.size() * found[0].size());
    for (const auto& f : found) {
        for (const auto& x : f) {
            all_distances.push_back(x.second);
        }
    }

    // Computing the MAD from the lower half, to mitigate biases from a long right tail.
    Float med = median(all_distances.size(), all_distances.data());
    size_t counter = 0;
    for (auto& a : all_distances) {
        Float delta = med - a;
        if (delta > 0) {
            all_distances[counter] = delta;
            ++counter;
        }
    }
    Float mad = median(counter, all_distances.data());

    // Under normality, most of the distribution should be obtained
    // within 3 sigma of the correction vector. 
    return med + nmads * mad * static_cast<Float>(mad2sigma);
}

}

#endif
