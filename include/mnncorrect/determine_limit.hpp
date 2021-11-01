#ifndef MNNCORRECT_DETERMINE_LIMIT_HPP
#define MNNCORRECT_DETERMINE_LIMIT_HPP

#include <vector>
#include <algorithm>
#include <cmath>

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

    Float total = 0;
    for (auto w : weights) {
        total += (w > 0);
    }
    for (auto& o : output) {
        o /= total;
    }

    return output;
}

template<typename Index, typename Float>
Float limit_from_batch_vector(int ndim, size_t nobs, const Float* data, const std::vector<Float>& average, const std::vector<Index>& in_mnn, Float nmads = 3) {
    std::vector<Float> locations(nobs);
    for (size_t o = 0; o < nobs; ++o) {
        locations[o] = std::inner_product(average.begin(), average.end(), data + o * ndim);
    }

    std::vector<Float> mnn_loc(in_mnn.size());
    for (size_t p = 0; p < in_mnn.size(); ++p) {
        mnn_loc[p] = locations[in_mnn[p]];
    }

    // Deriving the median of all points.
    Float med = median(nobs, locations.begin());
    
    // Deriving the MAD of all points.
    for (auto& l : locations) {
        l = std::abs(l - med);
    }
    Float mad = median(nobs, locations.begin());

    // Getting the median of the MNN-paired points.
    Float mnn_med = median(mnn_loc.size(), mnn_loc.begin());

    // Under normality, most of the distribution should be obtained
    // within 3 sigma of the correction vector. We use this to define
    // the boundaries of the distribution as med +/- delta.
    Float delta = mad * nmads * static_cast<Float>(mad2sigma);

    // Computing the sigma based on the distance from the median
    // location to either one of the effective distribution boundaries.
    return std::max(med + delta - mnn_med, mnn_med - (mad - delta));
}

template<typename Index, typename Float>
Float limit_from_closest_distances(const NeighborSet<Index, Float>& found, Float nmads = 3) {
    // Pooling all distances together.
    std::vector<Float> all_distances;
    if (found.size()) {
        all_distances.reserve(found.size() * found[0].size());
    }

    for (const auto& f : found) {
        for (const auto& x : f) {
            all_distances.push_back(x.second);
        }
    }

    Float med = median(all_distances.size(), all_distances.begin());
    for (auto& a : all_distances) {
        a = std::abs(a - med);
    }
    Float mad = median(nobs, all_distances.begin());

    // Under normality, most of the distribution should be obtained
    // within 3 sigma of the correction vector. 
    return med + nmads * mad * static_cast<Float>(mad2sigma);
}

}

}

#endif
