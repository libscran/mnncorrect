#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_mutual_nns.hpp"
#include "RobustAverage.hpp"

#include <algorithm>
#include <vector>

namespace mnncorrect {

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
    if (found.empty()) {
        return 0;        
    }

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

template<typename Index, typename Float>
void compute_center_of_mass(int ndim, size_t num_mnns, const NeighborSet<Index, Float>& closest_mnn, const Float* data, Float* buffer, int iterations, double trim, Float limit) {
    auto inverted = invert_neighbors(num_mnns, closest_mnn);
    #pragma omp parallel
    {
        RobustAverage<Index, Float> rbave(iterations, trim, limit);
        #pragma omp for
        for (size_t g = 0; g < num_mnns; ++g) {
            rbave.run(ndim, inverted[g], data, buffer + g * ndim);
        }
    }
    return;
}

template<typename Index, typename Float, class Builder>
void correct_target(
    int ndim, 
    size_t nref, 
    const Float* ref, 
    size_t ntarget, 
    const Float* target, 
    const MnnPairs<Index>& pairings, 
    Builder bfun, 
    int k, 
    Float nmads,
    int robust_iterations,
    Float robust_trim,
    Float* output) 
{
    auto uniq_ref = unique_left(pairings);
    auto uniq_target = unique_right(pairings);

    // Determine the expected width to use. 
    std::vector<Float> buffer_ref(uniq_ref.size() * ndim);
    auto mnn_ref = identify_closest_mnn(ndim, nref, ref, uniq_ref, bfun, k, buffer_ref.data());
    Float limit_closest_ref = limit_from_closest_distances(mnn_ref, nmads);

    std::vector<Float> buffer_target(uniq_target.size() * ndim);
    auto mnn_target = identify_closest_mnn(ndim, ntarget, target, uniq_target, bfun, k, buffer_target.data());
    Float limit_closest_target = limit_from_closest_distances(mnn_target, nmads);

    // Computing the centers of mass, stored in the buffers.
    compute_center_of_mass(ndim, uniq_ref.size(), mnn_ref, ref, buffer_ref.data(), robust_iterations, robust_trim, limit_closest_ref);
    compute_center_of_mass(ndim, uniq_target.size(), mnn_target, target, buffer_target.data(), robust_iterations, robust_trim, limit_closest_target);

    // Computing the correction vector for each target point, 
    // And then applying it to the target data.
    auto remap_ref = invert_indices(nref, uniq_ref);
    auto remap_target = invert_indices(ntarget, uniq_target);

    #pragma omp parallel
    {
        std::vector<Float> corrections;
        corrections.reserve(ndim * 100); // just filling it with something to avoid initial allocations.
        RobustAverage<Index, Float> rbave(robust_iterations, robust_trim);
        
        #pragma omp for
        for (size_t t = 0; t < ntarget; ++t) {
            const auto& target_closest = mnn_target[t];
            corrections.clear();
            int ncorrections = 0;

            for (auto tc : target_closest) {
                const Float* ptptr = buffer_target.data() + tc.first * ndim;
                const auto& ref_partners = pairings.matches.at(uniq_target[tc.first]);

                for (auto rp : ref_partners) {
                    const Float* prptr = buffer_ref.data() + remap_ref[rp] * ndim;
                    for (int d = 0; d < ndim; ++d) {
                        corrections.push_back(prptr[d] - ptptr[d]);
                    }
                    ++ncorrections;
                }
            }

            auto optr = output + t * ndim;
            rbave.run(ndim, ncorrections, corrections.data(), optr);

            // Actually applying the correction.
            auto tptr = target + t * ndim;
            for (int d = 0; d < ndim; ++d) {
                optr[d] += tptr[d];
            }
        }
    }

    return;
}

}

#endif
