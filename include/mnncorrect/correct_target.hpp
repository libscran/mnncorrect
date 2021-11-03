#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include "utils.hpp"
#include "knncolle/knncolle.hpp"
#include "determine_limits.hpp"
#include <algorithm>
#include <vector>

namespace mnncorrect {

template<typename Index, typename Float>
void compute_center_of_mass(int ndim, size_t nmnns, const NeighborSet<Index, Float>& closest_mnn, const Float* data, Float limit, Float* output) {
    std::fill(output, output + nmnns * ndim, 0);

    std::vector<Float> total(nmnns);
    for (size_t f = 0; f < closest_mnn.size(); ++f) {
        const Float* curdata = data + f * ndim;
        const auto& my_mnns = closest_mnn[f];

        for (const auto& x : my_mnns) {
            if (x.second <= limit) {
                Float* curout = output + x.first * ndim;
                for (int d = 0; d < ndim; ++d) {
                    curout[d] += curdata[d];
                }
                ++total[x.first];
            }
        }
    }

    for (size_t i = 0; i < nmnns; ++i) {
        Float* curout = output + i * ndim;
        for (int d = 0; d < ndim; ++d) {
            curout[d] /= total[i];
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
    Float* output) 
{
    auto uniq_ref = unique(pairings.left);
    auto uniq_target = unique(pairings.right);

    // Determine the expected width to use. 
    auto ave_vector = average_batch_vector(ndim, nref, ref, ntarget, target, pairings);
    Float limit_batch_ref = limit_from_batch_vector(ndim, nref, ref, ave_vector, uniq_ref, nmads); 
    Float limit_batch_target = limit_from_batch_vector(ndim, ntarget, target, ave_vector, uniq_target, nmads); 

    std::vector<Float> buffer_ref(uniq_ref.size() * ndim);
    auto mnn_ref = identify_closest_mnn(ndim, nref, ref, uniq_ref, bfun, k, buffer_ref.data());
    Float limit_closest_ref = limit_from_closest_distances(mnn_ref, nmads);

    std::vector<Float> buffer_target(uniq_target.size() * ndim);
    auto mnn_target = identify_closest_mnn(ndim, ntarget, target, uniq_target, bfun, k, buffer_target.data());
    Float limit_closest_target = limit_from_closest_distances(mnn_target, nmads);

    // Computing the centers of mass, stored in the buffers.
    Float limit_ref = std::min(limit_batch_ref, limit_closest_ref);
    compute_center_of_mass(ndim, uniq_ref.size(), mnn_ref, ref, limit_ref, buffer_ref.data());

    Float limit_target = std::min(limit_batch_target, limit_closest_target);
    compute_center_of_mass(ndim, uniq_target.size(), mnn_target, target, limit_target, buffer_target.data());

    // Computing the correction vector for each target point in an MNN pair, stored in the target buffer.
    auto remap_ref = invert_index(nref, uniq_ref);
    auto remap_target = invert_index(ntarget, uniq_target);

    std::vector<Float> weights(ntarget);
    for (auto x : pairings.right) {
        ++weights[x];
    }

    for (size_t p = 0; p < pairings.size(); ++p) {
        auto r = buffer_ref.data() + ndim * remap_ref[pairings.left[p]];
        auto t = buffer_target.data() + ndim * remap_target[pairings.right[p]];
        auto w = 1 / weights[pairings.right[p]];
        for (int d = 0; d < ndim; ++d) {
            t[d] -= r[d] * w;
        }
    }

    // And then applying it to the target data.
    for (size_t t = 0; t < ntarget; ++t) {
        auto src = target + t * ndim;
        auto out = output + t * ndim;
        std::copy(src, src + ndim, out);

        const auto& my_mnns = mnn_target[t];
        for (const auto& x : my_mnns) {
            auto corr = buffer_target.data() + x.first * ndim;
            for (int d = 0; d < ndim; ++d) {
                out[d] -= corr[d] / my_mnns.size();
            }
        }
    }

    return;
}

/* For testing purposes only. */
template<typename Index, typename Float>
void correct_target(int ndim, size_t nref, const Float* ref, size_t ntarget, const Float* target, const MnnPairs<Index>& pairings, int k, Float nmads, Float* output) {
    typedef knncolle::Base<Index, Float> knncolleBase;
    auto builder = [](int nd, size_t no, const Float* d) -> auto { 
        return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d)); 
    };
    correct_target(ndim, nref, ref, ntarget, target, pairings, builder, k, nmads, output);
    return;
}

}

#endif
