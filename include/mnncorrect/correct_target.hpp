#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "fuse_nn_results.hpp"
#include "find_mutual_nns.hpp"
#include "robust_average.hpp"
#include "parallelize.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <cassert>
#include <cstddef>

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_> 
void subset_matrix(std::size_t ndim, const Float_* data, const std::vector<Index_>& in_mnn, std::vector<Float_>& buffer) {
    auto num_in_mnn = in_mnn.size();
    buffer.resize(static_cast<std::size_t>(num_in_mnn) * ndim); // cast to size_t's to avoid overflow.
    for (decltype(num_in_mnn) f = 0; f < num_in_mnn; ++f) {
        auto curdata = data + static_cast<std::size_t>(in_mnn[f]) * ndim; // ditto.
        std::copy_n(curdata, ndim, buffer.data() + static_cast<std::size_t>(f) * ndim); // ditto.
    }
}

template<typename Index_, typename Float_, typename Matrix_>
void correct_target(
    std::size_t ndim, 
    Index_ nref, 
    const Float_* ref,
    const std::vector<Index_>& ref_centers,
    Index_ ntarget, 
    const Float_* target, 
    const std::vector<Index_>& target_centers,
    const MnnPairs<Index_>& pairings, 
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder, 
    int k, 
    int robust_iterations,
    double robust_trim, // yes, this is a double, not a Float_. Doesn't really matter given where we're using it.
    Float_* output,
    int nthreads) 
{
    std::vector<Float_> buffer_ref, buffer_target;

    // Parallelized building of the center-only indices.
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > center_index_ref, center_index_target;
    parallelize(nthreads, 2, [&](int, int start, int length) -> void {
        for (int opt = start, end = start + length; opt < end; ++opt) {
            auto obs_ptr = (opt == 0 ? ref : target);
            const auto& uniq = (opt == 0 ? ref_centers : target_centers);
            auto& buffer = (opt == 0 ? buffer_ref : buffer_target);
            auto& index = (opt == 0 ? center_index_ref : center_index_target);
            subset_matrix(ndim, obs_ptr, uniq, buffer);
            index = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(ndim, uniq.size(), buffer.data()));
        }
    });

    // Find closest center for each MNN-involved point.
    auto uniq_mnn_ref = unique_left(pairings);
    auto uniq_mnn_target = unique_right(pairings);
    std::vector<Index_> ref_chosen_center(nref, -1);
    std::vector<Index_> target_chosen_center(ntarget, -1);

    for (int i = 0; i < 2; ++i) {
        const auto& uniq_mnn = (i == 0 ? uniq_mnn_ref : uniq_mnn_target);
        const auto& index = (i == 0 ? center_index_ref : center_index_target);
        const auto values = (i == 0 ? ref : target);
        const auto& centers = (i == 0 ? ref_centers : target_centers);
        auto& chosen = (i == 0 ? ref_chosen_center : target_chosen_center);

        auto num_uniq = uniq_mnn.size();
        parallelize(nthreads, num_uniq, [&](int, decltype(num_uniq) start, decltype(num_uniq) length) -> void {
            std::vector<Index_> curindices;
            auto searcher = index->initialize();
            for (decltype(start) i = start, end = start + length; i < end; ++i) {
                auto curmnn = uniq_mnn[i];
                searcher->search(values + static_cast<std::size_t>(curmnn) * ndim, 1, &curindices, NULL); // cast to avoid overflow.
                chosen[curmnn] = centers[curindices.back()];
            }
        });
    }

    // Computing the correction vector for each target point as a robust
    // average of the correction vectors for the closest MNN-involved cells,
    // and then applying it to the target data.
    subset_matrix(ndim, target, uniq_mnn_target, buffer_target);
    auto mnn_index_target = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(ndim, uniq_mnn_target.size(), buffer_target.data()));

    RobustAverageOptions raopt(robust_iterations, robust_trim);
    parallelize(nthreads, ntarget, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> target_closest;
        auto searcher = mnn_index_target->initialize();
        std::vector<Float_> corrections;
        RobustAverageWorkspace<Float_> rawork;

        for (Index_ t = start, end = start + length; t < end; ++t) {
            std::size_t toffset = static_cast<std::size_t>(t) * ndim; // cast to avoid overflow.
            searcher->search(target + toffset, k, &target_closest, NULL);

            corrections.clear();
            decltype(corrections.size()) ncorrections = 0; // the number of correction vectors (i.e., pairs), which could be larger than the number of observations and beyond an Index_.
            for (const auto& tc : target_closest) {
                auto mnn = uniq_mnn_target[tc];
                const auto& ref_partners = pairings.matches.at(mnn);

                auto old_size = corrections.size();
                std::size_t num_to_add = ref_partners.size();
                corrections.insert( // unlike resize(), insert() gives us a chance to bad_alloc() if the request exceeds the max capacity of std::vector.
                    corrections.end(),
                    num_to_add * ndim, // both size_t's, no need to cast.
                    static_cast<Float_>(0)
                );
                auto corptr = corrections.data() + old_size; // make sure this is after the insert(), otherwise we could invalidate on reallocation.
                ncorrections += num_to_add;

                const Float_* ptptr = target + static_cast<std::size_t>(target_chosen_center[mnn]) * ndim; // cast to avoid overflow.
                for (auto rp : ref_partners) {
                    const Float_* prptr = ref + static_cast<std::size_t>(ref_chosen_center[rp]) * ndim; // cast to avoid overflow.
                    for (std::size_t d = 0; d < ndim; ++d) {
                        corptr[d] = prptr[d] - ptptr[d];
                    }
                    corptr += ndim;
                }
            }

            auto optr = output + toffset;
            robust_average(ndim, ncorrections, corrections.data(), optr, rawork, raopt);
            auto tptr = target + toffset;
            for (std::size_t d = 0; d < ndim; ++d) {
                optr[d] += tptr[d];
            }
        }
    });

    return;
}

}

}

#endif
