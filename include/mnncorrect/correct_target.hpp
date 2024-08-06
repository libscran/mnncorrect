#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "fuse_nn_results.hpp"
#include "find_mutual_nns.hpp"
#include "robust_average.hpp"

#include <algorithm>
#include <vector>
#include <memory>

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_> 
void subset_to_mnns(size_t ndim, const Float_* data, const std::vector<Index_>& in_mnn, Float_* buffer) {
    size_t num_in_mnn = in_mnn.size();
    for (size_t f = 0; f < num_in_mnn; ++f) {
        size_t current = in_mnn[f];
        auto curdata = data + current * ndim; // already size_t's so there's no chance of overflow.
        std::copy_n(curdata, ndim, buffer);
        buffer += ndim;
    }
}

template<typename Dim_, typename Index_, typename Float_>
NeighborSet<Index_, Float_> identify_closest_mnn(size_t nobs, const Float_* data, const knncolle::Prebuilt<Dim_, Index_, Float_>& index, int k, size_t nobs_cap, int nthreads) {
    if (nobs_cap >= nobs) {
        return quick_find_nns(nobs, data, index, k, nthreads);
    }

    NeighborSet<Index_, Float_> output(nobs);
    size_t ndim = index.num_dimensions();

    // The gap guaranteed to be > 1 here, so there's no chance of jobs
    // overlapping if we apply any type of truncation or rounding.
    double gap = static_cast<double>(nobs) / nobs_cap; 

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#else
    MNNCORRECT_CUSTOM_PARALLEL(nobs_cap, [&](size_t start, size_t end) -> void {
#endif

        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = index.initialize();

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp for
#endif
        for (size_t o_ = 0; o_ < nobs_cap; ++o_) {
#else
        for (size_t o_ = start; o_ < end; ++o_) {
#endif

            size_t o = gap * o_; // truncation
            searcher->search(data + o * ndim, k, &indices, &distances);
            fill_pair_vector(indices, distances, output[o]);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    return output;
}

template<typename Index_, typename Float_>
Float_ limit_from_closest_distances(const NeighborSet<Index_, Float_>& found, Float_ nmads) {
    if (found.empty()) {
        return 0;        
    }

    // Pooling all distances together.
    std::vector<Float_> all_distances;
    {
        size_t full_size = 0;
        for (const auto& f : found) {
            full_size += f.size();
        }
        all_distances.reserve(full_size);
    }
    for (const auto& f : found) {
        for (const auto& x : f) {
            all_distances.push_back(x.second);
        }
    }

    // Computing the median and MAD. 
    Float_ med = median(all_distances.size(), all_distances.data());
    for (auto& a : all_distances) {
        a = std::abs(a - med);
    }
    Float_ mad = median(all_distances.size(), all_distances.data());

    // Under normality, most of the distribution should be obtained
    // within 3 sigma of the correction vector. 
    return med + nmads * mad * static_cast<Float_>(mad2sigma);
}

template<typename Index_, typename Float_>
void compute_center_of_mass(
    size_t ndim,
    const std::vector<Index_>& mnn_ids,
    const NeighborSet<Index_, Float_>& closest_mnn,
    const Float_* data,
    Float_* buffer,
    const RobustAverageOptions& raopt,
    Float_ limit,
    [[maybe_unused]] int nthreads)
{
    size_t num_mnns = mnn_ids.size();
    auto inverted = invert_neighbors(num_mnns, closest_mnn, limit);

#ifndef MNNCORRECT_CUSTOM_PARALLEL    
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#else
    MNNCORRECT_CUSTOM_PARALLEL(num_mnns, [&](size_t start, size_t end) -> void {
#endif

        std::vector<std::pair<Float_, size_t> > deltas;

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp for
#endif
        for (size_t g = 0; g < num_mnns; ++g) {
#else
        for (size_t g = start; g < end; ++g) {
#endif

            // Usually, the MNN is always included in its own neighbor list.
            // However, this may not be the case if a cap is applied. We don't
            // want to force it into the neighbor list, because that biases the
            // subsample towards the MNN (thus causing kissing effects). So, in
            // the unfortunate case when the MNN's neighbor list is empty, we
            // fall back to just setting the center of mass to the MNN itself.
            const auto& inv = inverted[g];
            auto output = buffer + g * ndim;
            if (inv.empty()) {
                auto ptr = data + static_cast<size_t>(mnn_ids[g]) * ndim;
                std::copy_n(ptr, ndim, output);
            } else {
                robust_average(ndim, inv, data, output, deltas, raopt);
            }
        }

#ifndef MNNCORRECT_CUSTOM_PARALLEL
    }
#else
    }, nthreads);
#endif

    return;
}

template<typename Dim_, typename Index_, typename Float_>
void correct_target(
    size_t ndim, 
    size_t nref, 
    const Float_* ref, 
    size_t ntarget, 
    const Float_* target, 
    const MnnPairs<Index_>& pairings, 
    const knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_>& builder, 
    int k, 
    Float_ nmads,
    int robust_iterations,
    double robust_trim, // yes, this is a double, not a Float_. Doesn't really matter given where we're using it.
    Float_* output,
    size_t nobs_cap,
    int nthreads) 
{
    auto uniq_ref = unique_left(pairings);
    auto uniq_target = unique_right(pairings);

    // Identify the closest MNN, with parallelized index building.
    std::vector<Float_> buffer_ref(uniq_ref.size() * ndim);
    std::vector<Float_> buffer_target(uniq_target.size() * ndim);
    std::unique_ptr<knncolle::Prebuilt<Dim_, Index_, Float_> > index_ref, index_target;

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for (int opt = 0; opt < 2; ++opt) {
#else
    MNNCORRECT_CUSTOM_PARALLEL(2, [&](int start, int end) -> void {
        for (int opt = start; opt < end; ++opt) {
#endif

            auto obs_ptr = (opt == 0 ? ref : target);
            const auto& uniq = (opt == 0 ? uniq_ref : uniq_target);
            auto& buffer = (opt == 0 ? buffer_ref : buffer_target);
            auto& index = (opt == 0 ? index_ref : index_target);

            subset_to_mnns(ndim, obs_ptr, uniq, buffer.data());
            index = builder.build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(ndim, uniq.size(), buffer.data()));

#ifndef MNNCORRECT_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    auto mnn_ref = identify_closest_mnn(nref, ref, *index_ref, k, nobs_cap, nthreads);
    index_ref.reset();
    auto mnn_target = identify_closest_mnn(ntarget, target, *index_target, k, nobs_cap, nthreads);
    index_target.reset();

    // Determine the expected width to use, again in parallel. 
    Float_ limit_closest_ref, limit_closest_target;

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        #pragma omp for
#endif
        for (int opt = 0; opt < 2; ++opt) {
#else
    MNNCORRECT_CUSTOM_PARALLEL(2, [&](int start, int end) -> void {
        for (int opt = start; opt < end; ++opt) {
#endif

            auto& limit = (opt == 0 ? limit_closest_ref : limit_closest_target);
            const auto& mnn = (opt == 0 ? mnn_ref : mnn_target);
            limit = limit_from_closest_distances(mnn, nmads);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    // Computing the centers of mass. We reuse the buffers to store the center coordinates.
    RobustAverageOptions raopt(robust_iterations, robust_trim);
    compute_center_of_mass(ndim, uniq_ref, mnn_ref, ref, buffer_ref.data(), raopt, limit_closest_ref, nthreads);
    compute_center_of_mass(ndim, uniq_target, mnn_target, target, buffer_target.data(), raopt, limit_closest_target, nthreads);

    // Computing the correction vector for each target point as a robust
    // average of the correction vectors for the closest MNN-involved cells,
    // and then applying it to the target data.
    auto remap_ref = invert_indices(nref, uniq_ref);
    auto remap_target = invert_indices(ntarget, uniq_target);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#else
    MNNCORRECT_CUSTOM_PARALLEL(ntarget, [&](size_t start, size_t end) -> void {
#endif

        std::vector<Float_> corrections;
        std::vector<std::pair<Float_, size_t> > deltas;

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp for
#endif
        for (size_t t = 0; t < ntarget; ++t) {
#else
        for (size_t t = start; t < end; ++t) {
#endif

            const auto& target_closest = mnn_target[t];
            corrections.clear();
            size_t ncorrections = 0;

            for (const auto& tc : target_closest) {
                const Float_* ptptr = buffer_target.data() + static_cast<size_t>(tc.first) * ndim; // cast to avoid overflow.
                const auto& ref_partners = pairings.matches.at(uniq_target[tc.first]);

                size_t old_size = corrections.size();
                corrections.resize(corrections.size() + ref_partners.size() * ndim);
                auto corptr = corrections.data() + old_size;

                for (auto rp : ref_partners) {
                    const Float_* prptr = buffer_ref.data() + static_cast<size_t>(remap_ref[rp]) * ndim; // cast to avoid overflow.
#ifdef _OPENMP
                    #pragma omp simd
#endif
                    for (size_t d = 0; d < ndim; ++d) {
                        corptr[d] = prptr[d] - ptptr[d];
                    }
                    corptr += ndim;
                    ++ncorrections;
                }
            }

            size_t toffset = static_cast<size_t>(t) * ndim; // cast to avoid overflow.
            auto optr = output + toffset;
            robust_average(ndim, ncorrections, corrections.data(), optr, deltas, raopt);

            auto tptr = target + toffset;
            for (size_t d = 0; d < ndim; ++d) {
                optr[d] += tptr[d];
            }

#ifndef MNNCORRECT_CUSTOM_PARALLEL
        }
    }
#else
        }
    }, nthreads);
#endif

    return;
}

}

}

#endif
