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
void subset_to_mnns(std::size_t ndim, const Float_* data, const std::vector<Index_>& in_mnn, Float_* buffer) {
    auto num_in_mnn = in_mnn.size();
    for (decltype(num_in_mnn) f = 0; f < num_in_mnn; ++f) {
        auto curdata = data + static_cast<std::size_t>(in_mnn[f]) * ndim; // cast to size_t's to avoid overflow.
        std::copy_n(curdata, ndim, buffer + static_cast<std::size_t>(f) * ndim); // also casting to avoid overflow.
    }
}

template<typename Index_>
Index_ capped_index(Index_ i, double gap) {
    return static_cast<double>(i) * gap; // truncation.
}

template<typename Index_, typename Float_>
std::pair<double, NeighborSet<Index_, Float_> > capped_find_nns(
    Index_ nobs,
    const Float_* data,
    const knncolle::Prebuilt<Index_, Float_, Float_>& index,
    int k,
    Index_ mass_cap,
    [[maybe_unused]] int nthreads) 
{
    // This function should only be called if nobs > mass_cap, so the gap is
    // guaranteed to be > 1 here; there's no chance of jobs overlapping if
    // we apply any type of truncation or rounding.
    assert(nobs > mass_cap);
    double gap = static_cast<double>(nobs) / mass_cap; 

    NeighborSet<Index_, Float_> output(mass_cap);
    std::size_t ndim = index.num_dimensions();

    parallelize(nthreads, mass_cap, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = index.initialize();

        for (Index_ raw_o = start, end = start + length; raw_o < end; ++raw_o) {
            Index_ o = capped_index(raw_o, gap);
            searcher->search(data + static_cast<std::size_t>(o) * ndim, k, &indices, &distances);
            fill_pair_vector(indices, distances, output[raw_o]);
        }
    });

    return std::make_pair(gap, std::move(output));
}

template<typename Float_>
Float_ median(std::size_t n, Float_* ptr) {
    if (!n) {
        return std::numeric_limits<Float_>::quiet_NaN();
    }
    std::size_t half = n / 2;
    bool is_even = n % 2 == 0;

    std::nth_element(ptr, ptr + half, ptr + n);
    Float_ mid = *(ptr + half);
    if (!is_even) {
        return mid;
    }

    std::nth_element(ptr, ptr + half - 1, ptr + n);
    return (mid + *(ptr + half - 1)) / 2;
}

template<typename Index_, typename Float_>
Float_ limit_from_closest_distances(const NeighborSet<Index_, Float_>& found, Float_ nmads) {
    if (found.empty()) {
        return 0;        
    }

    // Pooling all distances together.
    std::vector<Float_> all_distances;
    {
        std::size_t full_size = 0;
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
    constexpr double mad2sigma = 1.4826;
    return med + nmads * mad * static_cast<Float_>(mad2sigma);
}

template<typename Index_, typename Distance_>
std::vector<std::vector<Index_> > invert_neighbors(std::size_t n, const NeighborSet<Index_, Distance_>& neighbors, Distance_ limit) {
    std::vector<std::vector<Index_> > output(n);
    const Index_ num_neighbors = neighbors.size();
    for (Index_ i = 0; i < num_neighbors; ++i) {
        for (const auto& x : neighbors[i]) {
            if (x.second <= limit) {
                output[x.first].push_back(i);
            }
        }
    }
    return output;
}

template<typename Index_>
std::vector<Index_> invert_indices(std::size_t n, const std::vector<Index_>& uniq) {
    std::vector<Index_> output(n, static_cast<Index_>(-1)); // any value is fine, we don't check this anyway.
    Index_ num_uniq = uniq.size();
    for (Index_ u = 0; u < num_uniq; ++u) {
        output[uniq[u]] = u;
    }
    return output;
}

template<typename Index_, typename Float_>
void compute_center_of_mass(
    std::size_t ndim,
    const std::vector<Index_>& mnn_ids,
    const std::vector<std::vector<Index_> >& mnn_neighbors,
    const Float_* data,
    Float_* buffer,
    const RobustAverageOptions& raopt,
    int nthreads)
{
    Index_ num_mnns = mnn_ids.size();

    parallelize(nthreads, num_mnns, [&](int, Index_ start, Index_ length) -> void {
        RobustAverageWorkspace<Float_> rawork;

        for (Index_ g = start, end = start + length; g < end; ++g) {
            const auto& inv = mnn_neighbors[g];
            auto output = buffer + static_cast<std::size_t>(g) * ndim; // cast to avoid overflow.

            // Usually, the MNN is always included in its own neighbor list.
            // However, this may not be the case if a cap is applied. We don't
            // want to force it into the neighbor list, because that biases the
            // subsample towards the MNN (thus causing kissing effects). So, in
            // the unfortunate case when the MNN's neighbor list is empty, we
            // fall back to just setting the center of mass to the MNN itself.
            if (inv.empty()) {
                auto ptr = data + static_cast<std::size_t>(mnn_ids[g]) * ndim; // cast to avoid overflow. 
                std::copy_n(ptr, ndim, output);
            } else {
                robust_average(ndim, inv, data, output, rawork, raopt);
            }
        }
    });

    return;
}

template<typename Index_, typename Float_, typename Matrix_>
void correct_target(
    size_t ndim, 
    Index_ nref, 
    const Float_* ref, 
    Index_ ntarget, 
    const Float_* target, 
    const MnnPairs<Index_>& pairings, 
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder, 
    int k, 
    Float_ nmads,
    int robust_iterations,
    double robust_trim, // yes, this is a double, not a Float_. Doesn't really matter given where we're using it.
    Float_* output,
    Index_ mass_cap,
    int nthreads) 
{
    auto uniq_mnn_ref = unique_left(pairings);
    auto uniq_mnn_target = unique_right(pairings);

    // Parallelized building of the MNN-only indices.
    std::vector<Float_> buffer_ref(uniq_mnn_ref.size() * ndim);
    std::vector<Float_> buffer_target(uniq_mnn_target.size() * ndim);
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > index_ref, index_target;

    parallelize(nthreads, 2, [&](int, int start, int length) -> void {
        for (int opt = start, end = start + length; opt < end; ++opt) {
            auto obs_ptr = (opt == 0 ? ref : target);
            const auto& uniq = (opt == 0 ? uniq_mnn_ref : uniq_mnn_target);
            auto& buffer = (opt == 0 ? buffer_ref : buffer_target);
            auto& index = (opt == 0 ? index_ref : index_target);

            subset_to_mnns(ndim, obs_ptr, uniq, buffer.data());
            index = builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(ndim, uniq.size(), buffer.data()));
        }
    });

    // Finding the closest MNN-involved point for each point in the reference
    // dataset. If 'mass_cap' is smaller than 'nobs', we only do this search
    // for every x-th point where 'x = nobs / mass_cap'; this subsamples the
    // reference to save time in the mass calculation. In particular, this
    // amortizes the cost of the mass calculation over multiple merge steps to
    // counter the growth of the reference itself after each step (which would
    // otherwise cause each mass calculation to take longer).
    bool is_capped = mass_cap > 0 && mass_cap < nref;
    double gap = 0;
    NeighborSet<Index_, Float_> closest_mnn_ref;
    if (is_capped) {
        auto capped_out = capped_find_nns(nref, ref, *index_ref, k, mass_cap, nthreads);
        gap = capped_out.first;
        closest_mnn_ref = std::move(capped_out.second);
    } else {
        closest_mnn_ref = quick_find_nns(nref, ref, *index_ref, k, nthreads);
    }
    index_ref.reset();

    // Don't apply the cap when searching for the closest MNN to each target
    // point, as we need the MNN-neighbor information for each target point to
    // compute its correction. Besides, the cap is only intended to avoid
    // issues with the growth of the reference.
    auto closest_mnn_target = quick_find_nns(ntarget, target, *index_target, k, nthreads);
    index_target.reset();

    // Parallelized limit calculation for reference/target.
    Float_ limit_closest_ref, limit_closest_target;
    parallelize(nthreads, 2, [&](int, int start, int length) -> void {
        for (int opt = start, end = start + length; opt < end; ++opt) {
            auto& limit = (opt == 0 ? limit_closest_ref : limit_closest_target);
            const auto& mnn = (opt == 0 ? closest_mnn_ref : closest_mnn_target);
            limit = limit_from_closest_distances(mnn, nmads);
        }
    });

    // Computing the centers of mass. We reuse the buffers to store the center coordinates.
    RobustAverageOptions raopt(robust_iterations, robust_trim);
    {
        auto ref_inverted = invert_neighbors(uniq_mnn_ref.size(), closest_mnn_ref, limit_closest_ref);
        if (is_capped) {
            for (auto& ref_neighbors : ref_inverted) {
                for (auto& x : ref_neighbors) {
                    x = capped_index(x, gap);
                }
            }
        }
        compute_center_of_mass(ndim, uniq_mnn_ref, ref_inverted, ref, buffer_ref.data(), raopt, nthreads);

        auto target_inverted = invert_neighbors(uniq_mnn_target.size(), closest_mnn_target, limit_closest_target);
        compute_center_of_mass(ndim, uniq_mnn_target, target_inverted, target, buffer_target.data(), raopt, nthreads);
    }

    auto remap_ref = invert_indices(nref, uniq_mnn_ref);

    // Computing the correction vector for each target point as a robust
    // average of the correction vectors for the closest MNN-involved cells,
    // and then applying it to the target data.
    parallelize(nthreads, ntarget, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Float_> corrections;
        RobustAverageWorkspace<Float_> rawork;

        for (Index_ t = start, end = start + length; t < end; ++t) {
            const auto& target_closest = closest_mnn_target[t];
            corrections.clear();
            std::size_t ncorrections = 0; // the number of correction vectors (i.e., pairs), which could be larger than the number of observations and beyond an Index_.

            for (const auto& tc : target_closest) {
                const Float_* ptptr = buffer_target.data() + static_cast<std::size_t>(tc.first) * ndim; // cast to avoid overflow.
                const auto& ref_partners = pairings.matches.at(uniq_mnn_target[tc.first]);

                auto old_size = corrections.size();
                std::size_t num_to_add = ref_partners.size();
                corrections.resize(old_size + num_to_add * ndim); // both size_t's no need to cast.
                auto corptr = corrections.data() + old_size;
                ncorrections += num_to_add;

                for (auto rp : ref_partners) {
                    const Float_* prptr = buffer_ref.data() + static_cast<std::size_t>(remap_ref[rp]) * ndim; // cast to avoid overflow.
                    for (std::size_t d = 0; d < ndim; ++d) {
                        corptr[d] = prptr[d] - ptptr[d];
                    }
                    corptr += ndim;
                }
            }

            std::size_t toffset = static_cast<std::size_t>(t) * ndim; // cast to avoid overflow.
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
