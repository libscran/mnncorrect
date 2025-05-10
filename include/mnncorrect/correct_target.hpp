#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "populate_cross_neighbors.hpp"
#include "parallelize.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct CorrectTargetWorkspace {
    std::vector<Float_> ref_buffer, target_buffer;
    NeighborSet<Index_, Float_> neighbor_from, neighbor_to;
    std::vector<Index_> mapping;
    std::vector<Float_> ref_var_buffer, target_var_buffer; 
};

template<typename Index_, typename Float_, class Matrix_> 
std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > build_mnn_only_index(
    std::size_t ndim,
    const Float_* data,
    const std::vector<Index_>& in_mnn,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    std::vector<Float_>& buffer)
{
    auto num_in_mnn = in_mnn.size();
    buffer.resize(ndim * static_cast<std::size_t>(num_in_mnn));
    for (decltype(num_in_mnn) f = 0; f < num_in_mnn; ++f) {
        auto curdata = data + static_cast<std::size_t>(in_mnn[f]) * ndim; // cast to size_t's to avoid overflow.
        std::copy_n(curdata, ndim, buffer.begin() + static_cast<std::size_t>(f) * ndim); // also casting to avoid overflow.
    }
    return builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(ndim, uniq.size(), buffer.data()));
}

// Ensure each MNN-involved observation is part of its own neighbor set,
// for simplicity. This is also more consistent with the expectation that
// 'num_neighbors' is the lower bound on any subpopulation size; if we
// excluded the self-neighbor, the lower bound with be 'num_neighbors + 1'.
template<typename Index_, typename Float_>
void force_self(std::vector<std::pair<Index_, Float_> >& current_neighbors, Index_ self, int num_neighbors) { 
    for (const auto& y : current_neighbors) {
        // Technically we could break out if y.second is not 0. But we should
        // aim to be robust against NN algorithms where the distance to self is
        // not exactly zero due to numerical precision issues.
        if (y.first == self) {
            return;
        }
    }

    if (current_neighbors.size() == static_cast<std::size_t>(num_neighbors)) {
        current_neighbors.pop_back(); // pop first to avoid re-allocation. 
    }
    current_neighbors.insert(current_neighbors.begin(), std::pair<Index_, Float_>(self, 0));
}

// Find all neighbors of each MNN-involved observation, only in its corresponding metabatch (i.e., reference or target). 
// Here, 'output' will contain neighbor indices relative to the entire dataset.
template<typename Index_, typename Float_>
void search_for_neighbors_from_mnns(
    Index_ num_total,
    const std::vector<Index_>& ref_mnns_unique,
    const std::vector<Index_>& target_mnns,
    const Float_* data,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const BatchInfo<Index_, Float_>& target,
    int num_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    output.resize(num_total);
    populate_batch_neighbors<Index_, Float_>(
        target_mnns.size(),
        [&](Index_ i) -> Index_ { return target_mnn[i]; },
        data,
        target,
        num_neighbors,
        false,
        num_threads,
        output
    );
    for (auto t : target_mnns) {
        force_self(output[t], t, num_neighbors);
    }

    for (decltype(references.size()) b = 0, end = references.size(); b < end; ++b) {
        populate_batch_neighbors<Index_, Float_>(
            ref_mnns_unique.size(),
            [&](Index_ i) -> Index_ { return ref_mnns_unique[i]; },
            data,
            references[b],
            num_neighbors,
            b > 0,
            num_threads,
            output
        );
    }
    for (auto r : ref_mnns_unique) {
        force_self(output[r], r, num_neighbors);
    }
}

// Find the MNN-involved observations that are neighbors of each observation in its corresponding metabatch (i.e., reference or target). 
// Here, 'output' will contain neighbor indices relative to the subset of MNNs, i.e., 'mnns[output[0][0].first]' is the original index.
template<typename Index_, typename Float_>
void search_for_neighbors_to_mnns(
    const std::vector<Index_>& ids,
    const Float_* data,
    const std::vector<Index_>& mnns,
    const knncolle::Prebuilt<Index_, Float_, Float_>& mnn_index,
    int num_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    Index_ num_obs = ids.size();
    parallelize(num_threads, num_obs, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = mnn_index->initialize();

        for (Index_ l = start, end = start + length; l < end; ++l) {
            auto k = ids[l];
            auto ptr = data + static_cast<std::size_t>(k) * ndim;
            searcher->search(ptr, num_neighbors, &indices, &distances);
            auto& curnn = output[k];
            fill_pair_vector(indices, distances, curnn);
        }
    });

    for (Index_ m = 0, mend = mnns.size(); m < mend; ++m) {
        force_self(output[mnns[m]], m, num_neighbors);
    }
}

template<typename Index_, typename Float_>
void search_for_neighbors_to_mnns(
    Index_ num_total,
    const std::vector<Index_>& ref_ids,
    const std::vector<Index_>& target_ids,
    const Float_* data,
    const std::vector<Index_>& ref_mnns_unique,
    const knncolle::Prebuilt<Index_, Float_, Float_>& ref_mnn_index,
    const std::vector<Index_>& target_mnns,
    const knncolle::Prebuilt<Index_, Float_, Float_>& target_mnn_index,
    int num_neighbors,
    int num_threads,
    NeighborSet<Index_, Float_>& output)
{
    output.resize(num_total);
    search_for_neighbors_to_mnns(ref_ids, data, ref_mnns_unique, ref_mnn_index, num_neighbors, num_threads, output);
    search_for_neighbors_to_mnns(target_ids, data, target_mnns, target_mnn_index, num_neighbors, num_threads, output);
}

// Here, output contains indices relative to the entire dataset again.  We
// return a NeighborSet rather than reusing it, because the inner vectors could
// be arbitrarily long and of variable length between merge steps; we want to
// avoid accumulating very long allocations for the inner vectors.
template<typename Index_, typename Distance_>
NeighborSet<Index_, Distance_> invert_neighbors(Index_ num_mnns, const std::vector<Index_>& in_batch, const NeighborSet<Index_, Distance_>& neighbors, int num_threads) {
    NeighborSet<Index_, Distance_> output(num_mnns);
    Index_ num_in_batch = in_batch.size();
    for (Index_ i = 0; i < num_in_batch; ++i) {
        for (const auto& x : neighbors[i]) {
            output[x.first].emplace_back(i, x.second);
        }
    }

    parallelize(num_threads, num_mnns, [&](int, Index_ start, Index_ length) -> void {
        for (Index_ i = start, end = start + length; i < end; ++i) {
            auto& curout = output[i];
            std::sort(
                curout.begin(),
                curout.end(),
                [](const std::pair<Index_, Distance_>& left, const std::pair<Index_, Distance_>& right) -> bool {
                    if (left.second == right.second) {
                        return left.first < right.first;
                    } else {
                        return left.second < right.second;
                    }
                }
            );
        }
    });

    return output;
}

template<typename Index_, typename Float_>
void compute_initial_center_of_mass(
    std::size_t ndim,
    const std::vector<Index_>& mnns,
    const NeighborSet<Index_, Float_>& neighbors,
    const Float_* data,
    int num_threads,
    double tolerance,
    std::vector<double>& running_mean,
    std::vector<double>& running_variance)
{
    Index_ num_mnns = mnns.size();
    running_mean.resize(static_cast<std::size_t>(num_mnns) * ndim);
    running_variance.resize(running_mean.size());

    parallelize(nthreads, num_mnns, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Float_> mean(ndim), sum_squares(ndim);

        for (Index_ g = start, end = start + length; g < end; ++g) {
            auto m = mnns[g];
            const auto& inv = neighbors[m];

            // Using Welford's algorithm to compute the running mean and
            // variance around the current center of mass. We filter out
            // points that are outliers in any dimension. 
            std::fill(mean.begin(), mean.end(), 0);
            std::fill(sum_squares.begin(), sum_squares.end(), 0);
            Index_ counter = 0;

            for (auto nn : inv) {
                auto target = data + static_cast<std::size_t>(nn.first) * ndim; // cast to avoid overflow.

                ++counter;
                for (std::size_t d = 0; d < ndim; ++d) {
                    auto val = target[d];
                    auto& curmean = mean[d];
                    double delta = val - curmean;
                    curmean += delta / counter;
                    sum_squares[d] += delta * (val - curmean);
                }
            }

            std::size_t output_offset = ndim * static_cast<std::size_t>(g);
            std::copy(mean.begin(), mean.end(), running_mean.begin() + output_offset);
            std::copy(sum_squares.begin(), sum_squares.end(), running_variance.begin() + output_offset);
        }
    });

    return;
}

template<typename Index_, typename Float_>
void correct_target(
    Index_ num_total,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const std::vector<Index_>& ref_ids,
    const BatchInfo<Index_, Float_>& target,
    const std::vector<Index_>& target_ids,
    Float_* data,
    const FindClosestMnnWorkspace<Index_>& mnns,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder, 
    int num_neighbors,
    int num_threads,
    double tolerance,
    CorrectTargetWorkspace<Index_, Float_>& workspace) 
{
    // Build this first so that we can re-use the ref_buffers and target_buffers for the center of mass calculations.
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > ref_mnn_index;
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > target_mnn_index;
    parallelize(nthreads, 2, [&](int, int start, int length) -> void {
        for (int opt = start, end = start + length; opt < end; ++opt) {
            const auto& uniq = (opt == 0 ? mnns.ref_mnns_unique : mnns.target_mnns);
            auto& buffer = (opt == 0 ? workspace.ref_buffer : workspace.target_buffer);
            auto& index = (opt == 0 ? ref_mnn_index: target_mnn_index);
            index = build_mnn_only_index(ndim, data, uniq, builder, buffer);
        }
    });

    workspace.mapping.resize(num_total);
    for (decltype(mnns.ref_ids.size()) rx = 0, rend = mnns.ref_ids.size(); r < rend; ++r) {
        workspace.mapping[mnns.ref_ids[rx]] = rx;
    }
    for (decltype(mnns.target_ids.size()) tx = 0, tend = mnns.target_ids.size(); t < tend; ++t) {
        workspace.mapping[mnns.target_ids[tx]] = tx;
    }

    // Computing an initial center of mass for each MNN-involved observation.
    search_for_neighbors_from_mnns(
        num_total,
        mnns.ref_mnns_unique,
        mnns.target_mnns,
        data,
        mnn_indices.ref_index,
        mnn_indices.target_index,
        num_neighbors,
        num_threads,
        workspace.neighbors_from
    );
    compute_initial_center_of_mass(mnns.ref_mnns_unique, workspace.neighbors, data, workspace.ref_buffer, num_threads, workspace.ref_var_buffer);
    compute_initial_center_of_mass(mnns.target_mnns, workspace.neighbors, data, workspace.target_buffer, num_threads, workspace.target_var_buffer);

    // Find the closest MNN-involved observation(s) to each observation in either metabatch.
    search_for_neighbors_to_mnns(
        num_total,
        ref_ids,
        target_ids,
        data,
        mnns.ref_mnns_unique,
        ref_mnn_index,
        mnns.target_mnns,
        target_mnn_index,
        num_neighbors,
        num_threads,
        workspace.neighbors_to
    );

    // Refining the center of mass.
    auto ref_inverted = invert_neighbors<Index_, Float_>(mnns.ref_mnns_unique.size(), ref_ids, workspace.neighbors_to, num_threads);
    auto target_inverted = invert_neighbors<Index_, Float_>(mnns.target_mnns.size(), target_ids, workspace.neighbors_to, num_threads);
    refine_center_of_mass(mnns.ref_mnns_unique, correct_workspace.neighbors, data, correct_workspace.ref_buffer, num_threads, correct_workspace.ref_var_buffer);
    refine_center_of_mass(mnns.target_mnns, correct_workspace.neighbors, data, correct_workspace.target_buffer, num_threads, correct_workspace.target_var_buffer);

    // Apply the correction in the target based on its closest MNN-involved cell, and report the index of the closest MNN-involved cell.

}

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
