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
    std::vector<Index_> chosen_batch;
};

template<typename Index_, typename Float_, class Matrix_> 
std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > build_mnn_only_index(
    std::size_t num_dim,
    const Float_* data,
    const std::vector<Index_>& in_mnn,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    std::vector<Float_>& buffer)
{
    auto num_in_mnn = in_mnn.size();
    buffer.resize(num_dim * static_cast<std::size_t>(num_in_mnn));
    for (decltype(num_in_mnn) f = 0; f < num_in_mnn; ++f) {
        auto curdata = data + static_cast<std::size_t>(in_mnn[f]) * num_dim; // cast to size_t's to avoid overflow.
        std::copy_n(curdata, num_dim, buffer.begin() + static_cast<std::size_t>(f) * num_dim); // also casting to avoid overflow.
    }
    return builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, uniq.size(), buffer.data()));
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

// While knncolle's interface guarantees that the output neighbors are sorted
// by distance, it doesn't say anything about the order of neighbors when they
// are tied. So we just make sure that earlier indices are sorted first,
// which is useful for ensuring that sorted vectors can be compared safely.
template<typename Index_, typename Distance_>
struct SortBySecond {
    bool operator()(const std::pair<Index_, Distance_>& left, const std::pair<Index_, Distance_>& right) const {
        if (left.second == right.second) {
            return left.first < right.first;
        } else {
            return left.second < right.second;
        }
    }
};

template<typename Index_, typename Distance_>
void ensure_sort(std::vector<std::pair<Index_, Float_> >& current_neighbors) { 
    SortBySecond<Index_, Distance_> cmp;
    if (std::is_unsorted(current_neighbors.begin(), current_neighbors.end(), cmp)) {
        std::sort(current_neighbors.begin(), current_neighbors.end(), cmp);
    }
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
        ensure_sort(output[t]);
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
        ensure_sort(output[r]);
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
            auto ptr = data + static_cast<std::size_t>(k) * num_dim;
            searcher->search(ptr, num_neighbors, &indices, &distances);
            auto& curnn = output[k];
            fill_pair_vector(indices, distances, curnn);
        }
    });

    // We'll guarantee sortedness for tied distances during the inversion, so
    // no need to do it here. There's also no need to force each MNN-involved
    // cell to be reported as its own neighbor as that should have been done in
    // the neighbors_from function. So, each MNN-involved cell should already
    // participate in the calculation of its own center of mas.
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
            std::sort(curout.begin(), curout.end(), SortBySecond<Index_, Distance_>());
        }
    });

    return output;
}

template<typename Index_, typename Float_>
void compute_center_of_mass(
    std::size_t num_dim,
    const std::vector<Index_>& mnns,
    const NeighborSet<Index_, Float_>& neighbors_from,
    const NeighborSet<Index_, Float_>& inverted_neighbors_to,
    const Float_* data,
    int num_threads,
    double tolerance,
    std::vector<double>& running_mean)
{
    Index_ num_mnns = mnns.size();
    running_mean.resize(static_cast<std::size_t>(num_mnns) * num_dim); // cast to avoid overflow.

    parallelize(num_threads, num_mnns, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Float_> mean(num_dim), sum_squares(num_dim);

        for (Index_ g = start, end = start + length; g < end; ++g) {
            // Using Welford's algorithm to compute the running mean and
            // variance around the current center of mass. We filter out
            // points that are outliers in any dimension. 
            std::fill(mean.begin(), mean.end(), 0);
            std::fill(sum_squares.begin(), sum_squares.end(), 0);
            Index_ counter = 0;

            /**
             * Computing the initial seed.
             */
            const auto& nn_from = neighbors_from[mnns[g]];
            for (auto nn : nn_from) {
                auto target = data + static_cast<std::size_t>(nn.first) * num_dim; // cast to avoid overflow.
                ++counter;
                for (std::size_t d = 0; d < num_dim; ++d) {
                    auto val = target[d];
                    auto& curmean = mean[d];
                    double delta = val - curmean;
                    curmean += delta / counter;
                    sum_squares[d] += delta * (val - curmean);
                }
            }

            /**
             * Refining the seed based on the inverted neighbors.
             */
            decltype(nn_from.size()) checked = 0, limit = nn_from.size();
            SortBySecond<Index_, Distance_> cmp;
            for (auto nn : inverted_neighbors_to[g]) {
                // Iterating over both to/from sorted lists, to avoid re-adding
                // observations that were added in the seed.
                bool found = false;
                while (checked < limit) {
                    if (cmp(nn, nn_from[checked])) {
                        break;
                    if (nn_from[checked].first == nn.first) {
                        found = true;
                        break;
                    }
                    ++checked;
                }
                if (found) {
                    continue;
                }

                // Checking if the new observation is beyond the tolerance on any dimension.
                bool outside_range = false;
                Float_ multiplier = tolerance / std::sqrt(counter - 1.0);
                for (std::size_t d = 0; d < num_dim; ++d) {
                    if (std::abs(target[d] - mean[d]) > multiplier * std::sqrt(sum_squares[d])) {
                        outside_range = true;
                        break;
                    }
                }
                if (outside_range) {
                    continue;
                }

                // If it's good, we proceed to add it.
                auto target = data + static_cast<std::size_t>(nn.first) * num_dim; // cast to avoid overflow.
                ++counter;
                for (std::size_t d = 0; d < num_dim; ++d) {
                    auto val = target[d];
                    auto& curmean = mean[d];
                    double delta = val - curmean;
                    curmean += delta / counter;
                    sum_squares[d] += delta * (val - curmean);
                }
            }

            /**
             * Copying the result into the output vector.
             */
            std::size_t output_offset = num_dim * static_cast<std::size_t>(g); // cast to avoid overflow.
            std::copy(mean.begin(), mean.end(), running_mean.begin() + output_offset);
        }
    });

    return;
}

template<typename Index_, typename Float_>
void correct_target(
    std::size_t num_dim,
    Index_ num_total,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const BatchInfo<Index_, Float_>& target,
    const PopulateCrossNeighborsWorkspace<Index_, Float_>& pop,
    const FindClosestMnnWorkspace<Index_>& mnns,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder, 
    int num_neighbors,
    int num_threads,
    double tolerance,
    Float_* data,
    CorrectTargetWorkspace<Index_, Float_>& workspace) 
{
    // Build this first so that we can re-use the ref_buffers and target_buffers for the center of mass calculations.
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > ref_mnn_index;
    std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > target_mnn_index;
    parallelize(num_threads, 2, [&](int, int start, int length) -> void {
        for (int opt = start, end = start + length; opt < end; ++opt) {
            const auto& uniq = (opt == 0 ? mnns.ref_mnns_unique : mnns.target_mnns);
            auto& buffer = (opt == 0 ? workspace.ref_buffer : workspace.target_buffer);
            auto& index = (opt == 0 ? ref_mnn_index: target_mnn_index);
            index = build_mnn_only_index(num_dim, data, uniq, builder, buffer);
        }
    });

    workspace.mapping.resize(num_total);
    for (decltype(mnns.ref_mnns_unique.size()) rx = 0, rend = mnns.ref_mnns_unique.size(); r < rend; ++r) {
        workspace.mapping[mnns.ref_mnns_unique[rx]] = rx;
    }

    // Find the closest neighbors of each MNN-involved observation in either metabatch.
    search_for_neighbors_from_mnns(
        num_total,
        mnns.ref_mnns_unique,
        mnns.target_mnns,
        data,
        references,
        target,
        num_neighbors,
        num_threads,
        workspace.neighbors_from
    );

    // Find the closest MNN-involved observation(s) of each observation in either metabatch.
    search_for_neighbors_to_mnns(
        num_total,
        pop.ref_ids,
        pop.target_ids,
        data,
        mnns.ref_mnns_unique,
        ref_mnn_index,
        mnns.target_mnns,
        target_mnn_index,
        num_neighbors,
        num_threads,
        workspace.neighbors_to
    );

    // Computing the center of mass.
    auto ref_inverted = invert_neighbors<Index_, Float_>(mnns.ref_mnns_unique.size(), pop.ref_ids, workspace.neighbors_to, num_threads);
    auto target_inverted = invert_neighbors<Index_, Float_>(mnns.target_mnns.size(), pop.target_ids, workspace.neighbors_to, num_threads);
    compute_center_of_mass(num_dim, mnns.ref_mnns_unique, workspace.neighbors_from, ref_inverted, data, num_threads, tolerance, workspace.ref_buffer);
    compute_center_of_mass(num_dim, mnns.target_mnns, workspace.neighbors_from, target_inverted, data, num_threads, tolerance, workspace.target_buffer);

    // Apply the correction in the target based on its closest MNN-involved cell.
    Index_ num_target = pop.target_ids.size();
    workspace.chosen_batch.resize(num_total);
    parallelize(num_threads, num_target, [&](int, int start, int length) -> void {
        for (Index_ i = start, end = start + length; i < end; ++i) {
            auto t = pop.target_ids[i];
            auto tptr = data + static_cast<std::size_t>(t) * num_dim; // cast to avoid overflow.

            auto closest_mnn = workspace.neighbors_to[t][0].first; // this index is already defined with respect to the target_mnns subset.
            auto tcenter = workspace.target_buffer.data() + static_cast<std::size_t>(closest_mnn) * num_dim; // casting again.

            auto ref_partner = mnn.ref_mnns_partner[closest_mnn];
            auto ridx = workspace.mapping[ref_partner];
            auto rcenter = workspace.ref_buffer.data() + static_cast<std::size_t>(ridx) * num_dim; // ditto.

            for (std::size_t d = 0; d < num_dim; ++d) {
                tptr[d] += (recenter[d] - tcenter[d]);
            }

            workspace.chosen_batch[t] = pop.batch[ref_partner]; // report the full index of the closest MNN-involved cell.
        }
    });

    return;
}

}

}

#endif
