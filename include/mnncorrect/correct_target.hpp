#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include <algorithm>
#include <vector>
#include <memory>
#include <cstddef>
#include <numeric>
#include <unordered_set>
#include <unordered_map>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"
#include "find_closest_mnn.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct CorrectTargetWorkspace {
    // Intermediates for stepwise neighbor search.
    NeighborSet<Index_, Float_> neighbors;
    std::unordered_set<Index_> visited;
    std::vector<Index_> ids, next_visit;

    // Buffers for storing the centers of mass and correction vectors.
    std::vector<Float_> correction_buffer, ref_center_buffer;

    // For the correction itself.
    std::unordered_map<Index_, Index_> ref_remapping;
    std::vector<BatchIndex> new_target_batch;
};

// Find all neighbors of each MNN-involved observation within its own batch.
// Here, 'output' will contain neighbor indices relative to the entire dataset.
template<typename Index_, typename Float_>
void walk_around_neighborhood(
    const std::size_t num_dim,
    const Index_ num_total,
    const std::vector<Index_>& ids,
    const Float_* data,
    const BatchInfo<Index_, Float_>& batch,
    const int num_neighbors,
    const int num_steps,
    const int num_threads,
    CorrectTargetWorkspace<Index_, Float_>& workspace)
{
    workspace.neighbors.resize(num_total);
    find_batch_neighbors(
        num_dim,
        static_cast<Index_>(ids.size()),
        [&](Index_ i) -> Index_ { return ids[i]; },
        data,
        batch,
        num_neighbors,
        false,
        num_threads,
        workspace.neighbors 
    );

    workspace.visited.clear();
    workspace.visited.insert(ids.begin(), ids.end());

    for (int s = 0; s < num_steps; ++s) {
        workspace.next_visit.clear();
        const auto& current_visit = (s == 0 ? ids : workspace.ids);

        for (const auto i : current_visit) {
            const auto& curneighbors = workspace.neighbors[i];
            for (const auto& pair : curneighbors) {
                if (workspace.visited.find(pair.first) == workspace.visited.end()) {
                    workspace.next_visit.push_back(pair.first);
                    workspace.visited.insert(pair.first);
                }
            }
        }
        if (workspace.next_visit.empty()) {
            break;
        }

        find_batch_neighbors(
            num_dim,
            static_cast<Index_>(workspace.next_visit.size()),
            [&](Index_ i) -> Index_ { return workspace.next_visit[i]; },
            data,
            batch,
            num_neighbors,
            false,
            num_threads,
            workspace.neighbors 
        );

        workspace.ids.swap(workspace.next_visit);
    }
}

template<typename Index_, typename Float_>
void compute_center_of_mass(
    const std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* const data,
    const int num_steps,
    const int num_threads,
    NeighborSet<Index_, Float_>& neighbors,
    Float_* const buffer)
{
    const Index_ num = ids.size();
    parallelize(num_threads, num, [&](const int, const Index_ start, const Index_ length) -> void {
        auto mean = sanisizer::create<std::vector<Float_> >(num_dim);
        std::unordered_set<Index_> visited;
        std::vector<Index_> current_processed, next_processed;

        for (Index_ g = start, end = start + length; g < end; ++g) {
            std::fill(mean.begin(), mean.end(), 0);
            visited.clear();
            current_processed.clear();
            const auto curmnn = ids[g];

            for (const auto& nn : neighbors[curmnn]) {
                const auto ptr = data + sanisizer::product_unsafe<std::size_t>(nn.first, num_dim);
                for (std::size_t d = 0; d < num_dim; ++d) {
                    mean[d] += ptr[d];
                }
                visited.insert(nn.first);
                current_processed.push_back(nn.first);
            }

            for (int s = 0; s < num_steps; ++s) {
                next_processed.clear();
                for (const auto y : current_processed) {
                    for (const auto& nn : neighbors[y]) {
                        if (visited.find(nn.first) == visited.end()) {
                            const auto ptr = data + sanisizer::product_unsafe<std::size_t>(nn.first, num_dim);
                            for (std::size_t d = 0; d < num_dim; ++d) {
                                mean[d] += ptr[d];
                            }
                            visited.insert(nn.first);
                            next_processed.push_back(nn.first);
                        }
                    }
                }

                current_processed.swap(next_processed);
                if (current_processed.empty()) {
                    break;
                }
            }

            const double denom = visited.size();
            for (std::size_t d = 0; d < num_dim; ++d) {
                buffer[sanisizer::nd_offset<std::size_t>(d, num_dim, g)] = mean[d] / denom;
            }
        }
    });
}

template<typename Index_, typename Float_, class Matrix_>
std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > build_mnn_only_index(
    const std::size_t num_dim,
    const Float_* const data,
    const std::vector<Index_>& in_mnn,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder,
    std::vector<Float_>& buffer)
{
    const auto num_in_mnn = in_mnn.size();
    buffer.resize(sanisizer::product<decltype(I(buffer.size()))>(num_dim, num_in_mnn));
    for (decltype(I(num_in_mnn)) f = 0; f < num_in_mnn; ++f) {
        const auto curdata = data + sanisizer::product_unsafe<std::size_t>(in_mnn[f], num_dim);
        std::copy_n(curdata, num_dim, buffer.begin() + sanisizer::product_unsafe<std::size_t>(f, num_dim));
    }
    return builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_in_mnn, buffer.data()));
}

template<typename Index_>
struct CorrectTargetResults {
    std::vector<std::vector<Index_> > reassignments;
};

template<typename Index_, typename Float_, class Matrix_>
CorrectTargetResults<Index_> correct_target(
    const std::size_t num_dim,
    const Index_ num_total,
    const std::vector<BatchInfo<Index_, Float_> >& references,
    const BatchInfo<Index_, Float_>& target,
    const std::vector<Index_>& target_ids,
    const std::vector<BatchIndex>& batch_of_origin,
    const FindClosestMnnResults<Index_>& mnns,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder, 
    const int num_neighbors,
    const int num_steps,
    const int num_threads,
    Float_* const data,
    CorrectTargetWorkspace<Index_, Float_>& workspace)
{
    CorrectTargetResults<Index_> results;

    // Allocate reference MNNs into their batches. Here we use the
    // 'results.reassignments' as a temporary place to put this information;
    // we will overwrite it before we return from this function.
    const auto num_refs = references.size();
    sanisizer::resize(results.reassignments, num_refs);
    for (auto& reass : results.reassignments) {
        reass.clear();
    }
    workspace.visited.clear();
    for (const auto r : mnns.ref_mnns) {
        if (workspace.visited.find(r) == workspace.visited.end()) {
            results.reassignments[batch_of_origin[r]].push_back(r);
            workspace.visited.insert(r);
        }
    }

    workspace.ref_center_buffer.resize(sanisizer::product<decltype(I(workspace.ref_center_buffer.size()))>(num_dim, workspace.visited.size()));
    workspace.ref_remapping.clear();
    Index_ counter = 0;

    // Find neighbors in each of the reference batches.
    for (decltype(I(num_refs)) r = 0; r < num_refs; ++r) {
        const auto& curass = results.reassignments[r];

        walk_around_neighborhood(
            num_dim,
            num_total,
            curass,
            data,
            references[r],
            num_neighbors,
            num_steps,
            num_threads,
            workspace
        );

        compute_center_of_mass(
            num_dim,
            curass,
            data,
            num_steps,
            num_threads,
            workspace.neighbors,
            workspace.ref_center_buffer.data() + static_cast<std::size_t>(counter) * num_dim
        );

        for (auto x : curass) {
            workspace.ref_remapping[x] = counter;
            ++counter;
        }
    }

    // Build this first so that we can re-use the correction_buffer for the center of mass calculations.
    const auto target_mnn_index = build_mnn_only_index(
        num_dim,
        data,
        mnns.target_mnns,
        builder,
        workspace.correction_buffer
    );

    // Now computing the correction vector for each MNN pair.
    walk_around_neighborhood(
        num_dim,
        num_total,
        mnns.target_mnns,
        data,
        target,
        num_neighbors,
        num_steps,
        num_threads,
        workspace
    );

    workspace.correction_buffer.resize(sanisizer::product<decltype(I(workspace.correction_buffer.size()))>(num_dim, mnns.target_mnns.size()));
    compute_center_of_mass(
        num_dim,
        mnns.target_mnns,
        data,
        num_steps,
        num_threads,
        workspace.neighbors,
        workspace.correction_buffer.data() // using the correction buffer to hold the center of mass for now.
    );

    const auto num_pairs = mnns.target_mnns.size();
    for (decltype(I(num_pairs)) p = 0; p < num_pairs; ++p) {
        const auto ref_index = workspace.ref_remapping.find(mnns.ref_mnns[p])->second;
        for (std::size_t d = 0; d < num_dim; ++d) {
            auto& correction = workspace.correction_buffer[sanisizer::nd_offset<std::size_t>(d, num_dim, p)];
            correction = workspace.ref_center_buffer[sanisizer::nd_offset<std::size_t>(d, num_dim, ref_index)] - correction;
        }
    }

    // Apply the correction in the target based on its closest MNN-involved cell.
    const Index_ num_target = target_ids.size();
    sanisizer::resize(workspace.new_target_batch, num_target);

    parallelize(num_threads, num_target, [&](const int, const Index_ start, const Index_ length) -> void {
        auto searcher = target_mnn_index->initialize();
        std::vector<Index_> indices;

        for (Index_ i = start, end = start + length; i < end; ++i) {
            const auto tptr = data + sanisizer::product_unsafe<std::size_t>(target_ids[i], num_dim);
            searcher->search(tptr, 1, &indices, NULL); // no need to cap, we had better have at least one observation in each batch.

            const auto chosen = indices.front();
            const auto correct_ptr = workspace.correction_buffer.data() + sanisizer::product_unsafe<std::size_t>(num_dim, chosen);
            for (std::size_t d = 0; d < num_dim; ++d) {
                tptr[d] += correct_ptr[d];
            }

            workspace.new_target_batch[i] = batch_of_origin[mnns.ref_mnns[chosen]];
        }
    });

    for (auto& reass : results.reassignments) {
        reass.clear();
    }
    for (decltype(I(num_target)) i = 0; i < num_target; ++i) {
        results.reassignments[workspace.new_target_batch[i]].push_back(target_ids[i]);
    }

    return results;
}

}

}

#endif
