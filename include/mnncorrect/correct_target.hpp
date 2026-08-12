#ifndef MNNCORRECT_CORRECT_TARGET_HPP
#define MNNCORRECT_CORRECT_TARGET_HPP

#include <algorithm>
#include <vector>
#include <memory>
#include <cstddef>
#include <numeric>
#include <cassert>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"
#include "find_closest_mnn.hpp"

namespace mnncorrect {

template<typename Index_>
struct NeighborhoodWalkWorkspace {
    NeighborhoodWalkWorkspace() = default;
    NeighborhoodWalkWorkspace(Index_ num_total) : visited(sanisizer::cast<decltype(visited.size())>(num_total)) {}

    // 'visited' and 'all_ids' combine to form an unordered_set for integers in [0, num_total).
    // If 'visited[i] == true', 'i' was already added; otherwise, we set 'visited[i] = true' and 'all_ids.push_back(i)'.
    // Once we're done in a given scope, we must run through 'all_ids' and set all positions of 'visited' back to false prior to next use.
    std::vector<char> visited;
    std::vector<Index_> ids, next_ids, all_ids;
};

template<typename Index_, typename Float_>
struct CorrectTargetWorkspace {
    CorrectTargetWorkspace() = default;
    CorrectTargetWorkspace(Index_ num_total) :
        neighbors(sanisizer::cast<decltype(neighbors.size())>(num_total)),
        walk(num_total),
        ref_remapping(sanisizer::cast<decltype(ref_remapping.size())>(num_total))
    {}

    // Intermediates for stepwise neighbor search.
    NeighborSet<Index_, Float_> neighbors;
    NeighborhoodWalkWorkspace<Index_> walk;

    // Buffers for storing the centers of mass and correction vectors.
    std::vector<Float_> correction_buffer, ref_center_buffer;

    // For the correction itself.
    std::vector<Index_> ref_remapping;
    std::vector<BatchIndex> new_target_meta_batch;
};

// Find all neighbors of each MNN-involved observation within its own meta-batch.
// Here, 'output' will contain neighbor indices relative to the entire dataset.
template<typename Index_, typename Float_>
void walk_around_neighborhood(
    const std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* data,
    const MetaBatch<Index_, Float_>& meta_batch,
    const int num_neighbors,
    const int num_steps,
    const int num_threads,
    NeighborhoodWalkWorkspace<Index_>& walkspace, 
    NeighborSet<Index_, Float_>& neighbors
) {
    find_neighbors(
        num_dim,
        static_cast<Index_>(ids.size()),
        [&](Index_ i) -> Index_ { return ids[i]; },
        meta_batch,
        data,
        num_neighbors,
        false,
        num_threads,
        neighbors 
    );

    walkspace.all_ids.clear();
    walkspace.all_ids.insert(walkspace.all_ids.end(), ids.begin(), ids.end());

    for (int s = 0; s < num_steps; ++s) {
        walkspace.next_ids.clear();
        const auto& current_visit = (s == 0 ? ids : walkspace.ids);

        for (const auto i : current_visit) {
            const auto& curneighbors = neighbors[i];
            for (const auto& pair : curneighbors) {
                if (walkspace.visited[pair.first]) {
                    continue;
                }
                walkspace.next_ids.push_back(pair.first);
                walkspace.visited[pair.first] = true;
            }
        }
        if (walkspace.next_ids.empty()) {
            break;
        }

        find_neighbors(
            num_dim,
            static_cast<Index_>(walkspace.next_ids.size()),
            [&](Index_ i) -> Index_ { return walkspace.next_ids[i]; },
            meta_batch,
            data,
            num_neighbors,
            false,
            num_threads,
            neighbors 
        );

        walkspace.all_ids.insert(walkspace.all_ids.end(), walkspace.next_ids.begin(), walkspace.next_ids.end());
        walkspace.ids.swap(walkspace.next_ids);
    }

    // Set it back to an all-zero vector for downstream use.
    for (const auto x : walkspace.all_ids) {
        walkspace.visited[x] = false;
    }
}

template<typename Index_, typename Float_>
void compute_center_of_mass(
    const std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* const data,
    const int num_steps,
    const int num_threads,
    const NeighborSet<Index_, Float_>& neighbors,
    NeighborhoodWalkWorkspace<Index_>& walkspace,
    Float_* const buffer
) {
    const Index_ num = ids.size();
    parallelize(num_threads, num, [&](const int t, const Index_ start, const Index_ length) -> void {
        // Using a separate mean array to minimize false sharing.
        auto mean = sanisizer::create<std::vector<Float_> >(num_dim);

        // Reusing the workspace's memory for the first thread, otherwise allocating anew.
        std::optional<std::vector<Index_> > tmp_current_processed, tmp_next_processed, tmp_all_processed;
        std::optional<std::vector<char> > tmp_visited;
        if (t > 0) {
            tmp_visited.emplace(walkspace.visited.size()); // same size_type, no cast.
            tmp_current_processed.emplace();
            tmp_next_processed.emplace();
            tmp_all_processed.emplace();
        } else {
            walkspace.ids.clear();
            walkspace.next_ids.clear();
            walkspace.all_ids.clear();
        }
        auto& visited = (t > 0 ? *tmp_visited : walkspace.visited);
        auto& current_processed = (t > 0 ? *tmp_current_processed : walkspace.ids);
        auto& next_processed = (t > 0 ? *tmp_next_processed : walkspace.next_ids);
        auto& all_processed = (t > 0 ? *tmp_all_processed : walkspace.all_ids);

        for (Index_ g = start, end = start + length; g < end; ++g) {
            std::fill(mean.begin(), mean.end(), 0);
            current_processed.clear();
            const auto curmnn = ids[g];

            for (const auto& nn : neighbors[curmnn]) {
                const auto ptr = data + sanisizer::product_unsafe<std::size_t>(nn.first, num_dim);
                for (std::size_t d = 0; d < num_dim; ++d) {
                    mean[d] += ptr[d];
                }
                visited[nn.first] = true;
                current_processed.push_back(nn.first);
            }
            all_processed.insert(all_processed.end(), current_processed.begin(), current_processed.end());

            for (int s = 0; s < num_steps; ++s) {
                next_processed.clear();
                for (const auto y : current_processed) {
                    for (const auto& nn : neighbors[y]) {
                        if (visited[nn.first]) {
                            continue;
                        }
                        const auto ptr = data + sanisizer::product_unsafe<std::size_t>(nn.first, num_dim);
                        for (std::size_t d = 0; d < num_dim; ++d) {
                            mean[d] += ptr[d];
                        }
                        visited[nn.first] = true;
                        next_processed.push_back(nn.first);
                    }
                }

                if (next_processed.empty()) {
                    break;
                }
                all_processed.insert(all_processed.end(), next_processed.begin(), next_processed.end());
                current_processed.swap(next_processed);
            }

            const double denom = all_processed.size();
            for (std::size_t d = 0; d < num_dim; ++d) {
                buffer[sanisizer::nd_offset<std::size_t>(d, num_dim, g)] = mean[d] / denom;
            }

            for (const auto x : all_processed) {
                visited[x] = false;
            }
            all_processed.clear();
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
    buffer.resize(sanisizer::product<I<decltype(buffer.size())> >(num_dim, num_in_mnn));
    for (I<decltype(num_in_mnn)> f = 0; f < num_in_mnn; ++f) {
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
    const std::vector<MetaBatch<Index_, Float_> >& reference_meta_batches,
    const MetaBatch<Index_, Float_>& target_meta_batch,
    const std::vector<BatchIndex>& meta_batch_assignments,
    const std::vector<Index_>& target_ids,
    const FindClosestMnnResults<Index_>& mnns,
    const knncolle::Builder<Index_, Float_, Float_, Matrix_>& builder, 
    const int num_neighbors,
    const int num_steps,
    const int num_threads,
    Float_* const data,
    CorrectTargetWorkspace<Index_, Float_>& workspace
) {
    CorrectTargetResults<Index_> results;

    // Split reference MNNs back into their meta-batches of origin.
    // Here we use the 'results.reassignments' as a temporary place to put this information; we will overwrite it before we return from this function.
    // We also abuse 'workspace.walk' as a proxy for a hashmap.
    const auto num_refs = reference_meta_batches.size();
    {
        sanisizer::resize(results.reassignments, num_refs);
        for (auto& reass : results.reassignments) {
            reass.clear();
        }

        workspace.walk.all_ids.clear();
        for (const auto r : mnns.ref_mnns) {
            if (workspace.walk.visited[r]) {
                continue;
            }
            results.reassignments[meta_batch_assignments[r]].push_back(r);
            workspace.walk.visited[r] = true;
            workspace.walk.all_ids.push_back(r);
        }
        for (const auto r : workspace.walk.all_ids) {
            workspace.walk.visited[r] = false;
        }

        const auto num_unique_ref_mnns = workspace.walk.all_ids.size();
        workspace.ref_center_buffer.resize(sanisizer::product<I<decltype(workspace.ref_center_buffer.size())> >(num_dim, num_unique_ref_mnns));
    }

    // Find neighbors in each of the reference meta-batches.
    Index_ counter = 0;
    for (I<decltype(num_refs)> r = 0; r < num_refs; ++r) {
        const auto& curass = results.reassignments[r];

        walk_around_neighborhood(
            num_dim,
            curass,
            data,
            reference_meta_batches[r],
            num_neighbors,
            num_steps,
            num_threads,
            workspace.walk,
            workspace.neighbors
        );

        compute_center_of_mass(
            num_dim,
            curass,
            data,
            num_steps,
            num_threads,
            workspace.neighbors,
            workspace.walk,
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
        mnns.target_mnns,
        data,
        target_meta_batch,
        num_neighbors,
        num_steps,
        num_threads,
        workspace.walk,
        workspace.neighbors
    );

    workspace.correction_buffer.resize(sanisizer::product<I<decltype(workspace.correction_buffer.size())> >(num_dim, mnns.target_mnns.size()));
    compute_center_of_mass(
        num_dim,
        mnns.target_mnns,
        data,
        num_steps,
        num_threads,
        workspace.neighbors,
        workspace.walk,
        workspace.correction_buffer.data() // using the correction buffer to hold the center of mass for now.
    );

    const auto num_pairs = mnns.target_mnns.size();
    for (I<decltype(num_pairs)> p = 0; p < num_pairs; ++p) {
        const auto ref_index = workspace.ref_remapping[mnns.ref_mnns[p]];
        for (std::size_t d = 0; d < num_dim; ++d) {
            auto& correction = workspace.correction_buffer[sanisizer::nd_offset<std::size_t>(d, num_dim, p)];
            correction = workspace.ref_center_buffer[sanisizer::nd_offset<std::size_t>(d, num_dim, ref_index)] - correction;
        }
    }

    // Apply the correction in the target meta-batch based on its closest MNN-involved cell.
    const Index_ num_target = target_ids.size();
    sanisizer::resize(workspace.new_target_meta_batch, num_target);

    parallelize(num_threads, num_target, [&](const int, const Index_ start, const Index_ length) -> void {
        auto searcher = target_mnn_index->initialize();
        std::vector<Index_> indices;
        assert(target_mnn_index->num_observations() > 0);

        for (Index_ i = start, end = start + length; i < end; ++i) {
            const auto tptr = data + sanisizer::product_unsafe<std::size_t>(target_ids[i], num_dim);

            // No need to cap the number of neighbors to a value below 1.
            // Each batch is expected to be non-empty at this point.
            searcher->search(tptr, 1, &indices, NULL);

            const auto chosen = indices.front();
            const auto correct_ptr = workspace.correction_buffer.data() + sanisizer::product_unsafe<std::size_t>(num_dim, chosen);
            for (std::size_t d = 0; d < num_dim; ++d) {
                tptr[d] += correct_ptr[d];
            }

            workspace.new_target_meta_batch[i] = meta_batch_assignments[mnns.ref_mnns[chosen]];
        }
    });

    for (auto& reass : results.reassignments) {
        reass.clear();
    }
    for (I<decltype(num_target)> i = 0; i < num_target; ++i) {
        results.reassignments[workspace.new_target_meta_batch[i]].push_back(target_ids[i]);
    }

    return results;
}

}

#endif
