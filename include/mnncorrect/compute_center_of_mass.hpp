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
#include "find_neighbors.hpp"

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

// Find all (indirect) neighbors of each MNN-involved observation within its own meta-batch.
// Here, 'neighbors[i]' will be populated if 'i' is within 'num_steps' of any entry of 'mnn_ids'.
template<typename Index_, typename Float_>
void walk_around_neighborhood(
    const std::size_t num_dim,
    const std::vector<Index_>& mnn_ids,
    const MetaBatch<Index_, Float_>& meta_batch,
    const Float_* data,
    const int num_neighbors,
    const int num_steps,
    const int num_threads,
    NeighborhoodWalkWorkspace<Index_>& walkspace, 
    NeighborSet<Index_, Float_>& neighbors
) {
    find_neighbors(
        num_dim,
        static_cast<Index_>(mnn_ids.size()),
        [&](Index_ i) -> Index_ { return mnn_ids[i]; },
        meta_batch,
        data,
        num_neighbors,
        false,
        num_threads,
        neighbors 
    );

    walkspace.all_ids.clear();
    walkspace.all_ids.insert(walkspace.all_ids.end(), mnn_ids.begin(), mnn_ids.end());
    assert(std::accumulate(walkspace.visited.begin(), walkspace.visited.end(), static_cast<Index_>(0)) == 0);
    for (auto& x : mnn_ids) {
        walkspace.visited[x] = true;
    }

    for (int s = 0; s < num_steps; ++s) {
        walkspace.next_ids.clear();
        const auto& current_visit = (s == 0 ? mnn_ids : walkspace.ids);

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
    const std::vector<Index_>& mnn_ids,
    const MetaBatch<Index_, Float_>& meta_batch,
    const Float_* const data,
    const int num_neighbors,
    const int num_steps,
    const int num_threads,
    NeighborhoodWalkWorkspace<Index_>& walkspace,
    NeighborSet<Index_, Float_>& neighbors,
    Float_* const centers
) {
    walk_around_neighborhood(
        num_dim,
        mnn_ids,
        meta_batch,
        data,
        num_neighbors,
        num_steps,
        num_threads,
        walkspace,
        neighbors
    );

    const Index_ num_mnns = mnn_ids.size();
    parallelize(num_threads, num_mnns, [&](const int t, const Index_ start, const Index_ length) -> void {
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
            assert(std::accumulate(walkspace.visited.begin(), walkspace.visited.end(), static_cast<Index_>(0)) == 0);
        }
        auto& visited = (t > 0 ? *tmp_visited : walkspace.visited);
        auto& current_processed = (t > 0 ? *tmp_current_processed : walkspace.ids);
        auto& next_processed = (t > 0 ? *tmp_next_processed : walkspace.next_ids);
        auto& all_processed = (t > 0 ? *tmp_all_processed : walkspace.all_ids);

        for (Index_ i = start, end = start + length; i < end; ++i) {
            std::fill(mean.begin(), mean.end(), 0);
            current_processed.clear();
            const auto curmnn = mnn_ids[i];

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
                centers[sanisizer::nd_offset<std::size_t>(d, num_dim, curmnn)] = mean[d] / denom;
            }

            for (const auto x : all_processed) {
                visited[x] = false;
            }
            all_processed.clear();
        }
    });
}

}

#endif
