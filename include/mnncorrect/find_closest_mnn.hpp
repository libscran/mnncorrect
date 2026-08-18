#ifndef MNNCORRECT_FIND_CLOSEST_MNN_HPP
#define MNNCORRECT_FIND_CLOSEST_MNN_HPP

#include <vector>
#include <algorithm>

#include "utils.hpp"

namespace mnncorrect {

template<typename Index_>
struct FindClosestMnnResults {
    // Index of the observation of the target metabatch in each MNN pair.
    // This is guaranteed to be sorted and unique.
    std::vector<Index_> target_mnns; 

    // 1:1 with target_mnns, specifying the other observation of the MNN pair in the reference metabatch.
    // This is not guaranteed to be sorted or unique.
    std::vector<Index_> ref_mnns;
};

template<typename Index_>
struct FindClosestMnnWorkspace {
    FindClosestMnnWorkspace() = default;
    FindClosestMnnWorkspace(const Index_ num_total) :
        resorted_neighbors(sanisizer::cast<I<decltype(resorted_neighbors.size())> >(num_total)),
        last_checked(sanisizer::cast<I<decltype(last_checked.size())> >(num_total))
    {
        used.reserve(num_total);
    }

public:
    std::vector<std::vector<Index_> > resorted_neighbors;

    // Length of each vector in 'neighbors' must be less than the number of
    // points, thus each 'last' position must fit in an Index_ type.
    std::vector<Index_> last_checked;

    std::vector<Index_> used;
};

template<typename Index_, typename Float_>
void find_closest_mnn(
    const std::vector<Index_>& target_ids,
    const NeighborSet<Index_, Float_>& neighbors,
    FindClosestMnnWorkspace<Index_>& workspace,
    FindClosestMnnResults<Index_>& results
) {
    workspace.used.clear();
    for ([[maybe_unused]] const auto& rev : workspace.resorted_neighbors) {
        assert(rev.empty());
    }
    for ([[maybe_unused]] const auto lu : workspace.last_checked) {
        assert(lu == 0);
    }

    results.ref_mnns.clear();
    results.target_mnns.clear();

    // Target IDs should be sorted and unique at this point.
    const auto num_targets = target_ids.size();
    for (I<decltype(num_targets)> i = 1; i < num_targets; ++i) {
        assert(target_ids[i - 1] < target_ids[i]);
    }

    for (const auto t : target_ids) {
        const auto& tvals = neighbors[t];
        bool best_found = false;
        Index_ best_ref = 0;

        // tvals should be sorted by distance, so we can quit early when we find the first (and thus closest) MNN.
        for (const auto& tpair : tvals) {
            const auto tneighbor = tpair.first;
            auto& other = workspace.resorted_neighbors[tneighbor];

            if (other.empty()) { // Only instantiate this when needed.
                const auto& rvals = neighbors[tneighbor];
                workspace.used.push_back(tneighbor);
                other.reserve(rvals.size());
                for (const auto& rpair : rvals) {
                    other.push_back(rpair.first);
                }
                std::sort(other.begin(), other.end());
            }

            // Picking up our search from the last checked position.
            // We don't need to search earlier indices, because there were already processed by an earlier iteration of 't'.
            auto& position = workspace.last_checked[tneighbor];
            const Index_ num_other = other.size();
            for (; position < num_other; ++position) {
                if (other[position] >= t) {
                    if (other[position] == t) {
                        best_ref = tpair.first;
                        best_found = true;
                    }
                    break;
                }
            }

            if (best_found) {
                results.target_mnns.push_back(t);
                results.ref_mnns.push_back(best_ref);
                break;
            }
        }
    }

    // Wiping everything in preparation for the next call to this function.
    for (const auto u : workspace.used) {
        workspace.resorted_neighbors[u].clear();
        workspace.last_checked[u] = 0;
    }
}

}

#endif
