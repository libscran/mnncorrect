#ifndef MNNCORRECT_FIND_CLOSEST_MNN_HPP
#define MNNCORRECT_FIND_CLOSEST_MNN_HPP

#include <vector>
#include <algorithm>

#include "utils.hpp"
#include "find_batch_neighbors.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_>
struct FindClosestMnnResults {
    std::vector<Index_> target_mnns; // observation of the target batch in the MNN pair.
    std::vector<Index_> ref_mnns; // 1:1 with target_mnns, specifying the other observation of the MNN pair in the reference metabatch.
};

template<typename Index_>
struct FindClosestMnnWorkspace {
    std::vector<std::vector<Index_> > reverse_neighbor_buffer;

    // Length of each vector in 'neighbors' must be less than the number of
    // points, thus each 'last' position must fit in an Index_ type.
    std::vector<Index_> last_checked;
};

template<typename Index_, typename Float_>
void find_closest_mnn(
    const std::vector<Index_>& target_ids,
    const NeighborSet<Index_, Float_>& neighbors,
    FindClosestMnnWorkspace<Index_>& workspace,
    FindClosestMnnResults<Index_>& results)
{
    const auto num_total = neighbors.size();
    for (auto& rev : workspace.reverse_neighbor_buffer) {
        rev.clear();
    }
    sanisizer::resize(workspace.reverse_neighbor_buffer, num_total);
    workspace.last_checked.clear();
    sanisizer::resize(workspace.last_checked, num_total);

    results.ref_mnns.clear();
    results.target_mnns.clear();

    for (const auto t : target_ids) {
        const auto& tvals = neighbors[t];
        bool best_found = false;
        Index_ best_ref = 0;

        // tvals should be sorted by distance, so we can quit early when
        // we find the first (and thus closest) MNN.
        for (const auto& tpair : tvals) {
            const auto tneighbor = tpair.first;
            auto& other = workspace.reverse_neighbor_buffer[tneighbor];

            if (other.empty()) { // Only instantiate this when needed.
                const auto& rvals = neighbors[tneighbor];
                other.reserve(rvals.size());
                for (const auto& rpair : rvals) {
                    other.push_back(rpair.first);
                }
                std::sort(other.begin(), other.end());
            }

            // Picking up our search from the last checked position; we don't
            // need to search earlier indices, because there were already
            // processed by an earlier iteration of 't'.
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
}

}

}

#endif
