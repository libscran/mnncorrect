#ifndef MNNCORRECT_FIND_CLOSEST_MNN_HPP
#define MNNCORRECT_FIND_CLOSEST_MNN_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct FindClosestMnnWorkspace {
    std::vector<Index_> ref_mnns;
    std::vector<Index_> target_mnns;
    std::vector<Index_> ref_mnns_unique;

    std::vector<std::vector<Index_> > reverse_neighbor_buffer;
    std::vector<unsigned char> ref_mnn_buffer;

    // Length of each vector in 'neighbors' must be less than the number of
    // points, thus each 'last' position must fit in Index_.
    std::vector<Index_> last_checked;
};

template<typename Index_, typename Float_>
void find_closest_mnn(
    const NeighborSet<Index_, Float_>& neighbors,
    const std::vector<Index_>& ref_ids,
    const std::vector<Index_>& target_ids,
    FindClosestMnnWorkspace<Index_, Float_>& workspace)
{
    Index_ num_ref = ref_ids.size();
    Index_ num_target = target_ids.size();

    workspace.reverse_neighbor_buffer.resize(neighbors.size());
    workspace.last_checked.clear();
    workspace.last_checked.resize(neighbors.size());

    workspace.ref_mnns.clear();
    workspace.target_mnns.clear();

    for (auto t : target_ids) {
        const auto& tvals = neighbors[t];
        bool best_found = false;
        Index_ best_ref = 0;
        Float_ best_distance = std::numeric_limits<Float_>::max();

        // tvals should be sorted by distance, so we can quit early when
        // we find the closest MNN.
        for (auto tpair : tvals) {
            auto tneighbor = tpair.first;
            auto& other = workspace.reverse_neighbor_buffer[tneighbor];

            if (other.empty()) {
                // Only instantiate this when needed.
                const auto& rvals = neighbors[tneighbor];
                if (!rvals.empty()) {
                    other.reserve(rvals.size());
                    for (const auto& rpair : rvals) {
                        other.push_back(rpairs.first);
                    }
                    std::sort(other.begin(), other.end());
                }
            }

            // Picking up our search from the last position; we don't need to
            // search earlier indices, because there were already processed
            // by an earlier iteration of 'r'.
            auto& position = workspace.last_checked[tneighbor];
            Index_ num_other = other.size();
            for (; position < num_other; ++position) {
                if (other[position] >= t) {
                    if (other[position] == t) {
                        if (best_distance <= tpair.second) {
                            best_ref = tpair.first;
                            best_found = true;
                            break;
                        }
                    }
                }
            }

            if (best_found) {
                workspace.target_mnns.push_back(t);
                workspace.ref_mnns.push_back(rbest);
                break;
            }
        }
    }

    // Uniquifying.
    workspace.ref_mnn_buffer.clear();
    workspace.ref_mnn_buffer.resize(neighbors.size());
    for (auto r : ref_mnns) {
        ref_mnn_buffer[r] = true;
    }
    Index_ n = ref_mnn_buffer.size();
    workspace.ref_mnns_unique.clear();
    for (Index_ r = 0; r < n; ++r) {
        ref_mnns_unique.push_back(r);
    }
}

}

}

#endif
