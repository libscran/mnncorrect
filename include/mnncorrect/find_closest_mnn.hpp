#ifndef MNNCORRECT_FIND_CLOSEST_MNN_HPP
#define MNNCORRECT_FIND_CLOSEST_MNN_HPP

#include <vector>
#include <algorithm>

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
struct FindClosestMnnWorkspace {
    std::vector<Index_> target_mnns; // observation of the target metabatch in the MNN pair.
    std::vector<Index_> ref_mnns_partner; // 1:1 with target_mnns, specifying the other observation of the MNN pair.
    std::vector<Index_> ref_mnns_unique; // unique and sorted version of 'ref_mnns_partner'

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
    auto num_total = neighbors.size();
    for (auto& rev : workspace.reverse_neighbor_buffer) {
        rev.clear();
    }
    workspace.reverse_neighbor_buffer.resize(num_total);
    std::fill(workspace.last_checked.begin(), workspace.last_checked.end(), 0);
    workspace.last_checked.resize(num_total);

    workspace.ref_mnns_partner.clear();
    workspace.target_mnns.clear();

    for (auto t : target_ids) {
        const auto& tvals = neighbors[t];
        bool best_found = false;
        Index_ best_ref = 0;

        // tvals should be sorted by distance, so we can quit early when
        // we find the first (and thus closest) MNN.
        for (auto tpair : tvals) {
            auto tneighbor = tpair.first;
            auto& other = workspace.reverse_neighbor_buffer[tneighbor];

            if (other.empty()) { // Only instantiate this when needed.
                const auto& rvals = neighbors[tneighbor];
                if (!rvals.empty()) {
                    other.reserve(rvals.size());
                    for (const auto& rpair : rvals) {
                        other.push_back(rpairs.first);
                    }
                    std::sort(other.begin(), other.end());
                }
            }

            // Picking up our search from the last checked position; we don't
            // need to search earlier indices, because there were already
            // processed by an earlier iteration of 't'.
            auto& position = workspace.last_checked[tneighbor];
            Index_ num_other = other.size();
            for (; position < num_other; ++position) {
                if (other[position] >= t) {
                    if (other[position] == t) {
                        best_ref = tpair.first;
                        best_found = true;
                        break;
                    }
                }
            }

            if (best_found) {
                workspace.target_mnns.push_back(t);
                workspace.ref_mnns_partner.push_back(best_ref);
                break;
            }
        }
    }

    // Uniquifying.
    std::fill(workspace.ref_mnn_buffer.begin(), workspace.ref_mnn_buffer.end(), false);
    workspace.ref_mnn_buffer.resize(num_total);
    for (auto r : ref_mnns_partner) {
        ref_mnn_buffer[r] = true;
    }
    workspace.ref_mnns_unique.clear();
    for (Index_ r = 0, end = ref_mnn_buffer.size(); r < end; ++r) {
        if (ref_mnn_buffer[r]) {
            ref_mnns_unique.push_back(r);
        }
    }
}

}

}

#endif
