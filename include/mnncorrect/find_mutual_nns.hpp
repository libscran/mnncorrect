#ifndef MNNCORRECT_FIND_MUTUAL_NN_HPP
#define MNNCORRECT_FIND_MUTUAL_NN_HPP

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_>
struct MnnPairs {
    std::unordered_map<Index_, std::vector<Index_> > matches;
    size_t num_pairs = 0;
};

template<typename Index_>
std::vector<Index_> unique_left(const MnnPairs<Index_>& input) {
    std::unordered_set<Index_> tmp;
    for (const auto& x : input.matches) {
        for (auto y : x.second) {
            tmp.insert(y);
        }
    }

    std::vector<Index_> output(tmp.begin(), tmp.end());
    std::sort(output.begin(), output.end());
    return output;
}

template<typename Index_>
std::vector<Index_> unique_right(const MnnPairs<Index_>& input) {
    std::vector<Index_> output;
    output.reserve(input.matches.size());
    for (const auto& x : input.matches) {
        output.push_back(x.first);
    }
    std::sort(output.begin(), output.end());
    return output;
}

template<typename Index_>
struct MnnWorkspace {
    std::vector<std::vector<Index_> > neighbors_of_left;
    std::vector<size_t> last;
};

template<typename Index_, typename Float_>
void find_mutual_nns(const NeighborSet<Index_, Float_>& left, const NeighborSet<Index_, Float_>& right, MnnPairs<Index_>& output, MnnWorkspace<Index_>& workspace) {
    Index nleft = left.size();
    Index nright = right.size();
    output.matches.reserve(nright);

    // Clearing manually to preserve any existing allocations. 
    workspace.neighbors_of_left.resize(nleft);
    for (auto& x : workspace.neighbors_of_left) {
        x.clear();
    }
    workspace.last.clear();
    workspace.last.resize(nleft);

    for (Index r = 0; r < nright; ++r) {
        const auto& mine = right[r];
        std::vector<Index_> holder;

        for (auto left_pair : mine) {
            auto left_neighbor = left_pair.first;
            auto& other = workspace.neighbors_of_left[left_neighbor];

            if (other.empty()) {
                // Only instantiate this when needed.
                const auto& curleft = left[left_neighbor];
                if (curleft.size()) {
                    other.reserve(curleft.size());
                    for (const auto& f : curleft) {
                        other.push_back(f.first);
                    }
                    std::sort(other.begin(), other.end());
                }
            }

            auto& position = workspace.last[left_neighbor];
            size_t num_other = other.size();
            for (; position < num_other; ++position) {
                if (other[position] >= r) {
                    if (other[position] == r) {
                        holder.push_back(left_neighbor);
                        ++output.num_pairs;
                    }
                    break;
                }
            }
        }

        if (holder.size()) {
            output.matches[r] = std::move(holder);
        }
    }

    return output;
}

}

}

#endif
