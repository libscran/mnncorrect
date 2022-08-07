#ifndef MNNCORRECT_FIND_MUTUAL_NN_HPP
#define MNNCORRECT_FIND_MUTUAL_NN_HPP

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include "knncolle/knncolle.hpp"
#include "utils.hpp"

namespace mnncorrect {

template<typename Index>
struct MnnPairs {
    MnnPairs(size_t nright=0) {
        matches.reserve(nright);
        return;
    }

    std::unordered_map<Index, std::vector<Index> > matches;
    size_t num_pairs = 0;
};

template<typename Index>
std::vector<Index> unique_left(const MnnPairs<Index>& input) {
    std::set<Index> tmp; // yes, I would like it to be ordered, please.
    for (const auto& x : input.matches) {
        for (auto y : x.second) {
            tmp.insert(y);
        }
    }
    return std::vector<Index>(tmp.begin(), tmp.end());
}

template<typename Index>
std::vector<Index> unique_right(const MnnPairs<Index>& input) {
    std::vector<Index> output;
    output.reserve(input.matches.size());
    for (const auto& x : input.matches) {
        output.push_back(x.first);
    }
    std::sort(output.begin(), output.end());
    return output;
}

template<typename Index, typename Float>
MnnPairs<Index> find_mutual_nns(const NeighborSet<Index, Float>& left, const NeighborSet<Index, Float>& right) {
    Index nleft = left.size();
    Index nright = right.size();

    MnnPairs<Index> output(nright);
    std::vector<std::vector<Index> > neighbors_of_left(nleft);
    std::vector<size_t> last(nleft);

    for (Index r = 0; r < nright; ++r) {
        const auto& mine = right[r];
        std::vector<Index> holder;

        for (auto left_pair : mine) {
            auto left_neighbor = left_pair.first;
            auto& other = neighbors_of_left[left_neighbor];

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

            auto& position = last[left_neighbor];

            for (; position < other.size(); ++position) {
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

#endif
