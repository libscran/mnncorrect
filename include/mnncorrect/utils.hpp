#ifndef MNNCORRECT_UTILS_HPP
#define MNNCORRECT_UTILS_HPP

#include <deque>
#include <vector>

namespace mnncorrect {

template<typename Index>
struct MnnPairs {
    size_t size() const { return left.size(); }
    std::deque<Index> left, right;
};

template<typename Index, typename Dist>
using NeighborSet = std::vector<std::vector<std::pair<Index, Dist> > >;

}

#endif
