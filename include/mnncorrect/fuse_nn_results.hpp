#ifndef MNNCORRECT_FUSE_NN_RESULTS_HPP
#define MNNCORRECT_FUSE_NN_RESULTS_HPP

#include <vector>
#include <utility>

#include "knncolle/knncolle.hpp"

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Distance_>
void fuse_nn_results(
    const std::vector<std::pair<Index_, Distance_> >& base, 
    const std::vector<std::pair<Index_, Distance_> >& alt, 
    const int k,
    std::vector<std::pair<Index_, Distance_> >& output)
{
    output.clear();
    decltype(I(output.size())) num_neighbors = k; // converting into size_type for easier comparisons below.
    if (num_neighbors == 0) {
        return;
    }

    output.reserve(num_neighbors);
    auto bIt = base.begin();
    auto bEnd = base.end();
    auto aIt = alt.begin();
    auto aEnd = alt.end();

    if (bIt != bEnd && aIt != aEnd) {
        do {
            auto bval = bIt->second;
            auto aval = aIt->second;
            if (bval > aval) {
                output.push_back(*aIt);
                ++aIt;
                if (aIt == aEnd) {
                    break;
                }
            } else if (bval < aval) {
                output.push_back(*bIt);
                ++bIt;
                if (bIt == bEnd) {
                    break;
                }
               
            } else if (bIt->first > aIt->first) { // handling the unlikely cases of equal distances...
                output.push_back(*aIt);
                ++aIt;
                if (aIt == aEnd) {
                    break;
                }
            } else {
                output.push_back(*bIt);
                ++bIt;
                if (bIt == bEnd) {
                    break;
                }
            }
        } while (output.size() < num_neighbors);
    }

    while (bIt != bEnd && output.size() < num_neighbors) {
        output.push_back(*bIt);
        ++bIt;
    }

    while (aIt != aEnd && output.size() < num_neighbors) {
        output.push_back(*aIt);
        ++aIt;
    }
}

}

}

#endif
