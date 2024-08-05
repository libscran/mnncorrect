#ifndef MNNCORRECT_FUSE_NN_RESULTS_HPP
#define MNNCORRECT_FUSE_NN_RESULTS_HPP

#include <vector>
#include "knncolle/knncolle.hpp"
#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Dim_, typename Index_, typename Distance_>
NeighborSet<Index_, Distance_> quick_find_nns(size_t nobs, const Distance_* query, const knncolle::Prebuilt<Dim_, Index_, Distance_>& index, int k, [[maybe_unused]] int nthreads) {
    NeighborSet<Index_, Distance_> output(nobs);
    size_t ndim = index.num_dimensions();

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
    std::vector<Index_> indices;
    std::vector<Distance_> distances;
    auto searcher = index.initialize();
#ifdef _OPENMP
    #pragma omp for
#endif
    for (size_t l = 0; l < nobs; ++l) {
#else
    MNNCORRECT_CUSTOM_PARALLEL(nobs, [&](size_t start, size_t end) -> void {
    std::vector<Index_> indices;
    std::vector<Distance_> distances;
    auto searcher = index.initialize();
    for (size_t l = start; l < end; ++l) {
#endif

        searcher->search(query + ndim * l, k, &indices, &distances);
        size_t found = indices.size();

        auto& curout = output[l];
        curout.clear();
        curout.reserve(found);
        for (size_t i = 0; i < found; ++i) {
            curout.emplace_back(indices[i], distances[i]);
        }

#ifndef MNNCORRECT_CUSTOM_PARALLEL
    }
    }
#else
    }
    }, nthreads);
#endif

    return output;
}

template<typename Index_, typename Distance_>
void fuse_nn_results(
    const std::vector<std::pair<Index_, Distance_> >& base, 
    const std::vector<std::pair<Index_, Distance_> >& alt, 
    size_t num_neighbors, 
    std::vector<std::pair<Index_, Distance_> >& output,
    Index_ offset = 0) 
{
    output.clear();
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
                output.back().first += offset;
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
                output.back().first += offset;
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
        output.back().first += offset;
        ++aIt;
    }
}

}

}

#endif

