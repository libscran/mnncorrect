#ifndef MNNCORRECT_FUSE_NN_RESULTS_HPP
#define MNNCORRECT_FUSE_NN_RESULTS_HPP

#include <vector>

#include "knncolle/knncolle.hpp"
#include "utils.hpp"
#include "parallelize.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Distance_>
void fill_pair_vector(const std::vector<Index_>& indices, const std::vector<Distance_>& distances, std::vector<std::pair<Index_, Distance_> >& output) {
    size_t found = indices.size();
    output.clear();
    output.reserve(found);
    for (size_t i = 0; i < found; ++i) {
        output.emplace_back(indices[i], distances[i]);
    }
}

template<typename Dim_, typename Index_, typename Distance_>
void quick_find_nns(size_t nobs, const Distance_* query, const knncolle::Prebuilt<Dim_, Index_, Distance_>& index, int k, int nthreads, NeighborSet<Index_, Distance_>& output, size_t shift) {
    size_t ndim = index.num_dimensions();

    parallelize(nthreads, nobs, [&](int, size_t start, size_t length) -> void {
        std::vector<Index_> indices;
        std::vector<Distance_> distances;
        auto searcher = index.initialize();

        for (size_t l = start, end = start + length; l < end; ++l) {
            auto ptr = query + ndim * l; // everything is a size_t, so no chance of overflow.
            searcher->search(ptr, k, &indices, &distances);
            fill_pair_vector(indices, distances, output[l + shift]);
        }
    });
}

template<typename Dim_, typename Index_, typename Distance_>
NeighborSet<Index_, Distance_> quick_find_nns(size_t nobs, const Distance_* query, const knncolle::Prebuilt<Dim_, Index_, Distance_>& index, int k, [[maybe_unused]] int nthreads) {
    NeighborSet<Index_, Distance_> output(nobs);
    quick_find_nns(nobs, query, index, k, nthreads, output, /* shift = */ 0);
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

template<typename Dim_, typename Index_, typename Distance_>
void quick_fuse_nns(NeighborSet<Index_, Distance_>& existing, const Distance_* query, const knncolle::Prebuilt<Dim_, Index_, Distance_>& index, int k, int nthreads, Index_ offset) {
    size_t nobs = existing.size();
    size_t ndim = index.num_dimensions();

    parallelize(nthreads, nobs, [&](int, size_t start, size_t length) -> void {
        std::vector<Index_> indices;
        std::vector<Distance_> distances;
        auto searcher = index.initialize();
        std::vector<std::pair<Index_, Distance_> > search_buffer, fuse_buffer;

        for (size_t l = start, end = start + length; l < end; ++l) {
            auto ptr = query + ndim * l; // everything is a size_t, so no chance of overflow.
            searcher->search(ptr, k, &indices, &distances);
            fill_pair_vector(indices, distances, search_buffer);

            auto& curexisting = existing[l];
            fuse_nn_results(curexisting, search_buffer, k, fuse_buffer, offset);
            fuse_buffer.swap(curexisting);
        }
    });
}

}

}

#endif

