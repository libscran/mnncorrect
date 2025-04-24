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
    auto found = indices.size();
    output.clear();
    output.reserve(found);
    for (decltype(found) i = 0; i < found; ++i) {
        output.emplace_back(indices[i], distances[i]);
    }
}

template<typename Index_, typename Float_>
void quick_find_nns(Index_ nobs, const Float_* query, const knncolle::Prebuilt<Index_, Float_, Float_>& index, int k, int nthreads, NeighborSet<Index_, Float_>& output, Index_ shift) {
    std::size_t ndim = index.num_dimensions();

    parallelize(nthreads, nobs, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = index.initialize();

        for (Index_ l = start, end = start + length; l < end; ++l) {
            auto ptr = query + ndim * static_cast<std::size_t>(l); // cast to avoid overflow.
            searcher->search(ptr, k, &indices, &distances);
            fill_pair_vector(indices, distances, output[l + shift]);
        }
    });
}

template<typename Index_, typename Float_>
NeighborSet<Index_, Float_> quick_find_nns(Index_ nobs, const Float_* query, const knncolle::Prebuilt<Index_, Float_, Float_>& index, int k, int nthreads) {
    NeighborSet<Index_, Float_> output(nobs);
    quick_find_nns(nobs, query, index, k, nthreads, output, /* shift = */ static_cast<Index_>(0));
    return output;
}

template<typename Index_, typename Distance_>
void fuse_nn_results(
    const std::vector<std::pair<Index_, Distance_> >& base, 
    const std::vector<std::pair<Index_, Distance_> >& alt, 
    int k,
    std::vector<std::pair<Index_, Distance_> >& output,
    Index_ offset) 
{
    output.clear();
    decltype(output.size()) num_neighbors = k; // converting into size_type for easier comparisons below.
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

template<typename Index_, typename Float_>
void quick_fuse_nns(NeighborSet<Index_, Float_>& existing, const Float_* query, const knncolle::Prebuilt<Index_, Float_, Float_>& index, int k, int nthreads, Index_ offset) {
    Index_ nobs = existing.size();
    std::size_t ndim = index.num_dimensions();

    parallelize(nthreads, nobs, [&](int, Index_ start, Index_ length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = index.initialize();
        std::vector<std::pair<Index_, Float_> > search_buffer, fuse_buffer;

        for (std::size_t l = start, end = start + length; l < end; ++l) {
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

