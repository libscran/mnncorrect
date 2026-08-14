#ifndef MNNCORRECT_FIND_NEIGHBORS_HPP
#define MNNCORRECT_FIND_NEIGHBORS_HPP

#include <vector>
#include <utility>
#include <cstddef>

#include "knncolle/knncolle.hpp"
#include "sanisizer/sanisizer.hpp"

#include "find_neighbors.hpp"
#include "utils.hpp"

namespace mnncorrect {

template<typename Index_, typename Float_>
void fill_pair_vector(const std::vector<Index_>& indices, const std::vector<Float_>& distances, std::vector<std::pair<Index_, Float_> >& output) {
    const auto found = indices.size();
    output.clear();
    output.reserve(found);
    for (I<decltype(found)> i = 0; i < found; ++i) {
        output.emplace_back(indices[i], distances[i]);
    }
}

template<typename Index_, typename Distance_>
void fuse_nn_results(
    const std::vector<std::pair<Index_, Distance_> >& base, 
    const std::vector<std::pair<Index_, Distance_> >& alt, 
    const int num_neighbors,
    std::vector<std::pair<Index_, Distance_> >& output
) {
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
        } while (sanisizer::is_less_than(output.size(), num_neighbors));
    }

    while (bIt != bEnd && sanisizer::is_less_than(output.size(), num_neighbors)) {
        output.push_back(*bIt);
        ++bIt;
    }

    while (aIt != aEnd && sanisizer::is_less_than(output.size(), num_neighbors)) {
        output.push_back(*aIt);
        ++aIt;
    }
}

template<typename Index_, class GetIndex_, typename Float_>
void find_neighbors(
    const std::size_t num_dim,
    const Index_ num_query,
    const GetIndex_ get_query_index,
    const MetaBatch<Index_, Float_>& subject,
    const Float_* const data,
    const int num_neighbors,
    const bool fuse_neighbors,
    const int num_threads,
    NeighborSet<Index_, Float_>& output
) {
    parallelize(num_threads, num_query, [&](const int, const Index_ query_start, const Index_ query_length) -> void {
        std::vector<Index_> indices;
        std::vector<Float_> distances;
        std::vector<std::pair<Index_, Float_> > fuse_buffer1, fuse_buffer2;

        { // scoped to prevent confusion from variable aliasing with subject.corrected.
            auto searcher = subject.original_index->initialize();
            const auto capped_neighbors = knncolle::cap_k_query(num_neighbors, subject.original_index->num_observations());

            for (Index_ q = query_start, query_end = query_start + query_length; q < query_end; ++q) {
                const auto qidx = get_query_index(q);
                const auto ptr = data + sanisizer::product_unsafe<std::size_t>(qidx, num_dim);
                searcher->search(ptr, capped_neighbors, &indices, &distances);
                for (auto& i : indices) {
                    i += subject.original_ids.start;
                }

                auto& curnn = output[qidx];
                if (!fuse_neighbors) {
                    fill_pair_vector(indices, distances, curnn);
                } else {
                    fuse_buffer1.swap(curnn);
                    fill_pair_vector(indices, distances, fuse_buffer2);
                    fuse_nn_results(fuse_buffer1, fuse_buffer2, num_neighbors, curnn);
                }
            }
        }

        for (const auto& corrected : subject.corrected) {
            auto searcher = corrected.index->initialize();
            const auto capped_neighbors = knncolle::cap_k_query(num_neighbors, corrected.index->num_observations());

            for (Index_ q = query_start, query_end = query_start + query_length; q < query_end; ++q) {
                const auto qidx = get_query_index(q);
                const auto ptr = data + sanisizer::product_unsafe<std::size_t>(qidx, num_dim);
                searcher->search(ptr, capped_neighbors, &indices, &distances);
                for (auto& i : indices) {
                    i = corrected.ids[i];
                }
                auto& curnn = output[qidx];
                fuse_buffer1.swap(curnn);
                fill_pair_vector(indices, distances, fuse_buffer2);
                fuse_nn_results(fuse_buffer1, fuse_buffer2, num_neighbors, curnn);
            }
        }
    });
}

template<typename Index_, typename Float_>
void find_neighbors(
    const std::size_t num_dim,
    const MetaBatch<Index_, Float_>& query,
    const MetaBatch<Index_, Float_>& subject,
    const Float_* const data,
    const int num_neighbors,
    const bool fuse_neighbors,
    const int num_threads,
    NeighborSet<Index_, Float_>& output
) {
    find_neighbors(
        num_dim,
        query.original_ids.size,
        [&](const Index_ q) -> Index_ { return q + query.original_ids.start; },
        subject,
        data,
        num_neighbors,
        fuse_neighbors,
        num_threads,
        output 
    );

    for (const auto& corrected : query.corrected) {
        find_neighbors(
            num_dim,
            static_cast<Index_>(corrected.ids.size()),
            [&](const Index_ q) -> Index_ { return corrected.ids[q]; },
            subject,
            data,
            num_neighbors,
            fuse_neighbors,
            num_threads,
            output
        );
    }
}

template<typename Index_, typename Float_>
void find_neighbors(
    const std::size_t num_dim,
    const std::vector<MetaBatch<Index_, Float_> >& references,
    const MetaBatch<Index_, Float_>& target,
    const Float_* const data,
    const int num_neighbors,
    const int num_threads,
    NeighborSet<Index_, Float_>& output
) {
    const auto num_refs = references.size();
    for (I<decltype(num_refs)> b = 0; b < num_refs; ++b) {
        const auto& curref = references[b];
        find_neighbors(num_dim, curref, target, data, num_neighbors, false, num_threads, output);
        find_neighbors(num_dim, target, curref, data, num_neighbors, b > 0, num_threads, output);
    }
}

}

#endif
