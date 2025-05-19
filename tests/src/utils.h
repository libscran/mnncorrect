#ifndef UTILS_H
#define UTILS_H

#include "knncolle/knncolle.hpp"
#include "mnncorrect/utils.hpp"

#include <algorithm>
#include <numeric>
#include <vector>
#include <cstddef>

template<typename Index_, typename Float_>
std::unique_ptr<knncolle::Prebuilt<Index_, Float_, Float_> > subset_and_index(
    std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* data,
    const knncolle::Builder<Index_, Float_, Float_>& builder,
    std::vector<Float_>& buffer)
{ 
    auto num_obs = ids.size();
    buffer.resize(static_cast<std::size_t>(num_obs) * num_dim);
    for (decltype(num_obs) e = 0; e < num_obs; ++e) {
        std::copy_n(
            data + static_cast<std::size_t>(ids[e]) * num_dim,
            num_dim,
            buffer.begin() + static_cast<std::size_t>(e) * num_dim
        );
    }
    return builder.build_unique(knncolle::SimpleMatrix<Index_, Float_>(num_dim, num_obs, buffer.data()));
}

template<typename Index_, typename Float_>
void find_neighbors(
    std::size_t num_dim,
    const std::vector<Index_>& ids,
    const Float_* data,
    const knncolle::Prebuilt<Index_, Float_, Float_>& index,
    const std::vector<Index_>& index_ids,
    int num_neighbors,
    mnncorrect::internal::NeighborSet<Index_, Float_>& output)
{ 
    std::vector<int> indices;
    std::vector<double> distances;
    auto searcher = index.initialize();
    for (auto i : ids) {
        searcher->search(data + static_cast<std::size_t>(i) * num_dim, num_neighbors, &indices, &distances);
        auto found = indices.size();
        for (decltype(found) j = 0; j < found; ++j) {
            output[i].emplace_back(index_ids[indices[j]], distances[j]);
        }
    }
}

template<typename Index_, typename Float_, class Rng_>
std::vector<mnncorrect::internal::BatchInfo<Index_, Float_> > mock_batches(
    std::size_t num_dim,
    const std::vector<Index_>& batch_sizes,
    const Float_* data,
    bool extras,
    Rng_& rng,
    const knncolle::Builder<Index_, Float_, Float_>& builder)
{
    auto num_batches = batch_sizes.size();
    std::vector<mnncorrect::internal::BatchInfo<int, double> > all_batches(num_batches);
    if (extras) {
        for (auto& batch : all_batches) {
            batch.extras.resize(num_batches);
        }
    }

    Index_ sofar = 0;
    for (decltype(num_batches) b = 0; b < num_batches; ++b) {
        auto bsize = batch_sizes[b];

        // Firstly adding the core stretch.
        int quarter = bsize / 4, half = bsize / 2;
        int start = rng() % quarter;
        int number = rng() % half + quarter;
        all_batches[b].offset = sofar + start;
        all_batches[b].num_obs = number;

        // Now adding anything before it.
        if (extras) {
            for (int i = 0; i < start; ++i) {
                auto chosen = rng() % num_batches;
                all_batches[chosen].extras[b].ids.push_back(i + sofar);
            }

            int remaining = bsize - number - start;
            for (int i = 0; i < remaining; ++i) {
                auto chosen = rng() % num_batches;
                all_batches[chosen].extras[b].ids.push_back(i + sofar + start + number);
            }
        }

        sofar += bsize;
    }

    // Creating the indices.
    std::vector<double> buffer;
    for (decltype(num_batches) b = 0; b < num_batches; ++b) {
        auto& batch = all_batches[b];
        batch.index = builder.build_unique(knncolle::SimpleMatrix<int, double>(num_dim, batch.num_obs, data + static_cast<std::size_t>(batch.offset) * num_dim));
        for (auto& extra : batch.extras) {
            extra.index = subset_and_index(num_dim, extra.ids, data, builder, buffer);
        }
    }

    return all_batches;
}

template<typename Index_, typename Float_>
std::vector<std::vector<Index_> > create_assignments(const std::vector<mnncorrect::internal::BatchInfo<Index_, Float_> >& all_batches) {
    std::vector<std::vector<Index_> > assignments;
    assignments.reserve(all_batches.size());
    for (const auto& batch : all_batches) {
        std::vector<Index_> current(batch.num_obs);
        std::iota(current.begin(), current.end(), batch.offset);
        for (const auto& extra : batch.extras) {
            current.insert(current.end(), extra.ids.begin(), extra.ids.end());
        }
        std::sort(current.begin(), current.end());
        assignments.emplace_back(std::move(current));
    }
    return assignments;
}

#endif
