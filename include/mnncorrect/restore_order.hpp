#ifndef MNNCORRECT_RESTORE_ORDER_HPP
#define MNNCORRECT_RESTORE_ORDER_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_>
std::pair<std::vector<Index_>, Index_> define_merge_order_offsets(const std::vector<BatchIndex>& merge_order, const std::vector<Index_>& sizes) {
    BatchIndex nbatches = merge_order.size();
    Index_ accumulated = 0;
    std::vector<Index_> offsets(nbatches);
    for (BatchIndex b = 0; b < nbatches; ++b) {
        offsets[merge_order[b]] = accumulated;
        accumulated += sizes[merge_order[b]];
    }
    return std::make_pair(std::move(offsets), accumulated);
}

template<typename Index_, typename Float_>
void reorder_data(std::size_t ndim, Index_ nobs, const std::vector<Index_>& reindex, Float_* output) {
    std::vector<unsigned char> used(nobs);
    std::vector<Float_> buffer(ndim);

    for (Index_ i = 0; i < nobs; ++i) {
        if (used[i]) {
            continue;
        }

        used[i] = true;
        auto target = reindex[i];
        if (target != i) {
            // Moving the current vector into a buffer to free up 
            // some space for the shuffling. This avoids the need
            // to do a bunch of std::swap() calls.
            auto current_ptr = output + i * ndim;
            std::copy_n(current_ptr, ndim, buffer.data());

            while (target != i) {
                auto tptr = output + target * ndim;
                std::copy_n(tptr, ndim, current_ptr);
                used[target] = true;
                current_ptr = tptr;
                target = reindex[target];
            }

            std::copy_n(buffer.data(), ndim, current_ptr);
        }
    }
}

template<typename Index_, typename Float_>
void restore_order(std::size_t ndim, const std::vector<BatchIndex>& merge_order, const std::vector<Index_>& sizes, Float_* output) {
    auto offset_out = define_merge_order_offsets(merge_order, sizes);
    const auto& offsets = offset_out.first;
    Index_ nobs = offset_out.second;
    BatchIndex nbatches = offsets.size();

    std::vector<Index_> reindex(nobs);
    auto ptr = reindex.data();
    for (BatchIndex b = 0; b < nbatches; ++b) {
        std::iota(ptr, ptr + sizes[b], offsets[b]);
        ptr += sizes[b];
    }

    reorder_data(ndim, nobs, reindex, output);
    return;    
}

template<typename Index_, typename Batch, typename Float_>
void restore_order(std::size_t ndim, const std::vector<BatchIndex>& merge_order, const std::vector<Index_>& sizes, const Batch* batch, Float_* output) {
    auto offset_out = define_merge_order_offsets(merge_order, sizes);
    auto& offsets = offset_out.first;
    Index_ nobs = offset_out.second;

    std::vector<Index_> reindex(nobs);
    for (Index_ o = 0; o < nobs; ++o) {
        auto& off = offsets[batch[o]];
        reindex[o] = off;
        ++off;
    }

    reorder_data(ndim, nobs, reindex, output);
    return;
}

}

}

#endif
