#ifndef MNNCORRECT_RESTORE_ORDER_HPP
#define MNNCORRECT_RESTORE_ORDER_HPP

#include <vector>
#include <algorithm>
#include <numeric>

namespace mnncorrect {

namespace internal {

inline std::pair<std::vector<size_t>, size_t> define_merge_order_offsets(const std::vector<size_t>& merge_order, const std::vector<size_t>& sizes) {
    size_t nbatches = merge_order.size();
    size_t accumulated = 0;
    std::vector<size_t> offsets(nbatches);
    for (size_t b = 0; b < nbatches; ++b) {
        offsets[merge_order[b]] = accumulated;
        accumulated += sizes[merge_order[b]];
    }
    return std::make_pair(std::move(offsets), accumulated);
}

template<typename Float_>
void reorder_data(size_t ndim, size_t nobs, const std::vector<size_t>& reindex, Float_* output) {
    std::vector<unsigned char> used(nobs);
    std::vector<Float_> buffer(ndim);

    for (size_t i = 0; i < nobs; ++i) {
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

template<typename Float_>
void restore_order(size_t ndim, const std::vector<size_t>& merge_order, const std::vector<size_t>& sizes, Float_* output) {
    auto offset_out = define_merge_order_offsets(merge_order, sizes);
    const auto& offsets = offset_out.first;
    size_t nobs = offset_out.second;
    size_t nbatches = offsets.size();

    std::vector<size_t> reindex(nobs);
    auto ptr = reindex.data();
    for (size_t b = 0; b < nbatches; ++b) {
        std::iota(ptr, ptr + sizes[b], offsets[b]);
        ptr += sizes[b];
    }

    reorder_data(ndim, nobs, reindex, output);
    return;    
}

template<typename Float_, typename Batch>
void restore_order(size_t ndim, const std::vector<size_t>& merge_order, const std::vector<size_t>& sizes, const Batch* batch, Float_* output) {
    auto offset_out = define_merge_order_offsets(merge_order, sizes);
    auto& offsets = offset_out.first;
    size_t nobs = offset_out.second;

    std::vector<size_t> reindex(nobs);
    for (size_t o = 0; o < nobs; ++o) {
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
