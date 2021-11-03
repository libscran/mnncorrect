#ifndef MNNCORRECT_RESTORE_ORDER_HPP
#define MNNCORRECT_RESTORE_ORDER_HPP

#include <vector>
#include <algorithm>
#include <numeric>

namespace mnncorrect {

namespace restore {

std::pair<std::vector<size_t>, size_t> define_offsets(const std::vector<int>& merge_order, const std::vector<size_t>& sizes) {
    size_t nbatches = merge_order.size();
    size_t accumulated = 0;
    std::vector<size_t> offsets(nbatches);
    for (size_t b = 0; b < nbatches; ++b) {
        offsets[merge_order[b]] = accumulated;
        accumulated += sizes[merge_order[b]];
    }
    return std::make_pair(std::move(offsets), accumulated);
}

template<typename Float>
void reorder(int ndim, size_t nobs, const std::vector<size_t>& reindex, Float* output) {
    std::vector<char> used(nobs);
    for (size_t i = 0; i < nobs; ++i) {
        if (used[i]) {
            continue;
        }

        used[i] = true;
        auto current = i;
        auto target = reindex[i];
        while (target != i) {
            auto cptr = output + current * ndim;
            auto tptr = output + target * ndim;
            for (int d = 0; d < ndim; ++d) {
                std::swap(*(cptr + d), *(tptr + d));
            }
            used[target] = true;
            current = target;
            target = reindex[target];
        }
    }
}

}

template<typename Float>
void restore_order(int ndim, const std::vector<int>& merge_order, const std::vector<size_t>& sizes, Float* output) {
    auto offset_out = restore::define_offsets(merge_order, sizes);
    const auto& offsets = offset_out.first;
    size_t nobs = offset_out.second;
    size_t nbatches = offsets.size();

    std::vector<size_t> reindex(nobs);
    auto ptr = reindex.data();
    for (size_t b = 0; b < nbatches; ++b) {
        std::iota(ptr, ptr + sizes[b], offsets[b]);
        ptr += sizes[b];
    }

    restore::reorder(ndim, nobs, reindex, output);
    return;    
}

template<typename Float, typename Batch>
void restore_order(int ndim, const std::vector<int>& merge_order, const std::vector<size_t>& sizes, const Batch* batch, Float* output) {
    auto offset_out = restore::define_offsets(merge_order, sizes);
    auto& offsets = offset_out.first;
    size_t nobs = offset_out.second;

    std::vector<size_t> reindex(nobs);
    for (size_t o = 0; o < nobs; ++o) {
        auto& off = offsets[batch[o]];
        reindex[o] = off;
        ++off;
    }

    restore::reorder(ndim, nobs, reindex, output);
    return;
}

}

#endif
