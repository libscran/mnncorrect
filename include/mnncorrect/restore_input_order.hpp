#ifndef MNNCORRECT_RESTORE_INPUT_ORDER_HPP
#define MNNCORRECT_RESTORE_INPUT_ORDER_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Batch, typename Float_>
void restore_input_order(const std::size_t ndim, const std::vector<Index_>& sizes, const Batch* const batch, Float_* const output) {
    const BatchIndex nbatches = sizes.size();
    Index_ nobs = 0;
    auto offsets = sanisizer::create<std::vector<Index_> >(nbatches);
    for (BatchIndex b = 0; b < nbatches; ++b) {
        offsets[b] = nobs;
        nobs += sizes[b]; // known to NOT overflow, see mnncorrect::compute().
    }

    auto reindex = sanisizer::create<std::vector<Index_> >(nobs);
    for (Index_ o = 0; o < nobs; ++o) {
        auto& off = offsets[batch[o]];
        reindex[o] = off;
        ++off;
    }

    auto used = sanisizer::create<std::vector<unsigned char> >(nobs);
    auto buffer = sanisizer::create<std::vector<Float_> >(ndim);
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
            auto current_ptr = output + sanisizer::product_unsafe<std::size_t>(i, ndim);
            std::copy_n(current_ptr, ndim, buffer.data());

            while (target != i) {
                const auto tptr = output + sanisizer::product_unsafe<std::size_t>(target, ndim);
                std::copy_n(tptr, ndim, current_ptr);
                used[target] = true;
                current_ptr = tptr;
                target = reindex[target];
            }

            std::copy_n(buffer.data(), ndim, current_ptr);
        }
    }

    return;
}

}

}

#endif
