#ifndef MNNCORRECT_RESTORE_INPUT_ORDER_HPP
#define MNNCORRECT_RESTORE_INPUT_ORDER_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Batch, typename Float_>
void restore_input_order(std::size_t ndim, const std::vector<Index_>& sizes, const Batch* batch, Float_* output) {
    BatchIndex nbatches = sizes.size();
    Index_ nobs = 0;
    std::vector<Index_> offsets(nbatches);
    for (BatchIndex b = 0; b < nbatches; ++b) {
        offsets[b] = nobs;
        nobs += sizes[b];
    }

    std::vector<Index_> reindex(nobs);
    for (Index_ o = 0; o < nobs; ++o) {
        auto& off = offsets[batch[o]];
        reindex[o] = off;
        ++off;
    }

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
            auto current_ptr = output + static_cast<std::size_t>(i) * ndim; // cast to avoid overflow.
            std::copy_n(current_ptr, ndim, buffer.data());

            while (target != i) {
                auto tptr = output + static_cast<std::size_t>(target) * ndim; // more casting to avoid overflow.
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
