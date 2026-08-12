#ifndef MNNCORRECT_RESTORE_INPUT_ORDER_HPP
#define MNNCORRECT_RESTORE_INPUT_ORDER_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>
#include <cassert>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace mnncorrect {

template<typename Index_, typename Batch_, typename Float_>
void restore_input_order(const std::size_t num_dim, const Index_ num_total, const std::vector<Batch<Index_> >& contiguous_batches, const Batch_* const batch_factor, Float_* const output) {
    assert([&]{
        Index_ actual = 0;
        for (auto& batch : contiguous_batches) {
            actual += batch.size;
        }
        return actual == num_total;
    }());

    auto reindex = sanisizer::create<std::vector<Index_> >(num_total);
    {
        const auto nbatches = contiguous_batches.size();
        auto offsets = sanisizer::create<std::vector<Index_> >(nbatches);
        for (I<decltype(nbatches)> b = 0; b < nbatches; ++b) {
            offsets[b] = contiguous_batches[b].offset;
        }
        for (Index_ o = 0; o < num_total; ++o) {
            auto& off = offsets[batch_factor[o]];
            reindex[o] = off;
            ++off;
        }
    }

    auto buffer = sanisizer::create<std::vector<Float_> >(num_dim);
    for (Index_ i = 0; i < num_total; ++i) {
        // We use 'num_total' as a sentinel to indicate that this observation has already been reindexed.
        if (reindex[i] == num_total) {
            continue;
        }

        auto target = reindex[i];
        reindex[i] = num_total;
        if (target == i) {
            continue;
        }

        // Moving the current vector into a buffer to free up some space for the shuffling.
        // This avoids the need/ to do a bunch of std::swap() calls.
        auto current_ptr = output + sanisizer::product_unsafe<std::size_t>(i, num_dim);
        std::copy_n(current_ptr, num_dim, buffer.data());

        do {
            const auto tptr = output + sanisizer::product_unsafe<std::size_t>(target, num_dim);
            std::copy_n(tptr, num_dim, current_ptr);
            const auto new_target = reindex[target];
            reindex[target] = num_total;
            target = new_target;
            current_ptr = tptr;
        } while (target != i);

        std::copy_n(buffer.data(), num_dim, current_ptr);
    }
}

}

#endif
