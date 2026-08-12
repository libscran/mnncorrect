#ifndef MNNCORRECT_REORDER_MATRIX_IN_PLACE_HPP
#define MNNCORRECT_REORDER_MATRIX_IN_PLACE_HPP

#include <algorithm>
#include <vector>

#include "sanisizer/sanisizer.hpp"

namespace mnncorrect {

template<typename Index_, typename Float_>
void reorder_matrix_in_place(const std::size_t num_dim, const Index_ num_total, std::vector<Index_>& order, Float_* const data, std::vector<Float_>& buffer) {
    for (Index_ i = 0; i < num_total; ++i) {
        // We use 'num_total' as a sentinel to indicate that this observation has already been reordered.
        if (order[i] == num_total) {
            continue;
        }

        auto target = order[i];
        order[i] = num_total;
        if (target == i) {
            continue;
        }

        // Moving the current vector into a buffer to free up some space for the shuffling.
        // This avoids the need/ to do a bunch of std::swap() calls.
        auto current_ptr = data + sanisizer::product_unsafe<std::size_t>(i, num_dim);
        std::copy_n(current_ptr, num_dim, buffer.data());

        do {
            const auto tptr = data + sanisizer::product_unsafe<std::size_t>(target, num_dim);
            std::copy_n(tptr, num_dim, current_ptr);
            const auto new_target = order[target];
            order[target] = num_total;
            target = new_target;
            current_ptr = tptr;
        } while (target != i);

        std::copy_n(buffer.data(), num_dim, current_ptr);
    }
}

}

#endif
