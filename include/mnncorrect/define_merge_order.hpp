#ifndef MNNCORRECT_DEFINE_MERGE_ORDER_HPP
#define MNNCORRECT_DEFINE_MERGE_ORDER_HPP

#include <algorithm>
#include <vector>
#include <cstddef>
#include <cassert>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace mnncorrect {

template<typename Index_, typename Float_>
Float_ compute_total_variance(const std::size_t ndim, const Index_ nobs, const Float_* const values, std::vector<Float_>& mean_buffer, const bool as_rss) {
    assert(mean_buffer.size() == ndim);
    std::fill(mean_buffer.begin(), mean_buffer.end(), 0);

    Float_ total = 0;
    for (Index_ i = 0; i < nobs; ++i) {
        for (std::size_t d = 0; d < ndim; ++d) {
            const auto curval = values[sanisizer::nd_offset<std::size_t>(d, ndim, i)];
            auto& curmean = mean_buffer[d];
            const Float_ delta = curval - curmean;
            curmean += delta/(i + 1);
            total += delta * (curval -  curmean);
        }
    }

    if (!as_rss) {
        if (nobs > 1) { // batches with fewer than 2 cells get a 'variance' of zero, to avoid problems during sorting.
            total /= nobs - 1;
        }
    }
    return total;
}

template<typename Index_, typename Float_>
std::vector<Float_> compute_total_variances(
    const std::size_t ndim,
    const std::vector<Batch<Index_> >& batches,
    const Float_* const data,
    const bool as_rss,
    const int num_threads
) {
    const auto num_batches = batches.size();
    auto output = sanisizer::create<std::vector<Float_> >(num_batches);
    parallelize(num_threads, num_batches, [&](const int, const I<decltype(num_batches)> start, const I<decltype(num_batches)> length) -> void {
        auto mean_buffer = sanisizer::create<std::vector<Float_> >(ndim);
        for (I<decltype(num_batches)> b = start, end = start + length; b < end; ++b) {
            output[b] = compute_total_variance(
                ndim,
                batches[b].size,
                data + sanisizer::product_unsafe<std::size_t>(batches[b].start, ndim),
                mean_buffer,
                as_rss
            );
        }
    });
    return output;
}

template<class Fun_>
void define_merge_order(const BatchIndex num_batches, Fun_ fun, std::vector<BatchIndex>& order) {
    sanisizer::resize(order, num_batches);
    std::iota(order.begin(), order.end(), static_cast<BatchIndex>(0));
    std::sort(
        order.begin(),
        order.end(),
        [&](BatchIndex left, BatchIndex right) -> bool {
            const auto lval = fun(left);
            const auto rval = fun(right);
            if (lval == rval) {
                return left < right;
            } else {
                return lval > rval;
            }
        }
    );
}

template<typename Index_>
void define_size_merge_order(const std::vector<Batch<Index_> >& batches, std::vector<BatchIndex>& order) {
    define_merge_order(
        batches.size(), // we already assume that number of batches can fit in a BatchIndex.
        [&](BatchIndex i) -> Index_ {
            return batches[i].size;
        },
        order
    );
}

template<class Float_>
void define_variance_merge_order(const std::vector<Float_>& variances, std::vector<BatchIndex>& order) {
    define_merge_order(
        variances.size(), // we already assume that number of batches can fit in a BatchIndex.
        [&](BatchIndex i) -> Float_ {
            return variances[i];
        },
        order
    );
}

}

#endif
