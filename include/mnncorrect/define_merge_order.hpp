#ifndef MNNCORRECT_DEFINE_MERGE_ORDER_HPP
#define MNNCORRECT_DEFINE_MERGE_ORDER_HPP

#include <algorithm>
#include <vector>
#include <cstddef>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace mnncorrect {

namespace internal {

template<typename Index_, typename Float_>
Float_ compute_total_variance(const std::size_t ndim, const Index_ nobs, const Float_* const values, std::vector<Float_>& mbuffer, const bool as_rss) {
    std::fill(mbuffer.begin(), mbuffer.end(), 0);

    Float_ total = 0;
    for (Index_ i = 0; i < nobs; ++i) {
        for (std::size_t d = 0; d < ndim; ++d) {
            const auto curval = values[sanisizer::nd_offset<std::size_t>(d, ndim, i)];
            auto& curmean = mbuffer[d];
            const Float_ delta = curval - curmean;
            curmean += delta/(i + 1);
            total += delta * (curval -  curmean);
        }
    }

    if (!as_rss) {
        total /= nobs - 1;
    }
    return total;
}

template<typename Index_, typename Float_>
std::vector<Float_> compute_total_variances(const std::size_t ndim, const std::vector<Index_>& nobs, const std::vector<const Float_*>& batches, const bool as_rss, const int num_threads) {
    const BatchIndex num_batches = nobs.size();
    auto vars = sanisizer::create<std::vector<Float_> >(num_batches);
    parallelize(num_threads, num_batches, [&](const int, const BatchIndex start, const BatchIndex length) -> void {
        auto mean_buffer = sanisizer::create<std::vector<Float_> >(ndim);
        for (BatchIndex b = start, end = start + length; b < end; ++b) {
            vars[b] = compute_total_variance(ndim, nobs[b], batches[b], mean_buffer, as_rss);
        }
    });

    return vars;
}

template<typename Stat_>
void define_merge_order(const std::vector<Stat_>& stat, std::vector<BatchIndex>& order) {
    const auto nbatches = stat.size();
    std::vector<std::pair<Stat_, BatchIndex> > ordering;
    ordering.reserve(nbatches);

    for (decltype(I(nbatches)) b = 0; b < nbatches; ++b) {
        ordering.emplace_back(stat[b], b);
    }
    std::sort(ordering.begin(), ordering.end(), std::greater<std::pair<Stat_, BatchIndex> >()); // batches with largest values for the statistic are more 'reference-y'. 

    order.clear();
    order.reserve(nbatches);
    for (const auto& batch : ordering) {
        order.push_back(batch.second);
    }
}

}

}

#endif
