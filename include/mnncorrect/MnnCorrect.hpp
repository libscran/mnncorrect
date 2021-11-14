#ifndef MNNCORRECT_MNNCORRECT_HPP
#define MNNCORRECT_MNNCORRECT_HPP

#include <algorithm>
#include <vector>
#include "AutomaticOrder.hpp"
#include "restore_order.hpp"
#include "knncolle/knncolle.hpp"

namespace mnncorrect {

template<typename Index = int, typename Float = double>
class MnnCorrect {
public:
    struct Defaults {
        static constexpr int num_neighbors = 15;

        static constexpr int num_clusters = 50;

        static constexpr bool approximate = false;

        static constexpr bool automatic_order = true;
    };

private:
    int num_neighbors = Defaults::num_neighbors;
    int num_clusters = Defaults::num_clusters;
    bool approximate = Defaults::approximate;
    bool automatic_order = Defaults::automatic_order;

public:
    MnnCorrect& set_num_neighbors(int n = Defaults::num_neighbors) {
        num_neighbors = n;
        return *this;
    }

    MnnCorrect& set_num_clusters(int n = Defaults::num_clusters) {
        num_clusters = n;
        return *this;
    }

    MnnCorrect& set_approximate(bool a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    MnnCorrect& set_automatic_order(bool a = Defaults::automatic_order) {
        automatic_order = a;
        return *this;
    }

public:
    struct Results {
        Results() {}
        Results(std::vector<int> mo, std::vector<int> np) : merge_order(std::move(mo)), num_pairs(std::move(np)) {}
        std::vector<int> merge_order;
        std::vector<int> num_pairs;
    };

private:
    template<class Builder>
    Results run_automatic_internal(int ndim, const std::vector<size_t>& nobs, const std::vector<const Float*>& batches, Builder bfun, Float* output) {
        AutomaticOrder<Index, Float, Builder> runner(ndim, nobs, batches, output, bfun, num_neighbors);
        runner.run(num_mads);
        return Results(runner.get_order(), runner.get_num_pairs());
    }

    Results run_internal(int ndim, const std::vector<size_t>& nobs, const std::vector<const Float*>& batches, Float* output) {
        typedef knncolle::Base<Index, Float> knncolleBase; 

        if (automatic_order) {
            if (approximate) {
                auto builder = [](int nd, size_t no, const Float* d) -> auto { 
                    return std::shared_ptr<knncolleBase>(new knncolle::AnnoyEuclidean<Index, Float>(nd, no, d)); 
                };
                return run_automatic_internal(ndim, nobs, batches, builder, output);
            } else {
                auto builder = [](int nd, size_t no, const Float* d) -> auto { 
                    return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d)); 
                };
                return run_automatic_internal(ndim, nobs, batches, builder, output);
            }
        }
    }

public:
    Results run(int ndim, const std::vector<size_t>& nobs, const std::vector<const Float*>& batches, Float* output) {
        auto stats = run_internal(ndim, nobs, batches, output);
        restore_order(ndim, stats.merge_order, nobs, output);
        return stats;
    }

    Results run(int ndim, const std::vector<size_t>& nobs, const Float* input, Float* output) {
        std::vector<const Float*> batches;
        batches.reserve(nobs.size());
        for (auto n : nobs) {
            batches.push_back(input);
            input += n * ndim;
        }
        return run(ndim, nobs, batches, output);
    }

    template<typename Batch>
    Results run(int ndim, size_t nobs, const Float* input, const Batch* batch, Float* output) {
        const Batch nbatches = (nobs ? *std::max_element(batch, batch + nobs) + 1 : 0);
        std::vector<size_t> sizes(nbatches);
        for (size_t o = 0; o < nobs; ++o) {
            ++sizes[batch[o]];
        }

        // Avoiding the need to allocate a temporary buffer
        // if we're already dealing with contiguous batches.
        bool already_sorted = true;
        for (size_t o = 1; o < nobs; ++o) {
           if (batch[o] < batch[o-1]) {
               already_sorted = false;
               break;
           }
        }
        if (already_sorted) {
            return run(ndim, sizes, input, output);
        }

        size_t accumulated = 0;
        std::vector<size_t> offsets(nbatches);
        for (size_t b = 0; b < nbatches; ++b) {
            offsets[b] = accumulated;
            accumulated += sizes[b];
        }

        // Dumping everything by order into another vector.
        std::vector<Float> tmp(ndim * nobs);
        std::vector<const Float*> ptrs(nbatches, tmp.data());
        for (size_t b = 0; b < nbatches; ++b) {
            ptrs[b] += offsets[b] * ndim;
        }
        for (size_t o = 0; o < nobs; ++o) {
            auto current = input + o * ndim;
            auto& offset = offsets[batch[o]];
            auto destination = tmp.data() + ndim * offset;
            std::copy(current, current + ndim, destination);
            ++offset;
        }

        auto stats = run_internal(ndim, sizes, ptrs, output);
        restore_order(ndim, stats.merge_order, sizes, batch, output);
        return stats;
    }
};

}

#endif
