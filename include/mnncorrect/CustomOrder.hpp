#ifndef MNNCORRECT_CUSTOM_ORDER_HPP
#define MNNCORRECT_CUSTOM_ORDER_HPP

#include <algorithm>
#include <unordered_set>
#include <stdexcept>

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_mutual_nns.hpp"
#include "fuse_nn_results.hpp"
#include "correct_target.hpp"

namespace mnncorrect {

namespace internal {

template<typename Dim_, typename Index_, typename Float_>
class CustomOrder {
public:
    template<typename Order_>
    CustomOrder(
        size_t ndim, 
        std::vector<size_t> nobs,
        std::vector<const Float_*> batches,
        Float_* corrected,
        std::unique_ptr<knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_> > builder,
        int num_neighbors,
        const Order_* order,
        size_t nobs_cap,
        int nthreads) 
    :
        my_ndim(ndim), 
        my_nobs(std::move(nobs)), 
        my_batches(std::move(batches)),
        my_builder(std::move(builder)),
        my_indices(my_batches.size()),
        my_num_neighbors(num_neighbors),
        my_corrected(corrected),
        my_order(order, order + my_batches.size()),
        my_nobs_cap(nobs_cap),
        my_nthreads(nthreads)
    {
        size_t nbatches = my_nobs.size();
        if (nbatches != my_batches.size()) {
            throw std::runtime_error("length of 'no' and 'b' must be equal");
        }
        if (nbatches == 0) {
            return;
        }

#ifndef MNNCORRECT_CUSTOM_PARALLEL
#ifdef _OPENMP
        #pragma omp parallel num_threads(my_nthreads)
#endif
        {
#ifdef _OPENMP
            #pragma omp for
#endif
            for (size_t b = 0; b < nbatches; ++b) {
#else
        MNNCORRECT_CUSTOM_PARALLEL(nbatches, [&](size_t start, size_t end) -> void {
            for (size_t b = start; b < end; ++b) {
#endif

                my_indices[b] = my_builder->build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(my_ndim, my_nobs[b], my_batches[b]));

#ifndef MNNCORRECT_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, my_nthreads);
#endif

        // Picking the first batch to be our reference.
        auto first = my_order[0];
        const size_t rnum = my_nobs[first];
        const Float_* rdata = my_batches[first];
        std::copy(rdata, rdata + my_ndim * rnum, corrected);
        my_ncorrected += rnum;

        if (my_nobs.size() > 1) {
            auto second = my_order[1];
            my_neighbors_target = quick_find_nns(my_nobs[second], my_batches[second], *(my_indices[first]), my_num_neighbors, my_nthreads);
            my_neighbors_ref = quick_find_nns(rnum, rdata, *(my_indices[second]), my_num_neighbors, my_nthreads);
        }
    }

protected:
    int my_ndim;
    std::vector<size_t> my_nobs;
    std::vector<const Float_*> my_batches;

    std::unique_ptr<knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_> > my_builder;
    std::vector<std::unique_ptr<knncolle::Prebuilt<Dim_, Index_, Float_> > > my_indices;

    int my_num_neighbors;
    NeighborSet<Index_, Float_> my_neighbors_ref;
    NeighborSet<Index_, Float_> my_neighbors_target;

    Float_* my_corrected;
    size_t my_ncorrected = 0;
    std::vector<size_t> my_order;
    std::vector<size_t> my_num_pairs;

    size_t my_nobs_cap;
    int my_nthreads;

protected:
    void update(size_t position) {
        auto latest = my_order[position];
        size_t lnum = my_nobs[latest]; 
        const Float_* ldata = my_corrected + my_ncorrected * my_ndim;
        my_ncorrected += lnum;

        ++position;
        if (position == my_batches.size()) { 
            return;
        }

        // Updating all statistics with the latest batch added to the corrected reference.
        my_indices[latest] = my_builder->build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(my_ndim, lnum, ldata));

        auto next = my_order[position];
        auto next_data = my_batches[next];
        auto next_num = my_nobs[next];
        const auto& next_index = my_indices[next];
        my_neighbors_ref.resize(my_ncorrected);

        // Progressively finding the best neighbors across the currently built batches.
        size_t previous_ncorrected = 0;
        for (size_t i = 0; i < position; ++i) {
            auto prev = my_order[i];
            const auto& prev_index = my_indices[prev];

            if (i == 0) {
                my_neighbors_target.resize(next_num);
                quick_find_nns(next_num, next_data, *prev_index, my_num_neighbors, my_nthreads, my_neighbors_target, 0);
            } else {
                quick_fuse_nns(my_neighbors_target, next_data, *prev_index, my_num_neighbors, my_nthreads, static_cast<Index_>(previous_ncorrected));
            }

            auto prev_num = my_nobs[prev];
            auto prev_data = my_corrected + previous_ncorrected * my_ndim;
            quick_find_nns(prev_num, prev_data, *next_index, my_num_neighbors, my_nthreads, my_neighbors_ref, previous_ncorrected);

            previous_ncorrected += prev_num;
        }

        return;
    }

public:
    void run(Float_ nmads, int robust_iterations, double robust_trim) {
        size_t nbatches = my_batches.size();
        for (size_t i = 1; i < nbatches; ++i) {
            auto mnns = find_mutual_nns(my_neighbors_ref, my_neighbors_target);
            auto tnum = my_nobs[my_order[i]];
            auto tdata = my_batches[my_order[i]];

            correct_target(
                my_ndim, 
                my_ncorrected, 
                my_corrected, 
                tnum, 
                tdata, 
                mnns,
                my_builder,
                my_num_neighbors,
                nmads,
                robust_iterations,
                robust_trim,
                my_corrected + my_ncorrected * my_ndim,
                my_nobs_cap,
                my_nthreads
            );

            update(i);
            my_num_pairs.push_back(mnns.num_pairs);
        }
    }

    const auto& get_num_pairs() const { return my_num_pairs; }

    const auto& get_order() const { return my_order; }
};

}

}

#endif
