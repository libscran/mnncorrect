#ifndef MNNCORRECT_CUSTOMORDER_HPP
#define MNNCORRECT_CUSTOMORDER_HPP

#include "utils.hpp"
#include "knncolle/knncolle.hpp"
#include "find_mutual_nns.hpp"
#include "fuse_nn_results.hpp"
#include "correct_target.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace mnncorrect {

template<typename Index, typename Float, class Builder>
class CustomOrder {
public:
    CustomOrder(int nd, std::vector<size_t> no, std::vector<const Float*> b, Float* c, Builder bfun, int k, const int* co) :
        ndim(nd), 
        nobs(std::move(no)), 
        batches(std::move(b)),
        indices(batches.size()),
        builder(bfun),
        num_neighbors(k),
        corrected(c),
        order(co, co + batches.size())
    {
        if (nobs.size() != batches.size()) {
            throw std::runtime_error("length of 'no' and 'b' must be equal");
        }

        if (!nobs.size()) {
            return;
        }

#ifndef MNNCORRECT_CUSTOM_PARALLEL
        #pragma omp parallel for
        for (size_t b = 0; b < nobs.size(); ++b) {
#else
        MNNCORRECT_CUSTOM_PARALLEL(nobs.size(), [&](size_t start, size_t end) -> void {
        for (size_t b = start; b < end; ++b) {
#endif

            indices[b] = bfun(ndim, nobs[b], batches[b]);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
        }
#else
        }
        });
#endif

        // Picking the first batch to be our reference.
        auto first = order[0];
        const size_t rnum = nobs[first];
        const Float* rdata = batches[first];
        std::copy(rdata, rdata + ndim * rnum, corrected);
        ncorrected += rnum;

        if (nobs.size() > 1) {
            auto second = order[1];
            neighbors_target = quick_find_nns(nobs[second], batches[second], indices[first].get(), num_neighbors);
            neighbors_ref = quick_find_nns(rnum, rdata, indices[second].get(), num_neighbors);
        }

        return;
    }

protected:
    template<bool testing = false>
    void update(size_t position) {
        auto latest = order[position];
        size_t lnum = nobs[latest]; 
        const Float* ldata = corrected + ncorrected * ndim;
        ncorrected += lnum;

        ++position;
        if (position == batches.size()) { 
            return;
        }

        // Updating all statistics with the latest batch added to the corrected reference.
        indices[latest] = builder(ndim, lnum, ldata);

        auto next = order[position];
        auto nxdata = batches[next];
        auto nxnum = nobs[next];
        const auto& nxdex = indices[next];
        neighbors_ref.resize(ncorrected);
        neighbors_target.resize(nxnum);

        // Progressively finding the best neighbors across the currently built batches.
        size_t previous_ncorrected = 0;
        for (size_t i = 0; i < position; ++i) {
            auto prev = order[i];
            const auto& prevdex = indices[prev];
            
            if (i) {
#ifndef MNNCORRECT_CUSTOM_PARALLEL
                #pragma omp parallel for
                for (size_t n = 0; n < nxnum; ++n) {
#else
                MNNCORRECT_CUSTOM_PARALLEL(nxnum, [&](size_t start, size_t end) -> void {
                for (size_t n = start; n < end; ++n) {
#endif

                    auto alt = prevdex->find_nearest_neighbors(nxdata + ndim * n, num_neighbors);
                    fuse_nn_results(neighbors_target[n], alt, num_neighbors, static_cast<Index>(previous_ncorrected));

#ifndef MNNCORRECT_CUSTOM_PARALLEL
                }
#else
                }
                });
#endif

            } else {
#ifndef MNNCORRECT_CUSTOM_PARALLEL
                #pragma omp parallel for
                for (size_t n = 0; n < nxnum; ++n) {
#else
                MNNCORRECT_CUSTOM_PARALLEL(nxnum, [&](size_t start, size_t end) -> void {
                for (size_t n = start; n < end; ++n) {
#endif

                    neighbors_target[n] = prevdex->find_nearest_neighbors(nxdata + ndim * n, num_neighbors);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
                }
#else
                }
                });
#endif
            }

            auto prevnum = nobs[prev];
            auto prevdata = corrected + previous_ncorrected * ndim;

#ifndef MNNCORRECT_CUSTOM_PARALLEL
            #pragma omp parallel for
            for (size_t p = 0; p < prevnum; ++p) {
#else
            MNNCORRECT_CUSTOM_PARALLEL(prevnum, [&](size_t start, size_t end) -> void {
            for (size_t p = start; p < end; ++p) {
#endif

                neighbors_ref[previous_ncorrected + p] = nxdex->find_nearest_neighbors(prevdata + ndim * p, num_neighbors);

#ifndef MNNCORRECT_CUSTOM_PARALLEL
            }
#else
            }
            });
#endif

            previous_ncorrected += prevnum;
        }

        return;
    }

public:
    void run(Float nmads, int robust_iterations, double robust_trim) {
        for (size_t i = 1; i < batches.size(); ++i) {
            auto mnns = find_mutual_nns(neighbors_ref, neighbors_target);
            auto tnum = nobs[order[i]];
            auto tdata = batches[order[i]];

            correct_target(
                ndim, 
                ncorrected, 
                corrected, 
                tnum, 
                tdata, 
                mnns,
                builder,
                num_neighbors,
                nmads,
                robust_iterations,
                robust_trim,
                corrected + ncorrected * ndim);

            update(i);
            num_pairs.push_back(mnns.num_pairs);
        }
    }

    const auto& get_num_pairs() const { return num_pairs; }

    const auto& get_order() const { return order; }

protected:
    int ndim;
    std::vector<size_t> nobs;
    std::vector<const Float*> batches;
    std::vector<std::shared_ptr<knncolle::Base<Index, Float> > > indices;

    Builder builder;
    int num_neighbors;
    NeighborSet<Index, Float> neighbors_ref;
    NeighborSet<Index, Float> neighbors_target;

    Float* corrected;
    size_t ncorrected = 0;
    std::vector<int> order;
    std::vector<int> num_pairs;
};

}

#endif
