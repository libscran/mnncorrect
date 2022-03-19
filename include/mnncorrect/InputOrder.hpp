#ifndef MNNCORRECT_INPUTORDER_HPP
#define MNNCORRECT_INPUTORDER_HPP

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
class InputOrder {
public:
    InputOrder(int nd, std::vector<size_t> no, std::vector<const Float*> b, Float* c, Builder bfun, int k) :
        ndim(nd), 
        nobs(std::move(no)), 
        batches(std::move(b)),
        indices(batches.size()),
        builder(bfun),
        num_neighbors(k),
        corrected(c)
    {
        if (nobs.size() != batches.size()) {
            throw std::runtime_error("length of 'no' and 'b' must be equal");
        }

        if (!nobs.size()) {
            return;
        }

        #pragma omp parallel for
        for (size_t b = 0; b < nobs.size(); ++b) {
            indices[b] = bfun(ndim, nobs[b], batches[b]);
        }

        // Picking the first batch to be our reference.
        const size_t rnum = nobs[0];
        const Float* rdata = batches[0];
        std::copy(rdata, rdata + ndim * rnum, corrected);
        ncorrected += rnum;

        if (nobs.size() > 1) {
            neighbors_target = quick_find_nns(nobs[1], batches[1], indices[0].get(), num_neighbors);
            neighbors_ref = quick_find_nns(rnum, rdata, indices[1].get(), num_neighbors);
        }

        return;
    }

protected:
    template<bool testing = false>
    void update(size_t latest) {
        size_t lnum = nobs[latest]; 
        const Float* ldata = corrected + ncorrected * ndim;
        ncorrected += lnum;

        auto next = latest + 1;
        if constexpr(!testing) { // testing = true for correct building of all indices.
            if (next == batches.size()) { 
                return;
            }
        }

        // Updating all statistics with the latest batch added to the corrected reference.
        indices[latest] = builder(ndim, lnum, ldata);
        const auto& lindex = indices[latest];

        if (next < batches.size()) {
            auto nxdata = batches[next];
            auto nxnum = nobs[next];
            const auto& nxdex = indices[next];
            neighbors_ref.resize(ncorrected);
            neighbors_target.resize(nxnum);

            // Progressively finding the best neighbors across the currently built batches.
            size_t previous_ncorrected = 0;
            for (size_t prev = 0; prev < next; ++prev) {
                const auto& prevdex = indices[prev];
                
                if (prev) {
                    #pragma omp parallel for
                    for (size_t n = 0; n < nxnum; ++n) {
                        auto alt = prevdex->find_nearest_neighbors(nxdata + ndim * n, num_neighbors);
                        fuse_nn_results(neighbors_target[n], alt, num_neighbors, static_cast<Index>(previous_ncorrected));
                    }
                } else {
                    #pragma omp parallel for
                    for (size_t n = 0; n < nxnum; ++n) {
                        neighbors_target[n] = prevdex->find_nearest_neighbors(nxdata + ndim * n, num_neighbors);
                    }
                }

                auto prevnum = nobs[prev];
                auto prevdata = corrected + previous_ncorrected * ndim;

                #pragma omp parallel for
                for (size_t p = 0; p < prevnum; ++p) {
                    neighbors_ref[previous_ncorrected + p] = nxdex->find_nearest_neighbors(prevdata + ndim * p, num_neighbors);
                }

                previous_ncorrected += prevnum;
            }
        }

        return;
    }

public:
    void run(Float nmads, int robust_iterations, double robust_trim) {
        for (size_t i = 1; i < batches.size(); ++i) {
            auto mnns = find_mutual_nns(neighbors_ref, neighbors_target);
            auto tnum = nobs[i];
            auto tdata = batches[i];

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
    std::vector<int> num_pairs;
};

}

#endif
