#ifndef MNNCORRECT_AUTOMATICORDER_HPP
#define MNNCORRECT_AUTOMATICORDER_HPP

#include "utils.hpp"
#include "knncolle/knncolle.hpp"
#include "find_mutual_nns.hpp"
#include "correct_target.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace mnncorrect {

template<typename Index, typename Float, class Builder>
class AutomaticOrder {
public:
    AutomaticOrder(int nd, std::vector<size_t> no, std::vector<const Float*> b, Float* c, Builder bfun, int k) :
        ndim(nd), 
        nobs(std::move(no)), 
        batches(std::move(b)),
        builder(bfun),
        num_neighbors(k),
        indices(batches.size()),
        neighbors_ref(batches.size()), 
        neighbors_target(batches.size()), 
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

        // Picking the largest batch to be our reference.
        size_t ref = std::max_element(nobs.begin(), nobs.end()) - nobs.begin();
        const size_t rnum = nobs[ref];
        const Float* rdata = batches[ref];
        std::copy(rdata, rdata + ndim * rnum, corrected);
        ncorrected += rnum;
        order.push_back(ref);

        for (size_t b = 0; b < nobs.size(); ++b) {
            if (b == ref) {
                continue;
            }
            remaining.insert(b);
            neighbors_target[b] = find_nns(nobs[b], batches[b], indices[ref].get(), num_neighbors);
            neighbors_ref[b] = find_nns(rnum, rdata, indices[b].get(), num_neighbors);
        }
        return;
    }

protected:
    void update(size_t latest, size_t npairs, bool testing=false) {
        size_t lnum = nobs[latest]; 
        const Float* ldata = corrected + ncorrected * ndim;

        order.push_back(latest);
        num_pairs.push_back(npairs);
        auto previous_ncorrected = ncorrected;
        ncorrected += lnum;

        remaining.erase(latest);
        if (!testing && remaining.empty()) {
            return;
        }

        // Updating all statistics with the latest batch added to the corrected reference.
        indices[latest] = builder(ndim, lnum, ldata);
        const auto& lindex = indices[latest];

        for (auto b : remaining){
            auto& rneighbors = neighbors_ref[b];
            rneighbors.resize(ncorrected);
            const auto& tindex = indices[b];

            #pragma omp parallel for
            for (size_t l = 0; l < lnum; ++l) {
                rneighbors[previous_ncorrected + l] = tindex->find_nearest_neighbors(ldata + ndim * l, num_neighbors);
            }

            const size_t tnum = nobs[b];
            const Float* tdata = batches[b];
            auto& tneighbors = neighbors_target[b];

            #pragma omp parallel for
            for (size_t t = 0; t < tnum; ++t) {
                auto& current = tneighbors[t];
                auto last = current;
                auto alt = lindex->find_nearest_neighbors(tdata + ndim * t, num_neighbors);

                current.clear();
                auto lIt = last.begin(), aIt = alt.begin();
                while (current.size() < num_neighbors) {
                    if (lIt != last.end() && aIt != alt.end()) {
                        if (lIt->second > aIt->second) {
                            current.push_back(*aIt);
                            current.back().first += previous_ncorrected;
                            ++aIt;
                        } else {
                            current.push_back(*lIt);
                            ++lIt;
                        }
                    } else if (lIt != last.end()) {
                        current.push_back(*lIt);
                        ++lIt;
                    } else if (aIt != alt.end()) {
                        current.push_back(*aIt);
                        current.back().first += previous_ncorrected;
                        ++aIt;
                    } else {
                        break;
                    }
                }
            }
        }

        return;
    }

    std::pair<size_t, MnnPairs<Index> > choose() const {
        MnnPairs<Index> output;
        size_t chosen = 0;
        for (auto b : remaining) {
            auto tmp = find_mutual_nns(neighbors_ref[b], neighbors_target[b]);
            if (tmp.size() > output.size()) {
                output = std::move(tmp);
                chosen = b;
            }
        }
        return std::pair<size_t, MnnPairs<Index> >(chosen, std::move(output));
    }

public:
    void run(Float nmads) {
        while (remaining.size()) {
            auto output = choose();
            auto target = output.first;
            auto tnum = nobs[target];
            auto tdata = batches[target];

            correct_target(
                ndim, 
                ncorrected, 
                corrected, 
                tnum, 
                tdata, 
                output.second, 
                num_neighbors,
                nmads,
                corrected + ncorrected * ndim);

            update(output.first, output.second.size());
        }
    }

    const auto& get_order() const { return order; }

    const auto& get_num_pairs() const { return num_pairs; }

protected:
    int ndim;
    std::vector<size_t> nobs;
    std::vector<const Float*> batches;
    std::vector<std::shared_ptr<knncolle::Base<Index, Float> > > indices;

    Builder builder;
    int num_neighbors;
    std::vector<NeighborSet<Index, Float> > neighbors_ref;
    std::vector<NeighborSet<Index, Float> > neighbors_target;

    Float* corrected;
    size_t ncorrected = 0;
    std::vector<int> order;
    std::vector<int> num_pairs;

    std::set<size_t> remaining;
};

}

#endif
