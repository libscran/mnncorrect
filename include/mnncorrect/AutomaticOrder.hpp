#ifndef MNNCORRECT_AUTOMATICORDER_HPP
#define MNNCORRECT_AUTOMATICORDER_HPP

#include "knncolle/knncolle.hpp"

#include "utils.hpp"
#include "find_mutual_nns.hpp"
#include "correct_target.hpp"
#include "IterativeMerger.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace mnncorrect {

template<typename Index, typename Float, class Builder>
class AutomaticOrder : public IterativeMerger<Index, Float> {
public:
    AutomaticOrder(int nd, std::vector<size_t> no, std::vector<const Float*> b, Float* c, int ncenters, uint64_t s, Builder bfun, int k) :
        IterativeMerger<Index, Float>(nd, std::move(no), std::move(b), c, ncenters, s),
        indices(this->nobs.size()),
        builder(std::move(bfun)),
        num_neighbors(k)
    {
        // Picking the largest batch to be our reference.
        size_t ref = std::max_element(this->nobs.begin(), this->nobs.end()) - this->nobs.begin();
        const size_t rnum = this->nobs[ref];
        const Float* rdata = this->batches[ref];

        std::copy(rdata, rdata + this->ndim * rnum, this->corrected);
        this->ncorrected += rnum;
        this->order.push_back(ref);

        // Building indices for every other batch. 
        #pragma omp parallel for
        for (size_t b = 0; b < this->nobs.size(); ++b) {
            if (b != ref) {
                auto ptr = bfun(this->ndim, this->nobs[b], this->batches[b]);
                indices[b] = ptr;
            }
        }

        // Separate loop, avoid race conditions.
        for (size_t b = 0; b < this->nobs.size(); ++b) {
            if (b != ref) {
                remaining.insert(b);
            }
        }

        return;
    }

protected:
    void choose() {
        this->cluster();
        ref_index = builder(this->ndim, this->num_centers, this->centers.data());
        NeighborSet<Index, Float> rneighbors(this->num_centers);
        pairings.clear();

        for (auto b : remaining) {
            const auto tnum = this->nobs[b];
            const auto tdata = this->batches[b];
            const auto& tindex = indices[b];
            std::vector<Index> tneighbors(tnum);

            auto tmp = find_mutual_nns(
                this->centers.data(), 
                tdata, 
                ref_index.get(), 
                tindex.get(), 
                this->num_neighbors, 
                rneighbors, 
                tneighbors.data()
            );

            if (tmp.size() > pairings.size()) {
                pairings = std::move(tmp);
                std::copy(tneighbors.begin(), tneighbors.end(), this->clusters.data() + this->ncorrected);
                latest = b;
            }
        }

        return;
    }

    void update(bool testing = false) {
        size_t lnum = this->nobs[latest]; 

        auto previous_index = this->order.back();
        this->order.push_back(latest);
        this->num_pairs.push_back(pairings.size());

        auto previous_ncorrected = this->ncorrected;
        this->ncorrected += lnum;

        indices[latest].reset(); // freeing some memory early.
        remaining.erase(latest);

        // Adding cluster assignments for the latest batch.
        if (testing || remaining.size()) {
            assign_to_cluster(
                this->ndim, 
                lnum, 
                this->corrected + previous_ncorrected * this->ndim, 
                ref_index.get(), 
                this->clusters.data() + previous_ncorrected
            );
        }

        return;
    }

public:
    void run(int min_mnns) {
        while (remaining.size()) {
            choose();

            correct_target(
                this->ndim,
                this->num_centers,
                this->centers.data(),
                this->radius,
                this->nobs[latest],
                this->batches[latest],
                pairings, 
                this->clusters.data() + this->ncorrected,
                min_mnns,
                this->corrected + this->ncorrected * this->ndim
            );

            update();
        }
    }

protected:
    Builder builder;
    std::vector<std::shared_ptr<knncolle::Base<Index, Float> > > indices;
    std::shared_ptr<knncolle::Base<Index, Float> > ref_index;

    int num_neighbors;
    MnnPairs<Index> pairings;
    size_t latest;

    std::set<size_t> remaining;
};

}

#endif
