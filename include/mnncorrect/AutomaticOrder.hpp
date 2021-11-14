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
    AutomaticOrder(int nd, std::vector<size_t> no, std::vector<const Float*> b, Float* c, Builder bfun, int ncenters, int k, uint64_t s) :
        ndim(nd), 
        nobs(std::move(no)), 
        batches(std::move(b)),
        builder(bfun),
        corrected(c),
        num_centers(ncenters),
        centers(num_centers * ndim),
        radius(num_centers),
        num_neighbors(k),
        seed(s)
    {
        if (nobs.size() != batches.size()) {
            throw std::runtime_error("length of 'no' and 'b' must be equal");
        }
        if (!nobs.size()) {
            return;
        }

        size_t ntotal = std::accumulate(nobs.begin(), nobs.end(), 0);
        clusters.resize(ntotal);

        // Picking the largest batch to be our reference.
        size_t ref = std::max_element(nobs.begin(), nobs.end()) - nobs.begin();
        const size_t rnum = nobs[ref];
        const Float* rdata = batches[ref];

        std::copy(rdata, rdata + ndim * rnum, corrected);
        ncorrected += rnum;
        order.push_back(ref);

        // Building indices for every other batch. 
        #pragma omp parallel for
        for (size_t b = 0; b < nobs.size(); ++b) {
            if (b != ref) {
                indices[b] = bfun(ndim, nobs[b], batches[b]);
            }
        }

        // Separate loop, avoid race conditions.
        for (size_t b = 0; b < nobs.size(); ++b) {
            if (b != ref) {
                remaining.insert(b);
            }
        }

        return;
    }

protected:
    void update(bool testing = false) {
        size_t lnum = nobs[latest]; 

        auto previous_index = order.back();
        order.push_back(latest);
        num_pairs.push_back(pairings.size());

        auto previous_ncorrected = ncorrected;
        ncorrected += lnum;

        indices[latest].reset(); // freeing some memory early.
        remaining.erase(latest);

        // Adding cluster assignments for the latest batch.
        if (testing || remaining.size()) {
            const Float* ldata = corrected + previous_ncorrected * ndim;
            for (size_t l = 0; l < lnum; ++l) {
                auto best = ref_index->find_nearest_neighbors(ldata + ndim * l, 1);
                clusters[previous_ncorrected + l] = best.front().first;
            }
        }

        return;
    }

    void choose() {
        // Either performing the clustering fresh (for the first batch)
        // or reinitializing with knowledge of the existing cluster info.
        // We bump the seed for every run to make sure we get somewhat different values.
        kmeans::Kmeans<Float, int, Index> clusterer;
        auto new_seed = seed + order.size();

        std::unique_ptr<kmeans::Initializer<> > iptr;
        if (order.size() > 1) {
            auto ptr = new kmeans::Reinitialize<Float, int, Index>;
            ptr->set_seed(new_seed);
            ptr->set_recompute_clusters(false); // relying on an accurate clusters.
            iptr.reset(ptr);
        } else {
            auto ptr = new kmeans::InitializeKmeansPP<Float, int, Index>;
            ptr->set_seed(new_seed);
            iptr.reset(ptr);
        }

        clusterer.run(ndim, ncorrected, corrected, num_centers, centers.data(), clusters.data(), iptr.get());
        median_distance_from_center(ndim, ncorrected, corrected, num_centers, centers.data(), clusters.data(), radius.data());

        NeighborSet<Index, Float> rneighbors(num_centers);
        ref_index.reset(builder(ndim, ncenters, centers.data()));
        pairings.clear();

        for (auto b : remaining) {
            const auto tnum = nobs[b];
            const auto& tindex = indices[b];
            NeighborSet<Index, Float> tneighbors(tnum);

            #pragma omp parallel for
            for (size_t t = 0; t < tnum; ++t) {
                tneighbors[t] = ref_index->find_nearest_neighbors(tdata + ndim * t, 1);
            }
           
            #pragma omp parallel for
            for (size_t r = 0; r < num_centers; ++r) {
                rneighbors[r] = tindex->find_nearest_neighbors(centers.data() + r * ndim, num_neighbors);
            }

            auto tmp = find_mutual_nns(rneighbors, tneighbors);
            if (tmp.size() > pairings.size()) {
                pairings = std::move(tmp);
                target_neighbors = std::move(tneighbors);
                latest = b;
            }
        }

        return;
    }

public:
    void run(Float nmads) {
        while (remaining.size()) {
            choose();
            size_t tnum = 
            const Float* tdata = ;

            correct_target(
                ndim, 
                ncorrected, 
                corrected,
                radius,
                nobs[latest],
                batches[latest],
                pairings, 
                target_neighbors[latest],
                corrected + ncorrected * ndim);

            update();
        }
    }

    const auto& get_order() const { return order; }

    const auto& get_num_pairs() const { return num_pairs; }

protected:
    int ndim;
    std::vector<size_t> nobs;
    std::vector<const Float*> batches;

    int num_centers;
    uint64_t seed;
    std::vector<Float> centers;
    std::vector<Index> clusters;
    std::vector<Float> radius;

    Builder builder;
    std::vector<std::shared_ptr<knncolle::Base<Index, Float> > > indices;
    std::shared_ptr<knncolle::Base<Index, Float> > ref_index;

    int num_neighbors;
    NeighborSet<Index, Float> target_neighbors;
    MnnPairs<Index> pairings;
    size_t latest;

    Float* corrected;
    size_t ncorrected = 0;
    std::vector<int> order;
    std::vector<int> num_pairs;

    std::set<size_t> remaining;
};

}

#endif
