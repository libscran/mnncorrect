#ifndef MNNCORRECT_ORDERING_HPP
#define MNNCORRECT_ORDERING_HPP

#include "utils.hpp"

#include <vector>
#include <memory>

#include "kmeans/Reinitialize.hpp"
#include "kmeans/Kmeans.hpp"

namespace mnncorrect {

template<typename Index, typename Float>
void median_distance_from_center(int ndim, size_t nobs, const Float* data, size_t ncenters, const Float* centers, const Index* clusters, Float* output) {
    std::vector<std::vector<Float> > collected(ncenters);
    for (size_t o = 0; o < nobs; ++o) {
        auto rptr = data + o * ndim;
        auto cptr = centers + clusters[o] * ndim;

        Float dist = 0;
        for (int d = 0; d < ndim; ++d) {
            Float diff = rptr[d] - cptr[d];
            dist += diff * diff;
        }

        collected[clusters[o]].push_back(std::sqrt(dist));
    }

    #pragma omp parallel for
    for (size_t r = 0; r < ncenters; ++r) {
        auto& current = collected[r];
        if (current.size()) {
            output[r] = median(current.size(), current.data());
        } else {
            output[r] = 0; // shouldn't be possible, but whatever, just in case.
        }
    }

    return;
}

template<typename Index, typename Float, class Searcher>
void assign_to_cluster(int ndim, size_t nobs, const Float* data, const Searcher& ref_index, Index* output) {
    #pragma omp parallel for
    for (size_t l = 0; l < nobs; ++l) {
        auto best = ref_index->find_nearest_neighbors(data + ndim * l, 1);
        output[l] = best.front().first;
    }
    return;
}

template<typename Index, typename Float>
class IterativeMerger {
public:
    IterativeMerger(int nd, std::vector<size_t> no, std::vector<const Float*> b, Float* c, int ncenters, uint64_t s) :
        ndim(nd), 
        nobs(std::move(no)), 
        batches(std::move(b)),
        corrected(c),
        num_centers(ncenters),
        centers(num_centers * ndim),
        radius(num_centers),
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
        return;
    }

public:
    const auto& get_order() const { return order; }

    const auto& get_num_pairs() const { return num_pairs; }

protected:
    void cluster() {
        // We bump the seed for every run to make sure we get somewhat different values.
        auto new_seed = seed + order.size();

        // Either performing the clustering fresh (for the first batch)
        // or reinitializing with knowledge of the existing cluster info.
        bool initial = (order.size() == 1);

        kmeans::Kmeans<Float, int, Index> clusterer;

        std::unique_ptr<kmeans::Initialize<> > iptr;
        if (!initial) {
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
        return;
    }

protected:
    int ndim;
    std::vector<size_t> nobs;
    std::vector<const Float*> batches;

    int num_centers;
    uint64_t seed;
    std::vector<Float> centers;
    std::vector<Index> clusters;
    std::vector<Float> radius;

    Float* corrected;
    size_t ncorrected = 0;

    std::vector<int> order;
    std::vector<int> num_pairs;
};

}

#endif
