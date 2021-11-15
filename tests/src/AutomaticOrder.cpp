#include <gtest/gtest.h>
#include "mnncorrect/AutomaticOrder.hpp"
#include <random>
#include <algorithm>

struct Builder {
    std::shared_ptr<knncolle::Base<int, double> > operator()(int ndim, size_t nobs, const double* stuff) const {
        return std::shared_ptr<knncolle::Base<int, double> >(new knncolle::VpTreeEuclidean<int, double>(ndim, nobs, stuff));
    }
};

struct AutomaticOrder2 : public mnncorrect::AutomaticOrder<int, double, Builder> {
    static constexpr int num_centers = 50;

    AutomaticOrder2(int nd, std::vector<size_t> no, std::vector<const double*> b, double* c, int k) :
        AutomaticOrder<int, double, Builder>(nd, std::move(no), std::move(b), c, num_centers, 5678u, Builder(), k)
    {
        std::fill(clusters.begin(), clusters.end(), -1); // fail fast if this isn't properly filled. 
    }

    const std::vector<int>& get_clusters() const { 
        return this->clusters;
    }

    const std::vector<double>& get_centers() const { 
        return this->centers;
    }

    const mnncorrect::MnnPairs<int>& get_pairings() const { 
        return pairings;
    }

    size_t get_ncorrected() const { 
        return ncorrected;
    }

    size_t get_latest() const { 
        return latest;
    }

    const std::set<size_t>& get_remaining () const { 
        return remaining; 
    }

    auto test_choose(){
        return choose();        
    }

    void test_update() {
        update(true);
        return;
    }

    void reset_clusters(size_t from, size_t to) {
        std::fill(clusters.begin() + from, clusters.begin() + to, -1);
        return;
    }
};

class AutomaticOrderTest : public ::testing::TestWithParam<std::tuple<int, int, std::vector<size_t> > > {
protected:
    template<class Param>
    void assemble(Param param) {
        // Simulating values.
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist;

        ndim = std::get<0>(param);
        k = std::get<1>(param);
        sizes = std::get<2>(param);

        data.resize(sizes.size());
        ptrs.resize(sizes.size());
        for (size_t b = 0; b < sizes.size(); ++b) {
            for (size_t s = 0; s < sizes[b]; ++s) {
                for (int d = 0; d < ndim; ++d) {
                    data[b].push_back(dist(rng));
                }
            }
            ptrs[b] = data[b].data();
        }

        size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
        output.resize(total_size * ndim);
        return;
    }

    int ndim, k;
    std::vector<size_t> sizes;
    std::vector<std::vector<double> > data;
    std::vector<const double*> ptrs;
    std::vector<double> output;
};

TEST_P(AutomaticOrderTest, Initialization) {
    assemble(GetParam());
    AutomaticOrder2 coords(ndim, sizes, ptrs, output.data(), k);

    size_t maxed = std::max_element(sizes.begin(), sizes.end()) - sizes.begin();
    const auto& ord = coords.get_order();
    EXPECT_EQ(ord.size(), 1);
    EXPECT_EQ(ord[0], maxed);

    size_t ncorrected = coords.get_ncorrected();
    EXPECT_EQ(ncorrected, sizes[maxed]);
    EXPECT_EQ(std::vector<double>(output.begin(), output.begin() + ncorrected * ndim), data[maxed]);
    EXPECT_EQ(coords.get_remaining().size(), sizes.size() - 1);
}

TEST_P(AutomaticOrderTest, ChoiceAndUpdate) {
    assemble(GetParam());
    AutomaticOrder2 coords(ndim, sizes, ptrs, output.data(), k);
    std::vector<char> used(sizes.size());
    used[coords.get_order()[0]] = true;

    std::mt19937_64 rng(123456);
    std::normal_distribution<> dist;

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto copy_centers = coords.get_centers();
        coords.test_choose();

        auto chosen = coords.get_latest();
        EXPECT_FALSE(used[chosen]);
        used[chosen] = true;

        size_t chosen_size = sizes[chosen];
        size_t sofar = coords.get_ncorrected();

        // Check that the MNN pair indices are correct.
        const auto& pairings = coords.get_pairings();
        EXPECT_TRUE(pairings.size() > 0);
        const auto& left = pairings.ref;
        for (auto l : left) {
            EXPECT_TRUE(l < AutomaticOrder2::num_centers);
        }

        const auto& right = pairings.target;
        for (auto r : right) {
            EXPECT_TRUE(r < chosen_size);
        }

        const auto& clusters = coords.get_clusters();
        for (size_t r = 0 ; r < chosen_size; ++r) {
            EXPECT_TRUE(clusters[r + sofar] >= 0); 
        }

        EXPECT_NE(copy_centers, coords.get_centers());

        // Applying an update. We mock up some corrected data.
        double* fixed = output.data() + sofar * ndim;
        for (size_t s = 0; s < sizes[chosen]; ++s) {
            for (int d = 0; d < ndim; ++d) {
                fixed[s * ndim + d] = dist(rng);
            }
        }

        std::vector<int> copy_clusters_prev(clusters.begin(), clusters.begin() + sofar);
        std::vector<int> copy_clusters_now(clusters.begin() + sofar, clusters.begin() + sofar + chosen_size);
        coords.reset_clusters(sofar, sofar + chosen_size);

        coords.test_update();

        // Check that the update works as expected.
        const auto& remaining = coords.get_remaining();
        EXPECT_EQ(remaining.size(), sizes.size() - b - 1);
        EXPECT_EQ(sofar + sizes[chosen], coords.get_ncorrected());

        const auto& ord = coords.get_order();
        EXPECT_EQ(ord.size(), b + 1);
        EXPECT_EQ(ord.back(), chosen);

        for (size_t r = 0 ; r < chosen_size; ++r) {
            EXPECT_TRUE(clusters[r + sofar] >= 0); 
        }

        EXPECT_EQ(copy_clusters_prev, std::vector<int>(clusters.begin(), clusters.begin() + sofar)); // should be unchanged.
        EXPECT_NE(copy_clusters_now, std::vector<int>(clusters.begin() + sofar, clusters.begin() + sofar + chosen_size)); 
    }

    EXPECT_EQ(sizes.size(), coords.get_num_pairs().size() + 1);
    for (auto np : coords.get_num_pairs()) {
        EXPECT_TRUE(np > 0);
    }
}

INSTANTIATE_TEST_CASE_P(
    AutomaticOrder,
    AutomaticOrderTest,
    ::testing::Combine(
        ::testing::Values(5), // Number of dimensions
        ::testing::Values(1, 5, 10), // Number of neighbors
        ::testing::Values(
            std::vector<size_t>{10, 20},        
            std::vector<size_t>{10, 20, 30}, 
            std::vector<size_t>{100, 50, 80}, 
            std::vector<size_t>{50, 30, 100, 90} 
        )
    )
);
