#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/CustomOrder.hpp"
#include "mnncorrect/find_mutual_nns.hpp"
#include <random>
#include <algorithm>
#include "order_utils.h"

struct CustomOrder2 : public mnncorrect::CustomOrder<int, double, Builder> {
    CustomOrder2(int nd, std::vector<size_t> no, std::vector<const double*> b, double* c, int k, const int* co, int nthreads = 1) :
        CustomOrder<int, double, Builder>(nd, std::move(no), std::move(b), c, Builder(), k, co, nthreads) {}

    const mnncorrect::NeighborSet<int, double>& get_neighbors_ref () const { 
        return neighbors_ref;
    }
    const mnncorrect::NeighborSet<int, double>& get_neighbors_target () const { 
        return neighbors_target;
    }

    size_t get_ncorrected() const { 
        return ncorrected;
    }

    void test_update(size_t latest) {
        update(latest);
        return;
    }
};

class CustomOrderTest : public ::testing::TestWithParam<std::tuple<int, int, std::vector<size_t> > > {
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

TEST_P(CustomOrderTest, CheckInitializationSimple) {
    assemble(GetParam());

    std::vector<int> forward(sizes.size());
    std::iota(forward.begin(), forward.end(), 0);
    CustomOrder2 coords(ndim, sizes, ptrs, output.data(), k, forward.data());

    size_t ncorrected = coords.get_ncorrected();
    EXPECT_EQ(ncorrected, sizes[0]);
    EXPECT_EQ(std::vector<double>(output.begin(), output.begin() + ncorrected * ndim), data[0]);

    const auto& rneighbors = coords.get_neighbors_ref();
    EXPECT_EQ(rneighbors.size(), ncorrected);
    EXPECT_EQ(rneighbors[0].size(), k);

    const auto& lneighbors = coords.get_neighbors_target();
    EXPECT_EQ(lneighbors.size(), sizes[1]);
    EXPECT_EQ(lneighbors[0].size(), k);
}

TEST_P(CustomOrderTest, CheckUpdateSimple) {
    assemble(GetParam());

    std::vector<int> forward(sizes.size());
    std::iota(forward.begin(), forward.end(), 0);
    CustomOrder2 coords(ndim, sizes, ptrs, output.data(), k, forward.data());

    std::vector<double> par_output(output.size());
    CustomOrder2 par_coords(ndim, sizes, ptrs, par_output.data(), k, forward.data(), /* nthreads = */ 3);

    std::mt19937_64 rng(123456);
    std::normal_distribution<> dist;

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto mnns = mnncorrect::find_mutual_nns(coords.get_neighbors_ref(), coords.get_neighbors_target());

        // Check that the MNN pair indices are correct.
        const auto& m = mnns.matches;
        EXPECT_TRUE(m.size() > 0);
        for (const auto& x : m) {
            EXPECT_TRUE(x.first < sizes[b]);
            for (const auto& y : x.second) {
                EXPECT_TRUE(y < coords.get_ncorrected());
            }
        }

        // Applying an update. We mock up some corrected data so that the builders work correctly.
        size_t sofar = coords.get_ncorrected();
        double* fixed = output.data() + sofar * ndim;
        for (size_t s = 0; s < sizes[b]; ++s) {
            for (int d = 0; d < ndim; ++d) {
                fixed[s * ndim + d] = dist(rng);
            }
        }
        coords.test_update(b);

        // Check that the update works as expected.
        EXPECT_EQ(sofar + sizes[b], coords.get_ncorrected());

        auto next = b + 1;
        if (next != sizes.size()) {
            const auto& rneighbors = coords.get_neighbors_ref();
            knncolle::VpTreeEuclidean<int, double> target_index(ndim, sizes[next], data[next].data());

            for (size_t x = 0; x < coords.get_ncorrected(); ++x) {
                auto naive = target_index.find_nearest_neighbors(output.data() + x * ndim, k);
                const auto& updated = rneighbors[x];
                compare_to_naive(naive, updated);
            }

            const auto& tneighbors = coords.get_neighbors_target();
            const auto& tdata = data[next];
            EXPECT_EQ(tneighbors.size(), sizes[next]);
            knncolle::VpTreeEuclidean<int, double> ref_index(ndim, coords.get_ncorrected(), output.data());

            for (size_t x = 0; x < sizes[next]; ++x) {
                auto naive = ref_index.find_nearest_neighbors(tdata.data() + x * ndim, k);
                const auto& updated = tneighbors[x];
                compare_to_naive(naive, updated);
            }
        }

        // Doing the same for the parallelized run.
        double* par_fixed = par_output.data() + sofar * ndim;
        std::copy(fixed, fixed + sizes[b] * ndim, par_fixed);
        par_coords.test_update(b);
    }

    // Same results when run in parallel.
    EXPECT_EQ(output, par_output);
}

TEST_P(CustomOrderTest, CheckInitializationReverse) {
    assemble(GetParam());

    // Checking it all works in reverse.
    std::vector<int> reverse(sizes.size());
    std::iota(reverse.begin(), reverse.end(), 0);
    std::reverse(reverse.begin(), reverse.end());
    CustomOrder2 coords(ndim, sizes, ptrs, output.data(), k, reverse.data());

    size_t ncorrected = coords.get_ncorrected();
    EXPECT_EQ(ncorrected, sizes.back());
    EXPECT_EQ(std::vector<double>(output.begin(), output.begin() + ncorrected * ndim), data.back());

    const auto& rneighbors = coords.get_neighbors_ref();
    EXPECT_EQ(rneighbors.size(), ncorrected);
    EXPECT_EQ(rneighbors[0].size(), k);

    const auto& lneighbors = coords.get_neighbors_target();
    EXPECT_EQ(lneighbors.size(), sizes[reverse.size() - 2]);
    EXPECT_EQ(lneighbors[0].size(), k);
}

TEST_P(CustomOrderTest, CheckUpdateReverse) {
    assemble(GetParam());

    // Checking it all works in reverse.
    std::vector<int> reverse(sizes.size());
    std::iota(reverse.begin(), reverse.end(), 0);
    std::reverse(reverse.begin(), reverse.end());
    CustomOrder2 coords(ndim, sizes, ptrs, output.data(), k, reverse.data());

    std::mt19937_64 rng(654321);
    std::normal_distribution<> dist;

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto actual = sizes.size() - b - 1;
        auto mnns = mnncorrect::find_mutual_nns(coords.get_neighbors_ref(), coords.get_neighbors_target());

        // Check that the MNN pair indices are correct.
        const auto& m = mnns.matches;
        EXPECT_TRUE(m.size() > 0);
        for (const auto& x : m) {
            EXPECT_TRUE(x.first < sizes[actual]);
            for (const auto& y : x.second) {
                EXPECT_TRUE(y < coords.get_ncorrected());
            }
        }

        // Applying an update. We mock up some corrected data so that the builders work correctly.
        size_t sofar = coords.get_ncorrected();
        double* fixed = output.data() + sofar * ndim;
        for (size_t s = 0; s < sizes[actual]; ++s) {
            for (int d = 0; d < ndim; ++d) {
                fixed[s * ndim + d] = dist(rng);
            }
        }
        coords.test_update(b);

        // Check that the update works as expected.
        EXPECT_EQ(sofar + sizes[actual], coords.get_ncorrected());

        if (actual > 0) {
            auto next = actual - 1;
            const auto& rneighbors = coords.get_neighbors_ref();
            knncolle::VpTreeEuclidean<int, double> target_index(ndim, sizes[next], data[next].data());

            for (size_t x = 0; x < coords.get_ncorrected(); ++x) {
                auto naive = target_index.find_nearest_neighbors(output.data() + x * ndim, k);
                const auto& updated = rneighbors[x];
                compare_to_naive(naive, updated);
            }

            const auto& tneighbors = coords.get_neighbors_target();
            const auto& tdata = data[next];
            EXPECT_EQ(tneighbors.size(), sizes[next]);
            knncolle::VpTreeEuclidean<int, double> ref_index(ndim, coords.get_ncorrected(), output.data());

            for (size_t x = 0; x < sizes[next]; ++x) {
                auto naive = ref_index.find_nearest_neighbors(tdata.data() + x * ndim, k);
                const auto& updated = tneighbors[x];
                compare_to_naive(naive, updated);
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    CustomOrder,
    CustomOrderTest,
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
