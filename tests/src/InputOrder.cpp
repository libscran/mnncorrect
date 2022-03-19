#include <gtest/gtest.h>
#include "mnncorrect/InputOrder.hpp"
#include "mnncorrect/find_mutual_nns.hpp"
#include <random>
#include <algorithm>
#include "order_utils.h"

struct InputOrder2 : public mnncorrect::InputOrder<int, double, Builder> {
    InputOrder2(int nd, std::vector<size_t> no, std::vector<const double*> b, double* c, int k) :
        InputOrder<int, double, Builder>(nd, std::move(no), std::move(b), c, Builder(), k) {}

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
        update<true>(latest);
        return;
    }
};

class InputOrderTest : public ::testing::TestWithParam<std::tuple<int, int, std::vector<size_t> > > {
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

TEST_P(InputOrderTest, CheckInitialization) {
    assemble(GetParam());
    InputOrder2 coords(ndim, sizes, ptrs, output.data(), k);

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

TEST_P(InputOrderTest, CheckUpdate) {
    assemble(GetParam());
    InputOrder2 coords(ndim, sizes, ptrs, output.data(), k);

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
    }
}

INSTANTIATE_TEST_CASE_P(
    InputOrder,
    InputOrderTest,
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
