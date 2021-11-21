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
    AutomaticOrder2(int nd, std::vector<size_t> no, std::vector<const double*> b, double* c, int k) :
        AutomaticOrder<int, double, Builder>(nd, std::move(no), std::move(b), c, Builder(), k) {}

    const std::vector<mnncorrect::NeighborSet<int, double> >& get_neighbors_ref () const { 
        return neighbors_ref;
    }
    const std::vector<mnncorrect::NeighborSet<int, double> >& get_neighbors_target () const { 
        return neighbors_target;
    }

    size_t get_ncorrected() const { 
        return ncorrected;
    }

    const std::set<size_t>& get_remaining () const { 
        return remaining; 
    }

    auto test_choose(){
        return choose();        
    }

    void test_update(size_t latest) {
        update(latest, 1000, true);
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

TEST_P(AutomaticOrderTest, CheckInitialization) {
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

    const auto& rneighbors = coords.get_neighbors_ref(); 
    const auto& lneighbors = coords.get_neighbors_target();

    for (size_t b = 0; b < sizes.size(); ++b) {
        if (b == maxed) { 
            continue; 
        }

        EXPECT_EQ(rneighbors[b].size(), ncorrected);
        EXPECT_EQ(rneighbors[b][0].size(), k);
        EXPECT_EQ(lneighbors[b].size(), sizes[b]);
        EXPECT_EQ(lneighbors[b][0].size(), k);
    }
}

TEST_P(AutomaticOrderTest, CheckUpdate) {
    assemble(GetParam());
    AutomaticOrder2 coords(ndim, sizes, ptrs, output.data(), k);
    std::vector<char> used(sizes.size());
    used[coords.get_order()[0]] = true;

    std::mt19937_64 rng(123456);
    std::normal_distribution<> dist;

    for (size_t b = 1; b < sizes.size(); ++b) {
        auto chosen = coords.test_choose();
        EXPECT_FALSE(used[chosen.first]);
        used[chosen.first] = true;
        size_t sofar = coords.get_ncorrected();

        // Check that the MNN pair indices are correct.
        const auto& m = chosen.second.matches;
        EXPECT_TRUE(m.size() > 0);
        for (const auto& x : m) {
            EXPECT_TRUE(x.first < sizes[chosen.first]);
            for (const auto& y : x.second) {
                EXPECT_TRUE(y < coords.get_ncorrected());
            }
        }

        // Applying an update. We mock up some corrected data.
        double* fixed = output.data() + sofar * ndim;
        for (size_t s = 0; s < sizes[chosen.first]; ++s) {
            for (int d = 0; d < ndim; ++d) {
                fixed[s * ndim + d] = dist(rng);
            }
        }
        coords.test_update(chosen.first);

        // Check that the update works as expected.
        const auto& remaining = coords.get_remaining();
        EXPECT_EQ(remaining.size(), sizes.size() - b - 1);
        EXPECT_EQ(sofar + sizes[chosen.first], coords.get_ncorrected());

        const auto& ord = coords.get_order();
        EXPECT_EQ(ord.size(), b + 1);
        EXPECT_EQ(ord.back(), chosen.first);

        const auto& rneighbors = coords.get_neighbors_ref();
        for (auto r : remaining) {
            const auto& rcurrent = rneighbors[r];
            knncolle::VpTreeEuclidean<int, double> target_index(ndim, sizes[r], data[r].data());
            EXPECT_EQ(rcurrent.size(), coords.get_ncorrected());

            for (size_t x = 0; x < coords.get_ncorrected(); ++x) {
                auto naive = target_index.find_nearest_neighbors(output.data() + x * ndim, k);
                const auto& updated = rcurrent[x];
                EXPECT_EQ(naive.size(), updated.size());

                for (size_t i = 0; i < std::min(naive.size(), updated.size()); ++i) {
                    EXPECT_EQ(naive[i].first, updated[i].first);
                    EXPECT_EQ(naive[i].second, updated[i].second);
                }
            }
        }

        knncolle::VpTreeEuclidean<> ref_index(ndim, coords.get_ncorrected(), output.data());
        const auto& tneighbors = coords.get_neighbors_target();
        for (auto r : remaining) {
            const auto& current = data[r];
            const auto& tcurrent = tneighbors[r];
            EXPECT_EQ(tcurrent.size(), sizes[r]);

            for (size_t x = 0; x < sizes[r]; ++x) {
                auto naive = ref_index.find_nearest_neighbors(current.data() + x * ndim, k);
                const auto& updated = tcurrent[x];
                EXPECT_EQ(naive.size(), updated.size());

                for (size_t i = 0; i < std::min(naive.size(), updated.size()); ++i) {
                    EXPECT_EQ(naive[i].first, updated[i].first);
                    EXPECT_EQ(naive[i].second, updated[i].second);
                }
            }
        }
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
