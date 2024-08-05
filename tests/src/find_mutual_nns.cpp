#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "scran_tests/scran_tests.hpp"
#include "knncolle/knncolle.hpp"

#include <random>
#include <vector>
#include <algorithm>

#include "mnncorrect/find_mutual_nns.hpp"
#include "mnncorrect/fuse_nn_results.hpp"

class FindMutualNNsTest : public ::testing::TestWithParam<std::tuple<int, int, int, int> > {
protected:
    static mnncorrect::internal::MnnPairs<int> compute_reference(
        const mnncorrect::internal::NeighborSet<int, double>& left, 
        const mnncorrect::internal::NeighborSet<int, double>& right) 
    {
        size_t nleft = left.size();
        std::set<std::pair<size_t, size_t> > found;
        for (size_t l = 0; l < nleft; ++l) {
            const auto& current = left[l];
            for (const auto& x : current) {
                found.insert(std::pair<size_t, size_t>(l, x.first));
            }
        }

        size_t nright = right.size();
        mnncorrect::internal::MnnPairs<int> output;
        for (size_t r = 0; r < nright; ++r) {
            const auto& current = right[r];

            std::vector<int> holder; 
            for (const auto& x : current) {
                auto it = found.find(std::pair<size_t, size_t>(x.first, r));
                if (it != found.end()) {
                    holder.push_back(x.first);
                }
            }

            if (holder.size()) {
                output.matches[static_cast<int>(r)] = std::move(holder);
            }
        }

        return output;
    }
};

TEST_P(FindMutualNNsTest, Check) {
    auto param = GetParam();
    int nleft = std::get<0>(param);
    int nright = std::get<1>(param);
    int k1 = std::get<2>(param);
    int k2 = std::get<3>(param);

    int ndim = 5;
    auto left = scran_tests::simulate_vector(nleft * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 42 + nleft + nright * 10 + k1 + k2 * 10;
        return sparams;
    }());
    auto right = scran_tests::simulate_vector(nright * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 69 + nleft + nright * 10 + k1 + k2 * 10;
        return sparams;
    }());

    // Reference calculation here.
    auto left_index = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix(ndim, nleft, left.data()));
    auto right_index = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix(ndim, nright, right.data()));
    auto neighbors_of_left = mnncorrect::internal::quick_find_nns(nleft, left.data(), *right_index, k1, /* nthreads = */ 1);
    auto neighbors_of_right = mnncorrect::internal::quick_find_nns(nright, right.data(), *left_index, k2, /* nthreads = */ 1);
    auto ref = compute_reference(neighbors_of_left, neighbors_of_right);

    // Checking that we do have some MNNs.
    size_t np = 0;
    for (const auto& x : ref.matches) {
        np += x.second.size();
    }
    EXPECT_TRUE(np > 0);

    auto output = mnncorrect::internal::find_mutual_nns(neighbors_of_left, neighbors_of_right);
    EXPECT_EQ(output.num_pairs, np);

    EXPECT_EQ(output.matches.size(), ref.matches.size());
    for (const auto& x : output.matches) {
        auto rIt = ref.matches.find(x.first);
        ASSERT_TRUE(rIt != ref.matches.end());
        EXPECT_EQ(rIt->second, x.second);
    }

    // Checking the uniques.
    auto ul = mnncorrect::internal::unique_left(output);
    EXPECT_TRUE(ul.size() && *std::max_element(ul.begin(), ul.end()) < nleft);

    auto ur = mnncorrect::internal::unique_right(output);
    EXPECT_EQ(ur.size(), output.matches.size());
    EXPECT_TRUE(ur.size() && *std::max_element(ur.begin(), ur.end()) < nright);
}

INSTANTIATE_TEST_CASE_P(
    FindMutualNNs,
    FindMutualNNsTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50), // first k
        ::testing::Values(10, 50)  // second k
    )
);


