#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "mnncorrect/find_mutual_nns.hpp"
#include "helper_find_mutual_nns.hpp"

class FindMutualNNsTest : public ::testing::TestWithParam<std::tuple<int, int, int, int> > {
protected:
    template<class Param>
    void assemble(Param param) {
        nleft = std::get<0>(param);
        nright = std::get<1>(param);
        k1 = std::get<2>(param);
        k2 = std::get<3>(param);

        // Simulating values.
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist;

        left.resize(nleft * ndim);
        for (auto& l : left) {
            l = dist(rng);
        }

        right.resize(nright * ndim);
        for (auto& r : right) {
            r = dist(rng);
        }
    }

    mnncorrect::MnnPairs<int> compute_reference() {
        knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
        knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());

        std::set<std::pair<size_t, size_t> > found;
        for (size_t l = 0; l < nleft; ++l) {
            auto current = right_index.find_nearest_neighbors(left.data() + l * ndim, k1);
            for (const auto& x : current) {
                found.insert(std::pair<size_t, size_t>(l, x.first));
            }
        }

        mnncorrect::MnnPairs<int> output(nright);
        for (size_t r = 0; r < nright; ++r) {
            auto current = left_index.find_nearest_neighbors(right.data() + r * ndim, k2);
            for (const auto& x : current) {
                auto it = found.find(std::pair<size_t, size_t>(x.first, r));
                if (it != found.end()) {
                    output.matches[r].push_back(x.first);
                }
            }
        }

        return output;
    }

    int ndim = 5;
    size_t nleft, nright;
    std::vector<double> left, right;
    int k1, k2;
};

TEST_P(FindMutualNNsTest, Check) {
    assemble(GetParam());
    auto ref = compute_reference();

    // Checking that we do have some MNNs.
    size_t np = 0;
    for (const auto& x : ref.matches) {
        np += x.size();
    }
    EXPECT_TRUE(np > 0);

    knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
    knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
    auto obs = find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k1, k2);

    EXPECT_EQ(obs.matches.size(), ref.matches.size());
    for (size_t r = 0; r < obs.matches.size(); ++r) {
        EXPECT_EQ(ref.matches[r], obs.matches[r]);
    }
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
