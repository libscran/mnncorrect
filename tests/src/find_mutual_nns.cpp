#include <gtest/gtest.h>
#include "mnncorrect/find_mutual_nns.hpp"
#include <random>
#include <vector>

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

    mnncorrect::MnnPairs<size_t> compute_reference() {
        knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
        knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());

        std::set<std::pair<size_t, size_t> > found;
        for (size_t l = 0; l < nleft; ++l) {
            auto current = right_index.find_nearest_neighbors(left.data() + l * ndim, k1);
            for (const auto& x : current) {
                found.insert(std::pair<size_t, size_t>(l, x.first));
            }
        }

        mnncorrect::MnnPairs<size_t> output;
        for (size_t r = 0; r < nright; ++r) {
            auto current = left_index.find_nearest_neighbors(right.data() + r * ndim, k2);
            for (const auto& x : current) {
                auto it = found.find(std::pair<size_t, size_t>(x.first, r));
                if (it != found.end()) {
                    output.left.push_back(x.first);
                    output.right.push_back(r);
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
    EXPECT_TRUE(ref.left.size() > 0);
    EXPECT_TRUE(ref.right.size() > 0);

    knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
    knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
    auto obs = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k1, k2);

    EXPECT_EQ(
        std::vector<size_t>(ref.left.begin(), ref.left.end()), 
        std::vector<size_t>(obs.left.begin(), obs.left.end())
    );
    EXPECT_EQ(
        std::vector<size_t>(ref.right.begin(), ref.right.end()), 
        std::vector<size_t>(obs.right.begin(), obs.right.end())
    );
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
