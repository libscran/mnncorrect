#include <gtest/gtest.h>
#include "mnncorrect/find_mutual_nns.hpp"
#include "mnncorrect/determine_limit.hpp"
#include "knncolle/knncolle.hpp"
#include "aarand/aarand.hpp"
#include <random>

class DetermineLimitTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    template<class Param>
    void assemble(Param param) {
        nleft = std::get<0>(param);
        nright = std::get<1>(param);
        k = std::get<2>(param);

        std::mt19937_64 rng(nleft * nright * k);

        left.resize(nleft * ndim);
        for (auto& l : left) {
            l = aarand::standard_normal(rng).first;
        }

        right.resize(nright * ndim);
        for (auto& r : right) {
            r = aarand::standard_normal(rng).first + 5; // throw in a batch effect.
        }
    }

    int ndim = 5, k;
    size_t nleft, nright;
    std::vector<double> left, right;
};

TEST_P(DetermineLimitTest, AverageBatch) {
    assemble(GetParam());

    knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
    knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
    auto pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);
    auto averaged = mnncorrect::average_batch_vector(ndim, nleft, left.data(), nright, right.data(), pairings);
    
    // Reference calculation.
    auto right_mnn = mnncorrect::unique(pairings.right);
    auto map = mnncorrect::invert_index(nright, right_mnn);
    std::vector<double> per_mnn(right_mnn.size() * ndim), per_weight(right_mnn.size());

    for (size_t p = 0; p < pairings.size(); ++p) {
        const double* left_ptr = left.data() + pairings.left[p] * ndim;
        const double* right_ptr = right.data() + pairings.right[p] * ndim;

        size_t o = map[pairings.right[p]];
        ++per_weight[o];
        double* out_ptr = per_mnn.data() + o * ndim;
        for (int d = 0; d < ndim; ++d) {
            out_ptr[d] += right_ptr[d] - left_ptr[d];
        }
    }

    std::vector<double> reference(ndim);
    for (size_t o = 0; o < per_weight.size(); ++o) {
        const double* ptr = per_mnn.data() + o * ndim;
        for (int d = 0; d < ndim; ++d) {
            reference[d] += ptr[d] / per_weight[o];
        }
    }

    double l2norm = 0;
    for (size_t r = 0; r < reference.size(); ++r) {
        l2norm += reference[r] * reference[r];
    }
    l2norm = std::sqrt(l2norm);

    for (size_t r = 0; r < reference.size(); ++r) {
        double val = reference[r] / l2norm;
        EXPECT_FLOAT_EQ(val, averaged[r]);
    }
}

TEST_P(DetermineLimitTest, LimitByBatch) {
    assemble(GetParam());

    knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
    knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
    auto pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);
    auto averaged = mnncorrect::average_batch_vector(ndim, nleft, left.data(), nright, right.data(), pairings);

    auto right_mnn = mnncorrect::unique(pairings.right);
    auto limit = mnncorrect::limit_from_batch_vector(ndim, nright, right.data(), averaged, right_mnn);
    EXPECT_TRUE(limit > 0);

    // Let's assume that we have a MAD ~= 1, based on the fact 
    // that we simulate from a standard Normal distribution.

    EXPECT_TRUE(limit > 3); // MNN median should be on one side, so the threshold should be at least 3 MADs away. 
    EXPECT_TRUE(limit < 6); // Even if the MNNs are stuck on the extreme edge of the distribution, the threshold should not be more than 6 MADs away.
}

TEST_P(DetermineLimitTest, LimitByClosest) {
    assemble(GetParam());

    knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
    knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
    auto pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);

    auto right_mnn = mnncorrect::unique(pairings.right);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto builder = [](int nd, size_t no, const double* d) -> auto { return std::shared_ptr<knncolle::Base<> >(new knncolle::VpTreeEuclidean<>(nd, no, d)); };
    auto self_mnn = mnncorrect::identify_closest_mnn(ndim, nright, right.data(), right_mnn, builder, k, buffer.data());

    double limit = mnncorrect::limit_from_closest_distances(self_mnn);

    // Check that everything is reasonable. The expected values are more
    // difficult to derive than in the batch vector approach, but the figures
    // should be in the same ballpark, so we'll just re-use it.
    EXPECT_TRUE(limit < 6);
    EXPECT_TRUE(limit > 3); 
}

INSTANTIATE_TEST_CASE_P(
    DetermineLimit,
    DetermineLimitTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);

TEST(MoreDetermineLimitTest, LimitByClosest) {
    mnncorrect::NeighborSet<int, double> closest(2);

    closest[0] = std::vector<std::pair<int, double> >{
        std::make_pair(0, 0.1),
        std::make_pair(0, 0.5),
        std::make_pair(0, 0.2),
        std::make_pair(0, 0.8)
    };
    closest[1] = std::vector<std::pair<int, double> >{
        std::make_pair(0, 0.7),
        std::make_pair(0, 0.3),
        std::make_pair(0, 0.5),
        std::make_pair(0, 0.1)
    };

    double limit = mnncorrect::limit_from_closest_distances(closest);

    // Should be the same as:
    // x <- c(0.1, 0.1, 0.2, 0.3, 0.5, 0.5, 0.7, 0.8)
    // med <- median(x)
    // diff <- med - x
    // mad <- median(abs(diff[diff > 0]))
    // med + 3 * mad * 1.4826
    EXPECT_FLOAT_EQ(limit, 1.51195);
}
