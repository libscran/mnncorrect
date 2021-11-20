#include <gtest/gtest.h>
#include "mnncorrect/find_mutual_nns.hpp"
#include "mnncorrect/determine_limits.hpp"
#include "mnncorrect/correct_target.hpp"
#include "aarand/aarand.hpp"
#include "knncolle/knncolle.hpp"
#include <random>

template<typename Index, typename Float>
NeighborSet<Index, Float> identify_closest_mnn(int ndim, size_t nobs, const Float* data, const std::vector<Index>& in_mnn, int k, Float* buffer) {
    typedef knncolle::Base<Index, Float> knncolleBase;
    auto builder = [](int nd, size_t no, const Float* d) -> auto { 
        return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d));
    };
    return mnncorrect::identify_closest_mnn(ndim, nobs, data, in_mnn, builder, k, buffer);
}

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

TEST_P(DetermineLimitTest, LimitByClosest) {
    assemble(GetParam());

    knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
    knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
    auto pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);

    auto right_mnn = mnncorrect::unique(pairings.right);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

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

class CorrectTargetTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
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

        // Setting up the values for a reasonable comparison.
        knncolle::VpTreeEuclidean<> left_index(ndim, nleft, left.data());
        knncolle::VpTreeEuclidean<> right_index(ndim, nright, right.data());
        pairings = mnncorrect::find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);
    }

    int ndim = 5, k;
    size_t nleft, nright;
    std::vector<double> left, right;
    mnncorrect::MnnPairs<int> pairings;
};

template<typename Index, typename Float>
void correct_target(int ndim, size_t nref, const Float* ref, size_t ntarget, const Float* target, const MnnPairs<Index>& pairings, int k, Float nmads, Float* output) {
    typedef knncolle::Base<Index, Float> knncolleBase;
    auto builder = [](int nd, size_t no, const Float* d) -> auto { 
        return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d)); 
    };
    mnncorrect::correct_target(ndim, nref, ref, ntarget, target, pairings, builder, k, nmads, output);
    return;
}

TEST_P(CorrectTargetTest, CenterOfMass) {
    assemble(GetParam());

    // Setting up the values for a reasonable comparison.
    auto right_mnn = mnncorrect::unique(pairings.right);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

    double limit = mnncorrect::limit_from_closest_distances(self_mnn);
    mnncorrect::compute_center_of_mass(ndim, right_mnn.size(), self_mnn, right.data(), limit, buffer.data());

    // Reference calculation for each MNN.
    std::vector<std::vector<int> > inverted(right_mnn.size());
    for (size_t s = 0; s < self_mnn.size(); ++s) {
        for (const auto& p : self_mnn[s]) {
            if (p.second <= limit) {
                inverted[p.first].push_back(s);
            }
        }
    }

    for (size_t i = 0; i < inverted.size(); ++i) {
        const auto& inv = inverted[i];
        std::vector<double> ref(ndim);

        for (auto x : inv) {
            const double* current = right.data() + x * ndim;
            for (int d = 0; d < ndim; ++d) {
                ref[d] += current[d];
            }
        }

        const double* obs = buffer.data() + i * ndim;
        for (int d = 0; d < ndim; ++d) {
            EXPECT_FLOAT_EQ(ref[d] / inv.size(), obs[d]);
        }
    }
}

TEST_P(CorrectTargetTest, Correction) {
    assemble(GetParam());
    std::vector<double> buffer(nright * ndim);
    correct_target(ndim, nleft, left.data(), nright, right.data(), pairings, k, 3.0, buffer.data());

    // Not entirely sure how to check for correctness here; 
    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
    std::vector<double> left_means(ndim), right_means(ndim);
    for (size_t l = 0; l < nleft; ++l) {
        for (int d = 0; d < ndim; ++d) {
            left_means[d] += left[l * ndim + d];
        }
    }
    for (size_t r = 0; r < nright; ++r) {
        for (int d = 0; d < ndim; ++d) {
            right_means[d] += buffer[r * ndim + d];
        }
    }
    for (int d = 0; d < ndim; ++d) {
        left_means[d] /= nleft;
        right_means[d] /= nright;
        double delta = std::abs(left_means[d] - right_means[d]);
        EXPECT_TRUE(delta < 1);
    }
}

INSTANTIATE_TEST_CASE_P(
    CorrectTarget,
    CorrectTargetTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);
