#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/correct_target.hpp"
#include "aarand/aarand.hpp"
#include "knncolle/knncolle.hpp"
#include "helper_find_mutual_nns.hpp"
#include <random>

template<typename Index, typename Float>
mnncorrect::NeighborSet<Index, Float> identify_closest_mnn(int ndim, size_t nobs, const Float* data, const std::vector<Index>& in_mnn, int k, Float* buffer, size_t cap = -1, int nthreads = 1) {
    typedef knncolle::Base<Index, Float> knncolleBase;
    auto builder = [](int nd, size_t no, const Float* d) -> auto { 
        return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d));
    };
    return mnncorrect::identify_closest_mnn(ndim, nobs, data, in_mnn, builder, k, buffer, cap, nthreads);
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
        pairings = find_mutual_nns<int>(left.data(), right.data(), &left_index, &right_index, k, k);
    }

    int ndim = 5, k;
    size_t nleft, nright;
    std::vector<double> left, right;
    mnncorrect::MnnPairs<int> pairings;
};

TEST_P(CorrectTargetTest, IdentifyClosestMnns) {
    assemble(GetParam());

    auto right_mnn = mnncorrect::unique_right(pairings);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

    // Buffer is filled with the MNN data.
    EXPECT_TRUE(buffer.front() != 0);
    EXPECT_TRUE(buffer.back() != 0);

    // Nearest neighbors are identified in range.
    EXPECT_EQ(self_mnn.size(), nright);
    for (const auto& current : self_mnn) {
        for (const auto& p : current) {
            EXPECT_TRUE(p.first < right_mnn.size());
        }
    }

    // Same results in parallel.
    auto par_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data(), /* cap = */ -1, /* nthreads = */ 3);
    EXPECT_EQ(self_mnn.size(), par_mnn.size());
    for (size_t i = 0; i < self_mnn.size(); ++i) {
        EXPECT_EQ(self_mnn[i], par_mnn[i]);
    }
}

TEST_P(CorrectTargetTest, IdentifyClosestMnnsCapped) {
    assemble(GetParam());

    auto right_mnn = mnncorrect::unique_right(pairings);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

    std::vector<double> buffer2(right_mnn.size() * ndim);
    size_t ncap = 20;
    EXPECT_TRUE(ncap < nright);
    auto self_mnn2 = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer2.data(), ncap);

    {
        EXPECT_EQ(buffer, buffer2);
        size_t obs_ncap = 0;
        for (size_t c = 0; c < self_mnn2.size(); ++c) {
            if (!self_mnn2[c].empty()) {
                EXPECT_EQ(self_mnn[c], self_mnn2[c]);
                ++obs_ncap;
            }
        }
        EXPECT_EQ(obs_ncap, ncap);

        // Remaining steps run without issue.
        double limit = mnncorrect::limit_from_closest_distances(self_mnn2, 3.0);
        EXPECT_TRUE(limit > 0);
        auto inverted = mnncorrect::invert_neighbors(right_mnn.size(), self_mnn2, limit);
        EXPECT_EQ(inverted.size(), right_mnn.size());
    }

    // Same results in parallel.
    {
        auto par_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data(), 100, /* nthreads = */ 3);
        size_t obs_ncap = 0;
        for (size_t c = 0; c < self_mnn2.size(); ++c) {
            if (!self_mnn2[c].empty()) {
                EXPECT_EQ(par_mnn[c], self_mnn2[c]);
                ++obs_ncap;
            }
        }
        EXPECT_EQ(obs_ncap, ncap);
    }
}

TEST(DetermineLimitTest, LimitByClosest) {
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

TEST_P(CorrectTargetTest, CenterOfMass) {
    assemble(GetParam());

    const int iterations = 2;
    const double trim = 0.2;
    const int nthreads = 1;

    // Setting up the values for a reasonable comparison.
    auto left_mnn = mnncorrect::unique_left(pairings);
    std::vector<double> buffer_left(left_mnn.size() * ndim);
    {
        auto self_mnn = identify_closest_mnn(ndim, nleft, left.data(), left_mnn, k, buffer_left.data());
        double limit = mnncorrect::limit_from_closest_distances(self_mnn);
        mnncorrect::compute_center_of_mass(ndim, left_mnn.size(), self_mnn, left.data(), buffer_left.data(), iterations, trim, limit, nthreads);

        // Same results in parallel.
        {
            std::vector<double> par_buffer_left(left_mnn.size() * ndim);
            mnncorrect::compute_center_of_mass(ndim, left_mnn.size(), self_mnn, left.data(), par_buffer_left.data(), iterations, trim, limit, /* nthreads = */ 3);
            EXPECT_EQ(par_buffer_left, buffer_left);
        }
    }

    auto right_mnn = mnncorrect::unique_right(pairings);
    std::vector<double> buffer_right(right_mnn.size() * ndim);
    {
        auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer_right.data());
        double limit = mnncorrect::limit_from_closest_distances(self_mnn);
        mnncorrect::compute_center_of_mass(ndim, right_mnn.size(), self_mnn, right.data(), buffer_right.data(), iterations, trim, limit, nthreads);
    }

    // Checking that the centroids are all around about the expectations.
    std::vector<double> left_means(ndim);
    for (size_t s = 0; s < left_mnn.size(); ++s) {
        for (int d = 0; d < ndim; ++d) {
            left_means[d] += buffer_left[s * ndim + d];
        }
    }
    for (auto m : left_means) {
        EXPECT_TRUE(std::abs(m / left_mnn.size()) < 0.5);
    }

    std::vector<double> right_means(ndim);
    for (size_t s = 0; s < right_mnn.size(); ++s) {
        for (int d = 0; d < ndim; ++d) {
            right_means[d] += buffer_right[s * ndim + d];
        }
    }
    for (auto m : right_means) {
        EXPECT_TRUE(std::abs(m / right_mnn.size() - 5) < 0.5);
    }
}

template<typename Index, typename Float>
void correct_target(int ndim, size_t nref, const Float* ref, size_t ntarget, const Float* target, const mnncorrect::MnnPairs<Index>& pairings, int k, Float* output, size_t cap = -1, int nthreads = 1) {
    typedef knncolle::Base<Index, Float> knncolleBase;
    auto builder = [](int nd, size_t no, const Float* d) -> auto { 
        return std::shared_ptr<knncolleBase>(new knncolle::VpTreeEuclidean<Index, Float>(nd, no, d)); 
    };

    const double nmads = 3;
    const int iterations = 2;
    const double trim = 0.2;
    mnncorrect::correct_target(ndim, nref, ref, ntarget, target, pairings, builder, k, nmads, iterations, trim, output, cap, nthreads);
    return;
}

TEST_P(CorrectTargetTest, Correction) {
    assemble(GetParam());
    std::vector<double> buffer(nright * ndim);
    correct_target(ndim, nleft, left.data(), nright, right.data(), pairings, k, buffer.data());

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

    // Same result with multiple threads.
    std::vector<double> par_buffer(nright * ndim);
    correct_target(ndim, nleft, left.data(), nright, right.data(), pairings, k, par_buffer.data(), /* cap = */ -1, /* nthreads = */ 3);
    EXPECT_EQ(par_buffer, buffer);

    // Different results with a cap.
    std::vector<double> cap_buffer(nright * ndim);
    correct_target(ndim, nleft, left.data(), nright, right.data(), pairings, k, cap_buffer.data(), /* cap = */ 50);
    EXPECT_NE(cap_buffer, buffer);
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
