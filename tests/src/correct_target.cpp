#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/correct_target.hpp"
#include "mnncorrect/fuse_nn_results.hpp"
#include "knncolle/knncolle.hpp"

TEST(DetermineLimitTest, LimitByClosest) {
    mnncorrect::internal::NeighborSet<int, double> closest(2);

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

    double limit = mnncorrect::internal::limit_from_closest_distances(closest, /* nmads = */ 3.0);

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
    void SetUp() {
        auto param = GetParam();
        nleft = std::get<0>(param);
        nright = std::get<1>(param);
        k = std::get<2>(param);

        left = scran_tests::simulate_vector(nleft * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.lower = -2;
            sparams.upper = 2;
            sparams.seed = 42 + nleft * 10 + nright + k;
            return sparams;
        }());

        right = scran_tests::simulate_vector(nright * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.lower = -2 + 5; // throw in a batch effect.
            sparams.upper = 2 + 5;
            sparams.seed = 69 + nleft * 10 + nright + k;
            return sparams;
        }());

        auto left_index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix(ndim, nleft, left.data()));
        auto right_index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix(ndim, nright, right.data()));
        auto neighbors_of_left = mnncorrect::internal::quick_find_nns(nleft, left.data(), *right_index, k, /* nthreads = */ 1);
        auto neighbors_of_right = mnncorrect::internal::quick_find_nns(nright, right.data(), *left_index, k, /* nthreads = */ 1);
        pairings = mnncorrect::internal::find_mutual_nns(neighbors_of_left, neighbors_of_right);
    }

    int ndim = 5;
    int nleft, nright;
    int k;
    std::vector<double> left, right;
    mnncorrect::internal::MnnPairs<int> pairings;

protected:
    template<typename Index_, typename Float_>
    static mnncorrect::internal::NeighborSet<Index_, Float_> identify_closest_mnn(
        size_t ndim,
        size_t nobs,
        const Float_* data,
        const std::vector<Index_>& in_mnn,
        int k,
        Float_* buffer,
        size_t cap = -1,
        int nthreads = 1) 
    {
        mnncorrect::internal::subset_to_mnns(ndim, data, in_mnn, buffer);
        auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, Float_>(ndim, in_mnn.size(), buffer));
        return mnncorrect::internal::identify_closest_mnn(nobs, data, *index, k, cap, nthreads);
    }
};

TEST_P(CorrectTargetTest, IdentifyClosestMnns) {
    auto right_mnn = mnncorrect::internal::unique_right(pairings);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

    // Buffer is filled with the MNN data.
    EXPECT_TRUE(buffer.front() != 0);
    EXPECT_TRUE(buffer.back() != 0);

    // Nearest neighbors are identified in range.
    EXPECT_EQ(self_mnn.size(), nright);
    for (const auto& current : self_mnn) {
        for (const auto& p : current) {
            EXPECT_LT(p.first, right_mnn.size());
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
    auto right_mnn = mnncorrect::internal::unique_right(pairings);
    std::vector<double> buffer(right_mnn.size() * ndim);
    auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer.data());

    std::vector<double> buffer2(right_mnn.size() * ndim);
    size_t ncap = 20;
    EXPECT_LT(ncap, nright);
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
        double limit = mnncorrect::internal::limit_from_closest_distances(self_mnn2, 3.0);
        EXPECT_GT(limit, 0);
        auto inverted = mnncorrect::internal::invert_neighbors(right_mnn.size(), self_mnn2, limit);
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

TEST_P(CorrectTargetTest, CenterOfMass) {
    mnncorrect::internal::RobustAverageOptions raopt(/* iterations = */ 2, /* trim = */ 0.2);

    // Setting up the values for a reasonable comparison.
    auto left_mnn = mnncorrect::internal::unique_left(pairings);
    std::vector<double> buffer_left(left_mnn.size() * ndim);
    {
        auto self_mnn = identify_closest_mnn(ndim, nleft, left.data(), left_mnn, k, buffer_left.data());
        double limit = mnncorrect::internal::limit_from_closest_distances(self_mnn, /* nmads = */ 3.0);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, self_mnn, left.data(), buffer_left.data(), raopt, limit, /* nthreads = */ 1);

        // Same results in parallel.
        {
            std::vector<double> par_buffer_left(left_mnn.size() * ndim);
            mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, self_mnn, left.data(), par_buffer_left.data(), raopt, limit, /* nthreads = */ 3);
            EXPECT_EQ(par_buffer_left, buffer_left);
        }
    }

    auto right_mnn = mnncorrect::internal::unique_right(pairings);
    std::vector<double> buffer_right(right_mnn.size() * ndim);
    {
        auto self_mnn = identify_closest_mnn(ndim, nright, right.data(), right_mnn, k, buffer_right.data());
        double limit = mnncorrect::internal::limit_from_closest_distances(self_mnn, /* nmads = */ 3.0);
        mnncorrect::internal::compute_center_of_mass(ndim, right_mnn, self_mnn, right.data(), buffer_right.data(), raopt, limit, /* nthreads = */ 1);
    }

    // Checking that the centroids are all close to the expected values.
    std::vector<double> left_means(ndim);
    for (size_t s = 0; s < left_mnn.size(); ++s) {
        for (int d = 0; d < ndim; ++d) {
            left_means[d] += buffer_left[s * ndim + d];
        }
    }
    for (auto m : left_means) {
        EXPECT_LT(std::abs(m / left_mnn.size()), 0.5);
    }

    std::vector<double> right_means(ndim);
    for (size_t s = 0; s < right_mnn.size(); ++s) {
        for (int d = 0; d < ndim; ++d) {
            right_means[d] += buffer_right[s * ndim + d];
        }
    }
    for (auto m : right_means) {
        EXPECT_LT(std::abs(m / right_mnn.size() - 5), 0.5);
    }
}

TEST_P(CorrectTargetTest, CenterOfMassCapped) {
    mnncorrect::internal::RobustAverageOptions raopt(/* iterations = */ 2, /* trim = */ 0.2);

    auto left_mnn = mnncorrect::internal::unique_left(pairings);
    std::vector<double> buffer_left(left_mnn.size() * ndim);

    // Reference value.
    {
        auto self_mnn = identify_closest_mnn(ndim, nleft, left.data(), left_mnn, k, buffer_left.data());
        double limit = mnncorrect::internal::limit_from_closest_distances(self_mnn, /* nmads = */ 3.0);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, self_mnn, left.data(), buffer_left.data(), raopt, limit, /* nthreads = */ 1);
    }

    // Forcing a cap to get different results.
    {
        std::vector<double> buffer_left2(left_mnn.size() * ndim);
        auto self_mnn2 = identify_closest_mnn(ndim, nleft, left.data(), left_mnn, k, buffer_left2.data(), 50);
        double limit2 = mnncorrect::internal::limit_from_closest_distances(self_mnn2, /* nmads = */ 3.0);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, self_mnn2, left.data(), buffer_left2.data(), raopt, limit2, /* nthreads = */ 1);
        EXPECT_NE(buffer_left, buffer_left2);
    }

    // Checking what happens when the cap is onerous.
    {
        std::vector<double> buffer_left2(left_mnn.size() * ndim);
        auto self_mnn2 = identify_closest_mnn(ndim, nleft, left.data(), left_mnn, k, buffer_left2.data(), 0);
        double limit2 = mnncorrect::internal::limit_from_closest_distances(self_mnn2, /* nmads = */ 3.0);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, self_mnn2, left.data(), buffer_left2.data(), raopt, limit2, /* nthreads = */ 1);

        std::vector<double> expected;
        for (auto x : left_mnn) {
            auto it = left.data() + x * ndim;
            expected.insert(expected.end(), it, it + ndim);
        }
        EXPECT_EQ(expected, buffer_left2);
    }
}

TEST_P(CorrectTargetTest, Correction) {
    std::vector<double> buffer(nright * ndim);
    mnncorrect::internal::correct_target(
        ndim,
        nleft,
        left.data(),
        nright,
        right.data(),
        pairings,
        knncolle::VptreeBuilder(),
        k,
        /* nmads = */ 3.0,
        /* iterations = */ 2,
        /* trim = */ 0.2,
        buffer.data(),
        /* cap = */ -1, 
        /* nthreads = */ 1
    );

    // Not entirely sure how to check for correctness here; 
    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
    std::vector<double> left_means(ndim), right_means(ndim);
    for (int l = 0; l < nleft; ++l) {
        for (int d = 0; d < ndim; ++d) {
            left_means[d] += left[l * ndim + d];
        }
    }
    for (int r = 0; r < nright; ++r) {
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
    mnncorrect::internal::correct_target(
        ndim,
        nleft,
        left.data(),
        nright,
        right.data(),
        pairings,
        knncolle::VptreeBuilder(),
        k,
        /* nmads = */ 3.0,
        /* iterations = */ 2,
        /* trim = */ 0.2,
        par_buffer.data(),
        /* cap = */ -1, 
        /* nthreads = */ 3 
    );
    EXPECT_EQ(par_buffer, buffer);

    // Different results with a cap.
    std::vector<double> cap_buffer(nright * ndim);
    mnncorrect::internal::correct_target(
        ndim,
        nleft,
        left.data(),
        nright,
        right.data(),
        pairings,
        knncolle::VptreeBuilder(),
        k,
        /* nmads = */ 3.0,
        /* iterations = */ 2,
        /* trim = */ 0.2,
        cap_buffer.data(),
        /* cap = */ 50, 
        /* nthreads = */ 3 
    );
    EXPECT_NE(cap_buffer, buffer);
}

INSTANTIATE_TEST_SUITE_P(
    CorrectTarget,
    CorrectTargetTest,
    ::testing::Combine(
        ::testing::Values(100, 1000), // left
        ::testing::Values(100, 1000), // right
        ::testing::Values(10, 50)  // choice of k
    )
);
