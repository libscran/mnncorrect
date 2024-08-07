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
};

TEST_P(CorrectTargetTest, CappedFindNns) {
    auto right_mnn = mnncorrect::internal::unique_right(pairings);
    std::vector<double> subbuffer(right_mnn.size() * ndim);
    mnncorrect::internal::subset_to_mnns(ndim, right.data(), right_mnn, subbuffer.data());
    auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, right_mnn.size(), subbuffer.data()));

    auto full = mnncorrect::internal::quick_find_nns(nright, right.data(), *index, k, /* nthreads = */ 1);

    auto cap_out = mnncorrect::internal::capped_find_nns(nright, right.data(), *index, k, 23, /* nthreads = */ 1);
    auto gap = cap_out.first;
    const auto& capped = cap_out.second;

    EXPECT_EQ(capped.size(), 23);
    EXPECT_GT(gap, 1);
    for (size_t c = 0; c < capped.size(); ++c) {
        EXPECT_EQ(full[static_cast<size_t>(c * gap)], capped[c]);
    }

    // Same results in parallel.
    auto pcap_out = mnncorrect::internal::capped_find_nns(nright, right.data(), *index, k, 23, /* nthreads = */ 3);
    EXPECT_EQ(pcap_out.first, cap_out.first);
    EXPECT_EQ(pcap_out.second, cap_out.second);
}

TEST_P(CorrectTargetTest, CenterOfMass) {
    mnncorrect::internal::RobustAverageOptions raopt(/* iterations = */ 2, /* trim = */ 0.2);

    // Setting up the values for a reasonable comparison.
    auto left_mnn = mnncorrect::internal::unique_left(pairings);
    std::vector<double> buffer_left(left_mnn.size() * ndim);
    {
        mnncorrect::internal::subset_to_mnns(ndim, left.data(), left_mnn, buffer_left.data());
        auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, left_mnn.size(), buffer_left.data()));
        auto closest_mnn = mnncorrect::internal::quick_find_nns(nleft, left.data(), *index, k, /* nthreads = */ 1);
        auto inverted = mnncorrect::internal::invert_neighbors(left_mnn.size(), closest_mnn, /* limit = */ 1e8);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, inverted, left.data(), buffer_left.data(), raopt, /* nthreads = */ 1);

        // Same results in parallel.
        std::vector<double> par_buffer_left(left_mnn.size() * ndim);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, inverted, left.data(), par_buffer_left.data(), raopt, /* nthreads = */ 3);
        EXPECT_EQ(par_buffer_left, buffer_left);
    }

    auto right_mnn = mnncorrect::internal::unique_right(pairings);
    std::vector<double> buffer_right(right_mnn.size() * ndim);
    {
        mnncorrect::internal::subset_to_mnns(ndim, right.data(), right_mnn, buffer_right.data());
        auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix<int, int, double>(ndim, right_mnn.size(), buffer_right.data()));
        auto closest_mnn = mnncorrect::internal::quick_find_nns(nright, right.data(), *index, k, /* nthreads = */ 1);
        auto inverted = mnncorrect::internal::invert_neighbors(right_mnn.size(), closest_mnn, /* limit = */ 1e8);
        mnncorrect::internal::compute_center_of_mass(ndim, right_mnn, inverted, right.data(), buffer_right.data(), raopt, /* nthreads = */ 1);
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

    // Center of mass calculations work correctly if it's all empty.
    {
        std::vector<std::vector<int> > empty_inverted(left_mnn.size());
        std::vector<double> empty_buffer_left(left_mnn.size() * ndim);
        mnncorrect::internal::compute_center_of_mass(ndim, left_mnn, empty_inverted, left.data(), empty_buffer_left.data(), raopt, /* nthreads = */ 1);

        std::vector<double> expected(left_mnn.size() * ndim);
        mnncorrect::internal::subset_to_mnns(ndim, left.data(), left_mnn, expected.data());
        EXPECT_EQ(empty_buffer_left, expected);
    }
}

TEST_P(CorrectTargetTest, Correction) {
    double nmads = 3;
    int iterations = 2;
    double trim = 0.2;

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
        nmads,
        iterations,
        trim,
        buffer.data(),
        /* mass_cap = */ -1, 
        /* nthreads = */ 1
    );

    // Not entirely sure how to check for correctness here; 
    // we'll heuristically check for a delta less than 1 on the mean in each dimension.
    {
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
    }

    // Same result with multiple threads.
    {
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
            nmads,
            iterations,
            trim,
            par_buffer.data(),
            /* mass_cap = */ -1, 
            /* nthreads = */ 3 
        );
        EXPECT_EQ(par_buffer, buffer);
    }

    // Different results with a cap.
    {
        int mass_cap = 50;

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
            nmads,
            iterations,
            trim,
            cap_buffer.data(),
            /* mass_cap = */ mass_cap, 
            /* nthreads = */ 1 
        );
        EXPECT_NE(cap_buffer, buffer);
    }
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
