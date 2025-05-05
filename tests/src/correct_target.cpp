#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/correct_target.hpp"
#include "mnncorrect/find_local_centers.hpp"
#include "mnncorrect/fuse_nn_results.hpp"
#include "knncolle/knncolle.hpp"

#include <cstddef>
#include <utility>
#include <vector>

class CorrectTargetTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    void SetUp() {
        auto param = GetParam();
        nleft = std::get<0>(param);
        nright = std::get<1>(param);
        k = std::get<2>(param);

        left = scran_tests::simulate_vector(nleft * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.lower = -0.5;
            sparams.upper = 0.5;
            sparams.seed = 42 + nleft * 10 + nright + k;
            return sparams;
        }());

        right = scran_tests::simulate_vector(nright * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.lower = -0.5 + 5; // throw in a batch effect.
            sparams.upper = 0.5 + 5;
            sparams.seed = 69 + nleft * 10 + nright + k;
            return sparams;
        }());

        knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
        auto left_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nleft, left.data()));
        auto right_index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nright, right.data()));

        auto neighbors_of_left = mnncorrect::internal::quick_find_nns(nleft, left.data(), *right_index, k, /* nthreads = */ 1);
        auto neighbors_of_right = mnncorrect::internal::quick_find_nns(nright, right.data(), *left_index, k, /* nthreads = */ 1);
        pairings = mnncorrect::internal::find_mutual_nns(neighbors_of_left, neighbors_of_right);

        auto left_self_neighbors = mnncorrect::internal::quick_find_nns(nleft, left.data(), *left_index, k, /* nthreads = */ 1);
        mnncorrect::internal::find_local_centers(left_self_neighbors, left_centers);
        auto right_self_neighbors = mnncorrect::internal::quick_find_nns(nright, right.data(), *right_index, k, /* nthreads = */ 1);
        mnncorrect::internal::find_local_centers(right_self_neighbors, right_centers);
    }

    int ndim = 5;
    int nleft, nright;
    int k;
    std::vector<double> left, right;
    mnncorrect::internal::MnnPairs<int> pairings;
    std::vector<int> left_centers;
    std::vector<int> right_centers;
};

TEST_P(CorrectTargetTest, Correction) {
    int iterations = 2;
    double trim = 0.2;
    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    std::vector<double> buffer(nright * ndim);
    mnncorrect::internal::correct_target(
        ndim,
        nleft,
        left.data(),
        left_centers,
        nright,
        right.data(),
        right_centers,
        pairings,
        builder,
        k,
        iterations,
        trim,
        buffer.data(),
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
            EXPECT_LT(delta, 1);
        }
    }

    // Same result with multiple threads.
    {
        std::vector<double> par_buffer(nright * ndim);
        mnncorrect::internal::correct_target(
            ndim,
            nleft,
            left.data(),
            left_centers,
            nright,
            right.data(),
            right_centers,
            pairings,
            builder,
            k,
            iterations,
            trim,
            par_buffer.data(),
            /* nthreads = */ 3 
        );
        EXPECT_EQ(par_buffer, buffer);
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
