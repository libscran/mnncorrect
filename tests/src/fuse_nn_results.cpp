#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/fuse_nn_results.hpp"

#include "scran_tests/scran_tests.hpp"
#include "knncolle/knncolle.hpp"

#include <cmath>
#include <vector>
#include <cstddef>

TEST(FuseNnResults, Basic) {
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 }, { 7, 1.7 }, { 5, 2.5 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 7, 1.7 }, { 2, 2.2 } };
        EXPECT_EQ(output, expected);
    }

    // Not enough of 'base'.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 }, { 7, 1.7 }, { 5, 2.5 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 7, 1.7 }, { 5, 2.5 } };
        EXPECT_EQ(output, expected);
    }

    // Not enough of 'alt'.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        EXPECT_EQ(output, expected);
    }

    // Ties.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 4, 2.2 }, { 5, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 2, 1.1 }, { 3, 2.2 }, { 6, 3.3 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output);
        std::vector<std::pair<int, double> > expected { { 1, 1.1 }, { 2, 1.1 }, { 3, 2.2 }, { 4, 2.2 } };
        EXPECT_EQ(output, expected);
    }
}

class FuseNnResultsTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {};

TEST_P(FuseNnResultsTest, Randomized) {
    auto param = GetParam();
    auto nleft = std::get<0>(param);
    auto nright = std::get<1>(param);
    auto nkeep = std::get<2>(param);

    std::mt19937_64 rng(nleft * nright + nkeep);
    std::normal_distribution ndist;
    std::uniform_int_distribution udist(0, 10000000);
    auto comp = [](const auto& l, const auto& r) -> bool { return l.second < r.second; };

    std::vector<std::pair<int, double> > base;
    for (int l = 0; l < nleft; ++l) {
        base.emplace_back(udist(rng), ndist(rng));
    }
    std::sort(base.begin(), base.end(), comp);

    std::vector<std::pair<int, double> > alt;
    for (int r = 0; r < nright; ++r) {
        alt.emplace_back(udist(rng), ndist(rng));
    }
    std::sort(alt.begin(), alt.end(), comp);
    
    auto ref = base;
    ref.insert(ref.end(), alt.begin(), alt.end());
    std::sort(ref.begin(), ref.end(), comp);
    if (static_cast<std::size_t>(nkeep) < ref.size()) {
        ref.resize(nkeep);
    }

    std::vector<std::pair<int, double> > output;
    mnncorrect::internal::fuse_nn_results(base, alt, nkeep, output);
    EXPECT_EQ(ref, output);
}

INSTANTIATE_TEST_SUITE_P(
    FuseNnResults,
    FuseNnResultsTest,
    ::testing::Combine(
        ::testing::Values(1, 5, 10), // left
        ::testing::Values(1, 5, 10), // right
        ::testing::Values(1, 5, 10) // number to keep
    )
);

TEST(FuseNnResults, Recovery) {
    // Recover the same NN results as just a direct search.
    std::size_t NR = 10;
    int NC = 100;
    auto contents = scran_tests::simulate_vector(NR * NC, []{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 69;
        return sparams;
    }());

    knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());

    auto prebuilt_full = builder.build_unique(knncolle::SimpleMatrix<int, double>(NR, NC, contents.data()));
    auto prebuilt_first = builder.build_unique(knncolle::SimpleMatrix<int, double>(NR, 50, contents.data()));
    auto prebuilt_second = builder.build_unique(knncolle::SimpleMatrix<int, double>(NR, NC - 50, contents.data() + 50 * NR));

    int k = 7;
    auto full_res = knncolle::find_nearest_neighbors(*prebuilt_full, k);
    auto first_res = knncolle::find_nearest_neighbors(*prebuilt_first, k);
    auto second_res = knncolle::find_nearest_neighbors(*prebuilt_second, k);

    std::vector<std::pair<int, double> > fused;
    for (int c = 0; c < NC; ++c) {
        mnncorrect::internal::fuse_nn_results(first_res[c], second_res[c], k, fused);
        EXPECT_EQ(full_res[c], fused);
    }
}
