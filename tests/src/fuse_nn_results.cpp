#include <gtest/gtest.h>

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/utils.hpp"
#include <cmath>
#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/fuse_nn_results.hpp"
#include "knncolle/knncolle.hpp"
#include <vector>

TEST(QuickFindNns, Basic) {
    size_t NR = 10;
    size_t NC = 100;
    auto contents = scran_tests::simulate_vector(NR * NC, []{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 999;
        return sparams;
    }());

    auto prebuilt = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix<int, int, double>(NR, NC, contents.data()));
    int k = 5;
    auto output = mnncorrect::internal::quick_find_nns(NC, contents.data(), *prebuilt, /* k = */ k, /* num_threads = */ 1);
    ASSERT_EQ(output.size(), NC);

    // Comparing to the reference.
    auto ref = knncolle::find_nearest_neighbors(*prebuilt, 5);
    for (size_t c = 0; c < NC; ++c) {
        EXPECT_EQ(output[c].size(), k);
        EXPECT_EQ(output[c][0].first, c); // self, duh...
        EXPECT_EQ(output[c][0].second, 0);

        for (int i = 0; i < k - 1; ++i) {
            EXPECT_EQ(ref[c].first[i], output[c][i + 1].first);
            EXPECT_EQ(ref[c].second[i], output[c][i + 1].second);
        }
    }

    // Works with multiple threads.
    auto poutput = mnncorrect::internal::quick_find_nns(NC, contents.data(), *prebuilt, /* k = */ k, /* num_threads = */ 3);
    ASSERT_EQ(output.size(), poutput.size());
    for (size_t c = 0; c < NC; ++c) {
        EXPECT_EQ(output[c], poutput[c]);
    }

    // Step by step construction works as expected.
    mnncorrect::internal::NeighborSet<int, double> stepwise(NC);
    mnncorrect::internal::quick_find_nns(50, contents.data(), *prebuilt, k, /* num_threads = */ 1, stepwise, 0);
    mnncorrect::internal::quick_find_nns(NC - 50, contents.data() + 50 * NR, *prebuilt, k, /* num_threads = */ 1, stepwise, 50);
    ASSERT_EQ(output.size(), stepwise.size());
    for (size_t c = 0; c < NC; ++c) {
        EXPECT_EQ(output[c], stepwise[c]);
    }
}

TEST(FuseNnResults, Basic) {
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 }, { 7, 1.7 }, { 5, 2.5 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 0);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 7, 1.7 }, { 2, 2.2 } };
        EXPECT_EQ(output, expected);

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 10);
        expected = std::vector<std::pair<int, double> >{ { 19, 0.9 }, { 1, 1.1 }, { 17, 1.7 }, { 2, 2.2 } };
        EXPECT_EQ(output, expected);
    }

    // Not enough of 'base'.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 }, { 7, 1.7 }, { 5, 2.5 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 0);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 7, 1.7 }, { 5, 2.5 } };
        EXPECT_EQ(output, expected);

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 10);
        expected = std::vector<std::pair<int, double> >{ { 19, 0.9 }, { 1, 1.1 }, { 17, 1.7 }, { 15, 2.5 } };
        EXPECT_EQ(output, expected);
    }

    // Not enough of 'alt'.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 9, 0.9 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 0);
        std::vector<std::pair<int, double> > expected { { 9, 0.9 }, { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        EXPECT_EQ(output, expected);

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 10);
        expected = std::vector<std::pair<int, double> > { { 19, 0.9 }, { 1, 1.1 }, { 2, 2.2 }, { 3, 3.3 } };
        EXPECT_EQ(output, expected);
    }

    // Ties.
    {
        std::vector<std::pair<int, double> > base { { 1, 1.1 }, { 4, 2.2 }, { 5, 3.3 } };
        std::vector<std::pair<int, double> > alt { { 2, 1.1 }, { 3, 2.2 }, { 6, 3.3 } };
        std::vector<std::pair<int, double> > output;

        mnncorrect::internal::fuse_nn_results(base, alt, 4, output, 0);
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
    if (static_cast<size_t>(nkeep) < ref.size()) {
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
    size_t NR = 10;
    size_t NC = 100;
    auto contents = scran_tests::simulate_vector(NR * NC, []{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 69;
        return sparams;
    }());

    int k = 5;
    auto prebuilt_full = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix<int, int, double>(NR, NC, contents.data()));
    auto ref = mnncorrect::internal::quick_find_nns(NC, contents.data(), *prebuilt_full, /* k = */ k, /* num_threads = */ 1);

    auto prebuilt_first = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix<int, int, double>(NR, 50, contents.data()));
    auto output = mnncorrect::internal::quick_find_nns(NC, contents.data(), *prebuilt_first, /* k = */ k, /* num_threads = */ 1);
    auto prebuilt_second = knncolle::VptreeBuilder<>().build_unique(knncolle::SimpleMatrix<int, int, double>(NR, NC - 50, contents.data() + 50 * NR));
    mnncorrect::internal::quick_fuse_nns(output, contents.data(), *prebuilt_second, /* k = */ k, /* num_threads = */ 1, /* offset = */ 50);

    ASSERT_EQ(output.size(), ref.size());
    for (size_t c = 0; c < NC; ++c) {
        EXPECT_EQ(output[c], ref[c]);
    }
}
