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

    knncolle::VptreeBuilder builder;
    auto prebuilt = builder.build_unique(knncolle::SimpleMatrix<int, int, double>(NR, NC, contents.data()));
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
