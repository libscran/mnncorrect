#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/define_merge_order.hpp"

#include <random>
#include <algorithm>
#include <cstddef>

TEST(DefineMergeOrder, RunningVariances) {
    int ndim = 12;
    int nobs = 34;
    auto data = scran_tests::simulate_vector(ndim * nobs, scran_tests::SimulateVectorParameters());

    double ref = 0;
    for (int d = 0; d < ndim; ++d) {
        // First pass for the mean.
        double* pos = data.data() + d;
        double mean = 0;
        for (int s = 0; s < nobs; ++s, pos += ndim) {
            mean += *pos;
        }
        mean /= nobs;

        // Second pass for the variance.
        pos = data.data() + d;
        double variance = 0;
        for (int s = 0; s < nobs; ++s, pos += ndim) {
            variance += (*pos - mean) * (*pos - mean);
        }
        variance /= nobs - 1;
        ref += variance;
    }

    std::vector<double> buffer(ndim);
    double running = mnncorrect::internal::compute_total_variance(ndim, nobs, data.data(), buffer, false);
    EXPECT_FLOAT_EQ(running, ref);

    double rss = mnncorrect::internal::compute_total_variance(ndim, nobs, data.data(), buffer, true);
    EXPECT_FLOAT_EQ(rss, ref * (nobs - 1));

    // Overlord function works, even with multiple threads.
    int nobs2 = 100;
    auto data2 = scran_tests::simulate_vector(ndim * nobs2, scran_tests::SimulateVectorParameters());

    auto vars = mnncorrect::internal::compute_total_variances<int, double>(ndim, { nobs, nobs2 }, { data.data(), data2.data() }, false, /* num_threads = */ 1);
    EXPECT_FLOAT_EQ(vars[0], running);
    EXPECT_FLOAT_EQ(vars[1], mnncorrect::internal::compute_total_variance(ndim, nobs2, data2.data(), buffer, false));

    auto pvars = mnncorrect::internal::compute_total_variances<int, double>(ndim, { nobs, nobs2 }, { data.data(), data2.data() }, false, /* num_threads = */ 3);
    EXPECT_EQ(vars, pvars);
}

TEST(DefineMergeOrder, Basic) {
    {
        std::vector<double> stat{ 1.2, 0.5, 3.5, 0.1 };
        std::vector<mnncorrect::BatchIndex> indices;
        mnncorrect::internal::define_merge_order(stat, indices);
        std::vector<mnncorrect::BatchIndex> expected { 2, 0, 1, 3 };
        EXPECT_EQ(indices, expected);
    }

    // Still works if the input 'indices' has something in it.
    {
        std::vector<double> stat{ 0.0, 1.1, 2.2, 3.3 };
        std::vector<mnncorrect::BatchIndex> indices{ 0, 1, 2, 3 };
        mnncorrect::internal::define_merge_order(stat, indices);
        std::vector<mnncorrect::BatchIndex> expected{ 3, 2, 1, 0};
        EXPECT_EQ(indices, expected);
    }
}
