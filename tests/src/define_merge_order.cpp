#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/define_merge_order.hpp"

#include <random>
#include <algorithm>
#include <cstddef>

TEST(ComputeTotalVariance, Simple) {
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
    double running = mnncorrect::compute_total_variance(ndim, nobs, data.data(), buffer, false);
    EXPECT_FLOAT_EQ(running, ref);

    double rss = mnncorrect::compute_total_variance(ndim, nobs, data.data(), buffer, true);
    EXPECT_FLOAT_EQ(rss, ref * (nobs - 1));

    // Variance is set to zero if there aren't enough cells.
    EXPECT_EQ(mnncorrect::compute_total_variance(ndim, 0, data.data(), buffer, false), 0);
    EXPECT_EQ(mnncorrect::compute_total_variance(ndim, 1, data.data(), buffer, false), 0);
}

TEST(ComputeTotalVariance, Combined) {
    int ndim = 7;
    std::vector<int> sizes { 23, 73, 66 };
    int num_total = 0;
    for (auto s : sizes) {
        num_total += s;
    }
    auto data = scran_tests::simulate_vector(ndim * num_total, scran_tests::SimulateVectorParameters());

    std::vector<double> buffer(ndim);
    std::vector<mnncorrect::Batch<int> > batches;
    std::vector<double> expected_variances, expected_rss;
    int accumulated = 0;
    for (auto s : sizes) {
        mnncorrect::Batch<int> curbatch;
        curbatch.start = accumulated;
        curbatch.size = s;
        batches.push_back(curbatch);
        expected_variances.push_back(mnncorrect::compute_total_variance(ndim, s, data.data() + accumulated * ndim, buffer, false));
        expected_rss.push_back(mnncorrect::compute_total_variance(ndim, s, data.data() + accumulated * ndim, buffer, true));
        accumulated += s;
    }

    auto vars = mnncorrect::compute_total_variances<int, double>(ndim, batches, data.data(), false, 1);
    EXPECT_EQ(vars, expected_variances);
    auto rss = mnncorrect::compute_total_variances<int, double>(ndim, batches, data.data(), true, 1);
    EXPECT_EQ(rss, expected_rss);

    // Changing the order to check that it doesn't rely on batches being sorted.
    auto revbatches = batches;
    std::reverse(revbatches.begin(), revbatches.end());
    auto revvars = mnncorrect::compute_total_variances<int, double>(ndim, revbatches, data.data(), false, 1);
    std::reverse(revvars.begin(), revvars.end());
    EXPECT_EQ(revvars, expected_variances);

    // Same result with multiple threads.
    auto pvars = mnncorrect::compute_total_variances<int, double>(ndim, batches, data.data(), false, 3);
    EXPECT_EQ(vars, pvars);
}

TEST(DefineMergeOrder, Variance) {
    {
        std::vector<double> stat{ 1.2, 0.5, 3.5, 0.1 };
        std::vector<mnncorrect::BatchIndex> indices;
        mnncorrect::define_variance_merge_order(stat, indices);
        std::vector<mnncorrect::BatchIndex> expected { 2, 0, 1, 3 };
        EXPECT_EQ(indices, expected);
    }

    // Still works if the input 'indices' has something in it.
    {
        std::vector<double> stat{ 0.0, 1.1, 2.2, 3.3 };
        std::vector<mnncorrect::BatchIndex> indices{ 0, 1, 2, 3 };
        mnncorrect::define_variance_merge_order(stat, indices);
        std::vector<mnncorrect::BatchIndex> expected{ 3, 2, 1, 0};
        EXPECT_EQ(indices, expected);
    }
}

TEST(DefineMergeOrder, Size) {
    std::vector<mnncorrect::Batch<int> > batches(4);
    batches[0].start = 0;
    batches[0].size = 10;
    batches[1].start = 10;
    batches[1].size = 5;
    batches[2].start = 15;
    batches[2].size = 20;
    batches[3].start = 35;
    batches[3].size = 15;

    std::vector<mnncorrect::BatchIndex> indices;
    mnncorrect::define_size_merge_order(batches, indices);
    std::vector<mnncorrect::BatchIndex> expected { 2, 3, 0, 1 };
    EXPECT_EQ(indices, expected);
}
