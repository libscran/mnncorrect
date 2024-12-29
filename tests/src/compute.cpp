#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/compute.hpp"
#include <random>
#include <algorithm>
#include <cmath>

class OverallTest : public ::testing::TestWithParam<std::tuple<int, int, std::vector<size_t> > > {
protected:
    constexpr static double multiplier = 10;

    void SetUp() {
        auto param = GetParam();
        ndim = std::get<0>(param);
        k = std::get<1>(param);
        sizes = std::get<2>(param);

        nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
        data = scran_tests::simulate_vector(nobs * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = ndim * k + nobs;
            sparams.lower = -2;
            sparams.upper = 2;
            return sparams;
        }());

        ptrs.resize(sizes.size());
        size_t sofar = 0;
        for (size_t b = 0; b < sizes.size(); ++b) {
            auto current = data.data() + sofar;
            size_t len = sizes[b] * ndim;
            for (size_t i = 0; i < len; ++i) { // introducing our own batch effect.
                current[i] += multiplier * b;
            }
            ptrs[b] = current;
            sofar += len;
        }

        return;
    }

protected:
    // Parameters.
    int ndim, nobs, k;
    std::vector<size_t> sizes;

    // Simulated.
    std::vector<double> data;
    std::vector<const double*> ptrs;
};

TEST_P(OverallTest, Basic) {
    std::vector<double> output(nobs * ndim);
    auto ordering = mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        return opt;
    }());
    size_t refbatch = ordering.merge_order.front();

    // Heuristic: check that the differences in the mean are less than the
    // standard deviation (default 1) in each dimension.
    size_t sofar = 0;
    for (size_t b = 0; b < sizes.size(); ++b) {
        auto ptr = output.data() + sofar * ndim;
        std::vector<double> ref(ndim);

        for (size_t s = 0; s < sizes[b]; ++s) {
            for (int d = 0; d < ndim; ++d) {
                ref[d] += ptr[d];                
            }
            ptr += ndim;
        }

        for (auto& r : ref) {
            r /= sizes[b];
            double delta = std::abs(r - refbatch * multiplier);
            EXPECT_TRUE(delta < 1);
        }
    }

    // Check that the first batch is indeed unchanged.
    std::vector<double> original(ptrs[refbatch], ptrs[refbatch] + sizes[refbatch] * ndim);
    size_t offset = 0;
    for (size_t b = 0; b < refbatch; ++b) {
        offset += sizes[b];
    }
    std::vector<double> corrected(output.begin() + offset * ndim, output.begin() + (offset + sizes[refbatch]) * ndim);
    EXPECT_EQ(original, corrected);

    // Same results when multiple threads are in use.
    std::vector<double> par_output(nobs * ndim);
    auto par_ordering = mnncorrect::compute(ndim, sizes, ptrs, par_output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        opt.num_threads = 3;
        return opt;
    }());
    EXPECT_EQ(par_ordering.merge_order, ordering.merge_order);
    EXPECT_EQ(par_output, output);
}

TEST_P(OverallTest, Iterative) {
    std::vector<double> output(nobs * ndim);
    auto ordering = mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        return opt;
    }());

    // Now trying to run it iteratively as a reference.
    size_t previous = ordering.merge_order[0];
    std::vector<double> ref(nobs * ndim), buffer(nobs * ndim);
    std::vector<const double*> ref_ptrs { ptrs[previous], NULL };
    std::vector<size_t> ref_sizes{ sizes[previous], 0 };

    for (size_t i = 1; i < ordering.merge_order.size(); ++i) {
        if (i != 1) {
            std::copy(ref.begin(), ref.end(), buffer.begin());
            ref_ptrs[0] = buffer.data();
            ref_sizes[0] += sizes[previous];
        }

        size_t current = ordering.merge_order[i];
        ref_ptrs[1] = ptrs[current];
        ref_sizes[1] = sizes[current];

        mnncorrect::compute(ndim, ref_sizes, ref_ptrs, ref.data(), [&]{
            mnncorrect::Options opt;
            opt.num_neighbors = k;
            opt.automatic_order = false;
            return opt;
        }());
        previous = current;
    }

    mnncorrect::internal::restore_order(ndim, ordering.merge_order, sizes, ref.data());
    EXPECT_EQ(output, ref);
}

TEST_P(OverallTest, Linear) {
    std::vector<double> output(nobs * ndim);
    auto ordering = mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        opt.automatic_order = false;
        return opt;
    }());

    // Checking that the order is as expected.
    EXPECT_EQ(ordering.merge_order.size(), sizes.size());
    EXPECT_EQ(ordering.merge_order[0], 0);
    EXPECT_EQ(ordering.merge_order.back(), sizes.size() - 1);

    // Now trying to run it iteratively.
    size_t previous = 0;
    std::vector<double> ref(nobs * ndim), buffer(nobs * ndim);
    std::vector<const double*> ref_ptrs { ptrs[previous], NULL };
    std::vector<size_t> ref_sizes{ sizes[previous], 0 };

    for (size_t i = 1; i < sizes.size(); ++i) {
        if (i != 1) {
            std::copy(ref.begin(), ref.end(), buffer.begin());
            ref_ptrs[0] = buffer.data();
            ref_sizes[0] += sizes[previous];
        }

        ref_ptrs[1] = ptrs[i];
        ref_sizes[1] = sizes[i];

        mnncorrect::compute(ndim, ref_sizes, ref_ptrs, ref.data(), [&]{
            mnncorrect::Options opt;
            opt.num_neighbors = k;
            opt.automatic_order = false;
            return opt;
        }());
        previous = i;
    }

    EXPECT_EQ(output, ref);
}

TEST_P(OverallTest, Reverse) {
    std::vector<double> output(nobs * ndim);
    auto ordering = mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        opt.automatic_order = false;
        opt.order.resize(sizes.size());
        std::iota(opt.order.begin(), opt.order.end(), 0);
        std::reverse(opt.order.begin(), opt.order.end());
        return opt;
    }());

    // Checking that the order is as expected.
    EXPECT_EQ(ordering.merge_order.size(), sizes.size());
    EXPECT_EQ(ordering.merge_order[0], sizes.size() - 1);
    EXPECT_EQ(ordering.merge_order.back(), 0);

    // Now trying to run it iteratively.
    size_t previous = sizes.size() - 1;
    std::vector<double> ref(nobs * ndim), buffer(nobs * ndim);
    std::vector<const double*> ref_ptrs { ptrs[previous], NULL };
    std::vector<size_t> ref_sizes{ sizes[previous], 0 };

    for (size_t i = 1; i < sizes.size(); ++i) {
        if (i != 1) {
            std::copy(ref.begin(), ref.end(), buffer.begin());
            ref_ptrs[0] = buffer.data();
            ref_sizes[0] += sizes[previous];
        }

        size_t next = sizes.size() - i - 1;
        ref_ptrs[1] = ptrs[next];
        ref_sizes[1] = sizes[next];

        mnncorrect::compute(ndim, ref_sizes, ref_ptrs, ref.data(), [&]{
            mnncorrect::Options opt;
            opt.num_neighbors = k;
            opt.automatic_order = false;
            return opt;
        }());
        previous = next;
    }

    mnncorrect::internal::restore_order(ndim, ordering.merge_order, sizes, ref.data());
    EXPECT_EQ(output, ref);
}

TEST_P(OverallTest, OtherInputs) {
    std::vector<double> output(nobs * ndim);
    auto ordering = mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        return opt;
    }());

    // Just getting some coverage on the other input approach.
    std::vector<double> output2(nobs * ndim);
    auto ordering2 = mnncorrect::compute(ndim, sizes, data.data(), output2.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        return opt;
    }());
    EXPECT_EQ(output, output2);
    EXPECT_EQ(ordering.merge_order, ordering2.merge_order);

    // Creating a mock batch permutation.
    size_t nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (size_t b = 0; b < sizes.size(); ++b) {
        std::fill(bIt, bIt + sizes[b], b);
        bIt += sizes[b];
    }
    std::shuffle(batch.begin(), batch.end(), std::default_random_engine(nobs * sizes.size())); // just varying the seed a bit.

    // Scrambling both the data and the expected results to match the scrambled batches.
    std::vector<size_t> mock_order(sizes.size());
    std::iota(mock_order.begin(), mock_order.end(), 0);

    auto copy = data;
    mnncorrect::internal::restore_order(ndim, mock_order, sizes, batch.data(), copy.data());

    auto ref = output;
    mnncorrect::internal::restore_order(ndim, mock_order, sizes, batch.data(), ref.data());

    // Actually running the test.
    std::vector<double> output3(nobs * ndim);
    auto ordering3 = mnncorrect::compute(ndim, nobs, copy.data(), batch.data(), output3.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        return opt;
    }());
    EXPECT_EQ(ref, output3);
    EXPECT_EQ(ordering.merge_order, ordering3.merge_order);
}

TEST_P(OverallTest, OtherParams) {
    std::vector<double> output(nobs * ndim);
    mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options opt;
        opt.num_neighbors = k;
        return opt;
    }());

    // Trying different options to check they have some effect.    
    {
        std::vector<double> output2(nobs * ndim);
        mnncorrect::compute(ndim, sizes, ptrs, output2.data(), [&]{
            mnncorrect::Options opt;
            opt.num_neighbors = k;
            opt.mass_cap = 50;
            return opt;
        }());
        EXPECT_NE(output2, output);
    }

    {
        std::vector<double> output2(nobs * ndim);
        mnncorrect::compute(ndim, sizes, ptrs, output2.data(), [&]{
            mnncorrect::Options opt;
            opt.num_neighbors = k;
            opt.robust_trim = 0;
            return opt;
        }());
        EXPECT_NE(output2, output);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Overall,
    OverallTest,
    ::testing::Combine(
        ::testing::Values(5), // Number of dimensions
        ::testing::Values(10, 50), // Number of neighbors
        ::testing::Values( // Batch sizes
            std::vector<size_t>{100, 200},        
            std::vector<size_t>{100, 200, 300}, 
            std::vector<size_t>{100, 500, 80}, 
            std::vector<size_t>{60, 300, 100, 80} 
        )
    )
);
