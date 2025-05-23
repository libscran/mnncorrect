#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/mnncorrect.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <cstddef>

class OverallTest : public ::testing::TestWithParam<std::tuple<int, std::vector<int> > > {
protected:
    constexpr static double multiplier = 10;
    constexpr static std::size_t ndim = 5;

    void SetUp() {
        auto param = GetParam();
        k = std::get<0>(param);
        sizes = std::get<1>(param);

        nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
        data = scran_tests::simulate_vector(nobs * ndim, [&]{
            scran_tests::SimulationParameters sparams;
            sparams.seed = ndim * k + nobs;
            sparams.lower = -2;
            sparams.upper = 2;
            return sparams;
        }());

        ptrs.resize(sizes.size());
        std::size_t sofar = 0;
        for (std::size_t b = 0, bend = sizes.size(); b < bend; ++b) {
            auto current = data.data() + sofar;
            std::size_t len = sizes[b] * ndim;
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
    int nobs, k;
    std::vector<int> sizes;

    // Simulated.
    std::vector<double> data;
    std::vector<const double*> ptrs;
};

TEST_P(OverallTest, Basic) {
    mnncorrect::Options<int, double> opt;
    opt.merge_policy = mnncorrect::MergePolicy::INPUT;
    opt.num_neighbors = k;
    opt.num_steps = 4; // bumping it up to guarantee a good merge.

    std::vector<double> output(nobs * ndim);
    mnncorrect::compute(ndim, sizes, ptrs, output.data(), opt);

    // Reference batch is the first, as we set an INPUT policy.
    size_t refbatch = 0;

    // Heuristic: check that the differences in the mean are much less than the
    // range of simulated values within each batch (-2 to 2) in each dimension.
    std::size_t sofar = 0;
    for (std::size_t b = 0, bend = sizes.size(); b < bend; ++b) {
        auto ptr = output.data() + sofar;
        std::vector<double> ref(ndim);

        auto num = sizes[b];
        for (int s = 0; s < num; ++s) {
            for (std::size_t d = 0; d < ndim; ++d) {
                ref[d] += ptr[d];                
            }
            ptr += ndim;
        }

        for (auto r : ref) {
            auto mean = r / sizes[b];
            double expected = refbatch * multiplier;
            double err = std::abs(mean - expected);
            EXPECT_LT(err, 1); // The upper bound on this threshold is 4 (-2 to 2) but we are more stringent here.
        }

        sofar += ndim * num;
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
    mnncorrect::compute(ndim, sizes, ptrs, par_output.data(), [&]{
        mnncorrect::Options<int, double> opt2 = opt;
        opt2.num_threads = 3;
        return opt2;
    }());
    EXPECT_EQ(par_output, output);
}

TEST_P(OverallTest, OtherInputs) {
    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;

    std::vector<double> output(nobs * ndim);
    mnncorrect::compute(ndim, sizes, ptrs, output.data(), opt);

    {
        // Getting some coverage on the other input approach.
        std::vector<double> output2(nobs * ndim);
        mnncorrect::compute(ndim, sizes, data.data(), output2.data(), opt);
        EXPECT_EQ(output, output2);
    }

    // Creating a mock batch vector.
    int nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (size_t b = 0; b < sizes.size(); ++b) {
        std::fill(bIt, bIt + sizes[b], b);
        bIt += sizes[b];
    }

    {
        // Vanilla checks first, where the batch vector is ordered.
        std::vector<double> output3(nobs * ndim);
        mnncorrect::compute(ndim, nobs, data.data(), batch.data(), output3.data(), opt);
        EXPECT_EQ(output, output3);
    }

    // Trying again after shuffling the batch vector.
    std::shuffle(batch.begin(), batch.end(), std::default_random_engine(nobs * sizes.size())); // just varying the seed a bit.
    {
        // Scrambling both the data and the expected results to match the scrambled batches.
        auto copy = data;
        mnncorrect::internal::restore_input_order(ndim, sizes, batch.data(), copy.data());
        auto ref = output;
        mnncorrect::internal::restore_input_order(ndim, sizes, batch.data(), ref.data());

        std::vector<double> output3(nobs * ndim);
        mnncorrect::compute(ndim, nobs, copy.data(), batch.data(), output3.data(), opt);
        EXPECT_EQ(ref, output3);
    }
}

TEST_P(OverallTest, OtherParams) {
    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;

    std::vector<double> output(nobs * ndim);
    mnncorrect::compute(ndim, sizes, ptrs, output.data(), opt);

    // Trying different options to check they have some effect.    
    {
        std::vector<double> output2(nobs * ndim);
        mnncorrect::compute(ndim, sizes, ptrs, output2.data(), [&]{
            auto opt2 = opt;
            opt2.num_steps = 2;
            return opt2;
        }());
        EXPECT_NE(output2, output);
    }

    {
        std::vector<double> output2(nobs * ndim);
        mnncorrect::compute(ndim, sizes, ptrs, output2.data(), [&]{
            auto opt2 = opt;
            opt2.builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::ManhattanDistance<double, double> >()));
            return opt2;
        }());
        EXPECT_NE(output2, output);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Overall,
    OverallTest,
    ::testing::Combine(
        ::testing::Values(10, 50), // Number of neighbors
        ::testing::Values( // Batch sizes
            std::vector<int>{100, 200},        
            std::vector<int>{100, 200, 300}, 
            std::vector<int>{100, 500, 80}, 
            std::vector<int>{60, 300, 100, 80} 
        )
    )
);

TEST(Overall, Sanity) {
    const std::size_t ndim = 4;
    std::vector<int> sizes{ 300, 400, 110 };
    auto nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulationParameters sparams;
        sparams.seed = 9999;
        sparams.lower = -0.5;
        sparams.upper = 0.5;
        return sparams;
    }());

    constexpr double batch_multiplier = 10, within_multiplier = 20;
    std::vector<const double*> ptrs(sizes.size());
    std::size_t sofar = 0;
    for (std::size_t b = 0, bend = sizes.size(); b < bend; ++b) {
        auto current = data.data() + sofar;
        auto len = sizes[b]; 
        for (int c = 0; c < len; ++c) {
            current[c * ndim] += batch_multiplier * b; // first dimension represents the batch effect.
            current[c * ndim + b + 1] += (c % 2 == 1) * within_multiplier; // some other dimension represents within-batch structure, shifted for every second observation.
        }
        ptrs[b] = current;
        sofar += ndim * len;
    }

    std::vector<double> output(ndim * nobs);
    mnncorrect::compute(ndim, sizes, ptrs, output.data(), [&]{
        mnncorrect::Options<int, double> opt;
        opt.num_steps = 4; // bumping it up to guarantee a good merge.
        return opt;
    }());

    size_t refbatch = 1; // highest RSS, as it has the most observations.
    sofar = 0;
    for (std::size_t b = 0, bend = sizes.size(); b < bend; ++b) {
        auto len = sizes[b];
        auto ptr = output.data() + sofar;
        std::vector<double> common(ndim);
        std::vector<double> unique(ndim);

        // Check that the differences in the mean for each common population are much less than the
        // range of simulated values within each batch (-2 to 2) in each dimension.
        for (int s = 0; s < len; ++s) {
            auto cptr = (s % 2 == 0 ? common.data() : unique.data());
            for (std::size_t d = 0; d < ndim; ++d) {
                cptr[d] += ptr[d];                
            }
            ptr += ndim;
        }

        for (std::size_t d = 0; d < ndim; ++d) {
            double expected = 0;
            if (d == 0) {
                expected = refbatch * batch_multiplier;
            }
            auto mean = common[d]/(len/2.0);
            double err = std::abs(mean - expected);
            EXPECT_LT(err, 1); // The upper bound on this threshold is 4 (-2 to 2) but we are more stringent here.
        }

        for (std::size_t d = 0; d < ndim; ++d) {
            double expected = 0;
            if (d == 0) {
                expected = refbatch * batch_multiplier;
            } else if (static_cast<std::size_t>(d) == b + 1) {
                expected = within_multiplier;
            }
            auto mean = unique[d]/(len/2.0);
            double err = std::abs(mean - expected);
            EXPECT_LT(err, 1); // The upper bound on this threshold is 4 (-2 to 2) but we are more stringent here.
        }

        sofar += ndim * len;
    }
}
