#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/mnncorrect.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <cstddef>

class OverallSimpleSanityTest : public ::testing::TestWithParam<std::tuple<int, std::vector<int>, mnncorrect::MergePolicy> > {};

TEST_P(OverallSimpleSanityTest, Basic) {
    auto param = GetParam();
    const auto k = std::get<0>(param);
    const auto sizes = std::get<1>(param);
    const auto policy = std::get<2>(param);

    const std::size_t ndim = 5;
    auto nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = ndim * k + nobs + static_cast<int>(policy);
        sparams.lower = -2;
        sparams.upper = 2;
        return sparams;
    }());

    const std::size_t nbatches = sizes.size();
    std::vector<mnncorrect::Batch<int> > batches(nbatches);
    constexpr double batch_multiplier = 10;
    int accumulated = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        auto current = data.data() + accumulated * ndim;
        std::size_t len = sizes[b] * ndim;
        for (size_t i = 0; i < len; ++i) { // introducing our own batch effect.
            current[i] += batch_multiplier * b;
        }
        batches[b].start = accumulated;
        batches[b].size = sizes[b];
        accumulated += sizes[b];
    }

    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;
    opt.num_steps = 4; // bumping it up to guarantee a good merge.
    opt.merge_policy = policy;

    std::vector<double> output = data;
    mnncorrect::compute(ndim, batches, output.data(), opt);

    std::size_t refbatch = (policy == mnncorrect::MergePolicy::SIZE ? std::max_element(sizes.begin(), sizes.end()) - sizes.begin() : 0);

    // Heuristic: check that the differences in the mean are much less than the
    // range of simulated values within each batch (-2 to 2) in each dimension.
    std::size_t sofar = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
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
            double expected = refbatch * batch_multiplier;
            double err = std::abs(mean - expected);
            EXPECT_LT(err, 1); // The upper bound on this threshold is 4 (-2 to 2) but we are more stringent here.
        }

        sofar += ndim * num;
    }

    // Check that the reference batch is indeed unchanged.
    const auto ptr = data.data() + batches[refbatch].start * ndim;
    std::vector<double> original(ptr, ptr + batches[refbatch].size * ndim);
    size_t offset = 0;
    for (size_t b = 0; b < refbatch; ++b) {
        offset += sizes[b];
    }
    std::vector<double> corrected(output.begin() + offset * ndim, output.begin() + (offset + sizes[refbatch]) * ndim);
    EXPECT_EQ(original, corrected);

    // Same results when multiple threads are in use.
    std::vector<double> par_output = data;
    mnncorrect::compute(ndim, batches, par_output.data(), [&]{
        mnncorrect::Options<int, double> opt2 = opt;
        opt2.num_threads = 3;
        return opt2;
    }());
    EXPECT_EQ(par_output, output);
}

INSTANTIATE_TEST_SUITE_P(
    Overall,
    OverallSimpleSanityTest,
    ::testing::Combine(
        ::testing::Values(10, 50), // Number of neighbors
        ::testing::Values( // Batch sizes
            std::vector<int>{100, 200},        
            std::vector<int>{100, 200, 300}, 
            std::vector<int>{100, 500, 80}, 
            std::vector<int>{60, 300, 100, 80} 
        ),
        ::testing::Values(
            mnncorrect::MergePolicy::INPUT,
            mnncorrect::MergePolicy::SIZE
        )
    )
);

/***************************************************/

class OverallComplexSanityTest : public ::testing::TestWithParam<std::tuple<int, std::vector<int>, mnncorrect::MergePolicy> > {};

TEST_P(OverallComplexSanityTest, Basic) {
    auto param = GetParam();
    const auto k = std::get<0>(param);
    const auto sizes = std::get<1>(param);
    const auto policy = std::get<2>(param);

    const std::size_t ndim = 6; 
    auto nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 10 + ndim * k + nobs + static_cast<int>(policy);
        sparams.lower = -0.5;
        sparams.upper = 0.5;
        return sparams;
    }());

    constexpr double batch_multiplier = 10, within_multiplier = 20;
    const std::size_t nbatches = sizes.size();
    ASSERT_TRUE(ndim > nbatches); 
    std::vector<mnncorrect::Batch<int> > batches(nbatches);
    int accumulated = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        auto current = data.data() + accumulated * ndim;
        auto len = sizes[b]; 
        for (int c = 0; c < len; ++c) {
            current[c * ndim] += batch_multiplier * b; // first dimension represents the batch effect.
            current[c * ndim + b + 1] += (c % 2 == 1) * within_multiplier; // some other dimension represents within-batch structure, shifted for every second observation.
        }
        batches[b].start = accumulated;
        batches[b].size = len;
        accumulated += len;
    }

    mnncorrect::Options<int, double> opt;
    opt.num_neighbors = k;
    opt.num_steps = 4; // bumping it up to guarantee a good merge.
    opt.merge_policy = policy;
    std::vector<double> output = data;
    mnncorrect::compute(ndim, batches, output.data(), opt);

    std::size_t refbatch = (policy == mnncorrect::MergePolicy::SIZE ? std::max_element(sizes.begin(), sizes.end()) - sizes.begin() : 0);

    // Heuristic: check that the differences in the mean for each common population are much less than the
    // range of simulated values within each batch (-2 to 2) in each dimension.
    // We also check that the unique population is still distinct.
    std::size_t sofar = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        auto len = sizes[b];
        auto ptr = output.data() + sofar;
        std::vector<double> common(ndim);
        std::vector<double> unique(ndim);

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
            EXPECT_LT(err, 1) << b; // The upper bound on this threshold is 4 (-2 to 2) but we are more stringent here.
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

INSTANTIATE_TEST_SUITE_P(
    Overall,
    OverallComplexSanityTest,
    ::testing::Combine(
        ::testing::Values(10, 25), // Number of neighbors
        ::testing::Values( // Batch sizes, note the smallest size must be at least twice the number of neighbors.
            std::vector<int>{100, 200},        
            std::vector<int>{100, 200, 300}, 
            std::vector<int>{100, 500, 80}, 
            std::vector<int>{60, 300, 100, 80} 
        ),
        ::testing::Values(
            mnncorrect::MergePolicy::INPUT,
            mnncorrect::MergePolicy::SIZE
        )
    )
);

/***************************************************/

TEST(Overall, BatchFactorInput) {
    const std::size_t ndim = 3;
    std::vector<int> sizes{ 300, 400, 110 };

    const std::size_t nbatches = sizes.size();
    std::vector<mnncorrect::Batch<int> > batches(nbatches);
    int nobs = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        batches[b].start = nobs;
        batches[b].size = sizes[b];
        nobs += sizes[b];
    }

    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 9999;
        sparams.lower = -0.5;
        sparams.upper = 0.5;
        return sparams;
    }());

    // Check that input order is respected when we go from a batch factor to a vector<Batch>.
    mnncorrect::Options<int, double> opt;
    opt.merge_policy = mnncorrect::MergePolicy::INPUT;

    std::vector<double> ref = data;
    mnncorrect::compute(ndim, batches, ref.data(), opt);

    // Vanilla checks first, where the batch vector is contiguous.
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (std::size_t b = 0; b < nbatches; ++b) {
        std::fill(bIt, bIt + sizes[b], b);
        bIt += sizes[b];
    }

    {
        std::vector<double> output2 = data;
        mnncorrect::compute(ndim, nobs, output2.data(), batch.data(), nbatches, opt);
        EXPECT_EQ(ref, output2);
    }

    // Trying again after shuffling the batch vector.
    {
        std::vector<int> shuffler(nobs);
        std::iota(shuffler.begin(), shuffler.end(), 0);
        std::shuffle(shuffler.begin(), shuffler.end(), std::default_random_engine(nobs * nbatches)); // just varying the seed a bit.
 
        std::vector<double> shuffled_data(data.size());
        std::vector<double> shuffled_ref(data.size());
        std::vector<int> shuffled_batch(nobs);
        for (int o = 0; o < nobs; ++o) {
            const auto chosen = shuffler[o];
            shuffled_batch[o] = batch[chosen];
            std::copy_n(data.begin() + chosen * ndim, ndim, shuffled_data.begin() + o * ndim);
            std::copy_n(ref.begin() + chosen * ndim, ndim, shuffled_ref.begin() + o * ndim);
        }

        mnncorrect::compute(ndim, nobs, shuffled_data.data(), shuffled_batch.data(), nbatches, opt);
        EXPECT_EQ(shuffled_ref, shuffled_data);
    }
}

TEST(Overall, BatchFactorRss) {
    const std::size_t ndim = 7;
    std::vector<int> sizes{ 25, 40, 30, 50 };

    const std::size_t nbatches = sizes.size();
    std::vector<mnncorrect::Batch<int> > batches(nbatches);
    int nobs = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        batches[b].start = nobs;
        batches[b].size = sizes[b];
        nobs += sizes[b];
    }

    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 888;
        sparams.lower = -0.5;
        sparams.upper = 0.5;
        return sparams;
    }());

    mnncorrect::Options<int, double> opt;
    opt.merge_policy = mnncorrect::MergePolicy::RSS;

    std::vector<double> ref = data;
    mnncorrect::compute(ndim, batches, ref.data(), opt);

    // Still contiguous, but the IDs in the batch vector are flipped. 
    // This checks that we still use the contiguous short-circuit.
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (std::size_t b = 0; b < nbatches; ++b) {
        std::fill(bIt, bIt + sizes[b], nbatches - b - 1);
        bIt += sizes[b];
    }

    {
        std::vector<double> output2 = data;
        mnncorrect::compute(ndim, nobs, output2.data(), batch.data(), nbatches, opt);
        EXPECT_EQ(ref, output2);
    }

    // Trying again after shuffling the batch vector.
    {
        std::vector<int> shuffler(nobs);
        std::iota(shuffler.begin(), shuffler.end(), 0);
        std::shuffle(shuffler.begin(), shuffler.end(), std::default_random_engine(nobs * nbatches)); // just varying the seed a bit.
 
        std::vector<double> shuffled_data(data.size());
        std::vector<double> shuffled_ref(data.size());
        std::vector<int> shuffled_batch(nobs);
        for (int o = 0; o < nobs; ++o) {
            const auto chosen = shuffler[o];
            shuffled_batch[o] = batch[chosen];
            std::copy_n(data.begin() + chosen * ndim, ndim, shuffled_data.begin() + o * ndim);
            std::copy_n(ref.begin() + chosen * ndim, ndim, shuffled_ref.begin() + o * ndim);
        }

        mnncorrect::compute(ndim, nobs, shuffled_data.data(), shuffled_batch.data(), nbatches, opt);
        EXPECT_EQ(shuffled_ref, shuffled_data);
    }
}

TEST(Overall, EmptyBatches) {
    const std::size_t ndim = 7;
    std::vector<int> sizes{ 250, 200, 110 };

    const std::size_t nbatches = sizes.size();
    std::vector<mnncorrect::Batch<int> > batches(nbatches);
    int nobs = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        batches[b].start = nobs;
        batches[b].size = sizes[b];
        nobs += sizes[b];
    }

    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 9999;
        sparams.lower = -0.5;
        sparams.upper = 0.5;
        return sparams;
    }());

    // Computing the reference first.
    std::vector<double> ref = data;
    mnncorrect::Options<int, double> opt;
    mnncorrect::compute(ndim, batches, ref.data(), opt);

    // Creating a mock batch vector where every even batch is empty.
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (size_t b = 0; b < nbatches; ++b) {
        std::fill(bIt, bIt + sizes[b], b * 2 + 1);
        bIt += sizes[b];
    }

    std::vector<double> output2 = data;
    mnncorrect::compute(ndim, nobs, output2.data(), batch.data(), nbatches * 2 + 1, opt);
    EXPECT_EQ(ref, output2);
}

TEST(Overall, OtherParams) {
    const std::size_t ndim = 7;
    std::vector<int> sizes{ 100, 50, 300 };

    const std::size_t nbatches = sizes.size();
    std::vector<mnncorrect::Batch<int> > batches(nbatches);
    int nobs = 0;
    for (std::size_t b = 0; b < nbatches; ++b) {
        batches[b].start = nobs;
        batches[b].size = sizes[b];
        nobs += sizes[b];
    }

    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulateVectorParameters sparams;
        sparams.seed = 9999;
        sparams.lower = -0.5;
        sparams.upper = 0.5;
        return sparams;
    }());

    // Computing the reference first.
    std::vector<double> ref = data;
    mnncorrect::Options<int, double> opt;
    mnncorrect::compute(ndim, batches, ref.data(), opt);

    // Trying different options to check they have some effect.
    // First, the number of steps:
    {
        std::vector<double> output2 = data;
        mnncorrect::compute(ndim, batches, output2.data(), [&]{
            auto opt2 = opt;
            opt2.num_steps = 2;
            return opt2;
        }());
        EXPECT_NE(ref, output2);
    }

    // Trying a different distance metric.
    {
        std::vector<double> output2 = data;
        mnncorrect::compute(ndim, batches, output2.data(), [&]{
            auto opt2 = opt;
            opt2.builder.reset(new knncolle::VptreeBuilder<int, double, double>(std::make_shared<knncolle::ManhattanDistance<double, double> >()));
            return opt2;
        }());
        EXPECT_NE(ref, output2);
    }
}
