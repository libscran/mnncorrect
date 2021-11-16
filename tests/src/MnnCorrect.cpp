#include <gtest/gtest.h>
#include "mnncorrect/MnnCorrect.hpp"
#include <random>
#include <algorithm>
#include <cmath>

class MnnCorrectTest : public ::testing::TestWithParam<std::tuple<int, int, std::vector<size_t> > > {
protected:
    constexpr static double multiplier = 10;

    template<class Param>
    void assemble(Param param) {
        // Simulating values.
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist;

        ndim = std::get<0>(param);
        k = std::get<1>(param);
        sizes = std::get<2>(param);

        nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
        data.resize(nobs * ndim);
        ptrs.resize(sizes.size());

        auto ptr = data.data();
        for (size_t b = 0; b < sizes.size(); ++b) {
            ptrs[b] = ptr;
            for (size_t s = 0; s < sizes[b]; ++s) {
                for (int d = 0; d < ndim; ++d) {
                    *ptr = dist(rng) + multiplier * b;
                    ++ptr;
                }
            }
        }

        return;
    }

    int ndim, nobs, k;
    std::vector<size_t> sizes;
    std::vector<double> data;
    std::vector<const double*> ptrs;
};

TEST_P(MnnCorrectTest, Basic) {
    assemble(GetParam());

    mnncorrect::MnnCorrect<> mnnrun;
    mnnrun.set_num_neighbors(k).set_num_clusters(1); // setting to a single cluster for the 'perfect' correction.

    std::vector<double> output(nobs * ndim);
    auto ordering = mnnrun.run(ndim, sizes, ptrs, output.data());
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
}

TEST_P(MnnCorrectTest, OtherInputs) {
    assemble(GetParam());

    mnncorrect::MnnCorrect<> mnnrun;
    mnnrun.set_num_neighbors(k);
    std::vector<double> output(nobs * ndim);
    auto ordering = mnnrun.run(ndim, sizes, ptrs, output.data());

    // Just getting some coverage on the other input approach.
    std::vector<double> output2(nobs * ndim);
    auto ordering2 = mnnrun.run(ndim, sizes, data.data(), output2.data());
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
    std::vector<int> mock_order(sizes.size());
    std::iota(mock_order.begin(), mock_order.end(), 0);

    auto copy = data;
    mnncorrect::restore_order(ndim, mock_order, sizes, batch.data(), copy.data());

    auto ref = output;
    mnncorrect::restore_order(ndim, mock_order, sizes, batch.data(), ref.data());

    // Actually running the test.
    std::vector<double> output3(nobs * ndim);
    auto ordering3 = mnnrun.run(ndim, nobs, copy.data(), batch.data(), output3.data());
    EXPECT_EQ(ref, output3);
    EXPECT_EQ(ordering.merge_order, ordering3.merge_order);
}

INSTANTIATE_TEST_CASE_P(
    MnnCorrect,
    MnnCorrectTest,
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
