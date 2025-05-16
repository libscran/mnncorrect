#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/restore_input_order.hpp"

#include <numeric>
#include <algorithm>
#include <random>
#include <cstddef>

class RestoreInputOrderTest : public ::testing::TestWithParam<std::vector<int> > {
protected:
    static std::size_t init(std::size_t batch, std::size_t index) {
        return (batch + 1) * (index + 1);
    } 
};

TEST_P(RestoreInputOrderTest, Basic) {
    auto sizes = GetParam();
    std::size_t num_batches = sizes.size();

    std::size_t ndim = 5;
    int nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<double> data(nobs * ndim);

    // Creating a mock permuted dataset.
    {
        auto ptr = data.data();
        for (decltype(num_batches) b = 0; b < num_batches; ++b) {
            for (int i = 0; i < sizes[b]; ++i) {
                std::iota(ptr, ptr + ndim, init(b, i));
                ptr += ndim;
            }
        }
    }

    // Creating a mock batch permutation.
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (std::size_t b = 0; b < sizes.size(); ++b) {
        std::fill(bIt, bIt + sizes[b], b);
        bIt += sizes[b];
    }
    std::shuffle(batch.begin(), batch.end(), std::default_random_engine(nobs * 10 + num_batches)); // just varying the seed a bit.

    mnncorrect::internal::restore_input_order(ndim, sizes, batch.data(), data.data());

    const double* ptr = data.data();
    std::vector<int> sofar(sizes.size());
    for (int o = 0; o < nobs; ++o) {
        std::size_t b = batch[o];
        std::vector<double> ref(ndim);
        std::iota(ref.begin(), ref.end(), init(b, sofar[b]));
        std::vector<double> obs(ptr, ptr + ndim);
        EXPECT_EQ(ref, obs);
        ptr += ndim;
        ++sofar[b];
    }
}

INSTANTIATE_TEST_SUITE_P(
    RestoreInputOrder,
    RestoreInputOrderTest,
    ::testing::Values(
        std::vector<int>{ 50, 20 },
        std::vector<int>{ 10, 20, 30 },
        std::vector<int>{ 9, 11, 7 },
        std::vector<int>{ 5, 2, 9, 4 }
    )
);
