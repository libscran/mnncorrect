#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/restore_order.hpp"

#include <numeric>
#include <algorithm>
#include <random>
#include <cstddef>

class RestoreOrderTest : public ::testing::TestWithParam<std::tuple<std::vector<std::size_t>, std::vector<int> > > {
protected:
    static std::size_t init(std::size_t batch, std::size_t index) {
        return (batch + 1) * (index + 1);
    } 
};

TEST_P(RestoreOrderTest, Simple) {
    auto param = GetParam();
    auto merge_order = std::get<0>(param);
    auto sizes = std::get<1>(param);

    std::size_t ndim = 5;
    int nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<double> data(nobs * ndim);

    // Creating a mock permuted dataset.
    {
        auto ptr = data.data();
        for (auto b : merge_order) {
            for (int i = 0; i < sizes[b]; ++i) {
                std::iota(ptr, ptr + ndim, init(b, i));
                ptr += ndim;
            }
        }
    }

    mnncorrect::internal::restore_order(ndim, merge_order, sizes, data.data());

    const double* ptr = data.data();
    for (std::size_t b = 0, bend = sizes.size(); b < bend; ++b) {
        for (int i = 0; i < sizes[b]; ++i) {
            std::vector<double> ref(ndim);
            std::iota(ref.begin(), ref.end(), init(b, i));
            std::vector<double> obs(ptr, ptr + ndim);
            EXPECT_EQ(ref, obs);
            ptr += ndim;
        }
    }
}

TEST_P(RestoreOrderTest, Batch) {
    auto param = GetParam();
    auto merge_order = std::get<0>(param);
    auto sizes = std::get<1>(param);

    std::size_t ndim = 5;
    int nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<double> data(nobs * ndim);

    // Creating a mock permuted dataset.
    {
        auto ptr = data.data();
        for (auto b : merge_order) {
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
    std::shuffle(batch.begin(), batch.end(), std::default_random_engine(nobs * sizes.size())); // just varying the seed a bit.

    mnncorrect::internal::restore_order(ndim, merge_order, sizes, batch.data(), data.data());

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
    RestoreOrder,
    RestoreOrderTest,
    ::testing::Values(
        std::make_tuple(std::vector<std::size_t>{0, 1}, std::vector<int>{ 50, 20 }),
        std::make_tuple(std::vector<std::size_t>{0, 1, 2}, std::vector<int>{ 10, 20, 30 }),
        std::make_tuple(std::vector<std::size_t>{2, 0, 1}, std::vector<int>{ 9, 11, 7 }),
        std::make_tuple(std::vector<std::size_t>{2, 1, 3, 0}, std::vector<int>{ 5, 2, 9, 4 })
    )
);
