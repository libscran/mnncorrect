#include <gtest/gtest.h>
#include "mnncorrect/restore_order.hpp"
#include <numeric>
#include <algorithm>
#include <random>

class RestoreOrderTest : public ::testing::TestWithParam<std::tuple<std::vector<int>, std::vector<size_t> > > {
protected:
    template<class Param>
    void assemble(Param param) {
        merge_order = std::get<0>(param);
        sizes = std::get<1>(param);

        size_t nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
        data.resize(nobs * ndim);

        // Creating a mock permuted dataset.
        auto ptr = data.data();
        for (auto b : merge_order) {
            for (size_t i = 0; i < sizes[b]; ++i) {
                std::iota(ptr, ptr + ndim, init(b, i));
                ptr += ndim;
            }
        }

        return;
    }

    size_t init(size_t batch, size_t index) {
        return (batch + 1) * (index + 1);
    } 

    int ndim = 5;
    std::vector<int> merge_order;
    std::vector<size_t> sizes;
    std::vector<double> data;
};

TEST_P(RestoreOrderTest, Simple) {
    assemble(GetParam());
    mnncorrect::restore_order(ndim, merge_order, sizes, data.data());

    const double* ptr = data.data();
    for (size_t b = 0; b < sizes.size(); ++b) {
        for (size_t i = 0; i < sizes[b]; ++i) {
            std::vector<double> ref(ndim);
            std::iota(ref.begin(), ref.end(), init(b, i));
            std::vector<double> obs(ptr, ptr + ndim);
            EXPECT_EQ(ref, obs);
            ptr += ndim;
        }
    }
}

TEST_P(RestoreOrderTest, Batch) {
    assemble(GetParam());

    // Creating a mock batch permutation.
    size_t nobs = std::accumulate(sizes.begin(), sizes.end(), 0);
    std::vector<int> batch(nobs);
    auto bIt = batch.begin();
    for (size_t b = 0; b < sizes.size(); ++b) {
        std::fill(bIt, bIt + sizes[b], b);
        bIt += sizes[b];
    }
    std::shuffle(batch.begin(), batch.end(), std::default_random_engine(nobs * sizes.size())); // just varying the seed a bit.

    mnncorrect::restore_order(ndim, merge_order, sizes, batch.data(), data.data());

    const double* ptr = data.data();
    std::vector<size_t> sofar(sizes.size());
    for (size_t o = 0; o < nobs; ++o) {
        size_t b = batch[o];
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
        std::make_tuple(std::vector<int>{0, 1}, std::vector<size_t>{ 50, 20 }),
        std::make_tuple(std::vector<int>{0, 1, 2}, std::vector<size_t>{ 10, 20, 30 }),
        std::make_tuple(std::vector<int>{2, 0, 1}, std::vector<size_t>{ 9, 11, 7 }),
        std::make_tuple(std::vector<int>{2, 1, 3, 0}, std::vector<size_t>{ 5, 2, 9, 4 })
    )
);


