#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/reorder_matrix_in_place.hpp"

#include <numeric>
#include <algorithm>
#include <random>
#include <cstddef>

class ReorderMatrixInPlaceTest : public ::testing::TestWithParam<int> {};

TEST_P(ReorderMatrixInPlaceTest, Basic) {
    int ndim = 5;
    int nobs = GetParam();
    auto data = scran_tests::simulate_vector(ndim * nobs, scran_tests::SimulateVectorParameters());

    // Creating a mock permuted dataset.
    std::vector<int> ordering(nobs);
    std::iota(ordering.begin(), ordering.end(), 0);

    std::mt19937_64 rng(12398 + nobs);
    std::shuffle(ordering.begin(), ordering.end(), rng);

    std::vector<double> expected;
    expected.reserve(data.size());
    for (int o = 0; o < nobs; ++o) {
        auto dIt = data.begin() + ndim * ordering[o];
        expected.insert(expected.end(), dIt, dIt + ndim);
    }

    std::vector<double> mbuffer(ndim);
    mnncorrect::reorder_matrix_in_place(ndim, nobs, ordering, data.data(), mbuffer);
    EXPECT_EQ(data, expected);
}

INSTANTIATE_TEST_SUITE_P(
    ReorderMatrixInPlace,
    ReorderMatrixInPlaceTest,
    ::testing::Values(154, 237, 32)
);
