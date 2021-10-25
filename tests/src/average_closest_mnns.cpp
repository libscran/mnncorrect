#include <gtest/gtest.h>
#include "mnncorrect/average_closest_mnns.hpp"
#include <random>
#include <vector>
#include <algorithm>

class AverageClosestMNNsTest : public ::testing::Test {
protected:
    std::vector<double> create_data(int ndim, size_t nobs) {
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist;
        std::vector<double> data(nobs* ndim);
        for (auto& d : data) {
            d = dist(rng);
        }
        return data;
    }
};

TEST_F(AverageClosestMNNsTest, AverageCheck) {
    size_t nobs = 100;
    int ndim = 4;
    auto data = create_data(nobs, ndim);

    // Creating 10 batches of 10 observations each. We shift each batch by a
    // large amount so that each batch is independent of the others when
    // considering NNs. 
    auto ptr = data.data();
    for (size_t b = 0; b < 10; ++b) {
        for (size_t r = 0; r < 10; ++r) {
            for (int d = 0; d < ndim; ++d, ++ptr) {
                *ptr += b * 10;
            }
        }
    }

    // Computing averages.
    std::vector<double> averaged(nobs * ndim);
    auto aptr = averaged.data();
    ptr = data.data();
    for (size_t b = 0; b < 10; ++b) {
        for (size_t r = 0; r < 10; ++r) {
            for (int d = 0; d < ndim; ++d, ++ptr) {
                aptr[d] += *ptr;
            }
        }
        for (int d = 0; d < ndim; ++d) {
            aptr[d] /= 10;
        }
        for (size_t r = 1; r < 10; ++r) {
            std::copy(aptr, aptr + ndim, aptr + r * ndim);
        }
        aptr += 10 * ndim;
    }

    // Comparing the values.
    std::deque<size_t> involved(nobs);
    std::iota(involved.begin(), involved.end(), 0);
    std::vector<double> output(nobs * ndim);

    mnncorrect::AverageClosestMNNs<double> averager;
    averager.find(ndim, nobs, data.data(), involved, 10, output.data());
   
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], averaged[i]);
    }
}

TEST_F(AverageClosestMNNsTest, FilterCheck) {
    size_t nobs = 100;
    int ndim = 4;
    auto data = create_data(nobs, ndim);

    std::vector<double> half(ndim * 50);
    for (size_t o = 0; o < 50; ++o) {
        auto start = data.data() + o * 2 * ndim;
        std::copy(start, start + ndim, half.data() + o * ndim);
    }

    // Computing the all-to-all matches for the first half.
    mnncorrect::AverageClosestMNNs<double> averager;

    std::vector<double> output(half.size());
    {
        std::deque<size_t> involved(50);
        std::iota(involved.begin(), involved.end(), 0);
        averager.find(ndim, 50, half.data(), involved, 5, output.data());
    }

    // Computing this again with a full run.
    std::vector<double> ref(data.size());
    {
        std::deque<size_t> involved;
        for (size_t o = 0; o < 50; ++o) {
            involved.push_back(2 * o);
        }
        averager.find(ndim, nobs, data.data(), involved, 5, ref.data());
    }

    for (size_t o = 0; o < 50; ++o) {
        auto curref = ref.data() + o * 2 * ndim;
        auto curout = output.data() + o * ndim;
        for (int d = 0; d < ndim; ++d) {
            EXPECT_FLOAT_EQ(curref[d], curout[d]);
        }
    }
}

TEST_F(AverageClosestMNNsTest, AnotherFilterCheck) {
    size_t nobs = 100;
    int ndim = 4;
    auto data = create_data(nobs, ndim);

    for (size_t o = 0; o < 50; ++o) {
        auto start = data.data() + o * 2 * ndim;
        std::copy(start, start + ndim, start + ndim); // clone every second observation
    }

    // Computing the all-to-all matches for the first half.
    mnncorrect::AverageClosestMNNs<double> averager;

    std::vector<double> output(data.size());
    std::deque<size_t> involved;
    for (size_t o = 0; o < 50; ++o) {
        involved.push_back(2 * o);
    }
    averager.find(ndim, nobs, data.data(), involved, 5, output.data());

    // Checking that the cloned entries have the same averages.
    for (size_t o = 0; o < 50; ++o) {
        auto curref = output.data() + o * 2 * ndim;
        auto curout = output.data() + o * 2 * ndim + ndim;
        for (int d = 0; d < ndim; ++d) {
            EXPECT_FLOAT_EQ(curref[d], curout[d]);
        }
    }
}

TEST_F(AverageClosestMNNsTest, Duplicates) {
    size_t nobs = 100;
    int ndim = 4;
    auto data = create_data(nobs, ndim);

    mnncorrect::AverageClosestMNNs<double> averager;

    std::vector<double> output(data.size());
    std::deque<size_t> involved1, involved2;
    for (size_t o = 0; o < 50; ++o) {
        involved1.push_back(2 * o);
        involved2.insert(involved2.end(), o % 5 + 1, 2 * o);
    }

    std::vector<double> output1(data.size()), output2(data.size());
    averager.find(ndim, nobs, data.data(), involved1, 5, output1.data());
    averager.find(ndim, nobs, data.data(), involved2, 5, output2.data());

    EXPECT_EQ(output1, output2);
}
