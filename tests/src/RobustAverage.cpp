#include "gtest/gtest.h"
#include "mnncorrect/RobustAverage.hpp"
#include "aarand/aarand.hpp"
#include <vector>
#include <numeric>
#include <random>

TEST(RobustAverageTest, Basic) {
    std::vector<double> data { 0.1, 0.5, 0.2, 0.9, 0.12 };

    // Simple mean if there are no robustness iterations.
    {
        mnncorrect::RobustAverage<int, double> test(0, 0.25);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_EQ(output, std::accumulate(data.begin(), data.end(), 0.0)/data.size());
    }

    // Keeping the mean of the closest three observations to the first mean.
    {
        mnncorrect::RobustAverage<int, double> test(1, 0.3);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, (0.5 + 0.2 + 0.12) / 3.0);
    }

    // This can be iterated, in which case the closest three observations changes.
    {
        mnncorrect::RobustAverage<int, double> test(2, 0.3);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, (0.1 + 0.2 + 0.12) / 3.0);
    }

    // With a trim of 0.25, we remove exactly one observation, as we keep the value at the third quartile.
    {
        mnncorrect::RobustAverage<int, double> test(1, 0.25);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, (0.1 + 0.5 + 0.2 + 0.12) / 4.0);
    }

    // With a trim of 0.5, we make sure we keep the median.
    {
        mnncorrect::RobustAverage<int, double> test(1, 0.5);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, (0.5 + 0.2 + 0.12) / 3.0);
    }
}

TEST(RobustAverageTest, EdgeCases) {
    std::vector<double> data { 0.1, 0.5, 0.2, 0.9, 0.12 };

    // Taking the average if the trim is zero.
    {
        mnncorrect::RobustAverage<int, double> test(1, 0);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, std::accumulate(data.begin(), data.end(), 0.0)/data.size());
    }

    // With a trim of 1, we keep the closest observation only.
    {
        mnncorrect::RobustAverage<int, double> test(1, 1);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, 0.5);
    }

    // Doing the right things with only one observation.
    {
        mnncorrect::RobustAverage<int, double> test(1, 1);
        double output;
        test.run(1, 1, data.data(), &output);
        EXPECT_FLOAT_EQ(output, 0.1);
    }
    {
        mnncorrect::RobustAverage<int, double> test(1, 0);
        double output;
        test.run(1, 1, data.data(), &output);
        EXPECT_FLOAT_EQ(output, 0.1);
    }
}

TEST(RobustAverageTest, Ties) {
    std::vector<double> data { 1, 2, 3, 4, 5 };

    // This should remove the furthest element, but as it's tied, we don't remove anything.
    {
        mnncorrect::RobustAverage<int, double> test(1, 0.1);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, 3);
    }

    // This should remove 3 elements, but we only remove the furthest 2 due to the tie.
    {
        mnncorrect::RobustAverage<int, double> test(1, 0.6);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, 3);
    }

    // With only two elements, both of them should be tied to the mean and
    // never removed.  We use lots of significant figures to ensure that the
    // tolerance mechanisms are working correctly.
    {
        std::vector<double> data { .28376783287177263475, .43984537534872874 };
        mnncorrect::RobustAverage<int, double> test(1, 0.5);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, (data[0] + data[1]) / 2);
    }
    
    {
        std::vector<double> data { 0.6363161874823, 10.2347625487411981 };
        mnncorrect::RobustAverage<int, double> test(1, 0.5);
        double output;
        test.run(1, data.size(), data.data(), &output);
        EXPECT_FLOAT_EQ(output, (data[0] + data[1]) / 2);
    }
}

class RobustAverageTest : public ::testing::TestWithParam<std::tuple<int, int, double> > {
protected:
    template<class Param>
    void assemble(Param param, int ndim) {
        nobs = std::get<0>(param);
        iterations = std::get<1>(param);
        trim = std::get<2>(param);

        std::mt19937_64 rng(nobs);
        data.resize(nobs* ndim);
        for (auto& l : data) {
            l = aarand::standard_normal(rng).first;
        }
    }

    size_t nobs;
    int iterations;
    double trim;
    std::vector<double> data;
};

TEST_P(RobustAverageTest, Multidimensional) {
    assemble(GetParam(), 1);

    int ndim = 3;
    std::vector<double> copy (ndim * nobs);
    for (int i = 0; i < nobs; ++i) {
        for (int d = 0; d < ndim; ++d) {
            copy[d + i * ndim] = data[i] + d;
        }
    }

    mnncorrect::RobustAverage<int, double> test(iterations, trim);
    double output;
    test.run(1, nobs, data.data(), &output);

    std::vector<double> output_vector(ndim);
    test.run(ndim, nobs, copy.data(), output_vector.data());

    for (int d = 0; d < ndim; ++d) {
        EXPECT_FLOAT_EQ(output_vector[d], output +d);
    }
}

TEST_P(RobustAverageTest, Indexed) {
    int ndim = 5;
    assemble(GetParam(), ndim);

    // Indexed calculation.
    std::mt19937_64 rng(nobs * iterations * trim);
    std::vector<int> indices;
    for (int i = 0; i < nobs; ++i) {
        indices.push_back(aarand::discrete_uniform(rng, nobs));
    }

    mnncorrect::RobustAverage<int, double> test(iterations, trim);
    std::vector<double> output(ndim);
    test.run(ndim, indices, data.data(), output.data()); 

    // Reference calculation.
    std::vector<double> copy(ndim * indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        auto target = data.data() + indices[i] * ndim;
        std::copy(target, target + ndim, copy.begin() + i * ndim);
    }

    std::vector<double> output2(ndim);
    test.run(ndim, indices.size(), copy.data(), output2.data()); 
    EXPECT_EQ(output, output2);
}

INSTANTIATE_TEST_CASE_P(
    RobustAverage,
    RobustAverageTest,
    ::testing::Combine(
        ::testing::Values(5, 13, 50), // number of observations.
        ::testing::Values(0, 1, 2), // number of iterations 
        ::testing::Values(0.1, 0.3) // trim proportion.
    )
);


