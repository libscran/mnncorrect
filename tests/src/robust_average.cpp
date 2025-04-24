#include "scran_tests/scran_tests.hpp"

#include "custom_parallel.h" // Must be before any mnncorrect includes.

#include "mnncorrect/robust_average.hpp"

#include <vector>
#include <numeric>
#include <random>

std::vector<double> robust_average(int num_dim, int num_pts, const double* data, const mnncorrect::internal::RobustAverageOptions& options) {
    std::vector<double> output(num_dim);
    std::vector<std::pair<double, std::size_t> > deltas;
    mnncorrect::internal::robust_average(num_dim, num_pts, data, output.data(), deltas, options);
    return output;
}

std::vector<double> robust_average(int num_dim, const std::vector<int>& indices, const double* data, const mnncorrect::internal::RobustAverageOptions& options) {
    std::vector<double> output(num_dim);
    std::vector<std::pair<double, std::size_t> > deltas;
    robust_average(num_dim, indices, data, output.data(), deltas, options);
    return output;
}

TEST(RobustAverageTest, Basic) {
    std::vector<double> data { 0.1, 0.5, 0.2, 0.9, 0.12 };

    // Simple mean if there are no robustness iterations.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(0, 0.25))[0];
        EXPECT_FLOAT_EQ(output, std::accumulate(data.begin(), data.end(), 0.0)/data.size());
    }

    // Keeping the mean of the closest three observations to the first mean.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.3))[0];
        EXPECT_FLOAT_EQ(output, (0.5 + 0.2 + 0.12) / 3.0);
    }

    // This can be iterated, in which case the closest three observations changes.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(2, 0.3))[0];
        EXPECT_FLOAT_EQ(output, (0.1 + 0.2 + 0.12) / 3.0);
    }

    // With a trim of 0.25, we remove exactly one observation, as we keep the value at the third quartile.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.25))[0];
        EXPECT_FLOAT_EQ(output, (0.1 + 0.5 + 0.2 + 0.12) / 4.0);
    }

    // With a trim of 0.5, we make sure we keep the median.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.5))[0];
        EXPECT_FLOAT_EQ(output, (0.5 + 0.2 + 0.12) / 3.0);
    }
}

TEST(RobustAverageTest, Persistence) {
    std::vector<double> data { 0.1, 0.5, 0.2, 0.9, 0.12 };

    std::vector<std::pair<double, std::size_t> > deltas;
    deltas.emplace_back(0.9, 1);
    deltas.emplace_back(0.99, 10);
    deltas.emplace_back(0.9999, -5);
    deltas.emplace_back(0.01, 100);
    deltas.emplace_back(0.05, 6);

    double ref;
    mnncorrect::internal::RobustAverageOptions raopt(0, 0.25);
    mnncorrect::internal::robust_average(1, data.size(), data.data(), &ref, deltas, raopt);

    // Gunk in the delta buffer is of no consequence.
    double output;
    mnncorrect::internal::robust_average(1, data.size(), data.data(), &output, deltas, raopt);

    EXPECT_EQ(ref, output);
}

TEST(RobustAverageTest, EdgeCases) {
    std::vector<double> data { 0.1, 0.5, 0.2, 0.9, 0.12 };

    // Taking the average if the trim is zero.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0))[0];
        EXPECT_FLOAT_EQ(output, std::accumulate(data.begin(), data.end(), 0.0)/data.size());
    }

    // With a trim of 1, we keep the closest observation only.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 1))[0];
        EXPECT_FLOAT_EQ(output, 0.5);
    }

    // Doing the right things with only one observation.
    {
        double output = robust_average(1, 1, data.data(), mnncorrect::internal::RobustAverageOptions(1, 1))[0];
        EXPECT_FLOAT_EQ(output, 0.1);
    }

    {
        double output = robust_average(1, 1, data.data(), mnncorrect::internal::RobustAverageOptions(1, 0))[0];
        EXPECT_FLOAT_EQ(output, 0.1);
    }
}

TEST(RobustAverageTest, Ties) {
    std::vector<double> data { 1, 2, 3, 4, 5 };

    // This should remove the furthest element, but as it's tied, we don't remove anything.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.1))[0];
        EXPECT_FLOAT_EQ(output, 3);
    }

    // This should remove 3 elements, but we only remove the furthest 2 due to the tie.
    {
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.6))[0];
        EXPECT_FLOAT_EQ(output, 3);
    }

    // With only two elements, both of them should be tied to the mean and
    // never removed.  We use lots of significant figures to ensure that the
    // tolerance mechanisms are working correctly.
    {
        std::vector<double> data { .28376783287177263475, .43984537534872874 };
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.5))[0];
        EXPECT_FLOAT_EQ(output, (data[0] + data[1]) / 2);
    }
    
    {
        std::vector<double> data { 0.6363161874823, 10.2347625487411981 };
        double output = robust_average(1, data.size(), data.data(), mnncorrect::internal::RobustAverageOptions(1, 0.5))[0];
        EXPECT_FLOAT_EQ(output, (data[0] + data[1]) / 2);
    }
}

class RobustAverageTest : public ::testing::TestWithParam<std::tuple<int, int, double> > {};

TEST_P(RobustAverageTest, Multidimensional) {
    auto param = GetParam();
    std::size_t nobs = std::get<0>(param);
    int iterations = std::get<1>(param);
    double trim = std::get<2>(param);

    auto data = scran_tests::simulate_vector(nobs, [&]{
        scran_tests::SimulationParameters opt;
        opt.seed = nobs * trim + iterations;
        return opt;
    }());

    mnncorrect::internal::RobustAverageOptions opt(iterations, trim);

    int ndim = 3;
    std::vector<double> copy(ndim * nobs);
    for (std::size_t i = 0; i < nobs; ++i) {
        for (int d = 0; d < ndim; ++d) {
            copy[d + i * ndim] = data[i] + d;
        }
    }

    double output = robust_average(1, nobs, data.data(), opt)[0];
    auto output_vector = robust_average(ndim, nobs, copy.data(), opt);
    for (int d = 0; d < ndim; ++d) {
        EXPECT_FLOAT_EQ(output_vector[d], output + d);
    }
}

TEST_P(RobustAverageTest, Indexed) {
    auto param = GetParam();
    std::size_t nobs = std::get<0>(param);
    int iterations = std::get<1>(param);
    double trim = std::get<2>(param);

    int ndim = 5;
    auto data = scran_tests::simulate_vector(nobs * ndim, [&]{
        scran_tests::SimulationParameters opt;
        opt.seed = nobs * trim + iterations;
        return opt;
    }());

    // Indexed calculation.
    std::mt19937_64 rng(nobs * trim + iterations);
    std::uniform_int_distribution dist(0, static_cast<int>(nobs) - 1);
    std::vector<int> indices;
    for (std::size_t i = 0; i < nobs; ++i) {
        indices.push_back(dist(rng));
    }

    mnncorrect::internal::RobustAverageOptions raopt(iterations, trim);
    std::vector<double> output = robust_average(ndim, indices, data.data(), raopt); 

    // Reference calculation.
    std::vector<double> copy(ndim * indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto target = data.data() + indices[i] * ndim;
        std::copy(target, target + ndim, copy.begin() + i * ndim);
    }

    std::vector<double> output2 = robust_average(ndim, indices.size(), copy.data(), raopt); 
    scran_tests::compare_almost_equal(output, output2);
}

INSTANTIATE_TEST_SUITE_P(
    RobustAverage,
    RobustAverageTest,
    ::testing::Combine(
        ::testing::Values(5, 13, 50), // number of observations.
        ::testing::Values(0, 1, 2), // number of iterations 
        ::testing::Values(0.1, 0.3) // trim proportion.
    )
);
